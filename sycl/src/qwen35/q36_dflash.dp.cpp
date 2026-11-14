// q36_dflash.dp.cpp - the DFlash 2 block drafter for the dense Qwen3.8-27B (q36_dflash.hpp; spec: tools/qwen35/dflash_ref.py).
//
// Device meta (int32 words / float words), staged by prepare(): [0] P  [1] anchor token  [2] first valid context position  [3] rows
// [16..47] context positions  [48..79] context ring slots  [80..87] block tokens  [96..4191] RoPE (cos[64], sin[64]) of 32 context
// rows  [4192..5215] RoPE of the 8 block rows.  Every kernel reads its variable inputs from here, so a recorded graph replays.
//
// Layouts: x16 = the q36xr W4A16 operand: element (col, i) of an [ncols][n_in] matrix at ((i / 16) * ncp + col) * 16 + i % 16, ncp =
// columns rounded up to 8.  A product's y is [col][n_out].  K/V rings: [layer][kv head][kRing][128] fp16.
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>

#include "q36_dflash.hpp"
#include "q36_xr_dense.hpp"

namespace strata::q36df {
using namespace k_consts;
namespace {

constexpr int kKvRows = 2 * kKvHeads * kHd;   // 2048: one layer's [k | v] rows of kv_all
constexpr int kKvAllRows = kLayers * kKvRows; // 10240
constexpr int kChunk = 128;                   // attention keys a work-group
constexpr int kTopChunk = 4096;               // logits a top-k work-group

inline uint16_t f2bf(float f) {
    uint32_t u = __builtin_bit_cast(uint32_t, f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
inline uint16_t h2u(sycl::half h) { return sycl::bit_cast<uint16_t>(h); }
inline float bf2f(uint16_t b) { return __builtin_bit_cast(float, (uint32_t) b << 16); }
inline size_t x16_at(int col, int i, int ncp) { return ((size_t) (i >> 4) * (size_t) ncp + (size_t) col) * 16u + (size_t) (i & 15); }
inline uint32_t key32(float f) {
    const uint32_t u = __builtin_bit_cast(uint32_t, f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
inline float unkey32(uint32_t k) { return __builtin_bit_cast(float, (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k); }
inline int round8(int n) { return (n + 7) / 8 * 8; }
inline uint32_t hash32(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return (uint32_t) x;
}

template <int ID> struct Tag;

}   // namespace

namespace k {

// ---------------------------------------------------------------------------------------------------------------- capture
// staging row of position pos: [kRing][5 x kN] bf16; tap k's segment at k * kN
void capture(sycl::queue& q, uint16_t* stag, int k, const float* x, int T, const int32_t* pos0) {
    const size_t items = (size_t) T * (kN / 4);
    q.parallel_for<Tag<1>>(sycl::range<1>(items), [=](sycl::id<1> id) {
        const int r = (int) (id[0] / (kN / 4)), c4 = (int) (id[0] % (kN / 4));
        if (r < T - kRing) return;   // a call longer than the ring: its last kRing rows count
        const int slot = (int) ((pos0[0] + r) % kRing);
        const sycl::float4 v = *reinterpret_cast<const sycl::float4*>(x + (size_t) r * kN + c4 * 4);
        uint16_t* d = stag + ((size_t) slot * kTaps + k) * kN + c4 * 4;
        const uint32_t lo = (uint32_t) f2bf(v.x()) | ((uint32_t) f2bf(v.y()) << 16);
        const uint32_t hi = (uint32_t) f2bf(v.z()) | ((uint32_t) f2bf(v.w()) << 16);
        *reinterpret_cast<sycl::uint2*>(d) = sycl::uint2(lo, hi);
    });
}

// ---------------------------------------------------------------------------------------------------------------- context rows
// staging rows of the n context rows -> fc's x16 operand, with a per-row power-of-two scale that keeps the fp16 range (the fc is
// linear and the next op is a norm: hidden_norm runs with eps / s^2, which is exact).
void ctx_prep(sycl::queue& q, const uint16_t* stag, const int32_t* meta, int n, uint16_t* x16, float* scale) {
    const int ncp = round8(n);
    q.submit([&](sycl::handler& h) {
        h.parallel_for<Tag<2>>(sycl::nd_range<1>((size_t) n * 256, 256), [=](sycl::nd_item<1> it) {
            const int j = (int) it.get_group(0), t = (int) it.get_local_id(0);
            const uint16_t* row = stag + (size_t) meta[kMetaCtxSlot + j] * (kTaps * kN);
            float am = 0.f;
            for (int i = t; i < kTaps * kN; i += 256) am = sycl::fmax(am, sycl::fabs(bf2f(row[i])));
            am = sycl::reduce_over_group(it.get_group(), am, sycl::maximum<float>());
            float s = 1.f;
            if (am > 8192.f) s = sycl::exp2(sycl::ceil(sycl::log2(am / 8192.f)));
            const float inv = 1.f / s;
            for (int i = t; i < kTaps * kN; i += 256) x16[x16_at(j, i, ncp)] = h2u(sycl::half(bf2f(row[i]) * inv));
            if (t == 0) scale[j] = s;
        });
    });
}

// y = fc output [n][kN] -> hidden_norm -> x16 [n_in = kN]
void ctx_norm(sycl::queue& q, const float* y, const float* scale, const float* w, float eps, int n, uint16_t* x16) {
    const int ncp = round8(n);
    q.parallel_for<Tag<3>>(sycl::nd_range<1>((size_t) n * 256, 256), [=](sycl::nd_item<1> it) {
        const int j = (int) it.get_group(0), t = (int) it.get_local_id(0);
        float v[kN / 256], ss = 0.f;
#pragma unroll
        for (int k = 0; k < kN / 256; ++k) { v[k] = y[(size_t) j * kN + t + k * 256]; ss += v[k] * v[k]; }
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float s = scale[j];
        const float r = 1.f / sycl::sqrt(ss * (1.f / kN) + eps / (s * s));
#pragma unroll
        for (int k = 0; k < kN / 256; ++k) {
            const int i = t + k * 256;
            x16[x16_at(j, i, ncp)] = h2u(sycl::half(v[k] * r * w[i]));
        }
    });
}

// kv_y [n][10240] (layer l: [k 1024 | v 1024] at l * 2048) -> k_norm + RoPE of K, V as is -> the rings at slot[j]
void ctx_store(sycl::queue& q, const float* kv, const int32_t* meta, int n, const float* const* kn, sycl::half* kring, sycl::half* vring, float eps) {
    struct KnPtrs { const float* p[kLayers]; } kp;
    for (int l = 0; l < kLayers; ++l) kp.p[l] = kn[l];
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh(sycl::range<1>(kHd), h);
        h.parallel_for<Tag<4>>(sycl::nd_range<1>((size_t) n * kLayers * kKvHeads * kHd, kHd), [=](sycl::nd_item<1> it) {
            const int g = (int) it.get_group(0), d = (int) it.get_local_id(0);
            const int kvh = g % kKvHeads, l = (g / kKvHeads) % kLayers, j = g / (kKvHeads * kLayers);
            const int slot = meta[kMetaCtxSlot + j];
            const float* src = kv + (size_t) j * kKvAllRows + (size_t) l * kKvRows;
            const float x = src[kvh * kHd + d];
            float ss = sycl::reduce_over_group(it.get_group(), x * x, sycl::plus<float>());
            const float xn = x * (1.f / sycl::sqrt(ss * (1.f / kHd) + eps)) * kp.p[l][d];
            sh[d] = xn;
            sycl::group_barrier(it.get_group());
            const float* rope = reinterpret_cast<const float*>(meta + kMetaRopeCtx) + (size_t) j * 128;
            const int i = d & 63;
            const float c = rope[i], s = rope[64 + i];
            const float o = d < 64 ? xn * c - sh[d + 64] * s : xn * c + sh[d - 64] * s;
            const size_t at = (((size_t) l * kKvHeads + kvh) * kRing + slot) * kHd + d;
            kring[at] = sycl::half(o);
            vring[at] = sycl::half(src[kKvRows / 2 + kvh * kHd + d]);
        });
    });
}

// ---------------------------------------------------------------------------------------------------------------- block rows
// h = rms(x) * w, also packed (8 columns)
void rms_pack(sycl::queue& q, const float* x, const float* w, float eps, float* h, uint16_t* x16) {
    q.parallel_for<Tag<5>>(sycl::nd_range<1>((size_t) kRows * 256, 256), [=](sycl::nd_item<1> it) {
        const int row = (int) it.get_group(0), t = (int) it.get_local_id(0);
        float v[kN / 256], ss = 0.f;
#pragma unroll
        for (int k = 0; k < kN / 256; ++k) { v[k] = x[(size_t) row * kN + t + k * 256]; ss += v[k] * v[k]; }
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float r = 1.f / sycl::sqrt(ss * (1.f / kN) + eps);
#pragma unroll
        for (int k = 0; k < kN / 256; ++k) {
            const int i = t + k * 256;
            const float o = v[k] * r * w[i];
            h[(size_t) row * kN + i] = o;
            x16[x16_at(row, i, kRows)] = h2u(sycl::half(o));
        }
    });
}

// the two-tap grouped conv of side `side` (0: before the block's products, 1: after): y[r] = (b0 + d0[r]) x[r] + (r >= 1)(b1 + d1[r]) x[r-1]
// with dyn [row][1280] (side x tap x group) and base [2 side][2 tap][kN]; the result packed (8 columns, n_in = kN)
void conv_pack(sycl::queue& q, const float* x, const float* dyn, int side, const float* base, uint16_t* x16) {
    q.parallel_for<Tag<6>>(sycl::nd_range<1>((size_t) kRows * kN, 256), [=](sycl::nd_item<1> it) {
        const size_t id = it.get_global_id(0);
        const int row = (int) (id / kN), ch = (int) (id % kN), grp = ch >> 4;
        const float* dr = dyn + (size_t) row * kKpOut + side * 640;
        float y = (base[(side * 2 + 0) * kN + ch] + dr[grp]) * x[(size_t) row * kN + ch];
        if (row >= 1) y += (base[(side * 2 + 1) * kN + ch] + dr[320 + grp]) * x[(size_t) (row - 1) * kN + ch];
        x16[x16_at(row, ch, kRows)] = h2u(sycl::half(y));
    });
}

// the finish conv of side 1 on a product's output, + residual, then the next norm: x += conv(o); h = rms(x) * w; h packed
void finish_rms(sycl::queue& q, const float* o, const float* dyn, const float* base, float* x, const float* w, float eps, float* h, uint16_t* x16) {
    q.parallel_for<Tag<7>>(sycl::nd_range<1>((size_t) kRows * 256, 256), [=](sycl::nd_item<1> it) {
        const int row = (int) it.get_group(0), t = (int) it.get_local_id(0);
        const float* dr = dyn + (size_t) row * kKpOut + 640;
        float v[kN / 256], ss = 0.f;
#pragma unroll
        for (int k = 0; k < kN / 256; ++k) {
            const int ch = t + k * 256, grp = ch >> 4;
            float y = (base[2 * kN + ch] + dr[grp]) * o[(size_t) row * kN + ch];
            if (row >= 1) y += (base[3 * kN + ch] + dr[320 + grp]) * o[(size_t) (row - 1) * kN + ch];
            const float xr = x[(size_t) row * kN + ch] + y;
            x[(size_t) row * kN + ch] = xr;
            v[k] = xr;
            ss += xr * xr;
        }
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float r = 1.f / sycl::sqrt(ss * (1.f / kN) + eps);
#pragma unroll
        for (int k = 0; k < kN / 256; ++k) {
            const int i = t + k * 256;
            const float hv = v[k] * r * w[i];
            h[(size_t) row * kN + i] = hv;
            x16[x16_at(row, i, kRows)] = h2u(sycl::half(hv));
        }
    });
}

// q [8][4096], kv [8][2048] -> per head RMS norm + RoPE (the block's positions): qb [8][32][128], kb [8][8][128]; vb [8][8][128] = V as is
void qk_prep(sycl::queue& q, const float* qy, const float* kvy, const float* qn, const float* kn, const int32_t* meta, float eps, float* qb, float* kb, float* vb) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh(sycl::range<1>(kHd), h);
        h.parallel_for<Tag<8>>(sycl::nd_range<1>((size_t) kRows * 48 * kHd, kHd), [=](sycl::nd_item<1> it) {
            const int g = (int) it.get_group(0), d = (int) it.get_local_id(0);
            const int row = g / 48, hs = g % 48;
            if (hs >= 40) {
                const int vh = hs - 40;
                vb[((size_t) row * kKvHeads + vh) * kHd + d] = kvy[(size_t) row * 2048 + 1024 + vh * kHd + d];
                return;
            }
            const bool isq = hs < 32;
            const float x = isq ? qy[(size_t) row * 4096 + hs * kHd + d] : kvy[(size_t) row * 2048 + (hs - 32) * kHd + d];
            float ss = sycl::reduce_over_group(it.get_group(), x * x, sycl::plus<float>());
            const float xn = x * (1.f / sycl::sqrt(ss * (1.f / kHd) + eps)) * (isq ? qn[d] : kn[d]);
            sh[d] = xn;
            sycl::group_barrier(it.get_group());
            const float* rope = reinterpret_cast<const float*>(meta + kMetaRopeBlk) + (size_t) row * 128;
            const int i = d & 63;
            const float c = rope[i], s = rope[64 + i];
            const float o = d < 64 ? xn * c - sh[d + 64] * s : xn * c + sh[d - 64] * s;
            if (isq) qb[((size_t) row * kHeads + hs) * kHd + d] = o;
            else kb[((size_t) row * kKvHeads + (hs - 32)) * kHd + d] = o;
        });
    });
}

// attention of the 8 block rows (non causal).  A work-group: one kv head, 2 of its 4 query heads x 8 rows = 16 queries, 128 keys.
// chunks 0 .. 15: context keys (positions lo + 128 c + t, lo = max(vlo, P - 2047)); chunk 16: the block's 8 keys.  Row i sees context
// position j when j >= P + i - 2047 (and j >= vlo) and every block key.  Writes the chunk's (m, l, acc) partials.
void attn_part(sycl::queue& q, const float* qb, const float* kb, const float* vb, const sycl::half* kring, const sycl::half* vring, int layer,
                 const int32_t* meta, float* part_acc, float* part_m, float* part_l) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>(16 * kHd), h);
        sycl::local_accessor<float, 1> ps(sycl::range<1>(16 * kChunk), h);
        h.parallel_for<Tag<9>>(sycl::nd_range<1>((size_t) kKvHeads * 2 * kChunks * kChunk, kChunk), [=](sycl::nd_item<1> it) {
            const int g = (int) it.get_group(0), t = (int) it.get_local_id(0);
            const int kvh = g / (2 * kChunks), rem = g % (2 * kChunks), qsp = rem / kChunks, c = rem % kChunks;
            const int P = meta[kMetaP], vlo = meta[kMetaVlo];
            const int lo = vlo > P - (kWindow - 1) ? vlo : P - (kWindow - 1);
            const int nctx = P - lo > 0 ? P - lo : 0;
            for (int i = t; i < 16 * kHd; i += kChunk) {
                const int qq = i / kHd, d = i % kHd;
                const int head = kvh * 4 + qsp * 2 + qq / 8, row = qq % 8;
                qs[i] = qb[((size_t) row * kHeads + head) * kHd + d];
            }
            sycl::group_barrier(it.get_group());
            // this thread's key
            const bool blk = c == kChunks - 1;
            int pos = 0;
            bool valid;
            const sycl::half* kh = nullptr;
            const float* kf = nullptr;
            if (!blk) {
                const int idx = c * kChunk + t;
                valid = idx < nctx;
                pos = lo + idx;
                if (valid) kh = kring + (((size_t) layer * kKvHeads + kvh) * kRing + (size_t) (pos % kRing)) * kHd;
            } else {
                valid = t < kRows;
                pos = P + t;
                if (valid) kf = kb + ((size_t) t * kKvHeads + kvh) * kHd;
            }
            float s[16];
#pragma unroll
            for (int qq = 0; qq < 16; ++qq) s[qq] = 0.f;
            if (valid) {
                for (int dd = 0; dd < kHd; dd += 32) {
                    float kv[32];
                    if (!blk) {
#pragma unroll
                        for (int e = 0; e < 32; ++e) kv[e] = (float) kh[dd + e];
                    } else {
#pragma unroll
                        for (int e = 0; e < 32; ++e) kv[e] = kf[dd + e];
                    }
#pragma unroll
                    for (int qq = 0; qq < 16; ++qq) {
                        float a = 0.f;
#pragma unroll
                        for (int e = 0; e < 32; ++e) a += qs[qq * kHd + dd + e] * kv[e];
                        s[qq] += a;
                    }
                }
            }
            const float scale = 0.08838834764831845f;   // 128 ^ -1/2
            const sycl::group<1> grp = it.get_group();
            for (int qq = 0; qq < 16; ++qq) {
                const int row = qq % 8;
                const bool vis = valid && (blk || pos >= P + row - (kWindow - 1));
                const float sc = vis ? s[qq] * scale : -INFINITY;
                const float m = sycl::reduce_over_group(grp, sc, sycl::maximum<float>());
                const float p = (vis && m > -INFINITY) ? sycl::exp(sc - m) : 0.f;
                const float l = sycl::reduce_over_group(grp, p, sycl::plus<float>());
                ps[qq * kChunk + t] = p;
                if (t == 0) {
                    const int head = kvh * 4 + qsp * 2 + qq / 8;
                    const size_t pi = ((size_t) c * kHeads + head) * kRows + row;
                    part_m[pi] = m;
                    part_l[pi] = l;
                }
            }
            sycl::group_barrier(grp);
            // PV: thread d = output dim
            float acc[16];
#pragma unroll
            for (int qq = 0; qq < 16; ++qq) acc[qq] = 0.f;
            const int nk = blk ? kRows : (nctx - c * kChunk < kChunk ? (nctx - c * kChunk > 0 ? nctx - c * kChunk : 0) : kChunk);
            for (int k = 0; k < nk; ++k) {
                float v;
                if (!blk) v = (float) vring[(((size_t) layer * kKvHeads + kvh) * kRing + (size_t) ((lo + c * kChunk + k) % kRing)) * kHd + t];
                else v = vb[((size_t) k * kKvHeads + kvh) * kHd + t];
#pragma unroll
                for (int qq = 0; qq < 16; ++qq) acc[qq] += ps[qq * kChunk + k] * v;
            }
#pragma unroll
            for (int qq = 0; qq < 16; ++qq) {
                const int head = kvh * 4 + qsp * 2 + qq / 8, row = qq % 8;
                part_acc[(((size_t) c * kHeads + head) * kRows + row) * kHd + t] = acc[qq];
            }
        });
    });
}

// combine the chunks' partials; the result packed as o_proj's operand (n_in 4096, 8 columns)
void attn_merge(sycl::queue& q, const float* part_acc, const float* part_m, const float* part_l, uint16_t* x16) {
    q.parallel_for<Tag<10>>(sycl::nd_range<1>((size_t) kRows * kHeads * kHd, kHd), [=](sycl::nd_item<1> it) {
        const int g = (int) it.get_group(0), d = (int) it.get_local_id(0);
        const int row = g / kHeads, head = g % kHeads;
        float M = -INFINITY;
        for (int c = 0; c < kChunks; ++c) M = sycl::fmax(M, part_m[((size_t) c * kHeads + head) * kRows + row]);
        float L = 0.f, a = 0.f;
        if (M > -INFINITY) {
            for (int c = 0; c < kChunks; ++c) {
                const size_t pi = ((size_t) c * kHeads + head) * kRows + row;
                const float mc = part_m[pi];
                if (mc == -INFINITY) continue;
                const float e = sycl::exp(mc - M);
                L += part_l[pi] * e;
                a += part_acc[pi * kHd + d] * e;
            }
        }
        const float o = L > 0.f ? a / L : 0.f;
        x16[x16_at(row, head * kHd + d, kRows)] = h2u(sycl::half(o));
    });
}

// ---------------------------------------------------------------------------------------------------------------- head: top-16 and the selector
// stage 1: per (row, chunk of 4096 logits) the 16 best 64-bit keys (key32(value) << 32 | ~index): ties go to the lower index
void top1(sycl::queue& q, const float* logits, int nd, int rows, int nch, uint64_t* part) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> sk(sycl::range<1>(kTopChunk), h);
        h.parallel_for<Tag<11>>(sycl::nd_range<2>(sycl::range<2>(rows, (size_t) nch * 256), sycl::range<2>(1, 256)), [=](sycl::nd_item<2> it) {
            const int row = (int) it.get_group(0), ch = (int) it.get_group(1), tid = (int) it.get_local_id(1);
            const int base = ch * kTopChunk, n = nd - base < kTopChunk ? nd - base : kTopChunk;
            const float* x = logits + (size_t) row * nd + base;
            for (int i = tid; i < kTopChunk; i += 256) sk[i] = i < n ? key32(x[i]) : 0u;
            sycl::group_barrier(it.get_group());
            uint64_t best = 0;
            for (int i = tid; i < kTopChunk; i += 256) {
                const uint32_t k = sk[i];
                if (k != 0u) { const uint64_t key = ((uint64_t) k << 32) | (0xFFFFFFFFu - (uint32_t) (base + i)); if (key > best) best = key; }
            }
            uint64_t* out = part + ((size_t) row * nch + ch) * kTopK;
            for (int r = 0; r < kTopK; ++r) {
                const uint64_t w = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<uint64_t>());
                if (tid == 0) out[r] = w;
                if (w != 0) {
                    const int e = (int) (0xFFFFFFFFu - (uint32_t) (w & 0xFFFFFFFFu)) - base;
                    if ((e & 255) == tid) {
                        sk[e] = 0u;
                        best = 0;
                        for (int i = tid; i < kTopChunk; i += 256) {
                            const uint32_t k = sk[i];
                            if (k != 0u) { const uint64_t key = ((uint64_t) k << 32) | (0xFFFFFFFFu - (uint32_t) (base + i)); if (key > best) best = key; }
                        }
                    }
                }
            }
        });
    });
}

// stage 2: the 16 best of a row's chunk winners -> token ids (through idmap) and raw logit values
void top2(sycl::queue& q, const uint64_t* part, int nch, int rows, const int32_t* idmap, int32_t* cand, float* unary) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint64_t, 1> ent(sycl::range<1>((size_t) 64 * kTopK), h);
        h.parallel_for<Tag<12>>(sycl::nd_range<1>((size_t) rows * 256, 256), [=](sycl::nd_item<1> it) {
            const int row = (int) it.get_group(0), tid = (int) it.get_local_id(0);
            const int ne = nch * kTopK;
            for (int i = tid; i < ne; i += 256) ent[i] = part[(size_t) row * ne + i];
            sycl::group_barrier(it.get_group());
            for (int r = 0; r < kTopK; ++r) {
                uint64_t best = 0;
                int bi = -1;
                for (int i = tid; i < ne; i += 256) if (ent[i] > best) { best = ent[i]; bi = i; }
                const uint64_t w = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<uint64_t>());
                if (w != 0 && best == w && bi >= 0) ent[bi] = 0;
                if (tid == 0) {
                    const int idx = w != 0 ? (int) (0xFFFFFFFFu - (uint32_t) (w & 0xFFFFFFFFu)) : 0;
                    cand[row * kTopK + r] = idmap ? idmap[idx] : idx;
                    unary[row * kTopK + r] = w != 0 ? unkey32((uint32_t) (w >> 32)) : -INFINITY;
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

// the selector's walk: step l (row l + 1): s[c] = unary[l][c] + sum_r pred[prev][r] hp[l + 1][r] succ[cand[l][c]][r]; pick the best c
// (ties: the lowest); prev = its token.  hp [8][256] (column = block row).  Writes the 7 picks and their softmax probabilities.
void selector_walk(sycl::queue& q, const float* hp, const int32_t* cand, const float* unary, const uint16_t* pred, const uint16_t* succ, const int32_t* meta,
              int32_t* out_tok, float* out_prob) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> w(sycl::range<1>(kRank), h), part(sycl::range<1>(256), h), sc(sycl::range<1>(kTopK), h);
        sycl::local_accessor<int32_t, 1> st(sycl::range<1>(2), h);
        h.parallel_for<Tag<13>>(sycl::nd_range<1>(256, 256), [=](sycl::nd_item<1> it) {
            const int t = (int) it.get_local_id(0);
            if (t == 0) st[0] = meta[kMetaAnchor];
            sycl::group_barrier(it.get_group());
            for (int l = 0; l < kDrafts; ++l) {
                const int prev = st[0];
                w[t] = bf2f(pred[(size_t) prev * kRank + t]) * hp[(size_t) (l + 1) * kRank + t];
                sycl::group_barrier(it.get_group());
                const int c = t >> 4, p = t & 15;
                const uint16_t* srow = succ + (size_t) cand[l * kTopK + c] * kRank + p * 16;
                float a = 0.f;
#pragma unroll
                for (int e = 0; e < 16; ++e) a += w[p * 16 + e] * bf2f(srow[e]);
                part[t] = a;
                sycl::group_barrier(it.get_group());
                if (t < kTopK) {
                    float s = unary[l * kTopK + t];
                    for (int e = 0; e < 16; ++e) s += part[t * 16 + e];
                    sc[t] = s;
                }
                sycl::group_barrier(it.get_group());
                if (t == 0) {
                    int bj = 0;
                    float bm = sc[0];
                    for (int c2 = 1; c2 < kTopK; ++c2) if (sc[c2] > bm) { bm = sc[c2]; bj = c2; }
                    float den = 0.f;
                    for (int c2 = 0; c2 < kTopK; ++c2) den += sycl::exp(sc[c2] - bm);
                    out_tok[l] = cand[l * kTopK + bj];
                    out_prob[l] = 1.f / den;
                    st[0] = out_tok[l];
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

}   // namespace k

namespace {

// parity / bench support: random bytes
void k_fill_random(sycl::queue& q, uint8_t* p, uint64_t bytes, uint64_t seed) {
    q.parallel_for<Tag<14>>(sycl::range<1>((bytes + 3) / 4), [=](sycl::id<1> id) {
        const uint32_t h = hash32(seed * 0x9E3779B97F4A7C15ULL + id[0]);
        const uint64_t o = id[0] * 4;
        for (int b = 0; b < 4; ++b) if (o + b < bytes) p[o + b] = (uint8_t) (h >> (8 * b));
    }).wait();
}
// a random w4 matrix: the nibbles random, the group meta sane (fp16 scale ~ 2^-8, zero 1 .. 16)
void k_fix_w4_meta(sycl::queue& q, uint8_t* p, uint64_t records) {
    q.parallel_for<Tag<15>>(sycl::range<1>(records * 32), [=](sycl::id<1> id) {
        uint32_t* rec = reinterpret_cast<uint32_t*>(p + (id[0] / 32) * 2176) + 512;
        const uint32_t h = hash32(0xABCDEFULL + id[0]);
        rec[id[0] % 32] = (0x1C00u + (h & 0x3FFu)) | ((1u + ((h >> 10) & 15u)) << 16);
    }).wait();
}
void k_fix_f32(sycl::queue& q, float* p, uint64_t n, float base, float amp, uint64_t seed) {
    q.parallel_for<Tag<16>>(sycl::range<1>(n), [=](sycl::id<1> id) {
        const uint32_t h = hash32(seed + id[0]);
        p[id[0]] = base + amp * ((float) (h & 0xFFFFu) / 32768.f - 1.f);
    }).wait();
}
void k_fix_bf16(sycl::queue& q, uint16_t* p, uint64_t n, float amp, uint64_t seed) {
    q.parallel_for<Tag<17>>(sycl::range<1>(n), [=](sycl::id<1> id) {
        const uint32_t h = hash32(seed + id[0]);
        p[id[0]] = f2bf(amp * ((float) (h & 0xFFFFu) / 32768.f - 1.f));
    }).wait();
}

}   // namespace

// ================================================================================================================ Dflash
Dflash::~Dflash() = default;   // (the process owns the device memory until exit: the engine never unloads)

const void* Dflash::tensor(const std::string& name) const { return w(name); }
const void* Dflash::w(const std::string& name) const {
    for (const auto& e : ents_) if (e.first == name) return blob_ + e.second.off;
    return nullptr;
}

// the index text: {"format": "DFLASH2", ..., "config": {...}, "tensors": {"name": {"off": N, "bytes": N, "kind": "w4", "n_in": N, "n_out": N}, ...}}
bool Dflash::parse_index(const std::string& js, std::string& err) {
    auto num = [&](const char* key, double dflt) {
        const size_t p = js.find(std::string("\"") + key + "\": ");
        return p == std::string::npos ? dflt : std::atof(js.c_str() + p + std::strlen(key) + 4);
    };
    if ((int) num("hidden", 0) != kN || (int) num("layers", 0) != kLayers || (int) num("ffn", 0) != kFF || (int) num("heads", 0) != kHeads ||
        (int) num("kv_heads", 0) != kKvHeads || (int) num("head_dim", 0) != kHd || (int) num("rank", 0) != kRank || (int) num("top_k", 0) != kTopK ||
        (int) num("window", 0) != kWindow || (int) num("block", 0) != kRows) {
        err = "the pack's config is not the 27B drafter's (hidden 5120, 5 layers, 32/8 heads of 128, window 2048, block 8)";
        return false;
    }
    mask_token_ = (int) num("mask_token", 248070);
    eps_ = (float) num("eps", 1e-6);
    if (std::fabs(num("rope_theta", 0) - 1.0e7) > 1.0) { err = "the pack's rope_theta is not 1e7"; return false; }
    size_t at = js.find("\"tensors\"");
    if (at == std::string::npos) { err = "no tensor index"; return false; }
    at = js.find('{', at);
    for (;;) {
        const size_t q0 = js.find('"', at + 1);
        if (q0 == std::string::npos) break;
        const size_t q1 = js.find('"', q0 + 1);
        if (q1 == std::string::npos) break;
        Ent e;
        unsigned long long off = 0, bytes = 0;
        char kind[16] = {0};
        if (std::sscanf(js.c_str() + q1 + 1, ": {\"off\": %llu, \"bytes\": %llu, \"kind\": \"%15[a-z0-9]\", \"n_in\": %d, \"n_out\": %d}", &off, &bytes, kind, &e.n_in,
                        &e.n_out) != 5) break;
        e.off = off;
        e.bytes = bytes;
        e.kind = !std::strcmp(kind, "w4") ? 0 : !std::strcmp(kind, "f32") ? 1 : 2;
        ents_.push_back({js.substr(q0 + 1, q1 - q0 - 1), e});
        at = js.find('}', q1);
        if (at == std::string::npos) break;
    }
    if (ents_.empty()) { err = "empty tensor index"; return false; }
    return true;
}

bool Dflash::alloc(sycl::queue& q, std::string& err) {
    auto need = [&](const char* n) -> const void* {
        const void* p = w(n);
        if (!p) err = std::string("the pack lacks ") + n;
        return p;
    };
    // shapes the products need
    auto shape_ok = [&](const std::string& n, int n_in, int n_out, int kind) {
        for (const auto& e : ents_)
            if (e.first == n) {
                const bool ok = e.second.n_in == n_in && e.second.n_out == n_out && e.second.kind == kind &&
                                (kind != 0 || e.second.bytes == strata::q36xr::w4_bytes(n_in, n_out));
                if (!ok) err = n + ": unexpected shape";
                return ok;
            }
        err = "the pack lacks " + n;
        return false;
    };
    if (!shape_ok("fc", kTaps * kN, kN, 0) || !shape_ok("kv_all", kN, kKvAllRows, 0) || !shape_ok("hproj", kN, kRank, 0) ||
        !shape_ok("hidden_norm", kN, 1, 1) || !shape_ok("norm", kN, 1, 1) || !shape_ok("pred_cb", kRank, 248320, 2) || !shape_ok("succ_cb", kRank, 248320, 2))
        return false;
    for (int l = 0; l < kLayers; ++l) {
        const std::string p = "l" + std::to_string(l) + ".";
        if (!shape_ok(p + "q", kN, kHeads * kHd, 0) || !shape_ok(p + "o", kHeads * kHd, kN, 0) || !shape_ok(p + "gate", kN, kFF, 0) ||
            !shape_ok(p + "up", kN, kFF, 0) || !shape_ok(p + "down", kFF, kN, 0) || !shape_ok(p + "akp", kN, kKpOut, 0) || !shape_ok(p + "mkp", kN, kKpOut, 0) ||
            !shape_ok(p + "in_ln", kN, 1, 1) || !shape_ok(p + "post_ln", kN, 1, 1) || !shape_ok(p + "qn", kHd, 1, 1) || !shape_ok(p + "kn", kHd, 1, 1) ||
            !shape_ok(p + "abase", kN, 4, 1) || !shape_ok(p + "mbase", kN, 4, 1))
            return false;
    }
    fc_ = need("fc"); kv_ = need("kv_all"); hproj_ = need("hproj");
    hidden_norm_ = static_cast<const float*>(need("hidden_norm")); norm_ = static_cast<const float*>(need("norm"));
    pred_cb_ = static_cast<const uint16_t*>(need("pred_cb")); succ_cb_ = static_cast<const uint16_t*>(need("succ_cb"));
    for (int l = 0; l < kLayers; ++l) {
        const std::string p = "l" + std::to_string(l) + ".";
        wq_[l] = w(p + "q"); wo_[l] = w(p + "o"); wg_[l] = w(p + "gate"); wu_[l] = w(p + "up"); wd_[l] = w(p + "down");
        akp_[l] = w(p + "akp"); mkp_[l] = w(p + "mkp");
        in_ln_[l] = static_cast<const float*>(w(p + "in_ln")); post_ln_[l] = static_cast<const float*>(w(p + "post_ln"));
        qn_[l] = static_cast<const float*>(w(p + "qn")); kn_[l] = static_cast<const float*>(w(p + "kn"));
        abase_[l] = static_cast<const float*>(w(p + "abase")); mbase_[l] = static_cast<const float*>(w(p + "mbase"));
    }
    auto dev = [&](size_t bytes) { vram_ += bytes; return sycl::aligned_alloc_device(4096, (bytes + 4095) / 4096 * 4096, q); };
    stag_ = static_cast<uint16_t*>(dev((size_t) kRing * kTaps * kN * 2));
    kring_ = static_cast<sycl::half*>(dev((size_t) kLayers * kKvHeads * kRing * kHd * 2));
    vring_ = static_cast<sycl::half*>(dev((size_t) kLayers * kKvHeads * kRing * kHd * 2));
    meta_ = static_cast<int32_t*>(dev(kMetaWords * 4));
    hmeta_ = sycl::malloc_host<int32_t>(kMetaWords, q);
    ctx_x16_ = static_cast<uint16_t*>(dev((size_t) kTaps * kN * kEagerCols * 2));
    ctx_y_ = static_cast<float*>(dev((size_t) kEagerCols * kN * 4));
    ctx_scale_ = static_cast<float*>(dev(kEagerCols * 4));
    ctx_h16_ = static_cast<uint16_t*>(dev((size_t) kN * kEagerCols * 2));
    kv_y_ = static_cast<float*>(dev((size_t) kEagerCols * kKvAllRows * 4));
    x_ = static_cast<float*>(dev(kRows * kN * 4)); h_ = static_cast<float*>(dev(kRows * kN * 4));
    h16_ = static_cast<uint16_t*>(dev(kRows * kN * 2)); h16b_ = static_cast<uint16_t*>(dev(kRows * kN * 2));
    att16_ = static_cast<uint16_t*>(dev(kRows * kHeads * kHd * 2));
    sw16_ = static_cast<uint16_t*>(dev((size_t) kRows * kFF * 2));
    dyn_a_ = static_cast<float*>(dev(kRows * kKpOut * 4)); dyn_m_ = static_cast<float*>(dev(kRows * kKpOut * 4));
    qy_ = static_cast<float*>(dev(kRows * kHeads * kHd * 4)); kvy_ = static_cast<float*>(dev(kRows * kKvRows * 4));
    qb_ = static_cast<float*>(dev(kRows * kHeads * kHd * 4)); kb_ = static_cast<float*>(dev(kRows * kKvHeads * kHd * 4));
    vb_ = static_cast<float*>(dev(kRows * kKvHeads * kHd * 4));
    part_acc_ = static_cast<float*>(dev((size_t) kChunks * kHeads * kRows * kHd * 4));
    part_m_ = static_cast<float*>(dev((size_t) kChunks * kHeads * kRows * 4)); part_l_ = static_cast<float*>(dev((size_t) kChunks * kHeads * kRows * 4));
    oy_ = static_cast<float*>(dev(kRows * kN * 4));
    gy_ = static_cast<float*>(dev((size_t) kRows * kFF * 4)); uy_ = static_cast<float*>(dev((size_t) kRows * kFF * 4));
    dny_ = static_cast<float*>(dev(kRows * kN * 4)); hp_y_ = static_cast<float*>(dev(kRows * kRank * 4));
    cand_ = static_cast<int32_t*>(dev(kDrafts * kTopK * 4)); unary_ = static_cast<float*>(dev(kDrafts * kTopK * 4));
    out_tok_ = static_cast<int32_t*>(dev(kDrafts * 4)); out_prob_ = static_cast<float*>(dev(kDrafts * 4));
    dbg_embed_ = static_cast<float*>(dev(2 * kN * 4));
    if (!stag_ || !kring_ || !vring_ || !meta_ || !hmeta_ || !ctx_x16_ || !kv_y_ || !part_acc_ || !gy_ || !dbg_embed_) { err = "out of device memory for the drafter's buffers"; return false; }
    q.memset(meta_, 0, kMetaWords * 4).wait();
    return true;
}

bool Dflash::load(sycl::queue& q, const std::string& path, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    char magic[8];
    uint64_t n = 0;
    if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "DFL2W4\0\1", 8) != 0 || std::fread(&n, 8, 1, f) != 1 || n > (64u << 20)) {
        std::fclose(f);
        err = path + " is not a DFLASH2 pack (tools/qwen35/dflash_pack.py)";
        return false;
    }
    std::string js((size_t) n, '\0');
    if (std::fread(js.data(), 1, (size_t) n, f) != (size_t) n) { std::fclose(f); err = path + ": short index"; return false; }
    if (!parse_index(js, err)) { std::fclose(f); return false; }
    const uint64_t base = (16 + n + 4095) / 4096 * 4096;
    uint64_t end = 0;
    for (const auto& e : ents_) end = std::max<uint64_t>(end, e.second.off + e.second.bytes);
    blob_bytes_ = end;
    blob_ = static_cast<uint8_t*>(sycl::aligned_alloc_device(4096, (end + 4095) / 4096 * 4096, q));
    if (!blob_) { std::fclose(f); err = "out of device memory for the drafter's weights"; return false; }
    vram_ += end;
    const size_t chunk = 64u << 20;
    uint8_t* host = sycl::malloc_host<uint8_t>(chunk, q);
    if (!host) { std::fclose(f); err = "no pinned memory for the load"; return false; }
    if (std::fseek(f, (long) base, SEEK_SET) != 0) { std::fclose(f); sycl::free(host, q); err = "seek"; return false; }
    for (uint64_t at = 0; at < end;) {
        const size_t k = (size_t) std::min<uint64_t>(chunk, end - at);
        if (std::fread(host, 1, k, f) != k) { std::fclose(f); sycl::free(host, q); err = path + ": the file is shorter than its index"; return false; }
        q.memcpy(blob_ + at, host, k).wait();
        at += k;
    }
    std::fclose(f);
    sycl::free(host, q);
    return alloc(q, err);
}

bool Dflash::load_synthetic(sycl::queue& q, uint64_t seed, std::string& err) {
    uint64_t off = 0;
    auto add = [&](const std::string& name, int kind, int n_in, int n_out) {
        Ent e;
        e.kind = kind; e.n_in = n_in; e.n_out = n_out;
        e.bytes = kind == 0 ? strata::q36xr::w4_bytes(n_in, n_out) : (uint64_t) n_in * n_out * (kind == 1 ? 4 : 2);
        e.off = off;
        off += (e.bytes + 4095) / 4096 * 4096;
        ents_.push_back({name, e});
    };
    add("fc", 0, kTaps * kN, kN); add("kv_all", 0, kN, kKvAllRows);
    for (int l = 0; l < kLayers; ++l) {
        const std::string p = "l" + std::to_string(l) + ".";
        add(p + "q", 0, kN, kHeads * kHd); add(p + "o", 0, kHeads * kHd, kN); add(p + "gate", 0, kN, kFF); add(p + "up", 0, kN, kFF);
        add(p + "down", 0, kFF, kN); add(p + "akp", 0, kN, kKpOut); add(p + "mkp", 0, kN, kKpOut);
        add(p + "in_ln", 1, kN, 1); add(p + "post_ln", 1, kN, 1); add(p + "qn", 1, kHd, 1); add(p + "kn", 1, kHd, 1); add(p + "abase", 1, kN, 4); add(p + "mbase", 1, kN, 4);
    }
    add("hproj", 0, kN, kRank); add("hidden_norm", 1, kN, 1); add("norm", 1, kN, 1); add("pred_cb", 2, kRank, 248320); add("succ_cb", 2, kRank, 248320);
    blob_bytes_ = off;
    blob_ = static_cast<uint8_t*>(sycl::aligned_alloc_device(4096, (off + 4095) / 4096 * 4096, q));
    if (!blob_) { err = "out of device memory for the synthetic pack"; return false; }
    vram_ += off;
    uint64_t s = seed;
    for (const auto& e : ents_) {
        uint8_t* p = blob_ + e.second.off;
        ++s;
        if (e.second.kind == 0) { k_fill_random(q, p, e.second.bytes, s); k_fix_w4_meta(q, p, e.second.bytes / 2176); }
        else if (e.second.kind == 1) {
            const std::string& nm = e.first;
            const bool ln = nm.find("_ln") != std::string::npos || nm.find("norm") != std::string::npos || nm.find(".qn") != std::string::npos ||
                            nm.find(".kn") != std::string::npos;
            k_fix_f32(q, reinterpret_cast<float*>(p), (uint64_t) e.second.n_in * e.second.n_out, ln ? 1.f : 0.f, ln ? 0.1f : 0.3f, s * 7919);
        } else k_fix_bf16(q, reinterpret_cast<uint16_t*>(p), (uint64_t) e.second.n_in * e.second.n_out, 0.5f, s * 104729);
    }
    return alloc(q, err);
}

void Dflash::set_hooks(sycl::queue& q, const Hooks& h) {
    hooks_ = h;
    nch_ = (h.max_nd + kTopChunk - 1) / kTopChunk;
    logits_ = static_cast<float*>(sycl::aligned_alloc_device(4096, ((size_t) kDrafts * h.max_nd * 4 + 4095) / 4096 * 4096, q));
    top_part_ = static_cast<uint64_t*>(sycl::aligned_alloc_device(4096, ((size_t) kDrafts * nch_ * kTopK * 8 + 4095) / 4096 * 4096, q));
    vram_ += (size_t) kDrafts * h.max_nd * 4;
}

bool Dflash::prod(sycl::queue& q, const void* wt, const uint16_t* x16, float* y, int n_in, int n_out, int ncols) {
    if (!strata::q36xr::mmvq_w4h(wt, x16, y, n_in, n_out, ncols, &q)) {
        static int shown = 0;
        if (shown++ < 3) std::fprintf(stderr, "strata-q36: dflash: mmvq_w4h declined %d x %d (%d columns)\n", n_in, n_out, ncols);
        return false;
    }
    return true;
}

// ---- target side
void Dflash::capture(sycl::queue& q, int tap, const float* x, int T, const int32_t* pos0_dev) {
    if (stag_) k::capture(q, stag_, tap, x, T, pos0_dev);
}

void Dflash::invalidate() { have_cap_ = false; cap_lo_ = cap_hi_ = ctx_vlo_ = ctx_hi_ = 0; }

void Dflash::note_forward(int64_t p0, int64_t n) {
    if (n <= 0) return;
    if (!have_cap_) {
        have_cap_ = true;
        cap_lo_ = p0; cap_hi_ = p0 + n; ctx_vlo_ = ctx_hi_ = p0;
    } else if (p0 == cap_hi_) {
        cap_hi_ += n;
    } else if (p0 < cap_hi_ && p0 >= cap_lo_) {   // a rewind inside the captured run: what is below p0 stays
        cap_hi_ = p0 + n;
        ctx_hi_ = std::min(ctx_hi_, p0);
        ctx_vlo_ = std::min(ctx_vlo_, ctx_hi_);
    } else {                                       // below the run or past it: nothing known about the rows in between
        cap_lo_ = p0; cap_hi_ = p0 + n; ctx_vlo_ = ctx_hi_ = p0;
    }
}

// ---- meta
void Dflash::stage_meta(sycl::queue& q, int64_t P, int32_t anchor, int64_t c0, int nrows) {
    int32_t* m = hmeta_;
    std::memset(m, 0, sizeof(int32_t) * 96);
    m[kMetaP] = (int32_t) P;
    m[kMetaAnchor] = anchor;
    m[kMetaVlo] = (int32_t) std::max<int64_t>(ctx_vlo_, 0);
    m[kMetaN] = nrows;
    float* rc = reinterpret_cast<float*>(m + kMetaRopeCtx);
    float* rb = reinterpret_cast<float*>(m + kMetaRopeBlk);
    const double theta = 1.0e7;
    auto table = [&](int64_t pos, float* out) {
        for (int i = 0; i < 64; ++i) {
            const double ang = (double) pos * std::pow(theta, -2.0 * i / kHd);
            out[i] = (float) std::cos(ang);
            out[64 + i] = (float) std::sin(ang);
        }
    };
    for (int j = 0; j < nrows; ++j) {
        const int64_t pos = c0 + j;
        m[kMetaCtxPos + j] = (int32_t) pos;
        m[kMetaCtxSlot + j] = (int32_t) (pos % kRing);
        table(pos, rc + (size_t) j * 128);
    }
    for (int r = 0; r < kRows; ++r) {
        m[kMetaTok + r] = r == 0 ? anchor : mask_token_;
        table(P + r, rb + (size_t) r * 128);
    }
    q.memcpy(meta_, hmeta_, sizeof(int32_t) * kMetaWords);
}

void Dflash::flush_eager(sycl::queue& q, int64_t c0, int nrows) {
    stage_meta(q, c0 + nrows, 0, c0, nrows);   // (P and the block rows' tables are not read by the context update)
    ctx_update(q, nrows, true);
    q.wait();                                  // the host's meta may be rewritten
}

int Dflash::prepare(sycl::queue& q, int64_t P, int32_t anchor) {
    if (!have_cap_ || cap_hi_ != P) {   // the captured run does not end at P (a path that never called note_forward, a session switch)
        have_cap_ = true;
        cap_lo_ = cap_hi_ = P;
        ctx_vlo_ = ctx_hi_ = P;
    }
    const int64_t lo_need = std::max<int64_t>({cap_lo_, P - (kWindow - 1), 0});
    int64_t s = ctx_hi_;
    if (s < lo_need) { s = lo_need; ctx_vlo_ = s; }          // the ring's run does not reach the rows we still can get: restart it
    if (s > P) s = P;
    int64_t total = P - s;
    const int ng = (int) std::min<int64_t>(total, kRows);
    int64_t at = s;
    while (total - ng > 0) {
        const int k = (int) std::min<int64_t>(total - ng, kEagerCols);
        dump_rows(q, at, k);
        flush_eager(q, at, k);
        at += k;
        total -= k;
    }
    dump_rows(q, at, ng);
    stage_meta(q, P, anchor, at, ng);
    ctx_hi_ = P;
    return ng;
}

// ---- passes
void Dflash::ctx_update(sycl::queue& q, int n, bool /*eager*/) {
    if (n <= 0) return;
    k::ctx_prep(q, stag_, meta_, n, ctx_x16_, ctx_scale_);
    prod(q, fc_, ctx_x16_, ctx_y_, kTaps * kN, kN, n);
    k::ctx_norm(q, ctx_y_, ctx_scale_, hidden_norm_, eps_, n, ctx_h16_);
    prod(q, kv_, ctx_h16_, kv_y_, kN, kKvAllRows, n);
    k::ctx_store(q, kv_y_, meta_, n, kn_, kring_, vring_, eps_);
}

void Dflash::block(sycl::queue& q) {
    hooks_.embed(meta_ + kMetaTok, kRows, x_);
    if (debug_) q.memcpy(dbg_embed_, x_, 2 * kN * sizeof(float));
    k::rms_pack(q, x_, in_ln_[0], eps_, h_, h16_);
    for (int l = 0; l < kLayers; ++l) {
        prod(q, akp_[l], h16_, dyn_a_, kN, kKpOut, kRows);
        k::conv_pack(q, h_, dyn_a_, 0, abase_[l], h16b_);
        prod(q, wq_[l], h16b_, qy_, kN, kHeads * kHd, kRows);
        prod(q, static_cast<const uint8_t*>(kv_) + (size_t) l * (kKvRows / 16) * (kN / 256) * 2176, h16b_, kvy_, kN, kKvRows, kRows);
        k::qk_prep(q, qy_, kvy_, qn_[l], kn_[l], meta_, eps_, qb_, kb_, vb_);
        k::attn_part(q, qb_, kb_, vb_, kring_, vring_, l, meta_, part_acc_, part_m_, part_l_);
        k::attn_merge(q, part_acc_, part_m_, part_l_, att16_);
        prod(q, wo_[l], att16_, oy_, kHeads * kHd, kN, kRows);
        k::finish_rms(q, oy_, dyn_a_, abase_[l], x_, post_ln_[l], eps_, h_, h16_);
        prod(q, mkp_[l], h16_, dyn_m_, kN, kKpOut, kRows);
        k::conv_pack(q, h_, dyn_m_, 0, mbase_[l], h16b_);
        prod(q, wg_[l], h16b_, gy_, kN, kFF, kRows);
        prod(q, wu_[l], h16b_, uy_, kN, kFF, kRows);
        strata::q36xr::w4h_swiglu_pack(gy_, uy_, sw16_, kFF, kRows, &q);
        prod(q, wd_[l], sw16_, dny_, kFF, kN, kRows);
        k::finish_rms(q, dny_, dyn_m_, mbase_[l], x_, l + 1 < kLayers ? in_ln_[l + 1] : norm_, eps_, h_, h16_);
    }
    // h_ = the final-normed block rows
    prod(q, hproj_, h16_, hp_y_, kN, kRank, kRows);
    int nd = 0;
    const int32_t* idmap = nullptr;
    hooks_.head(h_ + kN, kDrafts, logits_, nd, idmap);
    k::top1(q, logits_, nd, kDrafts, (nd + kTopChunk - 1) / kTopChunk, top_part_);
    k::top2(q, top_part_, (nd + kTopChunk - 1) / kTopChunk, kDrafts, idmap, cand_, unary_);
    k::selector_walk(q, hp_y_, cand_, unary_, pred_cb_, succ_cb_, meta_, out_tok_, out_prob_);
}

void Dflash::run(sycl::queue& q, int n) {
    ctx_update(q, n, false);
    block(q);
}

bool Dflash::debug_last(sycl::queue& q, Debug& d) {
    q.wait();
    d.embed.resize(2 * kN); d.hn.resize((size_t) kRows * kN); d.cand.resize(kDrafts * kTopK); d.unary.resize(kDrafts * kTopK);
    d.path.resize(kDrafts); d.prob.resize(kDrafts);
    q.memcpy(d.embed.data(), dbg_embed_, d.embed.size() * 4);
    q.memcpy(d.hn.data(), h_, d.hn.size() * 4);
    q.memcpy(d.cand.data(), cand_, d.cand.size() * 4);
    q.memcpy(d.unary.data(), unary_, d.unary.size() * 4);
    q.memcpy(d.path.data(), out_tok_, d.path.size() * 4);
    q.memcpy(d.prob.data(), out_prob_, d.prob.size() * 4).wait();
    return true;
}

bool Dflash::debug_stage_rows(sycl::queue& q, int64_t p0, int n, std::vector<uint16_t>& out) {
    if (!have_cap_ || p0 < cap_lo_ || p0 + n > cap_hi_) return false;
    out.resize((size_t) n * kTaps * kN);
    q.wait();
    for (int j = 0; j < n; ++j)
        q.memcpy(out.data() + (size_t) j * kTaps * kN, stag_ + (size_t) ((p0 + j) % kRing) * kTaps * kN, (size_t) kTaps * kN * 2);
    q.wait();
    return true;
}

// ---- dump (tools/qwen35/dflash_ref.py replay)
void Dflash::enable_dump(const std::string& dir, int max_steps, int max_rows) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    dump_dir_ = dir;
    f_ctx_ = std::fopen((dir + "/ctx.bin").c_str(), "wb");
    f_steps_ = std::fopen((dir + "/steps.jsonl").c_str(), "wb");
    f_blocks_ = std::fopen((dir + "/blocks.bin").c_str(), "wb");
    dump_ = f_ctx_ && f_steps_ && f_blocks_;
    dump_max_ = max_steps;
    dump_rows_max_ = max_rows;
    debug_ = dump_;
    if (!dump_) std::fprintf(stderr, "strata-q36: dflash: cannot write the dump folder %s\n", dir.c_str());
    else dump_meta();
}

void Dflash::dump_meta() {
    FILE* f = std::fopen((dump_dir_ + "/meta.json").c_str(), "wb");
    if (!f) return;
    std::fprintf(f, "{\"n_embd\": %d, \"taps\": [5, 19, 33, 47, 61], \"mask_token\": %d, \"block\": %d, \"window\": %d, \"head_subset\": %s, \"n_steps\": %d, \"ctx_records\": %lld}\n",
                 kN, mask_token_, kRows, kWindow, dump_subset_ ? "true" : "false", dump_steps_, (long long) dump_records_);
    std::fclose(f);
}

void Dflash::dump_subset(const int32_t* ids, int n) {
    if (!dump_) return;
    if (FILE* f = std::fopen((dump_dir_ + "/subset.bin").c_str(), "wb")) {
        std::fwrite(ids, 4, (size_t) n, f);
        std::fclose(f);
        dump_subset_ = true;
        dump_meta();
    }
}

void Dflash::dump_rows(sycl::queue& q, int64_t c0, int n) {
    if (!dump_ || n <= 0 || dump_records_ + n > dump_rows_max_) return;
    std::vector<uint16_t> rows;
    if (!debug_stage_rows(q, c0, n, rows)) return;
    for (int j = 0; j < n; ++j) {
        const int32_t hd[2] = {(int32_t) (c0 + j), 0};
        std::fwrite(hd, 4, 2, f_ctx_);
        std::fwrite(rows.data() + (size_t) j * kTaps * kN, 2, (size_t) kTaps * kN, f_ctx_);
    }
    std::fflush(f_ctx_);
    dump_records_ += n;
}

void Dflash::dump_step(sycl::queue& q, int64_t P, int32_t anchor) {
    if (!dump_ || dump_steps_ >= dump_max_) return;
    Debug d;
    debug_last(q, d);
    std::fprintf(f_steps_, "{\"step\": %d, \"P\": %lld, \"anchor\": %d, \"ctx_records\": %lld, \"valid_lo\": %lld, \"cand\": [", dump_steps_, (long long) P, anchor,
                 (long long) dump_records_, (long long) ctx_vlo_);
    for (int l = 0; l < kDrafts; ++l) {
        std::fprintf(f_steps_, "%s[", l ? ", " : "");
        for (int c = 0; c < kTopK; ++c) std::fprintf(f_steps_, "%s%d", c ? ", " : "", d.cand[l * kTopK + c]);
        std::fprintf(f_steps_, "]");
    }
    std::fprintf(f_steps_, "], \"unary\": [");
    for (int l = 0; l < kDrafts; ++l) {
        std::fprintf(f_steps_, "%s[", l ? ", " : "");
        for (int c = 0; c < kTopK; ++c) std::fprintf(f_steps_, "%s%.9g", c ? ", " : "", std::isfinite(d.unary[l * kTopK + c]) ? d.unary[l * kTopK + c] : -1.0e30);
        std::fprintf(f_steps_, "]");
    }
    std::fprintf(f_steps_, "], \"path\": [");
    for (int l = 0; l < kDrafts; ++l) std::fprintf(f_steps_, "%s%d", l ? ", " : "", d.path[l]);
    std::fprintf(f_steps_, "], \"prob\": [");
    for (int l = 0; l < kDrafts; ++l) std::fprintf(f_steps_, "%s%.5f", l ? ", " : "", d.prob[l]);
    std::fprintf(f_steps_, "]}\n");
    std::fflush(f_steps_);
    std::fwrite(d.embed.data(), 4, d.embed.size(), f_blocks_);
    std::fwrite(d.hn.data(), 4, d.hn.size(), f_blocks_);
    std::fflush(f_blocks_);
    ++dump_steps_;
    dump_meta();
}

}   // namespace strata::q36df
