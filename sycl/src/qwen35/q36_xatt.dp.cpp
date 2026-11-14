// q36_xatt.dp.cpp - strata-q36's decode attention on the Arc Pro B70's matrix engine (XMX, ESIMD).  In libq36_w4.
//
// k_attn_gqa (q36_engine.dp.cpp) scores and sums with FP32 multiply-adds: a window of 5 tokens is 30 rows a KV head, 15,360
// multiply-adds a cell, and at 25K cells the 16 layers take about 16 ms of a 51 ms step while their K / V bytes would pass
// in 3.  Here both products are DPAS: 512 int8 multiply-adds an instruction.
//
// What it reads (the mirror of a layer's cache, q36_xatt.hpp): K as int8 codes with one scale a cell and KV head, V as int8
// codes transposed in blocks of 32 cells (a DPAS sums along 32 contiguous bytes: for the value sums those must be 32 cells
// of one dimension), one scale a cell.
//
// A row's cells split into chunks exactly as k_attn_gqa's (att_chunk_cells: the ROW's own size, so a row's result does not
// depend on the window it stands in), and each chunk gives the partial sums, max and sum that k_attn_merge_gqa merges.
// Three kernels a call:
//   queries   each row's query as 256 int8 codes and a scale, laid out as DPAS operands (lanes = rows, 16 a set)
//   scores    a thread per (KV head, chunk).  Pass 1, 8 cells at a time: the cells' K by block loads, one DPAS per 32
//             dimensions and set of 16 rows, the scores (masked outside the row's cells) to scratch, the row's max.
//             Pass 2, 32 cells at a time: p = exp(score - max), the row's sum, then w = p x the cell's V scale as a
//             16-bit integer against the row's largest V scale in the chunk, its low and high bytes packed as DPAS
//             operands (4 cells a dword).
//   values    a thread per (KV head, chunk, 32 dimensions, pair of row sets): per block of 32 cells the V codes of its
//             dimensions (4 loads) and the P operands, two DPAS (low and high bytes) per 8 dimensions and row set.  The
//             sums stay integers to the end of the chunk; then (256 x high + low) x the row's scale.
// Every step is a fixed function of the row's own cells, in cell order: a T-row window is bitwise T one-row windows.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "strata/sycl_queue.hpp"
#include "q36_xatt.hpp"

namespace strata::q36xatt {
namespace {
using namespace sycl::ext::intel::esimd;
namespace es = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

constexpr float kNeg = -1e30f;   // a masked score / an empty max (k_attn_gqa's kAttNeg)

inline int chunk_cells_host(int n, int nchk) {
    const int per = (n + nchk - 1) / nchk;
    const int cs = (per + 63) / 64 * 64;
    return cs > 64 ? cs : 64;
}
// att_chunk_cells on the device (no integer division by a run-time value in an ESIMD kernel): n < 2^24
SYCL_ESIMD_FUNCTION inline int chunk_cells(int n, int nchk, float inv) {
    const int v = n + nchk - 1;
    int per = (int) ((float) v * inv);
    if (per * nchk > v) --per;
    if ((per + 1) * nchk <= v) ++per;
    const int cs = ((per + 63) >> 6) << 6;
    return cs > 64 ? cs : 64;
}
// e^x for x in [-87, 0]: 2^n x e^z, n the nearest integer to x / ln 2, z in [-0.35, 0.35] by its series to z^6 (1.2e-7)
SYCL_ESIMD_FUNCTION inline simd<float, 16> exp16(simd<float, 16> x) {
    simd<float, 16> y = x * 1.44269504f;
    simd<int, 16> n = convert<int>(y - 0.5f);
    simd<float, 16> nf = n;
    simd<float, 16> z = (y - nf) * 0.693147181f;
    simd<float, 16> e = 1.0f + z * (1.0f + z * (0.5f + z * (0.16666667f + z * (0.041666668f + z * (0.008333334f + z * 0.0013888889f)))));
    simd<int, 16> b = (n + 127) << 23;
    simd<float, 16> s = b.template bit_cast_view<float>();
    return e * s;
}
SYCL_ESIMD_FUNCTION inline simd<float, 128> exp128(simd<float, 128> x) {
    simd<float, 128> y = x * 1.44269504f;
    simd<int, 128> n = convert<int>(y - 0.5f);
    simd<float, 128> nf = n;
    simd<float, 128> z = (y - nf) * 0.693147181f;
    simd<float, 128> e = 1.0f + z * (1.0f + z * (0.5f + z * (0.16666667f + z * (0.041666668f + z * (0.008333334f + z * 0.0013888889f)))));
    simd<int, 128> b = (n + 127) << 23;
    simd<float, 128> s = b.template bit_cast_view<float>();
    return e * s;
}
// a class of a call's tokens: the ones whose chunk size is cs.  Chunk `ch` of them is [a, hik[k]) for token k (hik == a:
// no cell), ub the largest end.  (At most two sizes in a window: the size steps every nchk x 64 cells.)
struct Range {
    int a, ub;
    int lok[16], hik[16];
};
SYCL_ESIMD_FUNCTION inline void class_range(Range& r, const int* csk, int cs, int ch, int n0, int nt) {
    r.a = ch * cs;
    r.ub = r.a;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        int lo = 0, hi = 0;
        if (k < nt && csk[k] == cs) {
            const int e = r.a + cs < n0 + k ? r.a + cs : n0 + k;
            if (e > r.a) {
                lo = r.a;
                hi = e;
                if (e > r.ub) r.ub = e;
            }
        }
        r.lok[k] = lo;
        r.hik[k] = hi;
    }
}

// ---- queries: row r = 16 ls + lane of a KV head is token r / kG, query head r % kG.  Slice s (32 dimensions) of row set
// ls: 128 dwords, dword d of row `lane` at [16 d + lane] (a DPAS's 16-lane operand).  A thread a row; the lanes of rows the
// call does not have are not written (their scores are masked).
template <int NLS> struct TagXaQ;
template <int NLS> void launch_queries(sycl::queue& q, const Call c) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagXaQ<NLS>>(sycl::nd_range<2>(sycl::range<2>((size_t) c.HKV, (size_t) c.nt * kG), sycl::range<2>(1, 1)),
                                    [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
            const int kvh = (int) it.get_global_id(0), r = (int) it.get_global_id(1);
            const int ls = r >> 4, lane = r & 15;
            const int k = (r * 43691) >> 18, hd = r - k * kG;   // r / kG, r % kG
            const float* src = c.q + ((size_t) (c.t0 + k) * c.H + kvh * kG + hd) * kHD;
            simd<float, 64> x[4];
            float mx = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                x[i] = block_load<float, 64>(src + 64 * i);
                const float m1 = hmax<float>(es::abs(x[i]));
                mx = m1 > mx ? m1 : mx;
            }
            const float sc = mx / 127.0f, inv = sc > 0.f ? 1.0f / sc : 0.f;
            uint32_t* dst = c.qb + (((size_t) kvh * 8) * NLS + ls) * 128 + lane;
#pragma unroll
            for (int i = 0; i < 4; ++i) {   // 64 values: 16 dwords = 2 slices
                simd<float, 64> y = x[i] * inv, half(0.5f);   // the nearest code, a tie away from zero
                half.merge(simd<float, 64>(-0.5f), y < 0.0f);
                simd<int, 64> ci = convert<int>(y + half);
                ci = (es::min)((es::max)(ci, simd<int, 64>(-127)), simd<int, 64>(127));
                simd<uint32_t, 64> cu = ci.template bit_cast_view<uint32_t>();
                simd<uint32_t, 16> b0 = cu.template select<16, 4>(0), b1 = cu.template select<16, 4>(1), b2 = cu.template select<16, 4>(2),
                                   b3 = cu.template select<16, 4>(3);
                simd<uint32_t, 16> w = (b0 & 255u) | ((b1 & 255u) << 8) | ((b2 & 255u) << 16) | (b3 << 24);
#pragma unroll
                for (int hs = 0; hs < 2; ++hs) {
                    simd<uint32_t, 8> off(0u, 64u);   // dword d at [16 d + lane]
                    simd<uint32_t, 8> v8 = w.template select<8, 1>(8 * hs);
                    scatter<uint32_t, 8>(dst + (size_t) (2 * i + hs) * NLS * 128, off, v8);
                }
            }
            c.qs[kvh * kMaxRows + r] = sc * c.scale;
        });
    });
}

// ---- scores and each chunk's softmax: kW threads a (KV head, chunk).  Thread w takes the chunk's tiles (8 cells) w,
// w + kW, ... in pass 1 and its blocks (32 cells) w, w + kW, ... in pass 2; the rows' max and largest V scale go through
// shared local memory between the passes, the rows' sums after pass 2 (added in thread order: a fixed order per row).
constexpr int kW = 4;
constexpr auto kAl16 = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::alignment<16>};
constexpr auto kPf = sycl::ext::oneapi::experimental::properties{
    cache_hint_L1<cache_hint::cached>, cache_hint_L2<cache_hint::cached>};
// NS: the K scales a cell (1: the mirror's; 4: the engine's 8-bit cache, a scale per 64 values - the integer dots are then
// taken per 64 values and scaled before they are added)
template <int NLS, int NS> struct TagXaS;
template <int NLS, int NS> void launch_scores(sycl::queue& q, const Call c) {
    const float inv = 1.0f / (float) c.nchk;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagXaS<NLS, NS>>(sycl::nd_range<3>(sycl::range<3>((size_t) c.HKV, (size_t) c.nchk, (size_t) kW), sycl::range<3>(1, 1, (size_t) kW)),
                                    [=](sycl::nd_item<3> it) SYCL_ESIMD_KERNEL {
            slm_init<2 * kW * NLS * 3 * 64>();
            const int kvh = (int) it.get_global_id(0), ch = (int) it.get_global_id(1), w = (int) it.get_local_id(2);
            const int n0 = c.pos0[0] + c.t0 + 1, nt = c.nt, nchk = c.nchk;
            int csk[16];
#pragma unroll
            for (int k = 0; k < 16; ++k) csk[k] = k < nt ? chunk_cells(n0 + k, nchk, inv) : 0;
            const bool has_tmask = c.tmask != nullptr;
            const int p0_tree = has_tmask ? (c.tp0 ? c.tp0[0] : c.pos0[0]) : 0x3fffffff;
            simd<uint32_t, 16> tmk[NLS];
            if (has_tmask) {
                for (int ls = 0; ls < NLS; ++ls)
                    for (int lane = 0; lane < 16; ++lane) {
                        const int tok_k = (16 * ls + lane) / kG;
                        tmk[ls][lane] = tok_k < nt ? c.tmask[c.t0 + tok_k] : 0u;
                    }
            }
            const int kst = c.kstride > 0 ? c.kstride : c.ctxp;
            const uint32_t* kb = c.k8 + (size_t) kvh * kst * 64;
            const float* ksb = NS == 1 ? c.ks + (size_t) kvh * kst : nullptr;
            const uint32_t* ks16b = NS == 4 ? reinterpret_cast<const uint32_t*>(c.ks16) + (size_t) kvh * kst * 2 : nullptr;
            const float* vsb = c.vs + (size_t) kvh * c.ctxp;
            float* scb = c.sc + ((size_t) (kvh * nchk + ch) * c.capc) * (NLS * 16);
            const uint32_t* qbb = c.qb + (size_t) kvh * 8 * NLS * 128;
            simd<float, 16> m[NLS], l[NLS], vmx[NLS], qsc[NLS];
#pragma unroll
            for (int ls = 0; ls < NLS; ++ls) {
                m[ls] = kNeg;
                l[ls] = 0.f;
                vmx[ls] = 0.f;
                qsc[ls] = block_load<float, 16>(c.qs + kvh * kMaxRows + 16 * ls);
            }
            // (one row set: the query operands stay in registers; more: loaded at each use)
            constexpr int NH = 1;
            simd<int, 128> bq[8][NH];
            if constexpr (NLS <= 1) {
#pragma unroll
                for (int s = 0; s < 8; ++s)
#pragma unroll
                    for (int ls = 0; ls < NLS; ++ls) {
                        const uint32_t* p = qbb + ((size_t) s * NLS + ls) * 128;
                        simd<uint32_t, 128> u;
                        u.template select<64, 1>(0) = block_load<uint32_t, 64>(p);
                        u.template select<64, 1>(64) = block_load<uint32_t, 64>(p + 64);
                        bq[s][ls] = u.template bit_cast_view<int>();
                    }
            }
            const int dbg = c.dbg;
            if (dbg & 1) return;
            for (int cls = 0; cls < 2; ++cls) {
                Range rg;
                class_range(rg, csk, csk[0] + 64 * cls, ch, n0, nt);
                if (rg.ub <= rg.a) continue;   // (the same in every thread of the group: the barriers below stay paired)
                const uint32_t slm0 = (uint32_t) (cls * kW * NLS * 3 * 64);
                simd<int, 16> lo[NLS], hi[NLS];
#pragma unroll
                for (int ls = 0; ls < NLS; ++ls)
#pragma unroll
                    for (int lane = 0; lane < 16; ++lane) {
                        lo[ls][lane] = rg.lok[(16 * ls + lane) / kG];
                        hi[ls][lane] = rg.hik[(16 * ls + lane) / kG];
                    }
                // pass 1: the scores of 8 cells x the rows, the rows' max and largest V scale
                simd<float, 16> mw[NLS], vw[NLS];
#pragma unroll
                for (int ls = 0; ls < NLS; ++ls) {
                    mw[ls] = kNeg;
                    vw[ls] = 0.f;
                }
                for (int p = rg.a + 8 * w; p < rg.ub; p += 8 * kW) {
                    simd<uint32_t, 64> kc[8];
#pragma unroll
                    for (int c8 = 0; c8 < 8; ++c8) kc[c8] = block_load<uint32_t, 64>(kb + (size_t) (p + c8) * 64);
                    simd<float, 8> vs8 = block_load<float, 8>(vsb + p);
                    simd<float, 8> ks8(1.0f);
                    simd<float, 32> kf(0.f);   // NS == 4: [cell * 4 + group]
                    if constexpr (NS == 1) {
                        ks8 = block_load<float, 8>(ksb + p);
                    } else {
                        simd<uint32_t, 16> w16 = block_load<uint32_t, 16>(ks16b + (size_t) p * 2);
                        simd<uint16_t, 32> u16 = w16.template bit_cast_view<uint16_t>();
                        simd<sycl::half, 32> h16 = u16.template bit_cast_view<sycl::half>();
                        kf = h16;
                    }
                    simd<int, 128> acc[NLS];
                    simd<float, 128> sf[NLS];
#pragma unroll
                    for (int ls = 0; ls < NLS; ++ls) {
                        acc[ls] = 0;
                        sf[ls] = 0.f;
                    }
                    if (!(dbg & 2))
#pragma unroll
                    for (int s = 0; s < 8; ++s) {
                        simd<int, 64> av;
#pragma unroll
                        for (int c8 = 0; c8 < 8; ++c8) {
                            simd<uint32_t, 8> wk = kc[c8].template select<8, 1>(8 * s);
                            av.template select<8, 1>(8 * c8) = wk.template bit_cast_view<int>();
                        }
                        if (dbg & 64) av = kc[s].template bit_cast_view<int>();
#pragma unroll
                        for (int ls = 0; ls < NLS; ++ls) {
                            simd<int, 128> b;
                            if constexpr (NLS <= 1) {
                                b = bq[s][ls];
                            } else {
                                const uint32_t* pq = qbb + ((size_t) s * NLS + ls) * 128;
                                simd<uint32_t, 128> u;
                                u.template select<64, 1>(0) = block_load<uint32_t, 64>(pq);
                                u.template select<64, 1>(64) = block_load<uint32_t, 64>(pq + 64);
                                b = u.template bit_cast_view<int>();
                            }
                            acc[ls] = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(acc[ls], b, av);
                        }
                        if constexpr (NS == 4) {
                            if (s % 2 == 1) {   // 64 values done: their dots times the cells' scale of this group
#pragma unroll
                                for (int ls = 0; ls < NLS; ++ls) {
#pragma unroll
                                    for (int c8 = 0; c8 < 8; ++c8) {
                                        simd<int, 16> ai = acc[ls].template select<16, 1>(16 * c8);
                                        simd<float, 16> t = ai;
                                        simd<float, 16> o = sf[ls].template select<16, 1>(16 * c8);
                                        const float kg = kf[c8 * 4 + s / 2];
                                        o += t * kg;
                                        sf[ls].template select<16, 1>(16 * c8) = o;
                                    }
                                    acc[ls] = 0;
                                }
                            }
                        }
                    }
#pragma unroll
                    for (int ls = 0; ls < NLS; ++ls) {
                        simd<float, 128> out;
#pragma unroll
                        for (int c8 = 0; c8 < 8; ++c8) {
                            simd<float, 16> sv;
                            const float kc1 = ks8[c8], vc1 = vs8[c8];
                            if constexpr (NS == 1) {
                                simd<int, 16> ai = acc[ls].template select<16, 1>(16 * c8);
                                sv = ai;
                                sv = sv * (qsc[ls] * kc1);
                            } else {
                                sv = sf[ls].template select<16, 1>(16 * c8);
                                sv = sv * qsc[ls];
                            }
                            simd<int, 16> pv(p + c8);
                            simd_mask<16> off = (lo[ls] > pv) | (hi[ls] <= pv);
                            if (has_tmask && (p + c8) >= p0_tree) {
                                const int u = (p + c8) - p0_tree;
                                const uint32_t bit = (uint32_t) u < 32u ? (1u << (uint32_t) u) : 0u;
                                off = off | ((tmk[ls] & bit) == 0u);
                            }
                            sv.merge(simd<float, 16>(kNeg), off);
                            mw[ls] = (es::max)(mw[ls], sv);
                            simd<float, 16> vv(vc1);
                            vv.merge(simd<float, 16>(0.f), off);
                            vw[ls] = (es::max)(vw[ls], vv);
                            out.template select<16, 1>(16 * c8) = sv;
                        }
                        float* o = scb + ((size_t) (p - rg.a) * NLS * 16) + ls * 128;
                        block_store<float, 64>(o, out.template select<64, 1>(0));
                        block_store<float, 64>(o + 64, out.template select<64, 1>(64));
                    }
                }
#pragma unroll
                for (int ls = 0; ls < NLS; ++ls) {
                    slm_block_store<float, 16>(slm0 + (uint32_t) ((w * NLS + ls) * 3) * 64u, mw[ls], kAl16);
                    slm_block_store<float, 16>(slm0 + (uint32_t) ((w * NLS + ls) * 3 + 1) * 64u, vw[ls], kAl16);
                }
                barrier();
#pragma unroll
                for (int ww = 0; ww < kW; ++ww)
#pragma unroll
                    for (int ls = 0; ls < NLS; ++ls) {
                        simd<float, 16> a1 = slm_block_load<float, 16>(slm0 + (uint32_t) ((ww * NLS + ls) * 3) * 64u, kAl16);
                        simd<float, 16> a2 = slm_block_load<float, 16>(slm0 + (uint32_t) ((ww * NLS + ls) * 3 + 1) * 64u, kAl16);
                        m[ls] = (es::max)(m[ls], a1);
                        vmx[ls] = (es::max)(vmx[ls], a2);
                    }
                // pass 2: p = exp(score - max) and its sum; w = p x V scale as 16 bits against the row's largest V scale
                // (a tile of 8 cells at once, 128 lanes: the steps of the exp are then 8 independent registers each; cell by
                // cell they are one chain and the thread waits on every step)
                simd<float, 128> i65[NLS], m128[NLS], l128[NLS];
#pragma unroll
                for (int ls = 0; ls < NLS; ++ls) {
                    simd<float, 16> iv = 65535.0f / (es::max)(vmx[ls], simd<float, 16>(1e-30f));
#pragma unroll
                    for (int c8 = 0; c8 < 8; ++c8) {
                        i65[ls].template select<16, 1>(16 * c8) = iv;
                        m128[ls].template select<16, 1>(16 * c8) = m[ls];
                    }
                    l128[ls] = 0.f;
                }
                uint32_t* pbb = c.pb + ((((size_t) cls * c.HKV + kvh) * nchk + ch) * (size_t) (c.capc / kBlk)) * NLS * 256;
                for (int p = rg.a + kBlk * w; p < rg.ub && !(dbg & 4); p += kBlk * kW) {
                    const int bi = (p - rg.a) >> 5;
                    simd<float, 32> vs32 = block_load<float, 32>(vsb + p);
                    simd<float, 128> vsr[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j)
#pragma unroll
                        for (int c8 = 0; c8 < 8; ++c8) {
                            const float vc1 = vs32[8 * j + c8];
                            vsr[j].template select<16, 1>(16 * c8) = simd<float, 16>(vc1);
                        }
#pragma unroll
                    for (int ls = 0; ls < NLS; ++ls) {
                        simd<uint32_t, 128> blo(0u), bhi(0u);
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            if (p + 8 * j < rg.ub) {
                                const float* o = scb + ((size_t) (p + 8 * j - rg.a) * NLS * 16) + ls * 128;
                                simd<float, 128> s128;
                                s128.template select<64, 1>(0) = block_load<float, 64>(o);
                                s128.template select<64, 1>(64) = block_load<float, 64>(o + 64);
                                simd<float, 128> x = (es::max)(s128 - m128[ls], simd<float, 128>(-87.0f));
                                simd<float, 128> pr = (dbg & 32) ? x : exp128(x);
                                pr.merge(simd<float, 128>(0.f), s128 < -1e29f);
                                l128[ls] += pr;
                                simd<float, 128> wq = pr * (i65[ls] * vsr[j]) + 0.5f;
                                wq = (es::min)(wq, simd<float, 128>(65535.0f));
                                simd<uint32_t, 128> qv = convert<uint32_t>(wq);
#pragma unroll
                                for (int hh = 0; hh < 2; ++hh) {
                                    simd<uint32_t, 16> q0 = qv.template select<16, 1>(16 * (4 * hh)), q1 = qv.template select<16, 1>(16 * (4 * hh + 1)),
                                                       q2 = qv.template select<16, 1>(16 * (4 * hh + 2)), q3 = qv.template select<16, 1>(16 * (4 * hh + 3));
                                    simd<uint32_t, 16> wl = (q0 & 255u) | ((q1 & 255u) << 8) | ((q2 & 255u) << 16) | (q3 << 24);
                                    simd<uint32_t, 16> wh = (q0 >> 8) | ((q1 >> 8) << 8) | ((q2 >> 8) << 16) | ((q3 >> 8) << 24);
                                    blo.template select<16, 1>(16 * (2 * j + hh)) = wl;
                                    bhi.template select<16, 1>(16 * (2 * j + hh)) = wh;
                                }
                            }
                        }
                        uint32_t* o = pbb + ((size_t) bi * NLS + ls) * 256;
                        if (dbg & 128) continue;
                        block_store<uint32_t, 64>(o, blo.template select<64, 1>(0));
                        block_store<uint32_t, 64>(o + 64, blo.template select<64, 1>(64));
                        block_store<uint32_t, 64>(o + 128, bhi.template select<64, 1>(0));
                        block_store<uint32_t, 64>(o + 192, bhi.template select<64, 1>(64));
                    }
                }
                // the thread's sum of each row: its 8 cell positions of a tile added in order
                simd<float, 16> lw[NLS];
#pragma unroll
                for (int ls = 0; ls < NLS; ++ls) {
                    lw[ls] = l128[ls].template select<16, 1>(0);
#pragma unroll
                    for (int c8 = 1; c8 < 8; ++c8) lw[ls] += l128[ls].template select<16, 1>(16 * c8);
                }
#pragma unroll
                for (int ls = 0; ls < NLS; ++ls) slm_block_store<float, 16>(slm0 + (uint32_t) ((w * NLS + ls) * 3 + 2) * 64u, lw[ls], kAl16);
                barrier();
                if (w == 0) {
#pragma unroll
                    for (int ww = 0; ww < kW; ++ww)
#pragma unroll
                        for (int ls = 0; ls < NLS; ++ls) {
                            simd<float, 16> a3 = slm_block_load<float, 16>(slm0 + (uint32_t) ((ww * NLS + ls) * 3 + 2) * 64u, kAl16);
                            l[ls] += a3;
                        }
                }
            }
            // the rows' max and sum of this chunk (k_attn_merge_gqa's), the scale of their value sums
            if (w == 0) {
#pragma unroll
                for (int ls = 0; ls < NLS; ++ls) {
                    simd<uint32_t, 16> tok, hd;
#pragma unroll
                    for (int lane = 0; lane < 16; ++lane) {
                        tok[lane] = (uint32_t) ((16 * ls + lane) / kG);
                        hd[lane] = (uint32_t) ((16 * ls + lane) % kG);
                    }
                    simd_mask<16> live = tok < (uint32_t) nt;
                    simd<uint32_t, 16> off = ((tok * (uint32_t) c.H + hd + (uint32_t) (kvh * kG)) * (uint32_t) nchk + (uint32_t) ch) * 4u;
                    scatter<float, 16>(c.pm, off, m[ls], live);
                    scatter<float, 16>(c.pl, off, l[ls], live);
                    simd<float, 16> srw = vmx[ls] * (1.0f / 65535.0f);
                    block_store<float, 16>(c.srow + ((size_t) (kvh * nchk + ch) * kMaxRows) + 16 * ls, srw);
                }
            }
        });
    });
}

// ---- value sums: thread (KV head, chunk, 32 dimensions j, pair of row sets pr)
template <int NLS> struct TagXaV;
template <int NLS> void launch_values(sycl::queue& q, const Call c) {
    constexpr int NP = (NLS + 1) / 2;
    const float inv = 1.0f / (float) c.nchk;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagXaV<NLS>>(sycl::nd_range<3>(sycl::range<3>((size_t) c.HKV, (size_t) c.nchk, (size_t) 8 * NP), sycl::range<3>(1, 1, 1)),
                                    [=](sycl::nd_item<3> it) SYCL_ESIMD_KERNEL {
            const int kvh = (int) it.get_global_id(0), ch = (int) it.get_global_id(1), x = (int) it.get_global_id(2);
            const int j = x & 7, pr = x >> 3;
            const int ls0 = 2 * pr, ls1 = ls0 + 1 < NLS ? ls0 + 1 : ls0;
            const bool two = ls0 + 1 < NLS;
            const int n0 = c.pos0[0] + c.t0 + 1, nt = c.nt, nchk = c.nchk;
            int csk[16];
#pragma unroll
            for (int k = 0; k < 16; ++k) csk[k] = k < nt ? chunk_cells(n0 + k, nchk, inv) : 0;
            const uint32_t* vb = c.v8t + ((size_t) kvh * (c.ctxp / kBlk) * 256 + 32 * j) * 8;
            simd<int, 128> alo[4][2], ahi[4][2];
#pragma unroll
            for (int ag = 0; ag < 4; ++ag)
#pragma unroll
                for (int u = 0; u < 2; ++u) {
                    alo[ag][u] = 0;
                    ahi[ag][u] = 0;
                }
            const int dbg = c.dbg;
            if (dbg & 8) return;
            for (int cls = 0; cls < 2; ++cls) {
                Range rg;
                class_range(rg, csk, csk[0] + 64 * cls, ch, n0, nt);
                if (rg.ub <= rg.a) continue;
                const uint32_t* pbb = c.pb + ((((size_t) cls * c.HKV + kvh) * nchk + ch) * (size_t) (c.capc / kBlk)) * NLS * 256;
                int bi = 0;
                for (int p = rg.a; p < rg.ub; p += kBlk, ++bi) {
                    const uint32_t* vp = vb + (size_t) (p >> 5) * 256 * 8;
                    simd<int, 64> av[4];
#pragma unroll
                    for (int ag = 0; ag < 4; ++ag) {
                        simd<uint32_t, 64> w = block_load<uint32_t, 64>(vp + 64 * ag);
                        av[ag] = w.template bit_cast_view<int>();
                    }
#pragma unroll
                    for (int u = 0; u < 2; ++u) {
                        if (u == 0 || two) {
                            const uint32_t* o = pbb + ((size_t) bi * NLS + (u == 0 ? ls0 : ls1)) * 256;
                            simd<uint32_t, 128> b;
                            b.template select<64, 1>(0) = block_load<uint32_t, 64>(o);
                            b.template select<64, 1>(64) = block_load<uint32_t, 64>(o + 64);
                            simd<int, 128> bl = b.template bit_cast_view<int>();
#pragma unroll
                            for (int ag = 0; ag < 4; ++ag)
                                alo[ag][u] = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::u8, xmx::dpas_argument_type::s8>(alo[ag][u], bl, av[ag]);
                            b.template select<64, 1>(0) = block_load<uint32_t, 64>(o + 128);
                            b.template select<64, 1>(64) = block_load<uint32_t, 64>(o + 192);
                            simd<int, 128> bh = b.template bit_cast_view<int>();
#pragma unroll
                            for (int ag = 0; ag < 4; ++ag)
                                ahi[ag][u] = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::u8, xmx::dpas_argument_type::s8>(ahi[ag][u], bh, av[ag]);
                        }
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                if (u == 0 || two) {
                    const int ls = u == 0 ? ls0 : ls1;
                    simd<float, 16> srw = block_load<float, 16>(c.srow + ((size_t) (kvh * nchk + ch) * kMaxRows) + 16 * ls);
                    simd<uint32_t, 16> lane(0u, 1u);
                    simd<uint32_t, 16> r = lane + (uint32_t) (16 * ls);
                    // row r is token r / kG, head r % kG: r / 6 as (r x 43691) >> 18 (exact below 2^17)
                    simd<uint32_t, 16> tok = (r * 43691u) >> 18;
                    simd<uint32_t, 16> hd = r - tok * (uint32_t) kG;
                    simd_mask<16> live = tok < (uint32_t) nt;
                    simd<uint32_t, 16> row = (tok * (uint32_t) c.H + hd + (uint32_t) (kvh * kG)) * (uint32_t) nchk + (uint32_t) ch;
#pragma unroll
                    for (int ag = 0; ag < 4; ++ag)
#pragma unroll
                        for (int r8 = 0; r8 < 8; ++r8) {
                            simd<int, 16> il = alo[ag][u].template select<16, 1>(16 * r8), ih = ahi[ag][u].template select<16, 1>(16 * r8);
                            simd<float, 16> fl = il, fh = ih;
                            simd<float, 16> f = (fh * 256.0f + fl) * srw;
                            simd<uint32_t, 16> off = (row * 256u + (uint32_t) (32 * j + 8 * ag + r8)) * 4u;
                            if (!(dbg & 16)) scatter<float, 16>(c.pacc, off, f, live);
                        }
                }
            }
        });
    });
}

// STRATA_XATT_PARTS (a bench switch): the kernels a call runs, a sum of 1 (queries), 2 (scores), 4 (values); default 7
template <int NLS> void run(sycl::queue& q, const Call& c) {
    static const int parts = [] { const char* e = std::getenv("STRATA_XATT_PARTS"); return e && *e ? std::atoi(e) : 7; }();
    if (parts & 1) launch_queries<NLS>(q, c);
    if (parts & 2) {
        if (c.ns == 4) launch_scores<NLS, 4>(q, c);
        else launch_scores<NLS, 1>(q, c);
    }
    if (parts & 4) launch_values<NLS>(q, c);
}
}   // namespace

int chunk_cap(int ctx, int nchk) { return chunk_cells_host(ctx + 64, nchk) + 64; }

bool attend(const Call& c0, void* stream) {
    static const int dbg = [] { const char* e = std::getenv("STRATA_XATT_DBG"); return e && *e ? std::atoi(e) : 0; }();
    Call c = c0;
    c.dbg = dbg;
    if (c.nt < 1 || c.nt > 16 || c.H != kG * c.HKV || c.nchk < 1 || (c.ns != 1 && c.ns != 4) || (c.nt * kG + 15) / 16 > c.max_ls) return false;
    sycl::queue& q = *strata::q_of(stream);
    switch ((c.nt * kG + 15) / 16) {
        case 1: run<1>(q, c); break;
        case 2: run<2>(q, c); break;
        case 3: run<3>(q, c); break;
        case 4: run<4>(q, c); break;
        case 5: run<5>(q, c); break;
        case 6: run<6>(q, c); break;
        default: return false;
    }
    return true;
}

// =================================================================================================
// XATT2 (STRATA_Q36_XATT2=1): single-pass online-softmax ESIMD+DPAS split-K attention + ESIMD merge
// =================================================================================================
namespace {

constexpr int kFormS1 = 0;
constexpr int kFormD2 = 1;
constexpr int kFormS8 = 2;
constexpr int kFormL8 = 3;

SYCL_ESIMD_FUNCTION inline simd<float, 64> exp64(simd<float, 64> x) {
    simd<float, 64> y = x * 1.44269504f;
    simd<int, 64> n = convert<int>(y - 0.5f);
    simd<float, 64> nf = n;
    simd<float, 64> z = (y - nf) * 0.693147181f;
    simd<float, 64> e = 1.0f + z * (1.0f + z * (0.5f + z * (0.16666667f + z * (0.041666668f + z * (0.008333334f + z * 0.0013888889f)))));
    simd<int, 64> b = (n + 127) << 23;
    simd<float, 64> s = b.template bit_cast_view<float>();
    return e * s;
}

SYCL_ESIMD_FUNCTION inline float hsum16_f(simd<float, 16> v) {
    simd<float, 8> a = v.template select<8, 1>(0) + v.template select<8, 1>(8);
    simd<float, 4> b = a.template select<4, 1>(0) + a.template select<4, 1>(4);
    simd<float, 2> c = b.template select<2, 1>(0) + b.template select<2, 1>(2);
    return float(c[0]) + float(c[1]);
}

SYCL_ESIMD_FUNCTION inline simd<float, 64> sigm64(simd<float, 64> x) {
    simd<float, 64> ax = -es::abs(x);
    ax = (es::max)(ax, simd<float, 64>(-87.0f));
    simd<float, 64> ex = exp64(ax);
    simd<float, 64> sp = 1.0f / (1.0f + ex);
    simd<float, 64> sn = ex * sp;
    sp.merge(sn, x < 0.0f);
    return sp;
}

struct AttnGrp2 {
    uint8_t slot = 0;
    uint8_t r0 = 0;
    uint8_t cnt = 0;
    uint8_t ls0 = 0;
    uint8_t nrg = 1;
    uint8_t nseg = 1;
    uint16_t pad = 0;
};

struct QPlan2 {
    int32_t ngrp = 1;
    int32_t nt = 1;
    int32_t t0 = 0;
    int32_t H = 24;
    int32_t HKV = 4;
    int32_t per_tok_pos = 0;
    float scale = 1.0f / 16.0f;
    const float* q = nullptr;
    uint32_t* qb = nullptr;
    float* qs = nullptr;
    uint8_t g_r0[kMaxBatchGroups2]{};
    uint8_t g_cnt[kMaxBatchGroups2]{};
    uint8_t ls_grp[kMaxBatchGroups2]{};
    uint8_t ls_loc[kMaxBatchGroups2]{};
};

struct AttnPlan2 {
    const uint32_t* k8 = nullptr;
    const float* ks = nullptr;
    const uint16_t* ks16 = nullptr;
    const uint32_t* v8t = nullptr;
    const float* vs = nullptr;
    const int32_t* pos0 = nullptr;
    const uint32_t* tmask = nullptr;
    const int32_t* tp0 = nullptr;
    void* const* k_tab = nullptr;
    void* const* v_tab = nullptr;
    const void* const* mir_tab = nullptr;
    uint32_t* qb = nullptr;
    float* qs = nullptr;
    float* pacc = nullptr;
    float* pm = nullptr;
    float* pl = nullptr;
    int32_t kst = 0;
    int32_t ctxp = 0;
    int32_t wg_threads = 1;
    int32_t pfd = 2;
    int32_t zero_fence = 0;
    int32_t nseg_stride = 1;
    int32_t t0 = 0;
    int32_t H = 24;
    int32_t HKV = 4;
    float scale = 1.0f / 16.0f;
    uint8_t ngrp = 1;
    uint8_t total_tasks = 1;
    uint8_t per_tok_pos = 0;
    uint8_t pad = 0;
    AttnGrp2 g[kMaxBatchGroups2]{};
    uint8_t task_grp[kMaxSegs2]{};
    uint8_t task_seg[kMaxSegs2]{};
};

struct MPlan2 {
    int32_t nt = 1;
    int32_t H = 24;
    int32_t ngrp = 1;
    int32_t nseg_stride = 1;
    float* pacc = nullptr;
    float* pm = nullptr;
    float* pl = nullptr;
    const float* gate = nullptr;
    float* out = nullptr;
    uint8_t g_nseg[kMaxBatchGroups2]{};
    uint8_t tok_grp[kMaxBatchGroups2]{};
};

struct HostPlan2 {
    QPlan2 q{};
    AttnPlan2 a{};
    MPlan2 m{};
    int total_ls = 1;
};

template <int NS>
SYCL_ESIMD_FUNCTION inline void load_grp_ptrs(
    const AttnPlan2& pl, int gi, int kvh,
    const uint32_t*& kb, const float*& ksb, const uint32_t*& ks16b,
    const uint32_t*& vb, const float*& vsb) {
    const uint32_t* k8 = pl.k8;
    const float* ks = pl.ks;
    const uint16_t* ks16 = pl.ks16;
    const uint32_t* v8t = pl.v8t;
    const float* vs = pl.vs;
    if (!k8) {
        const int slot = (int) pl.g[gi].slot;
        if (pl.k_tab) {
            const int8_t* kp = reinterpret_cast<const int8_t*>(pl.k_tab[slot]);
            k8 = reinterpret_cast<const uint32_t*>(kp);
            ks16 = reinterpret_cast<const uint16_t*>(kp + (size_t) pl.HKV * (size_t) pl.kst * kHD);
        } else if (pl.mir_tab) {
            const size_t b = (size_t) slot * 4u;
            k8 = reinterpret_cast<const uint32_t*>(pl.mir_tab[b + 0]);
            if constexpr (NS == 1) ks = reinterpret_cast<const float*>(pl.mir_tab[b + 1]);
            else ks16 = reinterpret_cast<const uint16_t*>(pl.mir_tab[b + 1]);
        }
        if (pl.v_tab) {
            const int8_t* vp = reinterpret_cast<const int8_t*>(pl.v_tab[slot]);
            v8t = reinterpret_cast<const uint32_t*>(vp);
            vs = reinterpret_cast<const float*>(vp + v8t_dwords(pl.HKV, pl.ctxp) * 4);
        } else if (pl.mir_tab) {
            const size_t b = (size_t) slot * 4u;
            v8t = reinterpret_cast<const uint32_t*>(pl.mir_tab[b + 2]);
            vs = reinterpret_cast<const float*>(pl.mir_tab[b + 3]);
        }
    }
    kb = k8 + (size_t) kvh * pl.kst * 64;
    ksb = (NS == 1 && ks) ? (ks + (size_t) kvh * pl.kst) : nullptr;
    ks16b = (NS == 4 && ks16) ? (reinterpret_cast<const uint32_t*>(ks16) + (size_t) kvh * pl.kst * 2) : nullptr;
    vb = v8t + (size_t) kvh * (pl.ctxp / kBlk) * 256 * 8;
    vsb = vs + (size_t) kvh * pl.ctxp;
}

SYCL_ESIMD_FUNCTION inline void seg_block_range(int ub, int seg, int nseg, int& blk_a, int& blk_b) {
    const int nblk = (ub + kBlk - 1) >> 5;
    if (nblk <= 0 || seg >= nseg) {
        blk_a = 0;
        blk_b = 0;
        return;
    }
    if (nseg <= 1) {
        blk_a = 0;
        blk_b = nblk;
        return;
    }
    const float inv = 1.0f / (float) nseg;
    const int v = nblk + nseg - 1;
    int per = (int) ((float) v * inv);
    if (per * nseg > v) --per;
    if ((per + 1) * nseg <= v) ++per;
    if (per < 1) per = 1;
    blk_a = seg * per;
    if (blk_a >= nblk) {
        blk_a = 0;
        blk_b = 0;
        return;
    }
    const int eb = blk_a + per;
    blk_b = eb < nblk ? eb : nblk;
}

// Quantize queries into 16-row DPAS operand format per (kvh, ls_global), zeroing unused rows.
struct TagXa2Q;
void launch_queries2(sycl::queue& q, const QPlan2& pl, int total_ls) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagXa2Q>(
            sycl::nd_range<3>(sycl::range<3>((size_t) pl.HKV, (size_t) total_ls, 16),
                              sycl::range<3>(1, 1, 16)),
            [=](sycl::nd_item<3> it) SYCL_ESIMD_KERNEL {
                slm_init<16u * 64u * 4u>();
                const int kvh = (int) it.get_global_id(0);
                const int lsg = (int) it.get_global_id(1);
                const int lane = (int) it.get_local_id(2);
                const int gi = (int) pl.ls_grp[lsg];
                const int ls_loc = (int) pl.ls_loc[lsg];
                const int gp_r0 = (int) pl.g_r0[gi];
                const int gp_cnt = (int) pl.g_cnt[gi];
                const int r = ls_loc * 16 + lane;
                const int k = (r * 43691) >> 18;
                const int hd = r - k * kG;
                const bool live = k < gp_cnt;

                simd<uint32_t, 64> w(0u);
                float qsv = 0.0f;
                if (live) {
                    const int qt = pl.per_tok_pos ? (gp_r0 + k) : (pl.t0 + k);
                    const float* src = pl.q + ((size_t) qt * pl.H + kvh * kG + hd) * kHD;
                    simd<float, 64> x[4];
                    float mx = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        x[i] = block_load<float, 64>(src + 64 * i);
                        const float m1 = hmax<float>(es::abs(x[i]));
                        mx = m1 > mx ? m1 : mx;
                    }
                    const float sc = mx / 127.0f;
                    const float inv = sc > 0.0f ? 1.0f / sc : 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        simd<float, 64> y = x[i] * inv;
                        simd<float, 64> half(0.5f);
                        half.merge(simd<float, 64>(-0.5f), y < 0.0f);
                        simd<int, 64> ci = convert<int>(y + half);
                        ci = (es::min)((es::max)(ci, simd<int, 64>(-127)), simd<int, 64>(127));
                        simd<uint32_t, 64> cu = ci.template bit_cast_view<uint32_t>();
                        simd<uint32_t, 16> b0 = cu.template select<16, 4>(0);
                        simd<uint32_t, 16> b1 = cu.template select<16, 4>(1);
                        simd<uint32_t, 16> b2 = cu.template select<16, 4>(2);
                        simd<uint32_t, 16> b3 = cu.template select<16, 4>(3);
                        w.template select<16, 1>(16 * i) = (b0 & 255u) | ((b1 & 255u) << 8) | ((b2 & 255u) << 16) | (b3 << 24);
                    }
                    qsv = sc * pl.scale;
                }
                slm_block_store<uint32_t, 64>((uint32_t) lane * 256u, w, kAl16);
                pl.qs[((size_t) kvh * kMaxBatchGroups2 + (size_t) lsg) * 16u + (size_t) lane] = qsv;
                barrier();

                // Thread `lane` handles slice `s = lane >> 1`, 4 dwords `d0 = (lane & 1) * 4`, 16 lanes = 64 dwords.
                const int s = lane >> 1;
                const int d0 = (lane & 1) * 4;
                simd<uint32_t, 64> out64;
                const simd<uint32_t, 16> row_off(0u, 256u);
#pragma unroll
                for (int dd = 0; dd < 4; ++dd) {
                    const uint32_t col_off = (uint32_t) (8 * s + d0 + dd) * 4u;
                    out64.template select<16, 1>(16 * dd) = slm_gather<uint32_t, 16>(row_off + col_off);
                }
                uint32_t* dst = pl.qb + (((size_t) kvh * 8u + (size_t) s) * kMaxBatchGroups2 + (size_t) lsg) * 128u + (size_t) d0 * 16u;
                block_store<uint32_t, 64>(dst, out64);
            });
    });
}

// Helper: compute valid-cell upper bound and per-row [0, hi) lengths for 8-row or 16-row groups.
template <int R>
SYCL_ESIMD_FUNCTION inline void load_row_bounds(
    const AttnPlan2& pl, int gi, int r_base,
    simd<int, R>& hi_v, simd<uint32_t, R>& tmk_v, int& ub, bool& has_tmask, int& p0_tree) {
    const AttnGrp2& gp = pl.g[gi];
    int tok_len[16];
    ub = 0;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        if (k < (int) gp.cnt) {
            const int len = pl.per_tok_pos ? (pl.pos0[(int) gp.r0 + k] + 1) : (pl.pos0[0] + pl.t0 + 1 + k);
            tok_len[k] = len;
            if (len > ub) ub = len;
        } else {
            tok_len[k] = 0;
        }
    }
    has_tmask = (pl.tmask != nullptr) && (pl.ngrp == 1);
    p0_tree = has_tmask ? (pl.tp0 ? pl.tp0[0] : pl.pos0[0]) : 0x3fffffff;
#pragma unroll
    for (int lane = 0; lane < R; ++lane) {
        const int r = r_base + lane;
        const int tok = (r * 43691) >> 18;
        if (tok < (int) gp.cnt) {
            hi_v[lane] = tok_len[tok];
            tmk_v[lane] = has_tmask ? pl.tmask[pl.t0 + tok] : 0xffffffffu;
        } else {
            hi_v[lane] = 0;
            tmk_v[lane] = 0u;
        }
    }
}

// Helper: store 8-row outputs (o[32] of [ag=0..31][dim8][row8]) + pm/pl into split-K partials.
SYCL_ESIMD_FUNCTION inline void store_partials_r8(
    const AttnPlan2& pl, int gi, int seg, int kvh, int r_base,
    simd<float, 64> o[32], simd<float, 8> m, simd<float, 8> l) {
    const AttnGrp2& gp = pl.g[gi];
    const int nseg_s = pl.nseg_stride;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int rg = r_base + r;
        const int tok = (rg * 43691) >> 18;
        if (tok < (int) gp.cnt) {
            const int hd = rg - tok * kG;
            const size_t row = (size_t) ((int) gp.r0 + tok) * (size_t) pl.H + (size_t) (kvh * kG + hd);
            const size_t p_idx = row * (size_t) nseg_s + (size_t) seg;
            pl.pm[p_idx] = m[r];
            pl.pl[p_idx] = l[r];
            float* pout = pl.pacc + p_idx * (size_t) kHD;
#pragma unroll
            for (int c4 = 0; c4 < 4; ++c4) {
                simd<float, 64> v64;
#pragma unroll
                for (int u = 0; u < 8; ++u) v64.template select<8, 1>(8 * u) = o[8 * c4 + u].template select<8, 8>(r);
                block_store<float, 64>(pout + 64 * c4, v64);
            }
        }
    }
}

// Shared online-softmax + V-quantization + 8-row value accumulation for 1 block (32 cells).
// Uses runtime fences inside unrolled loops so sc[j]/o[ag] keep constant GRF indices while preventing load hoisting.
template <int U16, bool FROM_SLM = false>
SYCL_ESIMD_FUNCTION inline void softmax_v_r8_blk(
    simd<float, 64> sc[4], simd<float, 8> m_blk, simd<float, 32> vs32,
    const uint32_t* vb, size_t v_blk_off, int zero_fence, float* fence_ptr,
    simd<float, 64> o[32], simd<float, 8>& m, simd<float, 8>& l) {
    if (hmax<float>(m_blk) <= -1e29f) return;

    simd<float, 8> m_new = (es::max)(m, m_blk);
    simd<float, 16> mdiff16(0.0f);
    mdiff16.template select<8, 1>(0) = (es::max)(m - m_new, simd<float, 8>(-87.0f));
    simd<float, 8> alpha = exp16(mdiff16).template select<8, 1>(0);
    alpha.merge(simd<float, 8>(0.0f), m <= -1e29f);

    // Pass 1: find per-row max valid V scale (avoids holding pr[4]).
    simd<float, 8> vmx(0.0f);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int c8 = 0; c8 < 8; ++c8) {
            simd<float, 8> vv(vs32[8 * j + c8]);
            vv.merge(simd<float, 8>(0.0f), sc[j].template select<8, 1>(8 * c8) < -1e29f);
            vmx = (es::max)(vmx, vv);
        }
    }

    const float max_q = U16 ? 65535.0f : 255.0f;
    const simd<float, 8> inv_v = max_q / (es::max)(vmx, simd<float, 8>(1e-30f));

    // Pass 2: compute exp64 and quantize into DPAS operand blo/bhi one 8-cell tile at a time.
    simd<float, 8> l_blk(0.0f);
    simd<uint32_t, 128> blo(0u), bhi(0u);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        simd<float, 64> x;
#pragma unroll
        for (int c8 = 0; c8 < 8; ++c8) x.template select<8, 1>(8 * c8) = sc[j].template select<8, 1>(8 * c8) - m_new;
        x = (es::max)(x, simd<float, 64>(-87.0f));
        simd<float, 64> e = exp64(x);
        e.merge(simd<float, 64>(0.0f), sc[j] < -1e29f);
        simd<float, 64> wq;
#pragma unroll
        for (int c8 = 0; c8 < 8; ++c8) {
            simd<float, 8> er = e.template select<8, 1>(8 * c8);
            l_blk += er;
            wq.template select<8, 1>(8 * c8) = er * (inv_v * float(vs32[8 * j + c8])) + 0.5f;
        }
        wq = (es::min)(wq, simd<float, 64>(max_q));
        simd<uint32_t, 64> qv = convert<uint32_t>(wq);
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            simd<uint32_t, 8> q0 = qv.template select<8, 1>(8 * (4 * hh));
            simd<uint32_t, 8> q1 = qv.template select<8, 1>(8 * (4 * hh + 1));
            simd<uint32_t, 8> q2 = qv.template select<8, 1>(8 * (4 * hh + 2));
            simd<uint32_t, 8> q3 = qv.template select<8, 1>(8 * (4 * hh + 3));
            blo.template select<8, 1>(16 * (2 * j + hh)) = (q0 & 255u) | ((q1 & 255u) << 8) | ((q2 & 255u) << 16) | (q3 << 24);
            if constexpr (U16) {
                bhi.template select<8, 1>(16 * (2 * j + hh)) = (q0 >> 8) | ((q1 >> 8) << 8) | ((q2 >> 8) << 16) | ((q3 >> 8) << 24);
            }
        }
        if (zero_fence == -10 - j) fence_ptr[0] = 0.0f;
    }
    l = l * alpha + l_blk;
    m = m_new;

    simd<float, 64> alpha64;
#pragma unroll
    for (int u = 0; u < 8; ++u) alpha64.template select<8, 1>(8 * u) = alpha;

    const simd<float, 8> srw = vmx * (1.0f / max_q);
    simd<float, 64> srw64;
#pragma unroll
    for (int u = 0; u < 8; ++u) srw64.template select<8, 1>(8 * u) = srw;

    const uint32_t* vp = vb + v_blk_off + (size_t) zero_fence;
    const uint32_t v_slm_base = (uint32_t) v_blk_off;
    simd<int, 128> bl = blo.template bit_cast_view<int>();
    simd<int, 128> bh = bhi.template bit_cast_view<int>();
#pragma unroll
    for (int ag = 0; ag < 32; ++ag) {
        simd<uint32_t, 64> wv = FROM_SLM
            ? slm_block_load<uint32_t, 64>(v_slm_base + (uint32_t) ag * 256u, kAl16)
            : block_load<uint32_t, 64>(vp + (size_t) ag * 64u);
        simd<int, 64> av = wv.template bit_cast_view<int>();
        simd<int, 128> alo = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::u8, xmx::dpas_argument_type::s8>(simd<int, 128>(0), bl, av);
        simd<int, 64> lo64;
#pragma unroll
        for (int r8 = 0; r8 < 8; ++r8) lo64.template select<8, 1>(8 * r8) = alo.template select<8, 1>(16 * r8);
        simd<float, 64> fl = lo64;
        if constexpr (U16) {
            simd<int, 128> ahi = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::u8, xmx::dpas_argument_type::s8>(simd<int, 128>(0), bh, av);
            simd<int, 64> hi64;
#pragma unroll
            for (int r8 = 0; r8 < 8; ++r8) hi64.template select<8, 1>(8 * r8) = ahi.template select<8, 1>(16 * r8);
            simd<float, 64> fh = hi64;
            o[ag] = o[ag] * alpha64 + (fh * 256.0f + fl) * srw64;
        } else {
            o[ag] = o[ag] * alpha64 + fl * srw64;
        }
        if ((ag & 1) == 1 && zero_fence == -1 - ag) fence_ptr[0] = 0.0f;
    }
}

// Compute 32 cells x 8 rows score tile directly from DRAM K (used by S1 and L8).
// Streams Q slice from L1 (`qbb`) and fences j/s loops to eliminate all GRF spills.
template <int NS>
SYCL_ESIMD_FUNCTION inline void score_dram_r8_blk(
    const uint32_t* qbb, int half_off, simd<float, 8> qsc,
    const uint32_t* kb, const float* ksb, const uint32_t* ks16b,
    int p0, int ub, bool has_tmask, int p0_tree,
    simd<int, 8> hi_v, simd<uint32_t, 8> tmk_v, int zero_fence, float* fence_ptr,
    simd<float, 64> sc[4], simd<float, 8>& m_blk) {
    const bool full_blk = (p0 + kBlk <= ub);
    m_blk = kNeg;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int p = p0 + 8 * j;
        simd<uint32_t, 64> kc[8];
        simd<float, 8> ks8(1.0f);
        simd<float, 32> kf(0.0f);
        if (full_blk) {
#pragma unroll
            for (int c8 = 0; c8 < 8; ++c8) kc[c8] = block_load<uint32_t, 64>(kb + (size_t) (p + c8) * 64 + (size_t) zero_fence);
            if constexpr (NS == 1) {
                ks8 = block_load<float, 8>(ksb + p);
            } else {
                simd<uint32_t, 16> w16 = block_load<uint32_t, 16>(ks16b + (size_t) p * 2);
                simd<uint16_t, 32> u16 = w16.template bit_cast_view<uint16_t>();
                simd<sycl::half, 32> h16 = u16.template bit_cast_view<sycl::half>();
                kf = h16;
            }
        } else {
#pragma unroll
            for (int c8 = 0; c8 < 8; ++c8) {
                if (p + c8 < ub) {
                    kc[c8] = block_load<uint32_t, 64>(kb + (size_t) (p + c8) * 64 + (size_t) zero_fence);
                    if constexpr (NS == 1) {
                        ks8[c8] = ksb[p + c8];
                    } else {
                        simd<uint32_t, 2> w2 = block_load<uint32_t, 2>(ks16b + (size_t) (p + c8) * 2);
                        simd<uint16_t, 4> u4 = w2.template bit_cast_view<uint16_t>();
                        simd<sycl::half, 4> h4 = u4.template bit_cast_view<sycl::half>();
                        kf.template select<4, 1>(4 * c8) = h4;
                    }
                } else {
                    kc[c8] = 0u;
                }
            }
        }

        simd<int, 128> acc(0);
        simd<float, 64> sf(0.0f);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            simd<int, 64> av;
#pragma unroll
            for (int c8 = 0; c8 < 8; ++c8) {
                av.template select<8, 1>(8 * c8) = kc[c8].template select<8, 1>(8 * s).template bit_cast_view<int>();
            }
            const uint32_t* qp = qbb + (size_t) s * kMaxBatchGroups2 * 128u;
            simd<uint32_t, 128> u;
            u.template select<64, 1>(0) = block_load<uint32_t, 64>(qp);
            u.template select<64, 1>(64) = block_load<uint32_t, 64>(qp + 64);
            simd<int, 128> b(0);
            if (half_off == 0) {
                b = u.template bit_cast_view<int>();
            } else {
#pragma unroll
                for (int d = 0; d < 8; ++d)
                    b.template select<8, 1>(16 * d) = u.template select<8, 1>(16 * d + 8).template bit_cast_view<int>();
            }
            acc = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(acc, b, av);
            if constexpr (NS == 4) {
                if ((s & 1) == 1) {
#pragma unroll
                    for (int c8 = 0; c8 < 8; ++c8) {
                        simd<int, 8> ai = acc.template select<8, 1>(16 * c8);
                        simd<float, 8> t = ai;
                        const float kg = kf[c8 * 4 + (s >> 1)];
                        sf.template select<8, 1>(8 * c8) += t * kg;
                    }
                    acc = 0;
                }
            }
            if ((s & 1) == 1 && zero_fence == -30 - (j * 8 + s)) fence_ptr[0] = 0.0f;
        }
#pragma unroll
        for (int c8 = 0; c8 < 8; ++c8) {
            simd<float, 8> sv;
            if constexpr (NS == 1) {
                simd<int, 8> ai = acc.template select<8, 1>(16 * c8);
                sv = ai;
                const float kc1 = ks8[c8];
                sv = sv * (qsc * kc1);
            } else {
                sv = sf.template select<8, 1>(8 * c8);
                sv = sv * qsc;
            }
            simd_mask<8> off = (hi_v <= (p + c8));
            if (has_tmask && (p + c8) >= p0_tree) {
                const int u = (p + c8) - p0_tree;
                const uint32_t bit = (uint32_t) u < 32u ? (1u << (uint32_t) u) : 0u;
                off = off | ((tmk_v & bit) == 0u);
            }
            sv.merge(simd<float, 8>(kNeg), off);
            m_blk = (es::max)(m_blk, sv);
            sc[j].template select<8, 1>(8 * c8) = sv;
        }
        if (zero_fence == -20 - j) fence_ptr[0] = 0.0f;
    }
}

SYCL_ESIMD_FUNCTION inline void prefetch_blk_kv(
    const uint32_t* kb, const uint32_t* vb, int blk, int ub) {
    const int p0 = blk * kBlk;
    if (p0 + kBlk > ub) return;
#pragma unroll
    for (int c = 0; c < 32; ++c) es::prefetch<uint32_t, 64>(kb + (size_t) (p0 + c) * 64u, kPf);
#pragma unroll
    for (int c = 0; c < 32; ++c) es::prefetch<uint32_t, 64>(vb + (size_t) blk * 2048u + (size_t) c * 64u, kPf);
}

// Form 0: S1 (<= 8 rows per group, 1 thread per WG, no SLM/barriers).
template <int NS, int U16> struct TagXa2S1;
template <int NS, int U16> void launch_s1(sycl::queue& q, const AttnPlan2& pl) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagXa2S1<NS, U16>>(
            sycl::nd_range<3>(sycl::range<3>(1, (size_t) pl.HKV, (size_t) pl.total_tasks),
                              sycl::range<3>(1, 1, 1)),
            [=](sycl::nd_item<3> it) SYCL_ESIMD_KERNEL {
                const int kvh = (int) it.get_global_id(1);
                const int task = (int) it.get_global_id(2);
                const int gi = (int) pl.task_grp[task];
                const int seg = (int) pl.task_seg[task];
                const AttnGrp2& gp = pl.g[gi];

                const uint32_t* kb = nullptr;
                const float* ksb = nullptr;
                const uint32_t* ks16b = nullptr;
                const uint32_t* vb = nullptr;
                const float* vsb = nullptr;
                load_grp_ptrs<NS>(pl, gi, kvh, kb, ksb, ks16b, vb, vsb);

                simd<int, 8> hi_v;
                simd<uint32_t, 8> tmk_v;
                int ub = 0, p0_tree = 0;
                bool has_tmask = false;
                load_row_bounds<8>(pl, gi, 0, hi_v, tmk_v, ub, has_tmask, p0_tree);

                int blk_a = 0, blk_b = 0;
                seg_block_range(ub, seg, (int) gp.nseg, blk_a, blk_b);

                simd<float, 64> o[32];
#pragma unroll
                for (int k = 0; k < 32; ++k) o[k] = 0.0f;
                simd<float, 8> m(kNeg), l(0.0f);

                if (blk_a < blk_b) {
                    const uint32_t* qbb = pl.qb + ((size_t) kvh * 8u) * kMaxBatchGroups2 * 128u + (size_t) gp.ls0 * 128u;
                    const simd<float, 8> qsc = block_load<float, 8>(pl.qs + ((size_t) kvh * kMaxBatchGroups2 + (size_t) gp.ls0) * 16u);

                    for (int pb = 0; pb < pl.pfd && blk_a + pb < blk_b; ++pb) prefetch_blk_kv(kb, vb, blk_a + pb, ub);

                    for (int blk = blk_a; blk < blk_b; ++blk) {
                        if (blk + pl.pfd < blk_b) prefetch_blk_kv(kb, vb, blk + pl.pfd, ub);
                        const int p0 = blk * kBlk;
                        simd<float, 64> sc[4];
                        simd<float, 8> m_blk;
                        score_dram_r8_blk<NS>(qbb, 0, qsc, kb, ksb, ks16b, p0, ub, has_tmask, p0_tree, hi_v, tmk_v, pl.zero_fence, pl.pl, sc, m_blk);
                        const simd<float, 32> vs32 = block_load<float, 32>(vsb + p0);
                        softmax_v_r8_blk<U16, false>(sc, m_blk, vs32, vb, (size_t) blk * 2048u, pl.zero_fence, pl.pl, o, m, l);
                    }
                }
                store_partials_r8(pl, gi, seg, kvh, 0, o, m, l);
            });
    });
}

// Form 1: D2 (TD=2 D-split of 128 dims/thread, 16 rows per pair, NG pairs -> W = 2*NG threads/WG).
template <int NG, int NS, int U16> struct TagXa2D2;
template <int NG, int NS, int U16> void launch_d2(sycl::queue& q, const AttnPlan2& pl) {
    constexpr int W = 2 * NG;
    constexpr uint32_t kHalfKv = 16384u;                  // 8 KiB K + 8 KiB V
    constexpr uint32_t kSlmScOff = 2u * kHalfKv;         // 32768 B
    constexpr uint32_t kSlmBytes = kSlmScOff + (uint32_t) NG * 4096u;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagXa2D2<NG, NS, U16>>(
            sycl::nd_range<3>(sycl::range<3>(1, (size_t) pl.HKV, (size_t) pl.total_tasks * W),
                              sycl::range<3>(1, 1, (size_t) W)),
            [=](sycl::nd_item<3> it) SYCL_ESIMD_KERNEL {
                slm_init<kSlmBytes>();
                const int kvh = (int) it.get_global_id(1);
                const int task = (int) it.get_group(2);
                const int tid = (int) it.get_local_id(2);
                const int ls = tid >> 1;
                const int td = tid & 1;

                const int gi = (int) pl.task_grp[task];
                const int seg = (int) pl.task_seg[task];
                const AttnGrp2& gp = pl.g[gi];
                const bool active_ls = (ls < (int) gp.nrg);

                const uint32_t* kb = nullptr;
                const float* ksb = nullptr;
                const uint32_t* ks16b = nullptr;
                const uint32_t* vb = nullptr;
                const float* vsb = nullptr;
                load_grp_ptrs<NS>(pl, gi, kvh, kb, ksb, ks16b, vb, vsb);

                simd<int, 16> hi_v;
                simd<uint32_t, 16> tmk_v;
                int ub = 0, p0_tree = 0;
                bool has_tmask = false;
                load_row_bounds<16>(pl, gi, ls * 16, hi_v, tmk_v, ub, has_tmask, p0_tree);

                int blk_a = 0, blk_b = 0;
                seg_block_range(ub, seg, (int) gp.nseg, blk_a, blk_b);

                simd<float, 128> o[16];
#pragma unroll
                for (int k = 0; k < 16; ++k) o[k] = 0.0f;
                simd<float, 16> m(kNeg), l(0.0f);

                if (blk_a < blk_b) {
                    const uint32_t* qbb = pl.qb + ((size_t) kvh * 8u) * kMaxBatchGroups2 * 128u + (size_t) ((int) gp.ls0 + ls) * 128u;
                    simd<int, 128> bq[4];
                    if (active_ls) {
#pragma unroll
                        for (int sl = 0; sl < 4; ++sl) {
                            const int s = td * 4 + sl;
                            const uint32_t* p = qbb + (size_t) s * kMaxBatchGroups2 * 128u;
                            simd<uint32_t, 128> u;
                            u.template select<64, 1>(0) = block_load<uint32_t, 64>(p);
                            u.template select<64, 1>(64) = block_load<uint32_t, 64>(p + 64);
                            bq[sl] = u.template bit_cast_view<int>();
                        }
                    }
                    const simd<float, 16> qsc = active_ls
                        ? block_load<float, 16>(pl.qs + ((size_t) kvh * kMaxBatchGroups2 + (size_t) ((int) gp.ls0 + ls)) * 16u)
                        : simd<float, 16>(0.0f);

                    for (int pb = 0; pb < pl.pfd && blk_a + pb < blk_b; ++pb) {
                        const int p_blk = blk_a + pb;
                        const int pf0 = p_blk * kBlk;
                        if (pf0 + kBlk <= ub) {
                            for (int c = tid; c < 32; c += W) {
                                es::prefetch<uint32_t, 64>(kb + (size_t) (pf0 + c) * 64u, kPf);
                                es::prefetch<uint32_t, 64>(vb + (size_t) p_blk * 2048u + (size_t) c * 64u, kPf);
                            }
                        }
                    }

                    for (int blk = blk_a; blk < blk_b; ++blk) {
                        const int p0 = blk * kBlk;
                        const uint32_t slm_k = (uint32_t) (blk & 1) * kHalfKv;
                        const uint32_t slm_v = slm_k + 8192u;
                        const bool full_blk = (p0 + kBlk <= ub);

                        if (full_blk) {
                            for (int c = tid; c < 32; c += W) {
                                const simd<uint32_t, 64> kd = block_load<uint32_t, 64>(kb + (size_t) (p0 + c) * 64u + (size_t) pl.zero_fence);
                                slm_block_store<uint32_t, 64>(slm_k + (uint32_t) c * 256u, kd, kAl16);
                                const simd<uint32_t, 64> vd = block_load<uint32_t, 64>(vb + (size_t) blk * 2048u + (size_t) c * 64u + (size_t) pl.zero_fence);
                                slm_block_store<uint32_t, 64>(slm_v + (uint32_t) c * 256u, vd, kAl16);
                            }
                        } else {
                            for (int c = tid; c < 32; c += W) {
                                const simd<uint32_t, 64> kd = (p0 + c < ub)
                                    ? block_load<uint32_t, 64>(kb + (size_t) (p0 + c) * 64u + (size_t) pl.zero_fence)
                                    : simd<uint32_t, 64>(0u);
                                slm_block_store<uint32_t, 64>(slm_k + (uint32_t) c * 256u, kd, kAl16);
                                const simd<uint32_t, 64> vd = block_load<uint32_t, 64>(vb + (size_t) blk * 2048u + (size_t) c * 64u + (size_t) pl.zero_fence);
                                slm_block_store<uint32_t, 64>(slm_v + (uint32_t) c * 256u, vd, kAl16);
                            }
                        }

                        if (blk + pl.pfd < blk_b) {
                            const int p_blk = blk + pl.pfd;
                            const int pf0 = p_blk * kBlk;
                            if (pf0 + kBlk <= ub) {
                                for (int c = tid; c < 32; c += W) {
                                    es::prefetch<uint32_t, 64>(kb + (size_t) (pf0 + c) * 64u, kPf);
                                    es::prefetch<uint32_t, 64>(vb + (size_t) p_blk * 2048u + (size_t) c * 64u, kPf);
                                }
                            }
                        }
                        barrier();

                        const uint32_t my_base = kSlmScOff + (uint32_t) ((2 * ls + td) * 2048);
                        const uint32_t oth_base = kSlmScOff + (uint32_t) ((2 * ls + (td ^ 1)) * 2048);

                        // Stage 2: score 4 tiles -> store directly to SLM (zero my_sc[4] array in GRFs).
                        if (active_ls) {
                            const uint32_t slm_td = slm_k + (uint32_t) td * 128u;
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                simd<float, 32> kf(0.0f);
                                if constexpr (NS == 4) {
                                    if (full_blk) {
                                        simd<uint32_t, 16> w16 = block_load<uint32_t, 16>(ks16b + (size_t) (p0 + 8 * j) * 2);
                                        simd<uint16_t, 32> u16 = w16.template bit_cast_view<uint16_t>();
                                        simd<sycl::half, 32> h16 = u16.template bit_cast_view<sycl::half>();
                                        kf = h16;
                                    } else {
#pragma unroll
                                        for (int c8 = 0; c8 < 8; ++c8) {
                                            if (p0 + 8 * j + c8 < ub) {
                                                simd<uint32_t, 2> w2 = block_load<uint32_t, 2>(ks16b + (size_t) (p0 + 8 * j + c8) * 2);
                                                simd<uint16_t, 4> u4 = w2.template bit_cast_view<uint16_t>();
                                                simd<sycl::half, 4> h4 = u4.template bit_cast_view<sycl::half>();
                                                kf.template select<4, 1>(4 * c8) = h4;
                                            }
                                        }
                                    }
                                }
                                simd<uint32_t, 32> kc[8];
#pragma unroll
                                for (int c8 = 0; c8 < 8; ++c8) kc[c8] = slm_block_load<uint32_t, 32>(slm_td + (uint32_t) (8 * j + c8) * 256u, kAl16);
                                simd<int, 128> acc(0);
                                if constexpr (NS == 1) {
#pragma unroll
                                    for (int sl = 0; sl < 4; ++sl) {
                                        simd<int, 64> av;
#pragma unroll
                                        for (int c8 = 0; c8 < 8; ++c8) {
                                            av.template select<8, 1>(8 * c8) = kc[c8].template select<8, 1>(8 * sl).template bit_cast_view<int>();
                                        }
                                        acc = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(acc, bq[sl], av);
                                    }
                                    const uint32_t my_off = my_base + (uint32_t) j * 512u;
                                    slm_block_store<float, 64>(my_off, simd<float, 64>(acc.template select<64, 1>(0)), kAl16);
                                    slm_block_store<float, 64>(my_off + 256u, simd<float, 64>(acc.template select<64, 1>(64)), kAl16);
                                } else {
                                    simd<float, 128> sf(0.0f);
#pragma unroll
                                    for (int sl = 0; sl < 4; ++sl) {
                                        simd<int, 64> av;
#pragma unroll
                                        for (int c8 = 0; c8 < 8; ++c8) {
                                            av.template select<8, 1>(8 * c8) = kc[c8].template select<8, 1>(8 * sl).template bit_cast_view<int>();
                                        }
                                        acc = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(acc, bq[sl], av);
                                        if ((sl & 1) == 1) {
                                            const int g_idx = td * 2 + (sl >> 1);
#pragma unroll
                                            for (int c8 = 0; c8 < 8; ++c8) {
                                                simd<int, 16> ai = acc.template select<16, 1>(16 * c8);
                                                simd<float, 16> t = ai;
                                                const float kg = kf[c8 * 4 + g_idx];
                                                sf.template select<16, 1>(16 * c8) += t * kg;
                                            }
                                            acc = 0;
                                        }
                                    }
                                    const uint32_t my_off = my_base + (uint32_t) j * 512u;
                                    slm_block_store<float, 64>(my_off, sf.template select<64, 1>(0), kAl16);
                                    slm_block_store<float, 64>(my_off + 256u, sf.template select<64, 1>(64), kAl16);
                                }
                                if (pl.zero_fence == -20 - j) pl.pl[0] = 0.0f;
                            }
                        }
                        barrier();

                        if (active_ls) {
                            simd<float, 32> ks32(1.0f);
                            if constexpr (NS == 1) {
                                if (full_blk) {
                                    ks32 = block_load<float, 32>(ksb + p0);
                                } else {
#pragma unroll
                                    for (int c = 0; c < 32; ++c) ks32[c] = (p0 + c < ub) ? ksb[p0 + c] : 0.0f;
                                }
                            }
                            const simd<float, 32> vs32 = block_load<float, 32>(vsb + p0);

                            // Stage 3a: find m_blk and vmx across 4 tiles (64 lanes at a time, zero sv[4]).
                            simd<float, 16> m_blk(kNeg), vmx(0.0f);
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
#pragma unroll
                                for (int h2 = 0; h2 < 2; ++h2) {
                                    const uint32_t off_slm = (uint32_t) j * 512u + (uint32_t) h2 * 256u;
                                    simd<float, 64> my64 = slm_block_load<float, 64>(my_base + off_slm, kAl16);
                                    simd<float, 64> oth64 = slm_block_load<float, 64>(oth_base + off_slm, kAl16);
                                    my64 += oth64;
#pragma unroll
                                    for (int c4 = 0; c4 < 4; ++c4) {
                                        const int cc = 8 * j + 4 * h2 + c4;
                                        simd<float, 16> sv = my64.template select<16, 1>(16 * c4);
                                        if constexpr (NS == 1) sv = sv * (qsc * float(ks32[cc]));
                                        else sv = sv * qsc;
                                        const int pv = p0 + cc;
                                        simd_mask<16> off = (hi_v <= pv);
                                        if (has_tmask && pv >= p0_tree) {
                                            const int u = pv - p0_tree;
                                            const uint32_t bit = (uint32_t) u < 32u ? (1u << (uint32_t) u) : 0u;
                                            off = off | ((tmk_v & bit) == 0u);
                                        }
                                        sv.merge(simd<float, 16>(kNeg), off);
                                        m_blk = (es::max)(m_blk, sv);
                                        simd<float, 16> vv(vs32[cc]);
                                        vv.merge(simd<float, 16>(0.0f), off);
                                        vmx = (es::max)(vmx, vv);
                                    }
                                    if (pl.zero_fence == -40 - (j * 2 + h2)) pl.pl[0] = 0.0f;
                                }
                            }

                            if (hmax<float>(m_blk) > -1e29f) {
                                simd<float, 16> m_new = (es::max)(m, m_blk);
                                simd<float, 16> alpha = exp16((es::max)(m - m_new, simd<float, 16>(-87.0f)));
                                alpha.merge(simd<float, 16>(0.0f), m <= -1e29f);
                                const float max_q = U16 ? 65535.0f : 255.0f;
                                const simd<float, 16> inv_v = max_q / (es::max)(vmx, simd<float, 16>(1e-30f));
                                simd<float, 16> l_blk(0.0f);
                                simd<uint32_t, 128> blo(0u), bhi(0u);

                                // Stage 3b: compute exp64 + quantize 4 cells x 16 rows per half-tile (zero exp128, zero pr[4]).
#pragma unroll
                                for (int j = 0; j < 4; ++j) {
#pragma unroll
                                    for (int h2 = 0; h2 < 2; ++h2) {
                                        const uint32_t off_slm = (uint32_t) j * 512u + (uint32_t) h2 * 256u;
                                        simd<float, 64> my64 = slm_block_load<float, 64>(my_base + off_slm, kAl16);
                                        simd<float, 64> oth64 = slm_block_load<float, 64>(oth_base + off_slm, kAl16);
                                        my64 += oth64;
                                        simd<float, 64> x64;
#pragma unroll
                                        for (int c4 = 0; c4 < 4; ++c4) {
                                            const int cc = 8 * j + 4 * h2 + c4;
                                            simd<float, 16> sv = my64.template select<16, 1>(16 * c4);
                                            if constexpr (NS == 1) sv = sv * (qsc * float(ks32[cc]));
                                            else sv = sv * qsc;
                                            const int pv = p0 + cc;
                                            simd_mask<16> off = (hi_v <= pv);
                                            if (has_tmask && pv >= p0_tree) {
                                                const int u = pv - p0_tree;
                                                const uint32_t bit = (uint32_t) u < 32u ? (1u << (uint32_t) u) : 0u;
                                                off = off | ((tmk_v & bit) == 0u);
                                            }
                                            sv.merge(simd<float, 16>(kNeg), off);
                                            my64.template select<16, 1>(16 * c4) = sv;
                                            x64.template select<16, 1>(16 * c4) = sv - m_new;
                                        }
                                        simd_mask<64> inv64 = (my64 < -1e29f);
                                        x64 = (es::max)(x64, simd<float, 64>(-87.0f));
                                        simd<float, 64> e64 = exp64(x64);
                                        e64.merge(simd<float, 64>(0.0f), inv64);
#pragma unroll
                                        for (int c4 = 0; c4 < 4; ++c4) {
                                            simd<float, 16> er = e64.template select<16, 1>(16 * c4);
                                            l_blk += er;
                                            my64.template select<16, 1>(16 * c4) = er * (inv_v * float(vs32[8 * j + 4 * h2 + c4])) + 0.5f;
                                        }
                                        my64 = (es::min)(my64, simd<float, 64>(max_q));
                                        simd<uint32_t, 64> qv = convert<uint32_t>(my64);
                                        simd<uint32_t, 16> q0 = qv.template select<16, 1>(0);
                                        simd<uint32_t, 16> q1 = qv.template select<16, 1>(16);
                                        simd<uint32_t, 16> q2 = qv.template select<16, 1>(32);
                                        simd<uint32_t, 16> q3 = qv.template select<16, 1>(48);
                                        blo.template select<16, 1>(16 * (2 * j + h2)) =
                                            (q0 & 255u) | ((q1 & 255u) << 8) | ((q2 & 255u) << 16) | (q3 << 24);
                                        if constexpr (U16) {
                                            bhi.template select<16, 1>(16 * (2 * j + h2)) =
                                                (q0 >> 8) | ((q1 >> 8) << 8) | ((q2 >> 8) << 16) | ((q3 >> 8) << 24);
                                        }
                                        if (pl.zero_fence == -50 - (j * 2 + h2)) pl.pl[0] = 0.0f;
                                    }
                                }
                                l = l * alpha + l_blk;
                                m = m_new;

                                simd<float, 128> alpha128;
#pragma unroll
                                for (int u = 0; u < 8; ++u) alpha128.template select<16, 1>(16 * u) = alpha;

                                const simd<float, 16> srw = vmx * (1.0f / max_q);
                                simd<float, 128> srw128;
#pragma unroll
                                for (int u = 0; u < 8; ++u) srw128.template select<16, 1>(16 * u) = srw;

                                // Stage 3c: V DPAS with runtime fence every 2 iterations to cap load hoisting; fuse alpha scale in-place.
                                simd<int, 128> bl = blo.template bit_cast_view<int>();
                                simd<int, 128> bh = bhi.template bit_cast_view<int>();
                                const uint32_t v_base = slm_v + (uint32_t) td * 4096u;
#pragma unroll
                                for (int ag = 0; ag < 16; ++ag) {
                                    simd<uint32_t, 64> wv = slm_block_load<uint32_t, 64>(v_base + (uint32_t) ag * 256u, kAl16);
                                    simd<int, 64> av = wv.template bit_cast_view<int>();
                                    simd<int, 128> alo = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::u8, xmx::dpas_argument_type::s8>(simd<int, 128>(0), bl, av);
                                    simd<float, 128> fl = alo;
                                    if constexpr (U16) {
                                        simd<int, 128> ahi = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::u8, xmx::dpas_argument_type::s8>(simd<int, 128>(0), bh, av);
                                        simd<float, 128> fh = ahi;
                                        o[ag] = o[ag] * alpha128 + (fh * 256.0f + fl) * srw128;
                                    } else {
                                        o[ag] = o[ag] * alpha128 + fl * srw128;
                                    }
                                    if ((ag & 1) == 1 && pl.zero_fence == -70 - ag) pl.pl[0] = 0.0f;
                                }
                            }
                        }
                    }
                }

                if (active_ls) {
                    const int nseg_s = pl.nseg_stride;
                    const int d0 = td * 128;
#pragma unroll
                    for (int r = 0; r < 16; ++r) {
                        const int rg = ls * 16 + r;
                        const int tok = (rg * 43691) >> 18;
                        if (tok < (int) gp.cnt) {
                            const int hd = rg - tok * kG;
                            const size_t row = (size_t) ((int) gp.r0 + tok) * (size_t) pl.H + (size_t) (kvh * kG + hd);
                            const size_t p_idx = row * (size_t) nseg_s + (size_t) seg;
                            if (td == 0) {
                                pl.pm[p_idx] = m[r];
                                pl.pl[p_idx] = l[r];
                            }
                            float* pout = pl.pacc + p_idx * (size_t) kHD + (size_t) d0;
#pragma unroll
                            for (int h2 = 0; h2 < 2; ++h2) {
                                simd<float, 64> v64;
#pragma unroll
                                for (int ag = 0; ag < 8; ++ag) v64.template select<8, 1>(8 * ag) = o[8 * h2 + ag].template select<8, 16>(r);
                                block_store<float, 64>(pout + 64 * h2, v64);
                            }
                        }
                    }
                }
            });
    });
}

// Form 2: S8 (8 rows/thread, ALL 256 dims, 1 barrier Coop SLM K+V, W = max_nrg8).
template <int NS, int U16> struct TagXa2S8;
template <int NS, int U16> void launch_s8(sycl::queue& q, const AttnPlan2& pl) {
    const int W = pl.wg_threads;
    constexpr uint32_t kHalfKv = 16384u;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagXa2S8<NS, U16>>(
            sycl::nd_range<3>(sycl::range<3>(1, (size_t) pl.HKV, (size_t) pl.total_tasks * (size_t) W),
                              sycl::range<3>(1, 1, (size_t) W)),
            [=](sycl::nd_item<3> it) SYCL_ESIMD_KERNEL {
                slm_init<2u * kHalfKv>();
                const int kvh = (int) it.get_global_id(1);
                const int task = (int) it.get_group(2);
                const int tid = (int) it.get_local_id(2);

                const int gi = (int) pl.task_grp[task];
                const int seg = (int) pl.task_seg[task];
                const AttnGrp2& gp = pl.g[gi];
                const int r_base = tid * 8;
                const bool active = r_base < (int) gp.cnt * kG;

                const uint32_t* kb = nullptr;
                const float* ksb = nullptr;
                const uint32_t* ks16b = nullptr;
                const uint32_t* vb = nullptr;
                const float* vsb = nullptr;
                load_grp_ptrs<NS>(pl, gi, kvh, kb, ksb, ks16b, vb, vsb);

                simd<int, 8> hi_v;
                simd<uint32_t, 8> tmk_v;
                int ub = 0, p0_tree = 0;
                bool has_tmask = false;
                load_row_bounds<8>(pl, gi, r_base, hi_v, tmk_v, ub, has_tmask, p0_tree);

                int blk_a = 0, blk_b = 0;
                seg_block_range(ub, seg, (int) gp.nseg, blk_a, blk_b);

                simd<float, 64> o[32];
#pragma unroll
                for (int k = 0; k < 32; ++k) o[k] = 0.0f;
                simd<float, 8> m(kNeg), l(0.0f);

                if (blk_a < blk_b) {
                    const int ls_loc = r_base >> 4;
                    const int half_off = r_base & 8;
                    const uint32_t* qbb = pl.qb + ((size_t) kvh * 8u) * kMaxBatchGroups2 * 128u + (size_t) ((int) gp.ls0 + ls_loc) * 128u;
                    const simd<float, 8> qsc = active
                        ? block_load<float, 8>(pl.qs + ((size_t) kvh * kMaxBatchGroups2 + (size_t) ((int) gp.ls0 + ls_loc)) * 16u + (size_t) half_off)
                        : simd<float, 8>(0.0f);

                    for (int pb = 0; pb < pl.pfd && blk_a + pb < blk_b; ++pb) {
                        const int p_blk = blk_a + pb;
                        const int pf0 = p_blk * kBlk;
                        if (pf0 + kBlk <= ub) {
                            for (int c = tid; c < 32; c += W) {
                                es::prefetch<uint32_t, 64>(kb + (size_t) (pf0 + c) * 64u, kPf);
                                es::prefetch<uint32_t, 64>(vb + (size_t) p_blk * 2048u + (size_t) c * 64u, kPf);
                            }
                        }
                    }

                    for (int blk = blk_a; blk < blk_b; ++blk) {
                        const int p0 = blk * kBlk;
                        const uint32_t slm_k = (uint32_t) (blk & 1) * kHalfKv;
                        const uint32_t slm_v = slm_k + 8192u;
                        const bool full_blk = (p0 + kBlk <= ub);

                        for (int c = tid; c < 32; c += W) {
                            const simd<uint32_t, 64> kd = (full_blk || p0 + c < ub)
                                ? block_load<uint32_t, 64>(kb + (size_t) (p0 + c) * 64u + (size_t) pl.zero_fence)
                                : simd<uint32_t, 64>(0u);
                            slm_block_store<uint32_t, 64>(slm_k + (uint32_t) c * 256u, kd, kAl16);
                            const simd<uint32_t, 64> vd = block_load<uint32_t, 64>(vb + (size_t) blk * 2048u + (size_t) c * 64u + (size_t) pl.zero_fence);
                            slm_block_store<uint32_t, 64>(slm_v + (uint32_t) c * 256u, vd, kAl16);
                        }
                        if (blk + pl.pfd < blk_b) {
                            const int p_blk = blk + pl.pfd;
                            const int pf0 = p_blk * kBlk;
                            if (pf0 + kBlk <= ub) {
                                for (int c = tid; c < 32; c += W) {
                                    es::prefetch<uint32_t, 64>(kb + (size_t) (pf0 + c) * 64u, kPf);
                                    es::prefetch<uint32_t, 64>(vb + (size_t) p_blk * 2048u + (size_t) c * 64u, kPf);
                                }
                            }
                        }
                        barrier();

                        if (active) {
                            simd<float, 64> sc[4];
                            simd<float, 8> m_blk(kNeg);
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const int p = p0 + 8 * j;
                                simd<float, 8> ks8(1.0f);
                                simd<float, 32> kf(0.0f);
                                if constexpr (NS == 1) {
                                    if (full_blk) ks8 = block_load<float, 8>(ksb + p);
                                    else {
#pragma unroll
                                        for (int c8 = 0; c8 < 8; ++c8) ks8[c8] = (p + c8 < ub) ? ksb[p + c8] : 0.0f;
                                    }
                                } else {
                                    if (full_blk) {
                                        simd<uint32_t, 16> w16 = block_load<uint32_t, 16>(ks16b + (size_t) p * 2);
                                        simd<uint16_t, 32> u16 = w16.template bit_cast_view<uint16_t>();
                                        simd<sycl::half, 32> h16 = u16.template bit_cast_view<sycl::half>();
                                        kf = h16;
                                    } else {
#pragma unroll
                                        for (int c8 = 0; c8 < 8; ++c8) {
                                            if (p + c8 < ub) {
                                                simd<uint32_t, 2> w2 = block_load<uint32_t, 2>(ks16b + (size_t) (p + c8) * 2);
                                                simd<uint16_t, 4> u4 = w2.template bit_cast_view<uint16_t>();
                                                simd<sycl::half, 4> h4 = u4.template bit_cast_view<sycl::half>();
                                                kf.template select<4, 1>(4 * c8) = h4;
                                            }
                                        }
                                    }
                                }
                                simd<int, 128> acc(0);
                                simd<float, 64> sf(0.0f);
#pragma unroll
                                for (int s = 0; s < 8; ++s) {
                                    simd<int, 64> av;
#pragma unroll
                                    for (int c8 = 0; c8 < 8; ++c8) {
                                        simd<uint32_t, 8> kw = slm_block_load<uint32_t, 8>(
                                            slm_k + (uint32_t) (8 * j + c8) * 256u + (uint32_t) s * 32u, kAl16);
                                        av.template select<8, 1>(8 * c8) = kw.template bit_cast_view<int>();
                                    }
                                    const uint32_t* qp = qbb + (size_t) s * kMaxBatchGroups2 * 128u;
                                    simd<uint32_t, 128> u;
                                    u.template select<64, 1>(0) = block_load<uint32_t, 64>(qp);
                                    u.template select<64, 1>(64) = block_load<uint32_t, 64>(qp + 64);
                                    simd<int, 128> b(0);
                                    if (half_off == 0) {
                                        b = u.template bit_cast_view<int>();
                                    } else {
#pragma unroll
                                        for (int d = 0; d < 8; ++d)
                                            b.template select<8, 1>(16 * d) = u.template select<8, 1>(16 * d + 8).template bit_cast_view<int>();
                                    }
                                    acc = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(acc, b, av);
                                    if constexpr (NS == 4) {
                                        if ((s & 1) == 1) {
#pragma unroll
                                            for (int c8 = 0; c8 < 8; ++c8) {
                                                simd<int, 8> ai = acc.template select<8, 1>(16 * c8);
                                                simd<float, 8> t = ai;
                                                const float kg = kf[c8 * 4 + (s >> 1)];
                                                sf.template select<8, 1>(8 * c8) += t * kg;
                                            }
                                            acc = 0;
                                        }
                                    }
                                    if ((s & 1) == 1 && pl.zero_fence == -90 - (j * 8 + s)) pl.pl[0] = 0.0f;
                                }
#pragma unroll
                                for (int c8 = 0; c8 < 8; ++c8) {
                                    simd<float, 8> sv;
                                    if constexpr (NS == 1) {
                                        simd<int, 8> ai = acc.template select<8, 1>(16 * c8);
                                        sv = ai;
                                        const float kc1 = ks8[c8];
                                        sv = sv * (qsc * kc1);
                                    } else {
                                        sv = sf.template select<8, 1>(8 * c8);
                                        sv = sv * qsc;
                                    }
                                    simd_mask<8> off = (hi_v <= (p + c8));
                                    if (has_tmask && (p + c8) >= p0_tree) {
                                        const int u = (p + c8) - p0_tree;
                                        const uint32_t bit = (uint32_t) u < 32u ? (1u << (uint32_t) u) : 0u;
                                        off = off | ((tmk_v & bit) == 0u);
                                    }
                                    sv.merge(simd<float, 8>(kNeg), off);
                                    m_blk = (es::max)(m_blk, sv);
                                    sc[j].template select<8, 1>(8 * c8) = sv;
                                }
                                if (pl.zero_fence == -80 - j) pl.pl[0] = 0.0f;
                            }

                            const simd<float, 32> vs32 = block_load<float, 32>(vsb + p0);
                            softmax_v_r8_blk<U16, true>(sc, m_blk, vs32, nullptr, (size_t) slm_v, pl.zero_fence, pl.pl, o, m, l);
                        }
                    }
                }
                if (active) store_partials_r8(pl, gi, seg, kvh, r_base, o, m, l);
            });
    });
}

// Form 3: L8 (8 rows/thread, ALL 256 dims, 0 SLM, 0 barriers, L1-shared DRAM + deep prefetch).
template <int NS, int U16> struct TagXa2L8;
template <int NS, int U16> void launch_l8(sycl::queue& q, const AttnPlan2& pl) {
    const int W = pl.wg_threads;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagXa2L8<NS, U16>>(
            sycl::nd_range<3>(sycl::range<3>(1, (size_t) pl.HKV, (size_t) pl.total_tasks * (size_t) W),
                              sycl::range<3>(1, 1, (size_t) W)),
            [=](sycl::nd_item<3> it) SYCL_ESIMD_KERNEL {
                const int kvh = (int) it.get_global_id(1);
                const int task = (int) it.get_group(2);
                const int tid = (int) it.get_local_id(2);

                const int gi = (int) pl.task_grp[task];
                const int seg = (int) pl.task_seg[task];
                const AttnGrp2& gp = pl.g[gi];
                const int r_base = tid * 8;
                if (r_base >= (int) gp.cnt * kG) return;

                const uint32_t* kb = nullptr;
                const float* ksb = nullptr;
                const uint32_t* ks16b = nullptr;
                const uint32_t* vb = nullptr;
                const float* vsb = nullptr;
                load_grp_ptrs<NS>(pl, gi, kvh, kb, ksb, ks16b, vb, vsb);

                simd<int, 8> hi_v;
                simd<uint32_t, 8> tmk_v;
                int ub = 0, p0_tree = 0;
                bool has_tmask = false;
                load_row_bounds<8>(pl, gi, r_base, hi_v, tmk_v, ub, has_tmask, p0_tree);

                int blk_a = 0, blk_b = 0;
                seg_block_range(ub, seg, (int) gp.nseg, blk_a, blk_b);

                simd<float, 64> o[32];
#pragma unroll
                for (int k = 0; k < 32; ++k) o[k] = 0.0f;
                simd<float, 8> m(kNeg), l(0.0f);

                if (blk_a < blk_b) {
                    const int ls_loc = r_base >> 4;
                    const int half_off = r_base & 8;
                    const uint32_t* qbb = pl.qb + ((size_t) kvh * 8u) * kMaxBatchGroups2 * 128u + (size_t) ((int) gp.ls0 + ls_loc) * 128u;
                    const simd<float, 8> qsc = block_load<float, 8>(pl.qs + ((size_t) kvh * kMaxBatchGroups2 + (size_t) ((int) gp.ls0 + ls_loc)) * 16u + (size_t) half_off);

                    if (tid == 0) {
                        for (int pb = 0; pb < pl.pfd && blk_a + pb < blk_b; ++pb) prefetch_blk_kv(kb, vb, blk_a + pb, ub);
                    }
                    for (int blk = blk_a; blk < blk_b; ++blk) {
                        if (tid == 0 && blk + pl.pfd < blk_b) prefetch_blk_kv(kb, vb, blk + pl.pfd, ub);
                        const int p0 = blk * kBlk;
                        simd<float, 64> sc[4];
                        simd<float, 8> m_blk;
                        score_dram_r8_blk<NS>(qbb, half_off, qsc, kb, ksb, ks16b, p0, ub, has_tmask, p0_tree, hi_v, tmk_v, pl.zero_fence, pl.pl, sc, m_blk);
                        const simd<float, 32> vs32 = block_load<float, 32>(vsb + p0);
                        softmax_v_r8_blk<U16, false>(sc, m_blk, vs32, vb, (size_t) blk * 2048u, pl.zero_fence, pl.pl, o, m, l);
                    }
                }
                store_partials_r8(pl, gi, seg, kvh, r_base, o, m, l);
            });
    });
}

// Coalesced ESIMD segment merge (optional fused sigm(gate)).
template <int HAS_GATE> struct TagXa2Merge;
template <int HAS_GATE> void launch_merge2(sycl::queue& q, const MPlan2& pl) {
    const int rows = pl.nt * pl.H;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagXa2Merge<HAS_GATE>>(
            sycl::nd_range<2>(sycl::range<2>((size_t) rows, 4u),
                              sycl::range<2>(1u, 4u)),
            [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
                const int row = (int) it.get_global_id(0);
                const int d_chk = (int) it.get_local_id(1);
                const int d0 = d_chk * 64;
                const int tok = row / pl.H;
                const int gi = (int) pl.tok_grp[tok];
                if (gi >= pl.ngrp) return;
                const int nseg = (int) pl.g_nseg[gi];
                const size_t stride = (size_t) pl.nseg_stride;
                const size_t row_off = (size_t) row * stride;

                simd<float, 16> mx(kNeg);
                const simd<int, 16> lane(0, 1);
                for (int s0 = 0; s0 < nseg; s0 += 16) {
                    simd<float, 16> mv(kNeg);
                    if (s0 + 16 <= nseg) {
                        mv = block_load<float, 16>(pl.pm + row_off + (size_t) s0);
                    } else {
                        for (int k = 0; k < 16 && s0 + k < nseg; ++k) mv[k] = pl.pm[row_off + (size_t) (s0 + k)];
                    }
                    mx = (es::max)(mx, mv);
                }
                const float m_all = hmax<float>(mx);

                simd<float, 64> num(0.0f);
                float den = 0.0f;
                for (int s0 = 0; s0 < nseg; s0 += 16) {
                    simd<float, 16> mv(kNeg), lv(0.0f);
                    if (s0 + 16 <= nseg) {
                        mv = block_load<float, 16>(pl.pm + row_off + (size_t) s0);
                        lv = block_load<float, 16>(pl.pl + row_off + (size_t) s0);
                    } else {
                        for (int k = 0; k < 16 && s0 + k < nseg; ++k) {
                            mv[k] = pl.pm[row_off + (size_t) (s0 + k)];
                            lv[k] = pl.pl[row_off + (size_t) (s0 + k)];
                        }
                    }
                    simd<float, 16> w = exp16((es::max)(mv - m_all, simd<float, 16>(-87.0f)));
                    w.merge(simd<float, 16>(0.0f), (mv <= -1e29f) | ((s0 + lane) >= nseg));
                    den += hsum16_f(w * lv);
                    const float* pa_base = pl.pacc + (row_off + (size_t) s0) * (size_t) kHD + (size_t) d0;
                    for (int k = 0; k < 16 && s0 + k < nseg; ++k) {
                        const float wk = w[k];
                        if (wk != 0.0f) {
                            const simd<float, 64> pv = block_load<float, 64>(pa_base + (size_t) k * (size_t) kHD);
                            num += wk * pv;
                        }
                    }
                }
                simd<float, 64> out = (den > 0.0f) ? (num * (1.0f / den)) : simd<float, 64>(0.0f);
                const size_t gi_off = (size_t) row * (size_t) kHD + (size_t) d0;
                if constexpr (HAS_GATE) {
                    const simd<float, 64> g = block_load<float, 64>(pl.gate + gi_off);
                    out = out * sigm64(g);
                }
                block_store<float, 64>(pl.out + gi_off, out);
            });
    });
}

int build_plan2(const Call2& c, HostPlan2& hp, int& form_out, int& ng_d2_out, int& u16_out) {
    static const int env_form = [] { const char* e = std::getenv("STRATA_Q36_XATT2_FORM"); return e && *e ? std::atoi(e) : -1; }();
    static const int env_segs = [] { const char* e = std::getenv("STRATA_Q36_XATT2_SEGS"); return e && *e ? std::atoi(e) : 0; }();
    static const int env_u16 = [] { const char* e = std::getenv("STRATA_Q36_XATT2_U16"); return e && *e ? std::atoi(e) : 1; }();
    static const int env_pfd = [] { const char* e = std::getenv("STRATA_Q36_XATT2_PFD"); return e && *e ? std::atoi(e) : 2; }();

    if (c.nt < 1 || c.nt > 16 || c.H != kG * c.HKV || (c.ns != 1 && c.ns != 4)) return -1;
    if (!c.q || !c.pos0 || !c.qb || !c.qs || !c.pacc || !c.pm || !c.pl) return -1;

    hp = HostPlan2{};
    AttnPlan2& pl = hp.a;
    QPlan2& qp = hp.q;
    MPlan2& mp = hp.m;

    pl.t0 = c.t0;
    pl.H = c.H;
    pl.HKV = c.HKV;
    pl.kst = c.kstride > 0 ? c.kstride : (c.ctx > 0 && c.ns == 4 && c.k_tab ? c.ctx : c.ctxp);
    pl.ctxp = c.ctxp;
    pl.pfd = env_pfd >= 0 ? env_pfd : 0;
    pl.zero_fence = c.zero_fence;
    pl.scale = c.scale;
    pl.pos0 = c.pos0;
    pl.tmask = c.tmask;
    pl.tp0 = c.tp0;
    pl.k_tab = c.k_tab;
    pl.v_tab = c.v_tab;
    pl.mir_tab = c.mir_tab;
    pl.qb = c.qb;
    pl.qs = c.qs;
    pl.pacc = c.pacc;
    pl.pm = c.pm;
    pl.pl = c.pl;

    qp.nt = c.nt;
    qp.t0 = c.t0;
    qp.H = c.H;
    qp.HKV = c.HKV;
    qp.scale = c.scale;
    qp.q = c.q;
    qp.qb = c.qb;
    qp.qs = c.qs;

    mp.nt = c.nt;
    mp.H = c.H;
    mp.pacc = c.pacc;
    mp.pm = c.pm;
    mp.pl = c.pl;
    mp.gate = c.gate;
    mp.out = c.out;

    int grp_hint[kMaxBatchGroups2]{};
    if (c.ngrp > 0) {
        if (c.ngrp > kMaxBatchGroups2) return -1;
        pl.ngrp = (uint8_t) c.ngrp;
        pl.per_tok_pos = 1;
        for (int i = 0; i < c.ngrp; ++i) {
            if (c.grp[i].slot < 0 || c.grp[i].r0 < 0 || c.grp[i].cnt <= 0) return -1;
            pl.g[i].slot = (uint8_t) c.grp[i].slot;
            pl.g[i].r0 = (uint8_t) c.grp[i].r0;
            pl.g[i].cnt = (uint8_t) c.grp[i].cnt;
            grp_hint[i] = c.grp[i].cell_hint;
        }
    } else {
        if (!c.k8 || !c.v8t || !c.vs) return -1;
        if (c.ns == 1 && !c.ks) return -1;
        if (c.ns == 4 && !c.ks16) return -1;
        pl.ngrp = 1;
        pl.per_tok_pos = 0;
        pl.k8 = c.k8;
        pl.ks = c.ks;
        pl.ks16 = c.ks16;
        pl.v8t = c.v8t;
        pl.vs = c.vs;
        pl.g[0].slot = 0;
        pl.g[0].r0 = 0;
        pl.g[0].cnt = (uint8_t) c.nt;
        grp_hint[0] = c.pos_hint;
    }

    qp.ngrp = (int32_t) pl.ngrp;
    qp.per_tok_pos = (int32_t) pl.per_tok_pos;
    mp.ngrp = (int32_t) pl.ngrp;

    int max_rows = 0;
    int ls_cur = 0;
    std::fill_n(mp.tok_grp, kMaxBatchGroups2, (uint8_t) 0xff);
    for (int i = 0; i < (int) pl.ngrp; ++i) {
        const int cnt_i = (int) pl.g[i].cnt;
        const int r0_i = (int) pl.g[i].r0;
        const int rows = cnt_i * kG;
        if (rows < 1 || rows > kMaxRows) return -1;
        if (rows > max_rows) max_rows = rows;
        const int nrg = (rows + 15) / 16;
        if (ls_cur + nrg > kMaxBatchGroups2) return -1;
        pl.g[i].ls0 = (uint8_t) ls_cur;
        pl.g[i].nrg = (uint8_t) nrg;
        qp.g_r0[i] = pl.g[i].r0;
        qp.g_cnt[i] = pl.g[i].cnt;
        for (int l = 0; l < nrg; ++l) {
            qp.ls_grp[ls_cur + l] = (uint8_t) i;
            qp.ls_loc[ls_cur + l] = (uint8_t) l;
        }
        ls_cur += nrg;
        for (int k = 0; k < cnt_i; ++k) {
            const int tok = r0_i + k;
            if (tok < kMaxBatchGroups2) mp.tok_grp[tok] = (uint8_t) i;
        }
    }
    hp.total_ls = ls_cur;

    u16_out = (c.u16_override >= 0) ? (c.u16_override != 0 ? 1 : 0) : (env_u16 != 0 ? 1 : 0);

    int f = (c.form_override >= 0) ? c.form_override : env_form;
    if (f < 0 || f > 3) {
        f = (max_rows <= 8) ? kFormS1 : kFormD2;
    }
    if (f == kFormS1 && max_rows > 8) f = kFormD2;
    form_out = f;

    ng_d2_out = (max_rows + 15) / 16;
    if (ng_d2_out < 1) ng_d2_out = 1;
    if (ng_d2_out > 6) ng_d2_out = 6;
    const int w = (f == kFormS1) ? 1 : (f == kFormD2 ? 2 * ng_d2_out : ((max_rows + 7) / 8));
    pl.wg_threads = w;

    const int seg_cap = std::min(kMaxSegs2, std::max(1, c.nseg_max));
    const int nseg_req = c.nseg_override > 0 ? c.nseg_override : env_segs;

    int total_tasks = 0;
    int max_nseg = 1;
    const int tgt_threads = 256;
    const int ngrp_i = (int) pl.ngrp;
    const int budget_per_kv = std::max(ngrp_i, std::min(seg_cap, std::max(1, tgt_threads / (pl.HKV * w))));

    for (int i = 0; i < ngrp_i; ++i) {
        const int hint = grp_hint[i] > 0 ? grp_hint[i] : (c.pos_hint > 0 ? c.pos_hint : 256);
        const int nblk = std::max(1, (hint + kBlk - 1) / kBlk);
        int ns_i = 1;
        if (nseg_req > 0) {
            ns_i = std::min(seg_cap / ngrp_i, nseg_req);
        } else {
            const int blk_floor = std::max(1, nblk / 2);
            const int share = (ngrp_i == 1) ? budget_per_kv : std::max(1, budget_per_kv / ngrp_i);
            ns_i = std::min(share, blk_floor);
        }
        if (ns_i < 1) ns_i = 1;
        if (total_tasks + ns_i > seg_cap) ns_i = std::max(1, seg_cap - total_tasks);
        pl.g[i].nseg = (uint8_t) ns_i;
        mp.g_nseg[i] = (uint8_t) ns_i;
        if (ns_i > max_nseg) max_nseg = ns_i;
        for (int s = 0; s < ns_i && total_tasks < kMaxSegs2; ++s) {
            pl.task_grp[total_tasks] = (uint8_t) i;
            pl.task_seg[total_tasks] = (uint8_t) s;
            ++total_tasks;
        }
    }
    pl.total_tasks = (uint8_t) total_tasks;
    pl.nseg_stride = max_nseg;
    mp.nseg_stride = max_nseg;
    return 0;
}

template <int NS, int U16> void dispatch_form(sycl::queue& q, const AttnPlan2& pl, int form, int ng_d2) {
    if (form == kFormS1) {
        launch_s1<NS, U16>(q, pl);
        return;
    }
    if (form == kFormS8) {
        launch_s8<NS, U16>(q, pl);
        return;
    }
    if (form == kFormL8) {
        launch_l8<NS, U16>(q, pl);
        return;
    }
    switch (ng_d2) {
        case 1: launch_d2<1, NS, U16>(q, pl); break;
        case 2: launch_d2<2, NS, U16>(q, pl); break;
        case 3: launch_d2<3, NS, U16>(q, pl); break;
        case 4: launch_d2<4, NS, U16>(q, pl); break;
        case 5: launch_d2<5, NS, U16>(q, pl); break;
        case 6: launch_d2<6, NS, U16>(q, pl); break;
        default: break;
    }
}

}   // namespace

const char* xatt2_form_name(int form) {
    switch (form) {
        case kFormS1: return "S1";
        case kFormD2: return "D2";
        case kFormS8: return "S8";
        case kFormL8: return "L8";
        default: return "?";
    }
}

bool attend2(const Call2& c, void* stream) {
    HostPlan2 hp;
    int form = 0, ng_d2 = 1, u16 = 1;
    if (build_plan2(c, hp, form, ng_d2, u16) != 0) return false;
    sycl::queue& q = *strata::q_of(stream);
    launch_queries2(q, hp.q, hp.total_ls);
    if (c.ns == 4) {
        if (u16) dispatch_form<4, 1>(q, hp.a, form, ng_d2);
        else dispatch_form<4, 0>(q, hp.a, form, ng_d2);
    } else {
        if (u16) dispatch_form<1, 1>(q, hp.a, form, ng_d2);
        else dispatch_form<1, 0>(q, hp.a, form, ng_d2);
    }
    if (c.out) {
        if (c.gate) launch_merge2<1>(q, hp.m);
        else launch_merge2<0>(q, hp.m);
    }
    return true;
}

}   // namespace strata::q36xatt
