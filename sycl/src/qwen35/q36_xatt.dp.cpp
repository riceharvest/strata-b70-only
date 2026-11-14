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

constexpr int kFormR8 = 0;
constexpr int kFormRS16 = 1;
constexpr int kFormRS8 = 2;
constexpr int kFormRS16K = 3;
constexpr float kLog2e2 = 1.4426950408889634f;

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

SYCL_ESIMD_FUNCTION inline float hsum32_f(simd<float, 32> v) {
    simd<float, 16> a = v.template select<16, 1>(0) + v.template select<16, 1>(16);
    simd<float, 8> b = a.template select<8, 1>(0) + a.template select<8, 1>(8);
    simd<float, 4> c = b.template select<4, 1>(0) + b.template select<4, 1>(4);
    simd<float, 2> d = c.template select<2, 1>(0) + c.template select<2, 1>(2);
    return float(d[0]) + float(d[1]);
}

SYCL_ESIMD_FUNCTION inline int hmin8_i(simd<int, 8> v) {
    simd<int, 4> a = (es::min)(simd<int, 4>(v.template select<4, 1>(0)), simd<int, 4>(v.template select<4, 1>(4)));
    simd<int, 2> b = (es::min)(simd<int, 2>(a.template select<2, 1>(0)), simd<int, 2>(a.template select<2, 1>(2)));
    return int(b[0]) < int(b[1]) ? int(b[0]) : int(b[1]);
}

inline int xatt2_env_form() {
    static const int v = [] { const char* e = std::getenv("STRATA_Q36_XATT2_FORM"); return e && *e ? std::atoi(e) : -1; }();
    return v;
}
inline int xatt2_env_segs() {
    static const int v = [] { const char* e = std::getenv("STRATA_Q36_XATT2_SEGS"); return e && *e ? std::atoi(e) : 0; }();
    return v;
}
inline int xatt2_env_u16() {
    static const int v = [] { const char* e = std::getenv("STRATA_Q36_XATT2_U16"); return e && *e ? std::atoi(e) : 1; }();
    return v;
}
inline int xatt2_env_exp() {
    static const int v = [] { const char* e = std::getenv("STRATA_Q36_XATT2_EXP"); return e && *e ? (std::atoi(e) != 0 ? 1 : 0) : 0; }();
    return v;
}
inline int xatt2_env_pfd() {
    static const int v = [] { const char* e = std::getenv("STRATA_Q36_XATT2_PFD"); return e && *e ? std::atoi(e) : 1; }();
    return v;
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

inline bool xatt2_dbg_on() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_Q36_XATT2_DBG");
        return e && e[0] == '1';
    }();
    return on;
}

SYCL_ESIMD_FUNCTION inline float xatt2_nan_f32() {
    simd<uint32_t, 1> nu(0x7fc00000u);
    return nu.template bit_cast_view<float>()[0];
}

SYCL_ESIMD_FUNCTION inline void xatt2_mark_oob(uint32_t* dbg_flag, uint32_t code, float* fallback_nan) {
    if (dbg_flag) dbg_flag[0] = code;
    if (fallback_nan) fallback_nan[0] = xatt2_nan_f32();
}

SYCL_ESIMD_FUNCTION inline uint32_t unpack_u8_16(uint64_t lo, uint64_t hi, int idx) {
    const uint64_t w = (idx < 8) ? lo : hi;
    return (uint32_t) ((w >> ((uint32_t) (idx & 7) * 8u)) & 0xffu);
}

SYCL_ESIMD_FUNCTION inline uint32_t unpack_grp_u32(
    int gi, uint64_t g0, uint64_t g1, uint64_t g2, uint64_t g3,
    uint64_t g4, uint64_t g5, uint64_t g6, uint64_t g7) {
    uint64_t w = g0;
    const int p = gi >> 1;
    if (p == 1) w = g1;
    else if (p == 2) w = g2;
    else if (p == 3) w = g3;
    else if (p == 4) w = g4;
    else if (p == 5) w = g5;
    else if (p == 6) w = g6;
    else if (p == 7) w = g7;
    return (gi & 1) ? (uint32_t) (w >> 32) : (uint32_t) (w & 0xffffffffull);
}

inline uint32_t pack_grp_u32(int slot, int r0, int cnt, int ls0, int nrg, int nseg) {
    return (uint32_t) (slot & 0xff)
         | ((uint32_t) (r0 & 0xf) << 8)
         | ((uint32_t) ((cnt - 1) & 0xf) << 12)
         | ((uint32_t) (ls0 & 0xf) << 16)
         | ((uint32_t) ((nrg - 1) & 0x7) << 20)
         | ((uint32_t) (nseg & 0x1ff) << 23);
}

SYCL_ESIMD_FUNCTION inline void decode_grp_u32(
    uint32_t w, int& slot, int& r0, int& cnt, int& ls0, int& nrg, int& nseg) {
    slot = (int) (w & 0xffu);
    r0 = (int) ((w >> 8) & 0xfu);
    cnt = (int) (((w >> 12) & 0xfu) + 1u);
    ls0 = (int) ((w >> 16) & 0xfu);
    nrg = (int) (((w >> 20) & 0x7u) + 1u);
    nseg = (int) ((w >> 23) & 0x1ffu);
}

struct QPlan2 {
    int32_t ngrp = 1;
    int32_t nt = 1;
    int32_t t0 = 0;
    int32_t H = 24;
    int32_t HKV = 4;
    int32_t per_tok_pos = 0;
    float scale = 1.0f / 16.0f;
    uint32_t dbg = 0;
    const float* q = nullptr;
    uint32_t* qb = nullptr;
    float* qs = nullptr;
    uint32_t* dbg_flag = nullptr;
    uint64_t q_floats = 0;
    uint64_t qb_dwords = 0;
    uint64_t qs_floats = 0;
    uint64_t g_r0_cnt_lo = 0;
    uint64_t g_r0_cnt_hi = 0;
    uint64_t ls_gl_lo = 0;
    uint64_t ls_gl_hi = 0;
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
    uint32_t* dbg_flag = nullptr;
    int32_t kst = 0;
    int32_t ctxp = 0;
    int32_t wg_threads = 1;
    int32_t pfd = 2;
    int32_t exp_poly = 0;   // STRATA_Q36_XATT2_EXP=1: the softmax's exp by its series, not the hardware exp2
    int32_t zero_fence = 0;
    int32_t nseg_stride = 1;
    int32_t t0 = 0;
    int32_t H = 24;
    int32_t HKV = 4;
    float scale = 1.0f / 16.0f;
    uint32_t dbg = 0;
    uint32_t k8_max_cells = 0;
    uint32_t v8t_max_blks = 0;
    uint32_t vs_max_cells = 0;
    uint64_t qb_dwords = 0;
    uint64_t qs_floats = 0;
    uint64_t p_elems = 0;
    uint64_t pacc_elems = 0;
    int32_t ngrp = 1;
    int32_t total_tasks = 1;
    int32_t per_tok_pos = 0;
    uint64_t gp0 = 0, gp1 = 0, gp2 = 0, gp3 = 0;
    uint64_t gp4 = 0, gp5 = 0, gp6 = 0, gp7 = 0;
};

struct MPlan2 {
    int32_t nt = 1;
    int32_t H = 24;
    int32_t ngrp = 1;
    int32_t nseg_stride = 1;
    float inv_H = 1.0f / 24.0f;
    uint32_t dbg = 0;
    float* pacc = nullptr;
    float* pm = nullptr;
    float* pl = nullptr;
    const float* gate = nullptr;
    float* out = nullptr;
    uint32_t* dbg_flag = nullptr;
    uint64_t p_elems = 0;
    uint64_t pacc_elems = 0;
    uint64_t out_elems = 0;
    uint64_t g_nseg_lo = 0;
    uint64_t g_nseg_hi = 0;
    uint64_t tok_grp_lo = 0;
    uint64_t tok_grp_hi = 0;
};

struct HostPlan2 {
    QPlan2 q{};
    AttnPlan2 a{};
    MPlan2 m{};
    int total_ls = 1;
};

template <int NS>
SYCL_ESIMD_FUNCTION inline void load_grp_ptrs_raw(
    const uint32_t* k8, const float* ks, const uint16_t* ks16, const uint32_t* v8t, const float* vs,
    void* const* k_tab, void* const* v_tab, const void* const* mir_tab,
    int HKV, int kst, int ctxp, int slot, int kvh,
    const uint32_t*& kb, const float*& ksb, const uint32_t*& ks16b,
    const uint32_t*& vb, const float*& vsb) {
    if (!k8) {
        if (k_tab) {
            const int8_t* kp = reinterpret_cast<const int8_t*>(k_tab[slot]);
            k8 = reinterpret_cast<const uint32_t*>(kp);
            ks16 = reinterpret_cast<const uint16_t*>(kp + (size_t) HKV * (size_t) kst * kHD);
        } else if (mir_tab) {
            const size_t b = (size_t) slot * 4u;
            k8 = reinterpret_cast<const uint32_t*>(mir_tab[b + 0]);
            if constexpr (NS == 1) ks = reinterpret_cast<const float*>(mir_tab[b + 1]);
            else ks16 = reinterpret_cast<const uint16_t*>(mir_tab[b + 1]);
        }
        if (v_tab) {
            const int8_t* vp = reinterpret_cast<const int8_t*>(v_tab[slot]);
            v8t = reinterpret_cast<const uint32_t*>(vp);
            vs = reinterpret_cast<const float*>(vp + v8t_dwords(HKV, ctxp) * 4);
        } else if (mir_tab) {
            const size_t b = (size_t) slot * 4u;
            v8t = reinterpret_cast<const uint32_t*>(mir_tab[b + 2]);
            vs = reinterpret_cast<const float*>(mir_tab[b + 3]);
        }
    }
    kb = k8 + (size_t) kvh * kst * 64;
    ksb = (NS == 1 && ks) ? (ks + (size_t) kvh * kst) : nullptr;
    ks16b = (NS == 4 && ks16) ? (reinterpret_cast<const uint32_t*>(ks16) + (size_t) kvh * kst * 2) : nullptr;
    vb = v8t + (size_t) kvh * (ctxp / kBlk) * 256 * 8;
    vsb = vs + (size_t) kvh * ctxp;
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
void launch_queries2(sycl::queue& q, const QPlan2& p, int total_ls) {
    const int nt = p.nt, t0 = p.t0, H = p.H, HKV = p.HKV, per_tok_pos = p.per_tok_pos;
    const float scale = p.scale;
    const uint32_t dbg = p.dbg;
    const float* pq = p.q;
    uint32_t* qb = p.qb;
    float* qs = p.qs;
    uint32_t* dbg_flag = p.dbg_flag;
    const uint64_t q_floats = p.q_floats, qb_dwords = p.qb_dwords, qs_floats = p.qs_floats;
    const uint64_t g_r0_cnt_lo = p.g_r0_cnt_lo, g_r0_cnt_hi = p.g_r0_cnt_hi;
    const uint64_t ls_gl_lo = p.ls_gl_lo, ls_gl_hi = p.ls_gl_hi;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagXa2Q>(
            sycl::nd_range<3>(sycl::range<3>((size_t) HKV, (size_t) total_ls, 16),
                              sycl::range<3>(1, 1, 16)),
            [=](sycl::nd_item<3> it) SYCL_ESIMD_KERNEL {
                slm_init<16u * 64u * 4u>();
                const int kvh = (int) it.get_global_id(0);
                const int lsg = (int) it.get_global_id(1);
                const int lane = (int) it.get_local_id(2);
                const uint32_t lsb = unpack_u8_16(ls_gl_lo, ls_gl_hi, lsg);
                const int gi = (int) (lsb >> 4);
                const int ls_loc = (int) (lsb & 0xfu);
                const uint32_t grb = unpack_u8_16(g_r0_cnt_lo, g_r0_cnt_hi, gi);
                const int gp_r0 = (int) (grb >> 4);
                const int gp_cnt = (int) ((grb & 0xfu) + 1u);
                const int r = ls_loc * 16 + lane;
                const int k = (r * 43691) >> 18;
                const int hd = r - k * kG;
                const bool live = k < gp_cnt;

                simd<uint32_t, 64> w(0u);
                float qsv = 0.0f;
                bool q_oob = false;
                if (live) {
                    const int qt = per_tok_pos ? (gp_r0 + k) : (t0 + k);
                    const size_t src_off = ((size_t) qt * H + kvh * kG + hd) * kHD;
                    if (dbg && (src_off + (size_t) kHD > q_floats)) {
                        q_oob = true;
                        xatt2_mark_oob(dbg_flag, 101u, nullptr);
                    } else {
                        const float* src = pq + src_off;
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
                        qsv = sc * scale;
                    }
                }
                slm_block_store<uint32_t, 64>((uint32_t) lane * 256u, w, kAl16);
                const size_t qs_off = ((size_t) kvh * kMaxBatchGroups2 + (size_t) lsg) * 16u + (size_t) lane;
                if (dbg && (q_oob || qs_off >= qs_floats)) {
                    xatt2_mark_oob(dbg_flag, 102u, qs_floats > 0 ? qs : nullptr);
                    if (qs_off < qs_floats) qs[qs_off] = xatt2_nan_f32();
                } else {
                    qs[qs_off] = qsv;
                }
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
                const size_t dst_off = (((size_t) kvh * 8u + (size_t) s) * kMaxBatchGroups2 + (size_t) lsg) * 128u + (size_t) d0 * 16u;
                if (dbg && (dst_off + 64u > qb_dwords)) {
                    xatt2_mark_oob(dbg_flag, 103u, qs_floats > 0 ? qs : nullptr);
                } else {
                    block_store<uint32_t, 64>(qb + dst_off, out64);
                }
            });
    });
}

// Helper: decode task -> (gi, seg, slot, r0, cnt, ls0, nrg, nseg) with zero C-array/alloca/stack spills.
SYCL_ESIMD_FUNCTION inline void decode_task_grp(
    int task, int ngrp,
    uint64_t gp0, uint64_t gp1, uint64_t gp2, uint64_t gp3,
    uint64_t gp4, uint64_t gp5, uint64_t gp6, uint64_t gp7,
    int& gi_out, int& seg_out, int& slot, int& r0, int& cnt, int& ls0, int& nrg, int& nseg) {
    if (ngrp == 1) {
        gi_out = 0;
        seg_out = task;
        decode_grp_u32((uint32_t) (gp0 & 0xffffffffull), slot, r0, cnt, ls0, nrg, nseg);
        return;
    }
    int rem = task;
    gi_out = 0;
    seg_out = 0;
    uint32_t found = (uint32_t) (gp0 & 0xffffffffull);
    for (int i = 0; i < 16; ++i) {
        if (i >= ngrp) break;
        const uint32_t w = unpack_grp_u32(i, gp0, gp1, gp2, gp3, gp4, gp5, gp6, gp7);
        const int ns_i = (int) ((w >> 23) & 0x1ffu);
        if (rem < ns_i) {
            gi_out = i;
            seg_out = rem;
            found = w;
            break;
        }
        rem -= ns_i;
    }
    decode_grp_u32(found, slot, r0, cnt, ls0, nrg, nseg);
}

// Helper: compute valid-cell upper bound and per-row [0, hi) lengths without host/stack arrays.
template <int R>
SYCL_ESIMD_FUNCTION inline void load_row_bounds_raw(
    const int32_t* pos0, const uint32_t* tmask, const int32_t* tp0,
    int t0, int ngrp, int per_tok_pos, int gp_r0, int gp_cnt, int r_base,
    simd<int, R>& hi_v, simd<uint32_t, R>& tmk_v, int& ub, bool& has_tmask, int& p0_tree) {
    const int pos0_base = pos0[0] + t0 + 1;
    if (per_tok_pos) {
        ub = 0;
        for (int k = 0; k < gp_cnt; ++k) {
            const int len = pos0[gp_r0 + k] + 1;
            if (len > ub) ub = len;
        }
    } else {
        ub = gp_cnt > 0 ? (pos0_base + gp_cnt - 1) : 0;
    }
    has_tmask = (tmask != nullptr) && (ngrp == 1);
    p0_tree = has_tmask ? (tp0 ? tp0[0] : pos0[0]) : 0x3fffffff;
    const int tok0 = (r_base * 43691) >> 18;
    int l0 = 0, l1 = 0, l2 = 0, l3 = 0;
    uint32_t m0 = 0xffffffffu, m1 = 0xffffffffu, m2 = 0xffffffffu, m3 = 0xffffffffu;
    if (tok0 + 0 < gp_cnt) {
        l0 = per_tok_pos ? (pos0[gp_r0 + tok0 + 0] + 1) : (pos0_base + tok0 + 0);
        if (has_tmask) m0 = tmask[t0 + tok0 + 0];
    }
    if (tok0 + 1 < gp_cnt) {
        l1 = per_tok_pos ? (pos0[gp_r0 + tok0 + 1] + 1) : (pos0_base + tok0 + 1);
        if (has_tmask) m1 = tmask[t0 + tok0 + 1];
    }
    if (tok0 + 2 < gp_cnt) {
        l2 = per_tok_pos ? (pos0[gp_r0 + tok0 + 2] + 1) : (pos0_base + tok0 + 2);
        if (has_tmask) m2 = tmask[t0 + tok0 + 2];
    }
    if (tok0 + 3 < gp_cnt) {
        l3 = per_tok_pos ? (pos0[gp_r0 + tok0 + 3] + 1) : (pos0_base + tok0 + 3);
        if (has_tmask) m3 = tmask[t0 + tok0 + 3];
    }
#pragma unroll
    for (int lane = 0; lane < R; ++lane) {
        const int r = r_base + lane;
        const int dt = ((r * 43691) >> 18) - tok0;
        int lv = 0;
        uint32_t mv = 0u;
        if (dt == 0) { lv = l0; mv = m0; }
        else if (dt == 1) { lv = l1; mv = m1; }
        else if (dt == 2) { lv = l2; mv = m2; }
        else if (dt == 3) { lv = l3; mv = m3; }
        hi_v[lane] = lv;
        tmk_v[lane] = mv;
    }
}

// Helper: clamp segment & KV cell bounds under STRATA_Q36_XATT2_DBG=1 so any OOB turns into NaN/FAIL instead of GPU fault.
SYCL_ESIMD_FUNCTION inline bool dbg_clamp_seg_bounds_raw(
    uint32_t dbg, uint32_t* dbg_flag, int HKV,
    uint32_t k8_max_cells, uint32_t v8t_max_blks, uint32_t vs_max_cells,
    uint64_t qb_dwords, uint64_t qs_floats,
    int kvh, int ls_idx, int& ub, int& blk_a, int& blk_b) {
    if (dbg == 0) return false;
    bool oob = false;
    uint32_t code = 0u;
    if ((size_t) kvh >= (size_t) HKV || (size_t) ls_idx >= kMaxBatchGroups2) {
        oob = true; code = 1u;
    }
    const size_t qb_need = (((size_t) kvh * 8u + 7u) * kMaxBatchGroups2 + (size_t) ls_idx) * 128u + 128u;
    const size_t qs_need = ((size_t) kvh * kMaxBatchGroups2 + (size_t) ls_idx) * 16u + 16u;
    if (qb_need > qb_dwords) { oob = true; code = 2u; }
    if (qs_need > qs_floats) { oob = true; code = 3u; }
    if (ub < 0) { oob = true; code = 4u; ub = 0; }
    if (ub > (int) k8_max_cells) { oob = true; code = 5u; ub = (int) k8_max_cells; }
    const int vs_cells = (int) vs_max_cells;
    const int v8t_cells = (int) (v8t_max_blks * (uint32_t) kBlk);
    const int max_v_cells = vs_cells < v8t_cells ? vs_cells : v8t_cells;
    if (ub > max_v_cells) { oob = true; code = 6u; ub = max_v_cells; }
    if (blk_b > blk_a) {
        const int max_full_blks = max_v_cells >> 5;
        if (blk_b > max_full_blks) { oob = true; code = 7u; blk_b = max_full_blks; }
        if (blk_a < 0) { oob = true; code = 8u; blk_a = 0; }
        if (blk_a > blk_b) blk_b = blk_a;
    }
    if (oob) xatt2_mark_oob(dbg_flag, code, nullptr);
    return oob;
}

// Helper: store 8-row outputs (o16[16] of [dl=0..15][row8][d16]) + pm/pl into split-K partials.
SYCL_ESIMD_FUNCTION inline void store_partials_o16_raw(
    float* pacc, float* pm, float* pl, uint32_t dbg, uint32_t* dbg_flag,
    uint64_t p_elems, uint64_t pacc_elems, int H, int nseg_stride,
    int gp_r0, int gp_cnt, int seg, int kvh, int r_base,
    simd<float, 128> o16[16], simd<float, 8> m, simd<float, 8> l, bool oob = false) {
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int rg = r_base + r;
        const int tok = (rg * 43691) >> 18;
        if (tok < gp_cnt) {
            const int hd = rg - tok * kG;
            const size_t row = (size_t) (gp_r0 + tok) * (size_t) H + (size_t) (kvh * kG + hd);
            const size_t p_idx = row * (size_t) nseg_stride + (size_t) seg;
            if (dbg && (oob || p_idx >= p_elems || p_idx * (size_t) kHD + (size_t) kHD > pacc_elems)) {
                xatt2_mark_oob(dbg_flag, oob ? 10u : 11u, (p_idx < p_elems) ? (pm + p_idx) : (p_elems > 0 ? pm : nullptr));
                continue;
            }
            pm[p_idx] = m[r];
            pl[p_idx] = l[r];
            float* pout = pacc + p_idx * (size_t) kHD;
#pragma unroll
            for (int c4 = 0; c4 < 4; ++c4) {
                simd<float, 64> v64;
#pragma unroll
                for (int u = 0; u < 4; ++u) v64.template select<16, 1>(16 * u) = o16[4 * c4 + u].template select<16, 1>(16 * r);
                block_store<float, 64>(pout + 64 * c4, v64);
            }
        }
    }
}

template <int HALF_OFF>
SYCL_ESIMD_FUNCTION inline void load_qa8_half(
    const uint32_t* qb, const float* qs, int kvh, int ls_glob,
    simd<int, 64> qa[8], simd<float, 8>& qsc) {
    const uint32_t* qbb = qb + ((size_t) kvh * 8u) * kMaxBatchGroups2 * 128u + (size_t) ls_glob * 128u;
    qsc = block_load<float, 8>(qs + ((size_t) kvh * kMaxBatchGroups2 + (size_t) ls_glob) * 16u + (size_t) HALF_OFF);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        const uint32_t* qp = qbb + (size_t) s * kMaxBatchGroups2 * 128u;
        simd<uint32_t, 128> u;
        u.template select<64, 1>(0) = block_load<uint32_t, 64>(qp);
        u.template select<64, 1>(64) = block_load<uint32_t, 64>(qp + 64);
        simd<uint32_t, 64> qt;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            qt.template select<8, 1>(8 * r) = u.template select<8, 16>(HALF_OFF + r);
        }
        qa[s] = qt.template bit_cast_view<int>();
    }
}

SYCL_ESIMD_FUNCTION inline void load_qa8_and_scale(
    const uint32_t* qb, const float* qs, int kvh, int ls_glob, int half_off,
    simd<int, 64> qa[8], simd<float, 8>& qsc) {
    if (half_off != 0) load_qa8_half<8>(qb, qs, kvh, ls_glob, qa, qsc);
    else               load_qa8_half<0>(qb, qs, kvh, ls_glob, qa, qsc);
}

SYCL_ESIMD_FUNCTION inline simd<float, 8> init_m_r8(simd<int, 8> hi_v) {
    simd<float, 8> m(kNeg);
    m.merge(simd<float, 8>(1e30f), hi_v <= 0);
    return m;
}

SYCL_ESIMD_FUNCTION inline int compute_min_active_hi(simd<int, 8> hi_v) {
    simd<int, 8> hm = hi_v;
    hm.merge(simd<int, 8>(0x3fffffff), hi_v <= 0);
    return hmin8_i(hm);
}

SYCL_ESIMD_FUNCTION inline void coop_prefetch_kv(
    const uint32_t* kb, const uint32_t* vb, int blk_pf, int ub, int tid, int W) {
    const int p0 = blk_pf * kBlk;
    if (p0 + kBlk > ub) return;
    for (int c = tid; c < 32; c += W) {
        es::prefetch<uint32_t, 64>(kb + (size_t) (p0 + c) * 64u, kPf);
        es::prefetch<uint32_t, 64>(vb + (size_t) blk_pf * 2048u + (size_t) c * 64u, kPf);
    }
}

SYCL_ESIMD_FUNCTION inline void coop_load_k_slm(
    const uint32_t* kb, int p0, int ub, bool full_kv_blk, int tid, int W, uint32_t slm_k_base, int zero_fence) {
    for (int c = tid; c < 32; c += W) {
        simd<uint32_t, 64> kc(0u);
        if (full_kv_blk || p0 + c < ub) {
            kc = block_load<uint32_t, 64>(kb + (size_t) (p0 + c) * 64u + (size_t) zero_fence);
        }
        slm_block_store<uint32_t, 64>(slm_k_base + (uint32_t) c * 256u, kc, kAl16);
    }
}

SYCL_ESIMD_FUNCTION inline void coop_load_kb_slm(
    const uint32_t* kb, int p0, int ub, bool full_kv_blk, int tid, int W, uint32_t slm_k_base, int zero_fence) {
    for (int c = tid; c < 32; c += W) {
        simd<uint32_t, 64> kc(0u);
        if (full_kv_blk || p0 + c < ub) {
            kc = block_load<uint32_t, 64>(kb + (size_t) (p0 + c) * 64u + (size_t) zero_fence);
        }
        const uint32_t ch = (uint32_t) c >> 4;
        const uint32_t c16 = (uint32_t) c & 15u;
        const simd<uint32_t, 16> offs_base(slm_k_base + (ch * 1024u + c16) * 4u, 16u * 4u);
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {
            slm_scatter<uint32_t, 16>(offs_base + (uint32_t) q4 * 1024u, kc.template select<16, 1>(16 * q4));
        }
    }
}

SYCL_ESIMD_FUNCTION inline void coop_load_vb_slm(
    const uint32_t* vb, int blk, int tid, int W, uint32_t slm_v_base, int zero_fence) {
    const uint32_t* vb_blk = vb + (size_t) blk * 2048u + (size_t) zero_fence;
    for (int dl = tid; dl < 16; dl += W) {
        const uint32_t* vp = vb_blk + (size_t) dl * 128u;
        simd<uint32_t, 128> vl, vb_t;
        vl.template select<64, 1>(0) = block_load<uint32_t, 64>(vp);
        vl.template select<64, 1>(64) = block_load<uint32_t, 64>(vp + 64);
#pragma unroll
        for (int dw = 0; dw < 8; ++dw) vb_t.template select<16, 1>(16 * dw) = vl.template select<16, 8>(dw);
        slm_block_store<uint32_t, 128>(slm_v_base + (uint32_t) dl * 512u, vb_t, kAl16);
    }
}

SYCL_ESIMD_FUNCTION inline simd<float, 32> load_vs32_safe(const float* vsb, int p0, int ub, bool full_kv_blk) {
    if (full_kv_blk) return block_load<float, 32>(vsb + p0);
    simd<float, 32> vs32(0.0f);
    for (int c = 0; c < 32; ++c) {
        if (p0 + c < ub) vs32[c] = vsb[p0 + c];
    }
    return vs32;
}

// e^x for x in [-87, 0] by its series (1.2e-7, exp16's): STRATA_Q36_XATT2_EXP=1 takes it instead of the hardware exp2 in the
// softmax (the u16 parity missed its 3e-4 bound by 2x; a per-cell relative error of the exp of eps costs about eps of the
// output - measured in a float32 emulation of the kernel: eps 1e-4 -> 1e-4, 1e-3 -> 1e-3).  The parity's exp probe
// (--xatt2, stage 0) prints both errors.
template <int N>
SYCL_ESIMD_FUNCTION inline simd<float, N> exp_series(simd<float, N> x) {
    simd<float, N> y = x * 1.44269504f;
    simd<int, N> n = convert<int>(y - 0.5f);
    simd<float, N> nf = n;
    simd<float, N> z = (y - nf) * 0.693147181f;
    simd<float, N> e = 1.0f + z * (1.0f + z * (0.5f + z * (0.16666667f + z * (0.041666668f + z * (0.008333334f + z * 0.0013888889f)))));
    simd<int, N> b = (n + 127) << 23;
    simd<float, N> sc = b.template bit_cast_view<float>();
    return e * sc;
}

// Shared online softmax + per-row vmx=max(p*vs) quantization + native stride-2 byte pack into pa_lo / pa_hi.
// Never contains a barrier(). Leaves o16 untouched when no row exceeds its running max (chg == false).
template <int U16>
SYCL_ESIMD_FUNCTION inline void softmax_pack_r8(
    simd<float, 16> sc0[8], simd<float, 16> sc1[8], simd<float, 32> vs32,
    int p0, bool full_row_blk, bool has_tmask, int p0_tree,
    simd<int, 8> hi_v, simd<uint32_t, 8> tmk_v,
    simd<float, 128> o16[16], simd<float, 8>& m, simd<float, 8>& l,
    simd<uint32_t, 64>& pa_lo, simd<uint32_t, 64>& pa_hi, simd<float, 8>& srw,
    bool& row_valid, bool exp_poly) {
    if (!full_row_blk) {
        const simd<int, 16> c_idx0(p0, 1);
        const simd<int, 16> c_idx1 = c_idx0 + 16;
        const bool check_tmask = has_tmask && (p0 + kBlk > p0_tree);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int hi_r = hi_v[r];
            simd_mask<16> m0 = (c_idx0 >= hi_r);
            simd_mask<16> m1 = (c_idx1 >= hi_r);
            if (check_tmask) {
                const uint32_t tr = tmk_v[r];
                simd<int, 16> u0 = c_idx0 - p0_tree;
                simd<int, 16> u1 = c_idx1 - p0_tree;
                simd<uint32_t, 16> su0 = (es::min)((es::max)(u0, simd<int, 16>(0)), simd<int, 16>(31));
                simd<uint32_t, 16> su1 = (es::min)((es::max)(u1, simd<int, 16>(0)), simd<int, 16>(31));
                m0 = m0 | ((u0 >= 0) & ((u0 >= 32) | (((simd<uint32_t, 16>(tr) >> su0) & 1u) == 0u)));
                m1 = m1 | ((u1 >= 0) & ((u1 >= 32) | (((simd<uint32_t, 16>(tr) >> su1) & 1u) == 0u)));
            }
            sc0[r].merge(simd<float, 16>(kNeg), m0);
            sc1[r].merge(simd<float, 16>(kNeg), m1);
        }
    }

    simd<float, 8> m_blk(kNeg);
#pragma unroll
    for (int r = 0; r < 8; ++r) m_blk[r] = hmax<float>((es::max)(sc0[r], sc1[r]));
    m_blk.merge(simd<float, 8>(kNeg), hi_v <= 0);
    row_valid = (hmax<float>(m_blk) > -1e29f);
    if (!row_valid) return;

    simd<float, 8> m_new = (es::max)(m, m_blk);
    const bool chg = (hmax<float>(m_blk - m) > 0.0f);
    if (chg) {
        const simd<float, 8> dm = (es::max)(m - m_new, simd<float, 8>(-87.0f));
        simd<float, 8> alpha = exp_poly ? exp_series<8>(dm) : es::exp2(dm * kLog2e2);
        alpha.merge(simd<float, 8>(0.0f), m <= -1e29f);
        l *= alpha;
        simd<float, 128> a128;
#pragma unroll
        for (int r = 0; r < 8; ++r) a128.template select<16, 1>(16 * r) = float(alpha[r]);
#pragma unroll
        for (int dl = 0; dl < 16; ++dl) o16[dl] *= a128;
        m = m_new;
    }

    const float max_q = U16 ? 65535.0f : 255.0f;
    simd<float, 8> l_blk(0.0f);
    simd<float, 8> vmx(0.0f);
    pa_lo = 0u;
    if constexpr (U16) pa_hi = 0u;
    auto pa_lo_u8 = pa_lo.template bit_cast_view<uint8_t>();
    auto pa_hi_u8 = pa_hi.template bit_cast_view<uint8_t>();

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        simd<float, 32> sr;
        sr.template select<16, 1>(0) = sc0[r];
        sr.template select<16, 1>(16) = sc1[r];
        const simd<float, 32> dx = (es::max)(sr - float(m_new[r]), simd<float, 32>(-87.0f));
        simd<float, 32> e = exp_poly ? exp_series<32>(dx) : es::exp2(dx * kLog2e2);
        if (!full_row_blk) e.merge(simd<float, 32>(0.0f), sr < -1e29f);
        l_blk[r] = hsum32_f(e);
        simd<float, 32> pvr = e * vs32;
        if (!full_row_blk) pvr.merge(simd<float, 32>(0.0f), sr < -1e29f);
        const float r_vmx = hmax<float>(pvr);
        vmx[r] = r_vmx;
        const float inv_r = r_vmx > 1e-30f ? (max_q / r_vmx) : 0.0f;
        simd<uint16_t, 32> q16 = convert<uint16_t>((es::min)((es::max)(pvr * inv_r + 0.5f, simd<float, 32>(0.0f)), simd<float, 32>(max_q)));
        auto qb_r = q16.template bit_cast_view<uint8_t>();
        pa_lo_u8.template select<32, 1>(32 * r) = qb_r.template select<32, 2>(0);
        if constexpr (U16) {
            pa_hi_u8.template select<32, 1>(32 * r) = qb_r.template select<32, 2>(1);
        }
    }
    l += l_blk;
    srw = vmx * (1.0f / max_q);
}

// Accumulate 8 rows x 256 dims using transposed V_B in SLM.
template <int U16>
SYCL_ESIMD_FUNCTION inline void accum_vb_slm_r8_blk(
    uint32_t slm_v_base, simd<uint32_t, 64>& pa_lo, simd<uint32_t, 64>& pa_hi,
    simd<float, 8> srw, simd<float, 128> o16[16], int zero_fence, float* fence_ptr) {
    simd<float, 128> srw128;
#pragma unroll
    for (int r = 0; r < 8; ++r) srw128.template select<16, 1>(16 * r) = float(srw[r]);
    const simd<int, 64> pa_l = pa_lo.template bit_cast_view<int>();
    const simd<int, 64> pa_h = pa_hi.template bit_cast_view<int>();
#pragma unroll
    for (int dl = 0; dl < 16; ++dl) {
        simd<int, 128> VB = slm_block_load<int, 128>(slm_v_base + (uint32_t) dl * 512u, kAl16);
        simd<int, 128> ov = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::u8>(simd<int, 128>(0), VB, pa_l);
        if constexpr (U16) {
            simd<int, 128> ovh = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::u8>(simd<int, 128>(0), VB, pa_h);
            ov = ovh * 256 + ov;
        }
        o16[dl] += simd<float, 128>(ov) * srw128;
        if ((dl & 7) == 7 && zero_fence == -10 - dl) fence_ptr[0] = 0.0f;
    }
}

// Accumulate 8 rows x 256 dims directly from DRAM/L1 with in-register V_B transpose & 1 KiB ring[2] overlap.
template <int U16>
SYCL_ESIMD_FUNCTION inline void accum_vb_dram_r8_blk(
    const uint32_t* vb, int blk, bool do_pf, int blk_pf,
    simd<uint32_t, 64>& pa_lo, simd<uint32_t, 64>& pa_hi,
    simd<float, 8> srw, simd<float, 128> o16[16], int zero_fence, float* fence_ptr) {
    simd<float, 128> srw128;
#pragma unroll
    for (int r = 0; r < 8; ++r) srw128.template select<16, 1>(16 * r) = float(srw[r]);
    const simd<int, 64> pa_l = pa_lo.template bit_cast_view<int>();
    const simd<int, 64> pa_h = pa_hi.template bit_cast_view<int>();
    const uint32_t* vb_blk = vb + (size_t) blk * 2048u + (size_t) zero_fence;
    const uint32_t* vb_pf = vb + (size_t) blk_pf * 2048u;

    simd<uint32_t, 128> vl_ring[2];
    vl_ring[0].template select<64, 1>(0) = block_load<uint32_t, 64>(vb_blk);
    vl_ring[0].template select<64, 1>(64) = block_load<uint32_t, 64>(vb_blk + 64);
    vl_ring[1].template select<64, 1>(0) = block_load<uint32_t, 64>(vb_blk + 128);
    vl_ring[1].template select<64, 1>(64) = block_load<uint32_t, 64>(vb_blk + 192);

#pragma unroll
    for (int dl = 0; dl < 16; ++dl) {
        simd<uint32_t, 128> vb_t;
#pragma unroll
        for (int dw = 0; dw < 8; ++dw) vb_t.template select<16, 1>(16 * dw) = vl_ring[dl & 1].template select<16, 8>(dw);
        if (dl + 2 < 16) {
            const uint32_t* vp_n = vb_blk + (size_t) (dl + 2) * 128u;
            vl_ring[dl & 1].template select<64, 1>(0) = block_load<uint32_t, 64>(vp_n);
            vl_ring[dl & 1].template select<64, 1>(64) = block_load<uint32_t, 64>(vp_n + 64);
        }
        if (do_pf) {
            es::prefetch<uint32_t, 64>(vb_pf + (size_t) dl * 128u, kPf);
            es::prefetch<uint32_t, 64>(vb_pf + (size_t) dl * 128u + 64u, kPf);
        }
        const simd<int, 128> VB = vb_t.template bit_cast_view<int>();
        simd<int, 128> ov = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::u8>(simd<int, 128>(0), VB, pa_l);
        if constexpr (U16) {
            simd<int, 128> ovh = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::u8>(simd<int, 128>(0), VB, pa_h);
            ov = ovh * 256 + ov;
        }
        o16[dl] += simd<float, 128>(ov) * srw128;
    }
    if (zero_fence == -20) fence_ptr[0] = 0.0f;
}

// Score 32 cells x 8 rows from DRAM K using permanent Q_A[8] (32 GRFs) and 4-cell kc[4] + L1 prefetch + dpas<8,8>.
template <int NS>
SYCL_ESIMD_FUNCTION inline void score_kraw_dram_r8_4c_blk(
    const simd<int, 64> qa[8], simd<float, 8> qsc,
    const uint32_t* kb, const float* ksb, const uint32_t* ks16b,
    int p0, int ub, bool full_kv_blk, bool do_pf, int p0_pf, int zero_fence, float* fence_ptr,
    simd<float, 16> sc0[8], simd<float, 16> sc1[8]) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int p = p0 + 4 * j;
        simd<uint32_t, 64> kc[4];
#pragma unroll
        for (int c4 = 0; c4 < 4; ++c4) {
            if (full_kv_blk || p + c4 < ub) {
                kc[c4] = block_load<uint32_t, 64>(kb + (size_t) (p + c4) * 64u + (size_t) zero_fence);
            } else {
                kc[c4] = 0u;
            }
        }
        if (j + 1 < 8) {
            const int pn = p0 + 4 * (j + 1);
            if (full_kv_blk || pn + 3 < ub) {
#pragma unroll
                for (int c4 = 0; c4 < 4; ++c4) es::prefetch<uint32_t, 64>(kb + (size_t) (pn + c4) * 64u, kPf);
            }
        }
        if (do_pf) {
#pragma unroll
            for (int c4 = 0; c4 < 4; ++c4) es::prefetch<uint32_t, 64>(kb + (size_t) (p0_pf + 4 * j + c4) * 64u, kPf);
        }
        simd<float, 4> ks4(0.0f);
        simd<float, 16> kf(0.0f);
        if (full_kv_blk) {
            if constexpr (NS == 1) {
#pragma unroll
                for (int c4 = 0; c4 < 4; ++c4) ks4[c4] = ksb[p + c4];
            } else {
                simd<uint32_t, 8> w8 = block_load<uint32_t, 8>(ks16b + (size_t) p * 2u);
                kf = w8.template bit_cast_view<sycl::half>();
            }
        } else {
#pragma unroll
            for (int c4 = 0; c4 < 4; ++c4) {
                if (p + c4 < ub) {
                    if constexpr (NS == 1) ks4[c4] = ksb[p + c4];
                    else {
                        simd<uint32_t, 2> w2 = block_load<uint32_t, 2>(ks16b + (size_t) (p + c4) * 2u);
                        simd<sycl::half, 4> h4 = w2.template bit_cast_view<sycl::half>();
                        kf.template select<4, 1>(4 * c4) = h4;
                    }
                }
            }
        }

        simd<int, 128> acc(0);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            simd<int, 128> KB(0);
#pragma unroll
            for (int c4 = 0; c4 < 4; ++c4) {
                KB.template select<8, 16>(c4) = kc[c4].template select<8, 1>(8 * s).template bit_cast_view<int>();
            }
            acc = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(acc, KB, qa[s]);
            if constexpr (NS == 4) {
                if ((s & 1) == 1) {
                    const int g = s >> 1;
                    const simd<float, 4> ksc = kf.template select<4, 4>(g);
                    const int sc_off = (j & 3) * 4;
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const simd<float, 4> t = simd<float, 4>(acc.template select<4, 1>(16 * r)) * ksc;
                        if (j < 4) {
                            if (g == 0) sc0[r].template select<4, 1>(sc_off) = t;
                            else        sc0[r].template select<4, 1>(sc_off) += t;
                        } else {
                            if (g == 0) sc1[r].template select<4, 1>(sc_off) = t;
                            else        sc1[r].template select<4, 1>(sc_off) += t;
                        }
                    }
                    if (g < 3) acc = 0;
                }
            }
        }
        if constexpr (NS == 1) {
            const int sc_off = (j & 3) * 4;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const simd<float, 4> t = simd<float, 4>(acc.template select<4, 1>(16 * r)) * (ks4 * float(qsc[r]));
                if (j < 4) sc0[r].template select<4, 1>(sc_off) = t;
                else       sc1[r].template select<4, 1>(sc_off) = t;
            }
        }
    }
    if constexpr (NS == 4) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            sc0[r] *= float(qsc[r]);
            sc1[r] *= float(qsc[r]);
        }
    }
    if (zero_fence == -30) fence_ptr[0] = 0.0f;
}

// Score 32 cells x 8 rows from DRAM K using permanent Q_A[8] (32 GRFs) and 8-cell kc[8] + dpas<8,8>.
template <int NS>
SYCL_ESIMD_FUNCTION inline void score_kraw_dram_r8_8c_blk(
    const simd<int, 64> qa[8], simd<float, 8> qsc,
    const uint32_t* kb, const float* ksb, const uint32_t* ks16b,
    int p0, int ub, bool full_kv_blk, bool do_pf, int p0_pf, int zero_fence, float* fence_ptr,
    simd<float, 16> sc0[8], simd<float, 16> sc1[8]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int p = p0 + 8 * j;
        simd<uint32_t, 64> kc[8];
#pragma unroll
        for (int c8 = 0; c8 < 8; ++c8) {
            if (full_kv_blk || p + c8 < ub) {
                kc[c8] = block_load<uint32_t, 64>(kb + (size_t) (p + c8) * 64u + (size_t) zero_fence);
            } else {
                kc[c8] = 0u;
            }
        }
        if (do_pf) {
#pragma unroll
            for (int c8 = 0; c8 < 8; ++c8) es::prefetch<uint32_t, 64>(kb + (size_t) (p0_pf + 8 * j + c8) * 64u, kPf);
        }
        simd<float, 8> ks8(0.0f);
        simd<float, 32> kf(0.0f);
        if (full_kv_blk) {
            if constexpr (NS == 1) {
                ks8 = block_load<float, 8>(ksb + p);
            } else {
                simd<uint32_t, 16> w16 = block_load<uint32_t, 16>(ks16b + (size_t) p * 2u);
                kf = w16.template bit_cast_view<sycl::half>();
            }
        } else {
#pragma unroll
            for (int c8 = 0; c8 < 8; ++c8) {
                if (p + c8 < ub) {
                    if constexpr (NS == 1) ks8[c8] = ksb[p + c8];
                    else {
                        simd<uint32_t, 2> w2 = block_load<uint32_t, 2>(ks16b + (size_t) (p + c8) * 2u);
                        simd<sycl::half, 4> h4 = w2.template bit_cast_view<sycl::half>();
                        kf.template select<4, 1>(4 * c8) = h4;
                    }
                }
            }
        }

        simd<int, 128> acc(0);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            simd<int, 128> KB(0);
#pragma unroll
            for (int c8 = 0; c8 < 8; ++c8) {
                KB.template select<8, 16>(c8) = kc[c8].template select<8, 1>(8 * s).template bit_cast_view<int>();
            }
            acc = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(acc, KB, qa[s]);
            if constexpr (NS == 4) {
                if ((s & 1) == 1) {
                    const int g = s >> 1;
                    const simd<float, 8> ksc = kf.template select<8, 4>(g);
                    const int sc_off = (j & 1) * 8;
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const simd<float, 8> t = simd<float, 8>(acc.template select<8, 1>(16 * r)) * ksc;
                        if (j < 2) {
                            if (g == 0) sc0[r].template select<8, 1>(sc_off) = t;
                            else        sc0[r].template select<8, 1>(sc_off) += t;
                        } else {
                            if (g == 0) sc1[r].template select<8, 1>(sc_off) = t;
                            else        sc1[r].template select<8, 1>(sc_off) += t;
                        }
                    }
                    if (g < 3) acc = 0;
                }
            }
        }
        if constexpr (NS == 1) {
            const int sc_off = (j & 1) * 8;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const simd<float, 8> t = simd<float, 8>(acc.template select<8, 1>(16 * r)) * (ks8 * float(qsc[r]));
                if (j < 2) sc0[r].template select<8, 1>(sc_off) = t;
                else       sc1[r].template select<8, 1>(sc_off) = t;
            }
        }
    }
    if constexpr (NS == 4) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            sc0[r] *= float(qsc[r]);
            sc1[r] *= float(qsc[r]);
        }
    }
    if (zero_fence == -30) fence_ptr[0] = 0.0f;
}

// Score 32 cells x 8 rows from untransposed K in SLM using permanent Q_A[8] (32 GRFs) + dpas<8,8>.
template <int NS>
SYCL_ESIMD_FUNCTION inline void score_kraw_slm_r8_4c_blk(
    const simd<int, 64> qa[8], simd<float, 8> qsc,
    uint32_t slm_k_base, const float* ksb, const uint32_t* ks16b,
    int p0, int ub, bool full_kv_blk, int zero_fence, float* fence_ptr,
    simd<float, 16> sc0[8], simd<float, 16> sc1[8]) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int p = p0 + 4 * j;
        simd<uint32_t, 64> kc[4];
#pragma unroll
        for (int c4 = 0; c4 < 4; ++c4) kc[c4] = slm_block_load<uint32_t, 64>(slm_k_base + (uint32_t) (4 * j + c4) * 256u, kAl16);

        simd<float, 4> ks4(0.0f);
        simd<float, 16> kf(0.0f);
        if (full_kv_blk) {
            if constexpr (NS == 1) {
#pragma unroll
                for (int c4 = 0; c4 < 4; ++c4) ks4[c4] = ksb[p + c4];
            } else {
                simd<uint32_t, 8> w8 = block_load<uint32_t, 8>(ks16b + (size_t) p * 2u);
                kf = w8.template bit_cast_view<sycl::half>();
            }
        } else {
#pragma unroll
            for (int c4 = 0; c4 < 4; ++c4) {
                if (p + c4 < ub) {
                    if constexpr (NS == 1) ks4[c4] = ksb[p + c4];
                    else {
                        simd<uint32_t, 2> w2 = block_load<uint32_t, 2>(ks16b + (size_t) (p + c4) * 2u);
                        simd<sycl::half, 4> h4 = w2.template bit_cast_view<sycl::half>();
                        kf.template select<4, 1>(4 * c4) = h4;
                    }
                }
            }
        }

        simd<int, 128> acc(0);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            simd<int, 128> KB(0);
#pragma unroll
            for (int c4 = 0; c4 < 4; ++c4) {
                KB.template select<8, 16>(c4) = kc[c4].template select<8, 1>(8 * s).template bit_cast_view<int>();
            }
            acc = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(acc, KB, qa[s]);
            if constexpr (NS == 4) {
                if ((s & 1) == 1) {
                    const int g = s >> 1;
                    const simd<float, 4> ksc = kf.template select<4, 4>(g);
                    const int sc_off = (j & 3) * 4;
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const simd<float, 4> t = simd<float, 4>(acc.template select<4, 1>(16 * r)) * ksc;
                        if (j < 4) {
                            if (g == 0) sc0[r].template select<4, 1>(sc_off) = t;
                            else        sc0[r].template select<4, 1>(sc_off) += t;
                        } else {
                            if (g == 0) sc1[r].template select<4, 1>(sc_off) = t;
                            else        sc1[r].template select<4, 1>(sc_off) += t;
                        }
                    }
                    if (g < 3) acc = 0;
                }
            }
        }
        if constexpr (NS == 1) {
            const int sc_off = (j & 3) * 4;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const simd<float, 4> t = simd<float, 4>(acc.template select<4, 1>(16 * r)) * (ks4 * float(qsc[r]));
                if (j < 4) sc0[r].template select<4, 1>(sc_off) = t;
                else       sc1[r].template select<4, 1>(sc_off) = t;
            }
        }
    }
    if constexpr (NS == 4) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            sc0[r] *= float(qsc[r]);
            sc1[r] *= float(qsc[r]);
        }
    }
    if (zero_fence == -30) fence_ptr[0] = 0.0f;
}

// Score 32 cells x 8 rows from transposed K_B in SLM using permanent Q_A[8] (16 total dpas<8,8>).
template <int NS>
SYCL_ESIMD_FUNCTION inline void score_kb_slm_r8_blk(
    const simd<int, 64> qa[8], simd<float, 8> qsc,
    uint32_t slm_k_base, const float* ksb, const uint32_t* ks16b,
    int p0, int ub, bool full_kv_blk, int zero_fence, float* fence_ptr,
    simd<float, 16> sc0[8], simd<float, 16> sc1[8]) {
#pragma unroll
    for (int ch = 0; ch < 2; ++ch) {
        const int pc = p0 + 16 * ch;
        simd<float, 16> ks16f(0.0f);
        simd<float, 64> kf(0.0f);
        if (full_kv_blk) {
            if constexpr (NS == 1) {
                ks16f = block_load<float, 16>(ksb + pc);
            } else {
                simd<uint32_t, 32> w32 = block_load<uint32_t, 32>(ks16b + (size_t) pc * 2u);
                kf = w32.template bit_cast_view<sycl::half>();
            }
        } else {
            for (int c16 = 0; c16 < 16; ++c16) {
                if (pc + c16 < ub) {
                    if constexpr (NS == 1) ks16f[c16] = ksb[pc + c16];
                    else {
                        simd<uint32_t, 2> w2 = block_load<uint32_t, 2>(ks16b + (size_t) (pc + c16) * 2u);
                        simd<sycl::half, 4> h4 = w2.template bit_cast_view<sycl::half>();
                        kf.template select<4, 1>(4 * c16) = h4;
                    }
                }
            }
        }

        simd<int, 128> acc(0);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            simd<int, 128> KB = slm_block_load<int, 128>(slm_k_base + (uint32_t) (ch * 8 + s) * 512u, kAl16);
            acc = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(acc, KB, qa[s]);
            if constexpr (NS == 4) {
                if ((s & 1) == 1) {
                    const int g = s >> 1;
                    simd<float, 16> ksc = kf.template select<16, 4>(g);
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const simd<float, 16> t = simd<float, 16>(acc.template select<16, 1>(16 * r)) * ksc;
                        if (ch == 0) {
                            if (g == 0) sc0[r] = t;
                            else        sc0[r] += t;
                        } else {
                            if (g == 0) sc1[r] = t;
                            else        sc1[r] += t;
                        }
                    }
                    if (g < 3) acc = 0;
                }
            }
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if constexpr (NS == 1) {
                const simd<float, 16> t = simd<float, 16>(acc.template select<16, 1>(16 * r)) * (ks16f * float(qsc[r]));
                if (ch == 0) sc0[r] = t;
                else         sc1[r] = t;
            } else {
                if (ch == 0) sc0[r] *= float(qsc[r]);
                else         sc1[r] *= float(qsc[r]);
            }
        }
        if (zero_fence == -30 - ch) fence_ptr[0] = 0.0f;
    }
}

// Form 0: R8 (0 SLM, W = (max_rows+7)/8 threads/WG; lockstep L1 share when W>=2; Q_A[8] + 4-cell dpas<8,8> + in-reg V_B).
template <int NS, int U16> struct TagXa2R8;
template <int NS, int U16> void launch_r8(sycl::queue& q, const AttnPlan2& pl) {
    const uint32_t* k8 = pl.k8;
    const float* ks = pl.ks;
    const uint16_t* ks16 = pl.ks16;
    const uint32_t* v8t = pl.v8t;
    const float* vs = pl.vs;
    const int32_t* pos0 = pl.pos0;
    const uint32_t* tmask = pl.tmask;
    const int32_t* tp0 = pl.tp0;
    void* const* k_tab = pl.k_tab;
    void* const* v_tab = pl.v_tab;
    const void* const* mir_tab = pl.mir_tab;
    uint32_t* qb = pl.qb;
    float* qs = pl.qs;
    float* pacc = pl.pacc;
    float* pm = pl.pm;
    float* pl_ptr = pl.pl;
    uint32_t* dbg_flag = pl.dbg_flag;
    const int kst = pl.kst, ctxp = pl.ctxp, W = pl.wg_threads, pfd = pl.pfd, zero_fence = pl.zero_fence;
    const int nseg_stride = pl.nseg_stride, t0 = pl.t0, H = pl.H, HKV = pl.HKV;
    const uint32_t dbg = pl.dbg, k8_max_cells = pl.k8_max_cells, v8t_max_blks = pl.v8t_max_blks, vs_max_cells = pl.vs_max_cells;
    const uint64_t qb_dwords = pl.qb_dwords, qs_floats = pl.qs_floats, p_elems = pl.p_elems, pacc_elems = pl.pacc_elems;
    const int ngrp = pl.ngrp, per_tok_pos = pl.per_tok_pos;
    const bool exp_poly = pl.exp_poly != 0;
    const uint64_t gp0 = pl.gp0, gp1 = pl.gp1, gp2 = pl.gp2, gp3 = pl.gp3;
    const uint64_t gp4 = pl.gp4, gp5 = pl.gp5, gp6 = pl.gp6, gp7 = pl.gp7;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagXa2R8<NS, U16>>(
            sycl::nd_range<3>(sycl::range<3>((size_t) W, (size_t) HKV, (size_t) pl.total_tasks),
                              sycl::range<3>((size_t) W, 1, 1)),
            [=](sycl::nd_item<3> it) SYCL_ESIMD_KERNEL {
                const int tid = (int) it.get_local_id(0);
                const int kvh = (int) it.get_global_id(1);
                const int task = (int) it.get_global_id(2);
                const int r_base = tid * 8;
                int gi = 0, seg = 0, gp_slot = 0, gp_r0 = 0, gp_cnt = 0, gp_ls0 = 0, gp_nrg = 0, gp_nseg = 0;
                decode_task_grp(task, ngrp, gp0, gp1, gp2, gp3, gp4, gp5, gp6, gp7,
                                gi, seg, gp_slot, gp_r0, gp_cnt, gp_ls0, gp_nrg, gp_nseg);

                const uint32_t* kb = nullptr;
                const float* ksb = nullptr;
                const uint32_t* ks16b = nullptr;
                const uint32_t* vb = nullptr;
                const float* vsb = nullptr;
                load_grp_ptrs_raw<NS>(k8, ks, ks16, v8t, vs, k_tab, v_tab, mir_tab,
                                      HKV, kst, ctxp, gp_slot, kvh, kb, ksb, ks16b, vb, vsb);

                const bool thread_has_rows = (r_base < gp_cnt * kG);
                simd<int, 8> hi_v(0);
                simd<uint32_t, 8> tmk_v(0u);
                int ub = 0, p0_tree = 0;
                bool has_tmask = false;
                load_row_bounds_raw<8>(pos0, tmask, tp0, t0, ngrp, per_tok_pos, gp_r0, gp_cnt, r_base,
                                       hi_v, tmk_v, ub, has_tmask, p0_tree);

                int blk_a = 0, blk_b = 0;
                seg_block_range(ub, seg, gp_nseg, blk_a, blk_b);
                const int ls_glob = thread_has_rows ? (gp_ls0 + (r_base >> 4)) : gp_ls0;
                const bool oob = dbg_clamp_seg_bounds_raw(
                    dbg, dbg_flag, HKV, k8_max_cells, v8t_max_blks, vs_max_cells,
                    qb_dwords, qs_floats, kvh, ls_glob, ub, blk_a, blk_b);

                simd<float, 128> o16[16];
#pragma unroll
                for (int k = 0; k < 16; ++k) o16[k] = 0.0f;
                simd<float, 8> m = init_m_r8(hi_v);
                simd<float, 8> l(0.0f);

                if (blk_a < blk_b && !oob) {
                    simd<int, 64> qa[8];
                    simd<float, 8> qsc(0.0f);
                    if (thread_has_rows) {
                        load_qa8_and_scale(qb, qs, kvh, ls_glob, r_base & 8, qa, qsc);
                    }
                    const int min_hi = compute_min_active_hi(hi_v);

                    for (int blk = blk_a; blk < blk_b; ++blk) {
                        if (W > 1 && blk > blk_a && ((blk - blk_a) & 1) == 0) barrier();
                        const int p0 = blk * kBlk;
                        const bool full_kv_blk = (p0 + kBlk <= ub);
                        const int blk_pf = blk + pfd;
                        const bool pf_ok = (pfd > 0) && (blk_pf < blk_b) && (blk_pf * kBlk + kBlk <= ub);
                        if (W > 1 && pf_ok) coop_prefetch_kv(kb, vb, blk_pf, ub, tid, W);

                        if (thread_has_rows) {
                            const bool full_row_blk = (p0 + kBlk <= min_hi) && !(has_tmask && p0 + kBlk > p0_tree);
                            simd<float, 16> sc0[8], sc1[8];
                            score_kraw_dram_r8_4c_blk<NS>(
                                qa, qsc, kb, ksb, ks16b, p0, ub, full_kv_blk,
                                (W == 1) && pf_ok, blk_pf * kBlk, zero_fence, pl_ptr, sc0, sc1);
                            if (zero_fence == -98) pl_ptr[0] = 0.0f;
                            const simd<float, 32> vs32 = load_vs32_safe(vsb, p0, ub, full_kv_blk);
                            simd<uint32_t, 64> pa_lo, pa_hi;
                            simd<float, 8> srw;
                            bool row_valid = false;
                            softmax_pack_r8<U16>(
                                sc0, sc1, vs32, p0, full_row_blk, has_tmask, p0_tree,
                                hi_v, tmk_v, o16, m, l, pa_lo, pa_hi, srw, row_valid, exp_poly);
                            if (zero_fence == -99) pl_ptr[0] = 0.0f;
                            if (row_valid) {
                                accum_vb_dram_r8_blk<U16>(
                                    vb, blk, (W == 1) && pf_ok, blk_pf, pa_lo, pa_hi, srw, o16, zero_fence, pl_ptr);
                            }
                        }
                    }
                }
                if (thread_has_rows) {
                    store_partials_o16_raw(pacc, pm, pl_ptr, dbg, dbg_flag, p_elems, pacc_elems, H, nseg_stride, gp_r0, gp_cnt, seg, kvh, r_base, o16, m, l, oob);
                }
            });
    });
}

// Form 1 (W==1) & Form 3 (W==1): R8D8 (0 SLM, W=1, Q_A[8] + 8-cell dpas<8,8> kc[8] + in-reg V_B).
template <int NS, int U16> struct TagXa2R8D8;
template <int NS, int U16> void launch_r8_d8(sycl::queue& q, const AttnPlan2& pl) {
    const uint32_t* k8 = pl.k8;
    const float* ks = pl.ks;
    const uint16_t* ks16 = pl.ks16;
    const uint32_t* v8t = pl.v8t;
    const float* vs = pl.vs;
    const int32_t* pos0 = pl.pos0;
    const uint32_t* tmask = pl.tmask;
    const int32_t* tp0 = pl.tp0;
    void* const* k_tab = pl.k_tab;
    void* const* v_tab = pl.v_tab;
    const void* const* mir_tab = pl.mir_tab;
    uint32_t* qb = pl.qb;
    float* qs = pl.qs;
    float* pacc = pl.pacc;
    float* pm = pl.pm;
    float* pl_ptr = pl.pl;
    uint32_t* dbg_flag = pl.dbg_flag;
    const int kst = pl.kst, ctxp = pl.ctxp, pfd = pl.pfd, zero_fence = pl.zero_fence;
    const int nseg_stride = pl.nseg_stride, t0 = pl.t0, H = pl.H, HKV = pl.HKV;
    const uint32_t dbg = pl.dbg, k8_max_cells = pl.k8_max_cells, v8t_max_blks = pl.v8t_max_blks, vs_max_cells = pl.vs_max_cells;
    const uint64_t qb_dwords = pl.qb_dwords, qs_floats = pl.qs_floats, p_elems = pl.p_elems, pacc_elems = pl.pacc_elems;
    const int ngrp = pl.ngrp, per_tok_pos = pl.per_tok_pos;
    const bool exp_poly = pl.exp_poly != 0;
    const uint64_t gp0 = pl.gp0, gp1 = pl.gp1, gp2 = pl.gp2, gp3 = pl.gp3;
    const uint64_t gp4 = pl.gp4, gp5 = pl.gp5, gp6 = pl.gp6, gp7 = pl.gp7;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagXa2R8D8<NS, U16>>(
            sycl::nd_range<3>(sycl::range<3>(1, (size_t) HKV, (size_t) pl.total_tasks),
                              sycl::range<3>(1, 1, 1)),
            [=](sycl::nd_item<3> it) SYCL_ESIMD_KERNEL {
                const int kvh = (int) it.get_global_id(1);
                const int task = (int) it.get_global_id(2);
                int gi = 0, seg = 0, gp_slot = 0, gp_r0 = 0, gp_cnt = 0, gp_ls0 = 0, gp_nrg = 0, gp_nseg = 0;
                decode_task_grp(task, ngrp, gp0, gp1, gp2, gp3, gp4, gp5, gp6, gp7,
                                gi, seg, gp_slot, gp_r0, gp_cnt, gp_ls0, gp_nrg, gp_nseg);

                const uint32_t* kb = nullptr;
                const float* ksb = nullptr;
                const uint32_t* ks16b = nullptr;
                const uint32_t* vb = nullptr;
                const float* vsb = nullptr;
                load_grp_ptrs_raw<NS>(k8, ks, ks16, v8t, vs, k_tab, v_tab, mir_tab,
                                      HKV, kst, ctxp, gp_slot, kvh, kb, ksb, ks16b, vb, vsb);

                simd<int, 8> hi_v(0);
                simd<uint32_t, 8> tmk_v(0u);
                int ub = 0, p0_tree = 0;
                bool has_tmask = false;
                load_row_bounds_raw<8>(pos0, tmask, tp0, t0, ngrp, per_tok_pos, gp_r0, gp_cnt, 0,
                                       hi_v, tmk_v, ub, has_tmask, p0_tree);

                int blk_a = 0, blk_b = 0;
                seg_block_range(ub, seg, gp_nseg, blk_a, blk_b);
                const bool oob = dbg_clamp_seg_bounds_raw(
                    dbg, dbg_flag, HKV, k8_max_cells, v8t_max_blks, vs_max_cells,
                    qb_dwords, qs_floats, kvh, gp_ls0, ub, blk_a, blk_b);

                simd<float, 128> o16[16];
#pragma unroll
                for (int k = 0; k < 16; ++k) o16[k] = 0.0f;
                simd<float, 8> m = init_m_r8(hi_v);
                simd<float, 8> l(0.0f);

                if (blk_a < blk_b && !oob) {
                    simd<int, 64> qa[8];
                    simd<float, 8> qsc(0.0f);
                    load_qa8_and_scale(qb, qs, kvh, gp_ls0, 0, qa, qsc);
                    const int min_hi = compute_min_active_hi(hi_v);

                    for (int blk = blk_a; blk < blk_b; ++blk) {
                        const int p0 = blk * kBlk;
                        const bool full_kv_blk = (p0 + kBlk <= ub);
                        const bool full_row_blk = (p0 + kBlk <= min_hi) && !(has_tmask && p0 + kBlk > p0_tree);
                        const int blk_pf = blk + pfd;
                        const bool pf_ok = (pfd > 0) && (blk_pf < blk_b) && (blk_pf * kBlk + kBlk <= ub);

                        simd<float, 16> sc0[8], sc1[8];
                        score_kraw_dram_r8_8c_blk<NS>(
                            qa, qsc, kb, ksb, ks16b, p0, ub, full_kv_blk,
                            pf_ok, blk_pf * kBlk, zero_fence, pl_ptr, sc0, sc1);
                        if (zero_fence == -98) pl_ptr[0] = 0.0f;
                        const simd<float, 32> vs32 = load_vs32_safe(vsb, p0, ub, full_kv_blk);
                        simd<uint32_t, 64> pa_lo, pa_hi;
                        simd<float, 8> srw;
                        bool row_valid = false;
                        softmax_pack_r8<U16>(
                            sc0, sc1, vs32, p0, full_row_blk, has_tmask, p0_tree,
                            hi_v, tmk_v, o16, m, l, pa_lo, pa_hi, srw, row_valid, exp_poly);
                        if (zero_fence == -99) pl_ptr[0] = 0.0f;
                        if (row_valid) {
                            accum_vb_dram_r8_blk<U16>(
                                vb, blk, pf_ok, blk_pf, pa_lo, pa_hi, srw, o16, zero_fence, pl_ptr);
                        }
                    }
                }
                store_partials_o16_raw(pacc, pm, pl_ptr, dbg, dbg_flag, p_elems, pacc_elems, H, nseg_stride, gp_r0, gp_cnt, seg, kvh, 0, o16, m, l, oob);
            });
    });
}

// Form 1 (W>=2): RS16 (16 KiB Coop SLM: 0..8191 K_raw, 8192..16383 V_B, 2 barriers/blk, Q_A[8] + dpas<8,8> K/V).
template <int NS, int U16> struct TagXa2RS16;
template <int NS, int U16> void launch_rs16(sycl::queue& q, const AttnPlan2& pl) {
    const uint32_t* k8 = pl.k8;
    const float* ks = pl.ks;
    const uint16_t* ks16 = pl.ks16;
    const uint32_t* v8t = pl.v8t;
    const float* vs = pl.vs;
    const int32_t* pos0 = pl.pos0;
    const uint32_t* tmask = pl.tmask;
    const int32_t* tp0 = pl.tp0;
    void* const* k_tab = pl.k_tab;
    void* const* v_tab = pl.v_tab;
    const void* const* mir_tab = pl.mir_tab;
    uint32_t* qb = pl.qb;
    float* qs = pl.qs;
    float* pacc = pl.pacc;
    float* pm = pl.pm;
    float* pl_ptr = pl.pl;
    uint32_t* dbg_flag = pl.dbg_flag;
    const int kst = pl.kst, ctxp = pl.ctxp, W = pl.wg_threads, pfd = pl.pfd, zero_fence = pl.zero_fence;
    const int nseg_stride = pl.nseg_stride, t0 = pl.t0, H = pl.H, HKV = pl.HKV;
    const uint32_t dbg = pl.dbg, k8_max_cells = pl.k8_max_cells, v8t_max_blks = pl.v8t_max_blks, vs_max_cells = pl.vs_max_cells;
    const uint64_t qb_dwords = pl.qb_dwords, qs_floats = pl.qs_floats, p_elems = pl.p_elems, pacc_elems = pl.pacc_elems;
    const int ngrp = pl.ngrp, per_tok_pos = pl.per_tok_pos;
    const bool exp_poly = pl.exp_poly != 0;
    const uint64_t gp0 = pl.gp0, gp1 = pl.gp1, gp2 = pl.gp2, gp3 = pl.gp3;
    const uint64_t gp4 = pl.gp4, gp5 = pl.gp5, gp6 = pl.gp6, gp7 = pl.gp7;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagXa2RS16<NS, U16>>(
            sycl::nd_range<3>(sycl::range<3>((size_t) W, (size_t) HKV, (size_t) pl.total_tasks),
                              sycl::range<3>((size_t) W, 1, 1)),
            [=](sycl::nd_item<3> it) SYCL_ESIMD_KERNEL {
                slm_init<16384u>();
                const int tid = (int) it.get_local_id(0);
                const int kvh = (int) it.get_global_id(1);
                const int task = (int) it.get_global_id(2);
                const int r_base = tid * 8;
                int gi = 0, seg = 0, gp_slot = 0, gp_r0 = 0, gp_cnt = 0, gp_ls0 = 0, gp_nrg = 0, gp_nseg = 0;
                decode_task_grp(task, ngrp, gp0, gp1, gp2, gp3, gp4, gp5, gp6, gp7,
                                gi, seg, gp_slot, gp_r0, gp_cnt, gp_ls0, gp_nrg, gp_nseg);

                const uint32_t* kb = nullptr;
                const float* ksb = nullptr;
                const uint32_t* ks16b = nullptr;
                const uint32_t* vb = nullptr;
                const float* vsb = nullptr;
                load_grp_ptrs_raw<NS>(k8, ks, ks16, v8t, vs, k_tab, v_tab, mir_tab,
                                      HKV, kst, ctxp, gp_slot, kvh, kb, ksb, ks16b, vb, vsb);

                const bool thread_has_rows = (r_base < gp_cnt * kG);
                simd<int, 8> hi_v(0);
                simd<uint32_t, 8> tmk_v(0u);
                int ub = 0, p0_tree = 0;
                bool has_tmask = false;
                load_row_bounds_raw<8>(pos0, tmask, tp0, t0, ngrp, per_tok_pos, gp_r0, gp_cnt, r_base,
                                       hi_v, tmk_v, ub, has_tmask, p0_tree);

                int blk_a = 0, blk_b = 0;
                seg_block_range(ub, seg, gp_nseg, blk_a, blk_b);
                const int ls_glob = thread_has_rows ? (gp_ls0 + (r_base >> 4)) : gp_ls0;
                const bool oob = dbg_clamp_seg_bounds_raw(
                    dbg, dbg_flag, HKV, k8_max_cells, v8t_max_blks, vs_max_cells,
                    qb_dwords, qs_floats, kvh, ls_glob, ub, blk_a, blk_b);

                simd<float, 128> o16[16];
#pragma unroll
                for (int k = 0; k < 16; ++k) o16[k] = 0.0f;
                simd<float, 8> m = init_m_r8(hi_v);
                simd<float, 8> l(0.0f);

                if (blk_a < blk_b && !oob) {
                    simd<int, 64> qa[8];
                    simd<float, 8> qsc(0.0f);
                    if (thread_has_rows) {
                        load_qa8_and_scale(qb, qs, kvh, ls_glob, r_base & 8, qa, qsc);
                    }
                    const int min_hi = compute_min_active_hi(hi_v);

                    for (int blk = blk_a; blk < blk_b; ++blk) {
                        if (blk > blk_a) barrier();
                        const int p0 = blk * kBlk;
                        const bool full_kv_blk = (p0 + kBlk <= ub);
                        coop_load_k_slm(kb, p0, ub, full_kv_blk, tid, W, 0u, zero_fence);
                        coop_load_vb_slm(vb, blk, tid, W, 8192u, zero_fence);
                        barrier();
                        if (pfd > 0 && blk + pfd < blk_b) coop_prefetch_kv(kb, vb, blk + pfd, ub, tid, W);

                        if (thread_has_rows) {
                            const bool full_row_blk = (p0 + kBlk <= min_hi) && !(has_tmask && p0 + kBlk > p0_tree);
                            simd<float, 16> sc0[8], sc1[8];
                            score_kraw_slm_r8_4c_blk<NS>(
                                qa, qsc, 0u, ksb, ks16b, p0, ub, full_kv_blk, zero_fence, pl_ptr, sc0, sc1);
                            if (zero_fence == -98) pl_ptr[0] = 0.0f;
                            const simd<float, 32> vs32 = load_vs32_safe(vsb, p0, ub, full_kv_blk);
                            simd<uint32_t, 64> pa_lo, pa_hi;
                            simd<float, 8> srw;
                            bool row_valid = false;
                            softmax_pack_r8<U16>(
                                sc0, sc1, vs32, p0, full_row_blk, has_tmask, p0_tree,
                                hi_v, tmk_v, o16, m, l, pa_lo, pa_hi, srw, row_valid, exp_poly);
                            if (zero_fence == -99) pl_ptr[0] = 0.0f;
                            if (row_valid) {
                                accum_vb_slm_r8_blk<U16>(8192u, pa_lo, pa_hi, srw, o16, zero_fence, pl_ptr);
                            }
                        }
                    }
                }
                if (thread_has_rows) {
                    store_partials_o16_raw(pacc, pm, pl_ptr, dbg, dbg_flag, p_elems, pacc_elems, H, nseg_stride, gp_r0, gp_cnt, seg, kvh, r_base, o16, m, l, oob);
                }
            });
    });
}

// Form 2 (W>=2): RS8 (8 KiB Coop SLM sequential reuse: Phase A K_B -> barrier -> Phase B V_B, Q_A[8] + dpas<8,8>).
template <int NS, int U16> struct TagXa2RS8;
template <int NS, int U16> void launch_rs8(sycl::queue& q, const AttnPlan2& pl) {
    const uint32_t* k8 = pl.k8;
    const float* ks = pl.ks;
    const uint16_t* ks16 = pl.ks16;
    const uint32_t* v8t = pl.v8t;
    const float* vs = pl.vs;
    const int32_t* pos0 = pl.pos0;
    const uint32_t* tmask = pl.tmask;
    const int32_t* tp0 = pl.tp0;
    void* const* k_tab = pl.k_tab;
    void* const* v_tab = pl.v_tab;
    const void* const* mir_tab = pl.mir_tab;
    uint32_t* qb = pl.qb;
    float* qs = pl.qs;
    float* pacc = pl.pacc;
    float* pm = pl.pm;
    float* pl_ptr = pl.pl;
    uint32_t* dbg_flag = pl.dbg_flag;
    const int kst = pl.kst, ctxp = pl.ctxp, W = pl.wg_threads, pfd = pl.pfd, zero_fence = pl.zero_fence;
    const int nseg_stride = pl.nseg_stride, t0 = pl.t0, H = pl.H, HKV = pl.HKV;
    const uint32_t dbg = pl.dbg, k8_max_cells = pl.k8_max_cells, v8t_max_blks = pl.v8t_max_blks, vs_max_cells = pl.vs_max_cells;
    const uint64_t qb_dwords = pl.qb_dwords, qs_floats = pl.qs_floats, p_elems = pl.p_elems, pacc_elems = pl.pacc_elems;
    const int ngrp = pl.ngrp, per_tok_pos = pl.per_tok_pos;
    const bool exp_poly = pl.exp_poly != 0;
    const uint64_t gp0 = pl.gp0, gp1 = pl.gp1, gp2 = pl.gp2, gp3 = pl.gp3;
    const uint64_t gp4 = pl.gp4, gp5 = pl.gp5, gp6 = pl.gp6, gp7 = pl.gp7;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagXa2RS8<NS, U16>>(
            sycl::nd_range<3>(sycl::range<3>((size_t) W, (size_t) HKV, (size_t) pl.total_tasks),
                              sycl::range<3>((size_t) W, 1, 1)),
            [=](sycl::nd_item<3> it) SYCL_ESIMD_KERNEL {
                slm_init<8192u>();
                const int tid = (int) it.get_local_id(0);
                const int kvh = (int) it.get_global_id(1);
                const int task = (int) it.get_global_id(2);
                const int r_base = tid * 8;
                int gi = 0, seg = 0, gp_slot = 0, gp_r0 = 0, gp_cnt = 0, gp_ls0 = 0, gp_nrg = 0, gp_nseg = 0;
                decode_task_grp(task, ngrp, gp0, gp1, gp2, gp3, gp4, gp5, gp6, gp7,
                                gi, seg, gp_slot, gp_r0, gp_cnt, gp_ls0, gp_nrg, gp_nseg);

                const uint32_t* kb = nullptr;
                const float* ksb = nullptr;
                const uint32_t* ks16b = nullptr;
                const uint32_t* vb = nullptr;
                const float* vsb = nullptr;
                load_grp_ptrs_raw<NS>(k8, ks, ks16, v8t, vs, k_tab, v_tab, mir_tab,
                                      HKV, kst, ctxp, gp_slot, kvh, kb, ksb, ks16b, vb, vsb);

                const bool thread_has_rows = (r_base < gp_cnt * kG);
                simd<int, 8> hi_v(0);
                simd<uint32_t, 8> tmk_v(0u);
                int ub = 0, p0_tree = 0;
                bool has_tmask = false;
                load_row_bounds_raw<8>(pos0, tmask, tp0, t0, ngrp, per_tok_pos, gp_r0, gp_cnt, r_base,
                                       hi_v, tmk_v, ub, has_tmask, p0_tree);

                int blk_a = 0, blk_b = 0;
                seg_block_range(ub, seg, gp_nseg, blk_a, blk_b);
                const int ls_glob = thread_has_rows ? (gp_ls0 + (r_base >> 4)) : gp_ls0;
                const bool oob = dbg_clamp_seg_bounds_raw(
                    dbg, dbg_flag, HKV, k8_max_cells, v8t_max_blks, vs_max_cells,
                    qb_dwords, qs_floats, kvh, ls_glob, ub, blk_a, blk_b);

                simd<float, 128> o16[16];
#pragma unroll
                for (int k = 0; k < 16; ++k) o16[k] = 0.0f;
                simd<float, 8> m = init_m_r8(hi_v);
                simd<float, 8> l(0.0f);

                if (blk_a < blk_b && !oob) {
                    simd<int, 64> qa[8];
                    simd<float, 8> qsc(0.0f);
                    if (thread_has_rows) {
                        load_qa8_and_scale(qb, qs, kvh, ls_glob, r_base & 8, qa, qsc);
                    }
                    const int min_hi = compute_min_active_hi(hi_v);

                    for (int blk = blk_a; blk < blk_b; ++blk) {
                        if (blk > blk_a) barrier();
                        const int p0 = blk * kBlk;
                        const bool full_kv_blk = (p0 + kBlk <= ub);
                        coop_load_kb_slm(kb, p0, ub, full_kv_blk, tid, W, 0u, zero_fence);
                        barrier();
                        if (pfd > 0 && blk + pfd < blk_b) coop_prefetch_kv(kb, vb, blk + pfd, ub, tid, W);

                        simd<uint32_t, 64> pa_lo, pa_hi;
                        simd<float, 8> srw;
                        bool row_valid = false;
                        if (thread_has_rows) {
                            const bool full_row_blk = (p0 + kBlk <= min_hi) && !(has_tmask && p0 + kBlk > p0_tree);
                            simd<float, 16> sc0[8], sc1[8];
                            score_kb_slm_r8_blk<NS>(
                                qa, qsc, 0u, ksb, ks16b, p0, ub, full_kv_blk, zero_fence, pl_ptr, sc0, sc1);
                            if (zero_fence == -98) pl_ptr[0] = 0.0f;
                            const simd<float, 32> vs32 = load_vs32_safe(vsb, p0, ub, full_kv_blk);
                            softmax_pack_r8<U16>(
                                sc0, sc1, vs32, p0, full_row_blk, has_tmask, p0_tree,
                                hi_v, tmk_v, o16, m, l, pa_lo, pa_hi, srw, row_valid, exp_poly);
                            if (zero_fence == -99) pl_ptr[0] = 0.0f;
                        }
                        barrier();
                        coop_load_vb_slm(vb, blk, tid, W, 0u, zero_fence);
                        barrier();
                        if (thread_has_rows && row_valid) {
                            accum_vb_slm_r8_blk<U16>(0u, pa_lo, pa_hi, srw, o16, zero_fence, pl_ptr);
                        }
                    }
                }
                if (thread_has_rows) {
                    store_partials_o16_raw(pacc, pm, pl_ptr, dbg, dbg_flag, p_elems, pacc_elems, H, nseg_stride, gp_r0, gp_cnt, seg, kvh, r_base, o16, m, l, oob);
                }
            });
    });
}

// Form 3 (W>=2): RS16K (16 KiB Coop SLM dual-B: 0..8191 K_B, 8192..16383 V_B; permanent Q_A[8]; 16 Score + 32 Value dpas<8,8>).
template <int NS, int U16> struct TagXa2RS16K;
template <int NS, int U16> void launch_rs16k(sycl::queue& q, const AttnPlan2& pl) {
    const uint32_t* k8 = pl.k8;
    const float* ks = pl.ks;
    const uint16_t* ks16 = pl.ks16;
    const uint32_t* v8t = pl.v8t;
    const float* vs = pl.vs;
    const int32_t* pos0 = pl.pos0;
    const uint32_t* tmask = pl.tmask;
    const int32_t* tp0 = pl.tp0;
    void* const* k_tab = pl.k_tab;
    void* const* v_tab = pl.v_tab;
    const void* const* mir_tab = pl.mir_tab;
    uint32_t* qb = pl.qb;
    float* qs = pl.qs;
    float* pacc = pl.pacc;
    float* pm = pl.pm;
    float* pl_ptr = pl.pl;
    uint32_t* dbg_flag = pl.dbg_flag;
    const int kst = pl.kst, ctxp = pl.ctxp, W = pl.wg_threads, pfd = pl.pfd, zero_fence = pl.zero_fence;
    const int nseg_stride = pl.nseg_stride, t0 = pl.t0, H = pl.H, HKV = pl.HKV;
    const uint32_t dbg = pl.dbg, k8_max_cells = pl.k8_max_cells, v8t_max_blks = pl.v8t_max_blks, vs_max_cells = pl.vs_max_cells;
    const uint64_t qb_dwords = pl.qb_dwords, qs_floats = pl.qs_floats, p_elems = pl.p_elems, pacc_elems = pl.pacc_elems;
    const int ngrp = pl.ngrp, per_tok_pos = pl.per_tok_pos;
    const bool exp_poly = pl.exp_poly != 0;
    const uint64_t gp0 = pl.gp0, gp1 = pl.gp1, gp2 = pl.gp2, gp3 = pl.gp3;
    const uint64_t gp4 = pl.gp4, gp5 = pl.gp5, gp6 = pl.gp6, gp7 = pl.gp7;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagXa2RS16K<NS, U16>>(
            sycl::nd_range<3>(sycl::range<3>((size_t) W, (size_t) HKV, (size_t) pl.total_tasks),
                              sycl::range<3>((size_t) W, 1, 1)),
            [=](sycl::nd_item<3> it) SYCL_ESIMD_KERNEL {
                slm_init<16384u>();
                const int tid = (int) it.get_local_id(0);
                const int kvh = (int) it.get_global_id(1);
                const int task = (int) it.get_global_id(2);
                const int r_base = tid * 8;
                int gi = 0, seg = 0, gp_slot = 0, gp_r0 = 0, gp_cnt = 0, gp_ls0 = 0, gp_nrg = 0, gp_nseg = 0;
                decode_task_grp(task, ngrp, gp0, gp1, gp2, gp3, gp4, gp5, gp6, gp7,
                                gi, seg, gp_slot, gp_r0, gp_cnt, gp_ls0, gp_nrg, gp_nseg);

                const uint32_t* kb = nullptr;
                const float* ksb = nullptr;
                const uint32_t* ks16b = nullptr;
                const uint32_t* vb = nullptr;
                const float* vsb = nullptr;
                load_grp_ptrs_raw<NS>(k8, ks, ks16, v8t, vs, k_tab, v_tab, mir_tab,
                                      HKV, kst, ctxp, gp_slot, kvh, kb, ksb, ks16b, vb, vsb);

                const bool thread_has_rows = (r_base < gp_cnt * kG);
                simd<int, 8> hi_v(0);
                simd<uint32_t, 8> tmk_v(0u);
                int ub = 0, p0_tree = 0;
                bool has_tmask = false;
                load_row_bounds_raw<8>(pos0, tmask, tp0, t0, ngrp, per_tok_pos, gp_r0, gp_cnt, r_base,
                                       hi_v, tmk_v, ub, has_tmask, p0_tree);

                int blk_a = 0, blk_b = 0;
                seg_block_range(ub, seg, gp_nseg, blk_a, blk_b);
                const int ls_glob = thread_has_rows ? (gp_ls0 + (r_base >> 4)) : gp_ls0;
                const bool oob = dbg_clamp_seg_bounds_raw(
                    dbg, dbg_flag, HKV, k8_max_cells, v8t_max_blks, vs_max_cells,
                    qb_dwords, qs_floats, kvh, ls_glob, ub, blk_a, blk_b);

                simd<float, 128> o16[16];
#pragma unroll
                for (int k = 0; k < 16; ++k) o16[k] = 0.0f;
                simd<float, 8> m = init_m_r8(hi_v);
                simd<float, 8> l(0.0f);

                if (blk_a < blk_b && !oob) {
                    simd<int, 64> qa[8];
                    simd<float, 8> qsc(0.0f);
                    if (thread_has_rows) {
                        load_qa8_and_scale(qb, qs, kvh, ls_glob, r_base & 8, qa, qsc);
                    }
                    const int min_hi = compute_min_active_hi(hi_v);

                    for (int blk = blk_a; blk < blk_b; ++blk) {
                        if (blk > blk_a) barrier();
                        const int p0 = blk * kBlk;
                        const bool full_kv_blk = (p0 + kBlk <= ub);
                        coop_load_kb_slm(kb, p0, ub, full_kv_blk, tid, W, 0u, zero_fence);
                        coop_load_vb_slm(vb, blk, tid, W, 8192u, zero_fence);
                        barrier();
                        if (pfd > 0 && blk + pfd < blk_b) coop_prefetch_kv(kb, vb, blk + pfd, ub, tid, W);

                        if (thread_has_rows) {
                            const bool full_row_blk = (p0 + kBlk <= min_hi) && !(has_tmask && p0 + kBlk > p0_tree);
                            simd<float, 16> sc0[8], sc1[8];
                            score_kb_slm_r8_blk<NS>(
                                qa, qsc, 0u, ksb, ks16b, p0, ub, full_kv_blk, zero_fence, pl_ptr, sc0, sc1);
                            const simd<float, 32> vs32 = load_vs32_safe(vsb, p0, ub, full_kv_blk);
                            simd<uint32_t, 64> pa_lo, pa_hi;
                            simd<float, 8> srw;
                            bool row_valid = false;
                            softmax_pack_r8<U16>(
                                sc0, sc1, vs32, p0, full_row_blk, has_tmask, p0_tree,
                                hi_v, tmk_v, o16, m, l, pa_lo, pa_hi, srw, row_valid, exp_poly);
                            if (row_valid) {
                                accum_vb_slm_r8_blk<U16>(8192u, pa_lo, pa_hi, srw, o16, zero_fence, pl_ptr);
                            }
                        }
                    }
                }
                if (thread_has_rows) {
                    store_partials_o16_raw(pacc, pm, pl_ptr, dbg, dbg_flag, p_elems, pacc_elems, H, nseg_stride, gp_r0, gp_cnt, seg, kvh, r_base, o16, m, l, oob);
                }
            });
    });
}

// Coalesced ESIMD segment merge (optional fused sigm(gate)).
template <int HAS_GATE> struct TagXa2Merge;
template <int HAS_GATE> void launch_merge2(sycl::queue& q, const MPlan2& pl) {
    const int rows = pl.nt * pl.H;
    const int H = pl.H, ngrp = pl.ngrp, nseg_stride = pl.nseg_stride;
    const float inv_H = pl.inv_H;
    const uint32_t dbg = pl.dbg;
    float* pacc = pl.pacc;
    float* pm = pl.pm;
    float* pl_ptr = pl.pl;
    const float* gate = pl.gate;
    float* out_ptr = pl.out;
    uint32_t* dbg_flag = pl.dbg_flag;
    const uint64_t p_elems = pl.p_elems, pacc_elems = pl.pacc_elems, out_elems = pl.out_elems;
    const uint64_t g_nseg_lo = pl.g_nseg_lo, g_nseg_hi = pl.g_nseg_hi;
    const uint64_t tok_grp_lo = pl.tok_grp_lo, tok_grp_hi = pl.tok_grp_hi;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagXa2Merge<HAS_GATE>>(
            sycl::nd_range<2>(sycl::range<2>((size_t) rows, 4u),
                              sycl::range<2>(1u, 4u)),
            [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
                const int row = (int) it.get_global_id(0);
                const int d_chk = (int) it.get_local_id(1);
                const int d0 = d_chk * 64;
                int tok = (int) ((float) row * inv_H);
                if ((tok + 1) * H <= row) ++tok;
                if (tok * H > row) --tok;
                const int gi = (int) unpack_u8_16(tok_grp_lo, tok_grp_hi, tok);
                if (gi >= ngrp) return;
                const int nseg = (int) unpack_u8_16(g_nseg_lo, g_nseg_hi, gi);
                const size_t stride = (size_t) nseg_stride;
                const size_t row_off = (size_t) row * stride;
                const size_t gi_off = (size_t) row * (size_t) kHD + (size_t) d0;
                if (dbg) {
                    const size_t max_p = row_off + (size_t) nseg;
                    if (nseg <= 0 || max_p > p_elems || max_p * (size_t) kHD > pacc_elems || gi_off + 64u > out_elems) {
                        xatt2_mark_oob(dbg_flag, 201u, nullptr);
                        if (gi_off + 64u <= out_elems) {
                            simd<float, 64> nv(xatt2_nan_f32());
                            block_store<float, 64>(out_ptr + gi_off, nv);
                        }
                        return;
                    }
                }

                simd<float, 16> mx(kNeg);
                const simd<int, 16> lane(0, 1);
                for (int s0 = 0; s0 < nseg; s0 += 16) {
                    simd<float, 16> mv(kNeg);
                    if (s0 + 16 <= nseg && (((row_off + (size_t) s0) & 3u) == 0u)) {
                        mv = block_load<float, 16>(pm + row_off + (size_t) s0);
                    } else {
                        for (int k = 0; k < 16 && s0 + k < nseg; ++k) mv[k] = pm[row_off + (size_t) (s0 + k)];
                    }
                    mx = (es::max)(mx, mv);
                }
                const float m_all = hmax<float>(mx);

                simd<float, 64> num(0.0f);
                float den = 0.0f;
                for (int s0 = 0; s0 < nseg; s0 += 16) {
                    simd<float, 16> mv(kNeg), lv(0.0f);
                    if (s0 + 16 <= nseg && (((row_off + (size_t) s0) & 3u) == 0u)) {
                        mv = block_load<float, 16>(pm + row_off + (size_t) s0);
                        lv = block_load<float, 16>(pl_ptr + row_off + (size_t) s0);
                    } else {
                        for (int k = 0; k < 16 && s0 + k < nseg; ++k) {
                            mv[k] = pm[row_off + (size_t) (s0 + k)];
                            lv[k] = pl_ptr[row_off + (size_t) (s0 + k)];
                        }
                    }
                    simd<float, 16> w = exp16((es::max)(mv - m_all, simd<float, 16>(-87.0f)));
                    w.merge(simd<float, 16>(0.0f), (mv <= -1e29f) | ((s0 + lane) >= nseg));
                    den += hsum16_f(w * lv);
                    const float* pa_base = pacc + (row_off + (size_t) s0) * (size_t) kHD + (size_t) d0;
                    if (s0 + 16 <= nseg) {
                        // all 16 partials' loads in flight at once (every segment of the row was stored: an empty one holds
                        // zeros and has weight 0), not one load per dependent branch
                        simd<float, 64> pv[16];
#pragma unroll
                        for (int k = 0; k < 16; ++k) pv[k] = block_load<float, 64>(pa_base + (size_t) k * (size_t) kHD);
#pragma unroll
                        for (int k = 0; k < 16; ++k) num += float(w[k]) * pv[k];
                    } else {
                        for (int k = 0; k < 16 && s0 + k < nseg; ++k) {
                            const float wk = w[k];
                            if (wk != 0.0f) {
                                const simd<float, 64> pv = block_load<float, 64>(pa_base + (size_t) k * (size_t) kHD);
                                num += wk * pv;
                            }
                        }
                    }
                }
                simd<float, 64> out = (den > 0.0f) ? (num * (1.0f / den)) : simd<float, 64>(0.0f);
                if constexpr (HAS_GATE) {
                    const simd<float, 64> g = block_load<float, 64>(gate + gi_off);
                    out = out * sigm64(g);
                }
                block_store<float, 64>(out_ptr + gi_off, out);
            });
    });
}

inline void pack_u8_16(uint64_t& lo, uint64_t& hi, int idx, uint8_t val) {
    uint64_t& w = (idx < 8) ? lo : hi;
    const uint32_t sh = (uint32_t) (idx & 7) * 8u;
    w = (w & ~(0xffull << sh)) | ((uint64_t) val << sh);
}

// Segments a row may have: the partial slots a call of nt tokens owns (16 x H x nseg_max of them, shared by its nt x H rows).
inline int plan2_seg_cap(const Call2& c) {
    const int nt = std::max(1, c.nt);
    return std::max(1, std::min(kMaxSegsHard2, std::max(1, c.nseg_max) * 16 / nt));
}

// The threads the segments should keep busy (-doubleGRF: 1024 in flight at most).  Measured at 118K (ns=4): T=5 (W=4, 48 segs,
// 768 threads) 480 us; T=1 (W=1, 96 segs, 384 threads) 714 us - a thread keeps about 1 KiB of loads in flight, so the bytes
// in flight follow the thread count: W=1 gets 1024 threads (hypothesis, Hermes measures: STRATA_Q36_XATT2_THREADS).
inline int xatt2_env_threads() {
    static const int v = [] { const char* e = std::getenv("STRATA_Q36_XATT2_THREADS"); return e && *e ? std::atoi(e) : 0; }();
    return v;
}

inline int plan2_choose_nseg(int HKV, int w, int ngrp_i, int seg_cap, int nseg_req, int nblk) {
    if (nseg_req > 0) return std::max(1, std::min(seg_cap / std::max(1, ngrp_i), nseg_req));
    // Target ~768 active threads across 32 Xe cores (<= 24 thr/core <= 32 -doubleGRF limit) for W >= 2:
    // W=4 (T=5) -> 48 segs (768 thr, 6 WGs/core); W=6 (T=8) -> 32 segs; W=12 (T=16) -> 16 segs.  W=1 (<= 8 rows) -> 1024 threads.
    // Align nseg to a multiple of (32 / HKV) so total WGs divide evenly across 32 Xe cores.
    const int env_thr = xatt2_env_threads();
    const int tgt_threads = env_thr > 0 ? env_thr : (w == 1 ? 1024 : 768);
    const int core_align = std::max(1, 32 / std::max(1, HKV));
    int budget_per_kv = std::min(seg_cap, std::max(1, tgt_threads / (std::max(1, HKV) * std::max(1, w))));
    if (ngrp_i == 1 && budget_per_kv >= core_align) {
        budget_per_kv = (budget_per_kv / core_align) * core_align;
    }
    // a segment of fewer than 4 blocks (W = 1) / 2 blocks costs more in its setup (queries, partials) than it streams
    const int blk_floor = std::max(1, nblk / (w == 1 ? 4 : 2));
    const int share = (ngrp_i == 1) ? budget_per_kv : std::max(1, std::min(seg_cap / ngrp_i, budget_per_kv / ngrp_i));
    return std::max(1, std::min(share, blk_floor));
}

inline void plan2_group_nsegs(const Call2& c, int ngrp_i, int max_rows, int g_nseg[kMaxBatchGroups2]) {
    const int w = std::max(1, (max_rows + 7) / 8);
    const int seg_cap = plan2_seg_cap(c);
    const int nseg_req = c.nseg_override > 0 ? c.nseg_override : xatt2_env_segs();
    int total_tasks = 0;
    for (int i = 0; i < ngrp_i; ++i) {
        const int hint_raw = (c.ngrp > 0) ? c.grp[i].cell_hint : c.pos_hint;
        const int hint = hint_raw > 0 ? hint_raw : (c.pos_hint > 0 ? c.pos_hint : 256);
        const int nblk = std::max(1, (hint + kBlk - 1) / kBlk);
        int ns_i = plan2_choose_nseg(c.HKV, w, ngrp_i, seg_cap, nseg_req, nblk);
        if (total_tasks + ns_i > seg_cap) ns_i = std::max(1, seg_cap - total_tasks);
        g_nseg[i] = ns_i;
        total_tasks += ns_i;
    }
}

int build_plan2(const Call2& c, HostPlan2& hp, int& form_out, int& u16_out) {
    const int env_form = xatt2_env_form();
    const int env_u16 = xatt2_env_u16();
    const int env_pfd = xatt2_env_pfd();
    const uint32_t dbg_on = xatt2_dbg_on() ? 1u : 0u;

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
    pl.exp_poly = c.exp_override >= 0 ? (c.exp_override != 0 ? 1 : 0) : xatt2_env_exp();
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
    pl.dbg = dbg_on;
    pl.dbg_flag = c.dbg_flag;

    qp.nt = c.nt;
    qp.t0 = c.t0;
    qp.H = c.H;
    qp.HKV = c.HKV;
    qp.scale = c.scale;
    qp.q = c.q;
    qp.qb = c.qb;
    qp.qs = c.qs;
    qp.dbg = dbg_on;
    qp.dbg_flag = c.dbg_flag;

    mp.nt = c.nt;
    mp.H = c.H;
    mp.inv_H = 1.0f / (float) c.H;
    mp.pacc = c.pacc;
    mp.pm = c.pm;
    mp.pl = c.pl;
    mp.gate = c.gate;
    mp.out = c.out;
    mp.dbg = dbg_on;
    mp.dbg_flag = c.dbg_flag;

    int g_slot[kMaxBatchGroups2]{};
    int g_r0[kMaxBatchGroups2]{};
    int g_cnt[kMaxBatchGroups2]{};
    int g_ls0[kMaxBatchGroups2]{};
    int g_nrg[kMaxBatchGroups2]{};
    int g_nseg[kMaxBatchGroups2]{};

    if (c.ngrp > 0) {
        if (c.ngrp > kMaxBatchGroups2) return -1;
        pl.ngrp = c.ngrp;
        pl.per_tok_pos = 1;
        for (int i = 0; i < c.ngrp; ++i) {
            if (c.grp[i].slot < 0 || c.grp[i].slot > 255 || c.grp[i].r0 < 0 || c.grp[i].cnt <= 0) return -1;
            if (c.grp[i].r0 + c.grp[i].cnt > kMaxBatchGroups2) return -1;
            g_slot[i] = c.grp[i].slot;
            g_r0[i] = c.grp[i].r0;
            g_cnt[i] = c.grp[i].cnt;
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
        g_slot[0] = 0;
        g_r0[0] = 0;
        g_cnt[0] = c.nt;
    }

    qp.ngrp = pl.ngrp;
    qp.per_tok_pos = pl.per_tok_pos;
    mp.ngrp = pl.ngrp;
    mp.tok_grp_lo = ~0ull;
    mp.tok_grp_hi = ~0ull;

    int max_rows = 0;
    int ls_cur = 0;
    for (int i = 0; i < pl.ngrp; ++i) {
        const int cnt_i = g_cnt[i];
        const int r0_i = g_r0[i];
        const int rows = cnt_i * kG;
        if (rows < 1 || rows > kMaxRows) return -1;
        if (rows > max_rows) max_rows = rows;
        const int nrg = (rows + 15) / 16;
        if (ls_cur + nrg > kMaxBatchGroups2) return -1;
        g_ls0[i] = ls_cur;
        g_nrg[i] = nrg;
        pack_u8_16(qp.g_r0_cnt_lo, qp.g_r0_cnt_hi, i, (uint8_t) (((r0_i & 0xf) << 4) | ((cnt_i - 1) & 0xf)));
        for (int l = 0; l < nrg; ++l) {
            pack_u8_16(qp.ls_gl_lo, qp.ls_gl_hi, ls_cur + l, (uint8_t) (((i & 0xf) << 4) | (l & 0xf)));
        }
        ls_cur += nrg;
        for (int k = 0; k < cnt_i; ++k) {
            const int tok = r0_i + k;
            if (tok < kMaxBatchGroups2) pack_u8_16(mp.tok_grp_lo, mp.tok_grp_hi, tok, (uint8_t) i);
        }
    }
    hp.total_ls = ls_cur;

    u16_out = (c.u16_override >= 0) ? (c.u16_override != 0 ? 1 : 0) : (env_u16 != 0 ? 1 : 0);

    int f = (c.form_override >= 0) ? c.form_override : env_form;
    if (f < 0 || f > 3) {
        f = (max_rows <= 8) ? kFormR8 : kFormRS16K;
    }
    form_out = f;

    const int w = (max_rows + 7) / 8;
    pl.wg_threads = w;

    const int seg_cap = plan2_seg_cap(c);
    const int ngrp_i = pl.ngrp;
    plan2_group_nsegs(c, ngrp_i, max_rows, g_nseg);

    int total_tasks = 0;
    int max_nseg = 1;
    uint64_t* gp_words[8] = {&pl.gp0, &pl.gp1, &pl.gp2, &pl.gp3, &pl.gp4, &pl.gp5, &pl.gp6, &pl.gp7};
    for (int i = 0; i < ngrp_i; ++i) {
        const int ns_i = g_nseg[i];
        pack_u8_16(mp.g_nseg_lo, mp.g_nseg_hi, i, (uint8_t) ns_i);
        const uint32_t packed = pack_grp_u32(g_slot[i], g_r0[i], g_cnt[i], g_ls0[i], g_nrg[i], ns_i);
        const uint32_t sh = (uint32_t) (i & 1) * 32u;
        *gp_words[i >> 1] |= ((uint64_t) packed << sh);
        if (ns_i > max_nseg) max_nseg = ns_i;
        total_tasks += ns_i;
    }
    pl.total_tasks = total_tasks;
    int aligned_stride = max_nseg;
    if (c.out != nullptr && aligned_stride > 1) {
        const int a4 = (aligned_stride + 3) & ~3;
        if (a4 <= seg_cap) aligned_stride = a4;
    }
    pl.nseg_stride = aligned_stride;
    mp.nseg_stride = aligned_stride;

    const size_t q_cap = c.dbg_q_floats ? c.dbg_q_floats : ((size_t) (c.t0 + c.nt) * (size_t) c.H * (size_t) kHD);
    const size_t qb_cap = qb2_dwords(c.HKV);
    const size_t qs_cap = qs2_floats(c.HKV);
    const size_t k8_cells_cap = c.dbg_k8_dwords ? (c.dbg_k8_dwords / ((size_t) c.HKV * 64u)) : (size_t) pl.kst;
    const size_t ks_cells_cap = (c.ns == 1)
        ? (c.dbg_ks_floats ? (c.dbg_ks_floats / (size_t) c.HKV) : (size_t) pl.kst)
        : (c.dbg_ks16_u16 ? (c.dbg_ks16_u16 / ((size_t) c.HKV * 4u)) : (size_t) pl.kst);
    const size_t v8t_blks_cap = c.dbg_v8t_dwords ? (c.dbg_v8t_dwords / ((size_t) c.HKV * 2048u)) : (size_t) (pl.ctxp / kBlk);
    const size_t vs_cells_cap = c.dbg_vs_floats ? (c.dbg_vs_floats / (size_t) c.HKV) : (size_t) pl.ctxp;
    const size_t p_cap = (size_t) c.nt * (size_t) c.H * (size_t) seg_cap;
    const size_t pacc_cap = p_cap * (size_t) kHD;
    const size_t out_cap = c.dbg_out_floats ? c.dbg_out_floats : ((size_t) c.nt * (size_t) c.H * (size_t) kHD);

    qp.q_floats = q_cap;
    qp.qb_dwords = qb_cap;
    qp.qs_floats = qs_cap;
    pl.qb_dwords = qb_cap;
    pl.qs_floats = qs_cap;
    pl.k8_max_cells = (uint32_t) std::min<size_t>(k8_cells_cap, ks_cells_cap);
    pl.v8t_max_blks = (uint32_t) v8t_blks_cap;
    pl.vs_max_cells = (uint32_t) vs_cells_cap;
    pl.p_elems = p_cap;
    pl.pacc_elems = pacc_cap;
    mp.p_elems = p_cap;
    mp.pacc_elems = pacc_cap;
    mp.out_elems = out_cap;
    return 0;
}

template <int NS, int U16> void dispatch_form(sycl::queue& q, const AttnPlan2& pl, int form) {
    if (form == kFormR8) {
        launch_r8<NS, U16>(q, pl);
        return;
    }
    if (pl.wg_threads <= 1) {
        launch_r8_d8<NS, U16>(q, pl);
        return;
    }
    if (form == kFormRS16) {
        launch_rs16<NS, U16>(q, pl);
        return;
    }
    if (form == kFormRS8) {
        launch_rs8<NS, U16>(q, pl);
        return;
    }
    launch_rs16k<NS, U16>(q, pl);
}

}   // namespace

const char* xatt2_form_name(int form) {
    switch (form) {
        case kFormR8: return "R8";
        case kFormRS16: return "RS16";
        case kFormRS8: return "RS8";
        case kFormRS16K: return "RS16K";
        default: return "?";
    }
}

int xatt2_plan_nseg(const Call2& c, int group_idx) {
    const int ngrp_i = (c.ngrp > 0) ? std::min(c.ngrp, kMaxBatchGroups2) : 1;
    if (group_idx < 0 || group_idx >= ngrp_i) return 1;
    int max_rows = 0;
    if (c.ngrp > 0) {
        for (int i = 0; i < ngrp_i; ++i) max_rows = std::max(max_rows, c.grp[i].cnt * kG);
    } else {
        max_rows = std::max(1, c.nt) * kG;
    }
    int g_nseg[kMaxBatchGroups2]{};
    plan2_group_nsegs(c, ngrp_i, max_rows, g_nseg);
    return g_nseg[group_idx];
}

static int plan2_max_rows(const Call2& c) {
    int max_rows = 0;
    if (c.ngrp > 0) {
        for (int i = 0; i < std::min(c.ngrp, kMaxBatchGroups2); ++i) max_rows = std::max(max_rows, c.grp[i].cnt * kG);
    } else {
        max_rows = std::max(1, c.nt) * kG;
    }
    return max_rows;
}

int xatt2_plan_stride(const Call2& c) {
    const int ngrp_i = (c.ngrp > 0) ? std::min(c.ngrp, kMaxBatchGroups2) : 1;
    int g_nseg[kMaxBatchGroups2]{};
    plan2_group_nsegs(c, ngrp_i, plan2_max_rows(c), g_nseg);
    int max_nseg = 1;
    for (int i = 0; i < ngrp_i; ++i) max_nseg = std::max(max_nseg, g_nseg[i]);
    int stride = max_nseg;
    if (c.out != nullptr && stride > 1) {
        const int a4 = (stride + 3) & ~3;
        if (a4 <= plan2_seg_cap(c)) stride = a4;
    }
    return stride;
}

int xatt2_plan_form(const Call2& c) {
    const int f = (c.form_override >= 0) ? c.form_override : xatt2_env_form();
    if (f >= 0 && f <= 3) return f;
    return plan2_max_rows(c) <= 8 ? kFormR8 : kFormRS16K;
}

bool attend2(const Call2& c, void* stream) {
    HostPlan2 hp;
    int form = 0, u16 = 1;
    if (build_plan2(c, hp, form, u16) != 0) return false;
    sycl::queue& q = *strata::q_of(stream);
    launch_queries2(q, hp.q, hp.total_ls);
    if (c.ns == 4) {
        if (u16) dispatch_form<4, 1>(q, hp.a, form);
        else dispatch_form<4, 0>(q, hp.a, form);
    } else {
        if (u16) dispatch_form<1, 1>(q, hp.a, form);
        else dispatch_form<1, 0>(q, hp.a, form);
    }
    if (c.out) {
        if (c.gate) launch_merge2<1>(q, hp.m);
        else launch_merge2<0>(q, hp.m);
    }
    return true;
}

struct TagXa2ExpProbe;
bool xatt2_exp_probe(void* stream, const float* d_x, float* d_hw, float* d_poly, int n) {
    if (!stream || !d_x || !d_hw || !d_poly || n <= 0 || (n % 32) != 0) return false;
    sycl::queue& q = *strata::q_of(stream);
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagXa2ExpProbe>(
            sycl::nd_range<1>(sycl::range<1>((size_t) (n / 32)), sycl::range<1>(1)),
            [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
                const size_t o = (size_t) it.get_global_id(0) * 32u;
                simd<float, 32> x = block_load<float, 32>(d_x + o);
                x = (es::max)(x, simd<float, 32>(-87.0f));
                simd<float, 32> hw = es::exp2(x * kLog2e2);
                simd<float, 32> po = exp_series<32>(x);
                block_store<float, 32>(d_hw + o, hw);
                block_store<float, 32>(d_poly + o, po);
            });
    });
    return true;
}

}   // namespace strata::q36xatt
