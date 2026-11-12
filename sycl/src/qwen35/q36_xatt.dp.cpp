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

}   // namespace strata::q36xatt
