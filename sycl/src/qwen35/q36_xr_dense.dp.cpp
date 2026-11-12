// q36_xr_dense.dp.cpp - the dense Qwen3.8-27B's decode products on the Arc Pro B70's matrix engine (XMX, ESIMD).  strata-q36 only.
//
// The KREG forms (q36_mmvq_dense.dp.cpp) give a lane 32 weights of ONE row and keep every column's activations in registers
// per lane (12 registers a column): 4 columns fit, the 5th spills, a 17408-wide row at 5+ columns is read twice.  Here a
// hardware thread owns 16 output ROWS and walks K in steps of 32 weights: it decodes the 16 rows' 32 weights into the B
// operand of one s8 DPAS 8x16x32, whose A operand is the same 32 activations of up to 8 COLUMNS (256 bytes, the same for
// every lane, one block load).  One instruction is the integer dot of 16 rows x 8 columns; a column more costs only its
// float scaling.  (The Flash-Next prompt path does the same for the expert matrices: src/prefill/esimd_moe.dp.cpp.)
//
//   - Arithmetic.  Per row and 32 weights: y += (d x scale) x (dx x dot) - (dmin x min) x sx, with dx the activations'
//     scale of that column and 32, sx = dx x their sum.  The int32 dot is exact.  The float sums run in K order, each column
//     on its own: a T-column call is bitwise T one-column calls.  Another summation order than KREG.
//   - Activations.  quantize() writes them once per input: [32-block][column 0..7][32 int8], and per 256 of K the 8 x 8
//     scales and 8 x 8 scaled sums.  q8_1 as llama.cpp (d = amax / 127, round half away from zero), d kept as a float.
//   - Shapes.  n_in a multiple of 256, n_out a multiple of 16, the matrix under 4 GiB (32-bit byte offsets).
//   - ESIMD on this compiler: constant indices only, no division by a run-time value (esimd_moe.dp.cpp's notes).
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "strata/sycl_queue.hpp"
#include "q36_xr_dense.hpp"

namespace strata::q36xr {
namespace {
using namespace sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

constexpr int T_Q4K = 12, T_Q5K = 13, T_IQ4XS = 23;
template <int F> constexpr uint32_t blk_bytes() { return F == T_Q4K ? 144u : F == T_Q5K ? 176u : 136u; }
template <int F> constexpr uint32_t qs_off() { return F == T_Q4K ? 16u : F == T_Q5K ? 48u : 8u; }
constexpr int8_t kIq4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
// IQ4_XS: a qs byte -> the codebook value of its low nibble in byte 0 and of its high nibble in byte 2 (256 dwords; a
// work-group copies them into shared local memory: a lookup in a table in ordinary memory ran the form at 190 GB/s)
const uint32_t* iq4_table(sycl::queue& q) {
    static const uint32_t* t = [&] {
        std::vector<uint32_t> h(256);
        for (uint32_t i = 0; i < 256; ++i) h[i] = (uint32_t) (uint8_t) kIq4[i & 15] | ((uint32_t) (uint8_t) kIq4[i >> 4] << 16);
        uint32_t* d = sycl::malloc_device<uint32_t>(256, q);
        q.memcpy(d, h.data(), 256 * 4).wait();
        return d;
    }();
    return t;
}

SYCL_ESIMD_FUNCTION inline simd<float, 16> half_of(simd<uint32_t, 16> h16) {
    simd<uint16_t, 16> lo = h16.template bit_cast_view<uint16_t>().template select<16, 2>(0);
    simd<sycl::half, 16> hh = lo.template bit_cast_view<sycl::half>();
    simd<float, 16> f = hh;
    return f;
}
// 8 dwords a lane at byte offset `off` (dword aligned); element d of lane i at [16 * d + i].  STRATA_Q36_XR_VS (a build
// macro, 8): the dwords one gather message carries per lane.  8: one message (2: four).
#ifndef STRATA_Q36_XR_VS
#define STRATA_Q36_XR_VS 8
#endif
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 128> ld8(const uint32_t* base, simd<uint32_t, 16> off) {
#if STRATA_Q36_XR_VS == 8
    return gather<uint32_t, 128, 8>(base, off);
#else
    simd<uint32_t, 128> r;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        simd<uint32_t, 32> v = gather<uint32_t, 32, 2>(base, off + (uint32_t) (8 * g));
        r.template select<16, 1>(16 * (2 * g)) = v.template select<16, 1>(0);
        r.template select<16, 1>(16 * (2 * g + 1)) = v.template select<16, 1>(16);
    }
    return r;
#endif
}
// 4 dwords a lane (a K-quant block's header); element d of lane i at [16 * d + i]
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 64> ld4(const uint32_t* base, simd<uint32_t, 16> off) {
#if STRATA_Q36_XR_VS == 8
    return gather<uint32_t, 64, 4>(base, off);
#else
    simd<uint32_t, 64> r;
    simd<uint32_t, 32> a = gather<uint32_t, 32, 2>(base, off), b = gather<uint32_t, 32, 2>(base, off + 8u);
    r.template select<32, 1>(0) = a;
    r.template select<32, 1>(32) = b;
    return r;
#endif
}
// A block's raw words of 16 rows: the header (IQ4_XS: its 2 dwords), Q5_K's qh, the four 32-byte qs groups.
struct Raw {
    simd<uint32_t, 64> h;
    simd<uint32_t, 128> qh, q0, q1, q2, q3;
};
template <int F> SYCL_ESIMD_FUNCTION inline void load_raw(Raw& r, const uint32_t* Wd, simd<uint32_t, 16> po) {
    if constexpr (F == T_IQ4XS) {
        simd<uint32_t, 32> h = gather<uint32_t, 32, 2>(Wd, po);   // d, scales_h | scales_l
        r.h.template select<32, 1>(0) = h;
    } else {
        r.h = ld4(Wd, po);
        if constexpr (F == T_Q5K) r.qh = ld8(Wd, po + 16u);
    }
    r.q0 = ld8(Wd, po + qs_off<F>());
    r.q1 = ld8(Wd, po + (qs_off<F>() + 32u));
    r.q2 = ld8(Wd, po + (qs_off<F>() + 64u));
    r.q3 = ld8(Wd, po + (qs_off<F>() + 96u));
}
// ggml get_scale_min_k4 for sub-block J of a K-quant block: the 12 scale bytes are dwords h1, h2, h3 of its header
template <int K> SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> sbyte(simd<uint32_t, 16> h1, simd<uint32_t, 16> h2, simd<uint32_t, 16> h3) {
    if constexpr (K < 4) return (h1 >> (8 * (K & 3))) & 0xffu;
    else if constexpr (K < 8) return (h2 >> (8 * (K & 3))) & 0xffu;
    else return (h3 >> (8 * (K & 3))) & 0xffu;
}
template <int J> SYCL_ESIMD_FUNCTION inline void scale_min(simd<uint32_t, 16> h1, simd<uint32_t, 16> h2, simd<uint32_t, 16> h3,
                                                         simd<float, 16> d, simd<float, 16> dmin, simd<float, 16>& dl, simd<float, 16>& ml) {
    simd<uint32_t, 16> sc, m;
    if constexpr (J < 4) {
        sc = sbyte<J>(h1, h2, h3) & 63u;
        m = sbyte<J + 4>(h1, h2, h3) & 63u;
    } else {
        sc = (sbyte<J + 4>(h1, h2, h3) & 0xfu) | ((sbyte<J - 4>(h1, h2, h3) >> 6) << 4);
        m = (sbyte<J + 4>(h1, h2, h3) >> 4) | ((sbyte<J>(h1, h2, h3) >> 6) << 4);
    }
    dl = d * simd<float, 16>(simd<int, 16>(sc));
    ml = dmin * simd<float, 16>(simd<int, 16>(m));
}
// sub-block S of a block: its 32 weights of 16 rows (B operand) x the 8 columns' 32 activations, scaled into acc.
// Q81 = false: `xi` is the block's [8 sub-blocks][8 columns][8 dwords] of quantize(), dxv / sxv its scales and scaled sums.
// Q81 = true: `xi` is column 0's q8_1 blocks of this 256 (9 dwords each: d, s | 32 int8), column r is xs9 dwords further:
// one 16-dword load a column and sub-block (it reads 28 bytes past the block: the buffer has that slack), the sums by a
// second DPAS against ones.
// The q8_1 blocks of one 256 of K for the columns of a call: column r's 72 dwords as A[r] (dwords 0 .. 63: sub-blocks 0 .. 6
// and the first dword of 7) and B[r] (dwords 63 .. 78: sub-block 7, then 28 bytes past it).  Two loads a column and block.
struct Acts {
    simd<int, 64> A[8];
    simd<int, 16> B[8];
};
template <int NC> SYCL_ESIMD_FUNCTION inline void load_acts(Acts& a, const int* xi, int xs9) {
#pragma unroll
    for (int r = 0; r < NC; ++r) {
        a.A[r] = block_load<int, 64>(xi + (size_t) r * xs9);
        a.B[r] = block_load<int, 16>(xi + (size_t) r * xs9 + 63);
    }
}
template <int NC, int S, bool MIN, bool Q81>
SYCL_ESIMD_FUNCTION inline void sub_block(simd<float, 128>& acc, simd<uint32_t, 128> fr, const int* xi, Acts& ac,
                                          simd<float, 16> dl, simd<float, 16> ml,
                                          const simd<float, 64>& dxv, const simd<float, 64>& sxv) {
    simd<int, 128> bi = fr.template bit_cast_view<int>();
    if constexpr (Q81) {
        simd<int, 64> av = 0;
        simd<uint32_t, 16> hd = 0u;
#pragma unroll
        for (int r = 0; r < NC; ++r) {
            if constexpr (S < 7) {
                av.template select<8, 1>(8 * r) = ac.A[r].template select<8, 1>(9 * S + 1);
                hd[r] = (uint32_t) (int) ac.A[r][9 * S];
            } else {
                av.template select<8, 1>(8 * r) = ac.B[r].template select<8, 1>(1);
                hd[r] = (uint32_t) (int) ac.B[r][0];
            }
        }
        simd<float, 16> dxf = half_of(hd & 0xffffu);
        simd<int, 128> tf = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(simd<int, 128>(0), bi, av);
        simd<int, 128> ts = 0;
        if constexpr (MIN)
            ts = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(simd<int, 128>(0), simd<int, 128>(0x01010101), av);
#pragma unroll
        for (int r = 0; r < NC; ++r) {
            const float dxr = dxf[r];
            simd<int, 16> ti = tf.template select<16, 1>(16 * r);
            simd<float, 16> t = ti;
            simd<float, 16> a = acc.template select<16, 1>(16 * r);
            if constexpr (MIN) {
                simd<int, 16> si = ts.template select<16, 1>(16 * r);
                simd<float, 16> sf = si;
                a += t * (dl * dxr) - ml * (sf * dxr);
            } else {
                a += t * (dl * dxr);
            }
            acc.template select<16, 1>(16 * r) = a;
        }
    } else {
        simd<int, 64> av = block_load<int, 64>(xi + 64 * S);
        simd<int, 128> tf = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(simd<int, 128>(0), bi, av);
#pragma unroll
        for (int r = 0; r < NC; ++r) {
            const float dxr = dxv[8 * S + r];
            simd<int, 16> ti = tf.template select<16, 1>(16 * r);
            simd<float, 16> t = ti;
            simd<float, 16> a = acc.template select<16, 1>(16 * r);
            if constexpr (MIN) {
                const float sxr = sxv[8 * S + r];
                a += t * (dl * dxr) - ml * sxr;
            } else {
                a += t * (dl * dxr);
            }
            acc.template select<16, 1>(16 * r) = a;
        }
    }
}
// the pair P of sub-blocks (2P: the low nibbles of its 32 qs bytes `qd`, 2P + 1: the high ones)
template <int F, int NC, int P, bool Q81>
SYCL_ESIMD_FUNCTION inline void pair(simd<float, 128>& acc, const simd<uint32_t, 128>& qd,
                                     const simd<uint32_t, 128>& qh, simd<uint32_t, 16> h1, simd<uint32_t, 16> h2,
                                     simd<uint32_t, 16> h3, simd<float, 16> d, simd<float, 16> dmin,
                                     const int* xi, Acts& ac, const simd<float, 64>& dxv, const simd<float, 64>& sxv) {
    simd<uint32_t, 128> lo = qd & 0x0f0f0f0fu, hi = (qd >> 4) & 0x0f0f0f0fu;
    if constexpr (F == T_Q5K) {   // the fifth bit: bit 2P / 2P + 1 of the qh byte of the same position
        lo |= ((qh >> (2 * P)) & 0x01010101u) << 4;
        hi |= ((qh >> (2 * P + 1)) & 0x01010101u) << 4;
    }
    simd<float, 16> dl, ml;
    scale_min<2 * P>(h1, h2, h3, d, dmin, dl, ml);
    sub_block<NC, 2 * P, true, Q81>(acc, lo, xi, ac, dl, ml, dxv, sxv);
    scale_min<2 * P + 1>(h1, h2, h3, d, dmin, dl, ml);
    sub_block<NC, 2 * P + 1, true, Q81>(acc, hi, xi, ac, dl, ml, dxv, sxv);
}

// IQ4_XS sub-block S: its 16 qs bytes are dwords 4 (S & 1) .. + 3 of `qq` (the 32 bytes of sub-blocks S and S ^ 1); a
// byte's low nibble is value j of the 32, its high nibble value j + 16.  The table is at offset 0 of shared local memory.
template <int NC, int S, bool Q81>
SYCL_ESIMD_FUNCTION inline void sub_iq4(simd<float, 128>& acc, simd<uint32_t, 128> qq, simd<uint32_t, 16> h0, simd<uint32_t, 16> h1,
                                        simd<float, 16> d, const int* xi, Acts& ac, const simd<float, 64>& dxv) {
    simd<uint32_t, 128> fr;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        simd<uint32_t, 16> v = qq.template select<16, 1>(16 * (4 * (S & 1) + k));
        simd<uint32_t, 16> t0 = slm_gather<uint32_t, 16>((v & 0xffu) << 2), t1 = slm_gather<uint32_t, 16>(((v >> 8) & 0xffu) << 2);
        simd<uint32_t, 16> t2 = slm_gather<uint32_t, 16>(((v >> 16) & 0xffu) << 2), t3 = slm_gather<uint32_t, 16>((v >> 24) << 2);
        simd<uint32_t, 16> P = t0 | (t1 << 8), Q = t2 | (t3 << 8);   // [lo0, lo1, hi0, hi1], [lo2, lo3, hi2, hi3]
        fr.template select<16, 1>(16 * k) = (P & 0xffffu) | (Q << 16);
        fr.template select<16, 1>(16 * (k + 4)) = (P >> 16) | (Q & 0xffff0000u);
    }
    simd<uint32_t, 16> ls = ((h1 >> (4 * S)) & 0xfu) | (((h0 >> (16 + 2 * S)) & 3u) << 4);
    simd<float, 16> dl = d * simd<float, 16>(simd<int, 16>(ls) - 32);
    sub_block<NC, S, false, Q81>(acc, fr, xi, ac, dl, dl, dxv, dxv);
}

constexpr auto kPf = sycl::ext::oneapi::experimental::properties{cache_hint_L1<cache_hint::cached>, cache_hint_L2<cache_hint::cached>};
constexpr auto kAl16 = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::alignment<16>};
// The schedule.  A fixed pool of work-groups (G of them, C threads each: `threads` hardware threads at most, so no second
// wave forms) walks the 16-row tiles; a tile's K is split over the C threads of its group (C = 1, 2 or 4 by the row count:
// few rows are few streams, and a stream alone does not hide the memory latency).  Thread c sums its blocks in K order,
// threads 1 .. C - 1 pass their sums through shared local memory, thread 0 adds them in thread order and stores the rows:
// ((p0 + p1) + p2) + p3, the same at every column count.  The next block's cache lines are prefetched at the top of a block
// (bench, Q4_K 5120 x 10240, 1 column: 74.6 -> 56.7 us), and all of a block's loads go out before its first decode.
// PIPE (the default; STRATA_Q36_XR_PIPE=0: off): a block's loads go out one block ahead, into registers, while the block
// before it is decoded: no prefetch messages, and the memory latency runs under the arithmetic.
template <int F, int NC, int C, bool Q81, bool PIPE> struct TagXr;
template <int F, int NC, int C, bool Q81, bool PIPE>
void launch(sycl::queue& q, const uint8_t* W, const int* Xi, const float* S, float* Y, int bpr, int n_out, int threads) {
    const uint32_t* T = F == T_IQ4XS ? iq4_table(q) : nullptr;
    constexpr uint32_t kTab = F == T_IQ4XS ? 1024u : 0u;   // the table's bytes of shared local memory, before the sums
    const int xs9 = bpr * 8 * 9;                           // (Q81) dwords from a column's q8_1 blocks to the next column's
    const uint32_t rowb = (uint32_t) bpr * blk_bytes<F>();
    const int tiles = n_out / 16, cb = (bpr + C - 1) / C;
    const int G = std::max(1, std::min(tiles, threads / C));
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagXr<F, NC, C, Q81, PIPE>>(sycl::nd_range<2>(sycl::range<2>((size_t) G, (size_t) C), sycl::range<2>(1, (size_t) C)),
                                        [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
            if constexpr (C > 1 || F == T_IQ4XS) slm_init<kTab + 2 * C * 512>();
            const int g = (int) it.get_group(0), c = (int) it.get_local_id(1);
            if constexpr (F == T_IQ4XS) {
                if (c == 0) {
#pragma unroll
                    for (int k = 0; k < 4; ++k) slm_block_store<uint32_t, 64>(256 * k, block_load<uint32_t, 64>(T + 64 * k), kAl16);
                }
                barrier();
            }
            const int kb0 = c * cb, kb1 = (kb0 + cb) < bpr ? (kb0 + cb) : bpr;
            const uint32_t* Wd = reinterpret_cast<const uint32_t*>(W);
            simd<uint32_t, 16> lane(0, 1);
            uint32_t buf = kTab;
            for (int t = g; t < tiles; t += G) {
                simd<uint32_t, 16> ro = (lane + 16u * (uint32_t) t) * rowb;
                simd<float, 128> acc = 0.f;
                Raw cur, nxt;
                if constexpr (PIPE) {
                    if (kb0 < kb1) load_raw<F>(cur, Wd, ro + (uint32_t) kb0 * blk_bytes<F>());
                }
                for (int kb = kb0; kb < kb1; ++kb) {
                    simd<uint32_t, 16> po = ro + (uint32_t) kb * blk_bytes<F>();
                    if constexpr (PIPE) {
                        if (kb + 1 < kb1) load_raw<F>(nxt, Wd, po + blk_bytes<F>());
                    } else {
                        load_raw<F>(cur, Wd, po);
                        if (kb + 1 < kb1) {   // the next block of these rows: its first, second, third and last cache line
#pragma unroll
                            for (int cl = 0; cl < 3; ++cl) prefetch<uint32_t, 16>(Wd, po + (uint32_t) (blk_bytes<F>() + 64 * cl), kPf);
                            prefetch<uint32_t, 16>(Wd, po + (uint32_t) (2 * blk_bytes<F>() - 4), kPf);
                        }
                    }
                    simd<float, 64> dxv = 0.f, sxv = 0.f;
                    if constexpr (!Q81) {
                        const float* sk = S + (size_t) kb * 128;
                        dxv = block_load<float, 64>(sk);
                        if constexpr (F != T_IQ4XS) sxv = block_load<float, 64>(sk + 64);
                    }
                    const int* xi = Q81 ? Xi + (size_t) kb * 72 : Xi + (size_t) kb * 512;
                    Acts ac;
                    if constexpr (Q81) load_acts<NC>(ac, xi, xs9);
                    simd<uint32_t, 16> h0 = cur.h.template select<16, 1>(0), h1 = cur.h.template select<16, 1>(16);
                    if constexpr (F == T_IQ4XS) {
                        simd<float, 16> d = half_of(h0 & 0xffffu);
                        sub_iq4<NC, 0, Q81>(acc, cur.q0, h0, h1, d, xi, ac, dxv);
                        sub_iq4<NC, 1, Q81>(acc, cur.q0, h0, h1, d, xi, ac, dxv);
                        sub_iq4<NC, 2, Q81>(acc, cur.q1, h0, h1, d, xi, ac, dxv);
                        sub_iq4<NC, 3, Q81>(acc, cur.q1, h0, h1, d, xi, ac, dxv);
                        sub_iq4<NC, 4, Q81>(acc, cur.q2, h0, h1, d, xi, ac, dxv);
                        sub_iq4<NC, 5, Q81>(acc, cur.q2, h0, h1, d, xi, ac, dxv);
                        sub_iq4<NC, 6, Q81>(acc, cur.q3, h0, h1, d, xi, ac, dxv);
                        sub_iq4<NC, 7, Q81>(acc, cur.q3, h0, h1, d, xi, ac, dxv);
                    } else {
                        simd<uint32_t, 16> h2 = cur.h.template select<16, 1>(32), h3 = cur.h.template select<16, 1>(48);
                        simd<float, 16> d = half_of(h0 & 0xffffu), dmin = half_of(h0 >> 16);
                        pair<F, NC, 0, Q81>(acc, cur.q0, cur.qh, h1, h2, h3, d, dmin, xi, ac, dxv, sxv);
                        pair<F, NC, 1, Q81>(acc, cur.q1, cur.qh, h1, h2, h3, d, dmin, xi, ac, dxv, sxv);
                        pair<F, NC, 2, Q81>(acc, cur.q2, cur.qh, h1, h2, h3, d, dmin, xi, ac, dxv, sxv);
                        pair<F, NC, 3, Q81>(acc, cur.q3, cur.qh, h1, h2, h3, d, dmin, xi, ac, dxv, sxv);
                    }
                    if constexpr (PIPE) cur = nxt;
                }
                if constexpr (C > 1) {
                    if (c != 0) {
                        slm_block_store<float, 64>(buf + (uint32_t) c * 512u, acc.template select<64, 1>(0), kAl16);
                        if constexpr (NC > 4) slm_block_store<float, 64>(buf + (uint32_t) c * 512u + 256u, acc.template select<64, 1>(64), kAl16);
                    }
                    barrier();
                    if (c == 0) {
#pragma unroll
                        for (int cc = 1; cc < C; ++cc) {
                            simd<float, 64> p0 = slm_block_load<float, 64>(buf + (uint32_t) cc * 512u, kAl16);
                            simd<float, 64> a0 = acc.template select<64, 1>(0);
                            a0 += p0;
                            acc.template select<64, 1>(0) = a0;
                            if constexpr (NC > 4) {
                                simd<float, 64> p1 = slm_block_load<float, 64>(buf + (uint32_t) cc * 512u + 256u, kAl16);
                                simd<float, 64> a1 = acc.template select<64, 1>(64);
                                a1 += p1;
                                acc.template select<64, 1>(64) = a1;
                            }
                        }
                    }
                    buf = buf == kTab ? kTab + (uint32_t) (C * 512) : kTab;   // the next tile's sums go to the other half: this one is read after the barrier
                }
                if (c == 0) {
#pragma unroll
                    for (int r = 0; r < NC; ++r)
                        block_store<float, 16>(Y + (size_t) r * n_out + 16 * (size_t) t, acc.template select<16, 1>(16 * r));
                }
            }
        });
    });
}
// STRATA_Q36_XR_THREADS (1024): the pool's hardware threads.  STRATA_Q36_XR_C (1, 2, 4; unset: by the row count): threads a tile.
inline int pool_threads() {
    static const int v = [] { const char* e = std::getenv("STRATA_Q36_XR_THREADS"); const int n = e ? std::atoi(e) : 0; return n >= 16 ? n : 1024; }();
    return v;
}
// The threads a tile, by the shape alone (never by the column count: the sums' order follows it).  Bench on the B70 (Q4_K,
// one-message loads, 2026-11-12, us at 1 / 4 / 5 columns): 4 threads a tile on every row of 16384 or less (5120 x 10240:
// 57.8 / 61.4 / 61.5 at 1 thread, 54.1 / 56.4 / 57.1 at 4, KREG 55.0 / 59.0 / 69.4; 5120 x 6144: 35.6 / 39.9 / 41.1 -> 33.2 /
// 34.2 / 34.7; 5120 x 17408: 90.4 / 93.7 / 94.8 at 4); 2 on a wider row (ffn_down, 17408 x 5120: 99.1 / 103.1 / 101.4 at 2,
// 98.3 / 103.4 / 104.9 at 4).  STRATA_Q36_XR_C=1, 2 or 4: every shape.
inline int chunks_of(int n_in, int n_out) {
    static const int e = [] { const char* v = std::getenv("STRATA_Q36_XR_C"); const int n = v ? std::atoi(v) : 0; return n == 1 || n == 2 || n == 4 ? n : 0; }();
    if (e) return e;
    (void) n_out;
    return n_in > 16384 ? 2 : 4;
}
template <int F, int NC>
void dispatch(sycl::queue& q, const uint8_t* W, const int* Xi, const float* S, float* Y, int bpr, int n_out, bool q81) {
    const int c = chunks_of(bpr * 256, n_out), nt = pool_threads();
    (void) q81;   // (the callers pass the q8_1 image: mmvq declines the other)
    // (PIPE = false: with the one-message loads, prefetch beat the register pipeline on every shape, 5120 x 17408 Q4_K at 4
    // columns 94 us against 107)
    if (c == 4) launch<F, NC, 4, true, false>(q, W, Xi, S, Y, bpr, n_out, nt);
    else if (c == 2) launch<F, NC, 2, true, false>(q, W, Xi, S, Y, bpr, n_out, nt);
    else launch<F, NC, 1, true, false>(q, W, Xi, S, Y, bpr, n_out, nt);
}
template <int F>
void by_cols(sycl::queue& q, const uint8_t* W, const int* Xi, const float* S, float* Y, int bpr, int n_out, int ncols, bool q81) {
    switch (ncols) {
        case 1: dispatch<F, 1>(q, W, Xi, S, Y, bpr, n_out, q81); break;
        case 2: dispatch<F, 2>(q, W, Xi, S, Y, bpr, n_out, q81); break;
        case 3: dispatch<F, 3>(q, W, Xi, S, Y, bpr, n_out, q81); break;
        case 4: dispatch<F, 4>(q, W, Xi, S, Y, bpr, n_out, q81); break;
        case 5: dispatch<F, 5>(q, W, Xi, S, Y, bpr, n_out, q81); break;
        case 6: dispatch<F, 6>(q, W, Xi, S, Y, bpr, n_out, q81); break;
        case 7: dispatch<F, 7>(q, W, Xi, S, Y, bpr, n_out, q81); break;
        default: dispatch<F, 8>(q, W, Xi, S, Y, bpr, n_out, q81); break;
    }
}
}   // namespace

void init(void* stream) { (void) iq4_table(*strata::q_of(stream)); }
bool handles(int t) { return t == T_Q4K || t == T_Q5K || t == T_IQ4XS; }
// [n_in / 32][8 columns][32 int8], then [n_in / 256][8 sub-blocks][8] scales and [..][8][8] scaled sums
std::size_t act_bytes(int n_in) { return (std::size_t) (n_in / 32) * 256 + (std::size_t) (n_in / 256) * 128 * sizeof(float); }

void quantize(const float* x, int n_in, int ncols, void* act, void* stream) {
    sycl::queue& q = *strata::q_of(stream);
    (void) iq4_table(q);   // made here, outside any graph recording a product call may be part of
    const int nsb = n_in / 32;
    int8_t* xi = static_cast<int8_t*>(act);
    float* sc = reinterpret_cast<float*>(xi + (std::size_t) nsb * 256);
    q.parallel_for<class q36xr_quantize>(sycl::range<2>((size_t) nsb, 8), [=](sycl::id<2> id) {
        const int sb = (int) id[0], r = (int) id[1];
        int8_t* o = xi + (std::size_t) sb * 256 + r * 32;
        float* so = sc + (std::size_t) (sb >> 3) * 128 + (sb & 7) * 8 + r;
        if (r >= ncols) {
            for (int i = 0; i < 32; ++i) o[i] = 0;
            so[0] = 0.0f; so[64] = 0.0f;
            return;
        }
        const float* xr = x + (std::size_t) r * n_in + (std::size_t) sb * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = sycl::fmax(amax, sycl::fabs(xr[i]));
        const float d = amax / 127.0f, idv = d != 0.0f ? 1.0f / d : 0.0f;
        int sum = 0;
        for (int i = 0; i < 32; ++i) {
            const int v = (int) sycl::round(xr[i] * idv);
            o[i] = (int8_t) v;
            sum += v;
        }
        so[0] = d;
        so[64] = d * (float) sum;
    });
}

bool mmvq(int type, const void* weights, const void* act, float* y, int n_in, int n_out, int ncols, void* stream, bool q81) {
    if (!q81) return false;   // only the q8_1 image is built in (quantize()'s own block: the code is kept, no kernel is)
    if (!handles(type) || ncols < 1 || ncols > 8 || n_in <= 0 || n_in % 256 != 0 || n_out <= 0 || n_out % 16 != 0) return false;
    // IQ4_XS: only a row wider than 16384 (ffn_down).  Its codebook lookups make the form slower than KREG2 on the other
    // shapes (5120 x 10240, 4 columns: 77-83 us against 64.5), and on the wide row it reads the weights once at 5-8 columns
    // (270 -> 127 us at 5).  STRATA_Q36_XR_IQ4=all: every shape.
    static const bool iq4_all = [] { const char* e = std::getenv("STRATA_Q36_XR_IQ4"); return e && e[0] == 'a'; }();
    if (type == T_IQ4XS && n_in <= 16384 && !iq4_all) return false;
    if (!weights || !act || !y || (reinterpret_cast<uintptr_t>(weights) & 3u) != 0) return false;
    const int bpr = n_in / 256;
    const std::size_t rowb = (std::size_t) bpr * (type == T_Q4K ? 144 : type == T_Q5K ? 176 : 136);
    if (rowb * (std::size_t) n_out >= ((std::size_t) 1 << 32)) return false;
    sycl::queue& q = *strata::q_of(stream);
    const auto* W = static_cast<const uint8_t*>(weights);
    const int* Xi = static_cast<const int*>(act);
    const float* S = reinterpret_cast<const float*>(static_cast<const uint8_t*>(act) + (std::size_t) (n_in / 32) * 256);
    if ((reinterpret_cast<uintptr_t>(act) & 3u) != 0) return false;
    if (type == T_Q4K) by_cols<T_Q4K>(q, W, Xi, S, y, bpr, n_out, ncols, q81);
    else if (type == T_Q5K) by_cols<T_Q5K>(q, W, Xi, S, y, bpr, n_out, ncols, q81);
    else by_cols<T_IQ4XS>(q, W, Xi, S, y, bpr, n_out, ncols, q81);
    return true;
}
}   // namespace strata::q36xr
