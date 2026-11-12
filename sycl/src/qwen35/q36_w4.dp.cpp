// q36_w4.dp.cpp - B70W4: Qwen3.8-27B's int4 pack on the Arc Pro B70's matrix engine (XMX, ESIMD).  strata-q36 only; its
// own library (libq36_w4), so a change here recompiles these kernels alone and they take their own register-file size.
// The pack: tools/qwen35/b70w4_pack.py.  The API: q36_xr_dense.hpp (w4_bytes, mmvq_w4, w4_dequant_f16).
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "strata/sycl_queue.hpp"
#include "q36_xr_dense.hpp"

namespace strata::q36xr {
namespace {
using namespace sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

SYCL_ESIMD_FUNCTION inline simd<float, 16> half_of(simd<uint32_t, 16> h16) {
    simd<uint16_t, 16> lo = h16.template bit_cast_view<uint16_t>().template select<16, 2>(0);
    simd<sycl::half, 16> hh = lo.template bit_cast_view<sycl::half>();
    simd<float, 16> f = hh;
    return f;
}
constexpr auto kPf = sycl::ext::oneapi::experimental::properties{cache_hint_L1<cache_hint::cached>, cache_hint_L2<cache_hint::cached>};
constexpr auto kAl16 = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::alignment<16>};
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
// ===== B70W4: the int4 pack (tools/qwen35/b70w4_pack.py).  A matrix is records of 16 rows x 256 inputs, tile after tile,
// chunk after chunk: 4 x 128 dwords of nibbles (pair p of the chunk's 32-weight sub-blocks: 2p low, 2p + 1 high; dword d of
// row i at [128 p + 16 d + i]) and 2 x 16 dwords (group g of 128 inputs, row i: fp16 scale | zero << 16).  A thread owns a
// tile and reads its records in order with block loads: no gather, no lookup, one format.  Per row and 32 weights:
// y += (scale x dx) x (dot - zero x sum), dot and sum the s8 DPAS of the nibbles (and of ones) with a column's activations:
// the integer part is exact.  K in order, each column on its own: a T-column call is bitwise T one-column calls.
constexpr int kW4Rec = 544;   // dwords a record
constexpr int kW4MaxCols = 32;
// Columns: a DPAS takes 8 (its repeat count).  A call of NC columns (up to 32) runs them as groups of 8: the chunk's weights
// are loaded once, then for each group its activations and the two DPAS a sub-block; a group more costs its arithmetic and
// its activation loads, not a second read of the weights.  NG = the groups, CG(g) = the columns of group g.
template <int NC> constexpr int w4_ng() { return (NC + 7) / 8; }
template <int NC, int GI> constexpr int w4_cg() { return NC - 8 * GI < 8 ? NC - 8 * GI : 8; }
// sub-block S of a chunk for one column group: the 32 nibbles of 16 rows x the group's columns
template <int CG> SYCL_ESIMD_FUNCTION inline void w4_sub(simd<float, 128>& acc, simd<uint32_t, 128> fr, simd<int, 64> av, simd<float, 16> dxf,
                                                        simd<float, 16> sc, simd<int, 16> z) {
    simd<int, 128> bi = fr.template bit_cast_view<int>();
    simd<int, 128> tf = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(simd<int, 128>(0), bi, av);
    simd<int, 128> ts = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(simd<int, 128>(0), simd<int, 128>(0x01010101), av);
#pragma unroll
    for (int r = 0; r < CG; ++r) {
        const float dxr = dxf[r];
        simd<int, 16> ti = tf.template select<16, 1>(16 * r), si = ts.template select<16, 1>(16 * r);
        simd<int, 16> u = ti - z * si;
        simd<float, 16> t = u;
        simd<float, 16> a = acc.template select<16, 1>(16 * r);
        a += t * (sc * dxr);
        acc.template select<16, 1>(16 * r) = a;
    }
}
// sub-block S's activations of a group's columns out of the chunk's preloaded q8_1 blocks: the A operand and the scales
template <int CG, int S> SYCL_ESIMD_FUNCTION inline void w4_acts(Acts& ac, simd<int, 64>& av, simd<float, 16>& dxf) {
    simd<uint32_t, 16> hd = 0u;
    av = 0;
#pragma unroll
    for (int r = 0; r < CG; ++r) {
        if constexpr (S < 7) {
            av.template select<8, 1>(8 * r) = ac.A[r].template select<8, 1>(9 * S + 1);
            hd[r] = (uint32_t) (int) ac.A[r][9 * S];
        } else {
            av.template select<8, 1>(8 * r) = ac.B[r].template select<8, 1>(1);
            hd[r] = (uint32_t) (int) ac.B[r][0];
        }
    }
    dxf = half_of(hd & 0xffffu);
}
// pair P of a chunk (sub-blocks 2P and 2P + 1, both of group P >> 1) for one column group
template <int CG, int P> SYCL_ESIMD_FUNCTION inline void w4_pair(simd<float, 128>& acc, const simd<uint32_t, 128>& qd, Acts& ac,
                                                                simd<float, 16> sc, simd<int, 16> z) {
    simd<int, 64> av;
    simd<float, 16> dxf;
    w4_acts<CG, 2 * P>(ac, av, dxf);
    w4_sub<CG>(acc, qd & 0x0f0f0f0fu, av, dxf, sc, z);
    w4_acts<CG, 2 * P + 1>(ac, av, dxf);
    w4_sub<CG>(acc, (qd >> 4) & 0x0f0f0f0fu, av, dxf, sc, z);
}
// one column group of a chunk: its activations (xi: the group's first column), then the four pairs
template <int CG> SYCL_ESIMD_FUNCTION inline void w4_group(simd<float, 128>& acc, const simd<uint32_t, 128>& q0, const simd<uint32_t, 128>& q1,
                                                         const simd<uint32_t, 128>& q2, const simd<uint32_t, 128>& q3, const int* xi, int xs9,
                                                         simd<float, 16> sc0, simd<int, 16> z0, simd<float, 16> sc1, simd<int, 16> z1) {
    Acts ac;
    load_acts<CG>(ac, xi, xs9);
    w4_pair<CG, 0>(acc, q0, ac, sc0, z0);
    w4_pair<CG, 1>(acc, q1, ac, sc0, z0);
    w4_pair<CG, 2>(acc, q2, ac, sc1, z1);
    w4_pair<CG, 3>(acc, q3, ac, sc1, z1);
}
template <int NC, int C> struct TagW4;
template <int NC, int C>
void launch_w4(sycl::queue& q, const uint32_t* W, const int* Xi, float* Y, int bpr, int n_out, int threads) {
    constexpr int NG = w4_ng<NC>();
    const int tiles = n_out / 16, cb = (bpr + C - 1) / C;
    const int G = std::max(1, std::min(tiles, threads / C));
    const int xs9 = bpr * 8 * 9;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagW4<NC, C>>(sycl::nd_range<2>(sycl::range<2>((size_t) G, (size_t) C), sycl::range<2>(1, (size_t) C)),
                                     [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
            if constexpr (C > 1) slm_init<2 * C * 512 * NG>();
            const int g = (int) it.get_group(0), c = (int) it.get_local_id(1);
            const int kb0 = c * cb, kb1 = (kb0 + cb) < bpr ? (kb0 + cb) : bpr;
            uint32_t buf = 0;
            for (int t = g; t < tiles; t += G) {
                const uint32_t* tile = W + (size_t) t * (size_t) bpr * kW4Rec;
                simd<float, 128> acc[NG];
#pragma unroll
                for (int gi = 0; gi < NG; ++gi) acc[gi] = 0.f;
                for (int kb = kb0; kb < kb1; ++kb) {
                    const uint32_t* rec = tile + (size_t) kb * kW4Rec;
                    simd<uint32_t, 128> q0, q1, q2, q3;
                    q0.template select<64, 1>(0) = block_load<uint32_t, 64>(rec);
                    q0.template select<64, 1>(64) = block_load<uint32_t, 64>(rec + 64);
                    q1.template select<64, 1>(0) = block_load<uint32_t, 64>(rec + 128);
                    q1.template select<64, 1>(64) = block_load<uint32_t, 64>(rec + 192);
                    q2.template select<64, 1>(0) = block_load<uint32_t, 64>(rec + 256);
                    q2.template select<64, 1>(64) = block_load<uint32_t, 64>(rec + 320);
                    q3.template select<64, 1>(0) = block_load<uint32_t, 64>(rec + 384);
                    q3.template select<64, 1>(64) = block_load<uint32_t, 64>(rec + 448);
                    simd<uint32_t, 32> m = block_load<uint32_t, 32>(rec + 512);
                    if (kb + 1 < kb1) {   // the next record: its 34 cache lines
                        prefetch<uint32_t, 16>(rec + kW4Rec, simd<uint32_t, 16>(0, 64), kPf);
                        prefetch<uint32_t, 16>(rec + kW4Rec + 256, simd<uint32_t, 16>(0, 64), kPf);
                        prefetch<uint32_t, 2>(rec + kW4Rec + 512, simd<uint32_t, 2>(0, 64), kPf);
                    }
                    simd<uint32_t, 16> m0 = m.template select<16, 1>(0), m1 = m.template select<16, 1>(16);
                    simd<float, 16> sc0 = half_of(m0 & 0xffffu), sc1 = half_of(m1 & 0xffffu);
                    simd<int, 16> z0 = simd<int, 16>(m0 >> 16), z1 = simd<int, 16>(m1 >> 16);
                    const int* xi = Xi + (size_t) kb * 72;
                    w4_group<w4_cg<NC, 0>()>(acc[0], q0, q1, q2, q3, xi, xs9, sc0, z0, sc1, z1);
                    if constexpr (NG > 1) w4_group<w4_cg<NC, 1>()>(acc[1], q0, q1, q2, q3, xi + (size_t) 8 * xs9, xs9, sc0, z0, sc1, z1);
                    if constexpr (NG > 2) w4_group<w4_cg<NC, 2>()>(acc[2], q0, q1, q2, q3, xi + (size_t) 16 * xs9, xs9, sc0, z0, sc1, z1);
                    if constexpr (NG > 3) w4_group<w4_cg<NC, 3>()>(acc[3], q0, q1, q2, q3, xi + (size_t) 24 * xs9, xs9, sc0, z0, sc1, z1);
                }
                if constexpr (C > 1) {
                    if (c != 0) {
#pragma unroll
                        for (int gi = 0; gi < NG; ++gi) {
                            slm_block_store<float, 64>(buf + ((uint32_t) c * NG + gi) * 512u, acc[gi].template select<64, 1>(0), kAl16);
                            slm_block_store<float, 64>(buf + ((uint32_t) c * NG + gi) * 512u + 256u, acc[gi].template select<64, 1>(64), kAl16);
                        }
                    }
                    barrier();
                    if (c == 0) {
#pragma unroll
                        for (int cc = 1; cc < C; ++cc)
#pragma unroll
                            for (int gi = 0; gi < NG; ++gi) {
                                simd<float, 64> p0 = slm_block_load<float, 64>(buf + ((uint32_t) cc * NG + gi) * 512u, kAl16);
                                simd<float, 64> p1 = slm_block_load<float, 64>(buf + ((uint32_t) cc * NG + gi) * 512u + 256u, kAl16);
                                simd<float, 64> a0 = acc[gi].template select<64, 1>(0), a1 = acc[gi].template select<64, 1>(64);
                                a0 += p0;
                                a1 += p1;
                                acc[gi].template select<64, 1>(0) = a0;
                                acc[gi].template select<64, 1>(64) = a1;
                            }
                    }
                    buf = buf == 0u ? (uint32_t) (C * NG * 512) : 0u;
                }
                if (c == 0) {
#pragma unroll
                    for (int r = 0; r < NC; ++r)
                        block_store<float, 16>(Y + (size_t) r * n_out + 16 * (size_t) t, acc[r / 8].template select<16, 1>(16 * (r % 8)));
                }
            }
        });
    });
}

// STRATA_Q36_XR_THREADS (1024): the pool's hardware threads
inline int pool_threads() {
    static const int v = [] { const char* e = std::getenv("STRATA_Q36_XR_THREADS"); const int n = e ? std::atoi(e) : 0; return n >= 16 ? n : 1024; }();
    return v;
}
// STRATA_Q36_W4_C (1, 2, 4; unset: 4, a row wider than 16384 2): the W4 form's threads a tile
inline int w4_chunks(int n_in) {
    static const int e = [] { const char* v = std::getenv("STRATA_Q36_W4_C"); const int n = v ? std::atoi(v) : 0; return n == 1 || n == 2 || n == 4 ? n : 0; }();
    return e ? e : n_in > 16384 ? 2 : 4;
}
template <int NC>
void dispatch_w4(sycl::queue& q, const uint32_t* W, const int* Xi, float* Y, int bpr, int n_out) {
    const int c = w4_chunks(bpr * 256), nt = pool_threads();
    if (c == 4) launch_w4<NC, 4>(q, W, Xi, Y, bpr, n_out, nt);
    else if (c == 2) launch_w4<NC, 2>(q, W, Xi, Y, bpr, n_out, nt);
    else launch_w4<NC, 1>(q, W, Xi, Y, bpr, n_out, nt);
}
}   // namespace

std::size_t w4_bytes(int n_in, int n_out) { return (std::size_t) (n_out / 16) * (n_in / 256) * kW4Rec * 4; }
bool mmvq_w4(const void* weights, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (ncols < 1 || ncols > kW4MaxCols || n_in <= 0 || n_in % 256 != 0 || n_out <= 0 || n_out % 16 != 0) return false;
    if (!weights || !xq || !y || (reinterpret_cast<uintptr_t>(weights) & 63u) != 0 || (reinterpret_cast<uintptr_t>(xq) & 3u) != 0) return false;
    sycl::queue& q = *strata::q_of(stream);
    const auto* W = static_cast<const uint32_t*>(weights);
    const int* Xi = static_cast<const int*>(xq);
    const int bpr = n_in / 256;
    switch (ncols) {
#define W4C(N) case N: dispatch_w4<N>(q, W, Xi, y, bpr, n_out); break;
        W4C(1) W4C(2) W4C(3) W4C(4) W4C(5) W4C(6) W4C(7) W4C(8) W4C(9) W4C(10) W4C(11) W4C(12) W4C(13) W4C(14) W4C(15) W4C(16)
        W4C(17) W4C(18) W4C(19) W4C(20) W4C(21) W4C(22) W4C(23) W4C(24) W4C(25) W4C(26) W4C(27) W4C(28) W4C(29) W4C(30) W4C(31) W4C(32)
#undef W4C
        default: return false;
    }
    return true;
}

// A prompt chunk's product is an FP16 GEMM: the pack as FP16 [n_out][n_in] (scale x (q - zero), rounded to FP16).
// One thread a record (16 rows x 256 inputs): the record by block loads, a row's 256 values assembled in registers and
// stored as two 256-byte blocks.  (A plain kernel with one work-item per 32 values wrote 81 GB/s of FP16: 317 ms for the
// 400 matrices of a prompt pass.)
struct TagW4Deq;
void w4_dequant_f16(const void* weights, uint16_t* out, int n_in, int n_out, void* stream) {
    sycl::queue& q = *strata::q_of(stream);
    const auto* W = static_cast<const uint32_t*>(weights);
    const int bpr = n_in / 256, tiles = n_out / 16;
    uint32_t* O = reinterpret_cast<uint32_t*>(out);
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagW4Deq>(sycl::range<2>((size_t) tiles, (size_t) bpr), [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {
            const int t = (int) id[0], c = (int) id[1];
            const uint32_t* rec = W + ((size_t) t * bpr + c) * kW4Rec;
            simd<uint32_t, 128> qd[4];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                qd[p].template select<64, 1>(0) = block_load<uint32_t, 64>(rec + 128 * p);
                qd[p].template select<64, 1>(64) = block_load<uint32_t, 64>(rec + 128 * p + 64);
            }
            simd<uint32_t, 32> m = block_load<uint32_t, 32>(rec + 512);
            simd<uint32_t, 16> m0 = m.template select<16, 1>(0), m1 = m.template select<16, 1>(16);
            simd<float, 16> sc0 = half_of(m0 & 0xffffu), sc1 = half_of(m1 & 0xffffu);
            simd<int, 16> z0 = simd<int, 16>(m0 >> 16), z1 = simd<int, 16>(m1 >> 16);
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                simd<sycl::half, 256> row;
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const float sc = p < 2 ? (float) sc0[i] : (float) sc1[i];
                    const int z = p < 2 ? (int) z0[i] : (int) z1[i];
                    simd<uint32_t, 8> w = qd[p].template select<8, 16>(i);   // row i's 32 bytes of the pair
                    simd<uint32_t, 8> lo = w & 0x0f0f0f0fu, hi = (w >> 4) & 0x0f0f0f0fu;
                    simd<uint8_t, 32> bl = lo.template bit_cast_view<uint8_t>(), bh = hi.template bit_cast_view<uint8_t>();
                    simd<int, 32> il = bl, ih = bh;
                    simd<float, 32> fl = simd<float, 32>(il - z) * sc, fh = simd<float, 32>(ih - z) * sc;
                    simd<sycl::half, 32> hl = fl, hh = fh;
                    row.template select<32, 1>(64 * p) = hl;
                    row.template select<32, 1>(64 * p + 32) = hh;
                }
                simd<uint32_t, 128> r32 = row.template bit_cast_view<uint32_t>();
                uint32_t* o = O + ((size_t) (16 * t + i) * n_in + (size_t) c * 256) / 2;
                block_store<uint32_t, 64>(o, r32.template select<64, 1>(0));
                block_store<uint32_t, 64>(o + 64, r32.template select<64, 1>(64));
            }
        });
    });
}

}   // namespace strata::q36xr
