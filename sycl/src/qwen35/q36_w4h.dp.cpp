// q36_w4h.dp.cpp - W4A16: Qwen3.8-27B's int4 pack against FP16 activations on the Arc Pro B70's matrix engine (the form
// vLLM runs).  In libq36_w4, compiled with -fp-model=precise: a column's result must not depend on how many columns the
// call has (a kernel per column count).  The API: q36_xr_dense.hpp (mmvq_w4h, w4h_pack, w4h_swiglu_pack).
//
// The int8-activation form (q36_w4.dp.cpp) pays float work per column and 32 weights (the activations' scale per 32
// values, the zero times their sum): 5120 x 17408 takes 82.7 / 115.1 / 184.3 us at 1 / 16 / 32 columns.  Here a
// sub-block's 32 nibbles become 32 half floats (q - zero: small integers, exact) in two DPAS operands of 16 inputs, the
// activations of 8 columns for those inputs are one block load, the sums stay unscaled over a group of 128 inputs and
// take the row's scale once: 87.2 / 93.9 / 114.8 us.  A column more costs its share of a DPAS.
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
constexpr int kW4Rec = 544;   // dwords a record (b70w4_pack.py)
constexpr int kW4MaxCols = 32;
template <int NC> constexpr int w4_ng() { return (NC + 7) / 8; }

// ===== W4A16: the pack against FP16 activations (q36_xr_dense.hpp, mmvq_w4h).  A sub-block's 32 nibbles become 32 half
// floats (q - zero: small integers, exact) in two DPAS operands of 16 inputs; the activations of 8 columns for those 16
// inputs are one block load (x16's layout); the sums stay unscaled over a group of 128 inputs, then one multiply-add a
// column by the row's scale.
template <int NC, int C> struct TagW4H;
template <int NC, int C>
void launch_w4h(sycl::queue& q, const uint32_t* W, const uint32_t* X, float* Y, int bpr, int n_out, int threads) {
    constexpr int NG = w4_ng<NC>();
    constexpr int NCP = NG * 8;   // the columns x16 holds
    const int tiles = n_out / 16, cb = (bpr + C - 1) / C;
    const int G = std::max(1, std::min(tiles, threads / C));
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagW4H<NC, C>>(sycl::nd_range<2>(sycl::range<2>((size_t) G, (size_t) C), sycl::range<2>(1, (size_t) C)),
                                      [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
            if constexpr (C > 1) slm_init<2 * C * 512 * NG>();
            const int g = (int) it.get_group(0), c = (int) it.get_local_id(1);
            const int kb0 = c * cb, kb1 = (kb0 + cb) < bpr ? (kb0 + cb) : bpr;
            uint32_t buf = 0;
            for (int t = g; t < tiles; t += G) {
                const uint32_t* tile = W + (size_t) t * (size_t) bpr * kW4Rec;
                simd<float, 128> y[NG];
#pragma unroll
                for (int gi = 0; gi < NG; ++gi) y[gi] = 0.f;
                for (int kb = kb0; kb < kb1; ++kb) {
                    const uint32_t* rec = tile + (size_t) kb * kW4Rec;
                    simd<uint32_t, 128> qd[4];
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        qd[p].template select<64, 1>(0) = block_load<uint32_t, 64>(rec + 128 * p);
                        qd[p].template select<64, 1>(64) = block_load<uint32_t, 64>(rec + 128 * p + 64);
                    }
                    simd<uint32_t, 32> m = block_load<uint32_t, 32>(rec + 512);
                    if (kb + 1 < kb1) {   // the next record: its 34 cache lines
                        prefetch<uint32_t, 16>(rec + kW4Rec, simd<uint32_t, 16>(0, 64), kPf);
                        prefetch<uint32_t, 16>(rec + kW4Rec + 256, simd<uint32_t, 16>(0, 64), kPf);
                        prefetch<uint32_t, 2>(rec + kW4Rec + 512, simd<uint32_t, 2>(0, 64), kPf);
                    }
                    const uint32_t* xb = X + (size_t) kb * 16 * NCP * 8;   // the chunk's 16 blocks of 16 inputs
#pragma unroll
                    for (int g2 = 0; g2 < 2; ++g2) {   // a group of 128 inputs: its scale and zero
                        simd<uint32_t, 16> mg = m.template select<16, 1>(16 * g2);
                        simd<float, 16> sc = half_of(mg & 0xffffu);
                        simd<uint16_t, 16> z16 = mg >> 16;
                        simd<int16_t, 32> zr;
                        zr.template select<16, 2>(0) = z16.template bit_cast_view<int16_t>();
                        zr.template select<16, 2>(1) = z16.template bit_cast_view<int16_t>();
                        simd<float, 128> acc[NG];
#pragma unroll
                        for (int gi = 0; gi < NG; ++gi) acc[gi] = 0.f;
#pragma unroll
                        for (int s4 = 0; s4 < 4; ++s4) {   // the group's 4 sub-blocks: pair 2 g2 + s4 / 2, low or high nibbles
                            const int p = 2 * g2 + s4 / 2, sb = 4 * g2 + s4;
                            simd<uint32_t, 128> v = (s4 & 1) ? simd<uint32_t, 128>((qd[p] >> 4) & 0x0f0f0f0fu) : simd<uint32_t, 128>(qd[p] & 0x0f0f0f0fu);
#pragma unroll
                            for (int hh = 0; hh < 2; ++hh) {   // 16 inputs: dwords 4 hh .. 4 hh + 3 of the sub-block
                                simd<sycl::half, 256> bw;
#pragma unroll
                                for (int dd = 0; dd < 8; ++dd) {
                                    simd<uint32_t, 16> qD = v.template select<16, 1>(16 * (4 * hh + dd / 2));
                                    simd<uint16_t, 32> w16 = qD.template bit_cast_view<uint16_t>();
                                    simd<uint16_t, 16> sel = w16.template select<16, 2>(dd % 2);
                                    simd<uint8_t, 32> b8 = sel.template bit_cast_view<uint8_t>();
                                    simd<int16_t, 32> vi = b8;
                                    vi -= zr;
                                    simd<sycl::half, 32> hv = vi;
                                    bw.template select<32, 1>(32 * dd) = hv;
                                }
                                const uint32_t* xk = xb + (size_t) (sb * 2 + hh) * NCP * 8;
#pragma unroll
                                for (int gi = 0; gi < NG; ++gi) {
                                    simd<uint32_t, 64> xa = block_load<uint32_t, 64>(xk + 64 * gi);
                                    simd<sycl::half, 128> ah = xa.template bit_cast_view<sycl::half>();
                                    acc[gi] = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(acc[gi], bw, ah);
                                }
                            }
                        }
#pragma unroll
                        for (int gi = 0; gi < NG; ++gi)
#pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                simd<float, 16> a = acc[gi].template select<16, 1>(16 * r);
                                simd<float, 16> yv = y[gi].template select<16, 1>(16 * r);
                                yv += a * sc;
                                y[gi].template select<16, 1>(16 * r) = yv;
                            }
                    }
                }
                if constexpr (C > 1) {
                    if (c != 0) {
#pragma unroll
                        for (int gi = 0; gi < NG; ++gi) {
                            slm_block_store<float, 64>(buf + ((uint32_t) c * NG + gi) * 512u, y[gi].template select<64, 1>(0), kAl16);
                            slm_block_store<float, 64>(buf + ((uint32_t) c * NG + gi) * 512u + 256u, y[gi].template select<64, 1>(64), kAl16);
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
                                simd<float, 64> a0 = y[gi].template select<64, 1>(0), a1 = y[gi].template select<64, 1>(64);
                                a0 += p0;
                                a1 += p1;
                                y[gi].template select<64, 1>(0) = a0;
                                y[gi].template select<64, 1>(64) = a1;
                            }
                    }
                    buf = buf == 0u ? (uint32_t) (C * NG * 512) : 0u;
                }
                if (c == 0) {
#pragma unroll
                    for (int r = 0; r < NC; ++r)
                        block_store<float, 16>(Y + (size_t) r * n_out + 16 * (size_t) t, y[r / 8].template select<16, 1>(16 * (r % 8)));
                }
            }
        });
    });
}

inline int pool_threads() {
    static const int v = [] { const char* e = std::getenv("STRATA_Q36_XR_THREADS"); const int n = e ? std::atoi(e) : 0; return n >= 16 ? n : 1024; }();
    return v;
}
// STRATA_Q36_W4H_C (1, 2, 4): the threads a tile (4: 17408 x 5120 at 16 / 32 columns 100.6 / 199.0 us, 2: 107.9 / 240.4)
inline int w4h_chunks() {
    static const int e = [] { const char* v = std::getenv("STRATA_Q36_W4H_C"); const int n = v ? std::atoi(v) : 0; return n == 1 || n == 2 || n == 4 ? n : 4; }();
    return e;
}
template <int NC>
void dispatch_w4h(sycl::queue& q, const uint32_t* W, const uint32_t* X, float* Y, int bpr, int n_out) {
    const int c = w4h_chunks(), nt = pool_threads();
    if (c == 4) launch_w4h<NC, 4>(q, W, X, Y, bpr, n_out, nt);
    else if (c == 2) launch_w4h<NC, 2>(q, W, X, Y, bpr, n_out, nt);
    else launch_w4h<NC, 1>(q, W, X, Y, bpr, n_out, nt);
}
}   // namespace

std::size_t w4h_act_halfs(int n_in, int ncols) { return (std::size_t) (n_in / 16) * ((ncols + 7) / 8 * 8) * 16; }

// (the row counts a window can have by groups of 8 columns: the padded columns of a group are computed and dropped)
bool mmvq_w4h(const void* weights, const uint16_t* x16, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (ncols < 1 || ncols > kW4MaxCols || n_in <= 0 || n_in % 256 != 0 || n_out <= 0 || n_out % 16 != 0) return false;
    if (!weights || !x16 || !y || (reinterpret_cast<uintptr_t>(weights) & 63u) != 0 || (reinterpret_cast<uintptr_t>(x16) & 3u) != 0) return false;
    sycl::queue& q = *strata::q_of(stream);
    const auto* W = static_cast<const uint32_t*>(weights);
    const auto* X = reinterpret_cast<const uint32_t*>(x16);
    const int bpr = n_in / 256;
    switch (ncols) {
#define W4H(N) case N: dispatch_w4h<N>(q, W, X, y, bpr, n_out); break;
        W4H(1) W4H(2) W4H(3) W4H(4) W4H(5) W4H(6) W4H(7) W4H(8) W4H(9) W4H(10) W4H(11) W4H(12) W4H(13) W4H(14) W4H(15) W4H(16)
        W4H(17) W4H(18) W4H(19) W4H(20) W4H(21) W4H(22) W4H(23) W4H(24) W4H(25) W4H(26) W4H(27) W4H(28) W4H(29) W4H(30) W4H(31) W4H(32)
#undef W4H
        default: return false;
    }
    return true;
}


// x [ncols][n_in] (rows of ldx floats) -> x16.  A work-item: 16 inputs of one column.
void w4h_pack(const float* x, int ldx, uint16_t* x16, int n_in, int ncols, void* stream) {
    sycl::queue& q = *strata::q_of(stream);
    const size_t ncp = (size_t) (ncols + 7) / 8 * 8;
    q.parallel_for<class q36_w4h_pack_k>(sycl::range<2>((size_t) n_in / 16, (size_t) ncols), [=](sycl::id<2> i) {
        const float* s = x + i[1] * (size_t) ldx + i[0] * 16;
        uint16_t* d = x16 + (i[0] * ncp + i[1]) * 16;
        sycl::vec<float, 16> v = *reinterpret_cast<const sycl::vec<float, 16>*>(s);
        sycl::vec<sycl::half, 16> hv = v.convert<sycl::half>();
        *reinterpret_cast<sycl::vec<sycl::half, 16>*>(d) = hv;
    });
}
// silu(g) * u of [ncols][n] -> x16 (the FFN's inner activation)
void w4h_swiglu_pack(const float* g, const float* u, uint16_t* x16, int n, int ncols, void* stream) {
    sycl::queue& q = *strata::q_of(stream);
    const size_t ncp = (size_t) (ncols + 7) / 8 * 8;
    q.parallel_for<class q36_w4h_swiglu_k>(sycl::range<2>((size_t) n / 16, (size_t) ncols), [=](sycl::id<2> i) {
        const size_t o = i[1] * (size_t) n + i[0] * 16;
        uint16_t* d = x16 + (i[0] * ncp + i[1]) * 16;
        sycl::vec<float, 16> gv = *reinterpret_cast<const sycl::vec<float, 16>*>(g + o);
        sycl::vec<float, 16> uv = *reinterpret_cast<const sycl::vec<float, 16>*>(u + o);
        sycl::vec<float, 16> v = gv / (1.0f + sycl::exp(-gv)) * uv;
        sycl::vec<sycl::half, 16> hv = v.convert<sycl::half>();
        *reinterpret_cast<sycl::vec<sycl::half, 16>*>(d) = hv;
    });
}

}   // namespace strata::q36xr
