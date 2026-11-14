// SPDX-License-Identifier: MIT
// W4xI8 prompt prefill GEMM (128-GRF ESIMD SplitPipe) and fused INT8/x128 quantization + RMSNorm kernels
// for Intel Arc Pro B70 (Xe2 / BMG-G31). Compiled in libq36_pp.so without -doubleGRF so all 64 HW
// threads/WG schedule cleanly inside one Xe2 core.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdint>
#include <cstdlib>
#include "strata/sycl_queue.hpp"
#include "q36_pp_i8.hpp"

namespace strata::q36pp {
namespace {

using namespace sycl::ext::intel::esimd;
namespace es  = sycl::ext::intel::esimd;
namespace esx = sycl::ext::intel::experimental::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

constexpr auto kAl16 = sycl::ext::oneapi::experimental::properties{
    sycl::ext::oneapi::experimental::alignment<16>};

constexpr auto kStreamW = sycl::ext::oneapi::experimental::properties{
    sycl::ext::oneapi::experimental::alignment<16>,
    es::cache_hint_L1<es::cache_hint::streaming>,
    es::cache_hint_L2<es::cache_hint::cached>};

namespace w4i8 {

template <bool SWIGLU> class TagQuantX128;

// Quantizes FP32 activations (or fused SwiGLU(gate, up)) into row-major xq8 [M, K] and padded-transposed
// group-128 scales dxT [K/128, Mp] where Mp = align_up(M, 128).
template <bool SWIGLU>
void launch_quant_x128(sycl::queue& q, const float* a, const float* b, void* xq8, float* dxT, int K, int M) {
    const int Mp       = (M + 127) & ~127;
    const int ng       = K / 128;
    const int n_row_wg = (Mp + 7) / 8;
    const int k4       = K / 4;
    auto* x4a = reinterpret_cast<const sycl::float4*>(a);
    auto* x4b = reinterpret_cast<const sycl::float4*>(b);
    auto* q4  = reinterpret_cast<uint32_t*>(xq8);
    q.parallel_for<TagQuantX128<SWIGLU>>(
        sycl::nd_range<2>(sycl::range<2>((size_t)ng, (size_t)n_row_wg * 256u), sycl::range<2>(1, 256)),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
            const int g    = (int)it.get_group(0);
            const auto sg  = it.get_sub_group();
            const int warp = (int)sg.get_group_linear_id();
            const int lane = (int)sg.get_local_linear_id();
            const int row  = (int)it.get_group(1) * 8 + warp;
            if (row >= M) {
                if (row < Mp && lane == 0) dxT[(size_t)g * Mp + row] = 0.0f;
                return;
            }
            const size_t idx4 = (size_t)row * k4 + (size_t)g * 32 + lane;
            sycl::float4 v = x4a[idx4];
            if constexpr (SWIGLU) {
                const sycl::float4 u = x4b[idx4];
                v.x() = (v.x() / (1.0f + sycl::native::exp(-v.x()))) * u.x();
                v.y() = (v.y() / (1.0f + sycl::native::exp(-v.y()))) * u.y();
                v.z() = (v.z() / (1.0f + sycl::native::exp(-v.z()))) * u.z();
                v.w() = (v.w() / (1.0f + sycl::native::exp(-v.w()))) * u.w();
            }
            float am = sycl::fmax(sycl::fmax(sycl::fabs(v.x()), sycl::fabs(v.y())),
                                  sycl::fmax(sycl::fabs(v.z()), sycl::fabs(v.w())));
            am = sycl::reduce_over_group(sg, am, sycl::maximum<float>());
            const float inv = (am > 0.0f) ? (127.0f / am) : 0.0f;
            const int8_t q0 = (int8_t)(int)sycl::clamp(sycl::rint(v.x() * inv), -127.0f, 127.0f);
            const int8_t q1 = (int8_t)(int)sycl::clamp(sycl::rint(v.y() * inv), -127.0f, 127.0f);
            const int8_t q2 = (int8_t)(int)sycl::clamp(sycl::rint(v.z() * inv), -127.0f, 127.0f);
            const int8_t q3 = (int8_t)(int)sycl::clamp(sycl::rint(v.w() * inv), -127.0f, 127.0f);
            q4[idx4] = (uint32_t)(uint8_t)q0 | ((uint32_t)(uint8_t)q1 << 8) |
                       ((uint32_t)(uint8_t)q2 << 16) | ((uint32_t)(uint8_t)q3 << 24);
            if (lane == 0) dxT[(size_t)g * Mp + row] = am / 127.0f;
        });
}

template <int N_FIXED, bool ADD, bool HAS_X128> class TagRmsFixed;
template <bool ADD, bool HAS_X128> class TagRmsGen;

template <int N_FIXED, bool ADD, bool HAS_X128>
void launch_rms_fixed(sycl::queue& q, float* x, const float* y, const float* w, float* h, void* xq8, float* dxT,
                      int T, float eps) {
    constexpr int PASSES = N_FIXED / 1024;
    const int Mp = (T + 127) & ~127;
    const int grid_rows = HAS_X128 ? Mp : T;
    auto* x4 = reinterpret_cast<sycl::float4*>(x);
    auto* y4 = reinterpret_cast<const sycl::float4*>(y);
    auto* w4 = reinterpret_cast<const sycl::float4*>(w);
    auto* h4 = reinterpret_cast<sycl::float4*>(h);
    auto* q4 = reinterpret_cast<uint32_t*>(xq8);
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> sh(sycl::range<1>(8), cgh);
        cgh.parallel_for<TagRmsFixed<N_FIXED, ADD, HAS_X128>>(
            sycl::nd_range<1>(sycl::range<1>((size_t)grid_rows * 256u), sycl::range<1>(256)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int row  = (int)it.get_group(0);
                const auto sg  = it.get_sub_group();
                const int warp = (int)sg.get_group_linear_id();
                const int lane = (int)sg.get_local_linear_id();
                if (row >= T) {
                    if constexpr (HAS_X128) {
                        if (lane == 0) {
#pragma unroll
                            for (int p = 0; p < PASSES; ++p) {
                                dxT[(size_t)(warp + p * 8) * Mp + row] = 0.0f;
                            }
                        }
                    }
                    return;
                }
                const size_t row4 = (size_t)row * (N_FIXED / 4);
                sycl::float4 xv[PASSES];
                float s = 0.0f;
#pragma unroll
                for (int p = 0; p < PASSES; ++p) {
                    const int g    = warp + p * 8;
                    const int col4 = g * 32 + lane;
                    sycl::float4 v = x4[row4 + col4];
                    if constexpr (ADD) {
                        const sycl::float4 b = y4[row4 + col4];
                        v = sycl::float4(v.x() + b.x(), v.y() + b.y(), v.z() + b.z(), v.w() + b.w());
                        x4[row4 + col4] = v;
                    }
                    xv[p] = v;
                    s += v.x() * v.x() + v.y() * v.y() + v.z() * v.z() + v.w() * v.w();
                }
                s = sycl::reduce_over_group(sg, s, sycl::plus<float>());
                if (lane == 0) sh[warp] = s;
                it.barrier(sycl::access::fence_space::local_space);
                float total = 0.0f;
#pragma unroll
                for (int u = 0; u < 8; ++u) total += sh[u];
                const float r = sycl::rsqrt(total / (float)N_FIXED + eps);
#pragma unroll
                for (int p = 0; p < PASSES; ++p) {
                    const int g    = warp + p * 8;
                    const int col4 = g * 32 + lane;
                    const sycl::float4 wv = w4[col4];
                    const sycl::float4 hv(xv[p].x() * r * wv.x(), xv[p].y() * r * wv.y(),
                                          xv[p].z() * r * wv.z(), xv[p].w() * r * wv.w());
                    h4[row4 + col4] = hv;
                    if constexpr (HAS_X128) {
                        float am = sycl::fmax(sycl::fmax(sycl::fabs(hv.x()), sycl::fabs(hv.y())),
                                              sycl::fmax(sycl::fabs(hv.z()), sycl::fabs(hv.w())));
                        am = sycl::reduce_over_group(sg, am, sycl::maximum<float>());
                        const float inv = (am > 0.0f) ? (127.0f / am) : 0.0f;
                        const int8_t q0 = (int8_t)(int)sycl::clamp(sycl::rint(hv.x() * inv), -127.0f, 127.0f);
                        const int8_t q1 = (int8_t)(int)sycl::clamp(sycl::rint(hv.y() * inv), -127.0f, 127.0f);
                        const int8_t q2 = (int8_t)(int)sycl::clamp(sycl::rint(hv.z() * inv), -127.0f, 127.0f);
                        const int8_t q3 = (int8_t)(int)sycl::clamp(sycl::rint(hv.w() * inv), -127.0f, 127.0f);
                        q4[row4 + col4] = (uint32_t)(uint8_t)q0 | ((uint32_t)(uint8_t)q1 << 8) |
                                          ((uint32_t)(uint8_t)q2 << 16) | ((uint32_t)(uint8_t)q3 << 24);
                        if (lane == 0) dxT[(size_t)g * Mp + row] = am / 127.0f;
                    }
                }
            });
    });
}

template <bool ADD, bool HAS_X128>
void launch_rms_gen(sycl::queue& q, float* x, const float* y, const float* w, float* h, void* xq8, float* dxT,
                    int n, int T, float eps) {
    const int Mp        = (T + 127) & ~127;
    const int grid_rows = HAS_X128 ? Mp : T;
    const int n_grp     = n >> 7;
    const int n4        = n >> 2;
    auto* x4 = reinterpret_cast<sycl::float4*>(x);
    auto* y4 = reinterpret_cast<const sycl::float4*>(y);
    auto* w4 = reinterpret_cast<const sycl::float4*>(w);
    auto* h4 = reinterpret_cast<sycl::float4*>(h);
    auto* q4 = reinterpret_cast<uint32_t*>(xq8);
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> sh(sycl::range<1>(8), cgh);
        cgh.parallel_for<TagRmsGen<ADD, HAS_X128>>(
            sycl::nd_range<1>(sycl::range<1>((size_t)grid_rows * 256u), sycl::range<1>(256)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int row  = (int)it.get_group(0);
                const auto sg  = it.get_sub_group();
                const int warp = (int)sg.get_group_linear_id();
                const int lane = (int)sg.get_local_linear_id();
                if (row >= T) {
                    if constexpr (HAS_X128) {
                        if (lane == 0) {
                            for (int g = warp; g < n_grp; g += 8) dxT[(size_t)g * Mp + row] = 0.0f;
                        }
                    }
                    return;
                }
                const size_t row4 = (size_t)row * n4;
                sycl::float4 xv[8];
                float s = 0.0f;
#pragma unroll
                for (int p = 0; p < 8; ++p) {
                    const int g = warp + p * 8;
                    if (g < n_grp) {
                        const int col4 = g * 32 + lane;
                        sycl::float4 v = x4[row4 + col4];
                        if constexpr (ADD) {
                            const sycl::float4 b = y4[row4 + col4];
                            v = sycl::float4(v.x() + b.x(), v.y() + b.y(), v.z() + b.z(), v.w() + b.w());
                            x4[row4 + col4] = v;
                        }
                        xv[p] = v;
                        s += v.x() * v.x() + v.y() * v.y() + v.z() * v.z() + v.w() * v.w();
                    }
                }
                s = sycl::reduce_over_group(sg, s, sycl::plus<float>());
                if (lane == 0) sh[warp] = s;
                it.barrier(sycl::access::fence_space::local_space);
                float total = 0.0f;
#pragma unroll
                for (int u = 0; u < 8; ++u) total += sh[u];
                const float r = sycl::rsqrt(total / (float)n + eps);
#pragma unroll
                for (int p = 0; p < 8; ++p) {
                    const int g = warp + p * 8;
                    if (g < n_grp) {
                        const int col4 = g * 32 + lane;
                        const sycl::float4 wv = w4[col4];
                        const sycl::float4 hv(xv[p].x() * r * wv.x(), xv[p].y() * r * wv.y(),
                                              xv[p].z() * r * wv.z(), xv[p].w() * r * wv.w());
                        h4[row4 + col4] = hv;
                        if constexpr (HAS_X128) {
                            float am = sycl::fmax(sycl::fmax(sycl::fabs(hv.x()), sycl::fabs(hv.y())),
                                                  sycl::fmax(sycl::fabs(hv.z()), sycl::fabs(hv.w())));
                            am = sycl::reduce_over_group(sg, am, sycl::maximum<float>());
                            const float inv = (am > 0.0f) ? (127.0f / am) : 0.0f;
                            const int8_t q0 = (int8_t)(int)sycl::clamp(sycl::rint(hv.x() * inv), -127.0f, 127.0f);
                            const int8_t q1 = (int8_t)(int)sycl::clamp(sycl::rint(hv.y() * inv), -127.0f, 127.0f);
                            const int8_t q2 = (int8_t)(int)sycl::clamp(sycl::rint(hv.z() * inv), -127.0f, 127.0f);
                            const int8_t q3 = (int8_t)(int)sycl::clamp(sycl::rint(hv.w() * inv), -127.0f, 127.0f);
                            q4[row4 + col4] = (uint32_t)(uint8_t)q0 | ((uint32_t)(uint8_t)q1 << 8) |
                                              ((uint32_t)(uint8_t)q2 << 16) | ((uint32_t)(uint8_t)q3 << 24);
                            if (lane == 0) dxT[(size_t)g * Mp + row] = am / 127.0f;
                        }
                    }
                }
            });
    });
}

// Stages one 128-K group into double-buffered SLM using all 64 HW threads for W4 unpack and tid < 32 for 2D X load.
SYCL_ESIMD_FUNCTION void load_tile_w4(
    uint32_t base, int step, int tid,
    int m0, int M, int K,
    const uint32_t* w_base, const int* xq8_i)
{
    const int un_n16 = tid >> 2, un_pr = (tid >> 1) & 1, un_hlf = tid & 1;
    const int x2_m8  = tid >> 1, x2_k64 = tid & 1;
    const int chk    = step >> 1, g = step & 1;
    const uint32_t* rec = w_base + (size_t)chk * 544u;
    simd<uint32_t, 16> meta = block_load<uint32_t, 16>(rec + 512 + 16 * g, kStreamW);
    simd<uint32_t, 64> w    = block_load<uint32_t, 64>(rec + 128 * (2 * g + un_pr) + 64 * un_hlf, kStreamW);
    if (tid < 32) {
        const int kx = (step * 128 + x2_k64 * 64) / 4;
        simd<int, 128> ax2 = esx::lsc_load_2d<int, 8, 8, 2, false, false, esx::cache_hint::cached, esx::cache_hint::cached>(
            xq8_i, K - 1, M - 1, K - 1, kx, m0 + x2_m8 * 8);
        slm_block_store<int, 64>(base + ((uint32_t)x2_m8 * 4u + (uint32_t)(x2_k64 * 2 + 0)) * 256u, ax2.template select<64, 1>(0), kAl16);
        slm_block_store<int, 64>(base + ((uint32_t)x2_m8 * 4u + (uint32_t)(x2_k64 * 2 + 1)) * 256u, ax2.template select<64, 1>(64), kAl16);
    }
    if ((tid & 3) == 0) {
        simd<uint16_t, 16> s16 = convert<uint16_t>(meta & 0xFFFFu);
        simd<sycl::half, 16> sh16 = s16.template bit_cast_view<sycl::half>();
        slm_block_store<float, 16>(base + 49152u + (uint32_t)un_n16 * 64u, simd<float, 16>(sh16), kAl16);
    }
    simd<uint32_t, 16> z4 = (meta >> 16) * 0x01010101u;
    simd<uint32_t, 64> lo, hi;
#pragma unroll
    for (int d = 0; d < 4; ++d) {
        simd<uint32_t, 16> x = w.template select<16, 1>(16 * d);
        lo.template select<16, 1>(16 * d) = (((x & 0x0F0F0F0Fu) | 0x80808080u) - z4) ^ 0x80808080u;
        hi.template select<16, 1>(16 * d) = ((((x >> 4) & 0x0F0F0F0Fu) | 0x80808080u) - z4) ^ 0x80808080u;
    }
    const uint32_t sb_lo = base + 16384u + ((uint32_t)un_n16 * 4u + (uint32_t)(2 * un_pr)) * 512u + 256u * (uint32_t)un_hlf;
    slm_block_store<uint32_t, 64>(sb_lo,        lo, kAl16);
    slm_block_store<uint32_t, 64>(sb_lo + 512u, hi, kAl16);
}

class TagW4I8;

void launch_w4i8(sycl::queue& q, const uint32_t* w4, const int8_t* xq8, const float* dxT,
                 float* Y, int M, int N, int K) {
    constexpr uint32_t kSlot = 50176u;
    constexpr int SwzM = 8;
    const int Mp     = (M + 127) & ~127;
    const int n_mb   = (M + 127) / 128;
    const int n_nb   = N / 256;
    const int* xq8_i = reinterpret_cast<const int*>(xq8);
    sycl::nd_range<2> nd(sycl::range<2>((size_t)n_mb * n_nb, 64), sycl::range<2>(1, 64));
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagW4I8>(nd, [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
            slm_init<kSlot * 2>();
            const int tid   = (int)it.get_local_id(1);
            const int g_idx = (int)it.get_group(0);
            const int panel = g_idx / (SwzM * n_nb);
            const int p_m   = ((panel + 1) * SwzM <= n_mb) ? SwzM : (n_mb - panel * SwzM);
            const int rem   = g_idx - panel * SwzM * n_nb;
            const int nb    = rem / p_m;
            const int mb    = panel * SwzM + (rem % p_m);
            const int m0    = mb * 128;
            const int n0    = nb * 256;
            const int m_th  = tid >> 3;
            const int n_th  = tid & 7;
            const int un_n16   = tid >> 2;
            const int n_grp    = K / 128;
            const int n_chunks = K / 256;
            const uint32_t* w_base = w4 + ((size_t)(n0 / 16 + un_n16) * n_chunks) * 544u;

            simd<float, 128> c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;
            load_tile_w4(0u, 0, tid, m0, M, K, w_base, xq8_i);
            const int row_b = m0 + m_th * 16;

            for (int gx = 0; gx < n_grp; ++gx) {
                barrier();
                const uint32_t base_cur = (gx & 1) ? kSlot : 0u;
                const int nx = gx + 1;
                simd<int, 128> a00 = 0, a01 = 0, a10 = 0, a11 = 0;
                // Keep k32 un-unrolled: unrolling k32 pushes live SLM loads over 128 GRFs and spills.
                for (int k32 = 0; k32 < 4; ++k32) {
                    simd<int, 64> A0 = slm_block_load<int, 64>(base_cur + ((m_th * 2 + 0) * 4 + k32) * 256u, kAl16);
                    simd<int, 64> A1 = slm_block_load<int, 64>(base_cur + ((m_th * 2 + 1) * 4 + k32) * 256u, kAl16);
                    const uint32_t sb0 = base_cur + 16384u + ((n_th * 2 + 0) * 4 + k32) * 512u;
                    const uint32_t sb1 = base_cur + 16384u + ((n_th * 2 + 1) * 4 + k32) * 512u;
                    simd<int, 128> B0, B1;
                    B0.template select<64, 1>(0)  = slm_block_load<int, 64>(sb0, kAl16);
                    B0.template select<64, 1>(64) = slm_block_load<int, 64>(sb0 + 256u, kAl16);
                    B1.template select<64, 1>(0)  = slm_block_load<int, 64>(sb1, kAl16);
                    B1.template select<64, 1>(64) = slm_block_load<int, 64>(sb1 + 256u, kAl16);
                    a00 = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(a00, B0, A0);
                    a10 = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(a10, B0, A1);
                    a01 = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(a01, B1, A0);
                    a11 = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(a11, B1, A1);
                }
                simd<float, 16> sc0 = slm_block_load<float, 16>(base_cur + 49152u + (uint32_t)(n_th * 2) * 64u, kAl16);
                simd<float, 16> sc1 = slm_block_load<float, 16>(base_cur + 49152u + (uint32_t)(n_th * 2 + 1) * 64u, kAl16);
                simd<float, 16> dxv = block_load<float, 16>(dxT + (size_t)gx * Mp + row_b, kAl16);
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const float d0 = float(dxv[r]), d1 = float(dxv[8 + r]);
                    c00.template select<16, 1>(16 * r) += simd<float, 16>(a00.template select<16, 1>(16 * r)) * (sc0 * d0);
                    c01.template select<16, 1>(16 * r) += simd<float, 16>(a01.template select<16, 1>(16 * r)) * (sc1 * d0);
                    c10.template select<16, 1>(16 * r) += simd<float, 16>(a10.template select<16, 1>(16 * r)) * (sc0 * d1);
                    c11.template select<16, 1>(16 * r) += simd<float, 16>(a11.template select<16, 1>(16 * r)) * (sc1 * d1);
                }
                if (nx < n_grp) {
                    load_tile_w4((nx & 1) ? kSlot : 0u, nx, tid, m0, M, K, w_base, xq8_i);
                }
            }

#pragma unroll
            for (int r = 0; r < 8; ++r) {
                if (row_b + r < M) {
                    simd<float, 32> v0;
                    v0.template select<16, 1>(0)  = c00.template select<16, 1>(16 * r);
                    v0.template select<16, 1>(16) = c01.template select<16, 1>(16 * r);
                    block_store<float, 32>(Y + (size_t)(row_b + r) * N + n0 + n_th * 32, v0);
                }
                if (row_b + 8 + r < M) {
                    simd<float, 32> v1;
                    v1.template select<16, 1>(0)  = c10.template select<16, 1>(16 * r);
                    v1.template select<16, 1>(16) = c11.template select<16, 1>(16 * r);
                    block_store<float, 32>(Y + (size_t)(row_b + 8 + r) * N + n0 + n_th * 32, v1);
                }
            }
        });
    });
}

class TagPackAb16;

}  // namespace w4i8

// ---------------------------------------------------------------------------
// PP2: 128-GRF ESIMD Coalesced Activation Quantizers + RMS5120 Producers
// ---------------------------------------------------------------------------
namespace pp2_prod {
namespace {

SYCL_ESIMD_FUNCTION inline float hmax2(float a, float b) { return a > b ? a : b; }

SYCL_ESIMD_FUNCTION inline float hmax8_f(simd<float, 8> v) {
    simd<float, 4> c = (es::max)(simd<float, 4>(v.template select<4, 1>(0)),
                                 simd<float, 4>(v.template select<4, 1>(4)));
    simd<float, 2> d = (es::max)(simd<float, 2>(c.template select<2, 1>(0)),
                                 simd<float, 2>(c.template select<2, 1>(2)));
    return hmax2(float(d[0]), float(d[1]));
}

SYCL_ESIMD_FUNCTION inline float hmax64_f(simd<float, 64> v) {
    simd<float, 32> a = (es::max)(simd<float, 32>(v.template select<32, 1>(0)),
                                  simd<float, 32>(v.template select<32, 1>(32)));
    simd<float, 16> b = (es::max)(simd<float, 16>(a.template select<16, 1>(0)),
                                  simd<float, 16>(a.template select<16, 1>(16)));
    simd<float, 8>  c = (es::max)(simd<float, 8>(b.template select<8, 1>(0)),
                                  simd<float, 8>(b.template select<8, 1>(8)));
    return hmax8_f(c);
}

SYCL_ESIMD_FUNCTION inline float hsum8_f(simd<float, 8> v) {
    simd<float, 4> c = v.template select<4, 1>(0) + v.template select<4, 1>(4);
    simd<float, 2> d = c.template select<2, 1>(0) + c.template select<2, 1>(2);
    return float(d[0]) + float(d[1]);
}

SYCL_ESIMD_FUNCTION inline float hsum64_f(simd<float, 64> v) {
    simd<float, 32> a = v.template select<32, 1>(0) + v.template select<32, 1>(32);
    simd<float, 16> b = a.template select<16, 1>(0) + a.template select<16, 1>(16);
    simd<float, 8>  c = b.template select<8, 1>(0)  + b.template select<8, 1>(8);
    return hsum8_f(c);
}

SYCL_ESIMD_FUNCTION inline int hsum32_i(simd<int, 32> v) {
    simd<int, 16> a = v.template select<16, 1>(0) + v.template select<16, 1>(16);
    simd<int, 8>  b = a.template select<8, 1>(0)  + a.template select<8, 1>(8);
    simd<int, 4>  c = b.template select<4, 1>(0)  + b.template select<4, 1>(4);
    simd<int, 2>  d = c.template select<2, 1>(0)  + c.template select<2, 1>(2);
    return int(d[0]) + int(d[1]);
}

SYCL_ESIMD_FUNCTION inline simd<int, 32> quant_rnde32(simd<float, 32> x, float inv_sc) {
    simd<float, 32> r = es::rnde<float, 32>(x * inv_sc);
    r = (es::min)((es::max)(r, simd<float, 32>(-127.0f)), simd<float, 32>(127.0f));
    return convert<int>(r);
}

SYCL_ESIMD_FUNCTION inline simd<float, 64> silu_mul_64(simd<float, 64> g, simd<float, 64> u) {
    simd<float, 64> out;
    simd<float, 32> g0 = g.template select<32, 1>(0);
    simd<float, 32> g1 = g.template select<32, 1>(32);
    simd<float, 32> s0 = g0 * (1.0f / (1.0f + es::exp<float, 32>(-g0)));
    simd<float, 32> s1 = g1 * (1.0f / (1.0f + es::exp<float, 32>(-g1)));
    out.template select<32, 1>(0)  = s0 * u.template select<32, 1>(0);
    out.template select<32, 1>(32) = s1 * u.template select<32, 1>(32);
    return out;
}

SYCL_ESIMD_FUNCTION inline void quant_pair_eo(simd<float, 64> v0, simd<float, 64> v1,
                                              simd<uint32_t, 16>& p0, simd<uint32_t, 16>& p1,
                                              float& dx, int& sumx) {
    const float am0  = hmax64_f(es::abs<float, 64>(v0));
    const float am1  = hmax64_f(es::abs<float, 64>(v1));
    const float amax = hmax2(am0, am1);
    dx = amax / 127.0f;
    const float inv  = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    simd<int, 32> qe0 = quant_rnde32(v0.template select<32, 1>(0),  inv);
    simd<int, 32> qo0 = quant_rnde32(v0.template select<32, 1>(32), inv);
    simd<int, 32> qe1 = quant_rnde32(v1.template select<32, 1>(0),  inv);
    simd<int, 32> qo1 = quant_rnde32(v1.template select<32, 1>(32), inv);
    sumx = hsum32_i(qe0) + hsum32_i(qo0) + hsum32_i(qe1) + hsum32_i(qo1);
    simd<int8_t, 64> eo0, eo1;
    eo0.template select<32, 2>(0) = convert<int8_t>(qe0);
    eo0.template select<32, 2>(1) = convert<int8_t>(qo0);
    eo1.template select<32, 2>(0) = convert<int8_t>(qe1);
    eo1.template select<32, 2>(1) = convert<int8_t>(qo1);
    p0 = eo0.template bit_cast_view<uint32_t>();
    p1 = eo1.template bit_cast_view<uint32_t>();
}

template <bool SWIGLU> class TagQuantX128Pp2;
template <bool ADD, bool HAS_X128> class TagRms5120Pp2;

template <bool SWIGLU>
void launch_quant_x128_pp2(sycl::queue& q, const float* g_ptr, const float* u_ptr,
                           void* xq8, float* dxT, int K, int M) {
    const uint32_t n_grp  = static_cast<uint32_t>(K) / 128u;
    const uint32_t n_ct32 = static_cast<uint32_t>(M + 31) / 32u;
    auto* xq8_u32 = static_cast<uint32_t*>(xq8);
    auto* dxT_u32 = reinterpret_cast<uint32_t*>(dxT);

    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagQuantX128Pp2<SWIGLU>>(
            sycl::nd_range<2>({static_cast<size_t>(n_ct32), static_cast<size_t>(n_grp) * 16u}, {1, 16}),
            [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
                slm_init<4352u>();
                const uint32_t ct32 = static_cast<uint32_t>(it.get_group(0));
                const uint32_t g    = static_cast<uint32_t>(it.get_group(1));
                const uint32_t lid  = static_cast<uint32_t>(it.get_local_id(1));  // 0..15

                const uint32_t ca   = lid;
                const uint32_t cb   = lid + 16u;
                const uint32_t cola = ct32 * 32u + ca;
                const uint32_t colb = ct32 * 32u + cb;
                const uint32_t k0   = g * 128u;
                const bool va = (cola < static_cast<uint32_t>(M));
                const bool vb = (colb < static_cast<uint32_t>(M));

                simd<float, 64> a0 = 0.0f, a1 = 0.0f, b0 = 0.0f, b1 = 0.0f;
                simd<float, 64> ua0 = 0.0f, ua1 = 0.0f, ub0 = 0.0f, ub1 = 0.0f;
                if (va) {
                    const size_t oa = static_cast<size_t>(cola) * static_cast<size_t>(K) + k0;
                    a0 = es::block_load<float, 64>(g_ptr + oa,       kAl16);
                    a1 = es::block_load<float, 64>(g_ptr + oa + 64u, kAl16);
                    if constexpr (SWIGLU) {
                        ua0 = es::block_load<float, 64>(u_ptr + oa,       kAl16);
                        ua1 = es::block_load<float, 64>(u_ptr + oa + 64u, kAl16);
                    }
                }
                if (vb) {
                    const size_t ob = static_cast<size_t>(colb) * static_cast<size_t>(K) + k0;
                    b0 = es::block_load<float, 64>(g_ptr + ob,       kAl16);
                    b1 = es::block_load<float, 64>(g_ptr + ob + 64u, kAl16);
                    if constexpr (SWIGLU) {
                        ub0 = es::block_load<float, 64>(u_ptr + ob,       kAl16);
                        ub1 = es::block_load<float, 64>(u_ptr + ob + 64u, kAl16);
                    }
                }

                simd<uint32_t, 16> pa0 = 0u, pa1 = 0u, pb0 = 0u, pb1 = 0u;
                float dxa = 0.0f, dxb = 0.0f;
                int sxa = 0, sxb = 0;

                if (va) {
                    if constexpr (SWIGLU) {
                        a0 = silu_mul_64(a0, ua0);
                        a1 = silu_mul_64(a1, ua1);
                    }
                    quant_pair_eo(a0, a1, pa0, pa1, dxa, sxa);
                }
                if (vb) {
                    if constexpr (SWIGLU) {
                        b0 = silu_mul_64(b0, ub0);
                        b1 = silu_mul_64(b1, ub1);
                    }
                    quant_pair_eo(b0, b1, pb0, pb1, dxb, sxb);
                }

                es::slm_block_store<uint32_t, 8>(0u    + ca * 32u, pa0.template select<8, 1>(0), kAl16);
                es::slm_block_store<uint32_t, 8>(1024u + ca * 32u, pa0.template select<8, 1>(8), kAl16);
                es::slm_block_store<uint32_t, 8>(2048u + ca * 32u, pa1.template select<8, 1>(0), kAl16);
                es::slm_block_store<uint32_t, 8>(3072u + ca * 32u, pa1.template select<8, 1>(8), kAl16);
                const uint32_t ja = ca >> 3, ra = ca & 7u;
                es::slm_scalar_store<float>(4096u + (16u * ja + ra) * 4u, dxa);
                es::slm_scalar_store<int>(4096u + (16u * ja + 8u + ra) * 4u, sxa);

                es::slm_block_store<uint32_t, 8>(0u    + cb * 32u, pb0.template select<8, 1>(0), kAl16);
                es::slm_block_store<uint32_t, 8>(1024u + cb * 32u, pb0.template select<8, 1>(8), kAl16);
                es::slm_block_store<uint32_t, 8>(2048u + cb * 32u, pb1.template select<8, 1>(0), kAl16);
                es::slm_block_store<uint32_t, 8>(3072u + cb * 32u, pb1.template select<8, 1>(8), kAl16);
                const uint32_t jb = cb >> 3, rb = cb & 7u;
                es::slm_scalar_store<float>(4096u + (16u * jb + rb) * 4u, dxb);
                es::slm_scalar_store<int>(4096u + (16u * jb + 8u + rb) * 4u, sxb);

                barrier();

                uint32_t* xq_blk = xq8_u32 + static_cast<size_t>(ct32 * n_grp + g) * 1024u;
                simd<uint32_t, 64> out = es::slm_block_load<uint32_t, 64>(lid * 256u, kAl16);
                es::block_store<uint32_t, 64>(xq_blk + lid * 64u, out, kAl16);

                if (lid == 0u) {
                    uint32_t* dx_blk = dxT_u32 + static_cast<size_t>(ct32 * n_grp + g) * 64u;
                    simd<uint32_t, 64> dout = es::slm_block_load<uint32_t, 64>(4096u, kAl16);
                    es::block_store<uint32_t, 64>(dx_blk, dout, kAl16);
                }
            });
    });
}

template <bool ADD, bool HAS_X128>
void launch_rms5120_pp2(sycl::queue& q, float* x, const float* y, const float* w,
                        float* h_out, void* xq8, float* dxT, int T, float eps) {
    const uint32_t n_ct32  = static_cast<uint32_t>(T + 31) / 32u;
    const uint32_t n_j     = HAS_X128 ? (4u * n_ct32) : static_cast<uint32_t>((T + 7) / 8);
    auto* xq8_u32 = static_cast<uint32_t*>(xq8);
    auto* dxT_u32 = reinterpret_cast<uint32_t*>(dxT);

    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagRms5120Pp2<ADD, HAS_X128>>(
            sycl::nd_range<1>(static_cast<size_t>(n_j) * 64u, 64u),
            [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
                constexpr uint32_t kSlm = HAS_X128 ? 8704u : 256u;
                slm_init<kSlm>();

                const uint32_t j_global  = static_cast<uint32_t>(it.get_group(0));
                const uint32_t lid       = static_cast<uint32_t>(it.get_local_id(0));
                const uint32_t r         = lid >> 3;      // 0..7
                const uint32_t k8        = lid & 7u;      // 0..7 (5 groups of 128 = 640 floats)
                const uint32_t ct32      = j_global >> 2;
                const uint32_t j_in_ct32 = j_global & 3u;
                const uint32_t col       = j_global * 8u + r;
                const bool valid_col     = (col < static_cast<uint32_t>(T));
                const uint32_t g0        = k8 * 5u;

                simd<float, 64> xlo[5], xhi[5];
                float ssq = 0.0f;

                if (valid_col) {
                    const size_t x_col_off = static_cast<size_t>(col) * 5120u;
                    #pragma unroll
                    for (int gi = 0; gi < 5; ++gi) {
                        const uint32_t k_off = (g0 + static_cast<uint32_t>(gi)) * 128u;
                        simd<float, 64> a0 = es::block_load<float, 64>(x + x_col_off + k_off,       kAl16);
                        simd<float, 64> a1 = es::block_load<float, 64>(x + x_col_off + k_off + 64u, kAl16);
                        if constexpr (ADD) {
                            simd<float, 64> y0 = es::block_load<float, 64>(y + x_col_off + k_off,       kAl16);
                            simd<float, 64> y1 = es::block_load<float, 64>(y + x_col_off + k_off + 64u, kAl16);
                            a0 += y0;
                            a1 += y1;
                            es::block_store<float, 64>(x + x_col_off + k_off,       a0, kAl16);
                            es::block_store<float, 64>(x + x_col_off + k_off + 64u, a1, kAl16);
                        }
                        xlo[gi] = a0;
                        xhi[gi] = a1;
                        ssq += hsum64_f(a0 * a0 + a1 * a1);
                    }
                }

                es::slm_scalar_store<float>((r * 8u + k8) * 4u, ssq);
                barrier();

                simd<float, 8> sq8 = es::slm_block_load<float, 8>(r * 32u, kAl16);
                const float mean_sq = hsum8_f(sq8) / 5120.0f + eps;
                const float inv_rms = 1.0f / float(es::sqrt_ieee<float, 1>(simd<float, 1>(mean_sq))[0]);

                if constexpr (HAS_X128) barrier();

                #pragma unroll
                for (int gi = 0; gi < 5; ++gi) {
                    const uint32_t k_off = (g0 + static_cast<uint32_t>(gi)) * 128u;
                    simd<float, 64> h0 = 0.0f, h1 = 0.0f;
                    if (valid_col) {
                        const size_t h_col_off = static_cast<size_t>(col) * 5120u;
                        simd<float, 64> w0 = es::block_load<float, 64>(w + k_off,       kAl16);
                        simd<float, 64> w1 = es::block_load<float, 64>(w + k_off + 64u, kAl16);
                        h0 = (xlo[gi] * inv_rms) * w0;
                        h1 = (xhi[gi] * inv_rms) * w1;
                        es::block_store<float, 64>(h_out + h_col_off + k_off,       h0, kAl16);
                        es::block_store<float, 64>(h_out + h_col_off + k_off + 64u, h1, kAl16);
                    }
                    if constexpr (HAS_X128) {
                        simd<uint32_t, 16> p0 = 0u, p1 = 0u;
                        float dx = 0.0f;
                        int sumx = 0;
                        if (valid_col) quant_pair_eo(h0, h1, p0, p1, dx, sumx);

                        const uint32_t slot_b = k8 * 1088u;
                        es::slm_block_store<uint32_t, 8>(slot_b + 0u   + r * 32u, p0.template select<8, 1>(0), kAl16);
                        es::slm_block_store<uint32_t, 8>(slot_b + 256u + r * 32u, p0.template select<8, 1>(8), kAl16);
                        es::slm_block_store<uint32_t, 8>(slot_b + 512u + r * 32u, p1.template select<8, 1>(0), kAl16);
                        es::slm_block_store<uint32_t, 8>(slot_b + 768u + r * 32u, p1.template select<8, 1>(8), kAl16);
                        es::slm_scalar_store<float>(slot_b + 1024u + r * 4u, dx);
                        es::slm_scalar_store<int>(slot_b + 1056u + r * 4u, sumx);
                        barrier();

                        if (lid < 32u) {
                            const uint32_t sl     = lid >> 2;
                            const uint32_t s4     = lid & 3u;
                            const uint32_t g_slot = sl * 5u + static_cast<uint32_t>(gi);
                            simd<uint32_t, 64> blk = es::slm_block_load<uint32_t, 64>(sl * 1088u + s4 * 256u, kAl16);
                            es::block_store<uint32_t, 64>(
                                xq8_u32 + (static_cast<size_t>(ct32 * 40u + g_slot) * 4u + s4) * 256u + j_in_ct32 * 64u,
                                blk, kAl16);
                        } else if (lid < 40u) {
                            const uint32_t sl     = lid - 32u;
                            const uint32_t g_slot = sl * 5u + static_cast<uint32_t>(gi);
                            simd<uint32_t, 16> dblk = es::slm_block_load<uint32_t, 16>(sl * 1088u + 1024u, kAl16);
                            es::block_store<uint32_t, 16>(
                                dxT_u32 + static_cast<size_t>(ct32 * 40u + g_slot) * 64u + j_in_ct32 * 16u,
                                dblk, kAl16);
                        }
                        if (gi < 4) barrier();
                    }
                }
            });
    });
}

}  // namespace
}  // namespace pp2_prod
}  // namespace

bool pp2_enabled() noexcept {
    static const bool on = []() {
        const char* e = std::getenv("STRATA_Q36_PP2");
        return e && std::atoi(e) != 0;
    }();
    return on;
}

bool pp2_fuse_enabled() noexcept {
    static const bool on = []() {
        if (!pp2_enabled() && !pp3_enabled()) return false;
        const char* e = std::getenv("STRATA_Q36_PP2_FUSE");
        return e ? (std::atoi(e) != 0) : true;
    }();
    return on;
}

size_t dxT_pp2_floats(int n_in, int T) noexcept {
    if (n_in <= 0 || T <= 0) return 0u;
    const size_t n_grp = static_cast<size_t>((n_in + 127) / 128);
    const size_t tp32  = static_cast<size_t>((T + 31) & ~31);
    return 2u * n_grp * tp32;
}

void quantize_x128_pp2(const float* x, void* xq8, float* dxT, int K, int M, void* stream) {
    if (!x || !xq8 || !dxT || K <= 0 || M <= 0 || (K % 128) != 0) return;
    pp2_prod::launch_quant_x128_pp2<false>(*strata::q_of(stream), x, nullptr, xq8, dxT, K, M);
}

void swiglu_x128_pp2(const float* g, const float* u, void* xq8, float* dxT, int K, int M, void* stream) {
    if (!g || !u || !xq8 || !dxT || K <= 0 || M <= 0 || (K % 128) != 0) return;
    pp2_prod::launch_quant_x128_pp2<true>(*strata::q_of(stream), g, u, xq8, dxT, K, M);
}

bool rms_pp2(float* x, const float* y, const float* w, float* h, void* xq8, float* dxT,
             int n, int T, float eps, void* stream) {
    if (!x || !w || !h || T <= 0 || n <= 0 || n > 8192 || (n % 128) != 0) return false;
    if (((uintptr_t)x & 15u) != 0u || ((uintptr_t)w & 15u) != 0u || ((uintptr_t)h & 15u) != 0u) return false;
    if (y && ((uintptr_t)y & 15u) != 0u) return false;
    const bool qx = (xq8 != nullptr && dxT != nullptr);
    if (qx && (((uintptr_t)xq8 & 15u) != 0u || ((uintptr_t)dxT & 15u) != 0u)) return false;
    sycl::queue& q = *strata::q_of(stream);
    if (n == 5120) {
        if (y) {
            if (qx) pp2_prod::launch_rms5120_pp2<true, true>(q, x, y, w, h, xq8, dxT, T, eps);
            else    pp2_prod::launch_rms5120_pp2<true, false>(q, x, y, w, h, nullptr, nullptr, T, eps);
        } else {
            if (qx) pp2_prod::launch_rms5120_pp2<false, true>(q, x, nullptr, w, h, xq8, dxT, T, eps);
            else    pp2_prod::launch_rms5120_pp2<false, false>(q, x, nullptr, w, h, nullptr, nullptr, T, eps);
        }
    } else {
        if (y) w4i8::launch_rms_gen<true, false>(q, x, y, w, h, nullptr, nullptr, n, T, eps);
        else   w4i8::launch_rms_gen<false, false>(q, x, nullptr, w, h, nullptr, nullptr, n, T, eps);
        if (qx) pp2_prod::launch_quant_x128_pp2<false>(q, h, nullptr, xq8, dxT, n, T);
    }
    return true;
}

bool handles_w4_i8(int n_in, int n_out, int T) noexcept {
    if (n_in <= 0 || n_out <= 0 || T <= 0) return false;
    if (pp2_enabled() || pp3_enabled()) return (n_in % 256) == 0 && (n_out % 16) == 0;
    return (n_in % 256) == 0 && (n_out % 256) == 0;
}

void quantize_x128(const float* x, void* xq8, float* dxT, int K, int M, void* stream) {
    if (pp2_enabled() || pp3_enabled()) {
        quantize_x128_pp2(x, xq8, dxT, K, M, stream);
        return;
    }
    w4i8::launch_quant_x128<false>(*strata::q_of(stream), x, nullptr, xq8, dxT, K, M);
}

void swiglu_x128(const float* g, const float* u, void* xq8, float* dxT, int K, int M, void* stream) {
    if (pp2_enabled() || pp3_enabled()) {
        swiglu_x128_pp2(g, u, xq8, dxT, K, M, stream);
        return;
    }
    w4i8::launch_quant_x128<true>(*strata::q_of(stream), g, u, xq8, dxT, K, M);
}

bool gemm_w4_i8(const void* weights, const void* xq8, const float* dxT, float* y, int n_in, int n_out, int T, void* stream) {
    if (!handles_w4_i8(n_in, n_out, T) || (n_out % 256) != 0 || !weights || !xq8 || !dxT || !y) return false;
    if ((reinterpret_cast<uintptr_t>(weights) & 15u) != 0u) return false;
    if ((reinterpret_cast<uintptr_t>(xq8) & 63u) != 0u) return false;
    if ((reinterpret_cast<uintptr_t>(dxT) & 15u) != 0u) return false;
    if ((reinterpret_cast<uintptr_t>(y) & 15u) != 0u) return false;
    w4i8::launch_w4i8(*strata::q_of(stream), static_cast<const uint32_t*>(weights),
                      static_cast<const int8_t*>(xq8), dxT, y, T, n_out, n_in);
    return true;
}

bool rms_pp(float* x, const float* y, const float* w, float* h, void* xq8, float* dxT,
            int n, int T, float eps, void* stream) {
    if (pp2_enabled() || pp3_enabled()) return rms_pp2(x, y, w, h, xq8, dxT, n, T, eps, stream);
    if (!x || !w || !h || T <= 0 || n <= 0 || n > 8192 || (n % 128) != 0) return false;
    if (((uintptr_t)x & 15u) != 0u || ((uintptr_t)w & 15u) != 0u || ((uintptr_t)h & 15u) != 0u) return false;
    if (y && ((uintptr_t)y & 15u) != 0u) return false;
    const bool qx = (xq8 != nullptr && dxT != nullptr);
    if (qx && (((uintptr_t)xq8 & 15u) != 0u || ((uintptr_t)dxT & 15u) != 0u)) return false;
    sycl::queue& q = *strata::q_of(stream);
    if (n == 5120) {
        if (y) {
            if (qx) w4i8::launch_rms_fixed<5120, true, true>(q, x, y, w, h, xq8, dxT, T, eps);
            else    w4i8::launch_rms_fixed<5120, true, false>(q, x, y, w, h, nullptr, nullptr, T, eps);
        } else {
            if (qx) w4i8::launch_rms_fixed<5120, false, true>(q, x, nullptr, w, h, xq8, dxT, T, eps);
            else    w4i8::launch_rms_fixed<5120, false, false>(q, x, nullptr, w, h, nullptr, nullptr, T, eps);
        }
    } else {
        if (y) {
            if (qx) w4i8::launch_rms_gen<true, true>(q, x, y, w, h, xq8, dxT, n, T, eps);
            else    w4i8::launch_rms_gen<true, false>(q, x, y, w, h, nullptr, nullptr, n, T, eps);
        } else {
            if (qx) w4i8::launch_rms_gen<false, true>(q, x, nullptr, w, h, xq8, dxT, n, T, eps);
            else    w4i8::launch_rms_gen<false, false>(q, x, nullptr, w, h, nullptr, nullptr, n, T, eps);
        }
    }
    return true;
}

void* pack_ab16(const float* alpha, const float* beta, int HV, int N, void* stream) {
    if (!alpha || !beta || HV <= 0 || (HV % 16) != 0 || HV > 64 || N <= 0 || (N % 32) != 0) return nullptr;
    sycl::queue& q = *strata::q_of(stream);
    const size_t n16       = (size_t)(HV / 16);
    const size_t k16_tot   = (size_t)(N / 16);
    const size_t total_u32 = 2u * n16 * k16_tot * 128u;
    auto* dst = sycl::malloc_device<uint32_t>(total_u32, q);
    if (!dst) return nullptr;
    const size_t branch_u32 = n16 * k16_tot * 128u;
    const size_t tile_u32   = k16_tot * 128u;
    q.parallel_for<w4i8::TagPackAb16>(sycl::range<1>(total_u32), [=](sycl::id<1> id) {
        const size_t idx    = id[0];
        const size_t branch = idx / branch_u32;
        const size_t rem    = idx - branch * branch_u32;
        const size_t t      = rem / tile_u32;
        const size_t rem2   = rem - t * tile_u32;
        const size_t k16    = rem2 >> 7;
        const size_t in_t   = rem2 & 127u;
        const size_t k2     = in_t >> 4;
        const size_t r      = in_t & 15u;
        const size_t row    = t * 16u + r;
        const size_t col    = k16 * 16u + k2 * 2u;
        const float* src    = (branch == 0 ? alpha : beta) + row * (size_t)N + col;
        const uint16_t b0   = sycl::bit_cast<uint16_t>(sycl::half(src[0]));
        const uint16_t b1   = sycl::bit_cast<uint16_t>(sycl::half(src[1]));
        dst[idx] = (uint32_t)b0 | ((uint32_t)b1 << 16);
    });
    return dst;
}

}  // namespace strata::q36pp
