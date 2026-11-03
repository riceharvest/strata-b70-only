#define DPCT_COMPAT_RT_VERSION 12080
// src/kernels/cuda/verify_kernels.cu - see
// include/strata/kernels/verify_kernels.hpp.
//
// The per-token arithmetic of every kernel here is transcribed from its
// single-token original (fused_gdn.cu, elementwise.cu) with the same operation
// order, so a verify window reproduces plain decode bit for bit.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "strata/core/geometry.hpp"
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/sycl_doorbell.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/resident_plan_mirror.hpp"
#include "strata/core/emulate.hpp"
// (#3948: STRATA_Q81_FASTMATH_DIV is the CUDA fast-math division; SYCL compiles -fp-model=precise, as native_mmvq does, so a / b)
#include "strata/kernels/q8_1_finite.hpp"   // #606: q8_1_ds
#include "strata/kernels/window_start.hpp"  // KVA: the draft layer's window floor
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/pdl.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <utility>
#include <vector>
#include <cmath>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#ifndef STRATA_PLAN_LOCAL
#define STRATA_PLAN_LOCAL 1   // 0: the original one-thread plan kernel (A/B)
#endif
namespace strata::kernels {
namespace {

constexpr int S = 128;          // GDN state size
static_assert(S == strata::core::kCompiledGeometry.ssm_state_size, "verify_kernels is compiled for core/geometry.hpp");
constexpr int RG = 4;
constexpr int RPG = S / RG;

void check(const char* what) {
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
}

// `commit` (one token): the history then keeps it - [hist1, hist2, x_0], each thread its channel's, after reading them
__dpct_inline__ void gdn_conv_l2_body(const int bx, const int by, float *hist,
                                      const float *__restrict__ qkv,
                                      const float *__restrict__ w,
                                      float *__restrict__ h, int C,
                                      int qk_heads, float eps, int t_begin,
                                      bool commit) {
    /*
    DPCT1115: The sycl::ext::oneapi::group_local_memory_for_overwrite is
    used to allocate group-local memory at the none kernel functor scope of a
    work-group data parallel kernel. You may need to adjust the code.
    */
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S / 32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = t_begin + by;
    const int c = bx * S + item_ct1.get_local_id(2);
    // the window of token t: [hist0, hist1, hist2, x_0, ..., x_t], its last four entries
    float win[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = t + j;          // index into [hist(3) | x...]
        win[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    const float v0 = win[0], v1 = win[1], v2 = win[2], x = qkv[(size_t) t * C + c];
    if (commit) {
        hist[c * 3] = v1;
        hist[c * 3 + 1] = v2;
        hist[c * 3 + 2] = x;
    }
    float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];
    float y = sum / (1.0f + sycl::native::exp(-sum));
    if ((int) bx < qk_heads) {
        float sq = y * y;
        /*
DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sq +=
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                sq, o);
        if ((item_ct1.get_local_id(2) & 31) == 0)
            part[item_ct1.get_local_id(2) >> 5] = sq;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= sycl::rsqrt(ss + eps);
    }
    h[(size_t) t * C + c] = y;
}

__dpct_inline__ void gdn_conv_l2_multi_kernel(float *hist,
                                              const float *__restrict__ qkv,
                                              const float *__restrict__ w,
                                              float *__restrict__ h, int C,
                                              int qk_heads, float eps,
                                              int t_begin, bool commit) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    gdn_conv_l2_body(item_ct1.get_group(2), item_ct1.get_group(1), hist, qkv, w,
                     h, C, qk_heads, eps, t_begin, commit);
}

__dpct_inline__ void
gdn_conv_commit_kernel(float *__restrict__ hist, const float *__restrict__ qkv,
                       int C, const int32_t *__restrict__ n_keep) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= C) return;
    const int n = *n_keep;
    if (n <= 0) return;
    float seq[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = n + j;          // the last three of [hist(3) | x_0..x_{n-1}]
        seq[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    hist[c * 3] = seq[0];
    hist[c * 3 + 1] = seq[1];
    hist[c * 3 + 2] = seq[2];
}

template <int MAX_T = kVerifyNarrowT, bool EXACT_T = false>
/*
DPCT1110: The total declared local variable size in device function
gdn_ab_body exceeds 128 bytes and may cause high register pressure. Consult with
your hardware vendor to find the total register size available and adjust the
code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
gdn_ab_body(const int bx, const float *__restrict__ x,
            const uint16_t *__restrict__ wa, const uint16_t *__restrict__ wb,
            const float *__restrict__ dt, const float *__restrict__ ssm_a,
            float *__restrict__ gate, float *__restrict__ beta, int n, int h_v,
            int T) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = bx * 2 + (item_ct1.get_local_id(2) >> 5),
              lane = item_ct1.get_local_id(2) & 31;
    if (row >= 2 * h_v) return;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const sycl::uint4 *w4 = reinterpret_cast<const sycl::uint4 *>(
        (is_beta ? wb : wa) + (size_t)r * n);
    float acc[MAX_T];
#pragma unroll
    for (int t = 0; t < MAX_T; ++t) acc[t] = 0.0f;
    for (int j = lane; j < n / 8; j += 32) {
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const sycl::uint4 wv = *(w4 + j);
        const float w0 = sycl::bit_cast<float>(wv.x() << 16),
                    w1 = sycl::bit_cast<float>(wv.x() & 0xffff0000u);
        const float w2 = sycl::bit_cast<float>(wv.y() << 16),
                    w3 = sycl::bit_cast<float>(wv.y() & 0xffff0000u);
        const float w4f = sycl::bit_cast<float>(wv.z() << 16),
                    w5 = sycl::bit_cast<float>(wv.z() & 0xffff0000u);
        const float w6 = sycl::bit_cast<float>(wv.w() << 16),
                    w7 = sycl::bit_cast<float>(wv.w() & 0xffff0000u);
#pragma unroll
        for (int t = 0; t < MAX_T; ++t) {
            if (!EXACT_T && t >= T) break;
            const float* xt = x + (size_t) t * n;
            /*
            DPCT1098: The '*' expression is used instead of the __ldg call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            const sycl::float4 xa = *reinterpret_cast<const sycl::float4 *>(xt + j * 8);
            /*
            DPCT1098: The '*' expression is used instead of the __ldg call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            const sycl::float4 xb = *reinterpret_cast<const sycl::float4 *>(xt + j * 8 + 4);
            float a = acc[t];
            a = sycl::fma((float)w0, (float)(xa.x()), a);
                a = sycl::fma((float)w1, (float)(xa.y()), a);
            a = sycl::fma((float)w2, (float)(xa.z()), a);
                a = sycl::fma((float)w3, (float)(xa.w()), a);
            a = sycl::fma((float)w4f, (float)(xb.x()), a);
                a = sycl::fma((float)w5, (float)(xb.y()), a);
            a = sycl::fma((float)w6, (float)(xb.z()), a);
                a = sycl::fma((float)w7, (float)(xb.w()), a);
            acc[t] = a;
        }
    }
#pragma unroll
    for (int t = 0; t < MAX_T; ++t) {
        if (!EXACT_T && t >= T) break;
        float a = acc[t];
#pragma unroll
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        for (int o = 16; o > 0; o >>= 1) a +=
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                a, o);
        if (lane != 0) continue;
        if (is_beta) {
            beta[(size_t)t * h_v + r] = 1.0f / (1.0f + sycl::native::exp(-a));
        } else {
            const float v = a + dt[r];
            const float sp = v > 20.0f ? v : sycl::log1p(sycl::native::exp(v));
            gate[(size_t) t * h_v + r] = sp * ssm_a[r];
        }
    }
}

template <int MAX_T = kVerifyNarrowT, bool EXACT_T = false>
__dpct_inline__ void gdn_ab_multi_kernel(
    const float *__restrict__ x, const uint16_t *__restrict__ wa,
    const uint16_t *__restrict__ wb, const float *__restrict__ dt,
    const float *__restrict__ ssm_a, float *__restrict__ gate,
    float *__restrict__ beta, int n, int h_v, int T) {
    gdn_ab_body<MAX_T, EXACT_T>(
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(2), x, wa,
        wb, dt, ssm_a, gate, beta, n, h_v, T);
}

// 143-amdk (STRATA_FUSE_CONVAB): the conv + L2 norm and the a/b projections in ONE launch.  They read different inputs (the qkv
// projection and the layer input) and write different outputs, so they only share a node: blocks [0, conv) run the conv body,
// the following blocks (token row 0 only) the a/b body, each exactly as its own kernel.
template <int MAX_T, bool EXACT_T>
__dpct_inline__ void gdn_conv_ab_kernel(
    int conv_blocks, float *hist, const float *__restrict__ qkv,
    const float *__restrict__ w, float *__restrict__ h, int C, int qk_heads,
    float eps, int t_begin, bool commit, const float *__restrict__ x,
    const uint16_t *__restrict__ wa, const uint16_t *__restrict__ wb,
    const float *__restrict__ dt, const float *__restrict__ ssm_a,
    float *__restrict__ gate, float *__restrict__ beta, int n, int h_v, int T) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int ab_blocks =
        (2 * h_v + 1) / 2; // the a/b blocks first: their dot chain is the
                           // longer one, so it starts with the first wave
    if ((int)item_ct1.get_group(2) >= ab_blocks) {
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        gdn_conv_l2_body((int)item_ct1.get_group(2) - ab_blocks,
                         (int)item_ct1.get_group(1), hist, qkv, w, h, C,
                         qk_heads, eps, t_begin, commit);
    } else if (item_ct1.get_group(1) == 0 && item_ct1.get_local_id(2) < 64) {
        gdn_ab_body<MAX_T, EXACT_T>((int)item_ct1.get_group(2), x, wa, wb, dt,
                                    ssm_a, gate, beta, n, h_v, T);
    }
}

// State-only commit kernel: each head's 128 independent state columns are split across 4 blocks of 32 columns
// (48 * 4 = 192 blocks of 128 threads across all SMs, vs 48 blocks of 512 threads), with no sq/o/norm/z/y work.
__dpct_inline__ void gdn_step_commit_body(float *__restrict__ state,
                                          const float *__restrict__ hbuf, int C,
                                          const float *__restrict__ gate,
                                          const float *__restrict__ beta,
                                          int h_k, int h_v, int n);

/*
DPCT1110: The total declared local variable size in device function
gdn_step_commit_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gdn_step_commit_kernel(
    float *__restrict__ state, const float *__restrict__ hbuf, int C,
    const float *__restrict__ gate, const float *__restrict__ beta, int h_k,
    int h_v, const int32_t *__restrict__ n_keep) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int n = *n_keep;
    if (n <= 0) return;
    gdn_step_commit_body(state, hbuf, C, gate, beta, h_k, h_v, n);
}

// engine-latency: gdn_step_commit_kernel for every GDN layer of the commit in one launch (blockIdx.z = the layer); each
// layer's blocks run gdn_step_commit_body on that layer's state and rows, exactly as its own launch did
__dpct_inline__ void gdn_step_commit_layers_kernel(
    float *__restrict__ state, long long state_stride,
    const float *__restrict__ hbuf, long long h_stride, int C,
    const float *__restrict__ gate, const float *__restrict__ beta,
    long long gb_stride, int h_k, int h_v, const int32_t *__restrict__ n_keep) {
    const int n = *n_keep;
    if (n <= 0) return;
    const long long z =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(0);
    gdn_step_commit_body(state + z * state_stride, hbuf + z * h_stride, C, gate + z * gb_stride, beta + z * gb_stride, h_k,
                         h_v, n);
}

// engine-latency: gdn_conv_commit_kernel for every GDN layer in one launch (blockIdx.y = the layer)
__dpct_inline__ void
gdn_conv_commit_layers_kernel(float *__restrict__ hist0, long long hist_stride,
                              const float *__restrict__ qkv0,
                              long long qkv_stride, int C,
                              const int32_t *__restrict__ n_keep) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= C) return;
    const int n = *n_keep;
    if (n <= 0) return;
    float *hist = hist0 + (long long)item_ct1.get_group(1) * hist_stride;
    const float *qkv = qkv0 + (long long)item_ct1.get_group(1) * qkv_stride;
    float seq[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = n + j;          // the last three of [hist(3) | x_0..x_{n-1}]
        seq[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    hist[c * 3] = seq[0];
    hist[c * 3 + 1] = seq[1];
    hist[c * 3 + 2] = seq[2];
}

/*
DPCT1110: The total declared local variable size in device function
gdn_step_commit_body exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gdn_step_commit_body(float *__restrict__ state,
                                          const float *__restrict__ hbuf, int C,
                                          const float *__restrict__ gate,
                                          const float *__restrict__ beta,
                                          int h_k, int h_v, int n) {
    /*
    DPCT1115: The sycl::ext::oneapi::group_local_memory_for_overwrite is
    used to allocate group-local memory at the none kernel functor scope of a
    work-group data parallel kernel. You may need to adjust the code.
    */
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sk = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    /*
    DPCT1115: The sycl::ext::oneapi::group_local_memory_for_overwrite is
    used to allocate group-local memory at the none kernel functor scope of a
    work-group data parallel kernel. You may need to adjust the code.
    */
    auto &red =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[RG][32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int head = item_ct1.get_group(2);
    const int col_local = item_ct1.get_local_id(2);
    const int col = item_ct1.get_group(1) * 32 + col_local;
    const int rg = item_ct1.get_local_id(1);
    const int tid = rg * 32 + col_local;    // 0 .. 127
    const int qh = head % h_k;
    const int qk = S * h_k;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        sk[tid] = ht[qk + qh * S + tid];
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const float g = sycl::native::exp(gate[(size_t)t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
        red[rg][col_local] = kv;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const float kv_col = red[0][col_local] + red[1][col_local] + red[2][col_local] + red[3][col_local];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            s[r] = sycl::fma((float)g, s[r], sk[rg * RPG + r] * delta);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
}

// S26 (STRATA_QFUSE=1): the q8_1 image of a 32-column block, written by the warp that holds those columns with
// native_quantize_q8_1_kernel's quantizer (the same XOR-tree max and sum, d = amax / 127, roundf(x / d), ds = (d, sum)).
struct GdnQ81 { sycl::half2 ds; int8_t qs[32]; };
__dpct_inline__ void gdn_q8_1_store(GdnQ81 *__restrict__ xq, size_t idx,
                                    float xi) {
    // no contraction: xi is a product here, and fma(a, b, shfl) would round the first sum unlike the separate
    // quantizer, which reads xi from memory (S26 harness: ds.sum differed by 1 ulp without this)
#pragma clang fp contract(off)
    float amax = sycl::fabs(xi), sum = xi;
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) amax = sycl::fmax(
        amax, dpct::experimental::permute_sub_group_by_xor(
                  0xffffffffu,
                  sycl::ext::oneapi::this_work_item::get_sub_group(), amax, o));
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) sum +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            sum, o);
    const float d = q8_1_finite(q81_div(amax, 127.0f));   // #606: as native_quantize_q8_1_kernel - finite blocks bit for bit (#3948: and its division)
    const int8_t q = q8_1_quant(xi, d, amax);
    xq[idx / 32].qs[idx % 32] = q;
    if (idx % 32 == 0) xq[idx / 32].ds = q8_1_ds(d, sum);   // #606: clamped (the QFUSE path bypassed the finite helper)
}

template <bool ALL_OUT, bool Q, bool QK_REG = false>
/*
DPCT1110: The total declared local variable size in device function
gdn_step_norm_multi_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
__dpct_inline__ void gdn_step_norm_multi_kernel(
    float *__restrict__ state, const float *__restrict__ hbuf, int C,
    const float *__restrict__ gate, const float *__restrict__ beta,
    const float *__restrict__ z, const float *__restrict__ gamma, float eps,
    float *__restrict__ y, int h_k, int h_v, int T,
    const int32_t *__restrict__ n_keep, int t_out_begin,
    GdnQ81 *__restrict__ xq, GdnGroups grp = {}) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (grp.n > 0) { // group blockIdx.y - its slot's state and its rows of the
                     // window's buffers
        const int gi = item_ct1.get_group(1), f = grp.first[gi];
        state = grp.state[gi];
        hbuf += (size_t) f * C;
        gate += (size_t) f * h_v;
        beta += (size_t) f * h_v;
        z += (size_t) f * S * h_v;
        y += (size_t) f * S * h_v;
        T = grp.ntok[gi];
        n_keep = grp.keep[gi];
    }
    auto &sk =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[2][S]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sq =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[2][S]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red_kv =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[RG][S]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red_o =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[RG][S]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &wsum =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S / 32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int head = item_ct1.get_group(2);
    const int col = item_ct1.get_local_id(2);
    const int rg = item_ct1.get_local_id(1);
    const int qh = head % h_k;
    const int qk = S * h_k;             // q at [0, qk), k at [qk, 2qk), v at [2qk, ...)
    const int value_dim = S * h_v;
    const int n = ALL_OUT ? T : (n_keep ? *n_keep : T);
    if (!QK_REG && n > 0) {
        if (rg == 0) sk[0][col] = hbuf[qk + qh * S + col];
        else if (rg == 1 && (ALL_OUT || 0 >= t_out_begin)) sq[0][col] = hbuf[qh * S + col];
    }
    const float gam = (rg == 0) ? gamma[col] : 0.0f;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    for (int r = 0; r < RPG; ++r) s[r] = ALL_OUT ? base[r * row_stride]
                                                 : base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const int cur = t & 1, nxt = (t + 1) & 1;
        const float* ht = hbuf + (size_t) t * C;
        const bool need_out = ALL_OUT || (t >= t_out_begin);
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        float k_own = 0.0f, q_own = 0.0f;
        auto sg = item_ct1.get_sub_group();
        if constexpr (QK_REG) {
            const int i = rg * RPG + sg.get_local_linear_id();
            k_own = ht[qk + qh * S + i];
            if (need_out) q_own = ht[qh * S + i];
        }
        if (!QK_REG && t + 1 < n) {
            const float* ht_next = ht + C;
            const bool need_next = ALL_OUT || (t + 1 >= t_out_begin);
            if (rg == 0) sk[nxt][col] = ht_next[qk + qh * S + col];
            else if (rg == 1 && need_next) sq[nxt][col] = ht_next[qh * S + col];
        }
        const float g = sycl::native::exp(gate[(size_t)t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            kv = sycl::fma(s[r], QK_REG ? sycl::select_from_group(sg, k_own, r) : sk[cur][rg * RPG + r], kv);
        red_kv[rg][col] = kv;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        if (!need_out) {
#pragma unroll
            for (int r = 0; r < RPG; ++r)
                s[r] = sycl::fma((float)g, s[r],
                                (QK_REG ? sycl::select_from_group(sg, k_own, r) : sk[cur][rg * RPG + r]) * delta);
            continue;
        }
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = sycl::fma((float)g, s[r],
                            (QK_REG ? sycl::select_from_group(sg, k_own, r) : sk[cur][rg * RPG + r]) * delta);
            o = sycl::fma(s[r], QK_REG ? sycl::select_from_group(sg, q_own, r) : sq[cur][rg * RPG + r], o);
        }
        red_o[rg][col] = o;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] +
                  red_o[3][col]) *
                 sycl::rsqrt((float)S);
            sq_part = oc * oc;
        }
        // every thread takes the shuffles, as the single-token kernel does: inside the `rg == 0` branch hipcc on
        // gfx1151 gave another rounding for 3% of the outputs (gdn_parity section 5 on Aurora)
        /*
DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part +=
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                sq_part, o2);
        if (rg == 0 && (col & 31) == 0) wsum[col >> 5] = sq_part;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = sycl::rsqrt(ss / (float)S + eps);
            const float zz = z[(size_t) t * value_dim + head * S + col];
            const float yv =
                oc * scale * gam * (1.0f / (1.0f + sycl::native::exp(-zz)));
            y[(size_t) t * value_dim + head * S + col] = yv;
            if constexpr (Q) gdn_q8_1_store(xq, (size_t) (t - t_out_begin) * value_dim + head * S + col, yv);
        }
    }
    if (!ALL_OUT && n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}

// S25 (STRATA_GDN_SPLIT=1): the same recurrence with each head's 128 state columns over 4 blocks of 32 columns
// (128 threads: the same (column, row group) threads, each with the same 32 state rows), so 4x the blocks of
// gdn_step_norm_multi_kernel - which ran one 512-thread block per head (32 blocks) and read the 2 MB state of a layer
// at ~60 GB/s. Every column's arithmetic is the original's; the output norm, which needs all 128 columns of a head,
// moves to gdn_out_norm_kernel: this kernel leaves the unnormalized output `oc` in y and the second kernel sums the
// squares by the same warps (columns 32w..32w+31, the same butterfly) in the same order.
// S26: OUT = false (the commit, whose outputs nobody reads) drops the output reduction and its two barriers; the
// norm kernel below has the old kernel's code shape (S x RG threads, rg 0 holding oc), which makes it bitwise equal -
// the 128-thread version differed from the old kernel by 1 ulp in ~7% of the outputs.
constexpr int GS_COLS = 32;
template <bool OUT>
/*
DPCT1110: The total declared local variable size in device function
gdn_step_split_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
gdn_step_split_kernel(float *__restrict__ state, const float *__restrict__ hbuf,
                      int C, const float *__restrict__ gate,
                      const float *__restrict__ beta, float *__restrict__ y,
                      int h_k, int h_v, int T,
                      const int32_t *__restrict__ n_keep, int t_out_begin) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sk = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sq = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[RG][GS_COLS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int head = item_ct1.get_group(2);
    const int c0 = item_ct1.get_group(1) * GS_COLS;
    const int lc = item_ct1.get_local_id(2); // local column
    const int col = c0 + lc;
    const int rg = item_ct1.get_local_id(1);
    const int tid = rg * GS_COLS + lc;
    const int qh = head % h_k;
    const int qk = S * h_k;
    const int value_dim = S * h_v;
    const int n = n_keep ? *n_keep : T;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (tid < S) { sk[tid] = ht[qk + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const float g = sycl::native::exp(gate[(size_t)t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
        red[rg][lc] = kv;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const float kv_col = red[0][lc] + red[1][lc] + red[2][lc] + red[3][lc];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = sycl::fma((float)g, s[r], sk[rg * RPG + r] * delta);
            o = sycl::fma(s[r], sq[rg * RPG + r], o);
        }
        if (!OUT) continue;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        red[rg][lc] = o;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (rg == 0 && t >= t_out_begin)
            y[(size_t)t * value_dim + head * S + col] =
                (red[0][lc] + red[1][lc] + red[2][lc] + red[3][lc]) *
                sycl::rsqrt((float)S);
    }
    if (n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}


// S26: the state-only replay (the commit) with its per-token inputs prefetched - gate, beta and the v column of every
// row up front, the next row's k / q while the current one computes. The arithmetic and its order are
// gdn_step_split_kernel<false>'s (gfx1151 harness: state bitwise equal; 44.5 -> 39.1 us at 1 row, 47.4 -> 40.5 at 2,
// 53.8 -> 42.9 at 4, h_v 48). With outputs the prefetch lost (register pressure), so the window keeps the plain split.
template <int MAXT>
/*
DPCT1110: The total declared local variable size in device function
gdn_state_replay_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gdn_state_replay_kernel(
    float *__restrict__ state, const float *__restrict__ hbuf, int C,
    const float *__restrict__ gate, const float *__restrict__ beta, int h_k,
    int h_v, int T, const int32_t *__restrict__ n_keep) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sk = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sq = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[RG][GS_COLS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int head = item_ct1.get_group(2);
    const int c0 = item_ct1.get_group(1) * GS_COLS;
    const int lc = item_ct1.get_local_id(2);
    const int col = c0 + lc;
    const int rg = item_ct1.get_local_id(1);
    const int tid = rg * GS_COLS + lc;
    const int qh = head % h_k;
    const int qk = S * h_k;
    const int n = n_keep ? *n_keep : T;
    float gv[MAXT], bv[MAXT], vv[MAXT];
#pragma unroll
    for (int t = 0; t < MAXT; ++t) {
        if (t < n) {
            gv[t] = gate[(size_t) t * h_v + head];
            bv[t] = beta[(size_t) t * h_v + head];
            vv[t] = hbuf[(size_t) t * C + 2 * qk + head * S + col];
        }
    }
    float kn = 0.0f, qn = 0.0f;
    if (n > 0) { kn = hbuf[qk + qh * S + tid]; qn = hbuf[qh * S + tid]; }
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
#pragma unroll
    for (int t = 0; t < MAXT; ++t) {
        if (t >= n) break;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        sk[tid] = kn; sq[tid] = qn;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (t + 1 < n) {
            const float* hn = hbuf + (size_t) (t + 1) * C;
            kn = hn[qk + qh * S + tid]; qn = hn[qh * S + tid];
        }
        const float g = sycl::native::exp(gv[t]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
        red[rg][lc] = kv;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const float kv_col = red[0][lc] + red[1][lc] + red[2][lc] + red[3][lc];
        const float delta = (vv[t] - g * kv_col) * bv[t];
        float o = 0.0f;   // unused: kept because the measured code had it (without it: 51 us instead of 39 at 1 row)
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = sycl::fma((float)g, s[r], sk[rg * RPG + r] * delta);
            o = sycl::fma(s[r], sq[rg * RPG + r], o);
        }
        (void) o;
    }
    if (n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}
// y = oc * rsqrt(mean(oc^2) + eps) * gamma * sigmoid(z) per head and token: gdn_step_norm_multi_kernel's tail with its
// code shape (S x RG threads; rg 0 holds oc, the other row groups contribute 0 to wsum), reading oc from y
template <bool Q>
__dpct_inline__ void gdn_out_norm_kernel(
    const float *__restrict__ z, const float *__restrict__ gamma, float eps,
    float *__restrict__ y, int h_v, int T, const int32_t *__restrict__ n_keep,
    int t_out_begin, GdnQ81 *__restrict__ xq) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &wsum =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S * RG / 32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int head = item_ct1.get_group(2),
              t = item_ct1.get_group(1) + t_out_begin,
              col = item_ct1.get_local_id(2), rg = item_ct1.get_local_id(1);
    const int tid = rg * S + col;
    const int n = n_keep ? *n_keep : T;
    if (t >= n) return;
    const int value_dim = S * h_v;
    float oc = 0.0f, sq_part = 0.0f;
    if (rg == 0) {
        oc = y[(size_t) t * value_dim + head * S + col];
        sq_part = oc * oc;
    }
    /*
DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            sq_part, o2);
    if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (rg == 0) {
        const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
        const float scale = sycl::rsqrt(ss / (float)S + eps);
        const float zz = z[(size_t) t * value_dim + head * S + col];
        if constexpr (Q) {
            const float yv = oc * scale * gamma[col] *
                             (1.0f / (1.0f + sycl::native::exp(-zz)));
            y[(size_t) t * value_dim + head * S + col] = yv;
            gdn_q8_1_store(xq, (size_t) (t - t_out_begin) * value_dim + head * S + col, yv);
        } else {
            y[(size_t)t * value_dim + head * S + col] =
                oc * scale * gamma[col] *
                (1.0f / (1.0f + sycl::native::exp(-zz)));
        }
    }
}

__dpct_inline__ void embedding_gather_dev_kernel(
    const uint8_t *__restrict__ codes, const float *__restrict__ scales,
    const float *__restrict__ offsets, const int32_t *__restrict__ tokens,
    int64_t n, int code_bits, int code_bias, int group_elems,
    unsigned long long row_codes, unsigned long long row_groups,
    float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n) return;
    const unsigned long long token = (unsigned long long) tokens[t];
    const uint8_t* c = codes + token * row_codes;
    const float* sc = scales + token * row_groups;
    const float* of = offsets ? offsets + token * row_groups : nullptr;
    const int per_byte = 8 / code_bits;
    const unsigned mask = (1u << code_bits) - 1u;
    const int code = (c[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
    const int64_t group = i / group_elems;
    /*
    DPCT1013: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    const float product = (float)(code + code_bias) * sc[group];
    /*
    DPCT1013: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    out[(size_t)t * n + i] = product + (of ? of[group] : 0.0f);
}

__dpct_inline__ void broadcast_streams_kernel(const float *__restrict__ x,
                                              float *__restrict__ R, int64_t n,
                                              int hc) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = x[(size_t) t * n + i % n];
}

__dpct_inline__ void hc_mean_streams_kernel(const float *__restrict__ R,
                                            float *__restrict__ out, int64_t n,
                                            int hc) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n) return;
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < hc; ++c) s += R[(size_t)c * n + i];
    out[i] = s / (float) hc;
}

__dpct_inline__ void
route_marginal_stats_kernel(const int32_t *__restrict__ ids, int n, int k,
                            unsigned long long *acc) {
    const int r =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (r >= n) return;
    int fresh = 0;
    for (int j = 0; j < k; ++j) {
        const int32_t e = ids[r * k + j];
        bool seen = false;
#pragma unroll
        for (int j2 = 0; j2 < j && !seen; ++j2) seen = ids[r * k + j2] == e;
#pragma unroll
        for (int r2 = 0; r2 < r && !seen; ++r2)
#pragma unroll
            for (int j2 = 0; j2 < k && !seen; ++j2)
                seen = ids[r2 * k + j2] == e;
        if (!seen) ++fresh;
    }
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        acc + r, (unsigned long long)fresh);
}

__dpct_inline__ void copy_indexed_kernel(float *__restrict__ dst,
                                         const float *__restrict__ src,
                                         int64_t stride,
                                         const int32_t *__restrict__ index,
                                         int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int idx = *index;
    if (idx < 0) return;
#pragma unroll
    for (int64_t i =
             (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < n; i += (int64_t)item_ct1.get_group_range(2) *
                     item_ct1.get_local_range(2))
        dst[i] = src[(size_t) idx * stride + i];
}

__dpct_inline__ void
fetch_blobs_kernel(const unsigned long long *__restrict__ src,
                   const int32_t *__restrict__ n, sycl::uint4 *__restrict__ dst,
                   long long per) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long total = (long long)*n * per;
    for (long long i =
             (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < total; i += (long long)item_ct1.get_group_range(2) *
                         item_ct1.get_local_range(2)) {
        const long long k = i / per, off = i - k * per;
        dst[i] = ((const sycl::uint4 *)src[k])[off];
    }
}

__dpct_inline__ void res_patch_kernel(int32_t *__restrict__ table,
                                      const volatile int32_t *pairs, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
#pragma unroll
    for (int k = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
         k < n; k += item_ct1.get_group_range(2) * item_ct1.get_local_range(2))
        table[pairs[2 * k]] = pairs[2 * k + 1];
}

__dpct_inline__ void rebase_ptrs_kernel(unsigned long long *ptr,
                                        const int32_t *n,
                                        unsigned long long base,
                                        long long bytes) {
    const int k =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (k < *n) ptr[k] = base + (unsigned long long) k * (unsigned long long) bytes;
}

__dpct_inline__ void add_streams_broadcast_kernel(const float *__restrict__ h,
                                                  const float *__restrict__ e,
                                                  float *__restrict__ R,
                                                  int64_t n, int hc) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = h[(size_t) t * n * hc + i] + e[(size_t) t * n + i % n];
}

__dpct_inline__ void ident_hits_kernel(const int32_t *__restrict__ ids, int n,
                                       int32_t *__restrict__ slot,
                                       int32_t *__restrict__ dst,
                                       int32_t *__restrict__ count) {
    const int i =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (i < n) { slot[i] = ids[i]; dst[i] = i; }
    if (i == 0) *count = n;
}

// E = the widest element the row size divides into (16, 4 or 1 bytes): a Q6_K head row of 2560 values is 2100 bytes
template <typename E>
__dpct_inline__ void gather_rows_kernel(const E *__restrict__ src,
                                        long long row_e,
                                        const int32_t *__restrict__ ids,
                                        long long n, E *__restrict__ dst) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long total = n * row_e;
    for (long long i =
             (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < total; i += (long long)item_ct1.get_group_range(2) *
                         item_ct1.get_local_range(2)) {
        const long long r = i / row_e, o = i - r * row_e;
        dst[i] = src[(long long) ids[r] * row_e + o];
    }
}

__dpct_inline__ void map_ids_kernel(int32_t *ids,
                                    const int32_t *__restrict__ table, int n) {
    const int i =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (i < n) ids[i] = table[ids[i]];
}

__dpct_inline__ void row_top_prob_kernel(const float *__restrict__ logits,
                                         int n_vocab,
                                         const int32_t *__restrict__ ids,
                                         float *__restrict__ probs) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_group(2);
    const float* l = logits + (size_t) t * n_vocab;
    const float m = l[ids[t]];
    float s = 0.0f;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < n_vocab;
         i += item_ct1.get_local_range(2)) s += sycl::native::exp(l[i] - m);
    /*
DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), s,
            o);
    if ((item_ct1.get_local_id(2) & 31) == 0)
        part[item_ct1.get_local_id(2) >> 5] = s;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (item_ct1.get_local_id(2) == 0) {
        float tot = 0.0f;
#pragma unroll
        for (int w = 0; w < (int)(item_ct1.get_local_range(2) >> 5); ++w) tot +=
            part[w];
        probs[t] = 1.0f / tot;
    }
}

__dpct_inline__ void
mtp_select_kernel(const float *__restrict__ R_src, int64_t stride,
                  const int32_t *__restrict__ ids,
                  const int32_t *__restrict__ row_dev,
                  float *__restrict__ R_dst, int32_t *__restrict__ tok_dst,
                  int32_t *out, int j, const float *probs, float *out_p) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = *row_dev;
#pragma unroll
    for (int64_t i =
             (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < stride; i += (int64_t)item_ct1.get_group_range(2) *
                          item_ct1.get_local_range(2))
        R_dst[i] = R_src[(size_t) row * stride + i];
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (item_ct1.get_group(2) == 0 && item_ct1.get_local_id(2) == 0) {
        const int32_t tok = ids[row];
        *tok_dst = tok;
        if (probs != nullptr && out_p != nullptr) ((volatile float*) out_p)[j] = probs[row];
        /*
        DPCT1078: Consider replacing memory_order::acq_rel with
        memory_order::seq_cst for correctness if strong memory order
        restrictions are needed.
        */
        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::system);
        if (out != nullptr) ((volatile int32_t*) out)[j] = tok;
    }
}

__dpct_inline__ void dense_steps_kernel(const int32_t *__restrict__ cells,
                                        int n, int32_t *__restrict__ steps) {
    const int i =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (i >= n) return;
    const int c = cells[i];
    steps[i * 4 + 0] = c;
    steps[i * 4 + 1] = c + 1;
    steps[i * 4 + 2] = (c + 1) / 4;
    steps[i * 4 + 3] = c + 1;
}

// --pipeline-windows: the drafter's chain, teacher forced (see force_token)
__dpct_inline__ void force_token_kernel(int32_t *tok, const int32_t *force,
                                        int j) {
    const int32_t f = ((const volatile int32_t*) force)[j];
    if (f >= 0) *tok = f;
}

}  // namespace

void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream) {
    if (cap <= 0) return;
    if (blob_bytes % 16 != 0) { std::fprintf(stderr, "fetch_blobs: blob size must be a multiple of 16\n"); std::exit(1); }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {
                auto blob_bytes_ct3 = (long long)(blob_bytes / 16);

                cgh.parallel_for<
                    dpct_kernel_name<class fetch_blobs_kernel_14b6f5>>(
                    sycl::nd_range<3>(sycl::range(1, 1, 48 * 8) *
                                          sycl::range(1, 1, 256),
                                      sycl::range(1, 1, 256)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        fetch_blobs_kernel(src, n, (sycl::uint4 *)dst,
                                           blob_bytes_ct3);
                    });
            });
    }
    check("fetch_blobs");
}

void res_patch(int32_t* table, const int32_t* pairs, int n, void* stream) {
    if (n <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class res_patch_kernel_dfb8fa>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1,
                                (n + 255) / 256 < 64 ? (n + 255) / 256 : 64) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    res_patch_kernel(table, pairs, n);
                });
    }
    check("res_patch");
}

void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class rebase_ptrs_kernel_436f27>>(
                sycl::nd_range<3>(sycl::range(1, 1, 128),
                                  sycl::range(1, 1, 128)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    rebase_ptrs_kernel(ptr, n, (unsigned long long)base,
                                       (long long)blob_bytes);
                });
    }
    check("rebase_ptrs");
}

void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class add_streams_broadcast_kernel_7de8f5>>(
                sycl::nd_range<3>(
                    sycl::range(1, (unsigned)n_tok,
                                (unsigned)((n_embd * hc + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    add_streams_broadcast_kernel(h, e, R, n_embd, hc);
                });
    }
    check("add_streams_broadcast");
}

void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream) {
    if (n < 1 || n > 1024) { std::fprintf(stderr, "ident_hits: n out of range\n"); std::exit(1); }
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class ident_hits_kernel_287489>>(
                sycl::nd_range<3>(sycl::range(1, 1, 1024),
                                  sycl::range(1, 1, 1024)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    ident_hits_kernel(ids, n, slot, dst, count);
                });
    }
    check("ident_hits");
}

namespace {
__dpct_inline__ void
copy_row_to_first_kernel(const int32_t *__restrict__ row_dev, float *a,
                         int64_t a_n, float *b, int64_t b_n, float *c,
                         int64_t c_n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t row = *row_dev;
    if (row == 0) return;
    const int64_t i0 = (int64_t)item_ct1.get_group(2) *
                           item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2),
                  st = (int64_t)item_ct1.get_group_range(2) *
                       item_ct1.get_local_range(2);
#pragma unroll
    for (int64_t i = i0; i < a_n; i += st) a[i] = a[row * a_n + i];
#pragma unroll
    for (int64_t i = i0; i < b_n; i += st) b[i] = b[row * b_n + i];
#pragma unroll
    for (int64_t i = i0; i < c_n; i += st) c[i] = c[row * c_n + i];
}
}  // namespace

void copy_row_to_first(const int32_t* row_dev, float* a, int64_t a_n, float* b, int64_t b_n, float* c, int64_t c_n,
                       void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class copy_row_to_first_kernel_993785>>(
                sycl::nd_range<3>(sycl::range(1, 1, 16) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    copy_row_to_first_kernel(row_dev, a, a_n, b, b_n, c, c_n);
                });
    }
    check("copy_row_to_first");
}

void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs, float* out_p) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class mtp_select_kernel_4611fe>>(
                sycl::nd_range<3>(sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    mtp_select_kernel(R_src, R_stride, ids, row_dev, R_dst,
                                      tok_dst, out, j, probs, out_p);
                });
    }
    check("mtp_select");
}

void force_token(int32_t* tok, const int32_t* force, int j, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class force_token_kernel_a9fa72>>(
                sycl::nd_range<3>(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    force_token_kernel(tok, force, j);
                });
    }
    check("force_token");
}

void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream) {
    dpct::queue_ptr s = strata::q_of(stream);
    if (row_bytes % 16 == 0)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        s->submit([&](sycl::handler &cgh) {
            long long row_bytes_ct1 = row_bytes / 16;

            cgh.parallel_for<
                dpct_kernel_name<class gather_rows_kernel_b3ae5a, sycl::uint4>>(
                sycl::nd_range<3>(sycl::range(1, 1, 48 * 8) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    gather_rows_kernel((const sycl::uint4 *)src, row_bytes_ct1,
                                       ids, n, (sycl::uint4 *)dst);
                });
        });
    } else if (row_bytes % 4 == 0)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        s->submit([&](sycl::handler &cgh) {
            long long row_bytes_ct1 = row_bytes / 4;

            cgh.parallel_for<
                dpct_kernel_name<class gather_rows_kernel_f256ac, uint32_t>>(
                sycl::nd_range<3>(sycl::range(1, 1, 48 * 8) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    gather_rows_kernel((const uint32_t *)src, row_bytes_ct1,
                                       ids, n, (uint32_t *)dst);
                });
        });
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        s->parallel_for<
            dpct_kernel_name<class gather_rows_kernel_c5b171, uint8_t>>(
            sycl::nd_range<3>(sycl::range(1, 1, 48 * 8) *
                                  sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            [=](sycl::nd_item<3> item_ct1) {
                gather_rows_kernel(src, row_bytes, ids, n, dst);
            });
    }
    check("gather_rows");
}

void map_ids(int32_t* ids, const int32_t* table, int n, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class map_ids_kernel_43a113>>(
                sycl::nd_range<3>(sycl::range(1, 1, 64), sycl::range(1, 1, 64)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    map_ids_kernel(ids, table, n);
                });
    }
    check("map_ids");
}

void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream) {
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class row_top_prob_kernel_6adbe2>>(
                sycl::nd_range<3>(sycl::range(1, 1, n_rows) *
                                      sycl::range(1, 1, 1024),
                                  sycl::range(1, 1, 1024)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        row_top_prob_kernel(logits, n_vocab, ids, probs);
                    });
    }
    check("row_top_prob");
}

#if !defined(__HIPCC__)   // warp-32 kernels: an AMD card keeps the one-block row_top_prob and sampler_greedy_kernel
namespace {

constexpr int kArgMaxBlocks = 128, kArgThreads = 256, kTopBlocks = 8;

// the larger value, on equality the lower index: sampler_greedy_kernel's order
__dpct_inline__ void arg_take(float ov, int oi, float &bv, int &bi) {
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
}

// a block's (value, index) pick, in thread 0; `none` is the index of no candidate
__dpct_inline__ void arg_block(float &bv, int &bi, int none) {
    /*
    DPCT1115: The sycl::ext::oneapi::group_local_memory_for_overwrite is
    used to allocate group-local memory at the none kernel functor scope of a
    work-group data parallel kernel. You may need to adjust the code.
    */
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sv = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    /*
    DPCT1115: The sycl::ext::oneapi::group_local_memory_for_overwrite is
    used to allocate group-local memory at the none kernel functor scope of a
    work-group data parallel kernel. You may need to adjust the code.
    */
    auto &si = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    for (int off = 16; off > 0; off >>= 1) {
        /*
        DPCT1108: '__shfl_down_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        /*
        DPCT1121: Make sure that the "bv" which is used in the SYCL group
        function/algorithm is initialized.
        */
        const float ov = dpct::experimental::shift_sub_group_left(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), bv,
            off);
        /*
        DPCT1108: '__shfl_down_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        /*
        DPCT1121: Make sure that the "bi" which is used in the SYCL group
        function/algorithm is initialized.
        */
        const int oi = dpct::experimental::shift_sub_group_left(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), bi,
            off);
        arg_take(ov, oi, bv, bi);
    }
    const int warp = (int)(item_ct1.get_local_id(2) >> 5),
              lane = (int)(item_ct1.get_local_id(2) & 31);
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (warp == 0) {
        const int nw = (int)(item_ct1.get_local_range(2) >> 5);
        bv = lane < nw ? sv[lane] : sycl::bit_cast<float, int>(0xff800000);
        bi = lane < nw ? si[lane] : none;
        for (int off = 16; off > 0; off >>= 1) {
            /*
            DPCT1108: '__shfl_down_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            /*
            DPCT1121: Make sure that the "bv" which is used in the SYCL
            group function/algorithm is initialized.
            */
            const float ov = dpct::experimental::shift_sub_group_left(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                bv, off);
            /*
            DPCT1108: '__shfl_down_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            /*
            DPCT1121: Make sure that the "bi" which is used in the SYCL
            group function/algorithm is initialized.
            */
            const int oi = dpct::experimental::shift_sub_group_left(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                bi, off);
            arg_take(ov, oi, bv, bi);
        }
    }
}

// A row's scratch, whatever the launch's row count (one scratch serves windows of every size): its counter, then
// the blocks' values and indices.
struct ArgRow {
    unsigned counter;
    unsigned pad[3];
    float v[kArgMaxBlocks];
    int i[kArgMaxBlocks];
};

// grid (blocks, rows): block b scans [b * per_block, +per_block) of its row
__dpct_inline__ void argmax_rows_kernel(const float *__restrict__ logits, int n,
                                        int per_block,
                                        ArgRow *__restrict__ rows,
                                        int32_t *out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = (int)item_ct1.get_group(1), b = (int)item_ct1.get_group(2),
              nb = (int)item_ct1.get_group_range(2);
    unsigned* counter = &rows[row].counter;
    float* pv = rows[row].v;
    int* pi = rows[row].i;
    const float* l = logits + (size_t) row * n;
    const int lo = b * per_block, hi = sycl::min(n, lo + per_block);
    float bv = sycl::bit_cast<float, int>(0xff800000); // -inf
    int bi = n;
    for (int v = lo + (int)item_ct1.get_local_id(2); v < hi;
         v += (int)item_ct1.get_local_range(2)) {
        const float s = l[v];
        if (s > bv) { bv = s; bi = v; }
    }
    arg_block(bv, bi, n);
    auto &last = *sycl::ext::oneapi::group_local_memory_for_overwrite<bool>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(2) == 0) {
        pv[b] = bv;
        pi[b] = bi;
        /*
        DPCT1078: Consider replacing memory_order::acq_rel with
        memory_order::seq_cst for correctness if strong memory order
        restrictions are needed.
        */
        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::device);
        last =
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                counter, 1u) == (unsigned)(nb - 1);
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (!last) return;
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    bv = sycl::bit_cast<float, int>(0xff800000);
    bi = n;
    if ((int)item_ct1.get_local_id(2) < nb) {
        bv = ((volatile const float *)pv)[item_ct1.get_local_id(2)];
        bi = ((volatile const int *)pi)[item_ct1.get_local_id(2)];
    }
    arg_block(bv, bi, n);
    if (item_ct1.get_local_id(2) == 0) {
        out[row] = bi < n ? bi : 0;   // no value above -inf: 0, as the one-block kernel
        *counter = 0;
    }
}

// a row's top-probability scratch: its counter, then its 32 warps' sums
struct TopRow {
    unsigned counter;
    unsigned pad[3];
    float parts[32];
};

// grid (kTopBlocks, rows), 1024 / kTopBlocks threads: thread j of block b is row_top_prob's thread b * blockDim + j
__dpct_inline__ void row_top_prob_split_kernel(const float *__restrict__ logits,
                                               int n_vocab,
                                               const int32_t *__restrict__ ids,
                                               float *__restrict__ probs,
                                               TopRow *__restrict__ rows) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = (int)item_ct1.get_group(1);
    float* parts = rows[row].parts;
    const float* l = logits + (size_t) row * n_vocab;
    const float m = l[ids[row]];
    const int vt = (int)(item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                         item_ct1.get_local_id(2));
    float s = 0.0f;
#pragma unroll
    for (int i = vt; i < n_vocab; i += 1024) s += sycl::native::exp(l[i] - m);
    /*
DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), s,
            o);
    if ((item_ct1.get_local_id(2) & 31) == 0) parts[vt >> 5] = s;
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    auto &last = *sycl::ext::oneapi::group_local_memory_for_overwrite<bool>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(2) == 0) last =
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &rows[row].counter, 1u) ==
        (unsigned)(item_ct1.get_group_range(2) - 1);
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (!last || item_ct1.get_local_id(2) != 0) return;
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    float tot = 0.0f;
#pragma unroll
    for (int w = 0; w < 32; ++w) tot += ((volatile const float *)parts)[w];
    probs[row] = 1.0f / tot;
    rows[row].counter = 0;
}

}  // namespace

uint64_t argmax_rows_scratch_bytes(int n_rows) { return (uint64_t) n_rows * sizeof(ArgRow); }

void argmax_rows(const float* logits, int n_rows, int n, void* scratch, int32_t* out, void* stream) {
    if (n_rows <= 0) return;
    const int nb = std::min(kArgMaxBlocks, std::max(1, (n + 4095) / 4096));
    const int per_block = (n + nb - 1) / nb;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class argmax_rows_kernel_d5bcf6>>(
                sycl::nd_range<3>(
                    sycl::range(1, (unsigned)n_rows, (unsigned)nb) *
                        sycl::range(1, 1, kArgThreads),
                    sycl::range(1, 1, kArgThreads)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        argmax_rows_kernel(logits, n, per_block,
                                           (ArgRow *)scratch, out);
                    });
    }
    check("argmax_rows");
}

uint64_t row_top_prob_scratch_bytes(int n_rows) { return (uint64_t) n_rows * sizeof(TopRow); }

void row_top_prob_split(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* scratch,
                        void* stream) {
    if (n_rows <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class row_top_prob_split_kernel_9495b0>>(
                sycl::nd_range<3>(sycl::range(1, (unsigned)n_rows, kTopBlocks) *
                                      sycl::range(1, 1, 1024 / kTopBlocks),
                                  sycl::range(1, 1, 1024 / kTopBlocks)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        row_top_prob_split_kernel(logits, n_vocab, ids, probs,
                                                  (TopRow *)scratch);
                    });
    }
    check("row_top_prob_split");
}
#else
uint64_t argmax_rows_scratch_bytes(int) { return 0; }
void argmax_rows(const float*, int, int, void*, int32_t*, void*) {
    std::fprintf(stderr, "argmax_rows: not built for this backend\n");
    std::exit(1);
}
uint64_t row_top_prob_scratch_bytes(int) { return 0; }
void row_top_prob_split(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void*,
                        void* stream) {
    row_top_prob(logits, n_rows, n_vocab, ids, probs, stream);
}
#endif

namespace {
// grid (kNonfiniteBlocks, rows): block b of row t scans [b * per_block, +per_block) and writes its own word.  The
// test is on the bits (exponent all ones), so --use_fast_math cannot fold it away.
__dpct_inline__ void logits_nonfinite_kernel(const float *__restrict__ logits,
                                             int n, int per_block,
                                             uint32_t *flags) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = (int)item_ct1.get_group(1), b = (int)item_ct1.get_group(2);
    const float* l = logits + (size_t) row * n;
    const int lo = b * per_block, hi = sycl::min(n, lo + per_block);
    int bad = 0;
#pragma unroll
    for (int v = lo + (int)item_ct1.get_local_id(2); v < hi;
         v += (int)item_ct1.get_local_range(2))
        bad |=
            (sycl::bit_cast<unsigned int>(l[v]) & 0x7f800000u) == 0x7f800000u;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    bad = (item_ct1.barrier(),
           sycl::any_of_group(
               sycl::ext::oneapi::this_work_item::get_work_group<3>(), bad));
    if (item_ct1.get_local_id(2) == 0) flags[row * kNonfiniteBlocks + b] =
        bad ? 1u : 0u;
}
}  // namespace

void logits_nonfinite_rows(const float* logits, int n_rows, int n, uint32_t* flags, void* stream) {
    if (n_rows <= 0 || n <= 0) return;
    const int per_block = (n + kNonfiniteBlocks - 1) / kNonfiniteBlocks;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class logits_nonfinite_kernel_263d09>>(
                sycl::nd_range<3>(sycl::range(1, (unsigned)n_rows,
                                              (unsigned)kNonfiniteBlocks) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    logits_nonfinite_kernel(logits, n, per_block, flags);
                });
    }
    check("logits_nonfinite_rows");
}

bool multi_block_head_ops() {
#if defined(__HIPCC__)
    return false;
#else
    static const bool on = [] {
        const char* v = std::getenv("STRATA_MULTI_BLOCK_ARGMAX");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
#endif
}

bool argmax_rows_wanted() try {
#if defined(__HIPCC__) || defined(SYCL_LANGUAGE_VERSION)
    return false;   // SYCL port: dpct's major version is not a CUDA compute capability (an Arc must not read as sm_8x)
#else
    // sm_80 to sm_89, and sm_7x with STRATA_ARGMAX_ROWS_SM70=1: sample_tokens' greedy pick there is the one-block kernel (sm_90+ has the cluster kernel)
    static const bool want = [] {
        int dev = 0, major = 0;
        if (DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) != 0 ||
            DPCT_CHECK_ERROR(
                major = dpct::get_device(dev).get_major_version()) != 0)
            return false;
        if (major == 7) {   // STRATA_ARGMAX_ROWS_SM70=1: opt-in on Volta (measured on sm_70 only)
            const char* e = std::getenv("STRATA_ARGMAX_ROWS_SM70");
            return e != nullptr && std::atoi(e) != 0;
        }
        return major == 8;
    }();
    return want && multi_block_head_ops();
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

namespace {
__dpct_inline__ void window_ids_kernel(int32_t *steps, int window, int32_t *ids,
                                       long long stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int q = item_ct1.get_group(1);
    int32_t* st = steps + q * 4;
    const int n_kv = st[1];
    const int start = n_kv > window ? n_kv - window : 0;
    const int width = n_kv - start;
#pragma unroll
    for (int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
         j < width;
         j += item_ct1.get_group_range(2) * item_ct1.get_local_range(2))
        ids[q * stride + j] = start + j;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_group(2) == 0 && item_ct1.get_local_id(2) == 0) st[3] =
        width;
}
}  // namespace

void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class window_ids_kernel_581536>>(
                sycl::nd_range<3>(sycl::range(1, (unsigned)n, 8) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    window_ids_kernel(steps, window, ids,
                                      (long long)ids_stride);
                });
    }
    check("window_ids");
}

namespace {
// window_ids with a floor read from (mapped) memory when the kernel RUNS: the captured graph keeps the pointer, the
// value changes between sessions.  A floor of 0 gives window_ids' selection exactly (strata::kernels::window_start).
__dpct_inline__ void
window_ids_floor_kernel(int32_t *steps, int window, int32_t *ids,
                        long long stride, const volatile int32_t *floor_cell) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int q = item_ct1.get_group(1);
    int32_t* st = steps + q * 4;
    const int n_kv = st[1];
    const int start = strata::kernels::window_start(n_kv, window, *floor_cell);
    const int width = n_kv - start;
#pragma unroll
    for (int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
         j < width;
         j += item_ct1.get_group_range(2) * item_ct1.get_local_range(2))
        ids[q * stride + j] = start + j;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_group(2) == 0 && item_ct1.get_local_id(2) == 0) st[3] =
        width;
}
}  // namespace

void window_ids_floor(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, const int32_t* floor_cell,
                      void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class window_ids_floor_kernel_c54a44>>(
                sycl::nd_range<3>(sycl::range(1, (unsigned)n, 8) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    window_ids_floor_kernel(steps, window, ids,
                                            (long long)ids_stride, floor_cell);
                });
    }
    check("window_ids_floor");
}

void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class dense_steps_kernel_fd7245>>(
                sycl::nd_range<3>(sycl::range(1, 1, 64), sycl::range(1, 1, 64)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    dense_steps_kernel(cells, n, steps);
                });
    }
    check("dense_steps");
}

void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin, bool commit) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || n_tok < 1 || n_tok > kVerifyMaxT ||
        (commit && (n_tok != 1 || t_begin != 0))) {
        std::fprintf(stderr, "gdn_conv_l2_multi: invalid arguments\n");
        std::exit(1);
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class gdn_conv_l2_multi_kernel_b3f737>>(
                sycl::nd_range<3>(
                    sycl::range(1, (unsigned)n_tok, (unsigned)(channels / S)) *
                        sycl::range(1, 1, S),
                    sycl::range(1, 1, S)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gdn_conv_l2_multi_kernel(
                            const_cast<float *>(history), qkv, conv_w, h,
                            channels, qk_heads, eps, t_begin, commit);
                    });
    }
    check("gdn_conv_l2_multi");
}

void gdn_conv_ab_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels, int qk_heads,
                       float eps, int n_tok, int t_begin, bool commit, const float* x, const uint16_t* w_alpha,
                       const uint16_t* w_beta, const float* dt, const float* ssm_a, float* gate, float* beta, int n_embd,
                       int h_v, void* stream) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || n_tok < 1 || n_tok > kVerifyMaxT || n_embd % 8 != 0 ||
        (commit && (n_tok != 1 || t_begin != 0))) {
        std::fprintf(stderr, "gdn_conv_ab_multi: invalid arguments\n");
        std::exit(1);
    }
    const int conv_blocks = channels / S, ab_blocks = (2 * h_v + 1) / 2;
    const dpct::dim3 grid((unsigned)(conv_blocks + ab_blocks), (unsigned)n_tok);
    dpct::queue_ptr st = strata::q_of(stream);
    float* hist = const_cast<float*>(history);
    switch (n_tok) {
    case 1: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        st->parallel_for<
            dpct_kernel_name<class gdn_conv_ab_kernel_f3b4a3,
                             dpct_kernel_scalar<1>, dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, S),
                              sycl::range(1, 1, S)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_conv_ab_kernel<1, true>(
                    conv_blocks, hist, qkv, conv_w, h, channels, qk_heads, eps,
                    t_begin, commit, x, w_alpha, w_beta, dt, ssm_a, gate, beta,
                    n_embd, h_v, n_tok);
            });
    } break;
    case 2: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        st->parallel_for<
            dpct_kernel_name<class gdn_conv_ab_kernel_33d48d,
                             dpct_kernel_scalar<2>, dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, S),
                              sycl::range(1, 1, S)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_conv_ab_kernel<2, true>(
                    conv_blocks, hist, qkv, conv_w, h, channels, qk_heads, eps,
                    t_begin, commit, x, w_alpha, w_beta, dt, ssm_a, gate, beta,
                    n_embd, h_v, n_tok);
            });
    } break;
    case 3: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        st->parallel_for<
            dpct_kernel_name<class gdn_conv_ab_kernel_59584e,
                             dpct_kernel_scalar<3>, dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, S),
                              sycl::range(1, 1, S)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_conv_ab_kernel<3, true>(
                    conv_blocks, hist, qkv, conv_w, h, channels, qk_heads, eps,
                    t_begin, commit, x, w_alpha, w_beta, dt, ssm_a, gate, beta,
                    n_embd, h_v, n_tok);
            });
    } break;
    case 4: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        st->parallel_for<
            dpct_kernel_name<class gdn_conv_ab_kernel_373e90,
                             dpct_kernel_scalar<4>, dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, S),
                              sycl::range(1, 1, S)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_conv_ab_kernel<4, true>(
                    conv_blocks, hist, qkv, conv_w, h, channels, qk_heads, eps,
                    t_begin, commit, x, w_alpha, w_beta, dt, ssm_a, gate, beta,
                    n_embd, h_v, n_tok);
            });
    } break;
    case 5: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        st->parallel_for<
            dpct_kernel_name<class gdn_conv_ab_kernel_c77ac9,
                             dpct_kernel_scalar<5>, dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, S),
                              sycl::range(1, 1, S)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_conv_ab_kernel<5, true>(
                    conv_blocks, hist, qkv, conv_w, h, channels, qk_heads, eps,
                    t_begin, commit, x, w_alpha, w_beta, dt, ssm_a, gate, beta,
                    n_embd, h_v, n_tok);
            });
    } break;
    case 6: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        st->parallel_for<
            dpct_kernel_name<class gdn_conv_ab_kernel_e2a528,
                             dpct_kernel_scalar<6>, dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, S),
                              sycl::range(1, 1, S)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_conv_ab_kernel<6, true>(
                    conv_blocks, hist, qkv, conv_w, h, channels, qk_heads, eps,
                    t_begin, commit, x, w_alpha, w_beta, dt, ssm_a, gate, beta,
                    n_embd, h_v, n_tok);
            });
    } break;
    case 7: case 8: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        st->parallel_for<dpct_kernel_name<class gdn_conv_ab_kernel_af6095,
                                          dpct_kernel_scalar<kVerifyNarrowT>,
                                          dpct_kernel_scalar<false>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, S),
                              sycl::range(1, 1, S)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_conv_ab_kernel<kVerifyNarrowT, false>(
                    conv_blocks, hist, qkv, conv_w, h, channels, qk_heads, eps,
                    t_begin, commit, x, w_alpha, w_beta, dt, ssm_a, gate, beta,
                    n_embd, h_v, n_tok);
            });
    } break;
    default: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        st->parallel_for<dpct_kernel_name<class gdn_conv_ab_kernel_6cc14f,
                                          dpct_kernel_scalar<kVerifyMaxT>,
                                          dpct_kernel_scalar<false>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, S),
                              sycl::range(1, 1, S)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_conv_ab_kernel<kVerifyMaxT, false>(
                    conv_blocks, hist, qkv, conv_w, h, channels, qk_heads, eps,
                    t_begin, commit, x, w_alpha, w_beta, dt, ssm_a, gate, beta,
                    n_embd, h_v, n_tok);
            });
    } break;
    }
    check("gdn_conv_ab_multi");
}

void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class gdn_conv_commit_kernel_dc1d7f>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((channels + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    gdn_conv_commit_kernel(history, qkv, channels, n_keep);
                });
    }
    check("gdn_conv_commit");
}

bool gdn_commit_layers(float* state, int64_t state_stride, int64_t conv_off, const float* qkv, int64_t qkv_stride,
                       const float* h, int64_t h_stride, const float* gate, const float* beta, int64_t gb_stride,
                       int conv_channels, int h_k, int h_v, int n_max, int n_layers, const int32_t* n_keep, void* stream) {
    // the per-layer commit's kernel choice (gdn_step_norm_multi below): only its state-only kernel is batched here
    static const bool per_layer = [] {
        const char* sp = std::getenv("STRATA_GDN_SPLIT");
        const char* cs = std::getenv("STRATA_GDN_COMMIT_SPLIT");
        const char* dbg = std::getenv("STRATA_DBG_GDN");
        return (sp && sp[0] == '1') || (cs && cs[0] == '0') || (dbg && dbg[0] == '1');
    }();
    if (per_layer || n_layers <= 0 || n_layers > 65535 || n_max < 1 || n_max > kVerifyMaxT || n_keep == nullptr ||
        h_v <= 0 || h_k <= 0 || h_v % h_k != 0)
        return false;
    dpct::queue_ptr st = strata::q_of(stream);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        st->submit([&](sycl::handler &cgh) {
            auto state_conv_off_ct0 = state + conv_off;

            cgh.parallel_for<
                dpct_kernel_name<class gdn_conv_commit_layers_kernel_7c47f1>>(
                sycl::nd_range<3>(
                    sycl::range(1, (unsigned)n_layers,
                                (unsigned)((conv_channels + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    gdn_conv_commit_layers_kernel(
                        state_conv_off_ct0, (long long)state_stride, qkv,
                        (long long)qkv_stride, conv_channels, n_keep);
                });
        });
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        st->parallel_for<
            dpct_kernel_name<class gdn_step_commit_layers_kernel_153ac8>>(
            sycl::nd_range<3>(
                sycl::range((unsigned)n_layers, 4u, (unsigned)h_v) *
                    sycl::range(1, RG, 32),
                sycl::range(1, RG, 32)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                gdn_step_commit_layers_kernel(
                    state, (long long)state_stride, h, (long long)h_stride,
                    conv_channels, gate, beta, (long long)gb_stride, h_k, h_v,
                    n_keep);
            });
    }
    check("gdn_commit_layers");
    return true;
}

void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream) {
    if (n_embd % 8 != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_ab_multi: invalid arguments\n");
        std::exit(1);
    }
    const unsigned blocks = (unsigned) ((2 * h_v + 1) / 2);
    dpct::queue_ptr st = strata::q_of(stream);
    switch (n_tok) {
    case 1: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        st->parallel_for<
            dpct_kernel_name<class gdn_ab_multi_kernel_34dcce,
                             dpct_kernel_scalar<1>, dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, 1, 64),
                              sycl::range(1, 1, 64)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_ab_multi_kernel<1, true>(x, w_alpha, w_beta, dt, ssm_a,
                                             gate, beta, n_embd, h_v, n_tok);
            });
    } break;
    case 2: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        st->parallel_for<
            dpct_kernel_name<class gdn_ab_multi_kernel_34dcce,
                             dpct_kernel_scalar<2>, dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, 1, 64),
                              sycl::range(1, 1, 64)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_ab_multi_kernel<2, true>(x, w_alpha, w_beta, dt, ssm_a,
                                             gate, beta, n_embd, h_v, n_tok);
            });
    } break;
    case 3: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        st->parallel_for<
            dpct_kernel_name<class gdn_ab_multi_kernel_34dcce,
                             dpct_kernel_scalar<3>, dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, 1, 64),
                              sycl::range(1, 1, 64)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_ab_multi_kernel<3, true>(x, w_alpha, w_beta, dt, ssm_a,
                                             gate, beta, n_embd, h_v, n_tok);
            });
    } break;
    case 4: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        st->parallel_for<
            dpct_kernel_name<class gdn_ab_multi_kernel_34dcce,
                             dpct_kernel_scalar<4>, dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, 1, 64),
                              sycl::range(1, 1, 64)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_ab_multi_kernel<4, true>(x, w_alpha, w_beta, dt, ssm_a,
                                             gate, beta, n_embd, h_v, n_tok);
            });
    } break;
    case 5: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        st->parallel_for<
            dpct_kernel_name<class gdn_ab_multi_kernel_34dcce,
                             dpct_kernel_scalar<5>, dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, 1, 64),
                              sycl::range(1, 1, 64)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_ab_multi_kernel<5, true>(x, w_alpha, w_beta, dt, ssm_a,
                                             gate, beta, n_embd, h_v, n_tok);
            });
    } break;
    case 6: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        st->parallel_for<
            dpct_kernel_name<class gdn_ab_multi_kernel_34dcce,
                             dpct_kernel_scalar<6>, dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, 1, 64),
                              sycl::range(1, 1, 64)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_ab_multi_kernel<6, true>(x, w_alpha, w_beta, dt, ssm_a,
                                             gate, beta, n_embd, h_v, n_tok);
            });
    } break;
    case 7: case 8: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        st->parallel_for<dpct_kernel_name<class gdn_ab_multi_kernel_34dcce,
                                          dpct_kernel_scalar<kVerifyNarrowT>,
                                          dpct_kernel_scalar<false>>>(
            sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, 1, 64),
                              sycl::range(1, 1, 64)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_ab_multi_kernel<kVerifyNarrowT, false>(
                    x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v,
                    n_tok);
            });
    } break;
    default: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        st->parallel_for<dpct_kernel_name<class gdn_ab_multi_kernel_34dccf,
                                          dpct_kernel_scalar<kVerifyMaxT>,
                                          dpct_kernel_scalar<false>>>(
            sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, 1, 64),
                              sycl::range(1, 1, 64)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_ab_multi_kernel<kVerifyMaxT, false>(x, w_alpha, w_beta, dt,
                                                        ssm_a, gate, beta,
                                                        n_embd, h_v, n_tok);
            });
    } break;
    }
    check("gdn_ab_multi");
}

void gdn_step_norm_groups(const GdnGroups& groups, const float* h, int conv_channels, const float* gate, const float* beta,
                          const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, bool commit,
                          void* stream) {
    if (groups.n < 1 || groups.n > 8 || !h || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v % h_k) {
        std::fprintf(stderr, "gdn_step_norm_groups: invalid arguments\n");
        std::exit(1);
    }
    for (int i = 0; i < groups.n; ++i)
        if (!groups.state[i] || groups.ntok[i] < 1 || groups.ntok[i] > kVerifyMaxT || (commit && !groups.keep[i])) {
            std::fprintf(stderr, "gdn_step_norm_groups: invalid group %d\n", i);
            std::exit(1);
        }
    const dpct::dim3 g1((unsigned)h_v, (unsigned)groups.n), b1(S, RG);
    // the kernel gdn_step_norm_multi picks for these calls: a window's rows (all outputs, no keep) or a commit of
    // one-row groups (keep, t_out_begin 0)
    if (commit)
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};
        dpct::has_capability_or_fail(
            strata::q_of(stream)->get_device(),
            {sycl::aspect::fp16});

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<
                class gdn_step_norm_multi_kernel_42ae27,
                dpct_kernel_scalar<false>, dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(g1 * b1, b1), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gdn_step_norm_multi_kernel<false, false>(
                            nullptr, h, conv_channels, gate, beta, z, gamma,
                            eps, y, h_k, h_v, 0, nullptr, 0, nullptr, groups);
                    });
    } else
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed
    the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};
        dpct::has_capability_or_fail(
            strata::q_of(stream)->get_device(),
            {sycl::aspect::fp16});

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<
                class gdn_step_norm_multi_kernel_42ae27,
                dpct_kernel_scalar<true>, dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(g1 * b1, b1), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gdn_step_norm_multi_kernel<true, false>(
                            nullptr, h, conv_channels, gate, beta, z, gamma,
                            eps, y, h_k, h_v, 0, nullptr, 0, nullptr, groups);
                    });
    }
    check("gdn_step_norm_groups");
}

void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin, void* xq_out) {
    GdnQ81* const xq = static_cast<GdnQ81*>(xq_out);   // S26: non-null = also the outputs' q8_1 (STRATA_QFUSE)
    if (!state || !h || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v % h_k || n_tok < 1 ||
        n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_step_norm_multi: invalid arguments\n");
        std::exit(1);
    }
    static const bool qk_reg = [] {
        const char* v = std::getenv("STRATA_GDN_QK_REG");
        return v && v[0] == '1';
    }();
    if (qk_reg && !xq && !n_keep && t_out_begin == 0 &&
        h_k == 16 && h_v == 48 && conv_channels == 10240 &&
        strata::intel_gpu_gen(strata::q_of(stream)->get_device()) == strata::IntelGpuGen::Battlemage) {
        static const bool note = [] {
            std::fprintf(stderr, "strata: GDN QK register broadcast enabled (GRF256)\n");
            return true;
        }();
        (void) note;
        auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
        strata::q_of(stream)->parallel_for<dpct_kernel_name<class gdn_qk_register>>(
            sycl::nd_range<3>(sycl::range(1, RG, (unsigned) h_v * S), sycl::range(1, RG, S)),
            props, [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
                gdn_step_norm_multi_kernel<true, false, true>(
                    state, h, conv_channels, gate, beta, z, gamma, eps, y,
                    h_k, h_v, n_tok, nullptr, 0, nullptr);
            });
        check("gdn QK register");
        return;
    }
    static const bool split = [] { const char* v = std::getenv("STRATA_GDN_SPLIT"); return v && v[0] == '1'; }();
    // S26: the split runs where it is faster (gfx1151 harness, h_v 48: the commit 8-19% per call; a window of 2+ rows
    // 3-13%; one row: no gain, the old kernel stays) - bitwise equal in every case
    if (split && (t_out_begin >= n_tok || n_tok - t_out_begin >= 2)) {
        const dpct::dim3 grid((unsigned)h_v, S / GS_COLS), block(GS_COLS, RG);
        if (t_out_begin >= n_tok) {   // no outputs (the commit): the state only
            if (n_tok > kVerifyNarrowT)
                /*
                DPCT1049: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                auto exp_props = sycl::ext::oneapi::experimental::properties{};

                strata::q_of(stream)
                    ->parallel_for<
                        dpct_kernel_name<class gdn_state_replay_kernel_e4d4c5,
                                         dpct_kernel_scalar<kVerifyMaxT>>>(
                        sycl::nd_range<3>(grid * block, block), exp_props,
                        [=](sycl::nd_item<3> item_ct1) {
                            gdn_state_replay_kernel<kVerifyMaxT>(
                                state, h, conv_channels, gate, beta, h_k, h_v,
                                n_tok, n_keep);
                        });
            } else
            /*
            DPCT1049: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            {
                auto exp_props = sycl::ext::oneapi::experimental::properties{};

                strata::q_of(stream)
                    ->parallel_for<
                        dpct_kernel_name<class gdn_state_replay_kernel_e4d4c4,
                                         dpct_kernel_scalar<kVerifyNarrowT>>>(
                        sycl::nd_range<3>(grid * block, block), exp_props,
                        [=](sycl::nd_item<3> item_ct1) {
                            gdn_state_replay_kernel<kVerifyNarrowT>(
                                state, h, conv_channels, gate, beta, h_k, h_v,
                                n_tok, n_keep);
                        });
            }
        } else {
            /*
            DPCT1049: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            {
                auto exp_props = sycl::ext::oneapi::experimental::properties{};

                strata::q_of(stream)
                    ->parallel_for<
                        dpct_kernel_name<class gdn_step_split_kernel_83c676,
                                         dpct_kernel_scalar<true>>>(
                        sycl::nd_range<3>(grid * block, block), exp_props,
                        [=](sycl::nd_item<3> item_ct1) {
                            gdn_step_split_kernel<true>(
                                state, h, conv_channels, gate, beta, y, h_k,
                                h_v, n_tok, n_keep, t_out_begin);
                        });
            }
            const dpct::dim3 ng((unsigned)h_v, (unsigned)(n_tok - t_out_begin));
            /*
            DPCT1049: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            if (xq) {
                auto exp_props = sycl::ext::oneapi::experimental::properties{};
                dpct::has_capability_or_fail(
                    strata::q_of(stream)->get_device(),
                    {sycl::aspect::fp16});

                strata::q_of(stream)
                    ->parallel_for<
                        dpct_kernel_name<class gdn_out_norm_kernel_43a701,
                                         dpct_kernel_scalar<true>>>(
                        sycl::nd_range<3>(ng * sycl::range(1, RG, S),
                                          sycl::range(1, RG, S)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                gdn_out_norm_kernel<true>(z, gamma, eps, y, h_v,
                                                          n_tok, n_keep,
                                                          t_out_begin, xq);
                            });
            }
            /*
            DPCT1049: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            else {
                auto exp_props = sycl::ext::oneapi::experimental::properties{};
                dpct::has_capability_or_fail(
                    strata::q_of(stream)->get_device(),
                    {sycl::aspect::fp16});

                strata::q_of(stream)
                    ->parallel_for<
                        dpct_kernel_name<class gdn_out_norm_kernel_e67d5f,
                                         dpct_kernel_scalar<false>>>(
                        sycl::nd_range<3>(ng * sycl::range(1, RG, S),
                                          sycl::range(1, RG, S)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                gdn_out_norm_kernel<false>(
                                    z, gamma, eps, y, h_v, n_tok, n_keep,
                                    t_out_begin, nullptr);
                            });
            }
        }
        check("gdn_step_norm_multi (split)");
        return;
    }
    static const bool commit_split = [] {
        const char* e = std::getenv("STRATA_GDN_COMMIT_SPLIT");
        return !e || e[0] != '0';
    }();
    if (commit_split && n_keep != nullptr && t_out_begin >= n_tok) {
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{};

            strata::q_of(stream)
                ->parallel_for<
                    dpct_kernel_name<class gdn_step_commit_kernel_5d9da7>>(
                    sycl::nd_range<3>(sycl::range(1, 4u, (unsigned)h_v) *
                                          sycl::range(1, RG, 32),
                                      sycl::range(1, RG, 32)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        gdn_step_commit_kernel(state, h, conv_channels, gate,
                                               beta, h_k, h_v, n_keep);
                    });
        }
        check("gdn_step_commit");
        return;
    }
    const dpct::dim3 g1((unsigned)h_v), b1(S, RG);
    const bool all_out = n_keep == nullptr && t_out_begin <= 0;
    if (xq && t_out_begin < n_tok) {
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        if (all_out) {
            auto exp_props = sycl::ext::oneapi::experimental::properties{};
            dpct::has_capability_or_fail(
                strata::q_of(stream)->get_device(),
                {sycl::aspect::fp16});

            strata::q_of(stream)
                ->parallel_for<dpct_kernel_name<
                    class gdn_step_norm_multi_kernel_42ae26,
                    dpct_kernel_scalar<true>, dpct_kernel_scalar<true>>>(
                    sycl::nd_range<3>(g1 * b1, b1), exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            gdn_step_norm_multi_kernel<true, true>(
                                state, h, conv_channels, gate, beta, z, gamma,
                                eps, y, h_k, h_v, n_tok, nullptr, 0, xq, {});
                        });
        }
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        else {
            auto exp_props = sycl::ext::oneapi::experimental::properties{};
            dpct::has_capability_or_fail(
                strata::q_of(stream)->get_device(),
                {sycl::aspect::fp16});

            strata::q_of(stream)
                ->parallel_for<dpct_kernel_name<
                    class gdn_step_norm_multi_kernel_42ae26,
                    dpct_kernel_scalar<false>, dpct_kernel_scalar<true>>>(
                    sycl::nd_range<3>(g1 * b1, b1), exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            gdn_step_norm_multi_kernel<false, true>(
                                state, h, conv_channels, gate, beta, z, gamma,
                                eps, y, h_k, h_v, n_tok, n_keep, t_out_begin,
                                xq, {});
                        });
        }
    } else if (all_out) {
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        auto exp_props = sycl::ext::oneapi::experimental::properties{};
        dpct::has_capability_or_fail(
            strata::q_of(stream)->get_device(),
            {sycl::aspect::fp16});

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<
                class gdn_step_norm_multi_kernel_42ae26,
                dpct_kernel_scalar<true>, dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(g1 * b1, b1), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gdn_step_norm_multi_kernel<true, false>(
                            state, h, conv_channels, gate, beta, z, gamma, eps,
                            y, h_k, h_v, n_tok, nullptr, 0, nullptr, {});
                    });
    } else {
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        auto exp_props = sycl::ext::oneapi::experimental::properties{};
        dpct::has_capability_or_fail(
            strata::q_of(stream)->get_device(),
            {sycl::aspect::fp16});

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<
                class gdn_step_norm_multi_kernel_42ae26,
                dpct_kernel_scalar<false>, dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(g1 * b1, b1), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gdn_step_norm_multi_kernel<false, false>(
                            state, h, conv_channels, gate, beta, z, gamma, eps,
                            y, h_k, h_v, n_tok, n_keep, t_out_begin, nullptr,
                            {});
                    });
    }
    check("gdn_step_norm_multi");
}

namespace {
__dpct_inline__ void wait_flag_ge_kernel(const volatile uint32_t *flag,
                                         uint32_t value, uint32_t spin_max) {
    for (uint32_t spin = 0; spin < spin_max && strata::sys_load(flag) < value; ++spin) strata_spin_pause();
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}

// the GPU's clock in ns (the same sources as gpu_stamp_kernel)
__dpct_inline__ unsigned long long strata_now_ns() {
    unsigned long long t;
#if defined(STRATA_HIP_GFX906)
    t = wall_clock64() * 40ull;
#elif defined(__HIPCC__)
    t = wall_clock64() * 10ull;
#else
    /*
    DPCT1053: Migration of device assembly code is not supported.
    */
    t = 0;   // SYCL: no %globaltimer equivalent; the stage profiler is inert on this backend
#endif
    return t;
}

// the overlapped split's hand-off wait, bounded: spin until *flag >= value, or give up after timeout_ns (0: never),
// marking *err = 1 (mapped) so the host fails the window instead of using a stale hand-off.  The window's later
// kernels still run (on whatever the hand-off buffer holds) so the graph completes and no stream is left hanging.
__dpct_inline__ void wait_handoff_kernel(const volatile uint32_t *flag,
                                         uint32_t value, uint32_t *err,
                                         unsigned long long timeout_ns) {
    const unsigned long long t0 = timeout_ns ? strata_now_ns() : 0ull;
    while (*flag < value) {
        strata_spin_pause();
        if (timeout_ns && strata_now_ns() - t0 > timeout_ns) {
            *(volatile uint32_t*) err = 1u;
            /*
            DPCT1078: Consider replacing memory_order::acq_rel with
            memory_order::seq_cst for correctness if strong memory order
            restrictions are needed.
            */
            sycl::atomic_fence(sycl::memory_order::acq_rel,
                               sycl::memory_scope::system);
            return;
        }
    }
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}

__dpct_inline__ void
handoff_publish_kernel(float *dst, const float *__restrict__ a, int64_t na,
                       const float *__restrict__ b, int64_t nb,
                       const float *__restrict__ c, int64_t nc,
                       uint32_t *counter, uint32_t *flag,
                       const volatile uint32_t *drop) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t n = na + nb + nc;
    volatile float* const out = dst;
#pragma unroll
    for (int64_t i =
             (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < n; i += (int64_t)item_ct1.get_group_range(2) *
                     item_ct1.get_local_range(2))
        out[i] = i < na ? a[i] : i < na + nb ? b[i - na] : c[i - na - nb];
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(
        sycl::memory_order::acq_rel,
        sycl::memory_scope::system); // this thread's stores reach host memory
                                     // before the block counts itself done
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (item_ct1.get_local_id(2) == 0) {
        const uint32_t prev =
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                counter, 1u);
        if (prev ==
            item_ct1.get_group_range(2) - 1) { // every block's payload is out:
                                               // raise the next stage's flag
            *(volatile uint32_t*) counter = 0u;
            /*
            DPCT1078: Consider replacing memory_order::acq_rel with
            memory_order::seq_cst for correctness if strong memory order
            restrictions are needed.
            */
            sycl::atomic_fence(sycl::memory_order::acq_rel,
                               sycl::memory_scope::system);
            if (drop == nullptr || *drop == 0u) {   // test hook (STRATA_TEST_HANDOFF_DROP): a lost publish
                *(volatile uint32_t*) flag = 1u;
                /*
                DPCT1078: Consider replacing memory_order::acq_rel with
                memory_order::seq_cst for correctness if strong memory order
                restrictions are needed.
                */
                sycl::atomic_fence(sycl::memory_order::acq_rel,
                                   sycl::memory_scope::system);
            }
        }
    }
}

// wait_flag_ge, with the GPU's clock as it arrives (t_in) and as it leaves (t_out), either may be null
__dpct_inline__ void
wait_flag_ge_stamped_kernel(const volatile uint32_t *flag, uint32_t value,
                            volatile unsigned long long *t_in,
                            volatile unsigned long long *t_out,
                            uint32_t spin_max) {
    if (t_in != nullptr) *t_in = strata_now_ns();
    for (uint32_t spin = 0; spin < spin_max && strata::sys_load(flag) < value; ++spin) strata_spin_pause();
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
    if (t_out != nullptr) *t_out = strata_now_ns();
}
}  // namespace

void wait_handoff(const uint32_t* flag, uint32_t value, uint32_t* err, unsigned long long timeout_ns, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class wait_handoff_kernel_2f7d33>>(
                sycl::nd_range<3>(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    wait_handoff_kernel(flag, value, err, timeout_ns);
                });
    }
    check("wait_handoff");
}

void handoff_publish(float* dst, const float* a, int64_t na, const float* b, int64_t nb, const float* c, int64_t nc,
                     uint32_t* counter, uint32_t* flag, void* stream, const uint32_t* drop) {
    const int64_t n = na + nb + nc;
    if (n <= 0 || dst == nullptr || counter == nullptr || flag == nullptr) {
        std::fprintf(stderr, "handoff_publish: invalid arguments\n");
        std::exit(1);
    }
    const int64_t nb256 = (n + 255) / 256;
    const int blocks = (int) (nb256 < 64 ? nb256 : 64);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class handoff_publish_kernel_d7f4e8>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    handoff_publish_kernel(dst, a, na, b, nb, c, nc, counter,
                                           flag, drop);
                });
    }
    check("handoff_publish");
}

namespace {
// one thread per window entry: up to kVerifyNarrowT tokens x 10 routed experts (80), so 128 (#646 had 64: a window of
// 7+ tokens lost its last entries).  A window of 9..16 rows (up to 160 entries) runs the same kernel on 256 threads;
// one block size per window class, so a narrow window launches exactly the 128-thread kernel it always did.
constexpr int kResidentPlanNarrow = 128;
constexpr int kResidentPlanMax = kVerifyMaxT > kVerifyNarrowT ? 256 : kResidentPlanNarrow;
static_assert(kVerifyNarrowT * 10 <= kResidentPlanNarrow, "resident_plan: one thread per entry");
static_assert(kVerifyMaxT * 10 <= kResidentPlanMax, "resident_plan: one thread per entry");
// the parallel scan keeps one partial sum per warp in s_wsum[NT / 32] and packs (entries << 16 | groups) in an int
static_assert(kResidentPlanMax <= 256 && kResidentPlanMax % 32 == 0, "resident_plan: at most 8 warps");
static_assert(kResidentPlanMax < 32768, "resident_plan: the packed entry count must stay below 2^15");
template <int NT>
/*
DPCT1110: The total declared local variable size in device function
resident_plan_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
resident_plan_kernel(const int32_t *__restrict__ ids, int n, int k,
                     const int32_t *__restrict__ res, int n_expert,
                     const uint8_t *cache_base,
                     const unsigned long long *slot_off, long long blob,
                     int32_t *__restrict__ pl, long long capx, uint32_t *skip,
                     uint32_t ring, const unsigned long long *__restrict__ mir,
                     volatile uint32_t *plan_err, int32_t *usage, bool mfirst,
                     uint8_t *stage, long long sstride, unsigned long long *msrc, uint8_t *rbase, int32_t *rof,
                     int32_t *rof_g, int32_t *rtag, uint8_t *rref, int32_t *rhand, int R, int key_base,
                     uint32_t *mstat) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int NW = NT / 32;
    auto &s_ids =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[NT]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_excl =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[NT]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_wsum =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[NW]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    // STRATA_PLAN_MIRROR_FIRST: the host-mirrored groups' own scan (they take the first group indices)
    auto &s_exclm =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[NT]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_wsumm =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[NW]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_bad = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int tid = item_ct1.get_local_id(2);
    if (tid == 0) s_bad = 0;
    item_ct1.barrier(sycl::access::fence_space::local_space);

    int32_t eid = -1;
    int32_t slot = -1;
    unsigned long long maddr = 0;   // SYCL port: a host-mirrored expert (not in VRAM): its pinned mirror, read over PCIe
    if (tid < n) {
        eid = ids[tid];
        s_ids[tid] = eid;
        slot = (eid >= 0 && eid < n_expert) ? res[eid] : -1;
        if (slot < 0 && eid >= 0 && eid < n_expert && mir != nullptr) maddr = mir[eid];
        if (usage != nullptr && eid >= 0 && eid < n_expert)   // the adaptive tier's routing counts (device side)
            sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(usage[eid]).fetch_add(1);
        if (slot < 0 && maddr == 0)
            dpct::atomic_fetch_or<sycl::access::address_space::generic_space>(
                &s_bad, 1);
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (s_bad) {
        if (tid == 0 && skip != nullptr) *skip = 0;
        if (tid == 0 && skip == nullptr) {   // #871: the all-resident graph has no host plan to fall back on
            pl[0] = 0; pl[1] = 0; pl[2] = 0; pl[3] = 0;   // an empty plan: no expert runs on a stale pointer
            /*
            DPCT1078: Consider replacing memory_order::acq_rel with
            memory_order::seq_cst for correctness if strong memory order
            restrictions are needed.
            */
            if (plan_err != nullptr) {
                    *plan_err = 1;
                    sycl::atomic_fence(sycl::memory_order::acq_rel,
                                       sycl::memory_scope::system);
            }
        }
        return;
    }

    int first_j = tid;
    int rank_in_group = 0;
    int count_same = 0;
    if (tid < n) {
#pragma unroll
        for (int j = 0; j < n; ++j) {
            if (s_ids[j] == eid) {
                if (j < first_j) first_j = j;
                if (j < tid) ++rank_in_group;
                ++count_same;
            }
        }
    }
    const bool is_first = (tid < n && first_j == tid && (slot >= 0 || maddr != 0));
    const int my_cnt = is_first ? count_same : 0;
    // STRATA_MIRROR_RING: a mirrored group whose expert is in the ring reads it there (no copy); one that is not takes a
    // ring slot for mirror_stage to copy into (a clock from the hand over NT candidates: hit by this call 2, inserted 1,
    // passed over 2 -> 1 -> 0; taken at 0, else at 1; never a slot this call hit).  The ring outlives the windows; an
    // expert's bytes never change, so a tag can only be missing, never wrong.  No slot left: the per-call staging blob.
    int rslot = -1;
    bool rnew = false;
    if (rbase != nullptr) {
        auto &s_cand = *sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[NT]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
        auto &s_pick = *sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[NT]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
        auto &s_rn = *sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[3]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
        if (tid == 0) { s_rn[0] = 0; s_rn[1] = 0; s_rn[2] = -1; }
        item_ct1.barrier(sycl::access::fence_space::global_and_local);
        int myj = -1;
        if (is_first && slot < 0) {
            const int rs = rof[eid];
            if (rs >= 0) { rslot = rs; rref[rs] = 2; }
            else myj = sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                        sycl::access::address_space::local_space>(s_rn[0]).fetch_add(1);
        }
        item_ct1.barrier(sycl::access::fence_space::global_and_local);
        const int nm = s_rn[0];
        if (nm > 0) {
            const int h = *rhand;
            const int c = (h + tid) % R;
            s_cand[tid] = rref[c];
            item_ct1.barrier(sycl::access::fence_space::local_space);
            if (tid == 0) {
                int got = 0, last = -1;
                for (int t = 0; t < NT && got < nm; ++t) if (s_cand[t] == 0) { s_pick[got++] = t; last = t > last ? t : last; s_cand[t] = 3; }
                for (int t = 0; t < NT && got < nm; ++t) if (s_cand[t] == 1) { s_pick[got++] = t; last = t > last ? t : last; s_cand[t] = 3; }
                s_rn[1] = got;
                s_rn[2] = last;
                *rhand = (h + last + 1) % R;
            }
            item_ct1.barrier(sycl::access::fence_space::local_space);
            const int cv = s_cand[tid];
            if (tid <= s_rn[2] && (cv == 1 || cv == 2)) rref[c] = (uint8_t) (cv - 1);
            if (myj >= 0 && myj < s_rn[1]) {
                const int cs = (h + s_pick[myj]) % R;
                const int old = rtag[cs];
                if (old >= 0) rof_g[old] = -1;
                rtag[cs] = key_base + eid;
                rof[eid] = cs;
                rref[cs] = 1;
                rslot = cs;
                rnew = true;
            }
        }
    }
    // STRATA_MIRROR_STATS (measurement): this layer's calls, entries, groups, VRAM groups, ring hits, ring inserts and
    // groups copied to (or read from) a per-call staging blob / the mirror
    if (mstat != nullptr) {
        using aref = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                      sycl::access::address_space::global_space>;
        if (tid == 0) { aref(mstat[0]).fetch_add(1u); aref(mstat[1]).fetch_add((uint32_t) n); }
        if (is_first) {
            aref(mstat[2]).fetch_add(1u);
            aref(mstat[slot >= 0 ? 3 : rnew ? 5 : rslot >= 0 ? 4 : 6]).fetch_add(1u);
        }
    }
    // mfirst: the groups read over PCIe (slot < 0: every entry of a group shares its expert's slot) are numbered
    // first, so their work-groups are dispatched at the start of the expert launch rather than wherever their
    // first entry falls; the group order changes no entry's arithmetic (each entry's rows are summed alone)
    // stage (STRATA_MIRROR_STAGE): the mirrored groups are numbered LAST instead, behind every VRAM group, so the VRAM
    // groups are a prefix of the list (counts[3] of them) and the mirrored ones a list of their own (ptr2 / start2 /
    // counts[2], pointing at their VRAM staging blobs; msrc[j] = where mirror_stage copies blob j from)
    const bool msplit = mfirst || stage != nullptr;
    const bool in_m = msplit && slot < 0;
    const int my_pack = in_m ? 0 : (my_cnt << 16) | (is_first ? 1 : 0);
    const int my_packm = in_m ? (my_cnt << 16) | (is_first ? 1 : 0) : 0;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int pref = sycl::inclusive_scan_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), my_pack,
        sycl::plus<int>());
    s_excl[tid] = pref - my_pack;
    if (lane == 31) {
        s_wsum[warp] = pref;
    }
    if (msplit) {
        const int prefm = sycl::inclusive_scan_over_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(), my_packm,
            sycl::plus<int>());
        s_exclm[tid] = prefm - my_packm;
        if (lane == 31) s_wsumm[warp] = prefm;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    int mtot = 0;   // the mirrored groups' (entries << 16 | groups), ahead of every device group
    if (msplit) {
#pragma unroll
        for (int w = 0; w < NW; ++w) mtot += s_wsumm[w];
    }
    int vtot = 0;   // stage: the VRAM groups' (entries << 16 | groups), ahead of every mirrored group
    if (stage != nullptr) {
#pragma unroll
        for (int w = 0; w < NW; ++w) vtot += s_wsum[w];
    }
    const int off_v = stage != nullptr ? 0 : mtot, off_m = stage != nullptr ? vtot : 0;

    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
    int32_t* start2 = pl + ptr_off + 4 * capx;

    if (is_first) {
        int tot = in_m ? s_exclm[tid] + off_m : s_excl[tid] + off_v;
#pragma unroll
        for (int w = 0; w < NW; ++w) {
            if (w < warp) tot += in_m ? s_wsumm[w] : s_wsum[w];
        }
        const int grp_idx = tot & 0xffff;
        const int ent_start = tot >> 16;
        const int mj = grp_idx - (vtot & 0xffff);   // stage: the mirrored group's index in its own list
        const unsigned long long sa = rslot >= 0 ? (unsigned long long) (rbase + (size_t) rslot * (size_t) sstride)
                                    : stage != nullptr ? (unsigned long long) (stage + (size_t) mj * (size_t) sstride) : 0ull;
        ptr[grp_idx] = slot >= 0 ? (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob))
                     : stage != nullptr ? sa : maddr;
        start[grp_idx] = ent_start;
        if (stage != nullptr && slot < 0) {
            ptr[capx + mj] = sa;
            start2[mj] = ent_start;
            msrc[mj] = (rslot >= 0 && !rnew) ? 0ull : maddr;   // a ring hit: nothing to copy
        }
    }
    if (tid < n) {
        const int fj_warp = first_j >> 5;
        int fj_tot = in_m ? s_exclm[first_j] + off_m : s_excl[first_j] + off_v;
#pragma unroll
        for (int w = 0; w < NW; ++w) {
            if (w < fj_warp) fj_tot += in_m ? s_wsumm[w] : s_wsum[w];
        }
        const int out_idx = (fj_tot >> 16) + rank_in_group;
        dst[out_idx] = tid;
        tok[out_idx] = tid / k;
    }
    if (tid == 0) {
        int groups;
        if constexpr (NW == 4) groups = (s_wsum[0] + s_wsum[1] + s_wsum[2] + s_wsum[3] + mtot) & 0xffff;   // (the 128-thread kernel as it was)
        else {
            groups = mtot;
#pragma unroll
            for (int w = 0; w < NW; ++w) groups += s_wsum[w];
            groups &= 0xffff;
        }
        start[groups] = n;
        const int gm = stage != nullptr ? (mtot & 0xffff) : 0;
        start2[gm] = n;
        counts[0] = groups;
        counts[1] = n;
        counts[2] = gm;
        counts[3] = stage != nullptr ? (vtot & 0xffff) : groups;
        if (skip != nullptr) {
            /*
            DPCT1078: Consider replacing memory_order::acq_rel with
            memory_order::seq_cst for correctness if strong memory order
            restrictions are needed.
            */
            sycl::atomic_fence(sycl::memory_order::acq_rel,
                               sycl::memory_scope::device);
            strata::sys_store(skip, ring);
        }
    }
}
// The same plan in one block of 128 threads (n <= 128): thread i owns entry i. Groups are the distinct experts in
// order of first occurrence; group g's entries are its occurrences in increasing i - exactly the loop above
// (S26: the one-thread loop took ~44 us per call on gfx1151, 48 per window).
void resident_plan_par_kernel(const int32_t* __restrict__ ids, int n, int k, const int32_t* __restrict__ res,
                                         int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off,
                                         long long blob, int32_t* __restrict__ pl, long long capx, uint32_t* skip,
                                         uint32_t ring) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &s_id = *sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[128]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_first =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[128]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_size =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[128]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_bad = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int i = item_ct1.get_local_id(2);
    if (i == 0) s_bad = 0;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    int32_t e = -1;
    if (i < n) {
        e = ids[i];
        s_id[i] = e;
        if (e < 0 || e >= n_expert || res[e] < 0) s_bad = 1;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (s_bad) { if (i == 0) *skip = 0; return; }
    int first = i, rank = 0, size = 0;
    if (i < n) {
        for (int j = 0; j < n; ++j) {
            const bool same = s_id[j] == e;
            if (same && j < first) first = j;
            if (same && j < i) ++rank;
            size += same;
        }
    }
    if (i < n) { s_first[i] = first; s_size[i] = size; }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
    int32_t* start2 = pl + ptr_off + 4 * capx;
    if (i < n) {
        // group of entry i = the number of first occurrences before `first`; its start = their sizes' sum
        int group = 0, gstart = 0;
#pragma unroll
        for (int j = 0; j < first; ++j)
            if (s_first[j] == j) { ++group; gstart += s_size[j]; }
        if (i == first) {
            const int32_t slot = res[e];
            ptr[group] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
            start[group] = gstart;
        }
        dst[gstart + rank] = i;
        tok[gstart + rank] = i / k;
        if (i == 0) {
            int groups = 0;
#pragma unroll
            for (int j = 0; j < n; ++j) groups += s_first[j] == j;
            start[groups] = n;
            start2[0] = n;
            counts[0] = groups;
            counts[1] = n;
            counts[2] = 0;
        }
    }
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (i == 0) *skip = ring;
}
__dpct_inline__ void wait_flag_ge_or_kernel(const volatile uint32_t *flag,
                                            uint32_t value,
                                            const volatile uint32_t *skip, uint32_t spin_max) {
    if (strata::sys_load(skip) == value) return;
    for (uint32_t spin = 0; spin < spin_max && strata::sys_load(flag) < value; ++spin) strata_spin_pause();
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}
__dpct_inline__ void copy_i32_unless_kernel(int32_t *__restrict__ dst,
                                            const volatile int32_t *src, int n,
                                            const uint32_t *skip,
                                            uint32_t value) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (strata::sys_load(skip) == value) return;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) dst[i] = src[i];
}
/*
DPCT1052: SYCL does not support the member access for a volatile qualified
vector type. The volatile qualifier was removed. You may need to rewrite the
code.
*/
__dpct_inline__ void copy_or_zero_kernel(sycl::float4 *__restrict__ dst,
                                         const sycl::float4 *src, long long n4,
                                         const uint32_t *skip, uint32_t value, bool pair) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const bool zero = *skip == value;
#pragma unroll
    for (long long i =
             (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < n4; i += (long long)item_ct1.get_group_range(2) *
                      item_ct1.get_local_range(2))
        dst[i] = zero ? sycl::float4(0.f, 0.f, 0.f, 0.f)
                      : (pair ? strata::load_mapped_float4(src + i) : const_cast<const sycl::float4 *>(src)[i]);
}
}  // namespace

namespace {
const int32_t* g_mirror_res = nullptr;
const unsigned long long* g_mirror_table = nullptr;
int32_t* g_usage = nullptr;
uint8_t* g_stage_base = nullptr;           // STRATA_MIRROR_STAGE: the next plans' VRAM staging blobs (resident_plan_set_stage)
long long g_stage_stride = 0;
unsigned long long* g_stage_src = nullptr; // and where their copies come from
uint8_t* g_ring_base = nullptr;            // STRATA_MIRROR_RING: the ring's blobs (stride g_stage_stride)
bool g_ring_on = false;                    // the next plans use it (resident_plan_set_stage's `ring`)
int32_t* g_rof = nullptr;                  // per (layer, expert): its ring slot or -1
int32_t* g_rtag = nullptr;                 // per ring slot: its (layer, expert) or -1
uint8_t* g_rref = nullptr;                 // per ring slot: the clock's count (0..2)
int32_t* g_rhand = nullptr;                // the clock hand
int g_ring_n = 0;
uint32_t* g_mstat = nullptr;               // STRATA_MIRROR_STATS: [n_layers][8] device counters (mirror_stats_*)
long long g_mirror_entries = 0;   // 0: unchecked
// #1980: the layer slice `res_layer` belongs to the table the mirror was set for (a later stage passes its own)
bool in_mirror_table(const int32_t* res_layer) {
    if (g_mirror_res == nullptr || res_layer < g_mirror_res) return false;
    return g_mirror_entries <= 0 || res_layer - g_mirror_res < g_mirror_entries;
}
}
void resident_plan_set_mirror(const int32_t* d_res, const unsigned long long* mirror_table, int32_t* usage, long long entries) {
    g_mirror_entries = entries;
    g_usage = usage;
    g_mirror_res = d_res;
    g_mirror_table = mirror_table;
}
int mirror_ring_slots() {
    static const int n = [] {
        auto set_nonzero = [](const char* k) { const char* e = std::getenv(k); return e != nullptr && e[0] != '\0' && e[0] != '0'; };
        const char* st = std::getenv("STRATA_MIRROR_STAGE");
        if ((st != nullptr && st[0] == '0') || !set_nonzero("STRATA_VERIFY_DEVICE_PLAN") || !set_nonzero("STRATA_VERIFY_NO_HOST")) return 0;
        const char* v = std::getenv("STRATA_MIRROR_RING");
        return v != nullptr && *v ? std::max(0, std::atoi(v)) : 1024;
    }();
    return n;
}
bool mirror_stats_enable(int n_layers) {
    if (g_mstat != nullptr) return true;
    if (n_layers <= 0) return false;
    sycl::queue& q = dpct::get_in_order_queue();
    try { g_mstat = sycl::malloc_device<uint32_t>((size_t) n_layers * 8, q); } catch (const sycl::exception&) { g_mstat = nullptr; }
    if (g_mstat == nullptr) return false;
    q.memset(g_mstat, 0, (size_t) n_layers * 8 * 4).wait();
    return true;
}
bool mirror_stats_read(uint32_t* out, int n_layers, bool reset) {
    if (g_mstat == nullptr) return false;
    sycl::queue& q = dpct::get_in_order_queue();
    q.memcpy(out, g_mstat, (size_t) n_layers * 8 * 4).wait();
    if (reset) q.memset(g_mstat, 0, (size_t) n_layers * 8 * 4).wait();
    return true;
}
static int g_ring_grant = 0;
void mirror_ring_grant(int slots) { g_ring_grant = std::max(0, slots); }
int mirror_ring_granted() { return g_ring_grant; }
bool mirror_ring_create(int slots, long long stride, std::string& err) {
    if (g_ring_base != nullptr) return true;
    if (g_mirror_entries <= 0 || slots < kResidentPlanMax) { err = "no mirror table, or fewer slots than a plan's entries"; return false; }
    sycl::queue& q = dpct::get_in_order_queue();
    try {
        g_ring_base = sycl::malloc_device<uint8_t>((size_t) slots * (size_t) stride, q);
        g_rof = sycl::malloc_device<int32_t>((size_t) g_mirror_entries, q);
        g_rtag = sycl::malloc_device<int32_t>((size_t) slots, q);
        g_rref = sycl::malloc_device<uint8_t>((size_t) slots, q);
        g_rhand = sycl::malloc_device<int32_t>(1, q);
    } catch (const sycl::exception& e) { err = e.what(); }
    if (!g_ring_base || !g_rof || !g_rtag || !g_rref || !g_rhand) { if (err.empty()) err = "out of VRAM"; g_ring_base = nullptr; return false; }
    q.memset(g_rof, 0xff, (size_t) g_mirror_entries * 4);
    q.memset(g_rtag, 0xff, (size_t) slots * 4);
    q.memset(g_rref, 0, (size_t) slots);
    q.memset(g_rhand, 0, 4).wait();
    g_ring_n = slots;
    return true;
}
void resident_plan_set_stage(uint8_t* base, long long stride, unsigned long long* src, bool ring) {
    g_ring_on = ring && g_ring_base != nullptr && src != nullptr;
    g_stage_base = src != nullptr ? base : nullptr;
    g_stage_stride = stride;
    g_stage_src = src;
}
// STRATA_MIRROR_STAGE: the plan's mirrored list (counts[2] blobs: msrc[j] -> ptr2[j], host -> VRAM) copied by a fixed
// grid of work-groups striding over each blob with 16-byte loads (U in flight per lane).  The B70 reads host memory at
// 16.6 GB/s with 4-byte loads (64 bytes per sub-group message: the expert kernels' rate), ~23.5 with 8-byte and 25-26
// with 16-byte loads (the copy engine: 27.5), whatever the number of loads in flight (pcie_read_bench).
void mirror_stage(const int32_t* plan, long long capx, long long blob_bytes, const unsigned long long* src, void* stream) {
    static const int nwg = [] { const char* v = std::getenv("STRATA_MIRROR_STAGE_WG"); return v && std::atoi(v) > 0 ? std::atoi(v) : 64; }();
    constexpr int WG = 256, U = 4;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    const unsigned long long* dstp = (const unsigned long long*) (plan + ptr_off) + capx;
    const int32_t* counts = plan;
    const long long n16 = blob_bytes / 16, tail = blob_bytes - n16 * 16;
    strata::q_of(stream)->parallel_for<dpct_kernel_name<class mirror_stage_kernel_dk3>>(
        sycl::nd_range<1>(sycl::range<1>((size_t) nwg * WG), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int m = counts[2];
            const long long T = (long long) it.get_global_range(0);
            const long long g0 = (long long) it.get_global_id(0);
            for (int j = 0; j < m; ++j) {
                if (src[j] == 0) continue;   // STRATA_MIRROR_RING: already in the ring
                const sycl::uint4* s4 = (const sycl::uint4*) src[j];
                sycl::uint4* d4 = (sycl::uint4*) dstp[j];
                long long i = g0;
                for (; i + (U - 1) * T < n16; i += U * T) {
                    sycl::uint4 v[U];
#pragma unroll
                    for (int u = 0; u < U; ++u) v[u] = s4[i + u * T];
#pragma unroll
                    for (int u = 0; u < U; ++u) d4[i + u * T] = v[u];
                }
                for (; i < n16; i += T) d4[i] = s4[i];
                if (g0 < tail) ((uint8_t*) d4)[n16 * 16 + g0] = ((const uint8_t*) s4)[n16 * 16 + g0];
            }
        });
    check("mirror_stage");
}
void resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, int n_expert,
                   const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                   long long capx, uint32_t* skip, uint32_t ring, void* stream, uint32_t* plan_err) {
    const unsigned long long* mir = nullptr;   // SYCL port: the layer's slice of the host-mirror table, if any
    int32_t* use = nullptr;                    // the layer's slice of the device routing counts, if on
    if (g_mirror_table != nullptr && in_mirror_table(res_layer)) mir = g_mirror_table + (res_layer - g_mirror_res);
    if (g_usage != nullptr && in_mirror_table(res_layer)) use = g_usage + (res_layer - g_mirror_res);
    // STRATA_PLAN_MIRROR_FIRST=1 (opt-in; default off): the groups read over PCIe (host-mirrored experts) take the first
    // group indices of the plan, so the expert launches start their PCIe reads at once and the VRAM groups run beside
    // them; with first-occurrence order a mirrored group can come last and its PCIe read then runs alone at the end of
    // the launch.  Bitwise the same outputs (each entry's sums are its own; only the order of the groups changes).  B70
    // kqbench, 2560 x 640 IQ2_S / Q2_0, 4-token window, 48 layers in a 20 GB arena, one mirrored group per layer: 203.2 ->
    // 132.7 us per call (no mirrored group: 126.9); two: 290.1 -> 216.2 us.  Engine-neutral at 15202 slots (Flash-Next
    // IQ2_XS, 8K prompt, ~9.4K pairs mirrored): per-window cost T2/T3/T4 29.47/43.88/57.48 ms off, 29.61/44.15/58.00 on -
    // there the layers carry many mirrored groups and the launch is PCIe-bound whatever their order.
    static const bool mfirst_env = [] {
        const char* v = std::getenv("STRATA_PLAN_MIRROR_FIRST");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    const bool mfirst = mfirst_env && mir != nullptr;
    uint8_t* stg = mir != nullptr ? g_stage_base : nullptr;
    const long long sstr = g_stage_stride;
    unsigned long long* ssrc = stg != nullptr ? g_stage_src : nullptr;
    const int kb = mir != nullptr ? (int) (res_layer - g_mirror_res) : 0;
    uint8_t* rng = stg != nullptr && g_ring_on ? g_ring_base : nullptr;
    int32_t* rof = rng != nullptr ? g_rof + kb : nullptr;
    int32_t* rofg = g_rof;
    int32_t* rtg = g_rtag;
    uint8_t* rrf = g_rref;
    int32_t* rhd = g_rhand;
    const int rn = g_ring_n;
    uint32_t* mst = g_mstat != nullptr && mir != nullptr ? g_mstat + (size_t) (kb / n_expert) * 8 : nullptr;
    if (n_entries > kResidentPlanNarrow) {   // a 9..16-row window (kResidentPlanMax == 256 only in a wide build)
        if (kResidentPlanMax > kResidentPlanNarrow && n_entries <= kResidentPlanMax)
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{};

            strata::q_of(stream)
                ->parallel_for<
                    dpct_kernel_name<class resident_plan_kernel_db6b7b,
                                     dpct_kernel_scalar<kResidentPlanMax>>>(
                    sycl::nd_range<3>(sycl::range(1, 1, kResidentPlanMax),
                                      sycl::range(1, 1, kResidentPlanMax)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            resident_plan_kernel<kResidentPlanMax>(
                                ids, n_entries, k, res_layer, n_expert,
                                cache_base, slot_off, blob, plan, capx, skip,
                                ring, mir, plan_err, use, mfirst, stg, sstr, ssrc, rng, rof, rofg, rtg, rrf, rhd, rn, kb, mst);
                        });
        } else {
            std::fprintf(stderr, "resident_plan: %d entries, at most %d\n", n_entries, kResidentPlanMax);
            std::exit(1);
        }
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class resident_plan_kernel_db6b7a,
                                 dpct_kernel_scalar<kResidentPlanNarrow>>>(
                sycl::nd_range<3>(sycl::range(1, 1, kResidentPlanNarrow),
                                  sycl::range(1, 1, kResidentPlanNarrow)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        resident_plan_kernel<kResidentPlanNarrow>(
                            ids, n_entries, k, res_layer, n_expert, cache_base,
                            slot_off, blob, plan, capx, skip, ring, mir, plan_err, use, mfirst, stg, sstr, ssrc, rng, rof, rofg, rtg, rrf, rhd, rn, kb, mst);
                    });
    }
    check("resident_plan");
}
void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* stream) {
    const uint32_t spin_max = strata::spin_max(*strata::q_of(stream));
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class wait_flag_ge_or_kernel_2b2de3>>(
                sycl::nd_range<3>(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    wait_flag_ge_or_kernel(flag, value, skip, spin_max);
                });
    }
    check("wait_flag_ge_or");
}
void copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip, uint32_t value,
                                 void* stream) {
    if (n <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class copy_i32_unless_kernel_d9d761>>(
                sycl::nd_range<3>(sycl::range(1, 1, 128),
                                  sycl::range(1, 1, 128)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    copy_i32_unless_kernel(dst, (const volatile int32_t *)src,
                                           (int)n, skip, value);
                });
    }
    check("copy_i32_from_mapped_unless");
}
void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream) {
    if (n <= 0) return;
    const long long n4 = n / 4;
    const bool pair = strata::a770_fast();
    const int blocks = (int) ((n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class copy_or_zero_kernel_5ef081>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    copy_or_zero_kernel((sycl::float4 *)dst,
                                        (const sycl::float4 *)src, n4,
                                        skip, value, pair);
                });
    }
    check("copy_or_zero_from_mapped");
}

namespace {
__dpct_inline__ void wait_copy_i32_kernel(const volatile uint32_t *flag,
                                          uint32_t value,
                                          int32_t *__restrict__ dst,
                                          const volatile int32_t *src, int n,
                                          uint32_t spin_max) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (item_ct1.get_local_id(2) == 0) {
        for (uint32_t spin = 0; spin < spin_max && strata::sys_load(flag) < value; ++spin) strata_spin_pause();
        /*
        DPCT1078: Consider replacing memory_order::acq_rel with
        memory_order::seq_cst for correctness if strong memory order
        restrictions are needed.
        */
        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::system);
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) dst[i] = src[i];
}
}  // namespace
void wait_copy_i32_from_mapped(const uint32_t* flag, uint32_t value, int32_t* dst, const int32_t* src, int64_t n,
                               void* stream) {
    const uint32_t spin_max = strata::spin_max(*strata::q_of(stream));
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class wait_copy_i32_kernel_6aeabe>>(
                sycl::nd_range<3>(sycl::range(1, 1, 128),
                                  sycl::range(1, 1, 128)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    wait_copy_i32_kernel(flag, value, dst, src, (int)n, spin_max);
                });
    }
    check("wait_copy_i32_from_mapped");
}

void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream) {
    const uint32_t spin_max = strata::spin_max(*strata::q_of(stream));
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class wait_flag_ge_kernel_d7debf>>(
                sycl::nd_range<3>(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    wait_flag_ge_kernel(flag, value, spin_max);
                });
    }
    check("wait_flag_ge");
}

void wait_flag_ge_stamped(const uint32_t* flag, uint32_t value, unsigned long long* t_in, unsigned long long* t_out,
                          void* stream) {
    const uint32_t spin_max = strata::spin_max(*strata::q_of(stream));
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class wait_flag_ge_stamped_kernel_29374d>>(
                sycl::nd_range<3>(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    wait_flag_ge_stamped_kernel(flag, value, t_in, t_out, spin_max);
                });
    }
    check("wait_flag_ge_stamped");
}

void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class embedding_gather_dev_kernel_600016>>(
                sycl::nd_range<3>(sycl::range(1, (unsigned)n_tok,
                                              (unsigned)((n + 255) / 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    embedding_gather_dev_kernel(
                        codes, scales, offsets, tokens, n, code_bits, code_bias,
                        group_elems, row_codes, row_groups, out);
                });
    }
    check("embedding_gather_dev");
}

void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class broadcast_streams_kernel_aa4f6f>>(
                sycl::nd_range<3>(
                    sycl::range(1, (unsigned)n_tok,
                                (unsigned)((n_embd * hc + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    broadcast_streams_kernel(x, R, n_embd, hc);
                });
    }
    check("broadcast_streams");
}

void hc_mean_streams(const float* R, float* out, int64_t n_embd, int hc, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class hc_mean_streams_kernel_c4fefa>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((n_embd + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    hc_mean_streams_kernel(R, out, n_embd, hc);
                });
    }
    check("hc_mean_streams");
}

void route_marginal_stats(const int32_t* ids, int n, int k, unsigned long long* acc, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class route_marginal_stats_kernel_7646d6>>(
                sycl::nd_range<3>(sycl::range(1, 1, 32), sycl::range(1, 1, 32)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    route_marginal_stats_kernel(ids, n, k, acc);
                });
    }
    check("route_marginal_stats");
}

void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream) {
    const unsigned blocks = (unsigned) ((n + 255) / 256 < 64 ? (n + 255) / 256 : 64);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class copy_indexed_kernel_b12e23>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    copy_indexed_kernel(dst, src, stride, index, n);
                });
    }
    check("copy_indexed");
}

// Opt-in GPU stage timestamps. DG2 has no comparable device-wide clock, so
// read a live host USM clock uncached (the fork's gpu_stamp implementation).
// The writer is joined before runtime teardown and exists only in profile mode.
namespace {
uint64_t profile_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct ProfileHostClock {
    sycl::context context;
    uint64_t* word;
    std::atomic<bool> stop{false};
    std::thread writer;
    explicit ProfileHostClock(const sycl::queue& q) : context(q.get_context()),
        word(sycl::malloc_host<uint64_t>(8, context)) {
        if (!word) throw std::runtime_error("gpu_stamp: host clock allocation failed");
        __atomic_store_n(word, profile_now_ns(), __ATOMIC_RELEASE);
        writer = std::thread([this] {
            while (!stop.load(std::memory_order_relaxed)) {
                __atomic_store_n(word, profile_now_ns(), __ATOMIC_RELEASE);
                for (int i = 0; i < 8; ++i) __builtin_ia32_pause();
            }
        });
    }
    ~ProfileHostClock() {
        stop.store(true, std::memory_order_relaxed);
        if (writer.joinable()) writer.join();
        // Keep the word alive until context teardown, as in the fork: a graph
        // may still retain its pointer when process-level statics are destroyed.
    }
};
uint64_t* profile_clock_word(const sycl::queue& q) {
    static std::mutex mu;
    static std::vector<std::unique_ptr<ProfileHostClock>> clocks;
    std::lock_guard<std::mutex> lock(mu);
    for (const auto& clock : clocks) if (clock->context == q.get_context()) return clock->word;
    clocks.push_back(std::make_unique<ProfileHostClock>(q));
    return clocks.back()->word;
}
}

namespace {
__dpct_inline__ void noop_kernel() {}
}  // namespace
void gpu_noop(void *stream) {
    auto exp_props = sycl::ext::oneapi::experimental::properties{};

  ((sycl::queue*)strata::q_of(stream))->parallel_for<dpct_kernel_name<class noop_kernel_2079ed>>(
    sycl::nd_range<3>(sycl::range(1, 1, 32), sycl::range(1, 1, 32)), 
exp_props,     [=](sycl::nd_item<3> item_ct1) {
        noop_kernel();
    }); }

void gpu_stamp(unsigned long long* buf, int i, void* stream) {
    auto& q = *strata::q_of(stream);
    uint64_t* word = profile_clock_word(q);
    q.single_task([=] {
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
        buf[i] = strata::sys_load_mapped(word);
    });
}

namespace {
__dpct_inline__ void
copy_rows_strided_kernel(sycl::float4 *__restrict__ dst,
                         const sycl::float4 *__restrict__ src, long long rows,
                         int w4, int src_w4) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long total = rows * w4;
    for (long long i =
             (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < total; i += (long long)item_ct1.get_group_range(2) *
                         item_ct1.get_local_range(2)) {
        const long long r = i / w4, c = i - r * w4;
        dst[i] = src[r * src_w4 + c];
    }
}
}  // namespace
void copy_rows_strided(float* dst, const float* src, int64_t rows, int64_t w, int64_t src_w, void* stream) {
    if (rows <= 0 || w <= 0) return;
    if ((w & 3) || (src_w & 3) || src_w < w || ((uintptr_t) dst & 15) || ((uintptr_t) src & 15)) {
        std::fprintf(stderr, "copy_rows_strided: widths must be multiples of 4 floats (src_w >= w), pointers 16-byte "
                             "aligned\n");
        std::exit(1);
    }
    const long long total = (long long) rows * (w / 4);
    const unsigned blocks = (unsigned) ((total + 255) / 256 < 256 ? (total + 255) / 256 : 256);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {
                auto w_ct3 = (int)(w / 4);
                auto src_w_ct4 = (int)(src_w / 4);

                cgh.parallel_for<
                    dpct_kernel_name<class copy_rows_strided_kernel_ec8c39>>(
                    sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                          sycl::range(1, 1, 256),
                                      sycl::range(1, 1, 256)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        copy_rows_strided_kernel((sycl::float4 *)dst,
                                                 (const sycl::float4 *)src,
                                                 rows, w_ct3, src_w_ct4);
                    });
            });
    }
    check("copy_rows_strided");
}

// ---- programmatic dependent launch (pdl.hpp): SYCL has no counterpart (griddepcontrol / the PDL launch attribute are
// sm_90 CUDA features); the verify window launches every kernel the ordinary way.
bool& pdl_scope() {
    static thread_local bool on = false;
    return on;
}

bool pdl_supported() { return false; }

#if !defined(__HIPCC__)
bool pdl_launch_ok(const void *kernel, dpct::queue_ptr stream) {
    (void) kernel; (void) stream;
    return false;
}
#endif

}  // namespace strata::kernels
