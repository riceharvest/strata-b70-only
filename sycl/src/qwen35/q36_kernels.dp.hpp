// q36_kernels.cuh - decode kernels of the Qwen3.6 engine (q36_engine.cu), written to give the SAME BITS as the shared
// kernels they replace (native_small_mmvq_kernel, k_moe_out, k_add_rms, k_group, native_quantize_q8_1) with a layout that
// keeps more bytes in flight and fewer launches.  Every kernel here is checked bit for bit against the old one on the
// real activations (STRATA_Q36_VERIFY=1, --no-graph).
#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace q36k {

struct QB {   // a q8_1 activation block (what native_quantize_q8_1 writes)
    sycl::half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(QB) == 36, "q8_1 block");

// (M2: the xor butterfly as sycl::permute_group_by_xor; dpct's masked form was a generic GroupNonUniformShuffle with a computed id.
// The same pairs in the same order: the same bits.)
__dpct_inline__ float wsum(float x) {
    const auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += sycl::permute_group_by_xor(sg, x, o);
    return x;
}
__dpct_inline__ float wmax(float x) {
    const auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x = sycl::fmax(x, sycl::permute_group_by_xor(sg, x, o));
    return x;
}
__dpct_inline__ float sigm(float v) {
    return 1.0f / (1.0f + sycl::native::exp(-v));
}

// The shared CUDA kernels these ones replace (native_mmvq.cu, native_router.cu) are compiled with --use_fast_math (CMakeLists.txt);
// this file is not, so on CUDA the division and the exponential they use are spelled out as the instructions that flag selects
// (div.approx.ftz, rcp.approx.ftz, ex2.approx.ftz via __expf).  HIP builds have no such flag: plain operators.
#if defined(SYCL_LANGUAGE_VERSION) && !defined(__HIPCC__) &&                   \
    !defined(STRATA_Q36_PORT) // (the SYCL port: precise division, as its shared
                              // quantizer)
__device__ __forceinline__ float fdiv_fm(float a, float b) {
    float r;
    asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float frcp_fm(float a) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}
__device__ __forceinline__ float fexp_fm(float a) { return __expf(a); }
#else
__dpct_inline__ float fdiv_fm(float a, float b) { return a / b; }
__dpct_inline__ float frcp_fm(float a) { return 1.0f / a; }
__dpct_inline__ float fexp_fm(float a) { return sycl::native::exp(a); }
#endif

/// One lane's element `v` of a 32-element q8_1 block (the warp is the block): native_quantize_q8_1_kernel's body.
/// `y` points at the row's blocks, `i` is the element index within the row.
__dpct_inline__ void q8_store(QB *__restrict__ y, int i, float v, int lane) {
    const float amax = wmax(sycl::fabs(v));
    const float sum = wsum(v);
    const float d = strata::kernels::q8_1_finite(fdiv_fm(amax, 127.0f));
    int8_t q = 0;   // strata::kernels::q8_1_quant with the division of the fast-math TU
    if (amax != 0.0f) {
        const float qf = sycl::round(fdiv_fm(v, d));
        q = (int8_t) (qf > 127.0f ? 127.0f : qf < -127.0f ? -127.0f : qf);
    }
    y[i >> 5].qs[lane] = q;
    if (lane == 0) y[i >> 5].ds = strata::kernels::q8_1_ds(d, sum);
}

/// A Q8_0 row (GGUF blocks of 34 bytes, or the qs / d planes with PK) against NC activation columns, with the arithmetic of
/// native_small_mmvq_kernel / native_mmvq_multi_kernel's exact layout: the 4 warps of the old block are the 4 partial sums
/// P_w of one lane here (block kbx = 8 w + lane / 4 + 32 j, ints 2 (lane % 4) and +1), added in warp order, then the xor
/// tree.  One warp per row, every load of a lane issued before the first dot (NJ = ceil(nb / 32) terms per partial).
/// res[c] = the row's value for column c in every lane.
template <int NC, int NJ, bool PK>
/*
DPCT1110:5: The total declared local variable size in device function q8_row_dot
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void q8_row_dot(const uint8_t *__restrict__ w,
                                const sycl::half *__restrict__ dpl,
                                const QB *__restrict__ x, int row, int nb,
                                int lane, float (&res)[NC]) {
    const int kqs = 2 * (lane & 3);
    const int kb0 = lane >> 2;
    int v0[4][NJ], v1[4][NJ];
    float d0[4][NJ];
    bool ok[4][NJ];
#pragma unroll
    for (int p = 0; p < 4; ++p)
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int kbx = kb0 + 8 * p + 32 * j;
            ok[p][j] = kbx < nb;
            if (!ok[p][j]) continue;
            const size_t blk = (size_t) row * nb + kbx;
            if (PK) {
                const int* q = reinterpret_cast<const int*>(w + blk * 32) + kqs;
                v0[p][j] = q[0];
                v1[p][j] = q[1];
                d0[p][j] =
                    sycl::vec<sycl::half, 1>(dpl[blk])
                        .convert<float, sycl::rounding_mode::automatic>()[0];
            } else {
                const uint16_t* b = reinterpret_cast<const uint16_t*>(w + blk * 34);
                d0[p][j] =
                    sycl::vec<sycl::half, 1>(
                        sycl::bit_cast<sycl::half, unsigned short>(b[0]))
                        .convert<float, sycl::rounding_mode::automatic>()[0];
                v0[p][j] = (int) b[1 + 2 * kqs] | ((int) b[2 + 2 * kqs] << 16);
                v1[p][j] = (int) b[3 + 2 * kqs] | ((int) b[4 + 2 * kqs] << 16);
            }
        }
    float P[NC][4];
#pragma unroll
    for (int c = 0; c < NC; ++c)
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            float t = 0.f;
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                if (!ok[p][j]) continue;
                const QB* xb = x + (size_t) c * nb + kb0 + 8 * p + 32 * j;
                const int* u = reinterpret_cast<const int*>(xb->qs);
                int sumi = 0;
                sumi = STRATA_DP4A(v0[p][j], u[kqs], sumi);
                sumi = STRATA_DP4A(v1[p][j], u[kqs + 1], sumi);
                const float d1 = xb->ds[0];
                t += d0[p][j] * d1 * float(sumi);
            }
            P[c][p] = t;
        }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float t = P[c][0];
        t += P[c][1];
        t += P[c][2];
        t += P[c][3];
        res[c] = wsum(t);
    }
}

/// y[c][row] = W[row] . x[c], NC columns, 8 rows (warps) per 256-thread block.
template <int NC, int NJ, bool PK>
__dpct_inline__ void
k_q8_rows(const uint8_t *__restrict__ w, const sycl::half *__restrict__ dpl,
          const QB *__restrict__ x, float *__restrict__ y, int nb, int n_out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    const int stride =
        item_ct1.get_group_range(2) *
        (item_ct1.get_local_range(2) >>
         5); // a grid-stride loop over the rows (see launch_q8_rows)
    for (int row =
             item_ct1.get_group(2) * (item_ct1.get_local_range(2) >> 5) + warp;
         row < n_out; row += stride) {
        float res[NC];
        q8_row_dot<NC, NJ, PK>(w, dpl, x, row, nb, lane, res);
        if (lane == 0)
#pragma unroll
            for (int c = 0; c < NC; ++c) y[(size_t) c * n_out + row] = res[c];
    }
}

template <int NC, int NJ, bool PK>
inline void launch_q8_rows_t(const void *w, const sycl::half *dpl,
                             const void *x, float *y, int nb, int n_out,
                             int warps, int max_warps, dpct::queue_ptr s) {
    const int rows = n_out < max_warps ? n_out : max_warps;
    /*
    DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};
        

        s->parallel_for<
            dpct_kernel_name<class k_q8_rows_3e89a3, dpct_kernel_scalar<NC>,
                             dpct_kernel_scalar<NJ>, dpct_kernel_scalar<PK>>>(
            sycl::nd_range<3>(
                sycl::range(1, 1, (unsigned)((rows + warps - 1) / warps)) *
                    sycl::range(1, 1, warps * 32),
                sycl::range(1, 1, warps * 32)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                k_q8_rows<NC, NJ, PK>(static_cast<const uint8_t *>(w), dpl,
                                      static_cast<const QB *>(x), y, nb, n_out);
            });
    }
}

template <int NC, int NJ, bool PK>
/// At most 1024 warps in the grid, a grid-stride loop over the rows beyond: a
/// 2048-row product took 21 us on gfx1201 as 2048 warps and 5.2 us as 1024 (a
/// grid above 32768 threads is several times slower there; the same rows, the
/// same sums).
inline void launch_q8_rows(const void *w, const sycl::half *dpl, const void *x,
                           float *y, int nb, int n_out, dpct::queue_ptr s) {
    launch_q8_rows_t<NC, NJ, PK>(w, dpl, x, y, nb, n_out, 8, 1024, s);
}

/// The shared expert's gate and up products and SwiGLU + q8_1 in one launch: 32 rows (one q8_1 block of h) per 1024-thread
/// block, warp w computing gate row and up row 32 b + w, then warp 0 applying native_swiglu_quantize_q8_1_kernel's body to
/// the 32 pairs.  Column c of the h image is at hq + c * (n_out / 32).  n_out % 32 == 0.
template <int NC, int NJ, bool PK>
__dpct_inline__ void
k_sh_gu(const uint8_t *__restrict__ wg, const sycl::half *__restrict__ dg,
        const uint8_t *__restrict__ wu, const sycl::half *__restrict__ du,
        const QB *__restrict__ x, QB *__restrict__ hq, int nb, int n_out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &g = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[NC][32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &u =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[NC][32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    const int row = item_ct1.get_group(2) * 32 + warp;
    float rg[NC], ru[NC];
    q8_row_dot<NC, NJ, PK>(wg, dg, x, row, nb, lane, rg);
    q8_row_dot<NC, NJ, PK>(wu, du, x, row, nb, lane, ru);
    if (lane == 0)
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            g[c][warp] = rg[c];
            u[c][warp] = ru[c];
        }
    /*
    DPCT1065:47: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (warp == 0) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float gi = g[c][lane];
            /*
            DPCT1013:46: The rounding mode could not be specified and the
            generated code may have different accuracy than the original code.
            Verify the correctness. SYCL math built-in function rounding mode is
            aligned with OpenCL C 1.2 standard.
            */
            const float xi = gi / (1.0f + sycl::native::exp(-gi)) * u[c][lane];
            q8_store(hq + (size_t)c * (n_out >> 5),
                     item_ct1.get_group(2) * 32 + lane, xi, lane);
        }
    }
}

/// k_add_rms (x += y; h = rms(x) * w, one block of 256 per row) and the q8_1 image of h in the same launch.
__dpct_inline__ void k_add_rms_q(float *x, const float *y, const float *w,
                                 float *h, QB *hq, int n, float eps) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sh = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    float *xr = x + (size_t)item_ct1.get_group(2) * n;
    const float *yr = y + (size_t)item_ct1.get_group(2) * n;
    float *hr = h + (size_t)item_ct1.get_group(2) * n;
    QB *qr = hq + (size_t)item_ct1.get_group(2) * (n >> 5);
    float s = 0.f;
    for (int i = item_ct1.get_local_id(2); i < n; i += 256) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        s += v * v;
    }
    sh[item_ct1.get_local_id(2)] = s;
    /*
    DPCT1065:48: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#pragma unroll
    for (int o = 128; o > 0; o >>= 1) {
        if ((int)item_ct1.get_local_id(2) < o) sh[item_ct1.get_local_id(2)] +=
            sh[item_ct1.get_local_id(2) + o];
        /*
        DPCT1118:7: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:49: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    const float r = sycl::rsqrt(sh[0] / (float)n + eps);
    for (int i = item_ct1.get_local_id(2); i < n; i += 256) {
        /*
        DPCT1013:50: The rounding mode could not be specified and the generated
        code may have different accuracy than the original code. Verify the
        correctness. SYCL math built-in function rounding mode is aligned with
        OpenCL C 1.2 standard.
        */
        const float hv = xr[i] * r * w[i]; // no FMA with the quantizer's sums:
                                           // the old h was stored first
        hr[i] = hv;
        q8_store(
            qr, i, ((volatile float *)hr)[i],
            item_ct1.get_local_id(2) &
                31); // the stored value, as the standalone quantizer reads it
    }
}

/// M2: k_add_rms_q with 1024 threads per row (one work-group, sub-group 32) and the row in SLM: the same bits.  The squares are
/// summed per old thread (element t, t + 256, ... in order), the old 256-wide tree's first three levels in SLM and its last five as
/// the xor butterfly of sub-group 0 (lane 0 adds the same pairs in the same order), h = (x * r) * w, and q8_store over the same
/// 32-element blocks with the same lanes.  n % 32 == 0, n <= NMAX.  (The old kernel: ~30 us a row on the B70, 127 calls a token.)
// M4: every global load of the row (x, y and the norm weight w) is issued up front into registers, before the first store:
// the old loops stored x before the next element's loads (x may alias y / w for the compiler, so they could not move) and
// loaded w[i] inside the quantizer loop, one dependent global load per element after the barriers.  The same values in the
// same order: the same bits.
template <int NMAX>
__dpct_inline__ void k_add_rms_q1k(float *x, const float *y, const float *w, float *h, QB *hq, int n, float eps) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto wg = sycl::ext::oneapi::this_work_item::get_work_group<3>();
    auto &vv = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[NMAX]>(wg);
    auto &sh = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(wg);
    constexpr int EPT = (NMAX + 1023) / 1024;   // elements a thread (n <= NMAX)
    const int t = (int) it.get_local_id(2);
    const size_t r = it.get_group(2);
    float *xr = x + r * n;
    const float *yr = y + r * n;
    float *hr = h + r * n;
    QB *qr = hq + r * (n >> 5);
    float xv[EPT], wv[EPT];
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        const int i = t + e * 1024;
        xv[e] = 0.f;
        wv[e] = 0.f;
        if (i < n) {
            xv[e] = xr[i] + yr[i];
            wv[e] = w[i];
        }
    }
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        const int i = t + e * 1024;
        if (i < n) {
            xr[i] = xv[e];
            vv[i] = xv[e];
        }
    }
    it.barrier(sycl::access::fence_space::local_space);
    if (t < 256) {
        float s = 0.f;
        for (int i = t; i < n; i += 256) s += vv[i] * vv[i];
        sh[t] = s;
    }
    it.barrier(sycl::access::fence_space::local_space);
    if (t < 128) sh[t] += sh[t + 128];
    it.barrier(sycl::access::fence_space::local_space);
    if (t < 64) sh[t] += sh[t + 64];
    it.barrier(sycl::access::fence_space::local_space);
    if (t < 32) {
        float v = sh[t] + sh[t + 32];
        const auto sg = it.get_sub_group();
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
        if (t == 0) sh[0] = v;
    }
    it.barrier(sycl::access::fence_space::local_space);
    const float rr = sycl::rsqrt(sh[0] / (float) n + eps);
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        const int i = t + e * 1024;
        if (i < n) {   // (n % 32 == 0: a sub-group's 32 elements are all in or all out, so q8_store's shuffles stay converged)
            const float hv = vv[i] * rr * wv[e];
            hr[i] = hv;
            // no FMA with the quantizer's sums: the stored value read back, as k_add_rms_q does - from the SLM copy (this
            // thread's own element) instead of global memory, so the loop does not wait on a global load per element
            vv[i] = hv;
            q8_store(qr, i, ((volatile float *) &vv[0])[i], t & 31);
        }
    }
}
/// x += y; h = rms(x) * w; hq = q8_1(h) for T rows: k_add_rms_q1k where it applies, else k_add_rms_q (bitwise the same).
inline void launch_add_rms_q(float *x, const float *y, const float *w, float *h, QB *hq, int n, int T, float eps,
                             sycl::queue *q) {
    static const bool k1 = [] { const char *e = std::getenv("STRATA_Q36_RMS1K"); return !(e && e[0] == '0'); }();   // 0: off
    if (k1 && n % 32 == 0 && n <= 8192) {
        q->parallel_for<class q36k_add_rms_q1k>(
            sycl::nd_range<3>(sycl::range<3>(1, 1, (size_t) T * 1024), sycl::range<3>(1, 1, 1024)),
            [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] { k_add_rms_q1k<8192>(x, y, w, h, hq, n, eps); });
        return;
    }
    q->parallel_for<class q36k_add_rms_q256>(
        sycl::nd_range<3>(sycl::range<3>(1, 1, (size_t) T * 256), sycl::range<3>(1, 1, 256)),
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] { k_add_rms_q(x, y, w, h, hq, n, eps); });
}

/// k_moe_out for T <= 8 rows (decode, verify windows) with n <= 2048 and n % 256 == 0: 1024 threads per row instead of 256
/// (the elements' loads in parallel), the squares summed per old thread (element t, t + 256, ... in order) and the
/// 256-wide tree as before: the same bits.  x += sum_k w_k part_k (+ gate * shared); h = rms(x) * wn; with hq the q8_1
/// image of h as well (the next layer's input).
__dpct_inline__ void k_moe_out_small(float *x, const float *parts,
                                     const int32_t *prow, const float *wts,
                                     int K_, const float *shared,
                                     const float *sgate, int sg_stride,
                                     const float *wn, float *h, int n,
                                     float eps, QB *hq = nullptr) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &vv = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[2048]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sh = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &pk =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<const float *[16]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_group(2);
    float* xr = x + (size_t) t * n;
    if (item_ct1.get_local_id(2) < K_) pk[item_ct1.get_local_id(2)] =
        parts + (size_t)(prow ? prow[t * K_ + item_ct1.get_local_id(2)]
                              : t * K_ + item_ct1.get_local_id(2)) *
                    n;
    /*
    DPCT1065:51: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const float* wr = wts + (size_t) t * K_;
    const float gs = sigm(sgate[(size_t) t * sg_stride]);
    for (int i = item_ct1.get_local_id(2); i < n; i += 1024) {
        float acc = wr[0] * pk[0][i];
#pragma unroll
        for (int k = 1; k < K_; ++k)
            acc = sycl::fma((float)(wr[k]), (float)(pk[k][i]), acc);
        acc += gs * shared[(size_t) t * n + i];
        const float v = xr[i] + acc;
        xr[i] = v;
        vv[i] = v;
    }
    if (wn == nullptr) return;
    /*
    DPCT1065:52: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) < 256) {
        float s = 0.f;
#pragma unroll
        for (int i = item_ct1.get_local_id(2); i < n; i += 256) s +=
            vv[i] * vv[i];
        sh[item_ct1.get_local_id(2)] = s;
    }
    /*
    DPCT1065:53: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#pragma unroll
    for (int s = 128; s > 0; s >>= 1) {
        if ((int)item_ct1.get_local_id(2) < s) sh[item_ct1.get_local_id(2)] +=
            sh[item_ct1.get_local_id(2) + s];
        /*
        DPCT1118:8: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:54: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    const float r = sycl::rsqrt(sh[0] / (float)n + eps);
    for (int i = item_ct1.get_local_id(2); i < n; i += 1024) {
        /*
        DPCT1013:55: The rounding mode could not be specified and the generated
        code may have different accuracy than the original code. Verify the
        correctness. SYCL math built-in function rounding mode is aligned with
        OpenCL C 1.2 standard.
        */
        const float hv = vv[i] * r * wn[i];
        h[(size_t) t * n + i] = hv;
        if (hq) q8_store(hq + (size_t)t * (n >> 5), i,
                         ((volatile float *)h)[(size_t)t * n + i],
                         item_ct1.get_local_id(2) & 31);
    }
}

/// k_group / k_group_hy for n_ent <= 64 entries: one block of 64 threads ranks each entry by (expert id, entry) in
/// parallel instead of one thread walking all the experts; the same group / entry arrays.  HY: only the entries whose
/// expert has a VRAM slot (slot_of >= 0), the group's blob at blobs + slot * blob.
template <bool HY>
__dpct_inline__ void
k_group_small(const int32_t *ids, int n_ent, int K_, const int32_t *slot_of,
              const uint8_t *blobs, unsigned long long blob,
              unsigned long long *grp_ptr, int32_t *grp_start,
              int32_t *n_groups, int32_t *ent_dst, int32_t *ent_tok) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sid = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[64]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int e = item_ct1.get_local_id(2);
    const int id = e < n_ent ? ids[e] : 0;
    const bool valid = e < n_ent && (!HY || slot_of[id] >= 0);
    int rank = 0;
    if (valid) {
        for (int e2 = 0; e2 < n_ent; ++e2) {
            const int id2 = ids[e2];
            if (HY && slot_of[id2] < 0) continue;
            if (id2 < id || (id2 == id && e2 < e)) ++rank;
        }
        ent_dst[rank] = e;
        ent_tok[rank] = e / K_;
        sid[rank] = id;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    int nv = 0;
#pragma unroll
    for (int e2 = 0; e2 < n_ent; ++e2) nv +=
        (!HY || slot_of[ids[e2]] >= 0) ? 1 : 0;
    if (valid && (rank == 0 || sid[rank - 1] != sid[rank])) {
        int g = 0;
#pragma unroll
        for (int q = 1; q < rank; ++q) g += sid[q - 1] != sid[q] ? 1 : 0;
        g += rank > 0 ? 1 : 0;
        grp_ptr[g] = (unsigned long long) (blobs + (size_t) (HY ? slot_of[id] : id) * blob);
        grp_start[g] = rank;
    }
    if (e == 0) {
        int G = nv > 0 ? 1 : 0;
#pragma unroll
        for (int q = 1; q < nv; ++q) G += sid[q - 1] != sid[q] ? 1 : 0;
        grp_start[G] = nv;
        *n_groups = G;
    }
}

/// route_topk<256, 8> (native_router.cu, copied: the same expressions) for n_tok <= 8 tokens (one warp each) and, in the same
/// launch, k_group_small's grouping of the picks.  block (32, 8).  HY as k_group_small.
template <bool HY>
/*
DPCT1110:9: The total declared local variable size in device function
k_route_group exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void k_route_group(
    const float *__restrict__ logits, int32_t *__restrict__ ids,
    float *__restrict__ weights, int n_tok, int ld, const int32_t *slot_of,
    const uint8_t *blobs, unsigned long long blob, unsigned long long *grp_ptr,
    int32_t *grp_start, int32_t *n_groups, int32_t *ent_dst, int32_t *ent_tok) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int NE = 256, K = 8, V = NE / 32;
    auto &sids = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[64]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sid = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[64]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int tk = item_ct1.get_local_id(1);
    const int lane = item_ct1.get_local_id(2);
    if (tk < n_tok) {
        const float* lg = logits + (size_t) tk * ld;
        float values[V];
#pragma unroll
        for (int i = 0; i < V; ++i) values[i] = lg[lane + i * 32];
        float maximum = -INFINITY;
#pragma unroll
        for (int i = 0; i < V; ++i) maximum = sycl::max(maximum, values[i]);
        maximum = wmax(maximum);
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < V; ++i) {
            values[i] = fexp_fm(values[i] - maximum);
            sum += values[i];
        }
        const float reciprocal = frcp_fm(wsum(sum));
#pragma unroll
        for (int i = 0; i < V; ++i) {
            values[i] *= reciprocal;
            if (sycl::isnan(values[i])) values[i] = -FLT_MAX;
        }
        float selected = 0.0f, selected_sum = 0.0f;
        for (int rank = 0; rank < K; ++rank) {
            float best = values[0];
            int expert = lane;
#pragma unroll
            for (int i = 1; i < V; ++i) {
                if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
            }
#pragma unroll
            for (int mask = 16; mask; mask >>= 1) {
                /*
                DPCT1108:10: '__shfl_xor_sync' was migrated with the
                experimental feature masked sub_group function which may not be
                supported by all compilers or runtimes. You may need to adjust
                the code.
                */
                const float other =
                    dpct::experimental::permute_sub_group_by_xor(
                        0xffffffffu,
                        sycl::ext::oneapi::this_work_item::get_sub_group(),
                        best, mask);
                /*
                DPCT1108:11: '__shfl_xor_sync' was migrated with the
                experimental feature masked sub_group function which may not be
                supported by all compilers or runtimes. You may need to adjust
                the code.
                */
                const int other_id =
                    dpct::experimental::permute_sub_group_by_xor(
                        0xffffffffu,
                        sycl::ext::oneapi::this_work_item::get_sub_group(),
                        expert, mask);
                if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
            }
            if ((expert & 31) == lane) {
                values[expert / 32] = -INFINITY;
                ids[(size_t) tk * K + rank] = expert;
                sids[tk * K + rank] = expert;
                selected_sum += best;
            }
            if (rank == lane) selected = best;
        }
        selected_sum = sycl::max(wsum(selected_sum), 6.103515625e-5f);
        const float inverse_selected_sum = frcp_fm(selected_sum);
        if (lane < K) weights[(size_t) tk * K + lane] = selected * inverse_selected_sum;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const int n_ent = n_tok * K;
    const int e = tk * 32 + lane;   // 0 .. 255: the entries are the first n_ent threads
    const int id = e < n_ent ? sids[e] : 0;
    const bool valid = e < n_ent && (!HY || slot_of[id] >= 0);
    int rank = 0;
    if (valid) {
        for (int e2 = 0; e2 < n_ent; ++e2) {
            const int id2 = sids[e2];
            if (HY && slot_of[id2] < 0) continue;
            if (id2 < id || (id2 == id && e2 < e)) ++rank;
        }
        ent_dst[rank] = e;
        ent_tok[rank] = e / K;
        sid[rank] = id;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    int nv = 0;
#pragma unroll
    for (int e2 = 0; e2 < n_ent; ++e2) nv +=
        (!HY || slot_of[sids[e2]] >= 0) ? 1 : 0;
    if (valid && (rank == 0 || sid[rank - 1] != sid[rank])) {
        int g = 0;
#pragma unroll
        for (int q = 1; q < rank; ++q) g += sid[q - 1] != sid[q] ? 1 : 0;
        g += rank > 0 ? 1 : 0;
        grp_ptr[g] = (unsigned long long) (blobs + (size_t) (HY ? slot_of[id] : id) * blob);
        grp_start[g] = rank;
    }
    if (e == 0) {
        int G = nv > 0 ? 1 : 0;
#pragma unroll
        for (int q = 1; q < nv; ++q) G += sid[q - 1] != sid[q] ? 1 : 0;
        grp_start[G] = nv;
        *n_groups = G;
    }
}

}  // namespace q36k
