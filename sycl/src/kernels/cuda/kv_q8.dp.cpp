// src/kernels/cuda/kv_q8.cu - see include/strata/kernels/kv_q8.hpp.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
}

void validate(const QsaShapes& s, const char* what) {
    if (s.head_dim % KV_Q8_GROUP != 0 || s.n_head_kv <= 0 || s.page_size <= 0) {
        std::fprintf(stderr, "kv_q8: %s: head_dim %lld must be a multiple of %d\n", what, (long long) s.head_dim,
                     KV_Q8_GROUP);
        std::exit(1);
    }
}

// STRATA_KV_APPEND_PACK (default on, 0 off): a streamed KV cache's host copy written eight codes to a store (the same
// bytes; 196K decode on one B70: the append took 20.8 us a call with one-byte stores over PCIe)
inline bool kv_append_pack_host() {
    static const bool on = [] { const char* v = std::getenv("STRATA_KV_APPEND_PACK"); return v == nullptr || std::atoi(v) != 0; }();
    return on;
}

// One block = one 64-value group of one KV head of K (blockIdx.z = 0) or V (1); 64 threads, one value each.
// MULTI (S26 STRATA_LFUSE): blockIdx.z = 2 * token + (K / V); token j reads step + j * step_stride and its rows at
// kcur / vcur + j * cur_stride - every token's code is the single launch's
template <bool MULTI = false>
__dpct_inline__ void kv_append_q8_kernel(
    int8_t *__restrict__ k_q, int8_t *__restrict__ v_q,
    uint16_t *__restrict__ k_scale, uint16_t *__restrict__ v_scale,
    const int32_t *__restrict__ table, const int32_t *__restrict__ step,
    const float *__restrict__ kcur, const float *__restrict__ vcur,
    int kv_heads, int head_dim, int page_size, KvHostPools host,
    int step_stride = 0, int cur_stride = 0, bool pack_host = false) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if constexpr (MULTI) {
        const int j = item_ct1.get_group(0) >> 1;
        step += (size_t) j * step_stride; kcur += (size_t) j * cur_stride; vcur += (size_t) j * cur_stride;
    }
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    const long long pos = (long long)*(step + kStepPos);
    const int h = item_ct1.get_group(2), g = item_ct1.get_group(1),
              t = item_ct1.get_local_id(2);
    const bool is_v =
        MULTI ? (item_ct1.get_group(0) & 1) == 1 : item_ct1.get_group(0) == 1;
    const int groups = head_dim / KV_Q8_GROUP;
    const float x = (is_v ? vcur : kcur)[h * head_dim + g * KV_Q8_GROUP + t];
    // max |x| over the 64 values: two warps, then combine through shared memory in a fixed order
    float a = sycl::fabs(x);
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        a = sycl::fmax(a, sycl::permute_group_by_xor(sg, a, o));
    auto &warp_max =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[2]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if ((t & 31) == 0) warp_max[t >> 5] = a;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const float amax = sycl::fmax(warp_max[0], warp_max[1]);
    const uint16_t sbits = f16_from_f32(amax / 127.0f);
    const float sf = f32_from_f16(sbits);                          // quantize against the STORED scale
    int q = 0;
    if (sf > 0.0f) {
        q = sycl::vec<float, 1>{(x / sf)}
                .convert<int, sycl::rounding_mode::rte>()[0];
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
    }
    // KV streaming: the host copy (identity layout) always, the VRAM page only if the block is resident
    const long long page = (long long) table[pos / page_size];
    if (page >= 0) {
        const long long row = (page * kv_heads + h) * page_size + (pos % page_size);
        (is_v ? v_q : k_q)[row * head_dim + g * KV_Q8_GROUP + t] = (int8_t) q;
        if (t == 0) (is_v ? v_scale : k_scale)[row * groups + g] = sbits;
    }
    if (host.k_q != nullptr) {
        const long long row = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
        if (pack_host) {
            // the same bytes, eight codes to a store: lane 8i gathers lanes 8i..8i+7 (one PCIe write instead of eight)
            uint64_t w = (uint64_t) (uint8_t) q;
#pragma unroll
            for (int j = 1; j < 8; ++j)
                w |= (uint64_t) (uint8_t) sycl::select_from_group(sg, q, (int) ((t & 31) | j)) << (8 * j);
            if ((t & 7) == 0)
                *reinterpret_cast<uint64_t *>((is_v ? host.v_q : host.k_q) + row * head_dim + g * KV_Q8_GROUP + t) = w;
        } else
        (is_v ? host.v_q : host.k_q)[row * head_dim + g * KV_Q8_GROUP + t] = (int8_t) q;
        if (t == 0) (is_v ? host.v_scale : host.k_scale)[row * groups + g] = sbits;
    }
}

// One thread = 4 consecutive values of one cell and head (as the FP16 gather does with uint2).  This is the gather
// the ready-made engine runs; the experimental build's 8-wide kernel is below it.
__dpct_inline__ void kv_gather_q8_kernel(
    const int8_t *__restrict__ k_q, const int8_t *__restrict__ v_q,
    const uint16_t *__restrict__ k_scale, const uint16_t *__restrict__ v_scale,
    const int32_t *__restrict__ table, const int32_t *__restrict__ ids,
    const int32_t *__restrict__ step, int kv_heads, int head_dim, int page_size,
    uint16_t *__restrict__ k_scratch, uint16_t *__restrict__ v_scratch) {
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long n_ids = (long long)*(step + kStepWidth);
    const int per = head_dim / 4;
    const long long total = n_ids * kv_heads * per;
    const long long i =
        item_ct1.get_group(2) * (long long)item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= total) return;
    const long long id = i / (kv_heads * (long long) per);
    const int rem = (int) (i % (kv_heads * (long long) per));
    const int h = rem / per, q4 = rem - h * per;
    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);
    const int d = q4 * 4;
    const int groups = head_dim / KV_Q8_GROUP;
    const float ks = f32_from_f16(k_scale[row * groups + d / KV_Q8_GROUP]);
    const float vs = f32_from_f16(v_scale[row * groups + d / KV_Q8_GROUP]);
    const sycl::char4 kc =
        reinterpret_cast<const sycl::char4 *>(k_q + row * head_dim)[q4];
    const sycl::char4 vc =
        reinterpret_cast<const sycl::char4 *>(v_q + row * head_dim)[q4];
    sycl::ushort4 ko, vo;
    ko.x() = f16_from_f32((float)kc.x() * ks);
        ko.y() = f16_from_f32((float)kc.y() * ks);
    ko.z() = f16_from_f32((float)kc.z() * ks);
        ko.w() = f16_from_f32((float)kc.w() * ks);
    vo.x() = f16_from_f32((float)vc.x() * vs);
        vo.y() = f16_from_f32((float)vc.y() * vs);
    vo.z() = f16_from_f32((float)vc.z() * vs);
        vo.w() = f16_from_f32((float)vc.w() * vs);
    const long long dst = (id * kv_heads + h) * (long long) per + q4;
    reinterpret_cast<sycl::ushort4 *>(k_scratch)[dst] = ko;
    reinterpret_cast<sycl::ushort4 *>(v_scratch)[dst] = vo;
}

#if defined(STRATA_EXPERIMENTAL_SM60)
// One thread = 8 consecutive values of one cell and head: an 8-byte `uint2` load of codes and a 16-byte `uint4`
// store of the dequantized fp16.  The 8 values share one 64-value group, so one scale is loaded per thread and
// the arithmetic stays `(float) code * scale -> f16_from_f32`, i.e. bitwise the same as the previous 4-wide
// version (whose char4 loads became these uint2 loads).  `load8_q8` in qsa_decode_attn.cu reads 8 int8 the
// same way; this is the gather's other half.  Only in the experimental build (-DSTRATA_EXPERIMENTAL_SM60=ON):
// the ready-made engine keeps exactly the 4-wide kernel above, so its gather is bit for bit the release engine's.
__device__ __forceinline__ uint32_t q8_pair_to_f16x2(const int8_t* c, float s) {
    return (uint32_t) f16_from_f32((float) c[0] * s) | ((uint32_t) f16_from_f32((float) c[1] * s) << 16);
}

__global__ void kv_gather_q8_wide_kernel(const int8_t* __restrict__ k_q, const int8_t* __restrict__ v_q,
                                         const uint16_t* __restrict__ k_scale, const uint16_t* __restrict__ v_scale,
                                         const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                                         const int32_t* __restrict__ step, int kv_heads, int head_dim, int page_size,
                                         uint16_t* __restrict__ k_scratch, uint16_t* __restrict__ v_scratch) {
    const long long n_ids = (long long) __ldg(step + kStepWidth);
    const int per = head_dim / 8;
    const long long total = n_ids * kv_heads * per;
    const long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x;
    if (i >= total) return;
    const long long id = i / (kv_heads * (long long) per);
    const int rem = (int) (i % (kv_heads * (long long) per));
    const int h = rem / per, q8 = rem - h * per;
    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);
    const int d = q8 * 8;                    // 8 divides KV_Q8_GROUP, so all 8 values share the group's scale
    const int groups = head_dim / KV_Q8_GROUP;
    const float ks = f32_from_f16(k_scale[row * groups + d / KV_Q8_GROUP]);
    const float vs = f32_from_f16(v_scale[row * groups + d / KV_Q8_GROUP]);
    const uint2 kraw = *reinterpret_cast<const uint2*>(k_q + row * head_dim + d);   // one 8-byte load
    const uint2 vraw = *reinterpret_cast<const uint2*>(v_q + row * head_dim + d);
    const int8_t* kc = reinterpret_cast<const int8_t*>(&kraw);
    const int8_t* vc = reinterpret_cast<const int8_t*>(&vraw);
    uint4 ko, vo;
    ko.x = q8_pair_to_f16x2(kc + 0, ks); ko.y = q8_pair_to_f16x2(kc + 2, ks);
    ko.z = q8_pair_to_f16x2(kc + 4, ks); ko.w = q8_pair_to_f16x2(kc + 6, ks);
    vo.x = q8_pair_to_f16x2(vc + 0, vs); vo.y = q8_pair_to_f16x2(vc + 2, vs);
    vo.z = q8_pair_to_f16x2(vc + 4, vs); vo.w = q8_pair_to_f16x2(vc + 6, vs);
    const long long dst = (id * kv_heads + h) * (long long) per + q8;
    reinterpret_cast<uint4*>(k_scratch)[dst] = ko;
    reinterpret_cast<uint4*>(v_scratch)[dst] = vo;
}

// The current device is below sm_75 (per device: a layer split can mix cards); STRATA_Q8_GATHER_WIDE=0 turns the
// 8-wide gather off (A/B).  The wide kernel is bitwise identical, so this only decides speed.  Mirrors
// qsa_decode_attn.cu's pre75_attn.
bool pre75_q8_gather() {
    static const bool off = [] {
        const char* e = std::getenv("STRATA_Q8_GATHER_WIDE");
        return e != nullptr && e[0] == '0';
    }();
    static int cc[64] = {};
    int dev = 0;
    if (off) return false;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (cc[dev] == 0) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        cc[dev] = 10 * major + minor;
    }
    return cc[dev] < 75;
}
#endif  // STRATA_EXPERIMENTAL_SM60

}  // namespace

void kv_append_q8_step(int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale, const int32_t* page_table,
                       const int32_t* step, const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    validate(s, "kv_append_q8");
    const dpct::dim3 grid((unsigned)s.n_head_kv,
                          (unsigned)(s.head_dim / KV_Q8_GROUP), 2);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {
                KvHostPools host_host_KvHostPools_ct11 =
                    host ? *host : KvHostPools{};
                const bool pack = kv_append_pack_host();

                /*
                DPCT1050: The template argument of the dpct_kernel_name
                could not be deduced. You need to update this code.
                */
                cgh.parallel_for<dpct_kernel_name<
                    class kv_append_q8_kernel_b60fdc,
                    dpct_kernel_scalar<false>>>(
                    sycl::nd_range<3>(grid * sycl::range(1, 1, KV_Q8_GROUP),
                                      sycl::range(1, 1, KV_Q8_GROUP)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            kv_append_q8_kernel(
                                k_q, v_q, k_scale, v_scale, page_table, step,
                                kcur, vcur, (int)s.n_head_kv, (int)s.head_dim,
                                (int)s.page_size, host_host_KvHostPools_ct11, 0,
                                0, pack);
                        });
            });
    }
    check("kv_append_q8 launch");
}

void kv_append_q8_steps(int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale, const int32_t* page_table,
                        const int32_t* step, int step_stride, const float* kcur, const float* vcur, int cur_stride,
                        int n_tok, const QsaShapes& s, void* stream, const KvHostPools* host) {
    validate(s, "kv_append_q8 (steps)");
    if (n_tok < 1) return;
    const dpct::dim3 grid((unsigned)s.n_head_kv,
                          (unsigned)(s.head_dim / KV_Q8_GROUP),
                          (unsigned)(2 * n_tok));
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {
                KvHostPools host_host_KvHostPools_ct11 =
                    host ? *host : KvHostPools{};
                const bool pack = kv_append_pack_host();

                cgh.parallel_for<
                    dpct_kernel_name<class kv_append_q8_kernel_b60fdc,
                                     dpct_kernel_scalar<true>>>(
                    sycl::nd_range<3>(grid * sycl::range(1, 1, KV_Q8_GROUP),
                                      sycl::range(1, 1, KV_Q8_GROUP)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            kv_append_q8_kernel<true>(
                                k_q, v_q, k_scale, v_scale, page_table, step,
                                kcur, vcur, (int)s.n_head_kv, (int)s.head_dim,
                                (int)s.page_size, host_host_KvHostPools_ct11,
                                step_stride, cur_stride, pack);
                        });
            });
    }
    check("kv_append_q8 (steps) launch");
}

void kv_gather_q8_step(const int8_t* k_q, const int8_t* v_q, const uint16_t* k_scale, const uint16_t* v_scale,
                       const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids,
                       const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    validate(s, "kv_gather_q8");
    if (max_ids <= 0) return;
#if defined(STRATA_EXPERIMENTAL_SM60)
    if (pre75_q8_gather()) {
        const long long total = max_ids * s.n_head_kv * (s.head_dim / 8);
        const unsigned blocks = (unsigned) ((total + 255) / 256);
        kv_gather_q8_wide_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(
            k_q, v_q, k_scale, v_scale, page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim,
            (int) s.page_size, k_scratch, v_scratch);
        check("kv_gather_q8 wide launch");
        return;
    }
#endif
    const long long total = max_ids * s.n_head_kv * (s.head_dim / 4);
    const unsigned blocks = (unsigned) ((total + 255) / 256);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class kv_gather_q8_kernel_84cbb3>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    kv_gather_q8_kernel(k_q, v_q, k_scale, v_scale, page_table,
                                        ids, step, (int)s.n_head_kv,
                                        (int)s.head_dim, (int)s.page_size,
                                        k_scratch, v_scratch);
                });
    }
    check("kv_gather_q8 launch");
}

}  // namespace strata::kernels
