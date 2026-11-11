// q36_tla.hpp - the interface of libq36_tla.so (q36_tla_fmha.cpp): vllm-xpu-kernels' Xe2 chunk_prefill attention
// (release 0.1.15.4, on sycl-tla 87f68506; third_party/vllm-xpu-kernels, third_party/sycl-tla), compiled from their
// templates as vllm-xpu-kernels builds libattn_kernels_xe_2 (its defines, its link options), for the one shape strata-q36
// asks: head 256, FP16 Q / K / V / O, one sequence (varlen, batch 1), no paging, no sinks, no sliding window, the
// natural-log softmax LSE written.  Included by q36_tla_fmha.cpp and by q36_prefill.dp.hpp (the engine side).
#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>

namespace q36tla {

/// One launch: out [T][H][256] = softmax(Q K^T * scale) V for the T queries over the L cells of K / V, causal or not.
/// Causal is bottom-right aligned (the chunk_prefill kernel's full_tile_offset = L - T): query t sees cells
/// [0, L - T + t].  lse [H][T] (stride T) gets each row's natural-log sum of exp of its scaled scores.
struct Fmha {
    const sycl::half* q = nullptr;   // [T][H][256] (row stride H * 256)
    const sycl::half* k = nullptr;   // [L][HKV][256] (row stride HKV * 256)
    const sycl::half* v = nullptr;   // same layout as k
    sycl::half* o = nullptr;         // [T][H][256]
    float* lse = nullptr;            // [H][T]
    const int32_t* cu_q = nullptr;   // device {0, T}: the kernel's cumulative query lengths (read by the kernel)
    const int32_t* cu_k = nullptr;   // device {0, L}
    int T = 0, L = 0, H = 0, HKV = 0;
    float scale = 0.0f;              // 1 / sqrt(256)
    bool causal = false;
};

/// The kernel's head size (the only one built).
constexpr int kHeadDim = 256;

/// Submits the kernel on q (no wait).
void fmha(sycl::queue& q, const Fmha& f);

}  // namespace q36tla
