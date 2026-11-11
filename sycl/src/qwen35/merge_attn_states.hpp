#pragma once
// SPDX-License-Identifier: Apache-2.0
// This file is a MODIFIED copy of csrc/attention/merge_attn_states.cpp of vllm-xpu-kernels
// (https://github.com/vllm-project/vllm-xpu-kernels, Apache License 2.0), commit ddf336d86e3c8602888572a3502f951abd51df12
// (v0.1.15 + 4).  What changed: the kernel only (no torch launcher), with the to_float / from_float helpers of csrc/utils.h;
// the FP8-output branches (USE_FP8_OUTPUT, never used here) are left out.  The merge itself (the BF16 / FP16 / FP32
// output path) is the released code.  The licence text is third_party/licenses/vllm-xpu-kernels-LICENSE; see
// THIRD_PARTY_NOTICES.md.

#include <sycl/sycl.hpp>
#include <sys/types.h>   // uint
#include <limits>

namespace vllm {
namespace xpu {

// From float to float.
inline void from_float(float& dst, float src) { dst = src; }
// From float32 to float16.
inline void from_float(sycl::half& dst, float src) { dst = sycl::half(src); }
// From float32 to bfloat16.
inline void from_float(sycl::ext::oneapi::bfloat16& dst, float src) {
  dst = sycl::ext::oneapi::bfloat16(src);
}

// From float to float.
inline float to_float(float u) { return u; }
// From float16 to float32.
inline float to_float(sycl::half u) { return float(u); }
// From bfloat16 to float32.
inline float to_float(sycl::ext::oneapi::bfloat16 u) { return float(u); }

}  // namespace xpu

// Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
// can be used to combine partial attention results (in the split-KV case)
template <typename scalar_t, typename output_t, const uint NUM_THREADS>
void merge_attn_states_kernel(
    output_t* output,
    float* output_lse,
    const scalar_t* prefix_output,
    const float* prefix_lse,
    const scalar_t* suffix_output,
    const float* suffix_lse,
    const uint num_tokens,
    const uint num_heads,
    const uint head_size,
    const uint prefix_head_stride,
    const uint output_head_stride,
    const uint prefix_num_tokens,
    const sycl::nd_item<3>& item_ct1) {
  // Inputs always load 128-bit packs (pack_size elements of scalar_t).
  using input_pack_t = sycl::uint4;
  using output_pack_t = sycl::uint4;
  const uint pack_size = 16 / sizeof(scalar_t);
  const uint threads_per_head = head_size / pack_size;

  const uint global_idx =
      item_ct1.get_group(2) * NUM_THREADS + item_ct1.get_local_id(2);
  const uint token_head_threads = num_tokens * num_heads * threads_per_head;

  if (global_idx >= token_head_threads) return;

  // global_idx -> token_idx + head_idx + pack_idx
  const uint token_head_idx = global_idx / threads_per_head;
  const uint pack_idx = global_idx % threads_per_head;

  const uint token_idx = token_head_idx / num_heads;
  const uint head_idx = token_head_idx % num_heads;

  const uint pack_offset = pack_idx * pack_size;  // (0~15)*8, etc.
  const uint src_head_offset = token_idx * num_heads * prefix_head_stride +
                               head_idx * prefix_head_stride;
  const uint dst_head_offset = token_idx * num_heads * output_head_stride +
                               head_idx * output_head_stride;
  const scalar_t* prefix_head_ptr = prefix_output + src_head_offset;
  const scalar_t* suffix_head_ptr = suffix_output + src_head_offset;
  output_t* output_head_ptr = output + dst_head_offset;

  // If token_idx >= prefix_num_tokens, just copy from suffix
  if (token_idx >= prefix_num_tokens) {
    if (pack_offset < head_size) {
      input_pack_t s_out_pack = reinterpret_cast<const input_pack_t*>(
          suffix_head_ptr)[pack_offset / pack_size];
      reinterpret_cast<output_pack_t*>(
          output_head_ptr)[pack_offset / pack_size] = s_out_pack;
    }
    if (output_lse != nullptr && pack_idx == 0) {
      float s_lse = suffix_lse[head_idx * num_tokens + token_idx];
      output_lse[head_idx * num_tokens + token_idx] = s_lse;
    }
    return;
  }

  float p_lse = prefix_lse[head_idx * num_tokens + token_idx];
  float s_lse = suffix_lse[head_idx * num_tokens + token_idx];
  p_lse = sycl::isinf(p_lse) ? -std::numeric_limits<float>::infinity() : p_lse;
  s_lse = sycl::isinf(s_lse) ? -std::numeric_limits<float>::infinity() : s_lse;

  const float max_lse = sycl::fmax(p_lse, s_lse);

  /* In certain edge cases, MLA can produce p_lse = s_lse = -inf;
     continuing the pipeline then yields NaN. Root cause: with chunked prefill
     a batch may be split into two chunks; if a request in that batch has no
     prefix hit, every LSE entry for that request's position is -inf, and at
     this moment we merge cross-attention at first. For now we simply emit
     prefix_output (expected to be all zeros) and prefix_lse (-inf) to fix
     this problem.
  */
  if (sycl::isinf(max_lse)) {
    if (pack_offset < head_size) {
      // Pack 128b load from prefix
      input_pack_t p_out_pack = reinterpret_cast<const input_pack_t*>(
          prefix_head_ptr)[pack_offset / pack_size];
      // Pack 128b storage
      reinterpret_cast<output_pack_t*>(
          output_head_ptr)[pack_offset / pack_size] = p_out_pack;
    }
    // We only need to write to output_lse once per head.
    if (output_lse != nullptr && pack_idx == 0) {
      output_lse[head_idx * num_tokens + token_idx] = max_lse;
    }
    return;
  }

  p_lse = p_lse - max_lse;
  s_lse = s_lse - max_lse;
  const float p_se = sycl::native::exp(p_lse);
  const float s_se = sycl::native::exp(s_lse);
  const float out_se = p_se + s_se;
  const float p_scale = p_se / out_se;
  const float s_scale = s_se / out_se;

  if (pack_offset < head_size) {
    // Pack 128b load
    input_pack_t p_out_pack = reinterpret_cast<const input_pack_t*>(
        prefix_head_ptr)[pack_offset / pack_size];
    input_pack_t s_out_pack = reinterpret_cast<const input_pack_t*>(
        suffix_head_ptr)[pack_offset / pack_size];

    // Compute merged values in float32
    float o_out_f[pack_size];
#pragma unroll
    for (uint i = 0; i < pack_size; ++i) {
      // Always use float for FMA to keep high precision.
      // half(uint16_t), bfloat16, float -> float.
      const float p_out_f = vllm::xpu::to_float(
          reinterpret_cast<const scalar_t*>(&p_out_pack)[i]);
      const float s_out_f = vllm::xpu::to_float(
          reinterpret_cast<const scalar_t*>(&s_out_pack)[i]);
      // fma: a * b + c = p_out_f * p_scale + (s_out_f * s_scale)
      o_out_f[i] = p_out_f * p_scale + s_out_f * s_scale;
    }

    // Convert and store
    output_pack_t o_out_pack;
#pragma unroll
    for (uint i = 0; i < pack_size; ++i) {
      // float -> half(uint16_t), bfloat16, float.
      vllm::xpu::from_float(
          reinterpret_cast<output_t*>(&o_out_pack)[i], o_out_f[i]);
    }
    // Pack 128b storage
    reinterpret_cast<output_pack_t*>(
        output_head_ptr)[pack_offset / pack_size] = o_out_pack;
  }
  // We only need to write to output_lse once per head.
  if (output_lse != nullptr && pack_idx == 0) {
    float out_lse = sycl::log((float)out_se) + max_lse;
    output_lse[head_idx * num_tokens + token_idx] = out_lse;
  }
}

}  // namespace vllm
