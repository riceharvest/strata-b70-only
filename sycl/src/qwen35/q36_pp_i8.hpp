// q36_pp_i8.hpp - Prompt-path (T > 16) int8 DPAS GEMM for dense Qwen3.8-27B (`qwen35`) on B70 (Xe2 XMX, ESIMD).
// Weights remain quantized in VRAM (Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS); activations use native_quantize_q8_1.
#pragma once

namespace strata::q36pp {

/// True when `q36pp::gemm` handles (ggml_type, n_in, n_out, T).
bool handles(int ggml_type, int n_in, int n_out, int T) noexcept;

/// y[col * n_out + row] = weights . x_q8_1 for T > 0 prompt columns, float32 output.
/// x_q8_1 holds T contiguous columns of (n_in / 32) block_q8_1 blocks (36 B each).
/// Returns true if launched on `stream`; false if unsupported (caller uses fallback FP16 GEMM).
bool gemm(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int T, void* stream);

/// True when `q36pp::gemm_w4_f16` handles B70W4 (n_in, n_out, T).
bool handles_w4_f16(int n_in, int n_out, int T) noexcept;

/// Fused B70W4 -> FP16 XMX DPAS GEMM: y[col * n_out + row] = W4 . x_f16.
/// x_f16 holds T contiguous rows of n_in sycl::half values; y holds T rows of n_out floats.
bool gemm_w4_f16(const void* weights, const void* x_f16, float* y, int n_in, int n_out, int T, void* stream);

}  // namespace strata::q36pp
