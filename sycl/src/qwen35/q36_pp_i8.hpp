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

// Phase 5c: W4 -> INT8 s8*s8->s32 SplitPipe XMX, fused RMS(+add)(+x128), and fused GDN alpha/beta DPAS.
bool handles_w4_i8(int n_in, int n_out, int T) noexcept;
void quantize_x128(const float* x, void* xq8, float* dxT, int K, int M, void* stream);
void swiglu_x128(const float* g, const float* u, void* xq8, float* dxT, int K, int M, void* stream);
bool gemm_w4_i8(const void* weights, const void* xq8, const float* dxT, float* y, int n_in, int n_out, int T, void* stream);
bool rms_pp(float* x, const float* y, const float* w, float* h, void* xq8, float* dxT,
            int n, int T, float eps, void* stream);
void* pack_ab16(const float* alpha, const float* beta, int HV, int N, void* stream);
void gdn_ab_pp(const void* ab16, const float* h, const float* dt, const float* ssm_a,
               float* al_out, float* be_out, int n_in, int HV, int T, void* stream);

}  // namespace strata::q36pp
