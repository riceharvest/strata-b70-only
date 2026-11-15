// strata/b70/fat.hpp - the B70 card library: fat products (many activation columns a weight pass: a prompt chunk) on the
// matrix engine (sycl/src/b70/fat_i8.dp.cpp, fat_w4i8.dp.cpp; Xe2 XMX, ESIMD).  Kernels of the card, not of a model: every
// entry takes its shape as arguments (n_in a multiple of 256, n_out of 16 for the int4 pack) and says with handles*()
// whether it takes a call.  The 27B's driver was their first caller (it knows them as strata::q36pp).
// Weights remain quantized in VRAM (Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS); activations use native_quantize_q8_1.
#pragma once
#include <cstddef>

namespace strata::b70::fat {

/// True when `gemm` handles (ggml_type, n_in, n_out, T).
bool handles(int ggml_type, int n_in, int n_out, int T) noexcept;

/// y[col * n_out + row] = weights . x_q8_1 for T > 0 prompt columns, float32 output.
/// x_q8_1 holds T contiguous columns of (n_in / 32) block_q8_1 blocks (36 B each).
/// Returns true if launched on `stream`; false if unsupported (caller uses fallback FP16 GEMM).
bool gemm(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int T, void* stream);

/// True when `gemm_w4_f16` handles B70W4 (n_in, n_out, T).
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

// PP2 (card-up B70W4 prompt GEMM + fused FFN gate/up+SwiGLU+x128 + coalesced RMS/x128): gated by STRATA_Q36_PP2.
bool pp2_enabled() noexcept;
bool pp2_fuse_enabled() noexcept;
std::size_t dxT_pp2_floats(int n_in, int T) noexcept;
void quantize_x128_pp2(const float* x, void* xq8, float* dxT, int K, int M, void* stream);
void swiglu_x128_pp2(const float* g, const float* u, void* xq8, float* dxT, int K, int M, void* stream);
bool gemm_w4_i8_pp2(const void* weights, const void* xq8, const float* dxT, float* y,
                    int n_in, int n_out, int T, void* stream);
bool rms_pp2(float* x, const float* y, const float* w, float* h, void* xq8, float* dxT,
             int n, int T, float eps, void* stream);
bool gemm_w4_i8_gu_swiglu_pp2(const void* w_gate, const void* w_up,
                              const void* xq8_in, const float* dxT_in,
                              void* xq8_out, float* dxT_out,
                              int n_in, int n_out, int T, void* stream);

// PP3 (fatgemm-paced 256-GRF ping-pong W4I8 GEMM + fused FFN gate/up+SwiGLU+x128): gated by STRATA_Q36_PP3.
bool pp3_enabled() noexcept;
bool gemm_w4_i8_pp3(const void* weights, const void* xq8, const float* dxT, float* y,
                    int n_in, int n_out, int T, void* stream);
bool gemm_w4_i8_gu_swiglu_pp3(const void* w_gate, const void* w_up,
                              const void* xq8_in, const float* dxT_in,
                              void* xq8_out, float* dxT_out,
                              int n_in, int n_out, int T, void* stream);

}  // namespace strata::b70::fat
