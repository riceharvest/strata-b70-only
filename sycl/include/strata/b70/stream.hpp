// strata/b70/stream.hpp - the B70 card library: stream products (a weight matrix read once from memory against 1-32 activation
// columns: a decode step, a verify window, a batch's rows) on the matrix engine (sycl/src/b70/stream_w4*.dp.cpp; Xe2 XMX, ESIMD).
//
// Kernels of the card, not of a model.  The weights are a pack matrix (strata/b70/pack.hpp: records of 16 output rows x 256
// inputs, the 4-bit values in DPAS operand order), so a hardware thread reads its records with plain block loads and a
// column more costs no second read of the weights.  Every entry takes its shape as arguments: n_in a multiple of 256, n_out a
// multiple of 16, 1-32 columns.  An entry returns false (nothing launched) for a call it does not take.
//
// Three activation forms, by what the caller already has:
//   mmvq_w4    the q8_1 image the shared quantizers write (native_quantize_q8_1): a drop-in for native_mmvq
//   mmvq_w4x   int8 activations with one scale per 128 inputs: the 4-bit values go to the matrix engine as they are
//   mmvq_w4h   FP16 activations (W4A16): no float work per column and 32 weights, for many columns
// The 27B's driver was their first caller (it knows them as strata::q36xr).
#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::b70::stream {

/// The bytes of a packed n_out x n_in matrix (n_in a multiple of 256, n_out of 16).
std::size_t w4_bytes(int n_in, int n_out);
/// y[col * n_out + row] = W4 weights . the q8_1 image `xq` (with 28 readable bytes after its last block) for 1-32 columns
/// (one read of the weights at any count; the columns run as groups of 8).
/// `weights`: the pack's records, 64-byte aligned.  false: not taken, nothing launched.
bool mmvq_w4(const void* weights, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream);
/// The pack as FP16 [n_out][n_in] (a prompt chunk's GEMM operand): n_out x n_in halves at `out`.
void w4_dequant_f16(const void* weights, uint16_t* out, int n_in, int n_out, void* stream);

/// The int4 pack against FP16 activations (W4A16): the weights as half floats (q - zero, exact), the DPAS of half floats (16
/// inputs an instruction), a row's scale once per 128 inputs.  No per-column float work per 32 weights: a column more costs
/// its share of a DPAS.  x16: the columns interleaved by 16 inputs, [n_in / 16][columns rounded up to 8][16] halfs
/// (w4h_act_halfs of them).  1 .. 32 columns.
std::size_t w4h_act_halfs(int n_in, int ncols);
bool mmvq_w4h(const void* weights, const uint16_t* x16, float* y, int n_in, int n_out, int ncols, void* stream);
/// 8 columns with a chosen decomposition (`chunks` in 1, 2, 4, 8, 16 threads a tile, `threads` in all): for a product with few
/// tiles, which would leave most of the card's threads idle at the default split.
bool mmvq_w4h_8(const void* weights, const uint16_t* x16, float* y, int n_in, int n_out, void* stream, int chunks, int threads);
/// x [ncols][n_in] floats (rows ldx apart) as x16; the FFN's silu(g) * u as x16.
void w4h_pack(const float* x, int ldx, uint16_t* x16, int n_in, int ncols, void* stream);
void w4h_swiglu_pack(const float* g, const float* u, uint16_t* x16, int n, int ncols, void* stream);

/// W4X: the native u4 x s8 form (gs = 128 or 32: the inputs an activation scale covers).
std::size_t w4x_act_bytes(int n_in, int ncols, int gs);
void w4x_quantize(const float* x, int ldx, void* act, int n_in, int ncols, int gs, void* stream);
void w4x_rms_apply_quantize(const float* x_res, int ldx, const float* norm_w, const float* inv_rms,
                            void* act, float* h_out, int ld_out, int n_in, int ncols, int gs, void* stream);
void w4x_swiglu_quantize(const float* g, const float* u, void* act, int n_in, int ncols, int gs, void* stream);
bool mmvq_w4x(const void* weights, const void* act, float* y, int n_in, int n_out, int ncols, int gs,
              void* stream, int ksplit_override = 0);

/// Optional merged gate+up interleaved pack (a gated FFN's two input matrices as one stream): one launch produces the gs = 128
/// activations of the FFN's output matrix.
std::size_t w4x_gu_merged_bytes(int n_in, int n_ff);
bool mmvq_w4x_gu_swiglu(const void* gu_weights, const void* act_in, void* act_down,
                        int n_in, int n_ff, int ncols, int gs, void* stream);

}   // namespace strata::b70::stream
