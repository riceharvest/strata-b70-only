// q36_xr_dense.hpp - strata-q36's XMX decode products for the dense Qwen3.8-27B on the Arc Pro B70 (q36_xr_dense.dp.cpp).
#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::q36xr {
/// Once per queue, outside any graph recording: the device tables.
void init(void* stream);
/// Whether `type` (a ggml type id) has an XMX form here.
bool handles(int ggml_type);
/// The bytes of the activation block quantize() writes for a width of n_in (a multiple of 256).
std::size_t act_bytes(int n_in);
/// x [ncols][n_in] floats -> `act`: per 32 activations the int8 values of up to 8 columns (q8_1 as llama.cpp: d = amax / 127),
/// each column's scale and its scale x sum.  Columns ncols .. 7 are zero.
void quantize(const float* x, int n_in, int ncols, void* act, void* stream);
/// y[col * n_out + row] = weights . activations for 1-8 columns.  false: not taken (type, shape), nothing launched.
/// q81 = false: `act` is quantize()'s block.  q81 = true: `act` is the q8_1 image native_quantize_q8_1 writes (ncols columns
/// of n_in / 32 blocks of 36 bytes), with at least 28 readable bytes after its last block.
bool mmvq(int ggml_type, const void* weights, const void* act, float* y, int n_in, int n_out, int ncols, void* stream, bool q81 = false);
/// B70W4 (tools/qwen35/b70w4_pack.py): the bytes of a packed n_out x n_in matrix (n_in a multiple of 256, n_out of 16).
std::size_t w4_bytes(int n_in, int n_out);
/// y[col * n_out + row] = W4 weights . the q8_1 image `xq` (with 28 readable bytes after its last block) for 1-32 columns
/// (one read of the weights at any count; the columns run as groups of 8).
/// `weights`: the pack's records, 64-byte aligned.  false: not taken, nothing launched.
bool mmvq_w4(const void* weights, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream);
/// The pack as FP16 [n_out][n_in] (a prompt chunk's GEMM operand): n_out x n_in halves at `out`.
/// The int4 pack against FP16 activations (W4A16, what vLLM runs): the weights as half floats (q - zero, exact), the
/// DPAS of half floats (16 inputs an instruction), a row's scale once per 128 inputs.  No per-column float work per 32
/// weights: a column more costs its share of a DPAS.  x16: the columns interleaved by 16 inputs, [n_in / 16][columns
/// rounded up to 8][16] halfs (w4h_act_halfs of them).  1 .. 32 columns.
std::size_t w4h_act_halfs(int n_in, int ncols);
bool mmvq_w4h(const void* weights, const uint16_t* x16, float* y, int n_in, int n_out, int ncols, void* stream);
/// x [ncols][n_in] floats (rows ldx apart) as x16; the FFN's silu(g) * u as x16.
void w4h_pack(const float* x, int ldx, uint16_t* x16, int n_in, int ncols, void* stream);
void w4h_swiglu_pack(const float* g, const float* u, uint16_t* x16, int n, int ncols, void* stream);
void w4_dequant_f16(const void* weights, uint16_t* out, int n_in, int n_out, void* stream);
}   // namespace strata::q36xr
