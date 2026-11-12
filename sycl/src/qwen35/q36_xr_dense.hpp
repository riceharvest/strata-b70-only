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
}   // namespace strata::q36xr
