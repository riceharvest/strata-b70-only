// q36_xr_dense.hpp - strata-q36's XMX decode products for the dense Qwen3.8-27B on the Arc Pro B70 (q36_xr_dense.dp.cpp).
#pragma once
#include <cstddef>
#include <cstdint>
#include "strata/b70/stream.hpp"

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
/// y[col * n_out + row] = weights . activations for 1-16 columns (9-16 read each weight tile once; STRATA_Q36_XR16=0 keeps
/// 9-16 off).  false: not taken (type, shape), nothing launched.
/// q81 = false: `act` is quantize()'s block.  q81 = true: `act` is the q8_1 image native_quantize_q8_1 writes (ncols columns
/// of n_in / 32 blocks of 36 bytes), with at least 28 readable bytes after its last block.
bool mmvq(int ggml_type, const void* weights, const void* act, float* y, int n_in, int n_out, int ncols, void* stream, bool q81 = false);
/// Whether mmvq() takes (ggml_type, n_in, n_out, ncols, q81), including STRATA_Q36_XR16* and STRATA_Q36_XR_IQ4.
bool handles_call(int ggml_type, int n_in, int n_out, int ncols, bool q81 = true);
// The int4 pack's stream products (w4_bytes, mmvq_w4, w4_dequant_f16, the W4A16 and W4X forms) are the card library's
// (strata/b70/stream.hpp); this family's sources go on calling them under their first names.
using namespace strata::b70::stream;
}   // namespace strata::q36xr
