// q36_mmvq_dense.hpp - strata-q36's decode matrix-vector forms for the dense Qwen3.8-27B (q36_mmvq_dense.dp.cpp).
// Only strata-q36 includes and links this; the kernel library `strata` links does not have it.
#pragma once

namespace strata::q36dense {

/// y[col * n_out + row] = weights . x_q8_1 for 1-8 columns, as strata::kernels::native_mmvq does, through the KREG forms tuned for
/// the 27B's shapes.  Q3_K (11), Q4_K (12), Q5_K (13), Q6_K (14), IQ4_NL (20) and IQ4_XS (23); true: the product was launched on
/// `stream`.  false: not this call (another type, STRATA_MMVQ_KREG not 1 for the KREG forms, arguments native_mmvq would reject, a
/// shape the forms do not take): nothing was launched, call native_mmvq.
bool mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);

}  // namespace strata::q36dense
