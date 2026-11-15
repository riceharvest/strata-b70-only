// strata/b70/fat_bench.hpp - the B70 card library: host entry points of the PARITY=ON pure-integer fat-product reference and the
// PP3-v2 layout benchmark (sycl/src/b70/fat_bench.dp.cpp).
#pragma once
#include <cstdint>
#include <sycl/sycl.hpp>

namespace strata::b70::fat::bench {

/// Convert `n_mat` packed B70W4 v1 matrices into contiguous `fatgemm` pure-u4 `[rt32][step][2][64]` layout.
void convert_b70w4_to_fg(sycl::queue& q, const uint32_t* src_v1, uint32_t* dst_fg,
                         int n_in, int n_out, int n_mat);

/// Convert `n_mat` packed B70W4 v1 matrices into contiguous PP3-v2 `[rt32][kb][8*128 + 64]` layout (same byte size).
void convert_b70w4_to_v2(sycl::queue& q, const uint32_t* src_v1, uint32_t* dst_v2,
                         int n_in, int n_out, int n_mat);

/// Exact pure-integer fatgemm (`fat<2,2,4,3>`): u4 x s8 -> int32 (no scales/zeros) on `[rt32][step][2][64]` + xq8_pp2.
void launch_fatgemm_u4i8_ref(sycl::queue& q, const uint32_t* W_fg, const int* xq8,
                             int* Y_out, int K, int R, int T, bool slack_ok);

/// Self-check `launch_fatgemm_u4i8_ref` against host exact int32 reference on 512 x 512 x 512 (0 mismatch).
bool verify_fatgemm_u4i8_ref(sycl::queue& q);

/// Scaled PP3-v2 GEMM on contiguous `[rt32][kb][1088]` weight layout + xq8_pp2 / dxT_pp2.
bool gemm_w4_i8_pp3_v2(const void* weights_v2, const void* xq8, const float* dxT, float* y,
                       int n_in, int n_out, int T, void* stream);

}  // namespace strata::b70::fat::bench
