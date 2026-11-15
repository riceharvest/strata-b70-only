// strata/b70/attn_prompt.hpp - the B70 card library: causal prompt attention over an 8-bit cache on the matrix engine
// (sycl/src/b70/attn_prompt_i8.dp.cpp).  A kernel of the card, not of a model: any engine whose cache has this layout calls it.
// Shapes it takes today: supports().
#pragma once
#include <cstdint>
#include <sycl/sycl.hpp>

namespace strata::b70::attn_prompt {

// Int8-native XMX GQA causal prompt attention (head dimension 256, 6 query heads per KV head, int8 KV cache with vt > 0).
// q16: [T, H, 256] half; kc: [HKV, ctx, 256] int8 + [HKV, ctx, 4] half scales;
// vc: [HKV, vt/32, 256, 32] int8 + [HKV, vt] float scales; out: [T, H, 256] float.
bool supports(int T, int H, int HKV, int HD, int vt);
void forward(sycl::queue& q, int vt, const sycl::half* q16, const int8_t* kc, const int8_t* vc, float* out,
             int T, int p0, int H, int HKV, int ctx, bool u16_p = false);

}  // namespace strata::b70::attn_prompt
