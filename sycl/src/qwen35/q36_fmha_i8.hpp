#pragma once
#include <cstdint>
#include <sycl/sycl.hpp>

namespace strata::q36fmha {

// Int8-native XMX GQA causal prompt attention for dense qwen35 (HD=256, G=6, int8 KV cache with vt > 0).
// q16: [T, H, 256] half; kc: [HKV, ctx, 256] int8 + [HKV, ctx, 4] half scales;
// vc: [HKV, vt/32, 256, 32] int8 + [HKV, vt] float scales; out: [T, H, 256] float.
bool supports(int T, int H, int HKV, int HD, int vt);
void forward(sycl::queue& q, int vt, const sycl::half* q16, const int8_t* kc, const int8_t* vc, float* out,
             int T, int p0, int H, int HKV, int ctx, bool u16_p = false);

}  // namespace strata::q36fmha
