// q36_gdn_wy.hpp - Chunkwise C=64 parallel WY GDN prefill recurrence on Xe2 XMX (ESIMD).
#pragma once

#include <cstddef>
#include <sycl/sycl.hpp>

namespace strata::q36wy {

constexpr int kChunk = 64;
constexpr int kS     = 128;
constexpr int kHK    = 16;
constexpr int kHV    = 48;
constexpr int kC     = 10240;

inline size_t workspace_bytes(int max_T) {
    const size_t nc = (size_t)((max_T + kChunk - 1) / kChunk);
    // per (chunk, V-head): ws_W [64][128] half + ws_U [64][128] half +
    // ws_Mqk [64][64] half + ws_g [64] float
    return nc * (size_t)kHV * (2ull * kChunk * kS * 2ull + (size_t)kChunk * kChunk * 2ull + (size_t)kChunk * 4ull);
}

bool ok(const sycl::queue& q, int S, int HK, int HV, int C);

void forward(sycl::queue& q, void* ws, int ws_max_T, float* state, const float* hc,
             const float* gate, const float* beta, float* out, int T);

} // namespace strata::q36wy
