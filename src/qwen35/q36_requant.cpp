// src/qwen35/q36_requant.cpp - a matrix of a GGUF type the native MMVQ does not take, requantized to Q8_0 at load through
// ggml's CPU dequantizer (the 27B's files need none: their IQ3_S runs natively).  Host code only.
#include "ggml.h"

#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace strata::qwen35 {

bool requant_q8_0(int type, const void* src, int64_t n_in, int64_t n_out, std::vector<uint8_t>& out) {
    if (type < 0 || type >= GGML_TYPE_COUNT || n_in % 32 != 0) return false;
    const ggml_type t = (ggml_type) type;
    const ggml_type_traits* tr = ggml_get_type_traits(t);
    if (tr == nullptr || tr->to_float == nullptr || n_in % ggml_blck_size(t) != 0) return false;
    const size_t src_row = ggml_row_size(t, n_in), dst_row = ggml_row_size(GGML_TYPE_Q8_0, n_in);
    out.assign((size_t) n_out * dst_row, 0);
    const int nt = (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int w = 0; w < nt; ++w)
        th.emplace_back([&, w] {
            std::vector<float> row((size_t) n_in);
            for (int64_t r = w; r < n_out; r += nt) {
                tr->to_float((const uint8_t*) src + (size_t) r * src_row, row.data(), n_in);
                ggml_quantize_chunk(GGML_TYPE_Q8_0, row.data(), out.data() + (size_t) r * dst_row, 0, 1, n_in, nullptr);
            }
        });
    for (auto& x : th) x.join();
    return true;
}

}  // namespace strata::qwen35
