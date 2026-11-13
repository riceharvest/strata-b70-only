// src/qwen35/q36_requant.cpp - a matrix of a GGUF type the native MMVQ does not take, requantized to Q8_0 at load through
// ggml's CPU dequantizer (the 27B's files need none: their IQ3_S runs natively).  Host code only.
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace strata::qwen35 {

/// The same to any ggml type `dst_type` (the draft side's matrices as Q4_K, STRATA_Q36_DRAFT_Q4): rows through ggml's
/// dequantizer and its quantizer for that type, no importance matrix.
namespace {

void quant128_b70w4(const float* w, uint16_t& scale_h, int& zero_out, uint8_t q_out[128]) {
    float wmin = w[0], wmax = w[0];
    for (int k = 1; k < 128; ++k) {
        wmin = std::min(wmin, w[k]);
        wmax = std::max(wmax, w[k]);
    }
    const float lo = std::min(0.0f, wmin), hi = std::max(0.0f, wmax);
    if (hi - lo <= 0.0f) {
        scale_h = 0;
        zero_out = 8;
        std::fill(q_out, q_out + 128, uint8_t(8));
        return;
    }

    const int z0 = std::clamp((int) std::lround(-15.0f * lo / (hi - lo)), 1, 16);
    float best_err = std::numeric_limits<float>::infinity();
    uint16_t best_sh = 0;
    int best_z = 8;
    std::fill(q_out, q_out + 128, uint8_t(8));

    auto eval_scale = [&](float s_cand, int z) {
        const uint16_t sh = ggml_fp32_to_fp16(s_cand);
        const float sq = ggml_fp16_to_fp32(sh);
        if (sq <= 0.0f) return;
        const float inv = 1.0f / sq;
        uint8_t qt[128];
        float err = 0.0f;
        for (int k = 0; k < 128; ++k) {
            const int q = std::clamp((int) std::lround(w[k] * inv + float(z)), 0, 15);
            qt[k] = (uint8_t) q;
            const float d = w[k] - sq * float(q - z);
            err += d * d;
        }
        if (err < best_err) {
            best_err = err;
            best_sh = sh;
            best_z = z;
            std::copy(qt, qt + 128, q_out);
        }
    };

    for (int dz = -1; dz <= 1; ++dz) {
        const int z = std::clamp(z0 + dz, 1, 16);
        if (dz == 0 || z != z0) {
            if (z >= 15 && hi > 0.0f) continue;
            const float s_mm = std::max(lo < 0.0f ? -lo / float(z) : 0.0f,
                                        (z < 15 && hi > 0.0f) ? hi / float(15 - z) : 0.0f);
            if (s_mm <= 0.0f) continue;
            eval_scale(s_mm, z);

            const float inv = 1.0f / s_mm;
            float swd = 0.0f, sdd = 0.0f;
            for (int k = 0; k < 128; ++k) {
                const int q = std::clamp((int) std::lround(w[k] * inv + float(z)), 0, 15);
                const float qz = float(q - z);
                swd += w[k] * qz;
                sdd += qz * qz;
            }
            if (sdd > 0.0f && swd > 0.0f) {
                eval_scale(swd / sdd, z);
            }
        }
    }

    scale_h = best_sh;
    zero_out = best_z;
}

}   // namespace

bool requant_b70w4(int type, const void* src, int64_t n_in, int64_t n_out, std::vector<uint8_t>& out) {
    if (type < 0 || type >= GGML_TYPE_COUNT || n_in <= 0 || n_out <= 0 || n_in % 256 != 0 || n_out % 16 != 0)
        return false;
    const ggml_type t = (ggml_type) type;
    const ggml_type_traits* tr = ggml_get_type_traits(t);
    if (tr == nullptr || tr->to_float == nullptr || n_in % ggml_blck_size(t) != 0) return false;

    const size_t src_row = ggml_row_size(t, n_in);
    const int64_t T = n_out / 16, C = n_in / 256;
    out.assign((size_t) T * (size_t) C * 544 * sizeof(uint32_t), 0);
    uint32_t* dst = reinterpret_cast<uint32_t*>(out.data());

    const unsigned hw = std::thread::hardware_concurrency();
    const int64_t nthreads = std::min<int64_t>(T, std::min<int64_t>(16, hw > 0 ? hw : 4));

    auto worker = [&](int64_t t_beg, int64_t t_end) {
        std::vector<float> row((size_t) n_in);
        uint8_t q[256];
        for (int64_t tile = t_beg; tile < t_end; ++tile) {
            for (int i = 0; i < 16; ++i) {
                const int64_t r = tile * 16 + i;
                tr->to_float(static_cast<const uint8_t*>(src) + (size_t) r * src_row, row.data(), n_in);
                for (int64_t c = 0; c < C; ++c) {
                    const float* wc = row.data() + c * 256;
                    uint16_t s0 = 0, s1 = 0;
                    int z0 = 8, z1 = 8;
                    quant128_b70w4(wc, s0, z0, q);
                    quant128_b70w4(wc + 128, s1, z1, q + 128);

                    uint32_t* rec = dst + ((size_t) tile * (size_t) C + (size_t) c) * 544;
                    rec[512 + i] = (uint32_t) s0 | ((uint32_t) z0 << 16);
                    rec[512 + 16 + i] = (uint32_t) s1 | ((uint32_t) z1 << 16);

                    for (int p = 0; p < 4; ++p) {
                        const uint8_t* qp = q + p * 64;
                        for (int d = 0; d < 8; ++d) {
                            uint32_t dw = 0;
                            for (int j = 0; j < 4; ++j) {
                                const int b = 4 * d + j;
                                const uint32_t byte = (uint32_t) (qp[b] | (qp[b + 32] << 4));
                                dw |= byte << (8 * j);
                            }
                            rec[128 * p + 16 * d + i] = dw;
                        }
                    }
                }
            }
        }
    };

    if (nthreads <= 1) {
        worker(0, T);
    } else {
        std::vector<std::thread> pool;
        pool.reserve((size_t) nthreads);
        for (int64_t tid = 0; tid < nthreads; ++tid) {
            const int64_t t_beg = (T * tid) / nthreads;
            const int64_t t_end = (T * (tid + 1)) / nthreads;
            pool.emplace_back(worker, t_beg, t_end);
        }
        for (auto& th : pool) th.join();
    }
    return true;
}

bool requant(int type, const void* src, int64_t n_in, int64_t n_out, int dst_type, std::vector<uint8_t>& out) {
    if (type < 0 || type >= GGML_TYPE_COUNT || dst_type < 0 || dst_type >= GGML_TYPE_COUNT || n_in % 32 != 0) return false;
    const ggml_type t = (ggml_type) type, dt = (ggml_type) dst_type;
    const ggml_type_traits* tr = ggml_get_type_traits(t);
    if (tr == nullptr || tr->to_float == nullptr || n_in % ggml_blck_size(t) != 0 || n_in % ggml_blck_size(dt) != 0) return false;
    const size_t src_row = ggml_row_size(t, n_in), dst_row = ggml_row_size(dt, n_in);
    out.assign((size_t) n_out * dst_row, 0);
    ggml_quantize_init(dt);   // (before the threads: its tables are made once)
    const int nt = (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int w = 0; w < nt; ++w)
        th.emplace_back([&, w] {
            std::vector<float> row((size_t) n_in);
            for (int64_t r = w; r < n_out; r += nt) {
                tr->to_float((const uint8_t*) src + (size_t) r * src_row, row.data(), n_in);
                ggml_quantize_chunk(dt, row.data(), out.data() + (size_t) r * dst_row, 0, 1, n_in, nullptr);
            }
        });
    for (auto& x : th) x.join();
    return true;
}

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
