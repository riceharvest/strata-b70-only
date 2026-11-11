// esimd_f16_parity: the FP16 expert products (prefill/esimd_moe.dp.cpp, gate_up_f16 / down_f16: IQ2_S / IQ2_XXS / Q2_0 by their own
// kernels, every other format by launch_x, STRATA_F16X_ANY) against
//   (a) the engine's FP16 chain they replace (iq_dequant_expert_f16 + oneMKL FP16 GEMM, swiglu_interleaved, oneMKL GEMM),
//   (b) an FP64 host reference from the blobs with the FP16 activations (gate/up: H = silu(g) u in FP64; down: from that H),
//   (c) the down product on the FP16 chain's own H (isolates the kernel arithmetic: only the summation order differs),
//   (d) themselves: a second run and the held-group launches at 8, 16 and 64 rows a thread must give the same bits,
// and times the pieces of a group: the FP16 chain (dequantize / gate-up GEMMs / SwiGLU / down GEMMs and the whole chain), the int8
// chain where it takes the formats, and the FP16 products, with TFLOP/s.
//   esimd_f16_parity [mean_rows=320] [experts=16] [reps=10] [seed=1] [gate_up_type=22 down_type=42 | all]
//   all: every format pair below, then one table (the sweep: no model file is needed, the experts are random blocks with sane scales;
//        random, because Xe2 compresses memory and a constant fill would stream faster than real weights).
//   types: gate/up 12 Q4_K, 13 Q5_K, 14 Q6_K, 8 Q8_0, 21 IQ3_S, 18 IQ3_XXS, 17 IQ2_XS, 23 IQ4_XS, 29 IQ1_M, 42 Q2_0, 2 Q4_0, 3 Q4_1,
//          6 Q5_0, 22 IQ2_S, 16 IQ2_XXS; down 20 IQ4_NL, 7 Q5_1, 8 Q8_0, 2 Q4_0, 3 Q4_1, 6 Q5_0, 42 Q2_0.
//   BENCH_NOREF=1: no FP64 reference (timing sweeps).
//   Row counts: gamma-distributed (a few large experts, many small), three edge counts (1, 33, 65), scaled to the mean.
// Exit status 1 when (c) is off by more than 1e-5 rms, (b) is more than 1.25x the FP16 chain's own error, or (d) differs.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/iq_dequant_expert.hpp"
#include "strata/prefill/esimd_moe.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/sycl_queue.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define GGML_TABLE_BEGIN(type, name, size) static const type name[size] = {
#define GGML_TABLE_END() };
#include "../prefill/iq_tables.inc"

namespace i8 = strata::prefill::i8;
static uint16_t f2h(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }
static float h2f(uint16_t h) { return (float) sycl::bit_cast<sycl::half>(h); }

static int block_bytes(int t) {
    switch (t) {
        case 2: return 18;
        case 3: return 20;
        case 6: return 22;
        case 7: return 24;
        case 8: return 34;
        case 12: return 144;
        case 13: return 176;
        case 14: return 210;
        case 16: return 66;
        case 17: return 74;
        case 18: return 98;
        case 20: return 18;
        case 21: return 110;
        case 22: return 82;
        case 23: return 136;
        case 29: return 56;
        default: return 18;   // 42 Q2_0
    }
}
static int block_vals(int t) {
    switch (t) {
        case 2: case 3: case 6: case 7: case 8: case 20: return 32;
        case 42: return 64;
        default: return 256;
    }
}
static const char* type_name(int t) {
    switch (t) {
        case 2: return "Q4_0";
        case 3: return "Q4_1";
        case 6: return "Q5_0";
        case 7: return "Q5_1";
        case 8: return "Q8_0";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS";
        case 20: return "IQ4_NL";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 23: return "IQ4_XS";
        case 29: return "IQ1_M";
        case 42: return "Q2_0";
        default: return "?";
    }
}
static double hval(const uint8_t* p) { uint16_t h; std::memcpy(&h, p, 2); return h2f(h); }
// llama.cpp's get_scale_min_k4
static void k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
// a block's FP16 fields set to sane values (the rest of the block stays random bytes): weights of roughly 0.05-0.5 at the largest codes
static void sane_block(int t, uint8_t* blk, std::mt19937& rng) {
    auto put = [&](int off, float v) { const uint16_t h = f2h(v); std::memcpy(blk + off, &h, 2); };
    const float j1 = (float) (rng() % 50), j2 = (float) (rng() % 50);
    switch (t) {
        case 12: case 13: put(0, 0.00012f + 0.000004f * j1); put(2, 0.00010f + 0.000004f * j2); break;   // d, dmin
        case 14: put(208, 0.00008f + 0.000002f * j1); break;                                             // d (the block's last field)
        case 3: case 7: put(0, 0.0004f + 0.00002f * j1); put(2, -0.004f - 0.0002f * j2); break;          // d, m
        case 8: case 20: put(0, 0.0001f + 0.000004f * j1); break;                                        // codes up to 127
        case 23: put(0, 0.00008f + 0.000001f * j1); break;                                               // codes up to 127 x 31
        case 29: {   // IQ1_M has no d field: the FP16 scale is the top nibbles of the four 16-bit scale words
            const uint16_t h = f2h(0.004f + 0.0002f * j1);
            uint16_t sc[4];
            std::memcpy(sc, blk + 48, 8);
            for (int k = 0; k < 4; ++k) sc[k] = (uint16_t) ((sc[k] & 0x0fff) | (((h >> (4 * k)) & 0xf) << 12));
            std::memcpy(blk + 48, sc, 8);
            break;
        }
        default: put(0, 0.0004f + 0.00002f * j1); break;
    }
}

// host decode of one row (K values) -> double (the exact values; the FP16 chain rounds them to FP16): llama.cpp's dequantize_row_*
static void dec_row(int t, const uint8_t* row, int K, double* out) {
    const int bb = block_bytes(t), bv = block_vals(t);
    auto sgn = [](uint8_t sb, int j, int g) { return (double) (((sb >> j) & 1) ? -g : g); };
    for (int b = 0; b < K / bv; ++b) {
        const uint8_t* blk = row + (size_t) b * bb;
        const double d = t == 14 ? hval(blk + 208) : hval(blk);
        double* o = out + (size_t) b * bv;
        if (t == 22) {
            const uint8_t *qs = blk + 2, *qh = blk + 66, *scl = blk + 74, *sg = blk + 34;
            for (int ib = 0; ib < 8; ++ib)
                for (int l = 0; l < 4; ++l) {
                    const int idx = qs[4 * ib + l] | ((qh[ib] << (8 - 2 * l)) & 0x300);
                    const uint8_t* g = (const uint8_t*) &iq2s_grid[idx];
                    const double dd = d * (0.5 + ((scl[ib] >> (4 * (l / 2))) & 0xf)) * 0.25;
                    const uint8_t sb = sg[4 * ib + l];
                    for (int j = 0; j < 8; ++j) o[ib * 32 + 8 * l + j] = dd * sgn(sb, j, g[j]);
                }
        } else if (t == 16) {
            const uint8_t* qs = blk + 2;
            for (int ib = 0; ib < 8; ++ib) {
                uint32_t aux; std::memcpy(&aux, qs + 8 * ib + 4, 4);
                const double dd = d * (0.5 + (aux >> 28)) * 0.25;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t* g = (const uint8_t*) &iq2xxs_grid[qs[8 * ib + l]];
                    const uint8_t sb = ksigns_iq2xs[(aux >> (7 * l)) & 127];
                    for (int j = 0; j < 8; ++j) o[ib * 32 + 8 * l + j] = dd * sgn(sb, j, g[j]);
                }
            }
        } else if (t == 17) {   // IQ2_XS: d, 32 x 16 bits (9 grid index, 7 signs), scales[8]
            const uint8_t* scl = blk + 66;
            for (int ib = 0; ib < 8; ++ib)
                for (int l = 0; l < 4; ++l) {
                    uint16_t qv; std::memcpy(&qv, blk + 2 + 2 * (4 * ib + l), 2);
                    const uint8_t* g = (const uint8_t*) &iq2xs_grid[qv & 511];
                    const double dd = d * (0.5 + ((scl[ib] >> (4 * (l / 2))) & 0xf)) * 0.25;
                    const uint8_t sb = ksigns_iq2xs[qv >> 9];
                    for (int j = 0; j < 8; ++j) o[ib * 32 + 8 * l + j] = dd * sgn(sb, j, g[j]);
                }
        } else if (t == 18) {   // IQ3_XXS: d, qs[64], 8 x 32 bits (4 x 7 signs, 4 scale)
            const uint8_t* qs = blk + 2;
            for (int ib = 0; ib < 8; ++ib) {
                uint32_t aux; std::memcpy(&aux, blk + 66 + 4 * ib, 4);
                const double dd = d * (0.5 + (aux >> 28)) * 0.5;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t* g1 = (const uint8_t*) &iq3xxs_grid[qs[8 * ib + 2 * l]];
                    const uint8_t* g2 = (const uint8_t*) &iq3xxs_grid[qs[8 * ib + 2 * l + 1]];
                    const uint8_t sb = ksigns_iq2xs[(aux >> (7 * l)) & 127];
                    for (int j = 0; j < 4; ++j) {
                        o[ib * 32 + 8 * l + j] = dd * sgn(sb, j, g1[j]);
                        o[ib * 32 + 8 * l + 4 + j] = dd * sgn(sb, j + 4, g2[j]);
                    }
                }
            }
        } else if (t == 21) {   // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]
            const uint8_t *qs = blk + 2, *qh = blk + 66, *sg = blk + 74, *scl = blk + 106;
            for (int ib = 0; ib < 8; ++ib) {
                const double dd = d * (1 + 2 * ((scl[ib / 2] >> (4 * (ib % 2))) & 0xf));
                for (int l = 0; l < 4; ++l) {
                    const uint8_t* g1 = (const uint8_t*) &iq3s_grid[qs[8 * ib + 2 * l] | ((qh[ib] << (8 - 2 * l)) & 256)];
                    const uint8_t* g2 = (const uint8_t*) &iq3s_grid[qs[8 * ib + 2 * l + 1] | ((qh[ib] << (7 - 2 * l)) & 256)];
                    const uint8_t sb = sg[4 * ib + l];
                    for (int j = 0; j < 4; ++j) {
                        o[ib * 32 + 8 * l + j] = dd * sgn(sb, j, g1[j]);
                        o[ib * 32 + 8 * l + 4 + j] = dd * sgn(sb, j + 4, g2[j]);
                    }
                }
            }
        } else if (t == 29) {   // IQ1_M: qs[32], qh[16], scales[8] (4 x 16 bits: 3-bit scales, the top nibbles the FP16 d)
            const uint8_t *qs = blk, *qh = blk + 32;
            uint16_t sc[4];
            std::memcpy(sc, blk + 48, 8);
            const uint16_t fb = (uint16_t) ((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
            const double d0 = h2f(fb);
            for (int ib = 0; ib < 8; ++ib)
                for (int l = 0; l < 4; ++l) {
                    const int ib16 = 2 * ib + l / 2;
                    const double dd = d0 * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 7) + 1);
                    const int qhb = qh[2 * ib + l / 2] >> (4 * (l % 2));
                    const double delta = (qhb & 8) ? -1.125 : -0.875;
                    const uint32_t g = iq1s_grid_gpu[qs[4 * ib + l] | ((qhb & 7) << 8)];
                    for (int j = 0; j < 8; ++j) {
                        const int qv = j < 4 ? (g >> (8 * j)) & 0xf : (g >> (8 * (j - 4) + 4)) & 0xf;
                        o[ib * 32 + 8 * l + j] = dd * (qv + delta);
                    }
                }
        } else if (t == 23) {   // IQ4_XS: d, scales_h, scales_l[4], qs[128]
            uint16_t sh; std::memcpy(&sh, blk + 2, 2);
            const uint8_t *sl = blk + 4, *qs = blk + 8;
            for (int ib = 0; ib < 8; ++ib) {
                const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
                const double dl = d * (ls - 32);
                for (int j = 0; j < 16; ++j) {
                    o[ib * 32 + j] = dl * kvalues_iq4nl[qs[16 * ib + j] & 0xf];
                    o[ib * 32 + 16 + j] = dl * kvalues_iq4nl[qs[16 * ib + j] >> 4];
                }
            }
        } else if (t == 12 || t == 13) {   // Q4_K: d, dmin, scales[12], qs[128]; Q5_K: d, dmin, scales[12], qh[32], qs[128]
            const double dmin = hval(blk + 2);
            const uint8_t *scl = blk + 4, *qh = blk + 16, *ql = blk + (t == 12 ? 16 : 48);
            for (int is = 0; is < 4; ++is) {
                int sc1, m1, sc2, m2;
                k4(2 * is, scl, sc1, m1);
                k4(2 * is + 1, scl, sc2, m2);
                for (int l = 0; l < 32; ++l) {
                    const int h1 = t == 13 && (qh[l] & (1 << (2 * is))) ? 16 : 0, h2 = t == 13 && (qh[l] & (2 << (2 * is))) ? 16 : 0;
                    o[64 * is + l] = d * sc1 * ((ql[32 * is + l] & 0xF) + h1) - dmin * m1;
                    o[64 * is + 32 + l] = d * sc2 * ((ql[32 * is + l] >> 4) + h2) - dmin * m2;
                }
            }
        } else if (t == 14) {   // Q6_K: ql[128], qh[64], scales[16] (int8), d
            const int8_t* sc = (const int8_t*) (blk + 192);
            for (int n = 0; n < 2; ++n) {
                const uint8_t *ql = blk + 64 * n, *qh = blk + 128 + 32 * n;
                for (int l = 0; l < 32; ++l) {
                    const int is = 8 * n + l / 16;
                    const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32, q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32, q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    o[128 * n + l] = d * sc[is] * q1;
                    o[128 * n + 32 + l] = d * sc[is + 2] * q2;
                    o[128 * n + 64 + l] = d * sc[is + 4] * q3;
                    o[128 * n + 96 + l] = d * sc[is + 6] * q4;
                }
            }
        } else if (t == 2 || t == 3 || t == 6 || t == 7) {   // d [, m] [, qh[4]], qs[16]: value j the low nibble of byte j, j + 16 its high
            const bool has_m = t == 3 || t == 7, has_h = t == 6 || t == 7;
            const double mo = has_m ? hval(blk + 2) : 0.0;
            const uint8_t* p = blk + 2 + (has_m ? 2 : 0);
            uint32_t qh = 0;
            if (has_h) { std::memcpy(&qh, p, 4); p += 4; }
            const int bias = t == 2 ? 8 : t == 6 ? 16 : 0;
            for (int j = 0; j < 16; ++j) {
                const int x0 = ((p[j] & 0xF) | (((qh >> j) & 1) << 4)) - bias, x1 = ((p[j] >> 4) | (((qh >> (j + 16)) & 1) << 4)) - bias;
                o[j] = d * x0 + mo;
                o[j + 16] = d * x1 + mo;
            }
        } else if (t == 8) {
            for (int j = 0; j < 32; ++j) o[j] = d * (int8_t) blk[2 + j];
        } else if (t == 20) {
            for (int j = 0; j < 16; ++j) {
                o[j] = d * kvalues_iq4nl[blk[2 + j] & 0xf];
                o[j + 16] = d * kvalues_iq4nl[blk[2 + j] >> 4];
            }
        } else {   // 42 Q2_0
            for (int i = 0; i < 64; ++i) o[i] = d * (((blk[2 + i / 4] >> ((i % 4) * 2)) & 3) - 1);
        }
    }
}

struct CaseResult { int gt = 0, dt = 0; bool ok = false; double chain = 0, i8 = 0, prod = 0, gu = 0, dn = 0; };

static bool run_case(sycl::queue* q, dpct::blas::descriptor_ptr hd, int gt, int dt, int mean, int n, int reps, int seed, CaseResult& res) {
    const int K = 2560, NFF = 640, NE = 2560;
    std::mt19937 rng(seed);
    std::vector<int32_t> cnt(n);
    std::gamma_distribution<double> gd(1.3, 1.0);
    std::vector<double> w(n);
    for (auto& x : w) x = gd(rng);
    const int nedge = n >= 6 ? 3 : 0;
    const int edge[3] = {1, 33, 65};
    double wsum = 0;
    for (int i = nedge; i < n; ++i) wsum += w[i];
    const double rest = (double) mean * n - (nedge ? 99 : 0);
    int tot = 0;
    for (int i = 0; i < n; ++i) { cnt[i] = i < nedge ? edge[i] : std::max(1, (int) std::lround(w[i] / wsum * rest)); tot += cnt[i]; }
    std::printf("  rows per expert:");
    for (int c : cnt) std::printf(" %d", c);
    std::printf("  (total %d, mean %.0f)\n", tot, (double) tot / n);
    i8::Layout lo;
    i8::make_layout(lo, cnt.data(), n);

    const size_t rowb_gu = (size_t) (K / block_vals(gt)) * block_bytes(gt), rowb_dn = (size_t) (NFF / block_vals(dt)) * block_bytes(dt);
    const size_t gate_b = NFF * rowb_gu, down_b = NE * rowb_dn, blob_b = 2 * gate_b + down_b;
    std::vector<std::vector<uint8_t>> hb(n);
    std::vector<uint8_t*> db(n);
    for (int i = 0; i < n; ++i) {
        hb[i].resize(blob_b);
        for (auto& x : hb[i]) x = (uint8_t) rng();
        for (size_t o = 0; o < 2 * gate_b; o += block_bytes(gt)) sane_block(gt, &hb[i][o], rng);
        for (size_t o = 2 * gate_b; o < blob_b; o += block_bytes(dt)) sane_block(dt, &hb[i][o], rng);
        db[i] = sycl::malloc_device<uint8_t>(blob_b, *q);   // (no slack: the kernels must not read past the blob)
        q->memcpy(db[i], hb[i].data(), blob_b);
    }
    std::vector<uint16_t> hx((size_t) tot * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& x : hx) { float v = nd(rng) * 0.5f; if (rng() % 97 == 0) v *= 12.f; x = f2h(v); }   // a few outliers
    uint16_t* dxh = sycl::malloc_device<uint16_t>(hx.size(), *q);
    q->memcpy(dxh, hx.data(), hx.size() * 2);
    float* GUr = sycl::malloc_device<float>((size_t) tot * 2 * NFF, *q);
    float *Dmr = sycl::malloc_device<float>((size_t) tot * NE, *q), *Dmn = sycl::malloc_device<float>((size_t) tot * NE, *q);
    float* Dmc = sycl::malloc_device<float>((size_t) tot * NE, *q);
    uint16_t *Hr = sycl::malloc_device<uint16_t>((size_t) tot * NFF, *q), *Hn = sycl::malloc_device<uint16_t>((size_t) tot * NFF, *q);
    uint16_t* dwgu = sycl::malloc_device<uint16_t>((size_t) 2 * NFF * K, *q);
    uint16_t* dwd = sycl::malloc_device<uint16_t>((size_t) NE * NFF, *q);
    i8::Workspace ws8;
    if (!i8::workspace(lo, K, NFF, ws8, q)) { std::printf("no int8 workspace\n"); std::exit(2); }
    q->wait();
    std::vector<const uint8_t*> pb(n);
    for (int i = 0; i < n; ++i) pb[i] = db[i];
    const float one = 1.f, zero = 0.f;
    auto gemm_gu = [&](int i, int64_t o) {
        dpct::blas::gemm(hd, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, 2 * NFF, cnt[i], K, &one, dwgu,
                         dpct::library_data_t::real_half, K, dxh + o * K, dpct::library_data_t::real_half, K, &zero,
                         GUr + o * 2 * NFF, dpct::library_data_t::real_float, 2 * NFF, dpct::compute_type::f32);
    };
    auto gemm_dn = [&](int i, int64_t o, const uint16_t* H, float* D) {
        dpct::blas::gemm(hd, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, NE, cnt[i], NFF, &one, dwd,
                         dpct::library_data_t::real_half, NFF, H + o * NFF, dpct::library_data_t::real_half, NFF, &zero,
                         D + o * NE, dpct::library_data_t::real_float, NE, dpct::compute_type::f32);
    };
    auto deq = [&](int i) { strata::kernels::iq_dequant_expert_f16(gt, dt, db[i], db[i] + gate_b, db[i] + 2 * gate_b, NFF, K, dwgu, dwd, q); };
    auto chain = [&]() {
        int64_t o = 0;
        for (int i = 0; i < n; ++i) {
            deq(i);
            gemm_gu(i, o);
            strata::prefill::swiglu_interleaved(GUr + o * 2 * NFF, Hr + o * NFF, cnt[i], q);
            gemm_dn(i, o, Hr, Dmr);
            o += cnt[i];
        }
    };
    auto newp = [&]() {
        if (!i8::gate_up_f16(gt, pb.data(), gate_b, NFF, lo, K, dxh, Hn, q)) { std::printf("gate_up_f16 refused (type %d)\n", gt); std::exit(2); }
        if (!i8::down_f16(dt, pb.data(), 2 * gate_b, NE, lo, NFF, Hn, Dmn, q)) { std::printf("down_f16 refused (type %d)\n", dt); std::exit(2); }
    };
    const bool has_i8 = (gt == 21 || gt == 18 || gt == 22 || gt == 17 || gt == 16 || gt == 23) && (dt == 20 || dt == 42 || dt == 8);
    auto int8 = [&]() {
        i8::quantize_rows(dxh, lo, K, ws8.xq, ws8.dx, q);
        if (!i8::gate_up(gt, pb.data(), gate_b, NFF, lo, K, ws8.xq, ws8.dx, GUr, q)) { std::printf("int8 gate_up refused (type %d)\n", gt); std::exit(2); }
        i8::swiglu_quantize(GUr, lo, NFF, ws8.hq, ws8.dh, q);
        if (!i8::down(dt, pb.data(), 2 * gate_b, NE, lo, NFF, ws8.hq, ws8.dh, Dmc, q)) { std::printf("int8 down refused (type %d)\n", dt); std::exit(2); }
    };
    auto tm = [&](auto f) {
        f(); q->wait();
        const auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.3) { f(); q->wait(); }   // GPU clocks up
        double best = 1e30;
        for (int k = 0; k < 3; ++k) {
            const auto a = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) f();
            q->wait();
            best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count() / reps);
        }
        return best;
    };

    chain(); q->wait();
    newp(); q->wait();
    // (c): the new down product on the FP16 chain's H
    if (!i8::down_f16(dt, pb.data(), 2 * gate_b, NE, lo, NFF, Hr, Dmc, q)) std::exit(2);
    q->wait();
    std::vector<uint16_t> hHr((size_t) tot * NFF), hHn((size_t) tot * NFF);
    std::vector<float> hDr((size_t) tot * NE), hDn((size_t) tot * NE), hDc((size_t) tot * NE);
    q->memcpy(hHr.data(), Hr, hHr.size() * 2);
    q->memcpy(hHn.data(), Hn, hHn.size() * 2);
    q->memcpy(hDr.data(), Dmr, hDr.size() * 4);
    q->memcpy(hDn.data(), Dmn, hDn.size() * 4);
    q->memcpy(hDc.data(), Dmc, hDc.size() * 4).wait();
    bool ok = true;
    {
        double e2 = 0, r2 = 0;
        size_t ulp_diff = 0;
        for (size_t i = 0; i < hHr.size(); ++i) {
            const double a = h2f(hHr[i]), b = h2f(hHn[i]);
            e2 += (a - b) * (a - b); r2 += a * a;
            ulp_diff += hHr[i] != hHn[i];
        }
        double f2 = 0, s2 = 0;
        for (size_t i = 0; i < hDr.size(); ++i) { f2 += (double) (hDr[i] - hDc[i]) * (hDr[i] - hDc[i]); s2 += (double) hDr[i] * hDr[i]; }
        double g2 = 0;
        for (size_t i = 0; i < hDr.size(); ++i) g2 += (double) (hDr[i] - hDn[i]) * (hDr[i] - hDn[i]);
        const bool okc = std::sqrt(f2 / s2) < 1e-5;
        ok = ok && okc;
        std::printf("  vs the FP16 chain: H rms rel diff %.2e (%zu of %zu FP16 values differ) | down on the chain's H rms rel diff %.2e %s | "
                    "chain end to end %.2e\n", std::sqrt(e2 / r2), ulp_diff, hHr.size(), std::sqrt(f2 / s2), okc ? "OK" : "FAIL",
                    std::sqrt(g2 / s2));
    }
    {   // (d) the same bits from a second run and from the held-group launches at every row tile (stale buffers: 0xFF bytes first)
        uint16_t* Hd = sycl::malloc_device<uint16_t>((size_t) tot * NFF, *q);
        float* Dd = sycl::malloc_device<float>((size_t) tot * NE, *q);
        std::vector<uint16_t> hHd((size_t) tot * NFF);
        std::vector<float> hDd((size_t) tot * NE);
        bool okd = true;
        auto same = [&](const char* what) {
            q->memcpy(hHd.data(), Hd, hHd.size() * 2);
            q->memcpy(hDd.data(), Dd, hDd.size() * 4).wait();
            const bool eh = std::memcmp(hHd.data(), hHn.data(), hHd.size() * 2) == 0, ed = std::memcmp(hDd.data(), hDn.data(), hDd.size() * 4) == 0;
            if (!eh || !ed) { okd = false; std::printf("  %s: %s%s differ from the first run\n", what, eh ? "" : "H ", ed ? "" : "down"); }
        };
        auto stale = [&]() {
            q->memset(Hd, 0xFF, hHd.size() * 2);
            q->memset(Dd, 0xFF, hDd.size() * 4).wait();
        };
        stale();
        if (!i8::gate_up_f16(gt, pb.data(), gate_b, NFF, lo, K, dxh, Hd, q) || !i8::down_f16(dt, pb.data(), 2 * gate_b, NE, lo, NFF, Hd, Dd, q)) std::exit(2);
        same("a second run");
        if (n <= i8::kMaxGroupH) {
            for (int mt : {1, 2, 8}) {
                stale();
                if (!i8::gate_up_f16_n(gt, pb.data(), gate_b, NFF, cnt.data(), n, K, dxh, Hd, q, mt) ||
                    !i8::down_f16_n(dt, pb.data(), 2 * gate_b, NE, cnt.data(), n, NFF, Hd, Dd, q, mt))
                    std::exit(2);
                char what[64];
                std::snprintf(what, sizeof what, "the held-group launches, %d rows a thread", 8 * mt);
                same(what);
            }
        }
        ok = ok && okd;
        std::printf("  a second run and the held-group launches (8, 16, 64 rows a thread): %s\n", okd ? "the same bits OK" : "FAIL");
        sycl::free(Hd, *q);
        sycl::free(Dd, *q);
    }
    if (std::getenv("BENCH_NOREF") == nullptr) {   // (b) FP64 (BENCH_NOREF=1: skipped, for timing sweeps)
        std::vector<double> wgu((size_t) 2 * NFF * K), wd((size_t) NE * NFF), H(NFF), x(K);
        double eh_r = 0, eh_n = 0, rh = 0, ed_r = 0, ed_n = 0, rd = 0;
        int smp = 0;
        for (int e = 0; e < n; ++e) {
            for (int j = 0; j < 2 * NFF; ++j) dec_row(gt, &hb[e][(size_t) j * rowb_gu], K, &wgu[(size_t) j * K]);   // gate rows, then up
            for (int v = 0; v < NE; v += 41) dec_row(dt, &hb[e][2 * gate_b + (size_t) v * rowb_dn], NFF, &wd[(size_t) v * NFF]);
            for (int rl = 0; rl < cnt[e]; rl += (cnt[e] > 24 ? 11 : 1)) {
                const int row = lo.row0[e] + rl;
                for (int k = 0; k < K; ++k) x[k] = h2f(hx[(size_t) row * K + k]);
                for (int j = 0; j < NFF; ++j) {
                    const double* wg = &wgu[(size_t) j * K];
                    const double* wu = &wgu[(size_t) (NFF + j) * K];
                    double g = 0, u = 0;
                    for (int k = 0; k < K; ++k) { g += x[k] * wg[k]; u += x[k] * wu[k]; }
                    H[j] = g / (1.0 + std::exp(-g)) * u;
                    const double a = h2f(hHr[(size_t) row * NFF + j]), b = h2f(hHn[(size_t) row * NFF + j]);
                    eh_r += (a - H[j]) * (a - H[j]); eh_n += (b - H[j]) * (b - H[j]); rh += H[j] * H[j];
                }
                for (int v = 0; v < NE; v += 41) {
                    double ref = 0;
                    for (int k = 0; k < NFF; ++k) ref += H[k] * wd[(size_t) v * NFF + k];
                    const double a = hDr[(size_t) row * NE + v], b = hDn[(size_t) row * NE + v];
                    ed_r += (a - ref) * (a - ref); ed_n += (b - ref) * (b - ref); rd += ref * ref;
                }
                ++smp;
            }
        }
        const double hr = std::sqrt(eh_r / rh), hn = std::sqrt(eh_n / rh), dr = std::sqrt(ed_r / rd), dn = std::sqrt(ed_n / rd);
        const bool okb = hn <= 1.25 * hr + 1e-6 && dn <= 1.25 * dr + 1e-6;
        ok = ok && okb;
        std::printf("  vs FP64 (%d rows): H rms rel err chain %.3e new %.3e | down chain %.3e new %.3e  %s\n", smp, hr, hn, dr, dn,
                    okb ? "OK" : "FAIL");
    }
    // timings (us per group) and TFLOP/s
    const double fl_gu = 2.0 * tot * 2 * NFF * K, fl_dn = 2.0 * tot * NE * NFF;
    const double t_chain = tm(chain);
    const double t_deq = tm([&] { for (int i = 0; i < n; ++i) deq(i); });
    const double t_ggu = tm([&] { int64_t o = 0; for (int i = 0; i < n; ++i) { gemm_gu(i, o); o += cnt[i]; } });
    const double t_sw = tm([&] { int64_t o = 0; for (int i = 0; i < n; ++i) { strata::prefill::swiglu_interleaved(GUr + o * 2 * NFF, Hr + o * NFF, cnt[i], q); o += cnt[i]; } });
    const double t_gdn = tm([&] { int64_t o = 0; for (int i = 0; i < n; ++i) { gemm_dn(i, o, Hr, Dmr); o += cnt[i]; } });
    const double t_i8 = has_i8 ? tm(int8) : 0.0;
    const double t_ngu = tm([&] { i8::gate_up_f16(gt, pb.data(), gate_b, NFF, lo, K, dxh, Hn, q); });
    const double t_ndn = tm([&] { i8::down_f16(dt, pb.data(), 2 * gate_b, NE, lo, NFF, Hn, Dmn, q); });
    const double t_new = tm(newp);
    const double fl = fl_gu + fl_dn;
    std::printf("  FP16 chain %.0f us (%.1f TFLOP/s): dequant %.0f, gate/up GEMMs %.0f (%.1f TFLOP/s), SwiGLU %.0f, down GEMMs %.0f "
                "(%.1f TFLOP/s), gaps %.0f\n", t_chain, fl / t_chain * 1e-6, t_deq, t_ggu, fl_gu / t_ggu * 1e-6, t_sw, t_gdn,
                fl_dn / t_gdn * 1e-6, t_chain - t_deq - t_ggu - t_sw - t_gdn);
    if (has_i8) std::printf("  int8 chain %.0f us (%.1f TFLOP/s)\n", t_i8, fl / t_i8 * 1e-6);
    else std::printf("  int8 chain: no kernels for these formats\n");
    std::printf("  FP16 products %.0f us (%.1f TFLOP/s): gate/up %.0f (%.1f TFLOP/s), down %.0f (%.1f TFLOP/s)  -> %.2fx the FP16 chain, %.2fx int8\n",
                t_new, fl / t_new * 1e-6, t_ngu, fl_gu / t_ngu * 1e-6, t_ndn, fl_dn / t_ndn * 1e-6, t_chain / t_new, t_i8 / t_new);
    res.gt = gt; res.dt = dt; res.ok = ok;
    res.chain = fl / t_chain * 1e-6; res.i8 = has_i8 ? fl / t_i8 * 1e-6 : 0.0; res.prod = fl / t_new * 1e-6;
    res.gu = fl_gu / t_ngu * 1e-6; res.dn = fl_dn / t_ndn * 1e-6;
    for (int i = 0; i < n; ++i) sycl::free(db[i], *q);
    sycl::free(dxh, *q); sycl::free(GUr, *q); sycl::free(Dmr, *q); sycl::free(Dmn, *q); sycl::free(Dmc, *q);
    sycl::free(Hr, *q); sycl::free(Hn, *q); sycl::free(dwgu, *q); sycl::free(dwd, *q);
    return ok;
}

int main(int argc, char** argv) {
    const int mean = argc > 1 ? std::atoi(argv[1]) : 320;
    const int n = argc > 2 ? std::atoi(argv[2]) : 16;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 10;
    const int seed = argc > 4 ? std::atoi(argv[4]) : 1;
    setenv("STRATA_F16X_ANY", "1", 0);   // the other formats' kernel (launch_x) is what this program checks; =0 in the environment keeps it off
    sycl::queue* q = &dpct::get_in_order_queue();
    std::vector<std::pair<int, int>> cases = {{22, 42}, {16, 42}};
    const bool all = argc > 5 && std::strcmp(argv[5], "all") == 0;
    if (all)   // every gate/up and down pair a layer has in Unsloth's files of the model (12 of them, two revisions) and in the GSQ-RCO
               // packs (tools/quant_formats.py --hf, from the headers), then the two formats of launch_x no file has (Q4_0, Q4_1)
        cases = {{16, 42}, {22, 42}, {17, 42}, {16, 20}, {22, 20}, {17, 20}, {22, 6},      // IQ2_XXS / IQ2_S / IQ2_XS with Q2_0, IQ4_NL, Q5_0
                 {18, 42}, {18, 20}, {18, 6}, {23, 20}, {23, 6}, {23, 8},                  // IQ3_XXS, IQ4_XS
                 {21, 42}, {21, 20}, {21, 8}, {29, 42}, {42, 42},                          // IQ3_S, IQ1_M, Q2_0 (the earlier packs)
                 {12, 7}, {12, 8}, {13, 8}, {14, 8}, {8, 8},                               // the K-quants and Q8_0
                 {2, 2}, {3, 3}};
    else if (argc > 6) cases = {{std::atoi(argv[5]), std::atoi(argv[6])}};
    if (!i8::f16_handles(22, 42, 640, 2560, q)) { std::printf("esimd_f16_parity: SKIP (no ESIMD DPAS device)\n"); return 0; }
    dpct::blas::descriptor_ptr hd = new dpct::blas::descriptor();
    hd->set_queue(q);
    int fails = 0;
    std::vector<CaseResult> rs;
    for (auto [gt, dt] : cases) {
        std::printf("gate/up %s (%d), down %s (%d), mean %d rows, %d experts\n", type_name(gt), gt, type_name(dt), dt, mean, n);
        if (!i8::f16_handles(gt, dt, 640, 2560, q)) { std::printf("  not handled (STRATA_F16X_ANY=0, or a format without FP16 products)\n"); ++fails; continue; }
        CaseResult r;
        fails += !run_case(q, hd, gt, dt, mean, n, reps, seed, r);
        rs.push_back(r);
    }
    if (rs.size() > 1) {
        std::printf("\nTFLOP/s at a mean of %d rows, %d experts (FP16 chain = dequantize + oneMKL, what these formats take without the products)\n", mean, n);
        std::printf("%-9s %-8s %6s %10s %6s %9s %8s %6s %10s\n", "gate/up", "down", "checks", "FP16 chain", "int8", "products", "gate/up", "down", "x chain");
        for (const auto& r : rs)
            std::printf("%-9s %-8s %6s %10.1f %6.1f %9.1f %8.1f %6.1f %10.2f\n", type_name(r.gt), type_name(r.dt), r.ok ? "OK" : "FAIL", r.chain, r.i8,
                        r.prod, r.gu, r.dn, r.prod / r.chain);
    }
    std::printf("%s\n", fails ? "FAILED" : "all OK");
    return fails ? 1 : 0;
}
