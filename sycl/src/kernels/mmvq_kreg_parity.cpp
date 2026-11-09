// src/kernels/mmvq_kreg_parity.cpp - the KREG small-batch MMVQ kernels (STRATA_MMVQ_KREG=1) against a float64
// reference, against the default kernels and against themselves one column at a time.
//
//     mmvq_kreg_parity [--selftest]                      the checks below (ctest runs it with --selftest)
//     mmvq_kreg_parity --bench [types] [ncols] [shape]   us per call, default kernels against KREG
//                                                        (types=12,13,14,23 ncols=1,2,3,4,6 shape=all|qkv|...|head)
//
// Everything goes through the public native_mmvq; native_mmvq_set_kreg switches the KREG kernels on and off (the
// STRATA_MMVQ_KREG latch is not read). For Q4_K, Q5_K, Q6_K and IQ4_XS at 1-8 columns:
//
//   1. KREG against a float64 reference that dequantizes the SAME weight blocks (ggml's dequantize_row_* formulas)
//      and the SAME Q8_1 bytes the kernels read, so only the kernels' FP32 arithmetic is under test. Bound per
//      output: |gpu - ref| <= 2e-5 * sum_i |w_i x_i| (an FP32 chain of ~25 roundings stays under 2e-6 of that sum).
//   2. the default kernels against the same reference and the same bound: the bound is one the kernels the engine
//      runs today meet.
//   3. KREG differs from the default kernels in some output of every case with n_in >= 2560 (it sums in another
//      order). If nothing differs, the call never reached KREG (a fallback) and checks 1 and 4 tested the default
//      kernels instead.
//   4. a T-column KREG call bitwise equal to T one-column KREG calls: native_mmvq_set_multi_exact's contract ("every
//      column bitwise equal to a single-column call"), which the verify window's bit-exactness rests on.
//   5. power: at 8 columns, row 0 of the KREG output against a reference with the scale of sub-block 0 of row 0's
//      first block one step off must fail the bound in some column. A kernel that decodes one sub-block scale wrong
//      would pass a bound that cannot see this.
//   6. STRATA_XE2_KQ (native_mmvq_set_xe2_kq): the KREG call again with STRATA_XE2_KQ=1, bitwise the KREG output (the
//      persistent grid and the 256-register Q5_K tiles change no sum). Over the run some launch must have taken the
//      grid and some the tile (native_mmvq_xe2_kq_seen), or the check has compared KREG with itself.
//   7. STRATA_XE2_KQ part 16: native_mmvq_pair on two Q8_0 matrices at 1-8 columns must run and be bitwise the two
//      native_mmvq calls.
//   8. STRATA_XE2_KQ part 8 (opt-in, not in 1): a planar Q6_K matrix (native_q6_k_planar_adopt) at 5-8 columns, the
//      launch with the part on (256 registers from 6 columns) bitwise the launch with it off.
//
// THE DATA. Random weight bytes, except each block's fp16 scales (d, and dmin for Q4_K/Q5_K), which are normal fp16
// values in +-[2^-10, 2^-5) as in mmvq_multi_parity, so every output is finite and a non-finite one fails. Random
// bytes matter for --bench too: Xe2 compresses memory, and a constant fill streams faster than real weights.
// Activations are N(0, 1), quantized once by native_quantize_q8_1 and shared by every call.
//
// --bench: NMAT distinct weight copies per graph (a 640 MB pool, so L2 never holds a matrix), in-order queue graph
// replay, median of 7-11 replays; prints us per call, GB/s of weight bytes and the ratio.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

namespace k = strata::kernels;
namespace sx = sycl::ext::oneapi::experimental;

struct Format {
    int type;           // GGML type id
    const char* name;
    int block_bytes;    // per 256 weights
    int scale_at[2];    // byte offsets of the block's fp16 scales; -1 = none
    int sub0_scale_at;  // byte whose bit 0 is the low bit of sub-block 0's scale (check 5)
};
const Format FORMATS[] = {
    {12, "Q4_K", 144, {0, 2}, 4},     // half2 dm, scales[12] at 4 (j < 4: scales[j] & 63), qs[128]
    {13, "Q5_K", 176, {0, 2}, 4},     // half2 dm, scales[12] at 4, qh[32], qs[128]
    {14, "Q6_K", 210, {208, -1}, 192},// ql[128], qh[64], int8 scales[16] at 192, half d at 208
    {23, "IQ4_XS", 136, {0, -1}, 4},  // half d, u16 scales_h, scales_l[4] at 4, qs[128]
};
const Format* format_of(int t) {
    for (const Format& f : FORMATS) if (f.type == t) return &f;
    return nullptr;
}

constexpr int8_t kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

float h2f(const uint8_t* p) { uint16_t h; std::memcpy(&h, p, 2); return k::f32_from_f16(h); }

// a normal fp16 in +-[2^-10, 2^-5): exponent field 5..9 (bias 15), any mantissa, either sign
uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

void k4_scale_min(const uint8_t* q, int j, int& d, int& m) {   // ggml get_scale_min_k4
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// one 256-weight block -> 256 doubles, ggml's dequantize_row_q4_K / q5_K / q6_K / iq4_xs
void dequant_block(int t, const uint8_t* b, double* y) {
    if (t == 12 || t == 13) {
        const double d = h2f(b), dmin = h2f(b + 2);
        const uint8_t* sc = b + 4;
        const uint8_t* qh = b + 16;
        const uint8_t* ql = t == 12 ? b + 16 : b + 48;
        for (int is = 0; is < 4; ++is) {
            int s1, m1, s2, m2;
            k4_scale_min(sc, 2 * is, s1, m1); k4_scale_min(sc, 2 * is + 1, s2, m2);
            for (int l = 0; l < 32; ++l) {
                int lo = ql[32 * is + l] & 0xF, hi = ql[32 * is + l] >> 4;
                if (t == 13) {
                    lo += (qh[l] >> (2 * is)) & 1 ? 16 : 0;
                    hi += (qh[l] >> (2 * is + 1)) & 1 ? 16 : 0;
                }
                y[64 * is + l] = d * s1 * lo - dmin * m1;
                y[64 * is + 32 + l] = d * s2 * hi - dmin * m2;
            }
        }
    } else if (t == 14) {
        const double d = h2f(b + 208);
        for (int n = 0; n < 2; ++n) {
            const uint8_t* ql = b + 64 * n;
            const uint8_t* qh = b + 128 + 32 * n;
            const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192) + 8 * n;
            double* o = y + 128 * n;
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                o[l] = d * sc[is] * q1; o[l + 32] = d * sc[is + 2] * q2;
                o[l + 64] = d * sc[is + 4] * q3; o[l + 96] = d * sc[is + 6] * q4;
            }
        }
    } else {   // IQ4_XS
        const double d = h2f(b);
        uint16_t sh; std::memcpy(&sh, b + 2, 2);
        const uint8_t* sl = b + 4;
        const uint8_t* qs = b + 8;
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
            const double dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[32 * ib + j] = dl * kIq4nl[qs[16 * ib + j] & 0xF];
                y[32 * ib + j + 16] = dl * kIq4nl[qs[16 * ib + j] >> 4];
            }
        }
    }
}

// Q8_1 column bytes -> doubles (d * q per 32-block; the block sum is not used by these kernels)
void dequant_q8_1(const uint8_t* xq, int n_in, int T, std::vector<double>& x) {
    x.assign((size_t) T * n_in, 0.0);
    for (int j = 0; j < T; ++j)
        for (int b = 0; b < n_in / 32; ++b) {
            const uint8_t* blk = xq + ((size_t) j * (n_in / 32) + b) * 36;
            const double d = h2f(blk);
            for (int i = 0; i < 32; ++i) x[(size_t) j * n_in + 32 * b + i] = d * (double) (int8_t) blk[4 + i];
        }
}

// rows [r0, r1): ref[j * n_out + r] and the bound's sum of |w_i x_i|
void reference(int t, const std::vector<uint8_t>& w, int n_in, int n_out, int r0, int r1, const std::vector<double>& x,
               int T, std::vector<double>& ref, std::vector<double>& mag) {
    const int bpr = n_in / 256;
    const int bb = format_of(t)->block_bytes;
    std::vector<double> row((size_t) n_in);
    ref.assign((size_t) T * n_out, 0.0);
    mag.assign((size_t) T * n_out, 0.0);
    for (int r = r0; r < r1; ++r) {
        for (int b = 0; b < bpr; ++b) dequant_block(t, &w[((size_t) r * bpr + b) * bb], &row[(size_t) 256 * b]);
        for (int j = 0; j < T; ++j) {
            const double* xj = &x[(size_t) j * n_in];
            double s = 0, m = 0;
            for (int i = 0; i < n_in; ++i) { s += row[i] * xj[i]; m += std::fabs(row[i] * xj[i]); }
            ref[(size_t) j * n_out + r] = s;
            mag[(size_t) j * n_out + r] = m;
        }
    }
}

constexpr double kBound = 2e-5;

struct Shape { int n_in, n_out; };
// n_in <= 4096 and > 4096 take different tiles; 768 and 256 leave lanes of the last sub-group idle; n_out 1000, 520,
// 300, 77 and 3 end in a partial tile.
// 2560 x 4099: more tiles than the persistent KREG2 grid (its default engages at 2-4 columns there, KREG at 1 column:
// check 4 then compares the two kernels bit for bit), and a partial last tile.
// 6144 x 2563: more tiles than a wave of the card holds 12-sub-group work-groups (check 6: the persistent grid at
// 3-8 columns), a partial last tile, and Q5_K rows wider than 4096 (check 6: the 256-register tile at 5-8 columns).
const Shape SHAPES[] = {{2560, 1000}, {6144, 520}, {768, 300}, {256, 77}, {2560, 3}, {2560, 4099}, {6144, 2563}};

struct Dev {
    sycl::queue& q;
    void* alloc(size_t n) { return sycl::malloc_device(n, q); }
    void put(void* d, const void* h, size_t n) { q.memcpy(d, h, n).wait(); }
    void get(void* h, const void* d, size_t n) { q.memcpy(h, d, n).wait(); }
};

std::vector<float> run(Dev& dev, int t, const void* dw, const uint8_t* dxq, float* dy, int n_in, int n_out, int T,
                       int kreg, bool per_column, int xe2 = 0) {
    k::native_mmvq_set_kreg(kreg);
    k::native_mmvq_set_xe2_kq(xe2);   // 0: the launches as they are without STRATA_XE2_KQ
    const size_t qcol = k::native_q8_1_bytes(n_in, 1);
    dev.q.memset(dy, 0xff, (size_t) T * n_out * 4).wait();   // NaN fill: an output not written fails
    if (per_column)
        for (int j = 0; j < T; ++j) k::native_mmvq(t, dw, dxq + (size_t) j * qcol, dy + (size_t) j * n_out, n_in, n_out, 1, &dev.q);
    else
        k::native_mmvq(t, dw, dxq, dy, n_in, n_out, T, &dev.q);
    dev.q.wait_and_throw();
    std::vector<float> y((size_t) T * n_out);
    dev.get(y.data(), dy, y.size() * 4);
    return y;
}

// 7. STRATA_XE2_KQ part 16: two Q8_0 matrices of one shape on one input, native_mmvq_pair against the two calls
int pair_selftest(sycl::queue& q) {
    Dev dev{q};
    int bad = 0;
    const Shape shapes[] = {{2560, 640}, {640, 2560}, {2560, 77}, {96, 5}};
    const int TMAX = 8;
    for (const Shape& sh : shapes) {
        const size_t wbytes = k::native_mmvq_weight_bytes(8, sh.n_in, sh.n_out);
        std::mt19937 rng(4321u + (unsigned) sh.n_in + 7u * (unsigned) sh.n_out);
        std::vector<uint8_t> w(2 * wbytes);
        for (auto& b : w) b = (uint8_t) (rng() & 0xff);
        for (size_t b = 0; b < w.size() / 34; ++b) {   // a Q8_0 block: its fp16 scale, then 32 quants
            const uint16_t h = sane_half(rng);
            std::memcpy(&w[b * 34], &h, 2);
        }
        std::vector<float> hx((size_t) TMAX * sh.n_in);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : hx) v = nd(rng);
        const size_t ny = (size_t) TMAX * sh.n_out;
        void* dw1 = dev.alloc(wbytes);
        void* dw2 = dev.alloc(wbytes);
        float* dx = (float*) dev.alloc(hx.size() * 4);
        uint8_t* dxq = (uint8_t*) dev.alloc(k::native_q8_1_bytes(sh.n_in, TMAX));
        float* dy = (float*) dev.alloc(4 * ny * 4);   // the two calls' outputs, then the pair's
        dev.put(dw1, w.data(), wbytes);
        dev.put(dw2, w.data() + wbytes, wbytes);
        dev.put(dx, hx.data(), hx.size() * 4);
        k::native_quantize_q8_1(dx, dxq, sh.n_in, TMAX, &q);
        q.wait_and_throw();
        for (int T = 1; T <= TMAX; ++T) {
            q.memset(dy, 0xff, 4 * ny * 4).wait();   // NaN fill: an output not written fails
            k::native_mmvq_set_xe2_kq(0);
            k::native_mmvq(8, dw1, dxq, dy, sh.n_in, sh.n_out, T, &q);
            k::native_mmvq(8, dw2, dxq, dy + ny, sh.n_in, sh.n_out, T, &q);
            k::native_mmvq_set_xe2_kq(16);
            k::native_mmvq_xe2_kq_seen();
            const bool took = k::native_mmvq_pair(8, dw1, dw2, dxq, dy + 2 * ny, dy + 3 * ny, sh.n_in, sh.n_out, T, &q);
            const bool seen = (k::native_mmvq_xe2_kq_seen() & 16) != 0;
            q.wait_and_throw();
            std::vector<float> y(4 * ny);
            dev.get(y.data(), dy, y.size() * 4);
            const size_t n = (size_t) T * sh.n_out;
            long long d1 = 0, d2 = 0, nf = 0;
            for (size_t i = 0; i < n; ++i) {
                if (!std::isfinite(y[i]) || !std::isfinite(y[ny + i])) ++nf;
                d1 += std::memcmp(&y[i], &y[2 * ny + i], 4) != 0;
                d2 += std::memcmp(&y[ny + i], &y[3 * ny + i], 4) != 0;
            }
            const bool ok = took && seen && nf == 0 && d1 == 0 && d2 == 0;
            std::printf("Q8_0 pair %5d x %5d %2d  ran %d (wide kernel %d)  pair!=calls %lld %lld  nonfinite %lld  %s\n", sh.n_in,
                        sh.n_out, T, took ? 1 : 0, seen ? 1 : 0, d1, d2, nf, ok ? "ok" : "FAIL");
            if (!ok) ++bad;
        }
        k::native_mmvq_set_xe2_kq(-1);
        sycl::free(dw1, q); sycl::free(dw2, q); sycl::free(dx, q); sycl::free(dxq, q); sycl::free(dy, q);
    }
    return bad;
}

// 8. STRATA_XE2_KQ part 8: a planar Q6_K matrix at 5-8 columns with the part on and off
int planar_selftest() {
    sycl::queue& q = dpct::get_in_order_queue();   // native_q6_k_planar_adopt allocates and frees on this queue
    Dev dev{q};
    const int n_in = 2560, n_out = 1501, TMAX = 8;
    const size_t wbytes = k::native_mmvq_weight_bytes(14, n_in, n_out);
    std::mt19937 rng(9001u);
    std::vector<uint8_t> w(wbytes);
    for (auto& b : w) b = (uint8_t) (rng() & 0xff);
    for (size_t b = 0; b < wbytes / 210; ++b) {   // a Q6_K block keeps its fp16 scale at byte 208
        const uint16_t h = sane_half(rng);
        std::memcpy(&w[b * 210 + 208], &h, 2);
    }
    std::vector<float> hx((size_t) TMAX * n_in);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : hx) v = nd(rng);
    void* dw = dev.alloc(wbytes);
    dev.put(dw, w.data(), wbytes);
    void* pw = k::native_q6_k_planar_adopt(dw, n_in, n_out, "parity matrix");   // frees dw when it converts
    if (!k::native_q6_k_is_planar(pw)) {
        std::printf("Q6_K planar: the matrix was not converted (STRATA_Q6_PLANAR=0?): check 8 skipped\n");
        sycl::free(pw, q);
        return 0;
    }
    float* dx = (float*) dev.alloc(hx.size() * 4);
    uint8_t* dxq = (uint8_t*) dev.alloc(k::native_q8_1_bytes(n_in, TMAX));
    float* dy = (float*) dev.alloc((size_t) TMAX * n_out * 4);
    dev.put(dx, hx.data(), hx.size() * 4);
    k::native_quantize_q8_1(dx, dxq, n_in, TMAX, &q);
    q.wait_and_throw();
    int bad = 0;
    for (int T = 5; T <= TMAX; ++T) {
        const std::vector<float> y0 = run(dev, 14, pw, dxq, dy, n_in, n_out, T, 1, false, 0);
        k::native_mmvq_xe2_kq_seen();
        const std::vector<float> y1 = run(dev, 14, pw, dxq, dy, n_in, n_out, T, 1, false, 8);
        const bool seen = (k::native_mmvq_xe2_kq_seen() & 8) != 0;
        long long d = 0, nf = 0;
        for (size_t i = 0; i < y0.size(); ++i) {
            if (!std::isfinite(y0[i]) || !std::isfinite(y1[i])) ++nf;
            d += std::memcmp(&y0[i], &y1[i], 4) != 0;
        }
        const bool ok = nf == 0 && d == 0 && seen == (T >= 6);
        std::printf("Q6_K planar %5d x %5d %2d  256 registers %d  on!=off %lld  nonfinite %lld  %s\n", n_in, n_out, T,
                    seen ? 1 : 0, d, nf, ok ? "ok" : "FAIL");
        if (!ok) ++bad;
    }
    k::native_mmvq_set_kreg(-1);
    k::native_mmvq_set_xe2_kq(-1);
    k::native_q6_k_unpack(pw);
    sycl::free(pw, q); sycl::free(dx, q); sycl::free(dxq, q); sycl::free(dy, q);
    return bad;
}

int selftest() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    Dev dev{q};
    std::printf("mmvq_kreg_parity on %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("%-7s %5s x %5s %2s  %-11s %-11s %9s %9s %9s %9s %s\n", "type", "n_in", "n_out", "T", "kreg/bound",
                "dflt/bound", "kreg!=dflt", "multi!=1c", "xe2!=kreg", "nonfinite", "");
    int bad = 0;
    int xe2_seen = 0;   // check 6: the STRATA_XE2_KQ parts some launch took
    for (const Format& f : FORMATS) {
        for (const Shape& sh : SHAPES) {
            const size_t wbytes = k::native_mmvq_weight_bytes(f.type, sh.n_in, sh.n_out);
            const size_t nblk = wbytes / f.block_bytes;
            std::mt19937 rng(777u + 31u * (unsigned) f.type + (unsigned) sh.n_in + 7u * (unsigned) sh.n_out);
            std::vector<uint8_t> w(wbytes);
            for (auto& b : w) b = (uint8_t) (rng() & 0xff);
            for (size_t b = 0; b < nblk; ++b)
                for (int at : f.scale_at)
                    if (at >= 0) { const uint16_t h = sane_half(rng); std::memcpy(&w[b * f.block_bytes + at], &h, 2); }
            const uint16_t big = 0x27FF;   // row 0, block 0: d near 2^-5, so check 5 has a step well above the bound
            std::memcpy(&w[f.scale_at[0]], &big, 2);
            const int TMAX = 8;
            std::vector<float> hx((size_t) TMAX * sh.n_in);
            std::normal_distribution<float> nd(0.f, 1.f);
            for (auto& v : hx) v = nd(rng);

            void* dw = dev.alloc(wbytes);
            float* dx = (float*) dev.alloc(hx.size() * 4);
            uint8_t* dxq = (uint8_t*) dev.alloc(k::native_q8_1_bytes(sh.n_in, TMAX));
            float* dy = (float*) dev.alloc((size_t) TMAX * sh.n_out * 4);
            dev.put(dw, w.data(), wbytes);
            dev.put(dx, hx.data(), hx.size() * 4);
            k::native_quantize_q8_1(dx, dxq, sh.n_in, TMAX, &q);
            q.wait_and_throw();
            std::vector<uint8_t> xq(k::native_q8_1_bytes(sh.n_in, TMAX));
            dev.get(xq.data(), dxq, xq.size());
            std::vector<double> x, ref, mag;
            dequant_q8_1(xq.data(), sh.n_in, TMAX, x);
            reference(f.type, w, sh.n_in, sh.n_out, 0, sh.n_out, x, TMAX, ref, mag);   // T columns = the first T of these

            long long case_diff = 0;
            for (int T = 1; T <= TMAX; ++T) {
                const std::vector<float> yk = run(dev, f.type, dw, dxq, dy, sh.n_in, sh.n_out, T, 1, false);
                const std::vector<float> yd = run(dev, f.type, dw, dxq, dy, sh.n_in, sh.n_out, T, 0, false);
                const std::vector<float> y1 = run(dev, f.type, dw, dxq, dy, sh.n_in, sh.n_out, T, 1, true);
                k::native_mmvq_xe2_kq_seen();   // 6. the KREG call with STRATA_XE2_KQ=1 (the grid, the tile, the pair)
                const std::vector<float> yx = run(dev, f.type, dw, dxq, dy, sh.n_in, sh.n_out, T, 1, false, 1);
                xe2_seen |= k::native_mmvq_xe2_kq_seen();
                double rk = 0, rd = 0;
                long long kd = 0, md = 0, nf = 0, xd = 0;
                for (int j = 0; j < T; ++j)
                    for (int r = 0; r < sh.n_out; ++r) {
                        const size_t i = (size_t) j * sh.n_out + r;
                        const double bnd = kBound * mag[i] + 1e-30;
                        if (!std::isfinite(yk[i]) || !std::isfinite(yd[i]) || !std::isfinite(y1[i]) || !std::isfinite(yx[i])) { ++nf; continue; }
                        rk = std::max(rk, std::fabs(yk[i] - ref[i]) / bnd);
                        rd = std::max(rd, std::fabs(yd[i] - ref[i]) / bnd);
                        uint32_t a, b, c, e;
                        std::memcpy(&a, &yk[i], 4); std::memcpy(&b, &yd[i], 4); std::memcpy(&c, &y1[i], 4);
                        std::memcpy(&e, &yx[i], 4);
                        kd += a != b;
                        md += a != c;
                        xd += a != e;
                    }
                case_diff += kd;
                const bool ok = nf == 0 && rk <= 1.0 && rd <= 1.0 && md == 0 && xd == 0;
                std::printf("%-7s %5d x %5d %2d  %-11.3f %-11.3f %9lld %9lld %9lld %9lld %s\n", f.name, sh.n_in, sh.n_out, T,
                            rk, rd, kd, md, xd, nf, ok ? "ok" : "FAIL");
                if (!ok) ++bad;
                if (T == TMAX) {   // 5. power: row 0 against a reference with sub-block 0's scale one step off
                    std::vector<uint8_t> w2(w.begin(), w.begin() + wbytes / sh.n_out);
                    w2[f.sub0_scale_at] ^= 1;
                    std::vector<double> ref2, mag2;
                    reference(f.type, w2, sh.n_in, 1, 0, 1, x, TMAX, ref2, mag2);
                    double seen = 0;
                    for (int j = 0; j < TMAX; ++j)
                        seen = std::max(seen, std::fabs(yk[(size_t) j * sh.n_out] - ref2[j]) / (kBound * mag[(size_t) j * sh.n_out] + 1e-30));
                    if (seen <= 1.0) {
                        std::printf("%-7s %5d x %5d     no power: one scale step of row 0 stays inside the bound (%.3f)\n",
                                    f.name, sh.n_in, sh.n_out, seen);
                        ++bad;
                    }
                }
            }
            if (sh.n_in >= 2560 && case_diff == 0) {
                std::printf("%-7s %5d x %5d     KREG output bitwise equal to the default kernels at every T: the call "
                            "did not reach KREG\n", f.name, sh.n_in, sh.n_out);
                ++bad;
            }
            sycl::free(dw, q); sycl::free(dx, q); sycl::free(dxq, q); sycl::free(dy, q);
        }
    }
    k::native_mmvq_set_kreg(-1);
    k::native_mmvq_set_xe2_kq(-1);
    if ((xe2_seen & 2) == 0) {
        std::printf("STRATA_XE2_KQ: no KREG launch took the persistent grid: check 6 compared KREG with itself\n");
        ++bad;
    }
    if ((xe2_seen & 4) == 0) {
        std::printf("STRATA_XE2_KQ: no Q5_K launch took the 256-register tile: check 6 compared KREG with itself\n");
        ++bad;
    }
    bad += pair_selftest(q);
    bad += planar_selftest();
    std::printf("\nmmvq_kreg_parity: %d failures\n", bad);
    return bad;
}

// ---------------------------------------------------------------- --bench
uint64_t rs = 0x9E3779B97F4A7C15ull;
inline uint64_t xr() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
void fill_pool(const Format& f, std::vector<uint8_t>& h) {   // random bytes, sane fp16 scales at the format's offsets
    uint64_t* p = (uint64_t*) h.data();
    for (size_t i = 0; i < h.size() / 8; ++i) p[i] = xr();
    const size_t nb = h.size() / f.block_bytes;
    for (size_t b = 0; b < nb; ++b)
        for (int at : f.scale_at)
            if (at >= 0) {
                const uint16_t d = (uint16_t) (0x1400 | (xr() & 0x3ff));
                std::memcpy(&h[b * f.block_bytes + at], &d, 2);
            }
}
std::vector<int> parse_list(const char* s) {
    std::vector<int> v; std::string t(s); size_t p = 0;
    while (p < t.size()) {
        size_t e = t.find(',', p); if (e == std::string::npos) e = t.size();
        v.push_back(std::atoi(t.substr(p, e - p).c_str())); p = e + 1;
    }
    return v;
}
double time_graph(sycl::queue& q, int nmat, const std::function<void(int)>& launch, int reps) {
    sx::command_graph g(q.get_context(), q.get_device());
    g.begin_recording(q);
    for (int m = 0; m < nmat; ++m) launch(m);
    g.end_recording(q);
    auto ge = g.finalize();
    for (int i = 0; i < 2; ++i) q.ext_oneapi_graph(ge);
    q.wait();
    std::vector<double> us;
    for (int i = 0; i < reps; ++i) {
        sycl::event e0 = q.ext_oneapi_submit_barrier();
        q.ext_oneapi_graph(ge);
        sycl::event e1 = q.ext_oneapi_submit_barrier();
        e1.wait();
        us.push_back(double(e1.get_profiling_info<sycl::info::event_profiling::command_end>() -
                            e0.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1e3 / nmat);
    }
    std::sort(us.begin(), us.end());
    return us[us.size() / 2];
}

int bench(const char* types_s, const char* ncols_s, const std::string& ssel) {
    const std::vector<int> types = parse_list(types_s), ncl = parse_list(ncols_s);
    const size_t POOL = (size_t) 640 << 20;
    sycl::queue q(sycl::gpu_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    struct BShape { int n_in, n_out; const char* name; };
    // Qwen3.8-Flash-Next decode shapes; the head is 2560 x 248320
    const BShape shapes[] = {{2560, 10240, "qkv"}, {2560, 6144, "gate"}, {6144, 2560, "out"}, {2560, 12288, "attn_q"},
                             {2560, 512, "kv"}, {2560, 640, "sh_gu"}, {2560, 248320, "head"}};
    uint8_t* dpool = (uint8_t*) sycl::malloc_device(POOL + 4096, q);
    std::vector<uint8_t> hpool(POOL);
    const int maxc = 8, maxout = 248320;
    float* y = sycl::malloc_device<float>((size_t) maxc * maxout, q);
    std::vector<float> hx((size_t) maxc * 6144);
    std::mt19937 rng(4242);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : hx) v = nd(rng);
    float* dx = sycl::malloc_device<float>(hx.size(), q);
    q.memcpy(dx, hx.data(), hx.size() * 4).wait();
    void* xq = sycl::malloc_device(k::native_q8_1_bytes(6144, maxc) + 256, q);
    std::printf("mmvq_kreg_parity --bench on %s: us per call (median), GB/s of weight bytes\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());
    for (const int t : types) {
        const Format* f = format_of(t);
        if (!f) { std::printf("type %d: not a KREG format (12, 13, 14, 23)\n", t); continue; }
        fill_pool(*f, hpool);
        q.memcpy(dpool, hpool.data(), POOL).wait();
        for (const BShape& sh : shapes) {
            if (ssel != "all" && ssel != sh.name) continue;
            const size_t wb = k::native_mmvq_weight_bytes(t, sh.n_in, sh.n_out);
            const size_t slot = (wb + 255) & ~(size_t) 255;
            const int ndist = (int) std::max<size_t>(1, std::min<size_t>(128, POOL / slot)), nmat = std::max(ndist, 16);
            for (const int nc : ncl) {
                if (nc < 1 || nc > maxc) continue;
                k::native_quantize_q8_1(dx, xq, sh.n_in, nc, &q);
                q.wait();
                auto timed = [&](int mode) {
                    k::native_mmvq_set_kreg(mode);
                    return time_graph(q, nmat, [&](int m) {
                        k::native_mmvq(t, dpool + (size_t) (m % ndist) * slot, xq, y, sh.n_in, sh.n_out, nc, &q);
                    }, wb > (100u << 20) ? 7 : 11);
                };
                if (std::getenv("STRATA_BENCH_AB")) {   // KREG alone (3: no KREG2) against the default (1), A B A B
                    const double a0 = timed(3), b0 = timed(1), a1 = timed(3), b1 = timed(1);
                    std::printf("%-6s %-6s %5d x %6d nc %d  kreg %8.2f %8.2f us %5.1f GB/s  kreg2 %8.2f %8.2f us %5.1f GB/s  x%.3f\n",
                                f->name, sh.name, sh.n_in, sh.n_out, nc, a0, a1, 2 * wb / (a0 + a1) / 1e3, b0, b1,
                                2 * wb / (b0 + b1) / 1e3, (a0 + a1) / (b0 + b1));
                    std::fflush(stdout);
                    continue;
                }
                double us[2];
                for (int kreg = 0; kreg < 2; ++kreg) us[kreg] = timed(kreg);
                std::printf("%-6s %-6s %5d x %6d nc %d  default %8.2f us %5.1f GB/s  kreg %8.2f us %5.1f GB/s  x%.2f\n",
                            f->name, sh.name, sh.n_in, sh.n_out, nc, us[0], wb / us[0] / 1e3, us[1], wb / us[1] / 1e3,
                            us[0] / us[1]);
                std::fflush(stdout);
            }
        }
    }
    k::native_mmvq_set_kreg(-1);
    sycl::free(dpool, q); sycl::free(y, q); sycl::free(dx, q); sycl::free(xq, q);
    return 0;
}

}  // namespace

// --floor: us per launch of a kernel that does nothing but one store, in the same graph replay as --bench
int floor_bench() {
    sycl::queue q(sycl::gpu_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    float* y = sycl::malloc_device<float>(1 << 20, q);
    const int grids[][2] = {{1, 16}, {160, 80}, {640, 80}, {2560, 80}, {384, 80}, {62080, 80}};
    for (const auto& gl : grids) {
        const int G = gl[0], L = gl[1];
        const double us = time_graph(q, 64, [&](int m) {
            q.parallel_for(sycl::nd_range<1>(std::size_t(G) * L, L), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                if (it.get_global_linear_id() == 0) y[m] = 1.0f;
            });
        }, 11);
        std::printf("floor: %6d work-groups x %3d lanes: %6.2f us per launch\n", G, L, us);
    }
    sycl::free(y, q);
    return 0;
}

int main(int argc, char** argv) try {
    bool selftest_flag = false;
    if (argc > 1 && std::string(argv[1]) == "--floor") return floor_bench();
    if (argc > 1 && std::string(argv[1]) == "--bench")
        return bench(argc > 2 ? argv[2] : "12,13,14,23", argc > 3 ? argv[3] : "1,2,3,4,6", argc > 4 ? argv[4] : "all");
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest_flag = true;
        else { std::fprintf(stderr, "usage: mmvq_kreg_parity [--selftest] | --bench [types] [ncols] [shape]\n"); return 2; }
    }
    const int bad = selftest();
    if (bad) return 1;
    if (selftest_flag) std::printf("mmvq_kreg_parity OK\n");
    return 0;
}
catch (std::exception const& exc) {   // sycl::exception included
    std::fprintf(stderr, "mmvq_kreg_parity: %s\n", exc.what());
    return 1;
}
