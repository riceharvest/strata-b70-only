// src/kernels/mmvq_kreg_parity.cpp - the KREG small-batch MMVQ kernels (STRATA_MMVQ_KREG=1) against a float64
// reference, against the default kernels and against themselves one column at a time.
//
//     mmvq_kreg_parity [--selftest]                      the checks below (ctest runs it with --selftest)
//     STRATA_PARITY_DENSE=1 mmvq_kreg_parity [--selftest]   the dense engine's forms (see below)
//     mmvq_kreg_parity --bench [types] [ncols] [shape]   us per call, default kernels against KREG
//                                                        (types=12,13,14,23 ncols=1,2,3,4,6 shape=all|qkv|...|head)
//                                                        STRATA_KREG_BENCH_STRIP=1|2|3: IQ4_XS with work taken out (kst:: below)
//
// Everything goes through the public native_mmvq; native_mmvq_set_kreg switches the KREG kernels on and off (the
// STRATA_MMVQ_KREG latch is not read). For Q4_K, Q5_K, Q6_K and IQ4_XS (and, dense mode only, Q3_K and IQ4_NL) at 1-8 columns:
//
//   1. KREG against a float64 reference that dequantizes the SAME weight blocks (ggml's dequantize_row_* formulas)
//      and the SAME Q8_1 bytes the kernels read, so only the kernels' FP32 arithmetic is under test. Bound per output:
//      |gpu - ref| <= 2e-5 * sum_i |w_i x_i| (a strata run, as it always was); in a dense run (STRATA_PARITY_DENSE=1) the
//      summation bound of KREG's own FP32 tree (M3b): |gpu - ref| <= 2 gamma(h) * sum_i (|a_i x_i| + |b_i x_i|), kreg_rel:
//      depth h = 15 + n_in / 512 at most, gamma(h) = h u / (1 - h u), u = 2^-24; w_i = a_i - b_i, the scale part and the min
//      term Q4_K / Q5_K subtract in FP32. It grows with the row's tree, not with a fixed 2e-5, so it stays tight enough for
//      check 5 at the 17408-wide rows.
//   2. the default kernels against the float64 reference with the loose bound |gpu - ref| <= 2e-5 * sum_i |w_i x_i|
//      (their summation trees differ): a bound the kernels the engine runs today meet.
//   3. KREG differs from the default kernels in some output of every case with n_in >= 2560 (it sums in another
//      order). If nothing differs, the call never reached KREG (a fallback) and checks 1 and 4 tested the default
//      kernels instead.
//   4. a T-column KREG call bitwise equal to T one-column KREG calls: native_mmvq_set_multi_exact's contract ("every
//      column bitwise equal to a single-column call"), which the verify window's bit-exactness rests on.
//   5. power: at 8 columns, row 0 of the KREG output against a reference with the scale of sub-block 0 of row 0's
//      first block one step off must fail check 1's bound in some column, at every width. A kernel that decodes one
//      sub-block scale wrong would pass a bound that cannot see this.
//   6. STRATA_XE2_KQ (native_mmvq_set_xe2_kq): the KREG call again with STRATA_XE2_KQ=1, bitwise the KREG output (the
//      persistent grid and the 256-register Q5_K tiles change no sum). Over the run some launch must have taken the
//      grid and some the tile (native_mmvq_xe2_kq_seen), or the check has compared KREG with itself.
//   7. STRATA_XE2_KQ part 16: native_mmvq_pair on two Q8_0 matrices at 1-8 columns must run and be bitwise the two
//      native_mmvq calls.
//   8. STRATA_XE2_KQ part 8 (opt-in, not in 1): a planar Q6_K matrix (native_q6_k_planar_adopt) at 5-8 columns, the
//      launch with the part on (256 registers from 6 columns) bitwise the launch with it off.
//
// STRATA_PARITY_DENSE=1 (selftest and --bench): native_mmvq_set_dense_mode(true), the forms strata-q36 runs the 27B with (the
// two-piece and K-split rows of 17408, Q3_K and IQ4_NL through KREG, other tiles at 5-8 columns).  Those formats and shapes
// are only run then (a dense run: checks 1 to 5; 6 to 8 are strata's and are not run again).  Without it every check is
// the one strata's own launches face.
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
    int scale_at[8];    // byte offsets of the block's fp16 scales; -1 = none
    int sub0_scale_at;  // byte whose bit 0 is the low bit of sub-block 0's scale (check 5)
    bool dense_only = false;   // KREG takes it in dense mode only (STRATA_PARITY_DENSE=1)
};
const Format FORMATS[] = {
    {12, "Q4_K", 144, {0, 2, -1, -1, -1, -1, -1, -1}, 4},     // half2 dm, scales[12] at 4 (j < 4: scales[j] & 63), qs[128]
    {13, "Q5_K", 176, {0, 2, -1, -1, -1, -1, -1, -1}, 4},     // half2 dm, scales[12] at 4, qh[32], qs[128]
    {14, "Q6_K", 210, {208, -1, -1, -1, -1, -1, -1, -1}, 192},// ql[128], qh[64], int8 scales[16] at 192, half d at 208
    {23, "IQ4_XS", 136, {0, -1, -1, -1, -1, -1, -1, -1}, 4},  // half d, u16 scales_h, scales_l[4] at 4, qs[128]
    {11, "Q3_K", 110, {108, -1, -1, -1, -1, -1, -1, -1}, 96, true},   // hmask[32], qs[64], scales[12] at 96 (6-bit, packed), half d at 108
    {20, "IQ4_NL", 144, {0, 18, 36, 54, 72, 90, 108, 126}, 1, true},   // 8 x (half d, qs[16]) per 256 weights; check 5: d's high byte
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

// one 256-weight block -> 256 doubles, ggml's dequantize_row_q4_K / q5_K / q6_K / iq4_xs; `ymin` (if given): each
// weight's min term (dmin * m, which Q4_K / Q5_K subtract: y = scale part - ymin), 0 for the other formats
void dequant_block(int t, const uint8_t* b, double* y, double* ymin = nullptr) {
    if (ymin) std::fill(ymin, ymin + 256, 0.0);
    if (t == 20) {   // ggml dequantize_row_iq4_nl, 8 blocks
        for (int ib = 0; ib < 8; ++ib) {
            const uint8_t* q = b + 18 * ib;
            const double d = h2f(q);
            for (int j = 0; j < 16; ++j) {
                y[32 * ib + j] = d * kIq4nl[q[2 + j] & 0xF];
                y[32 * ib + j + 16] = d * kIq4nl[q[2 + j] >> 4];
            }
        }
    } else if (t == 11) {   // ggml dequantize_row_q3_K
        const double d = h2f(b + 108);
        const uint8_t* hm = b;
        const uint8_t* q = b + 32;
        uint32_t aux[4];
        std::memcpy(aux, b + 96, 12);
        const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f, tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* sc = reinterpret_cast<const int8_t*>(aux);
        int is = 0, o = 0;
        uint8_t m = 1;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                double dl = d * (sc[is++] - 32);
                for (int l = 0; l < 16; ++l) y[o++] = dl * ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                dl = d * (sc[is++] - 32);
                for (int l = 0; l < 16; ++l) y[o++] = dl * ((int8_t) ((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    } else if (t == 12 || t == 13) {
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
                if (ymin) { ymin[64 * is + l] = dmin * m1; ymin[64 * is + 32 + l] = dmin * m2; }
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

// rows [r0, r1): ref[j * n_out + r], the loose bound's sum of |w_i x_i| (mag) and KREG's bound's sum of
// |a_i x_i| + |b_i x_i| (mag2: w_i = a_i - b_i, b_i the min term)
void reference(int t, const std::vector<uint8_t>& w, int n_in, int n_out, int r0, int r1, const std::vector<double>& x,
               int T, std::vector<double>& ref, std::vector<double>& mag, std::vector<double>& mag2) {
    const int bpr = n_in / 256;
    const int bb = format_of(t)->block_bytes;
    std::vector<double> row((size_t) n_in), rmin((size_t) n_in);
    ref.assign((size_t) T * n_out, 0.0);
    mag.assign((size_t) T * n_out, 0.0);
    mag2.assign((size_t) T * n_out, 0.0);
    for (int r = r0; r < r1; ++r) {
        for (int b = 0; b < bpr; ++b)
            dequant_block(t, &w[((size_t) r * bpr + b) * bb], &row[(size_t) 256 * b], &rmin[(size_t) 256 * b]);
        for (int j = 0; j < T; ++j) {
            const double* xj = &x[(size_t) j * n_in];
            double s = 0, m = 0, m2 = 0;
            for (int i = 0; i < n_in; ++i) {
                s += row[i] * xj[i];
                m += std::fabs(row[i] * xj[i]);
                m2 += std::fabs((row[i] + rmin[i]) * xj[i]) + std::fabs(rmin[i] * xj[i]);
            }
            ref[(size_t) j * n_out + r] = s;
            mag[(size_t) j * n_out + r] = m;
            mag2[(size_t) j * n_out + r] = m2;
        }
    }
}

constexpr double kBound = 2e-5;   // the default kernels' (check 2)

// KREG's bound (checks 1 and 5) relative to mag2, by its FP32 summation tree at width n_in: a lane's 32-weight dot sums
// integers exactly and then scales and adds them in at most 8 roundings; the row's second piece (H = 2) adds 2, the
// 16-lane butterfly 4, the sub-groups' SLM sum one after another at most n_in / 512 (one-piece: 34 sub-groups at 17408),
// the K split's parts (KS) 1.  (M4, STRATA_KREG_KS=4: KS in 4 parts at 17408 - 9 sub-groups a part, then 3 plane adds:
// 8 + 4 + 9 + 3 = 24, inside the one-piece depth this bound already allows.)  A summation tree of depth h errs by at most gamma(h) = h u / (1 - h u) times the sum of
// the magnitudes it adds (Higham, Accuracy and Stability of Numerical Algorithms, 4.2); checked against twice that.
double kreg_rel(int n_in) {
    const int h = 8 + 2 + 4 + (n_in + 511) / 512 + 1;
    const double u = std::ldexp(1.0, -24);
    return 2.0 * h * u / (1.0 - h * u);
}

struct Shape { int n_in, n_out; bool dense_only = false; };
bool g_dense = false;   // STRATA_PARITY_DENSE=1
// n_in <= 4096 and > 4096 take different tiles; 768 and 256 leave lanes of the last sub-group idle; n_out 1000, 520,
// 300, 77 and 3 end in a partial tile.
// 2560 x 4099: more tiles than the persistent KREG2 grid (its default engages at 2-4 columns there, KREG at 1 column:
// check 4 then compares the two kernels bit for bit), and a partial last tile.
// 6144 x 2563: more tiles than a wave of the card holds 12-sub-group work-groups (check 6: the persistent grid at
// 3-8 columns), a partial last tile, and Q5_K rows wider than 4096 (check 6: the 256-register tile at 5-8 columns).
// 5120 x 600 and 17408 x 300: the Qwen3.8-27B's widths (dense runs only for 17408: 544 lanes, past a 256-register group; with
// STRATA_KREG_KS=4 its K in 4 parts and 5-8 columns in one launch pair).
const Shape SHAPES[] = {{2560, 1000}, {6144, 520}, {768, 300}, {256, 77}, {2560, 3}, {2560, 4099}, {6144, 2563},
                        {5120, 600}, {17408, 300, true}};

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
            if (!g_dense && (f.dense_only || sh.dense_only)) continue;
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
            std::vector<double> x, ref, mag, mag2;
            dequant_q8_1(xq.data(), sh.n_in, TMAX, x);
            reference(f.type, w, sh.n_in, sh.n_out, 0, sh.n_out, x, TMAX, ref, mag, mag2);   // T columns = the first T of these
            const double krel = kreg_rel(sh.n_in);

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
                        const double bnd = kBound * mag[i] + 1e-30, bk = g_dense ? krel * mag2[i] + 1e-30 : bnd;
                        if (!std::isfinite(yk[i]) || !std::isfinite(yd[i]) || !std::isfinite(y1[i]) || !std::isfinite(yx[i])) { ++nf; continue; }
                        rk = std::max(rk, std::fabs(yk[i] - ref[i]) / bk);
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
                if (T == TMAX) {   // 5. power: row 0 against a reference with sub-block 0's scale one step off, check 1's bound
                    std::vector<uint8_t> w2(w.begin(), w.begin() + wbytes / sh.n_out);
                    w2[f.sub0_scale_at] ^= 1;
                    std::vector<double> ref2, m2a, m2b;
                    reference(f.type, w2, sh.n_in, 1, 0, 1, x, TMAX, ref2, m2a, m2b);
                    double seen = 0;
                    for (int j = 0; j < TMAX; ++j)
                        seen = std::max(seen, std::fabs(yk[(size_t) j * sh.n_out] - ref2[j]) / (g_dense ? krel * mag2[(size_t) j * sh.n_out] + 1e-30 : kBound * mag[(size_t) j * sh.n_out] + 1e-30));
                    std::printf("%-7s %5d x %5d     power: one scale step of row 0 is %.1f x the bound\n", f.name, sh.n_in,
                                sh.n_out, seen);
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
    if (!g_dense && (xe2_seen & 2) == 0) {
        std::printf("STRATA_XE2_KQ: no KREG launch took the persistent grid: check 6 compared KREG with itself\n");
        ++bad;
    }
    if (!g_dense && (xe2_seen & 4) == 0) {
        std::printf("STRATA_XE2_KQ: no Q5_K launch took the 256-register tile: check 6 compared KREG with itself\n");
        ++bad;
    }
    if (!g_dense) {   // (strata's checks; a dense run does not touch them)
        bad += pair_selftest(q);
        bad += planar_selftest();
    }
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

// ---------------------------------------------------------------- --bench STRIP kernels (IQ4_XS)
// STRATA_KREG_BENCH_STRIP (bench only): the IQ4_XS kernels the engine runs by default (STRATA_MMVQ_KREG=1) with work taken out,
// to tell whether a call is limited by the weight reads or by the decode.  The kernel is chosen as native_mmvq.dp.cpp's
// try_kreg<KregIQ4XS> chooses it (the bench's "kreg" column: native_mmvq_set_kreg(1)):
//   try_kreg2 (1-4 columns here: the strip modes do not follow STRATA_KREG2_R's entries 5-8, those column counts are
//   stripped as KREG; STRATA_KREG2=0 off; STRATA_KREG2_R="a,b,c,d" rows a tile per column count, default 2,4,4,4):
//     k2_go: grid = max(1, 60 / nsg) * cores (cores = max_compute_units / 8), tiles = ceil(n_out / R);
//     1 column takes KREG2 when tiles <= grid, 2-4 columns when tiles > grid; G = min(tiles, grid) work-groups,
//     kreg2_iq4xs_kernel<NCOLS, R, PF = NCOLS > 1>;
//   otherwise kreg_dispatch<KregIQ4XS, NCOLS>: native_mmvq_kreg_kernel with R = 8 at 1 column, 4 at 2-8, one work-group a tile.
// Work-group = L = ceil(n_in / 256 * 8 / 16) * 16 work-items (sub-group size 16), the same nd_range and the same local allocation
// (2 * nsg * P + the codebook words, as the real launches) as the real launch of the same kernel.  Every work-item loads what the
// real kernel's lane loads, in the order of the real kernel: weights per row three 8-byte loads (block header, then the two
// halves of the 16 qs bytes of sub-block g), activations per column two 16-byte vector loads and the fp16 scale of Q8_1 block g.
//   STRIP=3 weight reads only: the weight loads of every tile (KREG2: with its prefetch order), xor-folded; no activation loads,
//           no codebook fill, no barrier.
//   STRIP=1 reads only: STRIP=3 plus the activation loads and the codebook fill and its barrier (the real kernels' fixed
//           per-work-group part, as kqs keeps its tables in STRIP=1); one table word is folded; no lookups, no dp4a, no products.
//   STRIP=2 reads + decode, no products: STRIP=1 plus the real weight decode (the table lookups, the lo / hi recombination of
//           KREG, the two-table form of KREG2, the sub-block scale and the fp16 block scale as float), the decoded words
//           summed instead of dp4a with the activations.  The activation permute of KREG2 is not done (activation work).
// Dropped in every variant: the dot products, the 16-lane transposed butterfly, the SLM partial sums with their barrier per tile
// and the output stores.  A work-item's value is xor-folded over the sub-group and stored once (lane 0, one word a sub-group), so
// every load stays live.  Its stores are a few words a work-group against the real kernel's n_out * ncols floats.
namespace kst {

constexpr int kSg = 16;      // = KREG_SG
constexpr int kIqBlk = 136;  // IQ4XSBlock: half d, u16 scales_h, u8 scales_l[4], u8 qs[128]
constexpr int kQ81 = 36;     // Q81Block: half2 ds, int8 qs[32]

struct Raw { sycl::int2 hdr, qa, qb; };   // = K2Raw: the block's 8 header bytes, the 16 qs bytes of sub-block g as two 8-byte loads

__dpct_inline__ sycl::int4 ld_i4_a4(const int* p) {   // = ld_i4_a4: a 16-byte vector load of a 4-byte aligned address
    sycl::int4 v;
    v.load(0, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p));
    return v;
}
__dpct_inline__ uint32_t k2_code(int n) {   // = k2_code: kIq4nlTable[n] as a byte, from registers
    const uint32_t w0 = 0xBFAD9881u, w1 = 0xF6EADDCFu, w2 = 0x26190D01u, w3 = 0x71594535u;
    const uint32_t wd = n < 8 ? (n < 4 ? w0 : w1) : (n < 12 ? w2 : w3);
    return (wd >> (8 * (n & 3))) & 0xFFu;
}
__dpct_inline__ uint32_t table_kreg(int i) {   // = KregIQ4XS::table_entry
    return (uint32_t) (uint8_t) kIq4nl[i & 15] | ((uint32_t) (uint8_t) kIq4nl[i >> 4] << 16);
}
__dpct_inline__ uint32_t lanes_xor(const sycl::sub_group& sg, uint32_t v) {   // a 16-lane xor butterfly
#pragma unroll
    for (int o = kSg / 2; o > 0; o >>= 1) v ^= sycl::permute_group_by_xor(sg, v, o);
    return v;
}
// = F::act per column: Q8_1 block kb * 8 + g of column j (36 bytes: ds at 0, 32 quants at 4), its two 16-byte halves and its fp16 scale
template <int NC>
__dpct_inline__ uint32_t act_fold(const uint8_t* x, int x_stride, int kb, int g, bool live) {
    uint32_t a = 0u;
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        const uint8_t* xb = x + ((size_t) j * x_stride + (size_t) kb * 8 + g) * kQ81;
        const int* q = reinterpret_cast<const int*>(xb + 4);
        const sycl::int4 u0 = ld_i4_a4(q), u1 = ld_i4_a4(q + 4);
        const uint32_t ds = (uint32_t) *reinterpret_cast<const uint16_t*>(xb);
        const uint32_t v = (uint32_t) (u0.x() ^ u0.y() ^ u0.z() ^ u0.w() ^ u1.x() ^ u1.y() ^ u1.z() ^ u1.w()) ^ ds;
        a ^= live ? v : 0u;
    }
    return a;
}
// = k2_load_tile: the raw words of the R rows of tile t for this lane (a row past n_out repeats the last row)
template <int R>
__dpct_inline__ void load_tile(Raw (&raw)[R], const uint8_t* w, int t, int bpr, int kb, int g, int n_out) {
    const int row0 = t * R;
    const uint8_t* wt = w + ((size_t) row0 * bpr + kb) * kIqBlk;
    const int rlast = n_out - 1 - row0;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const uint8_t* b = wt + (size_t) (r < rlast ? r : rlast) * bpr * kIqBlk;
        raw[r].hdr = *reinterpret_cast<const sycl::int2*>(b);
        const sycl::int2* q2 = reinterpret_cast<const sycl::int2*>(b + 8 + 16 * g);
        raw[r].qa = q2[0];
        raw[r].qb = q2[1];
    }
}
__dpct_inline__ uint32_t fold_raw(const Raw& r) {
    return (uint32_t) (r.hdr.x() ^ r.hdr.y() ^ r.qa.x() ^ r.qa.y() ^ r.qb.x() ^ r.qb.y());
}
// STRIP=2: KregIQ4XS::load (K2 false: one table, the P / Q recombination) or k2_decode (K2 true: the table and the table << 8),
// without the dp4a; the eight decoded words, the sub-block scale and the block scale as float are summed
template <bool K2>
__dpct_inline__ uint32_t decode_sum(const Raw& r, int g, const uint32_t* tl) {
    const uint32_t v[4] = {(uint32_t) r.qa.x(), (uint32_t) r.qa.y(), (uint32_t) r.qb.x(), (uint32_t) r.qb.y()};
    uint32_t acc = 0u;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t lo, hi;
        if constexpr (K2) {
            const uint32_t* tb = tl + 256;
            lo = tl[v[i] & 0xFF] | tb[(v[i] >> 8) & 0xFF];
            hi = tl[(v[i] >> 16) & 0xFF] | tb[v[i] >> 24];
        } else {
            const uint32_t P = tl[v[i] & 0xFF] | (tl[(v[i] >> 8) & 0xFF] << 8);
            const uint32_t Q = tl[(v[i] >> 16) & 0xFF] | (tl[v[i] >> 24] << 8);
            lo = (P & 0xFFFFu) | (Q << 16);
            hi = (P >> 16) | (Q & 0xFFFF0000u);
        }
        acc += lo;
        acc += hi;
    }
    const uint32_t h0 = (uint32_t) r.hdr.x(), sl = (uint32_t) r.hdr.y();
    const int ls = (int) (((sl >> (8 * (g >> 1) + 4 * (g & 1))) & 0x0f) | (((h0 >> (16 + 2 * g)) & 0x03) << 4));
    const float d = (float) sycl::bit_cast<sycl::half>((uint16_t) (h0 & 0xFFFF)) * (float) (ls - 32);
    return acc + sycl::bit_cast<uint32_t>(d);
}

// native_mmvq_kreg_kernel<KregIQ4XS, NC, R>: activations, then the codebook table (SLM fill, barrier), then the tile loop
// (one iteration where the grid is one work-group a tile).  slm: 2 * nsg * P + 256 floats (the table after the reduction buffers).
template <int MODE, int NC, int R>
__dpct_inline__ void kreg_strip_kernel(sycl::nd_item<1> it, const uint8_t* __restrict__ w, const uint8_t* __restrict__ x,
                                       uint32_t* __restrict__ sink, int n_in, int n_out, float* __restrict__ slm) {
    const sycl::sub_group sg = it.get_sub_group();
    constexpr int P = (R * NC + kSg - 1) / kSg * kSg;
    const int lid = int(it.get_local_id(0)), lane = int(sg.get_local_linear_id()), sgi = lid / kSg;
    const int L = int(it.get_local_range(0)), nsg = L / kSg;
    const int bpr = n_in / 256;
    [[maybe_unused]] const int x_stride = n_in / 32;
    const bool live = (lid >> 3) < bpr;
    const int kb = live ? (lid >> 3) : 0, g = lid & 7;
    uint32_t af = 0u, tseed = 0u;
    [[maybe_unused]] uint32_t* tw = reinterpret_cast<uint32_t*>(slm + 2 * nsg * P);
    if constexpr (MODE != 3) {
        af = act_fold<NC>(x, x_stride, kb, g, live);
        for (int i = lid; i < 256; i += L) tw[i] = table_kreg(i);
        it.barrier(sycl::access::fence_space::local_space);
        tseed = tw[lid & 255];
    }
    const int ntiles = (n_out + R - 1) / R;
    uint32_t s = 0u;
    for (int t = int(it.get_group(0)); t < ntiles; t += int(it.get_group_range(0))) {
        Raw raw[R];
        load_tile<R>(raw, w, t, bpr, kb, g, n_out);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if constexpr (MODE == 2) s += decode_sum<false>(raw[r], g, tw);
            else s ^= fold_raw(raw[r]);
        }
    }
    s ^= af ^ tseed;
    slm[lid] = sycl::bit_cast<float>(s);   // keeps the reduction buffers' part of the local allocation in the kernel (2 * nsg * P >= L)
    s = lanes_xor(sg, s);
    if (lane == 0) sink[(size_t) it.get_group(0) * nsg + sgi] = s;
}

// kreg2_iq4xs_kernel<NC, R, PF>: the first tile's weight loads, then the activations, then the table (computed from constants:
// no loads; the table and the table << 8) and the barrier, then the persistent tile loop with the next tile's loads in flight
// (PF, 2-4 columns) while this one is folded.  slm: 2 * nsg * P + 512 floats.
template <int MODE, int NC, int R, bool PF>
__dpct_inline__ void kreg2_strip_kernel(sycl::nd_item<1> it, const uint8_t* __restrict__ w, const uint8_t* __restrict__ x,
                                        uint32_t* __restrict__ sink, int n_in, int n_out, float* __restrict__ slm) {
    const sycl::sub_group sg = it.get_sub_group();
    constexpr int P = (R * NC + kSg - 1) / kSg * kSg;
    const int lid = int(it.get_local_id(0)), lane = int(sg.get_local_linear_id()), sgi = lid / kSg;
    const int L = int(it.get_local_range(0)), nsg = L / kSg;
    const int bpr = n_in / 256;
    [[maybe_unused]] const int x_stride = n_in / 32;
    const bool live = (lid >> 3) < bpr;
    const int kb = live ? (lid >> 3) : 0, g = lid & 7;
    const int ntiles = (n_out + R - 1) / R, G = int(it.get_group_range(0));
    int t = int(it.get_group(0));   // < ntiles: the grid never exceeds the tiles
    Raw cur[R];
    load_tile<R>(cur, w, t, bpr, kb, g, n_out);
    uint32_t af = 0u, tseed = 0u;
    [[maybe_unused]] uint32_t* tw = reinterpret_cast<uint32_t*>(slm + 2 * nsg * P);
    if constexpr (MODE != 3) {
        af = act_fold<NC>(x, x_stride, kb, g, live);
        for (int i = lid; i < 256; i += L) {
            const uint32_t te = k2_code(i & 15) | (k2_code(i >> 4) << 16);
            tw[i] = te;
            tw[256 + i] = te << 8;
        }
        it.barrier(sycl::access::fence_space::local_space);
        tseed = tw[lid & 255];
    }
    uint32_t s = 0u;
    for (; t < ntiles; t += G) {
        const int tn = t + G;
        Raw nxt[R];
        if constexpr (PF) load_tile<R>(nxt, w, tn < ntiles ? tn : t, bpr, kb, g, n_out);   // (the real kernel's last prefetch repeats tile t)
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if constexpr (MODE == 2) s += decode_sum<true>(cur[r], g, tw);
            else s ^= fold_raw(cur[r]);
        }
        if constexpr (PF) {
#pragma unroll
            for (int r = 0; r < R; ++r) cur[r] = nxt[r];
        } else if (tn < ntiles) {
            load_tile<R>(cur, w, tn, bpr, kb, g, n_out);
        }
    }
    s ^= af ^ tseed;
    slm[lid] = sycl::bit_cast<float>(s);
    s = lanes_xor(sg, s);
    if (lane == 0) sink[(size_t) it.get_group(0) * nsg + sgi] = s;
}

struct Plan {
    bool k2;       // KREG2 (else KREG)
    int R, groups, L, nsg, grid;
};
// try_kreg2 / k2_go / kreg_dispatch for IQ4_XS: which kernel, rows a tile, work-groups
Plan make_plan(int nc, int n_in, int n_out, int cores, bool k2_on, const int (&kr)[5]) {
    Plan p;
    p.L = ((n_in / 256) * 8 + kSg - 1) / kSg * kSg;
    p.nsg = p.L / kSg;
    p.grid = std::max(1, 60 / p.nsg) * cores;   // 60 threads per Xe core
    p.k2 = false;
    p.R = nc == 1 ? 8 : 4;
    p.groups = (n_out + p.R - 1) / p.R;
    if (k2_on && nc >= 1 && nc <= 4) {
        const int R = kr[nc], tiles = (n_out + R - 1) / R;
        if (!(nc == 1 ? tiles > p.grid : tiles <= p.grid)) {
            p.k2 = true;
            p.R = R;
            p.groups = tiles < p.grid ? tiles : p.grid;
        }
    }
    return p;
}

template <int MODE, int NC, int R, bool K2>
void launch_one(sycl::queue& q, const Plan& p, const uint8_t* w, const uint8_t* x, uint32_t* sink, int n_in, int n_out) {
    constexpr int P = (R * NC + kSg - 1) / kSg * kSg;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>((size_t) (2 * p.nsg * P + (K2 ? 512 : 256))), h);
        const sycl::nd_range<1> nr(std::size_t(p.groups) * p.L, p.L);
        h.parallel_for(nr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            float* sp = slm.get_multi_ptr<sycl::access::decorated::no>().get();
            if constexpr (K2) kreg2_strip_kernel<MODE, NC, R, (NC > 1)>(it, w, x, sink, n_in, n_out, sp);
            else kreg_strip_kernel<MODE, NC, R>(it, w, x, sink, n_in, n_out, sp);
        });
    });
}
template <int MODE, int NC>
void launch_nc(sycl::queue& q, const Plan& p, const uint8_t* w, const uint8_t* x, uint32_t* sink, int n_in, int n_out) {
    if constexpr (NC <= 4) {
        if (p.k2) {
            if (p.R == 2) launch_one<MODE, NC, 2, true>(q, p, w, x, sink, n_in, n_out);
            else launch_one<MODE, NC, 4, true>(q, p, w, x, sink, n_in, n_out);
            return;
        }
    }
    if constexpr (NC == 1) launch_one<MODE, 1, 8, false>(q, p, w, x, sink, n_in, n_out);
    else launch_one<MODE, NC, 4, false>(q, p, w, x, sink, n_in, n_out);
}
template <int MODE>
void launch_mode(int nc, sycl::queue& q, const Plan& p, const uint8_t* w, const uint8_t* x, uint32_t* sink, int n_in, int n_out) {
    switch (nc) {
        case 1: launch_nc<MODE, 1>(q, p, w, x, sink, n_in, n_out); break;
        case 2: launch_nc<MODE, 2>(q, p, w, x, sink, n_in, n_out); break;
        case 3: launch_nc<MODE, 3>(q, p, w, x, sink, n_in, n_out); break;
        case 4: launch_nc<MODE, 4>(q, p, w, x, sink, n_in, n_out); break;
        case 5: launch_nc<MODE, 5>(q, p, w, x, sink, n_in, n_out); break;
        case 6: launch_nc<MODE, 6>(q, p, w, x, sink, n_in, n_out); break;
        case 7: launch_nc<MODE, 7>(q, p, w, x, sink, n_in, n_out); break;
        default: launch_nc<MODE, 8>(q, p, w, x, sink, n_in, n_out); break;
    }
}
void call(int mode, int nc, sycl::queue& q, const Plan& p, const uint8_t* w, const uint8_t* x, uint32_t* sink, int n_in, int n_out) {
    if (mode == 1) launch_mode<1>(nc, q, p, w, x, sink, n_in, n_out);
    else if (mode == 2) launch_mode<2>(nc, q, p, w, x, sink, n_in, n_out);
    else launch_mode<3>(nc, q, p, w, x, sink, n_in, n_out);
}

}  // namespace kst

int bench(const char* types_s, const char* ncols_s, const std::string& ssel) {
    const std::vector<int> types = parse_list(types_s), ncl = parse_list(ncols_s);
    // STRATA_KREG_BENCH_STRIP=1|2|3 (unset or 0: the real kernels, as always): the stripped IQ4_XS kernels (kst::) in place of the
    // "kreg" column's native_mmvq call; the "default" column (the pre-KREG kernels) is not run and prints "-".  Other types print a skip line.
    const int strip = std::getenv("STRATA_KREG_BENCH_STRIP") && *std::getenv("STRATA_KREG_BENCH_STRIP") ? std::atoi(std::getenv("STRATA_KREG_BENCH_STRIP")) : 0;
    if (strip < 0 || strip > 3) { std::fprintf(stderr, "--bench: STRATA_KREG_BENCH_STRIP must be 0, 1, 2 or 3\n"); std::exit(2); }
    if (strip != 0 && std::getenv("STRATA_BENCH_AB")) { std::fprintf(stderr, "--bench: STRATA_KREG_BENCH_STRIP and STRATA_BENCH_AB do not combine\n"); std::exit(2); }
    // the kernel choice of try_kreg2 at 1-4 columns (read as kreg2_cfg reads it): STRATA_KREG2=0 off, STRATA_KREG2_R="a,b,c,d"
    // rows a tile (2 or 4).  Entries 5-8 are not read: a strip run at 5-8 columns is the KREG kernel's.
    bool k2_on = true;
    int k2_r[5] = {0, 2, 4, 4, 4};
    if (strip != 0) {
        const char* e = std::getenv("STRATA_KREG2");
        k2_on = !(e && e[0] == '0');
        if (const char* rl = std::getenv("STRATA_KREG2_R")) {
            for (int i = 1; i <= 4 && *rl; ++i) {
                const int v = std::atoi(rl);
                if (v == 2 || v == 4) k2_r[i] = v;
                while (*rl && *rl != ',') ++rl;
                if (*rl == ',') ++rl;
            }
        }
    }
    const size_t POOL = (size_t) 640 << 20;
    sycl::queue q(sycl::gpu_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    struct BShape { int n_in, n_out; const char* name; };
    // Qwen3.8-Flash-Next decode shapes; the head is 2560 x 248320.  27B: Qwen3.8-27B's (5120 wide, FFN 17408, head 5120 x 248320)
    const BShape shapes[] = {{2560, 10240, "qkv"}, {2560, 6144, "gate"}, {6144, 2560, "out"}, {2560, 12288, "attn_q"},
                             {2560, 512, "kv"}, {2560, 640, "sh_gu"}, {2560, 248320, "head"},
                             {5120, 10240, "27qkv"}, {5120, 6144, "27gate"}, {6144, 5120, "27out"}, {5120, 12288, "27q"},
                             {5120, 17408, "27gu"}, {17408, 5120, "27down"}, {5120, 248320, "27head"}};
    uint8_t* dpool = (uint8_t*) sycl::malloc_device(POOL + 4096, q);
    std::vector<uint8_t> hpool(POOL);
    const int maxc = 8, maxout = 248320;
    float* y = sycl::malloc_device<float>((size_t) maxc * maxout, q);
    std::vector<float> hx((size_t) maxc * 17408);
    std::mt19937 rng(4242);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : hx) v = nd(rng);
    float* dx = sycl::malloc_device<float>(hx.size(), q);
    q.memcpy(dx, hx.data(), hx.size() * 4).wait();
    void* xq = sycl::malloc_device(k::native_q8_1_bytes(17408, maxc) + 256, q);
    std::printf("mmvq_kreg_parity --bench on %s: us per call (median), GB/s of weight bytes\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());
    const int cores = std::max(1, int(q.get_device().get_info<sycl::info::device::max_compute_units>()) / 8);   // k2_go's count
    if (strip != 0) {
        const char* what[] = {"", "reads only: weight and activation loads and the codebook fill + barrier, no lookups / decode / dp4a",
                              "reads + decode, no products: weight words decoded (lookups, scales) and summed, activations loaded",
                              "weight reads only: no activation loads, no codebook, no barrier"};
        std::printf("--bench: STRIP=%d (%s); IQ4_XS only; mirrors the engine's path (native_mmvq_set_kreg(1): KREG2 where try_kreg2 takes it, else KREG)\n",
                    strip, what[strip]);
        std::printf("--bench: STRIP=%d: no dot products, reduction or output stores, a few words a work-group are stored instead; "
                    "outputs are not results; compare with the kreg column of the same run without the switch\n", strip);
        std::printf("--bench: STRIP=%d: KREG2 %s, R per column count %d,%d,%d,%d, %d cores; each line ends with the kernel mirrored and the weight bytes a call reads\n",
                    strip, k2_on ? "on" : "off (STRATA_KREG2=0)", k2_r[1], k2_r[2], k2_r[3], k2_r[4], cores);
    }
    for (const int t : types) {
        const Format* f = format_of(t);
        if (!f) { std::printf("type %d: not a KREG format (11, 12, 13, 14, 20, 23)\n", t); continue; }
        if (f->dense_only && !g_dense) { std::printf("type %d (%s): dense mode only (STRATA_PARITY_DENSE=1)\n", t, f->name); continue; }
        if (strip != 0 && t != 23) { std::printf("type %d: STRIP=%d skipped (IQ4_XS only)\n", t, strip); continue; }
        fill_pool(*f, hpool);
        q.memcpy(dpool, hpool.data(), POOL).wait();
        for (const BShape& sh : shapes) {
            if (ssel == "27" ? sh.name[0] != '2' || sh.name[1] != '7' : (ssel != "all" && ssel != sh.name)) continue;
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
                if (strip != 0) {   // the stripped kernel in the kreg column; the default column is not run
                    const kst::Plan pl = kst::make_plan(nc, sh.n_in, sh.n_out, cores, k2_on, k2_r);
                    if ((size_t) pl.groups * pl.nsg > (size_t) maxc * maxout) {
                        std::fprintf(stderr, "--bench STRIP: %d work-groups of %d sub-groups do not fit the sink\n", pl.groups, pl.nsg);
                        std::exit(2);
                    }
                    const double us1 = time_graph(q, nmat, [&](int m) {
                        kst::call(strip, nc, q, pl, dpool + (size_t) (m % ndist) * slot, (const uint8_t*) xq, (uint32_t*) y, sh.n_in, sh.n_out);
                    }, wb > (100u << 20) ? 7 : 11);
                    std::printf("%-6s %-6s %5d x %6d nc %d  default %8s us %5s GB/s  kreg %8.2f us %5.1f GB/s  x%s  strip %d %s R%d groups %d wbytes %zu\n",
                                f->name, sh.name, sh.n_in, sh.n_out, nc, "-", "-", us1, wb / us1 / 1e3, "-", strip,
                                pl.k2 ? "kreg2" : "kreg", pl.R, pl.groups, wb);
                    std::fflush(stdout);
                    continue;
                }
                if (std::getenv("STRATA_BENCH_KREG_ONLY")) {   // KREG alone: us, GB/s, and us per column
                    const double u = timed(1);
                    std::printf("%-6s %-6s %5d x %6d nc %d  kreg %8.2f us %5.1f GB/s  %7.2f us/col\n", f->name, sh.name,
                                sh.n_in, sh.n_out, nc, u, wb / u / 1e3, u / nc);
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
    if (const char* d = std::getenv("STRATA_PARITY_DENSE"); d && d[0] == '1') {   // strata-q36's forms (see the top)
        g_dense = true;
        k::native_mmvq_set_dense_mode(true);
    }
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
