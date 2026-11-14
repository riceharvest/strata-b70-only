// sycl/src/qwen35/q36_mmvq_dense_parity.cpp - strata-q36's decode forms for the dense 27B (q36_mmvq_dense.dp.cpp, strata::q36dense::mmvq)
// against a float64 reference, against the default kernels (native_mmvq with the KREG kernels off) and against themselves one column
// at a time.  Derived from mmvq_kreg_parity.cpp, which checks the KREG kernels `strata` runs and is not changed for this.
//
//     q36_mmvq_dense_parity [--selftest]                      the checks below
//     q36_mmvq_dense_parity --bench [types] [ncols] [shape]   us per call, default kernels against the dense forms
//                                                             (types=11,12,13,14,20,23 ncols=1,2,3,4,6 shape=all|27|qkv|...|head)
//
// The dense forms through strata::q36dense::mmvq (STRATA_MMVQ_KREG=1 is set here), the default kernels through the public
// native_mmvq. For Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL and IQ4_XS at 1-8 columns:
//
//   1. KREG against a float64 reference that dequantizes the SAME weight blocks (ggml's dequantize_row_* formulas)
//      and the SAME Q8_1 bytes the kernels read, so only the kernels' FP32 arithmetic is under test. Bound per
//      output (M3b): |gpu - ref| <= 2 gamma(h) * sum_i (|a_i x_i| + |b_i x_i|), the summation bound of KREG's own
//      FP32 tree (kreg_rel: depth h = 15 + n_in / 512 at most, gamma(h) = h u / (1 - h u), u = 2^-24; w_i = a_i - b_i,
//      the scale part and the min term Q4_K / Q5_K subtract in FP32). It grows with the row's tree, not with a fixed
//      2e-5, so it stays tight enough for check 5 at the 17408-wide rows.
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
#include <dpct/blas_utils.hpp>
#include "strata/kernels/native_mmvq.hpp"
#include "q36_mmvq_dense.hpp"
#include "q36_xr_dense.hpp"
#include "q36_xatt.hpp"
#include "q36_pp_i8.hpp"

namespace strata::kernels {
void dequant_f16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out, void* stream);
}

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
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
};
const Format FORMATS[] = {
    {12, "Q4_K", 144, {0, 2, -1, -1, -1, -1, -1, -1}, 4},     // half2 dm, scales[12] at 4 (j < 4: scales[j] & 63), qs[128]
    {13, "Q5_K", 176, {0, 2, -1, -1, -1, -1, -1, -1}, 4},     // half2 dm, scales[12] at 4, qh[32], qs[128]
    {14, "Q6_K", 210, {208, -1, -1, -1, -1, -1, -1, -1}, 192},// ql[128], qh[64], int8 scales[16] at 192, half d at 208
    {23, "IQ4_XS", 136, {0, -1, -1, -1, -1, -1, -1, -1}, 4},  // half d, u16 scales_h, scales_l[4] at 4, qs[128]
    {11, "Q3_K", 110, {108, -1, -1, -1, -1, -1, -1, -1}, 96},
    {20, "IQ4_NL", 144, {0, 18, 36, 54, 72, 90, 108, 126}, 1},   // 8 x (half d, qs[16]) per 256 weights (M2); check 5: d's high byte // hmask[32], qs[64], scales[12] at 96 (6-bit, packed), half d at 108 (M2)
};
const Format* format_of(int t) {
    for (const Format& f : FORMATS) if (f.type == t) return &f;
    return nullptr;
}
const Format XR_EXTRA_FORMATS[] = {
    {8,  "Q8_0",  272, {0, 34, 68, 102, 136, 170, 204, 238}, 1},  // 8 x (half d, int8 qs[32]) per 256 weights; check 5: d's high byte
    {21, "IQ3_S", 110, {0, -1, -1, -1, -1, -1, -1, -1}, 106},      // half d, qs[64], qh[8], signs[32], scales[4] at 106
};
const Format* any_format_of(int t) {
    if (const Format* f = format_of(t)) return f;
    for (const Format& f : XR_EXTRA_FORMATS) if (f.type == t) return &f;
    return nullptr;
}

constexpr int8_t kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
constexpr uint32_t kIq3sGrid[512] = {
    0x01010101u, 0x01010103u, 0x01010105u, 0x0101010bu, 0x0101010fu, 0x01010301u, 0x01010303u, 0x01010305u,
    0x01010309u, 0x0101030du, 0x01010501u, 0x01010503u, 0x0101050bu, 0x01010707u, 0x01010901u, 0x01010905u,
    0x0101090bu, 0x0101090fu, 0x01010b03u, 0x01010b07u, 0x01010d01u, 0x01010d05u, 0x01010f03u, 0x01010f09u,
    0x01010f0fu, 0x01030101u, 0x01030103u, 0x01030105u, 0x01030109u, 0x01030301u, 0x01030303u, 0x0103030bu,
    0x01030501u, 0x01030507u, 0x0103050fu, 0x01030703u, 0x0103070bu, 0x01030909u, 0x01030d03u, 0x01030d0bu,
    0x01030f05u, 0x01050101u, 0x01050103u, 0x0105010bu, 0x0105010fu, 0x01050301u, 0x01050307u, 0x0105030du,
    0x01050503u, 0x0105050bu, 0x01050701u, 0x01050709u, 0x01050905u, 0x0105090bu, 0x0105090fu, 0x01050b03u,
    0x01050b07u, 0x01050f01u, 0x01050f07u, 0x01070107u, 0x01070303u, 0x0107030bu, 0x01070501u, 0x01070505u,
    0x01070703u, 0x01070707u, 0x0107070du, 0x01070909u, 0x01070b01u, 0x01070b05u, 0x01070d0fu, 0x01070f03u,
    0x01070f0bu, 0x01090101u, 0x01090307u, 0x0109030fu, 0x01090503u, 0x01090509u, 0x01090705u, 0x01090901u,
    0x01090907u, 0x01090b03u, 0x01090f01u, 0x010b0105u, 0x010b0109u, 0x010b0501u, 0x010b0505u, 0x010b050du,
    0x010b0707u, 0x010b0903u, 0x010b090bu, 0x010b090fu, 0x010b0d0du, 0x010b0f07u, 0x010d010du, 0x010d0303u,
    0x010d0307u, 0x010d0703u, 0x010d0b05u, 0x010d0f03u, 0x010f0101u, 0x010f0105u, 0x010f0109u, 0x010f0501u,
    0x010f0505u, 0x010f050du, 0x010f0707u, 0x010f0b01u, 0x010f0b09u, 0x03010101u, 0x03010103u, 0x03010105u,
    0x03010109u, 0x03010301u, 0x03010303u, 0x03010307u, 0x0301030bu, 0x0301030fu, 0x03010501u, 0x03010505u,
    0x03010703u, 0x03010709u, 0x0301070du, 0x03010b09u, 0x03010b0du, 0x03010d03u, 0x03010f05u, 0x03030101u,
    0x03030103u, 0x03030107u, 0x0303010du, 0x03030301u, 0x03030309u, 0x03030503u, 0x03030701u, 0x03030707u,
    0x03030903u, 0x03030b01u, 0x03030b05u, 0x03030f01u, 0x03030f0du, 0x03050101u, 0x03050305u, 0x0305030bu,
    0x0305030fu, 0x03050501u, 0x03050509u, 0x03050705u, 0x03050901u, 0x03050907u, 0x03050b0bu, 0x03050d01u,
    0x03050f05u, 0x03070103u, 0x03070109u, 0x0307010fu, 0x03070301u, 0x03070307u, 0x03070503u, 0x0307050fu,
    0x03070701u, 0x03070709u, 0x03070903u, 0x03070d05u, 0x03070f01u, 0x03090107u, 0x0309010bu, 0x03090305u,
    0x03090309u, 0x03090703u, 0x03090707u, 0x03090905u, 0x0309090du, 0x03090b01u, 0x03090b09u, 0x030b0103u,
    0x030b0301u, 0x030b0307u, 0x030b0503u, 0x030b0701u, 0x030b0705u, 0x030b0b03u, 0x030d0501u, 0x030d0509u,
    0x030d050fu, 0x030d0909u, 0x030d090du, 0x030f0103u, 0x030f0107u, 0x030f0301u, 0x030f0305u, 0x030f0503u,
    0x030f070bu, 0x030f0903u, 0x030f0d05u, 0x030f0f01u, 0x05010101u, 0x05010103u, 0x05010107u, 0x0501010bu,
    0x0501010fu, 0x05010301u, 0x05010305u, 0x05010309u, 0x0501030du, 0x05010503u, 0x05010507u, 0x0501050fu,
    0x05010701u, 0x05010705u, 0x05010903u, 0x05010907u, 0x0501090bu, 0x05010b01u, 0x05010b05u, 0x05010d0fu,
    0x05010f01u, 0x05010f07u, 0x05010f0bu, 0x05030101u, 0x05030105u, 0x05030301u, 0x05030307u, 0x0503030fu,
    0x05030505u, 0x0503050bu, 0x05030703u, 0x05030709u, 0x05030905u, 0x05030b03u, 0x05050103u, 0x05050109u,
    0x0505010fu, 0x05050503u, 0x05050507u, 0x05050701u, 0x0505070fu, 0x05050903u, 0x05050b07u, 0x05050b0fu,
    0x05050f03u, 0x05050f09u, 0x05070101u, 0x05070105u, 0x0507010bu, 0x05070303u, 0x05070505u, 0x05070509u,
    0x05070703u, 0x05070707u, 0x05070905u, 0x05070b01u, 0x05070d0du, 0x05090103u, 0x0509010fu, 0x05090501u,
    0x05090507u, 0x05090705u, 0x0509070bu, 0x05090903u, 0x05090f05u, 0x05090f0bu, 0x050b0109u, 0x050b0303u,
    0x050b0505u, 0x050b070fu, 0x050b0901u, 0x050b0b07u, 0x050b0f01u, 0x050d0101u, 0x050d0105u, 0x050d010fu,
    0x050d0503u, 0x050d0b0bu, 0x050d0d03u, 0x050f010bu, 0x050f0303u, 0x050f050du, 0x050f0701u, 0x050f0907u,
    0x050f0b01u, 0x07010105u, 0x07010303u, 0x07010307u, 0x0701030bu, 0x0701030fu, 0x07010505u, 0x07010703u,
    0x07010707u, 0x0701070bu, 0x07010905u, 0x07010909u, 0x0701090fu, 0x07010b03u, 0x07010d07u, 0x07010f03u,
    0x07030103u, 0x07030107u, 0x0703010bu, 0x07030309u, 0x07030503u, 0x07030507u, 0x07030901u, 0x07030d01u,
    0x07030f05u, 0x07030f0du, 0x07050101u, 0x07050305u, 0x07050501u, 0x07050705u, 0x07050709u, 0x07050b01u,
    0x07070103u, 0x07070301u, 0x07070309u, 0x07070503u, 0x07070507u, 0x0707050fu, 0x07070701u, 0x07070903u,
    0x07070907u, 0x0707090fu, 0x07070b0bu, 0x07070f07u, 0x07090107u, 0x07090303u, 0x0709030du, 0x07090505u,
    0x07090703u, 0x07090b05u, 0x07090d01u, 0x07090d09u, 0x070b0103u, 0x070b0301u, 0x070b0305u, 0x070b050bu,
    0x070b0705u, 0x070b0909u, 0x070b0b0du, 0x070b0f07u, 0x070d030du, 0x070d0903u, 0x070f0103u, 0x070f0107u,
    0x070f0501u, 0x070f0505u, 0x070f070bu, 0x09010101u, 0x09010109u, 0x09010305u, 0x09010501u, 0x09010509u,
    0x0901050fu, 0x09010705u, 0x09010903u, 0x09010b01u, 0x09010f01u, 0x09030105u, 0x0903010fu, 0x09030303u,
    0x09030307u, 0x09030505u, 0x09030701u, 0x0903070bu, 0x09030907u, 0x09030b03u, 0x09030b0bu, 0x09050103u,
    0x09050107u, 0x09050301u, 0x0905030bu, 0x09050503u, 0x09050707u, 0x09050901u, 0x09050b0fu, 0x09050d05u,
    0x09050f01u, 0x09070109u, 0x09070303u, 0x09070307u, 0x09070501u, 0x09070505u, 0x09070703u, 0x0907070bu,
    0x09090101u, 0x09090105u, 0x09090509u, 0x0909070fu, 0x09090901u, 0x09090f03u, 0x090b010bu, 0x090b010fu,
    0x090b0503u, 0x090b0d05u, 0x090d0307u, 0x090d0709u, 0x090d0d01u, 0x090f0301u, 0x090f030bu, 0x090f0701u,
    0x090f0907u, 0x090f0b03u, 0x0b010105u, 0x0b010301u, 0x0b010309u, 0x0b010505u, 0x0b010901u, 0x0b010909u,
    0x0b01090fu, 0x0b010b05u, 0x0b010d0du, 0x0b010f09u, 0x0b030103u, 0x0b030107u, 0x0b03010bu, 0x0b030305u,
    0x0b030503u, 0x0b030705u, 0x0b030f05u, 0x0b050101u, 0x0b050303u, 0x0b050507u, 0x0b050701u, 0x0b05070du,
    0x0b050b07u, 0x0b070105u, 0x0b07010fu, 0x0b070301u, 0x0b07050fu, 0x0b070909u, 0x0b070b03u, 0x0b070d0bu,
    0x0b070f07u, 0x0b090103u, 0x0b090109u, 0x0b090501u, 0x0b090705u, 0x0b09090du, 0x0b0b0305u, 0x0b0b050du,
    0x0b0b0b03u, 0x0b0b0b07u, 0x0b0d0905u, 0x0b0f0105u, 0x0b0f0109u, 0x0b0f0505u, 0x0d010303u, 0x0d010307u,
    0x0d01030bu, 0x0d010703u, 0x0d010707u, 0x0d010d01u, 0x0d030101u, 0x0d030501u, 0x0d03050fu, 0x0d030d09u,
    0x0d050305u, 0x0d050709u, 0x0d050905u, 0x0d050b0bu, 0x0d050d05u, 0x0d050f01u, 0x0d070101u, 0x0d070309u,
    0x0d070503u, 0x0d070901u, 0x0d09050bu, 0x0d090907u, 0x0d090d05u, 0x0d0b0101u, 0x0d0b0107u, 0x0d0b0709u,
    0x0d0b0d01u, 0x0d0d010bu, 0x0d0d0901u, 0x0d0f0303u, 0x0d0f0307u, 0x0f010101u, 0x0f010109u, 0x0f01010fu,
    0x0f010501u, 0x0f010505u, 0x0f01070du, 0x0f010901u, 0x0f010b09u, 0x0f010d05u, 0x0f030105u, 0x0f030303u,
    0x0f030509u, 0x0f030907u, 0x0f03090bu, 0x0f050103u, 0x0f050109u, 0x0f050301u, 0x0f05030du, 0x0f050503u,
    0x0f050701u, 0x0f050b03u, 0x0f070105u, 0x0f070705u, 0x0f07070bu, 0x0f070b07u, 0x0f090103u, 0x0f09010bu,
    0x0f090307u, 0x0f090501u, 0x0f090b01u, 0x0f0b0505u, 0x0f0b0905u, 0x0f0d0105u, 0x0f0d0703u, 0x0f0f0101u,
};

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
    } else if (t == 8) {   // ggml dequantize_row_q8_0, 8 blocks of 34 B (half d, int8 qs[32])
        for (int ib = 0; ib < 8; ++ib) {
            const uint8_t* q = b + 34 * ib;
            const double d = h2f(q);
            for (int j = 0; j < 32; ++j)
                y[32 * ib + j] = d * (int8_t) q[2 + j];
        }
    } else if (t == 21) {   // ggml dequantize_row_iq3_s: half d, qs[64], qh[8], signs[32], scales[4]
        const double d = h2f(b);
        const uint8_t* qs = b + 2;
        const uint8_t* qh = b + 66;
        const uint8_t* signs = b + 74;
        const uint8_t* scales = b + 106;
        for (int ib = 0; ib < 8; ++ib) {
            const double dl = d * (1 + 2 * ((scales[ib / 2] >> (4 * (ib & 1))) & 0xf));
            for (int l = 0; l < 8; ++l) {
                const uint32_t g = kIq3sGrid[qs[8 * ib + l] | (((qh[ib] >> l) & 1u) << 8)];
                const uint8_t sb = (signs[4 * ib + (l >> 1)] >> (4 * (l & 1))) & 0xfu;
                for (int j = 0; j < 4; ++j) {
                    const int v = (int) ((g >> (8 * j)) & 0xffu);
                    y[32 * ib + 4 * l + j] = dl * ((sb & (1u << j)) ? -v : v);
                }
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
    const int bb = any_format_of(t)->block_bytes;
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

struct Shape { int n_in, n_out; };
// n_in <= 4096 and > 4096 take different tiles; 768 and 256 leave lanes of the last sub-group idle; n_out 1000, 520,
// 300, 77 and 3 end in a partial tile.
// 2560 x 4099: more tiles than the persistent KREG2 grid (its default engages at 2-4 columns there, KREG at 1 column:
// check 4 then compares the two kernels bit for bit), and a partial last tile.
const Shape SHAPES[] = {{2560, 1000}, {6144, 520}, {768, 300}, {256, 77}, {2560, 3}, {2560, 4099},
                        {5120, 600}, {17408, 300}};   // (M2: the Qwen3.8-27B widths; 17408 is 544 lanes, past a 256-register group;
                                                      // M4, STRATA_KREG_KS=4: its K in 4 parts, 5-8 columns in one launch pair)

struct Dev {
    sycl::queue& q;
    void* alloc(size_t n) { return sycl::malloc_device(n, q); }
    void put(void* d, const void* h, size_t n) { q.memcpy(d, h, n).wait(); }
    void get(void* h, const void* d, size_t n) { q.memcpy(h, d, n).wait(); }
};

// one product: the dense forms (which must take the call: a decline fails the run) or the default kernels
void call(bool dense, int t, const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (!dense) { k::native_mmvq(t, w, xq, y, n_in, n_out, ncols, stream); return; }
    if (!strata::q36dense::mmvq(t, w, xq, y, n_in, n_out, ncols, stream)) throw std::runtime_error("the dense forms declined a call of this check");
}

// Engine's <=8 dispatch: XR when it handles (t, shape, snc), else KREG (q36dense), else native_mmvq.
void call_eng8(int t, const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (strata::q36xr::mmvq(t, w, xq, y, n_in, n_out, ncols, stream, true)) return;
    if (strata::q36dense::mmvq(t, w, xq, y, n_in, n_out, ncols, stream)) return;
    k::native_mmvq_set_kreg(0);
    k::native_mmvq(t, w, xq, y, n_in, n_out, ncols, stream);
}

std::vector<float> run(Dev& dev, int t, const void* dw, const uint8_t* dxq, float* dy, int n_in, int n_out, int T,
                       int kreg, bool per_column) {
    k::native_mmvq_set_kreg(0);   // (the default kernels: native_mmvq without KREG)
    const size_t qcol = k::native_q8_1_bytes(n_in, 1);
    dev.q.memset(dy, 0xff, (size_t) T * n_out * 4).wait();   // NaN fill: an output not written fails
    if (per_column)
        for (int j = 0; j < T; ++j) call(kreg != 0, t, dw, dxq + (size_t) j * qcol, dy + (size_t) j * n_out, n_in, n_out, 1, &dev.q);
    else
        call(kreg != 0, t, dw, dxq, dy, n_in, n_out, T, &dev.q);
    dev.q.wait_and_throw();
    std::vector<float> y((size_t) T * n_out);
    dev.get(y.data(), dy, y.size() * 4);
    return y;
}

double xr_rel(int n_in, int t = 0) {
    const int bpr = n_in / 256, c = n_in > 16384 ? 2 : 4, cb = (bpr + c - 1) / c;
    const int per_blk = (t == 14 || t == 11) ? 16 : 8;
    const int h = per_blk * cb + c + 6;
    const double u = std::ldexp(1.0, -24);
    return 2.0 * h * u / (1.0 - h * u);
}

const Shape XR_SHAPES[] = {{17408, 5120}, {5120, 17408}, {6144, 5120}, {5120, 6144}, {5120, 10240}, {5120, 1024}};

int selftest_xr(sycl::queue& q, Dev& dev) {
    const int xr_types[] = {8, 11, 12, 13, 14, 20, 21, 23};
    const char* xr16_env = std::getenv("STRATA_Q36_XR16");
    const bool xr16_off = xr16_env && xr16_env[0] == '0';
    if (xr16_off) std::printf("\nxr16 selftest skipped (STRATA_Q36_XR16=0)\n");
    strata::q36xr::init(&q);
    int bad = 0;
    bool h8 = false, h16 = false;
    for (int t : xr_types) {
        const Format* f = any_format_of(t);
        if (!f) continue;
        for (const Shape& sh : XR_SHAPES) {
            const bool do8 = strata::q36xr::handles_call(t, sh.n_in, sh.n_out, 8, true);
            const bool do16 = !xr16_off && strata::q36xr::handles_call(t, sh.n_in, sh.n_out, 16, true);
            if (!do8 && !do16) continue;
            const int TMAX = do16 ? 16 : 8;
            const size_t wbytes = k::native_mmvq_weight_bytes(t, sh.n_in, sh.n_out);
            const size_t nblk = wbytes / f->block_bytes;
            std::mt19937 rng(991u + 37u * (unsigned) t + (unsigned) sh.n_in + 13u * (unsigned) sh.n_out);
            std::vector<uint8_t> w(wbytes);
            for (auto& b : w) b = (uint8_t) (rng() & 0xff);
            for (size_t b = 0; b < nblk; ++b)
                for (int at : f->scale_at)
                    if (at >= 0) { const uint16_t h = sane_half(rng); std::memcpy(&w[b * f->block_bytes + at], &h, 2); }
            const uint16_t big = 0x27FF;
            std::memcpy(&w[f->scale_at[0]], &big, 2);

            std::vector<float> hx((size_t) TMAX * sh.n_in);
            std::normal_distribution<float> nd(0.f, 1.f);
            for (auto& v : hx) v = nd(rng);

            void* dw = dev.alloc(wbytes);
            float* dx = (float*) dev.alloc(hx.size() * 4);
            const size_t qcol = k::native_q8_1_bytes(sh.n_in, 1);
            uint8_t* dxq = (uint8_t*) dev.alloc(qcol * TMAX + 64);
            float* dy = (float*) dev.alloc((size_t) TMAX * sh.n_out * 4);
            dev.put(dw, w.data(), wbytes);
            dev.put(dx, hx.data(), hx.size() * 4);
            // Flatten contiguous columns as one vector (MAX_NCOLS = 8 unless STRATA_VERIFY_MAX_T > 8, same as Engine::mmq).
            k::native_quantize_q8_1(dx, dxq, sh.n_in * TMAX, 1, &q);
            q.wait_and_throw();
            std::vector<uint8_t> xq(qcol * TMAX);
            dev.get(xq.data(), dxq, xq.size());
            std::vector<double> x;
            dequant_q8_1(xq.data(), sh.n_in, TMAX, x);

            // Sample rows across start, middle and tail tiles for fp64 reference; compare 100% of elements across variants.
            std::vector<int> sample_rows;
            const int mid = (sh.n_out / 2) & ~15;
            for (int r = 0; r < sh.n_out; ++r) {
                if (r < 64 || (r >= mid && r < mid + 64) || r >= sh.n_out - 64) sample_rows.push_back(r);
            }
            std::vector<double> ref((size_t) TMAX * sample_rows.size(), 0.0), mag(ref.size(), 0.0), mag2(ref.size(), 0.0);
            {
                const int bpr = sh.n_in / 256, bb = f->block_bytes;
                std::vector<double> rrow(sh.n_in), rmin(sh.n_in);
                for (size_t ri = 0; ri < sample_rows.size(); ++ri) {
                    const int r = sample_rows[ri];
                    for (int b = 0; b < bpr; ++b)
                        dequant_block(t, &w[((size_t) r * bpr + b) * bb], &rrow[(size_t) 256 * b], &rmin[(size_t) 256 * b]);
                    for (int j = 0; j < TMAX; ++j) {
                        const double* xj = &x[(size_t) j * sh.n_in];
                        double s = 0, m = 0, m2 = 0;
                        for (int i = 0; i < sh.n_in; ++i) {
                            s += rrow[i] * xj[i];
                            m += std::fabs(rrow[i] * xj[i]);
                            m2 += std::fabs((rrow[i] + rmin[i]) * xj[i]) + std::fabs(rmin[i] * xj[i]);
                        }
                        ref[(size_t) j * sample_rows.size() + ri] = s;
                        mag[(size_t) j * sample_rows.size() + ri] = m;
                        mag2[(size_t) j * sample_rows.size() + ri] = m2;
                    }
                }
            }
            const double rel_new = xr_rel(sh.n_in, t);
            auto check_power = [&](const std::vector<float>& yvec, int T) {
                std::vector<uint8_t> w2(w.begin(), w.begin() + wbytes / sh.n_out);
                w2[f->sub0_scale_at] ^= 1;
                std::vector<double> ref2, m2a, m2b;
                reference(t, w2, sh.n_in, 1, 0, 1, x, T, ref2, m2a, m2b);
                double seen = 0;
                for (int j = 0; j < T; ++j)
                    seen = std::max(seen, std::fabs(yvec[(size_t) j * sh.n_out] - ref2[j]) / (rel_new * mag2[(size_t) j * sample_rows.size()] + 1e-30));
                std::printf("%-7s %5d x %5d     power: one scale step of row 0 is %.1f x the bound\n", f->name, sh.n_in, sh.n_out, seen);
                if (seen <= 1.0) ++bad;
            };
            if (do8) {
                if (!h8) {
                    std::printf("\n%-7s %5s x %5s %2s  %-11s %-11s %9s %9s %9s %s\n", "xr8", "n_in", "n_out", "T", "xr8/bound",
                                "dflt/bound", "xr8!=dflt", "multi!=1c", "nonfinite", "");
                    std::fflush(stdout);
                    h8 = true;
                }
                long long case_diff = 0;
                for (int T = 1; T <= 8; ++T) {
                    q.memset(dy, 0xff, (size_t) T * sh.n_out * 4).wait();
                    if (!strata::q36xr::mmvq(t, dw, dxq, dy, sh.n_in, sh.n_out, T, &q, true))
                        throw std::runtime_error("XR8 declined a selftest call");
                    q.wait_and_throw();
                    std::vector<float> yx((size_t) T * sh.n_out);
                    dev.get(yx.data(), dy, yx.size() * 4);

                    k::native_mmvq_set_kreg(0);
                    q.memset(dy, 0xff, (size_t) T * sh.n_out * 4).wait();
                    k::native_mmvq(t, dw, dxq, dy, sh.n_in, sh.n_out, T, &q);
                    q.wait_and_throw();
                    std::vector<float> yd((size_t) T * sh.n_out);
                    dev.get(yd.data(), dy, yd.size() * 4);

                    q.memset(dy, 0xff, (size_t) T * sh.n_out * 4).wait();
                    for (int j = 0; j < T; ++j)
                        if (!strata::q36xr::mmvq(t, dw, dxq + (size_t) j * qcol, dy + (size_t) j * sh.n_out, sh.n_in, sh.n_out, 1, &q, true))
                            throw std::runtime_error("XR8 1-col declined a selftest call");
                    q.wait_and_throw();
                    std::vector<float> y1((size_t) T * sh.n_out);
                    dev.get(y1.data(), dy, y1.size() * 4);

                    long long kd = 0, md = 0, nf = 0;
                    for (size_t i = 0; i < yx.size(); ++i) {
                        if (!std::isfinite(yx[i]) || !std::isfinite(yd[i]) || !std::isfinite(y1[i])) { ++nf; continue; }
                        uint32_t a, b, c;
                        std::memcpy(&a, &yx[i], 4); std::memcpy(&b, &yd[i], 4); std::memcpy(&c, &y1[i], 4);
                        kd += (a != b);
                        md += (a != c);
                    }
                    case_diff += kd;
                    double r_new = 0.0, r_old = 0.0;
                    for (int j = 0; j < T; ++j)
                        for (size_t ri = 0; ri < sample_rows.size(); ++ri) {
                            const size_t oi = (size_t) j * sh.n_out + sample_rows[ri];
                            const size_t si = (size_t) j * sample_rows.size() + ri;
                            if (std::isfinite(yx[oi])) r_new = std::max(r_new, std::fabs(yx[oi] - ref[si]) / (rel_new * mag2[si] + 1e-30));
                            if (std::isfinite(yd[oi])) r_old = std::max(r_old, std::fabs(yd[oi] - ref[si]) / (kBound * mag[si] + 1e-30));
                        }
                    const bool ok = nf == 0 && r_new <= 1.0 && r_old <= 1.0 && md == 0;
                    std::printf("%-7s %5d x %5d %2d  %-11.3f %-11.3f %9lld %9lld %9lld %s\n", f->name, sh.n_in, sh.n_out, T,
                                r_new, r_old, kd, md, nf, ok ? "ok" : "FAIL");
                    if (!ok) ++bad;
                    if (T == 8) check_power(yx, 8);
                }
                if (sh.n_in >= 2560 && case_diff == 0) {
                    std::printf("%-7s %5d x %5d     XR8 output bitwise equal to default kernels: did not reach XR\n",
                                f->name, sh.n_in, sh.n_out);
                    ++bad;
                }
            }
            if (do16) {
                if (!h16) {
                    std::printf("\n%-7s %5s x %5s %2s  %-11s %-11s %9s %9s %9s %s\n", "xr16", "n_in", "n_out", "T", "xr16/bound",
                                "split/bnd", "diff_vs8", "vs_split", "nonfinite", "");
                    std::fflush(stdout);
                    h16 = true;
                }
                const double rel_old = do8 ? rel_new : ((t == 8 || t == 21) ? kBound : kreg_rel(sh.n_in));
                const bool use_mag = !do8 && (t == 8 || t == 21);
                for (int T = 9; T <= 16; ++T) {
                    q.memset(dy, 0xff, (size_t) T * sh.n_out * 4).wait();
                    if (!strata::q36xr::mmvq(t, dw, dxq, dy, sh.n_in, sh.n_out, T, &q, true))
                        throw std::runtime_error("XR16 declined a selftest call");
                    q.wait_and_throw();
                    std::vector<float> y16((size_t) T * sh.n_out);
                    dev.get(y16.data(), dy, y16.size() * 4);

                    q.memset(dy, 0xff, (size_t) T * sh.n_out * 4).wait();
                    for (int c0 = 0; c0 < T; c0 += 8) {
                        call_eng8(t, dw, dxq + (size_t) c0 * qcol, dy + (size_t) c0 * sh.n_out, sh.n_in, sh.n_out,
                                  std::min(8, T - c0), &q);
                    }
                    q.wait_and_throw();
                    std::vector<float> y8((size_t) T * sh.n_out);
                    dev.get(y8.data(), dy, y8.size() * 4);

                    long long diff8 = 0, nf = 0;
                    double rms_ss = 0.0, rms_sd = 0.0;
                    for (size_t i = 0; i < y16.size(); ++i) {
                        if (!std::isfinite(y16[i]) || !std::isfinite(y8[i])) { ++nf; continue; }
                        uint32_t a, b;
                        std::memcpy(&a, &y16[i], 4);
                        std::memcpy(&b, &y8[i], 4);
                        diff8 += (a != b);
                        const double dlt = (double) y16[i] - (double) y8[i];
                        rms_ss += (double) y8[i] * y8[i];
                        rms_sd += dlt * dlt;
                    }
                    const double split_rel = std::sqrt(rms_sd / std::max(rms_ss, 1e-30));

                    double r_new = 0.0, r_old = 0.0;
                    for (int j = 0; j < T; ++j)
                        for (size_t ri = 0; ri < sample_rows.size(); ++ri) {
                            const size_t oi = (size_t) j * sh.n_out + sample_rows[ri];
                            const size_t si = (size_t) j * sample_rows.size() + ri;
                            const double m_old = use_mag ? mag[si] : mag2[si];
                            if (std::isfinite(y16[oi])) r_new = std::max(r_new, std::fabs(y16[oi] - ref[si]) / (rel_new * mag2[si] + 1e-30));
                            if (std::isfinite(y8[oi])) r_old = std::max(r_old, std::fabs(y8[oi] - ref[si]) / (rel_old * m_old + 1e-30));
                        }
                    const bool ok = nf == 0 && r_new <= 1.0 && r_old <= 1.0 && split_rel < (do8 ? 1e-5 : 5e-4);
                    std::printf("%-7s %5d x %5d %2d  %-11.3f %-11.3f %9lld %9.2e %9lld %s\n", f->name, sh.n_in, sh.n_out, T,
                                r_new, r_old, diff8, split_rel, nf, ok ? "ok" : "FAIL");
                    if (!ok) ++bad;
                    if (T == 16) check_power(y16, 16);
                }
            }
            sycl::free(dw, q); sycl::free(dx, q); sycl::free(dxq, q); sycl::free(dy, q);
        }
    }
    return bad;
}

int selftest() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    Dev dev{q};
    std::printf("q36_mmvq_dense_parity on %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("%-7s %5s x %5s %2s  %-11s %-11s %9s %9s %9s %s\n", "type", "n_in", "n_out", "T", "kreg/bound",
                "dflt/bound", "kreg!=dflt", "multi!=1c", "nonfinite", "");
    int bad = 0;
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
            std::vector<double> x, ref, mag, mag2;
            dequant_q8_1(xq.data(), sh.n_in, TMAX, x);
            reference(f.type, w, sh.n_in, sh.n_out, 0, sh.n_out, x, TMAX, ref, mag, mag2);   // T columns = the first T of these
            const double krel = kreg_rel(sh.n_in);

            long long case_diff = 0;
            for (int T = 1; T <= TMAX; ++T) {
                const std::vector<float> yk = run(dev, f.type, dw, dxq, dy, sh.n_in, sh.n_out, T, 1, false);
                const std::vector<float> yd = run(dev, f.type, dw, dxq, dy, sh.n_in, sh.n_out, T, 0, false);
                const std::vector<float> y1 = run(dev, f.type, dw, dxq, dy, sh.n_in, sh.n_out, T, 1, true);
                double rk = 0, rd = 0;
                long long kd = 0, md = 0, nf = 0;
                for (int j = 0; j < T; ++j)
                    for (int r = 0; r < sh.n_out; ++r) {
                        const size_t i = (size_t) j * sh.n_out + r;
                        const double bnd = kBound * mag[i] + 1e-30, bk = krel * mag2[i] + 1e-30;
                        if (!std::isfinite(yk[i]) || !std::isfinite(yd[i]) || !std::isfinite(y1[i])) { ++nf; continue; }
                        rk = std::max(rk, std::fabs(yk[i] - ref[i]) / bk);
                        rd = std::max(rd, std::fabs(yd[i] - ref[i]) / bnd);
                        uint32_t a, b, c;
                        std::memcpy(&a, &yk[i], 4); std::memcpy(&b, &yd[i], 4); std::memcpy(&c, &y1[i], 4);
                        kd += a != b;
                        md += a != c;
                    }
                case_diff += kd;
                const bool ok = nf == 0 && rk <= 1.0 && rd <= 1.0 && md == 0;
                std::printf("%-7s %5d x %5d %2d  %-11.3f %-11.3f %9lld %9lld %9lld %s\n", f.name, sh.n_in, sh.n_out, T,
                            rk, rd, kd, md, nf, ok ? "ok" : "FAIL");
                if (!ok) ++bad;
                if (T == TMAX) {   // 5. power: row 0 against a reference with sub-block 0's scale one step off, check 1's bound
                    std::vector<uint8_t> w2(w.begin(), w.begin() + wbytes / sh.n_out);
                    w2[f.sub0_scale_at] ^= 1;
                    std::vector<double> ref2, m2a, m2b;
                    reference(f.type, w2, sh.n_in, 1, 0, 1, x, TMAX, ref2, m2a, m2b);
                    double seen = 0;
                    for (int j = 0; j < TMAX; ++j)
                        seen = std::max(seen, std::fabs(yk[(size_t) j * sh.n_out] - ref2[j]) / (krel * mag2[(size_t) j * sh.n_out] + 1e-30));
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
    bad += selftest_xr(q, dev);
    k::native_mmvq_set_kreg(-1);
    std::printf("\nq36_mmvq_dense_parity: %d failures\n", bad);
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
    sycl::queue q(sycl::gpu_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    struct BShape { int n_in, n_out; const char* name; };
    // Qwen3.8-Flash-Next decode shapes; the head is 2560 x 248320.  27B: Qwen3.8-27B's (5120 wide, FFN 17408, head 5120 x 248320)
    const BShape shapes[] = {{2560, 10240, "qkv"}, {2560, 6144, "gate"}, {6144, 2560, "out"}, {2560, 12288, "attn_q"},
                             {2560, 512, "kv"}, {2560, 640, "sh_gu"}, {2560, 248320, "head"},
                             {5120, 10240, "27qkv"}, {5120, 1024, "27kv"}, {5120, 6144, "27gate"}, {6144, 5120, "27out"}, {5120, 12288, "27q"},
                             {5120, 17408, "27gu"}, {17408, 5120, "27down"}, {5120, 248320, "27head"}};
    const auto selected = [&](const BShape& sh) {
        return ssel == "27" ? sh.name[0] == '2' && sh.name[1] == '7' : (ssel == "all" || ssel == sh.name);
    };
    // THE POOL.  Every weight matrix of a launch must lie inside it: a read past the end of the allocation is a device page
    // fault, and on the xe driver that ends in "exec queue reset" and UR_RESULT_ERROR_DEVICE_LOST.  (2026-11-11: the pool was
    // a fixed 640 MiB; the 27head shape, 5120 x 248320, is 715 MB in Q4_K, 874 MB in Q5_K, 1043 MB in Q6_K, 675 MB in IQ4_XS,
    // one copy of it read past the end by 42 MiB (Q4_K) to 372 MB (Q6_K), and the card was lost on the first row after 27down.)  So the pool is
    // the largest selected matrix when that is above 640 MiB, and a launch whose matrix does not fit is skipped with a message.
    size_t pool = (size_t) 640 << 20;
    for (const int t : types) {
        if (!any_format_of(t)) continue;
        for (const BShape& sh : shapes)
            if (selected(sh)) pool = std::max(pool, (size_t) k::native_mmvq_weight_bytes(t, sh.n_in, sh.n_out));
    }
    const size_t max_alloc = (size_t) q.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
    if (pool + 4096 > max_alloc) pool = (max_alloc > 4096 ? max_alloc - 4096 : 0) & ~(size_t) 255;
    const size_t POOL = pool;
    const int maxc = 16, maxout = 248320;
    const size_t xq_bytes = k::native_q8_1_bytes(17408 * maxc, 1) + 256;
    uint8_t* dpool = (uint8_t*) sycl::malloc_device(POOL + 4096, q);
    float* y = sycl::malloc_device<float>((size_t) maxc * maxout, q);
    std::vector<float> hx((size_t) maxc * 17408);
    float* dx = sycl::malloc_device<float>(hx.size(), q);
    void* xq = sycl::malloc_device(xq_bytes, q);
    void* xact = sycl::malloc_device(strata::q36xr::act_bytes(17408) + 256, q);   // the XMX form's activation block
    if (!dpool || !y || !dx || !xq || !xact) {
        std::printf("bench: could not allocate the buffers (weight pool %.0f MiB): stopping before any launch\n", POOL / 1048576.0);
        return 1;
    }
    std::vector<uint8_t> hpool(POOL);
    std::mt19937 rng(4242);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : hx) v = nd(rng);
    q.memcpy(dx, hx.data(), hx.size() * 4).wait();
    strata::q36xr::init(&q);
    std::printf("q36_mmvq_dense_parity --bench on %s: us per call (median), GB/s of weight bytes\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());
    int bad = 0;
    for (const int t : types) {
        const Format* f = any_format_of(t);
        if (!f) { std::printf("type %d: not a supported parity format (8, 11, 12, 13, 14, 20, 21, 23)\n", t); continue; }
        const bool is_kreg = format_of(t) != nullptr;
        fill_pool(*f, hpool);
        q.memcpy(dpool, hpool.data(), POOL).wait();
        for (const BShape& sh : shapes) {
            if (!selected(sh)) continue;
            const size_t wb = k::native_mmvq_weight_bytes(t, sh.n_in, sh.n_out);
            const size_t slot = (wb + 255) & ~(size_t) 255;
            if (wb > POOL) {   // (a launch here would read past the pool: see the note above)
                std::printf("%-6s %-6s %5d x %6d: skipped, the weights (%.0f MiB) do not fit the pool (%.0f MiB)\n", f->name, sh.name,
                            sh.n_in, sh.n_out, wb / 1048576.0, POOL / 1048576.0);
                continue;
            }
            const int ndist = (int) std::max<size_t>(1, std::min<size_t>(128, POOL / slot)), nmat = std::max(ndist, 16);
            const size_t qb = k::native_q8_1_bytes(sh.n_in, 1);
            for (const int nc : ncl) {
                if (nc < 1 || nc > maxc) continue;
                if ((size_t) nc * sh.n_in > hx.size() || (size_t) nc * sh.n_out > (size_t) maxc * maxout ||
                    qb * nc > xq_bytes) {
                    std::printf("%-6s %-6s %5d x %6d nc %d: skipped, an activation or output buffer is too small\n", f->name, sh.name,
                                sh.n_in, sh.n_out, nc);
                    continue;
                }
                k::native_quantize_q8_1(dx, xq, sh.n_in * nc, 1, &q);
                q.wait();
                auto timed = [&](int mode) {
                    k::native_mmvq_set_kreg(0);
                    call(mode != 0, t, dpool, xq, y, sh.n_in, sh.n_out, nc, &q);
                    q.wait();
                    return time_graph(q, nmat, [&](int m) {
                        call(mode != 0, t, dpool + (size_t) (m % ndist) * slot, xq, y, sh.n_in, sh.n_out, nc, &q);
                    }, wb > (100u << 20) ? 7 : 11);
                };
                if (nc > 8) {
                    auto run_split = [&](int mode, const uint8_t* wptr) {
                        for (int c0 = 0; c0 < nc; c0 += 8) {
                            const int snc = std::min(8, nc - c0);
                            const void* qptr = (const char*) xq + (size_t) c0 * qb;
                            float* yptr = y + (size_t) c0 * sh.n_out;
                            if (mode == 2) call_eng8(t, wptr, qptr, yptr, sh.n_in, sh.n_out, snc, &q);
                            else call(mode != 0, t, wptr, qptr, yptr, sh.n_in, sh.n_out, snc, &q);
                        }
                    };
                    auto timed_split = [&](int mode) {
                        // mode: 0 = default split (8 + rest), 1 = KREG split, 2 = engine <=8 split (XR -> KREG -> native)
                        k::native_mmvq_set_kreg(0);
                        run_split(mode, dpool);   // warm up outside graph recording so lazy scratch buffers allocate safely
                        q.wait();
                        return time_graph(q, nmat, [&](int m) {
                            run_split(mode, dpool + (size_t) (m % ndist) * slot);
                        }, wb > (100u << 20) ? 7 : 11);
                    };
                    if (strata::q36xr::handles_call(t, sh.n_in, sh.n_out, nc, true)) {
                        const bool xr8 = strata::q36xr::handles_call(t, sh.n_in, sh.n_out, 8, true);
                        std::vector<float> hold((size_t) nc * sh.n_out), hnew((size_t) nc * sh.n_out);
                        run_split(2, dpool);
                        q.wait();
                        q.memcpy(hold.data(), y, hold.size() * 4).wait();
                        if (!strata::q36xr::mmvq(t, dpool, xq, y, sh.n_in, sh.n_out, nc, &q, true))
                            throw std::runtime_error("XR16 declined a bench call");
                        q.wait();
                        q.memcpy(hnew.data(), y, hnew.size() * 4).wait();
                        double ss = 0, sd = 0;
                        long long bdiff = 0;
                        for (size_t i = 0; i < hold.size(); ++i) {
                            uint32_t a, b;
                            std::memcpy(&a, &hold[i], 4); std::memcpy(&b, &hnew[i], 4);
                            bdiff += (a != b);
                            const double dlt = (double) hnew[i] - hold[i];
                            ss += (double) hold[i] * hold[i]; sd += dlt * dlt;
                        }
                        const double rel = std::sqrt(sd / std::max(ss, 1e-30));
                        const bool ok = rel < (xr8 ? 1e-4 : 5e-4);
                        if (!ok) ++bad;
                        const double u_old = timed_split(2);
                        const double u_new = time_graph(q, nmat, [&](int m) {
                            if (!strata::q36xr::mmvq(t, dpool + (size_t) (m % ndist) * slot, xq, y, sh.n_in, sh.n_out, nc, &q, true))
                                throw std::runtime_error("XR16 declined a bench call");
                        }, wb > (100u << 20) ? 7 : 11);
                        std::printf("%-6s %-6s %5d x %6d nc %2d  %-8s %8.2f us %5.1f GB/s  xr16 %8.2f us %5.1f GB/s  x%.2f  rms diff / rms %.2e (diff %lld/%zu)%s\n",
                                    f->name, sh.name, sh.n_in, sh.n_out, nc, xr8 ? "xr8+rest" : "eng8+rst",
                                    u_old, wb / u_old / 1e3, u_new, wb / u_new / 1e3,
                                    u_old / u_new, rel, bdiff, hold.size(), ok ? "" : "  FAIL");
                    } else if (!is_kreg) {
                        const double u0 = timed_split(0);
                        if (strata::q36xr::handles_call(t, sh.n_in, sh.n_out, 8, true)) {
                            const double u2 = timed_split(2);
                            std::printf("%-6s %-6s %5d x %6d nc %2d  dflt8+r %8.2f us %5.1f GB/s  xr8+r   %8.2f us %5.1f GB/s  x%.2f\n",
                                        f->name, sh.name, sh.n_in, sh.n_out, nc, u0, wb / u0 / 1e3, u2, wb / u2 / 1e3, u0 / u2);
                        } else {
                            std::printf("%-6s %-6s %5d x %6d nc %2d  dflt8+r %8.2f us %5.1f GB/s\n",
                                        f->name, sh.name, sh.n_in, sh.n_out, nc, u0, wb / u0 / 1e3);
                        }
                    } else {
                        const double u0 = timed_split(0);
                        const double u1 = timed_split(1);
                        std::printf("%-6s %-6s %5d x %6d nc %2d  dflt8+r %8.2f us %5.1f GB/s  kreg8+r %8.2f us %5.1f GB/s  x%.2f\n",
                                    f->name, sh.name, sh.n_in, sh.n_out, nc, u0, wb / u0 / 1e3, u1, wb / u1 / 1e3, u0 / u1);
                    }
                    std::fflush(stdout);
                    continue;
                }
                if ((!is_kreg || std::getenv("STRATA_BENCH_XR")) && strata::q36xr::handles_call(t, sh.n_in, sh.n_out, nc, true)) {
                    const int base_mode = is_kreg ? 1 : 0;
                    const char* base_name = is_kreg ? "kreg" : "dflt";
                    const double uk = timed(base_mode);
                    q.wait();
                    std::vector<float> hk((size_t) nc * sh.n_out), hx_((size_t) nc * sh.n_out);
                    q.memcpy(hk.data(), y, hk.size() * 4).wait();
                    const double ux = time_graph(q, nmat, [&](int m) {
                        if (!strata::q36xr::mmvq(t, dpool + (size_t) (m % ndist) * slot, xq, y, sh.n_in, sh.n_out, nc, &q, true))
                            throw std::runtime_error("the XMX form declined a bench call");
                    }, wb > (100u << 20) ? 7 : 11);
                    q.wait();
                    q.memcpy(hx_.data(), y, hx_.size() * 4).wait();
                    double ss = 0, sd = 0, md = 0;
                    for (size_t i = 0; i < hk.size(); ++i) {
                        const double dlt = (double) hx_[i] - hk[i];
                        ss += (double) hk[i] * hk[i]; sd += dlt * dlt; md = std::max(md, std::fabs(dlt));
                    }
                    std::printf("%-6s %-6s %5d x %6d nc %2d  %-4s %8.2f us %5.1f GB/s  xr %8.2f us %5.1f GB/s  x%.2f  rms diff / rms %.2e\n",
                                f->name, sh.name, sh.n_in, sh.n_out, nc, base_name, uk, wb / uk / 1e3, ux, wb / ux / 1e3, uk / ux,
                                std::sqrt(sd / std::max(ss, 1e-30)));
                    std::fflush(stdout);
                    continue;
                }
                if (!is_kreg) {
                    const double u = timed(0);
                    std::printf("%-6s %-6s %5d x %6d nc %2d  dflt %8.2f us %5.1f GB/s  %7.2f us/col\n", f->name, sh.name,
                                sh.n_in, sh.n_out, nc, u, wb / u / 1e3, u / nc);
                    std::fflush(stdout);
                    continue;
                }
                if (std::getenv("STRATA_BENCH_KREG_ONLY")) {   // (M2) KREG alone: us, GB/s, and us per column
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
    sycl::free(dpool, q); sycl::free(y, q); sycl::free(dx, q); sycl::free(xq, q); sycl::free(xact, q);
    return bad ? 1 : 0;
}

// --pp / --pp-bench [types] [ncols] [shape]: prompt-path (T > 16) int8 DPAS GEMM (q36pp::gemm) parity & throughput on 27B
// shapes vs float64 (sampled rows & columns) and vs dequant_f16 + oneMKL FP16 GEMM (including B70W4 = 1004).
struct PpFormat {
    int type;
    const char* name;
    int blk_bytes_256;
};
const PpFormat kPpFormats[] = {
    {1004, "B70W4",  136},
    {  12, "Q4_K",   144},
    {  13, "Q5_K",   176},
    {  23, "IQ4_XS", 136},
    {  14, "Q6_K",   210},
    {  11, "Q3_K",   110},
    {  20, "IQ4_NL", 144},
    {   8, "Q8_0",   272},
};
const PpFormat* pp_format_of(int t) {
    for (const PpFormat& f : kPpFormats) if (f.type == t) return &f;
    return nullptr;
}
void fill_pp_pool(int t, std::vector<uint8_t>& h) {
    if (t == 1004) {
        const size_t nrec = h.size() / 2176;
        uint32_t* dw = reinterpret_cast<uint32_t*>(h.data());
        for (size_t i = 0; i < (nrec * 544) / 2; ++i) {
            const uint64_t v = xr();
            dw[2 * i]     = (uint32_t) v;
            dw[2 * i + 1] = (uint32_t) (v >> 32);
        }
        for (size_t r = 0; r < nrec; ++r) {
            for (int k = 0; k < 32; ++k) {
                const uint64_t v = xr();
                dw[r * 544 + 512 + k] = (0x1400u | (uint32_t) (v & 0x3ff)) | ((1u + (uint32_t) ((v >> 20) & 15)) << 16);
            }
        }
        return;
    }
    if (const Format* f = format_of(t)) {
        fill_pool(*f, h);
        return;
    }
    // Q8_0 (272 B per 256 weights = 8 x 34-B blocks, half d at +34*b)
    uint64_t* p = (uint64_t*) h.data();
    for (size_t i = 0; i < h.size() / 8; ++i) p[i] = xr();
    const size_t nb = h.size() / 272;
    for (size_t b = 0; b < nb; ++b)
        for (int sb = 0; sb < 8; ++sb) {
            const uint16_t d = (uint16_t) (0x1400 | (xr() & 0x3ff));
            std::memcpy(&h[b * 272 + (size_t) sb * 34], &d, 2);
        }
}
void pp_ref_rows(int t, const std::vector<uint8_t>& w_all, int n_in, int n_out, const std::vector<int>& sample_rows,
                 const std::vector<double>& x_samp, int ns_cols, std::vector<double>& out_ref) {
    const int nsb = n_in / 256;
    const int nr = (int) sample_rows.size();
    if (t == 1004) {
        out_ref.assign((size_t) ns_cols * nr, 0.0);
        const uint32_t* wdw = reinterpret_cast<const uint32_t*>(w_all.data());
        for (int ri = 0; ri < nr; ++ri) {
            const int o = sample_rows[ri], tile = o / 16, row_i = o % 16;
            for (int c = 0; c < ns_cols; ++c) {
                double acc = 0.0;
                const double* xc = &x_samp[(size_t) c * n_in];
                for (int kb = 0; kb < nsb; ++kb) {
                    const uint32_t* rec = wdw + ((size_t) tile * nsb + kb) * 544;
                    for (int sb = 0; sb < 8; ++sb) {
                        const int pr = sb >> 1, g = sb >> 2;
                        const uint32_t meta = rec[512 + 16 * g + row_i];
                        const double sc = (double) (float) sycl::bit_cast<sycl::half>((uint16_t) (meta & 0xffff));
                        const int zero = (int) (meta >> 16);
                        for (int l = 0; l < 32; ++l) {
                            const uint32_t by = (rec[128 * pr + 16 * (l >> 2) + row_i] >> (8 * (l & 3))) & 0xff;
                            const int qv = (sb & 1) ? (int) (by >> 4) : (int) (by & 15);
                            acc += sc * (double) (qv - zero) * xc[(size_t) kb * 256 + sb * 32 + l];
                        }
                    }
                }
                out_ref[(size_t) c * nr + ri] = acc;
            }
        }
        return;
    }
    const PpFormat* pf = pp_format_of(t);
    const size_t rowb = (size_t) nsb * pf->blk_bytes_256;
    std::vector<uint8_t> w_sub((size_t) nr * rowb);
    for (int i = 0; i < nr; ++i) {
        std::memcpy(&w_sub[(size_t) i * rowb], &w_all[(size_t) sample_rows[i] * rowb], rowb);
    }
    if (t != 8) {
        std::vector<double> mag, mag2;
        reference(t, w_sub, n_in, nr, 0, nr, x_samp, ns_cols, out_ref, mag, mag2);
        return;
    }
    out_ref.assign((size_t) ns_cols * nr, 0.0);
    const int nb32 = n_in / 32;
    for (int r = 0; r < nr; ++r) {
        const uint8_t* row = &w_sub[(size_t) r * rowb];
        for (int c = 0; c < ns_cols; ++c) {
            double acc = 0.0;
            const double* xc = &x_samp[(size_t) c * n_in];
            for (int b = 0; b < nb32; ++b) {
                const uint8_t* blk = row + (size_t) b * 34;
                uint16_t dh;
                std::memcpy(&dh, blk, 2);
                const double d = (double) (float) sycl::bit_cast<sycl::half>(dh);
                for (int l = 0; l < 32; ++l) {
                    acc += d * (double) (int8_t) blk[2 + l] * xc[(size_t) b * 32 + l];
                }
            }
            out_ref[(size_t) c * nr + r] = acc;
        }
    }
}

int pp_bench(const char* types_s, const char* ncols_s, const std::string& ssel, bool full_bench) {
    const std::vector<int> types = parse_list(types_s);
    const std::vector<int> ncl   = parse_list(ncols_s);
    sycl::queue q(sycl::gpu_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    dpct::blas::descriptor_ptr blas = new dpct::blas::descriptor();
    blas->set_queue(&q);

    struct PpShape { int n_in, n_out; const char* name; };
    const PpShape shapes[] = {
        { 5120,  17408, "27gu"  },
        {17408,   5120, "27down"},
        { 5120,  10240, "27qkv" },
        { 5120,  12288, "27q"   },
        { 6144,   5120, "27out" },
        { 5120, 248320, "27head"},
        { 5120,   6144, "27gate"},
        { 5120,   1024, "27kv"  },
    };
    auto shape_in_27 = [](const std::string& n) {
        return n == "27gu" || n == "27down" || n == "27qkv" || n == "27q" || n == "27out" || n == "27head";
    };
    int max_in = 0, max_out = 0, max_T = 0;
    size_t max_wb = 0;
    for (const PpShape& sh : shapes) {
        const bool match = (ssel == "27" && shape_in_27(sh.name)) || ssel == "all" || ssel == sh.name;
        if (!match) continue;
        max_in  = std::max(max_in,  sh.n_in);
        max_out = std::max(max_out, sh.n_out);
        for (int t : types) {
            if (const PpFormat* pf = pp_format_of(t)) {
                max_wb = std::max(max_wb, (size_t) sh.n_out * (sh.n_in / 256) * pf->blk_bytes_256);
            }
        }
    }
    for (int nc : ncl) max_T = std::max(max_T, nc);
    if (max_wb == 0 || max_T <= 0) {
        std::printf("pp_bench: nothing matched (types=%s ncols=%s shape=%s)\n", types_s, ncols_s, ssel.c_str());
        return 2;
    }

    const size_t pool_bytes = std::max(max_wb, (size_t) 512u << 20);
    const size_t max_slice_rows = std::min<size_t>(max_out, 16384);
    const size_t dw16_elems = std::max<size_t>(max_slice_rows * (size_t) max_in, (size_t) 17408 * 5120);
    std::vector<uint8_t> hpool(pool_bytes);
    uint8_t*   dpool = sycl::malloc_device<uint8_t>(pool_bytes, q);
    uint16_t*  dw16  = sycl::malloc_device<uint16_t>(dw16_elems, q);
    float*     y     = sycl::malloc_device<float>((size_t) max_T * max_out, q);
    float*     dx    = sycl::malloc_device<float>((size_t) max_T * max_in, q);
    uint16_t*  dx16  = sycl::malloc_device<uint16_t>((size_t) max_T * max_in, q);
    void*      xq    = sycl::malloc_device(k::native_q8_1_bytes((size_t) max_T * max_in, 1) + 256, q);
    if (!dpool || !dw16 || !y || !dx || !dx16 || !xq) {
        std::printf("pp: device allocation failed\n");
        return 1;
    }

    std::vector<float> hx((size_t) max_T * max_in);
    std::mt19937 rng(202611);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : hx) v = nd(rng);
    q.memcpy(dx, hx.data(), hx.size() * 4).wait();

    std::printf("q36_mmvq_dense_parity --pp%s on %s\n", full_bench ? "-bench" : "",
                q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("%-6s %-6s %5s x %6s %4s | %9s %8s %7s | %9s %8s %7s %5s | %9s %10s %10s\n",
                "type", "shape", "K", "N", "T",
                "f16_us", "f16_GB/s", "f16_TF",
                "i8_us", "i8_GB/s", "i8_TF", "gain",
                "f64_rms", "f16_rms", "f16_max");
    std::fflush(stdout);

    int bad = 0;
    for (int t : types) {
        const PpFormat* pf = pp_format_of(t);
        if (!pf) { std::printf("type %d unsupported by --pp\n", t); continue; }
        fill_pp_pool(t, hpool);
        q.memcpy(dpool, hpool.data(), pool_bytes).wait();

        for (const PpShape& sh : shapes) {
            const bool match = (ssel == "27" && shape_in_27(sh.name)) || ssel == "all" || ssel == sh.name;
            if (!match) continue;
            const size_t wb = (size_t) sh.n_out * (sh.n_in / 256) * pf->blk_bytes_256;
            const bool has_f16 = (t != 1004) || ((size_t) sh.n_out * sh.n_in <= dw16_elems);
            const int ndist = (int) std::max<size_t>(1, std::min<size_t>(32, pool_bytes / wb));
            const int nmat  = full_bench ? std::max(ndist, (wb > (200u << 20) ? 3 : 7)) : 1;
            const int reps  = full_bench ? (wb > (200u << 20) ? 3 : 7) : 1;

            std::vector<int> srows;
            const int n_edge = std::min(16, sh.n_out);
            for (int i = 0; i < n_edge; ++i) srows.push_back(i);
            if (sh.n_out > 16) {
                const int b0 = std::max(n_edge, sh.n_out - 16);
                for (int i = b0; i < sh.n_out; ++i) srows.push_back(i);
            }

            for (const int T : ncl) {
                if (T <= 0 || T > max_T) continue;
                const size_t n_act = (size_t) T * sh.n_in;
                q.parallel_for(sycl::range<1>(n_act), [=](sycl::id<1> i) { dx16[i] = (uint16_t) sycl::bit_cast<uint16_t>(sycl::half(dx[i])); });
                k::native_quantize_q8_1(dx, xq, (int) n_act, 1, &q);
                q.wait();

                std::vector<int> scols;
                for (int c : {0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, T / 2, T - 4, T - 2, T - 1}) {
                    if (c >= 0 && c < T && std::find(scols.begin(), scols.end(), c) == scols.end()) scols.push_back(c);
                }

                auto run_f16 = [&](const uint8_t* wptr) {
                    const float alpha = 1.0f, beta = 0.0f;
                    if (t == 1004) {
                        strata::q36xr::w4_dequant_f16(wptr, dw16, sh.n_in, sh.n_out, &q);
                        dpct::blas::gemm(blas, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
                                         sh.n_out, T, sh.n_in, &alpha, dw16, dpct::library_data_t::real_half, sh.n_in,
                                         dx16, dpct::library_data_t::real_half, sh.n_in, &beta,
                                         y, dpct::library_data_t::real_float, sh.n_out, dpct::compute_type::f32);
                        return;
                    }
                    const int slice = (int) std::min<size_t>(sh.n_out, dw16_elems / sh.n_in);
                    for (int r0 = 0; r0 < sh.n_out; r0 += slice) {
                        const int rn = std::min(slice, sh.n_out - r0);
                        k::dequant_f16(t, wptr, r0, rn, sh.n_in, dw16, &q);
                        dpct::blas::gemm(blas, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
                                         rn, T, sh.n_in, &alpha, dw16, dpct::library_data_t::real_half, sh.n_in,
                                         dx16, dpct::library_data_t::real_half, sh.n_in, &beta,
                                         y + r0, dpct::library_data_t::real_float, sh.n_out, dpct::compute_type::f32);
                    }
                };
                auto run_i8 = [&](const uint8_t* wptr) {
                    if (!strata::q36pp::gemm(t, wptr, xq, y, sh.n_in, sh.n_out, T, &q)) {
                        throw std::runtime_error("q36pp::gemm declined a --pp case");
                    }
                };

                const size_t check_slot = (size_t) ((nmat - 1) % ndist) * wb;
                std::vector<float> hy_f16, hy_i8((size_t) T * sh.n_out);
                if (has_f16) {
                    hy_f16.resize((size_t) T * sh.n_out);
                    run_f16(dpool + check_slot);
                    q.wait();
                    q.memcpy(hy_f16.data(), y, hy_f16.size() * 4).wait();
                }

                run_i8(dpool + check_slot);
                q.wait();
                q.memcpy(hy_i8.data(), y, hy_i8.size() * 4).wait();

                const size_t row_qb = (size_t) (sh.n_in / 32) * 36;
                std::vector<uint8_t> xq_samp(scols.size() * row_qb);
                for (size_t ci = 0; ci < scols.size(); ++ci) {
                    q.memcpy(&xq_samp[ci * row_qb], (const uint8_t*) xq + (size_t) scols[ci] * row_qb, row_qb);
                }
                q.wait();
                std::vector<double> x_samp, ref_samp;
                dequant_q8_1(xq_samp.data(), sh.n_in, (int) scols.size(), x_samp);
                std::vector<uint8_t> w_check(wb);
                std::memcpy(w_check.data(), &hpool[check_slot], wb);
                pp_ref_rows(t, w_check, sh.n_in, sh.n_out, srows, x_samp, (int) scols.size(), ref_samp);

                double f64_ss = 0.0, f64_sd = 0.0;
                long long nonfinite = 0;
                const int nr_samp = (int) srows.size();
                for (size_t ci = 0; ci < scols.size(); ++ci) {
                    for (int ri = 0; ri < nr_samp; ++ri) {
                        const float got = hy_i8[(size_t) scols[ci] * sh.n_out + srows[ri]];
                        const double rf = ref_samp[ci * nr_samp + ri];
                        if (!std::isfinite(got)) { ++nonfinite; continue; }
                        f64_ss += rf * rf;
                        f64_sd += ((double) got - rf) * ((double) got - rf);
                    }
                }
                const double f64_rel = std::sqrt(f64_sd / std::max(f64_ss, 1e-30));

                double f16_ss = 0.0, f16_sd = 0.0, f16_max = 0.0;
                if (has_f16) {
                    for (size_t i = 0; i < hy_i8.size(); ++i) {
                        if (!std::isfinite(hy_i8[i]) || !std::isfinite(hy_f16[i])) { ++nonfinite; continue; }
                        const double dlt = (double) hy_i8[i] - hy_f16[i];
                        f16_ss += (double) hy_f16[i] * hy_f16[i];
                        f16_sd += dlt * dlt;
                        f16_max = std::max(f16_max, std::fabs(dlt));
                    }
                }
                const double f16_rel = has_f16 ? std::sqrt(f16_sd / std::max(f16_ss, 1e-30)) : 0.0;

                const double u_f16 = has_f16 ? time_graph(q, nmat, [&](int m) { run_f16(dpool + (size_t) (m % ndist) * wb); }, reps) : 0.0;
                const double u_i8  = time_graph(q, nmat, [&](int m) { run_i8(dpool + (size_t) (m % ndist) * wb); }, reps);

                const double xb    = (double) T * (sh.n_in / 32) * 36;
                const double bytes = (double) wb + xb;
                const double flops = 2.0 * (double) T * sh.n_out * sh.n_in;
                const double f16_gbs = has_f16 ? bytes / u_f16 / 1e3 : 0.0, i8_gbs = bytes / u_i8 / 1e3;
                const double f16_tf  = has_f16 ? flops / u_f16 / 1e6 : 0.0, i8_tf  = flops / u_i8 / 1e6;
                const bool ok = (nonfinite == 0 && f64_rel < 5e-4);
                if (!ok) ++bad;
                std::printf("%-6s %-6s %5d x %6d %4d | %9.1f %8.1f %7.2f | %9.1f %8.1f %7.2f %4.2fx | %9.2e %10.2e %10.3g %s\n",
                            pf->name, sh.name, sh.n_in, sh.n_out, T,
                            u_f16, f16_gbs, f16_tf,
                            u_i8,  i8_gbs,  i8_tf, has_f16 ? (u_f16 / u_i8) : 0.0,
                            f64_rel, f16_rel, f16_max, ok ? "ok" : "FAIL");

                if (t == 1004) {
                    auto run_w4f16 = [&](const uint8_t* wptr) {
                        if (!strata::q36pp::gemm_w4_f16(wptr, dx16, y, sh.n_in, sh.n_out, T, &q)) {
                            throw std::runtime_error("q36pp::gemm_w4_f16 declined a --pp case");
                        }
                    };
                    std::vector<float> hy_w4((size_t) T * sh.n_out);
                    run_w4f16(dpool + check_slot);
                    q.wait();
                    q.memcpy(hy_w4.data(), y, hy_w4.size() * 4).wait();

                    std::vector<uint16_t> x16_h(scols.size() * (size_t) sh.n_in);
                    for (size_t ci = 0; ci < scols.size(); ++ci) {
                        q.memcpy(&x16_h[ci * (size_t) sh.n_in], dx16 + (size_t) scols[ci] * sh.n_in, (size_t) sh.n_in * 2);
                    }
                    q.wait();
                    std::vector<double> x16_samp(x16_h.size()), ref16_samp;
                    for (size_t i = 0; i < x16_h.size(); ++i) {
                        x16_samp[i] = (double) (float) sycl::bit_cast<sycl::half>(x16_h[i]);
                    }
                    pp_ref_rows(1004, w_check, sh.n_in, sh.n_out, srows, x16_samp, (int) scols.size(), ref16_samp);

                    double w4_ss = 0.0, w4_sd = 0.0;
                    long long w4_nf = 0;
                    for (size_t ci = 0; ci < scols.size(); ++ci) {
                        for (int ri = 0; ri < nr_samp; ++ri) {
                            const float got = hy_w4[(size_t) scols[ci] * sh.n_out + srows[ri]];
                            const double rf = ref16_samp[ci * nr_samp + ri];
                            if (!std::isfinite(got)) { ++w4_nf; continue; }
                            w4_ss += rf * rf;
                            w4_sd += ((double) got - rf) * ((double) got - rf);
                        }
                    }
                    const double w4_f64_rel = std::sqrt(w4_sd / std::max(w4_ss, 1e-30));

                    double w4_f16_ss = 0.0, w4_f16_sd = 0.0, w4_f16_max = 0.0;
                    if (has_f16) {
                        for (size_t i = 0; i < hy_w4.size(); ++i) {
                            if (!std::isfinite(hy_w4[i]) || !std::isfinite(hy_f16[i])) { ++w4_nf; continue; }
                            const double dlt = (double) hy_w4[i] - hy_f16[i];
                            w4_f16_ss += (double) hy_f16[i] * hy_f16[i];
                            w4_f16_sd += dlt * dlt;
                            w4_f16_max = std::max(w4_f16_max, std::fabs(dlt));
                        }
                    }
                    const double w4_f16_rel = has_f16 ? std::sqrt(w4_f16_sd / std::max(w4_f16_ss, 1e-30)) : 0.0;
                    const double u_w4 = time_graph(q, nmat, [&](int m) { run_w4f16(dpool + (size_t) (m % ndist) * wb); }, reps);
                    const double w4_bytes = (double) wb + (double) T * sh.n_in * 2.0;
                    const double w4_gbs = w4_bytes / u_w4 / 1e3;
                    const double w4_tf  = flops / u_w4 / 1e6;
                    const bool ok_w4 = (w4_nf == 0 && w4_f64_rel < 5e-4);
                    if (!ok_w4) ++bad;
                    std::printf("%-6s %-6s %5d x %6d %4d | %9.1f %8.1f %7.2f | %9.1f %8.1f %7.2f %4.2fx | %9.2e %10.2e %10.3g %s\n",
                                "W4F16", sh.name, sh.n_in, sh.n_out, T,
                                u_f16, f16_gbs, f16_tf,
                                u_w4,  w4_gbs,  w4_tf, has_f16 ? (u_f16 / u_w4) : 0.0,
                                w4_f64_rel, w4_f16_rel, w4_f16_max, ok_w4 ? "ok" : "FAIL");
                }
                std::fflush(stdout);
            }
        }
    }
    delete blas;
    sycl::free(dpool, q); sycl::free(dw16, q); sycl::free(y, q); sycl::free(dx, q); sycl::free(dx16, q); sycl::free(xq, q);
    return bad ? 1 : 0;
}

}  // namespace

// --w4 [ncols] [shape]: the B70W4 form (q36_xr_dense.dp.cpp, tools/qwen35/b70w4_pack.py's records) on random packs: us per
// call and GB/s of pack bytes as --bench times them, and the output of the last launch against a float64 reference on the
// host for the first 32 and the last 16 rows (the pack's nibbles, scales and zeros x the q8_1 image's values).
int w4_bench(const char* ncols_s, const std::string& ssel) {
    const std::vector<int> ncl = parse_list(ncols_s);
    sycl::queue q(sycl::gpu_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    struct BShape { int n_in, n_out; const char* name; };
    const BShape shapes[] = {{5120, 10240, "27qkv"}, {5120, 6144, "27gate"}, {6144, 5120, "27out"}, {5120, 12288, "27q"},
                             {5120, 17408, "27gu"}, {17408, 5120, "27down"}, {5120, 1024, "27kv"}};
    const size_t REC = 544, POOL_DW = ((size_t) 640 << 20) / 4 / REC * REC;
    std::vector<uint32_t> hpool(POOL_DW);
    {
        uint64_t st = 0x9E3779B97F4A7C15ull;
        auto xr = [&] { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; };
        for (size_t i = 0; i < POOL_DW; i += 2) { const uint64_t v = xr(); hpool[i] = (uint32_t) v; hpool[i + 1] = (uint32_t) (v >> 32); }
        for (size_t r = 0; r < POOL_DW / REC; ++r)
            for (int k = 0; k < 32; ++k) {
                const uint64_t v = xr();
                hpool[r * REC + 512 + k] = (0x1400u | (uint32_t) (v & 0x3ff)) | ((1u + (uint32_t) ((v >> 20) & 15)) << 16);
            }
    }
    uint32_t* dpool = (uint32_t*) sycl::aligned_alloc_device(4096, POOL_DW * 4, q);
    const int maxc = 32, maxout = 17408, maxin = 17408;
    float* y = sycl::malloc_device<float>((size_t) maxc * maxout, q);
    std::vector<float> hx((size_t) maxc * maxin);
    float* dx = sycl::malloc_device<float>(hx.size(), q);
    const size_t xq_bytes = k::native_q8_1_bytes(maxin * maxc, 1) + 256;   // (the columns as one vector, as the engine quantizes them)
    void* xq = sycl::malloc_device(xq_bytes, q);
    uint16_t* df = sycl::malloc_device<uint16_t>((size_t) 17408 * 5120, q);   // the FP16 image of the largest matrix
    namespace FB16 = strata::kernels;
    uint16_t* dx16 = (uint16_t*) sycl::aligned_alloc_device(4096, strata::q36xr::w4h_act_halfs(maxin, maxc) * 2, q);
    if (!dpool || !y || !dx || !xq || !df) { std::printf("w4: could not allocate the buffers\n"); return 1; }
    std::mt19937 rng(4242);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : hx) v = nd(rng);
    q.memcpy(dx, hx.data(), hx.size() * 4).wait();
    q.memcpy(dpool, hpool.data(), POOL_DW * 4).wait();
    std::printf("q36_mmvq_dense_parity --w4 on %s: us per call (median), GB/s of pack bytes, the distance to a float64 reference\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());
    int bad = 0;
    for (const BShape& sh : shapes) {
        if (!(ssel == "27" || ssel == "all" || ssel == sh.name)) continue;
        const size_t wb = strata::q36xr::w4_bytes(sh.n_in, sh.n_out), wdw = wb / 4;
        const int ndist = (int) std::max<size_t>(1, std::min<size_t>(128, POOL_DW / wdw)), nmat = std::max(ndist, 16);
        const int bpr = sh.n_in / 256, nb = sh.n_in / 32;
        {   // the prompt path's dequantize: us per matrix, GB/s of FP16 written, the last matrix's first and last 16 rows bit for bit
            const double ud = time_graph(q, nmat, [&](int m) {
                strata::q36xr::w4_dequant_f16(dpool + (size_t) (m % ndist) * wdw, df, sh.n_in, sh.n_out, &q);
            }, 7);
            q.wait();
            const uint32_t* base = hpool.data() + (size_t) ((nmat - 1) % ndist) * wdw;
            std::vector<uint16_t> hf((size_t) sh.n_in * sh.n_out);
            q.memcpy(hf.data(), df, hf.size() * 2).wait();
            long long wrong = 0, seen = 0;
            for (int o = 0; o < sh.n_out; ++o) {
                if (!(o < 16 || o >= sh.n_out - 16)) continue;
                const int t = o / 16, i = o % 16;
                for (int c = 0; c < bpr; ++c) {
                    const uint32_t* rec = base + ((size_t) t * bpr + c) * REC;
                    for (int sb = 0; sb < 8; ++sb) {
                        const uint32_t meta = rec[512 + 16 * (sb >> 2) + i];
                        const float sc = (float) sycl::bit_cast<sycl::half>((uint16_t) (meta & 0xffff));
                        const int zero = (int) (meta >> 16);
                        for (int l = 0; l < 32; ++l) {
                            const uint32_t by = (rec[128 * (sb >> 1) + 16 * (l >> 2) + i] >> (8 * (l & 3))) & 0xff;
                            const int qv = (sb & 1) ? (int) (by >> 4) : (int) (by & 15);
                            const sycl::half want = (sycl::half) (sc * (float) (qv - zero));
                            ++seen;
                            if (sycl::bit_cast<uint16_t>(want) != hf[(size_t) o * sh.n_in + c * 256 + sb * 32 + l]) ++wrong;
                        }
                    }
                }
            }
            if (wrong) ++bad;
            std::printf("W4     %-6s %5d x %6d dequantize to FP16  %8.2f us %5.1f GB/s written  %lld of %lld values differ%s\n", sh.name,
                        sh.n_in, sh.n_out, ud, (double) sh.n_in * sh.n_out * 2 / ud / 1e3, wrong, seen, wrong ? "  FAIL" : "");
            std::fflush(stdout);
        }
        for (const int nc : ncl) {
            if (nc < 1 || nc > maxc) continue;
            k::native_quantize_q8_1(dx, xq, sh.n_in * nc, 1, &q);
            q.wait();
            const double us = time_graph(q, nmat, [&](int m) {
                if (!strata::q36xr::mmvq_w4(dpool + (size_t) (m % ndist) * wdw, xq, y, sh.n_in, sh.n_out, nc, &q))
                    throw std::runtime_error("the W4 form declined a bench call");
            }, wb > (100u << 20) ? 7 : 11);
            q.wait();
            std::vector<float> hy((size_t) nc * sh.n_out);
            std::vector<uint8_t> hq(k::native_q8_1_bytes(sh.n_in * nc, 1));
            q.memcpy(hy.data(), y, hy.size() * 4).wait();
            q.memcpy(hq.data(), xq, hq.size()).wait();
            const uint32_t* base = hpool.data() + (size_t) ((nmat - 1) % ndist) * wdw;
            std::vector<double> xd((size_t) nc * sh.n_in);
            for (int j = 0; j < nc; ++j)
                for (int b = 0; b < nb; ++b) {
                    const uint8_t* blk = hq.data() + ((size_t) j * nb + b) * 36;
                    uint16_t dh;
                    std::memcpy(&dh, blk, 2);
                    const double d = (double) (float) sycl::bit_cast<sycl::half>(dh);
                    for (int l = 0; l < 32; ++l) xd[(size_t) j * sh.n_in + b * 32 + l] = d * (double) (int8_t) blk[4 + l];
                }
            double ss = 0, sd = 0;
            for (int o = 0; o < sh.n_out; ++o) {
                if (!(o < 32 || o >= sh.n_out - 16)) continue;
                const int t = o / 16, i = o % 16;
                for (int j = 0; j < nc; ++j) {
                    double acc = 0;
                    for (int c = 0; c < bpr; ++c) {
                        const uint32_t* rec = base + ((size_t) t * bpr + c) * REC;
                        for (int sb = 0; sb < 8; ++sb) {
                            const int pr = sb >> 1, g = sb >> 2;
                            const uint32_t meta = rec[512 + 16 * g + i];
                            const double sc = (double) (float) sycl::bit_cast<sycl::half>((uint16_t) (meta & 0xffff));
                            const int zero = (int) (meta >> 16);
                            for (int l = 0; l < 32; ++l) {
                                const uint32_t by = (rec[128 * pr + 16 * (l >> 2) + i] >> (8 * (l & 3))) & 0xff;
                                const int qv = (sb & 1) ? (int) (by >> 4) : (int) (by & 15);
                                acc += sc * (double) (qv - zero) * xd[(size_t) j * sh.n_in + c * 256 + sb * 32 + l];
                            }
                        }
                    }
                    const double got = hy[(size_t) j * sh.n_out + o];
                    ss += acc * acc;
                    sd += (got - acc) * (got - acc);
                }
            }
            const double rel = std::sqrt(sd / std::max(ss, 1e-30));
            if (!(rel < 1e-4)) ++bad;
            std::printf("W4     %-6s %5d x %6d nc %d  %8.2f us %5.1f GB/s  rms diff / rms %.2e%s\n", sh.name, sh.n_in, sh.n_out, nc, us,
                        wb / us / 1e3, rel, rel < 1e-4 ? "" : "  FAIL");
            std::fflush(stdout);
        }
        // W4A16: the same matrices against FP16 activations (mmvq_w4h)
        for (const int nc : ncl) {
            if (nc < 1 || nc > maxc) continue;
            const int ncp = (nc + 7) / 8 * 8;
            std::vector<uint16_t> hx16(strata::q36xr::w4h_act_halfs(sh.n_in, nc), 0);
            std::vector<double> xd((size_t) nc * sh.n_in);
            for (int j = 0; j < nc; ++j)
                for (int kx = 0; kx < sh.n_in; ++kx) {
                    const uint16_t hb = FB16::f16_from_f32(hx[(size_t) j * sh.n_in + kx]);
                    hx16[((size_t) (kx / 16) * ncp + j) * 16 + kx % 16] = hb;
                    xd[(size_t) j * sh.n_in + kx] = (double) FB16::f32_from_f16(hb);
                }
            q.memcpy(dx16, hx16.data(), hx16.size() * 2).wait();
            const double us = time_graph(q, nmat, [&](int m) {
                if (!strata::q36xr::mmvq_w4h(dpool + (size_t) (m % ndist) * wdw, dx16, y, sh.n_in, sh.n_out, nc, &q))
                    throw std::runtime_error("the W4A16 form declined a bench call");
            }, wb > (100u << 20) ? 7 : 11);
            q.wait();
            std::vector<float> hy((size_t) nc * sh.n_out);
            q.memcpy(hy.data(), y, hy.size() * 4).wait();
            const uint32_t* base = hpool.data() + (size_t) ((nmat - 1) % ndist) * wdw;
            double ss = 0, sd = 0;
            for (int o = 0; o < sh.n_out; ++o) {
                if (!(o < 32 || o >= sh.n_out - 16)) continue;
                const int t = o / 16, i = o % 16;
                for (int j = 0; j < nc; ++j) {
                    double acc = 0;
                    for (int c = 0; c < bpr; ++c) {
                        const uint32_t* rec = base + ((size_t) t * bpr + c) * REC;
                        for (int sb = 0; sb < 8; ++sb) {
                            const int pr = sb >> 1, g = sb >> 2;
                            const uint32_t meta = rec[512 + 16 * g + i];
                            const double sc = (double) (float) sycl::bit_cast<sycl::half>((uint16_t) (meta & 0xffff));
                            const int zero = (int) (meta >> 16);
                            for (int l = 0; l < 32; ++l) {
                                const uint32_t by = (rec[128 * pr + 16 * (l >> 2) + i] >> (8 * (l & 3))) & 0xff;
                                const int qv = (sb & 1) ? (int) (by >> 4) : (int) (by & 15);
                                acc += sc * (double) (qv - zero) * xd[(size_t) j * sh.n_in + c * 256 + sb * 32 + l];
                            }
                        }
                    }
                    const double got = hy[(size_t) j * sh.n_out + o];
                    ss += acc * acc;
                    sd += (got - acc) * (got - acc);
                }
            }
            const double rel = std::sqrt(sd / std::max(ss, 1e-30));
            if (!(rel < 1e-4)) ++bad;
            std::printf("W4A16  %-6s %5d x %6d nc %d  %8.2f us %5.1f GB/s  rms diff / rms %.2e%s\n", sh.name, sh.n_in, sh.n_out, nc, us,
                        wb / us / 1e3, rel, rel < 1e-4 ? "" : "  FAIL");
            std::fflush(stdout);
        }
    }
    sycl::free(dpool, q); sycl::free(y, q); sycl::free(dx, q); sycl::free(xq, q); sycl::free(df, q);
    return bad ? 1 : 0;
}

double time_graph_warm(sycl::queue& q, int nmat, const std::function<void(int)>& launch, int reps, double warm_sec = 0.4) {
    sx::command_graph g(q.get_context(), q.get_device());
    g.begin_recording(q);
    for (int m = 0; m < nmat; ++m) launch(m);
    g.end_recording(q);
    auto ge = g.finalize();
    q.ext_oneapi_graph(ge);
    q.wait();
    const auto t0 = std::chrono::steady_clock::now();
    do {
        q.ext_oneapi_graph(ge);
        q.wait();
    } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < warm_sec);
    std::vector<double> us;
    us.reserve(reps);
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

std::vector<double> decode_w4x_act(const std::vector<uint8_t>& hact, int n_in, int nc, int gs) {
    const int n_g128 = n_in / 128;
    const int ng = (nc + 7) / 8;
    const size_t stride = (gs == 32) ? 1216u : 1088u;
    std::vector<double> xd((size_t) nc * n_in, 0.0);
    for (int g = 0; g < n_g128; ++g) {
        for (int col = 0; col < nc; ++col) {
            const int gi = col >> 3, r = col & 7;
            const uint8_t* gp = hact.data() + ((size_t) g * ng + gi) * stride;
            if (gs == 128) {
                float sa = 0.0f;
                int32_t sxi_stored = 0;
                std::memcpy(&sa, gp + 1024u + r * 4u, 4);
                std::memcpy(&sxi_stored, gp + 1056u + r * 4u, 4);
                int64_t sum_check = 0;
                for (int p = 0; p < 2; ++p) {
                    for (int hh = 0; hh < 2; ++hh) {
                        for (int da = 0; da < 8; ++da) {
                            uint32_t pk = 0;
                            std::memcpy(&pk, gp + ((p * 2 + hh) * 64 + r * 8 + da) * 4u, 4);
                            const int8_t qe0 = (int8_t) (uint8_t) (pk & 0xffu);
                            const int8_t qo0 = (int8_t) (uint8_t) ((pk >> 8) & 0xffu);
                            const int8_t qe1 = (int8_t) (uint8_t) ((pk >> 16) & 0xffu);
                            const int8_t qo1 = (int8_t) (uint8_t) ((pk >> 24) & 0xffu);
                            const int k_e = g * 128 + p * 64 + hh * 16 + da * 2;
                            const int k_o = k_e + 32;
                            xd[(size_t) col * n_in + k_e + 0] = (double) sa * (double) qe0;
                            xd[(size_t) col * n_in + k_e + 1] = (double) sa * (double) qe1;
                            xd[(size_t) col * n_in + k_o + 0] = (double) sa * (double) qo0;
                            xd[(size_t) col * n_in + k_o + 1] = (double) sa * (double) qo1;
                            sum_check += qe0 + qe1 + qo0 + qo1;
                        }
                    }
                }
                if (sum_check != sxi_stored) return {};
            } else {
                float sx_stored = 0.0f;
                std::memcpy(&sx_stored, gp + 1152u + r * 4u, 4);
                double sx_check = 0.0;
                for (int sb = 0; sb < 4; ++sb) {
                    float sa = 0.0f;
                    std::memcpy(&sa, gp + 1024u + (sb * 8 + r) * 4u, 4);
                    int64_t ssum = 0;
                    for (int da = 0; da < 8; ++da) {
                        uint32_t pk = 0;
                        std::memcpy(&pk, gp + (sb * 64 + r * 8 + da) * 4u, 4);
                        for (int b = 0; b < 4; ++b) {
                            const int8_t qv = (int8_t) (uint8_t) ((pk >> (8 * b)) & 0xffu);
                            xd[(size_t) col * n_in + g * 128 + sb * 32 + da * 4 + b] = (double) sa * (double) qv;
                            ssum += qv;
                        }
                    }
                    sx_check += (double) ((float) ssum * sa);
                }
                if (std::fabs(sx_check - (double) sx_stored) > 1e-3 * (std::fabs(sx_check) + 1.0)) return {};
            }
        }
    }
    return xd;
}

double w4x_f64_rel(const uint32_t* base, const std::vector<double>& xd, const std::vector<float>& hy,
                   int n_in, int n_out, int nc) {
    const int REC = 544, bpr = n_in / 256;
    double ss = 0.0, sd = 0.0;
    for (int o = 0; o < n_out; ++o) {
        if (!(o < 32 || o >= n_out - 16)) continue;
        const int t = o / 16, i = o % 16;
        for (int j = 0; j < nc; ++j) {
            double acc = 0.0;
            for (int c = 0; c < bpr; ++c) {
                const uint32_t* rec = base + ((size_t) t * bpr + c) * REC;
                for (int sb = 0; sb < 8; ++sb) {
                    const int pr = sb >> 1, g = sb >> 2;
                    const uint32_t meta = rec[512 + 16 * g + i];
                    const double sc = (double) (float) sycl::bit_cast<sycl::half>((uint16_t) (meta & 0xffff));
                    const int zero = (int) (meta >> 16);
                    for (int l = 0; l < 32; ++l) {
                        const uint32_t by = (rec[128 * pr + 16 * (l >> 2) + i] >> (8 * (l & 3))) & 0xff;
                        const int qv = (sb & 1) ? (int) (by >> 4) : (int) (by & 15);
                        acc += sc * (double) (qv - zero) * xd[(size_t) j * n_in + c * 256 + sb * 32 + l];
                    }
                }
            }
            const double got = hy[(size_t) j * n_out + o];
            ss += acc * acc;
            sd += (got - acc) * (got - acc);
        }
    }
    return std::sqrt(sd / std::max(ss, 1e-30));
}

// --w4x [128|32|all] [ncols] [shape] [ks]: native u4 x s8 parity vs float64 + B70 590 GB/s floor benchmark.
int w4x_bench(const std::string& mode, const char* ncols_s, const std::string& ssel, int ks_ovr) {
    const std::vector<int> ncl = parse_list(ncols_s);
    std::vector<int> gs_list;
    if (mode == "128") gs_list = {128};
    else if (mode == "32") gs_list = {32};
    else if (mode != "gu") gs_list = {128, 32};

    sycl::queue q(sycl::gpu_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    struct BShape { int n_in, n_out; const char* name; };
    const BShape shapes[] = {
        {5120, 17408, "27gu"}, {17408, 5120, "27down"}, {5120, 10240, "27qkv"},
        {5120, 6144, "27gate"}, {6144, 5120, "27out"}, {5120, 12288, "27q"},
        {5120, 1024, "27kv"}, {10240, 5120, "27eh"}, {5120, 248320, "27head"}};

    const size_t REC = 544, POOL_DW = ((size_t) 704 << 20) / 4 / REC * REC;   // ~704 MiB (>64 MiB and > 644 MiB head)
    std::vector<uint32_t> hpool(POOL_DW);
    {
        uint64_t st = 0x9E3779B97F4A7C15ull;
        auto xr = [&] { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; };
        for (size_t i = 0; i < POOL_DW; i += 2) { const uint64_t v = xr(); hpool[i] = (uint32_t) v; hpool[i + 1] = (uint32_t) (v >> 32); }
        for (size_t r = 0; r < POOL_DW / REC; ++r)
            for (int k = 0; k < 32; ++k) {
                const uint64_t v = xr();
                hpool[r * REC + 512 + k] = (0x1400u | (uint32_t) (v & 0x3ff)) | ((1u + (uint32_t) ((v >> 20) & 15)) << 16);
            }
    }
    uint32_t* dpool = (uint32_t*) sycl::aligned_alloc_device(4096, POOL_DW * 4, q);
    const int maxc = 32, maxout = 248320, maxin = 17408;
    float* y = sycl::malloc_device<float>((size_t) maxc * maxout, q);
    float* y_solo = sycl::malloc_device<float>((size_t) maxout, q);
    float* dh = sycl::malloc_device<float>((size_t) maxc * maxin, q);
    float* dn = sycl::malloc_device<float>((size_t) maxin, q);
    float* dir = sycl::malloc_device<float>((size_t) maxc, q);
    std::vector<float> hx((size_t) maxc * maxin), hu((size_t) maxc * maxin), hn(maxin), hir(maxc);
    float* dx = sycl::malloc_device<float>(hx.size(), q);
    float* du = sycl::malloc_device<float>(hu.size(), q);
    void* dact = sycl::aligned_alloc_device(4096, strata::q36xr::w4x_act_bytes(maxin, maxc, 32) + 4096, q);
    if (!dpool || !y || !y_solo || !dh || !dn || !dir || !dx || !du || !dact) {
        std::printf("w4x: could not allocate buffers\n");
        return 1;
    }
    std::mt19937 rng(4242);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : hx) v = nd(rng);
    for (auto& v : hu) v = nd(rng);
    for (auto& v : hn) v = 0.5f + 0.5f * std::fabs(nd(rng));
    for (auto& v : hir) v = 0.5f + 0.25f * std::fabs(nd(rng));
    q.memcpy(dx, hx.data(), hx.size() * 4).wait();
    q.memcpy(du, hu.data(), hu.size() * 4).wait();
    q.memcpy(dn, hn.data(), hn.size() * 4).wait();
    q.memcpy(dir, hir.data(), hir.size() * 4).wait();
    q.memcpy(dpool, hpool.data(), POOL_DW * 4).wait();

    std::printf("q36_mmvq_dense_parity --w4x on %s (floor 590 GB/s, >=0.4s warmup, ~704 MiB pool)\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());
    int bad = 0;

    for (const int gs : gs_list) {
        const double bound = (gs == 128) ? 5e-5 : 5e-4;
        // Contract check 1: RMSnorm-apply + quantize (with and without h_out) vs CPU float64
        {
            const int nin = 5120, nout = 5120, nc = 8;
            strata::q36xr::w4x_rms_apply_quantize(dx, nin, dn, dir, dact, dh, nin, nin, nc, gs, &q);
            strata::q36xr::mmvq_w4x(dpool, dact, y, nin, nout, nc, gs, &q, ks_ovr);
            q.wait();
            std::vector<float> hh((size_t) nc * nin), hy((size_t) nc * nout);
            std::vector<uint8_t> hact(strata::q36xr::w4x_act_bytes(nin, nc, gs));
            q.memcpy(hh.data(), dh, hh.size() * 4).wait();
            q.memcpy(hy.data(), y, hy.size() * 4).wait();
            q.memcpy(hact.data(), dact, hact.size()).wait();
            double h_ss = 0, h_sd = 0;
            for (int c = 0; c < nc; ++c)
                for (int k = 0; k < nin; ++k) {
                    const double want = (double) hx[(size_t) c * nin + k] * (double) hir[c] * (double) hn[k];
                    const double got = hh[(size_t) c * nin + k];
                    h_ss += want * want;
                    h_sd += (got - want) * (got - want);
                }
            const double h_rel = std::sqrt(h_sd / std::max(h_ss, 1e-30));
            const std::vector<double> xd = decode_w4x_act(hact, nin, nc, gs);
            const double rel = xd.empty() ? 1.0 : w4x_f64_rel(hpool.data(), xd, hy, nin, nout, nc);
            // Also verify h_out == nullptr variant writes identical quantized act bytes
            std::vector<uint8_t> hact_noh(hact.size());
            strata::q36xr::w4x_rms_apply_quantize(dx, nin, dn, dir, dact, nullptr, 0, nin, nc, gs, &q);
            q.memcpy(hact_noh.data(), dact, hact_noh.size()).wait();
            const bool noh_ok = (hact_noh == hact);
            if (!(h_rel < 1e-6 && rel < bound && noh_ok)) ++bad;
            std::printf("W4X-%-3d parity rms_apply_quant (h_out rel %.2e, mv rel %.2e, null_h %s)%s\n",
                        gs, h_rel, rel, noh_ok ? "OK" : "FAIL", (h_rel < 1e-6 && rel < bound && noh_ok) ? "" : "  FAIL");
        }
        // Contract check 2: SwiGLU + quantize vs CPU float64
        {
            const int nin = 17408, nout = 5120, nc = 8;
            strata::q36xr::w4x_swiglu_quantize(dx, du, dact, nin, nc, gs, &q);
            strata::q36xr::mmvq_w4x(dpool, dact, y, nin, nout, nc, gs, &q, ks_ovr);
            q.wait();
            std::vector<float> hy((size_t) nc * nout);
            std::vector<uint8_t> hact(strata::q36xr::w4x_act_bytes(nin, nc, gs));
            q.memcpy(hy.data(), y, hy.size() * 4).wait();
            q.memcpy(hact.data(), dact, hact.size()).wait();
            const std::vector<double> xd = decode_w4x_act(hact, nin, nc, gs);
            double s_ss = 0, s_sd = 0;
            if (!xd.empty()) {
                for (int c = 0; c < nc; ++c)
                    for (int k = 0; k < nin; ++k) {
                        const double g = hx[(size_t) c * nin + k], u = hu[(size_t) c * nin + k];
                        const double want = (g / (1.0 + std::exp(-g))) * u;
                        s_ss += want * want;
                        s_sd += (xd[(size_t) c * nin + k] - want) * (xd[(size_t) c * nin + k] - want);
                    }
            }
            const double s_rel = xd.empty() ? 1.0 : std::sqrt(s_sd / std::max(s_ss, 1e-30));
            const double rel = xd.empty() ? 1.0 : w4x_f64_rel(hpool.data(), xd, hy, nin, nout, nc);
            if (!(rel < bound && s_rel < 0.02)) ++bad;
            std::printf("W4X-%-3d parity swiglu_quant    (q_rms %.2e, mv rel %.2e)%s\n",
                        gs, s_rel, rel, (rel < bound && s_rel < 0.02) ? "" : "  FAIL");
        }
        // Contract check 3: Split-K sweep across all compiled KS in {1,2,3,4,5,10}
        for (const int ks : {1, 2, 3, 4, 5, 10}) {
            const int nin = 5120, nout = 1024, nc = 16;
            strata::q36xr::w4x_quantize(dx, nin, dact, nin, nc, gs, &q);
            strata::q36xr::mmvq_w4x(dpool, dact, y, nin, nout, nc, gs, &q, ks);
            q.wait();
            std::vector<float> hy((size_t) nc * nout);
            std::vector<uint8_t> hact(strata::q36xr::w4x_act_bytes(nin, nc, gs));
            q.memcpy(hy.data(), y, hy.size() * 4).wait();
            q.memcpy(hact.data(), dact, hact.size()).wait();
            const std::vector<double> xd = decode_w4x_act(hact, nin, nc, gs);
            const double rel = xd.empty() ? 1.0 : w4x_f64_rel(hpool.data(), xd, hy, nin, nout, nc);
            if (!(rel < bound)) ++bad;
            std::printf("W4X-%-3d parity ks=%-2d           (5120x1024 nc=16, rel %.2e)%s\n",
                        gs, ks, rel, rel < bound ? "" : "  FAIL");
        }

        // Main per-shape benchmark + parity + 32/8 ratio + multi-col vs 1-col check
        for (const BShape& sh : shapes) {
            if (!(ssel == "27" || ssel == "all" || ssel == sh.name)) continue;
            const size_t wb = strata::q36xr::w4_bytes(sh.n_in, sh.n_out), wdw = wb / 4;
            const size_t ndist_raw = std::max<size_t>(1, POOL_DW / wdw);
            const int ndist = (int) std::min<size_t>(128, ndist_raw);
            const int nmat = std::max(ndist, sh.n_out > 100000 ? 1 : 16);
            const double floor_us = (double) wb / 590e9 * 1e6;
            double us8 = 0.0;

            for (const int nc : ncl) {
                if (nc < 1 || nc > maxc) continue;
                strata::q36xr::w4x_quantize(dx, sh.n_in, dact, sh.n_in, nc, gs, &q);
                q.wait();
                const double us = time_graph_warm(q, nmat, [&](int m) {
                    if (!strata::q36xr::mmvq_w4x(dpool + (size_t) (m % ndist) * wdw, dact, y,
                                                 sh.n_in, sh.n_out, nc, gs, &q, ks_ovr))
                        throw std::runtime_error("mmvq_w4x declined bench call");
                }, wb > (100u << 20) ? 5 : 9, 0.4);
                q.wait();
                if (nc == 8) us8 = us;

                std::vector<float> hy((size_t) nc * sh.n_out);
                std::vector<uint8_t> hact(strata::q36xr::w4x_act_bytes(sh.n_in, nc, gs));
                q.memcpy(hy.data(), y, hy.size() * 4).wait();
                q.memcpy(hact.data(), dact, hact.size()).wait();
                const uint32_t* base = hpool.data() + (size_t) ((nmat - 1) % ndist) * wdw;
                const std::vector<double> xd = decode_w4x_act(hact, sh.n_in, nc, gs);
                const double rel = xd.empty() ? 1.0 : w4x_f64_rel(base, xd, hy, sh.n_in, sh.n_out, nc);
                if (!(rel < bound)) ++bad;

                // Contract check 4: batched column 0 matches solo nc=1 quant+gemv on the same weight copy
                double solo_rel = 0.0;
                if (nc > 1 && nc == ncl.back()) {
                    strata::q36xr::w4x_quantize(dx, sh.n_in, dact, sh.n_in, 1, gs, &q);
                    strata::q36xr::mmvq_w4x(dpool + (size_t) ((nmat - 1) % ndist) * wdw, dact, y_solo,
                                            sh.n_in, sh.n_out, 1, gs, &q, ks_ovr);
                    q.wait();
                    std::vector<float> hsolo(sh.n_out);
                    q.memcpy(hsolo.data(), y_solo, hsolo.size() * 4).wait();
                    double ss = 0.0, sd = 0.0;
                    for (int o = 0; o < sh.n_out; ++o) {
                        const double a = hsolo[o], b = hy[o];
                        ss += a * a;
                        sd += (a - b) * (a - b);
                    }
                    solo_rel = std::sqrt(sd / std::max(ss, 1e-30));
                    if (!(solo_rel < 1e-6)) ++bad;
                }

                const double gbps = (double) wb / us / 1e3;
                const double r_floor = us / floor_us;
                if (nc == 32 && us8 > 0.0) {
                    std::printf("W4X-%-3d %-6s %5d x %6d nc %2d  %8.2f us %5.1f GB/s (%4.2fx fl %7.2f)  32/8 %.2fx  rel %.2e (solo %.1e)%s\n",
                                gs, sh.name, sh.n_in, sh.n_out, nc, us, gbps, r_floor, floor_us, us / us8, rel, solo_rel,
                                (rel < bound && solo_rel < 1e-6) ? "" : "  FAIL");
                } else {
                    std::printf("W4X-%-3d %-6s %5d x %6d nc %2d  %8.2f us %5.1f GB/s (%4.2fx fl %7.2f)  rel %.2e%s\n",
                                gs, sh.name, sh.n_in, sh.n_out, nc, us, gbps, r_floor, floor_us, rel,
                                rel < bound ? "" : "  FAIL");
                }
                std::fflush(stdout);
            }
        }
    }

    // Optional merged gate+up SwiGLU kernel parity & bench (GS = 128 only)
    if (mode == "all" || mode == "128" || mode == "gu") {
        const int nin = 5120, nff = 17408, nout = 5120;
        const size_t gu_bytes = strata::q36xr::w4x_gu_merged_bytes(nin, nff);
        const size_t down_bytes = strata::q36xr::w4_bytes(nff, nout);
        const size_t gu_dw = gu_bytes / 4, dn_dw = down_bytes / 4;
        const int nd_gu = (int) std::max<size_t>(1, POOL_DW / gu_dw);
        const size_t act_dn_sz = strata::q36xr::w4x_act_bytes(nff, maxc, 128);
        if (gu_dw + dn_dw <= POOL_DW) {
            void* dact_dn = sycl::aligned_alloc_device(4096, act_dn_sz + 4096, q);
            void* dact_ref = sycl::aligned_alloc_device(4096, act_dn_sz + 4096, q);
            uint32_t* dgate = (uint32_t*) sycl::aligned_alloc_device(4096, strata::q36xr::w4_bytes(nin, nff), q);
            uint32_t* dup   = (uint32_t*) sycl::aligned_alloc_device(4096, strata::q36xr::w4_bytes(nin, nff), q);
            float* d_sg = sycl::malloc_device<float>((size_t) maxc * nff, q);
            float* d_su = sycl::malloc_device<float>((size_t) maxc * nff, q);
            const uint32_t* d_down = dpool + gu_dw;
            const uint32_t* h_down = hpool.data() + gu_dw;
            if (dact_dn && dact_ref && dgate && dup && d_sg && d_su) {
                const uint32_t* merged_h = hpool.data();
                const int n_fg = nff / 128, bpr = nin / 256;
                std::vector<uint32_t> hgate(strata::q36xr::w4_bytes(nin, nff) / 4);
                std::vector<uint32_t> hup(strata::q36xr::w4_bytes(nin, nff) / 4);
                for (int fg = 0; fg < n_fg; ++fg) {
                    for (int lt = 0; lt < 8; ++lt) {
                        const int tile = fg * 8 + lt;
                        for (int kb = 0; kb < bpr; ++kb) {
                            const uint32_t* mg = merged_h + (((size_t) fg * 8 + lt) * bpr + kb) * 2 * REC;
                            std::memcpy(&hgate[((size_t) tile * bpr + kb) * REC], mg, REC * 4);
                            std::memcpy(&hup[((size_t) tile * bpr + kb) * REC], mg + REC, REC * 4);
                        }
                    }
                }
                q.memcpy(dgate, hgate.data(), hgate.size() * 4).wait();
                q.memcpy(dup,   hup.data(),   hup.size() * 4).wait();
                const double floor_gu = (double) gu_bytes / 590e9 * 1e6;

                for (const int nc : ncl) {
                    if (nc < 1 || nc > maxc) continue;
                    strata::q36xr::w4x_quantize(dx, nin, dact, nin, nc, 128, &q);
                    const double us = time_graph_warm(q, std::max(nd_gu, 16), [&](int m) {
                        strata::q36xr::mmvq_w4x_gu_swiglu(dpool + (size_t) (m % nd_gu) * gu_dw, dact, dact_dn, nin, nff, nc, 128, &q);
                    }, 9, 0.4);
                    strata::q36xr::mmvq_w4x_gu_swiglu(dpool, dact, dact_dn, nin, nff, nc, 128, &q);
                    strata::q36xr::mmvq_w4x(d_down, dact_dn, y, nff, nout, nc, 128, &q, ks_ovr);
                    strata::q36xr::mmvq_w4x(dgate, dact, d_sg, nin, nff, nc, 128, &q);
                    strata::q36xr::mmvq_w4x(dup,   dact, d_su, nin, nff, nc, 128, &q);
                    strata::q36xr::w4x_swiglu_quantize(d_sg, d_su, dact_ref, nff, nc, 128, &q);
                    q.wait();

                    const size_t act_cur = strata::q36xr::w4x_act_bytes(nff, nc, 128);
                    std::vector<uint8_t> ha_dn(act_cur), ha_ref(act_cur);
                    std::vector<float> hy((size_t) nc * nout);
                    q.memcpy(ha_dn.data(), dact_dn, act_cur).wait();
                    q.memcpy(ha_ref.data(), dact_ref, act_cur).wait();
                    q.memcpy(hy.data(), y, hy.size() * 4).wait();
                    const std::vector<double> xd_dn = decode_w4x_act(ha_dn, nff, nc, 128);
                    const std::vector<double> xd_ref = decode_w4x_act(ha_ref, nff, nc, 128);
                    double act_ss = 0.0, act_sd = 0.0;
                    if (!xd_dn.empty() && !xd_ref.empty()) {
                        for (size_t i = 0; i < xd_dn.size(); ++i) {
                            act_ss += xd_ref[i] * xd_ref[i];
                            act_sd += (xd_dn[i] - xd_ref[i]) * (xd_dn[i] - xd_ref[i]);
                        }
                    }
                    const double act_rel = (xd_dn.empty() || xd_ref.empty()) ? 1.0 : std::sqrt(act_sd / std::max(act_ss, 1e-30));
                    const double down_rel = xd_dn.empty() ? 1.0 : w4x_f64_rel(h_down, xd_dn, hy, nff, nout, nc);
                    if (!(act_rel < 2e-3 && down_rel < 5e-5)) ++bad;
                    std::printf("W4X-GU  gu+swi %5d x %6d nc %2d  %8.2f us %5.1f GB/s (%4.2fx fl %7.2f)  act rel %.2e down rel %.2e%s\n",
                                nin, nff, nc, us, gu_bytes / us / 1e3, us / floor_gu, floor_gu, act_rel, down_rel,
                                (act_rel < 2e-3 && down_rel < 5e-5) ? "" : "  FAIL");
                    std::fflush(stdout);
                }
            }
            if (dact_dn) sycl::free(dact_dn, q);
            if (dact_ref) sycl::free(dact_ref, q);
            if (dgate) sycl::free(dgate, q);
            if (dup) sycl::free(dup, q);
            if (d_sg) sycl::free(d_sg, q);
            if (d_su) sycl::free(d_su, q);
        }
    }

    sycl::free(dpool, q); sycl::free(y, q); sycl::free(y_solo, q);
    sycl::free(dh, q); sycl::free(dn, q); sycl::free(dir, q);
    sycl::free(dx, q); sycl::free(du, q); sycl::free(dact, q);
    return bad ? 1 : 0;
}

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

// --xatt [T list] [cells]: the matrix-engine decode attention (q36_xatt.dp.cpp) on random K / V / queries.  Checks it
// against the same quantized math in float64 (the device's exp is a 1.2e-7 series: a 16-bit weight may differ by one),
// reports how far the quantized result is from the plain FP attention, checks that a window's rows are bitwise the
// one-row calls, and times a call (16 a graph: the 27B's attention layers).
int xatt_bench(const char* tl_s, int cells, int ns) {   // ns 4: K as the engine's 8-bit cache (a scale per 64 values)
    namespace xa = strata::q36xatt;
    namespace FB = strata::kernels;
    const std::vector<int> tl = parse_list(tl_s);
    sycl::queue q(sycl::gpu_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    const int HKV = 4, H = 24, HD = 256, NCHK = 96, TM = 16;
    const int ctx = cells + 64, ctxp = xa::ctx_pad(ctx), capc = xa::chunk_cap(ctx, NCHK);
    uint64_t st = 0x9E3779B97F4A7C15ull;
    auto xr = [&] { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; };
    auto nrm = [&] { double a = 0; for (int i = 0; i < 6; ++i) a += (double) (xr() >> 11) / 9007199254740992.0; return (float) ((a - 3.0) * 1.41421356); };
    std::vector<float> K((size_t) HKV * cells * HD), V(K.size()), Q((size_t) TM * H * HD);
    for (auto& v : K) v = nrm() * 1.5f;
    for (auto& v : V) v = nrm();
    for (auto& v : Q) v = nrm() * 1.5f;
    for (size_t c = 0; c < (size_t) HKV * cells; ++c) {   // (cells of unequal size, as a real cache's)
        const float a = 0.5f + (float) (xr() & 1023) / 512.0f;
        for (int d = 0; d < HD; ++d) V[c * HD + d] *= a;
    }
    std::vector<uint32_t> hk8(xa::k8_dwords(HKV, ctxp), 0), hv8(xa::v8t_dwords(HKV, ctxp), 0);
    std::vector<float> hks(xa::sc_floats(HKV, ctxp), 0.f), hvs(hks.size(), 0.f);
    std::vector<uint16_t> hk16((size_t) HKV * ctxp * 4, 0);
    auto* k8b = reinterpret_cast<int8_t*>(hk8.data());
    auto* v8b = reinterpret_cast<int8_t*>(hv8.data());
    auto code = [](float v, float sc) { int c = sc > 0.f ? (int) std::nearbyint(v / sc) : 0; return (int8_t) std::max(-127, std::min(127, c)); };
    for (int h = 0; h < HKV; ++h)
        for (int c = 0; c < cells; ++c) {
            const float* k = &K[((size_t) h * cells + c) * HD];
            const float* v = &V[((size_t) h * cells + c) * HD];
            float mk = 0.f, mv = 0.f;
            for (int d = 0; d < HD; ++d) { mk = std::max(mk, std::fabs(k[d])); mv = std::max(mv, std::fabs(v[d])); }
            const float sk = mk / 127.0f, sv = mv / 127.0f;
            hks[(size_t) h * ctxp + c] = sk;
            hvs[(size_t) h * ctxp + c] = sv;
            float sg[4];
            for (int g4 = 0; g4 < 4; ++g4) {
                float mg = 0.f;
                for (int d = 0; d < 64; ++d) mg = std::max(mg, std::fabs(k[64 * g4 + d]));
                const uint16_t hb = FB::f16_from_f32(mg / 127.0f);
                hk16[((size_t) h * ctxp + c) * 4 + g4] = hb;
                sg[g4] = FB::f32_from_f16(hb);
            }
            for (int d = 0; d < HD; ++d) {
                k8b[((size_t) h * ctxp + c) * HD + d] = ns == 4 ? code(k[d], sg[d / 64]) : code(k[d], sk);
                v8b[(((size_t) h * (ctxp / 32) + c / 32) * 256 + d) * 32 + c % 32] = code(v[d], sv);
            }
        }
    auto dev = [&](size_t bytes) { return sycl::aligned_alloc_device((size_t) 2 << 20, std::max<size_t>(bytes, 64), q); };
    auto* dk8 = (uint32_t*) dev(hk8.size() * 4); auto* dv8 = (uint32_t*) dev(hv8.size() * 4);
    auto* dks = (float*) dev(hks.size() * 4); auto* dvs = (float*) dev(hvs.size() * 4);
    auto* dq = (float*) dev(Q.size() * 4);
    auto* dk16 = (uint16_t*) dev(hk16.size() * 2);
    q.memcpy(dk16, hk16.data(), hk16.size() * 2);
    auto* dpos = (int32_t*) dev(64);
    auto* dqb = (uint32_t*) dev(xa::qb_dwords(HKV) * 4); auto* dqs = (float*) dev(xa::qs_floats(HKV) * 4);
    auto* dsc = (float*) dev(xa::score_floats(HKV, NCHK, capc) * 4);
    auto* dpb = (uint32_t*) dev(xa::pb_dwords(HKV, NCHK, capc) * 4);
    auto* dsr = (float*) dev(xa::srow_floats(HKV, NCHK) * 4);
    const size_t rows_m = (size_t) TM * H;
    auto* dpa = (float*) dev(rows_m * NCHK * HD * 4); auto* dpm = (float*) dev(rows_m * NCHK * 4); auto* dpl = (float*) dev(rows_m * NCHK * 4);
    q.memcpy(dk8, hk8.data(), hk8.size() * 4); q.memcpy(dv8, hv8.data(), hv8.size() * 4);
    q.memcpy(dks, hks.data(), hks.size() * 4); q.memcpy(dvs, hvs.data(), hvs.size() * 4);
    q.memcpy(dq, Q.data(), Q.size() * 4);
    q.wait();
    std::printf("xatt (K scales a cell: %d): %d cells, %d KV heads x %d query heads, %d chunks a row, a chunk's scratch %d cells; scratch %.0f MB\n", ns, cells, HKV, H / HKV, NCHK,
                capc, (double) (xa::score_floats(HKV, NCHK, capc) + xa::pb_dwords(HKV, NCHK, capc)) * 4 / 1048576.0);
    auto chunk_cells = [&](int n) { const int per = (n + NCHK - 1) / NCHK; const int cs = (per + 63) / 64 * 64; return cs > 64 ? cs : 64; };
    auto call_of = [&](const float* qrows, int t0, int nt) {
        xa::Call c{};
        c.q = qrows; c.pos0 = dpos; c.t0 = t0; c.nt = nt; c.H = H; c.HKV = HKV; c.nchk = NCHK; c.scale = 1.0f / 16.0f;
        c.k8 = dk8; c.ks = dks; c.v8t = dv8; c.vs = dvs; c.ctxp = ctxp;
        c.ns = ns; c.ks16 = dk16; c.kstride = ctxp;
        c.qb = dqb; c.qs = dqs; c.sc = dsc; c.pb = dpb; c.srow = dsr; c.capc = capc;
        c.pacc = dpa; c.pm = dpm; c.pl = dpl;
        return c;
    };
    // the device's partials merged on the host: out[row][256]
    std::vector<float> pa, pm, pl;   // (the last call's partials)
    auto merged = [&](int nt, int pos0, std::vector<float>& out) {
        const size_t rows = (size_t) nt * H;
        pa.assign(rows * NCHK * HD, 0.f); pm.assign(rows * NCHK, 0.f); pl.assign(rows * NCHK, 0.f);
        q.memcpy(pa.data(), dpa, pa.size() * 4); q.memcpy(pm.data(), dpm, pm.size() * 4); q.memcpy(pl.data(), dpl, pl.size() * 4);
        q.wait();
        out.assign(rows * HD, 0.f);
        for (size_t row = 0; row < rows; ++row) {
            const int n = pos0 + (int) (row / H) + 1, cs = chunk_cells(n), nc = (n + cs - 1) / cs;
            double M = -1e30;
            for (int c = 0; c < nc; ++c) M = std::max(M, (double) pm[row * NCHK + c]);
            double den = 0;
            std::vector<double> num(HD, 0.0);
            for (int c = 0; c < nc; ++c) {
                const double w = std::exp((double) pm[row * NCHK + c] - M);
                den += w * pl[row * NCHK + c];
                for (int d = 0; d < HD; ++d) num[d] += w * pa[(row * NCHK + c) * HD + d];
            }
            for (int d = 0; d < HD; ++d) out[row * HD + d] = (float) (num[d] / den);
        }
    };
    int bad = 0;
    for (int T : tl) {
        if (T < 1 || T > TM || cells < T + 64) continue;
        const int pos0 = cells - T;
        q.memcpy(dpos, &pos0, 4).wait();
        const xa::Call c = call_of(dq, 0, T);
        if (!xa::attend(c, &q)) { std::printf("xatt T=%d: not taken\n", T); ++bad; continue; }
        q.wait();
        std::vector<float> out;
        merged(T, pos0, out);
        const std::vector<float> wpa = pa, wpm = pm, wpl = pl;
        // the references for a sample of rows: the same quantized math in float64, and the plain attention
        double e_q = 0, e_x = 0, nrm2 = 0;
        const int samp[] = {0, 7, H - 1, (T - 1) * H + 3, T * H - 1};
        for (int row : samp) {
            const int t = row / H, head = row % H, kvh = head / (H / HKV), n = pos0 + t + 1;
            const float* qr = &Q[(size_t) row * HD];
            float mx = 0.f;
            for (int d = 0; d < HD; ++d) mx = std::max(mx, std::fabs(qr[d]));
            const float qsc0 = mx / 127.0f, inv = qsc0 > 0.f ? 1.0f / qsc0 : 0.f, qsc = qsc0 * (1.0f / 16.0f);
            int8_t qc[256];
            for (int d = 0; d < HD; ++d) {
                const float y = qr[d] * inv;
                qc[d] = (int8_t) std::max(-127, std::min(127, (int) (y + (y < 0.f ? -0.5f : 0.5f))));
            }
            const int cs = chunk_cells(n), nc = (n + cs - 1) / cs;
            std::vector<double> cm(nc), cl(nc), cacc((size_t) nc * HD, 0.0);
            std::vector<double> sx(n);
            for (int cc = 0; cc < n; ++cc) {
                double a = 0;
                for (int d = 0; d < HD; ++d) a += (double) qr[d] * K[((size_t) kvh * cells + cc) * HD + d];
                sx[cc] = a / 16.0;
            }
            for (int ci = 0; ci < nc; ++ci) {
                const int a = ci * cs, b = std::min(a + cs, n);
                std::vector<float> s(b - a);
                float m = -1e30f, vmx = 0.f;
                for (int cc = a; cc < b; ++cc) {
                    const int8_t* kk = &k8b[((size_t) kvh * ctxp + cc) * HD];
                    if (ns == 4) {
                        float o = 0.f;
                        for (int g4 = 0; g4 < 4; ++g4) {
                            int dot = 0;
                            for (int d = 0; d < 64; ++d) dot += (int) qc[64 * g4 + d] * kk[64 * g4 + d];
                            o += (float) dot * FB::f32_from_f16(hk16[((size_t) kvh * ctxp + cc) * 4 + g4]);
                        }
                        s[cc - a] = o * qsc;
                    } else {
                        int dot = 0;
                        for (int d = 0; d < HD; ++d) dot += (int) qc[d] * kk[d];
                        s[cc - a] = (float) dot * (qsc * hks[(size_t) kvh * ctxp + cc]);
                    }
                    m = std::max(m, s[cc - a]);
                    vmx = std::max(vmx, hvs[(size_t) kvh * ctxp + cc]);
                }
                const float i65 = 65535.0f / std::max(vmx, 1e-30f);
                double l = 0;
                std::vector<long long> acc(HD, 0);
                for (int cc = a; cc < b; ++cc) {
                    const float pr = (float) std::exp((double) std::max(s[cc - a] - m, -87.0f));
                    l += pr;
                    const float w = std::min(pr * (i65 * hvs[(size_t) kvh * ctxp + cc]) + 0.5f, 65535.0f);
                    const long long qw = (long long) w;
                    for (int d = 0; d < HD; ++d) acc[d] += qw * v8b[(((size_t) kvh * (ctxp / 32) + cc / 32) * 256 + d) * 32 + cc % 32];
                }
                cm[ci] = m; cl[ci] = l;
                for (int d = 0; d < HD; ++d) cacc[(size_t) ci * HD + d] = (double) acc[d] * (double) (vmx * (1.0f / 65535.0f));
            }
            double M = -1e30, MX = -1e300;
            for (int ci = 0; ci < nc; ++ci) M = std::max(M, cm[ci]);
            for (int cc = 0; cc < n; ++cc) MX = std::max(MX, sx[cc]);
            double den = 0, dx = 0;
            std::vector<double> num(HD, 0.0), nx(HD, 0.0);
            for (int ci = 0; ci < nc; ++ci) {
                const double w = std::exp(cm[ci] - M);
                den += w * cl[ci];
                for (int d = 0; d < HD; ++d) num[d] += w * cacc[(size_t) ci * HD + d];
            }
            for (int cc = 0; cc < n; ++cc) {
                const double w = std::exp(sx[cc] - MX);
                dx += w;
                for (int d = 0; d < HD; ++d) nx[d] += w * V[((size_t) kvh * cells + cc) * HD + d];
            }
            for (int d = 0; d < HD; ++d) {
                const double o = out[(size_t) row * HD + d], rq = num[d] / den, rx = nx[d] / dx;
                e_q += (o - rq) * (o - rq); e_x += (o - rx) * (o - rx); nrm2 += rx * rx;
            }
        }
        const double rq = std::sqrt(e_q / nrm2), rx = std::sqrt(e_x / nrm2);
        // a window's rows against the one-row calls of the same tokens
        int diff = 0;
        for (int t = 0; t < T && T > 1; ++t) {
            const int p1 = pos0 + t;
            q.memcpy(dpos, &p1, 4).wait();
            const xa::Call c1 = call_of(dq + (size_t) t * H * HD, 0, 1);
            xa::attend(c1, &q);
            q.wait();
            std::vector<float> o1;
            merged(1, p1, o1);
            if (std::memcmp(o1.data(), &out[(size_t) t * H * HD], (size_t) H * HD * 4) != 0) {
                ++diff;
                if (t == 0 || t == T - 1) {   // which part differs: the chunks' max, sum, value sums (the row's own chunks only)
                    const int n = p1 + 1, cs = chunk_cells(n), nc = (n + cs - 1) / cs;
                    long dm = 0, dl = 0, da = 0;
                    double worst = 0;
                    for (int row = 0; row < H; ++row)
                        for (int c2 = 0; c2 < nc; ++c2) {
                            const size_t i1 = (size_t) row * NCHK + c2, iw = ((size_t) t * H + row) * NCHK + c2;
                            dm += pm[i1] != wpm[iw];
                            dl += pl[i1] != wpl[iw];
                            for (int d = 0; d < HD; ++d) {
                                const float x1 = pa[i1 * HD + d], xw = wpa[iw * HD + d];
                                if (x1 != xw) { ++da; worst = std::max(worst, (double) std::fabs(x1 - xw) / (std::fabs(x1) + 1e-30)); }
                            }
                        }
                    std::printf("  token %d: chunk max differs %ld, sum %ld, value sums %ld (worst relative %.2e)\n", t, dm, dl, da, worst);
                }
            }
        }
        q.memcpy(dpos, &pos0, 4).wait();
        const double us = time_graph(q, 16, [&](int) { xa::attend(c, &q); }, 15);
        const bool ok = rq < 2e-4 && diff == 0;
        if (!ok) ++bad;
        std::printf("xatt T=%2d (%2d rows a KV head): %8.1f us a call | vs the quantized float64 %.2e, vs plain FP attention %.2e, "
                    "tokens differing from their one-row call %d %s\n", T, T * H / HKV, us, rq, rx, diff, ok ? "ok" : "FAIL");
        std::fflush(stdout);
    }
    return bad ? 1 : 0;
}

double time_graph_warm(sycl::queue& q, int nmat, const std::function<void(int)>& launch, double warm_sec, int reps) {
    sx::command_graph g(q.get_context(), q.get_device());
    g.begin_recording(q);
    for (int m = 0; m < nmat; ++m) launch(m);
    g.end_recording(q);
    auto ge = g.finalize();
    const auto t_start = std::chrono::steady_clock::now();
    int warm_iters = 0;
    do {
        q.ext_oneapi_graph(ge);
        q.wait();
        ++warm_iters;
    } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() < warm_sec || warm_iters < 2);
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

// --xatt2 [T_list] [ctx_list] [ns]   |   --xatt2-sweep [T_list] [ctx_list] [ns]
int xatt2_bench(bool sweep, const char* tl_s, const char* cl_s, int ns) {
    namespace xa = strata::q36xatt;
    namespace FB = strata::kernels;
    const std::vector<int> tl = parse_list(tl_s);
    const std::vector<int> cl = parse_list(cl_s);
    sycl::queue q(sycl::gpu_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    const int HKV = 4, H = 24, HD = 256, NSEG_MAX = xa::kMaxSegs2, TM = 16;

    auto dev = [&](size_t bytes) { return sycl::aligned_alloc_device((size_t) 2 << 20, std::max<size_t>(bytes, 64), q); };
    uint64_t st = 0xD1B54A32D192ED03ull;
    auto xr = [&] { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; };
    auto nrm = [&] { double a = 0; for (int i = 0; i < 6; ++i) a += (double) (xr() >> 11) / 9007199254740992.0; return (float) ((a - 3.0) * 1.41421356); };
    auto code = [](float v, float sc) { int c = sc > 0.f ? (int) std::nearbyint(v / sc) : 0; return (int8_t) std::max(-127, std::min(127, c)); };

    int max_ctx_cells = 0;
    for (int c : cl) max_ctx_cells = std::max(max_ctx_cells, c);

    // Scratch & partials shared across parity + timing.
    auto* dpos = (int32_t*) dev(TM * sizeof(int32_t));
    auto* dtp0 = (int32_t*) dev(sizeof(int32_t));
    auto* dtmask = (uint32_t*) dev(TM * sizeof(uint32_t));
    auto* dqb = (uint32_t*) dev(xa::qb2_dwords(HKV) * 4);
    auto* dqs = (float*) dev(xa::qs2_floats(HKV) * 4);
    auto* d_dbg_flag = (uint32_t*) dev(64);
    const size_t rows_m = (size_t) TM * H;
    auto* dpa = (float*) dev(rows_m * NSEG_MAX * HD * 4);
    auto* dpm = (float*) dev(rows_m * NSEG_MAX * 4);
    auto* dpl = (float*) dev(rows_m * NSEG_MAX * 4);
    auto* dout = (float*) dev(rows_m * HD * 4);
    auto* dgate = (float*) dev(rows_m * HD * 4);
    auto* dq = (float*) dev(rows_m * HD * 4);
    q.memset(d_dbg_flag, 0, 4).wait();

    std::vector<float> Q(rows_m * HD), G(rows_m * HD);
    for (auto& v : Q) v = nrm() * 1.5f;
    for (auto& v : G) v = nrm() * 1.2f;
    q.memcpy(dq, Q.data(), Q.size() * 4);
    q.memcpy(dgate, G.data(), G.size() * 4);
    q.wait();

    int bad = 0;
    // Reference quantized float64 attention matching XATT2 (int8 Q, int8/fp16 K scales, int8 V, u16/u8 per-32-cell V-scale quant, segment merge).
    struct SegRef { double M = -1e300, l = 0.0; std::vector<double> o; };
    auto ref_row = [&](int kvh, const float* qr, int n, uint32_t tmask_row, int p0_tree,
                       int kst, int ctxp, int ns_ref, bool u16, float gate_val_scale,
                       const std::vector<float>& K_raw, const std::vector<float>& V_raw,
                       const int8_t* k8b, const float* hks, const uint16_t* hk16,
                       const int8_t* v8b, const float* hvs, int cells_raw, int nseg_ref,
                       double& rq_out, double& rx_out, const float* got, double& nrm2_out,
                       std::vector<SegRef>* segs_out = nullptr) {
        (void) gate_val_scale;
        float mx = 0.f;
        for (int d = 0; d < HD; ++d) mx = std::max(mx, std::fabs(qr[d]));
        const float qsc0 = mx / 127.0f, inv = qsc0 > 0.f ? 1.0f / qsc0 : 0.f, qsc = qsc0 * (1.0f / 16.0f);
        int8_t qc[256];
        for (int d = 0; d < HD; ++d) {
            const float y = qr[d] * inv;
            qc[d] = (int8_t) std::max(-127, std::min(127, (int) (y + (y < 0.f ? -0.5f : 0.5f))));
        }
        // Segmented online softmax matching kernel nseg_ref split and per-32-cell vmx = max(p * vs) V quantization.
        const int nseg_eff = std::max(1, nseg_ref);
        const int nblk = (n + 31) / 32;
        const int per = std::max(1, (nblk + nseg_eff - 1) / nseg_eff);
        const int max_q = u16 ? 65535 : 255;
        double M_glob = -1e300, l_glob = 0.0;
        std::vector<double> o_glob(HD, 0.0);
        for (int seg = 0; seg < nseg_eff; ++seg) {
            const int blk_a = seg * per;
            if (blk_a >= nblk) break;
            const int blk_b = std::min(blk_a + per, nblk);
            double M = -1e300, l_acc = 0.0;
            double o_acc[HD] = {};
            for (int blk = blk_a; blk < blk_b; ++blk) {
                const int b0 = blk * 32, b1 = std::min(b0 + 32, n);
                double s[32];
                double mb = -1e300;
                for (int cc = b0; cc < b1; ++cc) {
                    bool valid = true;
                    if (tmask_row != 0xffffffffu && cc >= p0_tree) {
                        const int u = cc - p0_tree;
                        valid = (u >= 0 && u < 32) && ((tmask_row >> u) & 1u);
                    }
                    if (!valid) { s[cc - b0] = -1e30; continue; }
                    const int8_t* kk = &k8b[((size_t) kvh * kst + cc) * HD];
                    if (ns_ref == 4) {
                        float o = 0.f;
                        for (int g4 = 0; g4 < 4; ++g4) {
                            int dot = 0;
                            for (int d = 0; d < 64; ++d) dot += (int) qc[64 * g4 + d] * kk[64 * g4 + d];
                            o += (float) dot * FB::f32_from_f16(hk16[((size_t) kvh * kst + cc) * 4 + g4]);
                        }
                        s[cc - b0] = (double) (o * qsc);
                    } else {
                        int dot = 0;
                        for (int d = 0; d < HD; ++d) dot += (int) qc[d] * kk[d];
                        s[cc - b0] = (double) ((float) dot * (qsc * hks[(size_t) kvh * kst + cc]));
                    }
                    mb = std::max(mb, s[cc - b0]);
                }
                if (mb <= -1e29) continue;
                const double mn = std::max(M, mb);
                const double alpha = (M > -1e29) ? std::exp(M - mn) : 0.0;
                for (int d = 0; d < HD; ++d) o_acc[d] *= alpha;
                double lb = 0.0, vmx = 0.0;
                double pv[32] = {};
                for (int cc = b0; cc < b1; ++cc) {
                    if (s[cc - b0] <= -1e29) continue;
                    const double p = std::exp(std::max(s[cc - b0] - mn, -87.0));
                    lb += p;
                    const double pvs = p * (double) hvs[(size_t) kvh * ctxp + cc];
                    pv[cc - b0] = pvs;
                    vmx = std::max(vmx, pvs);
                }
                const double i_q = (vmx > 1e-30) ? ((double) max_q / vmx) : 0.0;
                long long acc[HD] = {};
                for (int cc = b0; cc < b1; ++cc) {
                    if (s[cc - b0] <= -1e29) continue;
                    const long long qw = std::min<long long>((long long) max_q, (long long) (pv[cc - b0] * i_q + 0.5));
                    for (int d = 0; d < HD; ++d)
                        acc[d] += qw * (long long) v8b[(((size_t) kvh * (ctxp / 32) + cc / 32) * 256 + d) * 32 + cc % 32];
                }
                const double srw = (vmx > 1e-30) ? (vmx / (double) max_q) : 0.0;
                for (int d = 0; d < HD; ++d) o_acc[d] += (double) acc[d] * srw;
                l_acc = l_acc * alpha + lb;
                M = mn;
            }
            if (segs_out) {   // the kernel's partials of this segment, for the stage-by-stage diagnostics
                SegRef sr;
                sr.M = M; sr.l = l_acc; sr.o.assign(o_acc, o_acc + HD);
                segs_out->push_back(std::move(sr));
            }
            if (l_acc > 0.0) {
                if (l_glob <= 0.0) {
                    M_glob = M;
                    l_glob = l_acc;
                    for (int d = 0; d < HD; ++d) o_glob[d] = o_acc[d];
                } else {
                    const double mn = std::max(M_glob, M);
                    const double ag = std::exp(M_glob - mn);
                    const double as = std::exp(M - mn);
                    for (int d = 0; d < HD; ++d) o_glob[d] = o_glob[d] * ag + o_acc[d] * as;
                    l_glob = l_glob * ag + l_acc * as;
                    M_glob = mn;
                }
            }
        }
        // Plain float64 unquantized attention reference:
        double MX = -1e300, dx = 0.0;
        std::vector<double> sx(n, -1e30), nx(HD, 0.0);
        for (int cc = 0; cc < n; ++cc) {
            bool valid = true;
            if (tmask_row != 0xffffffffu && cc >= p0_tree) {
                const int u = cc - p0_tree;
                valid = (u >= 0 && u < 32) && ((tmask_row >> u) & 1u);
            }
            if (!valid) continue;
            double a = 0.0;
            for (int d = 0; d < HD; ++d) a += (double) qr[d] * (double) K_raw[((size_t) kvh * cells_raw + cc) * HD + d];
            sx[cc] = a / 16.0;
            MX = std::max(MX, sx[cc]);
        }
        for (int cc = 0; cc < n; ++cc) {
            if (sx[cc] <= -1e29) continue;
            const double w = std::exp(sx[cc] - MX);
            dx += w;
            for (int d = 0; d < HD; ++d) nx[d] += w * (double) V_raw[((size_t) kvh * cells_raw + cc) * HD + d];
        }
        double eq = 0.0, ex = 0.0, n2 = 0.0;
        for (int d = 0; d < HD; ++d) {
            const double o = got[d];
            const double rq = l_glob > 0.0 ? o_glob[d] / l_glob : 0.0;
            const double rx = dx > 0.0 ? nx[d] / dx : 0.0;
            eq += (o - rq) * (o - rq);
            ex += (o - rx) * (o - rx);
            n2 += rx * rx;
        }
        rq_out += eq; rx_out += ex; nrm2_out += n2;
    };

    // The kernel's partials (pm, pl, pacc at index base + seg) of one row against the contract's per segment.
    auto diag_segs = [&](const std::vector<SegRef>& segs, const std::vector<float>& pm_h, const std::vector<float>& pl_h,
                         const std::vector<float>& pa_h, size_t base, double& d_m, double& d_l, double& d_o) {
        for (size_t sg = 0; sg < segs.size(); ++sg) {
            const SegRef& r = segs[sg];
            if (!(r.l > 0.0)) continue;
            const size_t i = base + sg;
            if (i >= pm_h.size()) continue;
            d_m = std::max(d_m, std::fabs((double) pm_h[i] - r.M));
            d_l = std::max(d_l, std::fabs((double) pl_h[i] - r.l) / r.l);
            double e2 = 0.0, n2s = 0.0;
            for (int d = 0; d < HD; ++d) {
                const double a = pa_h[i * HD + d], b = r.o[d];
                e2 += (a - b) * (a - b); n2s += b * b;
            }
            d_o = std::max(d_o, std::sqrt(e2 / std::max(n2s, 1e-300)));
        }
    };

    // Helper: quantize synthetic K/V for parity and allocate matching device buffers.
    struct ParityKv {
        int cells = 0, ctx = 0, ctxp = 0;
        std::vector<float> K, V, hks, hvs;
        std::vector<uint32_t> hk8, hv8;
        std::vector<uint16_t> hk16;
        int8_t* k8b = nullptr;
        int8_t* v8b = nullptr;
        uint32_t* dk8 = nullptr;
        uint32_t* dv8 = nullptr;
        float* dks = nullptr;
        float* dvs = nullptr;
        uint16_t* dk16 = nullptr;
    };
    auto build_kv = [&](int cells_in) -> ParityKv {
        ParityKv p{};
        p.cells = cells_in;
        p.ctx = cells_in;
        p.ctxp = xa::ctx_pad(p.ctx);
        p.K.resize((size_t) HKV * p.cells * HD);
        p.V.resize(p.K.size());
        for (auto& v : p.K) v = nrm() * 1.5f;
        for (auto& v : p.V) v = nrm();
        for (size_t c = 0; c < (size_t) HKV * p.cells; ++c) {
            const float a = 0.5f + (float) (xr() & 1023) / 512.0f;
            for (int d = 0; d < HD; ++d) p.V[c * HD + d] *= a;
        }
        p.hk8.assign((size_t) HKV * p.ctx * 64, 0);
        p.hv8.assign(xa::v8t_dwords(HKV, p.ctxp), 0);
        p.hks.assign((size_t) HKV * p.ctx, 0.f);
        p.hvs.assign(xa::sc_floats(HKV, p.ctxp), 0.f);
        p.hk16.assign((size_t) HKV * p.ctx * 4, 0);
        p.k8b = reinterpret_cast<int8_t*>(p.hk8.data());
        p.v8b = reinterpret_cast<int8_t*>(p.hv8.data());
        for (int h = 0; h < HKV; ++h)
            for (int c = 0; c < p.cells; ++c) {
                const float* k = &p.K[((size_t) h * p.cells + c) * HD];
                const float* v = &p.V[((size_t) h * p.cells + c) * HD];
                float mk = 0.f, mv = 0.f;
                for (int d = 0; d < HD; ++d) { mk = std::max(mk, std::fabs(k[d])); mv = std::max(mv, std::fabs(v[d])); }
                const float sk = mk / 127.0f, sv = mv / 127.0f;
                p.hks[(size_t) h * p.ctx + c] = sk;
                p.hvs[(size_t) h * p.ctxp + c] = sv;
                float sg[4];
                for (int g4 = 0; g4 < 4; ++g4) {
                    float mg = 0.f;
                    for (int d = 0; d < 64; ++d) mg = std::max(mg, std::fabs(k[64 * g4 + d]));
                    const uint16_t hb = FB::f16_from_f32(mg / 127.0f);
                    p.hk16[((size_t) h * p.ctx + c) * 4 + g4] = hb;
                    sg[g4] = FB::f32_from_f16(hb);
                }
                for (int d = 0; d < HD; ++d) {
                    p.k8b[((size_t) h * p.ctx + c) * HD + d] = ns == 4 ? code(k[d], sg[d / 64]) : code(k[d], sk);
                    p.v8b[(((size_t) h * (p.ctxp / 32) + c / 32) * 256 + d) * 32 + c % 32] = code(v[d], sv);
                }
            }
        p.dk8 = (uint32_t*) dev(p.hk8.size() * 4);
        p.dv8 = (uint32_t*) dev(p.hv8.size() * 4);
        p.dks = (float*) dev(p.hks.size() * 4);
        p.dvs = (float*) dev(p.hvs.size() * 4);
        p.dk16 = (uint16_t*) dev(p.hk16.size() * 2);
        q.memcpy(p.dk8, p.hk8.data(), p.hk8.size() * 4);
        q.memcpy(p.dv8, p.hv8.data(), p.hv8.size() * 4);
        q.memcpy(p.dks, p.hks.data(), p.hks.size() * 4);
        q.memcpy(p.dvs, p.hvs.data(), p.hvs.size() * 4);
        q.memcpy(p.dk16, p.hk16.data(), p.hk16.size() * 2);
        q.wait();
        return p;
    };
    auto free_kv = [&](ParityKv& p) {
        sycl::free(p.dk8, q); sycl::free(p.dv8, q); sycl::free(p.dks, q); sycl::free(p.dvs, q); sycl::free(p.dk16, q);
    };
    auto set_dbg_caps = [&](xa::Call2& c, const ParityKv& p, int T_call) {
        c.dbg_flag = d_dbg_flag;
        c.dbg_q_floats = rows_m * HD;
        c.dbg_k8_dwords = p.hk8.size();
        c.dbg_ks_floats = p.hks.size();
        c.dbg_ks16_u16 = p.hk16.size();
        c.dbg_v8t_dwords = p.hv8.size();
        c.dbg_vs_floats = p.hvs.size();
        c.dbg_out_floats = (size_t) T_call * H * HD;
    };

    // ---- Stage 0: Single small diagnostic case (T=1, cells=4096, form=S1) + host-side max-index vs alloc-size printout
    {
        const int cells = 4096;
        ParityKv p = build_kv(cells);
        const int T = 1, pos0_val = cells - 1;
        xa::Call2 c_small{};
        c_small.q = dq; c_small.pos0 = dpos; c_small.t0 = 0; c_small.nt = T; c_small.H = H; c_small.HKV = HKV; c_small.scale = 1.0f / 16.0f;
        c_small.k8 = p.dk8; c_small.ks = p.dks; c_small.ks16 = p.dk16; c_small.v8t = p.dv8; c_small.vs = p.dvs;
        c_small.ns = ns; c_small.kstride = p.ctx; c_small.ctxp = p.ctxp; c_small.pos_hint = pos0_val + T;
        c_small.qb = dqb; c_small.qs = dqs; c_small.pacc = dpa; c_small.pm = dpm; c_small.pl = dpl;
        c_small.out = dout; c_small.form_override = 0; c_small.u16_override = 1;
        set_dbg_caps(c_small, p, T);
        const int nseg = xa::xatt2_plan_nseg(c_small, 0);
        const int nseg_stride = ((nseg + 3) & ~3) <= NSEG_MAX ? ((nseg + 3) & ~3) : nseg;
        const int blk_end = ((cells + xa::kBlk - 1) / xa::kBlk);   // 128

        std::printf("xatt2 small diagnostic (T=1, cells=%d, kstride=%d, ctxp=%d, ns=%d, form=R8, nseg=%d, stride=%d):\n",
                    cells, p.ctx, p.ctxp, ns, nseg, nseg_stride);
        std::printf("  Host bounds check (max touched elem/byte vs allocated elems/bytes):\n");
        std::printf("    TagXa2Q    q: max=%zu / alloc=%zu floats | qb: max=%zu / alloc=%zu dwords | qs: max=%zu / alloc=%zu floats | SLM=4096 / 131072 B\n",
                    (size_t) H * HD - 1, rows_m * HD,
                    (((size_t) (HKV - 1) * 8u + 7u) * xa::kMaxBatchGroups2 + 0u) * 128u + 127u, p.ctx ? xa::qb2_dwords(HKV) : 0,
                    ((size_t) (HKV - 1) * xa::kMaxBatchGroups2 + 0u) * 16u + 15u, xa::qs2_floats(HKV));
        std::printf("    TagXa2R8   pos0: max=0 / alloc=%d int32 | qb: max=%zu / alloc=%zu dw | qs: max=%zu / alloc=%zu fl | SLM=0 / 131072 B\n",
                    TM,
                    (((size_t) (HKV - 1) * 8u + 7u) * xa::kMaxBatchGroups2 + 0u) * 128u + 127u, xa::qb2_dwords(HKV),
                    ((size_t) (HKV - 1) * xa::kMaxBatchGroups2 + 0u) * 16u + 7u, xa::qs2_floats(HKV));
        std::printf("               k8: max=%zu / alloc=%zu dw | ks16: max=%zu / alloc=%zu u16 | ks: max=%zu / alloc=%zu fl\n",
                    ((size_t) (HKV - 1) * p.ctx + (cells - 1)) * 64u + 63u, p.hk8.size(),
                    ((size_t) (HKV - 1) * p.ctx + (cells - 1)) * 4u + 3u, p.hk16.size(),
                    (size_t) (HKV - 1) * p.ctx + (cells - 1), p.hks.size());
        std::printf("               v8t: max=%zu / alloc=%zu dw (max_blk=%d/%d) | vs: max=%zu / alloc=%zu fl\n",
                    ((size_t) (HKV - 1) * (p.ctxp / xa::kBlk) + (blk_end - 1)) * 2048u + 2047u, p.hv8.size(),
                    blk_end - 1, p.ctxp / xa::kBlk,
                    (size_t) (HKV - 1) * p.ctxp + (blk_end - 1) * xa::kBlk + 31u, p.hvs.size());
        std::printf("               pacc: max=%zu / alloc=%zu fl | pm/pl: max=%zu / alloc=%zu fl\n",
                    ((size_t) (H - 1) * nseg_stride + (nseg - 1)) * HD + HD - 1, rows_m * NSEG_MAX * HD,
                    (size_t) (H - 1) * nseg_stride + (nseg - 1), rows_m * NSEG_MAX);
        std::printf("    TagXa2Merge pm/pl: max=%zu / alloc=%zu fl | pacc: max=%zu / alloc=%zu fl | out: max=%zu / alloc=%zu fl\n",
                    (size_t) (H - 1) * nseg_stride + (nseg - 1), rows_m * NSEG_MAX,
                    ((size_t) (H - 1) * nseg_stride + (nseg - 1)) * HD + HD - 1, rows_m * NSEG_MAX * HD,
                    (size_t) H * HD - 1, rows_m * HD);
        std::fflush(stdout);

        std::vector<float> host_out((size_t) T * H * HD, 0.0f);
        uint32_t h_dbg = 0;
        bool taken = false;
        std::string exc_msg;
        try {
            q.memcpy(dpos, &pos0_val, 4);
            q.memset(d_dbg_flag, 0, 4);
            q.wait();
            taken = xa::attend2(c_small, &q);
            q.wait();
            q.memcpy(host_out.data(), dout, (size_t) T * H * HD * 4);
            q.memcpy(&h_dbg, d_dbg_flag, 4);
            q.wait();
        } catch (const sycl::exception& exc) {
            exc_msg = exc.what();
        }
        double eq = 0, ex = 0, n2 = 0;
        for (int row : {0, 5, 7, H - 1}) {
            ref_row(row / (H / HKV), &Q[(size_t) row * HD], pos0_val + 1, 0xffffffffu, 0x3fffffff,
                    p.ctx, p.ctxp, ns, true, 1.f, p.K, p.V, p.k8b, p.hks.data(), p.hk16.data(),
                    p.v8b, p.hvs.data(), p.cells, nseg, eq, ex, &host_out[(size_t) row * HD], n2);
        }
        const double rq = std::sqrt(eq / n2), rx = std::sqrt(ex / n2);
        const bool ok = exc_msg.empty() && h_dbg == 0 && taken && rq < 3e-4;
        if (!ok) ++bad;
        std::printf("  small T= 1 form=R8 u16=1 pos0=%d | vs q_f64=%.2e (bound 3.0e-04) vs fp64=%.2e dbg=%u %s%s%s\n",
                    pos0_val, rq, rx, h_dbg, ok ? "ok" : "FAIL", exc_msg.empty() ? "" : " EXC=", exc_msg.c_str());
        std::fflush(stdout);
        free_kv(p);
        if (!exc_msg.empty()) {
            std::fprintf(stderr, "xatt2 small case threw sycl::exception; aborting further tests\n");
            return 1;
        }
        if (const char* so = std::getenv("STRATA_Q36_XATT2_SMALL_ONLY"); so && so[0] == '1') {
            sycl::free(dpos, q); sycl::free(dtp0, q); sycl::free(dtmask, q);
            sycl::free(dqb, q); sycl::free(dqs, q); sycl::free(d_dbg_flag, q);
            sycl::free(dpa, q); sycl::free(dpm, q); sycl::free(dpl, q);
            sycl::free(dout, q); sycl::free(dgate, q); sycl::free(dq, q);
            return bad ? 1 : 0;
        }
    }

    // ---- Stage 0b: the softmax's two exps on the device against float64 (the u16 parity's 2x miss of its 3e-4 bound is an exp
    // error of that size per cell if hw exp2 is coarse: error of the output ~ the per-cell relative error of p)
    {
        const int n = 1 << 20;
        std::vector<float> hx(n), hh(n), hp(n);
        for (int i = 0; i < n; ++i) hx[i] = (i < n / 2) ? -(float) ((double) (xr() >> 11) / 9007199254740992.0 * 30.0)
                                                         : -(float) ((double) (xr() >> 11) / 9007199254740992.0 * 87.0);
        auto* dxp = (float*) dev((size_t) n * 4); auto* dhw = (float*) dev((size_t) n * 4); auto* dpo = (float*) dev((size_t) n * 4);
        q.memcpy(dxp, hx.data(), (size_t) n * 4).wait();
        const bool okp = xa::xatt2_exp_probe(&q, dxp, dhw, dpo, n);
        q.wait();
        q.memcpy(hh.data(), dhw, (size_t) n * 4); q.memcpy(hp.data(), dpo, (size_t) n * 4); q.wait();
        double mh = 0, mp = 0, sh = 0, sp = 0, mh30 = 0;
        for (int i = 0; i < n; ++i) {
            const double ex = std::exp((double) hx[i]);
            const double eh = std::fabs((double) hh[i] - ex) / ex, ep = std::fabs((double) hp[i] - ex) / ex;
            mh = std::max(mh, eh); mp = std::max(mp, ep); sh += eh * eh; sp += ep * ep;
            if (i < n / 2) mh30 = std::max(mh30, eh);
        }
        std::printf("exp probe (%d x in [-87,0], half of them in [-30,0]): hardware exp2 relative error max %.2e rms %.2e (max on [-30,0] %.2e); "
                    "series max %.2e rms %.2e%s\n  (a per-cell relative error eps costs ~eps of the output: eps 5e-4 explains the 3e-4..7e-4 u16 misses; "
                    "STRATA_Q36_XATT2_EXP=1 takes the series)\n",
                    n, mh, std::sqrt(sh / n), mh30, mp, std::sqrt(sp / n), okp ? "" : " NOT RUN");
        std::fflush(stdout);
        sycl::free(dxp, q); sycl::free(dhw, q); sycl::free(dpo, q);
    }

    // ---- Stage 1: Functional parity (first context, odd unaligned kstride=cells tail guard + all forms + u8/u16 + tree mask + 4-slot batch)
    {
        const int cells = cl.empty() ? 25123 : (cl[0] | 3);   // force non-multiple-of-32 to exercise tail OOB guard
        ParityKv p = build_kv(cells);
        const int ctx = p.ctx, ctxp = p.ctxp;
        auto& K = p.K; auto& V = p.V; auto& hk8 = p.hk8; auto& hv8 = p.hv8;
        auto& hks = p.hks; auto& hvs = p.hvs; auto& hk16 = p.hk16;
        auto* k8b = p.k8b; auto* v8b = p.v8b;
        auto* dk8 = p.dk8; auto* dv8 = p.dv8; auto* dks = p.dks; auto* dvs = p.dvs; auto* dk16 = p.dk16;

        std::printf("xatt2 parity (ns=%d, cells=%d, kstride=%d tail test):\n", ns, cells, ctx);
        std::printf("  bound contract: NOT bitwise to XATT1 (exp2 + per-32-cell p*vs quant + online softmax across nseg segments); "
                    "expected RMS vs float64 quantized ref <= 3.0e-4 (u16), <= 2.5e-2 (u8)\n");
        std::fflush(stdout);
        std::vector<float> host_out(rows_m * HD);
        auto run_check = [&](int T, int form, int u16, bool use_tmask, int pos0_val, int exp_ov = -1) {
            std::vector<uint32_t> htm(TM, 0xffffffffu);
            int p0_tree = pos0_val;
            std::fill(host_out.begin(), host_out.end(), 0.0f);
            uint32_t h_dbg = 0;
            bool taken = false;
            int nseg_planned = 1, stride_planned = 1;
            std::string exc_msg;
            std::vector<float> hp_m, hp_l, hp_a;   // the kernel's partials (m, l, o a row's segment), for the diagnostics
            try {
                q.memcpy(dpos, &pos0_val, 4);
                q.memset(d_dbg_flag, 0, 4);
                q.wait();
                if (use_tmask) {
                    for (int t = 0; t < T; ++t) htm[t] = (1u << (t + 1)) - 1u;
                    if (T >= 4) htm[2] &= ~(1u << 1);   // non-linear tree branch
                    q.memcpy(dtmask, htm.data(), TM * 4);
                    q.memcpy(dtp0, &p0_tree, 4);
                    q.wait();
                }
                xa::Call2 c{};
                c.q = dq; c.pos0 = dpos; c.t0 = 0; c.nt = T; c.H = H; c.HKV = HKV; c.scale = 1.0f / 16.0f;
                c.k8 = dk8; c.ks = dks; c.ks16 = dk16; c.v8t = dv8; c.vs = dvs;
                c.ns = ns; c.kstride = ctx; c.ctxp = ctxp; c.pos_hint = pos0_val + T;
                c.qb = dqb; c.qs = dqs; c.pacc = dpa; c.pm = dpm; c.pl = dpl;
                c.out = dout;
                if (use_tmask) { c.tmask = dtmask; c.tp0 = dtp0; }
                c.form_override = form; c.u16_override = u16; c.exp_override = exp_ov;
                set_dbg_caps(c, p, T);
                nseg_planned = xa::xatt2_plan_nseg(c, 0);
                stride_planned = xa::xatt2_plan_stride(c);
                taken = xa::attend2(c, &q);
                q.wait();
                q.memcpy(host_out.data(), dout, (size_t) T * H * HD * 4);
                q.memcpy(&h_dbg, d_dbg_flag, 4);
                q.wait();
                const size_t np = (size_t) T * H * (size_t) stride_planned;
                hp_m.resize(np); hp_l.resize(np); hp_a.resize(np * HD);
                q.memcpy(hp_m.data(), dpm, np * 4);
                q.memcpy(hp_l.data(), dpl, np * 4);
                q.memcpy(hp_a.data(), dpa, np * HD * 4);
                q.wait();
            } catch (const sycl::exception& exc) {
                exc_msg = exc.what();
            }
            double eq = 0, ex = 0, n2 = 0;
            double d_m = 0, d_l = 0, d_o = 0;   // worst |dM|, relative dl, relative do over the sampled rows' segments
            const int samp[] = {0, 5, 7, H - 1, (T - 1) * H + 3, T * H - 1};
            for (int row : samp) {
                if (row >= T * H) continue;
                const int t = row / H, head = row % H, kvh = head / (H / HKV);
                const int n = pos0_val + t + 1;
                std::vector<SegRef> segs;
                ref_row(kvh, &Q[(size_t) row * HD], n, htm[t], p0_tree, ctx, ctxp, ns, u16 != 0, 1.f,
                        K, V, k8b, hks.data(), hk16.data(), v8b, hvs.data(), cells, nseg_planned,
                        eq, ex, &host_out[(size_t) row * HD], n2, &segs);
                if (!hp_m.empty()) diag_segs(segs, hp_m, hp_l, hp_a, (size_t) row * stride_planned, d_m, d_l, d_o);
            }
            const double rq = std::sqrt(eq / n2), rx = std::sqrt(ex / n2);
            const double limit = u16 ? 3e-4 : 2.5e-2;
            const bool ok = exc_msg.empty() && h_dbg == 0 && taken && rq < limit;
            if (!ok) ++bad;
            std::printf("  T=%2d form=%-5s u16=%d exp=%s tmask=%d nseg=%2d pos0=%d | vs q_f64=%.2e (bound %.1e) vs fp64=%.2e dbg=%u %s%s%s\n",
                        T, xa::xatt2_form_name(form), u16, exp_ov < 0 ? "env" : exp_ov ? "series" : "hw", use_tmask ? 1 : 0, nseg_planned, pos0_val, rq, limit, rx, h_dbg,
                        ok ? "ok" : "FAIL", exc_msg.empty() ? "" : " EXC=", exc_msg.c_str());
            // stage by stage: the segments' max M, sum l and value sum o against the float64 contract (which stage carries a miss)
            if (!ok || std::getenv("STRATA_Q36_XATT2_DIAG"))
                std::printf("      partials vs contract: max|dM|=%.2e  max rel dl=%.2e  max rel do=%.2e  (rows sampled, every segment)\n", d_m, d_l, d_o);
            std::fflush(stdout);
        };
        for (int T : tl) {
            if (T < 1 || T > TM || cells < T + 8) continue;
            const int pos0 = cells - T;   // exercises exact final cell = cells - 1!
            run_check(T, 0, 1, false, pos0);
            run_check(T, 1, 1, false, pos0);
            run_check(T, 2, 1, false, pos0);
            run_check(T, 3, 1, false, pos0);
            run_check(T, 0, 0, false, pos0);
            run_check(T, 3, 0, false, pos0);
            run_check(T, 0, 1, false, pos0, 0);   // the same u16 forms with the hardware exp2 / the series, whatever the env says
            run_check(T, 3, 1, false, pos0, 0);
            run_check(T, 0, 1, false, pos0, 1);
            run_check(T, 3, 1, false, pos0, 1);
        }
        if (cells > 16) {
            run_check(5, 0, 1, true, cells - 8);
            run_check(5, 3, 1, true, cells - 8);
        }

        // Batch mode parity (4 slots with heterogeneous positions & token counts in ONE launch):
        {
            const int ngrp = 4;
            const int cnts[4] = {1, 3, 2, 2};   // total T = 8
            const int pos_s[4] = {cells - 1, cells - 4, (cells * 3) / 4, cells / 2};
            std::vector<void*> h_ktab(ngrp), h_vtab(ngrp);
            std::vector<const void*> h_mtab(ngrp * 4);
            // Pack contiguous ns=4 slot blobs matching Engine's d_kc_tab / d_vc_tab format.
            const size_t k_blob_b = (size_t) HKV * ctx * HD + (size_t) HKV * ctx * 4 * 2;
            const size_t v_blob_b = xa::v8t_dwords(HKV, ctxp) * 4 + (size_t) HKV * ctxp * 4;
            void* k_blob = dev(k_blob_b);
            void* v_blob = dev(v_blob_b);
            q.memcpy(k_blob, hk8.data(), (size_t) HKV * ctx * HD);
            q.memcpy((int8_t*) k_blob + (size_t) HKV * ctx * HD, hk16.data(), (size_t) HKV * ctx * 4 * 2);
            q.memcpy(v_blob, hv8.data(), xa::v8t_dwords(HKV, ctxp) * 4);
            q.memcpy((int8_t*) v_blob + xa::v8t_dwords(HKV, ctxp) * 4, hvs.data(), (size_t) HKV * ctxp * 4);
            for (int i = 0; i < ngrp; ++i) {
                h_ktab[i] = k_blob;
                h_vtab[i] = v_blob;
                h_mtab[i * 4 + 0] = dk8;
                h_mtab[i * 4 + 1] = (ns == 1) ? (const void*) dks : (const void*) dk16;
                h_mtab[i * 4 + 2] = dv8;
                h_mtab[i * 4 + 3] = dvs;
            }
            void** d_ktab = (void**) dev(ngrp * sizeof(void*));
            void** d_vtab = (void**) dev(ngrp * sizeof(void*));
            const void** d_mtab = (const void**) dev(ngrp * 4 * sizeof(void*));
            q.memcpy(d_ktab, h_ktab.data(), ngrp * sizeof(void*));
            q.memcpy(d_vtab, h_vtab.data(), ngrp * sizeof(void*));
            q.memcpy(d_mtab, h_mtab.data(), ngrp * 4 * sizeof(void*));
            int32_t hrc[TM]{};
            xa::Call2 cb{};
            cb.q = dq; cb.pos0 = dpos; cb.H = H; cb.HKV = HKV; cb.scale = 1.0f / 16.0f;
            cb.ns = ns; cb.ctx = ctx; cb.kstride = ctx; cb.ctxp = ctxp;
            if (ns == 4) { cb.k_tab = d_ktab; cb.v_tab = d_vtab; }
            else { cb.mir_tab = d_mtab; }
            cb.ngrp = ngrp;
            int r0 = 0;
            for (int i = 0; i < ngrp; ++i) {
                cb.grp[i] = {i, r0, cnts[i], pos_s[i] + cnts[i]};
                for (int k = 0; k < cnts[i]; ++k) hrc[r0 + k] = pos_s[i] + k;
                r0 += cnts[i];
            }
            cb.nt = r0;
            cb.qb = dqb; cb.qs = dqs; cb.pacc = dpa; cb.pm = dpm; cb.pl = dpl;
            cb.out = dout; cb.gate = dgate;
            set_dbg_caps(cb, p, r0);
            // One batch run: `form` (-1: the auto policy), the fused gate or not.  Per group (its last token, 4 heads) the
            // error against the contract; the partials against the contract's segments, stage by stage.
            auto run_batch = [&](int form, bool with_gate) {
                uint32_t h_dbg = 0;
                bool taken = false;
                std::string exc_msg;
                std::fill(host_out.begin(), host_out.end(), 0.0f);
                std::vector<int> nseg_grp(ngrp, 1);
                xa::Call2 cbr = cb;
                cbr.form_override = form;
                if (!with_gate) cbr.gate = nullptr;
                int stride_b = 1;
                std::vector<float> hp_m, hp_l, hp_a;
                try {
                    q.memcpy(dpos, hrc, (size_t) r0 * 4);
                    q.memset(d_dbg_flag, 0, 4);
                    q.wait();
                    for (int gi = 0; gi < ngrp; ++gi) nseg_grp[gi] = xa::xatt2_plan_nseg(cbr, gi);
                    stride_b = xa::xatt2_plan_stride(cbr);
                    taken = xa::attend2(cbr, &q);
                    q.wait();
                    q.memcpy(host_out.data(), dout, (size_t) r0 * H * HD * 4);
                    q.memcpy(&h_dbg, d_dbg_flag, 4);
                    const size_t np = (size_t) r0 * H * (size_t) stride_b;
                    hp_m.resize(np); hp_l.resize(np); hp_a.resize(np * HD);
                    q.memcpy(hp_m.data(), dpm, np * 4);
                    q.memcpy(hp_l.data(), dpl, np * 4);
                    q.memcpy(hp_a.data(), dpa, np * HD * 4);
                    q.wait();
                } catch (const sycl::exception& exc) {
                    exc_msg = exc.what();
                }
                // With the gate: the un-gated output is out / sigmoid(gate) (the gate is the host's own float64 sigmoid).
                double eq = 0, ex = 0, n2 = 0;
                std::string per_group;
                for (int gi = 0; gi < ngrp; ++gi) {
                    const int tok = cb.grp[gi].r0 + cb.grp[gi].cnt - 1;
                    double gq = 0, gx = 0, gn = 0, g_m = 0, g_l = 0, g_o = 0;
                    for (int head : {0, 5, 13, 23}) {
                        const size_t row = (size_t) tok * H + head;
                        std::vector<float> ungated(HD);
                        for (int d = 0; d < HD; ++d) {
                            double sg = 1.0;
                            if (with_gate) sg = std::max(1.0 / (1.0 + std::exp(-(double) G[row * HD + d])), 1e-12);
                            ungated[d] = (float) ((double) host_out[row * HD + d] / sg);
                        }
                        std::vector<SegRef> segs;
                        ref_row(head / (H / HKV), &Q[row * HD], hrc[tok] + 1, 0xffffffffu, 0x3fffffff, ctx, ctxp, ns, true, 1.f,
                                K, V, k8b, hks.data(), hk16.data(), v8b, hvs.data(), cells, nseg_grp[gi],
                                gq, gx, ungated.data(), gn, &segs);
                        if (!hp_m.empty()) diag_segs(segs, hp_m, hp_l, hp_a, row * (size_t) stride_b, g_m, g_l, g_o);
                    }
                    eq += gq; ex += gx; n2 += gn;
                    char b[160];
                    std::snprintf(b, sizeof b, " g%d(cnt=%d nseg=%d): q=%.1e dl=%.1e do=%.1e;", gi, cb.grp[gi].cnt, nseg_grp[gi],
                                  std::sqrt(gq / std::max(gn, 1e-300)), g_l, g_o);
                    per_group += b;
                }
                const double rq = std::sqrt(eq / n2), rx = std::sqrt(ex / n2);
                const bool ok = exc_msg.empty() && h_dbg == 0 && taken && rq < 3e-4;
                if (!ok) ++bad;
                std::printf("  batch ngrp=%d nt=%d form=%-5s %s | vs q_f64=%.2e vs fp64=%.2e dbg=%u %s%s%s\n",
                            ngrp, cb.nt, xa::xatt2_form_name(form >= 0 ? form : xa::xatt2_plan_form(cb)),
                            with_gate ? "fused-gate" : "no-gate   ", rq, rx, h_dbg,
                            ok ? "ok" : "FAIL", exc_msg.empty() ? "" : " EXC=", exc_msg.c_str());
                if (!ok || std::getenv("STRATA_Q36_XATT2_DIAG")) std::printf("     per group:%s\n", per_group.c_str());
            };
            run_batch(-1, true);    // the auto policy with the fused gate (what the engine launches)
            run_batch(-1, false);   // the same without the gate (is it the gate, or the attention)
            for (int fm : {0, 1, 2, 3}) run_batch(fm, false);   // every form on the same batch (is it a form)
            std::fflush(stdout);
            sycl::free(k_blob, q); sycl::free(v_blob, q);
            sycl::free(d_ktab, q); sycl::free(d_vtab, q); sycl::free(d_mtab, q);
        }
        free_kv(p);
    }

    // ---- Stage 2: Timing benchmark across context lengths (>= 64 MiB rotating buffers, 0.4 s warmup, device timestamps)
    for (int cells : cl) {
        const int ctx = cells + 64, ctxp = xa::ctx_pad(ctx);
        const size_t kv_bytes_per_buf = (size_t) HKV * ctx * HD + (ns == 4 ? (size_t) HKV * ctx * 4 * 2 : (size_t) HKV * ctx * 4)
                                      + xa::v8t_dwords(HKV, ctxp) * 4 + (size_t) HKV * ctxp * 4;
        // Ensure rotating pool exceeds 64 MiB (e.g. >= 80 MiB) and avoids L2/LLC caching.
        const int n_bufs = std::min(16, std::max(2, (int) (((80ULL << 20) + kv_bytes_per_buf - 1) / kv_bytes_per_buf)));
        struct KvBuf { uint32_t* k8; float* ks; uint16_t* ks16; uint32_t* v8t; float* vs; };
        std::vector<KvBuf> bufs((size_t) n_bufs);
        // Generate one incompressible random buffer of size `kv` then salt across rotating buffers on device.
        const size_t k8_b = (size_t) HKV * ctx * 64 * 4;
        const size_t ks_b = (size_t) HKV * ctx * 4;
        const size_t k16_b = (size_t) HKV * ctx * 4 * 2;
        const size_t v8_b = xa::v8t_dwords(HKV, ctxp) * 4;
        const size_t vs_b = (size_t) HKV * ctxp * 4;
        std::vector<uint32_t> rk8(k8_b / 4), rv8(v8_b / 4);
        std::vector<float> rks(ks_b / 4, 0.01f), rvs(vs_b / 4, 0.01f);
        std::vector<uint16_t> rk16(k16_b / 2, 0x211f);
        for (size_t i = 0; i < rk8.size(); ++i) rk8[i] = (uint32_t) (xr() & 0x7f7f7f7full);
        for (size_t i = 0; i < rv8.size(); ++i) rv8[i] = (uint32_t) (xr() & 0x7f7f7f7full);
        for (int b = 0; b < n_bufs; ++b) {
            bufs[b].k8 = (uint32_t*) dev(k8_b);
            bufs[b].ks = (ns == 1) ? (float*) dev(ks_b) : nullptr;
            bufs[b].ks16 = (ns == 4) ? (uint16_t*) dev(k16_b) : nullptr;
            bufs[b].v8t = (uint32_t*) dev(v8_b);
            bufs[b].vs = (float*) dev(vs_b);
            q.memcpy(bufs[b].k8, rk8.data(), k8_b);
            if (ns == 1) q.memcpy(bufs[b].ks, rks.data(), ks_b);
            else q.memcpy(bufs[b].ks16, rk16.data(), k16_b);
            q.memcpy(bufs[b].v8t, rv8.data(), v8_b);
            q.memcpy(bufs[b].vs, rvs.data(), vs_b);
        }
        q.wait();

        const double logical_bytes = (double) HKV * (double) cells * (HD * 2.0 + (ns == 4 ? 8.0 : 4.0) + 4.0);
        const double pool_mb = (double) n_bufs * (double) kv_bytes_per_buf / 1048576.0;
        std::printf("\nxatt2 bench ctx=%d (logical %.1f MB/layer, rotating %d bufs = %.0f MB, ns=%d):\n",
                    cells, logical_bytes / 1e6, n_bufs, pool_mb, ns);

        // XATT1 on the same buffers for the crossover (the engine's XATT1 window = these three kernels + k_attn_merge_gqa, which
        // is not timed here: XATT2 below includes its merge, so a XATT1 time within ~15 us of the XATT2 one is a tie).
        // XATT2_NO_CMP1=1 leaves it out.
        constexpr int NCHK1 = 96;
        const bool cmp1 = !sweep && !std::getenv("XATT2_NO_CMP1") && cells >= 4096;
        const int capc1 = xa::chunk_cap(ctx, NCHK1);
        float* dsc1 = nullptr; uint32_t* dpb1 = nullptr; float* dsr1 = nullptr;
        if (cmp1) {
            dsc1 = (float*) dev(xa::score_floats(HKV, NCHK1, capc1) * 4);
            dpb1 = (uint32_t*) dev(xa::pb_dwords(HKV, NCHK1, capc1) * 4);
            dsr1 = (float*) dev(xa::srow_floats(HKV, NCHK1) * 4);
        }
        double t1_us = 0.0, t16_us = 0.0;
        for (int T : tl) {
            if (T < 1 || T > TM || cells < T + 8) continue;
            const int pos0 = cells - T;
            q.memcpy(dpos, &pos0, 4).wait();

            if (!sweep) {
                const double us = time_graph_warm(q, 16, [&](int m) {
                    const KvBuf& kb = bufs[m % n_bufs];
                    xa::Call2 c{};
                    c.q = dq; c.pos0 = dpos; c.t0 = 0; c.nt = T; c.H = H; c.HKV = HKV; c.scale = 1.0f / 16.0f;
                    c.k8 = kb.k8; c.ks = kb.ks; c.ks16 = kb.ks16; c.v8t = kb.v8t; c.vs = kb.vs;
                    c.ns = ns; c.kstride = ctx; c.ctxp = ctxp; c.pos_hint = cells;
                    c.qb = dqb; c.qs = dqs; c.pacc = dpa; c.pm = dpm; c.pl = dpl;
                    c.out = dout; c.gate = dgate;
                    xa::attend2(c, &q);
                }, 0.40, 15);
                if (T == 1) t1_us = us;
                if (T == 16) t16_us = us;
                const double gbps = (logical_bytes / (us * 1e-6)) / 1e9;
                const double pct = 100.0 * gbps / 590.0;
                const double x16_ms = us * 16.0 / 1000.0;
                std::printf("  T=%2d (%2d rows/kvh): %7.1f us/layer (%5.2f ms x16) | %5.1f GB/s (%5.1f%% of 590)",
                            T, T * (H / HKV), us, x16_ms, gbps, pct);
                if (cmp1) {
                    const double us1 = time_graph_warm(q, 16, [&](int m) {
                        const KvBuf& kb = bufs[m % n_bufs];
                        xa::Call c1{};
                        c1.q = dq; c1.pos0 = dpos; c1.t0 = 0; c1.nt = T; c1.H = H; c1.HKV = HKV; c1.nchk = NCHK1; c1.scale = 1.0f / 16.0f;
                        c1.k8 = kb.k8; c1.ks = kb.ks; c1.ks16 = kb.ks16; c1.ns = ns; c1.kstride = ctx;
                        c1.v8t = kb.v8t; c1.vs = kb.vs; c1.ctxp = ctxp;
                        c1.qb = dqb; c1.qs = dqs; c1.sc = dsc1; c1.pb = dpb1; c1.srow = dsr1; c1.capc = capc1;
                        c1.pacc = dpa; c1.pm = dpm; c1.pl = dpl;
                        xa::attend(c1, &q);
                    }, 0.40, 15);
                    std::printf(" | XATT1 %7.1f us (no merge) %s", us1, us1 + 15.0 < us ? "XATT1 wins" : "XATT2");
                }
                std::printf("\n");
            } else {
                const int forms_t1[] = {0, 1};
                const int forms_tn[] = {0, 1, 2, 3};
                const int* forms = (T == 1) ? forms_t1 : forms_tn;
                const int nforms = (T == 1) ? 2 : 4;
                const int segs_list[] = {8, 16, 24, 32, 48, 64, 96, 128, 192, 248};
                const int seg_cap_t = std::min(xa::kMaxSegsHard2, xa::kMaxSegs2 * 16 / T);
                for (int fi = 0; fi < nforms; ++fi) {
                    const int fm = forms[fi];
                    for (int nseg : segs_list) {
                        if (nseg > seg_cap_t) continue;   // (more segments than a T-token call has partial slots for)
                        const double us = time_graph_warm(q, 16, [&](int m) {
                            const KvBuf& kb = bufs[m % n_bufs];
                            xa::Call2 c{};
                            c.q = dq; c.pos0 = dpos; c.t0 = 0; c.nt = T; c.H = H; c.HKV = HKV; c.scale = 1.0f / 16.0f;
                            c.k8 = kb.k8; c.ks = kb.ks; c.ks16 = kb.ks16; c.v8t = kb.v8t; c.vs = kb.vs;
                            c.ns = ns; c.kstride = ctx; c.ctxp = ctxp; c.pos_hint = cells;
                            c.qb = dqb; c.qs = dqs; c.pacc = dpa; c.pm = dpm; c.pl = dpl;
                            c.out = dout; c.gate = dgate;
                            c.form_override = fm; c.u16_override = 1; c.nseg_override = nseg;
                            xa::attend2(c, &q);
                        }, 0.40, 11);
                        const double gbps = (logical_bytes / (us * 1e-6)) / 1e9;
                        std::printf("  T=%2d form=%-5s u16=1 nseg=%2d: %7.1f us/layer | %5.1f GB/s (%5.1f%% of 590)\n",
                                    T, (T == 1 && fm == 1) ? "R8D8" : xa::xatt2_form_name(fm), nseg, us, gbps, 100.0 * gbps / 590.0);
                    }
                }
                for (int fm : {0, 3}) {
                    xa::Call2 cp{};
                    cp.nt = T; cp.H = H; cp.HKV = HKV; cp.pos_hint = cells; cp.form_override = fm; cp.u16_override = 0;
                    const int nseg_def = xa::xatt2_plan_nseg(cp, 0);
                    const double us = time_graph_warm(q, 16, [&](int m) {
                        const KvBuf& kb = bufs[m % n_bufs];
                        xa::Call2 c{};
                        c.q = dq; c.pos0 = dpos; c.t0 = 0; c.nt = T; c.H = H; c.HKV = HKV; c.scale = 1.0f / 16.0f;
                        c.k8 = kb.k8; c.ks = kb.ks; c.ks16 = kb.ks16; c.v8t = kb.v8t; c.vs = kb.vs;
                        c.ns = ns; c.kstride = ctx; c.ctxp = ctxp; c.pos_hint = cells;
                        c.qb = dqb; c.qs = dqs; c.pacc = dpa; c.pm = dpm; c.pl = dpl;
                        c.out = dout; c.gate = dgate;
                        c.form_override = fm; c.u16_override = 0; c.nseg_override = -1;
                        xa::attend2(c, &q);
                    }, 0.40, 11);
                    const double gbps = (logical_bytes / (us * 1e-6)) / 1e9;
                    std::printf("  T=%2d form=%-5s u16=0 nseg=%2d: %7.1f us/layer | %5.1f GB/s (%5.1f%% of 590)\n",
                                T, (T == 1 && fm == 3) ? "R8D8" : xa::xatt2_form_name(fm), nseg_def, us, gbps, 100.0 * gbps / 590.0);
                }
            }
            std::fflush(stdout);
        }
        if (!sweep && t1_us > 0.0 && t16_us > 0.0) {
            std::printf("  ratio T=16 / T=1: %.3fx\n", t16_us / t1_us);
        }
        if (!sweep) {
            // 4-slot batched decode (T=1 per slot -> 4 tokens in ONE launch reading 4 separate caches).
            const int ngrp = 4;
            int32_t hrc[4] = {cells - 1, cells - 1, cells - 1, cells - 1};
            q.memcpy(dpos, hrc, 16).wait();
            void* d_ktab[4];
            void* d_vtab[4];
            const void* d_mtab[16];
            void** dk_t = (void**) dev(ngrp * sizeof(void*));
            void** dv_t = (void**) dev(ngrp * sizeof(void*));
            const void** dm_t = (const void**) dev(ngrp * 4 * sizeof(void*));
            std::vector<void*> k_blobs((size_t) ngrp), v_blobs((size_t) ngrp);
            for (int i = 0; i < ngrp; ++i) {
                const KvBuf& kb = bufs[i % n_bufs];
                k_blobs[i] = dev(k8_b + (ns == 4 ? k16_b : ks_b));
                v_blobs[i] = dev(v8_b + vs_b);
                q.memcpy(k_blobs[i], rk8.data(), k8_b);
                if (ns == 4) q.memcpy((int8_t*) k_blobs[i] + k8_b, rk16.data(), k16_b);
                else q.memcpy((int8_t*) k_blobs[i] + k8_b, rks.data(), ks_b);
                q.memcpy(v_blobs[i], rv8.data(), v8_b);
                q.memcpy((int8_t*) v_blobs[i] + v8_b, rvs.data(), vs_b);
                d_ktab[i] = k_blobs[i];
                d_vtab[i] = v_blobs[i];
                d_mtab[i * 4 + 0] = kb.k8;
                d_mtab[i * 4 + 1] = (ns == 4) ? (const void*) kb.ks16 : (const void*) kb.ks;
                d_mtab[i * 4 + 2] = kb.v8t;
                d_mtab[i * 4 + 3] = kb.vs;
            }
            q.memcpy(dk_t, d_ktab, ngrp * sizeof(void*));
            q.memcpy(dv_t, d_vtab, ngrp * sizeof(void*));
            q.memcpy(dm_t, d_mtab, ngrp * 4 * sizeof(void*));
            q.wait();
            const double bus = time_graph_warm(q, 16, [&](int) {
                xa::Call2 cb{};
                cb.q = dq; cb.pos0 = dpos; cb.nt = ngrp; cb.H = H; cb.HKV = HKV; cb.scale = 1.0f / 16.0f;
                cb.ns = ns; cb.ctx = ctx; cb.kstride = ctx; cb.ctxp = ctxp;
                if (ns == 4) { cb.k_tab = dk_t; cb.v_tab = dv_t; }
                else { cb.mir_tab = dm_t; }
                cb.ngrp = ngrp;
                for (int i = 0; i < ngrp; ++i) cb.grp[i] = {i, i, 1, cells};
                cb.qb = dqb; cb.qs = dqs; cb.pacc = dpa; cb.pm = dpm; cb.pl = dpl;
                cb.out = dout; cb.gate = dgate;
                xa::attend2(cb, &q);
            }, 0.40, 15);
            const double total_b = logical_bytes * (double) ngrp;
            const double bgbps = (total_b / (bus * 1e-6)) / 1e9;
            std::printf("  batch 4x1 (4 slots @ %d): %7.1f us/layer (%5.2f ms x16) | %5.1f GB/s (%5.1f%% of 590)\n",
                        cells, bus, bus * 16.0 / 1000.0, bgbps, 100.0 * bgbps / 590.0);
            for (int i = 0; i < ngrp; ++i) { sycl::free(k_blobs[i], q); sycl::free(v_blobs[i], q); }
            sycl::free(dk_t, q); sycl::free(dv_t, q); sycl::free(dm_t, q);
        }
        if (cmp1) { sycl::free(dsc1, q); sycl::free(dpb1, q); sycl::free(dsr1, q); }
        for (int b = 0; b < n_bufs; ++b) {
            sycl::free(bufs[b].k8, q);
            if (bufs[b].ks) sycl::free(bufs[b].ks, q);
            if (bufs[b].ks16) sycl::free(bufs[b].ks16, q);
            sycl::free(bufs[b].v8t, q);
            sycl::free(bufs[b].vs, q);
        }
    }

    sycl::free(dpos, q); sycl::free(dtp0, q); sycl::free(dtmask, q);
    sycl::free(dqb, q); sycl::free(dqs, q); sycl::free(d_dbg_flag, q);
    sycl::free(dpa, q); sycl::free(dpm, q); sycl::free(dpl, q);
    sycl::free(dout, q); sycl::free(dgate, q); sycl::free(dq, q);
    return bad ? 1 : 0;
}
namespace strata::q36df { int parity_main(int argc, char** argv); }   // q36_dflash_parity.cpp: the DFlash 2 drafter's kernels

int main(int argc, char** argv) try {
    setenv("STRATA_MMVQ_KREG", "1", 1);   // the dense forms' latch (strata-q36 sets it for the 27B)
    bool selftest_flag = false;
    if (argc > 1 && std::string(argv[1]) == "--dflash") return strata::q36df::parity_main(argc - 2, argv + 2);   // [--bench]
    if (argc > 1 && std::string(argv[1]) == "--floor") return floor_bench();
    if (argc > 1 && std::string(argv[1]) == "--xatt") return xatt_bench(argc > 2 ? argv[2] : "1,5,6,16", argc > 3 ? std::atoi(argv[3]) : 25120, argc > 4 ? std::atoi(argv[4]) : 1);
    if (argc > 1 && (std::string(argv[1]) == "--xatt2" || std::string(argv[1]) == "--xatt2-sweep")) {
        const bool sweep = std::string(argv[1]) == "--xatt2-sweep";
        return xatt2_bench(sweep,
                           argc > 2 ? argv[2] : "1,2,3,4,5,6,7,8,12,16",
                           argc > 3 ? argv[3] : "25120,65536,118000,256000",
                           argc > 4 ? std::atoi(argv[4]) : 4);
    }
    if (argc > 1 && std::string(argv[1]) == "--w4") return w4_bench(argc > 2 ? argv[2] : "1,4,5,6,8", argc > 3 ? argv[3] : "27");
    if (argc > 1 && std::string(argv[1]) == "--w4x") {
        const std::string mode = argc > 2 ? argv[2] : "all";
        const char* ncols_s    = argc > 3 ? argv[3] : "1,8,16,32";
        const std::string ssel = argc > 4 ? argv[4] : "27";
        const int ks_ovr       = argc > 5 ? std::atoi(argv[5]) : 0;
        return w4x_bench(mode, ncols_s, ssel, ks_ovr);
    }
    if (argc > 1 && (std::string(argv[1]) == "--pp" || std::string(argv[1]) == "--pp-bench")) {
        const bool is_pp_bench = std::string(argv[1]) == "--pp-bench";
        int idx = 2;
        bool bench_flag = is_pp_bench;
        if (idx < argc && std::string(argv[idx]) == "--bench") { bench_flag = true; ++idx; }
        const char* types_s = idx < argc ? argv[idx++] : "1004,12,13,23,14,11,20,8";
        const char* ncols_s = idx < argc ? argv[idx++] : "256,1024,2048";
        const std::string ssel = idx < argc ? argv[idx++] : "27";
        return pp_bench(types_s, ncols_s, ssel, bench_flag);
    }
    if (argc > 1 && std::string(argv[1]) == "--bench")
        return bench(argc > 2 ? argv[2] : "12,13,14,23", argc > 3 ? argv[3] : "1,2,3,4,6", argc > 4 ? argv[4] : "all");
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest_flag = true;
        else { std::fprintf(stderr, "usage: q36_mmvq_dense_parity [--selftest] | --bench [types] [ncols] [shape]\n"); return 2; }
    }
    const int bad = selftest();
    if (bad) return 1;
    if (selftest_flag) std::printf("q36_mmvq_dense_parity OK\n");
    return 0;
}
catch (std::exception const& exc) {   // sycl::exception included
    std::fprintf(stderr, "q36_mmvq_dense_parity: %s\n", exc.what());
    return 1;
}
