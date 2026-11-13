// q36_xr_dense.dp.cpp - the dense Qwen3.8-27B's decode products on the Arc Pro B70's matrix engine (XMX, ESIMD).  strata-q36 only.
//
// The KREG forms (q36_mmvq_dense.dp.cpp) give a lane 32 weights of ONE row and keep every column's activations in registers
// per lane (12 registers a column): 4 columns fit, the 5th spills, a 17408-wide row at 5+ columns is read twice.  Here a
// hardware thread owns 16 output ROWS and walks K in steps of 32 weights: it decodes the 16 rows' 32 weights into the B
// operand of one s8 DPAS 8x16x32, whose A operand is the same 32 activations of up to 8 COLUMNS (256 bytes, the same for
// every lane, one block load).  One instruction is the integer dot of 16 rows x 8 columns; a column more costs only its
// float scaling.  (The Flash-Next prompt path does the same for the expert matrices: src/prefill/esimd_moe.dp.cpp.)
//
//   - Arithmetic.  Per row and 32 weights: y += (d x scale) x (dx x dot) - (dmin x min) x sx, with dx the activations'
//     scale of that column and 32, sx = dx x their sum.  The int32 dot is exact.  The float sums run in K order, each column
//     on its own: a T-column call is bitwise T one-column calls.  Another summation order than KREG.
//   - Activations.  quantize() writes them once per input: [32-block][column 0..7][32 int8], and per 256 of K the 8 x 8
//     scales and 8 x 8 scaled sums.  q8_1 as llama.cpp (d = amax / 127, round half away from zero), d kept as a float.
//   - Shapes.  n_in a multiple of 256, n_out a multiple of 16, the matrix under 4 GiB (32-bit byte offsets).
//   - ESIMD on this compiler: constant indices only, no division by a run-time value (esimd_moe.dp.cpp's notes).
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "strata/sycl_queue.hpp"
#include "q36_xr_dense.hpp"

namespace strata::q36xr {
namespace {
using namespace sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

constexpr int T_Q8_0 = 8, T_Q3K = 11, T_Q4K = 12, T_Q5K = 13, T_Q6K = 14, T_IQ4NL = 20, T_IQ3S = 21, T_IQ4XS = 23;
template <int F> constexpr uint32_t blk_bytes() {
    if constexpr (F == T_Q4K || F == T_IQ4NL) return 144u;
    else if constexpr (F == T_Q5K) return 176u;
    else if constexpr (F == T_IQ4XS) return 136u;
    else if constexpr (F == T_Q6K) return 210u;
    else if constexpr (F == T_Q8_0) return 272u;
    else return 110u;
}
template <int F> constexpr uint32_t qs_off() { return F == T_Q4K ? 16u : F == T_Q5K ? 48u : 8u; }
constexpr int8_t kIq4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
// IQ3_S 9-bit codebook (ggml-common.h iq3s_grid[512]): 4 unsigned odd magnitudes per uint32_t; signs come from signs[32].
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
// IQ4_XS: a qs byte -> the codebook value of its low nibble in byte 0 and of its high nibble in byte 2 (256 dwords; a
// work-group copies them into shared local memory: a lookup in a table in ordinary memory ran the form at 190 GB/s)
const uint32_t* iq4_table(sycl::queue& q) {
    static const uint32_t* t = [&] {
        std::vector<uint32_t> h(256);
        for (uint32_t i = 0; i < 256; ++i) h[i] = (uint32_t) (uint8_t) kIq4[i & 15] | ((uint32_t) (uint8_t) kIq4[i >> 4] << 16);
        uint32_t* d = sycl::malloc_device<uint32_t>(256, q);
        q.memcpy(d, h.data(), 256 * 4).wait();
        return d;
    }();
    return t;
}
const uint32_t* iq3s_table(sycl::queue& q) {
    static const uint32_t* t = [&] {
        uint32_t* d = sycl::malloc_device<uint32_t>(512, q);
        q.memcpy(d, kIq3sGrid, 512 * sizeof(uint32_t)).wait();
        return d;
    }();
    return t;
}

SYCL_ESIMD_FUNCTION inline simd<float, 16> half_of(simd<uint32_t, 16> h16) {
    simd<uint16_t, 16> lo = h16.template bit_cast_view<uint16_t>().template select<16, 2>(0);
    simd<sycl::half, 16> hh = lo.template bit_cast_view<sycl::half>();
    simd<float, 16> f = hh;
    return f;
}
// 8 dwords a lane at byte offset `off` (dword aligned); element d of lane i at [16 * d + i].  STRATA_Q36_XR_VS (a build
// macro, 8): the dwords one gather message carries per lane.  8: one message (2: four).
#ifndef STRATA_Q36_XR_VS
#define STRATA_Q36_XR_VS 8
#endif
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 128> ld8(const uint32_t* base, simd<uint32_t, 16> off) {
#if STRATA_Q36_XR_VS == 8
    return gather<uint32_t, 128, 8>(base, off);
#else
    simd<uint32_t, 128> r;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        simd<uint32_t, 32> v = gather<uint32_t, 32, 2>(base, off + (uint32_t) (8 * g));
        r.template select<16, 1>(16 * (2 * g)) = v.template select<16, 1>(0);
        r.template select<16, 1>(16 * (2 * g + 1)) = v.template select<16, 1>(16);
    }
    return r;
#endif
}
// 4 dwords a lane (a K-quant block's header); element d of lane i at [16 * d + i]
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 64> ld4(const uint32_t* base, simd<uint32_t, 16> off) {
#if STRATA_Q36_XR_VS == 8
    return gather<uint32_t, 64, 4>(base, off);
#else
    simd<uint32_t, 64> r;
    simd<uint32_t, 32> a = gather<uint32_t, 32, 2>(base, off), b = gather<uint32_t, 32, 2>(base, off + 8u);
    r.template select<32, 1>(0) = a;
    r.template select<32, 1>(32) = b;
    return r;
#endif
}
// A block's raw words of 16 rows: the header (IQ4_XS: its 2 dwords), Q5_K's qh, the four 32-byte qs groups.
struct Raw {
    simd<uint32_t, 64> h;
    simd<uint32_t, 128> qh, q0, q1, q2, q3;
};
template <int F> SYCL_ESIMD_FUNCTION inline void load_raw(Raw& r, const uint32_t* Wd, simd<uint32_t, 16> po) {
    if constexpr (F == T_IQ4XS) {
        simd<uint32_t, 32> h = gather<uint32_t, 32, 2>(Wd, po);   // d, scales_h | scales_l
        r.h.template select<32, 1>(0) = h;
    } else {
        r.h = ld4(Wd, po);
        if constexpr (F == T_Q5K) r.qh = ld8(Wd, po + 16u);
    }
    r.q0 = ld8(Wd, po + qs_off<F>());
    r.q1 = ld8(Wd, po + (qs_off<F>() + 32u));
    r.q2 = ld8(Wd, po + (qs_off<F>() + 64u));
    r.q3 = ld8(Wd, po + (qs_off<F>() + 96u));
}
struct RawQ6 {
    simd<uint32_t, 128> q0, q1, q2, q3, qh0, qh1;
    simd<uint32_t, 64> sc;
    simd<uint32_t, 16> hd;
};
struct RawQ3 {
    simd<uint32_t, 128> h0, q0, q1;
    simd<uint32_t, 64> sc;
};
struct RawQ8 {
    simd<uint32_t, 128> g[8];
    simd<uint32_t, 64> tail;
};
template <int F> struct RawSel { using type = Raw; };
template <> struct RawSel<T_Q6K> { using type = RawQ6; };
template <> struct RawSel<T_Q3K> { using type = RawQ3; };
template <> struct RawSel<T_IQ3S> { using type = RawQ3; };
template <> struct RawSel<T_Q8_0> { using type = RawQ8; };
template <int DW> SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> q8_dw(RawQ8& r) {
    if constexpr (DW < 64) return r.g[DW >> 3].template select<16, 1>(16 * (DW & 7));
    else return r.tail.template select<16, 1>(16 * (DW - 64));
}
template <int DW> SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> q3_dw(RawQ3& r) {
    if constexpr (DW < 8) return r.h0.template select<16, 1>(16 * DW);
    else if constexpr (DW < 16) return r.q0.template select<16, 1>(16 * (DW - 8));
    else if constexpr (DW < 24) return r.q1.template select<16, 1>(16 * (DW - 16));
    else return r.sc.template select<16, 1>(16 * (DW - 24));
}
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> join16(simd<uint32_t, 16> a, simd<uint32_t, 16> b) {
    // Use 16-bit register regioning so LLVM does not lower (a >> 16) | (b << 16) to llvm.fshl,
    // which emits an unlowerable helper function in VC SPIR-V under -fsycl-device-code-split=per_kernel.
    simd<uint32_t, 16> out;
    auto o16 = out.template bit_cast_view<uint16_t>();
    o16.template select<16, 2>(0) = a.template bit_cast_view<uint16_t>().template select<16, 2>(1);
    o16.template select<16, 2>(1) = b.template bit_cast_view<uint16_t>().template select<16, 2>(0);
    return out;
}
template <int D> SYCL_ESIMD_FUNCTION inline void sh16(simd<uint32_t, 16 * D>& v, simd<uint32_t, 16> nxt0) {
#pragma unroll
    for (int d = 0; d < D - 1; ++d)
        v.template select<16, 1>(16 * d) = join16(v.template select<16, 1>(16 * d),
                                                  v.template select<16, 1>(16 * (d + 1)));
    v.template select<16, 1>(16 * (D - 1)) = join16(v.template select<16, 1>(16 * (D - 1)), nxt0);
}
template <int F, class R>
SYCL_ESIMD_FUNCTION inline void load_raw_blk(R& r, const uint32_t* Wd, simd<uint32_t, 16> po, int kb) {
    if constexpr (F == T_Q6K) {
        if ((kb & 1) == 0) {
            r.q0 = ld8(Wd, po);
            r.q1 = ld8(Wd, po + 32u);
            r.q2 = ld8(Wd, po + 64u);
            r.q3 = ld8(Wd, po + 96u);
            r.qh0 = ld8(Wd, po + 128u);
            r.qh1 = ld8(Wd, po + 160u);
            r.sc = ld4(Wd, po + 192u);
            r.hd = gather<uint32_t, 16>(Wd, po + 208u);
        } else {
            simd<uint32_t, 16> p = po - 2u;
            r.q0 = ld8(Wd, p);
            r.q1 = ld8(Wd, p + 32u);
            r.q2 = ld8(Wd, p + 64u);
            r.q3 = ld8(Wd, p + 96u);
            r.qh0 = ld8(Wd, p + 128u);
            r.qh1 = ld8(Wd, p + 160u);
            r.sc = ld4(Wd, p + 192u);
            simd<uint32_t, 16> tail = gather<uint32_t, 16>(Wd, p + 208u);
            sh16<8>(r.q0, r.q1.template select<16, 1>(0));
            sh16<8>(r.q1, r.q2.template select<16, 1>(0));
            sh16<8>(r.q2, r.q3.template select<16, 1>(0));
            sh16<8>(r.q3, r.qh0.template select<16, 1>(0));
            sh16<8>(r.qh0, r.qh1.template select<16, 1>(0));
            sh16<8>(r.qh1, r.sc.template select<16, 1>(0));
            sh16<4>(r.sc, tail);
            r.hd = tail >> 16;
        }
    } else if constexpr (F == T_Q3K || F == T_IQ3S) {
        if ((kb & 1) == 0) {
            r.h0 = ld8(Wd, po);
            r.q0 = ld8(Wd, po + 32u);
            r.q1 = ld8(Wd, po + 64u);
            r.sc = ld4(Wd, po + 96u);
        } else {
            simd<uint32_t, 16> p = po - 2u;
            r.h0 = ld8(Wd, p);
            r.q0 = ld8(Wd, p + 32u);
            r.q1 = ld8(Wd, p + 64u);
            r.sc = ld4(Wd, p + 96u);
            sh16<8>(r.h0, r.q0.template select<16, 1>(0));
            sh16<8>(r.q0, r.q1.template select<16, 1>(0));
            sh16<8>(r.q1, r.sc.template select<16, 1>(0));
            sh16<4>(r.sc, simd<uint32_t, 16>(0u));
        }
    } else if constexpr (F == T_Q8_0) {
        (void) kb;
#pragma unroll
        for (int k = 0; k < 8; ++k) r.g[k] = ld8(Wd, po + (uint32_t) (32 * k));
        r.tail = ld4(Wd, po + 256u);
    } else if constexpr (F == T_IQ4NL) {
        (void) kb;
        r.h = ld4(Wd, po);
        r.q0 = ld8(Wd, po + 16u);
        r.q1 = ld8(Wd, po + 48u);
        r.q2 = ld8(Wd, po + 80u);
        r.q3 = ld8(Wd, po + 112u);
    } else {
        (void) kb;
        load_raw<F>(r, Wd, po);
    }
}
// ggml get_scale_min_k4 for sub-block J of a K-quant block: the 12 scale bytes are dwords h1, h2, h3 of its header
template <int K> SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> sbyte(simd<uint32_t, 16> h1, simd<uint32_t, 16> h2, simd<uint32_t, 16> h3) {
    if constexpr (K < 4) return (h1 >> (8 * (K & 3))) & 0xffu;
    else if constexpr (K < 8) return (h2 >> (8 * (K & 3))) & 0xffu;
    else return (h3 >> (8 * (K & 3))) & 0xffu;
}
template <int J> SYCL_ESIMD_FUNCTION inline void scale_min(simd<uint32_t, 16> h1, simd<uint32_t, 16> h2, simd<uint32_t, 16> h3,
                                                         simd<float, 16> d, simd<float, 16> dmin, simd<float, 16>& dl, simd<float, 16>& ml) {
    simd<uint32_t, 16> sc, m;
    if constexpr (J < 4) {
        sc = sbyte<J>(h1, h2, h3) & 63u;
        m = sbyte<J + 4>(h1, h2, h3) & 63u;
    } else {
        sc = (sbyte<J + 4>(h1, h2, h3) & 0xfu) | ((sbyte<J - 4>(h1, h2, h3) >> 6) << 4);
        m = (sbyte<J + 4>(h1, h2, h3) >> 4) | ((sbyte<J>(h1, h2, h3) >> 6) << 4);
    }
    dl = d * simd<float, 16>(simd<int, 16>(sc));
    ml = dmin * simd<float, 16>(simd<int, 16>(m));
}
// sub-block S of a block: its 32 weights of 16 rows (B operand) x the 8 columns' 32 activations, scaled into acc.
// Q81 = false: `xi` is the block's [8 sub-blocks][8 columns][8 dwords] of quantize(), dxv / sxv its scales and scaled sums.
// Q81 = true: `xi` is column 0's q8_1 blocks of this 256 (9 dwords each: d, s | 32 int8), column r is xs9 dwords further:
// one 16-dword load a column and sub-block (it reads 28 bytes past the block: the buffer has that slack), the sums by a
// second DPAS against ones.
// The q8_1 blocks of one 256 of K for the columns of a call: column r's 72 dwords as A[r] (dwords 0 .. 63: sub-blocks 0 .. 6
// and the first dword of 7) and B[r] (dwords 63 .. 78: sub-block 7, then 28 bytes past it).  Two loads a column and block.
struct Acts {
    simd<int, 64> A[8];
    simd<int, 16> B[8];
};
template <int NC> SYCL_ESIMD_FUNCTION inline void load_acts(Acts& a, const int* xi, int xs9) {
#pragma unroll
    for (int r = 0; r < NC; ++r) {
        a.A[r] = block_load<int, 64>(xi + (size_t) r * xs9);
        a.B[r] = block_load<int, 16>(xi + (size_t) r * xs9 + 63);
    }
}
template <int NC, int S, bool MIN, bool Q81>
SYCL_ESIMD_FUNCTION inline void sub_block(simd<float, 128>& acc, simd<uint32_t, 128> fr, const int* xi, Acts& ac,
                                          simd<float, 16> dl, simd<float, 16> ml,
                                          const simd<float, 64>& dxv, const simd<float, 64>& sxv) {
    simd<int, 128> bi = fr.template bit_cast_view<int>();
    if constexpr (Q81) {
        simd<int, 64> av = 0;
        simd<uint32_t, 16> hd = 0u;
#pragma unroll
        for (int r = 0; r < NC; ++r) {
            if constexpr (S < 7) {
                av.template select<8, 1>(8 * r) = ac.A[r].template select<8, 1>(9 * S + 1);
                hd[r] = (uint32_t) (int) ac.A[r][9 * S];
            } else {
                av.template select<8, 1>(8 * r) = ac.B[r].template select<8, 1>(1);
                hd[r] = (uint32_t) (int) ac.B[r][0];
            }
        }
        simd<float, 16> dxf = half_of(hd & 0xffffu);
        simd<int, 128> tf = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(simd<int, 128>(0), bi, av);
        simd<int, 128> ts = 0;
        if constexpr (MIN)
            ts = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(simd<int, 128>(0), simd<int, 128>(0x01010101), av);
#pragma unroll
        for (int r = 0; r < NC; ++r) {
            const float dxr = dxf[r];
            simd<int, 16> ti = tf.template select<16, 1>(16 * r);
            simd<float, 16> t = ti;
            simd<float, 16> a = acc.template select<16, 1>(16 * r);
            if constexpr (MIN) {
                simd<int, 16> si = ts.template select<16, 1>(16 * r);
                simd<float, 16> sf = si;
                a += t * (dl * dxr) - ml * (sf * dxr);
            } else {
                a += t * (dl * dxr);
            }
            acc.template select<16, 1>(16 * r) = a;
        }
    } else {
        simd<int, 64> av = block_load<int, 64>(xi + 64 * S);
        simd<int, 128> tf = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(simd<int, 128>(0), bi, av);
#pragma unroll
        for (int r = 0; r < NC; ++r) {
            const float dxr = dxv[8 * S + r];
            simd<int, 16> ti = tf.template select<16, 1>(16 * r);
            simd<float, 16> t = ti;
            simd<float, 16> a = acc.template select<16, 1>(16 * r);
            if constexpr (MIN) {
                const float sxr = sxv[8 * S + r];
                a += t * (dl * dxr) - ml * sxr;
            } else {
                a += t * (dl * dxr);
            }
            acc.template select<16, 1>(16 * r) = a;
        }
    }
}
// the pair P of sub-blocks (2P: the low nibbles of its 32 qs bytes `qd`, 2P + 1: the high ones)
template <int F, int NC, int P, bool Q81>
SYCL_ESIMD_FUNCTION inline void pair(simd<float, 128>& acc, const simd<uint32_t, 128>& qd,
                                     const simd<uint32_t, 128>& qh, simd<uint32_t, 16> h1, simd<uint32_t, 16> h2,
                                     simd<uint32_t, 16> h3, simd<float, 16> d, simd<float, 16> dmin,
                                     const int* xi, Acts& ac, const simd<float, 64>& dxv, const simd<float, 64>& sxv) {
    simd<uint32_t, 128> lo = qd & 0x0f0f0f0fu, hi = (qd >> 4) & 0x0f0f0f0fu;
    if constexpr (F == T_Q5K) {   // the fifth bit: bit 2P / 2P + 1 of the qh byte of the same position
        lo |= ((qh >> (2 * P)) & 0x01010101u) << 4;
        hi |= ((qh >> (2 * P + 1)) & 0x01010101u) << 4;
    }
    simd<float, 16> dl, ml;
    scale_min<2 * P>(h1, h2, h3, d, dmin, dl, ml);
    sub_block<NC, 2 * P, true, Q81>(acc, lo, xi, ac, dl, ml, dxv, sxv);
    scale_min<2 * P + 1>(h1, h2, h3, d, dmin, dl, ml);
    sub_block<NC, 2 * P + 1, true, Q81>(acc, hi, xi, ac, dl, ml, dxv, sxv);
}

// IQ4_XS sub-block S: its 16 qs bytes are dwords 4 (S & 1) .. + 3 of `qq` (the 32 bytes of sub-blocks S and S ^ 1); a
// byte's low nibble is value j of the 32, its high nibble value j + 16.  The table is at offset 0 of shared local memory.
// iq4_fr: the 16 lookups of a sub-block (the B operand); sub_iq4: its scale and its product.  A block issues sub-block
// S + 1's lookups before it computes S, so their latency runs under S's arithmetic.
constexpr auto kPf = sycl::ext::oneapi::experimental::properties{cache_hint_L1<cache_hint::cached>, cache_hint_L2<cache_hint::cached>};
constexpr auto kAl16 = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::alignment<16>};
template <int F>
SYCL_ESIMD_FUNCTION inline void prefetch_next_blk(const uint32_t* Wd, simd<uint32_t, 16> ro, simd<uint32_t, 16> po, int kb) {
    if constexpr (F == T_Q6K || F == T_Q3K || F == T_IQ3S) {
        simd<uint32_t, 16> pn = ro + (uint32_t) (kb + 1) * blk_bytes<F>();
        if ((kb + 1) & 1) pn -= 2u;
        prefetch<uint32_t, 16>(Wd, pn, kPf);
        prefetch<uint32_t, 16>(Wd, pn + 64u, kPf);
        if constexpr (F == T_Q6K) {
            prefetch<uint32_t, 16>(Wd, pn + 128u, kPf);
            prefetch<uint32_t, 16>(Wd, pn + 208u, kPf);
        } else {
            prefetch<uint32_t, 16>(Wd, pn + 108u, kPf);
        }
    } else if constexpr (F == T_Q8_0) {
#pragma unroll
        for (int cl = 0; cl < 4; ++cl) prefetch<uint32_t, 16>(Wd, po + (uint32_t) (272 + 64 * cl), kPf);
        prefetch<uint32_t, 16>(Wd, po + 540u, kPf);
    } else {
#pragma unroll
        for (int cl = 0; cl < 3; ++cl) prefetch<uint32_t, 16>(Wd, po + (uint32_t) (blk_bytes<F>() + 64 * cl), kPf);
        prefetch<uint32_t, 16>(Wd, po + (uint32_t) (2 * blk_bytes<F>() - 4), kPf);
    }
}
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 128> iq4_pack(const simd<uint32_t, 16> t[16]) {
    simd<uint32_t, 128> fr;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        simd<uint32_t, 16> P = t[4 * k] | (t[4 * k + 1] << 8);
        simd<uint32_t, 16> Q = t[4 * k + 2] | (t[4 * k + 3] << 8);
        fr.template select<16, 1>(16 * k) = (P & 0xffffu) | (Q << 16);
        fr.template select<16, 1>(16 * (k + 4)) = (P >> 16) | (Q & 0xffff0000u);
    }
    return fr;
}
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 128> iq4_fr4(simd<uint32_t, 64> qq) {
    simd<uint32_t, 128> fr;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        simd<uint32_t, 16> v = qq.template select<16, 1>(16 * k);
        simd<uint32_t, 16> t0 = slm_gather<uint32_t, 16>((v & 0xffu) << 2), t1 = slm_gather<uint32_t, 16>(((v >> 8) & 0xffu) << 2);
        simd<uint32_t, 16> t2 = slm_gather<uint32_t, 16>(((v >> 16) & 0xffu) << 2), t3 = slm_gather<uint32_t, 16>((v >> 24) << 2);
        simd<uint32_t, 16> P = t0 | (t1 << 8), Q = t2 | (t3 << 8);   // [lo0, lo1, hi0, hi1], [lo2, lo3, hi2, hi3]
        fr.template select<16, 1>(16 * k) = (P & 0xffffu) | (Q << 16);
        fr.template select<16, 1>(16 * (k + 4)) = (P >> 16) | (Q & 0xffff0000u);
    }
    return fr;
}
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 128> iq4_slm_pre(simd<uint32_t, 64> qq) {
    simd<uint32_t, 16> t[16];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        simd<uint32_t, 16> v = qq.template select<16, 1>(16 * k);
        t[4 * k + 0] = slm_gather<uint32_t, 16>((v & 0xffu) << 2);
        t[4 * k + 1] = slm_gather<uint32_t, 16>(((v >> 8) & 0xffu) << 2);
        t[4 * k + 2] = slm_gather<uint32_t, 16>(((v >> 16) & 0xffu) << 2);
        t[4 * k + 3] = slm_gather<uint32_t, 16>((v >> 24) << 2);
    }
    return iq4_pack(t);
}
template <int J> constexpr uint16_t iq4_p16() {
    return (uint16_t) ((uint8_t) kIq4[2 * J] | ((uint16_t) (uint8_t) kIq4[2 * J + 1] << 8));
}
SYCL_ESIMD_FUNCTION inline simd<uint16_t, 32> iq4_lut32_raw(simd<uint16_t, 32> n) {
    auto b1 = (n & (uint16_t) 2) != (uint16_t) 0;
    auto b2 = (n & (uint16_t) 4) != (uint16_t) 0;
    auto b3 = (n & (uint16_t) 8) != (uint16_t) 0;
    simd<uint16_t, 32> a0(iq4_p16<0>()), a1(iq4_p16<2>()), a2(iq4_p16<4>()), a3(iq4_p16<6>());
    a0.merge(iq4_p16<1>(), b1);
    a1.merge(iq4_p16<3>(), b1);
    a2.merge(iq4_p16<5>(), b1);
    a3.merge(iq4_p16<7>(), b1);
    a0.merge(a1, b2);
    a2.merge(a3, b2);
    a0.merge(a2, b3);
    return a0 >> ((n & (uint16_t) 1) << 3);
}
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 128> iq4_w32_v2(simd<uint32_t, 64> qq) {
    simd<uint32_t, 128> fr;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        simd<uint16_t, 32> v16 = qq.template select<16, 1>(16 * k).template bit_cast_view<uint16_t>();
        simd<uint16_t, 32> r_lo = (iq4_lut32_raw(v16 & (uint16_t) 0xfu) & (uint16_t) 0xffu) |
                                  (iq4_lut32_raw((v16 >> 8) & (uint16_t) 0xfu) << 8);
        simd<uint16_t, 32> r_hi = (iq4_lut32_raw((v16 >> 4) & (uint16_t) 0xfu) & (uint16_t) 0xffu) |
                                  (iq4_lut32_raw(v16 >> 12) << 8);
        fr.template select<16, 1>(16 * k) = r_lo.template bit_cast_view<uint32_t>();
        fr.template select<16, 1>(16 * (k + 4)) = r_hi.template bit_cast_view<uint32_t>();
    }
    return fr;
}
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 128> iq4_l1_batched(simd<uint32_t, 64> qq, const uint32_t* T) {
    simd<uint32_t, 16> t[16];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        simd<uint32_t, 16> v = qq.template select<16, 1>(16 * k);
        t[4 * k + 0] = gather<uint32_t, 16>(T, (v & 0xffu) << 2, kPf);
        t[4 * k + 1] = gather<uint32_t, 16>(T, ((v >> 8) & 0xffu) << 2, kPf);
        t[4 * k + 2] = gather<uint32_t, 16>(T, ((v >> 16) & 0xffu) << 2, kPf);
        t[4 * k + 3] = gather<uint32_t, 16>(T, (v >> 24) << 2, kPf);
    }
    return iq4_pack(t);
}
template <int DEC>
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 128> iq4_dec4(simd<uint32_t, 64> qq, const uint32_t* T) {
    if constexpr (DEC == 0) {
        (void) T;
        return iq4_fr4(qq);
    } else if constexpr (DEC == 1) {
        (void) T;
        return iq4_slm_pre(qq);
    } else if constexpr (DEC == 2) {
        (void) T;
        return iq4_w32_v2(qq);
    } else {
        return iq4_l1_batched(qq, T);
    }
}
template <int S, int DEC>
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 128> iq4_dec(simd<uint32_t, 128> qq, const uint32_t* T) {
    return iq4_dec4<DEC>(qq.template select<64, 1>(64 * (S & 1)), T);
}
template <int NC, int S, bool Q81>
SYCL_ESIMD_FUNCTION inline void sub_iq4(simd<float, 128>& acc, simd<uint32_t, 128> fr, simd<uint32_t, 16> h0, simd<uint32_t, 16> h1,
                                        simd<float, 16> d, const int* xi, Acts& ac, const simd<float, 64>& dxv) {
    simd<uint32_t, 16> ls = ((h1 >> (4 * S)) & 0xfu) | (((h0 >> (16 + 2 * S)) & 3u) << 4);
    simd<float, 16> dl = d * simd<float, 16>(simd<int, 16>(ls) - 32);
    sub_block<NC, S, false, Q81>(acc, fr, xi, ac, dl, dl, dxv, dxv);
}
template <int P, int K = 0>
SYCL_ESIMD_FUNCTION inline void q8_fill_even(RawQ8& r, simd<uint32_t, 128>& fr) {
    fr.template select<16, 1>(16 * K) = join16(q8_dw<17 * P + K>(r), q8_dw<17 * P + K + 1>(r));
    if constexpr (K + 1 < 8) q8_fill_even<P, K + 1>(r, fr);
}
template <int P, int K = 0>
SYCL_ESIMD_FUNCTION inline void q8_fill_odd(RawQ8& r, simd<uint32_t, 128>& fr) {
    fr.template select<16, 1>(16 * K) = q8_dw<17 * P + 9 + K>(r);
    if constexpr (K + 1 < 8) q8_fill_odd<P, K + 1>(r, fr);
}
template <int S>
SYCL_ESIMD_FUNCTION inline void q8_sub_weights(RawQ8& r, simd<float, 16>& dl, simd<uint32_t, 128>& fr) {
    constexpr int P = S >> 1;
    if constexpr ((S & 1) == 0) {
        dl = half_of(q8_dw<17 * P>(r) & 0xffffu);
        q8_fill_even<P>(r, fr);
    } else {
        dl = half_of(q8_dw<17 * P + 8>(r) >> 16);
        q8_fill_odd<P>(r, fr);
    }
}
template <int NC, int S, bool Q81>
SYCL_ESIMD_FUNCTION inline void sub_q8(simd<float, 128>& acc, RawQ8& r, const int* xi, Acts& ac, const simd<float, 64>& dxv) {
    simd<float, 16> dl;
    simd<uint32_t, 128> fr;
    q8_sub_weights<S>(r, dl, fr);
    sub_block<NC, S, false, Q81>(acc, fr, xi, ac, dl, dl, dxv, dxv);
}
template <int S> SYCL_ESIMD_FUNCTION inline simd<uint32_t, 128> iq3s_fr(RawQ3& r) {
    simd<uint32_t, 16> qsw0 = join16(q3_dw<2 * S>(r), q3_dw<2 * S + 1>(r));
    simd<uint32_t, 16> qsw1 = join16(q3_dw<2 * S + 1>(r), q3_dw<2 * S + 2>(r));
    simd<uint32_t, 16> qhv = (q3_dw<16 + ((2 + S) >> 2)>(r) >> (8 * ((2 + S) & 3))) & 0xffu;
    simd<uint32_t, 16> g[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const simd<uint32_t, 16>& qsw = (k < 4) ? qsw0 : qsw1;
        simd<uint32_t, 16> qb = (qsw >> (8 * (k & 3))) & 0xffu;
        simd<uint32_t, 16> idx = qb | (((qhv >> k) & 1u) << 8);
        g[k] = slm_gather<uint32_t, 16>(idx << 2);
    }
    simd<uint32_t, 16> sg = join16(q3_dw<18 + S>(r), q3_dw<19 + S>(r));
    simd<uint32_t, 128> fr;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        simd<uint32_t, 16> nib = (sg >> (4 * k)) & 0xfu;
        simd<uint32_t, 16> m = (nib * 0x00204081u) & 0x01010101u;
        fr.template select<16, 1>(16 * k) = (g[k] ^ (m * 255u)) + m;
    }
    return fr;
}
template <int S>
SYCL_ESIMD_FUNCTION inline simd<float, 16> iq3s_dl(simd<uint32_t, 16> scales_dw, simd<float, 16> d) {
    simd<uint32_t, 16> scn = (scales_dw >> (4 * S)) & 0xfu;
    return d * simd<float, 16>(simd<int, 16>(scn * 2u + 1u));
}
template <int NC, int S, bool Q81>
SYCL_ESIMD_FUNCTION inline void sub_iq3s(simd<float, 128>& acc, simd<uint32_t, 128> fr, simd<uint32_t, 16> scales_dw,
                                         simd<float, 16> d, const int* xi, Acts& ac, const simd<float, 64>& dxv) {
    simd<float, 16> dl = iq3s_dl<S>(scales_dw, d);
    sub_block<NC, S, false, Q81>(acc, fr, xi, ac, dl, dl, dxv, dxv);
}
// The schedule.  A fixed pool of work-groups (G of them, C threads each: `threads` hardware threads at most, so no second
// wave forms) walks the 16-row tiles; a tile's K is split over the C threads of its group (C = 1, 2 or 4 by the row count:
// few rows are few streams, and a stream alone does not hide the memory latency).  Thread c sums its blocks in K order,
// threads 1 .. C - 1 pass their sums through shared local memory, thread 0 adds them in thread order and stores the rows:
// ((p0 + p1) + p2) + p3, the same at every column count.  The next block's cache lines are prefetched at the top of a block
// (bench, Q4_K 5120 x 10240, 1 column: 74.6 -> 56.7 us), and all of a block's loads go out before its first decode.
// PIPE (the default; STRATA_Q36_XR_PIPE=0: off): a block's loads go out one block ahead, into registers, while the block
// before it is decoded: no prefetch messages, and the memory latency runs under the arithmetic.
template <int F, int NC, int C, bool Q81, bool PIPE, int DEC = 0> struct TagXr;
template <int F, int NC, int C, bool Q81, bool PIPE, int DEC = 0>
void launch(sycl::queue& q, const uint8_t* W, const int* Xi, const float* S, float* Y, int bpr, int n_out, int threads) {
    constexpr bool kNeedIq4Ptr = (F == T_IQ4XS || F == T_IQ4NL) && (DEC == 0 || DEC == 1 || DEC == 3);
    constexpr bool kStageTab = ((F == T_IQ4XS || F == T_IQ4NL) && (DEC == 0 || DEC == 1)) || F == T_IQ3S;
    constexpr bool kSlmSingle = (DEC != 0) || F == T_Q8_0 || F == T_IQ3S;
    constexpr uint32_t kTab = ((F == T_IQ4XS || F == T_IQ4NL) && (DEC == 0 || DEC == 1)) ? 1024u : (F == T_IQ3S ? 2048u : 0u);
    const uint32_t* T = (F == T_IQ4XS || F == T_IQ4NL) ? (kNeedIq4Ptr ? iq4_table(q) : nullptr) : (F == T_IQ3S ? iq3s_table(q) : nullptr);
    (void) T;
    constexpr uint32_t kSlmPerT = 512u;
    constexpr uint32_t kSlmBytes = kTab + (kSlmSingle ? (uint32_t) (C - 1) * kSlmPerT : (uint32_t) (2 * C) * kSlmPerT);
    const int xs9 = bpr * 8 * 9;                           // (Q81) dwords from a column's q8_1 blocks to the next column's
    const uint32_t rowb = (uint32_t) bpr * blk_bytes<F>();
    const int tiles = n_out / 16, cb = (bpr + C - 1) / C;
    const int G = std::max(1, std::min(tiles, threads / C));
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagXr<F, NC, C, Q81, PIPE, DEC>>(sycl::nd_range<2>(sycl::range<2>((size_t) G, (size_t) C), sycl::range<2>(1, (size_t) C)), props,
                                        [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
            if constexpr (C > 1 || kStageTab) slm_init<kSlmBytes>();
            const int g = (int) it.get_group(0), c = (int) it.get_local_id(1);
            if constexpr (kStageTab) {
                if (c == 0) {
                    constexpr int kChunks = (int) (kTab / 256u);
#pragma unroll
                    for (int k = 0; k < kChunks; ++k) slm_block_store<uint32_t, 64>(256 * k, block_load<uint32_t, 64>(T + 64 * k), kAl16);
                }
                barrier();
            }
            const int kb0 = c * cb, kb1 = (kb0 + cb) < bpr ? (kb0 + cb) : bpr;
            const uint32_t* Wd = reinterpret_cast<const uint32_t*>(W);
            simd<uint32_t, 16> lane(0, 1);
            uint32_t buf = kTab;
            for (int t = g; t < tiles; t += G) {
                simd<uint32_t, 16> ro = (lane + 16u * (uint32_t) t) * rowb;
                simd<float, 128> acc = 0.f;
                typename RawSel<F>::type cur, nxt;
                if constexpr (PIPE) {
                    if (kb0 < kb1) load_raw_blk<F>(cur, Wd, ro + (uint32_t) kb0 * blk_bytes<F>(), kb0);
                }
                for (int kb = kb0; kb < kb1; ++kb) {
                    simd<uint32_t, 16> po = ro + (uint32_t) kb * blk_bytes<F>();
                    if constexpr (PIPE) {
                        if (kb + 1 < kb1) load_raw_blk<F>(nxt, Wd, po + blk_bytes<F>(), kb + 1);
                    } else {
                        load_raw_blk<F>(cur, Wd, po, kb);
                        if (kb + 1 < kb1) prefetch_next_blk<F>(Wd, ro, po, kb);
                    }
                    simd<float, 64> dxv = 0.f, sxv = 0.f;
                    if constexpr (!Q81) {
                        const float* sk = S + (size_t) kb * 128;
                        dxv = block_load<float, 64>(sk);
                        if constexpr (F != T_IQ4XS && F != T_Q8_0 && F != T_IQ3S) sxv = block_load<float, 64>(sk + 64);
                    }
                    const int* xi = Q81 ? Xi + (size_t) kb * 72 : Xi + (size_t) kb * 512;
                    Acts ac;
                    if constexpr (Q81) load_acts<NC>(ac, xi, xs9);
                    if constexpr (F == T_IQ4XS) {
                        simd<uint32_t, 16> h0 = cur.h.template select<16, 1>(0), h1 = cur.h.template select<16, 1>(16);
                        simd<float, 16> d = half_of(h0 & 0xffffu);
                        simd<uint32_t, 128> fa = iq4_dec<0, DEC>(cur.q0, T), fb = iq4_dec<1, DEC>(cur.q0, T);
                        sub_iq4<NC, 0, Q81>(acc, fa, h0, h1, d, xi, ac, dxv);
                        fa = iq4_dec<2, DEC>(cur.q1, T);
                        sub_iq4<NC, 1, Q81>(acc, fb, h0, h1, d, xi, ac, dxv);
                        fb = iq4_dec<3, DEC>(cur.q1, T);
                        sub_iq4<NC, 2, Q81>(acc, fa, h0, h1, d, xi, ac, dxv);
                        fa = iq4_dec<4, DEC>(cur.q2, T);
                        sub_iq4<NC, 3, Q81>(acc, fb, h0, h1, d, xi, ac, dxv);
                        fb = iq4_dec<5, DEC>(cur.q2, T);
                        sub_iq4<NC, 4, Q81>(acc, fa, h0, h1, d, xi, ac, dxv);
                        fa = iq4_dec<6, DEC>(cur.q3, T);
                        sub_iq4<NC, 5, Q81>(acc, fb, h0, h1, d, xi, ac, dxv);
                        fb = iq4_dec<7, DEC>(cur.q3, T);
                        sub_iq4<NC, 6, Q81>(acc, fa, h0, h1, d, xi, ac, dxv);
                        sub_iq4<NC, 7, Q81>(acc, fb, h0, h1, d, xi, ac, dxv);
                    } else if constexpr (F == T_Q8_0) {
                        sub_q8<NC, 0, Q81>(acc, cur, xi, ac, dxv);
                        sub_q8<NC, 1, Q81>(acc, cur, xi, ac, dxv);
                        sub_q8<NC, 2, Q81>(acc, cur, xi, ac, dxv);
                        sub_q8<NC, 3, Q81>(acc, cur, xi, ac, dxv);
                        sub_q8<NC, 4, Q81>(acc, cur, xi, ac, dxv);
                        sub_q8<NC, 5, Q81>(acc, cur, xi, ac, dxv);
                        sub_q8<NC, 6, Q81>(acc, cur, xi, ac, dxv);
                        sub_q8<NC, 7, Q81>(acc, cur, xi, ac, dxv);
                    } else if constexpr (F == T_IQ3S) {
                        simd<float, 16> d = half_of(q3_dw<0>(cur) & 0xffffu);
                        simd<uint32_t, 16> scales_dw = join16(q3_dw<26>(cur), q3_dw<27>(cur));
                        simd<uint32_t, 128> fa = iq3s_fr<0>(cur), fb = iq3s_fr<1>(cur);
                        sub_iq3s<NC, 0, Q81>(acc, fa, scales_dw, d, xi, ac, dxv);
                        fa = iq3s_fr<2>(cur);
                        sub_iq3s<NC, 1, Q81>(acc, fb, scales_dw, d, xi, ac, dxv);
                        fb = iq3s_fr<3>(cur);
                        sub_iq3s<NC, 2, Q81>(acc, fa, scales_dw, d, xi, ac, dxv);
                        fa = iq3s_fr<4>(cur);
                        sub_iq3s<NC, 3, Q81>(acc, fb, scales_dw, d, xi, ac, dxv);
                        fb = iq3s_fr<5>(cur);
                        sub_iq3s<NC, 4, Q81>(acc, fa, scales_dw, d, xi, ac, dxv);
                        fa = iq3s_fr<6>(cur);
                        sub_iq3s<NC, 5, Q81>(acc, fb, scales_dw, d, xi, ac, dxv);
                        fb = iq3s_fr<7>(cur);
                        sub_iq3s<NC, 6, Q81>(acc, fa, scales_dw, d, xi, ac, dxv);
                        sub_iq3s<NC, 7, Q81>(acc, fb, scales_dw, d, xi, ac, dxv);
                    } else if constexpr (F == T_Q4K || F == T_Q5K) {
                        simd<uint32_t, 16> h0 = cur.h.template select<16, 1>(0), h1 = cur.h.template select<16, 1>(16);
                        simd<uint32_t, 16> h2 = cur.h.template select<16, 1>(32), h3 = cur.h.template select<16, 1>(48);
                        simd<float, 16> d = half_of(h0 & 0xffffu), dmin = half_of(h0 >> 16);
                        pair<F, NC, 0, Q81>(acc, cur.q0, cur.qh, h1, h2, h3, d, dmin, xi, ac, dxv, sxv);
                        pair<F, NC, 1, Q81>(acc, cur.q1, cur.qh, h1, h2, h3, d, dmin, xi, ac, dxv, sxv);
                        pair<F, NC, 2, Q81>(acc, cur.q2, cur.qh, h1, h2, h3, d, dmin, xi, ac, dxv, sxv);
                        pair<F, NC, 3, Q81>(acc, cur.q3, cur.qh, h1, h2, h3, d, dmin, xi, ac, dxv, sxv);
                    }
                    if constexpr (PIPE) cur = nxt;
                }
                if constexpr (C > 1) {
                    if (c != 0) {
                        const uint32_t off = buf + (uint32_t) (kSlmSingle ? (c - 1) : c) * 512u;
                        slm_block_store<float, 64>(off, acc.template select<64, 1>(0), kAl16);
                        if constexpr (NC > 4) slm_block_store<float, 64>(off + 256u, acc.template select<64, 1>(64), kAl16);
                    }
                    barrier();
                    if (c == 0) {
#pragma unroll
                        for (int cc = 1; cc < C; ++cc) {
                            const uint32_t off = buf + (uint32_t) (kSlmSingle ? (cc - 1) : cc) * 512u;
                            simd<float, 64> p0 = slm_block_load<float, 64>(off, kAl16);
                            simd<float, 64> a0 = acc.template select<64, 1>(0);
                            a0 += p0;
                            acc.template select<64, 1>(0) = a0;
                            if constexpr (NC > 4) {
                                simd<float, 64> p1 = slm_block_load<float, 64>(off + 256u, kAl16);
                                simd<float, 64> a1 = acc.template select<64, 1>(64);
                                a1 += p1;
                                acc.template select<64, 1>(64) = a1;
                            }
                        }
                    }
                    if constexpr (kSlmSingle) {
                        if (t + G < tiles) barrier();
                    } else {
                        buf = buf == kTab ? kTab + (uint32_t) (C * 512) : kTab;   // the next tile's sums go to the other half: this one is read after the barrier
                    }
                }
                if (c == 0) {
#pragma unroll
                    for (int r = 0; r < NC; ++r)
                        block_store<float, 16>(Y + (size_t) r * n_out + 16 * (size_t) t, acc.template select<16, 1>(16 * r));
                }
            }
        });
    });
}
// STRATA_Q36_XR_THREADS (1024): the pool's hardware threads.  STRATA_Q36_XR_C (1, 2, 4; unset: by the row count): threads a tile.
inline int pool_threads() {
    static const int v = [] { const char* e = std::getenv("STRATA_Q36_XR_THREADS"); const int n = e ? std::atoi(e) : 0; return n >= 16 ? n : 1024; }();
    return v;
}
// The threads a tile, by the shape alone (never by the column count: the sums' order follows it).  Bench on the B70 (Q4_K,
// one-message loads, 2026-11-12, us at 1 / 4 / 5 columns): 4 threads a tile on every row of 16384 or less (5120 x 10240:
// 57.8 / 61.4 / 61.5 at 1 thread, 54.1 / 56.4 / 57.1 at 4, KREG 55.0 / 59.0 / 69.4; 5120 x 6144: 35.6 / 39.9 / 41.1 -> 33.2 /
// 34.2 / 34.7; 5120 x 17408: 90.4 / 93.7 / 94.8 at 4); 2 on a wider row (ffn_down, 17408 x 5120: 99.1 / 103.1 / 101.4 at 2,
// 98.3 / 103.4 / 104.9 at 4).  STRATA_Q36_XR_C=1, 2 or 4: every shape.
inline int chunks_of(int n_in, int n_out) {
    static const int e = [] { const char* v = std::getenv("STRATA_Q36_XR_C"); const int n = v ? std::atoi(v) : 0; return n == 1 || n == 2 || n == 4 ? n : 0; }();
    if (e) return e;
    (void) n_out;
    return n_in > 16384 ? 2 : 4;
}
// 9-16 columns in one read of the weights: each 256-weight block loads activations in two 4-sub-block halves (36 dwords per
// column instead of 72, so activations + weights + both column-group accumulators fit in 256 GRFs with 0 spill) and decodes
// each weight sub-block once for both DPAS column groups (0..7 and 8..NC-1).
template <int NC> struct ActsHalf {
    simd<int, 36> V[NC];
};
template <int NC, int HALF>
SYCL_ESIMD_FUNCTION inline void load_acts_half(ActsHalf<NC>& a, const int* xi, int xs9) {
    constexpr int base = HALF * 36;   // 16B aligned (xi is 32B aligned; 36 dwords = 144 B; +28 dwords = +112 B)
#pragma unroll
    for (int r = 0; r < NC; ++r) {
        const int* p = xi + (size_t) r * xs9 + base;
        a.V[r].template select<32, 1>(0) = block_load<int, 32>(p);
        simd<int, 8> tail = block_load<int, 8>(p + 28);   // HALF 1 ends at dword 71 of the block (0 overrun)
        a.V[r].template select<4, 1>(32) = tail.template select<4, 1>(4);
    }
}
template <int NC, int C0, int NG, int S_HALF, bool MIN>
SYCL_ESIMD_FUNCTION inline void dpas_group(simd<float, 128>& acc, const simd<int, 128>& bi, ActsHalf<NC>& ac,
                                           simd<float, 16> dl, simd<float, 16> ml) {
    simd<int, 64> av = 0;
    simd<uint32_t, 16> hd = 0u;
#pragma unroll
    for (int r = 0; r < NG; ++r) {
        av.template select<8, 1>(8 * r) = ac.V[C0 + r].template select<8, 1>(9 * S_HALF + 1);
        hd[r] = (uint32_t) (int) ac.V[C0 + r][9 * S_HALF];
    }
    simd<float, 16> dxf = half_of(hd & 0xffffu);
    simd<int, 128> tf = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(simd<int, 128>(0), bi, av);
    simd<int, 128> ts = 0;
    if constexpr (MIN)
        ts = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(simd<int, 128>(0), simd<int, 128>(0x01010101), av);
#pragma unroll
    for (int r = 0; r < NG; ++r) {
        const float dxr = dxf[r];
        simd<float, 16> t = tf.template select<16, 1>(16 * r);
        simd<float, 16> a = acc.template select<16, 1>(16 * r);
        if constexpr (MIN) {
            simd<float, 16> sf = ts.template select<16, 1>(16 * r);
            a += t * (dl * dxr) - ml * (sf * dxr);
        } else {
            a += t * (dl * dxr);
        }
        acc.template select<16, 1>(16 * r) = a;
    }
}
template <int NC, int S_HALF, bool MIN>
SYCL_ESIMD_FUNCTION inline void sub_block_y(simd<float, 128>& acc0, simd<float, 128>& acc1,
                                            simd<uint32_t, 128> fr, ActsHalf<NC>& ac,
                                            simd<float, 16> dl, simd<float, 16> ml) {
    simd<int, 128> bi = fr.template bit_cast_view<int>();
    dpas_group<NC, 0, 8, S_HALF, MIN>(acc0, bi, ac, dl, ml);
    dpas_group<NC, 8, NC - 8, S_HALF, MIN>(acc1, bi, ac, dl, ml);
}
template <int F, int NC, int P, int HALF>
SYCL_ESIMD_FUNCTION inline void pair_y(simd<float, 128>& acc0, simd<float, 128>& acc1, const simd<uint32_t, 128>& qd,
                                       const simd<uint32_t, 128>& qh, simd<uint32_t, 16> h1, simd<uint32_t, 16> h2,
                                       simd<uint32_t, 16> h3, simd<float, 16> d, simd<float, 16> dmin, ActsHalf<NC>& ac) {
    constexpr int S_IN_HALF = (2 * P) - 4 * HALF;
    simd<float, 16> dl, ml;
    {
        simd<uint32_t, 128> lo = qd & 0x0f0f0f0fu;
        if constexpr (F == T_Q5K) lo |= ((qh >> (2 * P)) & 0x01010101u) << 4;
        scale_min<2 * P>(h1, h2, h3, d, dmin, dl, ml);
        sub_block_y<NC, S_IN_HALF, true>(acc0, acc1, lo, ac, dl, ml);
    }
    {
        simd<uint32_t, 128> hi = (qd >> 4) & 0x0f0f0f0fu;
        if constexpr (F == T_Q5K) hi |= ((qh >> (2 * P + 1)) & 0x01010101u) << 4;
        scale_min<2 * P + 1>(h1, h2, h3, d, dmin, dl, ml);
        sub_block_y<NC, S_IN_HALF + 1, true>(acc0, acc1, hi, ac, dl, ml);
    }
}
template <int NC, int S>
SYCL_ESIMD_FUNCTION inline void sub_iq4_y(simd<float, 128>& acc0, simd<float, 128>& acc1, simd<uint32_t, 128> fr,
                                          simd<uint32_t, 16> h0, simd<uint32_t, 16> h1, simd<float, 16> d, ActsHalf<NC>& ac) {
    simd<uint32_t, 16> ls = ((h1 >> (4 * S)) & 0xfu) | (((h0 >> (16 + 2 * S)) & 3u) << 4);
    simd<float, 16> dl = d * simd<float, 16>(simd<int, 16>(ls) - 32);
    sub_block_y<NC, S & 3, false>(acc0, acc1, fr, ac, dl, dl);
}
template <int NG>
SYCL_ESIMD_FUNCTION inline void slm_store_acc(uint32_t off, simd<float, 128>& acc) {
    slm_block_store<float, 64>(off, acc.template select<64, 1>(0), kAl16);
    if constexpr (NG > 4) slm_block_store<float, 64>(off + 256u, acc.template select<64, 1>(64), kAl16);
}
template <int NG>
SYCL_ESIMD_FUNCTION inline void slm_add_acc(uint32_t off, simd<float, 128>& acc) {
    simd<float, 64> p0 = slm_block_load<float, 64>(off, kAl16);
    simd<float, 64> a0 = acc.template select<64, 1>(0);
    a0 += p0;
    acc.template select<64, 1>(0) = a0;
    if constexpr (NG > 4) {
        simd<float, 64> p1 = slm_block_load<float, 64>(off + 256u, kAl16);
        simd<float, 64> a1 = acc.template select<64, 1>(64);
        a1 += p1;
        acc.template select<64, 1>(64) = a1;
    }
}

template <int DW> SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> nl_dw(Raw& r) {
    if constexpr (DW < 4) return r.h.template select<16, 1>(16 * DW);
    else if constexpr (DW < 12) return r.q0.template select<16, 1>(16 * (DW - 4));
    else if constexpr (DW < 20) return r.q1.template select<16, 1>(16 * (DW - 12));
    else if constexpr (DW < 28) return r.q2.template select<16, 1>(16 * (DW - 20));
    else return r.q3.template select<16, 1>(16 * (DW - 28));
}
template <int S> SYCL_ESIMD_FUNCTION inline simd<float, 16> nl_d(Raw& r) {
    constexpr int B = 9 * (S >> 1);
    if constexpr ((S & 1) == 0) return half_of(nl_dw<B>(r) & 0xffffu);
    else return half_of(nl_dw<B + 4>(r) >> 16);
}
template <int S, int K> SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> nl_qs_dw(Raw& r) {
    constexpr int B = 9 * (S >> 1);
    if constexpr ((S & 1) == 0) return join16(nl_dw<B + K>(r), nl_dw<B + K + 1>(r));
    else return nl_dw<B + 5 + K>(r);
}
template <int S> SYCL_ESIMD_FUNCTION inline simd<uint32_t, 64> nl_qs(Raw& r) {
    simd<uint32_t, 64> qs;
    qs.template select<16, 1>(0) = nl_qs_dw<S, 0>(r);
    qs.template select<16, 1>(16) = nl_qs_dw<S, 1>(r);
    qs.template select<16, 1>(32) = nl_qs_dw<S, 2>(r);
    qs.template select<16, 1>(48) = nl_qs_dw<S, 3>(r);
    return qs;
}
// 16-weight sub-block scales (Q6_K, Q3_K): two DPAS calls per sub-block S_HALF (half 0 in d = 0..3, half 1 in d = 4..7, zeroing the other half of av) so bi fits in one register.
template <int NC, int C0, int NG, int S_HALF>
SYCL_ESIMD_FUNCTION inline void dpas_group16(simd<float, 128>& acc, const simd<int, 128>& bi, ActsHalf<NC>& ac,
                                             simd<float, 16> dl0, simd<float, 16> dl1) {
    simd<uint32_t, 16> hd = 0u;
#pragma unroll
    for (int r = 0; r < NG; ++r) hd[r] = (uint32_t) (int) ac.V[C0 + r][9 * S_HALF];
    simd<float, 16> dxf = half_of(hd & 0xffffu);
    {
        simd<int, 64> av = 0;
#pragma unroll
        for (int r = 0; r < NG; ++r) av.template select<4, 1>(8 * r) = ac.V[C0 + r].template select<4, 1>(9 * S_HALF + 1);
        simd<int, 128> tf = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(simd<int, 128>(0), bi, av);
#pragma unroll
        for (int r = 0; r < NG; ++r) {
            const float dxr = dxf[r];
            simd<float, 16> t = tf.template select<16, 1>(16 * r);
            simd<float, 16> a = acc.template select<16, 1>(16 * r);
            a += t * (dl0 * dxr);
            acc.template select<16, 1>(16 * r) = a;
        }
    }
    {
        simd<int, 64> av = 0;
#pragma unroll
        for (int r = 0; r < NG; ++r) av.template select<4, 1>(8 * r + 4) = ac.V[C0 + r].template select<4, 1>(9 * S_HALF + 5);
        simd<int, 128> tf = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(simd<int, 128>(0), bi, av);
#pragma unroll
        for (int r = 0; r < NG; ++r) {
            const float dxr = dxf[r];
            simd<float, 16> t = tf.template select<16, 1>(16 * r);
            simd<float, 16> a = acc.template select<16, 1>(16 * r);
            a += t * (dl1 * dxr);
            acc.template select<16, 1>(16 * r) = a;
        }
    }
}
template <int NC, int S_HALF>
SYCL_ESIMD_FUNCTION inline void sub_block16_y(simd<float, 128>& acc0, simd<float, 128>& acc1, simd<uint32_t, 128> fr,
                                              ActsHalf<NC>& ac, simd<float, 16> dl0, simd<float, 16> dl1) {
    simd<int, 128> bi = fr.template bit_cast_view<int>();
    dpas_group16<NC, 0, 8, S_HALF>(acc0, bi, ac, dl0, dl1);
    dpas_group16<NC, 8, NC - 8, S_HALF>(acc1, bi, ac, dl0, dl1);
}
template <int J>
SYCL_ESIMD_FUNCTION inline simd<float, 16> q6_scale(simd<uint32_t, 64>& sc, simd<float, 16> d) {
    simd<uint32_t, 16> w = sc.template select<16, 1>(16 * (J >> 2));
    simd<uint32_t, 16> u = w << (24 - 8 * (J & 3));
    simd<int, 16> s = u.template bit_cast_view<int>() >> 24;
    return d * simd<float, 16>(s);
}
template <int NC, int S>
SYCL_ESIMD_FUNCTION inline void sub_q6_y(simd<float, 128>& acc0, simd<float, 128>& acc1, RawQ6& r,
                                         simd<float, 16> d, ActsHalf<NC>& ac) {
    const simd<uint32_t, 128>& ql = (S & 1) == 0 ? ((S < 4) ? r.q0 : r.q2) : ((S < 4) ? r.q1 : r.q3);
    const simd<uint32_t, 128>& qh = (S < 4) ? r.qh0 : r.qh1;
    simd<uint32_t, 128> lo = (S & 2) ? ((ql >> 4) & 0x0f0f0f0fu) : (ql & 0x0f0f0f0fu);
    simd<uint32_t, 128> hi = ((qh >> (2 * (S & 3))) & 0x03030303u) << 4;
    simd<uint32_t, 128> fr = ((lo | hi) + 0x60606060u) ^ 0x80808080u;   // exact q - 32 in [-32, 31], no byte carry
    sub_block16_y<NC, S & 3>(acc0, acc1, fr, ac, q6_scale<2 * S>(r.sc, d), q6_scale<2 * S + 1>(r.sc, d));
}
template <int J>
SYCL_ESIMD_FUNCTION inline simd<float, 16> q3_scale(simd<uint32_t, 64>& sc, simd<float, 16> d) {
    simd<uint32_t, 16> sw0 = sc.template select<16, 1>(0), sw1 = sc.template select<16, 1>(16), sw2 = sc.template select<16, 1>(32);
    simd<uint32_t, 16> l = ((J < 4 || (J >= 8 && J < 12) ? sw0 : sw1) >> (8 * (J & 3) + (J >= 8 ? 4 : 0))) & 0xfu;
    simd<uint32_t, 16> h = (sw2 >> (8 * (J & 3) + 2 * (J >> 2))) & 3u;
    return d * simd<float, 16>(simd<int, 16>(l | (h << 4)) - 32);
}
template <int NC, int S>
SYCL_ESIMD_FUNCTION inline void sub_q3_y(simd<float, 128>& acc0, simd<float, 128>& acc1, RawQ3& r,
                                         simd<float, 16> d, ActsHalf<NC>& ac) {
    const simd<uint32_t, 128>& qs = (S < 4) ? r.q0 : r.q1;
    simd<uint32_t, 128> lo = (qs >> (2 * (S & 3))) & 0x03030303u;
    simd<uint32_t, 128> hi = ((r.h0 >> S) & 0x01010101u) << 2;
    simd<uint32_t, 128> fr = ((lo | hi) + 0x7c7c7c7cu) ^ 0x80808080u;   // exact v - 4 in [-4, 3], no byte carry
    sub_block16_y<NC, S & 3>(acc0, acc1, fr, ac, q3_scale<2 * S>(r.sc, d), q3_scale<2 * S + 1>(r.sc, d));
}

template <int NC, int S>
SYCL_ESIMD_FUNCTION inline void sub_q8_y(simd<float, 128>& acc0, simd<float, 128>& acc1, RawQ8& r, ActsHalf<NC>& ac) {
    simd<float, 16> dl;
    simd<uint32_t, 128> fr;
    q8_sub_weights<S>(r, dl, fr);
    sub_block_y<NC, S & 3, false>(acc0, acc1, fr, ac, dl, dl);
}
template <int NC, int S>
SYCL_ESIMD_FUNCTION inline void sub_iq3s_y(simd<float, 128>& acc0, simd<float, 128>& acc1, simd<uint32_t, 128> fr,
                                           simd<uint32_t, 16> scales_dw, simd<float, 16> d, ActsHalf<NC>& ac) {
    simd<float, 16> dl = iq3s_dl<S>(scales_dw, d);
    sub_block_y<NC, S & 3, false>(acc0, acc1, fr, ac, dl, dl);
}

template <int F, int NC, int C, bool Q81, bool PIPE, int DEC = 0> struct TagXr16;
template <int F, int NC, int C, bool Q81, bool PIPE, int DEC = 0>
void launch16(sycl::queue& q, const uint8_t* W, const int* Xi, const float* S, float* Y, int bpr, int n_out, int threads) {
    static_assert(NC >= 9 && NC <= 16 && Q81 && !PIPE);
    (void) S;
    constexpr int NC1 = NC - 8;
    constexpr bool kNeedIq4Ptr = (F == T_IQ4XS || F == T_IQ4NL) && (DEC == 0 || DEC == 1 || DEC == 3);
    constexpr bool kStageTab = ((F == T_IQ4XS || F == T_IQ4NL) && (DEC == 0 || DEC == 1)) || F == T_IQ3S;
    constexpr bool kSlmSingle = (DEC != 0) || F == T_Q8_0 || F == T_IQ3S;
    constexpr uint32_t kTab = ((F == T_IQ4XS || F == T_IQ4NL) && (DEC == 0 || DEC == 1)) ? 1024u : (F == T_IQ3S ? 2048u : 0u);
    const uint32_t* T = (F == T_IQ4XS || F == T_IQ4NL) ? (kNeedIq4Ptr ? iq4_table(q) : nullptr) : (F == T_IQ3S ? iq3s_table(q) : nullptr);
    (void) T;
    constexpr uint32_t kSlmPerT = NC <= 12 ? 768u : 1024u;
    constexpr uint32_t kSlmBytes = kTab + (kSlmSingle ? (uint32_t) (C - 1) * kSlmPerT : (uint32_t) (2 * C) * kSlmPerT);
    const int xs9 = bpr * 8 * 9;
    const uint32_t rowb = (uint32_t) bpr * blk_bytes<F>();
    const int tiles = n_out / 16, cb = (bpr + C - 1) / C;
    const int G = std::max(1, std::min(tiles, threads / C));
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagXr16<F, NC, C, Q81, PIPE, DEC>>(
            sycl::nd_range<2>(sycl::range<2>((size_t) G, (size_t) C), sycl::range<2>(1, (size_t) C)), props,
            [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
            if constexpr (C > 1 || kStageTab) slm_init<kSlmBytes>();
            const int g = (int) it.get_group(0), c = (int) it.get_local_id(1);
            if constexpr (kStageTab) {
                if (c == 0) {
                    constexpr int kChunks = (int) (kTab / 256u);
#pragma unroll
                    for (int k = 0; k < kChunks; ++k) slm_block_store<uint32_t, 64>(256 * k, block_load<uint32_t, 64>(T + 64 * k), kAl16);
                }
                barrier();
            }
            const int kb0 = c * cb, kb1 = (kb0 + cb) < bpr ? (kb0 + cb) : bpr;
            const uint32_t* Wd = reinterpret_cast<const uint32_t*>(W);
            simd<uint32_t, 16> lane(0, 1);
            uint32_t buf = kTab;
            for (int t = g; t < tiles; t += G) {
                simd<uint32_t, 16> ro = (lane + 16u * (uint32_t) t) * rowb;
                simd<float, 128> acc0 = 0.f, acc1 = 0.f;
                typename RawSel<F>::type cur;
                for (int kb = kb0; kb < kb1; ++kb) {
                    simd<uint32_t, 16> po = ro + (uint32_t) kb * blk_bytes<F>();
                    load_raw_blk<F>(cur, Wd, po, kb);
                    if (kb + 1 < kb1) prefetch_next_blk<F>(Wd, ro, po, kb);
                    const int* xi0 = Xi + (size_t) kb * 72;
                    if constexpr (F == T_IQ4XS) {
                        simd<uint32_t, 16> h0 = cur.h.template select<16, 1>(0), h1 = cur.h.template select<16, 1>(16);
                        simd<float, 16> d = half_of(h0 & 0xffffu);
                        {
                            ActsHalf<NC> ac;
                            load_acts_half<NC, 0>(ac, xi0, xs9);
                            simd<uint32_t, 128> fa = iq4_dec<0, DEC>(cur.q0, T), fb = iq4_dec<1, DEC>(cur.q0, T);
                            sub_iq4_y<NC, 0>(acc0, acc1, fa, h0, h1, d, ac);
                            fa = iq4_dec<2, DEC>(cur.q1, T);
                            sub_iq4_y<NC, 1>(acc0, acc1, fb, h0, h1, d, ac);
                            fb = iq4_dec<3, DEC>(cur.q1, T);
                            sub_iq4_y<NC, 2>(acc0, acc1, fa, h0, h1, d, ac);
                            sub_iq4_y<NC, 3>(acc0, acc1, fb, h0, h1, d, ac);
                        }
                        {
                            ActsHalf<NC> ac;
                            load_acts_half<NC, 1>(ac, xi0, xs9);
                            simd<uint32_t, 128> fa = iq4_dec<4, DEC>(cur.q2, T), fb = iq4_dec<5, DEC>(cur.q2, T);
                            sub_iq4_y<NC, 4>(acc0, acc1, fa, h0, h1, d, ac);
                            fa = iq4_dec<6, DEC>(cur.q3, T);
                            sub_iq4_y<NC, 5>(acc0, acc1, fb, h0, h1, d, ac);
                            fb = iq4_dec<7, DEC>(cur.q3, T);
                            sub_iq4_y<NC, 6>(acc0, acc1, fa, h0, h1, d, ac);
                            sub_iq4_y<NC, 7>(acc0, acc1, fb, h0, h1, d, ac);
                        }
                    } else if constexpr (F == T_IQ4NL) {
                        {
                            ActsHalf<NC> ac;
                            load_acts_half<NC, 0>(ac, xi0, xs9);
                            simd<uint32_t, 128> fa = iq4_dec4<DEC>(nl_qs<0>(cur), T), fb = iq4_dec4<DEC>(nl_qs<1>(cur), T);
                            sub_block_y<NC, 0, false>(acc0, acc1, fa, ac, nl_d<0>(cur), nl_d<0>(cur));
                            fa = iq4_dec4<DEC>(nl_qs<2>(cur), T);
                            sub_block_y<NC, 1, false>(acc0, acc1, fb, ac, nl_d<1>(cur), nl_d<1>(cur));
                            fb = iq4_dec4<DEC>(nl_qs<3>(cur), T);
                            sub_block_y<NC, 2, false>(acc0, acc1, fa, ac, nl_d<2>(cur), nl_d<2>(cur));
                            sub_block_y<NC, 3, false>(acc0, acc1, fb, ac, nl_d<3>(cur), nl_d<3>(cur));
                        }
                        {
                            ActsHalf<NC> ac;
                            load_acts_half<NC, 1>(ac, xi0, xs9);
                            simd<uint32_t, 128> fa = iq4_dec4<DEC>(nl_qs<4>(cur), T), fb = iq4_dec4<DEC>(nl_qs<5>(cur), T);
                            sub_block_y<NC, 0, false>(acc0, acc1, fa, ac, nl_d<4>(cur), nl_d<4>(cur));
                            fa = iq4_dec4<DEC>(nl_qs<6>(cur), T);
                            sub_block_y<NC, 1, false>(acc0, acc1, fb, ac, nl_d<5>(cur), nl_d<5>(cur));
                            fb = iq4_dec4<DEC>(nl_qs<7>(cur), T);
                            sub_block_y<NC, 2, false>(acc0, acc1, fa, ac, nl_d<6>(cur), nl_d<6>(cur));
                            sub_block_y<NC, 3, false>(acc0, acc1, fb, ac, nl_d<7>(cur), nl_d<7>(cur));
                        }
                    } else if constexpr (F == T_Q8_0) {
                        {
                            ActsHalf<NC> ac;
                            load_acts_half<NC, 0>(ac, xi0, xs9);
                            sub_q8_y<NC, 0>(acc0, acc1, cur, ac);
                            sub_q8_y<NC, 1>(acc0, acc1, cur, ac);
                            sub_q8_y<NC, 2>(acc0, acc1, cur, ac);
                            sub_q8_y<NC, 3>(acc0, acc1, cur, ac);
                        }
                        {
                            ActsHalf<NC> ac;
                            load_acts_half<NC, 1>(ac, xi0, xs9);
                            sub_q8_y<NC, 4>(acc0, acc1, cur, ac);
                            sub_q8_y<NC, 5>(acc0, acc1, cur, ac);
                            sub_q8_y<NC, 6>(acc0, acc1, cur, ac);
                            sub_q8_y<NC, 7>(acc0, acc1, cur, ac);
                        }
                    } else if constexpr (F == T_IQ3S) {
                        simd<float, 16> d = half_of(q3_dw<0>(cur) & 0xffffu);
                        simd<uint32_t, 16> scales_dw = join16(q3_dw<26>(cur), q3_dw<27>(cur));
                        {
                            ActsHalf<NC> ac;
                            load_acts_half<NC, 0>(ac, xi0, xs9);
                            simd<uint32_t, 128> fa = iq3s_fr<0>(cur), fb = iq3s_fr<1>(cur);
                            sub_iq3s_y<NC, 0>(acc0, acc1, fa, scales_dw, d, ac);
                            fa = iq3s_fr<2>(cur);
                            sub_iq3s_y<NC, 1>(acc0, acc1, fb, scales_dw, d, ac);
                            fb = iq3s_fr<3>(cur);
                            sub_iq3s_y<NC, 2>(acc0, acc1, fa, scales_dw, d, ac);
                            sub_iq3s_y<NC, 3>(acc0, acc1, fb, scales_dw, d, ac);
                        }
                        {
                            ActsHalf<NC> ac;
                            load_acts_half<NC, 1>(ac, xi0, xs9);
                            simd<uint32_t, 128> fa = iq3s_fr<4>(cur), fb = iq3s_fr<5>(cur);
                            sub_iq3s_y<NC, 4>(acc0, acc1, fa, scales_dw, d, ac);
                            fa = iq3s_fr<6>(cur);
                            sub_iq3s_y<NC, 5>(acc0, acc1, fb, scales_dw, d, ac);
                            fb = iq3s_fr<7>(cur);
                            sub_iq3s_y<NC, 6>(acc0, acc1, fa, scales_dw, d, ac);
                            sub_iq3s_y<NC, 7>(acc0, acc1, fb, scales_dw, d, ac);
                        }
                    } else if constexpr (F == T_Q6K) {
                        simd<float, 16> d = half_of(cur.hd & 0xffffu);
                        {
                            ActsHalf<NC> ac;
                            load_acts_half<NC, 0>(ac, xi0, xs9);
                            sub_q6_y<NC, 0>(acc0, acc1, cur, d, ac);
                            sub_q6_y<NC, 1>(acc0, acc1, cur, d, ac);
                            sub_q6_y<NC, 2>(acc0, acc1, cur, d, ac);
                            sub_q6_y<NC, 3>(acc0, acc1, cur, d, ac);
                        }
                        {
                            ActsHalf<NC> ac;
                            load_acts_half<NC, 1>(ac, xi0, xs9);
                            sub_q6_y<NC, 4>(acc0, acc1, cur, d, ac);
                            sub_q6_y<NC, 5>(acc0, acc1, cur, d, ac);
                            sub_q6_y<NC, 6>(acc0, acc1, cur, d, ac);
                            sub_q6_y<NC, 7>(acc0, acc1, cur, d, ac);
                        }
                    } else if constexpr (F == T_Q3K) {
                        simd<float, 16> d = half_of(cur.sc.template select<16, 1>(48) & 0xffffu);
                        {
                            ActsHalf<NC> ac;
                            load_acts_half<NC, 0>(ac, xi0, xs9);
                            sub_q3_y<NC, 0>(acc0, acc1, cur, d, ac);
                            sub_q3_y<NC, 1>(acc0, acc1, cur, d, ac);
                            sub_q3_y<NC, 2>(acc0, acc1, cur, d, ac);
                            sub_q3_y<NC, 3>(acc0, acc1, cur, d, ac);
                        }
                        {
                            ActsHalf<NC> ac;
                            load_acts_half<NC, 1>(ac, xi0, xs9);
                            sub_q3_y<NC, 4>(acc0, acc1, cur, d, ac);
                            sub_q3_y<NC, 5>(acc0, acc1, cur, d, ac);
                            sub_q3_y<NC, 6>(acc0, acc1, cur, d, ac);
                            sub_q3_y<NC, 7>(acc0, acc1, cur, d, ac);
                        }
                    } else {
                        simd<uint32_t, 16> h0 = cur.h.template select<16, 1>(0), h1 = cur.h.template select<16, 1>(16);
                        simd<uint32_t, 16> h2 = cur.h.template select<16, 1>(32), h3 = cur.h.template select<16, 1>(48);
                        simd<float, 16> d = half_of(h0 & 0xffffu), dmin = half_of(h0 >> 16);
                        {
                            ActsHalf<NC> ac;
                            load_acts_half<NC, 0>(ac, xi0, xs9);
                            pair_y<F, NC, 0, 0>(acc0, acc1, cur.q0, cur.qh, h1, h2, h3, d, dmin, ac);
                            pair_y<F, NC, 1, 0>(acc0, acc1, cur.q1, cur.qh, h1, h2, h3, d, dmin, ac);
                        }
                        {
                            ActsHalf<NC> ac;
                            load_acts_half<NC, 1>(ac, xi0, xs9);
                            pair_y<F, NC, 2, 1>(acc0, acc1, cur.q2, cur.qh, h1, h2, h3, d, dmin, ac);
                            pair_y<F, NC, 3, 1>(acc0, acc1, cur.q3, cur.qh, h1, h2, h3, d, dmin, ac);
                        }
                    }
                }
                if constexpr (C > 1) {
                    if (c != 0) {
                        const uint32_t base = buf + (uint32_t) (kSlmSingle ? (c - 1) : c) * kSlmPerT;
                        slm_store_acc<8>(base, acc0);
                        slm_store_acc<NC1>(base + 512u, acc1);
                    }
                    barrier();
                    if (c == 0) {
#pragma unroll
                        for (int cc = 1; cc < C; ++cc) {
                            const uint32_t base = buf + (uint32_t) (kSlmSingle ? (cc - 1) : cc) * kSlmPerT;
                            slm_add_acc<8>(base, acc0);
                            slm_add_acc<NC1>(base + 512u, acc1);
                        }
                    }
                    if constexpr (kSlmSingle) {
                        if (t + G < tiles) barrier();
                    } else {
                        buf = buf == kTab ? kTab + (uint32_t) (C * kSlmPerT) : kTab;
                    }
                }
                if (c == 0) {
#pragma unroll
                    for (int r = 0; r < 8; ++r)
                        block_store<float, 16>(Y + (size_t) r * n_out + 16 * (size_t) t, acc0.template select<16, 1>(16 * r));
#pragma unroll
                    for (int r = 0; r < NC1; ++r)
                        block_store<float, 16>(Y + (size_t) (8 + r) * n_out + 16 * (size_t) t, acc1.template select<16, 1>(16 * r));
                }
            }
        });
    });
}
template <int F, int NC, int DEC = 0>
void dispatch(sycl::queue& q, const uint8_t* W, const int* Xi, const float* S, float* Y, int bpr, int n_out, bool q81) {
    const int c = chunks_of(bpr * 256, n_out), nt = pool_threads();
    (void) q81;   // (the callers pass the q8_1 image: mmvq declines the other)
    // (PIPE = false: with the one-message loads, prefetch beat the register pipeline on every shape, 5120 x 17408 Q4_K at 4
    // columns 94 us against 107)
    if constexpr (NC <= 8) {
        if constexpr (F == T_Q4K || F == T_Q5K || F == T_IQ4XS || F == T_Q8_0 || F == T_IQ3S) {
            if (c == 4) launch<F, NC, 4, true, false, DEC>(q, W, Xi, S, Y, bpr, n_out, nt);
            else if (c == 2) launch<F, NC, 2, true, false, DEC>(q, W, Xi, S, Y, bpr, n_out, nt);
            else launch<F, NC, 1, true, false, DEC>(q, W, Xi, S, Y, bpr, n_out, nt);
        }
    } else {
        if (c == 4) launch16<F, NC, 4, true, false, DEC>(q, W, Xi, S, Y, bpr, n_out, nt);
        else if (c == 2) launch16<F, NC, 2, true, false, DEC>(q, W, Xi, S, Y, bpr, n_out, nt);
        else launch16<F, NC, 1, true, false, DEC>(q, W, Xi, S, Y, bpr, n_out, nt);
    }
}
template <int F, int DEC = 0>
void by_cols(sycl::queue& q, const uint8_t* W, const int* Xi, const float* S, float* Y, int bpr, int n_out, int ncols, bool q81) {
    switch (ncols) {
        case 1: dispatch<F, 1, DEC>(q, W, Xi, S, Y, bpr, n_out, q81); break;
        case 2: dispatch<F, 2, DEC>(q, W, Xi, S, Y, bpr, n_out, q81); break;
        case 3: dispatch<F, 3, DEC>(q, W, Xi, S, Y, bpr, n_out, q81); break;
        case 4: dispatch<F, 4, DEC>(q, W, Xi, S, Y, bpr, n_out, q81); break;
        case 5: dispatch<F, 5, DEC>(q, W, Xi, S, Y, bpr, n_out, q81); break;
        case 6: dispatch<F, 6, DEC>(q, W, Xi, S, Y, bpr, n_out, q81); break;
        case 7: dispatch<F, 7, DEC>(q, W, Xi, S, Y, bpr, n_out, q81); break;
        case 8: dispatch<F, 8, DEC>(q, W, Xi, S, Y, bpr, n_out, q81); break;
        case 9: dispatch<F, 9, DEC>(q, W, Xi, S, Y, bpr, n_out, q81); break;
        case 10: dispatch<F, 10, DEC>(q, W, Xi, S, Y, bpr, n_out, q81); break;
        case 11: dispatch<F, 11, DEC>(q, W, Xi, S, Y, bpr, n_out, q81); break;
        case 12: dispatch<F, 12, DEC>(q, W, Xi, S, Y, bpr, n_out, q81); break;
        case 13: dispatch<F, 13, DEC>(q, W, Xi, S, Y, bpr, n_out, q81); break;
        case 14: dispatch<F, 14, DEC>(q, W, Xi, S, Y, bpr, n_out, q81); break;
        case 15: dispatch<F, 15, DEC>(q, W, Xi, S, Y, bpr, n_out, q81); break;
        case 16:
        default: dispatch<F, 16, DEC>(q, W, Xi, S, Y, bpr, n_out, q81); break;
    }
}
inline bool xr16_on() {
    // 9-16 columns in one launch (one weight read, two DPAS column groups).  STRATA_Q36_XR16=0 keeps 9-16 off.
    static const bool on = [] { const char* e = std::getenv("STRATA_Q36_XR16"); return !(e && e[0] == '0'); }();
    return on;
}
inline bool iq4_all_on() {
    // IQ4_XS: by default, <=8 columns take KREG on n_in <= 16384 (codebook lookups make XR slower there: 77-83 us vs 64.5
    // at 4 cols) and XR on >16384 (270 -> 127 us at 5).  STRATA_Q36_XR_IQ4=all: every shape at 1-16 columns.
    static const bool all = [] { const char* e = std::getenv("STRATA_Q36_XR_IQ4"); return e && e[0] == 'a'; }();
    return all;
}
inline bool xr16_iq4_on() {
    // IQ4_XS on n_in <= 16384 at 9-16 columns: TagXr16 reads the 3.62 GB once instead of KREG running twice (8+rest).
    // STRATA_Q36_XR16_IQ4=0 keeps n_in <= 16384 on the 8+rest KREG split at 9-16 columns.
    static const bool on = [] { const char* e = std::getenv("STRATA_Q36_XR16_IQ4"); return !(e && e[0] == '0'); }();
    return on;
}
inline bool xr16_q6_on() {
    // Q6_K at 9-16 columns: TagXr16 reads head (1.04 GB) and main (0.58 GB effective) once instead of KREG splits.
    // STRATA_Q36_XR16_Q6=0 keeps Q6_K on KREG at 9-16 columns.
    static const bool on = [] { const char* e = std::getenv("STRATA_Q36_XR16_Q6"); return !(e && e[0] == '0'); }();
    return on;
}
inline bool xr16_iq4nl_on() {
    // IQ4_NL at 9-16 columns: TagXr16 reads weights once instead of KREG 8+rest / 4-col splits.  STRATA_Q36_XR16_IQ4NL=0: off.
    static const bool on = [] { const char* e = std::getenv("STRATA_Q36_XR16_IQ4NL"); return !(e && e[0] == '0'); }();
    return on;
}
inline bool xr16_q3k_on() {
    // Q3_K at 9-16 columns: TagXr16 reads weights once (no KregQ3K scratch spill).  STRATA_Q36_XR16_Q3K=0: off.
    static const bool on = [] { const char* e = std::getenv("STRATA_Q36_XR16_Q3K"); return !(e && e[0] == '0'); }();
    return on;
}
inline int iq4_dec_mode() {
    // IQ4_XS / IQ4_NL decode path + single-buffered SLM reduction:
    //   1 (pre, default): batched 16x slm_gather + single-buf SLM1
    //   0 (slm / old):    interleaved 4x slm_gather + double-buf SLM
    //   2 (reg):          register-resident 16-entry uint16_t mux LUT (0 SLM table) + SLM1
    //   3 (l1):           L1/L2-cached global table gather (0 SLM table) + SLM1
    static const int m = [] {
        const char* e = std::getenv("STRATA_Q36_XR_IQ4_DEC");
        if (!e || !e[0]) return 1;
        if (e[0] == '0' || e[0] == 's' || e[0] == 'o') return 0;
        if (e[0] == '2' || e[0] == 'r') return 2;
        if (e[0] == '3' || e[0] == 'l') return 3;
        return 1;
    }();
    return m;
}
inline bool q3k_slm1_on() {
    // Q3_K XR16 single-buffered SLM reduction (3072 B vs 8192 B at C=4, NC=16). STRATA_Q36_XR_Q3K_SLM1=0: off.
    static const bool on = [] { const char* e = std::getenv("STRATA_Q36_XR_Q3K_SLM1"); return !(e && e[0] == '0'); }();
    return on;
}
inline bool xr_q8_on() {
    // Q8_0 at 1-16 columns in one read of the weights. STRATA_Q36_XR_Q8=0 keeps Wide32Q8.
    static const bool on = [] { const char* e = std::getenv("STRATA_Q36_XR_Q8"); return !(e && e[0] == '0'); }();
    return on;
}
inline bool xr16_q8_on() {
    // Q8_0 at 9-16 columns in one read of the weights. STRATA_Q36_XR16_Q8=0 splits 8+rest through Wide32Q8 / XR.
    static const bool on = [] { const char* e = std::getenv("STRATA_Q36_XR16_Q8"); return !(e && e[0] == '0'); }();
    return on;
}
inline bool xr_iq3s_on() {
    // IQ3_S at 1-16 columns in one read of the weights. STRATA_Q36_XR_IQ3S=0 keeps launch_mmvq_multi_nc<21,false>.
    static const bool on = [] { const char* e = std::getenv("STRATA_Q36_XR_IQ3S"); return !(e && e[0] == '0'); }();
    return on;
}
inline bool xr16_iq3s_on() {
    // IQ3_S at 9-16 columns in one read of the weights. STRATA_Q36_XR16_IQ3S=0 keeps 8+rest split.
    static const bool on = [] { const char* e = std::getenv("STRATA_Q36_XR16_IQ3S"); return !(e && e[0] == '0'); }();
    return on;
}
}   // namespace

void init(void* stream) {
    sycl::queue& q = *strata::q_of(stream);
    (void) iq4_table(q);
    (void) iq3s_table(q);
}
bool handles(int t) {
    return t == T_Q8_0 || t == T_Q3K || t == T_Q4K || t == T_Q5K || t == T_Q6K || t == T_IQ4NL || t == T_IQ3S || t == T_IQ4XS;
}
bool handles_call(int type, int n_in, int n_out, int ncols, bool q81) {
    if (!q81 || !handles(type) || ncols < 1 || ncols > 16) return false;
    if (ncols > 8 && !xr16_on()) return false;
    if (n_in <= 0 || n_in % 256 != 0 || n_out <= 0 || n_out % 16 != 0) return false;
    if ((type == T_Q6K || type == T_Q3K || type == T_IQ3S) && n_in % 512 != 0) return false;
    if (type == T_Q8_0 && (!xr_q8_on() || (ncols > 8 && !xr16_q8_on()))) return false;
    if (type == T_IQ3S && (!xr_iq3s_on() || (ncols > 8 && !xr16_iq3s_on()))) return false;
    if (type == T_Q6K && (ncols <= 8 || !xr16_q6_on())) return false;
    if (type == T_IQ4NL && (ncols <= 8 || !xr16_iq4nl_on())) return false;
    if (type == T_Q3K && (ncols <= 8 || !xr16_q3k_on())) return false;
    if (type == T_IQ4XS && n_in <= 16384 && !iq4_all_on() && !(ncols > 8 && xr16_iq4_on())) return false;
    const std::size_t rowb = (std::size_t) (n_in / 256) *
        (type == T_Q4K || type == T_IQ4NL ? 144 :
         type == T_Q5K ? 176 :
         type == T_IQ4XS ? 136 :
         type == T_Q6K ? 210 :
         type == T_Q8_0 ? 272 : 110);
    if (rowb * (std::size_t) n_out >= ((std::size_t) 1 << 32)) return false;
    return true;
}
// [n_in / 32][8 columns][32 int8], then [n_in / 256][8 sub-blocks][8] scales and [..][8][8] scaled sums
std::size_t act_bytes(int n_in) { return (std::size_t) (n_in / 32) * 256 + (std::size_t) (n_in / 256) * 128 * sizeof(float); }

void quantize(const float* x, int n_in, int ncols, void* act, void* stream) {
    sycl::queue& q = *strata::q_of(stream);
    (void) iq4_table(q);    // made here, outside any graph recording a product call may be part of
    (void) iq3s_table(q);
    const int nsb = n_in / 32;
    int8_t* xi = static_cast<int8_t*>(act);
    float* sc = reinterpret_cast<float*>(xi + (std::size_t) nsb * 256);
    q.parallel_for<class q36xr_quantize>(sycl::range<2>((size_t) nsb, 8), [=](sycl::id<2> id) {
        const int sb = (int) id[0], r = (int) id[1];
        int8_t* o = xi + (std::size_t) sb * 256 + r * 32;
        float* so = sc + (std::size_t) (sb >> 3) * 128 + (sb & 7) * 8 + r;
        if (r >= ncols) {
            for (int i = 0; i < 32; ++i) o[i] = 0;
            so[0] = 0.0f; so[64] = 0.0f;
            return;
        }
        const float* xr = x + (std::size_t) r * n_in + (std::size_t) sb * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = sycl::fmax(amax, sycl::fabs(xr[i]));
        const float d = amax / 127.0f, idv = d != 0.0f ? 1.0f / d : 0.0f;
        int sum = 0;
        for (int i = 0; i < 32; ++i) {
            const int v = (int) sycl::round(xr[i] * idv);
            o[i] = (int8_t) v;
            sum += v;
        }
        so[0] = d;
        so[64] = d * (float) sum;
    });
}

bool mmvq(int type, const void* weights, const void* act, float* y, int n_in, int n_out, int ncols, void* stream, bool q81) {
    if (!handles_call(type, n_in, n_out, ncols, q81)) return false;
    if (!weights || !act || !y || (reinterpret_cast<uintptr_t>(weights) & 3u) != 0) return false;
    const int bpr = n_in / 256;
    sycl::queue& q = *strata::q_of(stream);
    const auto* W = static_cast<const uint8_t*>(weights);
    const int* Xi = static_cast<const int*>(act);
    const float* S = reinterpret_cast<const float*>(static_cast<const uint8_t*>(act) + (std::size_t) (n_in / 32) * 256);
    if ((reinterpret_cast<uintptr_t>(act) & 3u) != 0) return false;
    if (type == T_Q4K) by_cols<T_Q4K>(q, W, Xi, S, y, bpr, n_out, ncols, q81);
    else if (type == T_Q5K) by_cols<T_Q5K>(q, W, Xi, S, y, bpr, n_out, ncols, q81);
    else if (type == T_Q6K) by_cols<T_Q6K>(q, W, Xi, S, y, bpr, n_out, ncols, q81);
    else if (type == T_Q8_0) by_cols<T_Q8_0>(q, W, Xi, S, y, bpr, n_out, ncols, q81);
    else if (type == T_IQ3S) by_cols<T_IQ3S>(q, W, Xi, S, y, bpr, n_out, ncols, q81);
    else if (type == T_Q3K) {
        if (q3k_slm1_on()) by_cols<T_Q3K, 1>(q, W, Xi, S, y, bpr, n_out, ncols, q81);
        else by_cols<T_Q3K, 0>(q, W, Xi, S, y, bpr, n_out, ncols, q81);
    } else if (type == T_IQ4NL) {
        switch (iq4_dec_mode()) {
            case 0: by_cols<T_IQ4NL, 0>(q, W, Xi, S, y, bpr, n_out, ncols, q81); break;
            case 2: by_cols<T_IQ4NL, 2>(q, W, Xi, S, y, bpr, n_out, ncols, q81); break;
            case 3: by_cols<T_IQ4NL, 3>(q, W, Xi, S, y, bpr, n_out, ncols, q81); break;
            default: by_cols<T_IQ4NL, 1>(q, W, Xi, S, y, bpr, n_out, ncols, q81); break;
        }
    } else {
        switch (iq4_dec_mode()) {
            case 0: by_cols<T_IQ4XS, 0>(q, W, Xi, S, y, bpr, n_out, ncols, q81); break;
            case 2: by_cols<T_IQ4XS, 2>(q, W, Xi, S, y, bpr, n_out, ncols, q81); break;
            case 3: by_cols<T_IQ4XS, 3>(q, W, Xi, S, y, bpr, n_out, ncols, q81); break;
            default: by_cols<T_IQ4XS, 1>(q, W, Xi, S, y, bpr, n_out, ncols, q81); break;
        }
    }
    return true;
}
}   // namespace strata::q36xr
