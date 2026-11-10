// sycl/src/kernels/gr_q8_parity.cpp - STRATA_HC_Q8=1 on SYCL: fused_gr_read_multi's Q8_0 read (q8_down / q8_up
// set) against a CPU FP64 reference on the very Q8_0 values it reads, and against the port's BF16 read.  Real
// geometry (n_embd 2560, hc 4, hc_lr 320), T = 1..8, with and without the pending write and the inject rows (a layer
// read / the final mixer), direct and as a captured graph replayed.  GPU, synthetic, no model.
//
// What it checks:
//   1. the Q8_0 read is observable: its `mixed` differs from the BF16 read's by about the Q8_0 rounding of the
//      weights, and not more (the port used to ignore the Q8_0 copies and read BF16 whatever STRATA_HC_Q8 said);
//   2. lo, rs, inject and mixed match an FP64 reference computed from the Q8_0 bytes (d * q) within FP32 summation
//      rounding; R_out (the pending write, the same fma as the BF16 read) is bitwise the BF16 read's;
//   3. a captured graph replayed twice gives bitwise the direct call's outputs (the arrival counters reset);
//   4. arguments not at the callers' fixed strides fall back to the BF16 read, bitwise;
//   5. STRATA_HC_Q8_DOWN2 (fused_gr_set_q8_down2): the same read through the second down kernel, twice in a row
//      (its arrival counters reset too), bitwise the first kernel's outputs.  Checks 1-4 run with it off.
//
// --bench follows STRATA_HC_Q8_DOWN2 in the environment: run it with =0 and =1 and compare the times and the
// outputs hash (the hash must be the same).
// --bench with STRATA_HC_BENCH_STRIP=1|2|3 (unset or 0: the real read): the stripped read kernels of grs:: below in its place, to split
// the read's time into weight reads and the rest; it follows STRATA_HC_Q8_DOWN2 too; no outputs hash.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/fused_gr_q8.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {
constexpr int N = 2560, HC = 4, LR = 320, D = N * HC, TM = k::kFusedGrMaxT;

template <typename T>
T* up(const std::vector<T>& h) {
    T* p = sycl::malloc_device<T>(h.size() + 64, dpct::get_in_order_queue());
    dpct::get_in_order_queue().memcpy(p, h.data(), h.size() * sizeof(T)).wait();
    return p;
}
template <typename T>
T* dev(size_t n) {
    T* p = sycl::malloc_device<T>(n + 64, dpct::get_in_order_queue());
    dpct::get_in_order_queue().memset(p, 0, (n + 64) * sizeof(T)).wait();
    return p;
}
template <typename T>
std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    dpct::get_in_order_queue().memcpy(h.data(), d, n * sizeof(T)).wait();
    return h;
}

uint16_t bf16_of(float f) {
    uint32_t i;
    std::memcpy(&i, &f, 4);
    i = (i + ((i >> 16) & 1u) + 0x7FFFu) & 0xFFFF0000u;
    return (uint16_t) (i >> 16);
}
float bf16_to_f(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float half_to_f(uint16_t h) { return (float) sycl::bit_cast<sycl::half>(h); }

/// ggml's Q8_0 blocks of `n` bf16 values (d = amax / 127 as fp16, q = round(x / d)): the loader's rule
std::vector<uint8_t> q8_0_of(const std::vector<uint16_t>& w) {
    std::vector<uint8_t> out(w.size() / 32 * 34);
    for (size_t b = 0; b < w.size() / 32; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(bf16_to_f(w[b * 32 + j])));
        const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        const uint16_t hb = sycl::bit_cast<uint16_t>(sycl::half(d));
        uint8_t* o = out.data() + b * 34;
        std::memcpy(o, &hb, 2);
        for (int j = 0; j < 32; ++j) o[2 + j] = (uint8_t) (int8_t) std::lround(bf16_to_f(w[b * 32 + j]) * id);
    }
    return out;
}
/// the values a Q8_0 block holds (the fp16 scale as the kernel reads it, times the quant)
std::vector<float> dequant(const std::vector<uint8_t>& blocks) {
    std::vector<float> w(blocks.size() / 34 * 32);
    for (size_t b = 0; b < blocks.size() / 34; ++b) {
        uint16_t hb;
        std::memcpy(&hb, blocks.data() + b * 34, 2);
        const float d = half_to_f(hb);
        for (int j = 0; j < 32; ++j) w[b * 32 + j] = d * (float) (int8_t) blocks[b * 34 + 2 + j];
    }
    return w;
}

struct Out {
    std::vector<float> r_out, lo, rs, inject, mixed;
};

/// ref/gr.py::gr_read in FP64 for T rows: R' = R + bo * 2 sigmoid(inj_prev / hc) (apply), rs per stream,
/// lo = silu(w_down . xn / hc), inject = w_inject . xn, mixed = mean_c xn * sigmoid(w_up . lo)
Out reference(int T, bool apply, bool inject, const std::vector<float>& R, const std::vector<float>& bo,
              const std::vector<float>& ip, const std::vector<float>& wn, const std::vector<float>& wd,
              const std::vector<float>& wu, const std::vector<float>& wi, float eps) {
    Out o;
    o.r_out.resize((size_t) T * D); o.lo.resize((size_t) T * LR); o.rs.resize((size_t) T * HC);
    o.inject.resize((size_t) T * HC); o.mixed.resize((size_t) T * N);
    std::vector<double> rp(D), xn(D), lo(LR);
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < HC; ++c) {
            const double gw = apply ? 2.0 / (1.0 + std::exp(-(double) ip[t * HC + c] / HC)) : 0.0;
            double ss = 0.0;
            for (int d = 0; d < N; ++d) {
                const int i = c * N + d;
                double r = R[(size_t) t * D + i];
                if (apply) r += (double) bo[(size_t) t * N + d] * gw;
                rp[i] = r;
                ss += r * r;
            }
            const double rs = 1.0 / std::sqrt(ss / N + (double) eps);
            o.rs[t * HC + c] = (float) rs;
            for (int d = 0; d < N; ++d) xn[c * N + d] = rp[c * N + d] * wn[c * N + d] * rs;
        }
        for (int i = 0; i < D; ++i) o.r_out[(size_t) t * D + i] = (float) rp[i];
        for (int j = 0; j < LR; ++j) {
            double s = 0.0;
            for (int i = 0; i < D; ++i) s += (double) wd[(size_t) j * D + i] * xn[i];
            const double x = s / HC;
            lo[j] = x / (1.0 + std::exp(-x));
            o.lo[t * LR + j] = (float) lo[j];
        }
        for (int c = 0; c < HC; ++c) {
            double s = 0.0;
            if (inject)
                for (int i = 0; i < D; ++i) s += (double) wi[(size_t) c * D + i] * xn[i];
            o.inject[t * HC + c] = (float) s;
        }
        for (int d = 0; d < N; ++d) {
            double m = 0.0;
            for (int c = 0; c < HC; ++c) {
                const int i = c * N + d;
                double u = 0.0;
                for (int j = 0; j < LR; ++j) u += (double) wu[(size_t) i * LR + j] * lo[j];
                m += xn[i] / (1.0 + std::exp(-u));
            }
            o.mixed[(size_t) t * N + d] = (float) (m / HC);
        }
    }
    return o;
}

/// worst |x - y| over max |y|
double rel_worst(const std::vector<float>& x, const std::vector<float>& y) {
    double worst = 0.0, mag = 1e-30;
    for (size_t i = 0; i < x.size(); ++i) {
        worst = std::max(worst, (double) std::fabs(x[i] - y[i]));
        mag = std::max(mag, (double) std::fabs(y[i]));
    }
    return worst / mag;
}
/// RMS (x - y) over RMS y
double rel_rms(const std::vector<float>& x, const std::vector<float>& y) {
    double d = 0.0, m = 1e-60;
    for (size_t i = 0; i < x.size(); ++i) { d += (double) (x[i] - y[i]) * (x[i] - y[i]); m += (double) y[i] * y[i]; }
    return std::sqrt(d / m);
}
bool same(const std::vector<float>& x, const std::vector<float>& y) {
    return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0;
}

// ------------------------------------------------------------------------------------------------ --bench STRIP kernels
// STRATA_HC_BENCH_STRIP (bench only): the Q8_0 hyper-connection read with work taken out, to tell whether it is limited by the
// weight reads or by the arithmetic and the phases around them.  Two kernels per read as fused_gr.dp.cpp's gr_q8 namespace (read_t):
//   down: gr_q8_down_kernel (launch_down), or gr_q8_down2_kernel (launch_down2) when STRATA_HC_Q8_DOWN2 is set (read as down2_on reads
//         it): 5 row groups x 20 slabs = 100 work-groups of 256 work-items (16 sub-groups), 4 rows a lane x 32 columns a sub-group;
//         a lane loads 2 x 16 bytes of quants (uint4) and one fp16 scale of each of its 4 rows;
//   up:   gr_q8_up_kernel (launch_up): 160 work-groups of 320 work-items (20 sub-groups), one row a lane, K = 320 in 5 fifths;
//         a lane loads 4 x 16 bytes of quants and 2 fp16 scales.
// The bench's read is a layer read: pending write on, no inject rows, so the down kernel launches 5 row groups and holds no BF16 row.
//   STRIP=3 weight reads only: the quants and scales of both kernels, xor-folded; nothing else.
//   STRIP=1 reads: STRIP=3 plus the state rows the real kernels load.  Down: ip (gate input), and for the 128 staging work-items R,
//           the pending write's bo and w_norm, in the order of the kernel that runs (gr_q8_down_kernel: ip, R, bo, wn per token;
//           gr_q8_down2_kernel: wn once, then R of every token, bo of every token, ip), staged into local memory as the real
//           kernels stage them, one barrier, then the local reads of the dot (2 x 16 bytes per 8 columns and token).  Up: R, bo, ip,
//           w_norm and rs of the lane's epilogue elements, lo staged through local memory (one barrier) and read as the dot reads
//           it.  All folded by bit pattern; no sums of squares, no sigmoid, no products.
//   STRIP=2 reads + decode, no products: STRIP=1 plus q8x8 (fused_gr.dp.cpp): per weight the byte extract, the int to float
//           conversion and the multiply by the block scale, the 8 values folded by bit pattern (integer xor, so the compiler cannot
//           turn the products into one multiply of a sum).  Q8_0 has this much decode: no table, no scale unpacking.
// Dropped in every variant: the fma chains with the activations, the sums of squares, the sub-group and slab reductions, the local
// partials, the global partials (part, ssg), the arrival counter with its fences and the finishing work-group's reads, lo / rs /
// inject writes, the up kernel's R_out and mixed stores, and the other barriers.  So STRIP=1 is weight plus state reads, not all of
// the memory traffic of the real read; the real kernels' phases (their comment in fused_gr.dp.cpp: the down kernel is bound by its
// chain of phases) are what STRIP=1 minus the real call measures.  A work-item's value is xor-folded over its sub-group and stored
// once (one word a sub-group, 1,600 for down and 3,200 for up), so every load stays live.  The local allocations of the real launches
// are made (xs sized as there, a pad for the rest) and touched with one store.
namespace grs {

constexpr int SG = 16, DTH = 256, DSG = DTH / SG;   // = gr_q8
constexpr int RPL = 4, KS = 32;                     // = DOWN_RPL_Q8, DOWN_KS_Q8
constexpr int SLAB = DSG * KS, NSLAB = D / SLAB, SPS = N / SLAB, RPG = SG * RPL, NRG = (LR + RPG - 1) / RPG;
constexpr int UQ = 5, UKQ = LR / UQ, USG = 4 * UQ, UTH = USG * SG, UP_COLS = 16;   // = gr_q8_up_kernel's
constexpr int kUpSink = 2048;   // word offset of the up kernel's sink words (down stores NRG * NSLAB * DSG = 1600)
static_assert(D % SLAB == 0 && N % SLAB == 0 && NRG * RPG >= LR && (NRG - 1) * RPG < LR, "slabs never straddle streams");

struct Ptrs {
    const float *R, *bo, *ip, *wn, *lo, *rs;   // the bench's rows (token k at R + k * D, bo + k * N, ip + k * HC, lo + k * LR, rs + k * HC)
    uint32_t* sink;
};

__dpct_inline__ uint32_t x4(const sycl::uint4 v) { return v.x() ^ v.y() ^ v.z() ^ v.w(); }
__dpct_inline__ uint32_t fbits(float f) { return sycl::bit_cast<uint32_t>(f); }
__dpct_inline__ uint32_t lanes_xor(const sycl::sub_group& sg, uint32_t v) {
#pragma unroll
    for (int o = SG / 2; o > 0; o >>= 1) v ^= sycl::permute_group_by_xor(sg, v, o);
    return v;
}
// = q8x8: 8 Q8_0 values of a 16-byte chunk (half 0: bytes 0-7, half 1: bytes 8-15) times their block's scale, folded by bit pattern
__dpct_inline__ uint32_t q8x8_fold(const sycl::uint4 q, int half, float d) {
    const uint32_t a = half ? q.z() : q.x(), b = half ? q.w() : q.y();
    return fbits(d * (float) (int8_t) (a & 255u)) ^ fbits(d * (float) (int8_t) ((a >> 8) & 255u)) ^
           fbits(d * (float) (int8_t) ((a >> 16) & 255u)) ^ fbits(d * (float) (int8_t) (a >> 24)) ^
           fbits(d * (float) (int8_t) (b & 255u)) ^ fbits(d * (float) (int8_t) ((b >> 8) & 255u)) ^
           fbits(d * (float) (int8_t) ((b >> 16) & 255u)) ^ fbits(d * (float) (int8_t) (b >> 24));
}

// gr_q8_down_kernel (DN2 false) / gr_q8_down2_kernel (DN2 true) at T tokens: step 1 (the weights), step 2 (the staging) and the
// local reads of step 3
template <int MODE, bool DN2, int T>
__dpct_inline__ void down_strip(sycl::nd_item<3> item, const Ptrs& m, const uint8_t* qd, float* xs, float* pad) {
    constexpr int NB = KS / 32, NJ = KS / 8, NQ = KS / 16;   // scales, chunks of 8 values, 16-byte chunks
    const sycl::sub_group sgp = item.get_sub_group();
    const int t = (int) item.get_local_id(2), lane = t & (SG - 1), sg = t / SG;
    const int g = (int) item.get_group(2), rg = g / NSLAB, slab = g - rg * NSLAB;
    [[maybe_unused]] const int c = slab / SPS;
    const int col0 = slab * SLAB + sg * KS;
    // 1. this lane's 4 rows over the sub-group's 32 columns (every row of the launched row groups is below LR)
    const uint16_t* dbase = reinterpret_cast<const uint16_t*>(qd + (size_t) LR * D);
    sycl::uint4 qv[RPL][NQ];
    uint16_t dv[RPL][NB];
#pragma unroll
    for (int p = 0; p < RPL; ++p) {
        const int row = rg * RPG + p * SG + lane;
        const sycl::uint4* q4 = reinterpret_cast<const sycl::uint4*>(qd + (size_t) row * D + col0);
#pragma unroll
        for (int j = 0; j < NQ; ++j) qv[p][j] = q4[j];
#pragma unroll
        for (int b = 0; b < NB; ++b) dv[p][b] = dbase[(size_t) row * (D / 32) + col0 / 32 + b];
    }
    // 2. the staging of R' * w_norm for the T tokens (work-items 0..127: columns 4 t .. 4 t + 3), one barrier, then step 3's local reads
    uint32_t af = 0u;
    if constexpr (MODE != 3) {
        const int col = slab * SLAB + 4 * t;
        const bool stg = t < SLAB / 4;
        if constexpr (!DN2) {
#pragma unroll
            for (int k = 0; k < T; ++k) {
                const uint32_t ib = fbits(m.ip[k * HC + c]);   // the pending write's gate input: every work-item, as the kernel
                af ^= ib;
                if (stg) {
                    const sycl::uint4 r = *reinterpret_cast<const sycl::uint4*>(m.R + (size_t) k * D + col);
                    const sycl::uint4 b = *reinterpret_cast<const sycl::uint4*>(m.bo + (size_t) k * N + (col - c * N));
                    const sycl::uint4 gn = *reinterpret_cast<const sycl::uint4*>(m.wn + col);
                    *reinterpret_cast<sycl::uint4*>(xs + k * SLAB + 4 * t) =
                        sycl::uint4(r.x() ^ b.x() ^ gn.x() ^ ib, r.y() ^ b.y() ^ gn.y() ^ ib, r.z() ^ b.z() ^ gn.z() ^ ib, r.w() ^ b.w() ^ gn.w() ^ ib);
                }
            }
        } else if (stg) {
            const sycl::uint4 gn = *reinterpret_cast<const sycl::uint4*>(m.wn + col);   // w_norm once
            if constexpr (T <= 4) {   // every token's R, then every token's bo, then the gates
                sycl::uint4 rv[T], bv[T];
                uint32_t ibv[T];
#pragma unroll
                for (int k = 0; k < T; ++k) rv[k] = *reinterpret_cast<const sycl::uint4*>(m.R + (size_t) k * D + col);
#pragma unroll
                for (int k = 0; k < T; ++k) bv[k] = *reinterpret_cast<const sycl::uint4*>(m.bo + (size_t) k * N + (col - c * N));
#pragma unroll
                for (int k = 0; k < T; ++k) ibv[k] = fbits(m.ip[k * HC + c]);
#pragma unroll
                for (int k = 0; k < T; ++k) {
                    af ^= ibv[k];
                    *reinterpret_cast<sycl::uint4*>(xs + k * SLAB + 4 * t) =
                        sycl::uint4(rv[k].x() ^ bv[k].x() ^ gn.x() ^ ibv[k], rv[k].y() ^ bv[k].y() ^ gn.y() ^ ibv[k],
                                    rv[k].z() ^ bv[k].z() ^ gn.z() ^ ibv[k], rv[k].w() ^ bv[k].w() ^ gn.w() ^ ibv[k]);
                }
            } else {   // per token: R, the gate, bo
#pragma unroll
                for (int k = 0; k < T; ++k) {
                    const sycl::uint4 r = *reinterpret_cast<const sycl::uint4*>(m.R + (size_t) k * D + col);
                    const uint32_t ib = fbits(m.ip[k * HC + c]);
                    const sycl::uint4 b = *reinterpret_cast<const sycl::uint4*>(m.bo + (size_t) k * N + (col - c * N));
                    af ^= ib;
                    *reinterpret_cast<sycl::uint4*>(xs + k * SLAB + 4 * t) =
                        sycl::uint4(r.x() ^ b.x() ^ gn.x() ^ ib, r.y() ^ b.y() ^ gn.y() ^ ib, r.z() ^ b.z() ^ gn.z() ^ ib, r.w() ^ b.w() ^ gn.w() ^ ib);
                }
            }
        }
        item.barrier(sycl::access::fence_space::local_space);
        const float* xsg = xs + sg * KS;   // the dot's reads: the same address in every lane of the sub-group
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int k = 0; k < T; ++k) {
                af ^= x4(*reinterpret_cast<const sycl::uint4*>(xsg + k * SLAB + 8 * j));
                af ^= x4(*reinterpret_cast<const sycl::uint4*>(xsg + k * SLAB + 8 * j + 4));
            }
    }
    // the weights, folded (STRIP=2: decoded as q8x8 does)
    uint32_t wf = 0u;
    if constexpr (MODE == 2) {
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int p = 0; p < RPL; ++p) wf ^= q8x8_fold(qv[p][j / 2], j & 1, (float) sycl::bit_cast<sycl::half>(dv[p][j / 4]));
    } else {
#pragma unroll
        for (int p = 0; p < RPL; ++p) {
#pragma unroll
            for (int j = 0; j < NQ; ++j) wf ^= x4(qv[p][j]);
#pragma unroll
            for (int b = 0; b < NB; ++b) wf ^= (uint32_t) dv[p][b];
        }
    }
    uint32_t s = wf ^ af;
    pad[t & 127] = sycl::bit_cast<float>(s);   // touches the local allocation of the real launch
    s = lanes_xor(sgp, s);
    if (lane == 0) m.sink[g * DSG + sg] = s;
}

// gr_q8_up_kernel at T tokens: this lane's row of w_up (K fifth q), the epilogue elements' loads, the lo staging and the dot's local reads
template <int MODE, int T>
__dpct_inline__ void up_strip(sycl::nd_item<3> item, const Ptrs& m, const uint8_t* qu, float* lo, float* pad) {
    constexpr int NQ = UKQ / 16, NB = UKQ / 32, NJ = UKQ / 8;   // 4 chunks of 16 bytes, 2 blocks, 8 chunks of 8
    const sycl::sub_group sgp = item.get_sub_group();
    const int t = (int) item.get_local_id(2), lane = t & (SG - 1), sg = t / SG;
    const int cg = sg % 4, q = sg / 4;                        // column group of 4, K fifth
    const int g = (int) item.get_group(2), d0 = g * UP_COLS;
    const int c = lane / 4, d = d0 + cg * 4 + (lane & 3), i = c * N + d;   // this lane's row of w_up
    const uint16_t* dbase = reinterpret_cast<const uint16_t*>(qu + (size_t) D * LR);
    sycl::uint4 qv[NQ];
    uint16_t dv[NB];
    const sycl::uint4* q4 = reinterpret_cast<const sycl::uint4*>(qu + (size_t) i * LR + q * UKQ);
#pragma unroll
    for (int j = 0; j < NQ; ++j) qv[j] = q4[j];
#pragma unroll
    for (int b = 0; b < NB; ++b) dv[b] = dbase[(size_t) i * (LR / 32) + q * NB + b];
    uint32_t af = 0u;
    if constexpr (MODE != 3) {
        constexpr int NE = (T * 64 + UTH - 1) / UTH;
#pragma unroll
        for (int e = 0; e < NE; ++e) {   // the epilogue's element(s) of this work-item: R, bo, the gate input, w_norm, rs
            const int x = t + e * UTH;
            if (x < T * 64) {
                const int k = x / 64, s = x - k * 64, cgs = s / SG, l = s - cgs * SG;
                const int cs = l / 4, ds = d0 + cgs * 4 + (l & 3), is = cs * N + ds;
                af ^= fbits(m.R[(size_t) k * D + is]);
                af ^= fbits(m.bo[(size_t) k * N + ds]);
                af ^= fbits(m.ip[k * HC + cs]);
                af ^= fbits(m.wn[is]);
                af ^= fbits(m.rs[k * HC + cs]);
            }
        }
        for (int x = t; x < T * LR; x += UTH) lo[x] = m.lo[x];
        item.barrier(sycl::access::fence_space::local_space);
        const float* lq = lo + q * UKQ;
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int k = 0; k < T; ++k) {
                af ^= x4(*reinterpret_cast<const sycl::uint4*>(lq + k * LR + 8 * j));
                af ^= x4(*reinterpret_cast<const sycl::uint4*>(lq + k * LR + 8 * j + 4));
            }
    }
    uint32_t wf = 0u;
    if constexpr (MODE == 2) {
#pragma unroll
        for (int j = 0; j < NJ; ++j) wf ^= q8x8_fold(qv[j / 2], j & 1, (float) sycl::bit_cast<sycl::half>(dv[j / 4]));
    } else {
#pragma unroll
        for (int j = 0; j < NQ; ++j) wf ^= x4(qv[j]);
#pragma unroll
        for (int b = 0; b < NB; ++b) wf ^= (uint32_t) dv[b];
    }
    uint32_t s = wf ^ af;
    pad[t & 63] = sycl::bit_cast<float>(s);
    s = lanes_xor(sgp, s);
    if (lane == 0) m.sink[kUpSink + g * USG + sg] = s;
}

template <int MODE, bool DN2, int T>
void launch_pair(sycl::queue* st, const Ptrs& m, const uint8_t* qd, const uint8_t* qu) {
    // the real launches' local memory: down xs = max(T * SLAB, DSG * T * RPG) floats (down2: T * SLAB and the partials T * DTH * RPL),
    // red DSG * 8 floats and the flag; up lo, pr UQ * T * 64 and gs T * HC * UP_COLS floats
    constexpr size_t XS = DN2 ? (size_t) T * SLAB : (size_t) (T * SLAB > DSG * T * RPG ? T * SLAB : DSG * T * RPG);
    constexpr size_t PADD = 129 + (DN2 ? (size_t) DTH * T * RPL : 0), PADU = (size_t) UQ * T * 64 + (size_t) T * HC * UP_COLS;
    st->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> xs(sycl::range<1>(XS), cgh);
        sycl::local_accessor<float, 1> pad(sycl::range<1>(PADD), cgh);
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, (size_t) NRG * NSLAB * DTH), sycl::range<3>(1, 1, DTH)),
                         [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(16)]] {
                             down_strip<MODE, DN2, T>(item, m, qd, xs.get_multi_ptr<sycl::access::decorated::no>().get(),
                                                      pad.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
    });
    st->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> lo(sycl::range<1>((size_t) T * LR), cgh);
        sycl::local_accessor<float, 1> pad(sycl::range<1>(PADU), cgh);
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, (size_t) (N / UP_COLS) * UTH), sycl::range<3>(1, 1, UTH)),
                         [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(16)]] {
                             up_strip<MODE, T>(item, m, qu, lo.get_multi_ptr<sycl::access::decorated::no>().get(),
                                               pad.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
    });
}
template <int MODE, bool DN2>
void read_t(int T, sycl::queue* st, const Ptrs& m, const uint8_t* qd, const uint8_t* qu) {
    switch (T) {
        case 1: launch_pair<MODE, DN2, 1>(st, m, qd, qu); break;
        case 2: launch_pair<MODE, DN2, 2>(st, m, qd, qu); break;
        case 3: launch_pair<MODE, DN2, 3>(st, m, qd, qu); break;
        case 4: launch_pair<MODE, DN2, 4>(st, m, qd, qu); break;
        case 5: launch_pair<MODE, DN2, 5>(st, m, qd, qu); break;
        case 6: launch_pair<MODE, DN2, 6>(st, m, qd, qu); break;
        case 7: launch_pair<MODE, DN2, 7>(st, m, qd, qu); break;
        default: launch_pair<MODE, DN2, 8>(st, m, qd, qu); break;
    }
}
// one stripped read: the two kernels in order on `st`, as the real read launches them
void read(int mode, bool dn2, int T, sycl::queue* st, const Ptrs& m, const uint8_t* qd, const uint8_t* qu) {
    if (mode == 1) { if (dn2) read_t<1, true>(T, st, m, qd, qu); else read_t<1, false>(T, st, m, qd, qu); }
    else if (mode == 2) { if (dn2) read_t<2, true>(T, st, m, qd, qu); else read_t<2, false>(T, st, m, qd, qu); }
    else { if (dn2) read_t<3, true>(T, st, m, qd, qu); else read_t<3, false>(T, st, m, qd, qu); }
}

}  // namespace grs
}  // namespace

uint64_t fnv_add(uint64_t h, const std::vector<float>& v) {
    const uint32_t* u = (const uint32_t*) v.data();
    for (size_t i = 0; i < v.size(); ++i) { h ^= u[i]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char** argv) {
    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--bench") bench = true;
        else if (std::string(argv[i]) != "--selftest") { std::fprintf(stderr, "usage: gr_q8_parity [--selftest] | --bench\n"); return 2; }
    }
    const float eps = 1e-6f;
    std::mt19937 rng(20261016u);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    // the weights: BF16 (as the pack holds them) and their Q8_0 blocks in the read's split layout
    std::vector<float> w_norm((size_t) D);
    for (auto& x : w_norm) x = 1.0f + 0.1f * gauss(rng);
    std::vector<uint16_t> w_down((size_t) LR * D), w_up((size_t) D * LR), w_inj((size_t) HC * D);
    for (auto& x : w_down) x = bf16_of(gauss(rng) * 0.05f);
    for (auto& x : w_up) x = bf16_of(gauss(rng) * 0.05f);
    for (auto& x : w_inj) x = bf16_of(gauss(rng) * 0.02f);
    const std::vector<uint8_t> q_down = q8_0_of(w_down), q_up = q8_0_of(w_up);
    std::vector<uint8_t> s_down(q_down.size()), s_up(q_up.size());
    k::fused_gr_q8_0_split_host(q_down.data(), s_down.data(), LR, D);
    k::fused_gr_q8_0_split_host(q_up.data(), s_up.data(), D, LR);
    // the reference's weights: the Q8_0 values for the Q8_0 read, the BF16 ones for the inject rows
    const std::vector<float> wd_q = dequant(q_down), wu_q = dequant(q_up);
    std::vector<float> wi_f(w_inj.size());
    for (size_t i = 0; i < w_inj.size(); ++i) wi_f[i] = bf16_to_f(w_inj[i]);

    float* d_norm = up(w_norm);
    uint16_t *d_down = up(w_down), *d_up = up(w_up), *d_inj = up(w_inj);
    uint8_t *d_qdown = up(s_down), *d_qup = up(s_up);

    // the rows: R with stream scales 1, 4, 16, 64 (so a per-stream norm is a claim with content), bo, inj_prev
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), ip((size_t) TM * HC);
    for (int t = 0; t < TM; ++t)
        for (int c = 0; c < HC; ++c)
            for (int d = 0; d < N; ++d) R[(size_t) t * D + c * N + d] = gauss(rng) * (float) (1 << (2 * c));
    for (auto& x : bo) x = gauss(rng) * 0.3f;
    for (auto& x : ip) x = gauss(rng) * 0.3f;
    float *d_R = dev<float>((size_t) TM * D), *d_bo = up(bo), *d_ip = up(ip);
    float *d_lo = dev<float>((size_t) TM * LR), *d_rs = dev<float>((size_t) TM * HC), *d_io = dev<float>((size_t) TM * HC);
    float *d_mixed = dev<float>((size_t) TM * N), *d_xn = dev<float>((size_t) TM * D);

    dpct::queue_ptr stream = dpct::get_current_device().create_queue(true);
    auto reset = [&]() { stream->memcpy(d_R, R.data(), R.size() * sizeof(float)); };
    auto args_for = [&](int T, bool apply, bool inject, bool q8) {
        std::vector<k::FusedGrArgs> a(T);
        for (int t = 0; t < T; ++t) {
            k::FusedGrArgs& x = a[t];
            x.R = d_R + (size_t) t * D; x.R_out = d_R + (size_t) t * D; x.apply = apply;
            x.bo_prev = d_bo + (size_t) t * N; x.inj_prev = d_ip + (size_t) t * HC;
            x.w_norm = d_norm; x.w_down = d_down; x.w_up = d_up; x.w_inject = inject ? d_inj : nullptr;
            x.q8_down = q8 ? d_qdown : nullptr; x.q8_up = q8 ? d_qup : nullptr;
            x.eps = eps; x.lo = d_lo + (size_t) t * LR; x.rs = d_rs + (size_t) t * HC;
            x.inject_out = inject ? d_io + (size_t) t * HC : nullptr; x.mixed = d_mixed + (size_t) t * N;
        }
        return a;
    };
    auto snapshot = [&](int T, bool inject) {
        stream->wait();
        Out o;
        o.r_out = down(d_R, (size_t) T * D); o.lo = down(d_lo, (size_t) T * LR); o.rs = down(d_rs, (size_t) T * HC);
        o.inject = inject ? down(d_io, (size_t) T * HC) : std::vector<float>((size_t) T * HC, 0.0f);
        o.mixed = down(d_mixed, (size_t) T * N);
        return o;
    };
    auto clear = [&]() {
        stream->memset(d_lo, 0, (size_t) TM * LR * sizeof(float));
        stream->memset(d_rs, 0, (size_t) TM * HC * sizeof(float));
        stream->memset(d_io, 0, (size_t) TM * HC * sizeof(float));
        stream->memset(d_mixed, 0, (size_t) TM * N * sizeof(float));
    };

    if (bench) {
        // --bench: NB distinct Q8_0 weight copies (the window reads another layer's projections every call; 3.3 MB
        // each fits in L2), one graph of NB reads per T (pending write on, no inject: a layer read), us per read
        // (median of 30); then an FNV hash of every output of direct reads (T 1..8 x pending write x inject) to
        // compare two builds or two settings bit for bit.
        constexpr int NB = 24;
        std::vector<uint8_t*> qd(NB), qu(NB);
        for (int b = 0; b < NB; ++b) { qd[b] = up(s_down); qu[b] = up(s_up); }
        const double wbytes = (double) (s_down.size() + s_up.size());
        // STRATA_HC_BENCH_STRIP=1|2|3 (unset or 0: the real read, as always): the stripped kernels (grs::), mirroring the kernel
        // pair the real read runs (STRATA_HC_Q8_DOWN2 as down2_on reads it); STRIP 2 holds Q8_0's one decode step (q8x8)
        const char* strip_e = std::getenv("STRATA_HC_BENCH_STRIP");
        const int strip = strip_e && *strip_e ? std::atoi(strip_e) : 0;
        if (strip < 0 || strip > 3) { std::fprintf(stderr, "--bench: STRATA_HC_BENCH_STRIP must be 0, 1, 2 or 3\n"); return 2; }
        const char* dn2_e = std::getenv("STRATA_HC_Q8_DOWN2");
        const bool dn2 = dn2_e != nullptr && dn2_e[0] != '\0' && dn2_e[0] != '0';   // as down2_on
        grs::Ptrs gp{d_R, d_bo, d_ip, d_norm, d_lo, d_rs, nullptr};
        if (strip != 0) {
            gp.sink = dev<uint32_t>(8192);
            const char* what[] = {"", "reads: weights, and the state rows (R, bo, gate input, w_norm, rs, lo) staged and read as the real kernels do; no arithmetic",
                                  "reads + decode, no products: STRIP=1 plus the Q8_0 decode (byte extract, int to float, scale multiply) folded by bit pattern",
                                  "weight reads only: the quants and the fp16 scales of both kernels"};
            std::printf("gr_q8 --bench: STRIP=%d (%s); layer read (pending write on, no inject rows); mirrors gr_q8_down%s_kernel + gr_q8_up_kernel (STRATA_HC_Q8_DOWN2 %s)\n",
                        strip, what[strip], dn2 ? "2" : "", dn2 ? "on" : "off");
            std::printf("gr_q8 --bench: STRIP=%d: no arithmetic with the activations, no reductions, partials, arrival counter, lo / rs / mixed / R_out writes; "
                        "outputs are not results; compare with the same run without the switch\n", strip);
            std::printf("gr_q8 --bench: weight bytes a read loads: %.0f (down %zu + up %zu); GB/s below = bytes / us\n", wbytes, s_down.size(), s_up.size());
        }
        for (int T : {1, 2, 3, 4, 6, 8}) {
            reset();
            auto aq = args_for(T, true, false, true);
            dpct::experimental::command_graph_ptr g = nullptr;
            dpct::experimental::begin_recording(stream);
            for (int b = 0; b < NB; ++b) {
                if (strip != 0) { grs::read(strip, dn2, T, stream, gp, qd[b], qu[b]); continue; }
                for (auto& x : aq) { x.q8_down = qd[b]; x.q8_up = qu[b]; }
                k::fused_gr_read_multi(aq.data(), T, d_xn, stream);
            }
            dpct::experimental::end_recording(stream, &g);
            auto* gx = new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(g->finalize());
            std::vector<double> us;
            for (int i = 0; i < 40; ++i) {
                if (i % 8 == 0) { reset(); stream->wait(); }
                sycl::event e0 = stream->ext_oneapi_submit_barrier();
                stream->ext_oneapi_graph(*gx);
                sycl::event e1 = stream->ext_oneapi_submit_barrier();
                e1.wait();
                const double ns = (double) (e1.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                            e0.get_profiling_info<sycl::info::event_profiling::command_start>());
                if (i >= 10) us.push_back(ns / 1e3 / NB);
            }
            std::sort(us.begin(), us.end());
            std::printf("gr_q8 read T=%d: %8.2f us per read (min %8.2f)  %6.1f GB/s of Q8_0 weights", T, us[us.size() / 2],
                        us[0], wbytes / us[us.size() / 2] / 1e3);
            if (strip != 0) std::printf("  strip %d%s wbytes %.0f", strip, dn2 ? " down2" : "", wbytes);
            std::printf("\n");
            delete gx;
            delete g;
        }
        if (strip != 0) {
            std::printf("gr_q8 outputs hash skipped (STRIP=%d: the stripped kernels' stores are not results)\n", strip);
        } else {
        uint64_t h = 1469598103934665603ull;
        for (int T = 1; T <= TM; ++T)
            for (int apply = 0; apply < 2; ++apply)
                for (int inject = 0; inject < 2; ++inject) {
                    clear(); reset();
                    auto aq = args_for(T, apply, inject, true);
                    k::fused_gr_read_multi(aq.data(), T, d_xn, stream);
                    const Out o = snapshot(T, inject);
                    h = fnv_add(fnv_add(fnv_add(fnv_add(fnv_add(h, o.r_out), o.lo), o.rs), o.inject), o.mixed);
                }
        std::printf("gr_q8 outputs hash %016llx\n", (unsigned long long) h);
        }
        if (gp.sink != nullptr) sycl::free(gp.sink, dpct::get_in_order_queue());
        for (int b = 0; b < NB; ++b) { sycl::free(qd[b], dpct::get_in_order_queue()); sycl::free(qu[b], dpct::get_in_order_queue()); }
        dpct::get_current_device().destroy_queue(stream);
        return 0;
    }

    k::fused_gr_set_q8_down2(0);   // checks 1-4: the first down kernel, whatever the environment says
    int bad = 0, cases = 0;
    const double tol = 1e-4;   // FP32 summation order against FP64: a structural error is orders of magnitude larger
    double worst_lo = 0, worst_rs = 0, worst_inj = 0, worst_mixed = 0, min_obs = 1e9, max_obs = 0;
    for (int T = 1; T <= TM; ++T)
        for (int apply = 0; apply < 2; ++apply)
            for (int inject = 0; inject < 2; ++inject) {
                ++cases;
                const auto want = reference(T, apply, inject, R, bo, ip, w_norm, wd_q, wu_q, wi_f, eps);
                // the BF16 read (the default) on the same rows
                clear(); reset();
                auto ab = args_for(T, apply, inject, false);
                k::fused_gr_read_multi(ab.data(), T, d_xn, stream);
                const Out bf16 = snapshot(T, inject);
                // the Q8_0 read
                clear(); reset();
                auto aq = args_for(T, apply, inject, true);
                k::fused_gr_read_multi(aq.data(), T, d_xn, stream);
                const Out q8 = snapshot(T, inject);
                const double rl = rel_worst(q8.lo, want.lo), rr = rel_worst(q8.rs, want.rs),
                             ri = inject ? rel_worst(q8.inject, want.inject) : 0.0, rm = rel_worst(q8.mixed, want.mixed);
                worst_lo = std::max(worst_lo, rl); worst_rs = std::max(worst_rs, rr);
                worst_inj = std::max(worst_inj, ri); worst_mixed = std::max(worst_mixed, rm);
                bool ok = rl <= tol && rr <= tol && ri <= tol && rm <= tol;
                if (!ok) std::printf("  T=%d apply=%d inject=%d: Q8_0 read vs FP64 reference: lo %.2e rs %.2e inject %.2e mixed %.2e\n",
                                     T, apply, inject, rl, rr, ri, rm);
                if (!same(q8.r_out, bf16.r_out)) {
                    std::printf("  T=%d apply=%d inject=%d: R_out of the Q8_0 read differs from the BF16 read's\n", T, apply, inject);
                    ok = false;
                }
                // observable: the Q8_0 weights, not the BF16 ones, were read
                const double obs = rel_rms(q8.mixed, bf16.mixed);
                min_obs = std::min(min_obs, obs); max_obs = std::max(max_obs, obs);
                if (obs < 1e-5 || obs > 1e-1) {
                    std::printf("  T=%d apply=%d inject=%d: Q8_0 read vs BF16 read: mixed rel RMS %.2e (expected the Q8_0 rounding, ~1e-3)\n",
                                T, apply, inject, obs);
                    ok = false;
                }
                // a captured graph, replayed twice: bitwise the direct call
                dpct::experimental::command_graph_ptr g = nullptr;
                dpct::experimental::command_graph_exec_ptr gx = nullptr;
                dpct::experimental::begin_recording(stream);
                k::fused_gr_read_multi(aq.data(), T, d_xn, stream);
                dpct::experimental::end_recording(stream, &g);
                gx = new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(g->finalize());
                for (int rep = 0; rep < 2; ++rep) {
                    clear(); reset();
                    stream->ext_oneapi_graph(*gx);
                    const Out rp = snapshot(T, inject);
                    if (!same(rp.r_out, q8.r_out) || !same(rp.lo, q8.lo) || !same(rp.rs, q8.rs) || !same(rp.inject, q8.inject) ||
                        !same(rp.mixed, q8.mixed)) {
                        std::printf("  T=%d apply=%d inject=%d: graph replay %d differs from the direct Q8_0 read\n", T, apply, inject, rep);
                        ok = false;
                    }
                }
                delete gx;
                delete g;
                // STRATA_HC_Q8_DOWN2: the second down kernel, two direct reads in a row, bitwise the first kernel's
                k::fused_gr_set_q8_down2(1);
                for (int rep = 0; rep < 2; ++rep) {
                    clear(); reset();
                    k::fused_gr_read_multi(aq.data(), T, d_xn, stream);
                    const Out v2 = snapshot(T, inject);
                    if (!same(v2.r_out, q8.r_out) || !same(v2.lo, q8.lo) || !same(v2.rs, q8.rs) || !same(v2.inject, q8.inject) ||
                        !same(v2.mixed, q8.mixed)) {
                        std::printf("  T=%d apply=%d inject=%d: STRATA_HC_Q8_DOWN2 read %d differs from the Q8_0 read (lo %d rs %d inject %d mixed %d)\n",
                                    T, apply, inject, rep, same(v2.lo, q8.lo) ? 0 : 1, same(v2.rs, q8.rs) ? 0 : 1,
                                    same(v2.inject, q8.inject) ? 0 : 1, same(v2.mixed, q8.mixed) ? 0 : 1);
                        ok = false;
                    }
                }
                k::fused_gr_set_q8_down2(0);
                if (!ok) ++bad;
            }
    std::printf("  Q8_0 read vs FP64 reference on its Q8_0 values, %d cases (T 1..8 x pending write x inject): worst rel lo %.2e rs %.2e "
                "inject %.2e mixed %.2e (tolerance %.0e); R_out bitwise the BF16 read's; graph replays bitwise\n",
                cases, worst_lo, worst_rs, worst_inj, worst_mixed, tol);
    std::printf("  Q8_0 read vs BF16 read, mixed rel RMS %.2e..%.2e: the Q8_0 weights are read (not bitwise, by their rounding)\n",
                min_obs, max_obs);
    std::printf("  STRATA_HC_Q8_DOWN2: the second down kernel bitwise the first in every case that did not say otherwise above\n");

    // rows not at the fixed strides (token 1's mixed before token 0's): the BF16 read, bitwise
    {
        const int T = 2;
        clear(); reset();
        auto ab = args_for(T, true, true, false);
        std::swap(ab[0].mixed, ab[1].mixed);
        k::fused_gr_read_multi(ab.data(), T, d_xn, stream);
        const Out bf16 = snapshot(T, true);
        clear(); reset();
        auto aq = args_for(T, true, true, true);
        std::swap(aq[0].mixed, aq[1].mixed);
        k::fused_gr_read_multi(aq.data(), T, d_xn, stream);
        const Out q8 = snapshot(T, true);
        const bool ok = same(q8.r_out, bf16.r_out) && same(q8.lo, bf16.lo) && same(q8.rs, bf16.rs) &&
                        same(q8.inject, bf16.inject) && same(q8.mixed, bf16.mixed);
        std::printf("  rows not at the fixed strides: the BF16 read, bitwise: %s\n", ok ? "pass" : "FAIL");
        if (!ok) ++bad;
    }

    k::fused_gr_set_q8_down2(-1);
    std::printf("gr_q8_parity: %s (%d failing of %d cases)\n", bad ? "FAIL" : "pass", bad, cases + 1);
    dpct::get_current_device().destroy_queue(stream);
    for (void* p : {(void*) d_norm, (void*) d_down, (void*) d_up, (void*) d_inj, (void*) d_qdown, (void*) d_qup, (void*) d_R,
                    (void*) d_bo, (void*) d_ip, (void*) d_lo, (void*) d_rs, (void*) d_io, (void*) d_mixed, (void*) d_xn})
        sycl::free(p, dpct::get_in_order_queue());
    return bad ? 1 : 0;
}
