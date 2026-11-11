// 151-b70: fused int8 DPAS expert products (ESIMD, Battlemage) - see include/strata/prefill/esimd_moe.hpp and
// docs/INTEL_INT8_EXPERTS.md (the behaviour note).
//
//  * Tile.  One hardware thread owns 16 output rows (a DPAS N tile; for gate/up the 16 virtual rows alternate gate/up) and up to
//    MT x 8 activation rows (MT = 2, 4 or 8, picked per group from the experts' mean row count).  It walks K in steps of 32:
//    decode the 16 weight rows' 32 values into the B operand (registers only), then for each of its row tiles one s8 DPAS 8x16x32
//    into a zeroed int32 tile, converted to float and scaled by (weight scale of the row and 32-block) x (activation scale of the
//    row and 32-block) into a float accumulator.  There is no shared memory and no barrier.
//  * Decode once per tile: the B operand is decoded once per thread and applied to all its MT row tiles.  An expert with more rows
//    than 8 x MT is split over row-block threads, each decoding again.  A thread's accumulators are MT tiles of 512 bytes in an
//    8 KB register file (256 GRF of 32 bytes): MT 12, or two output tiles per thread, spilled (3-15 KB) and ran 2-5x slower.
//  * Scales.  Activations: q8_1 as llama.cpp (d = amax / 127, q = round half away from zero), one float per row and 32 of K.
//    Weights: IQ3_S d (1 + 2 s), IQ3_XXS d (2 s + 1) / 4, IQ2_S d (2 s + 1) / 8 per 16 (two DPAS per 32 with the other half
//    zeroed), IQ4_NL d per 32, Q2_0 d per 64; the int8 values are the codebook entries with their signs, never rounded.  The int32
//    dot product of a 32-block is exact and exactly convertible to float.  The float accumulation order is fixed (K ascending).
//  * Edge cases.  Rows are padded to tiles of 8 with zeros (stores are masked to the real rows); a group of up to 16 experts per
//    launch; K a multiple of 256 (IQ3_S / IQ3_XXS / IQ2_S) / 64 (Q2_0) / 32 (IQ4_NL); output rows a multiple of 16.  Blob reads
//    never go past a matrix end.
//  * Fallback.  available() is false without the s8 8x16x32 DPAS (a non-Battlemage device) and the caller keeps the dequantize +
//    oneMKL path; gate_up()/down() return false for a type / shape they do not handle.
//  * ESIMD on this compiler (oneAPI 2026.1, bmg-g31): a dynamic index into the kernel-argument arrays, and `/` or `%` by a run-time
//    value, gave wrong values / garbage pointers (the first weight read faulted the GPU); the kernels use constant-index reads
//    (pick_expert) and 2-D grids.  gather<T, N, VS> with VS > 1 is not available on Battlemage.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <type_traits>
#include <string>
#include <vector>
#include <unordered_map>
#include "strata/sycl_queue.hpp"
#include "strata/prefill/esimd_moe.hpp"

#ifdef STRATA_I8_NO_ESIMD
// an AOT build for a device without the s8 8x16x32 DPAS (Alchemist, Xe-HPC): no ESIMD kernels, available() is false
namespace strata::prefill::i8 {
void make_layout(Layout& lo, const int32_t* cnt, int n) {
    lo.n = n;
    int r = 0, t = 0;
    for (int i = 0; i < n; ++i) { lo.cnt[i] = cnt[i]; lo.row0[i] = r; lo.tile0[i] = t; r += cnt[i]; t += (cnt[i] + 7) / 8; }
    lo.row0[n] = r;
    lo.tile0[n] = t;
}
bool available(void*) { return false; }
bool battlemage(void*) { return false; }
bool any_battlemage() { return false; }
size_t xq_bytes(const Layout& lo, int K) { return (size_t) lo.tile0[lo.n] * 8 * K; }
size_t dx_bytes(const Layout& lo, int K) { return (size_t) lo.tile0[lo.n] * (K / 32) * 8 * sizeof(float); }
bool workspace(const Layout&, int, int, Workspace&, void*) { return false; }
void quantize_rows(const uint16_t*, const Layout&, int, int8_t*, float*, void*) {}
bool gate_up(int, const uint8_t* const*, size_t, int, const Layout&, int, const int8_t*, const float*, float*, void*) { return false; }
void swiglu_quantize(const float*, const Layout&, int, int8_t*, float*, void*) {}
bool down(int, const uint8_t* const*, size_t, int, const Layout&, int, const int8_t*, const float*, float*, void*) { return false; }
bool f16_handles(int, int, int, int, void*) { return false; }
bool gate_up_f16(int, const uint8_t* const*, size_t, int, const Layout&, int, const uint16_t*, uint16_t*, void*) { return false; }
bool down_f16(int, const uint8_t* const*, size_t, int, const Layout&, int, const uint16_t*, float*, void*) { return false; }
bool gate_up_f16_n(int, const uint8_t* const*, size_t, int, const int32_t*, int, int, const uint16_t*, uint16_t*, void*, int, uint64_t) { return false; }
bool down_f16_n(int, const uint8_t* const*, size_t, int, const int32_t*, int, int, const uint16_t*, float*, void*, int, uint64_t) { return false; }
}  // namespace strata::prefill::i8
#else
namespace strata::prefill::i8 {
namespace {
#define GGML_TABLE_BEGIN(type, name, size) static const type name[size] = {
#define GGML_TABLE_END() };
#include "iq_tables.inc"
#undef GGML_TABLE_BEGIN
#undef GGML_TABLE_END

using namespace sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using half = sycl::half;

// ggml types handled
constexpr int T_Q8_0 = 8, T_IQ2XXS = 16, T_IQ2XS = 17, T_IQ3XXS = 18, T_IQ4NL = 20, T_IQ3S = 21, T_IQ2S = 22, T_IQ4XS = 23, T_Q2_0 = 42;
template <int F> constexpr int blk_bytes() {
    return F == T_IQ4XS ? 136 : F == T_IQ3S ? 110 : F == T_IQ3XXS ? 98 : F == T_IQ2S ? 82 : F == T_IQ2XS ? 74 : F == T_IQ2XXS ? 66 : F == T_Q8_0 ? 34 : 18;
}
constexpr int block_bytes_of(int f) {
    return f == T_IQ4XS ? 136 : f == T_IQ3S ? 110 : f == T_IQ3XXS ? 98 : f == T_IQ2S ? 82 : f == T_IQ2XS ? 74 : f == T_IQ2XXS ? 66 : f == T_Q8_0 ? 34 : 18;
}
constexpr int block_vals_of(int f) {
    return (f == T_IQ4XS || f == T_IQ3S || f == T_IQ3XXS || f == T_IQ2S || f == T_IQ2XS || f == T_IQ2XXS) ? 256 : f == T_Q2_0 ? 64 : 32;
}

// One device table buffer (uint32 units): the codebooks the decoders gather from.
constexpr int O_IQ3S = 0;        // iq3s_grid[512]
constexpr int O_IQ3XXS = 512;    // iq3xxs_grid[256]
constexpr int O_IQ2S = 768;      // iq2s_grid[1024] as 2048 dwords (low dword of an entry first)
constexpr int O_KSIGNS = 2816;   // ksigns_iq2xs[128], one byte in the low byte of each dword
constexpr int O_IQ4 = 2944;      // [256]: byte b -> kvalues_iq4nl[b & 15] in byte 0, kvalues_iq4nl[b >> 4] in byte 1
constexpr int O_Q2 = 3200;       // [256]: byte b -> its four 2-bit codes minus 1, as four int8 (Q2_0 values)
constexpr int O_IQ2XS = 3456;    // iq2xs_grid[512] as 1024 dwords
constexpr int O_IQ2XXS = 4480;   // iq2xxs_grid[256] as 512 dwords
constexpr int N_TABLE = 4992;

uint32_t* tables(sycl::queue& q) {
    static std::mutex mu;
    static std::unordered_map<sycl::device, uint32_t*> per_dev;   // a layer split has one prompt path per card
    std::lock_guard<std::mutex> lk(mu);
    uint32_t*& t = per_dev[q.get_device()];
    if (!t) {
        std::vector<uint32_t> h(N_TABLE, 0);
        for (int i = 0; i < 512; ++i) h[O_IQ3S + i] = iq3s_grid[i];
        for (int i = 0; i < 256; ++i) h[O_IQ3XXS + i] = iq3xxs_grid[i];
        for (int i = 0; i < 1024; ++i) { h[O_IQ2S + 2 * i] = (uint32_t) iq2s_grid[i]; h[O_IQ2S + 2 * i + 1] = (uint32_t) (iq2s_grid[i] >> 32); }
        for (int i = 0; i < 128; ++i) h[O_KSIGNS + i] = ksigns_iq2xs[i];
        for (int b = 0; b < 256; ++b) {
            h[O_IQ4 + b] = (uint32_t) (uint8_t) kvalues_iq4nl[b & 15] | ((uint32_t) (uint8_t) kvalues_iq4nl[b >> 4] << 8);
            uint32_t q2 = 0;
            for (int k = 0; k < 4; ++k) q2 |= (uint32_t) (uint8_t) (int8_t) (((b >> (2 * k)) & 3) - 1) << (8 * k);
            h[O_Q2 + b] = q2;
        }
        for (int i = 0; i < 512; ++i) { h[O_IQ2XS + 2 * i] = (uint32_t) iq2xs_grid[i]; h[O_IQ2XS + 2 * i + 1] = (uint32_t) (iq2xs_grid[i] >> 32); }
        for (int i = 0; i < 256; ++i) { h[O_IQ2XXS + 2 * i] = (uint32_t) iq2xxs_grid[i]; h[O_IQ2XXS + 2 * i + 1] = (uint32_t) (iq2xxs_grid[i] >> 32); }
        uint32_t* d = sycl::malloc_device<uint32_t>(N_TABLE, q);
        q.memcpy(d, h.data(), N_TABLE * 4);
        q.wait();
        t = d;
    }
    return t;
}

// unaligned 32-bit read at a 2-byte aligned byte offset (per lane): the two dwords around it, one 2-dword gather
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> ld32(const uint8_t* base, simd<uint32_t, 16> off) {
    simd<uint32_t, 16> a = off & ~3u;
    simd<uint32_t, 32> v = gather<uint32_t, 32, 2>(reinterpret_cast<const uint32_t*>(base), a);   // both dwords in one message
    simd<uint32_t, 16> lo = v.template select<16, 1>(0), hi = v.template select<16, 1>(16);
    simd<uint32_t, 16> sh = (off & 2u) << 3;   // 0 or 16
    return (lo >> sh) | ((hi << 16) << (16u - sh));
}
// the same with the second dword clamped to byte offset `lim` (the last dword of the matrix): the down matrix ends the expert blob, and a block
// whose last field is not dword aligned would otherwise read up to 3 bytes past it
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> ld32c(const uint8_t* base, simd<uint32_t, 16> off, uint32_t lim) {
    simd<uint32_t, 16> a = off & ~3u;
    // one 2-dword gather; a lane whose dword is the last one (a == lim) reads the pair before it and takes its second dword
    // for both halves, as the two clamped reads did
    const simd_mask<16> end = a + 4u > lim;
    simd<uint32_t, 16> b = a;
    b.merge(simd<uint32_t, 16>(lim - 4u), end);
    simd<uint32_t, 32> v = gather<uint32_t, 32, 2>(reinterpret_cast<const uint32_t*>(base), b);
    simd<uint32_t, 16> lo = v.template select<16, 1>(0), hi = v.template select<16, 1>(16);
    lo.merge(hi, end);
    simd<uint32_t, 16> sh = (off & 2u) << 3;
    return (lo >> sh) | ((hi << 16) << (16u - sh));
}
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> ld16(const uint8_t* base, simd<uint32_t, 16> off) {
    simd<uint32_t, 16> a = off & ~3u;
    simd<uint32_t, 16> lo = gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(base), a);
    return (lo >> ((off & 2u) << 3)) & 0xffffu;
}
SYCL_ESIMD_FUNCTION inline simd<float, 16> half_of(simd<uint32_t, 16> h16) {
    simd<uint16_t, 16> lo = h16.template bit_cast_view<uint16_t>().template select<16, 2>(0);
    simd<half, 16> hh = lo.template bit_cast_view<half>();
    simd<float, 16> f = hh;
    return f;
}
SYCL_ESIMD_FUNCTION inline simd<int, 128> dpas_s8(simd<uint32_t, 128> b, simd<int, 64> a) {
    simd<int, 128> bi = b.template bit_cast_view<int>();
    return xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(simd<int, 128>(0), bi, a);
}
// sign nibble -> four negation bytes: (g ^ mask) + m negates the bytes whose sign bit is set (g is never 0xFF per byte)
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> apply_signs(simd<uint32_t, 16> g, simd<uint32_t, 16> nib) {
    simd<uint32_t, 16> m = ((nib * 0x00204081u) & 0x01010101u);
    simd<uint32_t, 16> mask = m * 255u;
    return (g ^ mask) + m;
}

// ksigns_iq2xs[s] for s < 128: s with bit 7 set to the parity of its bits (an even number of minus signs per 8 values)
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> ksigns(simd<uint32_t, 16> s7) {
    simd<uint32_t, 16> c = cbit(s7);
    return s7 | ((c & 1u) << 7);
}

struct GArgs {
    const uint8_t* blob[kMaxGroup];
    uint32_t sec_off;                 // gate/up: byte offset of the up rows; down: byte offset of the down rows
    int n;
    int row0[kMaxGroup + 1];
    int tile0[kMaxGroup + 1];
    int rb0[kMaxGroup + 1];           // first row block of expert i
    const int8_t* Xq;
    const float* dx;
    float* Y;
    const uint32_t* tab;
    int ntiles;                       // output tiles of 16
    int nkb;                          // K / 32
    int rowb;                         // bytes per weight row
    int ldy;
    uint32_t lim;                     // byte offset of the last dword of the matrix the clamped reads (ld32c) stay inside
    int dbg;                          // bring-up switches (STRATA_I8_DBG): 1 skip the decode, 2 skip the activation loads, 4 skip the stores, 8 skip the K loop
};

// The expert a work item belongs to, with constant-index reads of the kernel argument arrays.
struct Pick { int e, row0, cnt, tile0, rb0; const uint8_t* blob; };
SYCL_ESIMD_FUNCTION inline Pick pick_expert(const GArgs& a, int rbg) {
    Pick p{0, a.row0[0], a.row0[1] - a.row0[0], a.tile0[0], a.rb0[0], a.blob[0]};
#pragma unroll
    for (int i = 1; i < kMaxGroup; ++i)
        if (i < a.n && rbg >= a.rb0[i]) p = Pick{i, a.row0[i], a.row0[i + 1] - a.row0[i], a.tile0[i], a.rb0[i], a.blob[i]};
    return p;
}

int fill_args_mt(GArgs& a, const Layout& lo, int MT) {
    int rb = 0;
    for (int i = 0; i < lo.n; ++i) {
        a.row0[i] = lo.row0[i];
        a.tile0[i] = lo.tile0[i];
        a.rb0[i] = rb;
        const int ntl = (lo.cnt[i] + 7) / 8;
        rb += (ntl + MT - 1) / MT;
    }
    a.row0[lo.n] = lo.row0[lo.n];
    a.tile0[lo.n] = lo.tile0[lo.n];
    a.rb0[lo.n] = rb;
    a.n = lo.n;
    return rb;
}

template <int F, int MT, int G> struct TagGu;
template <int F, int MT, int G> struct TagDn;

// ------------------------------------------------------------------------------------------------ gate/up
template <int F, int MT, int G>
void launch_gu(sycl::queue& q, const GArgs a, int nrowblocks) {
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<G>};
    constexpr int NS = (F == T_IQ2S || F == T_IQ2XS) ? 2 : 1;   // IQ2_S / IQ2_XS: two scales per 32 values -> two DPAS with the other half zeroed
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagGu<F, MT, G>>(sycl::range<2>((size_t) nrowblocks, (size_t) a.ntiles), props, [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {
            const int rbg = (int) id[0], nt = (int) id[1];
            const Pick pk = pick_expert(a, rbg);
            const int cnt = pk.cnt;
            const int ntl = (cnt + 7) >> 3;
            const int tl0 = (rbg - pk.rb0) * MT;
            const int nvalid = (ntl - tl0) < MT ? (ntl - tl0) : MT;
            const int tbase = pk.tile0 + tl0;
            const uint8_t* We = pk.blob;
            const uint32_t* T = a.tab;
            simd<uint32_t, 16> lane(0, 1);
            simd<uint32_t, 16> ro = (lane & 1u) * a.sec_off + ((lane >> 1) + (uint32_t) (8 * nt)) * (uint32_t) a.rowb;
            simd<float, 128 * MT> acc = 0.f;
            const int* Xi = reinterpret_cast<const int*>(a.Xq);
            for (int sb = 0; sb < ((a.dbg & 8) ? 0 : (a.nkb >> 3)); ++sb) {
                const uint32_t P = (uint32_t) sb * blk_bytes<F>();
                const simd<float, 16> d = half_of(ld16(We, ro + P));
                simd<uint32_t, 16> w0 = 0, w1 = 0, w2 = 0, w3 = 0;   // sb-level fields
                if constexpr (F == T_IQ3S) {
                    w0 = ld32(We, ro + (P + 106));   // scales
                    w1 = ld32(We, ro + (P + 66));    // qh 0..3
                    w2 = ld32(We, ro + (P + 70));    // qh 4..7
                } else if constexpr (F == T_IQ2S) {
                    w0 = ld32(We, ro + (P + 74));    // scales 0..3
                    w1 = ld32(We, ro + (P + 78));    // scales 4..7
                    w2 = ld32(We, ro + (P + 66));    // qh 0..3
                    w3 = ld32(We, ro + (P + 70));    // qh 4..7
                } else if constexpr (F == T_IQ2XS) {
                    w0 = ld32(We, ro + (P + 66));    // scales 0..3
                    w1 = ld32(We, ro + (P + 70));    // scales 4..7
                } else if constexpr (F == T_IQ4XS) {
                    w0 = ld16(We, ro + (P + 2));     // scales_h
                    w1 = ld32(We, ro + (P + 4));     // scales_l 0..3
                }
#pragma unroll 1
                for (int ib = 0; ib < 8; ++ib) {
                    const int kb = sb * 8 + ib;
                    simd<uint32_t, 128> fr0 = 0x01010101u, fr1 = 0u;
                    simd<float, 16> sw0 = 1.f, sw1 = 1.f;
                    if (!(a.dbg & 1)) {
                        if constexpr (F == T_IQ3S) {
                            simd<uint32_t, 16> q0 = ld32(We, ro + (P + 2 + 8 * ib));
                            simd<uint32_t, 16> q1 = ld32(We, ro + (P + 6 + 8 * ib));
                            simd<uint32_t, 16> sg = ld32(We, ro + (P + 74 + 4 * ib));
                            simd<uint32_t, 16> qhv = (((ib < 4) ? w1 : w2) >> (8 * (ib & 3))) & 0xffu;
                            simd<uint32_t, 16> scn = ((w0 >> (8 * (ib >> 1))) >> (4 * (ib & 1))) & 0xfu;
                            sw0 = d * simd<float, 16>(simd<int, 16>(scn * 2u + 1u));
#pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                simd<uint32_t, 16> qsw = r < 4 ? q0 : q1;
                                simd<uint32_t, 16> idx = ((qsw >> (8 * (r & 3))) & 0xffu) | (((qhv >> r) & 1u) << 8);
                                simd<uint32_t, 16> g = gather<uint32_t, 16>(T + O_IQ3S, idx << 2);
                                simd<uint32_t, 16> nib = ((sg >> (8 * (r >> 1))) >> (4 * (r & 1))) & 0xfu;
                                fr0.template select<16, 1>(16 * r) = apply_signs(g, nib);
                            }
                        } else if constexpr (F == T_IQ3XXS) {
                            simd<uint32_t, 16> q0 = ld32(We, ro + (P + 2 + 8 * ib));
                            simd<uint32_t, 16> q1 = ld32(We, ro + (P + 6 + 8 * ib));
                            simd<uint32_t, 16> aux = ld32(We, ro + (P + 66 + 4 * ib));
                            sw0 = d * simd<float, 16>(simd<int, 16>((aux >> 28) * 2u + 1u)) * 0.25f;
#pragma unroll
                            for (int il = 0; il < 4; ++il) {
                                simd<uint32_t, 16> sbyte = ksigns((aux >> (7 * il)) & 127u);
#pragma unroll
                                for (int hh = 0; hh < 2; ++hh) {
                                    const int r = 2 * il + hh;
                                    simd<uint32_t, 16> qsw = r < 4 ? q0 : q1;
                                    simd<uint32_t, 16> idx = (qsw >> (8 * (r & 3))) & 0xffu;
                                    simd<uint32_t, 16> g = gather<uint32_t, 16>(T + O_IQ3XXS, idx << 2);
                                    fr0.template select<16, 1>(16 * r) = apply_signs(g, (sbyte >> (4 * hh)) & 0xfu);
                                }
                            }
                        } else if constexpr (F == T_IQ2S) {
                            simd<uint32_t, 16> idw = ld32(We, ro + (P + 2 + 4 * ib));
                            simd<uint32_t, 16> sgw = ld32(We, ro + (P + 34 + 4 * ib));
                            simd<uint32_t, 16> qhb = (((ib < 4) ? w2 : w3) >> (8 * (ib & 3))) & 0xffu;
                            simd<uint32_t, 16> scb = (((ib < 4) ? w0 : w1) >> (8 * (ib & 3))) & 0xffu;
                            sw0 = d * simd<float, 16>(simd<int, 16>((scb & 0xfu) * 2u + 1u)) * 0.125f;
                            sw1 = d * simd<float, 16>(simd<int, 16>((scb >> 4) * 2u + 1u)) * 0.125f;
#pragma unroll
                            for (int il = 0; il < 4; ++il) {
                                simd<uint32_t, 16> idx = ((idw >> (8 * il)) & 0xffu) | ((qhb << (8 - 2 * il)) & 0x300u);
                                simd<uint32_t, 32> g2 = gather<uint32_t, 32, 2>(T + O_IQ2S, idx << 3);   // the 8-byte entry in one message
                                simd<uint32_t, 16> gl = g2.template select<16, 1>(0), gh = g2.template select<16, 1>(16);
                                simd<uint32_t, 16> sbyte = (sgw >> (8 * il)) & 0xffu;
                                simd<uint32_t, 16> vl = apply_signs(gl, sbyte & 0xfu);
                                simd<uint32_t, 16> vh = apply_signs(gh, sbyte >> 4);
                                if (il < 2) {
                                    fr0.template select<16, 1>(16 * (2 * il)) = vl;
                                    fr0.template select<16, 1>(16 * (2 * il + 1)) = vh;
                                } else {
                                    fr1.template select<16, 1>(16 * (2 * il)) = vl;
                                    fr1.template select<16, 1>(16 * (2 * il + 1)) = vh;
                                }
                            }
                            // fr0 holds values 0..15 (dwords 0..3, the rest zero), fr1 values 16..31 (dwords 4..7)
                            fr0.template select<64, 1>(64) = 0u;
                            fr1.template select<64, 1>(0) = 0u;
                        } else if constexpr (F == T_IQ2XS) {
                            simd<uint32_t, 16> q0 = ld32(We, ro + (P + 2 + 8 * ib));
                            simd<uint32_t, 16> q1 = ld32(We, ro + (P + 6 + 8 * ib));
                            simd<uint32_t, 16> scb = (((ib < 4) ? w0 : w1) >> (8 * (ib & 3))) & 0xffu;
                            sw0 = d * simd<float, 16>(simd<int, 16>((scb & 0xfu) * 2u + 1u)) * 0.125f;
                            sw1 = d * simd<float, 16>(simd<int, 16>((scb >> 4) * 2u + 1u)) * 0.125f;
#pragma unroll
                            for (int il = 0; il < 4; ++il) {
                                simd<uint32_t, 16> qsw = (il < 2) ? q0 : q1;
                                simd<uint32_t, 16> qv = (il & 1) ? (qsw >> 16) : (qsw & 0xffffu);
                                simd<uint32_t, 16> idx = qv & 511u;
                                simd<uint32_t, 32> g2 = gather<uint32_t, 32, 2>(T + O_IQ2XS, idx << 3);   // the 8-byte entry in one message
                                simd<uint32_t, 16> gl = g2.template select<16, 1>(0), gh = g2.template select<16, 1>(16);
                                simd<uint32_t, 16> sbyte = ksigns(qv >> 9);
                                simd<uint32_t, 16> vl = apply_signs(gl, sbyte & 0xfu);
                                simd<uint32_t, 16> vh = apply_signs(gh, sbyte >> 4);
                                if (il < 2) {
                                    fr0.template select<16, 1>(16 * (2 * il)) = vl;
                                    fr0.template select<16, 1>(16 * (2 * il + 1)) = vh;
                                } else {
                                    fr1.template select<16, 1>(16 * (2 * il)) = vl;
                                    fr1.template select<16, 1>(16 * (2 * il + 1)) = vh;
                                }
                            }
                            fr0.template select<64, 1>(64) = 0u;
                            fr1.template select<64, 1>(0) = 0u;
                        } else if constexpr (F == T_IQ2XXS) {
                            simd<uint32_t, 16> q0 = ld32(We, ro + (P + 2 + 8 * ib));
                            simd<uint32_t, 16> aux = ld32(We, ro + (P + 6 + 8 * ib));
                            sw0 = d * simd<float, 16>(simd<int, 16>((aux >> 28) * 2u + 1u)) * 0.125f;
#pragma unroll
                            for (int il = 0; il < 4; ++il) {
                                simd<uint32_t, 16> idx = (q0 >> (8 * il)) & 0xffu;
                                simd<uint32_t, 32> g2 = gather<uint32_t, 32, 2>(T + O_IQ2XXS, idx << 3);   // the 8-byte entry in one message
                                simd<uint32_t, 16> gl = g2.template select<16, 1>(0), gh = g2.template select<16, 1>(16);
                                simd<uint32_t, 16> sbyte = ksigns((aux >> (7 * il)) & 127u);
                                fr0.template select<16, 1>(16 * (2 * il)) = apply_signs(gl, sbyte & 0xfu);
                                fr0.template select<16, 1>(16 * (2 * il + 1)) = apply_signs(gh, sbyte >> 4);
                            }
                        } else if constexpr (F == T_IQ4XS) {
                            simd<uint32_t, 16> ls = ((w1 >> (4 * ib)) & 0xfu) | (((w0 >> (2 * ib)) & 3u) << 4);
                            sw0 = d * simd<float, 16>(simd<int, 16>(ls) - 32);
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                simd<uint32_t, 16> qd = ld32(We, ro + (P + 8 + 16 * ib + 4 * j));
                                simd<uint32_t, 16> t0 = gather<uint32_t, 16>(T + O_IQ4, (qd & 0xffu) << 2);
                                simd<uint32_t, 16> t1 = gather<uint32_t, 16>(T + O_IQ4, ((qd >> 8) & 0xffu) << 2);
                                simd<uint32_t, 16> t2 = gather<uint32_t, 16>(T + O_IQ4, ((qd >> 16) & 0xffu) << 2);
                                simd<uint32_t, 16> t3 = gather<uint32_t, 16>(T + O_IQ4, (qd >> 24) << 2);
                                fr0.template select<16, 1>(16 * j) = (t0 & 0xffu) | ((t1 & 0xffu) << 8) | ((t2 & 0xffu) << 16) | (t3 << 24);
                                fr0.template select<16, 1>(16 * (j + 4)) =
                                    ((t0 >> 8) & 0xffu) | (t1 & 0xff00u) | ((t2 & 0xff00u) << 8) | ((t3 & 0xff00u) << 16);
                            }
                        }
                    }
#pragma unroll
                    for (int i = 0; i < MT; ++i) {
                        if (i < nvalid) {
                            simd<int, 64> av = 0x01010101;
                            simd<float, 8> dxv = 1.f;
                            if (!(a.dbg & 2)) {
                                av = block_load<int, 64>(Xi + ((size_t) (tbase + i) * a.nkb + kb) * 64);
                                dxv = block_load<float, 8>(a.dx + ((size_t) (tbase + i) * a.nkb + kb) * 8);
                            }
                            simd<float, 128> tf = dpas_s8(fr0, av);
#pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                const float dxr = dxv[r];
                                acc.template select<16, 1>(128 * i + 16 * r) += tf.template select<16, 1>(16 * r) * (sw0 * dxr);
                            }
                            if constexpr (NS == 2) {
                                simd<float, 128> tg = dpas_s8(fr1, av);
#pragma unroll
                                for (int r = 0; r < 8; ++r) {
                                    const float dxr = dxv[r];
                                    acc.template select<16, 1>(128 * i + 16 * r) += tg.template select<16, 1>(16 * r) * (sw1 * dxr);
                                }
                            }
                        }
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                if (i < nvalid) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int rl = (tl0 + i) * 8 + r;
                        if (rl < cnt && !(a.dbg & 4))
                            block_store<float, 16>(a.Y + ((size_t) (pk.row0 + rl)) * a.ldy + 16 * nt, acc.template select<16, 1>(128 * i + 16 * r));
                    }
                }
            }
        });
    });
}

// ------------------------------------------------------------------------------------------------ down
template <int F, int MT, int G>
void launch_dn(sycl::queue& q, const GArgs a, int nrowblocks) {
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<G>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagDn<F, MT, G>>(sycl::range<2>((size_t) nrowblocks, (size_t) a.ntiles), props, [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {
            const int rbg = (int) id[0], nt = (int) id[1];
            const Pick pk = pick_expert(a, rbg);
            const int cnt = pk.cnt;
            const int ntl = (cnt + 7) >> 3;
            const int tl0 = (rbg - pk.rb0) * MT;
            const int nvalid = (ntl - tl0) < MT ? (ntl - tl0) : MT;
            const int tbase = pk.tile0 + tl0;
            const uint8_t* We = pk.blob;
            const uint32_t* T = a.tab;
            simd<uint32_t, 16> lane(0, 1);
            simd<uint32_t, 16> ro = (lane + (uint32_t) (16 * nt)) * (uint32_t) a.rowb + a.sec_off;
            simd<float, 128 * MT> acc = 0.f;
            const int* Xi = reinterpret_cast<const int*>(a.Xq);
            for (int kb = 0; kb < ((a.dbg & 8) ? 0 : a.nkb); ++kb) {
                simd<float, 16> sw0 = 1.f;
                simd<uint32_t, 128> fr0 = 0x01010101u;
                if (!(a.dbg & 1)) {
                    if constexpr (F == T_IQ4NL) {
                        const uint32_t P = (uint32_t) kb * 18;
                        sw0 = half_of(ld16(We, ro + P));
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            simd<uint32_t, 16> qd = ld32c(We, ro + (P + 2 + 4 * j), a.lim);
                            // four byte-pair lookups (lo value in byte 0, hi value in byte 1), repacked into the two dwords
                            simd<uint32_t, 16> t0 = gather<uint32_t, 16>(T + O_IQ4, (qd & 0xffu) << 2);
                            simd<uint32_t, 16> t1 = gather<uint32_t, 16>(T + O_IQ4, ((qd >> 8) & 0xffu) << 2);
                            simd<uint32_t, 16> t2 = gather<uint32_t, 16>(T + O_IQ4, ((qd >> 16) & 0xffu) << 2);
                            simd<uint32_t, 16> t3 = gather<uint32_t, 16>(T + O_IQ4, (qd >> 24) << 2);
                            fr0.template select<16, 1>(16 * j) = (t0 & 0xffu) | ((t1 & 0xffu) << 8) | ((t2 & 0xffu) << 16) | (t3 << 24);
                            fr0.template select<16, 1>(16 * (j + 4)) =
                                ((t0 >> 8) & 0xffu) | (t1 & 0xff00u) | ((t2 & 0xff00u) << 8) | ((t3 & 0xff00u) << 16);
                        }
                    } else if constexpr (F == T_Q8_0) {   // T_Q8_0: 32 int8 values per 34-byte block (2B fp16 d + 32B qs), no codebook lookup
                        const uint32_t P = (uint32_t) kb * 34;
                        sw0 = half_of(ld16(We, ro + P));
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            fr0.template select<16, 1>(16 * j) = ld32c(We, ro + (P + 2 + 4 * j), a.lim);
                        }
                    } else {   // T_Q2_0: 64 values per block (2 + 16 bytes), two 32-blocks per block, one table lookup per byte
                        const uint32_t P = (uint32_t) (kb >> 1) * 18;
                        const int hb = kb & 1;
                        sw0 = half_of(ld16(We, ro + P));
                        simd<uint32_t, 16> q0 = ld32c(We, ro + (P + 2 + 8 * hb), a.lim);
                        simd<uint32_t, 16> q1 = ld32c(We, ro + (P + 6 + 8 * hb), a.lim);
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            simd<uint32_t, 16> qb = ((r < 4 ? q0 : q1) >> (8 * (r & 3))) & 0xffu;
                            fr0.template select<16, 1>(16 * r) = gather<uint32_t, 16>(T + O_Q2, qb << 2);
                        }
                    }
                }
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                    if (i < nvalid) {
                        simd<int, 64> av = 0x01010101;
                        simd<float, 8> dxv = 1.f;
                        if (!(a.dbg & 2)) {
                            av = block_load<int, 64>(Xi + ((size_t) (tbase + i) * a.nkb + kb) * 64);
                            dxv = block_load<float, 8>(a.dx + ((size_t) (tbase + i) * a.nkb + kb) * 8);
                        }
                        simd<float, 128> tf = dpas_s8(fr0, av);
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const float dxr = dxv[r];
                            acc.template select<16, 1>(128 * i + 16 * r) += tf.template select<16, 1>(16 * r) * (sw0 * dxr);
                        }
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                if (i < nvalid) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int rl = (tl0 + i) * 8 + r;
                        if (rl < cnt && !(a.dbg & 4))
                            block_store<float, 16>(a.Y + ((size_t) (pk.row0 + rl)) * a.ldy + 16 * nt, acc.template select<16, 1>(128 * i + 16 * r));
                    }
                }
            }
        });
    });
}

// ------------------------------------------------------------------------------------------------ work-group versions
// A work group is WGT = 8 threads side by side on 8 consecutive output tiles (128 outputs) of the same row block.  The
// activation tiles of the row block are read from global memory ONCE per group, in chunks of 4 steps of 32 (128 of K), into local
// memory (double buffered, one barrier per chunk) and every thread reads them from there, instead of every thread reading all of
// them from L2 (80 threads per expert re-read the same ~200 KB: the activation loads were 8 of 19 us of gate/up).  The codebook
// the decoders gather from is copied into local memory by the group once (its gathers then cost ~nothing).
constexpr int WGT = 8;
constexpr auto kAl16 = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::alignment<16>};

// bytes of the tables a format gathers from (local memory), and the segments copied from the global table buffer.
// For IQ2_S / IQ2_XS / IQ2_XXS / Q8_0, TB = 0: only activation tiles are staged in SLM (8x global activation traffic reduction),
// while codebooks stay hot in the XeCore 64 KB L1 data cache without paying a per-workgroup SLM table copy.
template <int F> constexpr int tab_bytes() {
    return F == T_IQ3S ? 2048 : F == T_IQ3XXS ? 1024 : (F == T_IQ4XS || F == T_IQ4NL || F == T_Q2_0) ? 1024 : 0;   // IQ3XXS: the grid (its signs are computed, ksigns)
}

template <int F, int MT>
struct WgLayout {
    static constexpr int TB = tab_bytes<F>();
    static constexpr int ASZ = MT * 1024;   // activation tiles of one stage: MT tiles x 4 steps x 256 bytes
    static constexpr int DSZ = MT * 128;    // their scales: MT tiles x 4 steps x 32 bytes
    static constexpr int STAGE = ASZ + DSZ;
    static constexpr int TOTAL = TB + 2 * STAGE;
    static constexpr int SPLIT = 8 / MT;    // threads per tile in the staging load
    static constexpr int PART = 1024 / SPLIT;   // bytes of one tile's activations a thread stages
};

// Staging of one chunk's activation tiles: global -> registers (LOAD_CHUNK) -> local memory stage SS (STORE_CHUNK).  Macros, not
// lambdas: a lambda capturing the staging registers by reference put them in private memory.
#define LOAD_CHUNK(CC)                                                                                                         \
    if (s_tile < nvalid) {                                                                                                     \
        const int* src_ = Xi + ((size_t) (tbase + s_tile) * a.nkb + (CC) * 4) * 64 + s_part * (L::PART / 4);                  \
        _Pragma("unroll")                                                                                                      \
        for (int k = 0; k < L::PART / 256; ++k) stg.template select<64, 1>(64 * k) = block_load<int, 64>(src_ + 64 * k, kAl16); \
        if (s_part == 0) stg_dx = block_load<float, 32>(a.dx + ((size_t) (tbase + s_tile) * a.nkb + (CC) * 4) * 8, kAl16);    \
    }
#define STORE_CHUNK(SS)                                                                                                        \
    if (s_tile < nvalid) {                                                                                                     \
        const uint32_t base_ = L::TB + (SS) * L::STAGE;                                                                        \
        _Pragma("unroll")                                                                                                      \
        for (int k = 0; k < L::PART / 256; ++k)                                                                                \
            slm_block_store<int, 64>(base_ + s_tile * 1024 + s_part * L::PART + 256 * k, stg.template select<64, 1>(64 * k), kAl16); \
        if (s_part == 0) slm_block_store<float, 32>(base_ + L::ASZ + s_tile * 128, stg_dx, kAl16);                             \
    }

template <int F, int MT, int G> struct TagGuW;
template <int F, int MT, int G> struct TagDnW;

template <int F, int MT, int G>
void launch_gu_wg(sycl::queue& q, const GArgs a, int nrowblocks) {
// IQ2_S / IQ2_XS gate/up: the next chunk is prefetched into L1 at the top of a chunk and loaded + stored to local memory at its
// end, so the staging registers are live only there, not across the decode and the DPAS.  At MT 8 they were 36 of the registers
// the IQ2_S kernel spilled (1344 B; none left): 28.5 -> 16.9 us per expert on the B70.  The other formats did not spill and keep
// the load at the top (prefetching cost them 2-4%).
constexpr auto kPfL1 = sycl::ext::oneapi::experimental::properties{cache_hint_L1<cache_hint::cached>, cache_hint_L2<cache_hint::cached>};
#define PREFETCH_CHUNK(CC)                                                                                                     \
    if (s_tile < nvalid) {                                                                                                     \
        const int* src_ = Xi + ((size_t) (tbase + s_tile) * a.nkb + (CC) * 4) * 64 + s_part * (L::PART / 4);                  \
        prefetch<int, L::PART / 64>(src_, simd<uint32_t, L::PART / 64>(0, 64), kPfL1);                                        \
        if (s_part == 0) prefetch<float, 2>(a.dx + ((size_t) (tbase + s_tile) * a.nkb + (CC) * 4) * 8, simd<uint32_t, 2>(0, 64), kPfL1); \
    }

    using L = WgLayout<F, MT>;
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<G>};
    constexpr int NS = (F == T_IQ2S || F == T_IQ2XS) ? 2 : 1;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagGuW<F, MT, G>>(sycl::nd_range<2>(sycl::range<2>((size_t) nrowblocks, (size_t) a.ntiles), sycl::range<2>(1, WGT)), props,
                                         [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
            slm_init<L::TOTAL>();
            const int rbg = (int) it.get_group(0), lt = (int) it.get_local_id(1), nt = (int) it.get_global_id(1);
            const Pick pk = pick_expert(a, rbg);
            const int cnt = pk.cnt;
            const int ntl = (cnt + 7) >> 3;
            const int tl0 = (rbg - pk.rb0) * MT;
            const int nvalid = (ntl - tl0) < MT ? (ntl - tl0) : MT;
            const int tbase = pk.tile0 + tl0;
            const uint8_t* We = pk.blob;
            const uint32_t* T = a.tab;
            const int* Xi = reinterpret_cast<const int*>(a.Xq);
            // the codebook into local memory: thread lt copies its eighth of each segment (for TB > 0)
            if constexpr (F == T_IQ3S) {
#pragma unroll
                for (int k = 0; k < tab_bytes<F>() / WGT / 64; ++k)
                    slm_block_store<uint32_t, 16>(lt * (tab_bytes<F>() / WGT) + 64 * k, block_load<uint32_t, 16>(T + O_IQ3S + (lt * (tab_bytes<F>() / WGT) + 64 * k) / 4, kAl16), kAl16);
            } else if constexpr (F == T_IQ3XXS) {
#pragma unroll
                for (int k = 0; k < 1024 / WGT / 64; ++k)
                    slm_block_store<uint32_t, 16>(lt * (1024 / WGT) + 64 * k, block_load<uint32_t, 16>(T + O_IQ3XXS + (lt * (1024 / WGT) + 64 * k) / 4, kAl16), kAl16);
            } else if constexpr (F == T_IQ4XS) {
#pragma unroll
                for (int k = 0; k < tab_bytes<F>() / WGT / 64; ++k)
                    slm_block_store<uint32_t, 16>(lt * (tab_bytes<F>() / WGT) + 64 * k, block_load<uint32_t, 16>(T + O_IQ4 + (lt * (tab_bytes<F>() / WGT) + 64 * k) / 4, kAl16), kAl16);
            }
            // staging: this thread's piece of the chunk c (tile, part) into registers, then into stage s
            const int s_tile = lt / L::SPLIT, s_part = lt % L::SPLIT;
            simd<int, L::PART / 4> stg;
            simd<float, 32> stg_dx;
            simd<uint32_t, 16> lane(0, 1);
            simd<uint32_t, 16> ro = (lane & 1u) * a.sec_off + ((lane >> 1) + (uint32_t) (8 * nt)) * (uint32_t) a.rowb;
            simd<float, 128 * MT> acc = 0.f;
            const int nchunks = (a.dbg & 8) ? 0 : (a.nkb >> 2);
            if (nchunks > 0) { LOAD_CHUNK(0); STORE_CHUNK(0); }
            barrier();   // the codebook and chunk 0 are in local memory
            simd<float, 16> d = 0.f;
            simd<uint32_t, 16> w0 = 0, w1 = 0, w2 = 0, w3 = 0;
            simd<uint32_t, NS == 2 ? 192 : 16> fb = 0x01010101u;   // NS == 2: [values 0..15 | zeros | values 16..31], see the step below
            if constexpr (NS == 2) fb.template select<64, 1>(64) = 0u;
            for (int c = 0; c < nchunks; ++c) {
                const int sb = c >> 1;
                const int stage_off = L::TB + (c & 1) * L::STAGE;
                if (c + 1 < nchunks) {
                    if constexpr (NS == 2) { PREFETCH_CHUNK(c + 1); } else { LOAD_CHUNK(c + 1); }
                }
                if ((c & 1) == 0) {   // new superblock: its scale and the fields shared by its 8 steps
                    const uint32_t P = (uint32_t) sb * blk_bytes<F>();
                    d = half_of(ld16(We, ro + P));
                    if constexpr (F == T_IQ3S) {
                        w0 = ld32(We, ro + (P + 106)); w1 = ld32(We, ro + (P + 66)); w2 = ld32(We, ro + (P + 70));
                    } else if constexpr (F == T_IQ2S) {
                        w0 = ld32(We, ro + (P + 74)); w1 = ld32(We, ro + (P + 78)); w2 = ld32(We, ro + (P + 66)); w3 = ld32(We, ro + (P + 70));
                    } else if constexpr (F == T_IQ2XS) {
                        w0 = ld32(We, ro + (P + 66)); w1 = ld32(We, ro + (P + 70));
                    } else if constexpr (F == T_IQ4XS) {
                        w0 = ld16(We, ro + (P + 2)); w1 = ld32(We, ro + (P + 4));
                    }
                }
                const uint32_t P = (uint32_t) sb * blk_bytes<F>();
#pragma unroll 1
                for (int ib4 = 0; ib4 < 4; ++ib4) {
                    const int ib = ((c & 1) << 2) + ib4;
                    if constexpr (NS == 2) {
                        // IQ2_S / IQ2_XS: two scales per 32 values.  The low half's values go to fb[0, 64) and its DPAS reads fb[0, 128)
                        // (the upper half zero), then the high half's go to fb[128, 192) and its DPAS reads fb[64, 192): one half's
                        // operand and products are live at a time, and every output gets the low half's term before the high half's as before
                        simd<uint32_t, 16> qa, qb, scb;
                        simd<float, 16> sw0 = 1.f, sw1 = 1.f;
                        if (!(a.dbg & 1)) {
                            if constexpr (F == T_IQ2S) {
                                qa = ld32(We, ro + (P + 2 + 4 * ib));    // grid index low bytes
                                qb = ld32(We, ro + (P + 34 + 4 * ib));   // signs
                                scb = (((ib < 4) ? w0 : w1) >> (8 * (ib & 3))) & 0xffu;
                            } else {
                                qa = ld32(We, ro + (P + 2 + 8 * ib));
                                qb = ld32(We, ro + (P + 6 + 8 * ib));
                                scb = (((ib < 4) ? w0 : w1) >> (8 * (ib & 3))) & 0xffu;
                            }
                            sw0 = d * simd<float, 16>(simd<int, 16>((scb & 0xfu) * 2u + 1u)) * 0.125f;
                            sw1 = d * simd<float, 16>(simd<int, 16>((scb >> 4) * 2u + 1u)) * 0.125f;
                        }
#pragma unroll
                        for (int hf = 0; hf < 2; ++hf) {
                            if (!(a.dbg & 1)) {
#pragma unroll
                                for (int il = 2 * hf; il < 2 * hf + 2; ++il) {
                                    simd<uint32_t, 16> gl, gh, sbyte;
                                    if constexpr (F == T_IQ2S) {
                                        simd<uint32_t, 16> qhb = (((ib < 4) ? w2 : w3) >> (8 * (ib & 3))) & 0xffu;
                                        simd<uint32_t, 16> idx = ((qa >> (8 * il)) & 0xffu) | ((qhb << (8 - 2 * il)) & 0x300u);
                                        simd<uint32_t, 32> g2 = gather<uint32_t, 32, 2>(T + O_IQ2S, idx << 3);   // the 8-byte entry in one message
                                        gl = g2.template select<16, 1>(0);
                                        gh = g2.template select<16, 1>(16);
                                        sbyte = (qb >> (8 * il)) & 0xffu;
                                    } else {
                                        simd<uint32_t, 16> qsw = (il < 2) ? qa : qb;
                                        simd<uint32_t, 16> qv = (il & 1) ? (qsw >> 16) : (qsw & 0xffffu);
                                        simd<uint32_t, 16> idx = qv & 511u;
                                        simd<uint32_t, 32> g2 = gather<uint32_t, 32, 2>(T + O_IQ2XS, idx << 3);   // the 8-byte entry in one message
                                        gl = g2.template select<16, 1>(0);
                                        gh = g2.template select<16, 1>(16);
                                        sbyte = ksigns(qv >> 9);
                                    }
                                    fb.template select<16, 1>(64 * hf + 32 * il) = apply_signs(gl, sbyte & 0xfu);
                                    fb.template select<16, 1>(64 * hf + 32 * il + 16) = apply_signs(gh, sbyte >> 4);
                                }
                            }
                            const simd<float, 16> sw = hf ? sw1 : sw0;
#pragma unroll
                            for (int i = 0; i < MT; ++i) {
                                if (i < nvalid) {
                                    simd<int, 64> av = 0x01010101;
                                    simd<float, 8> dxv = 1.f;
                                    if (!(a.dbg & 2)) {
                                        av = slm_block_load<int, 64>(stage_off + i * 1024 + ib4 * 256, kAl16);
                                        dxv = slm_block_load<float, 8>(stage_off + L::ASZ + i * 128 + ib4 * 32, kAl16);
                                    }
                                    simd<float, 128> tf = dpas_s8(fb.template select<128, 1>(64 * hf), av);
#pragma unroll
                                    for (int r = 0; r < 8; ++r) {
                                        const float dxr = dxv[r];
                                        acc.template select<16, 1>(128 * i + 16 * r) += tf.template select<16, 1>(16 * r) * (sw * dxr);
                                    }
                                }
                            }
                        }
                    } else {
                        simd<uint32_t, 128> fr0 = 0x01010101u;
                        simd<float, 16> sw0 = 1.f;
                        if (!(a.dbg & 1)) {
                            if constexpr (F == T_IQ3S) {
                                simd<uint32_t, 16> q0 = ld32(We, ro + (P + 2 + 8 * ib));
                                simd<uint32_t, 16> q1 = ld32(We, ro + (P + 6 + 8 * ib));
                                simd<uint32_t, 16> sg = ld32(We, ro + (P + 74 + 4 * ib));
                                simd<uint32_t, 16> qhv = (((ib < 4) ? w1 : w2) >> (8 * (ib & 3))) & 0xffu;
                                simd<uint32_t, 16> scn = ((w0 >> (8 * (ib >> 1))) >> (4 * (ib & 1))) & 0xfu;
                                sw0 = d * simd<float, 16>(simd<int, 16>(scn * 2u + 1u));
#pragma unroll
                                for (int r = 0; r < 8; ++r) {
                                    simd<uint32_t, 16> qsw = r < 4 ? q0 : q1;
                                    simd<uint32_t, 16> idx = ((qsw >> (8 * (r & 3))) & 0xffu) | (((qhv >> r) & 1u) << 8);
                                    simd<uint32_t, 16> g = slm_gather<uint32_t, 16>(idx << 2);
                                    simd<uint32_t, 16> nib = ((sg >> (8 * (r >> 1))) >> (4 * (r & 1))) & 0xfu;
                                    fr0.template select<16, 1>(16 * r) = apply_signs(g, nib);
                                }
                            } else if constexpr (F == T_IQ3XXS) {
                                simd<uint32_t, 16> q0 = ld32(We, ro + (P + 2 + 8 * ib));
                                simd<uint32_t, 16> q1 = ld32(We, ro + (P + 6 + 8 * ib));
                                simd<uint32_t, 16> aux = ld32(We, ro + (P + 66 + 4 * ib));
                                sw0 = d * simd<float, 16>(simd<int, 16>((aux >> 28) * 2u + 1u)) * 0.25f;
#pragma unroll
                                for (int il = 0; il < 4; ++il) {
                                    simd<uint32_t, 16> sbyte = ksigns((aux >> (7 * il)) & 127u);
#pragma unroll
                                    for (int hh = 0; hh < 2; ++hh) {
                                        const int r = 2 * il + hh;
                                        simd<uint32_t, 16> qsw = r < 4 ? q0 : q1;
                                        simd<uint32_t, 16> idx = (qsw >> (8 * (r & 3))) & 0xffu;
                                        simd<uint32_t, 16> g = slm_gather<uint32_t, 16>(idx << 2);
                                        fr0.template select<16, 1>(16 * r) = apply_signs(g, (sbyte >> (4 * hh)) & 0xfu);
                                    }
                                }
                            } else if constexpr (F == T_IQ2XXS) {
                                simd<uint32_t, 16> q0 = ld32(We, ro + (P + 2 + 8 * ib));
                                simd<uint32_t, 16> aux = ld32(We, ro + (P + 6 + 8 * ib));
                                sw0 = d * simd<float, 16>(simd<int, 16>((aux >> 28) * 2u + 1u)) * 0.125f;
#pragma unroll
                                for (int il = 0; il < 4; ++il) {
                                    simd<uint32_t, 16> idx = (q0 >> (8 * il)) & 0xffu;
                                    simd<uint32_t, 32> g2 = gather<uint32_t, 32, 2>(T + O_IQ2XXS, idx << 3);   // the 8-byte entry in one message
                                    simd<uint32_t, 16> gl = g2.template select<16, 1>(0), gh = g2.template select<16, 1>(16);
                                    simd<uint32_t, 16> sbyte = ksigns((aux >> (7 * il)) & 127u);
                                    fr0.template select<16, 1>(16 * (2 * il)) = apply_signs(gl, sbyte & 0xfu);
                                    fr0.template select<16, 1>(16 * (2 * il + 1)) = apply_signs(gh, sbyte >> 4);
                                }
                            } else if constexpr (F == T_IQ4XS) {
                                simd<uint32_t, 16> ls = ((w1 >> (4 * ib)) & 0xfu) | (((w0 >> (2 * ib)) & 3u) << 4);
                                sw0 = d * simd<float, 16>(simd<int, 16>(ls) - 32);
#pragma unroll
                                for (int j = 0; j < 4; ++j) {
                                    simd<uint32_t, 16> qd = ld32(We, ro + (P + 8 + 16 * ib + 4 * j));
                                    simd<uint32_t, 16> t0 = slm_gather<uint32_t, 16>((qd & 0xffu) << 2);
                                    simd<uint32_t, 16> t1 = slm_gather<uint32_t, 16>(((qd >> 8) & 0xffu) << 2);
                                    simd<uint32_t, 16> t2 = slm_gather<uint32_t, 16>(((qd >> 16) & 0xffu) << 2);
                                    simd<uint32_t, 16> t3 = slm_gather<uint32_t, 16>((qd >> 24) << 2);
                                    fr0.template select<16, 1>(16 * j) = (t0 & 0xffu) | ((t1 & 0xffu) << 8) | ((t2 & 0xffu) << 16) | (t3 << 24);
                                    fr0.template select<16, 1>(16 * (j + 4)) =
                                        ((t0 >> 8) & 0xffu) | (t1 & 0xff00u) | ((t2 & 0xff00u) << 8) | ((t3 & 0xff00u) << 16);
                                }
                            }
                        }
#pragma unroll
                        for (int i = 0; i < MT; ++i) {
                            if (i < nvalid) {
                                simd<int, 64> av = 0x01010101;
                                simd<float, 8> dxv = 1.f;
                                if (!(a.dbg & 2)) {
                                    av = slm_block_load<int, 64>(stage_off + i * 1024 + ib4 * 256, kAl16);
                                    dxv = slm_block_load<float, 8>(stage_off + L::ASZ + i * 128 + ib4 * 32, kAl16);
                                }
                                simd<float, 128> tf = dpas_s8(fr0, av);
#pragma unroll
                                for (int r = 0; r < 8; ++r) {
                                    const float dxr = dxv[r];
                                    acc.template select<16, 1>(128 * i + 16 * r) += tf.template select<16, 1>(16 * r) * (sw0 * dxr);
                                }
                            }
                        }
                    }
                }
                if (c + 1 < nchunks) {
                    if constexpr (NS == 2) { LOAD_CHUNK(c + 1); }
                    STORE_CHUNK((c + 1) & 1);
                }
                barrier();
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                if (i < nvalid) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int rl = (tl0 + i) * 8 + r;
                        if (rl < cnt && !(a.dbg & 4))
                            block_store<float, 16>(a.Y + ((size_t) (pk.row0 + rl)) * a.ldy + 16 * nt, acc.template select<16, 1>(128 * i + 16 * r));
                    }
                }
            }
        });
    });
}

template <int F, int MT, int G>
void launch_dn_wg(sycl::queue& q, const GArgs a, int nrowblocks) {
    using L = WgLayout<F, MT>;
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<G>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagDnW<F, MT, G>>(sycl::nd_range<2>(sycl::range<2>((size_t) nrowblocks, (size_t) a.ntiles), sycl::range<2>(1, WGT)), props,
                                         [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
            slm_init<L::TOTAL>();
            const int rbg = (int) it.get_group(0), lt = (int) it.get_local_id(1), nt = (int) it.get_global_id(1);
            const Pick pk = pick_expert(a, rbg);
            const int cnt = pk.cnt;
            const int ntl = (cnt + 7) >> 3;
            const int tl0 = (rbg - pk.rb0) * MT;
            const int nvalid = (ntl - tl0) < MT ? (ntl - tl0) : MT;
            const int tbase = pk.tile0 + tl0;
            const uint8_t* We = pk.blob;
            const uint32_t* T = a.tab;
            const int* Xi = reinterpret_cast<const int*>(a.Xq);
            if constexpr (L::TB > 0) {
                constexpr uint32_t TOFF = (F == T_IQ4NL) ? O_IQ4 : O_Q2;
#pragma unroll
                for (int k = 0; k < tab_bytes<F>() / WGT / 64; ++k)
                    slm_block_store<uint32_t, 16>(lt * (tab_bytes<F>() / WGT) + 64 * k, block_load<uint32_t, 16>(T + TOFF + (lt * (tab_bytes<F>() / WGT) + 64 * k) / 4, kAl16), kAl16);
            }
            const int s_tile = lt / L::SPLIT, s_part = lt % L::SPLIT;
            simd<int, L::PART / 4> stg;
            simd<float, 32> stg_dx;
            simd<uint32_t, 16> lane(0, 1);
            simd<uint32_t, 16> ro = (lane + (uint32_t) (16 * nt)) * (uint32_t) a.rowb + a.sec_off;
            simd<float, 128 * MT> acc = 0.f;
            const int nchunks = (a.dbg & 8) ? 0 : (a.nkb >> 2);
            if (nchunks > 0) { LOAD_CHUNK(0); STORE_CHUNK(0); }
            barrier();
            for (int c = 0; c < nchunks; ++c) {
                const int stage_off = L::TB + (c & 1) * L::STAGE;
                if (c + 1 < nchunks) { LOAD_CHUNK(c + 1); }
#pragma unroll 1
                for (int ib4 = 0; ib4 < 4; ++ib4) {
                    const int kb = c * 4 + ib4;
                    simd<float, 16> sw0 = 1.f;
                    simd<uint32_t, 128> fr0 = 0x01010101u;
                    if (!(a.dbg & 1)) {
                        if constexpr (F == T_IQ4NL) {
                            const uint32_t P = (uint32_t) kb * 18;
                            sw0 = half_of(ld16(We, ro + P));
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                simd<uint32_t, 16> qd = ld32c(We, ro + (P + 2 + 4 * j), a.lim);
                                simd<uint32_t, 16> t0 = slm_gather<uint32_t, 16>((qd & 0xffu) << 2);
                                simd<uint32_t, 16> t1 = slm_gather<uint32_t, 16>(((qd >> 8) & 0xffu) << 2);
                                simd<uint32_t, 16> t2 = slm_gather<uint32_t, 16>(((qd >> 16) & 0xffu) << 2);
                                simd<uint32_t, 16> t3 = slm_gather<uint32_t, 16>((qd >> 24) << 2);
                                fr0.template select<16, 1>(16 * j) = (t0 & 0xffu) | ((t1 & 0xffu) << 8) | ((t2 & 0xffu) << 16) | (t3 << 24);
                                fr0.template select<16, 1>(16 * (j + 4)) =
                                    ((t0 >> 8) & 0xffu) | (t1 & 0xff00u) | ((t2 & 0xff00u) << 8) | ((t3 & 0xff00u) << 16);
                            }
                        } else if constexpr (F == T_Q8_0) {
                            const uint32_t P = (uint32_t) kb * 34;
                            sw0 = half_of(ld16(We, ro + P));
#pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                fr0.template select<16, 1>(16 * j) = ld32c(We, ro + (P + 2 + 4 * j), a.lim);
                            }
                        } else {   // T_Q2_0
                            const uint32_t P = (uint32_t) (kb >> 1) * 18;
                            const int hb = kb & 1;
                            sw0 = half_of(ld16(We, ro + P));
                            simd<uint32_t, 16> q0 = ld32c(We, ro + (P + 2 + 8 * hb), a.lim);
                            simd<uint32_t, 16> q1 = ld32c(We, ro + (P + 6 + 8 * hb), a.lim);
#pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                simd<uint32_t, 16> qb = ((r < 4 ? q0 : q1) >> (8 * (r & 3))) & 0xffu;
                                fr0.template select<16, 1>(16 * r) = slm_gather<uint32_t, 16>(qb << 2);
                            }
                        }
                    }
#pragma unroll
                    for (int i = 0; i < MT; ++i) {
                        if (i < nvalid) {
                            simd<int, 64> av = 0x01010101;
                            simd<float, 8> dxv = 1.f;
                            if (!(a.dbg & 2)) {
                                av = slm_block_load<int, 64>(stage_off + i * 1024 + ib4 * 256, kAl16);
                                dxv = slm_block_load<float, 8>(stage_off + L::ASZ + i * 128 + ib4 * 32, kAl16);
                            }
                            simd<float, 128> tf = dpas_s8(fr0, av);
#pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                const float dxr = dxv[r];
                                acc.template select<16, 1>(128 * i + 16 * r) += tf.template select<16, 1>(16 * r) * (sw0 * dxr);
                            }
                        }
                    }
                }
                if (c + 1 < nchunks) { STORE_CHUNK((c + 1) & 1); }
                barrier();
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                if (i < nvalid) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int rl = (tl0 + i) * 8 + r;
                        if (rl < cnt && !(a.dbg & 4))
                            block_store<float, 16>(a.Y + ((size_t) (pk.row0 + rl)) * a.ldy + 16 * nt, acc.template select<16, 1>(128 * i + 16 * r));
                    }
                }
            }
        });
    });
}

// MT from the group's mean number of 8-row tiles per expert.
int pick_mt(const Layout& lo) {
    const double mean = lo.n ? (double) lo.tile0[lo.n] / lo.n : 0;
    if (mean <= 2.0) return 2;
    if (mean <= 4.0) return 4;
    return 8;
}
int dbg_flags() {
    static const int v = [] { const char* s = std::getenv("STRATA_I8_DBG"); return s ? std::atoi(s) : 0; }();
    return v;
}
int mt_for(const Layout& lo) {
    static const int mo = [] { const char* s = std::getenv("STRATA_I8_MT"); return s ? std::atoi(s) : 0; }();
    return mo ? mo : pick_mt(lo);
}

// register mode per tile height: STRATA_I8_GRF=128 (more threads per EU, spills at MT 8) or 256; default 256
int wg_for() {
    static const int v = [] { const char* s = std::getenv("STRATA_I8_WG"); return s ? std::atoi(s) : 1; }();
    return v;
}
int grf_for(int mt) {
    static const int g = [] { const char* s = std::getenv("STRATA_I8_GRF"); return s ? std::atoi(s) : 256; }();
    (void) mt;
    return g == 128 ? 128 : 256;
}

template <class F>
bool dispatch_mt(int mt, F&& f) {
    switch (mt) {
        case 2: f(std::integral_constant<int, 2>{}); return true;
        case 4: f(std::integral_constant<int, 4>{}); return true;
        case 8: f(std::integral_constant<int, 8>{}); return true;
        default: return false;
    }
}

struct QArgs {
    int n;
    int row0[kMaxGroup + 1];
    int tile0[kMaxGroup + 1];
    int nkb;
    int ld;   // row length in elements of the source
};
class QuantTag;
class SwigluTag;

}  // namespace

void make_layout(Layout& lo, const int32_t* cnt, int n) {
    lo.n = n;
    int r = 0, t = 0;
    for (int i = 0; i < n; ++i) {
        lo.cnt[i] = cnt[i];
        lo.row0[i] = r;
        lo.tile0[i] = t;
        r += cnt[i];
        t += (cnt[i] + 7) / 8;
    }
    lo.row0[n] = r;
    lo.tile0[n] = t;
}

namespace {
bool bmg_arch(const sycl::device& d) {
    namespace ex = sycl::ext::oneapi::experimental;
    if (!d.is_gpu()) return false;
    const auto a = d.get_info<ex::info::device::architecture>();
    return a == ex::architecture::intel_gpu_bmg_g21 || a == ex::architecture::intel_gpu_bmg_g31;
}
}  // namespace

bool battlemage(void* stream) {
    sycl::queue& q = *strata::q_of(stream);
    return bmg_arch(q.get_device());
}

bool any_battlemage() {
    static const bool v = [] {
        try {
            for (const auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
                if (bmg_arch(d)) return true;
        } catch (...) {}
        return false;
    }();
    return v;
}

bool available(void* stream) {
    sycl::queue& q = *strata::q_of(stream);
    static std::mutex mu;
    static std::unordered_map<sycl::device, bool> per_dev;
    std::lock_guard<std::mutex> lk(mu);
    const sycl::device d = q.get_device();
    if (auto it = per_dev.find(d); it != per_dev.end()) return it->second;
    const bool ok = [&] {
        const char* v = std::getenv("STRATA_I8_FORCE");
        if (v && v[0] == '0') return false;
        if (!d.has(sycl::aspect::ext_intel_esimd)) return false;
        namespace mx = sycl::ext::oneapi::experimental::matrix;
        bool s8 = false;
        for (const auto& c : d.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>())
            if (c.atype == mx::matrix_type::sint8 && c.btype == mx::matrix_type::sint8 && c.nsize == 16 && c.ksize == 32) s8 = true;
        return s8;
    }();
    per_dev[d] = ok;
    return ok;
}

size_t xq_bytes(const Layout& lo, int K) { return (size_t) lo.tile0[lo.n] * 8 * K; }
size_t dx_bytes(const Layout& lo, int K) { return (size_t) lo.tile0[lo.n] * (K / 32) * 8 * sizeof(float); }

bool workspace(const Layout& lo, int K, int n_ff, Workspace& ws, void* stream) {
    sycl::queue& q = *strata::q_of(stream);
    static std::mutex mu;
    struct St { Workspace cur; size_t cap_x = 0, cap_h = 0; };
    static std::unordered_map<sycl::device, St> per_dev;
    std::lock_guard<std::mutex> lk(mu);
    St& st = per_dev[q.get_device()];
    Workspace& cur = st.cur;
    size_t& cap_x = st.cap_x;
    size_t& cap_h = st.cap_h;
    const size_t nx = xq_bytes(lo, K), nh = xq_bytes(lo, n_ff);
    if (nx > cap_x || nh > cap_h) {
        q.wait();
        if (cur.xq) { sycl::free(cur.xq, q); sycl::free(cur.dx, q); sycl::free(cur.hq, q); sycl::free(cur.dh, q); }
        cap_x = std::max(nx, cap_x + cap_x / 4);
        cap_h = std::max(nh, cap_h + cap_h / 4);
        cur.xq = sycl::malloc_device<int8_t>(cap_x, q);
        cur.dx = sycl::malloc_device<float>(cap_x / 32 / 8 * 8 + 64, q);   // rows * K/32 floats = bytes / 32
        cur.hq = sycl::malloc_device<int8_t>(cap_h, q);
        cur.dh = sycl::malloc_device<float>(cap_h / 32 + 64, q);
        if (!cur.xq || !cur.dx || !cur.hq || !cur.dh) { cap_x = cap_h = 0; cur = Workspace{}; ws = Workspace{}; return false; }
    }
    ws = cur;
    return true;
}

void quantize_rows(const uint16_t* X, const Layout& lo, int K, int8_t* Xq, float* dx, void* stream) {
    if (lo.tile0[lo.n] == 0) return;
    sycl::queue& q = *strata::q_of(stream);
    QArgs a{};
    a.n = lo.n;
    for (int i = 0; i <= lo.n; ++i) { a.row0[i] = lo.row0[i]; a.tile0[i] = lo.tile0[i]; }
    a.nkb = K / 32;
    a.ld = K;
    const size_t nthreads = (size_t) lo.tile0[lo.n] * a.nkb;
    const half* Xh = reinterpret_cast<const half*>(X);
    uint32_t* Xd = reinterpret_cast<uint32_t*>(Xq);
    q.submit([&](sycl::handler& h) {
        h.parallel_for<QuantTag>(sycl::range<2>((size_t) lo.tile0[lo.n], (size_t) a.nkb), [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {
            const int T = (int) id[0], kb = (int) id[1];
            // the expert of tile T, by constant-index reads of the argument arrays (see pick_expert)
            int t0e = a.tile0[0], r0e = a.row0[0], r1e = a.row0[1];
#pragma unroll
            for (int i = 1; i < kMaxGroup; ++i)
                if (i < a.n && T >= a.tile0[i]) { t0e = a.tile0[i]; r0e = a.row0[i]; r1e = a.row0[i + 1]; }
            const int cnt = r1e - r0e;
            simd<float, 8> dxv = 0.f;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int rl = (T - t0e) * 8 + r;
                simd<uint32_t, 8> qd = 0u;
                if (rl < cnt) {
                    simd<half, 32> xh = block_load<half, 32>(Xh + ((size_t) (r0e + rl)) * a.ld + kb * 32);
                    simd<float, 32> x = xh;
                    float amax = hmax<float>(abs(x));
                    const float dd = amax / 127.f;
                    const float id127 = amax > 0.f ? 127.f / amax : 0.f;
                    simd<float, 32> y = x * id127;
                    simd<float, 32> adj = 0.5f;
                    adj.merge(simd<float, 32>(-0.5f), y < 0.f);
                    simd<int, 32> qi = convert<int>(y + adj);
                    simd<int8_t, 32> q8 = convert<int8_t>(qi);
                    qd = q8.template bit_cast_view<uint32_t>();
                    dxv[r] = dd;
                }
                block_store<uint32_t, 8>(Xd + ((size_t) T * a.nkb + kb) * 64 + r * 8, qd);
            }
            block_store<float, 8>(dx + ((size_t) T * a.nkb + kb) * 8, dxv);
        });
    });
}

void swiglu_quantize(const float* GU, const Layout& lo, int n_ff, int8_t* Hq, float* dh, void* stream) {
    if (lo.tile0[lo.n] == 0) return;
    sycl::queue& q = *strata::q_of(stream);
    QArgs a{};
    a.n = lo.n;
    for (int i = 0; i <= lo.n; ++i) { a.row0[i] = lo.row0[i]; a.tile0[i] = lo.tile0[i]; }
    a.nkb = n_ff / 32;
    a.ld = 2 * n_ff;
    const size_t nthreads = (size_t) lo.tile0[lo.n] * a.nkb;
    uint32_t* Hd = reinterpret_cast<uint32_t*>(Hq);
    q.submit([&](sycl::handler& h) {
        h.parallel_for<SwigluTag>(sycl::range<2>((size_t) lo.tile0[lo.n], (size_t) a.nkb), [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {
            const int T = (int) id[0], kb = (int) id[1];
            // the expert of tile T, by constant-index reads of the argument arrays (see pick_expert)
            int t0e = a.tile0[0], r0e = a.row0[0], r1e = a.row0[1];
#pragma unroll
            for (int i = 1; i < kMaxGroup; ++i)
                if (i < a.n && T >= a.tile0[i]) { t0e = a.tile0[i]; r0e = a.row0[i]; r1e = a.row0[i + 1]; }
            const int cnt = r1e - r0e;
            simd<float, 8> dxv = 0.f;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int rl = (T - t0e) * 8 + r;
                simd<uint32_t, 8> qd = 0u;
                if (rl < cnt) {
                    simd<float, 64> v = block_load<float, 64>(GU + ((size_t) (r0e + rl)) * a.ld + kb * 64);
                    simd<float, 32> g = v.template select<32, 2>(0);
                    simd<float, 32> u = v.template select<32, 2>(1);
                    simd<float, 32> x = g * u / (1.f + exp(-g));
                    float amax = hmax<float>(abs(x));
                    const float dd = amax / 127.f;
                    const float id127 = amax > 0.f ? 127.f / amax : 0.f;
                    simd<float, 32> y = x * id127;
                    simd<float, 32> adj = 0.5f;
                    adj.merge(simd<float, 32>(-0.5f), y < 0.f);
                    simd<int, 32> qi = convert<int>(y + adj);
                    simd<int8_t, 32> q8 = convert<int8_t>(qi);
                    qd = q8.template bit_cast_view<uint32_t>();
                    dxv[r] = dd;
                }
                block_store<uint32_t, 8>(Hd + ((size_t) T * a.nkb + kb) * 64 + r * 8, qd);
            }
            block_store<float, 8>(dh + ((size_t) T * a.nkb + kb) * 8, dxv);
        });
    });
}

bool gate_up(int gu_type, const uint8_t* const* blob, size_t up_off, int n_ff, const Layout& lo, int K, const int8_t* Xq,
             const float* dx, float* GU, void* stream) {
    if (lo.n == 0 || lo.tile0[lo.n] == 0) return true;
    if ((gu_type != T_IQ3S && gu_type != T_IQ3XXS && gu_type != T_IQ2S && gu_type != T_IQ2XS && gu_type != T_IQ2XXS && gu_type != T_IQ4XS) ||
        K % 256 != 0 || (2 * n_ff) % 16 != 0 || (up_off & 3) != 0 || lo.n > kMaxGroup)
        return false;
    sycl::queue& q = *strata::q_of(stream);
    GArgs a{};
    for (int i = 0; i < lo.n; ++i) a.blob[i] = blob[i];
    a.sec_off = (uint32_t) up_off;
    a.Xq = Xq; a.dx = dx; a.Y = GU;
    a.tab = tables(q);
    a.ntiles = 2 * n_ff / 16;
    a.nkb = K / 32;
    a.rowb = (K / 256) * block_bytes_of(gu_type);
    a.ldy = 2 * n_ff;
    a.dbg = dbg_flags();
    const int mt = mt_for(lo);
    const int rb = fill_args_mt(a, lo, mt);
    return dispatch_mt(mt, [&](auto mtc) {
        constexpr int MT = decltype(mtc)::value;
        if (wg_for() && a.nkb % 4 == 0 && a.ntiles % WGT == 0) {
            if (gu_type == T_IQ3S) launch_gu_wg<T_IQ3S, MT, 256>(q, a, rb);
            else if (gu_type == T_IQ3XXS) launch_gu_wg<T_IQ3XXS, MT, 256>(q, a, rb);
            else if (gu_type == T_IQ4XS) launch_gu_wg<T_IQ4XS, MT, 256>(q, a, rb);
            else if (gu_type == T_IQ2XS) launch_gu_wg<T_IQ2XS, MT, 256>(q, a, rb);
            else if (gu_type == T_IQ2XXS) launch_gu_wg<T_IQ2XXS, MT, 256>(q, a, rb);
            else launch_gu_wg<T_IQ2S, MT, 256>(q, a, rb);
        } else if (grf_for(MT) == 128) {
            if (gu_type == T_IQ3S) launch_gu<T_IQ3S, MT, 128>(q, a, rb);
            else if (gu_type == T_IQ3XXS) launch_gu<T_IQ3XXS, MT, 128>(q, a, rb);
            else if (gu_type == T_IQ4XS) launch_gu<T_IQ4XS, MT, 128>(q, a, rb);
            else if (gu_type == T_IQ2XS) launch_gu<T_IQ2XS, MT, 128>(q, a, rb);
            else if (gu_type == T_IQ2XXS) launch_gu<T_IQ2XXS, MT, 128>(q, a, rb);
            else launch_gu<T_IQ2S, MT, 128>(q, a, rb);
        } else {
            if (gu_type == T_IQ3S) launch_gu<T_IQ3S, MT, 256>(q, a, rb);
            else if (gu_type == T_IQ3XXS) launch_gu<T_IQ3XXS, MT, 256>(q, a, rb);
            else if (gu_type == T_IQ4XS) launch_gu<T_IQ4XS, MT, 256>(q, a, rb);
            else if (gu_type == T_IQ2XS) launch_gu<T_IQ2XS, MT, 256>(q, a, rb);
            else if (gu_type == T_IQ2XXS) launch_gu<T_IQ2XXS, MT, 256>(q, a, rb);
            else launch_gu<T_IQ2S, MT, 256>(q, a, rb);
        }
    });
}

bool down(int d_type, const uint8_t* const* blob, size_t down_off, int n_embd, const Layout& lo, int n_ff, const int8_t* Hq,
          const float* dh, float* Dm, void* stream) {
    if (lo.n == 0 || lo.tile0[lo.n] == 0) return true;
    if ((d_type != T_IQ4NL && d_type != T_Q2_0 && d_type != T_Q8_0) || n_ff % block_vals_of(d_type) != 0 || n_embd % 16 != 0 ||
        (down_off & 3) != 0 || lo.n > kMaxGroup)
        return false;
    sycl::queue& q = *strata::q_of(stream);
    GArgs a{};
    for (int i = 0; i < lo.n; ++i) a.blob[i] = blob[i];
    a.sec_off = (uint32_t) down_off;
    a.Xq = Hq; a.dx = dh; a.Y = Dm;
    a.tab = tables(q);
    a.ntiles = n_embd / 16;
    a.nkb = n_ff / 32;
    a.rowb = (n_ff / block_vals_of(d_type)) * block_bytes_of(d_type);
    a.lim = (uint32_t) (down_off + (size_t) n_embd * (size_t) a.rowb) - 4u;
    a.ldy = n_embd;
    a.dbg = dbg_flags();
    const int mt = mt_for(lo);
    const int rb = fill_args_mt(a, lo, mt);
    return dispatch_mt(mt, [&](auto mtc) {
        constexpr int MT = decltype(mtc)::value;
        if (wg_for() && a.nkb % 4 == 0 && a.ntiles % WGT == 0) {
            if (d_type == T_IQ4NL) launch_dn_wg<T_IQ4NL, MT, 256>(q, a, rb);
            else if (d_type == T_Q8_0) launch_dn_wg<T_Q8_0, MT, 256>(q, a, rb);
            else launch_dn_wg<T_Q2_0, MT, 256>(q, a, rb);
        } else if (grf_for(MT) == 128) {
            if (d_type == T_IQ4NL) launch_dn<T_IQ4NL, MT, 128>(q, a, rb);
            else if (d_type == T_Q8_0) launch_dn<T_Q8_0, MT, 128>(q, a, rb);
            else launch_dn<T_Q2_0, MT, 128>(q, a, rb);
        } else {
            if (d_type == T_IQ4NL) launch_dn<T_IQ4NL, MT, 256>(q, a, rb);
            else if (d_type == T_Q8_0) launch_dn<T_Q8_0, MT, 256>(q, a, rb);
            else launch_dn<T_Q2_0, MT, 256>(q, a, rb);
        }
    });
}

// ------------------------------------------------------------------------------------------------ FP16 products (larger groups)
// The FP16 chain (dequantize the expert to FP16, oneMKL FP16 GEMM, SwiGLU, oneMKL GEMM) without the FP16 copy of the expert and
// with one launch per product for the group:
//  * Tile.  A hardware thread owns 64 rows (MT = 8 row tiles) of one expert and 32 output columns (gate/up: 16 gate rows and the
//    same 16 up rows, SwiGLU in the epilogue; down: 2 tiles of 16).  It walks K in steps of 16: decode the step's B operand (lane =
//    output column, VNNI: a dword holds two consecutive K values) in registers, read the activations with 2-D block loads (rows past
//    the expert's end read as zero, stores past it are dropped), one FP16 DPAS 8x16x16 per row tile and column tile into FP32
//    accumulators (128 of the 256 registers).  A weight is decoded once per 64 rows.  No shared memory, no barrier; the 8 threads of a
//    work-group share their rows (L1).  Measured on gate/up (esimd_f16_parity, 16 experts x 320 rows, B70, the first version with a sign
//    table): the DPAS loop alone 111 TFLOP/s, with the decode 72, with the activation loads 79, with both 53; with the computed signs
//    and no per-tile branches 64; 32 rows a thread (MT 4) was half as fast, 32 output columns more (64 x 64) spilled.
//  * Weights.  The values iq_dequant_expert_f16 writes: IQ2_S / IQ2_XXS w = d (2 s + 1) g / 8 rounded to FP16 once (its float
//    product is exact).  Here g (2 s + 1) is an exact FP16 integer (<= 1333) and times d it rounds once to 8 w; the 1 / 8 is applied
//    to the FP32 accumulators (exact).  So the B operand is the FP16 chain's weights times 8, bit for bit, unless a weight is an FP16
//    subnormal (|w| < 6.1e-5; this one is then the more precise).  Q2_0 w = d (c - 1) is exact.  The codebooks are FP16 copies
//    of the grids (4 dwords an entry, one 16-byte gather a lane) gathered from global memory (L1); the sign bits are computed.
//  * Outputs: gate/up H = silu(g) u in FP16, saturated at +-65504 as swiglu_interleaved; down FP32.  The products differ from the
//    FP16 chain only in the summation order.
namespace {
constexpr int O_HG2S = 0;       // iq2s_grid as FP16: 1024 entries x 4 dwords (values 2j, 2j + 1 in dword j)
constexpr int O_HG2XXS = 4096;  // iq2xxs_grid as FP16: 256 x 4
constexpr int O_HQ2 = 5120;     // Q2_0 byte -> its four (code - 1) as FP16: 256 x 2 dwords
constexpr int N_HTABLE = 5632;

uint32_t* htables(sycl::queue& q) {
    static std::mutex mu;
    static std::unordered_map<sycl::device, uint32_t*> per_dev;
    std::lock_guard<std::mutex> lk(mu);
    uint32_t*& t = per_dev[q.get_device()];
    if (!t) {
        auto hb = [](int v) { return (uint32_t) sycl::bit_cast<uint16_t>(sycl::half((float) v)); };
        std::vector<uint32_t> h(N_HTABLE, 0);
        auto grid = [&](int off, const uint64_t* g, int n) {
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < 4; ++j)
                    h[off + 4 * i + j] = hb((int) ((g[i] >> (16 * j)) & 0xff)) | (hb((int) ((g[i] >> (16 * j + 8)) & 0xff)) << 16);
        };
        grid(O_HG2S, iq2s_grid, 1024);
        grid(O_HG2XXS, iq2xxs_grid, 256);
        for (int b = 0; b < 256; ++b)
            for (int j = 0; j < 2; ++j)
                h[O_HQ2 + 2 * b + j] = hb(((b >> (4 * j)) & 3) - 1) | (hb(((b >> (4 * j + 2)) & 3) - 1) << 16);
        uint32_t* d = sycl::malloc_device<uint32_t>(N_HTABLE, q);
        q.memcpy(d, h.data(), N_HTABLE * 4);
        q.wait();
        t = d;
    }
    return t;
}

template <int G> struct HArgsT {
    const uint8_t* blob[G];
    uint32_t sec_off;                 // gate/up: byte offset of the up rows; down: byte offset of the down rows
    int n;
    int row0[G + 1];
    int rb0[G + 1];                   // first row block of expert i
    const half* X;                    // A rows [row][K]
    void* Y;                          // gate/up: H (FP16) [row][n_out]; down: Dm (FP32) [row][n_out]
    const uint32_t* tab;
    int K;
    int n_out;                        // gate/up: n_ff; down: n_embd
    int rowb;                         // bytes per weight row
    uint32_t lim;                     // down: byte offset of the last dword of the matrix (ld32c)
    int dbg;                          // STRATA_F16X_DBG: 4 skip the stores, 8 no L1 prefetch of the weight rows
};
using HArgs = HArgsT<kMaxGroup>;
struct HPick { int row0, cnt, rb0; const uint8_t* blob; };
template <int G>
SYCL_ESIMD_FUNCTION inline HPick pick_h(const HArgsT<G>& a, int rbg) {
    HPick p{a.row0[0], a.row0[1] - a.row0[0], a.rb0[0], a.blob[0]};
#pragma unroll
    for (int i = 1; i < G; ++i)
        if (i < a.n && rbg >= a.rb0[i]) p = HPick{a.row0[i], a.row0[i + 1] - a.row0[i], a.rb0[i], a.blob[i]};
    return p;
}
// the held group's (STRATA_PF_F16X_GROUP): expert i's rows from the running sum of cnt, as make_layout and fill_hargs give them
template <int G>
int fill_hargs_n(HArgsT<G>& a, const int32_t* cnt, int n, int BM, uint64_t skip = 0) {
    int rb = 0, r = 0;
    for (int i = 0; i < n; ++i) {
        a.row0[i] = r;
        a.rb0[i] = rb;
        r += cnt[i];
        if (!((skip >> i) & 1)) rb += (cnt[i] + BM - 1) / BM;   // a skipped expert keeps its rows' place, no row block (pick_h passes it)
    }
    a.row0[n] = r;
    a.rb0[n] = rb;
    a.n = n;
    return rb;
}
int fill_hargs(HArgs& a, const Layout& lo, int BM) {
    int rb = 0;
    for (int i = 0; i < lo.n; ++i) {
        a.row0[i] = lo.row0[i];
        a.rb0[i] = rb;
        rb += (lo.cnt[i] + BM - 1) / BM;
    }
    a.row0[lo.n] = lo.row0[lo.n];
    a.rb0[lo.n] = rb;
    a.n = lo.n;
    return rb;
}

SYCL_ESIMD_FUNCTION inline simd<float, 128> dpas_h(simd<float, 128> c, simd<uint32_t, 128> b, simd<half, 128> a) {
    simd<int, 128> bi = b.template bit_cast_view<int>();
    simd<int, 64> ai = a.template bit_cast_view<int>();
    return xmx::dpas<8, 8, float, float, int, int, xmx::dpas_argument_type::fp16, xmx::dpas_argument_type::fp16>(c, bi, ai);
}
// FP16 per-lane scale -> the 32 halves of one VNNI row (lane l: halves 2 l, 2 l + 1)
SYCL_ESIMD_FUNCTION inline simd<half, 32> pair_of(simd<half, 16> s) {
    simd<half, 32> r;
    r.template select<16, 2>(0) = s;
    r.template select<16, 2>(1) = s;
    return r;
}
// One B operand (16 K values x 16 lanes) of IQ2_S / IQ2_XXS: the two 8-value grid entries idx0, idx1 with their sign bytes, times
// c1 = 2 s + 1 (exact), times d (one rounding): 8 w.  The sign bits are computed: dword j of an entry takes bits 2 j, 2 j + 1 of the
// sign byte to bits 15, 31 as (s >> 2 j) * 0x40008000 & 0x80008000 (a sign table gathered like the grid: gate/up 622 against 524 us,
// 601 against 569 with per-tile branches, esimd_f16_parity 320 16 on the B70).
template <int OG>
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 128> dec_iq2_h(const uint32_t* T, simd<uint32_t, 16> idx0, simd<uint32_t, 16> idx1,
                                                         simd<uint32_t, 16> s0, simd<uint32_t, 16> s1, simd<half, 32> c2,
                                                         simd<half, 32> d2) {
    simd<uint32_t, 128> b;
    b.template select<64, 1>(0) = gather<uint32_t, 64, 4>(T + OG, idx0 << 4);
    b.template select<64, 1>(64) = gather<uint32_t, 64, 4>(T + OG, idx1 << 4);
    simd<uint32_t, 128> sv;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        sv.template select<16, 1>(16 * j) = s0 >> (2 * j);
        sv.template select<16, 1>(64 + 16 * j) = s1 >> (2 * j);
    }
    b ^= (sv * 0x40008000u) & 0x80008000u;
    auto bh = b.template bit_cast_view<half>();
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        simd<half, 32> v = bh.template select<32, 1>(32 * j);
        v = v * c2;
        bh.template select<32, 1>(32 * j) = v * d2;
    }
    return b;
}
constexpr auto kPfL1h = sycl::ext::oneapi::experimental::properties{cache_hint_L1<cache_hint::cached>, cache_hint_L2<cache_hint::cached>};
SYCL_ESIMD_FUNCTION inline simd<half, 16> half16(simd<uint32_t, 16> h16) {
    simd<uint16_t, 16> lo = h16.template bit_cast_view<uint16_t>().template select<16, 2>(0);
    return lo.template bit_cast_view<half>();
}

template <int F, int MT, int NG, int G> struct TagGuH;
template <int F, int MT, int NT, int G> struct TagDnH;

// gate/up: rows (rbg's MT x 8) x (NG gate tiles of 16 rows from 16 NG cb, the same up rows), H out
template <int F, int MT, int NG, int G = kMaxGroup>
void launch_gu_h(sycl::queue& q, const HArgsT<G> a, int nrb, int ncb, int wgn) {
    constexpr int NT = 2 * NG;
    constexpr int BB = blk_bytes<F>();
    constexpr int OG = F == T_IQ2S ? O_HG2S : O_HG2XXS;
    static_assert(MT == 1 || MT == 2 || MT == 4 || MT == 8, "row tiles: 8-row block loads (MT 1, 2) or 32-row ones (4, 8)");
    constexpr int LR = MT >= 4 ? 32 : 8;   // rows per activation block load
    constexpr int NL = MT * 8 / LR;        // block loads per step
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagGuH<F, MT, NG, G>>(sycl::nd_range<2>(sycl::range<2>((size_t) nrb, (size_t) ncb), sycl::range<2>(1, (size_t) wgn)), props,
                                          [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
            const int rbg = (int) it.get_group(0), cb = (int) it.get_global_id(1);
            const HPick pk = pick_h(a, rbg);
            const int r0 = (rbg - pk.rb0) * (8 * MT);
            const int left = pk.cnt - r0;
            const int nvalid = left >= 8 * MT ? MT : (left + 7) >> 3;
            const half* Xe = a.X + (size_t) pk.row0 * a.K;
            const uint32_t sw = (uint32_t) a.K * 2u - 1u, sh = (uint32_t) pk.cnt - 1u;
            const uint8_t* We = pk.blob;
            const uint32_t* T = a.tab;
            simd<uint32_t, 16> lane(0, 1);
            simd<uint32_t, 16> ro[NT];
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                ro[g] = (lane + (uint32_t) (16 * (NG * cb + g))) * (uint32_t) a.rowb;
                ro[NG + g] = ro[g] + a.sec_off;
            }
            simd<float, 128> acc[MT][NT];
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int t = 0; t < NT; ++t) acc[i][t] = 0.f;
            const int nsb = a.K / 256;
            for (int sb = 0; sb < nsb; ++sb) {
                const uint32_t P = (uint32_t) sb * BB;
                if (sb + 1 < nsb && !(a.dbg & 8)) {   // the next superblock's bytes of each lane's row into L1 (its loads then hit it)
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        simd<uint32_t, 16> l0 = (ro[t] + (P + BB)) & ~63u;
#pragma unroll
                        for (int c = 0; c < (BB + 63) / 64 + 1; ++c)
                            prefetch<uint32_t, 16>(reinterpret_cast<const uint32_t*>(We), l0 + 64u * c, kPfL1h);
                    }
                }
                simd<half, 32> d2[NT];
                simd<uint32_t, 16> w0[NT], w1[NT], w2[NT], w3[NT];
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    d2[t] = pair_of(half16(ld16(We, ro[t] + P)));
                    if constexpr (F == T_IQ2S) {
                        w0[t] = ld32(We, ro[t] + (P + 74));   // scales 0..3
                        w1[t] = ld32(We, ro[t] + (P + 78));   // scales 4..7
                        w2[t] = ld32(We, ro[t] + (P + 66));   // qh 0..3
                        w3[t] = ld32(We, ro[t] + (P + 70));   // qh 4..7
                    }
                }
                // a 32-block's index and sign dwords, loaded one 32-block ahead (a macro, not a lambda: see LOAD_CHUNK)
#define LDQ_H(IB, XA, XB)                                                                                    \
    _Pragma("unroll") for (int t = 0; t < NT; ++t) {                                                         \
        if constexpr (F == T_IQ2S) {                                                                         \
            XA[t] = ld32(We, ro[t] + (P + 2 + 4 * (IB)));    /* grid index low bytes */                      \
            XB[t] = ld32(We, ro[t] + (P + 34 + 4 * (IB)));   /* signs */                                     \
        } else {                                                                                             \
            XA[t] = ld32(We, ro[t] + (P + 2 + 8 * (IB)));    /* grid indices */                              \
            XB[t] = ld32(We, ro[t] + (P + 6 + 8 * (IB)));    /* signs (4 x 7 bits) + scale */                \
        }                                                                                                    \
    }
                simd<uint32_t, 16> qa[NT], qb[NT];
                LDQ_H(0, qa, qb)
#pragma unroll 1
                for (int ib = 0; ib < 8; ++ib) {
                    simd<uint32_t, 16> na[NT], nb[NT];
                    if (ib < 7) { LDQ_H(ib + 1, na, nb) }
#pragma unroll
                    for (int hf = 0; hf < 2; ++hf) {
                        const int k0 = 256 * sb + 32 * ib + 16 * hf;
                        simd<half, 16 * LR> av[NL];
#pragma unroll
                        for (int p = 0; p < NL; ++p) av[p] = load_2d<half, 16, LR, 1>(Xe, sw, sh, sw, k0, r0 + LR * p);   // (past the expert: zeros)
                        simd<uint32_t, 128> bt[NT];
#pragma unroll
                        for (int t = 0; t < NT; ++t) {
                            simd<uint32_t, 16> i0, i1, s0, s1, c1;
                            if constexpr (F == T_IQ2S) {
                                const int il = 2 * hf;
                                simd<uint32_t, 16> qhb = (((ib < 4) ? w2[t] : w3[t]) >> (8 * (ib & 3))) & 0xffu;
                                simd<uint32_t, 16> scb = (((ib < 4) ? w0[t] : w1[t]) >> (8 * (ib & 3))) & 0xffu;
                                i0 = ((qa[t] >> (8 * il)) & 0xffu) | ((qhb << (8 - 2 * il)) & 0x300u);
                                i1 = ((qa[t] >> (8 * il + 8)) & 0xffu) | ((qhb << (6 - 2 * il)) & 0x300u);
                                s0 = (qb[t] >> (8 * il)) & 0xffu;
                                s1 = (qb[t] >> (8 * il + 8)) & 0xffu;
                                c1 = ((scb >> (4 * hf)) & 0xfu) * 2u + 1u;
                            } else {
                                const int il = 2 * hf;
                                i0 = (qa[t] >> (8 * il)) & 0xffu;
                                i1 = (qa[t] >> (8 * il + 8)) & 0xffu;
                                s0 = ksigns((qb[t] >> (7 * il)) & 127u);
                                s1 = ksigns((qb[t] >> (7 * il + 7)) & 127u);
                                c1 = (qb[t] >> 28) * 2u + 1u;
                            }
                            simd<half, 16> c1h = simd<half, 16>(simd<float, 16>(simd<int, 16>(c1)));
                            bt[t] = dec_iq2_h<OG>(T, i0, i1, s0, s1, pair_of(c1h), d2[t]);
                        }
                        // every row tile, also the ones past the expert's rows (zeros, not stored): a branch per tile was slower (569
                        // against 524 us)
#pragma unroll
                        for (int i = 0; i < MT; ++i) {
                            simd<half, 128> ai = av[i * 8 / LR].template select<128, 1>(128 * (i % (LR / 8)));
#pragma unroll
                            for (int t = 0; t < NT; ++t) acc[i][t] = dpas_h(acc[i][t], bt[t], ai);
                        }
                    }
#pragma unroll
                    for (int t = 0; t < NT; ++t) { qa[t] = na[t]; qb[t] = nb[t]; }
                }
#undef LDQ_H
            }
            if (a.dbg & 4) return;
            half* He = reinterpret_cast<half*>(a.Y) + (size_t) pk.row0 * a.n_out;
            const uint32_t hw = (uint32_t) a.n_out * 2u - 1u;
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                if (i < nvalid) {
#pragma unroll
                    for (int g = 0; g < NG; ++g) {
                        simd<float, 128> gg = acc[i][g] * 0.125f, uu = acc[i][NG + g] * 0.125f;
                        simd<float, 128> hv = gg / (1.f + exp(-gg)) * uu;
                        hv = max(min(hv, 65504.f), -65504.f);
                        simd<half, 128> hh = hv;
                        store_2d<half, 16, 8>(He, hw, sh, hw, 16 * (NG * cb + g), r0 + 8 * i, hh);
                    }
                }
            }
        });
    });
}

// down (Q2_0): rows (rbg's MT x 8) x NT tiles of 16 output columns from 16 NT cb, FP32 out
template <int F, int MT, int NT, int G = kMaxGroup>
void launch_dn_h(sycl::queue& q, const HArgsT<G> a, int nrb, int ncb, int wgn) {
    static_assert(F == T_Q2_0, "down: Q2_0");
    static_assert(MT == 1 || MT == 2 || MT == 4 || MT == 8, "row tiles: 8-row block loads (MT 1, 2) or 32-row ones (4, 8)");
    constexpr int LR = MT >= 4 ? 32 : 8;   // rows per activation block load
    constexpr int NL = MT * 8 / LR;        // block loads per step
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagDnH<F, MT, NT, G>>(sycl::nd_range<2>(sycl::range<2>((size_t) nrb, (size_t) ncb), sycl::range<2>(1, (size_t) wgn)), props,
                                          [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
            const int rbg = (int) it.get_group(0), cb = (int) it.get_global_id(1);
            const HPick pk = pick_h(a, rbg);
            const int r0 = (rbg - pk.rb0) * (8 * MT);
            const int left = pk.cnt - r0;
            const int nvalid = left >= 8 * MT ? MT : (left + 7) >> 3;
            const half* Xe = a.X + (size_t) pk.row0 * a.K;
            const uint32_t sw = (uint32_t) a.K * 2u - 1u, sh = (uint32_t) pk.cnt - 1u;
            const uint8_t* We = pk.blob;
            const uint32_t* T = a.tab;
            simd<uint32_t, 16> lane(0, 1);
            simd<uint32_t, 16> ro[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) ro[t] = (lane + (uint32_t) (16 * (NT * cb + t))) * (uint32_t) a.rowb + a.sec_off;
            if (!(a.dbg & 8)) {   // each lane's whole row (Q2_0: 180 bytes) into L1, kept inside the matrix
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    simd<uint32_t, 16> l0 = ro[t] & ~63u;
#pragma unroll
                    for (int c = 0; c < 4; ++c)
                        prefetch<uint32_t, 16>(reinterpret_cast<const uint32_t*>(We), min(l0 + 64u * c, simd<uint32_t, 16>(a.lim & ~63u)), kPfL1h);
                }
            }
            simd<float, 128> acc[MT][NT];
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int t = 0; t < NT; ++t) acc[i][t] = 0.f;
            const int nblk = a.K / 64;
#pragma unroll 1
            for (int kb = 0; kb < nblk; ++kb) {
                const uint32_t P = (uint32_t) kb * 18;
                simd<half, 32> d2[NT];
                simd<uint32_t, 16> qw[NT][4];
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    d2[t] = pair_of(half16(ld16(We, ro[t] + P)));
#pragma unroll
                    for (int j = 0; j < 4; ++j) qw[t][j] = ld32c(We, ro[t] + (P + 2 + 4 * j), a.lim);
                }
#pragma unroll
                for (int ks = 0; ks < 4; ++ks) {
                    const int k0 = 64 * kb + 16 * ks;
                    simd<half, 16 * LR> av[NL];
#pragma unroll
                    for (int p = 0; p < NL; ++p) av[p] = load_2d<half, 16, LR, 1>(Xe, sw, sh, sw, k0, r0 + LR * p);   // (past the expert: zeros)
                    simd<uint32_t, 128> bt[NT];
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
#pragma unroll
                        for (int j = 0; j < 4; ++j)
                            bt[t].template select<32, 1>(32 * j) = gather<uint32_t, 32, 2>(T + O_HQ2, ((qw[t][ks] >> (8 * j)) & 0xffu) << 3);
                        auto bh = bt[t].template bit_cast_view<half>();
#pragma unroll
                        for (int j = 0; j < 8; ++j) bh.template select<32, 1>(32 * j) = bh.template select<32, 1>(32 * j) * d2[t];
                    }
#pragma unroll
                    for (int i = 0; i < MT; ++i) {
                        simd<half, 128> ai = av[i * 8 / LR].template select<128, 1>(128 * (i % (LR / 8)));
#pragma unroll
                        for (int t = 0; t < NT; ++t) acc[i][t] = dpas_h(acc[i][t], bt[t], ai);
                    }
                }
            }
            if (a.dbg & 4) return;
            float* De = reinterpret_cast<float*>(a.Y) + (size_t) pk.row0 * a.n_out;
            const uint32_t dw = (uint32_t) a.n_out * 4u - 1u;
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                if (i < nvalid) {
#pragma unroll
                    for (int t = 0; t < NT; ++t) store_2d<float, 16, 8>(De, dw, sh, dw, 16 * (NT * cb + t), r0 + 8 * i, acc[i][t]);
                }
            }
        });
    });
}

// ------------------------------------------------------------------------------------------------ FP16 products, the other formats
// STRATA_F16X_ANY=1 (off by default; f16_handles): the FP16 products for expert formats launch_gu_h / launch_dn_h do not take.  One
// kernel, launch_x, for gate/up and for down, with their tile: a thread owns 8 MT rows of one expert and two tiles of 16 output
// columns (16 gate rows and the same 16 up rows with SwiGLU in the epilogue, or 32 down columns), walks K in steps of 16, reads the
// FP16 activations with 2-D block loads and adds one FP16 DPAS per row tile and column tile into FP32 accumulators.
//  * One decode for every format.  Per 32 of K and tile the weights become 32 signed code bytes (cq: dword j holds values 4 j ..
//    4 j + 3), a float scale per 16 of K (cs) and a float offset (cm); x_pack makes the B operand as FP16(cs * code + cm).  That is
//    the float expression iq_dequant_expert_f16 evaluates for the format, and its products are exact in FP32 (an 11-bit d times
//    integer factors, 24 bits at most), so the weights are the FP16 chain's and a product differs from the chain in its summation
//    order only.  What a format adds is its unpack to code bytes and its scales: the blocks in the K loop below.
//  * Formats.  Blocks of 256 values (gate/up; K a multiple of 256): Q4_K, Q5_K, Q6_K, IQ4_XS, IQ3_S, IQ3_XXS, IQ2_XS.  Blocks of 32
//    values (gate/up and down; K a multiple of 32): Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL.  IQ2_S, IQ2_XXS and Q2_0 keep the kernels
//    above; IQ1_M and the trellis types have no FP16 products.
//  * Reads stay inside the matrix (a.lim: the byte offset of its last dword): a dword gather starts at a dword inside it, ld32c and
//    ld4u clamp the one dword they may read past the bytes they need.  Row bytes are a multiple of 4 (x_shape).
//  * Not tuned: the next 32-block is not loaded ahead as in launch_gu_h, and the IQ4 table costs four gathers a dword.
constexpr int T_Q4_0 = 2, T_Q4_1 = 3, T_Q5_0 = 6, T_Q5_1 = 7, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14;
constexpr int x_b8(int f) {   // bytes per 256 values (0: not a format of launch_x)
    return f == T_Q4_0 || f == T_IQ4NL || f == T_Q4_K ? 144 : f == T_Q4_1 ? 160 : f == T_Q5_0 || f == T_Q5_K ? 176 : f == T_Q5_1 ? 192 :
           f == T_Q8_0 ? 272 : f == T_Q6_K ? 210 : f == T_IQ4XS ? 136 : f == T_IQ3S ? 110 : f == T_IQ3XXS ? 98 : f == T_IQ2XS ? 74 : 0;
}
constexpr bool x_blk32(int f) {   // blocks of 32 values (the others: 256)
    return f == T_Q4_0 || f == T_Q4_1 || f == T_Q5_0 || f == T_Q5_1 || f == T_Q8_0 || f == T_IQ4NL;
}
constexpr bool x_has_m(int f) { return f == T_Q4_1 || f == T_Q5_1 || f == T_Q4_K || f == T_Q5_K; }   // w = scale * code + offset
// the roles launch_x is built for (the engine's gate/up and down formats, native_expert_supported, without the ones named above)
constexpr bool x_gu(int f) {
    return f == T_Q4_K || f == T_Q5_K || f == T_Q6_K || f == T_Q8_0 || f == T_IQ3S || f == T_IQ3XXS || f == T_IQ2XS || f == T_IQ4XS ||
           f == T_Q4_0 || f == T_Q4_1 || f == T_Q5_0;
}
constexpr bool x_dn(int f) { return f == T_IQ4NL || f == T_Q5_1 || f == T_Q8_0 || f == T_Q4_0 || f == T_Q4_1 || f == T_Q5_0; }

// four consecutive dwords ([dword j][lane]) at a dword-aligned byte offset
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 64> ld4a(const uint8_t* base, simd<uint32_t, 16> off) {
    return gather<uint32_t, 64, 4>(reinterpret_cast<const uint32_t*>(base), off);
}
// the same at a 2-byte aligned byte offset: the four dwords from the dword below it and, for a lane 2 bytes off, the low half of the
// fifth.  The 16 bytes are inside the matrix, so a fifth dword past `lim` belongs to a lane that is not 2 bytes off and does not use it.
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 64> ld4u(const uint8_t* base, simd<uint32_t, 16> off, uint32_t lim) {
    simd<uint32_t, 16> a = off & ~3u;
    simd<uint32_t, 64> v = gather<uint32_t, 64, 4>(reinterpret_cast<const uint32_t*>(base), a);
    simd<uint32_t, 16> e = gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(base), min(a + 16u, simd<uint32_t, 16>(lim)));
    simd<uint32_t, 16> s = (off & 2u) << 3;   // 0 or 16
    simd<uint32_t, 16> d0 = v.template select<16, 1>(0), d1 = v.template select<16, 1>(16);
    simd<uint32_t, 16> d2 = v.template select<16, 1>(32), d3 = v.template select<16, 1>(48);
    simd<uint32_t, 64> r;
    r.template select<16, 1>(0) = (d0 >> s) | ((d1 << 16) << (16u - s));
    r.template select<16, 1>(16) = (d1 >> s) | ((d2 << 16) << (16u - s));
    r.template select<16, 1>(32) = (d2 >> s) | ((d3 << 16) << (16u - s));
    r.template select<16, 1>(48) = (d3 >> s) | ((e << 16) << (16u - s));
    return r;
}
// each byte u (< 128) of the dwords as the int8 u - bias (bias4: the bias, at most 127, in its four bytes)
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> x_sub8(simd<uint32_t, 16> u, uint32_t bias4) {
    return ((u | 0x80808080u) - bias4) ^ 0x80808080u;
}
// the four bits of n (one nibble a lane) to bit 4 of the dword's four bytes
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> x_bit4(simd<uint32_t, 16> n) {
    return ((n * 0x00204081u) & 0x01010101u) << 4;
}
// byte i (0..11) of the 12 bytes in w0, w1, w2
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> x_b12(simd<uint32_t, 16> w0, simd<uint32_t, 16> w1, simd<uint32_t, 16> w2, int i) {
    simd<uint32_t, 16> w = w2;
    if (i < 4) w = w0;
    else if (i < 8) w = w1;
    return (w >> (uint32_t) (8 * (i & 3))) & 0xffu;
}
// a dword of IQ4_NL / IQ4_XS codes: [0, 16) the values of its four low nibbles, [16, 32) of its four high nibbles (O_IQ4: a byte's two)
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 32> x_iq4(const uint32_t* T, simd<uint32_t, 16> qd) {
    simd<uint32_t, 16> t0 = gather<uint32_t, 16>(T + O_IQ4, (qd & 0xffu) << 2);
    simd<uint32_t, 16> t1 = gather<uint32_t, 16>(T + O_IQ4, ((qd >> 8) & 0xffu) << 2);
    simd<uint32_t, 16> t2 = gather<uint32_t, 16>(T + O_IQ4, ((qd >> 16) & 0xffu) << 2);
    simd<uint32_t, 16> t3 = gather<uint32_t, 16>(T + O_IQ4, (qd >> 24) << 2);
    simd<uint32_t, 32> r;
    r.template select<16, 1>(0) = (t0 & 0xffu) | ((t1 & 0xffu) << 8) | ((t2 & 0xffu) << 16) | (t3 << 24);
    r.template select<16, 1>(16) = ((t0 >> 8) & 0xffu) | (t1 & 0xff00u) | ((t2 & 0xff00u) << 8) | ((t3 & 0xff00u) << 16);
    return r;
}
// Four values' code bytes (one dword a lane) as FP16(s * code + m), rounded once: two K pairs in VNNI order ([pair][lane], a dword
// holding the pair's two values).
template <bool HASM>
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 32> x_pack4(simd<uint32_t, 16> w, simd<float, 16> s, simd<float, 16> m) {
    auto cb = w.template bit_cast_view<int8_t>();
    simd<uint32_t, 32> b;
    auto bh = b.template bit_cast_view<half>();
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        simd<int8_t, 16> c0 = cb.template select<16, 4>(2 * p);
        simd<int8_t, 16> c1 = cb.template select<16, 4>(2 * p + 1);
        simd<float, 16> f0 = simd<float, 16>(simd<int, 16>(c0)) * s;
        simd<float, 16> f1 = simd<float, 16>(simd<int, 16>(c1)) * s;
        if constexpr (HASM) {
            f0 += m;
            f1 += m;
        }
        simd<half, 32> v;
        v.template select<16, 2>(0) = simd<half, 16>(f0);
        v.template select<16, 2>(1) = simd<half, 16>(f1);
        bh.template select<32, 1>(32 * p) = v;
    }
    return b;
}
// The B operand of 16 K values (lane = output column) from their code bytes in q0..q3 (dword j: values 4 j .. 4 j + 3), in VNNI
// order (dword j of a lane holds K values 2 j and 2 j + 1).
template <bool HASM>
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 128> x_pack(simd<uint32_t, 16> q0, simd<uint32_t, 16> q1, simd<uint32_t, 16> q2,
                                                      simd<uint32_t, 16> q3, simd<float, 16> s, simd<float, 16> m) {
    simd<uint32_t, 128> b;
    b.template select<32, 1>(0) = x_pack4<HASM>(q0, s, m);
    b.template select<32, 1>(32) = x_pack4<HASM>(q1, s, m);
    b.template select<32, 1>(64) = x_pack4<HASM>(q2, s, m);
    b.template select<32, 1>(96) = x_pack4<HASM>(q3, s, m);
    return b;
}

template <int F, int MT, bool GU, int G> struct TagX;

// GU: gate/up (the group's rows x 16 gate rows from 16 cb and the same up rows, H out); else down (x 32 columns from 32 cb, FP32 out)
template <int F, int MT, bool GU, int G = kMaxGroupH>
void launch_x(sycl::queue& q, const HArgsT<G> a, int nrb, int ncb, int wgn) {
    constexpr int NT = 2;
    constexpr int B8 = x_b8(F);
    constexpr int BS = B8 / 8;   // a block's bytes (the formats of x_blk32)
    constexpr bool HASM = x_has_m(F);
    static_assert(B8 != 0, "a format of x_b8");
    static_assert(MT == 1 || MT == 2 || MT == 4 || MT == 8, "row tiles: 8-row block loads (MT 1, 2) or 32-row ones (4, 8)");
    constexpr int LR = MT >= 4 ? 32 : 8;   // rows per activation block load
    constexpr int NL = MT * 8 / LR;        // block loads per step
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagX<F, MT, GU, G>>(sycl::nd_range<2>(sycl::range<2>((size_t) nrb, (size_t) ncb), sycl::range<2>(1, (size_t) wgn)), props,
                                         [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
            const int rbg = (int) it.get_group(0), cb = (int) it.get_global_id(1);
            const HPick pk = pick_h(a, rbg);
            const int r0 = (rbg - pk.rb0) * (8 * MT);
            const int left = pk.cnt - r0;
            const int nvalid = left >= 8 * MT ? MT : (left + 7) >> 3;
            const half* Xe = a.X + (size_t) pk.row0 * a.K;
            const uint32_t sw = (uint32_t) a.K * 2u - 1u, sh = (uint32_t) pk.cnt - 1u;
            const uint8_t* We = pk.blob;
            const uint32_t* T = a.tab;
            const uint32_t lim = a.lim;
            simd<uint32_t, 16> lane(0, 1);
            simd<uint32_t, 16> ro[NT];
            if constexpr (GU) {
                ro[0] = (lane + (uint32_t) (16 * cb)) * (uint32_t) a.rowb;
                ro[1] = ro[0] + a.sec_off;
            } else {
#pragma unroll
                for (int t = 0; t < NT; ++t) ro[t] = (lane + (uint32_t) (16 * (NT * cb + t))) * (uint32_t) a.rowb + a.sec_off;
            }
            simd<float, 128> acc[MT][NT];
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int t = 0; t < NT; ++t) acc[i][t] = 0.f;
            const int nkb = a.K >> 5;   // 32-blocks of K
            for (int sb = 0; 8 * sb < nkb; ++sb) {
                const uint32_t P = (uint32_t) sb * B8;
                const int nib = nkb - 8 * sb < 8 ? nkb - 8 * sb : 8;
                if (8 * (sb + 1) < nkb && !(a.dbg & 8)) {   // the next 256 values' bytes of each lane's row into L1, kept inside the matrix
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        simd<uint32_t, 16> l0 = (ro[t] + (P + B8)) & ~63u;
#pragma unroll
                        for (int c = 0; c < (B8 + 63) / 64 + 1; ++c)
                            prefetch<uint32_t, 16>(reinterpret_cast<const uint32_t*>(We), min(l0 + 64u * c, simd<uint32_t, 16>(lim & ~63u)), kPfL1h);
                    }
                }
                // the fields a 256-value block has once: hd its scale d, hm the K-quants' dmin, h0..h3 packed scales / high bits
                simd<float, 16> hd[NT], hm[NT];
                simd<uint32_t, 16> h0[NT], h1[NT], h2[NT], h3[NT];
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    simd<uint32_t, 16> o = ro[t] + P;
                    hd[t] = 0.f;
                    hm[t] = 0.f;
                    h0[t] = 0u;
                    h1[t] = 0u;
                    h2[t] = 0u;
                    h3[t] = 0u;
                    if constexpr (F == T_Q4_K || F == T_Q5_K) {   // d, dmin (FP16), scales[12]
                        simd<uint32_t, 64> v = ld4a(We, o);
                        simd<uint32_t, 16> dm = v.template select<16, 1>(0);
                        hd[t] = half_of(dm);
                        hm[t] = half_of(dm >> 16);
                        h0[t] = v.template select<16, 1>(16);
                        h1[t] = v.template select<16, 1>(32);
                        h2[t] = v.template select<16, 1>(48);
                    } else if constexpr (F == T_Q6_K) {           // ql[128], qh[64], scales[16] (int8), d
                        hd[t] = half_of(ld16(We, o + 208u));
                        simd<uint32_t, 64> v = ld4u(We, o + 192u, lim);
                        h0[t] = v.template select<16, 1>(0);
                        h1[t] = v.template select<16, 1>(16);
                        h2[t] = v.template select<16, 1>(32);
                        h3[t] = v.template select<16, 1>(48);
                    } else if constexpr (F == T_IQ4XS) {          // d, scales_h, scales_l[4], qs[128]
                        hd[t] = half_of(ld16(We, o));
                        h0[t] = ld16(We, o + 2u);
                        h1[t] = ld32c(We, o + 4u, lim);
                    } else if constexpr (F == T_IQ3S) {           // d, qs[64], qh[8], signs[32], scales[4]
                        hd[t] = half_of(ld16(We, o));
                        h0[t] = ld32c(We, o + 106u, lim);   // scales
                        h1[t] = ld32c(We, o + 66u, lim);    // qh 0..3
                        h2[t] = ld32c(We, o + 70u, lim);    // qh 4..7
                    } else if constexpr (F == T_IQ3XXS) {         // d, qs[64], scales and signs[32]
                        hd[t] = half_of(ld16(We, o));
                    } else if constexpr (F == T_IQ2XS) {          // d, qs[32] (16 bit), scales[8]
                        hd[t] = half_of(ld16(We, o));
                        h0[t] = ld32c(We, o + 66u, lim);    // scales 0..3
                        h1[t] = ld32c(We, o + 70u, lim);    // scales 4..7
                    }
                }
#pragma unroll 1
                for (int ib = 0; ib < nib; ++ib) {
                    simd<uint32_t, 16> cq[NT][8];
                    simd<float, 16> cs[NT][2], cm[NT];
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        simd<uint32_t, 16> o = ro[t] + P;
                        cm[t] = 0.f;
                        if constexpr (F == T_Q4_K || F == T_Q5_K) {
                            // 64 values share 32 bytes: the low nibbles are its first 32 values, the high nibbles its last 32.  Q5_K: bit ib
                            // of qh's byte l is the fifth bit of value l of this 32.  Scale and min: llama.cpp's get_scale_min_k4(ib).
                            constexpr uint32_t QS = F == T_Q4_K ? 16u : 48u;
                            simd<uint32_t, 16> qo = o + (QS + 32u * (uint32_t) (ib >> 1));
                            simd<uint32_t, 64> x0 = ld4a(We, qo), x1 = ld4a(We, qo + 16u);
                            const uint32_t nsh = 4u * (uint32_t) (ib & 1);
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                simd<uint32_t, 16> xa = x0.template select<16, 1>(16 * j), xb = x1.template select<16, 1>(16 * j);
                                cq[t][j] = (xa >> nsh) & 0x0F0F0F0Fu;
                                cq[t][4 + j] = (xb >> nsh) & 0x0F0F0F0Fu;
                            }
                            if constexpr (F == T_Q5_K) {
                                simd<uint32_t, 64> hb0 = ld4a(We, o + 16u), hb1 = ld4a(We, o + 32u);
#pragma unroll
                                for (int j = 0; j < 4; ++j) {
                                    simd<uint32_t, 16> ya = hb0.template select<16, 1>(16 * j), yb = hb1.template select<16, 1>(16 * j);
                                    cq[t][j] |= ((ya >> (uint32_t) ib) & 0x01010101u) << 4;
                                    cq[t][4 + j] |= ((yb >> (uint32_t) ib) & 0x01010101u) << 4;
                                }
                            }
                            simd<uint32_t, 16> sc, mn;
                            if (ib < 4) {
                                sc = x_b12(h0[t], h1[t], h2[t], ib) & 63u;
                                mn = x_b12(h0[t], h1[t], h2[t], ib + 4) & 63u;
                            } else {
                                sc = (x_b12(h0[t], h1[t], h2[t], ib + 4) & 0xFu) | ((x_b12(h0[t], h1[t], h2[t], ib - 4) >> 6) << 4);
                                mn = (x_b12(h0[t], h1[t], h2[t], ib + 4) >> 4) | ((x_b12(h0[t], h1[t], h2[t], ib) >> 6) << 4);
                            }
                            cs[t][0] = hd[t] * simd<float, 16>(simd<int, 16>(sc));
                            cs[t][1] = cs[t][0];
                            cm[t] = 0.f - hm[t] * simd<float, 16>(simd<int, 16>(mn));
                        } else if constexpr (F == T_Q6_K) {
                            // llama.cpp's dequantize_q6_K: 128 values share 64 bytes of ql and 32 of qh; the 32 values c4 of them take ql's bytes
                            // [32 (c4 & 1), + 32), low nibbles for c4 < 2, and bits 2 c4, 2 c4 + 1 of qh's bytes; scale 2 ib + hf, minus 32
                            const int ip = ib >> 2, c4 = ib & 3;
                            simd<uint32_t, 16> lo_ = o + (uint32_t) (64 * ip + 32 * (c4 & 1));
                            simd<uint32_t, 16> ho_ = o + (uint32_t) (128 + 32 * ip);
                            simd<uint32_t, 64> x0 = ld4u(We, lo_, lim), x1 = ld4u(We, lo_ + 16u, lim);
                            simd<uint32_t, 64> hb0 = ld4u(We, ho_, lim), hb1 = ld4u(We, ho_ + 16u, lim);
                            const uint32_t nsh = 4u * (uint32_t) (c4 >> 1), qsh = 2u * (uint32_t) c4;
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                simd<uint32_t, 16> xa = x0.template select<16, 1>(16 * j), xb = x1.template select<16, 1>(16 * j);
                                simd<uint32_t, 16> ya = hb0.template select<16, 1>(16 * j), yb = hb1.template select<16, 1>(16 * j);
                                cq[t][j] = x_sub8(((xa >> nsh) & 0x0F0F0F0Fu) | (((ya >> qsh) & 0x03030303u) << 4), 0x20202020u);
                                cq[t][4 + j] = x_sub8(((xb >> nsh) & 0x0F0F0F0Fu) | (((yb >> qsh) & 0x03030303u) << 4), 0x20202020u);
                            }
                            simd<uint32_t, 16> scw = h3[t];   // scales 2 ib, 2 ib + 1: bytes 2 (ib & 1), + 1 of dword ib / 2
                            if (ib < 2) scw = h0[t];
                            else if (ib < 4) scw = h1[t];
                            else if (ib < 6) scw = h2[t];
#pragma unroll
                            for (int hf = 0; hf < 2; ++hf) {
                                simd<uint32_t, 16> s8 = (scw >> (uint32_t) (8 * (2 * (ib & 1) + hf))) & 0xffu;
                                cs[t][hf] = hd[t] * simd<float, 16>(simd<int, 16>(s8 ^ 0x80u) - 128);
                            }
                        } else if constexpr (F == T_IQ4XS) {
                            simd<uint32_t, 16> ls = ((h1[t] >> (uint32_t) (4 * ib)) & 0xfu) | (((h0[t] >> (uint32_t) (2 * ib)) & 3u) << 4);
                            cs[t][0] = hd[t] * simd<float, 16>(simd<int, 16>(ls) - 32);
                            cs[t][1] = cs[t][0];
                            simd<uint32_t, 64> x0 = ld4a(We, o + (8u + 16u * (uint32_t) ib));
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                simd<uint32_t, 16> xa = x0.template select<16, 1>(16 * j);
                                simd<uint32_t, 32> kv = x_iq4(T, xa);
                                cq[t][j] = kv.template select<16, 1>(0);
                                cq[t][4 + j] = kv.template select<16, 1>(16);
                            }
                        } else if constexpr (F == T_IQ3S) {       // as launch_gu's
                            simd<uint32_t, 16> q0 = ld32c(We, o + (2u + 8u * (uint32_t) ib), lim);
                            simd<uint32_t, 16> q1 = ld32c(We, o + (6u + 8u * (uint32_t) ib), lim);
                            simd<uint32_t, 16> sg = ld32c(We, o + (74u + 4u * (uint32_t) ib), lim);
                            simd<uint32_t, 16> qhw = h2[t];
                            if (ib < 4) qhw = h1[t];
                            simd<uint32_t, 16> qhv = (qhw >> (uint32_t) (8 * (ib & 3))) & 0xffu;
                            simd<uint32_t, 16> scn = ((h0[t] >> (uint32_t) (8 * (ib >> 1))) >> (uint32_t) (4 * (ib & 1))) & 0xfu;
                            cs[t][0] = hd[t] * simd<float, 16>(simd<int, 16>(scn * 2u + 1u));
                            cs[t][1] = cs[t][0];
#pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                simd<uint32_t, 16> qsw = r < 4 ? q0 : q1;
                                simd<uint32_t, 16> idx = ((qsw >> (8 * (r & 3))) & 0xffu) | (((qhv >> r) & 1u) << 8);
                                simd<uint32_t, 16> g = gather<uint32_t, 16>(T + O_IQ3S, idx << 2);
                                simd<uint32_t, 16> sgn = ((sg >> (8 * (r >> 1))) >> (4 * (r & 1))) & 0xfu;
                                cq[t][r] = apply_signs(g, sgn);
                            }
                        } else if constexpr (F == T_IQ3XXS) {     // as launch_gu's
                            simd<uint32_t, 16> q0 = ld32c(We, o + (2u + 8u * (uint32_t) ib), lim);
                            simd<uint32_t, 16> q1 = ld32c(We, o + (6u + 8u * (uint32_t) ib), lim);
                            simd<uint32_t, 16> aux = ld32c(We, o + (66u + 4u * (uint32_t) ib), lim);
                            cs[t][0] = hd[t] * simd<float, 16>(simd<int, 16>((aux >> 28) * 2u + 1u)) * 0.25f;
                            cs[t][1] = cs[t][0];
#pragma unroll
                            for (int il = 0; il < 4; ++il) {
                                simd<uint32_t, 16> sbyte = ksigns((aux >> (7 * il)) & 127u);
#pragma unroll
                                for (int hh = 0; hh < 2; ++hh) {
                                    const int r = 2 * il + hh;
                                    simd<uint32_t, 16> qsw = r < 4 ? q0 : q1;
                                    simd<uint32_t, 16> idx = (qsw >> (8 * (r & 3))) & 0xffu;
                                    simd<uint32_t, 16> g = gather<uint32_t, 16>(T + O_IQ3XXS, idx << 2);
                                    cq[t][r] = apply_signs(g, (sbyte >> (4 * hh)) & 0xfu);
                                }
                            }
                        } else if constexpr (F == T_IQ2XS) {      // as launch_gu's: a scale per 16 values
                            simd<uint32_t, 16> q0 = ld32c(We, o + (2u + 8u * (uint32_t) ib), lim);
                            simd<uint32_t, 16> q1 = ld32c(We, o + (6u + 8u * (uint32_t) ib), lim);
                            simd<uint32_t, 16> scw = h1[t];
                            if (ib < 4) scw = h0[t];
                            simd<uint32_t, 16> scb = (scw >> (uint32_t) (8 * (ib & 3))) & 0xffu;
                            cs[t][0] = hd[t] * simd<float, 16>(simd<int, 16>((scb & 0xfu) * 2u + 1u)) * 0.125f;
                            cs[t][1] = hd[t] * simd<float, 16>(simd<int, 16>((scb >> 4) * 2u + 1u)) * 0.125f;
#pragma unroll
                            for (int il = 0; il < 4; ++il) {
                                simd<uint32_t, 16> qsw = (il < 2) ? q0 : q1;
                                simd<uint32_t, 16> qv = (il & 1) ? (qsw >> 16) : (qsw & 0xffffu);
                                simd<uint32_t, 16> idx = qv & 511u;
                                simd<uint32_t, 32> g2 = gather<uint32_t, 32, 2>(T + O_IQ2XS, idx << 3);   // the 8-byte entry in one message
                                simd<uint32_t, 16> gl = g2.template select<16, 1>(0), gh = g2.template select<16, 1>(16);
                                simd<uint32_t, 16> sbyte = ksigns(qv >> 9);
                                cq[t][2 * il] = apply_signs(gl, sbyte & 0xfu);
                                cq[t][2 * il + 1] = apply_signs(gh, sbyte >> 4);
                            }
                        } else {
                            // the blocks of 32 values: d (FP16) first, then Q4_1 / Q5_1's offset m (FP16), Q5_0 / Q5_1's 32 high bits, the codes.
                            // A code byte's low nibble is one of the block's first 16 values, its high nibble one of the last 16.
                            simd<uint32_t, 16> bo = o + (uint32_t) (BS * ib);
                            if constexpr (F == T_Q8_0) {
                                cs[t][0] = half_of(ld16(We, bo));
                                simd<uint32_t, 64> x0 = ld4u(We, bo + 2u, lim), x1 = ld4u(We, bo + 18u, lim);
#pragma unroll
                                for (int j = 0; j < 4; ++j) {
                                    cq[t][j] = x0.template select<16, 1>(16 * j);
                                    cq[t][4 + j] = x1.template select<16, 1>(16 * j);
                                }
                            } else if constexpr (F == T_IQ4NL) {
                                cs[t][0] = half_of(ld16(We, bo));
                                simd<uint32_t, 64> x0 = ld4u(We, bo + 2u, lim);
#pragma unroll
                                for (int j = 0; j < 4; ++j) {
                                    simd<uint32_t, 16> xa = x0.template select<16, 1>(16 * j);
                                    simd<uint32_t, 32> kv = x_iq4(T, xa);
                                    cq[t][j] = kv.template select<16, 1>(0);
                                    cq[t][4 + j] = kv.template select<16, 1>(16);
                                }
                            } else if constexpr (F == T_Q4_0) {
                                cs[t][0] = half_of(ld16(We, bo));
                                simd<uint32_t, 64> x0 = ld4u(We, bo + 2u, lim);
#pragma unroll
                                for (int j = 0; j < 4; ++j) {
                                    simd<uint32_t, 16> xa = x0.template select<16, 1>(16 * j);
                                    cq[t][j] = x_sub8(xa & 0x0F0F0F0Fu, 0x08080808u);
                                    cq[t][4 + j] = x_sub8((xa >> 4) & 0x0F0F0F0Fu, 0x08080808u);
                                }
                            } else if constexpr (F == T_Q4_1) {
                                simd<uint32_t, 16> dm = gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(We), bo);
                                cs[t][0] = half_of(dm);
                                cm[t] = half_of(dm >> 16);
                                simd<uint32_t, 64> x0 = ld4a(We, bo + 4u);
#pragma unroll
                                for (int j = 0; j < 4; ++j) {
                                    simd<uint32_t, 16> xa = x0.template select<16, 1>(16 * j);
                                    cq[t][j] = xa & 0x0F0F0F0Fu;
                                    cq[t][4 + j] = (xa >> 4) & 0x0F0F0F0Fu;
                                }
                            } else if constexpr (F == T_Q5_0) {
                                cs[t][0] = half_of(ld16(We, bo));
                                simd<uint32_t, 16> qh = ld32c(We, bo + 2u, lim);
                                simd<uint32_t, 64> x0 = ld4u(We, bo + 6u, lim);
#pragma unroll
                                for (int j = 0; j < 4; ++j) {
                                    simd<uint32_t, 16> xa = x0.template select<16, 1>(16 * j);
                                    cq[t][j] = x_sub8((xa & 0x0F0F0F0Fu) | x_bit4((qh >> (4 * j)) & 0xfu), 0x10101010u);
                                    cq[t][4 + j] = x_sub8(((xa >> 4) & 0x0F0F0F0Fu) | x_bit4((qh >> (16 + 4 * j)) & 0xfu), 0x10101010u);
                                }
                            } else {   // T_Q5_1
                                simd<uint32_t, 16> dm = gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(We), bo);
                                cs[t][0] = half_of(dm);
                                cm[t] = half_of(dm >> 16);
                                simd<uint32_t, 16> qh = gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(We), bo + 4u);
                                simd<uint32_t, 64> x0 = ld4a(We, bo + 8u);
#pragma unroll
                                for (int j = 0; j < 4; ++j) {
                                    simd<uint32_t, 16> xa = x0.template select<16, 1>(16 * j);
                                    cq[t][j] = (xa & 0x0F0F0F0Fu) | x_bit4((qh >> (4 * j)) & 0xfu);
                                    cq[t][4 + j] = ((xa >> 4) & 0x0F0F0F0Fu) | x_bit4((qh >> (16 + 4 * j)) & 0xfu);
                                }
                            }
                            cs[t][1] = cs[t][0];
                        }
                    }
#pragma unroll
                    for (int hf = 0; hf < 2; ++hf) {
                        const int k0 = 256 * sb + 32 * ib + 16 * hf;
                        simd<half, 16 * LR> av[NL];
#pragma unroll
                        for (int p = 0; p < NL; ++p) av[p] = load_2d<half, 16, LR, 1>(Xe, sw, sh, sw, k0, r0 + LR * p);   // (past the expert: zeros)
                        simd<uint32_t, 128> bt[NT];
#pragma unroll
                        for (int t = 0; t < NT; ++t)
                            bt[t] = x_pack<HASM>(cq[t][4 * hf], cq[t][4 * hf + 1], cq[t][4 * hf + 2], cq[t][4 * hf + 3], cs[t][hf], cm[t]);
#pragma unroll
                        for (int i = 0; i < MT; ++i) {
                            simd<half, 128> ai = av[i * 8 / LR].template select<128, 1>(128 * (i % (LR / 8)));
#pragma unroll
                            for (int t = 0; t < NT; ++t) acc[i][t] = dpas_h(acc[i][t], bt[t], ai);
                        }
                    }
                }
            }
            if (a.dbg & 4) return;
            if constexpr (GU) {
                half* He = reinterpret_cast<half*>(a.Y) + (size_t) pk.row0 * a.n_out;
                const uint32_t hw = (uint32_t) a.n_out * 2u - 1u;
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                    if (i < nvalid) {
                        simd<float, 128> gg = acc[i][0], uu = acc[i][1];
                        simd<float, 128> hv = gg / (1.f + exp(-gg)) * uu;
                        hv = max(min(hv, 65504.f), -65504.f);
                        simd<half, 128> hh = hv;
                        store_2d<half, 16, 8>(He, hw, sh, hw, 16 * cb, r0 + 8 * i, hh);
                    }
                }
            } else {
                float* De = reinterpret_cast<float*>(a.Y) + (size_t) pk.row0 * a.n_out;
                const uint32_t dw = (uint32_t) a.n_out * 4u - 1u;
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                    if (i < nvalid) {
#pragma unroll
                        for (int t = 0; t < NT; ++t) store_2d<float, 16, 8>(De, dw, sh, dw, 16 * (NT * cb + t), r0 + 8 * i, acc[i][t]);
                    }
                }
            }
        });
    });
}

int hdbg_flags() {
    static const int v = [] { const char* s = std::getenv("STRATA_F16X_DBG"); return s ? std::atoi(s) : 0; }();
    return v;
}
// tuning: STRATA_F16X_WG, the threads of a work-group (8: 5 % faster than 4 or 10 at 16 experts x 320 rows, 1 was 40 % slower)
int henv(const char* name, int def) {
    const char* s = std::getenv(name);
    return s ? std::atoi(s) : def;
}
bool al64(const void* p) { return ((uintptr_t) p & 63) == 0; }

// STRATA_F16X_ANY (read once; unset or 0: off): f16_handles also takes the formats of launch_x
bool x_any_on() {
    static const bool v = [] { const char* e = std::getenv("STRATA_F16X_ANY"); return e != nullptr && e[0] != '\0' && e[0] != '0'; }();
    return v;
}
int x_rowb(int f, int K) { return x_blk32(f) ? (K / 32) * (x_b8(f) / 8) : (K / 256) * x_b8(f); }   // bytes of a row of K values
// K is whole blocks of the format and a row is whole dwords (ld4a / ld4u, and the matrix then ends on a dword)
bool x_shape(int f, int K) { return K > 0 && K % (x_blk32(f) ? 32 : 256) == 0 && (x_rowb(f, K) & 3) == 0; }
// launch_x for a group of up to kMaxGroupH experts (cnt[i] rows each, back to back): gate/up (GU; sec_off the up rows' byte offset, n_out
// = n_ff, Y = H) or down (sec_off the down rows' byte offset, n_out = n_embd, Y = Dm).  `mt` 1, 2 or 8; `skip` as gate_up_f16_n's.
template <bool GU>
void x_n(int type, const uint8_t* const* blob, size_t sec_off, int n_out, const int32_t* cnt, int n, int K, const uint16_t* X, void* Y,
         sycl::queue& q, int mt, uint64_t skip) {
    HArgsT<kMaxGroupH> a{};
    for (int i = 0; i < n; ++i) a.blob[i] = blob[i];
    a.sec_off = (uint32_t) sec_off;
    a.X = reinterpret_cast<const half*>(X);
    a.Y = Y;
    a.tab = tables(q);
    a.K = K;
    a.n_out = n_out;
    a.rowb = x_rowb(type, K);
    a.lim = (uint32_t) (sec_off + (size_t) n_out * (size_t) a.rowb) - 4u;   // the last dword of the up / down matrix
    a.dbg = hdbg_flags();
    static const int wg = henv("STRATA_F16X_WG", 8);
    const int nrb = fill_hargs_n(a, cnt, n, 8 * mt, skip);
    if (nrb == 0) return;
    const int ncb = GU ? n_out / 16 : n_out / 32;
    const int wgn = ncb % wg == 0 ? wg : 1;
#define STRATA_X(F_)                                                    \
    case F_:                                                            \
        if (mt == 1) launch_x<F_, 1, GU>(q, a, nrb, ncb, wgn);          \
        else if (mt == 2) launch_x<F_, 2, GU>(q, a, nrb, ncb, wgn);     \
        else launch_x<F_, 8, GU>(q, a, nrb, ncb, wgn);                  \
        break;
    if constexpr (GU) {   // the formats of x_gu
        switch (type) {
            STRATA_X(T_Q4_K) STRATA_X(T_Q5_K) STRATA_X(T_Q6_K) STRATA_X(T_Q8_0) STRATA_X(T_IQ3S) STRATA_X(T_IQ3XXS) STRATA_X(T_IQ2XS)
            STRATA_X(T_IQ4XS) STRATA_X(T_Q4_0) STRATA_X(T_Q4_1) STRATA_X(T_Q5_0)
            default: break;
        }
    } else {              // the formats of x_dn
        switch (type) {
            STRATA_X(T_IQ4NL) STRATA_X(T_Q5_1) STRATA_X(T_Q8_0) STRATA_X(T_Q4_0) STRATA_X(T_Q4_1) STRATA_X(T_Q5_0)
            default: break;
        }
    }
#undef STRATA_X
}
}  // namespace

bool f16_handles(int gu_type, int d_type, int n_ff, int n_embd, void* stream) {
    const bool gu_h = gu_type == T_IQ2S || gu_type == T_IQ2XXS, d_h = d_type == T_Q2_0;
    if (!gu_h || !d_h) {   // a format of launch_x: STRATA_F16X_ANY
        if (!x_any_on()) return false;
        if (!gu_h && !(x_gu(gu_type) && x_shape(gu_type, n_embd))) return false;
        if (!d_h && !(x_dn(d_type) && x_shape(d_type, n_ff))) return false;
    }
    return n_embd % 256 == 0 && n_ff % 64 == 0 && n_ff % 32 == 0 && n_embd % 64 == 0 && available(stream);
}

bool gate_up_f16(int gu_type, const uint8_t* const* blob, size_t up_off, int n_ff, const Layout& lo, int K, const uint16_t* X,
                 uint16_t* H, void* stream) {
    if (lo.n == 0 || lo.row0[lo.n] == 0) return true;
    if (gu_type != T_IQ2S && gu_type != T_IQ2XXS) {   // launch_x (the 64-expert form at 64 rows a thread: the same bits as a 16-expert one)
        if (!x_gu(gu_type) || !x_shape(gu_type, K) || K % 256 != 0 || n_ff % 32 != 0 || (up_off & 3) != 0 || lo.n > kMaxGroup ||
            !al64(X) || !al64(H))
            return false;
        int32_t cnt[kMaxGroup];
        for (int i = 0; i < lo.n; ++i) cnt[i] = lo.cnt[i];
        x_n<true>(gu_type, blob, up_off, n_ff, cnt, lo.n, K, X, H, *strata::q_of(stream), 8, 0);
        return true;
    }
    if (K % 256 != 0 || n_ff % 32 != 0 || (up_off & 3) != 0 || lo.n > kMaxGroup || !al64(X) || !al64(H)) return false;
    sycl::queue& q = *strata::q_of(stream);
    HArgs a{};
    for (int i = 0; i < lo.n; ++i) a.blob[i] = blob[i];
    a.sec_off = (uint32_t) up_off;
    a.X = reinterpret_cast<const half*>(X);
    a.Y = H;
    a.tab = htables(q);
    a.K = K;
    a.n_out = n_ff;
    a.rowb = (K / 256) * block_bytes_of(gu_type);
    a.dbg = hdbg_flags();
    static const int wg = henv("STRATA_F16X_WG", 8);
    const int nrb = fill_hargs(a, lo, 64);
    const int ncb = n_ff / 16;
    const int wgn = ncb % wg == 0 ? wg : 1;
#define STRATA_GU_H(F_) launch_gu_h<F_, 8, 1>(q, a, nrb, ncb, wgn);
    if (gu_type == T_IQ2S) { STRATA_GU_H(T_IQ2S) } else { STRATA_GU_H(T_IQ2XXS) }
#undef STRATA_GU_H
    return true;
}

bool down_f16(int d_type, const uint8_t* const* blob, size_t down_off, int n_embd, const Layout& lo, int n_ff, const uint16_t* H,
              float* Dm, void* stream) {
    if (lo.n == 0 || lo.row0[lo.n] == 0) return true;
    if (d_type != T_Q2_0) {   // launch_x
        if (!x_dn(d_type) || !x_shape(d_type, n_ff) || n_ff % 64 != 0 || n_embd % 64 != 0 || (down_off & 3) != 0 || lo.n > kMaxGroup ||
            !al64(H) || !al64(Dm))
            return false;
        int32_t cnt[kMaxGroup];
        for (int i = 0; i < lo.n; ++i) cnt[i] = lo.cnt[i];
        x_n<false>(d_type, blob, down_off, n_embd, cnt, lo.n, n_ff, H, Dm, *strata::q_of(stream), 8, 0);
        return true;
    }
    if (n_ff % 64 != 0 || n_embd % 64 != 0 || (down_off & 3) != 0 || lo.n > kMaxGroup || !al64(H) || !al64(Dm)) return false;
    sycl::queue& q = *strata::q_of(stream);
    HArgs a{};
    for (int i = 0; i < lo.n; ++i) a.blob[i] = blob[i];
    a.sec_off = (uint32_t) down_off;
    a.X = reinterpret_cast<const half*>(H);
    a.Y = Dm;
    a.tab = htables(q);
    a.K = n_ff;
    a.n_out = n_embd;
    a.rowb = (n_ff / 64) * 18;
    a.lim = (uint32_t) (down_off + (size_t) n_embd * (size_t) a.rowb) - 4u;
    a.dbg = hdbg_flags();
    static const int wg = henv("STRATA_F16X_WG", 8);
    const int nrb = fill_hargs(a, lo, 64);
    const int ncb = n_embd / 16 / 2;
    const int wgn = ncb % wg == 0 ? wg : 1;
    launch_dn_h<T_Q2_0, 8, 2>(q, a, nrb, ncb, wgn);
    return true;
}

namespace {
// STRATA_F16X_MT (1, 2 or 8; default 0 = auto): the row tile of the held groups' FP16 products.  The bits do not depend on it (each
// output is the same DPAS chain over K); a smaller tile pads an expert of a few rows to 8 or 16 rows instead of 64, at a weight decode
// per 8 MT rows.  Auto (STRATA_F16X_SMALL > 0): the tile with the least modelled time for the group's row counts, from
// native_grouped_parity --pfbench (400 IQ2_S experts, held groups of 32, B70; us an expert of b row blocks): MT 1 3.4 + 4.3 b, MT 2
// 3.0 + 5.4 b, MT 8 12.4 b - an expert's first block reads its weights from VRAM, the others mostly from L2.  A mean-based choice lost
// on real groups, where a few experts of 20-60 rows sit among many of 1-4.  Groups averaging more than STRATA_F16X_SMALL rows keep MT 8.
int f16_mt_env() {
    static const int v = [] { const char* e = std::getenv("STRATA_F16X_MT"); const int m = e ? std::atoi(e) : 0; return (m == 1 || m == 2 || m == 8) ? m : 0; }();
    return v;
}
int f16_small_rows() {
    static const int v = [] { const char* e = std::getenv("STRATA_F16X_SMALL"); return e ? std::max(0, std::atoi(e)) : 64; }();   // default 64 (prefill.cpp pf_f16_small)
    return v;
}
int f16_mt_for(const int32_t* cnt, int n, int mt, uint64_t skip = 0) {
    if (mt == 1 || mt == 2 || mt == 8) return mt;
    if (f16_mt_env() != 0) return f16_mt_env();
    int64_t rows = 0;
    int ne = 0;
    double c1 = 0, c2 = 0, c8 = 0;
    for (int i = 0; i < n; ++i) {
        if ((skip >> i) & 1 || cnt[i] <= 0) continue;
        rows += cnt[i];
        ++ne;
        c1 += 3.4 + 4.3 * ((cnt[i] + 7) / 8);
        c2 += 3.0 + 5.4 * ((cnt[i] + 15) / 16);
        c8 += 12.4 * ((cnt[i] + 63) / 64);
    }
    const int lim = f16_small_rows();
    if (lim <= 0 || ne <= 0 || rows > (int64_t) lim * ne) return 8;
    return c1 <= c2 && c1 <= c8 ? 1 : c2 <= c8 ? 2 : 8;
}
template <int G>
void gu_h_n(int gu_type, const uint8_t* const* blob, size_t up_off, int n_ff, const int32_t* cnt, int n, int K, const uint16_t* X,
            uint16_t* H, sycl::queue& q, int mt = 8, uint64_t skip = 0) {
    HArgsT<G> a{};
    for (int i = 0; i < n; ++i) a.blob[i] = blob[i];
    a.sec_off = (uint32_t) up_off;
    a.X = reinterpret_cast<const half*>(X);
    a.Y = H;
    a.tab = htables(q);
    a.K = K;
    a.n_out = n_ff;
    a.rowb = (K / 256) * block_bytes_of(gu_type);
    a.dbg = hdbg_flags();
    static const int wg = henv("STRATA_F16X_WG", 8);
    const int nrb = fill_hargs_n(a, cnt, n, 8 * mt, skip);
    if (nrb == 0) return;
    const int ncb = n_ff / 16;
    const int wgn = ncb % wg == 0 ? wg : 1;
    if constexpr (G == kMaxGroupH) {
        if (mt == 1) {
            if (gu_type == T_IQ2S) launch_gu_h<T_IQ2S, 1, 1, G>(q, a, nrb, ncb, wgn);
            else launch_gu_h<T_IQ2XXS, 1, 1, G>(q, a, nrb, ncb, wgn);
            return;
        }
        if (mt == 2) {
            if (gu_type == T_IQ2S) launch_gu_h<T_IQ2S, 2, 1, G>(q, a, nrb, ncb, wgn);
            else launch_gu_h<T_IQ2XXS, 2, 1, G>(q, a, nrb, ncb, wgn);
            return;
        }
    }
    if (gu_type == T_IQ2S) launch_gu_h<T_IQ2S, 8, 1, G>(q, a, nrb, ncb, wgn);
    else launch_gu_h<T_IQ2XXS, 8, 1, G>(q, a, nrb, ncb, wgn);
}
template <int G>
void dn_h_n(const uint8_t* const* blob, size_t down_off, int n_embd, const int32_t* cnt, int n, int n_ff, const uint16_t* H, float* Dm,
            sycl::queue& q, int mt = 8, uint64_t skip = 0) {
    HArgsT<G> a{};
    for (int i = 0; i < n; ++i) a.blob[i] = blob[i];
    a.sec_off = (uint32_t) down_off;
    a.X = reinterpret_cast<const half*>(H);
    a.Y = Dm;
    a.tab = htables(q);
    a.K = n_ff;
    a.n_out = n_embd;
    a.rowb = (n_ff / 64) * 18;
    a.lim = (uint32_t) (down_off + (size_t) n_embd * (size_t) a.rowb) - 4u;
    a.dbg = hdbg_flags();
    static const int wg = henv("STRATA_F16X_WG", 8);
    const int nrb = fill_hargs_n(a, cnt, n, 8 * mt, skip);
    if (nrb == 0) return;
    const int ncb = n_embd / 16 / 2;
    const int wgn = ncb % wg == 0 ? wg : 1;
    if constexpr (G == kMaxGroupH) {
        if (mt == 1) { launch_dn_h<T_Q2_0, 1, 2, G>(q, a, nrb, ncb, wgn); return; }
        if (mt == 2) { launch_dn_h<T_Q2_0, 2, 2, G>(q, a, nrb, ncb, wgn); return; }
    }
    launch_dn_h<T_Q2_0, 8, 2, G>(q, a, nrb, ncb, wgn);
}
}  // namespace

bool gate_up_f16_n(int gu_type, const uint8_t* const* blob, size_t up_off, int n_ff, const int32_t* cnt, int n, int K,
                   const uint16_t* X, uint16_t* H, void* stream, int mt, uint64_t skip) {
    int64_t rows = 0;
    for (int i = 0; i < n; ++i) rows += cnt[i];
    if (n <= 0 || rows == 0) return true;
    const bool gen = gu_type != T_IQ2S && gu_type != T_IQ2XXS;   // launch_x
    if ((gen && !(x_gu(gu_type) && x_shape(gu_type, K))) || K % 256 != 0 || n_ff % 32 != 0 || (up_off & 3) != 0 || n > kMaxGroupH ||
        !al64(X) || !al64(H))
        return false;
    sycl::queue& q = *strata::q_of(stream);
    mt = f16_mt_for(cnt, n, mt, skip);
    if (gen) {
        x_n<true>(gu_type, blob, up_off, n_ff, cnt, n, K, X, H, q, mt, skip);
        return true;
    }
    if (mt != 8) gu_h_n<kMaxGroupH>(gu_type, blob, up_off, n_ff, cnt, n, K, X, H, q, mt, skip);   // (the small tiles: the 64-expert form only)
    else if (n <= 32) gu_h_n<32>(gu_type, blob, up_off, n_ff, cnt, n, K, X, H, q, 8, skip);
    else gu_h_n<kMaxGroupH>(gu_type, blob, up_off, n_ff, cnt, n, K, X, H, q, 8, skip);
    return true;
}

bool down_f16_n(int d_type, const uint8_t* const* blob, size_t down_off, int n_embd, const int32_t* cnt, int n, int n_ff,
                const uint16_t* H, float* Dm, void* stream, int mt, uint64_t skip) {
    int64_t rows = 0;
    for (int i = 0; i < n; ++i) rows += cnt[i];
    if (n <= 0 || rows == 0) return true;
    const bool gen = d_type != T_Q2_0;   // launch_x
    if ((gen && !(x_dn(d_type) && x_shape(d_type, n_ff))) || n_ff % 64 != 0 || n_embd % 64 != 0 || (down_off & 3) != 0 || n > kMaxGroupH ||
        !al64(H) || !al64(Dm))
        return false;
    sycl::queue& q = *strata::q_of(stream);
    mt = f16_mt_for(cnt, n, mt, skip);
    if (gen) {
        x_n<false>(d_type, blob, down_off, n_embd, cnt, n, n_ff, H, Dm, q, mt, skip);
        return true;
    }
    if (mt != 8) dn_h_n<kMaxGroupH>(blob, down_off, n_embd, cnt, n, n_ff, H, Dm, q, mt, skip);
    else if (n <= 32) dn_h_n<32>(blob, down_off, n_embd, cnt, n, n_ff, H, Dm, q, 8, skip);
    else dn_h_n<kMaxGroupH>(blob, down_off, n_embd, cnt, n, n_ff, H, Dm, q, 8, skip);
    return true;
}

}  // namespace strata::prefill::i8
#endif  // STRATA_I8_NO_ESIMD
