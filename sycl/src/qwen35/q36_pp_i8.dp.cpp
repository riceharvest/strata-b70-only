// q36_pp_i8.dp.cpp - Dense Qwen3.8-27B prompt-path (T > 16) quantized weight x int8 q8_1 GEMM on Xe2 XMX (ESIMD).
//
// Design:
//   - Two tiled ESIMD kernels parameterized on <Format F, int WGT_N, int WGT_M, int MT> in 256 GRFs (`grf_size<256>`):
//       Large (128x128 tile): WGT_N=8, WGT_M=2, MT=8 (16 threads/WG, 16x64 per thread = 64 GRF accs)
//       Small (64x32 tile):   WGT_N=4, WGT_M=1, MT=4 ( 4 threads/WG, 16x32 per thread = 32 GRF accs)
//   - Both maintain COLS_PER_TH = (WGT_M * MT * 8) / (WGT_N * WGT_M) = 8 so activation staging is identical.
//   - 2D workgroup swizzle (nd_range<3>(ngrp, mtiles, 4 * wgt)): pairs 4 adjacent N-tiles with M-tiles for L2 reuse
//     across weights (ffn_gate/up) and activations (ffn_down).
//   - Reads native_quantize_q8_1 layout [T][n_in/32] directly: 128 K-elements (4 sub-blocks = 144 B/col, 16-byte aligned)
//     staged per barrier into SLM in DPAS-A layout ([u][col8][64 int32]), with d8 and exact integer sum(q8) * d8
//     computed once per WG via a single `dpas(0, 0x01010101, av)` for min-bearing formats (Q4_K, Q5_K).
//   - Zero dynamic register array indexing, zero runtime integer division inside the kernel.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include "strata/sycl_queue.hpp"
#include "q36_pp_i8.hpp"

namespace strata::q36pp {
namespace {
using namespace sycl::ext::intel::esimd;
namespace es = sycl::ext::intel::esimd;
namespace esx = sycl::ext::intel::experimental::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

constexpr int T_Q8_0  = 8;
constexpr int T_Q3_K  = 11;
constexpr int T_Q4_K  = 12;
constexpr int T_Q5_K  = 13;
constexpr int T_Q6_K  = 14;
constexpr int T_IQ4NL = 20;
constexpr int T_IQ4XS = 23;
constexpr int T_W4    = 1004;

constexpr auto kAl16 = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::alignment<16>};
constexpr auto kPfL1 = sycl::ext::oneapi::experimental::properties{cache_hint_L1<cache_hint::cached>, cache_hint_L2<cache_hint::cached>};
constexpr auto kStreamW = sycl::ext::oneapi::experimental::properties{
    sycl::ext::oneapi::experimental::alignment<16>,
    es::cache_hint_L1<es::cache_hint::streaming>,
    es::cache_hint_L2<es::cache_hint::cached>};

constexpr uint32_t blk_bytes_256(int f) {
    return f == T_Q3_K  ? 110u :
           f == T_Q4_K  ? 144u :
           f == T_Q5_K  ? 176u :
           f == T_Q6_K  ? 210u :
           f == T_IQ4XS ? 136u :
           f == T_IQ4NL ? 144u :   // 8 x 18 B
           f == T_Q8_0  ? 272u :   // 8 x 34 B
           f == T_W4    ? 136u : 0u; // 2176 B per 16 rows
}
constexpr bool has_min(int f) { return f == T_Q4_K || f == T_Q5_K || f == T_W4; }
constexpr bool split_16(int f) { return f == T_Q6_K || f == T_Q3_K; }
constexpr bool uses_iq4_tab(int f) { return f == T_IQ4XS || f == T_IQ4NL; }

SYCL_ESIMD_FUNCTION inline simd<float, 16> half_of(simd<uint32_t, 16> h16) {
    simd<uint16_t, 16> lo = h16.template bit_cast_view<uint16_t>().template select<16, 2>(0);
    simd<sycl::half, 16> hh = lo.template bit_cast_view<sycl::half>();
    return simd<float, 16>(hh);
}

SYCL_ESIMD_FUNCTION inline simd<float, 8> half_of_8(simd<uint32_t, 8> h8) {
    simd<uint16_t, 8> lo = h8.template bit_cast_view<uint16_t>().template select<8, 2>(0);
    simd<sycl::half, 8> hh = lo.template bit_cast_view<sycl::half>();
    return simd<float, 8>(hh);
}

// 4 consecutive aligned dwords across 16 rows: result[16*j + lane].
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 64> ld4a(const uint8_t* base, simd<uint32_t, 16> off) {
    return gather<uint32_t, 64, 4>(reinterpret_cast<const uint32_t*>(base), off);
}

// 16-bit value at 2-byte aligned offset `off`. Requires `(off & ~3u) <= lim`.
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> ld16(const uint8_t* base, simd<uint32_t, 16> off) {
    simd<uint32_t, 16> a = off & ~3u;
    simd<uint32_t, 16> v = gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(base), a);
    return (v >> ((off & 2u) << 3)) & 0xffffu;
}

// 4 consecutive dwords at 2-byte aligned offset `off`, clamped to last dword `lim`.
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 64> ld4u(const uint8_t* base, simd<uint32_t, 16> off, uint32_t lim) {
    simd<uint32_t, 16> a = off & ~3u;
    simd<uint32_t, 64> v = gather<uint32_t, 64, 4>(reinterpret_cast<const uint32_t*>(base), a);
    simd<uint32_t, 16> e = gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(base), min(a + 16u, simd<uint32_t, 16>(lim)));
    simd<uint32_t, 16> s = (off & 2u) << 3;
    simd<uint32_t, 16> d0 = v.template select<16, 1>(0), d1 = v.template select<16, 1>(16);
    simd<uint32_t, 16> d2 = v.template select<16, 1>(32), d3 = v.template select<16, 1>(48);
    simd<uint32_t, 64> r;
    r.template select<16, 1>(0)  = (d0 >> s) | ((d1 << 16) << (16u - s));
    r.template select<16, 1>(16) = (d1 >> s) | ((d2 << 16) << (16u - s));
    r.template select<16, 1>(32) = (d2 >> s) | ((d3 << 16) << (16u - s));
    r.template select<16, 1>(48) = (d3 >> s) | ((e  << 16) << (16u - s));
    return r;
}

// u_bytes (< 128) - bias4 -> signed int8 packed into dwords.
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> sub8(simd<uint32_t, 16> u, uint32_t bias4) {
    return ((u | 0x80808080u) - bias4) ^ 0x80808080u;
}

// Vectorized 16-entry signed int8 LUT for IQ4_NL nibble lookup (N = 16 or 64).
template <int N>
SYCL_ESIMD_FUNCTION inline simd<uint32_t, N> iq4_nibble_u8(simd<uint32_t, N> nib) {
    simd<uint32_t, N> v(0x81u);  // -127
    v.merge(0x98u, nib == 1u);   // -104
    v.merge(0xADu, nib == 2u);   // -83
    v.merge(0xBFu, nib == 3u);   // -65
    v.merge(0xCFu, nib == 4u);   // -49
    v.merge(0xDDu, nib == 5u);   // -35
    v.merge(0xEAu, nib == 6u);   // -22
    v.merge(0xF6u, nib == 7u);   // -10
    v.merge(0x01u, nib == 8u);   // 1
    v.merge(0x0Du, nib == 9u);   // 13
    v.merge(0x19u, nib == 10u);  // 25
    v.merge(0x26u, nib == 11u);  // 38
    v.merge(0x35u, nib == 12u);  // 53
    v.merge(0x45u, nib == 13u);  // 69
    v.merge(0x59u, nib == 14u);  // 89
    v.merge(0x71u, nib == 15u);  // 113
    return v;
}

// Extract byte `i` (`0..11`) from three scale dwords `w0, w1, w2` (compile-time constant `i`).
template <int I>
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> scale_b12(simd<uint32_t, 16> w0, simd<uint32_t, 16> w1, simd<uint32_t, 16> w2) {
    simd<uint32_t, 16> w = (I < 4) ? w0 : (I < 8) ? w1 : w2;
    return (w >> (uint32_t)(8 * (I & 3))) & 0xffu;
}

// Compute Q3_K 6-bit signed scale (`is` in `0..15`, compile-time constant) -> `[-32..31]`.
template <int IS>
SYCL_ESIMD_FUNCTION inline simd<int, 16> q3_scale(simd<uint32_t, 16> w0, simd<uint32_t, 16> w1, simd<uint32_t, 16> w2) {
    constexpr int lo_idx = IS < 8 ? IS : IS - 8;
    simd<uint32_t, 16> wlo = lo_idx < 4 ? w0 : w1;
    simd<uint32_t, 16> blo = (wlo >> (uint32_t)(8 * (lo_idx & 3))) & 0xffu;
    simd<uint32_t, 16> lo  = (IS < 8) ? (blo & 15u) : (blo >> 4);
    simd<uint32_t, 16> bhi = (w2 >> (uint32_t)(8 * (IS & 3))) & 0xffu;
    simd<uint32_t, 16> hi  = (bhi >> (uint32_t)(2 * (IS >> 2))) & 3u;
    return simd<int, 16>(lo | (hi << 4)) - 32;
}

// Unpack one dword `qd` (16 lanes x 4 bytes) via 256-entry SLM table into:
//   lo16 = 4 low nibbles, hi16 = 4 high nibbles.
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 32> slm_iq4_dw(uint32_t slm_iq4_off, simd<uint32_t, 16> qd) {
    simd<uint32_t, 16> t0 = slm_gather<uint32_t, 16>(slm_iq4_off + ((qd & 0xffu) << 2));
    simd<uint32_t, 16> t1 = slm_gather<uint32_t, 16>(slm_iq4_off + (((qd >> 8) & 0xffu) << 2));
    simd<uint32_t, 16> t2 = slm_gather<uint32_t, 16>(slm_iq4_off + (((qd >> 16) & 0xffu) << 2));
    simd<uint32_t, 16> t3 = slm_gather<uint32_t, 16>(slm_iq4_off + ((qd >> 24) << 2));
    simd<uint32_t, 32> r;
    r.template select<16, 1>(0)  = (t0 & 0xffu) | ((t1 & 0xffu) << 8) | ((t2 & 0xffu) << 16) | (t3 << 24);
    r.template select<16, 1>(16) = ((t0 >> 8) & 0xffu) | (t1 & 0xff00u) | ((t2 & 0xff00u) << 8) | ((t3 & 0xff00u) << 16);
    return r;
}

template <int MT>
struct AccTile {
    simd<float, 128> col[MT];
};

template <int U, uint32_t WG_M, bool HASM, uint32_t SLM_D8_OFF, uint32_t SLM_DS_OFF>
SYCL_ESIMD_FUNCTION inline void pp_stage_sub(int tid, simd<uint32_t, 32> (&v0)[8], simd<uint32_t, 4> (&v1)[8]) {
    simd<uint32_t, 64> qv;
    simd<uint32_t, 8>  hv;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        if constexpr (U < 3) {
            constexpr int base = U * 9;
            hv[c] = v0[c][base];
            qv.template select<8, 1>(8 * c) = v0[c].template select<8, 1>(base + 1);
        } else {
            hv[c] = v0[c][27];
            qv.template select<4, 1>(8 * c)     = v0[c].template select<4, 1>(28);
            qv.template select<4, 1>(8 * c + 4) = v1[c].template select<4, 1>(0);
        }
    }
    simd<float, 8> d8 = half_of_8(hv);
    slm_block_store<uint32_t, 64>(
        (uint32_t)U * (WG_M * 32u) + (uint32_t)tid * 256u, qv, kAl16);
    slm_block_store<float, 8>(
        SLM_D8_OFF + ((uint32_t)U * WG_M + (uint32_t)tid * 8u) * 4u, d8, kAl16);
    if constexpr (HASM) {
        simd<int, 128> ts = xmx::dpas<8, 8, int, int, int, int,
            xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(
            simd<int, 128>(0), simd<int, 128>(0x01010101),
            simd<int, 64>(qv.template bit_cast_view<int>()));
        simd<float, 8> ds = d8 * simd<float, 8>(ts.template select<8, 16>(0));
        slm_block_store<float, 8>(
            SLM_DS_OFF + ((uint32_t)U * WG_M + (uint32_t)tid * 8u) * 4u, ds, kAl16);
    }
}

template <int F, int MT, int IB, uint32_t WG_M, uint32_t SLM_D8_OFF, uint32_t SLM_DS_OFF, uint32_t SLM_IQ4_OFF>
SYCL_ESIMD_FUNCTION inline void pp_compute_ib(
    int u, int mt_id, const uint8_t* W_tile, uint32_t lim,
    simd<uint32_t, 16> o, simd<float, 16> hd, simd<float, 16> hm,
    simd<uint32_t, 16> h0, simd<uint32_t, 16> h1, simd<uint32_t, 16> h2, simd<uint32_t, 16> h3,
    AccTile<MT>& acc) {
    constexpr bool HASM  = has_min(F);
    constexpr bool SPL16 = split_16(F);

    simd<float, MT * 8> d8_tile = slm_block_load<float, MT * 8>(
        SLM_D8_OFF + ((uint32_t)u * WG_M + (uint32_t)(mt_id * MT * 8)) * 4u, kAl16);
    simd<float, MT * 8> ds_tile;
    if constexpr (HASM) {
        ds_tile = slm_block_load<float, MT * 8>(
            SLM_DS_OFF + ((uint32_t)u * WG_M + (uint32_t)(mt_id * MT * 8)) * 4u, kAl16);
    }

    simd<uint32_t, 128> w_b;
    simd<float, 16> sc = 0.f, mn = 0.f;
    simd<int, 16> s0 = 0, s1 = 0;

    if constexpr (F == T_Q4_K || F == T_Q5_K) {
        constexpr uint32_t QS = F == T_Q4_K ? 16u : 48u;
        constexpr uint32_t nsh = 4u * (uint32_t)(IB & 1);
        simd<uint32_t, 16> qo = o + (QS + 32u * (uint32_t)(IB >> 1));
        simd<uint32_t, 64> x0 = ld4a(W_tile, qo), x1 = ld4a(W_tile, qo + 16u);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            w_b.template select<16, 1>(16 * j)       = (x0.template select<16, 1>(16 * j) >> nsh) & 0x0F0F0F0Fu;
            w_b.template select<16, 1>(16 * (4 + j)) = (x1.template select<16, 1>(16 * j) >> nsh) & 0x0F0F0F0Fu;
        }
        if constexpr (F == T_Q5_K) {
            simd<uint32_t, 64> hb0 = ld4a(W_tile, o + 16u), hb1 = ld4a(W_tile, o + 32u);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                w_b.template select<16, 1>(16 * j)       |= ((hb0.template select<16, 1>(16 * j) >> (uint32_t)IB) & 0x01010101u) << 4;
                w_b.template select<16, 1>(16 * (4 + j)) |= ((hb1.template select<16, 1>(16 * j) >> (uint32_t)IB) & 0x01010101u) << 4;
            }
        }
        simd<uint32_t, 16> ksc, kmn;
        if constexpr (IB < 4) {
            ksc = scale_b12<IB>(h0, h1, h2) & 63u;
            kmn = scale_b12<IB + 4>(h0, h1, h2) & 63u;
        } else {
            ksc = (scale_b12<IB + 4>(h0, h1, h2) & 0xFu) | ((scale_b12<IB - 4>(h0, h1, h2) >> 6) << 4);
            kmn = (scale_b12<IB + 4>(h0, h1, h2) >> 4)  | ((scale_b12<IB>(h0, h1, h2)     >> 6) << 4);
        }
        sc = hd * convert<float>(simd<int, 16>(ksc));
        mn = hm * convert<float>(simd<int, 16>(kmn));
    } else if constexpr (F == T_Q6_K) {
        constexpr int ip = IB >> 2, c4 = IB & 3;
        constexpr uint32_t nsh = 4u * (uint32_t)(c4 >> 1);
        constexpr uint32_t qsh = 2u * (uint32_t)c4;
        simd<uint32_t, 16> lo_ = o + (uint32_t)(64 * ip + 32 * (c4 & 1));
        simd<uint32_t, 16> ho_ = o + (uint32_t)(128 + 32 * ip);
        simd<uint32_t, 64> x0 = ld4u(W_tile, lo_, lim), x1 = ld4u(W_tile, lo_ + 16u, lim);
        simd<uint32_t, 64> hb0 = ld4u(W_tile, ho_, lim), hb1 = ld4u(W_tile, ho_ + 16u, lim);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            w_b.template select<16, 1>(16 * j) =
                sub8(((x0.template select<16, 1>(16 * j) >> nsh) & 0x0F0F0F0Fu) |
                     (((hb0.template select<16, 1>(16 * j) >> qsh) & 0x03030303u) << 4), 0x20202020u);
            w_b.template select<16, 1>(16 * (4 + j)) =
                sub8(((x1.template select<16, 1>(16 * j) >> nsh) & 0x0F0F0F0Fu) |
                     (((hb1.template select<16, 1>(16 * j) >> qsh) & 0x03030303u) << 4), 0x20202020u);
        }
        simd<uint32_t, 16> scw = (IB < 2) ? h0 : (IB < 4) ? h1 : (IB < 6) ? h2 : h3;
        simd<uint32_t, 16> sb0 = (scw >> (uint32_t)(8 * (2 * (IB & 1)))) & 0xffu;
        simd<uint32_t, 16> sb1 = (scw >> (uint32_t)(8 * (2 * (IB & 1) + 1))) & 0xffu;
        s0 = simd<int, 16>(sb0 ^ 0x80u) - 128;
        s1 = simd<int, 16>(sb1 ^ 0x80u) - 128;
    } else if constexpr (F == T_Q3_K) {
        constexpr int n = IB >> 2, j = IB & 3;
        constexpr uint32_t qsh = 2u * (uint32_t)j;
        constexpr uint32_t hsh = (uint32_t)(4 * n + j);
        simd<uint32_t, 16> qo = o + (uint32_t)(32 + 32 * n);
        simd<uint32_t, 64> x0 = ld4u(W_tile, qo, lim), x1 = ld4u(W_tile, qo + 16u, lim);
        simd<uint32_t, 64> hm0 = ld4u(W_tile, o, lim), hm1 = ld4u(W_tile, o + 16u, lim);
#pragma unroll
        for (int qj = 0; qj < 4; ++qj) {
            simd<uint32_t, 16> qla = (x0.template select<16, 1>(16 * qj) >> qsh) & 0x03030303u;
            simd<uint32_t, 16> qlb = (x1.template select<16, 1>(16 * qj) >> qsh) & 0x03030303u;
            simd<uint32_t, 16> nga = ((~hm0.template select<16, 1>(16 * qj)) >> hsh) & 0x01010101u;
            simd<uint32_t, 16> ngb = ((~hm1.template select<16, 1>(16 * qj)) >> hsh) & 0x01010101u;
            w_b.template select<16, 1>(16 * qj)      = qla | (nga * 0xFCu);
            w_b.template select<16, 1>(16 * (4 + qj)) = qlb | (ngb * 0xFCu);
        }
        s0 = q3_scale<2 * IB>(h0, h1, h2);
        s1 = q3_scale<2 * IB + 1>(h0, h1, h2);
    } else if constexpr (F == T_IQ4XS) {
        simd<uint32_t, 16> ls = ((h1 >> (uint32_t)(4 * IB)) & 0xfu) | (((h0 >> (uint32_t)(2 * IB)) & 3u) << 4);
        sc = hd * convert<float>(simd<int, 16>(ls) - 32);
        simd<uint32_t, 64> x0 = ld4a(W_tile, o + (8u + 16u * (uint32_t)IB));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            simd<uint32_t, 32> kv = slm_iq4_dw(SLM_IQ4_OFF, x0.template select<16, 1>(16 * j));
            w_b.template select<16, 1>(16 * j)       = kv.template select<16, 1>(0);
            w_b.template select<16, 1>(16 * (4 + j)) = kv.template select<16, 1>(16);
        }
    } else if constexpr (F == T_IQ4NL) {
        simd<uint32_t, 16> bo = o + (uint32_t)(18 * IB);
        sc = half_of(ld16(W_tile, bo));
        simd<uint32_t, 64> x0 = ld4u(W_tile, bo + 2u, lim);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            simd<uint32_t, 32> kv = slm_iq4_dw(SLM_IQ4_OFF, x0.template select<16, 1>(16 * j));
            w_b.template select<16, 1>(16 * j)       = kv.template select<16, 1>(0);
            w_b.template select<16, 1>(16 * (4 + j)) = kv.template select<16, 1>(16);
        }
    } else if constexpr (F == T_W4) {
        constexpr int P = IB >> 1;
        const uint32_t* rp = reinterpret_cast<const uint32_t*>(W_tile) + 128 * P;
        simd<uint32_t, 64> x0 = block_load<uint32_t, 64>(rp, kAl16);
        simd<uint32_t, 64> x1 = block_load<uint32_t, 64>(rp + 64, kAl16);
        if constexpr ((IB & 1) == 0) {
            w_b.template select<64, 1>(0)  = x0 & 0x0F0F0F0Fu;
            w_b.template select<64, 1>(64) = x1 & 0x0F0F0F0Fu;
        } else {
            w_b.template select<64, 1>(0)  = (x0 >> 4) & 0x0F0F0F0Fu;
            w_b.template select<64, 1>(64) = (x1 >> 4) & 0x0F0F0F0Fu;
        }
        if constexpr (IB < 4) {
            sc = hd;
            mn = hm;
        } else {
            sc = h0.template bit_cast_view<float>();
            mn = h1.template bit_cast_view<float>();
        }
    } else {  // T_Q8_0
        simd<uint32_t, 16> bo = o + (uint32_t)(34 * IB);
        sc = half_of(ld16(W_tile, bo));
        simd<uint32_t, 64> x0 = ld4u(W_tile, bo + 2u, lim), x1 = ld4u(W_tile, bo + 18u, lim);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            w_b.template select<16, 1>(16 * j)       = x0.template select<16, 1>(16 * j);
            w_b.template select<16, 1>(16 * (4 + j)) = x1.template select<16, 1>(16 * j);
        }
    }

    simd<int, 128> wb_i = w_b.template bit_cast_view<int>();
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int col_grp = mt_id * MT + m;
        simd<int, 64> av = slm_block_load<int, 64>(
            (uint32_t)u * (WG_M * 32u) + (uint32_t)col_grp * 256u, kAl16);
        if constexpr (SPL16) {
            simd<int, 64> av0 = av, av1 = av;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                av0.template select<4, 1>(8 * r + 4) = 0;
                av1.template select<4, 1>(8 * r)     = 0;
            }
            simd<int, 128> d0 = xmx::dpas<8, 8, int, int, int, int,
                xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(
                simd<int, 128>(0), wb_i, av0);
            simd<int, 128> d1 = xmx::dpas<8, 8, int, int, int, int,
                xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(
                simd<int, 128>(0), wb_i, av1);
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const float d8 = d8_tile[8 * m + r];
                simd<int, 16> c = s0 * d0.template select<16, 1>(16 * r)
                               + s1 * d1.template select<16, 1>(16 * r);
                simd<float, 16> a = acc.col[m].template select<16, 1>(16 * r);
                a += convert<float>(c) * (hd * d8);
                acc.col[m].template select<16, 1>(16 * r) = a;
            }
        } else {
            simd<int, 128> dot = xmx::dpas<8, 8, int, int, int, int,
                xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(
                simd<int, 128>(0), wb_i, av);
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int c_off = 8 * m + r;
                const float d8  = d8_tile[c_off];
                simd<float, 16> tf(dot.template select<16, 1>(16 * r));
                simd<float, 16> a  = acc.col[m].template select<16, 1>(16 * r);
                a += tf * (sc * d8);
                if constexpr (HASM) {
                    const float dsm = ds_tile[c_off];
                    a -= mn * dsm;
                }
                acc.col[m].template select<16, 1>(16 * r) = a;
            }
        }
    }
}

template <int F, int MT, int H2, uint32_t WG_M, uint32_t SLM_D8_OFF, uint32_t SLM_DS_OFF, uint32_t SLM_IQ4_OFF>
SYCL_ESIMD_FUNCTION inline void pp_run_half(
    int sb, int tid, int mt_id, int col0, int st_c0, int T, size_t row_q_dw,
    const uint32_t* X_base, const uint8_t* W_tile, uint32_t lim,
    simd<uint32_t, 16> o, simd<float, 16> hd, simd<float, 16> hm,
    simd<uint32_t, 16> h0, simd<uint32_t, 16> h1, simd<uint32_t, 16> h2, simd<uint32_t, 16> h3,
    AccTile<MT>& acc) {
    constexpr bool HASM = has_min(F);
    if constexpr (H2 == 1) barrier();

    {
        simd<uint32_t, 32> v0[8];
        simd<uint32_t, 4>  v1[8];
        const uint32_t k_dw_off = (uint32_t)sb * 72u + (uint32_t)H2 * 36u;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            if (col0 + st_c0 + c < T) {
                const uint32_t* xp = X_base + (size_t)c * row_q_dw + k_dw_off;
                v0[c] = block_load<uint32_t, 32>(xp);
                v1[c] = block_load<uint32_t, 4>(xp + 32);
            } else {
                v0[c] = 0u;
                v1[c] = 0u;
            }
        }
        pp_stage_sub<0, WG_M, HASM, SLM_D8_OFF, SLM_DS_OFF>(tid, v0, v1);
        pp_stage_sub<1, WG_M, HASM, SLM_D8_OFF, SLM_DS_OFF>(tid, v0, v1);
        pp_stage_sub<2, WG_M, HASM, SLM_D8_OFF, SLM_DS_OFF>(tid, v0, v1);
        pp_stage_sub<3, WG_M, HASM, SLM_D8_OFF, SLM_DS_OFF>(tid, v0, v1);
    }
    barrier();

    pp_compute_ib<F, MT, 4 * H2 + 0, WG_M, SLM_D8_OFF, SLM_DS_OFF, SLM_IQ4_OFF>(0, mt_id, W_tile, lim, o, hd, hm, h0, h1, h2, h3, acc);
    pp_compute_ib<F, MT, 4 * H2 + 1, WG_M, SLM_D8_OFF, SLM_DS_OFF, SLM_IQ4_OFF>(1, mt_id, W_tile, lim, o, hd, hm, h0, h1, h2, h3, acc);
    pp_compute_ib<F, MT, 4 * H2 + 2, WG_M, SLM_D8_OFF, SLM_DS_OFF, SLM_IQ4_OFF>(2, mt_id, W_tile, lim, o, hd, hm, h0, h1, h2, h3, acc);
    pp_compute_ib<F, MT, 4 * H2 + 3, WG_M, SLM_D8_OFF, SLM_DS_OFF, SLM_IQ4_OFF>(3, mt_id, W_tile, lim, o, hd, hm, h0, h1, h2, h3, acc);
}

template <int F, int WGT_N, int WGT_M, int MT> struct TagPp;

template <int F, int WGT_N, int WGT_M, int MT>
void launch_pp(sycl::queue& q, const void* w_ptr, const void* x_ptr, float* y_ptr, int n_in, int n_out, int T) {
    constexpr uint32_t B256  = blk_bytes_256(F);
    constexpr bool HASM      = has_min(F);
    constexpr bool USE_IQ4   = uses_iq4_tab(F);
    constexpr int WGT        = WGT_N * WGT_M;
    constexpr int WG_N       = WGT_N * 16;
    constexpr int WG_M       = WGT_M * MT * 8;
    constexpr int COLS_TH    = WG_M / WGT;
    static_assert(COLS_TH == 8, "SLM producer expects 8 columns per thread (one DPAS A-tile)");

    // SLM layout (bytes):
    //   [0 .. SLM_QS_BYTES):                   4 sub-blocks x WG_M cols x 32 B = 128 * WG_M B
    //   [SLM_D8_OFF .. +4 * WG_M * 4):         4 sub-blocks x WG_M floats     =  16 * WG_M B
    //   [SLM_DS_OFF .. +HASM * 4 * WG_M * 4):  4 sub-blocks x WG_M (d8 * sum) =  16 * WG_M B (when HASM)
    //   [SLM_IQ4_OFF .. +USE_IQ4 * 1024):      256 x uint32_t IQ4 pair LUT    =  1024 B     (when USE_IQ4)
    constexpr uint32_t SLM_QS_BYTES = 4u * (uint32_t)WG_M * 32u;
    constexpr uint32_t SLM_D8_OFF   = SLM_QS_BYTES;
    constexpr uint32_t SLM_D8_BYTES = 4u * (uint32_t)WG_M * 4u;
    constexpr uint32_t SLM_DS_OFF   = SLM_D8_OFF + SLM_D8_BYTES;
    constexpr uint32_t SLM_DS_BYTES = HASM ? (4u * (uint32_t)WG_M * 4u) : 0u;
    constexpr uint32_t SLM_IQ4_OFF  = SLM_DS_OFF + SLM_DS_BYTES;
    constexpr uint32_t SLM_TOTAL    = SLM_IQ4_OFF + (USE_IQ4 ? 1024u : 0u);

    const int bpr      = n_in / 256;
    const size_t rowb  = (size_t)bpr * B256;
    const size_t row_q_dw = (size_t)(n_in / 32) * 9u;
    const int n_tiles  = (n_out + WG_N - 1) / WG_N;
    const int m_tiles  = (T + WG_M - 1) / WG_M;
    constexpr int SWZ  = 4;
    const int n_groups = (n_tiles + SWZ - 1) / SWZ;

    const uint8_t* W   = reinterpret_cast<const uint8_t*>(w_ptr);
    const uint32_t* X  = reinterpret_cast<const uint32_t*>(x_ptr);

    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& h) {
        sycl::nd_range<3> nd(sycl::range<3>((size_t)n_groups, (size_t)m_tiles, (size_t)(SWZ * WGT)),
                             sycl::range<3>(1, 1, (size_t)WGT));
        h.parallel_for<TagPp<F, WGT_N, WGT_M, MT>>(nd, props, [=](sycl::nd_item<3> it) SYCL_ESIMD_KERNEL {
            slm_init<SLM_TOTAL>();
            const int nt  = (int)it.get_group(0) * SWZ + (int)it.get_group(2);
            if (nt >= n_tiles) return;
            const int mt  = (int)it.get_group(1);
            const int tid = (int)it.get_local_id(2);
            const int nt_id = tid / WGT_M;
            const int mt_id = tid - nt_id * WGT_M;

            // Populate SLM IQ4 lookup table once per WG (before the first barrier, zero extra barriers).
            if constexpr (USE_IQ4) {
                constexpr int EPT = 256 / WGT;
                simd<uint32_t, EPT> idx((uint32_t)(tid * EPT), 1u);
                simd<uint32_t, EPT> lo = iq4_nibble_u8<EPT>(idx & 0x0Fu);
                simd<uint32_t, EPT> hi = iq4_nibble_u8<EPT>(idx >> 4);
                slm_block_store<uint32_t, EPT>(SLM_IQ4_OFF + (uint32_t)(tid * EPT) * 4u, lo | (hi << 8), kAl16);
            }

            const int r_base = nt * WG_N + nt_id * 16;
            const int col0   = mt * WG_M;
            const bool row_any     = (r_base < n_out);
            const bool row_full    = (r_base + 16 <= n_out);
            const int r_clamp_base = row_any ? r_base : 0;
            const int valid_rem    = row_any ? (n_out - r_base) : n_out;
            const uint8_t* W_tile  = (F == T_W4)
                ? (W + (size_t)(r_clamp_base / 16) * (size_t)bpr * 2176u)
                : (W + (size_t)r_clamp_base * rowb);
            const uint32_t lim     = (F == T_W4) ? 0u : (uint32_t)((size_t)valid_rem * rowb - 4u);

            simd<uint32_t, 16> lane(0u, 1u);
            simd<uint32_t, 16> valid_lanes = min(lane, simd<uint32_t, 16>((uint32_t)(valid_rem - 1)));
            simd<uint32_t, 16> ro = valid_lanes * (uint32_t)rowb;

            // Accumulators: MT * 8 columns x 16 rows
            AccTile<MT> acc;
#pragma unroll
            for (int m = 0; m < MT; ++m) acc.col[m] = 0.f;

            // Thread's 8 columns to stage into SLM per 128-K chunk
            const int st_c0 = tid * 8;
            const uint32_t* X_base = X + (size_t)(col0 + st_c0) * row_q_dw;

            for (int sb = 0; sb < bpr; ++sb) {
                const uint32_t P = (uint32_t)sb * B256;
                if (sb + 1 < bpr) {
                    if constexpr (F == T_W4) {
                        const uint32_t* next_rec = reinterpret_cast<const uint32_t*>(W_tile) + (size_t)(sb + 1) * 544u;
                        prefetch<uint32_t, 16>(next_rec, simd<uint32_t, 16>(0u, 64u), kPfL1);
                        prefetch<uint32_t, 16>(next_rec + 256, simd<uint32_t, 16>(0u, 64u), kPfL1);
                        prefetch<uint32_t, 2>(next_rec + 512, simd<uint32_t, 2>(0u, 64u), kPfL1);
                    } else {
                        simd<uint32_t, 16> np = (ro + (P + B256)) & ~63u;
#pragma unroll
                        for (uint32_t c = 0; c < (B256 + 63u) / 64u + 1u; ++c)
                            prefetch<uint32_t, 16>(reinterpret_cast<const uint32_t*>(W_tile), min(np + 64u * c, simd<uint32_t, 16>(lim & ~63u)), kPfL1);
                    }
                }

                // 256-K superblock header (loaded once for both 128-K halves)
                simd<float, 16> hd = 0.f, hm = 0.f;
                simd<uint32_t, 16> h0 = 0u, h1 = 0u, h2 = 0u, h3 = 0u;
                simd<uint32_t, 16> o = ro + P;
                if constexpr (F == T_Q4_K || F == T_Q5_K) {
                    simd<uint32_t, 64> v = ld4a(W_tile, o);
                    simd<uint32_t, 16> dm = v.template select<16, 1>(0);
                    hd = half_of(dm);
                    hm = half_of(dm >> 16);
                    h0 = v.template select<16, 1>(16);
                    h1 = v.template select<16, 1>(32);
                    h2 = v.template select<16, 1>(48);
                } else if constexpr (F == T_Q6_K) {
                    hd = half_of(ld16(W_tile, o + 208u));
                    simd<uint32_t, 64> v = ld4u(W_tile, o + 192u, lim);
                    h0 = v.template select<16, 1>(0);
                    h1 = v.template select<16, 1>(16);
                    h2 = v.template select<16, 1>(32);
                    h3 = v.template select<16, 1>(48);
                } else if constexpr (F == T_Q3_K) {
                    hd = half_of(ld16(W_tile, o + 108u));
                    simd<uint32_t, 64> sv = ld4u(W_tile, o + 96u, lim);
                    h0 = sv.template select<16, 1>(0);
                    h1 = sv.template select<16, 1>(16);
                    h2 = sv.template select<16, 1>(32);
                } else if constexpr (F == T_IQ4XS) {
                    simd<uint32_t, 16> dw0 = gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(W_tile), o);
                    hd = half_of(dw0);
                    h0 = dw0 >> 16;  // scales_h
                    h1 = gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(W_tile), o + 4u);  // scales_l
                } else if constexpr (F == T_W4) {
                    const uint32_t* w4_rec = reinterpret_cast<const uint32_t*>(W_tile) + (size_t)sb * 544u;
                    simd<uint32_t, 32> scz = block_load<uint32_t, 32>(w4_rec + 512, kAl16);
                    simd<uint32_t, 16> m0 = scz.template select<16, 1>(0);
                    simd<uint32_t, 16> m1 = scz.template select<16, 1>(16);
                    hd = half_of(m0);
                    hm = hd * convert<float>(simd<int, 16>(m0 >> 16));
                    simd<float, 16> sc1 = half_of(m1);
                    simd<float, 16> mn1 = sc1 * convert<float>(simd<int, 16>(m1 >> 16));
                    h0 = sc1.template bit_cast_view<uint32_t>();
                    h1 = mn1.template bit_cast_view<uint32_t>();
                }

                // Wait for previous half's SLM reads to complete before overwriting SLM.
                if (sb > 0) barrier();

                const uint8_t* W_arg = (F == T_W4) ? (W_tile + (size_t)sb * 2176u) : W_tile;
                pp_run_half<F, MT, 0, (uint32_t)WG_M, SLM_D8_OFF, SLM_DS_OFF, SLM_IQ4_OFF>(
                    sb, tid, mt_id, col0, st_c0, T, row_q_dw, X_base, W_arg, lim, o, hd, hm, h0, h1, h2, h3, acc);
                pp_run_half<F, MT, 1, (uint32_t)WG_M, SLM_D8_OFF, SLM_DS_OFF, SLM_IQ4_OFF>(
                    sb, tid, mt_id, col0, st_c0, T, row_q_dw, X_base, W_arg, lim, o, hd, hm, h0, h1, h2, h3, acc);
            }

            // Epilogue: store valid (col < T, row < n_out) output floats
            const int th_col0 = col0 + mt_id * MT * 8;
            if (r_base < n_out) {
                if (row_full) {
#pragma unroll
                    for (int m = 0; m < MT; ++m)
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const int c = th_col0 + 8 * m + r;
                            if (c < T) block_store<float, 16>(y_ptr + (size_t)c * n_out + r_base, acc.col[m].template select<16, 1>(16 * r));
                        }
                } else {
                    const int left = n_out - r_base;
                    simd_mask<16> rmask = lane < (uint32_t)left;
                    simd<uint32_t, 16> boff = lane * 4u;
#pragma unroll
                    for (int m = 0; m < MT; ++m)
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const int c = th_col0 + 8 * m + r;
                            if (c < T) {
                                scatter<float, 16>(y_ptr + (size_t)c * n_out + r_base, boff,
                                                   acc.col[m].template select<16, 1>(16 * r), rmask);
                            }
                        }
                }
            }
        });
    });
}

template <int F>
void dispatch_tiled(sycl::queue& q, const void* W, const void* X, float* Y, int n_in, int n_out, int T) {
    // Large (128x128 WG, 16 threads): saturates 32+ Xe2 cores when >= 48 WGs exist, cutting weight traffic 2x.
    // Small (64x32 WG, 4 threads): high parallelism for small N (1024) x small T (17..256).
    const int n128 = (n_out + 127) / 128;
    const int m128 = (T + 127) / 128;
    if ((int64_t)n128 * m128 >= 48) {
        launch_pp<F, 8, 2, 8>(q, W, X, Y, n_in, n_out, T);
    } else {
        launch_pp<F, 4, 1, 4>(q, W, X, Y, n_in, n_out, T);
    }
}

}  // namespace

bool handles(int ggml_type, int n_in, int n_out, int T) noexcept {
    if (n_in <= 0 || n_out <= 0 || T <= 0) return false;
    if ((n_in % 256) != 0) return false;
    if (ggml_type == T_W4 && (n_out % 16) != 0) return false;
    const uint32_t b256 = blk_bytes_256(ggml_type);
    if (b256 == 0u) return false;
    // Q6_K (210 B) and Q3_K (110 B) are 2-byte aligned per 256-block; ensure total matrix bytes % 4 == 0 for safe ld4u/ld16.
    if (((size_t)n_out * (size_t)(n_in / 256) * b256) % 4u != 0u) return false;
    return true;
}

bool gemm(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int T, void* stream) {
    if (!handles(ggml_type, n_in, n_out, T)) return false;
    sycl::queue& q = *strata::q_of(stream);
    switch (ggml_type) {
        case T_Q4_K:  dispatch_tiled<T_Q4_K>(q, weights, x_q8_1, y, n_in, n_out, T);  return true;
        case T_Q5_K:  dispatch_tiled<T_Q5_K>(q, weights, x_q8_1, y, n_in, n_out, T);  return true;
        case T_IQ4XS: dispatch_tiled<T_IQ4XS>(q, weights, x_q8_1, y, n_in, n_out, T); return true;
        case T_Q6_K:  dispatch_tiled<T_Q6_K>(q, weights, x_q8_1, y, n_in, n_out, T);  return true;
        case T_Q3_K:  dispatch_tiled<T_Q3_K>(q, weights, x_q8_1, y, n_in, n_out, T);  return true;
        case T_IQ4NL: dispatch_tiled<T_IQ4NL>(q, weights, x_q8_1, y, n_in, n_out, T); return true;
        case T_Q8_0:  dispatch_tiled<T_Q8_0>(q, weights, x_q8_1, y, n_in, n_out, T);  return true;
        case T_W4:    dispatch_tiled<T_W4>(q, weights, x_q8_1, y, n_in, n_out, T);    return true;
        default: return false;
    }
}

namespace w4f16 {
using namespace sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

constexpr auto kAl16 = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::alignment<16>};
constexpr auto kPf   = sycl::ext::oneapi::experimental::properties{cache_hint_L1<cache_hint::cached>, cache_hint_L2<cache_hint::cached>};

SYCL_ESIMD_FUNCTION inline simd<float, 16> half_of(simd<uint32_t, 16> h16) {
    simd<uint16_t, 16> lo = h16.template bit_cast_view<uint16_t>().template select<16, 2>(0);
    simd<sycl::half, 16> hh = lo.template bit_cast_view<sycl::half>();
    return simd<float, 16>(hh);
}

template <int C_TH>
SYCL_ESIMD_FUNCTION inline void load_x64(const uint32_t* X16, int n_in, int T, int col_wg, int tid, int s64,
                                         simd<uint32_t, 32>* xr) {
    const size_t k_dw = (size_t)s64 * 32u;
    const size_t stride_dw = (size_t)(n_in >> 1);
#pragma unroll
    for (int ci = 0; ci < C_TH; ++ci) {
        const int c = col_wg + tid * C_TH + ci;
        if (c < T) {
            xr[ci] = block_load<uint32_t, 32>(X16 + (size_t)c * stride_dw + k_dw);
        } else {
            xr[ci] = 0u;
        }
    }
}

template <int C_TH, int N_M8>
SYCL_ESIMD_FUNCTION inline void store_x64(uint32_t slm_off, int tid, simd<uint32_t, 32>* xr) {
    if constexpr (C_TH == 16) {
#pragma unroll
        for (int b = 0; b < 2; ++b) {
#pragma unroll
            for (int k16 = 0; k16 < 4; ++k16) {
                simd<uint32_t, 64> blk;
#pragma unroll
                for (int ci = 0; ci < 8; ++ci) {
                    blk.template select<8, 1>(8 * ci) = xr[8 * b + ci].template select<8, 1>(8 * k16);
                }
                slm_block_store<uint32_t, 64>(slm_off + (uint32_t)(k16 * N_M8 + (2 * tid + b)) * 256u, blk, kAl16);
            }
        }
    } else if constexpr (C_TH == 8) {
#pragma unroll
        for (int k16 = 0; k16 < 4; ++k16) {
            simd<uint32_t, 64> blk;
#pragma unroll
            for (int ci = 0; ci < 8; ++ci) {
                blk.template select<8, 1>(8 * ci) = xr[ci].template select<8, 1>(8 * k16);
            }
            slm_block_store<uint32_t, 64>(slm_off + (uint32_t)(k16 * N_M8 + tid) * 256u, blk, kAl16);
        }
    } else {
        static_assert(C_TH == 4, "unsupported C_TH");
        const uint32_t m8 = (uint32_t)(tid >> 1);
        const uint32_t sub_off = (uint32_t)(tid & 1) * 128u;
#pragma unroll
        for (int k16 = 0; k16 < 4; ++k16) {
            simd<uint32_t, 32> blk;
#pragma unroll
            for (int ci = 0; ci < 4; ++ci) {
                blk.template select<8, 1>(8 * ci) = xr[ci].template select<8, 1>(8 * k16);
            }
            slm_block_store<uint32_t, 32>(slm_off + (uint32_t)(k16 * N_M8 + m8) * 256u + sub_off, blk, kAl16);
        }
    }
}

template <bool SCALE_BW, int HH, int DD>
SYCL_ESIMD_FUNCTION inline void unpack_bw_dd(simd<uint32_t, 128>& v,
                                             simd<sycl::half, 32>& hz32,
                                             simd<sycl::half, 32>& sc32,
                                             simd<sycl::half, 256>& bw_out) {
    simd<uint32_t, 16> qD = v.template select<16, 1>(16 * (4 * HH + (DD >> 1)));
    simd<uint16_t, 16> sel = qD.template bit_cast_view<uint16_t>().template select<16, 2>(DD & 1);
    simd<uint8_t, 32> b8 = sel.template bit_cast_view<uint8_t>();
    simd<uint16_t, 32> vi = b8;
    vi |= uint16_t(0x6400u);
    simd<sycl::half, 32> diff = vi.template bit_cast_view<sycl::half>() - hz32;
    if constexpr (SCALE_BW) {
        bw_out.template select<32, 1>(32 * DD) = diff * sc32;
    } else {
        bw_out.template select<32, 1>(32 * DD) = diff;
    }
}

template <bool SCALE_BW, int HH>
SYCL_ESIMD_FUNCTION inline void unpack_bw_all(simd<uint32_t, 128>& v,
                                              simd<sycl::half, 32>& hz32,
                                              simd<sycl::half, 32>& sc32,
                                              simd<sycl::half, 256>& bw_out) {
    unpack_bw_dd<SCALE_BW, HH, 0>(v, hz32, sc32, bw_out);
    unpack_bw_dd<SCALE_BW, HH, 1>(v, hz32, sc32, bw_out);
    unpack_bw_dd<SCALE_BW, HH, 2>(v, hz32, sc32, bw_out);
    unpack_bw_dd<SCALE_BW, HH, 3>(v, hz32, sc32, bw_out);
    unpack_bw_dd<SCALE_BW, HH, 4>(v, hz32, sc32, bw_out);
    unpack_bw_dd<SCALE_BW, HH, 5>(v, hz32, sc32, bw_out);
    unpack_bw_dd<SCALE_BW, HH, 6>(v, hz32, sc32, bw_out);
    unpack_bw_dd<SCALE_BW, HH, 7>(v, hz32, sc32, bw_out);
}

template <bool SCALE_BW, int MT, int K16, int M>
SYCL_ESIMD_FUNCTION inline void unpack_bw_for_m(simd<uint32_t, 128>& v,
                                                simd<sycl::half, 32>& hz32,
                                                simd<sycl::half, 32>& sc32,
                                                simd<sycl::half, 256>& bw_nxt) {
    if constexpr (K16 < 3 && (MT == 4 || MT == 8 || MT == 16)) {
        constexpr int NXT_HH = (K16 + 1) & 1;
        if constexpr (MT == 16) {
            if constexpr ((M & 1) == 0) {
                unpack_bw_dd<SCALE_BW, NXT_HH, (M >> 1)>(v, hz32, sc32, bw_nxt);
            }
        } else if constexpr (MT == 8) {
            unpack_bw_dd<SCALE_BW, NXT_HH, M>(v, hz32, sc32, bw_nxt);
        } else {
            unpack_bw_dd<SCALE_BW, NXT_HH, 2 * M>(v, hz32, sc32, bw_nxt);
            unpack_bw_dd<SCALE_BW, NXT_HH, 2 * M + 1>(v, hz32, sc32, bw_nxt);
        }
    }
}

template <bool SCALE_BW, int N_M8, int MT, int K16, int... Ms>
SYCL_ESIMD_FUNCTION inline void run_k16(
    uint32_t cur_slm, int wm_m0, simd<uint32_t, 128>& v,
    simd<sycl::half, 32>& hz32, simd<sycl::half, 32>& sc32,
    simd<sycl::half, 256>& bw_cur, simd<sycl::half, 256>& bw_nxt,
    simd<float, 128>* y_acc, simd<float, 128>* acc,
    std::integer_sequence<int, Ms...>) {
    const uint32_t slm_k = cur_slm + (uint32_t)(K16 * N_M8) * 256u;
    simd<uint32_t, 64> xa0 = slm_block_load<uint32_t, 64>(slm_k + (uint32_t)wm_m0 * 256u, kAl16);
    simd<uint32_t, 64> xa1 = 0u;
    auto step = [&](auto mc) {
        constexpr int M = decltype(mc)::value;
        if constexpr (M + 1 < MT) {
            if constexpr ((M & 1) == 0) {
                xa1 = slm_block_load<uint32_t, 64>(slm_k + (uint32_t)(wm_m0 + M + 1) * 256u, kAl16);
            } else {
                xa0 = slm_block_load<uint32_t, 64>(slm_k + (uint32_t)(wm_m0 + M + 1) * 256u, kAl16);
            }
        }
        simd<sycl::half, 128> ah = (((M & 1) == 0) ? xa0 : xa1).template bit_cast_view<sycl::half>();
        if constexpr (SCALE_BW) {
            y_acc[M] = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(y_acc[M], bw_cur, ah);
        } else {
            acc[M]   = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(acc[M], bw_cur, ah);
        }
        unpack_bw_for_m<SCALE_BW, MT, K16, M>(v, hz32, sc32, bw_nxt);
    };
    (step(std::integral_constant<int, Ms>{}), ...);
}

template <int WGT_N, int WGT_M, int MT, bool SCALE_BW>
struct TagW4F16;

template <int WGT_N, int WGT_M, int MT, bool SCALE_BW>
void launch_w4f16(sycl::queue& q, const uint32_t* W, const uint32_t* X16, float* Y, int n_in, int n_out, int T) {
    constexpr int NTHREADS = WGT_N * WGT_M;
    constexpr int WG_N     = WGT_N * 16;
    constexpr int WG_M     = WGT_M * MT * 8;
    constexpr int N_M8     = WGT_M * MT;
    constexpr int C_TH     = WG_M / NTHREADS;
    constexpr uint32_t SLM_STEP_BYTES = (uint32_t)(4 * N_M8) * 256u;

    const int n_tiles  = n_out / 16;
    const int n_wg_n   = (n_out + WG_N - 1) / WG_N;
    const int n_wg_m   = (T + WG_M - 1) / WG_M;
    const int total_wg = n_wg_n * n_wg_m;
    const int bpr      = n_in / 256;
    const int n_steps  = bpr * 4;

    q.submit([&](sycl::handler& h) {
        auto run_kernel = [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            slm_init<2u * SLM_STEP_BYTES>();
            const int wg_id = (int)it.get_group(0);
            const int tid   = (int)it.get_local_id(0);

            constexpr int GM = 8;
            const int num_in_group = GM * n_wg_n;
            const int grp_id  = wg_id / num_in_group;
            const int first_m = grp_id * GM;
            const int grp_m   = (n_wg_m - first_m) < GM ? (n_wg_m - first_m) : GM;
            const int in_grp  = wg_id % num_in_group;
            const int wg_m    = first_m + (in_grp % grp_m);
            const int wg_n    = in_grp / grp_m;

            const int col_wg = wg_m * WG_M;
            const int wn = tid / WGT_M;
            const int wm = tid % WGT_M;
            const int tile_idx = wg_n * WGT_N + wn;
            const bool row_ok  = (tile_idx < n_tiles);
            const uint32_t* w_tile = row_ok ? (W + (size_t)tile_idx * (size_t)bpr * 544u) : nullptr;

            {
                simd<uint32_t, 32> xr[C_TH];
                load_x64<C_TH>(X16, n_in, T, col_wg, tid, 0, xr);
                store_x64<C_TH, N_M8>(0u, tid, xr);
                barrier();
            }

            simd<float, 128> y_acc[MT];
            simd<float, 128> acc[SCALE_BW ? 1 : MT];
#pragma unroll
            for (int m = 0; m < MT; ++m) y_acc[m] = 0.f;
            if constexpr (!SCALE_BW) {
#pragma unroll
                for (int m = 0; m < MT; ++m) acc[m] = 0.f;
            }
            simd<float, 16> sc = 0.f;
            simd<sycl::half, 32> sc32 = 0.f;
            simd<sycl::half, 32> hz32 = 0.f;

            for (int s64 = 0; s64 < n_steps; ++s64) {
                const uint32_t cur_slm = (s64 & 1) ? SLM_STEP_BYTES : 0u;
                const uint32_t nxt_slm = (s64 & 1) ? 0u : SLM_STEP_BYTES;

                simd<uint32_t, 32> xr_nxt[C_TH];
                if (s64 + 1 < n_steps) {
                    load_x64<C_TH>(X16, n_in, T, col_wg, tid, s64 + 1, xr_nxt);
                }

                if (row_ok) {
                    const int kb = s64 >> 2;
                    const int g2 = (s64 >> 1) & 1;
                    const int p  = s64 & 3;
                    const uint32_t* rec = w_tile + (size_t)kb * 544u;

                    if ((s64 & 1) == 0) {
                        simd<uint32_t, 16> mg = block_load<uint32_t, 16>(rec + 512u + (uint32_t)(16 * g2));
                        if constexpr (SCALE_BW) {
                            simd<sycl::half, 16> sc16 = mg.template bit_cast_view<sycl::half>().template select<16, 2>(0);
                            sc32.template select<16, 2>(0) = sc16;
                            sc32.template select<16, 2>(1) = sc16;
                        } else {
                            sc = half_of(mg);
#pragma unroll
                            for (int m = 0; m < MT; ++m) acc[m] = 0.f;
                        }
                        simd<uint16_t, 16> z16 = mg >> 16;
                        simd<float, 16> zf = z16.template bit_cast_view<int16_t>();
                        simd<sycl::half, 16> hz16 = 1024.0f + zf;
                        hz32.template select<16, 2>(0) = hz16;
                        hz32.template select<16, 2>(1) = hz16;
                    }

                    if (s64 + 1 < n_steps) {
                        const int nxt_kb = (s64 + 1) >> 2;
                        const int nxt_p  = (s64 + 1) & 3;
                        const uint32_t* nxt_rec = w_tile + (size_t)nxt_kb * 544u;
                        prefetch<uint32_t, 8>(nxt_rec + (uint32_t)(128 * nxt_p), simd<uint32_t, 8>(0, 64), kPf);
                        if (nxt_p == 0) {
                            prefetch<uint32_t, 2>(nxt_rec + 512u, simd<uint32_t, 2>(0, 64), kPf);
                        }
                    }

                    simd<uint32_t, 128> qd;
                    qd.template select<64, 1>(0)  = block_load<uint32_t, 64>(rec + (uint32_t)(128 * p));
                    qd.template select<64, 1>(64) = block_load<uint32_t, 64>(rec + (uint32_t)(128 * p + 64));

                    simd<uint32_t, 128> v = qd & 0x0f0f0f0fu;
                    simd<sycl::half, 256> bw0, bw1;
                    unpack_bw_all<SCALE_BW, 0>(v, hz32, sc32, bw0);

                    const int wm_m0 = wm * MT;
                    run_k16<SCALE_BW, N_M8, MT, 0>(cur_slm, wm_m0, v, hz32, sc32, bw0, bw1, y_acc, acc,
                                                   std::make_integer_sequence<int, MT>{});
                    v = (qd >> 4) & 0x0f0f0f0fu;
                    run_k16<SCALE_BW, N_M8, MT, 1>(cur_slm, wm_m0, v, hz32, sc32, bw1, bw0, y_acc, acc,
                                                   std::make_integer_sequence<int, MT>{});
                    run_k16<SCALE_BW, N_M8, MT, 2>(cur_slm, wm_m0, v, hz32, sc32, bw0, bw1, y_acc, acc,
                                                   std::make_integer_sequence<int, MT>{});
                    run_k16<SCALE_BW, N_M8, MT, 3>(cur_slm, wm_m0, v, hz32, sc32, bw1, bw0, y_acc, acc,
                                                   std::make_integer_sequence<int, MT>{});

                    if constexpr (!SCALE_BW) {
                        if ((s64 & 1) == 1) {
#pragma unroll
                            for (int m = 0; m < MT; ++m) {
#pragma unroll
                                for (int r = 0; r < 8; ++r) {
                                    simd<float, 16> a  = acc[m].template select<16, 1>(16 * r);
                                    simd<float, 16> yv = y_acc[m].template select<16, 1>(16 * r);
                                    yv += a * sc;
                                    y_acc[m].template select<16, 1>(16 * r) = yv;
                                }
                            }
                        }
                    }
                }

                if (s64 + 1 < n_steps) {
                    store_x64<C_TH, N_M8>(nxt_slm, tid, xr_nxt);
                    barrier();
                }
            }

            if (row_ok) {
                const int col0 = col_wg + wm * MT * 8;
                const size_t row_off = (size_t)tile_idx * 16u;
#pragma unroll
                for (int m = 0; m < MT; ++m) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int c = col0 + 8 * m + r;
                        if (c < T) {
                            block_store<float, 16>(Y + (size_t)c * (size_t)n_out + row_off,
                                                   y_acc[m].template select<16, 1>(16 * r));
                        }
                    }
                }
            }
        };
        auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
        const sycl::nd_range<1> nd((size_t)total_wg * NTHREADS, (size_t)NTHREADS);
        h.parallel_for<TagW4F16<WGT_N, WGT_M, MT, SCALE_BW>>(nd, props, run_kernel);
    });
}

inline int w4f16_cfg_override() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_Q36_W4F16_CFG");
        return e ? std::atoi(e) : -1;
    }();
    return v;
}

}  // namespace w4f16

bool handles_w4_f16(int n_in, int n_out, int T) noexcept {
    if (n_in <= 0 || n_out <= 0 || T <= 0) return false;
    return (n_in % 256) == 0 && (n_out % 16) == 0;
}

bool gemm_w4_f16(const void* weights, const void* x_f16, float* y, int n_in, int n_out, int T, void* stream) {
    if (!handles_w4_f16(n_in, n_out, T) || !weights || !x_f16 || !y) return false;
    if ((reinterpret_cast<uintptr_t>(weights) & 63u) != 0u) return false;
    if ((reinterpret_cast<uintptr_t>(x_f16) & 3u) != 0u) return false;
    if ((reinterpret_cast<uintptr_t>(y) & 15u) != 0u) return false;
    sycl::queue& q = *strata::q_of(stream);
    const auto* W = static_cast<const uint32_t*>(weights);
    const auto* X = static_cast<const uint32_t*>(x_f16);
    const int ov = w4f16::w4f16_cfg_override();
    const int n256 = (n_out + 255) / 256;
    const int m128 = (T + 127) / 128;
    int cfg = ov;
    if (cfg < 0) {
        if (n_out < 4096 || (T <= 128 && (int64_t)n_out * T <= 2097152)) {
            cfg = 3;
        } else if ((int64_t)n256 * m128 >= 32) {
            cfg = 0;
        } else {
            cfg = 2;
        }
    }
    if (cfg == 0) {
        w4f16::launch_w4f16<16, 1, 16, true>(q, W, X, y, n_in, n_out, T);
    } else if (cfg == 1) {
        w4f16::launch_w4f16<8, 2, 8, true>(q, W, X, y, n_in, n_out, T);
    } else if (cfg == 2) {
        w4f16::launch_w4f16<8, 1, 16, true>(q, W, X, y, n_in, n_out, T);
    } else {
        w4f16::launch_w4f16<4, 2, 4, false>(q, W, X, y, n_in, n_out, T);
    }
    return true;
}

namespace gdn_ab {

template <int N16> class TagGdnAbPp;

template <int N16>
SYCL_ESIMD_FUNCTION inline void load_h_k32(const float* h, int n_in, int T, int m0, int tid, int k32,
                                           simd<uint32_t, 32>& b0, simd<uint32_t, 32>& b1) {
    const int row0 = m0 + tid * 4;
    const int koff = k32 * 32;
#pragma unroll
    for (int rr = 0; rr < 4; ++rr) {
        simd<uint32_t, 16> vu = 0u;
        if (row0 + rr < T) {
            simd<float, 32> vf = block_load<float, 32>(h + (size_t)(row0 + rr) * n_in + koff, kAl16);
            simd<sycl::half, 32> vh = convert<sycl::half>(vf);
            vu = vh.template bit_cast_view<uint32_t>();
        }
        b0.template select<8, 1>(8 * rr) = vu.template select<8, 1>(0);
        b1.template select<8, 1>(8 * rr) = vu.template select<8, 1>(8);
    }
}

template <int N16>
void launch_gdn_ab(sycl::queue& q, const uint32_t* ab16, const float* h, const float* dt, const float* ssm_a,
                   float* al_out, float* be_out, int n_in, int T) {
    constexpr int HV = N16 * 16;
    constexpr int M_WG = 32;
    constexpr int WG = 8;
    const int n_wg = (T + M_WG - 1) / M_WG;
    const int k32_cnt = n_in / 32;
    const int k16_cnt = n_in / 16;
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<TagGdnAbPp<N16>>(
            sycl::nd_range<2>(sycl::range<2>((size_t)n_wg, (size_t)WG), sycl::range<2>(1, (size_t)WG)),
            props, [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
                slm_init<4096u>();
                const int tid    = (int)it.get_local_id(1);
                const int m0     = (int)it.get_group(0) * M_WG;
                const int m_th   = tid >> 1;
                const int branch = tid & 1;
                const uint32_t* Wb = ab16 + (size_t)branch * N16 * k16_cnt * 128u;

                {
                    simd<uint32_t, 32> b0, b1;
                    load_h_k32<N16>(h, n_in, T, m0, tid, 0, b0, b1);
                    slm_block_store<uint32_t, 32>((0u * 4u + (uint32_t)m_th) * 256u + (uint32_t)branch * 128u, b0, kAl16);
                    slm_block_store<uint32_t, 32>((1u * 4u + (uint32_t)m_th) * 256u + (uint32_t)branch * 128u, b1, kAl16);
                }

                simd<float, 128> c[N16];
#pragma unroll
                for (int t = 0; t < N16; ++t) c[t] = 0.0f;

                for (int k32 = 0; k32 < k32_cnt; ++k32) {
                    barrier();
                    const uint32_t cur_slm = (k32 & 1) ? 2048u : 0u;
                    simd<uint32_t, 32> nb0, nb1;
                    if (k32 + 1 < k32_cnt) load_h_k32<N16>(h, n_in, T, m0, tid, k32 + 1, nb0, nb1);

#pragma unroll
                    for (int k16 = 0; k16 < 2; ++k16) {
                        simd<uint32_t, 64> ahu = slm_block_load<uint32_t, 64>(
                            cur_slm + ((uint32_t)k16 * 4u + (uint32_t)m_th) * 256u, kAl16);
                        simd<sycl::half, 128> ah = ahu.template bit_cast_view<sycl::half>();
#pragma unroll
                        for (int t = 0; t < N16; ++t) {
                            simd<uint32_t, 128> bwu = block_load<uint32_t, 128>(
                                Wb + ((size_t)t * k16_cnt + (size_t)(2 * k32 + k16)) * 128u, kAl16);
                            simd<sycl::half, 256> bw = bwu.template bit_cast_view<sycl::half>();
                            c[t] = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(c[t], bw, ah);
                        }
                    }

                    if (k32 + 1 < k32_cnt) {
                        const uint32_t nxt_slm = (k32 & 1) ? 0u : 2048u;
                        slm_block_store<uint32_t, 32>(nxt_slm + (0u * 4u + (uint32_t)m_th) * 256u + (uint32_t)branch * 128u, nb0, kAl16);
                        slm_block_store<uint32_t, 32>(nxt_slm + (1u * 4u + (uint32_t)m_th) * 256u + (uint32_t)branch * 128u, nb1, kAl16);
                    }
                }

                simd<float, 16> dtv[N16], sav[N16];
                if (branch == 0) {
#pragma unroll
                    for (int t = 0; t < N16; ++t) {
                        dtv[t] = block_load<float, 16>(dt + t * 16, kAl16);
                        sav[t] = block_load<float, 16>(ssm_a + t * 16, kAl16);
                    }
                }
                const int row_base = m0 + m_th * 8;
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int row = row_base + r;
                    if (row < T) {
                        if (branch == 0) {
#pragma unroll
                            for (int t = 0; t < N16; ++t) {
                                simd<float, 16> v = c[t].template select<16, 1>(16 * r) + dtv[t];
                                simd<float, 16> u = es::exp(v);
                                simd<float, 16> sp = es::log(1.0f + u);
                                sp.merge(u - 0.5f * u * u, u < 1e-4f);
                                sp.merge(v, v > 20.0f);
                                block_store<float, 16>(al_out + (size_t)row * HV + t * 16, sp * sav[t], kAl16);
                            }
                        } else {
#pragma unroll
                            for (int t = 0; t < N16; ++t) {
                                simd<float, 16> v = c[t].template select<16, 1>(16 * r);
                                simd<float, 16> sg = 1.0f / (1.0f + es::exp(-v));
                                block_store<float, 16>(be_out + (size_t)row * HV + t * 16, sg, kAl16);
                            }
                        }
                    }
                }
            });
    });
}

}  // namespace gdn_ab

void gdn_ab_pp(const void* ab16, const float* h, const float* dt, const float* ssm_a,
               float* al_out, float* be_out, int n_in, int HV, int T, void* stream) {
    sycl::queue& q = *strata::q_of(stream);
    const auto* w = static_cast<const uint32_t*>(ab16);
    if (HV == 48)      gdn_ab::launch_gdn_ab<3>(q, w, h, dt, ssm_a, al_out, be_out, n_in, T);
    else if (HV == 16) gdn_ab::launch_gdn_ab<1>(q, w, h, dt, ssm_a, al_out, be_out, n_in, T);
    else if (HV == 32) gdn_ab::launch_gdn_ab<2>(q, w, h, dt, ssm_a, al_out, be_out, n_in, T);
    else if (HV == 64) gdn_ab::launch_gdn_ab<4>(q, w, h, dt, ssm_a, al_out, be_out, n_in, T);
}

// Shared host/ESIMD helpers for PP2 and PP3
namespace pp_util {
struct RecipDiv {
    uint32_t d;
    uint64_t mul;
};

inline RecipDiv make_recip(uint32_t d) noexcept {
    return RecipDiv{d, ((1ULL << 40) + (uint64_t)d - 1ULL) / (uint64_t)d};
}

SYCL_ESIMD_FUNCTION inline uint32_t fast_div(uint32_t x, const RecipDiv& r) {
    return static_cast<uint32_t>((static_cast<uint64_t>(x) * r.mul) >> 40);
}

SYCL_ESIMD_FUNCTION inline float hmax2(float a, float b) { return a > b ? a : b; }

SYCL_ESIMD_FUNCTION inline float hmax8_f(simd<float, 8> v) {
    simd<float, 4> c = (es::max)(simd<float, 4>(v.template select<4, 1>(0)),
                                 simd<float, 4>(v.template select<4, 1>(4)));
    simd<float, 2> d = (es::max)(simd<float, 2>(c.template select<2, 1>(0)),
                                 simd<float, 2>(c.template select<2, 1>(2)));
    return hmax2(float(d[0]), float(d[1]));
}

SYCL_ESIMD_FUNCTION inline float hmax16_f(simd<float, 16> v) {
    simd<float, 8> b = (es::max)(simd<float, 8>(v.template select<8, 1>(0)),
                                 simd<float, 8>(v.template select<8, 1>(8)));
    return hmax8_f(b);
}

SYCL_ESIMD_FUNCTION inline int hsum32_i(simd<int, 32> v) {
    simd<int, 16> a = v.template select<16, 1>(0) + v.template select<16, 1>(16);
    simd<int, 8>  b = a.template select<8, 1>(0)  + a.template select<8, 1>(8);
    simd<int, 4>  c = b.template select<4, 1>(0)  + b.template select<4, 1>(4);
    simd<int, 2>  d = c.template select<2, 1>(0)  + c.template select<2, 1>(2);
    return int(d[0]) + int(d[1]);
}

SYCL_ESIMD_FUNCTION inline simd<int, 32> quant_rnde32(simd<float, 32> x, float inv_sc) {
    simd<float, 32> r = es::rnde<float, 32>(x * inv_sc);
    r = (es::min)((es::max)(r, simd<float, 32>(-127.0f)), simd<float, 32>(127.0f));
    return convert<int>(r);
}

SYCL_ESIMD_FUNCTION inline void gu_swiglu_epilogue(
    simd<float, 128> (&cg)[4],
    simd<float, 128> (&cu)[4],
    uint32_t* xq8_out,
    uint32_t* dxT_out,
    uint32_t lid,
    uint32_t ri,
    uint32_t ci,
    uint32_t wgr,
    uint32_t wgc,
    uint32_t n_g_out,
    uint32_t n_ct32,
    uint32_t T) {
    const uint32_t ct32_b = wgc * 4u;
    #pragma unroll
    for (int jt = 0; jt < 4; ++jt) {
        const uint32_t ct32 = ct32_b + static_cast<uint32_t>(jt);
        if (ct32 >= n_ct32) break;

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            simd<float, 16> vg  = cg[jt].template select<16, 1>(16 * r);
            simd<float, 16> vu  = cu[jt].template select<16, 1>(16 * r);
            simd<float, 16> sig = 1.0f / (1.0f + es::exp<float, 16>(-vg));
            simd<float, 16> sw  = (vg * sig) * vu;
            const uint32_t col32 = ci * 8u + static_cast<uint32_t>(r);
            es::slm_block_store<float, 16>((col32 * 128u + 16u * ri) * 4u, sw, kAl16);
            es::slm_scalar_store<float>(16384u + (col32 * 8u + ri) * 4u, hmax16_f(es::abs<float, 16>(sw)));
        }
        barrier();

        const uint32_t col = ct32 * 32u + lid;
        simd<uint32_t, 16> p0 = 0u, p1 = 0u;
        float dx = 0.0f;
        int sumx = 0;
        if (col < T) {
            simd<float, 8> am8 = es::slm_block_load<float, 8>(16384u + lid * 32u, kAl16);
            const float amax   = hmax8_f(am8);
            dx = amax / 127.0f;
            const float inv    = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            simd<float, 32> e0 = es::slm_block_load<float, 32>(lid * 512u + 0u,   kAl16);
            simd<float, 32> o0 = es::slm_block_load<float, 32>(lid * 512u + 128u, kAl16);
            simd<float, 32> e1 = es::slm_block_load<float, 32>(lid * 512u + 256u, kAl16);
            simd<float, 32> o1 = es::slm_block_load<float, 32>(lid * 512u + 384u, kAl16);
            simd<int, 32> qe0  = quant_rnde32(e0, inv);
            simd<int, 32> qo0  = quant_rnde32(o0, inv);
            simd<int, 32> qe1  = quant_rnde32(e1, inv);
            simd<int, 32> qo1  = quant_rnde32(o1, inv);
            sumx = hsum32_i(qe0) + hsum32_i(qo0) + hsum32_i(qe1) + hsum32_i(qo1);
            simd<int8_t, 64> eo0, eo1;
            eo0.template select<32, 2>(0) = convert<int8_t>(qe0);
            eo0.template select<32, 2>(1) = convert<int8_t>(qo0);
            eo1.template select<32, 2>(0) = convert<int8_t>(qe1);
            eo1.template select<32, 2>(1) = convert<int8_t>(qo1);
            p0 = eo0.template bit_cast_view<uint32_t>();
            p1 = eo1.template bit_cast_view<uint32_t>();
        }
        barrier();

        es::slm_block_store<uint32_t, 8>(0u    + lid * 32u, p0.template select<8, 1>(0), kAl16);
        es::slm_block_store<uint32_t, 8>(1024u + lid * 32u, p0.template select<8, 1>(8), kAl16);
        es::slm_block_store<uint32_t, 8>(2048u + lid * 32u, p1.template select<8, 1>(0), kAl16);
        es::slm_block_store<uint32_t, 8>(3072u + lid * 32u, p1.template select<8, 1>(8), kAl16);
        const uint32_t jc = lid >> 3, rc = lid & 7u;
        es::slm_scalar_store<float>(4096u + (16u * jc + rc) * 4u, dx);
        es::slm_scalar_store<int>(4096u + (16u * jc + 8u + rc) * 4u, sumx);
        barrier();

        uint32_t* xq_blk = xq8_out + static_cast<size_t>(ct32 * n_g_out + wgr) * 1024u;
        uint32_t* dx_blk = dxT_out + static_cast<size_t>(ct32 * n_g_out + wgr) * 64u;
        if (lid < 16u) {
            simd<uint32_t, 64> out = es::slm_block_load<uint32_t, 64>(lid * 256u, kAl16);
            es::block_store<uint32_t, 64>(xq_blk + lid * 64u, out, kAl16);
        } else if (lid == 16u) {
            simd<uint32_t, 64> dout = es::slm_block_load<uint32_t, 64>(4096u, kAl16);
            es::block_store<uint32_t, 64>(dx_blk, dout, kAl16);
        }
        barrier();
    }
}
}  // namespace pp_util

// ---------------------------------------------------------------------------
// PP2: 256-GRF Card-Up Fat GEMM + Fused FFN Gate/Up+SwiGLU+QuantX128 (Xe2 BMG)
// ---------------------------------------------------------------------------
namespace pp2 {
namespace {
using pp_util::RecipDiv;
using pp_util::make_recip;
using pp_util::fast_div;

inline uint32_t env_pp2_wgs() noexcept {
    static const uint32_t v = []() -> uint32_t {
        const char* e = std::getenv("STRATA_Q36_PP2_WGS");
        const int x = e ? std::atoi(e) : 0;
        return x > 0 ? static_cast<uint32_t>(x) : 32u;
    }();
    return v;
}

class TagW4I8Pp2;
class TagW4I8GuSwigluPp2;

template <int S4>
SYCL_ESIMD_FUNCTION inline void pp2_w4_substep(
    simd<int, 128> (&i_acc)[2][4],
    simd<int, 64> (&wr)[2][2],
    const uint32_t* w_tile0,
    const uint32_t* w_tile1,
    const int* xp,
    float* y_base,
    uint32_t g,
    uint32_t n_grp,
    uint32_t last_gs,
    uint32_t nv) {
    constexpr int u = S4 & 1;
    const uint32_t gs = g * 4u + static_cast<uint32_t>(S4);
    const int* xp_cur = xp + static_cast<size_t>(gs) * 256u;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        simd<int, 64> a = es::block_load<int, 64>(xp_cur + static_cast<uint32_t>(j) * 64u, kAl16);
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            i_acc[i][j] = xmx::dpas<8, 8, int, int, int, int,
                xmx::dpas_argument_type::u4, xmx::dpas_argument_type::s8>(
                i_acc[i][j], wr[u][i], a);
        }
    }
    const uint32_t gs_n = (gs + 2u <= last_gs) ? (gs + 2u) : last_gs;
    const uint32_t w_n  = ((gs_n >> 3) * 544u) + ((gs_n & 7u) << 6);
    wr[u][0] = es::block_load<int, 64>(reinterpret_cast<const int*>(w_tile0 + w_n), kAl16);
    wr[u][1] = es::block_load<int, 64>(reinterpret_cast<const int*>(w_tile1 + w_n), kAl16);

    const uint32_t gs_pa = (gs + 1u <= last_gs) ? (gs + 1u) : last_gs;
    const int* xp_pa = xp + static_cast<size_t>(gs_pa) * 256u;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        es::prefetch<int, 64>(xp_pa + static_cast<uint32_t>(j) * 64u, 0u, kPfL1);
    }
    const uint32_t gs_pw = (gs + 4u <= last_gs) ? (gs + 4u) : last_gs;
    const uint32_t w_pf  = ((gs_pw >> 3) * 544u) + ((gs_pw & 7u) << 6);
    es::prefetch<uint32_t, 64>(w_tile0 + w_pf, 0u, kPfL1);
    es::prefetch<uint32_t, 64>(w_tile1 + w_pf, 0u, kPfL1);
    if constexpr (S4 == 0) {
        const uint32_t g_pf = (g + 1u < n_grp) ? (g + 1u) : (n_grp - 1u);
        const uint32_t m_pf = ((g_pf >> 1) * 544u) + 512u + ((g_pf & 1u) << 4);
        es::prefetch<uint32_t, 16>(w_tile0 + m_pf, 0u, kPfL1);
        es::prefetch<uint32_t, 16>(w_tile1 + m_pf, 0u, kPfL1);
    }
    if (gs == nv) y_base[0] = 0.0f;
}

template <int S4>
SYCL_ESIMD_FUNCTION inline void pp2_gu_substep(
    simd<int, 128> (&ig)[4],
    simd<int, 128> (&iu)[4],
    simd<int, 64> (&wr_g)[2],
    simd<int, 64> (&wr_u)[2],
    const uint32_t* wg_tile,
    const uint32_t* wu_tile,
    const int* const (&xp)[4],
    uint32_t* dxT_out,
    uint32_t ci,
    uint32_t g,
    uint32_t n_grp,
    uint32_t last_gs,
    uint32_t nv) {
    constexpr int u = S4 & 1;
    const uint32_t gs = g * 4u + static_cast<uint32_t>(S4);
    const size_t x_s_off = static_cast<size_t>(gs) * 256u + ci * 64u;
    #pragma unroll
    for (int jt = 0; jt < 4; ++jt) {
        simd<int, 64> a = es::block_load<int, 64>(xp[jt] + x_s_off, kAl16);
        ig[jt] = xmx::dpas<8, 8, int, int, int, int,
            xmx::dpas_argument_type::u4, xmx::dpas_argument_type::s8>(ig[jt], wr_g[u], a);
        iu[jt] = xmx::dpas<8, 8, int, int, int, int,
            xmx::dpas_argument_type::u4, xmx::dpas_argument_type::s8>(iu[jt], wr_u[u], a);
    }
    const uint32_t gs_n = (gs + 2u <= last_gs) ? (gs + 2u) : last_gs;
    const uint32_t w_n  = ((gs_n >> 3) * 544u) + ((gs_n & 7u) << 6);
    wr_g[u] = es::block_load<int, 64>(reinterpret_cast<const int*>(wg_tile + w_n), kAl16);
    wr_u[u] = es::block_load<int, 64>(reinterpret_cast<const int*>(wu_tile + w_n), kAl16);

    const uint32_t gs_pa = (gs + 1u <= last_gs) ? (gs + 1u) : last_gs;
    const size_t x_pa_off = static_cast<size_t>(gs_pa) * 256u + ci * 64u;
    #pragma unroll
    for (int jt = 0; jt < 4; ++jt) {
        es::prefetch<int, 64>(xp[jt] + x_pa_off, 0u, kPfL1);
    }
    const uint32_t gs_pw = (gs + 4u <= last_gs) ? (gs + 4u) : last_gs;
    const uint32_t w_pf  = ((gs_pw >> 3) * 544u) + ((gs_pw & 7u) << 6);
    es::prefetch<uint32_t, 64>(wg_tile + w_pf, 0u, kPfL1);
    es::prefetch<uint32_t, 64>(wu_tile + w_pf, 0u, kPfL1);
    if constexpr (S4 == 0) {
        const uint32_t g_pf = (g + 1u < n_grp) ? (g + 1u) : (n_grp - 1u);
        const uint32_t m_pf = ((g_pf >> 1) * 544u) + 512u + ((g_pf & 1u) << 4);
        es::prefetch<uint32_t, 16>(wg_tile + m_pf, 0u, kPfL1);
        es::prefetch<uint32_t, 16>(wu_tile + m_pf, 0u, kPfL1);
    }
    if (gs == nv) dxT_out[0] = 0u;
}

void launch_w4i8_pp2(sycl::queue& q, const uint32_t* w_base, const int* xq8_base,
                     const uint32_t* dxT_base, float* y_base,
                     int K, int R, int T) {
    const uint32_t n_rt       = static_cast<uint32_t>(R) / 16u;
    const uint32_t n_kb       = static_cast<uint32_t>(K) / 256u;
    const uint32_t n_grp      = n_kb * 2u;
    const uint32_t n_steps    = n_grp * 4u;
    const uint32_t last_gs    = n_steps - 1u;
    const uint32_t n_ct32     = static_cast<uint32_t>(T + 31) / 32u;
    const uint32_t n_wgr      = (n_rt + 7u) / 8u;         // 128 rows per WG (4 row-tiles of 32)
    const uint32_t n_wgc_tot  = (n_ct32 + 7u) / 8u;       // 256 cols per WG (8 col-tiles of 32)

    // Size column panel to <= 12 MiB activations so panel fits cleanly in the 24 MiB LLC.
    const uint32_t target_lg = (K <= 12288) ? 2u : 1u;    // 4 WG-cols = 1024 cols or 2 WG-cols = 512 cols
    uint32_t lg_wgc_p = 0u;
    if (n_wgc_tot > 2u && target_lg >= 2u)      lg_wgc_p = 2u;
    else if (n_wgc_tot > 1u && target_lg >= 1u) lg_wgc_p = 1u;

    const uint32_t wgc_panel  = 1u << lg_wgc_p;
    const uint32_t wgc_mask   = wgc_panel - 1u;
    const uint32_t n_cp       = (n_wgc_tot + wgc_panel - 1u) >> lg_wgc_p;
    const uint32_t panel_wgs  = n_wgr << lg_wgc_p;
    const uint32_t nw         = n_cp * panel_wgs;
    const RecipDiv div_pw     = make_recip(panel_wgs);
    const uint32_t g_wgs      = std::min<uint32_t>(nw, env_pp2_wgs());
    const uint32_t nv         = 0xffffffffu;

    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagW4I8Pp2>(
            sycl::nd_range<1>(static_cast<size_t>(g_wgs) * 32u, 32u), props,
            [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
                const uint32_t wg0 = static_cast<uint32_t>(it.get_group(0));
                const uint32_t l   = static_cast<uint32_t>(it.get_local_id(0));
                const uint32_t ri  = l >> 3;     // 0..3 (32 rows each)
                const uint32_t ci  = l & 7u;     // 0..7 (32 cols each)

                for (uint32_t w = wg0; w < nw; w += g_wgs) {
                    const uint32_t cp     = fast_div(w, div_pw);
                    const uint32_t rem_p  = w - cp * panel_wgs;
                    const uint32_t wgr    = rem_p >> lg_wgc_p;
                    const uint32_t wgc    = (cp << lg_wgc_p) + (rem_p & wgc_mask);
                    if (wgc >= n_wgc_tot) continue;

                    const uint32_t rt0    = (wgr * 4u + ri) * 2u;
                    const uint32_t rt1    = rt0 + 1u;
                    const uint32_t ct32   = wgc * 8u + ci;
                    if (rt0 >= n_rt || ct32 >= n_ct32) continue;

                    const uint32_t* w_tile0 = w_base + static_cast<size_t>(rt0) * static_cast<size_t>(n_kb) * 544u;
                    const uint32_t* w_tile1 = (rt1 < n_rt) ? (w_tile0 + static_cast<size_t>(n_kb) * 544u) : w_tile0;
                    const int* xp           = xq8_base + static_cast<size_t>(ct32) * static_cast<size_t>(n_steps) * 256u;
                    const uint32_t* dxp_ct  = dxT_base + static_cast<size_t>(ct32) * static_cast<size_t>(n_grp) * 64u;

                    simd<float, 128> c_acc[2][4];
                    #pragma unroll
                    for (int i = 0; i < 2; ++i)
                        #pragma unroll
                        for (int j = 0; j < 4; ++j)
                            c_acc[i][j] = 0.0f;

                    simd<int, 64> wr[2][2];
                    wr[0][0] = es::block_load<int, 64>(reinterpret_cast<const int*>(w_tile0), kAl16);
                    wr[0][1] = es::block_load<int, 64>(reinterpret_cast<const int*>(w_tile1), kAl16);
                    const uint32_t gs1 = (1u <= last_gs) ? 1u : last_gs;
                    wr[1][0] = es::block_load<int, 64>(reinterpret_cast<const int*>(w_tile0 + 64u * gs1), kAl16);
                    wr[1][1] = es::block_load<int, 64>(reinterpret_cast<const int*>(w_tile1 + 64u * gs1), kAl16);

                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        es::prefetch<int, 64>(xp + static_cast<uint32_t>(j) * 64u, 0u, kPfL1);
                    }
                    for (uint32_t pf = 2u; pf <= 3u; ++pf) {
                        const uint32_t gs_p = (pf <= last_gs) ? pf : last_gs;
                        const uint32_t w_p  = ((gs_p >> 3) * 544u) + ((gs_p & 7u) << 6);
                        es::prefetch<uint32_t, 64>(w_tile0 + w_p, 0u, kPfL1);
                        es::prefetch<uint32_t, 64>(w_tile1 + w_p, 0u, kPfL1);
                    }
                    es::prefetch<uint32_t, 16>(w_tile0 + 512u, 0u, kPfL1);
                    es::prefetch<uint32_t, 16>(w_tile1 + 512u, 0u, kPfL1);

                    #pragma nounroll
                    for (uint32_t g = 0; g < n_grp; ++g) {
                        simd<int, 128> i_acc[2][4];
                        #pragma unroll
                        for (int i = 0; i < 2; ++i)
                            #pragma unroll
                            for (int j = 0; j < 4; ++j)
                                i_acc[i][j] = 0;

                        pp2_w4_substep<0>(i_acc, wr, w_tile0, w_tile1, xp, y_base, g, n_grp, last_gs, nv);
                        pp2_w4_substep<1>(i_acc, wr, w_tile0, w_tile1, xp, y_base, g, n_grp, last_gs, nv);
                        pp2_w4_substep<2>(i_acc, wr, w_tile0, w_tile1, xp, y_base, g, n_grp, last_gs, nv);
                        pp2_w4_substep<3>(i_acc, wr, w_tile0, w_tile1, xp, y_base, g, n_grp, last_gs, nv);

                        const uint32_t m_cur = ((g >> 1) * 544u) + 512u + ((g & 1u) << 4);
                        simd<uint32_t, 16> meta0 = es::block_load<uint32_t, 16>(w_tile0 + m_cur, kAl16);
                        simd<uint32_t, 16> meta1 = es::block_load<uint32_t, 16>(w_tile1 + m_cur, kAl16);
                        simd<float, 16> s0 = half_of(meta0);
                        simd<float, 16> z0 = simd<float, 16>(meta0.template bit_cast_view<uint16_t>().template select<16, 2>(1));
                        simd<float, 16> s1 = half_of(meta1);
                        simd<float, 16> z1 = simd<float, 16>(meta1.template bit_cast_view<uint16_t>().template select<16, 2>(1));
                        simd<uint32_t, 64> dx_all = es::block_load<uint32_t, 64>(dxp_ct + static_cast<size_t>(g) * 64u, kAl16);

                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            simd<float, 8> dx_v = dx_all.template select<8, 1>(16 * j).template bit_cast_view<float>();
                            simd<int, 8>   sx_v = dx_all.template select<8, 1>(16 * j + 8).template bit_cast_view<int>();
                            #pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                const float d_r  = float(dx_v[r]);
                                const float sx_r = float(int(sx_v[r]));
                                const simd<float, 16> sd0  = s0 * d_r;
                                const simd<float, 16> sd1  = s1 * d_r;
                                const simd<float, 16> zsx0 = z0 * sx_r;
                                const simd<float, 16> zsx1 = z1 * sx_r;
                                c_acc[0][j].template select<16, 1>(16 * r) +=
                                    (simd<float, 16>(i_acc[0][j].template select<16, 1>(16 * r)) - zsx0) * sd0;
                                c_acc[1][j].template select<16, 1>(16 * r) +=
                                    (simd<float, 16>(i_acc[1][j].template select<16, 1>(16 * r)) - zsx1) * sd1;
                            }
                        }
                        if (g == nv) y_base[0] = 0.0f;
                    }

                    const uint32_t row0     = rt0 * 16u;
                    const bool has_r1       = (rt1 < n_rt);
                    const uint32_t col_base = ct32 * 32u;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        #pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const uint32_t col = col_base + static_cast<uint32_t>(j * 8 + r);
                            if (col < static_cast<uint32_t>(T)) {
                                float* yp = y_base + static_cast<size_t>(col) * static_cast<size_t>(R) + row0;
                                if (has_r1) {
                                    simd<float, 32> out32;
                                    out32.template select<16, 1>(0)  = c_acc[0][j].template select<16, 1>(16 * r);
                                    out32.template select<16, 1>(16) = c_acc[1][j].template select<16, 1>(16 * r);
                                    es::block_store<float, 32>(yp, out32, kAl16);
                                } else {
                                    es::block_store<float, 16>(yp, c_acc[0][j].template select<16, 1>(16 * r), kAl16);
                                }
                            }
                        }
                    }
                }
            });
    });
}

void launch_w4i8_gu_swiglu_pp2(sycl::queue& q, const uint32_t* w_gate, const uint32_t* w_up,
                               const int* xq8_in, const uint32_t* dxT_in,
                               uint32_t* xq8_out, uint32_t* dxT_out,
                               int K, int R, int T) {
    const uint32_t n_kb      = static_cast<uint32_t>(K) / 256u;
    const uint32_t n_grp     = n_kb * 2u;
    const uint32_t n_steps   = n_grp * 4u;
    const uint32_t last_gs   = n_steps - 1u;
    const uint32_t n_g_out   = static_cast<uint32_t>(R) / 128u;   // 128 rows per WG (8 row-tiles x 16)
    const uint32_t n_ct32    = static_cast<uint32_t>(T + 31) / 32u;
    const uint32_t n_wgc_tot = (n_ct32 + 3u) / 4u;                // 128 cols per WG (4 ct32)

    const uint32_t target_lg = (K <= 12288) ? 3u : 2u;            // 8 WG-cols = 1024 cols or 4 = 512 cols
    uint32_t lg_wgc_p = 0u;
    if (n_wgc_tot > 4u && target_lg >= 3u)      lg_wgc_p = 3u;
    else if (n_wgc_tot > 2u && target_lg >= 2u) lg_wgc_p = 2u;
    else if (n_wgc_tot > 1u && target_lg >= 1u) lg_wgc_p = 1u;

    const uint32_t wgc_panel = 1u << lg_wgc_p;
    const uint32_t wgc_mask  = wgc_panel - 1u;
    const uint32_t n_cp      = (n_wgc_tot + wgc_panel - 1u) >> lg_wgc_p;
    const uint32_t panel_wgs = n_g_out << lg_wgc_p;
    const uint32_t nw        = n_cp * panel_wgs;
    const RecipDiv div_pw    = make_recip(panel_wgs);
    const uint32_t g_wgs     = std::min<uint32_t>(nw, env_pp2_wgs());
    const uint32_t nv        = 0xffffffffu;

    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagW4I8GuSwigluPp2>(
            sycl::nd_range<1>(static_cast<size_t>(g_wgs) * 32u, 32u), props,
            [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
                // SLM: 16384 B sw[col32][row128] + 1024 B amax[col32][8] = 17408 B; epilogue transpose reuses 0..4351 B.
                slm_init<17408u>();
                const uint32_t wg0 = static_cast<uint32_t>(it.get_group(0));
                const uint32_t lid = static_cast<uint32_t>(it.get_local_id(0));
                const uint32_t ri  = lid >> 2;    // 0..7 (16 rows each)
                const uint32_t ci  = lid & 3u;    // 0..3 (8 cols inside each ct32)

                for (uint32_t w = wg0; w < nw; w += g_wgs) {
                    const uint32_t cp    = fast_div(w, div_pw);
                    const uint32_t rem_p = w - cp * panel_wgs;
                    const uint32_t wgr   = rem_p >> lg_wgc_p;
                    const uint32_t wgc   = (cp << lg_wgc_p) + (rem_p & wgc_mask);
                    if (wgc >= n_wgc_tot) continue;

                    const uint32_t rt       = wgr * 8u + ri;
                    const uint32_t ct32_b   = wgc * 4u;
                    const size_t w_off_tile = static_cast<size_t>(rt) * static_cast<size_t>(n_kb) * 544u;
                    const uint32_t* wg_tile = w_gate + w_off_tile;
                    const uint32_t* wu_tile = w_up   + w_off_tile;

                    const int* xp[4];
                    const uint32_t* dxp[4];
                    #pragma unroll
                    for (int jt = 0; jt < 4; ++jt) {
                        const uint32_t ct = (ct32_b + static_cast<uint32_t>(jt) < n_ct32)
                                          ? (ct32_b + static_cast<uint32_t>(jt)) : (n_ct32 - 1u);
                        xp[jt]  = xq8_in + static_cast<size_t>(ct) * static_cast<size_t>(n_steps) * 256u;
                        dxp[jt] = dxT_in + static_cast<size_t>(ct) * static_cast<size_t>(n_grp)   * 64u;
                    }

                    simd<float, 128> cg[4], cu[4];
                    #pragma unroll
                    for (int jt = 0; jt < 4; ++jt) { cg[jt] = 0.0f; cu[jt] = 0.0f; }

                    simd<int, 64> wr_g[2], wr_u[2];
                    wr_g[0] = es::block_load<int, 64>(reinterpret_cast<const int*>(wg_tile), kAl16);
                    wr_u[0] = es::block_load<int, 64>(reinterpret_cast<const int*>(wu_tile), kAl16);
                    const uint32_t gs1 = (1u <= last_gs) ? 1u : last_gs;
                    wr_g[1] = es::block_load<int, 64>(reinterpret_cast<const int*>(wg_tile + 64u * gs1), kAl16);
                    wr_u[1] = es::block_load<int, 64>(reinterpret_cast<const int*>(wu_tile + 64u * gs1), kAl16);

                    #pragma unroll
                    for (int jt = 0; jt < 4; ++jt) {
                        es::prefetch<int, 64>(xp[jt] + ci * 64u, 0u, kPfL1);
                    }
                    for (uint32_t pf = 2u; pf <= 3u; ++pf) {
                        const uint32_t gs_p = (pf <= last_gs) ? pf : last_gs;
                        const uint32_t w_p  = ((gs_p >> 3) * 544u) + ((gs_p & 7u) << 6);
                        es::prefetch<uint32_t, 64>(wg_tile + w_p, 0u, kPfL1);
                        es::prefetch<uint32_t, 64>(wu_tile + w_p, 0u, kPfL1);
                    }
                    es::prefetch<uint32_t, 16>(wg_tile + 512u, 0u, kPfL1);
                    es::prefetch<uint32_t, 16>(wu_tile + 512u, 0u, kPfL1);

                    #pragma nounroll
                    for (uint32_t g = 0; g < n_grp; ++g) {
                        simd<int, 128> ig[4], iu[4];
                        #pragma unroll
                        for (int jt = 0; jt < 4; ++jt) { ig[jt] = 0; iu[jt] = 0; }

                        pp2_gu_substep<0>(ig, iu, wr_g, wr_u, wg_tile, wu_tile, xp, dxT_out, ci, g, n_grp, last_gs, nv);
                        pp2_gu_substep<1>(ig, iu, wr_g, wr_u, wg_tile, wu_tile, xp, dxT_out, ci, g, n_grp, last_gs, nv);
                        pp2_gu_substep<2>(ig, iu, wr_g, wr_u, wg_tile, wu_tile, xp, dxT_out, ci, g, n_grp, last_gs, nv);
                        pp2_gu_substep<3>(ig, iu, wr_g, wr_u, wg_tile, wu_tile, xp, dxT_out, ci, g, n_grp, last_gs, nv);

                        const uint32_t m_cur = ((g >> 1) * 544u) + 512u + ((g & 1u) << 4);
                        simd<uint32_t, 16> mg = es::block_load<uint32_t, 16>(wg_tile + m_cur, kAl16);
                        simd<uint32_t, 16> mu = es::block_load<uint32_t, 16>(wu_tile + m_cur, kAl16);
                        simd<float, 16> sg = half_of(mg);
                        simd<float, 16> zg = simd<float, 16>(mg.template bit_cast_view<uint16_t>().template select<16, 2>(1));
                        simd<float, 16> su = half_of(mu);
                        simd<float, 16> zu = simd<float, 16>(mu.template bit_cast_view<uint16_t>().template select<16, 2>(1));

                        #pragma unroll
                        for (int jt = 0; jt < 4; ++jt) {
                            simd<uint32_t, 16> dxi = es::block_load<uint32_t, 16>(
                                dxp[jt] + static_cast<size_t>(g) * 64u + ci * 16u, kAl16);
                            simd<float, 8> dx_v = dxi.template select<8, 1>(0).template bit_cast_view<float>();
                            simd<int, 8>   sx_v = dxi.template select<8, 1>(8).template bit_cast_view<int>();
                            #pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                const float d_r  = float(dx_v[r]);
                                const float sx_r = float(int(sx_v[r]));
                                cg[jt].template select<16, 1>(16 * r) +=
                                    (simd<float, 16>(ig[jt].template select<16, 1>(16 * r)) - zg * sx_r) * (sg * d_r);
                                cu[jt].template select<16, 1>(16 * r) +=
                                    (simd<float, 16>(iu[jt].template select<16, 1>(16 * r)) - zu * sx_r) * (su * d_r);
                            }
                        }
                        if (g == nv) dxT_out[0] = 0u;
                    }

                    // Epilogue: SwiGLU + quantize_x128 (pp2 layout) per ct32 = ct32_b + jt.
                    pp_util::gu_swiglu_epilogue(cg, cu, xq8_out, dxT_out, lid, ri, ci, wgr, wgc, n_g_out, n_ct32,
                                                static_cast<uint32_t>(T));
                }
            });
    });
}

}  // namespace
}  // namespace pp2

// ---------------------------------------------------------------------------
// PP3: Fatgemm-paced 256-GRF Ping-Pong W4I8 GEMM + Fused FFN (Xe2 BMG)
// ---------------------------------------------------------------------------
namespace pp3 {
namespace {
using pp_util::RecipDiv;
using pp_util::make_recip;
using pp_util::fast_div;

SYCL_ESIMD_FUNCTION inline simd<sycl::half, 16> half_raw_of(simd<uint32_t, 16> h16) {
    simd<uint16_t, 16> lo = h16.template bit_cast_view<uint16_t>().template select<16, 2>(0);
    return lo.template bit_cast_view<sycl::half>();
}

inline uint32_t env_pp3_max_thr() noexcept {
    static const uint32_t v = []() -> uint32_t {
        const char* e = std::getenv("STRATA_Q36_PP3_THR");
        const int x = e ? std::atoi(e) : 0;
        return x > 0 ? static_cast<uint32_t>(x) : 1024u;
    }();
    return v;
}

// Column panel in columns (STRATA_Q36_PP3_PANEL, read once, default 1024: 18 MiB of activations at K = 17408, inside the 24 MiB LLC).
inline uint32_t env_pp3_panel() noexcept {
    static const uint32_t v = []() -> uint32_t {
        const char* e = std::getenv("STRATA_Q36_PP3_PANEL");
        const int x = e ? std::atoi(e) : 0;
        return (x >= 128 && x <= 8192) ? static_cast<uint32_t>(x) : 1024u;
    }();
    return v;
}

class TagW4I8Pp3;
class TagW4I8GuSwigluPp3;

struct Pp3Grid {
    uint32_t lgRT, lgCT, RT, CT;
    uint32_t ncg, panel_wgs, nw, g_wgs;
    RecipDiv div_pw, div_ncg;
};

inline void pick_pp3_grid(uint32_t n_rt32, uint32_t n_ct32, uint32_t max_thr,
                          uint32_t& out_lgRT, uint32_t& out_lgCT) noexcept {
    if (n_ct32 >= 16u && n_rt32 >= 4u) {
        out_lgRT = 2u; out_lgCT = 3u;
        return;
    }
    struct Cand { uint32_t lgr, lgc; };
    static constexpr Cand kCands[] = {
        {2u, 3u}, {3u, 2u}, {4u, 1u}, {3u, 1u}, {2u, 2u}, {2u, 1u}, {3u, 0u}, {4u, 0u},
    };
    double best = -1.0;
    uint32_t blgr = 2u, blgc = 3u;
    for (const auto& c : kCands) {
        const uint32_t rt = 1u << c.lgr;
        const uint32_t ct = 1u << c.lgc;
        const uint32_t wg = rt * ct;
        if (wg > max_thr) continue;
        const uint32_t max_wgs = std::max<uint32_t>(1u, max_thr / wg);
        const uint32_t nwgr    = (n_rt32 + rt - 1u) / rt;
        const uint32_t nwgc    = (n_ct32 + ct - 1u) / ct;
        const uint32_t nw      = nwgr * nwgc;
        const uint32_t waves   = (nw + max_wgs - 1u) / max_wgs;
        const double util = static_cast<double>(n_rt32 * n_ct32) /
                            (0.75 * static_cast<double>(waves * max_thr) + 0.25 * static_cast<double>(nw * wg));
        const double tie  = 2e-3 * (wg == 32u ? 1.0 : 0.0) + 1e-3 * static_cast<double>(c.lgc);
        const double sc   = util + tie;
        if (sc > best) {
            best = sc;
            blgr = c.lgr;
            blgc = c.lgc;
        }
    }
    out_lgRT = blgr;
    out_lgCT = blgc;
}

inline Pp3Grid make_pp3_grid(uint32_t n_rt32, uint32_t n_ct32, bool force_ct4_rt8) noexcept {
    Pp3Grid g{};
    const uint32_t max_thr = env_pp3_max_thr();
    if (force_ct4_rt8) {
        g.lgRT = 3u;
        g.lgCT = 2u;
    } else {
        pick_pp3_grid(n_rt32, n_ct32, max_thr, g.lgRT, g.lgCT);
    }
    g.RT = 1u << g.lgRT;
    g.CT = 1u << g.lgCT;
    const uint32_t wg_thr    = g.RT * g.CT;
    const uint32_t max_wgs   = std::max<uint32_t>(1u, max_thr / wg_thr);
    const uint32_t n_wgr     = (n_rt32 + g.RT - 1u) / g.RT;
    const uint32_t n_wgc_tot = (n_ct32 + g.CT - 1u) / g.CT;
    const uint32_t ncg_t     = std::max<uint32_t>(1u, (env_pp3_panel() / 32u) >> g.lgCT);  // default 1024 cols per panel
    if (n_wgc_tot <= ncg_t) {
        g.ncg       = std::max<uint32_t>(1u, n_wgc_tot);
        g.panel_wgs = std::max<uint32_t>(1u, n_wgr * g.ncg);
        g.nw        = g.panel_wgs;
    } else {
        g.ncg       = ncg_t;
        const uint32_t n_cp = (n_wgc_tot + g.ncg - 1u) / g.ncg;
        g.panel_wgs = n_wgr * g.ncg;
        g.nw        = n_cp * g.panel_wgs;
    }
    g.div_pw  = make_recip(g.panel_wgs);
    g.div_ncg = make_recip(g.ncg);
    g.g_wgs   = std::min<uint32_t>(g.nw, max_wgs);
    return g;
}

template <int S>
SYCL_ESIMD_FUNCTION inline void pp3_w4_substep(
    simd<int, 128> (&i_act)[2][4],
    simd<int, 128> (&i_oth)[2][4],
    simd<float, 128> (&c_acc)[2][4],
    simd<int, 64> (&wr)[2][2],
    simd<float, 16> (&s_scale)[2],
    simd<float, 32>& dx_scale,
    simd<uint32_t, 16>& m0_nxt,
    simd<uint32_t, 16>& m1_nxt,
    simd<uint32_t, 64>& dx_nxt,
    const uint32_t* w0,
    const uint32_t* w1,
    uint32_t w_nxt_off,
    const int* xp,
    uint32_t xp_nxt_off,
    const uint32_t* dxp,
    uint32_t dxp_nxt_off,
    float* y_base,
    uint32_t kb,
    uint32_t nv) {
    constexpr int u  = S & 1;
    constexpr int js = S & 3;

    if constexpr (S == 0) {
        es::prefetch<uint32_t, 32>(w0 + w_nxt_off + 512u, 0u, kPfL1);
        es::prefetch<uint32_t, 32>(w1 + w_nxt_off + 512u, 0u, kPfL1);
        es::prefetch<uint32_t, 64>(dxp + dxp_nxt_off, 0u, kPfL1);
        es::prefetch<uint32_t, 64>(dxp + dxp_nxt_off + 64u, 0u, kPfL1);
    }

    const int* xp_s = xp + static_cast<size_t>(S) * 256u;
    simd<int, 64> a0 = es::block_load<int, 64>(xp_s);
    simd<int, 64> a1 = es::block_load<int, 64>(xp_s + 64u);
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        i_act[i][0] = xmx::dpas<8, 8, int, int, int, int,
            xmx::dpas_argument_type::u4, xmx::dpas_argument_type::s8>(
            i_act[i][0], wr[u][i], a0);
    }
    a0 = es::block_load<int, 64>(xp_s + 128u);
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        i_act[i][1] = xmx::dpas<8, 8, int, int, int, int,
            xmx::dpas_argument_type::u4, xmx::dpas_argument_type::s8>(
            i_act[i][1], wr[u][i], a1);
    }
    a1 = es::block_load<int, 64>(xp_s + 192u);
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        i_act[i][2] = xmx::dpas<8, 8, int, int, int, int,
            xmx::dpas_argument_type::u4, xmx::dpas_argument_type::s8>(
            i_act[i][2], wr[u][i], a0);
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        i_act[i][3] = xmx::dpas<8, 8, int, int, int, int,
            xmx::dpas_argument_type::u4, xmx::dpas_argument_type::s8>(
            i_act[i][3], wr[u][i], a1);
    }

    if constexpr (S < 6) {
        constexpr uint32_t w_off = static_cast<uint32_t>(S + 2) * 64u;
        wr[u][0] = es::block_load<int, 64>(reinterpret_cast<const int*>(w0 + w_off));
        wr[u][1] = es::block_load<int, 64>(reinterpret_cast<const int*>(w1 + w_off));
    } else {
        constexpr uint32_t w_off = static_cast<uint32_t>(S - 6) * 64u;
        wr[u][0] = es::block_load<int, 64>(reinterpret_cast<const int*>(w0 + w_nxt_off + w_off));
        wr[u][1] = es::block_load<int, 64>(reinterpret_cast<const int*>(w1 + w_nxt_off + w_off));
    }

    const int* xp_pa = (S < 7) ? (xp + static_cast<size_t>(S + 1) * 256u) : (xp + xp_nxt_off);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        es::prefetch<int, 64>(xp_pa + static_cast<uint32_t>(j) * 64u, 0u, kPfL1);
    }

    if constexpr (S < 4) {
        constexpr uint32_t pf_off = static_cast<uint32_t>(S + 4) * 64u;
        es::prefetch<uint32_t, 64>(w0 + pf_off, 0u, kPfL1);
        es::prefetch<uint32_t, 64>(w1 + pf_off, 0u, kPfL1);
    } else {
        constexpr uint32_t pf_off = static_cast<uint32_t>(S - 4) * 64u;
        es::prefetch<uint32_t, 64>(w0 + w_nxt_off + pf_off, 0u, kPfL1);
        es::prefetch<uint32_t, 64>(w1 + w_nxt_off + pf_off, 0u, kPfL1);
    }

    const simd<float, 8> dxs_v = dx_scale.template select<8, 1>(8 * js);
    const simd<int, 8>   sxn_v = dx_nxt.template select<8, 1>(16 * js + 8).template bit_cast_view<int>();
    const simd<int16_t, 16> nz0 = simd<int16_t, 16>(0) - simd<int16_t, 16>(m0_nxt.template bit_cast_view<uint16_t>().template select<16, 2>(1));
    const simd<int16_t, 16> nz1 = simd<int16_t, 16>(0) - simd<int16_t, 16>(m1_nxt.template bit_cast_view<uint16_t>().template select<16, 2>(1));
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        const float d_r           = float(dxs_v[r]);
        const simd<float, 16> sd0 = s_scale[0] * d_r;
        const simd<float, 16> sd1 = s_scale[1] * d_r;
        const simd<float, 16> f0  = simd<float, 16>(i_oth[0][js].template select<16, 1>(16 * r));
        const simd<float, 16> f1  = simd<float, 16>(i_oth[1][js].template select<16, 1>(16 * r));
        c_acc[0][js].template select<16, 1>(16 * r) += f0 * sd0;
        c_acc[1][js].template select<16, 1>(16 * r) += f1 * sd1;
        const int16_t sx_r = static_cast<int16_t>(sxn_v[r]);
        i_oth[0][js].template select<16, 1>(16 * r) = simd<int, 16>(nz0) * static_cast<int>(sx_r);
        i_oth[1][js].template select<16, 1>(16 * r) = simd<int, 16>(nz1) * static_cast<int>(sx_r);
    }

    if constexpr ((S & 1) == 1) {
        if (kb * 8u + static_cast<uint32_t>(S) == nv) y_base[0] = 0.0f;
    }
}

void launch_w4i8_pp3(sycl::queue& q, const uint32_t* w_base, const int* xq8_base,
                     const uint32_t* dxT_base, float* y_base,
                     int K, int R, int T) {
    const uint32_t n_rt      = static_cast<uint32_t>(R) / 16u;
    const uint32_t n_rt32    = (n_rt + 1u) / 2u;
    const uint32_t n_kb      = static_cast<uint32_t>(K) / 256u;
    const uint32_t n_grp     = n_kb * 2u;
    const uint32_t n_steps   = n_kb * 8u;
    const uint32_t n_ct32    = static_cast<uint32_t>(T + 31) / 32u;
    const Pp3Grid  g         = make_pp3_grid(n_rt32, n_ct32, false);
    const uint32_t n_wgc_tot = (n_ct32 + g.CT - 1u) / g.CT;
    const uint32_t nv        = static_cast<uint32_t>(K) | 0x80000000u;
    const uint32_t ct_mask   = g.CT - 1u;

    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagW4I8Pp3>(
            sycl::nd_range<1>(static_cast<size_t>(g.g_wgs) << (g.lgRT + g.lgCT),
                              static_cast<size_t>(1) << (g.lgRT + g.lgCT)),
            props,
            [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
                const uint32_t wg0 = static_cast<uint32_t>(it.get_group(0));
                const uint32_t l   = static_cast<uint32_t>(it.get_local_id(0));
                const uint32_t ri  = l >> g.lgCT;
                const uint32_t ci  = l & ct_mask;

                for (uint32_t w = wg0; w < g.nw; w += g.g_wgs) {
                    barrier();
                    const uint32_t cp   = fast_div(w, g.div_pw);
                    const uint32_t rem  = w - cp * g.panel_wgs;
                    const uint32_t wgr  = fast_div(rem, g.div_ncg);
                    const uint32_t cg   = rem - wgr * g.ncg;
                    const uint32_t wgc  = cp * g.ncg + cg;
                    const uint32_t rt0  = ((wgr << g.lgRT) + ri) * 2u;
                    const uint32_t rt1  = rt0 + 1u;
                    const uint32_t ct32 = (wgc << g.lgCT) + ci;
                    if (wgc < n_wgc_tot && rt0 < n_rt && ct32 < n_ct32) {
                        const uint32_t* w0  = w_base   + static_cast<size_t>(rt0)  * static_cast<size_t>(n_kb) * 544u;
                        const uint32_t* w1  = (rt1 < n_rt) ? (w0 + static_cast<size_t>(n_kb) * 544u) : w0;
                        const int*      xp  = xq8_base + static_cast<size_t>(ct32) * static_cast<size_t>(n_steps) * 256u;
                        const uint32_t* dxp = dxT_base + static_cast<size_t>(ct32) * static_cast<size_t>(n_grp)   * 64u;

                        simd<float, 128> c_acc[2][4];
                        simd<int, 128>   i0[2][4], i1[2][4];
                        #pragma unroll
                        for (int i = 0; i < 2; ++i) {
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                c_acc[i][j] = 0.0f;
                                i1[i][j]    = 0;
                            }
                        }

                        simd<int, 64> wr[2][2];
                        wr[0][0] = es::block_load<int, 64>(reinterpret_cast<const int*>(w0));
                        wr[0][1] = es::block_load<int, 64>(reinterpret_cast<const int*>(w1));
                        wr[1][0] = es::block_load<int, 64>(reinterpret_cast<const int*>(w0 + 64u));
                        wr[1][1] = es::block_load<int, 64>(reinterpret_cast<const int*>(w1 + 64u));

                        #pragma unroll
                        for (int s = 0; s < 2; ++s) {
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                es::prefetch<int, 64>(xp + static_cast<uint32_t>(s) * 256u + static_cast<uint32_t>(j) * 64u, 0u, kPfL1);
                            }
                        }
                        #pragma unroll
                        for (int pf = 2; pf <= 5; ++pf) {
                            es::prefetch<uint32_t, 64>(w0 + static_cast<uint32_t>(pf) * 64u, 0u, kPfL1);
                            es::prefetch<uint32_t, 64>(w1 + static_cast<uint32_t>(pf) * 64u, 0u, kPfL1);
                        }
                        es::prefetch<uint32_t, 32>(w0 + 512u, 0u, kPfL1);
                        es::prefetch<uint32_t, 32>(w1 + 512u, 0u, kPfL1);
                        es::prefetch<uint32_t, 64>(dxp, 0u, kPfL1);
                        es::prefetch<uint32_t, 64>(dxp + 64u, 0u, kPfL1);

                        // Preload g=0 metadata and initialize i0 = -z * sx; prepare (s_dpas, dx_dpas) for g=0.
                        simd<uint32_t, 16> m0_nxt = es::block_load<uint32_t, 16>(w0 + 512u, kAl16);
                        simd<uint32_t, 16> m1_nxt = es::block_load<uint32_t, 16>(w1 + 512u, kAl16);
                        simd<uint32_t, 64> dx_nxt = es::block_load<uint32_t, 64>(dxp, kAl16);

                        simd<sycl::half, 32> s_dpas_h;
                        simd<float, 16> s_scale[2];
                        simd<float, 32> dx_dpas, dx_scale = 0.0f;
                        s_scale[0] = 0.0f;
                        s_scale[1] = 0.0f;
                        s_dpas_h.template select<16, 1>(0)  = half_raw_of(m0_nxt);
                        s_dpas_h.template select<16, 1>(16) = half_raw_of(m1_nxt);
                        {
                            const simd<int16_t, 16> nz0 = simd<int16_t, 16>(0) - simd<int16_t, 16>(m0_nxt.template bit_cast_view<uint16_t>().template select<16, 2>(1));
                            const simd<int16_t, 16> nz1 = simd<int16_t, 16>(0) - simd<int16_t, 16>(m1_nxt.template bit_cast_view<uint16_t>().template select<16, 2>(1));
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                dx_dpas.template select<8, 1>(8 * j) =
                                    dx_nxt.template select<8, 1>(16 * j).template bit_cast_view<float>();
                                const simd<int, 8> sx_v = dx_nxt.template select<8, 1>(16 * j + 8).template bit_cast_view<int>();
                                #pragma unroll
                                for (int r = 0; r < 8; ++r) {
                                    const int16_t sx_r = static_cast<int16_t>(sx_v[r]);
                                    i0[0][j].template select<16, 1>(16 * r) = simd<int, 16>(nz0) * static_cast<int>(sx_r);
                                    i0[1][j].template select<16, 1>(16 * r) = simd<int, 16>(nz1) * static_cast<int>(sx_r);
                                }
                            }
                        }

                        // Preload g=1 metadata asynchronously before the kb loop.
                        m0_nxt = es::block_load<uint32_t, 16>(w0 + 528u, kAl16);
                        m1_nxt = es::block_load<uint32_t, 16>(w1 + 528u, kAl16);
                        dx_nxt = es::block_load<uint32_t, 64>(dxp + 64u, kAl16);

                        #pragma nounroll
                        for (uint32_t kb = 0; kb < n_kb; ++kb) {
                            const bool has_nxt         = (kb + 1u < n_kb);
                            const uint32_t w_nxt_off   = has_nxt ? 544u  : 0u;
                            const uint32_t xp_nxt_off  = has_nxt ? 2048u : 0u;
                            const uint32_t dxp_nxt_off = has_nxt ? 128u  : 0u;

                            pp3_w4_substep<0>(i0, i1, c_acc, wr, s_scale, dx_scale, m0_nxt, m1_nxt, dx_nxt,
                                              w0, w1, w_nxt_off, xp, xp_nxt_off, dxp, dxp_nxt_off, y_base, kb, nv);
                            pp3_w4_substep<1>(i0, i1, c_acc, wr, s_scale, dx_scale, m0_nxt, m1_nxt, dx_nxt,
                                              w0, w1, w_nxt_off, xp, xp_nxt_off, dxp, dxp_nxt_off, y_base, kb, nv);
                            pp3_w4_substep<2>(i0, i1, c_acc, wr, s_scale, dx_scale, m0_nxt, m1_nxt, dx_nxt,
                                              w0, w1, w_nxt_off, xp, xp_nxt_off, dxp, dxp_nxt_off, y_base, kb, nv);
                            pp3_w4_substep<3>(i0, i1, c_acc, wr, s_scale, dx_scale, m0_nxt, m1_nxt, dx_nxt,
                                              w0, w1, w_nxt_off, xp, xp_nxt_off, dxp, dxp_nxt_off, y_base, kb, nv);

                            // Handoff at end of g=2*kb: (s_dpas, dx_dpas) -> scale; m_nxt -> dpas; async load next (kb, g=0).
                            s_scale[0] = simd<float, 16>(s_dpas_h.template select<16, 1>(0));
                            s_scale[1] = simd<float, 16>(s_dpas_h.template select<16, 1>(16));
                            dx_scale   = dx_dpas;
                            s_dpas_h.template select<16, 1>(0)  = half_raw_of(m0_nxt);
                            s_dpas_h.template select<16, 1>(16) = half_raw_of(m1_nxt);
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                dx_dpas.template select<8, 1>(8 * j) =
                                    dx_nxt.template select<8, 1>(16 * j).template bit_cast_view<float>();
                            }
                            m0_nxt = es::block_load<uint32_t, 16>(w0 + w_nxt_off + 512u, kAl16);
                            m1_nxt = es::block_load<uint32_t, 16>(w1 + w_nxt_off + 512u, kAl16);
                            dx_nxt = es::block_load<uint32_t, 64>(dxp + dxp_nxt_off, kAl16);

                            pp3_w4_substep<4>(i1, i0, c_acc, wr, s_scale, dx_scale, m0_nxt, m1_nxt, dx_nxt,
                                              w0, w1, w_nxt_off, xp, xp_nxt_off, dxp, dxp_nxt_off, y_base, kb, nv);
                            pp3_w4_substep<5>(i1, i0, c_acc, wr, s_scale, dx_scale, m0_nxt, m1_nxt, dx_nxt,
                                              w0, w1, w_nxt_off, xp, xp_nxt_off, dxp, dxp_nxt_off, y_base, kb, nv);
                            pp3_w4_substep<6>(i1, i0, c_acc, wr, s_scale, dx_scale, m0_nxt, m1_nxt, dx_nxt,
                                              w0, w1, w_nxt_off, xp, xp_nxt_off, dxp, dxp_nxt_off, y_base, kb, nv);
                            pp3_w4_substep<7>(i1, i0, c_acc, wr, s_scale, dx_scale, m0_nxt, m1_nxt, dx_nxt,
                                              w0, w1, w_nxt_off, xp, xp_nxt_off, dxp, dxp_nxt_off, y_base, kb, nv);

                            // Handoff at end of g=2*kb+1: async load next (kb+1, g=1) if any.
                            s_scale[0] = simd<float, 16>(s_dpas_h.template select<16, 1>(0));
                            s_scale[1] = simd<float, 16>(s_dpas_h.template select<16, 1>(16));
                            dx_scale   = dx_dpas;
                            s_dpas_h.template select<16, 1>(0)  = half_raw_of(m0_nxt);
                            s_dpas_h.template select<16, 1>(16) = half_raw_of(m1_nxt);
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                dx_dpas.template select<8, 1>(8 * j) =
                                    dx_nxt.template select<8, 1>(16 * j).template bit_cast_view<float>();
                            }
                            if (has_nxt) {
                                m0_nxt = es::block_load<uint32_t, 16>(w0 + w_nxt_off + 528u, kAl16);
                                m1_nxt = es::block_load<uint32_t, 16>(w1 + w_nxt_off + 528u, kAl16);
                                dx_nxt = es::block_load<uint32_t, 64>(dxp + dxp_nxt_off + 64u, kAl16);
                            }
                            w0  += 544u;
                            w1  += 544u;
                            xp  += 2048u;
                            dxp += 128u;
                        }

                        // Final scaling epilogue for the last group (g = 2*n_kb - 1 in i1).
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const simd<float, 8> dxs_v = dx_scale.template select<8, 1>(8 * j);
                            #pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                const float d_r           = float(dxs_v[r]);
                                const simd<float, 16> sd0 = s_scale[0] * d_r;
                                const simd<float, 16> sd1 = s_scale[1] * d_r;
                                c_acc[0][j].template select<16, 1>(16 * r) +=
                                    simd<float, 16>(i1[0][j].template select<16, 1>(16 * r)) * sd0;
                                c_acc[1][j].template select<16, 1>(16 * r) +=
                                    simd<float, 16>(i1[1][j].template select<16, 1>(16 * r)) * sd1;
                            }
                        }

                        const uint32_t row0     = rt0 * 16u;
                        const bool has_r1       = (rt1 < n_rt);
                        const uint32_t col_base = ct32 * 32u;
                        if (col_base + 32u <= static_cast<uint32_t>(T) && has_r1) {
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                #pragma unroll
                                for (int r = 0; r < 8; ++r) {
                                    float* yp = y_base + static_cast<size_t>(col_base + static_cast<uint32_t>(j * 8 + r)) * static_cast<size_t>(R) + row0;
                                    simd<float, 32> out32;
                                    out32.template select<16, 1>(0)  = c_acc[0][j].template select<16, 1>(16 * r);
                                    out32.template select<16, 1>(16) = c_acc[1][j].template select<16, 1>(16 * r);
                                    es::block_store<float, 32>(yp, out32, kAl16);
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                #pragma unroll
                                for (int r = 0; r < 8; ++r) {
                                    const uint32_t col = col_base + static_cast<uint32_t>(j * 8 + r);
                                    if (col < static_cast<uint32_t>(T)) {
                                        float* yp = y_base + static_cast<size_t>(col) * static_cast<size_t>(R) + row0;
                                        if (has_r1) {
                                            simd<float, 32> out32;
                                            out32.template select<16, 1>(0)  = c_acc[0][j].template select<16, 1>(16 * r);
                                            out32.template select<16, 1>(16) = c_acc[1][j].template select<16, 1>(16 * r);
                                            es::block_store<float, 32>(yp, out32, kAl16);
                                        } else {
                                            es::block_store<float, 16>(yp, c_acc[0][j].template select<16, 1>(16 * r), kAl16);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            });
    });
}

template <int S>
SYCL_ESIMD_FUNCTION inline void pp3_gu_substep(
    simd<int, 128> (&ig_act)[4],
    simd<int, 128> (&iu_act)[4],
    simd<int, 128> (&ig_oth)[4],
    simd<int, 128> (&iu_oth)[4],
    simd<float, 128> (&cg)[4],
    simd<float, 128> (&cu)[4],
    simd<int, 64> (&wr_g)[2],
    simd<int, 64> (&wr_u)[2],
    simd<float, 16>& sg_scale,
    simd<float, 16>& su_scale,
    simd<float, 32>& dx_scale,
    simd<uint32_t, 16>& mg_nxt,
    simd<uint32_t, 16>& mu_nxt,
    simd<uint32_t, 64>& dx_nxt,
    const uint32_t* wg,
    const uint32_t* wu,
    const int* xp0,
    const uint32_t* dxp0,
    const simd<uint32_t, 4>& x_off_jt,
    const simd<uint32_t, 4>& dx_off_jt,
    bool has_nxt,
    uint32_t* dxT_out,
    uint32_t kb,
    uint32_t nv) {
    constexpr int u  = S & 1;
    constexpr int js = S & 3;
    const uint32_t w_nxt_off   = has_nxt ? 544u  : 0u;
    const uint32_t dxp_nxt_off = has_nxt ? 128u  : 0u;

    if constexpr (S == 0) {
        es::prefetch<uint32_t, 32>(wg + w_nxt_off + 512u, 0u, kPfL1);
        es::prefetch<uint32_t, 32>(wu + w_nxt_off + 512u, 0u, kPfL1);
        #pragma unroll
        for (int jt = 0; jt < 4; ++jt) {
            const uint32_t doff = uint32_t(dx_off_jt[jt]);
            es::prefetch<uint32_t, 16>(dxp0 + doff + dxp_nxt_off, 0u, kPfL1);
            es::prefetch<uint32_t, 16>(dxp0 + doff + dxp_nxt_off + 64u, 0u, kPfL1);
        }
    }

    const int* xp0_s = xp0 + static_cast<size_t>(S) * 256u;
    simd<int, 64> a0 = es::block_load<int, 64>(xp0_s + static_cast<size_t>(uint32_t(x_off_jt[0])));
    simd<int, 64> a1 = es::block_load<int, 64>(xp0_s + static_cast<size_t>(uint32_t(x_off_jt[1])));
    ig_act[0] = xmx::dpas<8, 8, int, int, int, int,
        xmx::dpas_argument_type::u4, xmx::dpas_argument_type::s8>(ig_act[0], wr_g[u], a0);
    iu_act[0] = xmx::dpas<8, 8, int, int, int, int,
        xmx::dpas_argument_type::u4, xmx::dpas_argument_type::s8>(iu_act[0], wr_u[u], a0);
    a0 = es::block_load<int, 64>(xp0_s + static_cast<size_t>(uint32_t(x_off_jt[2])));
    ig_act[1] = xmx::dpas<8, 8, int, int, int, int,
        xmx::dpas_argument_type::u4, xmx::dpas_argument_type::s8>(ig_act[1], wr_g[u], a1);
    iu_act[1] = xmx::dpas<8, 8, int, int, int, int,
        xmx::dpas_argument_type::u4, xmx::dpas_argument_type::s8>(iu_act[1], wr_u[u], a1);
    a1 = es::block_load<int, 64>(xp0_s + static_cast<size_t>(uint32_t(x_off_jt[3])));
    ig_act[2] = xmx::dpas<8, 8, int, int, int, int,
        xmx::dpas_argument_type::u4, xmx::dpas_argument_type::s8>(ig_act[2], wr_g[u], a0);
    iu_act[2] = xmx::dpas<8, 8, int, int, int, int,
        xmx::dpas_argument_type::u4, xmx::dpas_argument_type::s8>(iu_act[2], wr_u[u], a0);
    ig_act[3] = xmx::dpas<8, 8, int, int, int, int,
        xmx::dpas_argument_type::u4, xmx::dpas_argument_type::s8>(ig_act[3], wr_g[u], a1);
    iu_act[3] = xmx::dpas<8, 8, int, int, int, int,
        xmx::dpas_argument_type::u4, xmx::dpas_argument_type::s8>(iu_act[3], wr_u[u], a1);

    if constexpr (S < 6) {
        constexpr uint32_t w_off = static_cast<uint32_t>(S + 2) * 64u;
        wr_g[u] = es::block_load<int, 64>(reinterpret_cast<const int*>(wg + w_off));
        wr_u[u] = es::block_load<int, 64>(reinterpret_cast<const int*>(wu + w_off));
    } else {
        constexpr uint32_t w_off = static_cast<uint32_t>(S - 6) * 64u;
        wr_g[u] = es::block_load<int, 64>(reinterpret_cast<const int*>(wg + w_nxt_off + w_off));
        wr_u[u] = es::block_load<int, 64>(reinterpret_cast<const int*>(wu + w_nxt_off + w_off));
    }

    const size_t x_pa_off = (S < 7) ? (static_cast<size_t>(S + 1) * 256u)
                                    : static_cast<size_t>(has_nxt ? 2048u : 0u);
    #pragma unroll
    for (int jt = 0; jt < 4; ++jt) {
        es::prefetch<int, 64>(xp0 + static_cast<size_t>(uint32_t(x_off_jt[jt])) + x_pa_off, 0u, kPfL1);
    }

    if constexpr (S < 4) {
        constexpr uint32_t pf_off = static_cast<uint32_t>(S + 4) * 64u;
        es::prefetch<uint32_t, 64>(wg + pf_off, 0u, kPfL1);
        es::prefetch<uint32_t, 64>(wu + pf_off, 0u, kPfL1);
    } else {
        constexpr uint32_t pf_off = static_cast<uint32_t>(S - 4) * 64u;
        es::prefetch<uint32_t, 64>(wg + w_nxt_off + pf_off, 0u, kPfL1);
        es::prefetch<uint32_t, 64>(wu + w_nxt_off + pf_off, 0u, kPfL1);
    }

    const simd<float, 8> dxs_v = dx_scale.template select<8, 1>(8 * js);
    const simd<int, 8>   sxn_v = dx_nxt.template select<8, 1>(16 * js + 8).template bit_cast_view<int>();
    const simd<int16_t, 16> nzg = simd<int16_t, 16>(0) - simd<int16_t, 16>(mg_nxt.template bit_cast_view<uint16_t>().template select<16, 2>(1));
    const simd<int16_t, 16> nzu = simd<int16_t, 16>(0) - simd<int16_t, 16>(mu_nxt.template bit_cast_view<uint16_t>().template select<16, 2>(1));
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        const float d_r           = float(dxs_v[r]);
        const simd<float, 16> sdg = sg_scale * d_r;
        const simd<float, 16> sdu = su_scale * d_r;
        const simd<float, 16> fg  = simd<float, 16>(ig_oth[js].template select<16, 1>(16 * r));
        const simd<float, 16> fu  = simd<float, 16>(iu_oth[js].template select<16, 1>(16 * r));
        cg[js].template select<16, 1>(16 * r) += fg * sdg;
        cu[js].template select<16, 1>(16 * r) += fu * sdu;
        const int16_t sx_r = static_cast<int16_t>(sxn_v[r]);
        ig_oth[js].template select<16, 1>(16 * r) = simd<int, 16>(nzg) * static_cast<int>(sx_r);
        iu_oth[js].template select<16, 1>(16 * r) = simd<int, 16>(nzu) * static_cast<int>(sx_r);
    }

    if constexpr ((S & 1) == 1) {
        if (kb * 8u + static_cast<uint32_t>(S) == nv) dxT_out[0] = 0u;
    }
}

void launch_w4i8_gu_swiglu_pp3(sycl::queue& q, const uint32_t* w_gate, const uint32_t* w_up,
                               const int* xq8_in, const uint32_t* dxT_in,
                               uint32_t* xq8_out, uint32_t* dxT_out,
                               int K, int R, int T) {
    const uint32_t n_kb      = static_cast<uint32_t>(K) / 256u;
    const uint32_t n_grp     = n_kb * 2u;
    const uint32_t n_steps   = n_kb * 8u;
    const uint32_t n_g_out   = static_cast<uint32_t>(R) / 128u;
    const uint32_t n_ct32    = static_cast<uint32_t>(T + 31) / 32u;
    const uint32_t n_wgc_tot = (n_ct32 + 3u) / 4u;
    const uint32_t max_wgs   = std::max<uint32_t>(1u, env_pp3_max_thr() / 32u);
    const uint32_t ncg_t     = std::max<uint32_t>(1u, env_pp3_panel() / 128u);   // WG-cols of 128 cols; default 8 = 1024 cols per panel
    uint32_t ncg, panel_wgs, nw;
    if (n_wgc_tot <= ncg_t) {
        ncg       = std::max<uint32_t>(1u, n_wgc_tot);
        panel_wgs = std::max<uint32_t>(1u, n_g_out * ncg);
        nw        = panel_wgs;
    } else {
        ncg       = ncg_t;
        const uint32_t n_cp = (n_wgc_tot + ncg - 1u) / ncg;
        panel_wgs = n_g_out * ncg;
        nw        = n_cp * panel_wgs;
    }
    const RecipDiv div_pw    = make_recip(panel_wgs);
    const RecipDiv div_ncg   = make_recip(ncg);
    const uint32_t g_wgs     = std::min<uint32_t>(nw, max_wgs);
    const uint32_t nv        = static_cast<uint32_t>(K) | 0x80000000u;

    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagW4I8GuSwigluPp3>(
            sycl::nd_range<1>(static_cast<size_t>(g_wgs) * 32u, 32u), props,
            [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
                slm_init<17408u>();
                const uint32_t wg0 = static_cast<uint32_t>(it.get_group(0));
                const uint32_t lid = static_cast<uint32_t>(it.get_local_id(0));
                const uint32_t ri  = lid >> 2;
                const uint32_t ci  = lid & 3u;

                for (uint32_t w = wg0; w < nw; w += g_wgs) {
                    barrier();
                    const uint32_t cp   = fast_div(w, div_pw);
                    const uint32_t rem  = w - cp * panel_wgs;
                    const uint32_t wgr  = fast_div(rem, div_ncg);
                    const uint32_t cg   = rem - wgr * ncg;
                    const uint32_t wgc  = cp * ncg + cg;
                    if (wgc >= n_wgc_tot) continue;

                    const uint32_t rt       = wgr * 8u + ri;
                    const uint32_t ct32_b   = wgc * 4u;
                    const size_t w_off_tile = static_cast<size_t>(rt) * static_cast<size_t>(n_kb) * 544u;
                    const uint32_t* wg      = w_gate + w_off_tile;
                    const uint32_t* wu      = w_up   + w_off_tile;

                    const int*      xp0     = xq8_in + static_cast<size_t>(ct32_b) * static_cast<size_t>(n_steps) * 256u + ci * 64u;
                    const uint32_t* dxp0    = dxT_in + static_cast<size_t>(ct32_b) * static_cast<size_t>(n_grp)   * 64u  + ci * 16u;
                    const uint32_t x_cstride  = n_steps * 256u;
                    const uint32_t dx_cstride = n_grp * 64u;
                    simd<uint32_t, 4> x_off_jt, dx_off_jt;
                    #pragma unroll
                    for (int jt = 0; jt < 4; ++jt) {
                        const uint32_t k = (ct32_b + static_cast<uint32_t>(jt) < n_ct32)
                                         ? static_cast<uint32_t>(jt)
                                         : (n_ct32 - 1u - ct32_b);
                        x_off_jt[jt]  = k * x_cstride;
                        dx_off_jt[jt] = k * dx_cstride;
                    }

                    simd<float, 128> cg_acc[4], cu_acc[4];
                    simd<int, 128>   ig0[4], iu0[4], ig1[4], iu1[4];
                    #pragma unroll
                    for (int jt = 0; jt < 4; ++jt) {
                        cg_acc[jt] = 0.0f; cu_acc[jt] = 0.0f;
                        ig1[jt]    = 0;    iu1[jt]    = 0;
                    }

                    simd<int, 64> wr_g[2], wr_u[2];
                    wr_g[0] = es::block_load<int, 64>(reinterpret_cast<const int*>(wg));
                    wr_u[0] = es::block_load<int, 64>(reinterpret_cast<const int*>(wu));
                    wr_g[1] = es::block_load<int, 64>(reinterpret_cast<const int*>(wg + 64u));
                    wr_u[1] = es::block_load<int, 64>(reinterpret_cast<const int*>(wu + 64u));

                    #pragma unroll
                    for (int s = 0; s < 2; ++s) {
                        #pragma unroll
                        for (int jt = 0; jt < 4; ++jt) {
                            es::prefetch<int, 64>(xp0 + static_cast<size_t>(uint32_t(x_off_jt[jt])) + static_cast<size_t>(s) * 256u, 0u, kPfL1);
                        }
                    }
                    #pragma unroll
                    for (int pf = 2; pf <= 5; ++pf) {
                        es::prefetch<uint32_t, 64>(wg + static_cast<uint32_t>(pf) * 64u, 0u, kPfL1);
                        es::prefetch<uint32_t, 64>(wu + static_cast<uint32_t>(pf) * 64u, 0u, kPfL1);
                    }
                    es::prefetch<uint32_t, 32>(wg + 512u, 0u, kPfL1);
                    es::prefetch<uint32_t, 32>(wu + 512u, 0u, kPfL1);
                    #pragma unroll
                    for (int jt = 0; jt < 4; ++jt) {
                        const uint32_t doff = uint32_t(dx_off_jt[jt]);
                        es::prefetch<uint32_t, 16>(dxp0 + doff, 0u, kPfL1);
                        es::prefetch<uint32_t, 16>(dxp0 + doff + 64u, 0u, kPfL1);
                    }

                    simd<uint32_t, 16> mg_nxt = es::block_load<uint32_t, 16>(wg + 512u, kAl16);
                    simd<uint32_t, 16> mu_nxt = es::block_load<uint32_t, 16>(wu + 512u, kAl16);
                    simd<uint32_t, 64> dx_nxt;
                    #pragma unroll
                    for (int jt = 0; jt < 4; ++jt) {
                        dx_nxt.template select<16, 1>(16 * jt) = es::block_load<uint32_t, 16>(
                            dxp0 + static_cast<size_t>(uint32_t(dx_off_jt[jt])), kAl16);
                    }

                    simd<sycl::half, 32> sgu_dpas_h;
                    sgu_dpas_h.template select<16, 1>(0)  = half_raw_of(mg_nxt);
                    sgu_dpas_h.template select<16, 1>(16) = half_raw_of(mu_nxt);
                    simd<float, 16> sg_scale = 0.0f, su_scale = 0.0f;
                    simd<float, 32> dx_dpas, dx_scale = 0.0f;
                    {
                        const simd<int16_t, 16> nzg = simd<int16_t, 16>(0) - simd<int16_t, 16>(mg_nxt.template bit_cast_view<uint16_t>().template select<16, 2>(1));
                        const simd<int16_t, 16> nzu = simd<int16_t, 16>(0) - simd<int16_t, 16>(mu_nxt.template bit_cast_view<uint16_t>().template select<16, 2>(1));
                        #pragma unroll
                        for (int jt = 0; jt < 4; ++jt) {
                            dx_dpas.template select<8, 1>(8 * jt) =
                                dx_nxt.template select<8, 1>(16 * jt).template bit_cast_view<float>();
                            const simd<int, 8> sx_v = dx_nxt.template select<8, 1>(16 * jt + 8).template bit_cast_view<int>();
                            #pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                const int16_t sx_r = static_cast<int16_t>(sx_v[r]);
                                ig0[jt].template select<16, 1>(16 * r) = simd<int, 16>(nzg) * static_cast<int>(sx_r);
                                iu0[jt].template select<16, 1>(16 * r) = simd<int, 16>(nzu) * static_cast<int>(sx_r);
                            }
                        }
                    }

                    mg_nxt = es::block_load<uint32_t, 16>(wg + 528u, kAl16);
                    mu_nxt = es::block_load<uint32_t, 16>(wu + 528u, kAl16);
                    #pragma unroll
                    for (int jt = 0; jt < 4; ++jt) {
                        dx_nxt.template select<16, 1>(16 * jt) = es::block_load<uint32_t, 16>(
                            dxp0 + static_cast<size_t>(uint32_t(dx_off_jt[jt])) + 64u, kAl16);
                    }

                    #pragma nounroll
                    for (uint32_t kb = 0; kb < n_kb; ++kb) {
                        const bool has_nxt         = (kb + 1u < n_kb);
                        const uint32_t w_nxt_off   = has_nxt ? 544u : 0u;
                        const uint32_t dxp_nxt_off = has_nxt ? 128u : 0u;

                        pp3_gu_substep<0>(ig0, iu0, ig1, iu1, cg_acc, cu_acc, wr_g, wr_u, sg_scale, su_scale, dx_scale,
                                          mg_nxt, mu_nxt, dx_nxt, wg, wu, xp0, dxp0, x_off_jt, dx_off_jt, has_nxt, dxT_out, kb, nv);
                        pp3_gu_substep<1>(ig0, iu0, ig1, iu1, cg_acc, cu_acc, wr_g, wr_u, sg_scale, su_scale, dx_scale,
                                          mg_nxt, mu_nxt, dx_nxt, wg, wu, xp0, dxp0, x_off_jt, dx_off_jt, has_nxt, dxT_out, kb, nv);
                        pp3_gu_substep<2>(ig0, iu0, ig1, iu1, cg_acc, cu_acc, wr_g, wr_u, sg_scale, su_scale, dx_scale,
                                          mg_nxt, mu_nxt, dx_nxt, wg, wu, xp0, dxp0, x_off_jt, dx_off_jt, has_nxt, dxT_out, kb, nv);
                        pp3_gu_substep<3>(ig0, iu0, ig1, iu1, cg_acc, cu_acc, wr_g, wr_u, sg_scale, su_scale, dx_scale,
                                          mg_nxt, mu_nxt, dx_nxt, wg, wu, xp0, dxp0, x_off_jt, dx_off_jt, has_nxt, dxT_out, kb, nv);

                        sg_scale = simd<float, 16>(sgu_dpas_h.template select<16, 1>(0));
                        su_scale = simd<float, 16>(sgu_dpas_h.template select<16, 1>(16));
                        dx_scale = dx_dpas;
                        sgu_dpas_h.template select<16, 1>(0)  = half_raw_of(mg_nxt);
                        sgu_dpas_h.template select<16, 1>(16) = half_raw_of(mu_nxt);
                        #pragma unroll
                        for (int jt = 0; jt < 4; ++jt) {
                            dx_dpas.template select<8, 1>(8 * jt) =
                                dx_nxt.template select<8, 1>(16 * jt).template bit_cast_view<float>();
                        }
                        mg_nxt = es::block_load<uint32_t, 16>(wg + w_nxt_off + 512u, kAl16);
                        mu_nxt = es::block_load<uint32_t, 16>(wu + w_nxt_off + 512u, kAl16);
                        #pragma unroll
                        for (int jt = 0; jt < 4; ++jt) {
                            dx_nxt.template select<16, 1>(16 * jt) = es::block_load<uint32_t, 16>(
                                dxp0 + static_cast<size_t>(uint32_t(dx_off_jt[jt])) + dxp_nxt_off, kAl16);
                        }

                        pp3_gu_substep<4>(ig1, iu1, ig0, iu0, cg_acc, cu_acc, wr_g, wr_u, sg_scale, su_scale, dx_scale,
                                          mg_nxt, mu_nxt, dx_nxt, wg, wu, xp0, dxp0, x_off_jt, dx_off_jt, has_nxt, dxT_out, kb, nv);
                        pp3_gu_substep<5>(ig1, iu1, ig0, iu0, cg_acc, cu_acc, wr_g, wr_u, sg_scale, su_scale, dx_scale,
                                          mg_nxt, mu_nxt, dx_nxt, wg, wu, xp0, dxp0, x_off_jt, dx_off_jt, has_nxt, dxT_out, kb, nv);
                        pp3_gu_substep<6>(ig1, iu1, ig0, iu0, cg_acc, cu_acc, wr_g, wr_u, sg_scale, su_scale, dx_scale,
                                          mg_nxt, mu_nxt, dx_nxt, wg, wu, xp0, dxp0, x_off_jt, dx_off_jt, has_nxt, dxT_out, kb, nv);
                        pp3_gu_substep<7>(ig1, iu1, ig0, iu0, cg_acc, cu_acc, wr_g, wr_u, sg_scale, su_scale, dx_scale,
                                          mg_nxt, mu_nxt, dx_nxt, wg, wu, xp0, dxp0, x_off_jt, dx_off_jt, has_nxt, dxT_out, kb, nv);

                        sg_scale = simd<float, 16>(sgu_dpas_h.template select<16, 1>(0));
                        su_scale = simd<float, 16>(sgu_dpas_h.template select<16, 1>(16));
                        dx_scale = dx_dpas;
                        sgu_dpas_h.template select<16, 1>(0)  = half_raw_of(mg_nxt);
                        sgu_dpas_h.template select<16, 1>(16) = half_raw_of(mu_nxt);
                        #pragma unroll
                        for (int jt = 0; jt < 4; ++jt) {
                            dx_dpas.template select<8, 1>(8 * jt) =
                                dx_nxt.template select<8, 1>(16 * jt).template bit_cast_view<float>();
                        }
                        if (has_nxt) {
                            mg_nxt = es::block_load<uint32_t, 16>(wg + w_nxt_off + 528u, kAl16);
                            mu_nxt = es::block_load<uint32_t, 16>(wu + w_nxt_off + 528u, kAl16);
                            #pragma unroll
                            for (int jt = 0; jt < 4; ++jt) {
                                dx_nxt.template select<16, 1>(16 * jt) = es::block_load<uint32_t, 16>(
                                    dxp0 + static_cast<size_t>(uint32_t(dx_off_jt[jt])) + 192u, kAl16);
                            }
                        }
                        wg   += 544u;
                        wu   += 544u;
                        xp0  += 2048u;
                        dxp0 += 128u;
                    }

                    #pragma unroll
                    for (int jt = 0; jt < 4; ++jt) {
                        const simd<float, 8> dxs_v = dx_scale.template select<8, 1>(8 * jt);
                        #pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const float d_r           = float(dxs_v[r]);
                            const simd<float, 16> sdg = sg_scale * d_r;
                            const simd<float, 16> sdu = su_scale * d_r;
                            cg_acc[jt].template select<16, 1>(16 * r) +=
                                simd<float, 16>(ig1[jt].template select<16, 1>(16 * r)) * sdg;
                            cu_acc[jt].template select<16, 1>(16 * r) +=
                                simd<float, 16>(iu1[jt].template select<16, 1>(16 * r)) * sdu;
                        }
                    }

                    pp_util::gu_swiglu_epilogue(cg_acc, cu_acc, xq8_out, dxT_out, lid, ri, ci, wgr, wgc, n_g_out, n_ct32,
                                                static_cast<uint32_t>(T));
                }
            });
    });
}

}  // namespace
}  // namespace pp3

bool gemm_w4_i8_pp2(const void* weights, const void* xq8, const float* dxT, float* y,
                    int n_in, int n_out, int T, void* stream) {
    if (n_in <= 0 || n_out <= 0 || T <= 0 || (n_in % 256) != 0 || (n_out % 16) != 0) return false;
    if (!weights || !xq8 || !dxT || !y) return false;
    if ((reinterpret_cast<uintptr_t>(weights) & 15u) != 0u) return false;
    if ((reinterpret_cast<uintptr_t>(xq8) & 15u) != 0u) return false;
    if ((reinterpret_cast<uintptr_t>(dxT) & 15u) != 0u) return false;
    if ((reinterpret_cast<uintptr_t>(y) & 15u) != 0u) return false;
    pp2::launch_w4i8_pp2(*strata::q_of(stream),
                         static_cast<const uint32_t*>(weights),
                         static_cast<const int*>(xq8),
                         reinterpret_cast<const uint32_t*>(dxT),
                         y, n_in, n_out, T);
    return true;
}

bool gemm_w4_i8_gu_swiglu_pp2(const void* w_gate, const void* w_up,
                              const void* xq8_in, const float* dxT_in,
                              void* xq8_out, float* dxT_out,
                              int n_in, int n_out, int T, void* stream) {
    if (n_in <= 0 || n_out <= 0 || T <= 0 || (n_in % 256) != 0 || (n_out % 128) != 0) return false;
    if (!w_gate || !w_up || !xq8_in || !dxT_in || !xq8_out || !dxT_out) return false;
    if ((reinterpret_cast<uintptr_t>(w_gate)  & 15u) != 0u ||
        (reinterpret_cast<uintptr_t>(w_up)    & 15u) != 0u ||
        (reinterpret_cast<uintptr_t>(xq8_in)  & 15u) != 0u ||
        (reinterpret_cast<uintptr_t>(dxT_in)  & 15u) != 0u ||
        (reinterpret_cast<uintptr_t>(xq8_out) & 15u) != 0u ||
        (reinterpret_cast<uintptr_t>(dxT_out) & 15u) != 0u) return false;
    pp2::launch_w4i8_gu_swiglu_pp2(*strata::q_of(stream),
                                   static_cast<const uint32_t*>(w_gate),
                                   static_cast<const uint32_t*>(w_up),
                                   static_cast<const int*>(xq8_in),
                                   reinterpret_cast<const uint32_t*>(dxT_in),
                                   static_cast<uint32_t*>(xq8_out),
                                   reinterpret_cast<uint32_t*>(dxT_out),
                                   n_in, n_out, T);
    return true;
}

bool pp3_enabled() noexcept {
    static const bool on = []() {
        // default on since 2026-11-14 (card: 26K prompt read 1,815 -> 2,228 tok/s, 118K 1,320 -> 1,534; texts unchanged); 0 = off
        const char* e = std::getenv("STRATA_Q36_PP3");
        return !(e && std::atoi(e) == 0);
    }();
    return on;
}

bool gemm_w4_i8_pp3(const void* weights, const void* xq8, const float* dxT, float* y,
                    int n_in, int n_out, int T, void* stream) {
    if (n_in <= 0 || n_out <= 0 || T <= 0 || (n_in % 256) != 0 || (n_out % 16) != 0) return false;
    if (!weights || !xq8 || !dxT || !y) return false;
    if ((reinterpret_cast<uintptr_t>(weights) & 15u) != 0u) return false;
    if ((reinterpret_cast<uintptr_t>(xq8) & 15u) != 0u) return false;
    if ((reinterpret_cast<uintptr_t>(dxT) & 15u) != 0u) return false;
    if ((reinterpret_cast<uintptr_t>(y) & 15u) != 0u) return false;
    pp3::launch_w4i8_pp3(*strata::q_of(stream),
                         static_cast<const uint32_t*>(weights),
                         static_cast<const int*>(xq8),
                         reinterpret_cast<const uint32_t*>(dxT),
                         y, n_in, n_out, T);
    return true;
}

bool gemm_w4_i8_gu_swiglu_pp3(const void* w_gate, const void* w_up,
                              const void* xq8_in, const float* dxT_in,
                              void* xq8_out, float* dxT_out,
                              int n_in, int n_out, int T, void* stream) {
    if (n_in <= 0 || n_out <= 0 || T <= 0 || (n_in % 256) != 0 || (n_out % 128) != 0) return false;
    if (!w_gate || !w_up || !xq8_in || !dxT_in || !xq8_out || !dxT_out) return false;
    if ((reinterpret_cast<uintptr_t>(w_gate)  & 15u) != 0u ||
        (reinterpret_cast<uintptr_t>(w_up)    & 15u) != 0u ||
        (reinterpret_cast<uintptr_t>(xq8_in)  & 15u) != 0u ||
        (reinterpret_cast<uintptr_t>(dxT_in)  & 15u) != 0u ||
        (reinterpret_cast<uintptr_t>(xq8_out) & 15u) != 0u ||
        (reinterpret_cast<uintptr_t>(dxT_out) & 15u) != 0u) return false;
    pp3::launch_w4i8_gu_swiglu_pp3(*strata::q_of(stream),
                                   static_cast<const uint32_t*>(w_gate),
                                   static_cast<const uint32_t*>(w_up),
                                   static_cast<const int*>(xq8_in),
                                   reinterpret_cast<const uint32_t*>(dxT_in),
                                   static_cast<uint32_t*>(xq8_out),
                                   reinterpret_cast<uint32_t*>(dxT_out),
                                   n_in, n_out, T);
    return true;
}

}  // namespace strata::q36pp
