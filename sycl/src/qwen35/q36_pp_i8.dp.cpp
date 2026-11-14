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

}  // namespace strata::q36pp
