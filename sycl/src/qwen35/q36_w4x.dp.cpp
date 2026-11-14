// q36_w4x.dp.cpp - Card-up native u4 x s8 decode/verify engine for Qwen3.8-27B on Xe2 (Arc Pro B70).
//
// Reads B70W4 records directly (544 uint32_t = 2,176 B per 16 rows x 256 inputs) without unpacking weight nibbles
// when activation scales use GS = 128, and with register-only nibble de-interleave + group-factored zero subtraction
// when activation scales use GS = 32:
//   - GS = 128: quantizer writes paired [x[64p+0..31], x[64p+32..63]] int8 bytes (matches B70W4 lo/hi nibble layout)
//               plus per-group sa[8] and int32 sumx_int[8]; DPAS receives B verbatim (0 unpack ops).
//   - GS = 32:  quantizer writes sequential 32 int8 bytes per sub-block + sa[32] + pre-weighted sx_g[8] per 128-group;
//               kernel de-interleaves lo/hi nibbles with uint16 regioning (no llvm.fshl) and factors z * sx_g once/128.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <utility>

#include "q36_xr_dense.hpp"

namespace strata::q36xr {
namespace {

using namespace sycl::ext::intel::esimd;
namespace es = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

constexpr int kW4Rec = 544;   // 512 u32 nibbles + 32 u32 (fp16 scale | u16 zero)
constexpr auto kAl16 = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::alignment<16>};
constexpr auto kPfL1 = sycl::ext::oneapi::experimental::properties{
    cache_hint_L1<cache_hint::cached>, cache_hint_L2<cache_hint::cached>};

constexpr size_t act_group_stride(int gs) { return gs == 32 ? 1216u : 1088u; }

template <int GS, int MODE> struct TagW4xQuant {};
template <int NC_SLOT, int KS, int GS> struct TagW4xGemv {};
template <int NC_SLOT> struct TagW4xGuSwiglu {};

template <int I0, int N, typename Fn>
SYCL_ESIMD_FUNCTION __attribute__((always_inline)) inline void for_ce(Fn&& fn) {
    if constexpr (I0 < N) {
        [&]<int... Is>(std::integer_sequence<int, Is...>) SYCL_ESIMD_FUNCTION {
            (fn(std::integral_constant<int, I0 + Is>{}), ...);
        }(std::make_integer_sequence<int, N - I0>{});
    }
}

SYCL_ESIMD_FUNCTION inline float hmax2(float a, float b) { return a > b ? a : b; }

SYCL_ESIMD_FUNCTION inline simd<float, 16> half_of(simd<uint32_t, 16> h16) {
    simd<uint16_t, 16> lo = h16.template bit_cast_view<uint16_t>().template select<16, 2>(0);
    simd<sycl::half, 16> hh = lo.template bit_cast_view<sycl::half>();
    return simd<float, 16>(hh);
}

SYCL_ESIMD_FUNCTION inline simd<int, 16> join16(simd<uint32_t, 16> lo_u, simd<uint32_t, 16> lo_v) {
    simd<uint16_t, 32> out;
    out.template select<16, 2>(0) = lo_u.template bit_cast_view<uint16_t>().template select<16, 2>(0);
    out.template select<16, 2>(1) = lo_v.template bit_cast_view<uint16_t>().template select<16, 2>(0);
    return simd<int, 16>(out.template bit_cast_view<int>());
}

SYCL_ESIMD_FUNCTION inline void prefetch_rec(const uint32_t* p) {
    es::prefetch<uint32_t, 16>(p,       simd<uint32_t, 16>(0u, 64u), kPfL1);
    es::prefetch<uint32_t, 16>(p + 256, simd<uint32_t, 16>(0u, 64u), kPfL1);
    es::prefetch<uint32_t,  2>(p + 512, simd<uint32_t,  2>(0u, 64u), kPfL1);
}

template <int REC_STRIDE = kW4Rec>
SYCL_ESIMD_FUNCTION inline void prefetch_g128(const uint32_t* w_tile, int g_pf) {
    const uint32_t* rec_pf = w_tile + (size_t) (g_pf >> 1) * (size_t) REC_STRIDE;
    const int half = g_pf & 1;
    es::prefetch<uint32_t, 16>(rec_pf + 256 * half, simd<uint32_t, 16>(0u, 64u), kPfL1);
    es::prefetch<uint32_t,  1>(rec_pf + 512 + 16 * half, 0u, kPfL1);
}

SYCL_ESIMD_FUNCTION inline float hmax16_f(simd<float, 16> v) {
    simd<float, 8> b = (es::max)(simd<float, 8>(v.template select<8, 1>(0)),
                                 simd<float, 8>(v.template select<8, 1>(8)));
    simd<float, 4> c = (es::max)(simd<float, 4>(b.template select<4, 1>(0)),
                                 simd<float, 4>(b.template select<4, 1>(4)));
    simd<float, 2> d = (es::max)(simd<float, 2>(c.template select<2, 1>(0)),
                                 simd<float, 2>(c.template select<2, 1>(2)));
    return hmax2(float(d[0]), float(d[1]));
}

SYCL_ESIMD_FUNCTION inline simd<int, 16> quant_round16(simd<float, 16> x, float inv_sc) {
    simd<float, 16> y = x * inv_sc;
    simd<float, 16> rnd(0.5f);
    rnd.merge(simd<float, 16>(-0.5f), y < 0.0f);
    return (es::min)((es::max)(convert<int>(y + rnd), simd<int, 16>(-127)), simd<int, 16>(127));
}

SYCL_ESIMD_FUNCTION inline int hsum16_i(simd<int, 16> v) {
    simd<int, 8> b = v.template select<8, 1>(0) + v.template select<8, 1>(8);
    simd<int, 4> c = b.template select<4, 1>(0) + b.template select<4, 1>(4);
    simd<int, 2> d = c.template select<2, 1>(0) + c.template select<2, 1>(2);
    return int(d[0]) + int(d[1]);
}

inline int q_clip127(float v, float id) {
    const int q = (int) sycl::round(v * id);
    return sycl::min(sycl::max(q, -127), 127);
}

// MODE 0: plain x -> act; MODE 1: x_res * inv_rms * norm_w -> act (+ optional h_out); MODE 2: silu(g) * u -> act.
template <int GS, int MODE>
void launch_quantize(sycl::queue& q, const float* x0, int ldx, const float* p1, const float* p2,
                     void* act, float* h_out, int ld_out, int n_in, int ncols) {
    const int n_g128 = n_in / 128;
    const int ng = (ncols + 7) / 8;
    const int col_slots = ng * 8;
    const size_t nwg = (size_t) n_g128 * (size_t) col_slots;
    uint8_t* act_b = reinterpret_cast<uint8_t*>(act);

    q.submit([&](sycl::handler& h) {
        sycl::nd_range<1> nd(sycl::range<1>(nwg * 32u), sycl::range<1>(32u));
        h.parallel_for<TagW4xQuant<GS, MODE>>(nd, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int wg = (int) it.get_group(0);
            const int tid = (int) it.get_local_id(0);
            const auto sg = it.get_sub_group();
            const int col = wg / n_g128;
            const int g = wg - col * n_g128;
            const int gi = col >> 3;
            const int r = col & 7;
            uint32_t* dst = reinterpret_cast<uint32_t*>(act_b + ((size_t) g * ng + gi) * act_group_stride(GS));

            if (col >= ncols) {
                if constexpr (GS == 128) {
                    const int p = tid >> 4;
                    const int hhalf = (tid >> 3) & 1;
                    const int da = tid & 7;
                    dst[(p * 2 + hhalf) * 64 + r * 8 + da] = 0u;
                    if (tid == 0) {
                        reinterpret_cast<float*>(dst + 256)[r] = 0.0f;
                        reinterpret_cast<int32_t*>(dst + 264)[r] = 0;
                    }
                } else {
                    const int sb = tid >> 3;
                    const int da = tid & 7;
                    dst[sb * 64 + r * 8 + da] = 0u;
                    if (da == 0) reinterpret_cast<float*>(dst + 256)[sb * 8 + r] = 0.0f;
                    if (tid == 0) reinterpret_cast<float*>(dst + 288)[r] = 0.0f;
                }
                return;
            }

            auto load_elem = [&](int k) -> float {
                if constexpr (MODE == 0) {
                    return x0[(size_t) col * ldx + k];
                } else if constexpr (MODE == 1) {
                    const float v = x0[(size_t) col * ldx + k] * p2[col] * p1[k];
                    if (h_out) h_out[(size_t) col * ld_out + k] = v;
                    return v;
                } else {
                    const float gv = x0[(size_t) col * ldx + k];
                    const float uv = p1[(size_t) col * ldx + k];
                    return (gv / (1.0f + sycl::exp(sycl::fmin(-gv, 88.0f)))) * uv;
                }
            };

            if constexpr (GS == 128) {
                const int p = tid >> 4;
                const int hhalf = (tid >> 3) & 1;
                const int da = tid & 7;
                const int k_even = g * 128 + p * 64 + hhalf * 16 + da * 2;
                const int k_odd = k_even + 32;
                const float e0 = load_elem(k_even + 0);
                const float e1 = load_elem(k_even + 1);
                const float o0 = load_elem(k_odd + 0);
                const float o1 = load_elem(k_odd + 1);
                const float lamax = sycl::fmax(sycl::fmax(sycl::fabs(e0), sycl::fabs(e1)),
                                               sycl::fmax(sycl::fabs(o0), sycl::fabs(o1)));
                const float amax = sycl::reduce_over_group(sg, lamax, sycl::maximum<float>());
                const float d = amax / 127.0f;
                const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                const int qe0 = q_clip127(e0, id);
                const int qe1 = q_clip127(e1, id);
                const int qo0 = q_clip127(o0, id);
                const int qo1 = q_clip127(o1, id);
                const int sumx = sycl::reduce_over_group(sg, qe0 + qe1 + qo0 + qo1, sycl::plus<int>());
                const uint32_t pk = (uint32_t) (uint8_t) (int8_t) qe0
                                  | ((uint32_t) (uint8_t) (int8_t) qo0 << 8)
                                  | ((uint32_t) (uint8_t) (int8_t) qe1 << 16)
                                  | ((uint32_t) (uint8_t) (int8_t) qo1 << 24);
                dst[(p * 2 + hhalf) * 64 + r * 8 + da] = pk;
                if (tid == 0) {
                    reinterpret_cast<float*>(dst + 256)[r] = d;
                    reinterpret_cast<int32_t*>(dst + 264)[r] = sumx;
                }
            } else {
                const int sb = tid >> 3;
                const int da = tid & 7;
                const int k0 = g * 128 + sb * 32 + da * 4;
                const float v0 = load_elem(k0 + 0);
                const float v1 = load_elem(k0 + 1);
                const float v2 = load_elem(k0 + 2);
                const float v3 = load_elem(k0 + 3);
                float amax = sycl::fmax(sycl::fmax(sycl::fabs(v0), sycl::fabs(v1)),
                                        sycl::fmax(sycl::fabs(v2), sycl::fabs(v3)));
                amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 1u));
                amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 2u));
                amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 4u));
                const float d = amax / 127.0f;
                const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                const int q0 = q_clip127(v0, id);
                const int q1 = q_clip127(v1, id);
                const int q2 = q_clip127(v2, id);
                const int q3 = q_clip127(v3, id);
                int sum_sb = q0 + q1 + q2 + q3;
                sum_sb += sycl::permute_group_by_xor(sg, sum_sb, 1u);
                sum_sb += sycl::permute_group_by_xor(sg, sum_sb, 2u);
                sum_sb += sycl::permute_group_by_xor(sg, sum_sb, 4u);
                const float sx_sb = (da == 0) ? ((float) sum_sb * d) : 0.0f;
                const float sx_g = sycl::reduce_over_group(sg, sx_sb, sycl::plus<float>());
                const uint32_t pk = (uint32_t) (uint8_t) (int8_t) q0
                                  | ((uint32_t) (uint8_t) (int8_t) q1 << 8)
                                  | ((uint32_t) (uint8_t) (int8_t) q2 << 16)
                                  | ((uint32_t) (uint8_t) (int8_t) q3 << 24);
                dst[sb * 64 + r * 8 + da] = pk;
                if (da == 0) reinterpret_cast<float*>(dst + 256)[sb * 8 + r] = d;
                if (tid == 0) reinterpret_cast<float*>(dst + 288)[r] = sx_g;
            }
        });
    });
}

template <int KS>
constexpr int ks_row_tiles() {
    if constexpr (KS == 2)  return 8;
    if constexpr (KS == 3)  return 8;
    if constexpr (KS == 4)  return 4;
    if constexpr (KS == 5)  return 4;
    if constexpr (KS == 6)  return 4;
    if constexpr (KS == 10) return 2;
    return 1;
}

template <int NC_SLOT, int KS, int GS>
void launch_gemv(sycl::queue& q, const void* weights, const void* act, float* y_out,
                 int n_in, int n_out, int ncols) {
    constexpr int NG = (NC_SLOT + 7) / 8;
    constexpr int RT = ks_row_tiles<KS>();
    constexpr int RED_COLS = (NC_SLOT <= 16) ? NC_SLOT : 16;
    constexpr int RED_ROUNDS = (NC_SLOT + RED_COLS - 1) / RED_COLS;
    constexpr uint32_t SLM_TILE_BYTES = (uint32_t) (KS * RED_COLS * 64u);
    constexpr uint32_t SLM_BYTES = (KS > 1) ? ((uint32_t) RT * SLM_TILE_BYTES) : 0u;

    const int tiles = n_out / 16;
    const int bpr = n_in / 256;
    const int act_ng = (ncols + 7) / 8;
    const uint32_t* w_base = reinterpret_cast<const uint32_t*>(weights);
    const uint8_t* act_b = reinterpret_cast<const uint8_t*>(act);

    const size_t wg_ks1 = (tiles % 16 == 0) ? 16u : ((tiles % 8 == 0) ? 8u : ((tiles % 4 == 0) ? 4u : 1u));
    const size_t g_range = (KS == 1) ? (size_t) tiles : ((size_t) tiles * (size_t) KS);
    const size_t l_range = (KS == 1) ? wg_ks1 : (size_t) (RT * KS);

    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& h) {
        sycl::nd_range<1> nd{sycl::range<1>{g_range}, sycl::range<1>{l_range}};
        h.parallel_for<TagW4xGemv<NC_SLOT, KS, GS>>(nd, props, [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            if constexpr (KS > 1) slm_init<SLM_BYTES>();
            int tile = 0;
            int rt = 0;
            int lt = 0;
            if constexpr (KS == 1) {
                tile = (int) it.get_global_id(0);
            } else {
                const int lid = (int) it.get_local_id(0);
                rt = lid / KS;
                lt = lid - rt * KS;
                tile = (int) it.get_group(0) * RT + rt;
            }
            const int kb0 = (KS == 1) ? 0 : ((bpr * lt) / KS);
            const int kb1 = (KS == 1) ? bpr : ((bpr * (lt + 1)) / KS);
            const int g0 = kb0 * 2;
            const int g1 = kb1 * 2;

            simd<float, 16> y[NC_SLOT];
            #pragma unroll
            for (int c = 0; c < NC_SLOT; ++c) y[c] = 0.0f;

            const uint32_t* w_tile = w_base + (size_t) tile * (size_t) bpr * kW4Rec;
            if (g0 < g1) {
                prefetch_g128(w_tile, g0);
                prefetch_g128(w_tile, (g0 + 1 < g1) ? (g0 + 1) : (g1 - 1));
                prefetch_g128(w_tile, (g0 + 2 < g1) ? (g0 + 2) : (g1 - 1));

                #pragma nounroll
                for (int g_idx = g0; g_idx < g1; ++g_idx) {
                    const int kb = g_idx >> 1;
                    const int g = g_idx & 1;
                    const int p0 = 2 * g;
                    const uint32_t* rec = w_tile + (size_t) kb * kW4Rec;
                    const uint8_t* act_g = act_b + (size_t) g_idx * (size_t) act_ng * act_group_stride(GS);

                    if constexpr (GS == 128) {
                        simd<int, 128> acc[NG];
                        for_ce<0, NG>([&](auto gi_c) SYCL_ESIMD_FUNCTION {
                            constexpr int gi = decltype(gi_c)::value;
                            acc[gi] = 0;
                        });

                        simd<int, 64> bw0 = es::block_load<int, 64>(reinterpret_cast<const int*>(rec + 128 * p0), kAl16);
                        simd<int, 64> bw1 = es::block_load<int, 64>(reinterpret_cast<const int*>(rec + 128 * p0 + 64), kAl16);
                        simd<int, 64> bw2 = es::block_load<int, 64>(reinterpret_cast<const int*>(rec + 128 * (p0 + 1)), kAl16);
                        simd<int, 64> bw3 = es::block_load<int, 64>(reinterpret_cast<const int*>(rec + 128 * (p0 + 1) + 64), kAl16);
                        simd<uint32_t, 16> meta = es::block_load<uint32_t, 16>(rec + 512 + 16 * g, kAl16);

                        const int g_pf = (g_idx + 3 < g1) ? (g_idx + 3) : (g1 - 1);
                        prefetch_g128(w_tile, g_pf);

                        auto run_half = [&](const simd<int, 64>& B, uint32_t p_off) SYCL_ESIMD_FUNCTION __attribute__((always_inline)) {
                            for_ce<0, NG>([&](auto gi_c) SYCL_ESIMD_FUNCTION {
                                constexpr int gi = decltype(gi_c)::value;
                                const uint8_t* ap = act_g + (size_t) gi * 1088u + p_off;
                                simd<int, 64> A = es::block_load<int, 64>(reinterpret_cast<const int*>(ap), kAl16);
                                acc[gi] = xmx::dpas<8, 8, int, int, int, int,
                                    xmx::dpas_argument_type::u4, xmx::dpas_argument_type::s8>(acc[gi], B, A);
                            });
                        };

                        run_half(bw0, 0u);
                        run_half(bw1, 256u);
                        run_half(bw2, 512u);
                        run_half(bw3, 768u);

                        simd<float, 16> s_row = half_of(meta);
                        simd<int, 16> z_int(meta.template bit_cast_view<uint16_t>().template select<16, 2>(1));
                        for_ce<0, NG>([&](auto gi_c) SYCL_ESIMD_FUNCTION {
                            constexpr int gi = decltype(gi_c)::value;
                            constexpr int c_lim = (gi + 1) * 8 <= NC_SLOT ? 8 : (NC_SLOT - gi * 8);
                            const uint8_t* ap = act_g + (size_t) gi * 1088u;
                            simd<uint32_t, 16> sa_sxi = es::block_load<uint32_t, 16>(reinterpret_cast<const uint32_t*>(ap + 1024u), kAl16);
                            simd<float, 8> sa_v = sa_sxi.template select<8, 1>(0).template bit_cast_view<float>();
                            simd<int, 8> sxi_v  = sa_sxi.template select<8, 1>(8).template bit_cast_view<int>();
                            for_ce<0, c_lim>([&](auto rc_c) SYCL_ESIMD_FUNCTION {
                                constexpr int rc = decltype(rc_c)::value;
                                constexpr int c = gi * 8 + rc;
                                const float sa_c = float(sa_v[rc]);
                                const int sxi = int(sxi_v[rc]);
                                simd<int, 16> diff = acc[gi].template select<16, 1>(16 * rc) - z_int * sxi;
                                y[c] += convert<float>(diff) * (s_row * sa_c);
                            });
                        });
                    } else {
                        simd<uint32_t, 16> meta = es::block_load<uint32_t, 16>(rec + 512 + 16 * g, kAl16);
                        simd<float, 16> s_row = half_of(meta);
                        simd<float, 16> z_flt(meta.template bit_cast_view<uint16_t>().template select<16, 2>(1));
                        simd<float, 16> z_prod = s_row * z_flt;

                        const int g_pf = (g_idx + 3 < g1) ? (g_idx + 3) : (g1 - 1);
                        prefetch_g128(w_tile, g_pf);

                        simd<float, 32> sa_all[NG];
                        simd<float, 8> sx_all[NG];
                        for_ce<0, NG>([&](auto gi_c) SYCL_ESIMD_FUNCTION {
                            constexpr int gi = decltype(gi_c)::value;
                            const uint8_t* ap = act_g + (size_t) gi * 1216u;
                            sa_all[gi] = es::block_load<float, 32>(reinterpret_cast<const float*>(ap + 1024u), kAl16);
                            sx_all[gi] = es::block_load<float, 8>(reinterpret_cast<const float*>(ap + 1152u), kAl16);
                        });

                        #pragma nounroll
                        for (int lp = 0; lp < 2; ++lp) {
                            const int p = 2 * g + lp;
                            simd<uint32_t, 64> bw0 = es::block_load<uint32_t, 64>(rec + 128 * p, kAl16);
                            simd<uint32_t, 64> bw1 = es::block_load<uint32_t, 64>(rec + 128 * p + 64, kAl16);
                            simd<int, 64> B_even, B_odd;
                            for_ce<0, 4>([&](auto j_c) SYCL_ESIMD_FUNCTION {
                                constexpr int j = decltype(j_c)::value;
                                constexpr int off = (j & 1) * 32;
                                simd<uint32_t, 16> u = (j < 2) ? simd<uint32_t, 16>(bw0.template select<16, 1>(off))
                                                               : simd<uint32_t, 16>(bw1.template select<16, 1>(off));
                                simd<uint32_t, 16> v = (j < 2) ? simd<uint32_t, 16>(bw0.template select<16, 1>(off + 16))
                                                               : simd<uint32_t, 16>(bw1.template select<16, 1>(off + 16));
                                simd<uint32_t, 16> lu = u & 0x0f0f0f0fu;
                                lu = (lu | (lu >> 4)) & 0x00ff00ffu;
                                lu |= (lu >> 8);
                                simd<uint32_t, 16> lv = v & 0x0f0f0f0fu;
                                lv = (lv | (lv >> 4)) & 0x00ff00ffu;
                                lv |= (lv >> 8);
                                B_even.template select<16, 1>(16 * j) = join16(lu, lv);
                                simd<uint32_t, 16> hu = (u >> 4) & 0x0f0f0f0fu;
                                hu = (hu | (hu >> 4)) & 0x00ff00ffu;
                                hu |= (hu >> 8);
                                simd<uint32_t, 16> hv = (v >> 4) & 0x0f0f0f0fu;
                                hv = (hv | (hv >> 4)) & 0x00ff00ffu;
                                hv |= (hv >> 8);
                                B_odd.template select<16, 1>(16 * j) = join16(hu, hv);
                            });

                            const int sb_e = 2 * lp;
                            const int sb_o = 2 * lp + 1;
                            for_ce<0, NG>([&](auto gi_c) SYCL_ESIMD_FUNCTION {
                                constexpr int gi = decltype(gi_c)::value;
                                constexpr int c_lim = (gi + 1) * 8 <= NC_SLOT ? 8 : (NC_SLOT - gi * 8);
                                const uint8_t* ap = act_g + (size_t) gi * 1216u;
                                simd<int, 64> Ae = es::block_load<int, 64>(reinterpret_cast<const int*>(ap + sb_e * 256u), kAl16);
                                simd<int, 64> Ao = es::block_load<int, 64>(reinterpret_cast<const int*>(ap + sb_o * 256u), kAl16);
                                simd<int, 128> acc_e = xmx::dpas<8, 8, int, int, int, int,
                                    xmx::dpas_argument_type::u4, xmx::dpas_argument_type::s8>(simd<int, 128>(0), B_even, Ae);
                                simd<int, 128> acc_o = xmx::dpas<8, 8, int, int, int, int,
                                    xmx::dpas_argument_type::u4, xmx::dpas_argument_type::s8>(simd<int, 128>(0), B_odd, Ao);
                                for_ce<0, c_lim>([&](auto rc_c) SYCL_ESIMD_FUNCTION {
                                    constexpr int rc = decltype(rc_c)::value;
                                    constexpr int c = gi * 8 + rc;
                                    simd<float, 16> p_sum =
                                        convert<float>(simd<int, 16>(acc_e.template select<16, 1>(16 * rc))) * float(sa_all[gi][sb_e * 8 + rc])
                                      + convert<float>(simd<int, 16>(acc_o.template select<16, 1>(16 * rc))) * float(sa_all[gi][sb_o * 8 + rc]);
                                    y[c] += s_row * p_sum;
                                });
                            });
                        }

                        for_ce<0, NG>([&](auto gi_c) SYCL_ESIMD_FUNCTION {
                            constexpr int gi = decltype(gi_c)::value;
                            constexpr int c_lim = (gi + 1) * 8 <= NC_SLOT ? 8 : (NC_SLOT - gi * 8);
                            for_ce<0, c_lim>([&](auto rc_c) SYCL_ESIMD_FUNCTION {
                                constexpr int rc = decltype(rc_c)::value;
                                constexpr int c = gi * 8 + rc;
                                y[c] -= z_prod * float(sx_all[gi][rc]);
                            });
                        });
                    }
                }
            }

            if constexpr (KS == 1) {
                for_ce<0, NC_SLOT>([&](auto c_c) SYCL_ESIMD_FUNCTION {
                    constexpr int c = decltype(c_c)::value;
                    if (c < ncols) {
                        es::block_store<float, 16>(y_out + (size_t) c * (size_t) n_out + (size_t) tile * 16, y[c], kAl16);
                    }
                });
            } else {
                const uint32_t slm_tile = (uint32_t) rt * SLM_TILE_BYTES;
                for_ce<0, RED_ROUNDS>([&](auto rd_c) SYCL_ESIMD_FUNCTION {
                    constexpr int rd = decltype(rd_c)::value;
                    constexpr int c0 = rd * RED_COLS;
                    constexpr int c_cnt = (c0 + RED_COLS <= NC_SLOT) ? RED_COLS : (NC_SLOT - c0);
                    constexpr int MAX_OWN = (c_cnt + KS - 1) / KS;
                    for_ce<0, c_cnt>([&](auto rc_c) SYCL_ESIMD_FUNCTION {
                        constexpr int rc = decltype(rc_c)::value;
                        es::slm_block_store<float, 16>(
                            slm_tile + ((uint32_t) lt * c_cnt + (uint32_t) rc) * 64u,
                            y[c0 + rc], kAl16);
                    });
                    es::barrier();
                    for_ce<0, MAX_OWN>([&](auto q_c) SYCL_ESIMD_FUNCTION {
                        constexpr int q_idx = decltype(q_c)::value;
                        const int rc = q_idx * KS + lt;
                        const int c = c0 + rc;
                        if (rc < c_cnt && c < ncols) {
                            simd<float, 16> sum = es::slm_block_load<float, 16>(
                                slm_tile + (uint32_t) rc * 64u, kAl16);
                            for_ce<1, KS>([&](auto t_c) SYCL_ESIMD_FUNCTION {
                                constexpr int t = decltype(t_c)::value;
                                sum += es::slm_block_load<float, 16>(
                                    slm_tile + ((uint32_t) t * c_cnt + (uint32_t) rc) * 64u, kAl16);
                            });
                            es::block_store<float, 16>(
                                y_out + (size_t) c * (size_t) n_out + (size_t) tile * 16, sum, kAl16);
                        }
                    });
                    if constexpr (rd + 1 < RED_ROUNDS) es::barrier();
                });
            }
        });
    });
}

// One WG of 32 HW threads (8 tiles x 2 gate/up x 2 K-splits) computes 128 FFN rows (gate + up),
// reduces KS=2 in SLM, runs silu(g) * u across 8 columns at a time, and writes GS=128 act_down.
template <int NC_SLOT>
void launch_gu_swiglu(sycl::queue& q, const void* gu_weights, const void* act_in, void* act_down,
                      int n_in, int n_ff, int ncols) {
    constexpr int NG = (NC_SLOT + 7) / 8;
    constexpr int REC_STRIDE = 2 * kW4Rec;
    // SLM (18,048 B = 564 B/thread for 32 threads):
    //   [0 .. 16383]:     4 (ks*2 + is_up) x 64 (rc*8 + tile) x 64 B partials; reused at [0..4095] for sw[rc][tile]
    //   [16384 .. 17407]: 64 (rc*8 + tile) x 16 B tile maxes
    //   [17408 .. 17919]: 32 (rc*4 + sub)  x 16 B sub-block sum_p
    //   [17920 .. 18047]: 8  (rc)          x 16 B column scale d
    constexpr uint32_t SLM_BYTES = 18048u;
    const int n_fg = n_ff / 128;
    const int bpr = n_in / 256;
    const int act_ng = (ncols + 7) / 8;
    const uint32_t* w_base = reinterpret_cast<const uint32_t*>(gu_weights);
    const uint8_t* act_in_b = reinterpret_cast<const uint8_t*>(act_in);
    uint8_t* act_down_b = reinterpret_cast<uint8_t*>(act_down);

    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& h) {
        sycl::nd_range<1> nd(sycl::range<1>((size_t) n_fg * 32u), sycl::range<1>(32u));
        h.parallel_for<TagW4xGuSwiglu<NC_SLOT>>(nd, props, [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            slm_init<SLM_BYTES>();
            const int fg = (int) it.get_group(0);
            const int lt = (int) it.get_local_id(0);
            const int tile_local = lt & 7;
            const int is_up = (lt >> 3) & 1;
            const int ks = lt >> 4;
            const int kb0 = (bpr * ks) / 2;
            const int kb1 = (bpr * (ks + 1)) / 2;
            const int g0 = kb0 * 2;
            const int g1 = kb1 * 2;

            const uint32_t* w_tile = w_base
                + ((size_t) fg * 8u + (size_t) tile_local) * (size_t) bpr * (size_t) REC_STRIDE
                + (size_t) is_up * (size_t) kW4Rec;

            simd<float, 16> y[NC_SLOT];
            #pragma unroll
            for (int c = 0; c < NC_SLOT; ++c) y[c] = 0.0f;

            if (g0 < g1) {
                prefetch_g128<REC_STRIDE>(w_tile, g0);
                prefetch_g128<REC_STRIDE>(w_tile, (g0 + 1 < g1) ? (g0 + 1) : (g1 - 1));
                prefetch_g128<REC_STRIDE>(w_tile, (g0 + 2 < g1) ? (g0 + 2) : (g1 - 1));

                #pragma nounroll
                for (int g_idx = g0; g_idx < g1; ++g_idx) {
                    const int kb = g_idx >> 1;
                    const int g = g_idx & 1;
                    const int p0 = 2 * g;
                    const uint32_t* rec = w_tile + (size_t) kb * (size_t) REC_STRIDE;
                    const uint8_t* act_g = act_in_b + (size_t) g_idx * (size_t) act_ng * 1088u;

                    simd<int, 128> acc[NG];
                    for_ce<0, NG>([&](auto gi_c) SYCL_ESIMD_FUNCTION {
                        constexpr int gi = decltype(gi_c)::value;
                        acc[gi] = 0;
                    });

                    simd<int, 64> bw0 = es::block_load<int, 64>(reinterpret_cast<const int*>(rec + 128 * p0), kAl16);
                    simd<int, 64> bw1 = es::block_load<int, 64>(reinterpret_cast<const int*>(rec + 128 * p0 + 64), kAl16);
                    simd<int, 64> bw2 = es::block_load<int, 64>(reinterpret_cast<const int*>(rec + 128 * (p0 + 1)), kAl16);
                    simd<int, 64> bw3 = es::block_load<int, 64>(reinterpret_cast<const int*>(rec + 128 * (p0 + 1) + 64), kAl16);
                    simd<uint32_t, 16> meta = es::block_load<uint32_t, 16>(rec + 512 + 16 * g, kAl16);

                    const int g_pf = (g_idx + 3 < g1) ? (g_idx + 3) : (g1 - 1);
                    prefetch_g128<REC_STRIDE>(w_tile, g_pf);

                    auto run_half = [&](const simd<int, 64>& B, uint32_t p_off) SYCL_ESIMD_FUNCTION __attribute__((always_inline)) {
                        for_ce<0, NG>([&](auto gi_c) SYCL_ESIMD_FUNCTION {
                            constexpr int gi = decltype(gi_c)::value;
                            const uint8_t* ap = act_g + (size_t) gi * 1088u + p_off;
                            simd<int, 64> A = es::block_load<int, 64>(reinterpret_cast<const int*>(ap), kAl16);
                            acc[gi] = xmx::dpas<8, 8, int, int, int, int,
                                xmx::dpas_argument_type::u4, xmx::dpas_argument_type::s8>(acc[gi], B, A);
                        });
                    };

                    run_half(bw0, 0u);
                    run_half(bw1, 256u);
                    run_half(bw2, 512u);
                    run_half(bw3, 768u);

                    simd<float, 16> s_row = half_of(meta);
                    simd<int, 16> z_int(meta.template bit_cast_view<uint16_t>().template select<16, 2>(1));
                    for_ce<0, NG>([&](auto gi_c) SYCL_ESIMD_FUNCTION {
                        constexpr int gi = decltype(gi_c)::value;
                        constexpr int c_lim = (gi + 1) * 8 <= NC_SLOT ? 8 : (NC_SLOT - gi * 8);
                        const uint8_t* ap = act_g + (size_t) gi * 1088u;
                        simd<uint32_t, 16> sa_sxi = es::block_load<uint32_t, 16>(reinterpret_cast<const uint32_t*>(ap + 1024u), kAl16);
                        simd<float, 8> sa_v = sa_sxi.template select<8, 1>(0).template bit_cast_view<float>();
                        simd<int, 8> sxi_v  = sa_sxi.template select<8, 1>(8).template bit_cast_view<int>();
                        for_ce<0, c_lim>([&](auto rc_c) SYCL_ESIMD_FUNCTION {
                            constexpr int rc = decltype(rc_c)::value;
                            constexpr int c = gi * 8 + rc;
                            simd<int, 16> diff = acc[gi].template select<16, 1>(16 * rc) - z_int * int(sxi_v[rc]);
                            y[c] += convert<float>(diff) * (s_row * float(sa_v[rc]));
                        });
                    });
                }
            }

            // Epilogue: reduce KS=2, compute silu(yg) * yu, and quantize 8 columns per gi in parallel across 32 threads.
            const uint32_t plane_off = (uint32_t) (ks * 2 + is_up) * 4096u;
            const int t_loc = lt & 7;
            const int rc0 = (lt >> 3) * 2;
            const int rc1 = rc0 + 1;
            const int q_rc = lt >> 2;
            const int q_sub = lt & 3;

            for_ce<0, NG>([&](auto gi_c) SYCL_ESIMD_FUNCTION {
                constexpr int gi = decltype(gi_c)::value;
                constexpr int c_lim = (gi + 1) * 8 <= NC_SLOT ? 8 : (NC_SLOT - gi * 8);
                uint8_t* dst = act_down_b + ((size_t) fg * act_ng + (size_t) gi) * 1088u;

                for_ce<0, c_lim>([&](auto rc_c) SYCL_ESIMD_FUNCTION {
                    constexpr int rc = decltype(rc_c)::value;
                    es::slm_block_store<float, 16>(
                        plane_off + (uint32_t) (rc * 8 + tile_local) * 64u,
                        y[gi * 8 + rc], kAl16);
                });
                es::barrier();

                if (rc0 < c_lim && (gi * 8 + rc0) < ncols) {
                    const uint32_t s0_idx = (uint32_t) (rc0 * 8 + t_loc);
                    const uint32_t b0 = s0_idx * 64u;
                    simd<float, 16> gsum = es::slm_block_load<float, 16>(b0, kAl16)
                                         + es::slm_block_load<float, 16>(b0 + 8192u, kAl16);
                    simd<float, 16> usum = es::slm_block_load<float, 16>(b0 + 4096u, kAl16)
                                         + es::slm_block_load<float, 16>(b0 + 12288u, kAl16);
                    simd<float, 16> sw = (gsum / (1.0f + es::exp((es::min)(-gsum, simd<float, 16>(88.0f))))) * usum;
                    es::slm_block_store<float, 16>(b0, sw, kAl16);
                    es::slm_block_store<float, 4>(16384u + s0_idx * 16u, simd<float, 4>(hmax16_f(es::abs(sw))), kAl16);
                }
                if (rc1 < c_lim && (gi * 8 + rc1) < ncols) {
                    const uint32_t s1_idx = (uint32_t) (rc1 * 8 + t_loc);
                    const uint32_t b1 = s1_idx * 64u;
                    simd<float, 16> gsum = es::slm_block_load<float, 16>(b1, kAl16)
                                         + es::slm_block_load<float, 16>(b1 + 8192u, kAl16);
                    simd<float, 16> usum = es::slm_block_load<float, 16>(b1 + 4096u, kAl16)
                                         + es::slm_block_load<float, 16>(b1 + 12288u, kAl16);
                    simd<float, 16> sw = (gsum / (1.0f + es::exp((es::min)(-gsum, simd<float, 16>(88.0f))))) * usum;
                    es::slm_block_store<float, 16>(b1, sw, kAl16);
                    es::slm_block_store<float, 4>(16384u + s1_idx * 16u, simd<float, 4>(hmax16_f(es::abs(sw))), kAl16);
                }
                es::barrier();

                const int col = gi * 8 + q_rc;
                if (q_rc < c_lim && col < ncols) {
                    simd<float, 32> m32 = es::slm_block_load<float, 32>(16384u + (uint32_t) q_rc * 128u, kAl16);
                    simd<float, 8> m8 = m32.template select<8, 4>(0);
                    simd<float, 4> m4 = (es::max)(simd<float, 4>(m8.template select<4, 1>(0)),
                                                  simd<float, 4>(m8.template select<4, 1>(4)));
                    simd<float, 2> m2 = (es::max)(simd<float, 2>(m4.template select<2, 1>(0)),
                                                  simd<float, 2>(m4.template select<2, 1>(2)));
                    const float amax = hmax2(float(m2[0]), float(m2[1]));
                    const float d = amax / 127.0f;
                    const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

                    const int p = q_sub >> 1;
                    const int hh = q_sub & 1;
                    const uint32_t e_off = (uint32_t) (q_rc * 512 + (p * 4 + hh) * 64);
                    const uint32_t o_off = e_off + 128u;
                    simd<float, 16> ve = es::slm_block_load<float, 16>(e_off, kAl16);
                    simd<float, 16> vo = es::slm_block_load<float, 16>(o_off, kAl16);
                    simd<int, 16> qe = quant_round16(ve, id);
                    simd<int, 16> qo = quant_round16(vo, id);
                    const int sum_p = hsum16_i(qe) + hsum16_i(qo);
                    simd<uint32_t, 16> ue = qe.template bit_cast_view<uint32_t>();
                    simd<uint32_t, 16> uo = qo.template bit_cast_view<uint32_t>();
                    simd<uint32_t, 8> pk = (ue.template select<8, 2>(0) & 0xffu)
                                         | ((uo.template select<8, 2>(0) & 0xffu) << 8)
                                         | ((ue.template select<8, 2>(1) & 0xffu) << 16)
                                         | ((uo.template select<8, 2>(1) & 0xffu) << 24);
                    es::block_store<uint32_t, 8>(
                        reinterpret_cast<uint32_t*>(dst + (size_t) q_sub * 256u + (size_t) q_rc * 32u), pk, kAl16);
                    es::slm_block_store<int, 4>(17408u + (uint32_t) lt * 16u, simd<int, 4>(sum_p), kAl16);
                    if (q_sub == 0) {
                        es::slm_block_store<float, 4>(17920u + (uint32_t) q_rc * 16u, simd<float, 4>(d), kAl16);
                    }
                } else {
                    es::block_store<uint32_t, 8>(
                        reinterpret_cast<uint32_t*>(dst + (size_t) q_sub * 256u + (size_t) q_rc * 32u),
                        simd<uint32_t, 8>(0u), kAl16);
                    es::slm_block_store<int, 4>(17408u + (uint32_t) lt * 16u, simd<int, 4>(0), kAl16);
                    if (q_sub == 0) {
                        es::slm_block_store<float, 4>(17920u + (uint32_t) q_rc * 16u, simd<float, 4>(0.0f), kAl16);
                    }
                }
                es::barrier();

                if (lt == 0) {
                    simd<float, 32> d32 = es::slm_block_load<float, 32>(17920u, kAl16);
                    simd<float, 8> sa_out = d32.template select<8, 4>(0);
                    simd<int, 32> s0 = es::slm_block_load<int, 32>(17408u, kAl16);
                    simd<int, 32> s1 = es::slm_block_load<int, 32>(17536u, kAl16);
                    simd<int, 32> s2 = es::slm_block_load<int, 32>(17664u, kAl16);
                    simd<int, 32> s3 = es::slm_block_load<int, 32>(17792u, kAl16);
                    simd<int, 8> v0 = s0.template select<8, 4>(0);
                    simd<int, 8> v1 = s1.template select<8, 4>(0);
                    simd<int, 8> v2 = s2.template select<8, 4>(0);
                    simd<int, 8> v3 = s3.template select<8, 4>(0);
                    simd<int, 4> u0 = v0.template select<4, 2>(0) + v0.template select<4, 2>(1);
                    simd<int, 4> u1 = v1.template select<4, 2>(0) + v1.template select<4, 2>(1);
                    simd<int, 4> u2 = v2.template select<4, 2>(0) + v2.template select<4, 2>(1);
                    simd<int, 4> u3 = v3.template select<4, 2>(0) + v3.template select<4, 2>(1);
                    simd<int, 8> sxi_out;
                    sxi_out.template select<2, 1>(0) = u0.template select<2, 2>(0) + u0.template select<2, 2>(1);
                    sxi_out.template select<2, 1>(2) = u1.template select<2, 2>(0) + u1.template select<2, 2>(1);
                    sxi_out.template select<2, 1>(4) = u2.template select<2, 2>(0) + u2.template select<2, 2>(1);
                    sxi_out.template select<2, 1>(6) = u3.template select<2, 2>(0) + u3.template select<2, 2>(1);
                    es::block_store<float, 8>(reinterpret_cast<float*>(dst + 1024u), sa_out, kAl16);
                    es::block_store<int, 8>(reinterpret_cast<int*>(dst + 1056u), sxi_out, kAl16);
                }
                if constexpr (gi + 1 < NG) es::barrier();
            });
        });
    });
}

int read_w4x_ks_env() {
    static const int ks = []() {
        const char* e = std::getenv("STRATA_Q36_W4X_KS");
        return (e && *e) ? std::atoi(e) : 0;
    }();
    return ks;
}

int snap_ks(int req, int bpr, int tiles) {
    const int cap = std::min(req, bpr);
    auto ok = [&](int ks, int rt) { return cap >= ks && (tiles % rt == 0); };
    if (ok(10, 2)) return 10;
    if (ok(6, 4))  return 6;
    if (ok(5, 4))  return 5;
    if (ok(4, 4))  return 4;
    if (ok(3, 8))  return 3;
    if (ok(2, 8))  return 2;
    return 1;
}

int choose_ks(int n_in, int n_out, int ksplit_override) {
    const int bpr = n_in / 256;
    const int tiles = n_out / 16;
    const int env_ks = read_w4x_ks_env();
    if (ksplit_override > 0) return snap_ks(ksplit_override, bpr, tiles);
    if (env_ks > 0) return snap_ks(env_ks, bpr, tiles);

    int want = 1;
    if (tiles <= 64 && bpr >= 10) want = 10;
    else if (tiles <= 96 && bpr >= 5) want = 5;
    else if (tiles <= 256 && bpr >= 4) want = 4;
    else if (tiles <= 320 && bpr >= 3) want = 3;
    else if (tiles <= 512 && bpr >= 2) want = 2;
    else if (tiles == 1088 && bpr >= 5) want = 5;
    return snap_ks(want, bpr, tiles);
}

template <int NC_SLOT, int KS>
void dispatch_gs(sycl::queue& q, const void* weights, const void* act, float* y,
                 int n_in, int n_out, int ncols, int gs) {
    if (gs == 32) launch_gemv<NC_SLOT, KS, 32>(q, weights, act, y, n_in, n_out, ncols);
    else launch_gemv<NC_SLOT, KS, 128>(q, weights, act, y, n_in, n_out, ncols);
}

template <int NC_SLOT>
void dispatch_ks(sycl::queue& q, const void* weights, const void* act, float* y,
                 int n_in, int n_out, int ncols, int gs, int ks) {
    switch (ks) {
        case 1:  dispatch_gs<NC_SLOT, 1>(q, weights, act, y, n_in, n_out, ncols, gs); return;
        case 2:  dispatch_gs<NC_SLOT, 2>(q, weights, act, y, n_in, n_out, ncols, gs); return;
        case 3:  dispatch_gs<NC_SLOT, 3>(q, weights, act, y, n_in, n_out, ncols, gs); return;
        case 4:  dispatch_gs<NC_SLOT, 4>(q, weights, act, y, n_in, n_out, ncols, gs); return;
        case 5:  dispatch_gs<NC_SLOT, 5>(q, weights, act, y, n_in, n_out, ncols, gs); return;
        case 6:  dispatch_gs<NC_SLOT, 6>(q, weights, act, y, n_in, n_out, ncols, gs); return;
        case 10: dispatch_gs<NC_SLOT, 10>(q, weights, act, y, n_in, n_out, ncols, gs); return;
        default: dispatch_gs<NC_SLOT, 1>(q, weights, act, y, n_in, n_out, ncols, gs); return;
    }
}

}   // namespace

std::size_t w4x_act_bytes(int n_in, int ncols, int gs) {
    if (n_in <= 0 || ncols <= 0) return 0;
    const std::size_t n_g128 = (std::size_t) ((n_in + 127) / 128);
    const std::size_t ng = (std::size_t) ((ncols + 7) / 8);
    return n_g128 * ng * act_group_stride(gs == 32 ? 32 : 128);
}

void w4x_quantize(const float* x, int ldx, void* act, int n_in, int ncols, int gs, void* stream) {
    auto* q = reinterpret_cast<sycl::queue*>(stream);
    if (!q || !x || !act || n_in <= 0 || n_in % 128 != 0 || ncols <= 0 || ncols > 32) return;
    if (gs == 32) launch_quantize<32, 0>(*q, x, ldx, nullptr, nullptr, act, nullptr, 0, n_in, ncols);
    else launch_quantize<128, 0>(*q, x, ldx, nullptr, nullptr, act, nullptr, 0, n_in, ncols);
}

void w4x_rms_apply_quantize(const float* x_res, int ldx, const float* norm_w, const float* inv_rms,
                            void* act, float* h_out, int ld_out, int n_in, int ncols, int gs, void* stream) {
    auto* q = reinterpret_cast<sycl::queue*>(stream);
    if (!q || !x_res || !norm_w || !inv_rms || !act || n_in <= 0 || n_in % 128 != 0 || ncols <= 0 || ncols > 32) return;
    if (gs == 32) launch_quantize<32, 1>(*q, x_res, ldx, norm_w, inv_rms, act, h_out, ld_out, n_in, ncols);
    else launch_quantize<128, 1>(*q, x_res, ldx, norm_w, inv_rms, act, h_out, ld_out, n_in, ncols);
}

void w4x_swiglu_quantize(const float* g, const float* u, void* act, int n_in, int ncols, int gs, void* stream) {
    auto* q = reinterpret_cast<sycl::queue*>(stream);
    if (!q || !g || !u || !act || n_in <= 0 || n_in % 128 != 0 || ncols <= 0 || ncols > 32) return;
    if (gs == 32) launch_quantize<32, 2>(*q, g, n_in, u, nullptr, act, nullptr, 0, n_in, ncols);
    else launch_quantize<128, 2>(*q, g, n_in, u, nullptr, act, nullptr, 0, n_in, ncols);
}

bool mmvq_w4x(const void* weights, const void* act, float* y, int n_in, int n_out, int ncols, int gs,
              void* stream, int ksplit_override) {
    auto* q = reinterpret_cast<sycl::queue*>(stream);
    if (!q || !weights || !act || !y) return false;
    if (n_in <= 0 || n_out <= 0 || n_in % 256 != 0 || n_out % 16 != 0) return false;
    if (ncols < 1 || ncols > 32 || (gs != 128 && gs != 32)) return false;

    const int ks = choose_ks(n_in, n_out, ksplit_override);
    if (ncols == 1)      dispatch_ks<1>(*q, weights, act, y, n_in, n_out, ncols, gs, ks);
    else if (ncols <= 4) dispatch_ks<4>(*q, weights, act, y, n_in, n_out, ncols, gs, ks);
    else if (ncols <= 8) dispatch_ks<8>(*q, weights, act, y, n_in, n_out, ncols, gs, ks);
    else if (ncols <= 16) dispatch_ks<16>(*q, weights, act, y, n_in, n_out, ncols, gs, ks);
    else if (ncols <= 24) dispatch_ks<24>(*q, weights, act, y, n_in, n_out, ncols, gs, ks);
    else                 dispatch_ks<32>(*q, weights, act, y, n_in, n_out, ncols, gs, ks);
    return true;
}

std::size_t w4x_gu_merged_bytes(int n_in, int n_ff) {
    if (n_in <= 0 || n_ff <= 0 || n_in % 256 != 0 || n_ff % 128 != 0) return 0;
    return 2u * (std::size_t) (n_ff / 16) * (std::size_t) (n_in / 256) * (std::size_t) kW4Rec * 4u;
}

bool mmvq_w4x_gu_swiglu(const void* gu_weights, const void* act_in, void* act_down,
                        int n_in, int n_ff, int ncols, int gs, void* stream) {
    auto* q = reinterpret_cast<sycl::queue*>(stream);
    if (!q || !gu_weights || !act_in || !act_down) return false;
    if (n_in <= 0 || n_ff <= 0 || n_in % 256 != 0 || n_ff % 128 != 0) return false;
    if (ncols < 1 || ncols > 32 || gs != 128) return false;

    if (ncols == 1)      launch_gu_swiglu<1>(*q, gu_weights, act_in, act_down, n_in, n_ff, ncols);
    else if (ncols <= 4) launch_gu_swiglu<4>(*q, gu_weights, act_in, act_down, n_in, n_ff, ncols);
    else if (ncols <= 8) launch_gu_swiglu<8>(*q, gu_weights, act_in, act_down, n_in, n_ff, ncols);
    else if (ncols <= 16) launch_gu_swiglu<16>(*q, gu_weights, act_in, act_down, n_in, n_ff, ncols);
    else if (ncols <= 24) launch_gu_swiglu<24>(*q, gu_weights, act_in, act_down, n_in, n_ff, ncols);
    else                 launch_gu_swiglu<32>(*q, gu_weights, act_in, act_down, n_in, n_ff, ncols);
    return true;
}

}   // namespace strata::q36xr
