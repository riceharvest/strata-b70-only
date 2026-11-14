// q36_pp3_bench.dp.cpp - PARITY=ON 256-GRF fatgemm pure-int reference & PP3-v2 contiguous-weight benchmark kernels.
// Compiled into shared library `q36_pp3_bench` with `-O3 -fsycl-device-code-split=per_kernel -doubleGRF`, matching fatgemm.cpp.
#include "q36_pp3_bench.hpp"
#include "strata/sycl_queue.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

namespace strata::q36pp::bench {
namespace {
using namespace sycl::ext::intel::esimd;
namespace es  = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using DT      = xmx::dpas_argument_type;

constexpr auto kAl16 = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::alignment<16>};
constexpr auto kCC   = sycl::ext::oneapi::experimental::properties{cache_hint_L1<cache_hint::cached>, cache_hint_L2<cache_hint::cached>};

struct RecipDiv {
    uint32_t d;
    uint64_t mul;
};

inline RecipDiv make_recip(uint32_t d) noexcept {
    return RecipDiv{d, ((1ULL << 40) + static_cast<uint64_t>(d) - 1ULL) / static_cast<uint64_t>(d)};
}

SYCL_ESIMD_FUNCTION inline uint32_t fast_div(uint32_t x, const RecipDiv& r) {
    return static_cast<uint32_t>((static_cast<uint64_t>(x) * r.mul) >> 40);
}

SYCL_ESIMD_FUNCTION inline simd<float, 16> half_of(simd<uint32_t, 16> h16) {
    simd<uint16_t, 16> lo = h16.template bit_cast_view<uint16_t>().template select<16, 2>(0);
    simd<sycl::half, 16> hh = lo.template bit_cast_view<sycl::half>();
    return simd<float, 16>(hh);
}

SYCL_ESIMD_FUNCTION inline simd<sycl::half, 16> half_raw_of(simd<uint32_t, 16> h16) {
    simd<uint16_t, 16> lo = h16.template bit_cast_view<uint16_t>().template select<16, 2>(0);
    return lo.template bit_cast_view<sycl::half>();
}

struct BenchGrid {
    uint32_t lgRT, lgCT, RT, CT;
    uint32_t ncg, panel_wgs, nw, g_wgs;
    RecipDiv div_pw, div_ncg;
};

inline BenchGrid make_grid(uint32_t n_rt32, uint32_t n_ct32) noexcept {
    BenchGrid g{};
    constexpr uint32_t kMaxThr = 1024u;
    if (n_ct32 >= 16u && n_rt32 >= 4u) {
        g.lgRT = 2u; g.lgCT = 3u;
    } else {
        struct Cand { uint32_t lgr, lgc; };
        static constexpr Cand kCands[] = {
            {2u, 3u}, {3u, 2u}, {4u, 1u}, {3u, 1u}, {2u, 2u}, {2u, 1u}, {3u, 0u}, {4u, 0u},
        };
        double best = -1.0;
        uint32_t blgr = 2u, blgc = 3u;
        for (const auto& c : kCands) {
            const uint32_t rt = 1u << c.lgr, ct = 1u << c.lgc, wg = rt * ct;
            if (wg > kMaxThr) continue;
            const uint32_t max_wgs = std::max<uint32_t>(1u, kMaxThr / wg);
            const uint32_t nw = ((n_rt32 + rt - 1u) / rt) * ((n_ct32 + ct - 1u) / ct);
            const uint32_t waves = (nw + max_wgs - 1u) / max_wgs;
            const double util = static_cast<double>(n_rt32 * n_ct32) /
                                (0.75 * static_cast<double>(waves * kMaxThr) + 0.25 * static_cast<double>(nw * wg));
            const double sc = util + 2e-3 * (wg == 32u ? 1.0 : 0.0) + 1e-3 * static_cast<double>(c.lgc);
            if (sc > best) { best = sc; blgr = c.lgr; blgc = c.lgc; }
        }
        g.lgRT = blgr; g.lgCT = blgc;
    }
    g.RT = 1u << g.lgRT;
    g.CT = 1u << g.lgCT;
    const uint32_t wg_thr    = g.RT * g.CT;
    const uint32_t max_wgs   = std::max<uint32_t>(1u, kMaxThr / wg_thr);
    const uint32_t n_wgr     = (n_rt32 + g.RT - 1u) / g.RT;
    const uint32_t n_wgc_tot = (n_ct32 + g.CT - 1u) / g.CT;
    const uint32_t ncg_t     = 32u >> g.lgCT;
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

class TagFgPureRef;
class TagW4I8Pp3V2;
class TagConvFg;
class TagConvV2;

template <int S>
SYCL_ESIMD_FUNCTION inline void pp3_v2_substep(
    simd<int, 128> (&i_act)[2][4],
    simd<int, 128> (&i_oth)[2][4],
    simd<float, 128> (&c_acc)[2][4],
    simd<int, 64> (&wr)[2][2],
    simd<float, 16> (&s_scale)[2],
    simd<float, 32>& dx_scale,
    simd<uint32_t, 64>& meta_kb,
    simd<uint32_t, 64>& dx_nxt,
    const uint32_t* wp,
    const uint32_t* wp_nxt,
    const int* xp,
    const int* xp_nxt,
    const uint32_t* dxp_nxt,
    float* y_base,
    uint32_t kb,
    uint32_t nv) {
    constexpr int u  = S & 1;
    constexpr int js = S & 3;
    constexpr int m_off = (S < 4) ? 32 : 0;  // S=0..3 prepares g=1 of cur meta_kb; S=4..7 prepares g=0 of next meta_kb

    if constexpr (S == 0) {
        es::prefetch<uint32_t, 64>(wp_nxt + 1024u, 0u, kCC);
        es::prefetch<uint32_t, 64>(dxp_nxt, 0u, kCC);
        es::prefetch<uint32_t, 64>(dxp_nxt + 64u, 0u, kCC);
    }

    const int* xp_s = xp + static_cast<size_t>(S) * 256u;
    simd<int, 64> a0 = es::block_load<int, 64>(xp_s);
    simd<int, 64> a1 = es::block_load<int, 64>(xp_s + 64u);
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        i_act[i][0] = xmx::dpas<8, 8, int, int, int, int, DT::u4, DT::s8>(i_act[i][0], wr[u][i], a0);
    }
    a0 = es::block_load<int, 64>(xp_s + 128u);
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        i_act[i][1] = xmx::dpas<8, 8, int, int, int, int, DT::u4, DT::s8>(i_act[i][1], wr[u][i], a1);
    }
    a1 = es::block_load<int, 64>(xp_s + 192u);
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        i_act[i][2] = xmx::dpas<8, 8, int, int, int, int, DT::u4, DT::s8>(i_act[i][2], wr[u][i], a0);
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        i_act[i][3] = xmx::dpas<8, 8, int, int, int, int, DT::u4, DT::s8>(i_act[i][3], wr[u][i], a1);
    }

    if constexpr (S < 6) {
        constexpr uint32_t w_off = static_cast<uint32_t>(S + 2) * 128u;
        wr[u][0] = es::block_load<int, 64>(reinterpret_cast<const int*>(wp + w_off));
        wr[u][1] = es::block_load<int, 64>(reinterpret_cast<const int*>(wp + w_off + 64u));
    } else {
        constexpr uint32_t w_off = static_cast<uint32_t>(S - 6) * 128u;
        wr[u][0] = es::block_load<int, 64>(reinterpret_cast<const int*>(wp_nxt + w_off));
        wr[u][1] = es::block_load<int, 64>(reinterpret_cast<const int*>(wp_nxt + w_off + 64u));
    }

    const int* xp_pa = (S < 7) ? (xp + static_cast<size_t>(S + 1) * 256u) : xp_nxt;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        es::prefetch<int, 64>(xp_pa + static_cast<uint32_t>(j) * 64u, 0u, kCC);
    }

    if constexpr (S < 4) {
        constexpr uint32_t pf_off = static_cast<uint32_t>(S + 4) * 128u;
        es::prefetch<uint32_t, 64>(wp + pf_off, 0u, kCC);
        es::prefetch<uint32_t, 64>(wp + pf_off + 64u, 0u, kCC);
    } else {
        constexpr uint32_t pf_off = static_cast<uint32_t>(S - 4) * 128u;
        es::prefetch<uint32_t, 64>(wp_nxt + pf_off, 0u, kCC);
        es::prefetch<uint32_t, 64>(wp_nxt + pf_off + 64u, 0u, kCC);
    }

    const simd<float, 8> dxs_v = dx_scale.template select<8, 1>(8 * js);
    const simd<int, 8>   sxn_v = dx_nxt.template select<8, 1>(16 * js + 8).template bit_cast_view<int>();
    simd<uint32_t, 16> m0_u = meta_kb.template select<16, 1>(m_off);
    simd<uint32_t, 16> m1_u = meta_kb.template select<16, 1>(m_off + 16);
    const simd<int16_t, 16> nz0 = simd<int16_t, 16>(0) - simd<int16_t, 16>(
        m0_u.template bit_cast_view<uint16_t>().template select<16, 2>(1));
    const simd<int16_t, 16> nz1 = simd<int16_t, 16>(0) - simd<int16_t, 16>(
        m1_u.template bit_cast_view<uint16_t>().template select<16, 2>(1));
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

void launch_w4i8_pp3_v2(sycl::queue& q, const uint32_t* w_base, const int* xq8_base,
                        const uint32_t* dxT_base, float* y_base,
                        int K, int R, int T) {
    const uint32_t n_rt      = static_cast<uint32_t>(R) / 16u;
    const uint32_t n_rt32    = (n_rt + 1u) / 2u;
    const uint32_t n_kb      = static_cast<uint32_t>(K) / 256u;
    const uint32_t n_grp     = n_kb * 2u;
    const uint32_t n_steps   = n_kb * 8u;
    const uint32_t n_ct32    = static_cast<uint32_t>(T + 31) / 32u;
    const BenchGrid g        = make_grid(n_rt32, n_ct32);
    const uint32_t n_wgc_tot = (n_ct32 + g.CT - 1u) / g.CT;
    const uint32_t nv        = static_cast<uint32_t>(K) | 0x80000000u;
    const uint32_t ct_mask   = g.CT - 1u;

    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagW4I8Pp3V2>(
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
                    const uint32_t rt32 = (wgr << g.lgRT) + ri;
                    const uint32_t rt0  = rt32 * 2u;
                    const uint32_t rt1  = rt0 + 1u;
                    const uint32_t ct32 = (wgc << g.lgCT) + ci;
                    if (wgc < n_wgc_tot && rt0 < n_rt && ct32 < n_ct32) {
                        const uint32_t* wp  = w_base   + static_cast<size_t>(rt32) * static_cast<size_t>(n_kb) * 1088u;
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
                        wr[0][0] = es::block_load<int, 64>(reinterpret_cast<const int*>(wp));
                        wr[0][1] = es::block_load<int, 64>(reinterpret_cast<const int*>(wp + 64u));
                        wr[1][0] = es::block_load<int, 64>(reinterpret_cast<const int*>(wp + 128u));
                        wr[1][1] = es::block_load<int, 64>(reinterpret_cast<const int*>(wp + 192u));

                        #pragma unroll
                        for (int s = 0; s < 2; ++s) {
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                es::prefetch<int, 64>(xp + static_cast<uint32_t>(s) * 256u + static_cast<uint32_t>(j) * 64u, 0u, kCC);
                            }
                        }
                        #pragma unroll
                        for (int pf = 2; pf <= 5; ++pf) {
                            es::prefetch<uint32_t, 64>(wp + static_cast<uint32_t>(pf) * 128u, 0u, kCC);
                            es::prefetch<uint32_t, 64>(wp + static_cast<uint32_t>(pf) * 128u + 64u, 0u, kCC);
                        }
                        es::prefetch<uint32_t, 64>(wp + 1024u, 0u, kCC);
                        es::prefetch<uint32_t, 64>(dxp, 0u, kCC);
                        es::prefetch<uint32_t, 64>(dxp + 64u, 0u, kCC);

                        simd<uint32_t, 64> meta_kb = es::block_load<uint32_t, 64>(wp + 1024u, kAl16);
                        simd<uint32_t, 64> dx_nxt  = es::block_load<uint32_t, 64>(dxp, kAl16);

                        simd<sycl::half, 32> s_dpas_h;
                        simd<float, 16> s_scale[2];
                        simd<float, 32> dx_dpas, dx_scale = 0.0f;
                        s_scale[0] = 0.0f;
                        s_scale[1] = 0.0f;
                        s_dpas_h.template select<16, 1>(0)  = half_raw_of(meta_kb.template select<16, 1>(0));
                        s_dpas_h.template select<16, 1>(16) = half_raw_of(meta_kb.template select<16, 1>(16));
                        {
                            simd<uint32_t, 16> m0_u = meta_kb.template select<16, 1>(0);
                            simd<uint32_t, 16> m1_u = meta_kb.template select<16, 1>(16);
                            const simd<int16_t, 16> nz0 = simd<int16_t, 16>(0) - simd<int16_t, 16>(
                                m0_u.template bit_cast_view<uint16_t>().template select<16, 2>(1));
                            const simd<int16_t, 16> nz1 = simd<int16_t, 16>(0) - simd<int16_t, 16>(
                                m1_u.template bit_cast_view<uint16_t>().template select<16, 2>(1));
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

                        dx_nxt = es::block_load<uint32_t, 64>(dxp + 64u, kAl16);

                        #pragma nounroll
                        for (uint32_t kb = 0; kb < n_kb; ++kb) {
                            const bool has_nxt      = (kb + 1u < n_kb);
                            const uint32_t* wp_nxt  = has_nxt ? (wp  + 1088u) : wp;
                            const int*      xp_nxt  = has_nxt ? (xp  + 2048u) : xp;
                            const uint32_t* dxp_nxt = has_nxt ? (dxp + 128u)  : dxp;

                            pp3_v2_substep<0>(i0, i1, c_acc, wr, s_scale, dx_scale, meta_kb, dx_nxt,
                                              wp, wp_nxt, xp, xp_nxt, dxp_nxt, y_base, kb, nv);
                            pp3_v2_substep<1>(i0, i1, c_acc, wr, s_scale, dx_scale, meta_kb, dx_nxt,
                                              wp, wp_nxt, xp, xp_nxt, dxp_nxt, y_base, kb, nv);
                            pp3_v2_substep<2>(i0, i1, c_acc, wr, s_scale, dx_scale, meta_kb, dx_nxt,
                                              wp, wp_nxt, xp, xp_nxt, dxp_nxt, y_base, kb, nv);
                            pp3_v2_substep<3>(i0, i1, c_acc, wr, s_scale, dx_scale, meta_kb, dx_nxt,
                                              wp, wp_nxt, xp, xp_nxt, dxp_nxt, y_base, kb, nv);

                            s_scale[0] = simd<float, 16>(s_dpas_h.template select<16, 1>(0));
                            s_scale[1] = simd<float, 16>(s_dpas_h.template select<16, 1>(16));
                            dx_scale   = dx_dpas;
                            s_dpas_h.template select<16, 1>(0)  = half_raw_of(meta_kb.template select<16, 1>(32));
                            s_dpas_h.template select<16, 1>(16) = half_raw_of(meta_kb.template select<16, 1>(48));
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                dx_dpas.template select<8, 1>(8 * j) =
                                    dx_nxt.template select<8, 1>(16 * j).template bit_cast_view<float>();
                            }
                            meta_kb = es::block_load<uint32_t, 64>(wp_nxt + 1024u, kAl16);
                            dx_nxt  = es::block_load<uint32_t, 64>(dxp_nxt, kAl16);

                            pp3_v2_substep<4>(i1, i0, c_acc, wr, s_scale, dx_scale, meta_kb, dx_nxt,
                                              wp, wp_nxt, xp, xp_nxt, dxp_nxt, y_base, kb, nv);
                            pp3_v2_substep<5>(i1, i0, c_acc, wr, s_scale, dx_scale, meta_kb, dx_nxt,
                                              wp, wp_nxt, xp, xp_nxt, dxp_nxt, y_base, kb, nv);
                            pp3_v2_substep<6>(i1, i0, c_acc, wr, s_scale, dx_scale, meta_kb, dx_nxt,
                                              wp, wp_nxt, xp, xp_nxt, dxp_nxt, y_base, kb, nv);
                            pp3_v2_substep<7>(i1, i0, c_acc, wr, s_scale, dx_scale, meta_kb, dx_nxt,
                                              wp, wp_nxt, xp, xp_nxt, dxp_nxt, y_base, kb, nv);

                            s_scale[0] = simd<float, 16>(s_dpas_h.template select<16, 1>(0));
                            s_scale[1] = simd<float, 16>(s_dpas_h.template select<16, 1>(16));
                            dx_scale   = dx_dpas;
                            s_dpas_h.template select<16, 1>(0)  = half_raw_of(meta_kb.template select<16, 1>(0));
                            s_dpas_h.template select<16, 1>(16) = half_raw_of(meta_kb.template select<16, 1>(16));
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                dx_dpas.template select<8, 1>(8 * j) =
                                    dx_nxt.template select<8, 1>(16 * j).template bit_cast_view<float>();
                            }
                            if (has_nxt) {
                                dx_nxt = es::block_load<uint32_t, 64>(dxp_nxt + 64u, kAl16);
                            }
                            wp  += 1088u;
                            xp  += 2048u;
                            dxp += 128u;
                        }

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

}  // namespace

void convert_b70w4_to_fg(sycl::queue& q, const uint32_t* src_v1, uint32_t* dst_fg,
                         int n_in, int n_out, int n_mat) {
    const size_t n_rt32  = static_cast<size_t>(n_out) / 32u;
    const size_t n_kb    = static_cast<size_t>(n_in)  / 256u;
    const size_t n_steps = n_kb * 8u;
    const size_t per_mat = n_rt32 * n_steps * 128u;
    const size_t per_v1  = n_rt32 * 2u * n_kb * 544u;
    const size_t tot     = static_cast<size_t>(n_mat) * per_mat;
    q.parallel_for<TagConvFg>(sycl::range<1>(tot), [=](sycl::id<1> id) {
        const size_t idx  = id[0];
        const size_t m    = idx / per_mat;
        const size_t rem  = idx - m * per_mat;
        const size_t rt32 = rem / (n_steps * 128u);
        const size_t rem2 = rem - rt32 * (n_steps * 128u);
        const size_t step = rem2 >> 7;
        const size_t in_s = rem2 & 127u;
        const size_t i    = in_s >> 6;
        const size_t dw   = in_s & 63u;
        const size_t kb   = step >> 3;
        const size_t s    = step & 7u;
        const size_t rt   = rt32 * 2u + i;
        dst_fg[idx] = src_v1[m * per_v1 + (rt * n_kb + kb) * 544u + s * 64u + dw];
    }).wait();
}

void convert_b70w4_to_v2(sycl::queue& q, const uint32_t* src_v1, uint32_t* dst_v2,
                         int n_in, int n_out, int n_mat) {
    const size_t n_rt32  = static_cast<size_t>(n_out) / 32u;
    const size_t n_kb    = static_cast<size_t>(n_in)  / 256u;
    const size_t per_mat = n_rt32 * n_kb * 1088u;
    const size_t tot     = static_cast<size_t>(n_mat) * per_mat;
    q.parallel_for<TagConvV2>(sycl::range<1>(tot), [=](sycl::id<1> id) {
        const size_t idx  = id[0];
        const size_t m    = idx / per_mat;
        const size_t rem  = idx - m * per_mat;
        const size_t rt32 = rem / (n_kb * 1088u);
        const size_t rem2 = rem - rt32 * (n_kb * 1088u);
        const size_t kb   = rem2 / 1088u;
        const size_t off  = rem2 - kb * 1088u;
        const size_t v1_b = m * per_mat;
        if (off < 1024u) {
            const size_t s   = off >> 7;
            const size_t in  = off & 127u;
            const size_t i   = in >> 6;
            const size_t dw  = in & 63u;
            const size_t rt  = rt32 * 2u + i;
            dst_v2[idx] = src_v1[v1_b + (rt * n_kb + kb) * 544u + s * 64u + dw];
        } else {
            const size_t mo  = off - 1024u;
            const size_t grp = mo >> 5;
            const size_t i   = (mo >> 4) & 1u;
            const size_t dw  = mo & 15u;
            const size_t rt  = rt32 * 2u + i;
            dst_v2[idx] = src_v1[v1_b + (rt * n_kb + kb) * 544u + 512u + grp * 16u + dw];
        }
    }).wait();
}

void launch_fatgemm_u4i8_ref(sycl::queue& q, const uint32_t* W_fg, const int* xq8,
                             int* Y_out, int K, int R, int T, bool slack_ok) {
    const uint32_t n_rt32    = static_cast<uint32_t>(R) / 32u;
    const uint32_t n_steps   = static_cast<uint32_t>(K) / 32u;
    const uint32_t n_ct32    = static_cast<uint32_t>(T + 31) / 32u;
    const BenchGrid g        = make_grid(n_rt32, n_ct32);
    const uint32_t n_wgc_tot = (n_ct32 + g.CT - 1u) / g.CT;
    const uint32_t ct_mask   = g.CT - 1u;
    const uint32_t last_s    = n_steps - 1u;
    const uint32_t nv        = static_cast<uint32_t>(K) | 0x80000000u;

    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagFgPureRef>(
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
                    const uint32_t rt   = (wgr << g.lgRT) + ri;
                    const uint32_t ct   = (wgc << g.lgCT) + ci;
                    if (wgc < n_wgc_tot && rt < n_rt32 && ct < n_ct32) {
                        const uint32_t* wp = W_fg + static_cast<size_t>(rt) * n_steps * 128u;
                        const int*      xp = xq8  + static_cast<size_t>(ct) * n_steps * 256u;

                        simd<int, 64> wr[2][2];
                        #pragma unroll
                        for (int u = 0; u < 2; ++u) {
                            #pragma unroll
                            for (int i = 0; i < 2; ++i) {
                                wr[u][i] = es::block_load<int, 64>(reinterpret_cast<const int*>(wp + (u * 2 + i) * 64u));
                            }
                        }
                        simd<int, 128> acc[2][4];
                        #pragma unroll
                        for (int i = 0; i < 2; ++i)
                            #pragma unroll
                            for (int j = 0; j < 4; ++j)
                                acc[i][j] = 0;

                        #pragma nounroll
                        for (uint32_t s = 0; s < n_steps; s += 2u) {
                            #pragma unroll
                            for (int u = 0; u < 2; ++u) {
                                const uint32_t si = s + static_cast<uint32_t>(u);
                                #pragma unroll
                                for (int j = 0; j < 4; ++j) {
                                    simd<int, 64> a = es::block_load<int, 64>(xp + (static_cast<size_t>(si) * 4u + j) * 64u);
                                    #pragma unroll
                                    for (int i = 0; i < 2; ++i) {
                                        acc[i][j] = xmx::dpas<8, 8, int, int, int, int, DT::u4, DT::s8>(acc[i][j], wr[u][i], a);
                                    }
                                }
                                const uint32_t sn = slack_ok ? (si + 2u) : ((si + 2u <= last_s) ? (si + 2u) : last_s);
                                #pragma unroll
                                for (int i = 0; i < 2; ++i) {
                                    wr[u][i] = es::block_load<int, 64>(reinterpret_cast<const int*>(wp + (static_cast<size_t>(sn) * 2u + i) * 64u));
                                }
                                const uint32_t sp = slack_ok ? (si + 1u) : ((si + 1u <= last_s) ? (si + 1u) : last_s);
                                #pragma unroll
                                for (int j = 0; j < 4; ++j) {
                                    es::prefetch<int, 64>(xp + (static_cast<size_t>(sp) * 4u + j) * 64u, 0u, kCC);
                                }
                                const uint32_t sw = slack_ok ? (si + 4u) : ((si + 4u <= last_s) ? (si + 4u) : last_s);
                                #pragma unroll
                                for (int i = 0; i < 2; ++i) {
                                    es::prefetch<uint32_t, 64>(wp + (static_cast<size_t>(sw) * 2u + i) * 64u, 0u, kCC);
                                }
                                if (si == nv) Y_out[0] = 0;
                            }
                        }

                        int* yp = Y_out + (static_cast<size_t>(ct) * n_rt32 + rt) * 1024u;
                        #pragma unroll
                        for (int i = 0; i < 2; ++i) {
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                #pragma unroll
                                for (int r = 0; r < 8; ++r) {
                                    es::block_store<int, 16>(yp + (i * 4 + j) * 128 + 16 * r, acc[i][j].template select<16, 1>(16 * r));
                                }
                            }
                        }
                    }
                }
            });
    });
}

bool verify_fatgemm_u4i8_ref(sycl::queue& q) {
    constexpr int R = 512, K = 512, C = 512;
    const size_t n_rt32 = R / 32, n_steps = K / 32, n_ct32 = C / 32;
    const size_t w_dw = n_rt32 * n_steps * 128u;
    const size_t x_dw = n_ct32 * n_steps * 256u;
    const size_t y_dw = static_cast<size_t>(R) * C;

    std::vector<uint32_t> hW(w_dw);
    std::vector<int32_t>  hX(x_dw), hY(y_dw);
    std::mt19937_64 rng(0xB70F6ULL);
    for (auto& v : hW) v = static_cast<uint32_t>(rng());
    for (auto& v : hX) v = static_cast<int32_t>(rng());

    uint32_t* dW = sycl::aligned_alloc_device<uint32_t>(4096, w_dw + 4096, q);
    int*      dX = sycl::aligned_alloc_device<int>(4096, x_dw + 4096, q);
    int*      dY = sycl::aligned_alloc_device<int>(4096, y_dw + 4096, q);
    q.memcpy(dW, hW.data(), w_dw * 4);
    q.memcpy(dX, hX.data(), x_dw * 4);
    q.wait();
    launch_fatgemm_u4i8_ref(q, dW, dX, dY, K, R, C, true);
    q.wait();
    q.memcpy(hY.data(), dY, y_dw * 4).wait();
    sycl::free(dW, q); sycl::free(dX, q); sycl::free(dY, q);

    for (uint32_t r : {0u, 17u, 127u, 255u, 511u}) {
        for (uint32_t c : {0u, 7u, 31u, 259u, 511u}) {
            const uint32_t rt = r / 32u, i = (r / 16u) & 1u, lane = r & 15u;
            const uint32_t ct = c / 32u, j = (c / 8u) & 3u,  rr = c & 7u;
            int64_t acc = 0;
            for (uint32_t k = 0; k < static_cast<uint32_t>(K); ++k) {
                const uint32_t s = k >> 5, kk = k & 31u;
                const uint32_t wv = (hW[((rt * n_steps + s) * 2u + i) * 64u + (kk >> 3) * 16u + lane] >> ((kk & 7u) * 4u)) & 15u;
                const int32_t  xv = static_cast<int8_t>(hX[((ct * n_steps + s) * 4u + j) * 64u + rr * 8u + (kk >> 2)] >> ((kk & 3u) * 8u));
                acc += static_cast<int64_t>(wv) * xv;
            }
            const int32_t got = hY[((ct * n_rt32 + rt) * 8u + i * 4u + j) * 128u + rr * 16u + lane];
            if (acc != static_cast<int64_t>(got)) return false;
        }
    }
    return true;
}

bool gemm_w4_i8_pp3_v2(const void* weights_v2, const void* xq8, const float* dxT, float* y,
                       int n_in, int n_out, int T, void* stream) {
    if (n_in <= 0 || n_out <= 0 || T <= 0 || (n_in % 256) != 0 || (n_out % 32) != 0) return false;
    if (!weights_v2 || !xq8 || !dxT || !y) return false;
    if ((reinterpret_cast<uintptr_t>(weights_v2) & 15u) != 0u) return false;
    if ((reinterpret_cast<uintptr_t>(xq8) & 15u) != 0u) return false;
    if ((reinterpret_cast<uintptr_t>(dxT) & 15u) != 0u) return false;
    if ((reinterpret_cast<uintptr_t>(y) & 15u) != 0u) return false;
    launch_w4i8_pp3_v2(*strata::q_of(stream),
                       static_cast<const uint32_t*>(weights_v2),
                       static_cast<const int*>(xq8),
                       reinterpret_cast<const uint32_t*>(dxT),
                       y, n_in, n_out, T);
    return true;
}

}  // namespace strata::q36pp::bench
