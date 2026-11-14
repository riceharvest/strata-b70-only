#include "q36_fmha_i8.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cmath>
#include <cstdint>

namespace strata::q36fmha {
namespace {

namespace es = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using namespace sycl::ext::intel::esimd;

constexpr int kHD = 256;
constexpr int kG = 6;
constexpr int kBT = 16;
constexpr int kWG = 16;
constexpr float kLog2e = 1.4426950408889634f;
constexpr auto kAl16 = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::alignment<16>};
constexpr auto kPf = sycl::ext::oneapi::experimental::properties{
    cache_hint_L1<cache_hint::cached>, cache_hint_L2<cache_hint::cached>};

SYCL_ESIMD_FUNCTION inline float hmax64_f(simd<float, 64> v) {
    simd<float, 32> a = (es::max)(simd<float, 32>(v.template select<32, 1>(0)), simd<float, 32>(v.template select<32, 1>(32)));
    simd<float, 16> b = (es::max)(simd<float, 16>(a.template select<16, 1>(0)), simd<float, 16>(a.template select<16, 1>(16)));
    simd<float, 8> c = (es::max)(simd<float, 8>(b.template select<8, 1>(0)), simd<float, 8>(b.template select<8, 1>(8)));
    simd<float, 4> d = (es::max)(simd<float, 4>(c.template select<4, 1>(0)), simd<float, 4>(c.template select<4, 1>(4)));
    simd<float, 2> e = (es::max)(simd<float, 2>(d.template select<2, 1>(0)), simd<float, 2>(d.template select<2, 1>(2)));
    return fmaxf(float(e[0]), float(e[1]));
}

// Parallel cross-row max reduction for 6 rows x 32 lanes (eliminates 6x serial hmax32 scalar chains).
SYCL_ESIMD_FUNCTION inline simd<float, 8> hmax6x32(
    simd<float, 32> r0, simd<float, 32> r1, simd<float, 32> r2,
    simd<float, 32> r3, simd<float, 32> r4, simd<float, 32> r5) {
    simd<float, 16> a0 = (es::max)(simd<float, 16>(r0.template select<16, 1>(0)), simd<float, 16>(r0.template select<16, 1>(16)));
    simd<float, 16> a1 = (es::max)(simd<float, 16>(r1.template select<16, 1>(0)), simd<float, 16>(r1.template select<16, 1>(16)));
    simd<float, 16> a2 = (es::max)(simd<float, 16>(r2.template select<16, 1>(0)), simd<float, 16>(r2.template select<16, 1>(16)));
    simd<float, 16> a3 = (es::max)(simd<float, 16>(r3.template select<16, 1>(0)), simd<float, 16>(r3.template select<16, 1>(16)));
    simd<float, 16> a4 = (es::max)(simd<float, 16>(r4.template select<16, 1>(0)), simd<float, 16>(r4.template select<16, 1>(16)));
    simd<float, 16> a5 = (es::max)(simd<float, 16>(r5.template select<16, 1>(0)), simd<float, 16>(r5.template select<16, 1>(16)));
    simd<float, 16> x0, y0, x1, y1, x2, y2;
    x0.template select<8, 1>(0) = a0.template select<8, 1>(0); x0.template select<8, 1>(8) = a1.template select<8, 1>(0);
    y0.template select<8, 1>(0) = a0.template select<8, 1>(8); y0.template select<8, 1>(8) = a1.template select<8, 1>(8);
    x1.template select<8, 1>(0) = a2.template select<8, 1>(0); x1.template select<8, 1>(8) = a3.template select<8, 1>(0);
    y1.template select<8, 1>(0) = a2.template select<8, 1>(8); y1.template select<8, 1>(8) = a3.template select<8, 1>(8);
    x2.template select<8, 1>(0) = a4.template select<8, 1>(0); x2.template select<8, 1>(8) = a5.template select<8, 1>(0);
    y2.template select<8, 1>(0) = a4.template select<8, 1>(8); y2.template select<8, 1>(8) = a5.template select<8, 1>(8);
    simd<float, 16> b0 = (es::max)(x0, y0), b1 = (es::max)(x1, y1), b2 = (es::max)(x2, y2);
    simd<float, 16> px, py;
    px.template select<4, 1>(0) = b0.template select<4, 1>(0);  px.template select<4, 1>(4) = b0.template select<4, 1>(8);
    px.template select<4, 1>(8) = b1.template select<4, 1>(0);  px.template select<4, 1>(12) = b1.template select<4, 1>(8);
    py.template select<4, 1>(0) = b0.template select<4, 1>(4);  py.template select<4, 1>(4) = b0.template select<4, 1>(12);
    py.template select<4, 1>(8) = b1.template select<4, 1>(4);  py.template select<4, 1>(12) = b1.template select<4, 1>(12);
    simd<float, 16> c03 = (es::max)(px, py);
    simd<float, 8> p2x, p2y;
    p2x.template select<4, 1>(0) = b2.template select<4, 1>(0); p2x.template select<4, 1>(4) = b2.template select<4, 1>(8);
    p2y.template select<4, 1>(0) = b2.template select<4, 1>(4); p2y.template select<4, 1>(4) = b2.template select<4, 1>(12);
    simd<float, 8> c45 = (es::max)(p2x, p2y);
    simd<float, 16> dx(-1e30f), dy(-1e30f);
    dx.template select<8, 1>(0) = c03.template select<8, 2>(0); dx.template select<4, 1>(8) = c45.template select<4, 2>(0);
    dy.template select<8, 1>(0) = c03.template select<8, 2>(1); dy.template select<4, 1>(8) = c45.template select<4, 2>(1);
    simd<float, 16> d05 = (es::max)(dx, dy);
    return (es::max)(simd<float, 8>(d05.template select<8, 2>(0)), simd<float, 8>(d05.template select<8, 2>(1)));
}

// Parallel cross-row sum reduction for 6 rows x 32 lanes.
SYCL_ESIMD_FUNCTION inline simd<float, 8> hsum6x32(
    simd<float, 32> r0, simd<float, 32> r1, simd<float, 32> r2,
    simd<float, 32> r3, simd<float, 32> r4, simd<float, 32> r5) {
    simd<float, 16> a0 = r0.template select<16, 1>(0) + r0.template select<16, 1>(16);
    simd<float, 16> a1 = r1.template select<16, 1>(0) + r1.template select<16, 1>(16);
    simd<float, 16> a2 = r2.template select<16, 1>(0) + r2.template select<16, 1>(16);
    simd<float, 16> a3 = r3.template select<16, 1>(0) + r3.template select<16, 1>(16);
    simd<float, 16> a4 = r4.template select<16, 1>(0) + r4.template select<16, 1>(16);
    simd<float, 16> a5 = r5.template select<16, 1>(0) + r5.template select<16, 1>(16);
    simd<float, 16> x0, y0, x1, y1, x2, y2;
    x0.template select<8, 1>(0) = a0.template select<8, 1>(0); x0.template select<8, 1>(8) = a1.template select<8, 1>(0);
    y0.template select<8, 1>(0) = a0.template select<8, 1>(8); y0.template select<8, 1>(8) = a1.template select<8, 1>(8);
    x1.template select<8, 1>(0) = a2.template select<8, 1>(0); x1.template select<8, 1>(8) = a3.template select<8, 1>(0);
    y1.template select<8, 1>(0) = a2.template select<8, 1>(8); y1.template select<8, 1>(8) = a3.template select<8, 1>(8);
    x2.template select<8, 1>(0) = a4.template select<8, 1>(0); x2.template select<8, 1>(8) = a5.template select<8, 1>(0);
    y2.template select<8, 1>(0) = a4.template select<8, 1>(8); y2.template select<8, 1>(8) = a5.template select<8, 1>(8);
    simd<float, 16> b0 = x0 + y0, b1 = x1 + y1, b2 = x2 + y2;
    simd<float, 16> px, py;
    px.template select<4, 1>(0) = b0.template select<4, 1>(0);  px.template select<4, 1>(4) = b0.template select<4, 1>(8);
    px.template select<4, 1>(8) = b1.template select<4, 1>(0);  px.template select<4, 1>(12) = b1.template select<4, 1>(8);
    py.template select<4, 1>(0) = b0.template select<4, 1>(4);  py.template select<4, 1>(4) = b0.template select<4, 1>(12);
    py.template select<4, 1>(8) = b1.template select<4, 1>(4);  py.template select<4, 1>(12) = b1.template select<4, 1>(12);
    simd<float, 16> c03 = px + py;
    simd<float, 8> p2x, p2y;
    p2x.template select<4, 1>(0) = b2.template select<4, 1>(0); p2x.template select<4, 1>(4) = b2.template select<4, 1>(8);
    p2y.template select<4, 1>(0) = b2.template select<4, 1>(4); p2y.template select<4, 1>(4) = b2.template select<4, 1>(12);
    simd<float, 8> c45 = p2x + p2y;
    simd<float, 16> dx(0.0f), dy(0.0f);
    dx.template select<8, 1>(0) = c03.template select<8, 2>(0); dx.template select<4, 1>(8) = c45.template select<4, 2>(0);
    dy.template select<8, 1>(0) = c03.template select<8, 2>(1); dy.template select<4, 1>(8) = c45.template select<4, 2>(1);
    simd<float, 16> d05 = dx + dy;
    return d05.template select<8, 2>(0) + d05.template select<8, 2>(1);
}

SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> pack_f32_to_s8(simd<float, 64> x, float inv_sc) {
    simd<float, 64> y = x * inv_sc;
    simd<float, 64> rnd = 0.5f;
    rnd.merge(simd<float, 64>(-0.5f), y < 0.0f);
    simd<int, 64> ci = (es::min)((es::max)(convert<int>(y + rnd), simd<int, 64>(-127)), simd<int, 64>(127));
    simd<uint32_t, 64> cu = ci.template bit_cast_view<uint32_t>();
    return (cu.template select<16, 4>(0) & 255u) | ((cu.template select<16, 4>(1) & 255u) << 8) |
           ((cu.template select<16, 4>(2) & 255u) << 16) | ((cu.template select<16, 4>(3) & 255u) << 24);
}

SYCL_ESIMD_FUNCTION inline simd<int, 8> pack_p32_u8fast(simd<float, 32> pv, float inv) {
    simd<uint8_t, 32> q = convert<uint8_t>(es::min(pv * inv + 0.5f, simd<float, 32>(255.0f)));
    return q.template bit_cast_view<int>();
}

SYCL_ESIMD_FUNCTION inline void pack_p32_u16(simd<float, 32> pv, float inv_u16, simd<int, 8>& hi, simd<int, 8>& lo) {
    simd<uint32_t, 32> q = convert<uint32_t>(es::min(pv * inv_u16 + 0.5f, simd<float, 32>(65535.0f)));
    simd<uint32_t, 32> qh = q >> 8, ql = q & 255u;
    simd<uint32_t, 8> phi = qh.template select<8, 4>(0) | (qh.template select<8, 4>(1) << 8) |
                            (qh.template select<8, 4>(2) << 16) | (qh.template select<8, 4>(3) << 24);
    simd<uint32_t, 8> plo = ql.template select<8, 4>(0) | (ql.template select<8, 4>(1) << 8) |
                            (ql.template select<8, 4>(2) << 16) | (ql.template select<8, 4>(3) << 24);
    hi = phi.template bit_cast_view<int>();
    lo = plo.template bit_cast_view<int>();
}

template <bool U16P>
struct TagFmhaI8;

// 1-token x 6-head per thread (T1H6) int8 XMX FlashAttention:
// eliminates SLM_S and SLM_P completely by keeping all HD=256 Q and O slices register-resident,
// and ping-pongs SLM_VB (kBC=64 for u8, kBC=32 for u16_p) for a single barrier per KV block.
template <bool U16P>
void launch_fmha_i8(sycl::queue& q, int vt, const sycl::half* q16, const int8_t* kc, const int8_t* vc, float* out,
                    int T, int p0, int H, int HKV, int ctx) {
    constexpr int kBC = U16P ? 32 : 64;
    constexpr uint32_t kSlmHalf = U16P ? 8192u : 16384u;
    const int nqb = (T + kBT - 1) / kBT;
    const size_t vtb = (size_t)(vt / 32) * (size_t)kHD * 32u;
    const int* k_dw_base = reinterpret_cast<const int*>(kc);
    const uint16_t* ks_base = reinterpret_cast<const uint16_t*>(kc + (size_t)HKV * ctx * kHD);
    const uint32_t* v_dw_base = reinterpret_cast<const uint32_t*>(vc);
    const float* vs_base = reinterpret_cast<const float*>(vc + (size_t)HKV * vtb);
    const float qk_scale = 1.0f / std::sqrt((float)kHD);
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};

    q.submit([&](sycl::handler& h) {
        sycl::nd_range<2> nd(sycl::range<2>((size_t)HKV * nqb, kWG), sycl::range<2>(1, kWG));
        h.parallel_for<TagFmhaI8<U16P>>(nd, props, [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
            slm_init<kSlmHalf * 2u>();

            const int g_idx = (int)it.get_group(0), tid = (int)it.get_local_id(1);
            const int kvh = g_idx / nqb, qb = g_idx % nqb;
            const int t_wg = qb * kBT, tok = t_wg + tid;
            const int tok_clamp = (tok < T) ? tok : (T - 1);
            const int wg_t1 = (t_wg + kBT < T) ? (t_wg + kBT) : T;
            const int kv_end = p0 + wg_t1;
            const unsigned surf_h = (unsigned)(kv_end - 1);
            const int causal_lim = (tok < T) ? ((p0 + tok < kv_end - 1) ? (p0 + tok) : (kv_end - 1)) : -1;
            const int fast_end = (t_wg + kBT <= T) ? ((p0 + t_wg) & ~(kBC - 1)) : 0;

            const int* k_kv = k_dw_base + (size_t)kvh * ctx * 64u;
            const uint16_t* ks_kv = ks_base + (size_t)kvh * ctx * 4u;
            const uint32_t* v_kv = v_dw_base + (size_t)kvh * (vtb / 4u);
            const float* vs_kv = vs_base + (size_t)kvh * vt;

            simd<int, 64> Q_A[8];
            simd<float, 8> q_sc[4];
#pragma unroll
            for (int ds = 0; ds < 8; ++ds) Q_A[ds] = 0;
#pragma unroll
            for (int qtr = 0; qtr < 4; ++qtr) {
                q_sc[qtr] = 0.0f;
#pragma unroll
                for (int hl = 0; hl < kG; ++hl) {
                    const int qh = kvh * kG + hl;
                    const sycl::half* qrow = q16 + ((size_t)tok_clamp * H + qh) * kHD + qtr * 64;
                    simd<uint32_t, 32> raw = block_load<uint32_t, 32>(reinterpret_cast<const uint32_t*>(qrow), kAl16);
                    simd<float, 64> xf = raw.template bit_cast_view<sycl::half>();
                    float amax = hmax64_f(es::abs(xf));
                    q_sc[qtr][hl] = ((amax > 0.0f) ? (amax * (1.0f / 127.0f)) : 1.0f) * qk_scale;
                    simd<uint32_t, 16> pk = pack_f32_to_s8(xf, (amax > 0.0f) ? (127.0f / amax) : 0.0f);
                    Q_A[qtr * 2 + 0].template select<8, 1>(8 * hl) = pk.template select<8, 1>(0).template bit_cast_view<int>();
                    Q_A[qtr * 2 + 1].template select<8, 1>(8 * hl) = pk.template select<8, 1>(8).template bit_cast_view<int>();
                }
            }

            simd<float, 96> O[16];
#pragma unroll
            for (int dl = 0; dl < 16; ++dl) O[dl] = 0.0f;
            simd<float, 8> m_my(-1e30f), l_my(0.0f);

            int buf = 0;
            int blk_c0 = 0;

            // Unmasked causal fast path (all 16 tokens x kBC cols strictly inside causal prefix)
            for (; blk_c0 < fast_end; blk_c0 += kBC) {
                if constexpr (!U16P) {
                    const uint32_t* vp0 = v_kv + ((size_t)(blk_c0 >> 5) * 256u + (size_t)tid * 16u) * 8u;
                    const uint32_t* vp1 = vp0 + 256u * 8u;
                    simd<uint32_t, 128> vl0, vl1, vb0, vb1;
                    vl0.template select<64, 1>(0) = block_load<uint32_t, 64>(vp0, kAl16);
                    vl0.template select<64, 1>(64) = block_load<uint32_t, 64>(vp0 + 64, kAl16);
                    vl1.template select<64, 1>(0) = block_load<uint32_t, 64>(vp1, kAl16);
                    vl1.template select<64, 1>(64) = block_load<uint32_t, 64>(vp1 + 64, kAl16);
#pragma unroll
                    for (int dw = 0; dw < 8; ++dw) {
                        vb0.template select<16, 1>(16 * dw) = vl0.template select<16, 8>(dw);
                        vb1.template select<16, 1>(16 * dw) = vl1.template select<16, 8>(dw);
                    }
                    const uint32_t v_dst = (uint32_t)buf * kSlmHalf + (uint32_t)tid * 512u;
                    slm_block_store<uint32_t, 128>(v_dst, vb0, kAl16);
                    slm_block_store<uint32_t, 128>(v_dst + 8192u, vb1, kAl16);
                    if (blk_c0 + 64 < kv_end) {
                        prefetch_2d<int, 8, 16, 1>(k_kv, 255u, surf_h, 255u, (tid & 7) * 8, blk_c0 + 64 + ((tid >> 3) & 1) * 16, kPf);
                        prefetch_2d<int, 8, 16, 1>(k_kv, 255u, surf_h, 255u, (tid & 7) * 8, blk_c0 + 96 + ((tid >> 3) & 1) * 16, kPf);
                    }
                    simd<float, 64> vs_64 = block_load<float, 64>(vs_kv + blk_c0);
                    simd<uint16_t, 128> ksr0 = block_load<uint16_t, 128>(ks_kv + (size_t)blk_c0 * 4u, kAl16);
                    simd<uint16_t, 128> ksr1 = block_load<uint16_t, 128>(ks_kv + (size_t)(blk_c0 + 32) * 4u, kAl16);

                    simd<float, 32> sa[6], sb[6];
#pragma unroll
                    for (int r = 0; r < kG; ++r) { sa[r] = 0.0f; sb[r] = 0.0f; }

#pragma unroll
                    for (int c16 = 0; c16 < 4; ++c16) {
                        const int c_off = blk_c0 + c16 * 16;
                        auto& ksr = (c16 < 2) ? ksr0 : ksr1;
                        const int c_in = (c16 & 1);
#pragma unroll
                        for (int qtr = 0; qtr < 4; ++qtr) {
                            simd<int, 128> KB0 = load_2d<int, 8, 16, 1, true, false>(k_kv, 255u, surf_h, 255u, qtr * 16, c_off, kPf);
                            simd<int, 128> KB1 = load_2d<int, 8, 16, 1, true, false>(k_kv, 255u, surf_h, 255u, qtr * 16 + 8, c_off, kPf);
                            simd<int, 128> acc = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(simd<int, 128>(0), KB0, Q_A[qtr * 2 + 0]);
                            acc = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(acc, KB1, Q_A[qtr * 2 + 1]);
                            simd<uint16_t, 16> ks16(ksr.template select<16, 4>(c_in * 64 + qtr));
                            simd<float, 16> ksc = ks16.template bit_cast_view<sycl::half>();
                            if (c16 < 2) {
#pragma unroll
                                for (int r = 0; r < kG; ++r) sa[r].template select<16, 1>(16 * c_in) += simd<float, 16>(acc.template select<16, 1>(16 * r)) * (ksc * float(q_sc[qtr][r]));
                            } else {
#pragma unroll
                                for (int r = 0; r < kG; ++r) sb[r].template select<16, 1>(16 * c_in) += simd<float, 16>(acc.template select<16, 1>(16 * r)) * (ksc * float(q_sc[qtr][r]));
                            }
                        }
                    }

                    simd<float, 8> mx_v = hmax6x32(
                        (es::max)(sa[0], sb[0]), (es::max)(sa[1], sb[1]), (es::max)(sa[2], sb[2]),
                        (es::max)(sa[3], sb[3]), (es::max)(sa[4], sb[4]), (es::max)(sa[5], sb[5]));
                    simd<int, 8> need_scale_m = (mx_v > m_my);
                    simd<float, 8> m_new_v = (es::max)(m_my, mx_v);
                    simd<float, 8> alpha_v = es::exp2((m_my - m_new_v) * kLog2e);
                    alpha_v.merge(simd<float, 8>(0.0f), m_my <= -1e20f);
                    m_my = m_new_v;

                    const simd<float, 32> vs_a = vs_64.template select<32, 1>(0), vs_b = vs_64.template select<32, 1>(32);
#pragma unroll
                    for (int r = 0; r < kG; ++r) {
                        const float mv = float(m_new_v[r]);
                        sa[r] = es::exp2((sa[r] - mv) * kLog2e);
                        sb[r] = es::exp2((sb[r] - mv) * kLog2e);
                    }
                    simd<float, 8> sum_v = hsum6x32(
                        sa[0] + sb[0], sa[1] + sb[1], sa[2] + sb[2],
                        sa[3] + sb[3], sa[4] + sb[4], sa[5] + sb[5]);
                    l_my = l_my * alpha_v + sum_v;

#pragma unroll
                    for (int r = 0; r < kG; ++r) { sa[r] *= vs_a; sb[r] *= vs_b; }
                    simd<float, 8> vmx_v = hmax6x32(
                        (es::max)(sa[0], sb[0]), (es::max)(sa[1], sb[1]), (es::max)(sa[2], sb[2]),
                        (es::max)(sa[3], sb[3]), (es::max)(sa[4], sb[4]), (es::max)(sa[5], sb[5]));
                    simd<float, 8> psc_v = vmx_v * (1.0f / 255.0f);
                    simd<float, 8> inv_v = 255.0f / vmx_v;
                    inv_v.merge(simd<float, 8>(0.0f), vmx_v <= 0.0f);
                    simd<int, 64> pa0(0), pa1(0);
#pragma unroll
                    for (int r = 0; r < kG; ++r) {
                        const float iv = float(inv_v[r]);
                        pa0.template select<8, 1>(8 * r) = pack_p32_u8fast(sa[r], iv);
                        pa1.template select<8, 1>(8 * r) = pack_p32_u8fast(sb[r], iv);
                    }

                    barrier();

                    if (blk_c0 > 0 && need_scale_m.template select<6, 1>(0).any()) {
                        simd<float, 96> a96;
#pragma unroll
                        for (int r = 0; r < kG; ++r) a96.template select<16, 1>(16 * r) = float(alpha_v[r]);
#pragma unroll
                        for (int dl = 0; dl < 16; ++dl) O[dl] *= a96;
                    }
                    simd<float, 96> sc96;
#pragma unroll
                    for (int r = 0; r < kG; ++r) sc96.template select<16, 1>(16 * r) = float(psc_v[r]);
#pragma unroll
                    for (int dl = 0; dl < 16; dl += 2) {
                        const uint32_t vb_off0 = (uint32_t)buf * kSlmHalf + (uint32_t)dl * 512u;
                        simd<int, 128> VB0_a = slm_block_load<int, 128>(vb_off0, kAl16);
                        simd<int, 128> VB1_a = slm_block_load<int, 128>(vb_off0 + 512u, kAl16);
                        simd<int, 128> ov0 = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::u8>(simd<int, 128>(0), VB0_a, pa0);
                        simd<int, 128> ov1 = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::u8>(simd<int, 128>(0), VB1_a, pa0);
                        O[dl]     += simd<float, 96>(ov0.template select<96, 1>(0)) * sc96;
                        O[dl + 1] += simd<float, 96>(ov1.template select<96, 1>(0)) * sc96;
                        const uint32_t vb_off1 = vb_off0 + 8192u;
                        simd<int, 128> VB0_b = slm_block_load<int, 128>(vb_off1, kAl16);
                        simd<int, 128> VB1_b = slm_block_load<int, 128>(vb_off1 + 512u, kAl16);
                        ov0 = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::u8>(simd<int, 128>(0), VB0_b, pa1);
                        ov1 = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::u8>(simd<int, 128>(0), VB1_b, pa1);
                        O[dl]     += simd<float, 96>(ov0.template select<96, 1>(0)) * sc96;
                        O[dl + 1] += simd<float, 96>(ov1.template select<96, 1>(0)) * sc96;
                    }
                    buf ^= 1;
                } else {
                    const uint32_t* vp = v_kv + ((size_t)(blk_c0 >> 5) * 256u + (size_t)tid * 16u) * 8u;
                    simd<uint32_t, 128> vl;
                    vl.template select<64, 1>(0)  = block_load<uint32_t, 64>(vp, kAl16);
                    vl.template select<64, 1>(64) = block_load<uint32_t, 64>(vp + 64, kAl16);
                    if (blk_c0 + 32 < kv_end) {
                        prefetch_2d<int, 8, 16, 1>(k_kv, 255u, surf_h, 255u, (tid & 7) * 8, blk_c0 + 32 + ((tid >> 3) & 1) * 16, kPf);
                        es::prefetch<uint32_t, 64>(vp + 256u * 8u, kPf);
                        es::prefetch<uint32_t, 64>(vp + 256u * 8u + 64u, kPf);
                    }
                    simd<uint32_t, 128> vb;
#pragma unroll
                    for (int dw = 0; dw < 8; ++dw) vb.template select<16, 1>(16 * dw) = vl.template select<16, 8>(dw);
                    const uint32_t v_dst = (uint32_t)buf * kSlmHalf + (uint32_t)tid * 512u;
                    slm_block_store<uint32_t, 128>(v_dst, vb, kAl16);

                    simd<float, 32> vs_32 = block_load<float, 32>(vs_kv + blk_c0);
                    simd<uint16_t, 128> ksr = block_load<uint16_t, 128>(ks_kv + (size_t)blk_c0 * 4u, kAl16);

                    simd<float, 32> s[6];
#pragma unroll
                    for (int r = 0; r < kG; ++r) s[r] = 0.0f;

#pragma unroll
                    for (int qtr = 0; qtr < 4; ++qtr) {
                        simd<int, 128> KB0_a = load_2d<int, 8, 16, 1, true, false>(k_kv, 255u, surf_h, 255u, qtr * 16, blk_c0, kPf);
                        simd<int, 128> KB0_b = load_2d<int, 8, 16, 1, true, false>(k_kv, 255u, surf_h, 255u, qtr * 16, blk_c0 + 16, kPf);
                        simd<int, 128> acc_a = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(simd<int, 128>(0), KB0_a, Q_A[qtr * 2 + 0]);
                        simd<int, 128> acc_b = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(simd<int, 128>(0), KB0_b, Q_A[qtr * 2 + 0]);
                        simd<int, 128> KB1_a = load_2d<int, 8, 16, 1, true, false>(k_kv, 255u, surf_h, 255u, qtr * 16 + 8, blk_c0, kPf);
                        simd<int, 128> KB1_b = load_2d<int, 8, 16, 1, true, false>(k_kv, 255u, surf_h, 255u, qtr * 16 + 8, blk_c0 + 16, kPf);
                        acc_a = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(acc_a, KB1_a, Q_A[qtr * 2 + 1]);
                        acc_b = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(acc_b, KB1_b, Q_A[qtr * 2 + 1]);
                        simd<uint16_t, 16> ks_a(ksr.template select<16, 4>(qtr));
                        simd<uint16_t, 16> ks_b(ksr.template select<16, 4>(64 + qtr));
                        simd<float, 16> ksc_a = ks_a.template bit_cast_view<sycl::half>();
                        simd<float, 16> ksc_b = ks_b.template bit_cast_view<sycl::half>();
#pragma unroll
                        for (int r = 0; r < kG; ++r) {
                            const float qsr = float(q_sc[qtr][r]);
                            s[r].template select<16, 1>(0)  += simd<float, 16>(acc_a.template select<16, 1>(16 * r)) * (ksc_a * qsr);
                            s[r].template select<16, 1>(16) += simd<float, 16>(acc_b.template select<16, 1>(16 * r)) * (ksc_b * qsr);
                        }
                    }

                    simd<float, 8> mx_v = hmax6x32(s[0], s[1], s[2], s[3], s[4], s[5]);
                    simd<int, 8> need_scale_m = (mx_v > m_my);
                    simd<float, 8> m_new_v = (es::max)(m_my, mx_v);
                    simd<float, 8> alpha_v = es::exp2((m_my - m_new_v) * kLog2e);
                    alpha_v.merge(simd<float, 8>(0.0f), m_my <= -1e20f);
                    m_my = m_new_v;

                    simd<float, 32> e[6], pv[6];
#pragma unroll
                    for (int r = 0; r < kG; ++r) {
                        e[r] = es::exp2((s[r] - float(m_new_v[r])) * kLog2e);
                        pv[r] = e[r] * vs_32;
                    }
                    simd<float, 8> sum_v = hsum6x32(e[0], e[1], e[2], e[3], e[4], e[5]);
                    l_my = l_my * alpha_v + sum_v;

                    simd<float, 8> vmx_v = hmax6x32(pv[0], pv[1], pv[2], pv[3], pv[4], pv[5]);
                    simd<float, 8> psc_v = vmx_v * (1.0f / 65535.0f);
                    simd<float, 8> inv_v = 65535.0f / vmx_v;
                    inv_v.merge(simd<float, 8>(0.0f), vmx_v <= 0.0f);
                    simd<int, 64> pa_hi(0), pa_lo(0);
#pragma unroll
                    for (int r = 0; r < kG; ++r) {
                        simd<int, 8> ph, pl;
                        pack_p32_u16(pv[r], float(inv_v[r]), ph, pl);
                        pa_hi.template select<8, 1>(8 * r) = ph;
                        pa_lo.template select<8, 1>(8 * r) = pl;
                    }

                    barrier();

                    if (blk_c0 > 0 && need_scale_m.template select<6, 1>(0).any()) {
                        simd<float, 96> a96;
#pragma unroll
                        for (int r = 0; r < kG; ++r) a96.template select<16, 1>(16 * r) = float(alpha_v[r]);
#pragma unroll
                        for (int dl = 0; dl < 16; ++dl) O[dl] *= a96;
                    }
                    simd<float, 96> sc96;
#pragma unroll
                    for (int r = 0; r < kG; ++r) sc96.template select<16, 1>(16 * r) = float(psc_v[r]);
#pragma unroll
                    for (int dl = 0; dl < 16; dl += 2) {
                        const uint32_t vb_off = (uint32_t)buf * kSlmHalf + (uint32_t)dl * 512u;
                        simd<int, 128> VB0 = slm_block_load<int, 128>(vb_off, kAl16);
                        simd<int, 128> VB1 = slm_block_load<int, 128>(vb_off + 512u, kAl16);
                        simd<int, 128> ov0 = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::u8>(simd<int, 128>(0), VB0, pa_hi);
                        simd<int, 128> ov1 = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::u8>(simd<int, 128>(0), VB1, pa_hi);
                        simd<int, 128> ov0l = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::u8>(simd<int, 128>(0), VB0, pa_lo);
                        simd<int, 128> ov1l = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::u8>(simd<int, 128>(0), VB1, pa_lo);
                        ov0 = ov0 * 256 + ov0l;
                        ov1 = ov1 * 256 + ov1l;
                        O[dl]     += simd<float, 96>(ov0.template select<96, 1>(0)) * sc96;
                        O[dl + 1] += simd<float, 96>(ov1.template select<96, 1>(0)) * sc96;
                    }
                    buf ^= 1;
                }
            }

            // Masked causal diagonal & partial tail path (steps by 32 KV columns using kSlmHalf ping-pong)
            for (; blk_c0 < kv_end; blk_c0 += 32) {
                const uint32_t* vp = v_kv + ((size_t)(blk_c0 >> 5) * 256u + (size_t)tid * 16u) * 8u;
                simd<uint32_t, 128> vl;
                vl.template select<64, 1>(0)  = block_load<uint32_t, 64>(vp, kAl16);
                vl.template select<64, 1>(64) = block_load<uint32_t, 64>(vp + 64, kAl16);
                simd<uint32_t, 128> vb;
#pragma unroll
                for (int dw = 0; dw < 8; ++dw) vb.template select<16, 1>(16 * dw) = vl.template select<16, 8>(dw);
                const uint32_t v_dst = (uint32_t)buf * kSlmHalf + (uint32_t)tid * 512u;
                slm_block_store<uint32_t, 128>(v_dst, vb, kAl16);

                simd<float, 32> s[6];
                if (blk_c0 <= causal_lim) {
#pragma unroll
                    for (int r = 0; r < kG; ++r) s[r] = 0.0f;
#pragma unroll
                    for (int c16 = 0; c16 < 2; ++c16) {
                        const int c_off = blk_c0 + c16 * 16;
                        if (c_off <= causal_lim) {
                            simd<uint16_t, 64> ksr;
                            if (c_off + 16 <= kv_end) {
                                ksr = block_load<uint16_t, 64>(ks_kv + (size_t)c_off * 4u, kAl16);
                            } else {
                                ksr = 0u;
                                for (int i = 0; i < 16; ++i)
                                    if (c_off + i < kv_end)
                                        for (int q = 0; q < 4; ++q) ksr[i * 4 + q] = ks_kv[(size_t)(c_off + i) * 4u + q];
                            }
#pragma unroll
                            for (int qtr = 0; qtr < 4; ++qtr) {
                                simd<int, 128> KB0 = load_2d<int, 8, 16, 1, true, false>(k_kv, 255u, surf_h, 255u, qtr * 16, c_off, kPf);
                                simd<int, 128> KB1 = load_2d<int, 8, 16, 1, true, false>(k_kv, 255u, surf_h, 255u, qtr * 16 + 8, c_off, kPf);
                                simd<int, 128> acc = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(simd<int, 128>(0), KB0, Q_A[qtr * 2 + 0]);
                                acc = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(acc, KB1, Q_A[qtr * 2 + 1]);
                                simd<uint16_t, 16> ks16(ksr.template select<16, 4>(qtr));
                                simd<float, 16> ksc = ks16.template bit_cast_view<sycl::half>();
#pragma unroll
                                for (int r = 0; r < kG; ++r)
                                    s[r].template select<16, 1>(16 * c16) += simd<float, 16>(acc.template select<16, 1>(16 * r)) * (ksc * float(q_sc[qtr][r]));
                            }
                        } else {
#pragma unroll
                            for (int r = 0; r < kG; ++r) s[r].template select<16, 1>(16 * c16) = -1e30f;
                        }
                    }
                    if (blk_c0 + 31 > causal_lim) {
                        const simd<int, 32> cidx(blk_c0, 1);
                        const auto mask_c = (cidx > causal_lim);
#pragma unroll
                        for (int r = 0; r < kG; ++r) s[r].merge(simd<float, 32>(-1e30f), mask_c);
                    }
                }

                simd<float, 8> psc_v(0.0f), alpha_v(1.0f);
                simd<int, 64> pa_hi(0), pa_lo(0);
                simd<int, 8> need_scale_m(0);
                if (blk_c0 <= causal_lim) {
                    simd<float, 32> vs_32;
                    if (blk_c0 + 32 <= kv_end) {
                        vs_32 = block_load<float, 32>(vs_kv + blk_c0);
                    } else {
                        vs_32 = 0.0f;
                        for (int i = 0; i < 32; ++i)
                            if (blk_c0 + i < kv_end) vs_32[i] = vs_kv[blk_c0 + i];
                    }

                    simd<float, 8> mx_v = hmax6x32(s[0], s[1], s[2], s[3], s[4], s[5]);
                    need_scale_m = (mx_v > m_my);
                    simd<float, 8> m_new_v = (es::max)(m_my, mx_v);
                    alpha_v = es::exp2((m_my - m_new_v) * kLog2e);
                    alpha_v.merge(simd<float, 8>(0.0f), m_my <= -1e20f);
                    m_my = m_new_v;

                    simd<float, 32> e[6], pv[6];
#pragma unroll
                    for (int r = 0; r < kG; ++r) {
                        e[r] = es::exp2((s[r] - float(m_new_v[r])) * kLog2e);
                        e[r].merge(simd<float, 32>(0.0f), s[r] < -1e20f);
                        pv[r] = e[r] * vs_32;
                    }
                    simd<float, 8> sum_v = hsum6x32(e[0], e[1], e[2], e[3], e[4], e[5]);
                    l_my = l_my * alpha_v + sum_v;

                    simd<float, 8> vmx_v = hmax6x32(pv[0], pv[1], pv[2], pv[3], pv[4], pv[5]);
                    simd<float, 8> inv_v;
                    if constexpr (U16P) {
                        psc_v = vmx_v * (1.0f / 65535.0f);
                        inv_v = 65535.0f / vmx_v;
                    } else {
                        psc_v = vmx_v * (1.0f / 255.0f);
                        inv_v = 255.0f / vmx_v;
                    }
                    inv_v.merge(simd<float, 8>(0.0f), vmx_v <= 0.0f);
#pragma unroll
                    for (int r = 0; r < kG; ++r) {
                        if constexpr (U16P) {
                            simd<int, 8> ph, pl;
                            pack_p32_u16(pv[r], float(inv_v[r]), ph, pl);
                            pa_hi.template select<8, 1>(8 * r) = ph;
                            pa_lo.template select<8, 1>(8 * r) = pl;
                        } else {
                            pa_hi.template select<8, 1>(8 * r) = pack_p32_u8fast(pv[r], float(inv_v[r]));
                        }
                    }
                }

                barrier();

                if (blk_c0 <= causal_lim) {
                    if (blk_c0 > 0 && need_scale_m.template select<6, 1>(0).any()) {
                        simd<float, 96> a96;
#pragma unroll
                        for (int r = 0; r < kG; ++r) a96.template select<16, 1>(16 * r) = float(alpha_v[r]);
#pragma unroll
                        for (int dl = 0; dl < 16; ++dl) O[dl] *= a96;
                    }
                    simd<float, 96> sc96;
#pragma unroll
                    for (int r = 0; r < kG; ++r) sc96.template select<16, 1>(16 * r) = float(psc_v[r]);
#pragma unroll
                    for (int dl = 0; dl < 16; dl += 2) {
                        const uint32_t vb_off = (uint32_t)buf * kSlmHalf + (uint32_t)dl * 512u;
                        simd<int, 128> VB0 = slm_block_load<int, 128>(vb_off, kAl16);
                        simd<int, 128> VB1 = slm_block_load<int, 128>(vb_off + 512u, kAl16);
                        simd<int, 128> ov0 = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::u8>(simd<int, 128>(0), VB0, pa_hi);
                        simd<int, 128> ov1 = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::u8>(simd<int, 128>(0), VB1, pa_hi);
                        if constexpr (U16P) {
                            simd<int, 128> ov0l = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::u8>(simd<int, 128>(0), VB0, pa_lo);
                            simd<int, 128> ov1l = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::u8>(simd<int, 128>(0), VB1, pa_lo);
                            ov0 = ov0 * 256 + ov0l;
                            ov1 = ov1 * 256 + ov1l;
                        }
                        O[dl]     += simd<float, 96>(ov0.template select<96, 1>(0)) * sc96;
                        O[dl + 1] += simd<float, 96>(ov1.template select<96, 1>(0)) * sc96;
                    }
                }
                buf ^= 1;
            }

            if (tok < T) {
                simd<float, 8> inv_lv(0.0f);
#pragma unroll
                for (int r = 0; r < kG; ++r) inv_lv[r] = (float(l_my[r]) > 0.0f) ? (1.0f / float(l_my[r])) : 0.0f;
#pragma unroll
                for (int r = 0; r < kG; ++r) {
                    const int qh = kvh * kG + r;
                    const float il = float(inv_lv[r]);
                    simd<float, 64> out_blk;
#pragma unroll
                    for (int qtr = 0; qtr < 4; ++qtr) {
                        out_blk.template select<16, 1>(0)  = O[qtr * 4 + 0].template select<16, 1>(16 * r) * il;
                        out_blk.template select<16, 1>(16) = O[qtr * 4 + 1].template select<16, 1>(16 * r) * il;
                        out_blk.template select<16, 1>(32) = O[qtr * 4 + 2].template select<16, 1>(16 * r) * il;
                        out_blk.template select<16, 1>(48) = O[qtr * 4 + 3].template select<16, 1>(16 * r) * il;
                        block_store<float, 64>(out + ((size_t)tok * H + qh) * kHD + qtr * 64, out_blk);
                    }
                }
            }
        });
    });
}

}  // namespace

bool supports(int T, int H, int HKV, int HD, int vt) {
    return T >= 1 && H > 0 && HKV > 0 && H == HKV * kG && HD == kHD && vt > 0 && (vt % 32) == 0;
}

void forward(sycl::queue& q, int vt, const sycl::half* q16, const int8_t* kc, const int8_t* vc, float* out,
             int T, int p0, int H, int HKV, int ctx, bool u16_p) {
    if (u16_p) launch_fmha_i8<true>(q, vt, q16, kc, vc, out, T, p0, H, HKV, ctx);
    else       launch_fmha_i8<false>(q, vt, q16, kc, vc, out, T, p0, H, HKV, ctx);
}

}  // namespace strata::q36fmha
