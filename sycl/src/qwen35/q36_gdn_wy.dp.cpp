// q36_gdn_wy.dp.cpp - Chunkwise C=64 parallel WY GDN prefill recurrence on Xe2 XMX (ESIMD).
//
// Implements the gated delta rule over T prompt tokens for Qwen3.8-27B (S=128, HK=16, HV=48, C=10240)
// in two -doubleGRF ESIMD kernels:
//   1. k_gdn_wy_prep  (grid = nc * HV, 16 threads/WG): computes M_qk = causal(Q K^T * decay) and solves
//      A = (I + strict_lower(beta * K K^T * decay))^{-1} in FP32 via 16x16 block triangular inversion,
//      then forms W = A (beta * exp(gamma) * K) and U = A (beta * V) via xmx::dpas<8,8,float,float,half,half>.
//   2. k_gdn_wy_state (grid = HV, 32 threads/WG): keeps the entire [128][128] FP32 state S resident in GRF
//      accumulators across chunks c = 0..nc-1 and executes delta = U - W S, O_inter = Q_s S, O_intra = M_qk delta,
//      and S = exp(g_end) S + K_tail^T delta via xmx::dpas.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include "q36_gdn_wy.hpp"

namespace strata::q36wy {
namespace {
using namespace sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

constexpr auto kAl16 = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::alignment<16>};

struct TagWyPrep;
struct TagWyState;

SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> pack_vnni2_f(simd<float, 16> even_row, simd<float, 16> odd_row) {
    simd<sycl::half, 32> hp;
    hp.template select<16, 2>(0) = simd<sycl::half, 16>(even_row);
    hp.template select<16, 2>(1) = simd<sycl::half, 16>(odd_row);
    return simd<uint32_t, 16>(hp.template bit_cast_view<uint32_t>());
}

SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> pack_vnni2_h(simd<sycl::half, 16> even_row, simd<sycl::half, 16> odd_row) {
    simd<sycl::half, 32> hp;
    hp.template select<16, 2>(0) = even_row;
    hp.template select<16, 2>(1) = odd_row;
    return simd<uint32_t, 16>(hp.template bit_cast_view<uint32_t>());
}

SYCL_ESIMD_FUNCTION inline simd<sycl::half, 128> load_slm_a(uint32_t base_b, uint32_t stride_b, int r0, int k16) {
    simd<uint32_t, 64> a_u;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        a_u.template select<8, 1>(8 * r) =
            slm_block_load<uint32_t, 8>(base_b + (uint32_t)(r0 + r) * stride_b + (uint32_t)k16 * 32u, kAl16);
    }
    return simd<sycl::half, 128>(a_u.template bit_cast_view<sycl::half>());
}

SYCL_ESIMD_FUNCTION inline simd<sycl::half, 256> load_slm_b_h256(uint32_t off) {
    simd<uint32_t, 128> b_u = slm_block_load<uint32_t, 128>(off, kAl16);
    return simd<sycl::half, 256>(b_u.template bit_cast_view<sycl::half>());
}

SYCL_ESIMD_FUNCTION inline simd<sycl::half, 128> load_g_f16_128(const float* src) {
    simd<sycl::half, 128> out;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        out.template select<32, 1>(32 * i) = simd<sycl::half, 32>(block_load<float, 32>(src + 32 * i));
    }
    return out;
}

SYCL_ESIMD_FUNCTION inline simd<float, 16> esimd_exp16(simd<float, 16> x) {
    return sycl::ext::intel::esimd::exp(x);
}

SYCL_ESIMD_FUNCTION inline float esimd_exp1(float x) {
    simd<float, 8> xv(x);
    return float(sycl::ext::intel::esimd::exp(xv)[0]);
}

SYCL_ESIMD_FUNCTION inline void store_dec_tile(
    int gm, int j0, int r_base, simd<float, 128>& kk, simd<float, 128>& qk,
    const simd<float, 16>& gj, const simd<float, 16>& gt, const simd<float, 16>& bt,
    const simd<int, 16>& jidx, uint32_t slm_m, sycl::half* mqk_base) {
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int rr = r_base + r;
        const int t  = 16 * gm + rr;
        const simd<int, 16> tv(t);
        const auto m_gt = tv > jidx;
        const auto m_ge = tv >= jidx;
        simd<float, 16> sdiff = 0.f;
        sdiff.merge(float(gt[rr]) - gj, m_ge);
        const simd<float, 16> dec = esimd_exp16(sdiff);
        simd<float, 16> mr = 0.f;
        mr.merge(float(bt[rr]) * dec * kk.template select<16, 1>(16 * r), m_gt);
        slm_block_store<float, 16>(slm_m + (uint32_t)(t * 64 + j0) * 4u, mr, kAl16);
        simd<float, 16> qr = 0.f;
        qr.merge(dec * qk.template select<16, 1>(16 * r), m_ge);
        block_store<sycl::half, 16>(mqk_base + (size_t)t * kChunk + j0, simd<sycl::half, 16>(qr));
    }
}

SYCL_ESIMD_FUNCTION inline void solve_offdiag_blk(int tid, int rb, int cb, uint32_t slm_m, uint32_t slm_a, uint32_t slm_tmp) {
    const uint32_t mr_off = slm_m + (uint32_t)(16 * rb + tid) * 256u;
    simd<float, 16> sum = 0.f;
    for (int kb = cb; kb < rb; ++kb) {
        const simd<float, 16> mv = slm_block_load<float, 16>(mr_off + (uint32_t)(16 * kb) * 4u, kAl16);
        const uint32_t ab_off = slm_a + (uint32_t)(16 * kb * 64 + 16 * cb) * 4u;
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            sum += float(mv[m]) * slm_block_load<float, 16>(ab_off + (uint32_t)m * 256u, kAl16);
        }
    }
    slm_block_store<float, 16>(slm_tmp + (uint32_t)tid * 64u, sum, kAl16);
    barrier();
    const simd<float, 16> diag_r = slm_block_load<float, 16>(slm_a + (uint32_t)(16 * rb + tid) * 256u + (uint32_t)(16 * rb) * 4u, kAl16);
    simd<float, 16> out_r = 0.f;
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        out_r -= float(diag_r[m]) * slm_block_load<float, 16>(slm_tmp + (uint32_t)m * 64u, kAl16);
    }
    slm_block_store<float, 16>(slm_a + (uint32_t)(16 * rb + tid) * 256u + (uint32_t)(16 * cb) * 4u, out_r, kAl16);
    barrier();
}

SYCL_ESIMD_FUNCTION inline void flush_wu_tile(
    int row_base, int dg, simd<float, 128>& w0, simd<float, 128>& w1,
    simd<float, 128>& u0, simd<float, 128>& u1, sycl::half* w_dst, sycl::half* u_dst) {
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int lr = row_base + r;
        const size_t goff = (size_t)lr * kS + (size_t)(32 * dg);
        block_store<sycl::half, 16>(w_dst + goff,      simd<sycl::half, 16>(w0.template select<16, 1>(16 * r)));
        block_store<sycl::half, 16>(w_dst + goff + 16, simd<sycl::half, 16>(w1.template select<16, 1>(16 * r)));
        block_store<sycl::half, 16>(u_dst + goff,      simd<sycl::half, 16>(u0.template select<16, 1>(16 * r)));
        block_store<sycl::half, 16>(u_dst + goff + 16, simd<sycl::half, 16>(u1.template select<16, 1>(16 * r)));
    }
}

// Kernel 1: per (chunk c, head h), compute ws_Mqk, ws_W, ws_U, ws_g.
void launch_prep(sycl::queue& q, sycl::half* ws_W, sycl::half* ws_U, sycl::half* ws_Mqk, float* ws_g,
                 const float* hc, const float* gate, const float* beta, int T, int nc) {
    constexpr uint32_t SLM_K   = 0u;      // [64][128] half row-major (16 KB)
    constexpr uint32_t SLM_Q   = 16384u;  // [64][128] half row-major (16 KB), later SLM_M / SLM_WK
    constexpr uint32_t SLM_M   = 16384u;  // [64][64] float row-major (16 KB)
    constexpr uint32_t SLM_WK  = 16384u;  // [4 k16][8 d16][8 k2][16 n] u32 (16 KB)
    constexpr uint32_t SLM_KT  = 32768u;  // [8 kd][4 n16][8 k2][16 n] u32 (16 KB), later SLM_VB
    constexpr uint32_t SLM_VB  = 32768u;  // [4 k16][8 d16][8 k2][16 n] u32 (16 KB)
    constexpr uint32_t SLM_GAM = 49152u;  // [64] float (256 B)
    constexpr uint32_t SLM_BET = 49408u;  // [64] float (256 B)
    constexpr uint32_t SLM_A_H = 49664u;  // [64][64] half row-major (8 KB)
    constexpr uint32_t SLM_TOT = 57856u;  // 56.5 KB

    const size_t total_wg = (size_t)nc * kHV;
    const auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};

    q.submit([&](sycl::handler& ch) {
        ch.parallel_for<TagWyPrep>(
            sycl::nd_range<1>(total_wg * 16u, 16u), props, [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
                slm_init<SLM_TOT>();
                const int wg  = (int)it.get_group(0);
                const int tid = (int)it.get_local_id(0);
                const int c   = wg / kHV;
                const int h   = wg % kHV;
                const int qh  = h % kHK;
                const int t0  = c * kChunk;
                const int len = (T - t0) < kChunk ? (T - t0) : kChunk;

                // Stage 1: prefix log-decay gamma & beta into SLM, plus store ws_g.
                if (tid == 0) {
                    simd<float, 64> g_raw = 0.f;
                    simd<float, 64> b_raw = 0.f;
                    for (int i = 0; i < len; ++i) {
                        g_raw[i] = gate[(size_t)(t0 + i) * kHV + h];
                        b_raw[i] = beta[(size_t)(t0 + i) * kHV + h];
                    }
                    simd<float, 64> gam;
                    float acc = 0.f;
                    for (int i = 0; i < len; ++i) {
                        acc += float(g_raw[i]);
                        gam[i] = acc;
                    }
                    for (int i = len; i < kChunk; ++i) gam[i] = acc;
                    slm_block_store<float, 32>(SLM_GAM,        gam.template select<32, 1>(0),  kAl16);
                    slm_block_store<float, 32>(SLM_GAM + 128u, gam.template select<32, 1>(32), kAl16);
                    slm_block_store<float, 32>(SLM_BET,        b_raw.template select<32, 1>(0),  kAl16);
                    slm_block_store<float, 32>(SLM_BET + 128u, b_raw.template select<32, 1>(32), kAl16);
                    float* g_dst = ws_g + (size_t)wg * kChunk;
                    block_store<float, 32>(g_dst,      gam.template select<32, 1>(0));
                    block_store<float, 32>(g_dst + 32, gam.template select<32, 1>(32));
                }

                // Stage 1b: load 4 tokens/thread of K and Q, store row-major & VNNI-2 K^T.
                const int n16 = tid >> 2;
                const int sub = tid & 3;
                const int r0  = 4 * tid;
                simd<uint32_t, 64> kh_u[4];
#pragma unroll
                for (int dt = 0; dt < 4; ++dt) {
                    const int lr = r0 + dt;
                    simd<sycl::half, 128> kh(0.0f);
                    simd<sycl::half, 128> qh_v(0.0f);
                    if (lr < len) {
                        const float* hrow = hc + (size_t)(t0 + lr) * kC;
                        qh_v = load_g_f16_128(hrow + (size_t)qh * kS);
                        kh   = load_g_f16_128(hrow + (size_t)kHK * kS + (size_t)qh * kS);
                    }
                    kh_u[dt] = kh.template bit_cast_view<uint32_t>();
                    const simd<uint32_t, 64> qh_u = qh_v.template bit_cast_view<uint32_t>();
                    slm_block_store<uint32_t, 64>(SLM_K + (uint32_t)lr * 256u, kh_u[dt], kAl16);
                    slm_block_store<uint32_t, 64>(SLM_Q + (uint32_t)lr * 256u, qh_u,     kAl16);
                }
                for (int kd = 0; kd < 8; ++kd) {
                    const uint32_t kt_tile = SLM_KT + (uint32_t)(kd * 4 + n16) * 512u + (uint32_t)(sub * 4) * 4u;
#pragma unroll
                    for (int k2 = 0; k2 < 8; ++k2) {
                        simd<uint32_t, 4> col4;
                        col4[0] = kh_u[0][8 * kd + k2];
                        col4[1] = kh_u[1][8 * kd + k2];
                        col4[2] = kh_u[2][8 * kd + k2];
                        col4[3] = kh_u[3][8 * kd + k2];
                        slm_block_store<uint32_t, 4>(kt_tile + (uint32_t)k2 * 64u, col4, kAl16);
                    }
                }
                barrier();

                // Stage 2: KK = K @ K^T and QK = Q @ K^T via DPAS.
                const int gm = tid >> 2;  // 0..3 -> 16 rows
                const int gn = tid & 3;   // 0..3 -> 16 cols
                simd<float, 128> kk0 = 0.f, kk1 = 0.f, qk0 = 0.f, qk1 = 0.f;
                for (int kd = 0; kd < 8; ++kd) {
                    const simd<sycl::half, 256> b = load_slm_b_h256(SLM_KT + (uint32_t)(kd * 4 + gn) * 512u);
                    const simd<sycl::half, 128> ak0 = load_slm_a(SLM_K, 256u, 16 * gm,     kd);
                    const simd<sycl::half, 128> ak1 = load_slm_a(SLM_K, 256u, 16 * gm + 8, kd);
                    kk0 = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(kk0, b, ak0);
                    kk1 = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(kk1, b, ak1);
                    const simd<sycl::half, 128> aq0 = load_slm_a(SLM_Q, 256u, 16 * gm,     kd);
                    const simd<sycl::half, 128> aq1 = load_slm_a(SLM_Q, 256u, 16 * gm + 8, kd);
                    qk0 = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(qk0, b, aq0);
                    qk1 = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(qk1, b, aq1);
                }
                barrier(); // SLM_Q is free to become SLM_M.

                const int j0 = 16 * gn;
                const simd<float, 16> gj  = slm_block_load<float, 16>(SLM_GAM + (uint32_t)j0 * 4u, kAl16);
                const simd<float, 16> gt  = slm_block_load<float, 16>(SLM_GAM + (uint32_t)(16 * gm) * 4u, kAl16);
                const simd<float, 16> bt  = slm_block_load<float, 16>(SLM_BET + (uint32_t)(16 * gm) * 4u, kAl16);
                const simd<int, 16> jidx(j0, 1);
                sycl::half* const mqk_base = ws_Mqk + (size_t)wg * kChunk * kChunk;

                store_dec_tile(gm, j0, 0, kk0, qk0, gj, gt, bt, jidx, SLM_M, mqk_base);
                store_dec_tile(gm, j0, 8, kk1, qk1, gj, gt, bt, jidx, SLM_M, mqk_base);
                barrier();

                // Stage 3: exact FP32 block triangular inverse A = (I + M)^{-1} in SLM_A (SLM_KT), keeping SLM_M intact.
                constexpr uint32_t SLM_A = SLM_KT;
                {
                    const simd<float, 32> z32(0.f);
#pragma unroll
                    for (int dt = 0; dt < 4; ++dt) {
                        const int r = 4 * tid + dt;
                        slm_block_store<float, 32>(SLM_A + (uint32_t)r * 256u,        z32, kAl16);
                        slm_block_store<float, 32>(SLM_A + (uint32_t)r * 256u + 128u, z32, kAl16);
                    }
                }
                barrier();
                if (tid < 4) {
                    const int b = tid;
                    simd<float, 16> mb[16], ab[16];
#pragma unroll
                    for (int r = 0; r < 16; ++r) {
                        mb[r] = slm_block_load<float, 16>(SLM_M + (uint32_t)((16 * b + r) * 64 + 16 * b) * 4u, kAl16);
                        ab[r] = 0.f;
                    }
                    ab[0][0] = 1.f;
                    for (int r = 1; r < 16; ++r) {
                        simd<float, 16> acc = 0.f;
                        acc[r] = 1.f;
                        for (int k = 0; k < r; ++k) acc -= float(mb[r][k]) * ab[k];
                        ab[r] = acc;
                    }
#pragma unroll
                    for (int r = 0; r < 16; ++r) {
                        slm_block_store<float, 16>(SLM_A + (uint32_t)((16 * b + r) * 64 + 16 * b) * 4u, ab[r], kAl16);
                    }
                }
                barrier();

                solve_offdiag_blk(tid, 1, 0, SLM_M, SLM_A, SLM_A_H);
                solve_offdiag_blk(tid, 2, 1, SLM_M, SLM_A, SLM_A_H);
                solve_offdiag_blk(tid, 3, 2, SLM_M, SLM_A, SLM_A_H);
                solve_offdiag_blk(tid, 2, 0, SLM_M, SLM_A, SLM_A_H);
                solve_offdiag_blk(tid, 3, 1, SLM_M, SLM_A, SLM_A_H);
                solve_offdiag_blk(tid, 3, 0, SLM_M, SLM_A, SLM_A_H);

                // Pack A [64][64] -> SLM_A_H half row-major before overwriting SLM_M/SLM_KT with SLM_WK/SLM_VB.
#pragma unroll
                for (int dt = 0; dt < 4; ++dt) {
                    const int r = 4 * tid + dt;
                    simd<sycl::half, 64> ah;
                    ah.template select<32, 1>(0)  = simd<sycl::half, 32>(slm_block_load<float, 32>(SLM_A + (uint32_t)r * 256u, kAl16));
                    ah.template select<32, 1>(32) = simd<sycl::half, 32>(slm_block_load<float, 32>(SLM_A + (uint32_t)r * 256u + 128u, kAl16));
                    slm_block_store<sycl::half, 64>(SLM_A_H + (uint32_t)r * 128u, ah, kAl16);
                }
                barrier();

                // Stage 4: pack W_key = beta*exp(gamma)*K and V_beta = beta*V into VNNI-2 SLM.
                for (int hc_blk = 0; hc_blk < 2; ++hc_blk) {
                    const int te  = hc_blk * 32 + 2 * tid;
                    const int to  = te + 1;
                    const int k16 = hc_blk * 2 + (tid >> 3);
                    const int k2  = tid & 7;
                    const float ge = slm_block_load<float, 1>(SLM_GAM + (uint32_t)te * 4u)[0];
                    const float go = slm_block_load<float, 1>(SLM_GAM + (uint32_t)to * 4u)[0];
                    const float be = slm_block_load<float, 1>(SLM_BET + (uint32_t)te * 4u)[0];
                    const float bo = slm_block_load<float, 1>(SLM_BET + (uint32_t)to * 4u)[0];
                    const float wse = (te < len) ? be * esimd_exp1(ge) : 0.f;
                    const float wso = (to < len) ? bo * esimd_exp1(go) : 0.f;
                    const float bse = (te < len) ? be : 0.f;
                    const float bso = (to < len) ? bo : 0.f;

#pragma unroll
                    for (int d16 = 0; d16 < 8; ++d16) {
                        const simd<float, 16> ke = simd<float, 16>(slm_block_load<sycl::half, 16>(SLM_K + (uint32_t)te * 256u + (uint32_t)d16 * 32u, kAl16)) * wse;
                        const simd<float, 16> ko = simd<float, 16>(slm_block_load<sycl::half, 16>(SLM_K + (uint32_t)to * 256u + (uint32_t)d16 * 32u, kAl16)) * wso;
                        const uint32_t blk_off = (uint32_t)((k16 * 8 + d16) * 8 + k2) * 64u;
                        slm_block_store<uint32_t, 16>(SLM_WK + blk_off, pack_vnni2_f(ke, ko), kAl16);

                        simd<float, 16> ve = 0.f, vo = 0.f;
                        if (te < len)
                            ve = block_load<float, 16>(hc + (size_t)(t0 + te) * kC + (size_t)2u * kHK * kS + (size_t)h * kS + 16 * d16) * bse;
                        if (to < len)
                            vo = block_load<float, 16>(hc + (size_t)(t0 + to) * kC + (size_t)2u * kHK * kS + (size_t)h * kS + 16 * d16) * bso;
                        slm_block_store<uint32_t, 16>(SLM_VB + blk_off, pack_vnni2_f(ve, vo), kAl16);
                    }
                }
                barrier();

                // W = A @ W_key and U = A @ V_beta: each thread holds [16 rows][32 cols] (4 8x16 DPAS tiles).
                const int dg = tid & 3;
                simd<float, 128> w00 = 0.f, w01 = 0.f, w10 = 0.f, w11 = 0.f;
                simd<float, 128> u00 = 0.f, u01 = 0.f, u10 = 0.f, u11 = 0.f;
                for (int k16 = 0; k16 < 4; ++k16) {
                    const simd<sycl::half, 128> a0 = load_slm_a(SLM_A_H, 128u, 16 * gm,     k16);
                    const simd<sycl::half, 128> a1 = load_slm_a(SLM_A_H, 128u, 16 * gm + 8, k16);
                    const uint32_t off0 = (uint32_t)(k16 * 8 + 2 * dg) * 512u;
                    const uint32_t off1 = (uint32_t)(k16 * 8 + 2 * dg + 1) * 512u;
                    const simd<sycl::half, 256> bw0 = load_slm_b_h256(SLM_WK + off0);
                    const simd<sycl::half, 256> bw1 = load_slm_b_h256(SLM_WK + off1);
                    w00 = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(w00, bw0, a0);
                    w01 = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(w01, bw1, a0);
                    w10 = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(w10, bw0, a1);
                    w11 = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(w11, bw1, a1);

                    const simd<sycl::half, 256> bu0 = load_slm_b_h256(SLM_VB + off0);
                    const simd<sycl::half, 256> bu1 = load_slm_b_h256(SLM_VB + off1);
                    u00 = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(u00, bu0, a0);
                    u01 = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(u01, bu1, a0);
                    u10 = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(u10, bu0, a1);
                    u11 = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(u11, bu1, a1);
                }

                sycl::half* const w_dst = ws_W + (size_t)wg * kChunk * kS;
                sycl::half* const u_dst = ws_U + (size_t)wg * kChunk * kS;
                flush_wu_tile(16 * gm,     dg, w00, w01, u00, u01, w_dst, u_dst);
                flush_wu_tile(16 * gm + 8, dg, w10, w11, u10, u11, w_dst, u_dst);
            });
    });
}

// Kernel 2: 1 WG per head (h in 0..47), 32 threads/WG, keeping S[128][128] FP32 in GRF across all chunks.
void launch_state(sycl::queue& q, const sycl::half* ws_W, const sycl::half* ws_U, const sycl::half* ws_Mqk,
                  const float* ws_g, float* state, const float* hc, float* out, int T, int nc) {
    constexpr uint32_t SLM_S_VNNI = 0u;      // [8 k16][8 gn][8 k2][16 n] u32 (32 KB)
    constexpr uint32_t SLM_A_WQ   = 32768u;  // [128][128] half row-major (32 KB)
    constexpr uint32_t SLM_DLT_V  = 0u;      // [4 k16][8 gn][8 k2][16 n] u32 (16 KB)
    constexpr uint32_t SLM_KTAIL  = 16384u;  // [128 dims][64 tok] half row-major (16 KB)
    constexpr uint32_t SLM_MQK    = 32768u;  // [64][64] half row-major (8 KB)
    constexpr uint32_t SLM_O_LO   = 16384u;  // [32][128] float (16 KB)
    constexpr uint32_t SLM_O_HI   = 40960u;  // [32][128] float (16 KB), ends at 57344 < 65536
    constexpr uint32_t SLM_TOT    = 65536u;

    const float scale_q = 1.0f / std::sqrt((float)kS);
    const auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};

    q.submit([&](sycl::handler& ch) {
        ch.parallel_for<TagWyState>(
            sycl::nd_range<1>((size_t)kHV * 32u, 32u), props, [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
                slm_init<SLM_TOT>();
                const int h   = (int)it.get_group(0);
                const int tid = (int)it.get_local_id(0);
                const int gm  = tid >> 3;  // 0..3 -> 32 K-rows (or tokens)
                const int gn  = tid & 7;   // 0..7 -> 16 V-cols
                const int qh  = h % kHK;

                simd<float, 128> S_acc[4];
#pragma unroll
                for (int m = 0; m < 4; ++m) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int sr = 32 * gm + 8 * m + r;
                        S_acc[m].template select<16, 1>(16 * r) =
                            block_load<float, 16>(state + ((size_t)sr * kHV + h) * kS + (size_t)(16 * gn));
                    }
                }

                for (int c = 0; c < nc; ++c) {
                    const int wg  = c * kHV + h;
                    const int t0  = c * kChunk;
                    const int len = (T - t0) < kChunk ? (T - t0) : kChunk;
                    simd<float, 64> gam_all;
                    gam_all.template select<32, 1>(0)  = block_load<float, 32>(ws_g + (size_t)wg * kChunk);
                    gam_all.template select<32, 1>(32) = block_load<float, 32>(ws_g + (size_t)wg * kChunk + 32);
                    const float g_end = float(gam_all[63]);

                    // Stage 1: pack S -> SLM_S_VNNI, W -> SLM_A_WQ[0..63], Q*exp(gamma) -> SLM_A_WQ[64..127].
#pragma unroll
                    for (int sk = 0; sk < 2; ++sk) {
                        const uint32_t tile_off = SLM_S_VNNI + (uint32_t)((2 * gm + sk) * 8 + gn) * 512u;
#pragma unroll
                        for (int p = 0; p < 4; ++p) {
                            slm_block_store<uint32_t, 16>(
                                tile_off + (uint32_t)p * 64u,
                                pack_vnni2_f(S_acc[2 * sk].template select<16, 1>(16 * (2 * p)),
                                             S_acc[2 * sk].template select<16, 1>(16 * (2 * p + 1))),
                                kAl16);
                            slm_block_store<uint32_t, 16>(
                                tile_off + (uint32_t)(p + 4) * 64u,
                                pack_vnni2_f(S_acc[2 * sk + 1].template select<16, 1>(16 * (2 * p)),
                                             S_acc[2 * sk + 1].template select<16, 1>(16 * (2 * p + 1))),
                                kAl16);
                        }
                    }

                    const sycl::half* const wg_W = ws_W + (size_t)wg * kChunk * kS;
                    for (int dr = 0; dr < 2; ++dr) {
                        const int lr = 2 * tid + dr;
                        const simd<sycl::half, 128> w_row = block_load<sycl::half, 128>(wg_W + (size_t)lr * kS);
                        slm_block_store<sycl::half, 128>(SLM_A_WQ + (uint32_t)lr * 256u, w_row, kAl16);

                        simd<sycl::half, 128> q_row(0.0f);
                        if (lr < len) {
                            const float sc = esimd_exp1(float(gam_all[lr]));
                            const float* qsrc = hc + (size_t)(t0 + lr) * kC + (size_t)qh * kS;
#pragma unroll
                            for (int d = 0; d < 4; ++d) {
                                q_row.template select<32, 1>(32 * d) =
                                    simd<sycl::half, 32>(block_load<float, 32>(qsrc + 32 * d) * sc);
                            }
                        }
                        slm_block_store<sycl::half, 128>(SLM_A_WQ + 16384u + (uint32_t)lr * 256u, q_row, kAl16);
                    }
                    barrier();

                    // Stage 2: [W; Q_s] @ S -> wq_acc[4] (gm=0,1 holds WS; gm=2,3 holds O_inter).
                    simd<float, 128> wq_acc[4];
#pragma unroll
                    for (int m = 0; m < 4; ++m) wq_acc[m] = 0.f;
                    for (int k16 = 0; k16 < 8; ++k16) {
                        const simd<sycl::half, 256> b = load_slm_b_h256(SLM_S_VNNI + (uint32_t)(k16 * 8 + gn) * 512u);
#pragma unroll
                        for (int m = 0; m < 4; ++m) {
                            const simd<sycl::half, 128> a = load_slm_a(SLM_A_WQ, 256u, 32 * gm + 8 * m, k16);
                            wq_acc[m] = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(wq_acc[m], b, a);
                        }
                    }
                    barrier();

                    // Stage 3: gm=0,1 writes delta=U-WS into SLM_DLT_V; gm=2,3 stages M_qk and K_tail^T.
                    if (gm < 2) {
                        const sycl::half* const wg_U = ws_U + (size_t)wg * kChunk * kS;
#pragma unroll
                        for (int sk = 0; sk < 2; ++sk) {
                            const uint32_t tile_off = SLM_DLT_V + (uint32_t)((2 * gm + sk) * 8 + gn) * 512u;
#pragma unroll
                            for (int p = 0; p < 4; ++p) {
                                const int lr0 = 32 * gm + 16 * sk + 2 * p;
                                const int lr1 = lr0 + 8;
                                const simd<float, 16> d0e =
                                    simd<float, 16>(block_load<sycl::half, 16>(wg_U + (size_t)lr0 * kS + 16 * gn)) -
                                    wq_acc[2 * sk].template select<16, 1>(16 * (2 * p));
                                const simd<float, 16> d0o =
                                    simd<float, 16>(block_load<sycl::half, 16>(wg_U + (size_t)(lr0 + 1) * kS + 16 * gn)) -
                                    wq_acc[2 * sk].template select<16, 1>(16 * (2 * p + 1));
                                slm_block_store<uint32_t, 16>(tile_off + (uint32_t)p * 64u, pack_vnni2_f(d0e, d0o), kAl16);

                                const simd<float, 16> d1e =
                                    simd<float, 16>(block_load<sycl::half, 16>(wg_U + (size_t)lr1 * kS + 16 * gn)) -
                                    wq_acc[2 * sk + 1].template select<16, 1>(16 * (2 * p));
                                const simd<float, 16> d1o =
                                    simd<float, 16>(block_load<sycl::half, 16>(wg_U + (size_t)(lr1 + 1) * kS + 16 * gn)) -
                                    wq_acc[2 * sk + 1].template select<16, 1>(16 * (2 * p + 1));
                                slm_block_store<uint32_t, 16>(tile_off + (uint32_t)(p + 4) * 64u, pack_vnni2_f(d1e, d1o), kAl16);
                            }
                        }
                    } else {
                        const int tid_hi = tid - 16;
                        const sycl::half* const wg_Mqk = ws_Mqk + (size_t)wg * kChunk * kChunk;
#pragma unroll
                        for (int dt = 0; dt < 4; ++dt) {
                            const int r = 4 * tid_hi + dt;
                            slm_block_store<sycl::half, 64>(SLM_MQK + (uint32_t)r * 128u,
                                                            block_load<sycl::half, 64>(wg_Mqk + (size_t)r * kChunk), kAl16);
                        }
                        const int tid_t = tid_hi >> 1;  // 0..7 -> 8 tokens
                        const int tid_d = tid_hi & 1;   // 0..1 -> 64 dims
                        simd<sycl::half, 64> kt[8];
#pragma unroll
                        for (int tok = 0; tok < 8; ++tok) {
                            const int lr = 8 * tid_t + tok;
                            if (lr < len) {
                                const float sc = esimd_exp1(g_end - float(gam_all[lr]));
                                const float* ksrc = hc + (size_t)(t0 + lr) * kC + (size_t)kHK * kS + (size_t)qh * kS + 64 * tid_d;
                                kt[tok].template select<32, 1>(0)  = simd<sycl::half, 32>(block_load<float, 32>(ksrc) * sc);
                                kt[tok].template select<32, 1>(32) = simd<sycl::half, 32>(block_load<float, 32>(ksrc + 32) * sc);
                            } else {
                                kt[tok] = sycl::half(0.0f);
                            }
                        }
                        for (int d8 = 0; d8 < 8; ++d8) {
#pragma unroll
                            for (int dd = 0; dd < 8; ++dd) {
                                const int d = 8 * d8 + dd;
                                simd<sycl::half, 8> col8;
                                col8[0] = kt[0][d]; col8[1] = kt[1][d]; col8[2] = kt[2][d]; col8[3] = kt[3][d];
                                col8[4] = kt[4][d]; col8[5] = kt[5][d]; col8[6] = kt[6][d]; col8[7] = kt[7][d];
                                slm_block_store<sycl::half, 8>(
                                    SLM_KTAIL + (uint32_t)(64 * tid_d + d) * 128u + (uint32_t)tid_t * 16u, col8, kAl16);
                            }
                        }
                    }
                    barrier();

                    // Stage 4: S = exp(g_end) * S + K_tail^T @ delta.
                    const float dec_end = esimd_exp1(g_end);
#pragma unroll
                    for (int m = 0; m < 4; ++m) S_acc[m] *= dec_end;
                    for (int k16 = 0; k16 < 4; ++k16) {
                        const simd<sycl::half, 256> b = load_slm_b_h256(SLM_DLT_V + (uint32_t)(k16 * 8 + gn) * 512u);
#pragma unroll
                        for (int m = 0; m < 4; ++m) {
                            const simd<sycl::half, 128> a = load_slm_a(SLM_KTAIL, 128u, 32 * gm + 8 * m, k16);
                            S_acc[m] = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(S_acc[m], b, a);
                        }
                    }
                    barrier(); // SLM_KTAIL free for SLM_O_LO / SLM_O_HI.

                    // Stage 5: gm=0,1 computes O_intra = M_qk @ delta while gm=2,3 stages O_inter to SLM.
                    simd<float, 128> ointra[4];
                    if (gm < 2) {
#pragma unroll
                        for (int m = 0; m < 4; ++m) ointra[m] = 0.f;
                        for (int k16 = 0; k16 < 4; ++k16) {
                            const simd<sycl::half, 256> b = load_slm_b_h256(SLM_DLT_V + (uint32_t)(k16 * 8 + gn) * 512u);
#pragma unroll
                            for (int m = 0; m < 4; ++m) {
                                const simd<sycl::half, 128> a = load_slm_a(SLM_MQK, 128u, 32 * gm + 8 * m, k16);
                                ointra[m] = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(ointra[m], b, a);
                            }
                        }
                    } else {
                        const uint32_t o_dst_slm = (gm == 2) ? SLM_O_LO : SLM_O_HI;
#pragma unroll
                        for (int m = 0; m < 4; ++m) {
#pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                const int sr = 8 * m + r;
                                slm_block_store<float, 16>(
                                    o_dst_slm + (uint32_t)sr * 512u + (uint32_t)(16 * gn) * 4u,
                                    wq_acc[m].template select<16, 1>(16 * r), kAl16);
                            }
                        }
                    }
                    barrier();

                    if (gm < 2) {
                        const uint32_t o_src_slm = (gm == 0) ? SLM_O_LO : SLM_O_HI;
#pragma unroll
                        for (int m = 0; m < 4; ++m) {
#pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                const int sr = 8 * m + r;
                                const int lr = 32 * gm + sr;
                                if (lr < len) {
                                    const simd<float, 16> oi = slm_block_load<float, 16>(
                                        o_src_slm + (uint32_t)sr * 512u + (uint32_t)(16 * gn) * 4u, kAl16);
                                    const simd<float, 16> tot = (oi + ointra[m].template select<16, 1>(16 * r)) * scale_q;
                                    block_store<float, 16>(out + ((size_t)(t0 + lr) * kHV + h) * kS + 16 * gn, tot);
                                }
                            }
                        }
                    }
                    barrier();
                }

#pragma unroll
                for (int m = 0; m < 4; ++m) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int sr = 32 * gm + 8 * m + r;
                        block_store<float, 16>(state + ((size_t)sr * kHV + h) * kS + (size_t)(16 * gn),
                                               S_acc[m].template select<16, 1>(16 * r));
                    }
                }
            });
    });
}

} // namespace

bool ok(const sycl::queue& q, int S, int HK, int HV, int C) {
    if (S != kS || HK != kHK || HV != kHV || C != kC) return false;
    const auto dev = q.get_device();
    return dev.is_gpu() && dev.has(sycl::aspect::fp16);
}

void forward(sycl::queue& q, void* ws, int ws_max_T, float* state, const float* hc,
             const float* gate, const float* beta, float* out, int T) {
    const int max_step = std::max(kChunk, (ws_max_T / kChunk) * kChunk);
    for (int off = 0; off < T; off += max_step) {
        const int tc = std::min(max_step, T - off);
        const int nc = (tc + kChunk - 1) / kChunk;
        const size_t nch = (size_t)nc * kHV;
        sycl::half* const ws_W   = reinterpret_cast<sycl::half*>(ws);
        sycl::half* const ws_U   = ws_W + nch * kChunk * kS;
        sycl::half* const ws_Mqk = ws_U + nch * kChunk * kS;
        float*      const ws_g   = reinterpret_cast<float*>(ws_Mqk + nch * kChunk * kChunk);

        const float* const hc_c   = hc   + (size_t)off * kC;
        const float* const gate_c = gate + (size_t)off * kHV;
        const float* const beta_c = beta + (size_t)off * kHV;
        float*       const out_c  = out  + (size_t)off * kHV * kS;

        launch_prep(q, ws_W, ws_U, ws_Mqk, ws_g, hc_c, gate_c, beta_c, tc, nc);
        launch_state(q, ws_W, ws_U, ws_Mqk, ws_g, state, hc_c, out_c, tc, nc);
    }
}

} // namespace strata::q36wy
