// q36_dflash_parity.cpp - parity and bench of the DFlash 2 drafter's kernels (q36_dflash.dp.cpp), run from the existing parity binary:
//
//     q36_mmvq_dense_parity --dflash              every check below (a few seconds), exit 1 on a mismatch
//     q36_mmvq_dense_parity --dflash --bench      + the whole pass on a random 1.2 GB pack: ms a pass, GB/s of weights against the 590 GB/s floor
//
// Every kernel against a float64 (or exact) host reference on random data.  The W4 products are q36xr::mmvq_w4h's (checked by its own
// parity) and are left out of the per-kernel checks; the bench runs them.  Contract per kernel (the bound is what the check enforces):
//   capture      bitwise: the bf16 of x rounded to nearest even
//   ctx_prep     the packed half operand bitwise (the scale is a power of two); the scale exactly
//   ctx_norm     |d| <= 1 half ulp + 1e-6 (fp32 rsqrt against float64)
//   ctx_store    K: 1 half ulp + 1e-4 (norm + RoPE in fp32, the RoPE table from float64); V bitwise
//   rms_pack / conv_pack / finish_rms / qk_prep  1 half ulp + 1e-6 relative (fp32 sums of 5,120 / 128 terms)
//   attn         2e-3 relative + 2e-4 (fp32 softmax, fp16 K/V rings, the packed half output)
//   top1/top2    bitwise: the same 16 ids and raw values as a host sort (ties to the lower index)
//   selector     the 7 picks exactly, the probabilities to 1e-4
// Data: N(0, 1)-like from a fixed generator; the bench's pack is random bytes (Xe2 compresses a constant fill: not used).
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "q36_dflash.hpp"
#include "q36_xr_dense.hpp"

namespace strata::q36df {
namespace {
using namespace k_consts;
constexpr int kKvAllRows = kLayers * 2 * kKvHeads * kHd;   // 10240: kv_all's rows

std::mt19937_64 g_rng(12345);
float rnd() {   // ~N(0, 1) (sum of 4 uniforms, scaled)
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    return (u(g_rng) + u(g_rng) + u(g_rng) + u(g_rng)) * 0.866f;
}
std::vector<float> rvec(size_t n, float scale = 1.f) { std::vector<float> v(n); for (auto& x : v) x = rnd() * scale; return v; }
uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7FFFu + ((u >> 16) & 1u); return (uint16_t) (u >> 16); }
float bf2f(uint16_t b) { uint32_t u = (uint32_t) b << 16; float f; std::memcpy(&f, &u, 4); return f; }
uint16_t h2u(float f) { sycl::half h(f); uint16_t u; std::memcpy(&u, &h, 2); return u; }
float u2h(uint16_t u) { sycl::half h; std::memcpy(&h, &u, 2); return (float) h; }
size_t x16_at(int col, int i, int ncp) { return ((size_t) (i >> 4) * ncp + col) * 16 + (i & 15); }

sycl::queue* g_q = nullptr;
template <class T> T* dev(size_t n) { return sycl::aligned_alloc_device<T>(4096, n, *g_q); }
template <class T> T* up(const std::vector<T>& v) { T* d = dev<T>(v.size() ? v.size() : 1); g_q->memcpy(d, v.data(), v.size() * sizeof(T)).wait(); return d; }
template <class T> std::vector<T> down(const T* d, size_t n) { std::vector<T> v(n); g_q->memcpy(v.data(), d, n * sizeof(T)).wait(); return v; }

int g_bad = 0;
void check(const char* name, bool ok, const std::string& info = "") {
    std::printf("%-44s %s %s\n", name, ok ? "ok  " : "FAIL", info.c_str());
    if (!ok) ++g_bad;
}
std::string fmt(const char* f, double a, double b = 0, double c = 0) { char s[160]; std::snprintf(s, sizeof s, f, a, b, c); return s; }
bool near_half(float got, double ref, double abs_tol = 1e-6) { return std::fabs((double) got - ref) <= std::fabs(ref) * 1.0 / 1024.0 + abs_tol; }

void rope_table(int64_t pos, float* out) {
    for (int i = 0; i < 64; ++i) {
        const double ang = (double) pos * std::pow(1.0e7, -2.0 * i / kHd);
        out[i] = (float) std::cos(ang);
        out[64 + i] = (float) std::sin(ang);
    }
}

// ---- the checks
void test_capture() {
    const int64_t p0 = 5000;
    for (int T : {70, 2100}) {
        auto x = rvec((size_t) T * kN, 3.f);
        float* dx = up(x);
        int32_t pos = (int32_t) p0;
        int32_t* dpos = up(std::vector<int32_t>{pos});
        uint16_t* stag = dev<uint16_t>((size_t) kRing * kTaps * kN);
        g_q->memset(stag, 0, (size_t) kRing * kTaps * kN * 2).wait();
        const int tap = 2;
        k::capture(*g_q, stag, tap, dx, T, dpos);
        g_q->wait();
        auto h = down(stag, (size_t) kRing * kTaps * kN);
        size_t bad = 0, n = 0;
        for (int r = std::max(0, T - kRing); r < T; ++r) {
            const size_t slot = (size_t) ((p0 + r) % kRing);
            for (int i = 0; i < kN; ++i, ++n) bad += h[(slot * kTaps + tap) * kN + i] != f2bf(x[(size_t) r * kN + i]);
        }
        check(T == 70 ? "capture: 70 rows, bf16 RNE" : "capture: 2100 rows (> the ring): the last 2064", bad == 0, fmt("%.0f of %.0f differ", (double) bad, (double) n));
        sycl::free(dx, *g_q); sycl::free(dpos, *g_q); sycl::free(stag, *g_q);
    }
}

void test_ctx() {
    const int n = 5;
    std::vector<uint16_t> stg((size_t) kRing * kTaps * kN);
    for (auto& v : stg) v = f2bf(rnd());
    const int slots[n] = {7, 100, 2063, 0, 1500};
    for (int i = 0; i < kTaps * kN; i += 997) stg[(size_t) slots[2] * kTaps * kN + i] = f2bf(30000.f);   // a row with massive entries: scale 4
    std::vector<int32_t> meta(kMetaWords, 0);
    for (int j = 0; j < n; ++j) meta[kMetaCtxSlot + j] = slots[j];
    uint16_t* dstg = up(stg);
    int32_t* dmeta = up(meta);
    const int ncp = 8;
    uint16_t* x16 = dev<uint16_t>((size_t) kTaps * kN * ncp);
    g_q->memset(x16, 0, (size_t) kTaps * kN * ncp * 2).wait();
    float* sc = dev<float>(8);
    k::ctx_prep(*g_q, dstg, dmeta, n, x16, sc);
    g_q->wait();
    auto hx = down(x16, (size_t) kTaps * kN * ncp);
    auto hs = down(sc, n);
    size_t bad = 0, bad_s = 0;
    for (int j = 0; j < n; ++j) {
        float am = 0.f;
        for (int i = 0; i < kTaps * kN; ++i) am = std::max(am, std::fabs(bf2f(stg[(size_t) slots[j] * kTaps * kN + i])));
        const float s = am > 8192.f ? std::exp2(std::ceil(std::log2(am / 8192.f))) : 1.f;
        bad_s += hs[j] != s;
        for (int i = 0; i < kTaps * kN; ++i) bad += hx[x16_at(j, i, ncp)] != h2u(bf2f(stg[(size_t) slots[j] * kTaps * kN + i]) * (1.f / s));
    }
    check("ctx_prep: packed halves bitwise, pow2 scale", bad == 0 && bad_s == 0, fmt("%.0f elements, %.0f scales differ", (double) bad, (double) bad_s));
    // ctx_norm
    auto y = rvec((size_t) n * kN, 2.f);
    auto w = rvec(kN, 0.3f);
    for (auto& v : w) v += 1.f;
    const float scale_h[8] = {1, 1, 4, 1, 1, 1, 1, 1};
    float *dy = up(y), *dw = up(w), *dsc = up(std::vector<float>(scale_h, scale_h + 8));
    uint16_t* o16 = dev<uint16_t>((size_t) kN * ncp);
    g_q->memset(o16, 0, (size_t) kN * ncp * 2).wait();
    k::ctx_norm(*g_q, dy, dsc, dw, 1e-6f, n, o16);
    g_q->wait();
    auto ho = down(o16, (size_t) kN * ncp);
    size_t badn = 0;
    for (int j = 0; j < n; ++j) {
        double ss = 0;
        for (int i = 0; i < kN; ++i) ss += (double) y[(size_t) j * kN + i] * y[(size_t) j * kN + i];
        const double r = 1.0 / std::sqrt(ss / kN + 1e-6 / ((double) scale_h[j] * scale_h[j]));
        for (int i = 0; i < kN; ++i) badn += !near_half(u2h(ho[x16_at(j, i, ncp)]), y[(size_t) j * kN + i] * r * w[i]);
    }
    check("ctx_norm: eps / s^2, hidden_norm, packed", badn == 0, fmt("%.0f elements out of bound", (double) badn));
    // ctx_store
    auto kv = rvec((size_t) n * kKvAllRows * 1, 1.5f);
    std::vector<float> kn[kLayers];
    const float* knp[kLayers];
    std::vector<float*> dkn(kLayers);
    for (int l = 0; l < kLayers; ++l) { kn[l] = rvec(kHd, 0.2f); for (auto& v : kn[l]) v += 1.f; dkn[l] = up(kn[l]); knp[l] = dkn[l]; }
    const int64_t pos[n] = {9, 5000, 262000, 0, 777};
    float* rc = reinterpret_cast<float*>(meta.data() + kMetaRopeCtx);
    for (int j = 0; j < n; ++j) { rope_table(pos[j], rc + j * 128); }
    int32_t* dmeta2 = up(meta);
    float* dkv = up(kv);
    sycl::half *kr = dev<sycl::half>((size_t) kLayers * kKvHeads * kRing * kHd), *vr = dev<sycl::half>((size_t) kLayers * kKvHeads * kRing * kHd);
    g_q->memset(kr, 0, (size_t) kLayers * kKvHeads * kRing * kHd * 2).wait();
    g_q->memset(vr, 0, (size_t) kLayers * kKvHeads * kRing * kHd * 2).wait();
    const float** dknp = sycl::malloc_shared<const float*>(kLayers, *g_q);
    for (int l = 0; l < kLayers; ++l) dknp[l] = knp[l];
    k::ctx_store(*g_q, dkv, dmeta2, n, dknp, kr, vr, 1e-6f);
    g_q->wait();
    auto hk = down(kr, (size_t) kLayers * kKvHeads * kRing * kHd);
    auto hv = down(vr, (size_t) kLayers * kKvHeads * kRing * kHd);
    size_t badk = 0, badv = 0;
    for (int j = 0; j < n; ++j)
        for (int l = 0; l < kLayers; ++l)
            for (int h = 0; h < kKvHeads; ++h) {
                const float* src = &kv[(size_t) j * kKvAllRows + (size_t) l * 2048];
                double ss = 0;
                for (int d = 0; d < kHd; ++d) ss += (double) src[h * kHd + d] * src[h * kHd + d];
                const double r = 1.0 / std::sqrt(ss / kHd + 1e-6);
                std::vector<double> xn(kHd);
                for (int d = 0; d < kHd; ++d) xn[d] = src[h * kHd + d] * r * kn[l][d];
                for (int d = 0; d < kHd; ++d) {
                    const int i = d & 63;
                    const double c = rc[j * 128 + i], s = rc[j * 128 + 64 + i];
                    const double ref = d < 64 ? xn[d] * c - xn[d + 64] * s : xn[d] * c + xn[d - 64] * s;
                    const size_t at = (((size_t) l * kKvHeads + h) * kRing + (size_t) meta[kMetaCtxSlot + j]) * kHd + d;
                    badk += !near_half((float) hk[at], ref, 1e-4);
                    badv += (float) hv[at] != u2h(h2u(src[1024 + h * kHd + d]));
                }
            }
    check("ctx_store: k_norm + RoPE (pos up to 262000), V", badk == 0 && badv == 0, fmt("%.0f K, %.0f V out of bound", (double) badk, (double) badv));
    for (void* p : {(void*) dstg, (void*) dmeta, (void*) x16, (void*) sc, (void*) dy, (void*) dw, (void*) dsc, (void*) o16, (void*) dmeta2, (void*) dkv, (void*) kr, (void*) vr, (void*) dknp})
        sycl::free(p, *g_q);
    for (auto* p : dkn) sycl::free(p, *g_q);
}

void test_block_rows() {
    const int R = kRows;
    auto x = rvec((size_t) R * kN, 2.f);
    auto w = rvec(kN, 0.2f);
    for (auto& v : w) v += 1.f;
    float *dx = up(x), *dw = up(w);
    float* dh = dev<float>((size_t) R * kN);
    uint16_t* x16 = dev<uint16_t>((size_t) R * kN);
    k::rms_pack(*g_q, dx, dw, 1e-6f, dh, x16);
    g_q->wait();
    auto hh = down(dh, (size_t) R * kN);
    auto hx = down(x16, (size_t) R * kN);
    size_t bad = 0;
    for (int r = 0; r < R; ++r) {
        double ss = 0;
        for (int i = 0; i < kN; ++i) ss += (double) x[(size_t) r * kN + i] * x[(size_t) r * kN + i];
        const double rr = 1.0 / std::sqrt(ss / kN + 1e-6);
        for (int i = 0; i < kN; ++i) {
            const double ref = x[(size_t) r * kN + i] * rr * w[i];
            bad += std::fabs(hh[(size_t) r * kN + i] - ref) > 1e-5 * std::fabs(ref) + 1e-6 || !near_half(u2h(hx[x16_at(r, i, kRows)]), ref);
        }
    }
    check("rms_pack: rms * w, f32 and packed", bad == 0, fmt("%.0f elements out of bound", (double) bad));
    // conv_pack (both sides) and finish_rms
    auto dyn = rvec((size_t) R * kKpOut, 0.3f);
    auto base = rvec((size_t) 4 * kN, 0.5f);
    float *ddyn = up(dyn), *dbase = up(base);
    for (int side = 0; side < 2; ++side) {
        g_q->memset(x16, 0, (size_t) R * kN * 2).wait();
        k::conv_pack(*g_q, dx, ddyn, side, dbase, x16);
        g_q->wait();
        auto hc = down(x16, (size_t) R * kN);
        size_t badc = 0;
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < kN; ++c) {
                const int g = c / 16;
                double ref = ((double) base[(side * 2 + 0) * kN + c] + dyn[(size_t) r * kKpOut + side * 640 + g]) * x[(size_t) r * kN + c];
                if (r >= 1) ref += ((double) base[(side * 2 + 1) * kN + c] + dyn[(size_t) r * kKpOut + side * 640 + 320 + g]) * x[(size_t) (r - 1) * kN + c];
                badc += !near_half(u2h(hc[x16_at(r, c, kRows)]), ref, 1e-5);
            }
        check(side == 0 ? "conv_pack side 0 (taps stop at the block start)" : "conv_pack side 1", badc == 0, fmt("%.0f out of bound", (double) badc));
    }
    // finish_rms: x += conv(o, side 1); h = rms(x) * w
    auto o = rvec((size_t) R * kN, 1.f);
    float *dox = up(o), *dres = up(x);
    k::finish_rms(*g_q, dox, ddyn, dbase, dres, dw, 1e-6f, dh, x16);
    g_q->wait();
    auto hx2 = down(dres, (size_t) R * kN);
    auto hh2 = down(dh, (size_t) R * kN);
    size_t badf = 0;
    for (int r = 0; r < R; ++r) {
        std::vector<double> xr(kN);
        double ss = 0;
        for (int c = 0; c < kN; ++c) {
            const int g = c / 16;
            double y = ((double) base[2 * kN + c] + dyn[(size_t) r * kKpOut + 640 + g]) * o[(size_t) r * kN + c];
            if (r >= 1) y += ((double) base[3 * kN + c] + dyn[(size_t) r * kKpOut + 640 + 320 + g]) * o[(size_t) (r - 1) * kN + c];
            xr[c] = x[(size_t) r * kN + c] + y;
            ss += xr[c] * xr[c];
        }
        const double rr = 1.0 / std::sqrt(ss / kN + 1e-6);
        for (int c = 0; c < kN; ++c) {
            badf += std::fabs(hx2[(size_t) r * kN + c] - xr[c]) > 1e-5 * std::fabs(xr[c]) + 1e-5;
            badf += std::fabs(hh2[(size_t) r * kN + c] - xr[c] * rr * w[c]) > 1e-4 * std::fabs(xr[c] * rr * w[c]) + 1e-5;
        }
    }
    check("finish_rms: residual += conv(side 1), then rms", badf == 0, fmt("%.0f out of bound", (double) badf));
    // qk_prep
    auto qy = rvec((size_t) R * 4096, 1.5f), kvy = rvec((size_t) R * 2048, 1.5f), qn = rvec(kHd, 0.2f), kn = rvec(kHd, 0.2f);
    for (auto& v : qn) v += 1.f;
    for (auto& v : kn) v += 1.f;
    std::vector<int32_t> meta(kMetaWords, 0);
    float* rb = reinterpret_cast<float*>(meta.data() + kMetaRopeBlk);
    const int64_t P = 131071;
    for (int r = 0; r < R; ++r) rope_table(P + r, rb + r * 128);
    float *dqy = up(qy), *dkvy = up(kvy), *dqn = up(qn), *dkn = up(kn);
    int32_t* dmeta = up(meta);
    float *qb = dev<float>((size_t) R * kHeads * kHd), *kb = dev<float>((size_t) R * kKvHeads * kHd), *vb = dev<float>((size_t) R * kKvHeads * kHd);
    k::qk_prep(*g_q, dqy, dkvy, dqn, dkn, dmeta, 1e-6f, qb, kb, vb);
    g_q->wait();
    auto hq = down(qb, (size_t) R * kHeads * kHd), hk = down(kb, (size_t) R * kKvHeads * kHd), hv = down(vb, (size_t) R * kKvHeads * kHd);
    size_t badq = 0, badv = 0;
    for (int r = 0; r < R; ++r)
        for (int h = 0; h < 40; ++h) {
            const bool isq = h < 32;
            const float* src = isq ? &qy[(size_t) r * 4096 + h * kHd] : &kvy[(size_t) r * 2048 + (h - 32) * kHd];
            double ss = 0;
            for (int d = 0; d < kHd; ++d) ss += (double) src[d] * src[d];
            const double rr = 1.0 / std::sqrt(ss / kHd + 1e-6);
            for (int d = 0; d < kHd; ++d) {
                const int i = d & 63;
                const double c = rb[r * 128 + i], s = rb[r * 128 + 64 + i];
                auto nv = [&](int e) { return src[e] * rr * (isq ? qn[e] : kn[e]); };
                const double ref = d < 64 ? nv(d) * c - nv(d + 64) * s : nv(d) * c + nv(d - 64) * s;
                const float got = isq ? hq[((size_t) r * kHeads + h) * kHd + d] : hk[((size_t) r * kKvHeads + (h - 32)) * kHd + d];
                badq += std::fabs(got - ref) > 1e-4 * std::fabs(ref) + 1e-5;
            }
        }
    for (int r = 0; r < R; ++r)
        for (int h = 0; h < kKvHeads; ++h)
            for (int d = 0; d < kHd; ++d) badv += hv[((size_t) r * kKvHeads + h) * kHd + d] != kvy[(size_t) r * 2048 + 1024 + h * kHd + d];
    check("qk_prep: per-head rms + RoPE (P = 131071), V copy", badq == 0 && badv == 0, fmt("%.0f q/k, %.0f v differ", (double) badq, (double) badv));
    for (void* p : {(void*) dx, (void*) dw, (void*) dh, (void*) x16, (void*) ddyn, (void*) dbase, (void*) dox, (void*) dres, (void*) dqy, (void*) dkvy, (void*) dqn, (void*) dkn,
                    (void*) dmeta, (void*) qb, (void*) kb, (void*) vb})
        sycl::free(p, *g_q);
}

void test_attention() {
    const int layer = 2;
    const size_t ring = (size_t) kLayers * kKvHeads * kRing * kHd;
    std::vector<sycl::half> kr(ring), vr(ring);
    for (size_t i = 0; i < ring; ++i) { kr[i] = sycl::half(rnd()); vr[i] = sycl::half(rnd()); }
    auto qb = rvec((size_t) kRows * kHeads * kHd, 1.f), kb = rvec((size_t) kRows * kKvHeads * kHd, 1.f), vb = rvec((size_t) kRows * kKvHeads * kHd, 1.f);
    sycl::half *dkr = up(kr), *dvr = up(vr);
    float *dqb = up(qb), *dkb = up(kb), *dvb = up(vb);
    float *pa = dev<float>((size_t) kChunks * kHeads * kRows * kHd), *pm = dev<float>((size_t) kChunks * kHeads * kRows), *pl = dev<float>((size_t) kChunks * kHeads * kRows);
    uint16_t* x16 = dev<uint16_t>((size_t) kRows * kHeads * kHd);
    struct Case { int P, vlo; const char* name; } cases[] = {{3000, 0, "P = 3000 (the window cuts, the ring wrapped)"}, {100, 0, "P = 100 (a short context)"},
                                                              {6000, 5400, "P = 6000, valid from 5400 (a restarted run)"}, {2500, 2500, "P = 2500, nothing valid (block keys only)"}};
    for (const Case& cs : cases) {
        std::vector<int32_t> meta(kMetaWords, 0);
        meta[kMetaP] = cs.P;
        meta[kMetaVlo] = cs.vlo;
        int32_t* dmeta = up(meta);
        g_q->memset(x16, 0, (size_t) kRows * kHeads * kHd * 2).wait();
        k::attn_part(*g_q, dqb, dkb, dvb, dkr, dvr, layer, dmeta, pa, pm, pl);
        k::attn_merge(*g_q, pa, pm, pl, x16);
        g_q->wait();
        auto hx = down(x16, (size_t) kRows * kHeads * kHd);
        size_t bad = 0, n = 0;
        double worst = 0;
        const int lo = std::max(cs.vlo, cs.P - (kWindow - 1));
        for (int r = 0; r < kRows; ++r)
            for (int h = 0; h < kHeads; ++h) {
                const int kvh = h / 4;
                std::vector<double> sc;
                std::vector<const sycl::half*> kk, vv;
                std::vector<int> kind;   // 0 ring, 1 block
                std::vector<int> idx;
                for (int pos = lo; pos < cs.P; ++pos) {
                    if (pos < cs.P + r - (kWindow - 1)) continue;
                    const size_t at = (((size_t) layer * kKvHeads + kvh) * kRing + (size_t) (pos % kRing)) * kHd;
                    kk.push_back(&kr[at]); vv.push_back(&vr[at]); kind.push_back(0); idx.push_back(pos);
                }
                const size_t nring = kk.size();
                for (int j = 0; j < kRows; ++j) { kind.push_back(1); idx.push_back(j); }
                std::vector<double> s(nring + kRows);
                double mx = -1e300;
                for (size_t t = 0; t < s.size(); ++t) {
                    double a = 0;
                    for (int d = 0; d < kHd; ++d) {
                        const double kv = t < nring ? (double) (float) kk[t][d] : (double) kb[((size_t) idx[t] * kKvHeads + kvh) * kHd + d];
                        a += (double) qb[((size_t) r * kHeads + h) * kHd + d] * kv;
                    }
                    s[t] = a * 0.08838834764831845;
                    mx = std::max(mx, s[t]);
                }
                double den = 0;
                for (auto& v : s) { v = std::exp(v - mx); den += v; }
                for (int d = 0; d < kHd; ++d, ++n) {
                    double o = 0;
                    for (size_t t = 0; t < s.size(); ++t) {
                        const double vvv = t < nring ? (double) (float) vv[t][d] : (double) vb[((size_t) idx[t] * kKvHeads + kvh) * kHd + d];
                        o += s[t] / den * vvv;
                    }
                    const double got = u2h(hx[x16_at(r, h * kHd + d, kRows)]);
                    const double err = std::fabs(got - o);
                    worst = std::max(worst, err / (std::fabs(o) + 0.1));
                    bad += err > 2e-3 * std::fabs(o) + 2e-4;
                }
            }
        check(cs.name, bad == 0, fmt("%.0f of %.0f out of bound, worst %.2e", (double) bad, (double) n, worst));
        sycl::free(dmeta, *g_q);
    }
    // the mask is the point: a CAUSAL block would differ.  Row 0 must see block row 7: perturb vb row 7 and watch row 0's output move.
    {
        std::vector<int32_t> meta(kMetaWords, 0);
        meta[kMetaP] = 100;   // (a short context: a block key keeps a weight of ~1 / 100)
        int32_t* dmeta = up(meta);
        auto vb2 = vb;
        for (int h = 0; h < kKvHeads; ++h) for (int d = 0; d < kHd; ++d) vb2[((size_t) 7 * kKvHeads + h) * kHd + d] += 5.f;
        float* dvb2 = up(vb2);
        k::attn_part(*g_q, dqb, dkb, dvb2, dkr, dvr, layer, dmeta, pa, pm, pl);
        uint16_t* y16 = dev<uint16_t>((size_t) kRows * kHeads * kHd);
        k::attn_merge(*g_q, pa, pm, pl, y16);
        g_q->wait();
        auto hy = down(y16, (size_t) kRows * kHeads * kHd);
        k::attn_part(*g_q, dqb, dkb, dvb, dkr, dvr, layer, dmeta, pa, pm, pl);   // the baseline
        k::attn_merge(*g_q, pa, pm, pl, x16);
        g_q->wait();
        auto hx = down(x16, (size_t) kRows * kHeads * kHd);
        double moved = 0;
        for (int d = 0; d < kHd; ++d) moved += std::fabs(u2h(hy[x16_at(0, d, kRows)]) - u2h(hx[x16_at(0, d, kRows)]));
        check("attention is NOT causal (row 0 reads block row 7)", moved > 0.1, fmt("row 0 moved by %.3f", moved));
        sycl::free(dmeta, *g_q); sycl::free(dvb2, *g_q); sycl::free(y16, *g_q);
    }
    for (void* p : {(void*) dkr, (void*) dvr, (void*) dqb, (void*) dkb, (void*) dvb, (void*) pa, (void*) pm, (void*) pl, (void*) x16}) sycl::free(p, *g_q);
}

void test_topk_select() {
    const int nd = 100003, rows = kDrafts, nch = (nd + 4095) / 4096;
    auto lg = rvec((size_t) rows * nd, 2.f);
    for (int r = 0; r < rows; ++r) { lg[(size_t) r * nd + 777] = 50.f + r; lg[(size_t) r * nd + 50000] = 50.f + r; lg[(size_t) r * nd + nd - 1] = 49.f; }   // a tie, and the last column
    float* dlg = up(lg);
    uint64_t* part = dev<uint64_t>((size_t) rows * 64 * kTopK);
    int32_t *cand = dev<int32_t>(rows * kTopK);
    float* un = dev<float>(rows * kTopK);
    std::vector<int32_t> map(nd);
    for (int i = 0; i < nd; ++i) map[i] = (int32_t) (((int64_t) i * 7 + 3) % 248320);
    int32_t* dmap = up(map);
    for (int variant = 0; variant < 2; ++variant) {
        k::top1(*g_q, dlg, nd, rows, nch, part);
        k::top2(*g_q, part, nch, rows, variant ? dmap : nullptr, cand, un);
        g_q->wait();
        auto hc = down(cand, rows * kTopK);
        auto hu = down(un, rows * kTopK);
        size_t bad = 0;
        for (int r = 0; r < rows; ++r) {
            std::vector<int> o(nd);
            std::iota(o.begin(), o.end(), 0);
            std::partial_sort(o.begin(), o.begin() + kTopK, o.end(), [&](int a, int b) { const float x = lg[(size_t) r * nd + a], y = lg[(size_t) r * nd + b]; return x > y || (x == y && a < b); });
            for (int c = 0; c < kTopK; ++c) bad += hc[r * kTopK + c] != (variant ? map[o[c]] : o[c]) || hu[r * kTopK + c] != lg[(size_t) r * nd + o[c]];
        }
        check(variant ? "top-16 (2 stages) with an id map" : "top-16 (2 stages), tie -> the lower index", bad == 0, fmt("%.0f of %.0f entries differ", (double) bad, (double) rows * kTopK));
    }
    // selector walk
    auto hp = rvec((size_t) kRows * kRank, 1.f);
    std::vector<uint16_t> pred((size_t) 3000 * kRank), succ((size_t) 3000 * kRank);
    for (auto& v : pred) v = f2bf(rnd() * 0.5f);
    for (auto& v : succ) v = f2bf(rnd() * 0.5f);
    std::vector<int32_t> cd(kDrafts * kTopK);
    std::vector<float> uy(kDrafts * kTopK);
    for (auto& v : cd) v = (int32_t) (g_rng() % 3000);
    for (auto& v : uy) v = rnd() * 3.f;
    std::vector<int32_t> meta(kMetaWords, 0);
    meta[kMetaAnchor] = 1234;
    float* dhp = up(hp);
    uint16_t *dpred = up(pred), *dsucc = up(succ);
    int32_t *dcd = up(cd), *dmeta = up(meta), *otok = dev<int32_t>(kDrafts);
    float *duy = up(uy), *oprob = dev<float>(kDrafts);
    k::selector_walk(*g_q, dhp, dcd, duy, dpred, dsucc, dmeta, otok, oprob);
    g_q->wait();
    auto ht = down(otok, kDrafts);
    auto hpr = down(oprob, kDrafts);
    int prev = 1234;
    size_t bad = 0;
    double perr = 0;
    for (int l = 0; l < kDrafts; ++l) {
        std::vector<double> s(kTopK);
        for (int c = 0; c < kTopK; ++c) {
            double a = uy[l * kTopK + c];
            for (int r = 0; r < kRank; ++r) a += (double) bf2f(pred[(size_t) prev * kRank + r]) * hp[(size_t) (l + 1) * kRank + r] * bf2f(succ[(size_t) cd[l * kTopK + c] * kRank + r]);
            s[c] = a;
        }
        int bj = 0;
        for (int c = 1; c < kTopK; ++c) if (s[c] > s[bj]) bj = c;
        double den = 0;
        for (double v : s) den += std::exp(v - s[bj]);
        bad += ht[l] != cd[l * kTopK + bj];
        perr = std::max(perr, std::fabs(hpr[l] - 1.0 / den));
        prev = cd[l * kTopK + bj];
    }
    check("selector walk: 7 picks, probabilities", bad == 0 && perr < 1e-4, fmt("%.0f picks differ, prob err %.1e", (double) bad, perr));
    for (void* p : {(void*) dlg, (void*) part, (void*) cand, (void*) un, (void*) dmap, (void*) dhp, (void*) dpred, (void*) dsucc, (void*) dcd, (void*) dmeta, (void*) otok, (void*) duy, (void*) oprob})
        sycl::free(p, *g_q);
}

// ---- the whole pass on a random pack
struct Bench {
    Dflash d;
    float* emb = nullptr;
    int nd = 65536;
};

int whole(bool bench) {
    Bench* b = new Bench;
    std::string err;
    if (!b->d.load_synthetic(*g_q, 99, err)) { std::printf("load_synthetic: %s\n", err.c_str()); return 1; }
    b->emb = dev<float>((size_t) kRows * kN);
    g_q->parallel_for(sycl::range<1>((size_t) kRows * kN), [e = b->emb](sycl::id<1> i) { e[i] = ((float) ((i[0] * 2654435761u) & 0xFFFF) / 32768.f - 1.f) * 0.5f; }).wait();
    Hooks hk;
    hk.max_nd = b->nd;
    hk.embed = [&](const int32_t*, int n, float* out) { g_q->memcpy(out, b->emb, (size_t) n * kN * 4); };
    hk.head = [&](const float* hn, int n, float* lg, int& nd, const int32_t*& idmap) {
        const int ndv = b->nd;
        g_q->parallel_for(sycl::range<1>((size_t) n * ndv), [=](sycl::id<1> i) {   // a deterministic function of the row's first element, so the picks move with it
            const uint64_t h = (uint64_t) i[0] * 0x9E3779B97F4A7C15ULL;
            lg[i[0]] = ((float) ((h >> 40) & 0xFFFF) / 32768.f - 1.f) * 3.f + hn[(i[0] / ndv) * kN] * 0.01f;
        });
        nd = ndv;
        idmap = nullptr;
    };
    b->d.set_hooks(*g_q, hk);
    // 100 context rows captured through capture(): x rows of every tap
    const int T = 100;
    auto x = rvec((size_t) T * kN, 2.f);
    float* dx = up(x);
    int32_t* dpos = up(std::vector<int32_t>{0});
    for (int k = 0; k < kTaps; ++k) b->d.capture(*g_q, k, dx, T, dpos);
    b->d.note_forward(0, T);
    const int n = b->d.prepare(*g_q, T, 4242);
    b->d.run(*g_q, n);
    g_q->wait();
    auto tok = down(b->d.out_tokens(), kDrafts);
    auto pr = down(b->d.out_probs(), kDrafts);
    bool ok = n == kRows;
    for (int l = 0; l < kDrafts; ++l) ok = ok && tok[l] >= 0 && tok[l] < b->nd && pr[l] > 0.f && pr[l] <= 1.f;
    check("whole pass on a random pack: 7 tokens in range", ok, fmt("n = %.0f context rows, first token %.0f", (double) n, (double) tok[0]));
    if (!bench) return 0;
    // bytes a pass reads: the block's weights (every layer's q, o, gate, up, down, 2 kernel projections, its kv slice) + hproj; the context update's fc + kv_all
    auto w4 = [](int n_in, int n_out) { return (double) strata::q36xr::w4_bytes(n_in, n_out); };
    const double blk = kLayers * (w4(kN, 4096) + w4(4096, kN) + 2 * w4(kN, kFF) + w4(kFF, kN) + 2 * w4(kN, kKpOut) + w4(kN, 2048)) + w4(kN, kRank);
    const double ctx = w4(kTaps * kN, kN) + w4(kN, kKvAllRows);
    auto time_it = [&](const char* name, double bytes, auto&& fn) {
        const auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.4) { fn(); g_q->wait(); }   // warm-up
        std::vector<double> ms;
        for (int i = 0; i < 15; ++i) {
            const auto a = std::chrono::steady_clock::now();
            fn();
            g_q->wait();
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
        }
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2];
        std::printf("%-34s %8.3f ms (min %.3f)  %7.1f GB/s of %.0f MB   floor at 590 GB/s: %.3f ms\n", name, med, ms[0], bytes / med / 1e6, bytes / 1e6, bytes / 590e6);
    };
    time_it("block pass (8 rows), no head", blk, [&] { b->d.block(*g_q); });
    time_it("context update, 8 rows (fc + kv_all)", ctx, [&] { b->d.ctx_update(*g_q, 8, false); });
    time_it("both (run(8))", blk + ctx, [&] { b->d.run(*g_q, 8); });
    std::printf("(host-submitted kernels, not a graph: add ~0.59 us a kernel boundary less in the engine's replayed graph; the engine's draft head %s is not in these numbers)\n", "(Q6_K 1.04 GB / the 64K draft vocabulary)");
    return 0;
}

}   // namespace

int parity_main(int argc, char** argv) {
    bool bench = false;
    for (int i = 0; i < argc; ++i) if (std::string(argv[i]) == "--bench") bench = true;
    g_q = &dpct::get_in_order_queue();
    strata::q36xr::init(g_q);
    test_capture();
    test_ctx();
    test_block_rows();
    test_attention();
    test_topk_select();
    if (whole(bench)) ++g_bad;
    std::printf("q36_dflash parity: %s (%d failed)\n", g_bad ? "FAIL" : "OK", g_bad);
    return g_bad ? 1 : 0;
}

}   // namespace strata::q36df
