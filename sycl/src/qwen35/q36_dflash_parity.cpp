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
#include <map>
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
// float -> IEEE half, round to nearest even, every range.  NOT sycl::half(float) on the host: the DPC++ host conversion (half_type.hpp,
// detail::float2Half) rounds the normal range correctly but TRUNCATES results below 2^-13 (its subnormal branch shifts the bits out), while the
// device rounds to nearest even.  Against that reference a correct kernel "differs" in the one element in ~10^4 whose half is subnormal.
uint16_t h2u(float f) {
    uint32_t b; std::memcpy(&b, &f, 4);
    const uint16_t sign = (uint16_t) ((b >> 16) & 0x8000u);
    if (std::isnan(f)) return (uint16_t) (sign | 0x7E00u);
    if (std::isinf(f)) return (uint16_t) (sign | 0x7C00u);
    const double a = std::fabs((double) f);
    if (a < 6.103515625e-05) return (uint16_t) (sign | (uint16_t) std::nearbyint(std::ldexp(a, 24)));   // subnormal (1024 = the smallest normal, by carry)
    int e = std::ilogb(a);
    double q = std::nearbyint(std::ldexp(a, 10 - e));                                                      // [1024, 2048]
    if (q >= 2048.0) { q = 1024.0; ++e; }
    if (e > 15) return (uint16_t) (sign | 0x7C00u);
    return (uint16_t) (sign | (uint16_t) (((e + 15) << 10) + ((int) q - 1024)));
}
float u2h(uint16_t u) {
    const int e = (u >> 10) & 31, m = u & 1023;
    const double v = e == 0 ? std::ldexp((double) m, -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp(1024.0 + m, e - 25);
    return (float) ((u & 0x8000) ? -v : v);
}
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
// The device's float -> half is IEEE round-to-nearest-even in every range (ties and subnormals included): the premise of the bitwise checks below.
void test_half_convert() {
    std::vector<float> v = {1.f + 0x1p-11f, 1.f + 3 * 0x1p-11f, 0x1p-25f, 3 * 0x1p-25f, 1.5f * 0x1p-24f, 2.5f * 0x1p-24f, 0x1p-14f - 0x1p-25f, 65519.f, 65520.f, -0x1.ffcp-15f, 3.3e-6f, 1e-8f};
    for (int i = 0; i < 4096; ++i) v.push_back(rnd() * 1e-4f);   // the subnormal and near-subnormal range
    for (int i = 0; i < 4096; ++i) v.push_back(rnd() * 3.f);
    float* dv = up(v);
    uint16_t* dh = dev<uint16_t>(v.size());
    g_q->parallel_for(sycl::range<1>(v.size()), [=](sycl::id<1> i) { sycl::half h(dv[i[0]]); dh[i[0]] = sycl::bit_cast<uint16_t>(h); }).wait();
    auto h = down(dh, v.size());
    size_t bad = 0;
    for (size_t i = 0; i < v.size(); ++i) bad += h[i] != h2u(v[i]);
    check("float -> half on the device: IEEE RNE (ties, subnormals)", bad == 0, fmt("%.0f of %.0f differ from the exact RNE", (double) bad, (double) v.size()));
    sycl::free(dv, *g_q); sycl::free(dh, *g_q);
}

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
struct Bench;
void batch_vs_singles(Dflash& d);
struct Bench {
    Dflash d;
    float* emb = nullptr;
    int nd = 65536;
};

// A batch pass of 3 slots (different positions, anchors and context lengths) against the 3 single passes: the weights are read once for 24 columns and every
// kernel keeps its block to itself, so each block's 7 picks and probabilities must be BITWISE the single pass's (the products' columns are independent, the
// reductions run in the same order).  Batch first (the context columns of the 3 slots share one product), then the singles on the same staging rows.
void batch_vs_singles(Dflash& d) {
    const int NS = 3, Ts[NS] = {100, 70, 2300};
    const int32_t anchors[NS] = {4242, 17, 99999};
    std::vector<float*> dxs;
    int32_t* dpos = up(std::vector<int32_t>{0});
    for (int s = 0; s < NS; ++s) {
        auto x = rvec((size_t) Ts[s] * kN, 2.f);
        float* dx = up(x);
        dxs.push_back(dx);
        for (int k = 0; k < kTaps; ++k) d.capture(*g_q, k, dx, Ts[s], dpos, s);
    }
    g_q->wait();
    auto go_batch = [&](int nb, int32_t* tok, float* prob) {
        for (int s = 0; s < nb; ++s) { d.invalidate(s); d.note_forward(0, Ts[s], s); }
        Dflash::Req rq[NS];
        for (int s = 0; s < nb; ++s) rq[s] = {s, (int64_t) Ts[s], anchors[s]};
        int cols[NS] = {0, 0, 0};
        const bool ok = d.prepare_batch(*g_q, rq, nb, cols);
        int tot = 0;
        for (int s = 0; s < nb; ++s) tot += cols[s];
        d.run_batch(*g_q, nb, tot);
        g_q->wait();
        auto t = down(d.out_tokens(), (size_t) nb * kDrafts);
        auto p = down(d.out_probs(), (size_t) nb * kDrafts);
        std::copy(t.begin(), t.end(), tok);
        std::copy(p.begin(), p.end(), prob);
        return ok && tot == 8 * nb;
    };
    int32_t tb[NS * kDrafts], ts[NS * kDrafts];
    float pb[NS * kDrafts], ps[NS * kDrafts];
    const bool ok_b = go_batch(NS, tb, pb);
    // the singles: the same slots, one at a time, from scratch
    for (int s = 0; s < NS; ++s) {
        d.invalidate(s);
        d.note_forward(0, Ts[s], s);
        const int n = d.prepare(*g_q, Ts[s], anchors[s], s);
        d.run(*g_q, n);
        g_q->wait();
        auto t = down(d.out_tokens(), kDrafts);
        auto p = down(d.out_probs(), kDrafts);
        for (int l = 0; l < kDrafts; ++l) { ts[s * kDrafts + l] = t[l]; ps[s * kDrafts + l] = p[l]; }
    }
    size_t bad = 0, badp = 0;
    for (int i = 0; i < NS * kDrafts; ++i) { bad += tb[i] != ts[i]; badp += std::memcmp(&pb[i], &ps[i], 4) != 0; }
    bool distinct = false;
    for (int l = 0; l < kDrafts; ++l) distinct = distinct || tb[l] != tb[kDrafts + l];
    check("batch pass of 3 slots == 3 single passes (bitwise)", ok_b && bad == 0 && badp == 0 && distinct,
          fmt("%.0f picks and %.0f probabilities differ", (double) bad, (double) badp) + (distinct ? "" : " (the blocks gave the same picks: the test is blind)"));
    for (auto* p : dxs) sycl::free(p, *g_q);
    sycl::free(dpos, *g_q);
}

// STRATA_Q36_DFLASH_TREE: the gather image against the buffers it copies, the host's re-computation of the selector against the device's walk (the same 7 picks and
// probabilities: it proves the host scores the tree with the device's formula), the tree built from it.
void test_tree_gather(Dflash& d, const std::vector<int32_t>& tok, const std::vector<float>& pr, const Debug& dbg) {
    namespace dt = strata::q36dt;
    check("tree gather: switched on, image allocated", d.tree_on(), "");
    std::vector<char> gb(sizeof(dt::Gather));
    g_q->memcpy(gb.data(), d.tree_gather(), gb.size()).wait();
    const dt::Gather& g = *reinterpret_cast<const dt::Gather*>(gb.data());
    bool ok = g.anchor == 4242;
    for (int l = 0; l < kDrafts; ++l) ok = ok && g.out_tok[l] == tok[l];
    ok = ok && std::memcmp(g.cand, dbg.cand.data(), sizeof g.cand) == 0 && std::memcmp(g.unary, dbg.unary.data(), sizeof g.unary) == 0;
    check("tree gather: anchor, picks, candidates, logits", ok, ok ? "" : fmt("anchor %.0f first pick %.0f vs %.0f", (double) g.anchor, (double) g.out_tok[0], (double) tok[0]));
    // pred / succ rows: the pack's tensors, row by row
    const uint16_t* pc = static_cast<const uint16_t*>(d.tensor("pred_cb"));
    const uint16_t* sc = static_cast<const uint16_t*>(d.tensor("succ_cb"));
    size_t bad = 0;
    std::vector<uint16_t> row(kRank);
    for (int r = 0; r < dt::kPredRows; ++r) {
        const int32_t id = r == 0 ? g.anchor : g.cand[(r - 1) / dt::kTopK][(r - 1) % dt::kTopK];
        g_q->memcpy(row.data(), pc + (size_t) id * kRank, kRank * 2).wait();
        bad += std::memcmp(row.data(), g.pred[r], kRank * 2) != 0;
    }
    for (int l = 0; l < kDrafts; ++l) for (int c = 0; c < dt::kTopK; ++c) {
        g_q->memcpy(row.data(), sc + (size_t) g.cand[l][c] * kRank, kRank * 2).wait();
        bad += std::memcmp(row.data(), g.succ[l][c], kRank * 2) != 0;
    }
    check("tree gather: pred_cb / succ_cb rows", bad == 0, fmt("%.0f of %.0f rows differ", (double) bad, (double) (dt::kPredRows + kDrafts * dt::kTopK)));
    // the host's walk over the gather = the device's walk
    int row_i = 0, bad_pick = 0;
    double perr = 0;
    for (int l = 0; l < kDrafts; ++l) {
        float s[dt::kTopK];
        dt::scores(g, l, row_i, s);
        int bj = 0;
        for (int c = 1; c < dt::kTopK; ++c) if (s[c] > s[bj]) bj = c;
        double den = 0;
        for (int c = 0; c < dt::kTopK; ++c) den += std::exp((double) s[c] - (double) s[bj]);
        bad_pick += g.cand[l][bj] != g.out_tok[l];
        perr = std::max(perr, std::fabs(1.0 / den - (double) pr[(size_t) l]));
        row_i = 1 + l * dt::kTopK + bj;
    }
    check("tree: host selector walk == the device's", bad_pick == 0 && perr < 1e-4, fmt("%.0f picks differ, prob err %.1e", (double) bad_pick, perr));
    dt::Params p;
    const auto nodes = dt::build(g, p);
    const dt::Layout L = dt::layout(nodes);
    bool spine = nodes.size() >= 8;
    for (int i = 1; i <= kDrafts && spine; ++i) spine = nodes[(size_t) i].tok == g.out_tok[i - 1] && nodes[(size_t) i].parent == i - 1;
    check("tree: built from the gather (spine = the walk, layout ok)", spine && L.ok && L.T == (int) nodes.size(), fmt("%.0f rows, slots %.0f", (double) nodes.size(), (double) L.slots));
    std::printf("  tree from the random pack: %d rows (spine 7 + %d siblings / branches), checkpoint slots %d, deepest %d\n", (int) nodes.size(), (int) nodes.size() - 8, L.slots, L.max_depth);
}

// compact_stage: capture 14 distinct rows at positions that wrap the ring, keep a path, and the rows must sit at p0 + k bit for bit
void test_compact_stage(Dflash& d) {
    const int T = 14;
    const int64_t p0 = kRing - 5;   // rows land in cells kRing - 5 .. kRing - 1, 0 .. 8
    d.invalidate();
    auto x = rvec((size_t) T * kN, 2.f);
    float* dx = up(x);
    int32_t* dpos = up(std::vector<int32_t>{(int32_t) p0});
    for (int k = 0; k < kTaps; ++k) d.capture(*g_q, k, dx, T, dpos);
    d.note_forward(p0, T);
    g_q->wait();
    std::vector<uint16_t> before, after;
    const bool got0 = d.debug_stage_rows(*g_q, p0, T, before);
    const int32_t path[6] = {0, 2, 3, 7, 8, 13};
    d.compact_stage(*g_q, p0, path, 6);
    g_q->wait();
    const bool got1 = d.debug_stage_rows(*g_q, p0, T, after);
    bool ok = got0 && got1;
    size_t bad = 0;
    const size_t rowe = (size_t) kTaps * kN;
    for (int k = 0; ok && k < 6; ++k) bad += std::memcmp(after.data() + (size_t) k * rowe, before.data() + (size_t) path[k] * rowe, rowe * 2) != 0;
    for (int r = 6; ok && r < T; ++r) bad += std::memcmp(after.data() + (size_t) r * rowe, before.data() + (size_t) r * rowe, rowe * 2) != 0;   // the rows past the kept ones stay as they were
    check("compact_stage: kept rows at p0 + k across the ring's wrap", ok && bad == 0, fmt("%.0f rows differ", (double) bad));
    d.invalidate();
    sycl::free(dx, *g_q);
    sycl::free(dpos, *g_q);
}

// ---- STRATA_Q36_DFLASH_W4X: the int8 route against the fp16 one
constexpr int kKvAllRows_() { return kLayers * 2 * kKvHeads * kHd; }
struct W4Shape { const char* name; int n_in, n_out; std::vector<const void*> w; };
std::vector<W4Shape> block_shapes(Dflash& d) {
    std::vector<W4Shape> sh = {{"ckp  5120 -> 1280", kN, kKpOut, {}}, {"q    5120 -> 4096", kN, kHeads * kHd, {}}, {"kv   5120 -> 2048", kN, 2048, {}}, {"o    4096 -> 5120", kHeads * kHd, kN, {}},
                               {"gate/up 5120 -> 17408", kN, kFF, {}}, {"down 17408 -> 5120", kFF, kN, {}}, {"hproj 5120 -> 256", kN, kRank, {}}, {"fc   25600 -> 5120", kTaps * kN, kN, {}},
                               {"kv_all 5120 -> 10240", kN, kKvAllRows_(), {}}};
    for (int l = 0; l < kLayers; ++l) {
        const std::string p = "l" + std::to_string(l) + ".";
        sh[0].w.push_back(d.tensor(p + "akp")); sh[0].w.push_back(d.tensor(p + "mkp"));
        sh[1].w.push_back(d.tensor(p + "q"));
        sh[2].w.push_back(static_cast<const uint8_t*>(d.tensor("kv_all")) + (size_t) l * (2048 / 16) * (kN / 256) * 2176);
        sh[3].w.push_back(d.tensor(p + "o"));
        sh[4].w.push_back(d.tensor(p + "gate")); sh[4].w.push_back(d.tensor(p + "up"));
        sh[5].w.push_back(d.tensor(p + "down"));
    }
    sh[6].w.push_back(d.tensor("hproj"));
    sh[7].w.push_back(d.tensor("fc"));
    sh[8].w.push_back(d.tensor("kv_all"));
    return sh;
}

// mmvq_w4x on int8 activations against mmvq_w4h on fp16 operands, same weights, same random activations: the relative L2 distance of the outputs is the int8 quantization's
// cost (one scale a group of 128 inputs, rounding to +-127: ~0.5% of a group's largest value a term)
void test_w4x_products(Dflash& d) {
    auto shapes = block_shapes(d);
    double worst = 0;
    bool ok = true, declined = false;
    std::string where;
    for (auto& sh : shapes) {
        for (int nc : {16, 24, 32}) {
            auto x = rvec((size_t) nc * sh.n_in, 1.f);
            float* dxf = up(x);
            uint16_t* dx16 = dev<uint16_t>(strata::q36xr::w4h_act_halfs(sh.n_in, nc));
            strata::q36xr::w4h_pack(dxf, sh.n_in, dx16, sh.n_in, nc, g_q);
            void* act = dev<uint8_t>(strata::q36xr::w4x_act_bytes(sh.n_in, nc, 128));
            strata::q36xr::w4x_quantize(dxf, sh.n_in, act, sh.n_in, nc, 128, g_q);
            float* yh = dev<float>((size_t) nc * sh.n_out);
            float* yx = dev<float>((size_t) nc * sh.n_out);
            const bool a = strata::q36xr::mmvq_w4h(sh.w[0], dx16, yh, sh.n_in, sh.n_out, nc, g_q);
            const bool b = strata::q36xr::mmvq_w4x(sh.w[0], act, yx, sh.n_in, sh.n_out, nc, 128, g_q);
            g_q->wait();
            declined = declined || !a || !b;
            if (a && b) {
                auto h = down(yh, (size_t) nc * sh.n_out), xx = down(yx, (size_t) nc * sh.n_out);
                double num = 0, den = 0;
                size_t nonfin = 0;
                for (size_t i = 0; i < h.size(); ++i) {
                    if (!std::isfinite(xx[i])) { ++nonfin; continue; }
                    num += ((double) xx[i] - h[i]) * ((double) xx[i] - h[i]);
                    den += (double) h[i] * h[i];
                }
                const double rel = std::sqrt(num / std::max(den, 1e-30));
                if (nonfin || rel > 0.02) { ok = false; where += fmt(" [%.0f cols, rel %.4f]", nc, rel) + sh.name; }
                worst = std::max(worst, rel);
            }
            for (void* p : {(void*) dxf, (void*) dx16, act, (void*) yh, (void*) yx}) sycl::free(p, *g_q);
        }
    }
    check("W4X products == W4H products within int8 quantization (16/24/32 columns)", ok && !declined, fmt("worst relative L2 distance %.4f (bound 0.02)", worst) + where);
}

// the pass: a batch pass of 2 blocks with the int8 route against the fp16 one: the final-normed rows must stay close (cosine), and the int8 route must repeat bitwise
void test_w4x_pass(Dflash& d) {
    const int NS = 2, Ts[NS] = {100, 70};
    const int32_t anchors[NS] = {4242, 17};
    std::vector<float*> dxs;
    int32_t* dpos = up(std::vector<int32_t>{0});
    for (int s = 0; s < NS; ++s) {
        auto x = rvec((size_t) Ts[s] * kN, 2.f);
        float* dx = up(x);
        dxs.push_back(dx);
        for (int k = 0; k < kTaps; ++k) d.capture(*g_q, k, dx, Ts[s], dpos, s);
    }
    g_q->wait();
    auto go = [&](std::vector<float>& h) {
        for (int s = 0; s < NS; ++s) { d.invalidate(s); d.note_forward(0, Ts[s], s); }
        Dflash::Req rq[NS] = {{0, Ts[0], anchors[0]}, {1, Ts[1], anchors[1]}};
        int cols[NS] = {0, 0};
        d.prepare_batch(*g_q, rq, NS, cols);
        d.run_batch(*g_q, NS, cols[0] + cols[1]);
        g_q->wait();
        d.debug_h(*g_q, 8 * NS, h);
    };
    std::vector<float> h0, h1, h2;
    d.set_w4x(*g_q, false);
    go(h0);
    const bool on = d.set_w4x(*g_q, true, 16);
    go(h1);
    go(h2);
    d.set_w4x(*g_q, false);
    double mincos = 1, maxrel = 0;
    size_t nonfin = 0;
    for (int r = 0; on && r < 8 * NS; ++r) {
        double dot = 0, a2 = 0, b2 = 0, df = 0;
        for (int i = 0; i < kN; ++i) {
            const double a = h0[(size_t) r * kN + i], b = h1[(size_t) r * kN + i];
            if (!std::isfinite(b)) ++nonfin;
            dot += a * b; a2 += a * a; b2 += b * b; df += (a - b) * (a - b);
        }
        mincos = std::min(mincos, dot / std::sqrt(a2 * b2 + 1e-30));
        maxrel = std::max(maxrel, std::sqrt(df / std::max(a2, 1e-30)));
    }
    const bool same = on && std::memcmp(h1.data(), h2.data(), h1.size() * 4) == 0;
    std::printf("  W4X pass: final-normed rows vs the fp16 route: min cosine %.5f, max relative L2 %.4f, %.0f non-finite\n", mincos, maxrel, (double) nonfin);
    check("W4X batch pass: rows close to the fp16 route (cosine > 0.9: a sanity floor, the card's acceptance is the test)", on && nonfin == 0 && mincos > 0.9, fmt("min cosine %.4f", mincos));
    check("W4X batch pass repeats bitwise", same);
    for (auto* p : dxs) sycl::free(p, *g_q);
    sycl::free(dpos, *g_q);
}

template <class F> double median_us(F&& fn, int per = 40) {
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.2) { for (int i = 0; i < per; ++i) fn((size_t) i); g_q->wait(); }
    std::vector<double> us;
    for (int rep = 0; rep < 9; ++rep) {
        const auto a = std::chrono::steady_clock::now();
        for (int i = 0; i < per; ++i) fn((size_t) i);
        g_q->wait();
        us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count() / per);
    }
    std::sort(us.begin(), us.end());
    return us[us.size() / 2];
}

// the product table at 8 / 16 / 24 / 32 columns: mmvq_w4h, mmvq_w4x alone, quantize + mmvq_w4x (a rotation of distinct matrices: more than the 24 MiB LLC where the pack has them)
void bench_w4x_products(Dflash& d) {
    auto shapes = block_shapes(d);
    std::printf("\nproducts at 8 / 16 / 24 / 32 columns: us (GB/s of the weights) of mmvq_w4h | mmvq_w4x | quantize + mmvq_w4x   (floor = bytes / 590 GB/s)\n%-26s %8s %8s", "shape", "MB", "floor");
    for (int nc : {8, 16, 24, 32}) std::printf(" | %2d cols: %-26s", nc, "w4h | w4x | q+w4x");
    std::printf("\n");
    for (auto& sh : shapes) {
        const double bytes = (double) strata::q36xr::w4_bytes(sh.n_in, sh.n_out);
        std::printf("%-26s %8.1f %6.1fus", sh.name, bytes / 1e6, bytes / 590e3);
        for (int nc : {8, 16, 24, 32}) {
            auto x = rvec((size_t) nc * sh.n_in, 1.f);
            float* dxf = up(x);
            uint16_t* dx16 = dev<uint16_t>(strata::q36xr::w4h_act_halfs(sh.n_in, nc));
            strata::q36xr::w4h_pack(dxf, sh.n_in, dx16, sh.n_in, nc, g_q);
            void* act = dev<uint8_t>(strata::q36xr::w4x_act_bytes(sh.n_in, nc, 128));
            strata::q36xr::w4x_quantize(dxf, sh.n_in, act, sh.n_in, nc, 128, g_q);
            float* y = dev<float>((size_t) nc * sh.n_out);
            g_q->wait();
            const double uh = median_us([&](size_t k) { strata::q36xr::mmvq_w4h(sh.w[k % sh.w.size()], dx16, y, sh.n_in, sh.n_out, nc, g_q); });
            const double ux = median_us([&](size_t k) { strata::q36xr::mmvq_w4x(sh.w[k % sh.w.size()], act, y, sh.n_in, sh.n_out, nc, 128, g_q); });
            const double uq = median_us([&](size_t k) {
                strata::q36xr::w4x_quantize(dxf, sh.n_in, act, sh.n_in, nc, 128, g_q);
                strata::q36xr::mmvq_w4x(sh.w[k % sh.w.size()], act, y, sh.n_in, sh.n_out, nc, 128, g_q);
            });
            char c[96];
            std::snprintf(c, sizeof c, "%.1f(%.0f) | %.1f(%.0f) | %.1f", uh, bytes / uh / 1e3, ux, bytes / ux / 1e3, uq);
            std::printf(" | %-36s", c);
            for (void* p : {(void*) dxf, (void*) dx16, act, (void*) y}) sycl::free(p, *g_q);
        }
        std::printf("\n");
    }
}

// the time of each stage of a block pass, from the host-submitted pass on a profiling queue: a barrier after every stage, the stage's time = the gap between consecutive barrier
// completions (the kernel(s) + the gap before it).  Two things could make this a measurement of the HOST: the pass is ~180 submissions (a kernel and a barrier a stage), and small kernels
// run a few us.  So the GPU gets a head start (a ~0.7 GB memset queued first: ~2 ms) and the sum of the stages is printed against `ref_ms`, the same pass replayed as a graph: a sum
// more than 10% above it is flagged host-bound (the table then over-counts the small stages).  Barrier timestamps that do not rise are flagged too.
void profile_block(Dflash& d, int nb, const char* label, double ref_ms) {
    std::printf("\nblock pass stages, %s (%d block%s, %d columns): median over 20 passes of the stage's time a pass (us), calls a pass, share\n", label, nb, nb > 1 ? "s" : "", 8 * nb);
    try {
        sycl::queue pq(g_q->get_context(), g_q->get_device(), {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
        sycl::queue* saved = g_q;
        g_q = &pq;   // (the hooks of this test submit to g_q: one queue, in order)
        const size_t head_bytes = (size_t) 768 << 20;
        char* head = sycl::malloc_device<char>(head_bytes, pq);
        struct Rec { const char* name; sycl::event ev; };
        std::vector<Rec> recs;
        d.set_stage_hook([&](const char* n) { recs.push_back({n, pq.ext_oneapi_submit_barrier()}); });
        std::map<std::string, std::vector<double>> per;
        std::map<std::string, int> calls;
        std::vector<double> totals;
        std::vector<std::string> order;
        long long bad_ts = 0, n_ts = 0;
        for (int rep = 0; rep < 24; ++rep) {
            recs.clear();
            if (head != nullptr) pq.memset(head, rep & 0xFF, head_bytes);   // (the GPU's head start: the host runs ahead while this executes)
            sycl::event e0 = pq.ext_oneapi_submit_barrier();
            d.block_batch(pq, nb);
            pq.wait();
            if (rep < 4) continue;
            double prev = (double) e0.get_profiling_info<sycl::info::event_profiling::command_end>();
            const double first = prev;
            std::map<std::string, double> sum;
            std::map<std::string, int> cnt;
            for (auto& r : recs) {
                const double t = (double) r.ev.get_profiling_info<sycl::info::event_profiling::command_end>();
                ++n_ts;
                if (t <= prev || t <= 0.0) ++bad_ts;
                sum[r.name] += (t - prev) / 1e3;
                ++cnt[r.name];
                prev = t;
                if (rep == 4 && std::find(order.begin(), order.end(), r.name) == order.end()) order.push_back(r.name);
            }
            totals.push_back((prev - first) / 1e3);
            for (auto& kv : sum) { per[kv.first].push_back(kv.second); calls[kv.first] = cnt[kv.first]; }
        }
        d.set_stage_hook(nullptr);
        g_q = saved;
        if (head != nullptr) sycl::free(head, pq);
        std::sort(totals.begin(), totals.end());
        const double tot = totals[totals.size() / 2];
        double acc = 0;
        for (const auto& nm : order) {
            auto& v = per[nm];
            std::sort(v.begin(), v.end());
            const double med = v[v.size() / 2];
            acc += med;
            std::printf("  %-18s %9.1f us  x%-3d  %5.1f%%   (%.1f us a call)\n", nm.c_str(), med, calls[nm], 100.0 * med / tot, med / calls[nm]);
        }
        std::printf("  %-18s %9.1f us  (the stages add up to %.1f us); the same pass as a graph: %.1f us\n", "pass", tot, acc, ref_ms * 1e3);
        if (bad_ts > 0) std::printf("  WARNING: %lld of %lld barrier timestamps did not rise: this table is not usable\n", bad_ts, n_ts);
        else if (ref_ms > 0 && tot > 1.1 * ref_ms * 1e3) std::printf("  WARNING: the stages add up to %.0f%% of the graph's time: the host-submitted pass is host-bound, the small stages are over-counted\n", 100.0 * tot / (ref_ms * 1e3));
    } catch (const std::exception& e) {
        std::printf("  (profiling queue unavailable: %s)\n", e.what());
    }
}

int whole(bool bench) {
    Bench* b = new Bench;
    std::string err;
    if (!b->d.load_synthetic(*g_q, 99, err, 3)) { std::printf("load_synthetic: %s\n", err.c_str()); return 1; }
    Hooks hk;
    hk.max_nd = b->nd;
    // an embedding that is a function of (the row's token, its row in the block): a block of a batch pass gets what it gets alone
    hk.embed = [&](const int32_t* tok, int n, float* out) {
        g_q->parallel_for(sycl::range<1>((size_t) n * kN), [=](sycl::id<1> id) {
            const size_t r = id[0] / kN, i = id[0] % kN;
            const uint32_t h = (uint32_t) ((uint64_t) ((uint32_t) tok[r] * 8u + (uint32_t) (r & 7)) * 2654435761u + i * 40503u) * 2246822519u;
            out[id[0]] = ((float) ((h >> 8) & 0xFFFF) / 32768.f - 1.f) * 0.5f;
        });
    };
    hk.head = [&](const float* hn, int n, float* lg, int& nd, const int32_t*& idmap) {
        const int ndv = b->nd;
        g_q->parallel_for(sycl::range<1>((size_t) n * ndv), [=](sycl::id<1> i) {
            // block invariant (the hash sees the row's index inside its block: a block of a batch pass gets what it gets alone) and content dependent (every logit
            // takes one element of its own row of the final-normed rows, so the drafter's output moves the candidates)
            const size_t row = i[0] / (size_t) ndv, col = i[0] % (size_t) ndv;
            const uint64_t h = ((uint64_t) (row % kDrafts) * (uint64_t) ndv + col) * 0x9E3779B97F4A7C15ULL;
            lg[i[0]] = ((float) ((h >> 40) & 0xFFFF) / 32768.f - 1.f) * 3.f + hn[row * kN + col % kN] * 0.5f;
        });
        nd = ndv;
        idmap = nullptr;
    };
    b->d.set_hooks(*g_q, hk);
    b->d.enable_tree(*g_q, true);   // the tree gather runs in this block pass (checked below); switched off again before the bench
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
    Debug dbg;
    b->d.debug_last(*g_q, dbg);
    size_t nonfinite = 0;
    double hmax = 0;
    for (float v : dbg.hn) { if (!std::isfinite(v)) ++nonfinite; else hmax = std::max(hmax, (double) std::fabs(v)); }
    std::printf("  whole pass stats: final-normed rows: %.0f non-finite of %.0f, max |h| %.3g; unary[0][0] %.4g, prob[0] %.4g, first picks %d %d %d\n", (double) nonfinite,
                (double) dbg.hn.size(), hmax, (double) dbg.unary[0], (double) pr[0], (int) tok[0], (int) tok[1], (int) tok[2]);
    check("whole pass: block rows finite (random pack with a sane scale)", nonfinite == 0, fmt("%.0f non-finite", (double) nonfinite));
    check("whole pass on a random pack: 7 tokens in range", ok, fmt("n = %.0f context rows, first token %.0f", (double) n, (double) tok[0]));
    test_tree_gather(b->d, tok, pr, dbg);
    b->d.enable_tree(*g_q, false);
    test_compact_stage(b->d);
    test_w4x_products(b->d);
    test_w4x_pass(b->d);
    if (!bench) { batch_vs_singles(b->d); return 0; }
    // the bench passes start from the state of the whole-pass check again (the tests above reused slot 0's staging and the meta)
    for (int k = 0; k < kTaps; ++k) b->d.capture(*g_q, k, dx, T, dpos);
    b->d.invalidate();
    b->d.note_forward(0, T);
    b->d.prepare(*g_q, T, 4242);
    g_q->wait();
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
        return med;
    };
    time_it("block pass (8 rows), no head", blk, [&] { b->d.block(*g_q); });
    time_it("context update, 8 rows (fc + kv_all)", ctx, [&] { b->d.ctx_update(*g_q, 8, false); });
    time_it("both (run(8))", blk + ctx, [&] { b->d.run(*g_q, 8); });
    std::printf("(the three lines above: ~85 / 5 / ~90 kernels submitted from the host one by one)\n");
    // the same passes recorded once and replayed as the engine does (graph_run): the host's submission gone, the kernel boundaries at graph cost
    {
        auto graph_it = [&](const char* name, double bytes, auto&& body) {
            dpct::experimental::command_graph_ptr graph = nullptr;
            dpct::experimental::begin_recording(g_q);
            body();
            dpct::experimental::end_recording(g_q, &graph);
            auto* gx = new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(graph->finalize());
            delete graph;
            return time_it(name, bytes, [&] { g_q->ext_oneapi_graph(*gx); });
        };
        const double g8 = graph_it("GRAPH block pass (8 rows), no head", blk, [&] { b->d.block(*g_q); });
        profile_block(b->d, 1, "8 rows, 100 context rows", g8);
        graph_it("GRAPH context update, 8 rows", ctx, [&] { b->d.ctx_update(*g_q, 8, false); });
        graph_it("GRAPH run(8) = what a draft costs", blk + ctx, [&] { b->d.run(*g_q, 8); });
        // the attention at a full window: 3,000 captured rows, the drafter's ring holds the last 2,047 (the pass above had 100)
        const int T3 = 3000;
        auto x3 = rvec((size_t) T3 * kN, 2.f);
        float* dx3 = up(x3);
        b->d.invalidate();
        for (int k = 0; k < kTaps; ++k) b->d.capture(*g_q, k, dx3, T3, dpos);
        b->d.note_forward(0, T3);
        const int n3 = b->d.prepare(*g_q, T3, 4242);
        b->d.run(*g_q, n3);
        g_q->wait();
        const double g2047 = graph_it("GRAPH block pass, 2,047 context rows", blk, [&] { b->d.block(*g_q); });
        profile_block(b->d, 1, "8 rows, 2,047 context rows", g2047);
        std::printf("(the difference between the last line and 'GRAPH block pass (8 rows)' is the attention at a full window: floor of its K/V reads 42 MB = %.3f ms)\n", 42e6 / 590e6);
        sycl::free(dx3, *g_q);
    }
    // each product shape of the block at 8 columns on random incompressible weights (a rotation of distinct matrices, more than the 24 MiB LLC where the pack has them):
    // the engine's mmvq_w4h, then STRATA_Q36_DFLASH_W4C's decompositions (threads a tile 1 / 2 / 4 / 8 / 16 at 1024 and 2048 threads in all).  Back-to-back launches.
    {
        struct Shape { const char* name; int n_in, n_out; std::vector<const void*> w; };
        std::vector<Shape> shapes = {{"ckp  akp/mkp 5120 -> 1280", kN, kKpOut, {}}, {"q    5120 -> 4096", kN, kHeads * kHd, {}}, {"kv   5120 -> 2048 (a layer's slice)", kN, 2048, {}},
                                     {"o    4096 -> 5120", kHeads * kHd, kN, {}}, {"gate/up 5120 -> 17408", kN, kFF, {}}, {"down 17408 -> 5120", kFF, kN, {}}, {"hproj 5120 -> 256", kN, kRank, {}}};
        for (int l = 0; l < kLayers; ++l) {
            const std::string p = "l" + std::to_string(l) + ".";
            shapes[0].w.push_back(b->d.tensor(p + "akp")); shapes[0].w.push_back(b->d.tensor(p + "mkp"));
            shapes[1].w.push_back(b->d.tensor(p + "q"));
            shapes[2].w.push_back(static_cast<const uint8_t*>(b->d.tensor("kv_all")) + (size_t) l * (2048 / 16) * (kN / 256) * 2176);
            shapes[3].w.push_back(b->d.tensor(p + "o"));
            shapes[4].w.push_back(b->d.tensor(p + "gate")); shapes[4].w.push_back(b->d.tensor(p + "up"));
            shapes[5].w.push_back(b->d.tensor(p + "down"));
        }
        shapes[6].w.push_back(b->d.tensor("hproj"));
        std::printf("\nproducts at 8 columns (us per product; GB/s of its weights; floor at 590 GB/s)\n%-40s %9s %8s", "shape", "MB", "floor");
        const int cfgs[][2] = {{0, 0}, {1, 1024}, {2, 1024}, {4, 1024}, {8, 1024}, {16, 1024}, {4, 2048}, {8, 2048}, {16, 2048}};
        for (auto& c : cfgs) { char h[32]; if (c[0] == 0) std::snprintf(h, sizeof h, "engine"); else std::snprintf(h, sizeof h, "C%d/%d", c[0], c[1]); std::printf(" %14s", h); }
        std::printf("\n");
        for (auto& sh : shapes) {
            const double bytes = (double) strata::q36xr::w4_bytes(sh.n_in, sh.n_out);
            std::vector<uint16_t> xh((size_t) strata::q36xr::w4h_act_halfs(sh.n_in, 8));
            for (auto& v : xh) v = h2u(rnd());
            uint16_t* dx = up(xh);
            float* dy = dev<float>((size_t) sh.n_out * 8);
            std::printf("%-40s %9.1f %7.1fus", sh.name, bytes / 1e6, bytes / 590e3);
            for (auto& c : cfgs) {
                auto run1 = [&](size_t k) {
                    const void* w = sh.w[k % sh.w.size()];
                    if (c[0] == 0) return strata::q36xr::mmvq_w4h(w, dx, dy, sh.n_in, sh.n_out, 8, g_q);
                    return strata::q36xr::mmvq_w4h_8(w, dx, dy, sh.n_in, sh.n_out, g_q, c[0], c[1]);
                };
                if (!run1(0)) { std::printf(" %14s", "declined"); continue; }
                g_q->wait();
                const int per = 40;
                const auto t0 = std::chrono::steady_clock::now();
                while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.2) { for (int i = 0; i < per; ++i) run1((size_t) i); g_q->wait(); }
                std::vector<double> us;
                for (int rep = 0; rep < 9; ++rep) {
                    const auto a = std::chrono::steady_clock::now();
                    for (int i = 0; i < per; ++i) run1((size_t) i);
                    g_q->wait();
                    us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count() / per);
                }
                std::sort(us.begin(), us.end());
                char cell[48];
                std::snprintf(cell, sizeof cell, "%.1f/%.0f", us[us.size() / 2], bytes / us[us.size() / 2] / 1e3);
                std::printf(" %14s", cell);
            }
            std::printf("\n");
            sycl::free(dx, *g_q); sycl::free(dy, *g_q);
        }
        std::printf("(cell: us / GB/s.  The pass uses the best C of each class through STRATA_Q36_DFLASH_W4C=ckp,q,kv,o,gate_up,down,hproj and STRATA_Q36_DFLASH_W4T; unset = the engine's)\n");
    }
    batch_vs_singles(b->d);
    bench_w4x_products(b->d);
    // the batch pass: the block's weights read once for 8 nb columns (graph replay; the head's hook here is the synthetic one)
    double t4 = 0;
    for (int nb = 1; nb <= kMaxBlocks; ++nb) {
        char nm[64];
        std::snprintf(nm, sizeof nm, "GRAPH block pass, %d blocks (%d columns)", nb, 8 * nb);
        dpct::experimental::command_graph_ptr graph = nullptr;
        dpct::experimental::begin_recording(g_q);
        b->d.block_batch(*g_q, nb);
        dpct::experimental::end_recording(g_q, &graph);
        auto* gx = new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(graph->finalize());
        delete graph;
        t4 = time_it(nm, blk, [&] { g_q->ext_oneapi_graph(*gx); });
    }
    profile_block(b->d, 4, "w4h products", t4);
    // the same with the int8 route (STRATA_Q36_DFLASH_W4X): graph replay of the batch passes, then the stage profile
    double t4x = 0;
    if (b->d.set_w4x(*g_q, true, 16)) {
        for (int nb = 2; nb <= kMaxBlocks; ++nb) {
            char nm[64];
            std::snprintf(nm, sizeof nm, "GRAPH block pass, %d blocks, W4X", nb);
            dpct::experimental::command_graph_ptr graph = nullptr;
            dpct::experimental::begin_recording(g_q);
            b->d.block_batch(*g_q, nb);
            dpct::experimental::end_recording(g_q, &graph);
            auto* gx = new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(graph->finalize());
            delete graph;
            t4x = time_it(nm, blk, [&] { g_q->ext_oneapi_graph(*gx); });
        }
        profile_block(b->d, 4, "W4X products", t4x);
        b->d.set_w4x(*g_q, false);
    }
    return 0;
}

}   // namespace

int parity_main(int argc, char** argv) {
    bool bench = false;
    for (int i = 0; i < argc; ++i) if (std::string(argv[i]) == "--bench") bench = true;
    g_q = &dpct::get_in_order_queue();
    strata::q36xr::init(g_q);
    test_half_convert();
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
