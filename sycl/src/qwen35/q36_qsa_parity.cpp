// q36_qsa_parity.cpp - parity and bench of the QSA kernels (q36_qsa.dp.cpp), run from the existing parity binary:
//
//     q36_mmvq_dense_parity --qsa [--bench [ctx,ctx,...]]
//
// Every kernel against a float64 (or exact) host reference on random data; the bench times the decode-step kernels against the byte floor
// (590 GB/s) at 25,120 / 118,000 / 255,000 cells.  Contract per kernel (the bound is what the check enforces):
//   project + prep   queries / raw keys: |d| <= 6e-5 * max|ref| (fp32 sums of 5,120 fp16-weight products, rsqrt, sin/cos in fp32); a row of a
//                    T-row call BITWISE equal to the row of a 1-row call (T = 1, 16, 17, 40)
//   pool             pooled fp16 key: |d| <= 2^-10 |ref| + 5e-4 (the fp16 store of a fp32 value); the ring: bitwise the raw keys
//   score            |d| <= 4e-6 * sum|q k| + 1e-6 (fp32 chain of 128 FMAs per head); bitwise equal across 1 / 2 / 4 / 8 rows a launch
//   select           EXACT: the same 512 block ids as a host sort by (score desc, index asc), ascending; ties and -0.0 included;
//                    rows with at most 512 visible blocks: all of them
//   attend           |d| <= 3e-4 * max|ref| + 1e-6 against a float64 softmax over the same cells (native exp, fp32 sums); a row of a T-row window
//                    BITWISE equal to the same row run alone (T = 16, tree window of 11)
//   pool_tree        the ring bitwise, the pooled blocks as pool
// Data: random bytes / N(0, 1) from a fixed generator.  The bench uses 16 layer copies of the block keys and a new selection every layer so the
// 24 MiB cache never holds the step's bytes.  TIMING COLUMNS ARE FILLED BY THE CARD: nothing here has run yet.
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "q36_qsa.hpp"

namespace strata::q36qsa {
namespace {

std::mt19937_64 g_rng(777);
float rnd() {
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    return (u(g_rng) + u(g_rng) + u(g_rng) + u(g_rng)) * 0.866f;
}
uint32_t rbits() { return (uint32_t) g_rng(); }
uint16_t h2u(float f) {   // float -> half, nearest even, every range (the host's sycl::half truncates subnormals)
    uint32_t b; std::memcpy(&b, &f, 4);
    const uint16_t sign = (uint16_t) ((b >> 16) & 0x8000u);
    if (std::isnan(f)) return (uint16_t) (sign | 0x7E00u);
    if (std::isinf(f)) return (uint16_t) (sign | 0x7C00u);
    const double a = std::fabs((double) f);
    if (a < 6.103515625e-05) return (uint16_t) (sign | (uint16_t) std::nearbyint(std::ldexp(a, 24)));
    int e = std::ilogb(a);
    double q = std::nearbyint(std::ldexp(a, 10 - e));
    if (q >= 2048.0) { q = 1024.0; ++e; }
    if (e > 15) return (uint16_t) (sign | 0x7C00u);
    return (uint16_t) (sign | (uint16_t) (((e + 15) << 10) + ((int) q - 1024)));
}
float u2h(uint16_t u) {
    const int e = (u >> 10) & 31, m = u & 1023;
    const double v = e == 0 ? std::ldexp((double) m, -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp(1024.0 + m, e - 25);
    return (float) ((u & 0x8000) ? -v : v);
}

sycl::queue* g_q = nullptr;
template <class T> T* dev(size_t n) { return sycl::aligned_alloc_device<T>(4096, n ? n : 1, *g_q); }
template <class T> T* up(const std::vector<T>& v) { T* d = dev<T>(v.size()); g_q->memcpy(d, v.data(), v.size() * sizeof(T)).wait(); return d; }
template <class T> std::vector<T> down(const T* d, size_t n) { std::vector<T> v(n); g_q->memcpy(v.data(), d, n * sizeof(T)).wait(); return v; }
int32_t* dscalar(int32_t v) { int32_t* d = dev<int32_t>(1); g_q->memcpy(d, &v, 4).wait(); return d; }

int g_bad = 0;
void check(const std::string& name, bool ok, const std::string& info = "") {
    std::printf("%-52s %s %s\n", name.c_str(), ok ? "ok  " : "FAIL", info.c_str());
    if (!ok) ++g_bad;
}
std::string fmt(const char* f, double a = 0, double b = 0, double c = 0) { char s[200]; std::snprintf(s, sizeof s, f, a, b, c); return s; }

// ---- host references ---------------------------------------------------------------------------------------------------------------
void rope64(std::vector<double>& v, double pos, double base, int n_rot) {   // NEOX over the first n_rot of the vector's dims
    std::vector<double> o = v;
    const int hlf = n_rot / 2;
    for (int i = 0; i < n_rot; ++i) {
        const int p = i < hlf ? i : i - hlf;
        const double th = pos * std::pow(base, -2.0 * p / n_rot);
        o[i] = i < hlf ? v[i] * std::cos(th) - v[i + hlf] * std::sin(th) : v[i - hlf] * std::sin(th) + v[i] * std::cos(th);
    }
    v = o;
}
std::vector<double> rms_mul(const std::vector<double>& x, const float* w, double eps) {
    double s = 0;
    for (double v : x) s += v * v;
    const double r = 1.0 / std::sqrt(s / x.size() + eps);
    std::vector<double> o(x.size());
    for (size_t i = 0; i < x.size(); ++i) o[i] = x[i] * r * w[i];
    return o;
}

struct Weights16 {   // host copies (the fp16-rounded values) of one layer's weights
    std::vector<uint16_t> wq, wk;
    std::vector<float> qn, kp, kn;
};
Weights16 make_weights() {
    Weights16 w;
    w.wq.resize((size_t) kHeads * kD * kEmbd);
    w.wk.resize((size_t) kD * kEmbd);
    for (auto& x : w.wq) x = h2u(rnd() * 0.02f);
    for (auto& x : w.wk) x = h2u(rnd() * 0.02f);
    w.qn.resize(kD); w.kp.resize(kD); w.kn.resize(kD);
    for (int i = 0; i < kD; ++i) { w.qn[i] = 0.5f + 0.5f * std::fabs(rnd()); w.kp[i] = 0.5f + 0.5f * std::fabs(rnd()); w.kn[i] = 0.5f + 0.5f * std::fabs(rnd()); }
    return w;
}
LayerWeights upload(const Weights16& w) {
    LayerWeights l;
    l.wq = up(w.wq); l.wk = up(w.wk); l.qn = up(w.qn); l.kpre = up(w.kp); l.knorm = up(w.kn);
    return l;
}

// ---- project + prep + pool ---------------------------------------------------------------------------------------------------------
void test_index(const Weights16& w, const LayerWeights& lw) {
    const int ctx = 8192, tmax = 64;
    Scratch sc;
    LayerState ls;
    alloc_scratch(sc, tmax, kMaxWin, ctx, g_q);
    alloc_state(ls, ctx, g_q);
    Rope rp;
    const int T = 40, p0 = 1001;   // 40 rows from cell 1001: blocks 250 (cells 1000..1003: the first member is before the window) .. 260 complete
    std::vector<float> h((size_t) T * kEmbd);
    for (auto& x : h) x = rnd();
    float* dh = up(h);
    // the ring before the window: the 1 cell (1000) of block 250 that precedes it
    std::vector<float> ring0((size_t) kRing * kD);
    for (auto& x : ring0) x = rnd();
    g_q->memcpy(ls.ring, ring0.data(), ring0.size() * 4).wait();
    Win win;
    win.pos0 = dscalar(p0);
    win.T = T; win.nr = T;
    index_rows(lw, ls, sc, dh, win, rp, false, g_q);
    g_q->wait();
    const auto qidx = down(sc.qidx, (size_t) T * kHeads * kD), raw = down(sc.raw, (size_t) T * kD);
    const auto ringd = down(ls.ring, (size_t) kRing * kD);
    const auto pooled = down(static_cast<const uint16_t*>(ls.pooled), (size_t) nb_cap_for(ctx) * kD);
    double eq = 0, ek = 0, ep = 0, er = 0, mq = 1e-9, mk = 1e-9;
    std::vector<std::vector<double>> rawref(T);
    for (int t = 0; t < T; ++t) {
        std::vector<double> pr(kProj, 0.0);
        for (int o = 0; o < kProj; ++o) {
            const uint16_t* wr = o < kHeads * kD ? &w.wq[(size_t) o * kEmbd] : &w.wk[(size_t) (o - kHeads * kD) * kEmbd];
            double a = 0;
            for (int k = 0; k < kEmbd; ++k) a += (double) u2h(wr[k]) * (double) h[(size_t) t * kEmbd + k];
            pr[o] = a;
        }
        for (int hh = 0; hh < kHeads; ++hh) {
            std::vector<double> v(pr.begin() + hh * kD, pr.begin() + (hh + 1) * kD);
            v = rms_mul(v, w.qn.data(), rp.eps);
            rope64(v, (double) (p0 + t), rp.base, rp.n_rot);
            for (int i = 0; i < kD; ++i) { mq = std::max(mq, std::fabs(v[i])); eq = std::max(eq, std::fabs(v[i] - qidx[((size_t) t * kHeads + hh) * kD + i])); }
        }
        std::vector<double> k(pr.begin() + kHeads * kD, pr.end());
        k = rms_mul(k, w.kp.data(), rp.eps);
        rawref[t] = k;
        for (int i = 0; i < kD; ++i) { mk = std::max(mk, std::fabs(k[i])); ek = std::max(ek, std::fabs(k[i] - raw[(size_t) t * kD + i])); }
        if (t >= T - 4) for (int i = 0; i < kD; ++i) er = std::max(er, std::fabs((double) ringd[(size_t) ((p0 + t) % kRing) * kD + i] - raw[(size_t) t * kD + i]));
    }
    check("project+prep: queries vs float64", eq <= 6e-5 * mq, fmt("max |d| %.2e (max |q| %.2f)", eq, mq));
    check("project+prep: raw keys vs float64", ek <= 6e-5 * mk, fmt("max |d| %.2e", ek));
    check("project+prep: ring holds the last 4 rows (T > 32)", er == 0.0, fmt("max |d| %.1e", er));
    // pooled blocks: complete blocks whose last cell lies in [p0, p0 + T): b = 250 .. (p0 + T - 4) / 4
    int nbad = 0, nblk = 0;
    for (int b = p0 / kRatio; b * kRatio + 3 <= p0 + T - 1; ++b) {
        std::vector<double> m(kD, 0.0);
        for (int i = 0; i < kD; ++i) {
            double a[4];
            for (int j = 0; j < 4; ++j) {
                const int c = b * kRatio + j;
                a[j] = c >= p0 ? (double) raw[(size_t) (c - p0) * kD + i] : (double) ring0[(size_t) (c % kRing) * kD + i];
            }
            m[i] = (double) (float) ((((float) a[0] + (float) a[1]) + (float) a[2]) + (float) a[3]) * 0.25;
        }
        m = rms_mul(m, w.kn.data(), rp.eps);
        rope64(m, (double) (b * kRatio), rp.base, rp.n_rot);
        for (int i = 0; i < kD; ++i) {
            const double got = u2h(pooled[(size_t) b * kD + i]);
            ep = std::max(ep, std::fabs(got - m[i]));
            if (std::fabs(got - m[i]) > std::fabs(m[i]) / 1024.0 + 5e-4) ++nbad;
        }
        ++nblk;
    }
    check("pool: " + std::to_string(nblk) + " blocks vs float64", nbad == 0, fmt("max |d| %.2e", ep));
    // a row of a T-row call is bitwise the row of a 1-row call
    for (int Tn : {1, 16, 17}) {
        Scratch s2;
        LayerState l2;
        alloc_scratch(s2, tmax, kMaxWin, ctx, g_q);
        alloc_state(l2, ctx, g_q);
        g_q->memcpy(l2.ring, ring0.data(), ring0.size() * 4).wait();
        Win w2 = win;
        w2.T = Tn; w2.nr = Tn;
        index_rows(lw, l2, s2, dh, w2, rp, false, g_q);
        g_q->wait();
        const auto q2 = down(s2.qidx, (size_t) Tn * kHeads * kD), r2 = down(s2.raw, (size_t) Tn * kD);
        const bool same = std::memcmp(q2.data(), qidx.data(), q2.size() * 4) == 0 && std::memcmp(r2.data(), raw.data(), r2.size() * 4) == 0;
        check(fmt("project+prep: %.0f rows bitwise the first rows of the 40-row call", Tn), same);
        free_scratch(s2, g_q); free_state(l2, g_q);
    }
    {   // the key-only call (a dense prompt chunk): raw keys and pooled blocks bitwise those of the full call
        Scratch s3;
        LayerState l3;
        alloc_scratch(s3, tmax, kMaxWin, ctx, g_q);
        alloc_state(l3, ctx, g_q);
        g_q->memcpy(l3.ring, ring0.data(), ring0.size() * 4).wait();
        index_rows(lw, l3, s3, dh, win, rp, false, g_q, false);
        g_q->wait();
        const auto r3 = down(s3.raw, (size_t) T * kD);
        const auto p3 = down(static_cast<const uint16_t*>(l3.pooled), (size_t) (p0 / kRatio + T / kRatio + 1) * kD);
        const bool same = std::memcmp(r3.data(), raw.data(), r3.size() * 4) == 0 &&
                          std::memcmp(p3.data() + (size_t) (p0 / kRatio) * kD, pooled.data() + (size_t) (p0 / kRatio) * kD, (size_t) (T / kRatio) * kD * 2) == 0;
        check("project+prep: key-only call (with_q = false) bitwise the keys and blocks of the full call", same);
        free_scratch(s3, g_q); free_state(l3, g_q);
    }
    free_scratch(sc, g_q); free_state(ls, g_q);
}

// ---- select diagnostics: what differs, in words ---------------------------------------------------------------------------------------
uint32_t skey_h(float v) { uint32_t b; std::memcpy(&b, &v, 4); return b & 0x7FFFFFFFu; }
// The host's own selection of the first c scores s: ids ascending, the threshold key, how many are above it and how many equal to it.
struct HostSel { std::vector<int> ids; uint32_t thr = 0; int gt = 0, eq = 0; };
HostSel host_select(const std::vector<float>& s, int c) {
    HostSel h;
    std::vector<int> idx(c);
    for (int b = 0; b < c; ++b) idx[b] = b;
    if (c <= kTopBlocks) { h.ids = idx; return h; }
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return skey_h(s[a]) > skey_h(s[b]); });
    h.thr = skey_h(s[idx[kTopBlocks - 1]]);
    for (int b = 0; b < c; ++b) { const uint32_t k = skey_h(s[b]); h.gt += k > h.thr; h.eq += k == h.thr; }
    idx.resize(kTopBlocks);
    std::sort(idx.begin(), idx.end());
    h.ids = idx;
    return h;
}
// Compares a row's ids / count with the host's; the text names the first difference (empty when equal).  got_scores: the scores the card
// wrote (first c), dbg: the kernel's 16 ints of the row.
std::string select_diff(const std::vector<int>& got, int ns, const HostSel& h, const std::vector<float>& s, const std::vector<float>& got_scores,
                        const std::vector<int>& dbg, int c) {
    std::string out;
    char b[300];
    const int want = (int) h.ids.size();
    // 1. the scores the card wrote against the host's (the select's input)
    int sbad = 0, sfirst = -1;
    for (int i = 0; i < c && i < (int) got_scores.size(); ++i)
        if (std::memcmp(&got_scores[i], &s[i], 4) != 0) { if (!sbad) sfirst = i; ++sbad; }
    if (sbad) { std::snprintf(b, sizeof b, "[scores differ at %d blocks, first b=%d got %.9g want %.9g] ", sbad, sfirst, got_scores[sfirst], s[sfirst]); out += b; }
    // 2. the ids
    std::vector<char> in_got(c > 0 ? c : 1, 0);
    int oob = 0, dup = 0, unsorted = 0;
    for (int i = 0; i < want && i < (int) got.size(); ++i) {
        const int x = got[i];
        if (x < 0 || x >= c) { ++oob; continue; }
        if (in_got[x]) ++dup;
        in_got[x] = 1;
        if (i && got[i] <= got[i - 1]) ++unsorted;
    }
    std::vector<char> in_want(c > 0 ? c : 1, 0);
    for (int x : h.ids) in_want[x] = 1;
    int missing = 0, extra = 0, fm = -1, fe = -1;
    for (int x = 0; x < c; ++x) {
        if (in_want[x] && !in_got[x]) { if (!missing) fm = x; ++missing; }
        if (!in_want[x] && in_got[x]) { if (!extra) fe = x; ++extra; }
    }
    if (ns != want) { std::snprintf(b, sizeof b, "[count got %d want %d] ", ns, want); out += b; }
    if (oob || dup || unsorted || missing || extra) {
        std::snprintf(b, sizeof b, "[ids: %d out of range / unwritten, %d duplicates, %d not ascending, %d missing, %d extra] ", oob, dup, unsorted, missing, extra);
        out += b;
        if (fm >= 0) { std::snprintf(b, sizeof b, "[first missing b=%d key %08x] ", fm, skey_h(s[fm])); out += b; }
        if (fe >= 0) { std::snprintf(b, sizeof b, "[first extra b=%d key %08x] ", fe, skey_h(s[fe])); out += b; }
        std::snprintf(b, sizeof b, "[got[0..3] %d %d %d %d, want[0..3] %d %d %d %d] ", got[0], got[1], got[2], got[3], h.ids[0], h.ids[1], h.ids[2], h.ids[3]);
        out += b;
    }
    if (!out.empty() && c > kTopBlocks) {
        // 3. the kernel's decisions against the host's
        std::snprintf(b, sizeof b, "[thr key host %08x gt %d eq %d | card %08x rem %d gt %d eq %d (host rem %d)] ", h.thr, h.gt, h.eq,
                      (unsigned) dbg[0], dbg[1], dbg[2], dbg[3], kTopBlocks - h.gt);
        out += b;
        std::snprintf(b, sizeof b, "[card candidates per pass %d %d %d %d, bins %02x %02x %02x %02x, c %d]", dbg[4], dbg[5], dbg[6], dbg[7],
                      dbg[9], dbg[10], dbg[11], dbg[12], dbg[8]);
        out += b;
        // the scores around the threshold (host order)
        std::vector<uint32_t> ks;
        for (int i = 0; i < c; ++i) ks.push_back(skey_h(s[i]));
        std::sort(ks.begin(), ks.end(), std::greater<uint32_t>());
        std::snprintf(b, sizeof b, " [keys of the 509th..515th best: %08x %08x %08x %08x %08x %08x %08x]", ks[508], ks[509], ks[510], ks[511],
                      ks[512 < c ? 512 : c - 1], ks[513 < c ? 513 : c - 1], ks[514 < c ? 514 : c - 1]);
        out += b;
    }
    return out;
}

// ---- score ---------------------------------------------------------------------------------------------------------------------------
void test_score(const Weights16& w) {
    (void) w;
    const int ctx = 40000, nbc = nb_cap_for(ctx);
    LayerState ls;
    alloc_state(ls, ctx, g_q);
    Scratch sc;
    alloc_scratch(sc, 16, kMaxWin, ctx, g_q);
    std::vector<uint16_t> pk((size_t) nbc * kD);
    for (auto& x : pk) x = h2u(rnd());
    g_q->memcpy(ls.pooled, pk.data(), pk.size() * 2).wait();
    std::vector<float> qi((size_t) 16 * kHeads * kD);
    for (auto& x : qi) x = rnd();
    g_q->memcpy(sc.qidx, qi.data(), qi.size() * 4).wait();
    const int P = 30000;   // row t is cell P + t: visible blocks (P + t + 1) / 4 = 7500
    Win win;
    win.pos0 = dscalar(P);
    win.T = 16; win.nr = 16;
    select_rows(ls, sc, win, g_q);
    g_q->wait();
    const auto score = down(sc.score, (size_t) 16 * nbc);
    double worst = 0;
    bool all_ok = true;
    for (int t = 0; t < 16; ++t) {
        const int c = (P + t + 1) / 4;
        const int stride_blocks = 61;   // every 61st block: the whole range is covered over rows without an hour of float64
        for (int b = (t * 7) % stride_blocks; b < c; b += stride_blocks) {
            double sum = 0, mag = 0;
            for (int hh = 0; hh < kHeads; ++hh) {
                double a = 0, m = 0;
                for (int d = 0; d < kD; ++d) {
                    const double pr = (double) qi[((size_t) t * kHeads + hh) * kD + d] * (double) u2h(pk[(size_t) b * kD + d]);
                    a += pr; m += std::fabs(pr);
                }
                sum += std::max(a, 0.0);
                mag += m;
            }
            sum *= (double) kInvSqrtD;
            mag *= (double) kInvSqrtD;
            const double d = std::fabs(sum - (double) score[(size_t) t * nbc + b]);
            worst = std::max(worst, d / (4e-6 * mag + 1e-6));
            if (d > 4e-6 * mag + 1e-6) all_ok = false;
        }
    }
    check("score: 16 rows vs float64 (sampled blocks)", all_ok, fmt("worst error / bound %.3f", worst));
    // the 16 rows' scores and ids are the same whatever the rows of a launch (1, 2, 4, 8): select_rows launches 8 at a time; a 3-row call is a 4-row kernel
    const auto sel16 = down(sc.sel, (size_t) 16 * kTopBlocks);
    bool same = true;
    std::string first_call_diff;
    for (int nr : {1, 3, 5}) {
        for (int r0 : {0, 6, 13}) {
            if (r0 + nr > 16) continue;
            Win w1 = win;
            w1.row0 = r0; w1.nr = nr;
            Scratch s1;
            alloc_scratch(s1, 16, kMaxWin, ctx, g_q);
            g_q->memcpy(s1.qidx, qi.data(), qi.size() * 4).wait();
            select_rows(ls, s1, w1, g_q);
            g_q->wait();
            const auto sc1 = down(s1.score, (size_t) nr * nbc);
            const auto id1 = down(s1.sel, (size_t) nr * kTopBlocks);
            for (int r = 0; r < nr; ++r) {
                const int c = (P + r0 + r + 1) / 4;
                if (std::memcmp(&sc1[(size_t) r * nbc], &score[(size_t) (r0 + r) * nbc], (size_t) c * 4) != 0) {
                    same = false;
                    if (first_call_diff.empty()) {
                        int fb = 0;
                        while (fb < c && std::memcmp(&sc1[(size_t) r * nbc + fb], &score[(size_t) (r0 + r) * nbc + fb], 4) == 0) ++fb;
                        first_call_diff = fmt("call nr %.0f from row %.0f, its row %.0f: SCORES differ", nr, r0, r) +
                                          fmt(" at block %.0f: got %.9g, 16-row call %.9g", fb, sc1[(size_t) r * nbc + fb], score[(size_t) (r0 + r) * nbc + fb]);
                    }
                }
                if (std::memcmp(&id1[(size_t) r * kTopBlocks], &sel16[(size_t) (r0 + r) * kTopBlocks], kTopBlocks * 4) != 0) {
                    same = false;
                    if (first_call_diff.empty()) {
                        int fi = 0;
                        while (fi < kTopBlocks && id1[(size_t) r * kTopBlocks + fi] == sel16[(size_t) (r0 + r) * kTopBlocks + fi]) ++fi;
                        first_call_diff = fmt("call nr %.0f from row %.0f, its row %.0f: IDS differ (scores equal)", nr, r0, r) +
                                          fmt(" at slot %.0f: got %.0f, 16-row call %.0f", fi, id1[(size_t) r * kTopBlocks + fi], sel16[(size_t) (r0 + r) * kTopBlocks + fi]);
                    }
                }
            }
            free_scratch(s1, g_q);
        }
    }
    // the 16-row call's own ids against the host's selection of the scores the card wrote (is the select wrong, or only different between calls?)
    std::string sdiff;
    {
        const auto dbg16 = down(sc.dbg, 16 * 16);
        for (int r = 0; r < 16 && sdiff.empty(); ++r) {
            const int c = (P + r + 1) / 4;
            const std::vector<float> rs(score.begin() + (size_t) r * nbc, score.begin() + (size_t) r * nbc + c);
            const std::vector<int> row(sel16.begin() + (size_t) r * kTopBlocks, sel16.begin() + (size_t) (r + 1) * kTopBlocks);
            const std::vector<int> rd(dbg16.begin() + r * 16, dbg16.begin() + (r + 1) * 16);
            const std::string d = select_diff(row, kTopBlocks, host_select(rs, c), rs, rs, rd, c);
            if (!d.empty()) sdiff = fmt("row %.0f of the 16-row call: ", r) + d;
        }
    }
    check("score+select: 16 rows' ids = the host's top 512 of the card's scores", sdiff.empty(), sdiff);
    check("score+select: rows bitwise the same in calls of 1, 3, 5", same, first_call_diff);
    free_scratch(sc, g_q); free_state(ls, g_q);
}

// ---- the OLD select kernels (parity only, no part of the engine): the same 21 cases through each, scores copied in (the scoring kernel is out of
// the loop), one info line per variant.  Read: 916106b3 ok + 05573cfe FAIL -> the sub-group aggregation; both FAIL and the new kernel ok -> the
// library scan / generic-space atomics / shuffles; all three FAIL alike -> shared code (harness, row_vis_blocks, the 1,024-wide work-group).
struct TagOldA; struct TagOldB;
inline uint32_t skey_d(float v) { return sycl::bit_cast<uint32_t>(v) & 0x7FFFFFFFu; }
template <class Tag>
void old_select(sycl::queue& q, const float* score, int c_in, int32_t* sel, int32_t* nsel) {
    q.parallel_for<Tag>(sycl::nd_range<1>((size_t) 1024, 1024), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto grp = it.get_group();
        auto& hist = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[256]>(grp);
        auto& sh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[2]>(grp);
        const int t = (int) it.get_group(0);
        const int tid = (int) it.get_local_id(0);
        const int c = c_in;
        const float* sc = score;
        int32_t* out = sel + (size_t) t * kTopBlocks;
        if (c <= kTopBlocks) {
            for (int i = tid; i < c; i += 1024) out[i] = i;
            if (tid == 0) nsel[t] = c;
            return;
        }
        uint32_t prefix = 0;
        int rem = kTopBlocks;
        for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - 8 * pass;
            if (tid < 256) hist[tid] = 0;
            sycl::group_barrier(grp);
            for (int i = tid; i < c; i += 1024) {
                const uint32_t u = skey_d(sc[i]);
                if (pass == 0 || (u >> (shift + 8)) == prefix) {
                    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::generic_space>
                        a(hist[(u >> shift) & 255u]);
                    a.fetch_add(1);
                }
            }
            sycl::group_barrier(grp);
            if (tid == 0) {
                int cum = 0, bin = 255;
                for (; bin > 0; --bin) {
                    if (cum + hist[bin] >= rem) break;
                    cum += hist[bin];
                }
                sh[0] = bin; sh[1] = cum;
            }
            sycl::group_barrier(grp);
            prefix = (prefix << 8) | (uint32_t) sh[0];
            rem -= sh[1];
            sycl::group_barrier(grp);
        }
        // prefix is now the 512th best score's bits; rem of the scores equal to it are taken, the lowest block indices first
        const int S = (c + 1023) / 1024;
        const int lo = tid * S, hi = sycl::min(c, lo + S);
        int gt = 0, eq = 0;
        for (int i = lo; i < hi; ++i) {
            const uint32_t u = skey_d(sc[i]);
            gt += u > prefix;
            eq += u == prefix;
        }
        const int eq_before = sycl::exclusive_scan_over_group(grp, eq, sycl::plus<int>());
        const int take_eq = sycl::min(sycl::max(rem - eq_before, 0), eq);
        const int off = sycl::exclusive_scan_over_group(grp, gt + take_eq, sycl::plus<int>());
        int o = off, seen = 0;
        for (int i = lo; i < hi; ++i) {
            const uint32_t u = skey_d(sc[i]);
            if (u > prefix) {
                out[o++] = i;
            } else if (u == prefix) {
                if (seen < take_eq) out[o++] = i;
                ++seen;
            }
        }
        if (tid == 0) nsel[t] = kTopBlocks;
    });
}

template <class Tag>
void old_select_agg(sycl::queue& q, const float* score, int c_in, int32_t* sel, int32_t* nsel) {
    q.parallel_for<Tag>(sycl::nd_range<1>((size_t) 1024, 1024), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto grp = it.get_group();
        auto& hist = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[256]>(grp);
        auto& sh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[2]>(grp);
        auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_id()[0];
        const int t = (int) it.get_group(0);
        const int tid = (int) it.get_local_id(0);
        const int c = c_in;
        const float* sc = score;
        int32_t* out = sel + (size_t) t * kTopBlocks;
        if (c <= kTopBlocks) {
            for (int i = tid; i < c; i += 1024) out[i] = i;
            if (tid == 0) nsel[t] = c;
            return;
        }
        uint32_t prefix = 0;
        int rem = kTopBlocks;
        for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - 8 * pass;
            if (tid < 256) hist[tid] = 0;
            sycl::group_barrier(grp);
            // The first two digits are low in entropy (the scores of a row share their exponent: nearly all of the 64K elements fall in a
            // few bins), so there a sub-group first counts the lanes of equal bin and the lowest of them adds the count once; passes 2 and 3
            // see only the elements that match the prefix.  (Every lane takes the same trips: the shuffles are convergent.)
            for (int base = 0; base < c; base += 1024) {
                const int i = base + tid;
                int bin = -1;
                if (i < c) {
                    const uint32_t u = skey_d(sc[i]);
                    if (pass == 0 || (u >> (shift + 8)) == prefix) bin = (int) ((u >> shift) & 255u);
                }
                if (pass < 2) {
                    int cnt = 0, first = 16;
#pragma unroll
                    for (int k = 0; k < 16; ++k) {
                        const int ob = sycl::select_from_group(sg, bin, sycl::id<1>(k));
                        if (ob == bin) { ++cnt; if (first == 16) first = k; }
                    }
                    if (bin >= 0 && first == lane) {
                        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::generic_space> a(hist[bin]);
                        a.fetch_add(cnt);
                    }
                } else if (bin >= 0) {
                    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::generic_space> a(hist[bin]);
                    a.fetch_add(1);
                }
            }
            sycl::group_barrier(grp);
            if (tid == 0) {
                int cum = 0, bin = 255;
                for (; bin > 0; --bin) {
                    if (cum + hist[bin] >= rem) break;
                    cum += hist[bin];
                }
                sh[0] = bin; sh[1] = cum;
            }
            sycl::group_barrier(grp);
            prefix = (prefix << 8) | (uint32_t) sh[0];
            rem -= sh[1];
            sycl::group_barrier(grp);
        }
        // prefix is now the 512th best score's bits; rem of the scores equal to it are taken, the lowest block indices first
        const int S = (c + 1023) / 1024;
        const int lo = tid * S, hi = sycl::min(c, lo + S);
        int gt = 0, eq = 0;
        for (int i = lo; i < hi; ++i) {
            const uint32_t u = skey_d(sc[i]);
            gt += u > prefix;
            eq += u == prefix;
        }
        const int eq_before = sycl::exclusive_scan_over_group(grp, eq, sycl::plus<int>());
        const int take_eq = sycl::min(sycl::max(rem - eq_before, 0), eq);
        const int off = sycl::exclusive_scan_over_group(grp, gt + take_eq, sycl::plus<int>());
        int o = off, seen = 0;
        for (int i = lo; i < hi; ++i) {
            const uint32_t u = skey_d(sc[i]);
            if (u > prefix) {
                out[o++] = i;
            } else if (u == prefix) {
                if (seen < take_eq) out[o++] = i;
                ++seen;
            }
        }
        if (tid == 0) nsel[t] = kTopBlocks;
    });
}

// ---- select: exact, on scores that are given ---------------------------------------------------------------------------------------
void test_select() {
    const int ctx = 262144, nbc = nb_cap_for(ctx);
    LayerState ls;
    ls.pooled = dev<uint16_t>(16);   // (select_rows would score first: the check calls the select through scores written by hand, see below)
    Scratch sc;
    alloc_scratch(sc, 16, kMaxWin, ctx, g_q);
    // select_rows always scores first; to give the selection its own scores the pooled keys and queries are made so that the score of block b is
    // known: q row = e_0 repeated over the 4 heads' first dim 1.0, key b = (v_b, 0, ...) so the score of block b is 4 * relu(v_b) / sqrt(128) when v_b > 0
    // (the scores are then computable exactly: v_b in fp16, times 1.0, summed 4 heads, times 1/sqrt(128) in fp32).
    std::vector<uint16_t> pk((size_t) nbc * kD, 0);
    ls.pooled = dev<uint16_t>(pk.size());
    std::vector<float> qi((size_t) 16 * kHeads * kD, 0.f);
    for (int t = 0; t < 16; ++t) for (int hh = 0; hh < kHeads; ++hh) qi[((size_t) t * kHeads + hh) * kD] = 1.0f;
    g_q->memcpy(sc.qidx, qi.data(), qi.size() * 4).wait();
    int total = 0, bad = 0;
    float* dsc = dev<float>(nbc);
    int var_bad[2] = {0, 0};
    std::string var_first[2];
    for (int c : {100, 512, 513, 700, 5000, 65535, 65536}) {
        for (int mode = 0; mode < 3; ++mode) {   // 0: distinct random values, 1: only 16 distinct values (heavy ties), 2: ties plus negative values and -0.0
            std::vector<float> v(nbc);
            for (int b = 0; b < nbc; ++b) {
                if (mode == 0) v[b] = std::fabs(rnd()) + 1e-3f;
                else if (mode == 1) v[b] = 0.25f * (float) (rbits() % 16);
                else { const uint32_t r = rbits() % 20; v[b] = r == 0 ? -0.0f : r < 6 ? -0.5f : 0.25f * (float) (r - 5); }
            }
            for (int b = 0; b < nbc; ++b) pk[(size_t) b * kD] = h2u(v[b]);
            g_q->memcpy(ls.pooled, pk.data(), pk.size() * 2).wait();
            Win win;
            win.pos0 = dscalar(4 * c - 1);   // row 0: complete blocks (4c - 1 + 1) / 4 = c
            win.T = 1; win.nr = 1;
            g_q->memset(sc.sel, 0xFF, (size_t) 16 * kTopBlocks * 4).wait();   // unwritten ids read -1
            select_rows(ls, sc, win, g_q);
            g_q->wait();
            const auto sel = down(sc.sel, kTopBlocks);
            const int ns = down(sc.nsel, 1)[0];
            // the host's scores: exactly the kernel's (a chain of one product per head: q0 * k0 = k0, relu, summed in order, times 1/sqrt(128))
            std::vector<float> s(c);
            for (int b = 0; b < c; ++b) {
                const float k0 = u2h(pk[(size_t) b * kD]);
                float a = std::max(k0, 0.f);
                a += std::max(k0, 0.f); a += std::max(k0, 0.f); a += std::max(k0, 0.f);
                s[b] = a * kInvSqrtD;
            }
            const HostSel hs = host_select(s, c);
            const std::string diff = select_diff(sel, ns, hs, s, down(sc.score, (size_t) c), down(sc.dbg, 16), c);
            ++total;
            if (!diff.empty()) { ++bad; check(fmt("select: c %.0f blocks, mode %.0f", c, mode), false, diff); }
            {   // the old kernels on the same scores (information only)
                g_q->memcpy(dsc, s.data(), (size_t) c * 4).wait();
                const std::vector<int> zero16(16, 0);
                for (int var = 0; var < 2; ++var) {
                    g_q->memset(sc.sel, 0xFF, (size_t) kTopBlocks * 4).wait();
                    if (var == 0) old_select<TagOldA>(*g_q, dsc, c, sc.sel, sc.nsel);
                    else old_select_agg<TagOldB>(*g_q, dsc, c, sc.sel, sc.nsel);
                    g_q->wait();
                    const std::string d2 = select_diff(down(sc.sel, kTopBlocks), down(sc.nsel, 1)[0], hs, s, s, zero16, c);
                    if (!d2.empty()) { ++var_bad[var]; if (var_first[var].empty()) var_first[var] = fmt("c %.0f mode %.0f: ", c, mode) + d2; }
                }
            }
        }
    }
    for (int var = 0; var < 2; ++var)
        std::printf("info   old select %-44s on the same scores: %d of %d cases differ%s%s\n",
                    var == 0 ? "916106b3 (generic atomics, library scan)" : "05573cfe (+ sub-group aggregation)", var_bad[var], total,
                    var_first[var].empty() ? "" : "; first: ", var_first[var].c_str());
    check("select: " + std::to_string(total) + " cases (c 100..65536, distinct / tied / -0.0) exact", bad == 0);
    // a tree window: c = root / 4
    {
        Win win;
        win.pos0 = dscalar(8000);
        win.tp0 = dscalar(8000);
        win.T = 4; win.nr = 4;
        std::vector<float> v(nbc);
        for (int b = 0; b < nbc; ++b) { v[b] = std::fabs(rnd()) + 1e-3f; pk[(size_t) b * kD] = h2u(v[b]); }
        g_q->memcpy(ls.pooled, pk.data(), pk.size() * 2).wait();
        g_q->memset(sc.sel, 0xFF, (size_t) 16 * kTopBlocks * 4).wait();
        select_rows(ls, sc, win, g_q);
        g_q->wait();
        const auto sel = down(sc.sel, (size_t) 4 * kTopBlocks);
        const auto nsv = down(sc.nsel, 4);
        const auto dbg = down(sc.dbg, 4 * 16);
        const auto scv = down(sc.score, (size_t) 3 * nbc + 2000);
        std::vector<float> s(2000);
        for (int b = 0; b < 2000; ++b) {
            const float k0 = u2h(pk[(size_t) b * kD]);
            float a = std::max(k0, 0.f);
            a += std::max(k0, 0.f); a += std::max(k0, 0.f); a += std::max(k0, 0.f);
            s[b] = a * kInvSqrtD;
        }
        const HostSel hs = host_select(s, 2000);
        std::string diff;
        for (int r = 0; r < 4 && diff.empty(); ++r) {
            const std::vector<int> row(sel.begin() + (size_t) r * kTopBlocks, sel.begin() + (size_t) (r + 1) * kTopBlocks);
            const std::vector<float> rs(scv.begin() + (size_t) r * nbc, scv.begin() + (size_t) r * nbc + 2000);
            const std::vector<int> rd(dbg.begin() + r * 16, dbg.begin() + (r + 1) * 16);
            const std::string d = select_diff(row, nsv[r], hs, s, rs, rd, 2000);
            if (!d.empty()) diff = fmt("row %.0f: ", r) + d;
        }
        check("select: tree window rows share the committed blocks (c = root / 4)", diff.empty(), diff);
    }
    free_scratch(sc, g_q);
}

// ---- attend ----------------------------------------------------------------------------------------------------------------------------
struct Cache {
    int ctx, vt;
    std::vector<int8_t> kcodes;       // [HKV][ctx][256]
    std::vector<uint16_t> kscale;     // [HKV][ctx][4] half bits
    std::vector<int8_t> vcodes;       // [HKV][vt/32][256][32]
    std::vector<float> vscale;        // [HKV][vt]
    void* dk = nullptr; void* dv = nullptr;
    int8_t k(int kvh, int x, int d) const { return kcodes[((size_t) kvh * ctx + x) * 256 + d]; }
    float ks(int kvh, int x, int g) const { return u2h(kscale[((size_t) kvh * ctx + x) * 4 + g]); }
    int8_t v(int kvh, int x, int d) const { return vcodes[(((size_t) kvh * (vt / 32) + x / 32) * 256 + d) * 32 + x % 32]; }
    float vs(int kvh, int x) const { return vscale[(size_t) kvh * vt + x]; }
};
Cache make_cache(int ctx) {
    Cache c;
    c.ctx = ctx;
    c.vt = (ctx + 63) / 64 * 64 + 64;
    c.kcodes.resize((size_t) 4 * ctx * 256);
    c.kscale.resize((size_t) 4 * ctx * 4);
    c.vcodes.resize((size_t) 4 * (c.vt / 32) * 256 * 32);
    c.vscale.resize((size_t) 4 * c.vt);
    for (auto& x : c.kcodes) x = (int8_t) (int) (rbits() % 255) - 127;
    for (auto& x : c.vcodes) x = (int8_t) (int) (rbits() % 255) - 127;
    for (auto& x : c.kscale) x = h2u(0.01f + 0.02f * std::fabs(rnd()));
    for (auto& x : c.vscale) x = 0.01f + 0.02f * std::fabs(rnd());
    // the engine's allocation: codes, then the scales
    std::vector<unsigned char> kb((size_t) 4 * ctx * 256 + (size_t) 4 * ctx * 8);
    std::memcpy(kb.data(), c.kcodes.data(), c.kcodes.size());
    std::memcpy(kb.data() + c.kcodes.size(), c.kscale.data(), c.kscale.size() * 2);
    std::vector<unsigned char> vb(c.vcodes.size() + c.vscale.size() * 4);
    std::memcpy(vb.data(), c.vcodes.data(), c.vcodes.size());
    std::memcpy(vb.data() + c.vcodes.size(), c.vscale.data(), c.vscale.size() * 4);
    c.dk = up(kb); c.dv = up(vb);
    return c;
}

// the reference attention of row (cells given): out[h][d] for the 24 heads, float64 softmax over the listed cells
void ref_attend(const Cache& c, const float* q, const float* gate, const std::vector<int>& cells, std::vector<double>& out) {
    out.assign(24 * kHD, 0.0);
    for (int kvh = 0; kvh < 4; ++kvh) {
        for (int hh = 0; hh < 6; ++hh) {
            const int head = kvh * 6 + hh;
            std::vector<double> s(cells.size());
            double M = -1e300;
            for (size_t i = 0; i < cells.size(); ++i) {
                double a = 0;
                for (int g = 0; g < 4; ++g) {
                    double p = 0;
                    for (int d = g * 64; d < g * 64 + 64; ++d) p += (double) q[(size_t) head * kHD + d] * (double) c.k(kvh, cells[i], d);
                    a += p * (double) c.ks(kvh, cells[i], g);
                }
                s[i] = a / 16.0;
                M = std::max(M, s[i]);
            }
            double den = 0;
            std::vector<double> acc(kHD, 0.0);
            for (size_t i = 0; i < cells.size(); ++i) {
                const double w = std::exp(s[i] - M);
                den += w;
                for (int d = 0; d < kHD; ++d) acc[d] += w * (double) c.vs(kvh, cells[i]) * (double) c.v(kvh, cells[i], d);
            }
            for (int d = 0; d < kHD; ++d) {
                const double g = gate[(size_t) head * kHD + d];
                out[(size_t) head * kHD + d] = acc[d] / den / (1.0 + std::exp(-g));
            }
        }
    }
}

void test_attend() {
    const int ctx = 12000;
    Cache c = make_cache(ctx);
    Scratch sc;
    alloc_scratch(sc, 16, kMaxWin, ctx, g_q);
    for (int variant = 0; variant < 4; ++variant) {
        // 0: linear window of 16 at P = 1500 (c <= 512: everything), 1: linear window of 16 at P = 9001 (c > 512: 512 random blocks a row),
        // 2: one row at P = 8999 (own block complete), 3: a tree window of 11 rows over root 9001
        const bool tree = variant == 3;
        const int T = variant == 2 ? 1 : (tree ? 11 : 16);
        const int P = variant == 0 ? 1500 : variant == 2 ? 8999 : 9001;
        std::vector<float> q((size_t) T * 24 * kHD), gate((size_t) T * 24 * kHD);
        for (auto& x : q) x = rnd();
        for (auto& x : gate) x = rnd();
        std::vector<int32_t> sel((size_t) 16 * kTopBlocks, 0), nsel(16, 0);
        std::vector<std::vector<int>> cells(T);
        std::vector<uint32_t> tmask(T, 0);
        for (int t = 0; t < T; ++t) {
            const int cell = P + t;
            const int cvis = tree ? P / 4 : (cell + 1) / 4;
            std::vector<int> ids;
            if (cvis <= kTopBlocks) { for (int b = 0; b < cvis; ++b) ids.push_back(b); }
            else {
                std::vector<char> pick(cvis, 0);
                int n = 0;
                if (!tree && (cell + 1) % kRatio == 0 && (variant == 2 || (rbits() & 1))) { pick[cvis - 1] = 1; n = 1; }   // the own block (complete) selected: counted once
                while (n < kTopBlocks) { const int b = (int) (rbits() % cvis); if (!pick[b]) { pick[b] = 1; ++n; } }
                for (int b = 0; b < cvis; ++b) if (pick[b]) ids.push_back(b);
            }
            nsel[t] = (int) ids.size();
            for (size_t i = 0; i < ids.size(); ++i) sel[(size_t) t * kTopBlocks + i] = ids[i];
            // the reference cells: the selected blocks' cells, then the own region (the own block, if selected, once)
            std::vector<int> cl;
            const int own = cell / kRatio;
            for (int b : ids) { if (!tree && b == own) continue; for (int j = 0; j < 4; ++j) cl.push_back(b * 4 + j); }
            if (!tree) { for (int x = own * 4; x <= cell; ++x) cl.push_back(x); }
            else {
                uint32_t m = 0;
                for (int i = 0; i <= t; ++i) if (i == t || (rbits() & 3) == 0) m |= 1u << i;
                tmask[t] = m;
                for (int x = P / 4 * 4; x < P; ++x) cl.push_back(x);
                for (int i = 0; i < T; ++i) if (m >> i & 1u) cl.push_back(P + i);
            }
            cells[t] = cl;
        }
        float* dq = up(q); float* dg = up(gate);
        float* dout = dev<float>((size_t) T * 24 * kHD);
        int32_t* dsel = up(sel); int32_t* dns = up(nsel);
        uint32_t* dtm = up(tmask);
        Win win;
        win.pos0 = dscalar(P);
        win.T = T; win.nr = T;
        if (tree) { win.tp0 = dscalar(P); win.tmask = dtm; }
        AttendArgs a;
        a.q = dq; a.gate = dg; a.out = dout; a.kc = c.dk; a.vc = c.dv; a.ctx = ctx; a.vt = c.vt;
        a.sel = dsel; a.nsel = dns; a.nitems = sc.nitems; a.pacc = sc.pacc; a.pm = sc.pm; a.pl = sc.pl;
        attend_rows(a, win, g_q);
        g_q->wait();
        const auto out = down(dout, (size_t) T * 24 * kHD);
        double worst = 0;
        bool ok = true;
        for (int t = 0; t < T; ++t) {
            std::vector<double> ref;
            ref_attend(c, &q[(size_t) t * 24 * kHD], &gate[(size_t) t * 24 * kHD], cells[t], ref);
            double mx = 1e-9, e = 0;
            for (double v : ref) mx = std::max(mx, std::fabs(v));
            for (size_t i = 0; i < ref.size(); ++i) e = std::max(e, std::fabs(ref[i] - (double) out[(size_t) t * 24 * kHD + i]));
            worst = std::max(worst, e / (3e-4 * mx + 1e-6));
            ok = ok && e <= 3e-4 * mx + 1e-6;
        }
        const char* names[] = {"linear window of 16, c <= 512 (dense-equal)", "linear window of 16, 512 of ~2250 blocks", "one row, own block complete", "tree window of 11 (tmask)"};
        check(std::string("attend: ") + names[variant] + " vs float64", ok, fmt("worst error / bound %.3f", worst));
        // each row alone
        bool same = true;
        for (int t = 0; t < T; ++t) {
            Win w1 = win;
            w1.row0 = t; w1.nr = 1;
            AttendArgs a1 = a;
            a1.q = dq + (size_t) t * 24 * kHD; a1.gate = dg + (size_t) t * 24 * kHD; a1.out = dout + 0;   // (written to a scratch row)
            a1.sel = dsel + (size_t) t * kTopBlocks; a1.nsel = dns + t;
            float* d1 = dev<float>(24 * kHD);
            a1.out = d1;
            attend_rows(a1, w1, g_q);
            g_q->wait();
            const auto o1 = down(d1, 24 * kHD);
            same = same && std::memcmp(o1.data(), &out[(size_t) t * 24 * kHD], 24 * kHD * 4) == 0;
            sycl::free(d1, *g_q);
        }
        check(std::string("attend: rows alone bitwise the window's rows (") + names[variant] + ")", same);
        sycl::free(dq, *g_q); sycl::free(dg, *g_q); sycl::free(dout, *g_q); sycl::free(dsel, *g_q); sycl::free(dns, *g_q); sycl::free(dtm, *g_q);
    }
    free_scratch(sc, g_q);
}

// ---- tree commit ------------------------------------------------------------------------------------------------------------------------
void test_pool_tree(const Weights16& w, const LayerWeights& lw) {
    const int ctx = 4096;
    LayerState ls;
    alloc_state(ls, ctx, g_q);
    Rope rp;
    std::vector<float> ring0((size_t) kRing * kD), traw((size_t) kMaxWin * kD);
    for (auto& x : ring0) x = rnd();
    for (auto& x : traw) x = rnd();
    g_q->memcpy(ls.ring, ring0.data(), ring0.size() * 4);
    g_q->memcpy(ls.traw, traw.data(), traw.size() * 4).wait();
    const int p0 = 2002, n_keep = 7;
    std::vector<int32_t> path = {0, 1, 3, 4, 8, 9, 10};
    int32_t* dp0 = dscalar(p0);
    int32_t* dpath = up(path);
    pool_tree(lw, ls, dp0, dpath, n_keep, rp, nullptr, g_q);
    g_q->wait();
    const auto ring = down(ls.ring, (size_t) kRing * kD);
    const auto pooled = down(static_cast<const uint16_t*>(ls.pooled), (size_t) nb_cap_for(ctx) * kD);
    bool ring_ok = true;
    for (int k = 0; k < n_keep; ++k)
        for (int i = 0; i < kD; ++i) ring_ok = ring_ok && ring[(size_t) ((p0 + k) % kRing) * kD + i] == traw[(size_t) path[k] * kD + i];
    check("pool_tree: the kept rows' raw keys in the ring (bitwise)", ring_ok);
    int bad = 0, nblk = 0;
    for (int b = p0 / kRatio; b * kRatio + 3 <= p0 + n_keep - 1; ++b) {
        std::vector<double> m(kD);
        for (int i = 0; i < kD; ++i) {
            float a[4];
            for (int j = 0; j < 4; ++j) { const int c = b * kRatio + j; a[j] = c >= p0 ? traw[(size_t) path[c - p0] * kD + i] : ring0[(size_t) (c % kRing) * kD + i]; }
            m[i] = (double) ((((a[0] + a[1]) + a[2]) + a[3]) * 0.25f);
        }
        m = rms_mul(m, w.kn.data(), rp.eps);
        rope64(m, (double) (b * kRatio), rp.base, rp.n_rot);
        for (int i = 0; i < kD; ++i) if (std::fabs((double) u2h(pooled[(size_t) b * kD + i]) - m[i]) > std::fabs(m[i]) / 1024.0 + 5e-4) ++bad;
        ++nblk;
    }
    check("pool_tree: " + std::to_string(nblk) + " pooled block(s) vs float64", bad == 0 && nblk > 0);
    free_state(ls, g_q);
}

// ---- bench ------------------------------------------------------------------------------------------------------------------------------
double sequence_us(sycl::queue& q, const std::function<void()>& run, int reps = 12) {
    std::vector<double> ts;
    char* head = sycl::malloc_device<char>((size_t) 768 << 20, q);
    for (int rep = 0; rep < reps + 3; ++rep) {
        if (head) q.memset(head, rep & 0xFF, (size_t) 768 << 20);
        sycl::event e0 = q.ext_oneapi_submit_barrier();
        run();
        sycl::event e1 = q.ext_oneapi_submit_barrier();
        q.wait();
        if (rep < 3) continue;
        ts.push_back(((double) e1.get_profiling_info<sycl::info::event_profiling::command_end>() - (double) e0.get_profiling_info<sycl::info::event_profiling::command_end>()) / 1e3);
    }
    if (head) sycl::free(head, q);
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
}

void bench(const std::vector<int>& ctxs, const Weights16& w, const LayerWeights& lw) {
    (void) w;
    for (int ctx : ctxs) {
        const int nbc = nb_cap_for(ctx), P = ctx - 64;
        std::printf("\n--- ctx %d cells (%d blocks), position %d, 16 layers a step; the floor is bytes / 590 GB/s ---\n", ctx, nbc, P);
        Cache c = make_cache(ctx);
        std::vector<LayerState> L(16);
        std::vector<uint16_t> pk((size_t) nbc * kD);
        for (auto& l : L) {
            alloc_state(l, ctx, g_q);
            for (auto& x : pk) x = h2u(rnd());
            g_q->memcpy(l.pooled, pk.data(), pk.size() * 2).wait();
        }
        Scratch sc;
        alloc_scratch(sc, 64, kMaxWin, ctx, g_q);
        std::vector<float> qi((size_t) 64 * kHeads * kD), hrow((size_t) 64 * kEmbd);
        for (auto& x : qi) x = rnd();
        for (auto& x : hrow) x = rnd();
        g_q->memcpy(sc.qidx, qi.data(), qi.size() * 4).wait();
        float* dh = up(hrow);
        std::vector<float> qa((size_t) 16 * 24 * kHD), ga((size_t) 16 * 24 * kHD);
        for (auto& x : qa) x = rnd();
        for (auto& x : ga) x = rnd();
        float* dq = up(qa); float* dg = up(ga);
        float* dout = dev<float>((size_t) 16 * 24 * kHD);
        sycl::queue pq(g_q->get_context(), g_q->get_device(), {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
        sycl::queue* saved = g_q;
        g_q = &pq;
        std::printf("%4s | %9s %9s | %9s %9s | %9s %9s %9s | %9s\n", "rows", "index us", "GB/s", "score+sel", "GB/s", "attend", "GB/s K+V", "V groups", "step ms");
        double one_row_step = 0;
        for (int T : {1, 2, 4, 16}) {
            Win win;
            win.pos0 = dscalar(P);
            win.T = T; win.nr = T;
            Rope rp;
            const double t_index = sequence_us(pq, [&] { for (int l = 0; l < 16; ++l) index_rows(lw, L[l], sc, dh, win, rp, false, &pq); });
            const double t_sel = sequence_us(pq, [&] { for (int l = 0; l < 16; ++l) select_rows(L[l], sc, win, &pq); });
            AttendArgs a;
            a.q = dq; a.gate = dg; a.out = dout; a.kc = c.dk; a.vc = c.dv; a.ctx = ctx; a.vt = c.vt;
            a.sel = sc.sel; a.nsel = sc.nsel; a.nitems = sc.nitems; a.pacc = sc.pacc; a.pm = sc.pm; a.pl = sc.pl;
            // distinct 32-cell V groups and the cells of row 0's selection of the last layer (the implied bytes of the attention)
            select_rows(L[15], sc, win, &pq);
            pq.wait();
            const auto sel = down(sc.sel, kTopBlocks);
            std::vector<int> grp;
            for (int b : sel) grp.push_back(b / 8);
            std::sort(grp.begin(), grp.end());
            grp.erase(std::unique(grp.begin(), grp.end()), grp.end());
            const double cells = (double) (kTopBlocks * kRatio + kRatio);
            const double bytes_k = 4.0 * cells * (256 + 8), bytes_v = 4.0 * ((double) grp.size() * 8192.0 + cells * 4.0);
            const double bytes_keys = (double) (P + 1) / kRatio * 256.0 * ((T + kScoreRows - 1) / kScoreRows);
            const double t_all = sequence_us(pq, [&] { for (int l = 0; l < 16; ++l) { select_rows(L[l], sc, win, &pq); attend_rows(a, win, &pq); } });
            const double t_att = t_all - t_sel;
            const double w_bytes = (double) (kHeads * kD + kD) * kEmbd * 2.0;
            std::printf("%4d | %9.1f %9.0f | %9.1f %9.0f | %9.1f %9.0f %9zu | %9.3f\n", T, t_index, 16.0 * w_bytes / (t_index * 1e3), t_sel,
                        16.0 * bytes_keys / (t_sel * 1e3), t_att, 16.0 * (bytes_k + bytes_v) / (std::max(t_att, 1e-3) * 1e3), grp.size(),
                        (t_index + t_sel + std::max(t_att, 0.0)) / 1e3);
            if (T == 1) one_row_step = t_index + t_sel + t_att;
            else if (T == 16) std::printf("     16 rows / 1 row, whole indexer+attention step: %.2fx (target <= 1.15x for shared gathers; per-row work-groups re-read through the cache)\n",
                                          (t_index + t_sel + t_att) / one_row_step);
            sycl::free(const_cast<int32_t*>(win.pos0), pq);
        }
        g_q = saved;
        for (auto& l : L) free_state(l, g_q);
        free_scratch(sc, g_q);
        sycl::free(dh, *g_q); sycl::free(dq, *g_q); sycl::free(dg, *g_q); sycl::free(dout, *g_q);
        (void) one_row_step;
    }
}

}   // namespace

int parity_main(int argc, char** argv) {
    bool do_bench = false;
    std::vector<int> ctxs = {25120, 118000, 255000};
    for (int i = 0; i < argc; ++i) {
        if (std::string(argv[i]) == "--bench") {
            do_bench = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                ctxs.clear();
                std::string s = argv[i + 1];
                size_t p = 0;
                while (p < s.size()) { ctxs.push_back(std::atoi(s.c_str() + p)); p = s.find(',', p); if (p == std::string::npos) break; ++p; }
            }
        }
    }
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    g_q = &q;
    std::printf("QSA parity on %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const Weights16 w = make_weights();
    const LayerWeights lw = upload(w);
    test_index(w, lw);
    test_score(w);
    test_select();
    test_attend();
    test_pool_tree(w, lw);
    std::printf("\n%s: %d check(s) failed\n", g_bad ? "FAIL" : "PASS", g_bad);
    if (do_bench && g_bad == 0) bench(ctxs, w, lw);
    return g_bad ? 1 : 0;
}

}   // namespace strata::q36qsa
