// src/kernels/gdn_parity.cpp - P2.S2's test for the GDN recurrence, the conv and the two norms.
//
// `ref/gdn.py` carries eleven PROPERTY checks, several of which exist because a rival reading of the SOURCE
// produces well-formed output.  The four that the kernels here can be wrong about are re-pinned below, each
// asserted OBSERVABLE against a deliberately-wrong computation before the kernel is judged:
//
//   1. MODULO head pairing, h % h_k, not INTERLEAVE.  With H_k=2 and H_v=4 the two give [0,1,0,1] and
//      [0,0,1,1]: they differ at h=1 and h=2.  Both produce a full-rank state of the right shape.
//   2. Decay applied BEFORE the rank-1 update.  Applying it after is a one-line move that keeps every shape
//      and every magnitude.
//   3. The conv's SLIDE DIRECTION.  `kernel[0]` must read the OLDEST state row and the new input must land in
//      the LAST state row; reversing either is invisible on a symmetric state, so the state rows here are
//      labelled so each position is identifiable.
//   4. `l2_norm`'s `+ eps` is an absolute floor on the SQUARED NORM, not on the mean.  The mean reading
//      differs by sqrt(S) = 11.3x at S = 128.  `ref/gdn.py` PROPERTY 1 instruments the derivation, so it is
//      not taken on trust there either.
//
// It also checks the STATE LAYOUT, which is the one place these kernels intentionally differ from the
// reference: (S, h_v, S) with j fastest instead of (S, S, h_v).  A layout mix-up is silent, so the state is
// filled with a value that encodes its own coordinates.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "strata/prefill/kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

void check(dpct::err0 e, const char *what) {
}

float sigmoid_f(float x) { return 1.0f / (1.0f + std::exp(-x)); }

/// The reference's `gdn_step`, with switches for the rival readings.
///
/// `state` is passed in the REFERENCE layout (S, S, h_v) - i.e. (i, j, h) - so that this function is a
/// transcription of `ref/gdn.py` and not of the kernel.
void ref_step(std::vector<double>& state, const std::vector<float>& q, const std::vector<float>& k,
              const std::vector<float>& v, const std::vector<float>& gate, const std::vector<float>& beta,
              int S, int h_k, int h_v, std::vector<float>& o, bool modulo_heads, bool decay_first) {
    auto at = [&](int i, int j, int h) -> double& { return state[(size_t) (i * S + j) * h_v + h]; };
    std::vector<float> oo((size_t) h_v * S, 0.0f);
    for (int h = 0; h < h_v; ++h) {
        const int src = modulo_heads ? (h % h_k) : (h / (h_v / h_k));
        const double dec = std::exp((double) gate[(size_t) h]);
        const double b = (double) beta[(size_t) h];
        for (int j = 0; j < S; ++j) {
            if (decay_first) for (int i = 0; i < S; ++i) at(i, j, h) *= dec;
            double sk = 0;
            for (int i = 0; i < S; ++i) sk += at(i, j, h) * (double) k[(size_t) (src * S + i)];
            const double d = ((double) v[(size_t) (h * S + j)] - sk) * b;
            for (int i = 0; i < S; ++i) at(i, j, h) += (double) k[(size_t) (src * S + i)] * d;
            if (!decay_first) for (int i = 0; i < S; ++i) at(i, j, h) *= dec;
            double dot = 0;
            for (int i = 0; i < S; ++i) dot += at(i, j, h) * (double) q[(size_t) (src * S + i)];
            oo[(size_t) (h * S + j)] = (float) dot;
        }
    }
    o = oo;
}

/// `ref/gdn.py::conv_step` in its own (d_conv, C) row-major orientation.
void ref_conv(std::vector<float>& cs, const std::vector<float>& x, const std::vector<float>& kern, int C,
              int dc, std::vector<float>& out) {
    out.assign((size_t) C, 0.0f);
    for (int c = 0; c < C; ++c) {
        double acc = 0;
        for (int i = 0; i < dc - 1; ++i) acc += (double) cs[(size_t) (i * C + c)] * (double) kern[(size_t) (i * C + c)];
        acc += (double) x[(size_t) c] * (double) kern[(size_t) ((dc - 1) * C + c)];
        out[(size_t) c] = (float) acc;
        for (int i = 0; i < dc - 2; ++i) cs[(size_t) (i * C + c)] = cs[(size_t) ((i + 1) * C + c)];
        cs[(size_t) ((dc - 2) * C + c)] = x[(size_t) c];
    }
}

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs((double) a[i] - (double) b[i]);
        m += std::fabs((double) a[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

// --prefill-variants A,B: the two gdn_recurrence_variant kernels compared (A is the baseline, B must give its bits).
// --prefill-fp64 N: also run the recurrence in FP64 on the host for value heads 0..N-1 (the kernels' state layout) and
// print both variants' error against it, the state and the raw outputs (o / sqrt(S), what y holds before the norm).
int g_variant_a = 0, g_variant_b = 6, g_fp64_heads = 0;

// The FP64 recurrence of value head hv over T tokens: st (S, HV, S) in, updated; oc[t * HV * S + hv * S + j].
void fp64_head(int hv, int T, const std::vector<float>& st0, const std::vector<float>& h,
               const std::vector<float>& gate, const std::vector<float>& beta, std::vector<double>& st,
               std::vector<double>& oc) {
    constexpr int S = 128, HK = 16, HV = 48, C = 10240;
    const int qh = hv % HK;
    std::vector<double> s((size_t) S * S);   // s[i * S + j]: row i, column j of this head
    for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j) s[(size_t) i * S + j] = st0[((size_t) i * HV + hv) * S + j];
    std::vector<double> kv(S), o(S);
    for (int t = 0; t < T; ++t) {
        const float* ht = h.data() + (size_t) t * C;
        const float *q = ht + qh * S, *k = ht + HK * S + qh * S, *v = ht + 2 * HK * S + hv * S;
        const double g = std::exp((double) gate[(size_t) t * HV + hv]), b = beta[(size_t) t * HV + hv];
        std::fill(kv.begin(), kv.end(), 0.0);
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < S; ++j) kv[j] += s[(size_t) i * S + j] * (double) k[i];
        for (int j = 0; j < S; ++j) kv[j] = ((double) v[j] - g * kv[j]) * b;   // delta
        std::fill(o.begin(), o.end(), 0.0);
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < S; ++j) {
                double& x = s[(size_t) i * S + j];
                x = g * x + (double) k[i] * kv[j];
                o[j] += x * (double) q[i];
            }
        for (int j = 0; j < S; ++j) oc[(size_t) t * HV * S + (size_t) hv * S + j] = o[j] / std::sqrt((double) S);
    }
    for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j) st[((size_t) i * HV + hv) * S + j] = s[(size_t) i * S + j];
}

// Compare every state element and every token/head output, including FP16 bits.
// Fixtures have nonzero initial state, modulo-paired heads, signed q/k/v, and
// both shallow and deep decay. A tolerance is diagnostic only, never a pass.
int prefill_case(int T, int seed, int repeats) {
    constexpr int S = 128, HK = 16, HV = 48, C = 10240;
    sycl::queue q(dpct::get_in_order_queue().get_device(),
                  sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
    // Variants 6 and 7 run on Intel GPUs only (the engine throws elsewhere): skip, not fail, on any other device.
    const auto dev = q.get_device();
    const auto sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    if (!dev.is_gpu() || dev.get_info<sycl::info::device::vendor_id>() != 0x8086 ||
        std::find(sizes.begin(), sizes.end(), size_t(32)) == sizes.end() ||
        std::find(sizes.begin(), sizes.end(), size_t(16)) == sizes.end()) {
        static bool said = false;
        if (!said) std::printf("prefill gdn: skipped (needs an Intel GPU with sub-groups 16 and 32)\n");
        said = true;
        return 0;
    }
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> st(size_t(S) * HV * S), h(size_t(T) * C),
        gate(size_t(T) * HV), beta(gate.size()), z(size_t(T) * HV * S), gamma(S);
    for (auto& x : st) x = gauss(rng) * 0.03f;
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < C; ++j) h[size_t(t) * C + j] = gauss(rng) * (j < 2 * HK * S ? 0.08f : 0.7f);
    }
    for (auto& x : gate) x = -(seed % 2 ? 0.02f : 2.0f) * (1.0f + std::fabs(gauss(rng)));
    for (auto& x : beta) x = 0.1f + float(rng() % 800) / 1000.0f;
    for (auto& x : z) x = gauss(rng);
    for (auto& x : gamma) x = 1.0f + 0.2f * gauss(rng);
    auto copy = [&](const std::vector<float>& src) {
        auto* ptr = sycl::malloc_device<float>(src.size(), q);
        if (!ptr) throw std::bad_alloc();
        q.memcpy(ptr, src.data(), src.size() * sizeof(float)).wait_and_throw();
        return ptr;
    };
    float *dst = copy(st), *dh = copy(h), *dg = copy(gate), *db = copy(beta), *dz = copy(z), *dgm = copy(gamma);
    float* dy = sycl::malloc_device<float>(z.size(), q);
    uint16_t* dy16 = sycl::malloc_device<uint16_t>(z.size(), q);
    if (!dy || !dy16) throw std::bad_alloc();
    std::vector<float> refst(st.size()), refy(z.size()), gotst(st.size()), goty(z.size());
    std::vector<uint16_t> ref16(z.size()), got16(z.size());
    auto call = [&](int variant) {
        if (variant == -1)
            strata::prefill::gdn_recurrence(dst, dh, dg, db, dz, dgm, 1e-6f, dy, dy16, T, &q);
        else if (variant == -2)   // the production recurrence without the output norm
            strata::prefill::gdn_recurrence_state_only(dst, dh, dg, db, dy, T, &q);
        else
            strata::prefill::gdn_recurrence_variant(variant, dst, dh, dg, db, dz, dgm, 1e-6f, dy, dy16, T, &q);
    };
    // Device time of one call (every call starts from the same state): the span between two barriers' completions.
    auto device_ms = [&](int variant) {
        q.memcpy(dst, st.data(), st.size() * sizeof(float)).wait_and_throw();
        auto e0 = q.ext_oneapi_submit_barrier();
        call(variant);
        auto e1 = q.ext_oneapi_submit_barrier();
        e1.wait_and_throw();
        return 1e-6 * double(e1.get_profiling_info<sycl::info::event_profiling::command_end>() -
                             e0.get_profiling_info<sycl::info::event_profiling::command_end>());
    };
    auto run = [&](int variant, std::vector<float>& ost, std::vector<float>& oy, std::vector<uint16_t>& o16) {
        q.memcpy(dst, st.data(), st.size() * sizeof(float)).wait_and_throw();
        call(variant);
        q.wait_and_throw();
        q.memcpy(ost.data(), dst, ost.size() * sizeof(float)).wait_and_throw();
        q.memcpy(oy.data(), dy, oy.size() * sizeof(float)).wait_and_throw();
        q.memcpy(o16.data(), dy16, o16.size() * sizeof(uint16_t)).wait_and_throw();
    };
    run(g_variant_a, refst, refy, ref16);
    run(g_variant_b, gotst, goty, got16);
    // the fastest of `repeats` device times, A and B (and the production recurrence without the norm) alternating
    double base_ms = 1e30, xe_ms = 1e30, rec_ms = 1e30;
    for (int i = 0; i < repeats; ++i) {
        base_ms = std::min(base_ms, device_ms(g_variant_a));
        xe_ms = std::min(xe_ms, device_ms(g_variant_b));
        if (repeats > 1) rec_ms = std::min(rec_ms, device_ms(-2));
    }
    if (repeats > 1) std::printf("  production recurrence without the norm: %.3f ms\n", rec_ms);
    if (g_fp64_heads > 0) {
        const int nh = std::min(g_fp64_heads, HV);
        std::vector<double> st64(st.size()), oc64(z.size());
        std::vector<std::thread> pool;
        for (int hv = 0; hv < nh; ++hv)
            pool.emplace_back([&, hv] { fp64_head(hv, T, st, h, gate, beta, st64, oc64); });
        for (auto& th : pool) th.join();
        // max |err| over the heads checked, and that over max |value| (a relative error that ignores values near 0)
        auto err = [&](const std::vector<float>& got, const std::vector<double>& want, bool state, double& emax,
                       double& vmax) {
            emax = 0, vmax = 0;
            for (int hv = 0; hv < nh; ++hv)
                for (int64_t r = 0; r < (state ? S : T); ++r)
                    for (int j = 0; j < S; ++j) {
                        const size_t at = ((size_t) r * HV + hv) * S + j;
                        emax = std::max(emax, std::fabs((double) got[at] - want[at]));
                        vmax = std::max(vmax, std::fabs(want[at]));
                    }
        };
        double se, sv, ye, yv;
        err(refst, st64, true, se, sv);
        err(refy, oc64, false, ye, yv);
        std::printf("  fp64 (%d heads) variant %d: state max_abs=%.3g (max|s|=%.3g, rel %.3g)  o max_abs=%.3g (max|o|=%.3g, rel %.3g)\n",
                    nh, g_variant_a, se, sv, se / sv, ye, yv, ye / yv);
        err(gotst, st64, true, se, sv);
        err(goty, oc64, false, ye, yv);
        std::printf("  fp64 (%d heads) variant %d: state max_abs=%.3g (max|s|=%.3g, rel %.3g)  o max_abs=%.3g (max|o|=%.3g, rel %.3g)\n",
                    nh, g_variant_b, se, sv, se / sv, ye, yv, ye / yv);
    }
    size_t state_bad = 0, y_bad = 0, half_bad = 0;
    double max_abs = 0, max_rel = 0;
    auto compare = [&](const std::vector<float>& a, const std::vector<float>& b, size_t& mismatches) {
        for (size_t i = 0; i < a.size(); ++i) {
            if (!std::isfinite(a[i]) || !std::isfinite(b[i]) ||
                std::memcmp(&a[i], &b[i], sizeof(float)) != 0) ++mismatches;
            const double diff = std::fabs(double(a[i]) - b[i]);
            max_abs = std::max(max_abs, diff);
            max_rel = std::max(max_rel, diff / std::max(std::fabs(double(a[i])), 1e-20));
        }
    };
    compare(refst, gotst, state_bad);
    compare(refy, goty, y_bad);
    for (size_t i = 0; i < ref16.size(); ++i) half_bad += ref16[i] != got16[i];
    std::printf("prefill T=%d seed=%d heads=%d: state=%zu y=%zu fp16=%zu mismatches; max_abs=%.9g max_rel=%.9g; variant %d=%.3f ms variant %d=%.3f ms\n",
                T, seed, HV, state_bad, y_bad, half_bad, max_abs, max_rel, g_variant_a, base_ms, g_variant_b, xe_ms);
    // T=7 additionally checks the production entry point (variant 7 on Battlemage or under STRATA_GDN_REG=1, else
    // variant 6 on Alchemist or under STRATA_GDN_XE=1, else variant 0) and T=0's no-op contract.
    int extra_bad = 0;
    if (T == 7) {
        run(-1, gotst, goty, got16);
        // Both variants must match anyway; compare production to baseline.
        extra_bad += std::memcmp(refst.data(), gotst.data(), refst.size() * sizeof(float)) != 0;
        extra_bad += std::memcmp(refy.data(), goty.data(), refy.size() * sizeof(float)) != 0;
        extra_bad += ref16 != got16;
        q.memcpy(dst, st.data(), st.size() * sizeof(float)).wait_and_throw();
        strata::prefill::gdn_recurrence_variant(6, dst, dh, dg, db, dz, dgm,
                                               1e-6f, dy, dy16, 0, &q);
        q.memcpy(gotst.data(), dst, st.size() * sizeof(float)).wait_and_throw();
        extra_bad += gotst != st;
        const auto saved_y = goty;
        const auto saved16 = got16;
        q.memcpy(goty.data(), dy, goty.size() * sizeof(float)).wait_and_throw();
        q.memcpy(got16.data(), dy16, got16.size() * sizeof(uint16_t)).wait_and_throw();
        extra_bad += goty != saved_y || got16 != saved16;
        std::printf("prefill production + T=0: %s\n", extra_bad ? "FAIL" : "OK");
    }
    for (auto ptr : {dst, dh, dg, db, dz, dgm, dy}) sycl::free(ptr, q);
    sycl::free(dy16, q);
    return state_bad || y_bad || half_bad || extra_bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // (progress lines survive a crash)
    bool selftest = false, prefill_only = false;
    int prefill_tokens = 0, prefill_repeat = 1;
    bool variants_given = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--selftest") selftest = true;
        else if (arg == "--prefill-only") prefill_only = true;
        else if ((arg == "--prefill-tokens" || arg == "--prefill-repeat") && i + 1 < argc) {
            char* end = nullptr;
            const long n = std::strtol(argv[++i], &end, 10);
            if (!end || *end || n < 1 || n > (arg == "--prefill-tokens" ? 32768 : 1000)) return 2;
            (arg == "--prefill-tokens" ? prefill_tokens : prefill_repeat) = int(n);
        } else if (arg == "--prefill-variants" && i + 1 < argc) {
            if (std::sscanf(argv[++i], "%d,%d", &g_variant_a, &g_variant_b) != 2) return 2;
            variants_given = true;
        } else if (arg == "--prefill-fp64" && i + 1 < argc) {
            g_fp64_heads = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "usage: gdn_parity [--selftest] [--prefill-only] [--prefill-tokens N] [--prefill-repeat N]"
                                 " [--prefill-variants A,B] [--prefill-fp64 HEADS]\n");
            return 2;
        }
    }
    const auto device = dpct::get_in_order_queue().get_device();
    std::printf("device: %s (%s), sub-groups:", device.get_info<sycl::info::device::name>().c_str(),
                device.is_cpu() ? "CPU" : device.is_gpu() ? "GPU" : "other");
    for (auto n : device.get_info<sycl::info::device::sub_group_sizes>()) std::printf(" %zu", n);
    std::printf("\n");
    int prefill_bad = 0;
    // without --prefill-variants: variant 0 against 6 and against 7
    const std::vector<int> cand = variants_given ? std::vector<int>{g_variant_b} : std::vector<int>{6, 7};
    for (int b : cand) {
        g_variant_b = b;
        for (int t : prefill_tokens ? std::vector<int>{prefill_tokens} : std::vector<int>{1, 7, 256, 1000})
            for (int seed : {2025, 2026}) prefill_bad += prefill_case(t, seed, prefill_repeat);
    }
    std::printf("prefill gdn: %d failures\n", prefill_bad);
    if (prefill_only) {
        if (selftest && !prefill_bad) std::printf("gdn_parity OK\n");
        return prefill_bad ? 1 : 0;
    }

    const int S = 128, h_k = 16, h_v = 48;      // the artifact's real geometry
    std::mt19937 rng(2024);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    int bad = prefill_bad;

    // ================= 1. the recurrence =================
    {
        // The state encodes its own coordinates, so a layout mix-up cannot pass: reading (i,j,h) where the
        // buffer holds (i,h,j) gives a DIFFERENT number, not a wrong-but-plausible one.
        std::vector<double> st_ref((size_t) S * S * h_v);
        std::vector<float> st_dev((size_t) S * h_v * S);
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < S; ++j)
                for (int h = 0; h < h_v; ++h) {
                    const float val = (float) (0.001 * (i + 2 * j + 3 * h) - 1.0);
                    st_ref[(size_t) (i * S + j) * h_v + h] = val;
                    st_dev[(size_t) (i * h_v + h) * S + j] = val;   // (S, h_v, S), j fastest
                }

        std::vector<float> q((size_t) h_k * S), k((size_t) h_k * S), v((size_t) h_v * S), gate((size_t) h_v),
            beta((size_t) h_v);
        for (auto& x : q) x = gauss(rng) * 0.1f;
        for (auto& x : k) x = gauss(rng) * 0.1f;
        for (auto& x : v) x = gauss(rng);
        for (auto& x : gate) x = -(2.0f + 4.0f * (float) (rng() % 100) / 100.0f);   // dec in (0.0025, 0.135]
        for (auto& x : beta) x = (float) (rng() % 100) / 100.0f;

        std::vector<float> want_o;
        std::vector<double> st_mod = st_ref;
        ref_step(st_mod, q, k, v, gate, beta, S, h_k, h_v, want_o, true, true);

        // the traps
        {
            std::vector<double> a = st_ref;
            std::vector<float> oo;
            ref_step(a, q, k, v, gate, beta, S, h_k, h_v, oo, false, true);
            const double rel = rel_l1(want_o, oo);
            std::printf("  %-42s %-4s (%.2f%% apart)\n", "modulo vs interleave head pairing",
                        rel > 0.05 ? "yes" : "*** NO ***", rel * 100);
            if (!(rel > 0.05)) ++bad;
        }
        {
            std::vector<double> a = st_ref;
            std::vector<float> oo;
            ref_step(a, q, k, v, gate, beta, S, h_k, h_v, oo, true, false);
            const double rel = rel_l1(want_o, oo);
            std::printf("  %-42s %-4s (%.2f%% apart)\n", "decay before vs after the update",
                        rel > 0.05 ? "yes" : "*** NO ***", rel * 100);
            // The two readings differ by exactly `k (x) (beta * v * (1 - dec))`, so the fixture is only able
            // to see the trap when `dec` is far from 1 and `beta` is far from 0.  The first version used
            // gate in [-0.99, 0] - dec in [0.37, 1] - and managed 3.25%, which is a threshold the wrong
            // reading nearly clears.  Deep decay is also the realistic regime: `ssm_a = -exp(A_log) < 0` and
            // softplus is positive, so gate is comfortably negative in the real model.
            if (!(rel > 0.05)) ++bad;
        }

        // the kernel
        float *d_st = nullptr, *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_g = nullptr, *d_b = nullptr,
              *d_o = nullptr;
        check(DPCT_CHECK_ERROR(
                  d_st = (float *)sycl::malloc_device(
                      st_dev.size() * 4, dpct::get_in_order_queue())),
              "st");
        check(DPCT_CHECK_ERROR(d_q = (float *)sycl::malloc_device(
                                   q.size() * 4, dpct::get_in_order_queue())),
              "q");
        check(DPCT_CHECK_ERROR(d_k = (float *)sycl::malloc_device(
                                   k.size() * 4, dpct::get_in_order_queue())),
              "k");
        check(DPCT_CHECK_ERROR(d_v = (float *)sycl::malloc_device(
                                   v.size() * 4, dpct::get_in_order_queue())),
              "v");
        check(
            DPCT_CHECK_ERROR(d_g = (float *)sycl::malloc_device(
                                 gate.size() * 4, dpct::get_in_order_queue())),
            "g");
        check(
            DPCT_CHECK_ERROR(d_b = (float *)sycl::malloc_device(
                                 beta.size() * 4, dpct::get_in_order_queue())),
            "b");
        check(DPCT_CHECK_ERROR(
                  d_o = (float *)sycl::malloc_device(
                      (size_t)h_v * S * 4, dpct::get_in_order_queue())),
              "o");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                  d_st, st_dev.data(), st_dev.size() * 4).wait()),
              "cst");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_q, q.data(),
                                                                 q.size() * 4).wait()),
              "cq");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_k, k.data(),
                                                                 k.size() * 4).wait()),
              "ck");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_v, v.data(),
                                                                 v.size() * 4).wait()),
              "cv");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                  d_g, gate.data(), gate.size() * 4).wait()),
              "cg");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                  d_b, beta.data(), beta.size() * 4).wait()),
              "cb");

        strata::kernels::GdnShapes sh{S, h_k, h_v};
        strata::kernels::gdn_step(d_st, d_q, d_k, d_v, d_g, d_b, d_o, sh, nullptr);

        std::vector<float> got_o((size_t) h_v * S), got_st(st_dev.size());
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                   .memcpy(got_o.data(), d_o, got_o.size() * 4)
                                   .wait()),
              "cgo");
        check(
            DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                 .memcpy(got_st.data(), d_st, got_st.size() * 4)
                                 .wait()),
            "cgst");

        // o, and the STATE read back through the declared layout
        const double rel_o = rel_l1(want_o, got_o);
        double rel_st = 0, mag_st = 0;
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < S; ++j)
                for (int h = 0; h < h_v; ++h) {
                    const size_t dev = (size_t) (i * h_v + h) * S + j;
                    const double w = st_mod[(size_t) (i * S + j) * h_v + h];
                    rel_st += std::fabs(w - (double) got_st[dev]);
                    mag_st += std::fabs(w);
                }
        rel_st /= (mag_st > 1e-30 ? mag_st : 1e-30);
        std::printf("\n  %-42s rel %.3e\n", "o vs reference", rel_o);
        std::printf("  %-42s rel %.3e\n", "state (read via (S,h_v,S)) vs reference", rel_st);
        // The kernel accumulates in f32 and the reference in f64, and the state is a RECURRENCE over 128
        // terms, so 1e-5 is the arithmetic's floor here and not a structural allowance.
        if (!(rel_o <= 1e-5)) { std::printf("    *** o over 1e-5 ***\n"); ++bad; }
        if (!(rel_st <= 1e-5)) { std::printf("    *** state over 1e-5 ***\n"); ++bad; }

        // a zero state stays zero, and a zero gate/beta leaves the state untouched (the reference's
        // PROPERTY 4 setup) - a cheap independent invariant that no layout can fake
        {
            std::vector<float> zero(st_dev.size(), 0.0f), zg((size_t) h_v, 0.0f), zb((size_t) h_v, 0.0f);
            /*
            DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
            assuming in the original code the source host memory is pageable
            memory. If the memory is not pageable, call wait() on event return
            by memcpy API to ensure synchronization behavior.
            */
            check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                      d_st, zero.data(), zero.size() * 4).wait()),
                  "z0");
            /*
            DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
            assuming in the original code the source host memory is pageable
            memory. If the memory is not pageable, call wait() on event return
            by memcpy API to ensure synchronization behavior.
            */
            check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                      d_g, zg.data(), zg.size() * 4).wait()),
                  "z1");
            /*
            DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
            assuming in the original code the source host memory is pageable
            memory. If the memory is not pageable, call wait() on event return
            by memcpy API to ensure synchronization behavior.
            */
            check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                      d_b, zb.data(), zb.size() * 4).wait()),
                  "z2");
            strata::kernels::gdn_step(d_st, d_q, d_k, d_v, d_g, d_b, d_o, sh, nullptr);
            std::vector<float> zz(zero.size());
            check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                       .memcpy(zz.data(), d_st, zz.size() * 4)
                                       .wait()),
                  "z3");
            double nz = 0;
            for (float x : zz) nz += std::fabs(x);
            const bool ok = nz == 0.0;
            std::printf("  %-42s %s\n", "gate=beta=0 from a zero state is a no-op", ok ? "yes" : "*** NO ***");
            if (!ok) ++bad;
        }

        // ephemeral state: unchanged when the state is untouched but beta=1 — the delta rule writing k*d
        sycl::free(d_st, dpct::get_in_order_queue());
            sycl::free(d_q, dpct::get_in_order_queue());
            sycl::free(d_k, dpct::get_in_order_queue());
            sycl::free(d_v, dpct::get_in_order_queue());
            sycl::free(d_g, dpct::get_in_order_queue());
            sycl::free(d_b, dpct::get_in_order_queue());
            sycl::free(d_o, dpct::get_in_order_queue());
    }

    // ================= 2. the conv =================
    {
        const int C = 24, dc = 4;      // small, so the host reference is a plain loop
        // TWO DIFFERENT LAYOUTS MEET HERE, and this is where the first version of this test failed at 0.136.
        // `ref/gdn.py` carries the conv state as (d_conv-1, C) ROW-MAJOR - (i, c) at i*C + c - because it is
        // numpy.  The kernel carries it as (C, d_conv-1) with c fastest - c*(dc-1) + i - because that is
        // ggml's `ne = {d_conv-1, C}` with ne0 fast, and it is a runtime buffer so either is legal.  The
        // mapping between them is written out below rather than assumed, in both directions.
        std::vector<float> cs_ref((size_t) (dc - 1) * C), x((size_t) C), kern((size_t) dc * C);
        for (int i = 0; i < dc - 1; ++i)
            for (int c = 0; c < C; ++c) cs_ref[(size_t) (i * C + c)] = (float) (10 * (i + 1) + c);  // labelled
        for (int c = 0; c < C; ++c) x[(size_t) c] = (float) (100 + c);
        for (size_t i = 0; i < kern.size(); ++i) kern[i] = gauss(rng);

        auto to_dev_layout = [&](const std::vector<float>& r) {
            std::vector<float> d(r.size());
            for (int i = 0; i < dc - 1; ++i)
                for (int c = 0; c < C; ++c) d[(size_t) (c * (dc - 1) + i)] = r[(size_t) (i * C + c)];
            return d;
        };
        auto to_ref_layout = [&](const std::vector<float>& d) {
            std::vector<float> r(d.size());
            for (int i = 0; i < dc - 1; ++i)
                for (int c = 0; c < C; ++c) r[(size_t) (i * C + c)] = d[(size_t) (c * (dc - 1) + i)];
            return r;
        };

        std::vector<float> cs = to_dev_layout(cs_ref);
        std::vector<float> want;
        std::vector<float> cs_want_ref = cs_ref;
        ref_conv(cs_want_ref, x, kern, C, dc, want);
        const std::vector<float> cs_want = to_dev_layout(cs_want_ref);

        // THE LAYOUT: the manifest's `ssm_conv1d.weight` has ne = [d_conv, C], so ne0 = d_conv is fast.
        // The kernel takes `kW[c*dc + i]`; the reference's `kern` is (dc, C) row-major.  The transcription
        // below is the ONLY place the two meet.
        std::vector<float> kW((size_t) C * dc);
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < dc; ++i) kW[(size_t) (c * dc + i)] = kern[(size_t) (i * C + c)];
        // and the rival reading, which must differ or the fixture cannot see a transposed weight
        std::vector<float> kW_wrong((size_t) C * dc);
        for (size_t i = 0; i < kern.size(); ++i) kW_wrong[i] = kern[i];
        {
            // apply the WRONG layout by hand: tap i of channel c is kW[c*dc+i] read as kW[i*C+c]
            std::vector<float> cs2 = cs;
            std::vector<float> out_bad((size_t) C, 0.0f);
            for (int c = 0; c < C; ++c) {
                double a = 0;
                for (int i = 0; i < dc - 1; ++i) a += (double) cs2[(size_t) (i * C + c)] * (double) kW_wrong[(size_t) (c * dc + i)];
                a += (double) x[(size_t) c] * (double) kW_wrong[(size_t) (c * dc + dc - 1)];
                out_bad[(size_t) c] = (float) a;
            }
            const double rel = rel_l1(want, out_bad);
            std::printf("\n  %-42s %-4s (%.1f%% apart)\n", "conv weight layout is observable",
                        rel > 0.05 ? "yes" : "*** NO ***", rel * 100);
            if (!(rel > 0.05)) ++bad;
        }

        float *d_cs = nullptr, *d_x = nullptr, *d_w = nullptr, *d_o = nullptr;
        check(DPCT_CHECK_ERROR(d_cs = (float *)sycl::malloc_device(
                                   cs.size() * 4, dpct::get_in_order_queue())),
              "cs");
        check(DPCT_CHECK_ERROR(d_x = (float *)sycl::malloc_device(
                                   x.size() * 4, dpct::get_in_order_queue())),
              "cx");
        check(DPCT_CHECK_ERROR(d_w = (float *)sycl::malloc_device(
                                   kW.size() * 4, dpct::get_in_order_queue())),
              "cw");
        check(DPCT_CHECK_ERROR(d_o = (float *)sycl::malloc_device(
                                   (size_t)C * 4, dpct::get_in_order_queue())),
              "co");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                  d_cs, cs.data(), cs.size() * 4).wait()),
              "ccs");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_x, x.data(),
                                                                 x.size() * 4).wait()),
              "ccx");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                  d_w, kW.data(), kW.size() * 4).wait()),
              "ccw");
        strata::kernels::gdn_conv_step(d_cs, d_x, d_w, d_o, C, dc, nullptr);
        std::vector<float> got((size_t) C), got_cs(cs.size());
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                   .memcpy(got.data(), d_o, got.size() * 4)
                                   .wait()),
              "cgo");
        check(
            DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                 .memcpy(got_cs.data(), d_cs, got_cs.size() * 4)
                                 .wait()),
            "cgs");
        const double rel_out = rel_l1(want, got), rel_cs = rel_l1(cs_want, got_cs);
        std::printf("  %-42s rel %.3e\n", "conv out vs reference", rel_out);
        std::printf("  %-42s rel %.3e\n", "conv state slide (mapped) vs reference", rel_cs);
        if (!(rel_out <= 1e-6)) { std::printf("    *** conv out ***\n"); ++bad; }
        if (!(rel_cs <= 1e-6)) { std::printf("    *** conv state ***\n"); ++bad; }
        sycl::free(d_cs, dpct::get_in_order_queue());
            sycl::free(d_x, dpct::get_in_order_queue());
            sycl::free(d_w, dpct::get_in_order_queue());
            sycl::free(d_o, dpct::get_in_order_queue());
    }

    // ================= 3. l2_norm =================
    {
        const int rows = 16, cols = 128;
        std::vector<float> x((size_t) rows * cols), x_ref;
        for (auto& val : x) val = gauss(rng) * 3.0f;
        x_ref = x;
        const float eps = 1e-6f;
        for (int r = 0; r < rows; ++r) {
            double acc = 0;
            for (int i = 0; i < cols; ++i) acc += (double) x_ref[(size_t) (r * cols + i)] * (double) x_ref[(size_t) (r * cols + i)];
            const float inv = (float) (1.0 / std::sqrt(acc + (double) eps));
            for (int i = 0; i < cols; ++i) x_ref[(size_t) (r * cols + i)] *= inv;
        }
        // The rival reading puts THE SAME eps on the MEAN instead of on the squared norm, which at
        // ||x||^2 ~ 1152 and cols = 128 changes the divisor from sqrt(1152) to sqrt(9) - a factor of 11.3, so
        // the trap is large here.  It would be a much smaller correction on a near-zero row, where the floor
        // is the whole answer; that regime is not what this fixture exercises.
        std::vector<float> x_mean = x;
        for (int r = 0; r < rows; ++r) {
            double acc = 0;
            for (int i = 0; i < cols; ++i) acc += (double) x_mean[(size_t) (r * cols + i)] * (double) x_mean[(size_t) (r * cols + i)];
            const float inv = (float) (1.0 / std::sqrt(acc / cols + (double) eps));
            for (int i = 0; i < cols; ++i) x_mean[(size_t) (r * cols + i)] *= inv;
        }
        const double rel_trap = rel_l1(x_ref, x_mean);
        std::printf("\n  %-42s %-4s (%.1f%% apart)\n", "eps on sum vs on mean is observable",
                    rel_trap > 0.05 ? "yes" : "*** NO ***", rel_trap * 100);
        if (!(rel_trap > 0.05)) ++bad;

        float* d_x = nullptr;
        check(DPCT_CHECK_ERROR(d_x = (float *)sycl::malloc_device(
                                   x.size() * 4, dpct::get_in_order_queue())),
              "lx");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_x, x.data(),
                                                                 x.size() * 4).wait()),
              "lcx");
        strata::kernels::gdn_l2_norm(d_x, rows, cols, eps, nullptr);
        std::vector<float> got(x.size());
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                   .memcpy(got.data(), d_x, got.size() * 4)
                                   .wait()),
              "lcg");
        const double rel = rel_l1(x_ref, got);
        std::printf("  %-42s rel %.3e\n", "l2_norm vs reference", rel);
        if (!(rel <= 1e-6)) ++bad;
        sycl::free(d_x, dpct::get_in_order_queue());
    }

    // ================= 4. the closing norm =================
    {
        const int S2 = 128, hv2 = 8;
        std::vector<float> o((size_t) hv2 * S2), z((size_t) hv2 * S2), sn((size_t) S2), y_ref((size_t) hv2 * S2);
        for (auto& val : o) val = gauss(rng);
        for (auto& val : z) val = gauss(rng);
        for (auto& val : sn) val = 1.0f + 0.1f * gauss(rng);
        const float eps = 1e-6f;
        for (int h = 0; h < hv2; ++h) {
            double acc = 0;
            for (int i = 0; i < S2; ++i) acc += (double) o[(size_t) (h * S2 + i)] * (double) o[(size_t) (h * S2 + i)];
            const float inv = (float) (1.0 / std::sqrt(acc / S2 + (double) eps));
            for (int i = 0; i < S2; ++i)
                y_ref[(size_t) (h * S2 + i)] = o[(size_t) (h * S2 + i)] * inv * sn[(size_t) i] *
                                              sigmoid_f(z[(size_t) (h * S2 + i)]);
        }
        float *d_o = nullptr, *d_z = nullptr, *d_sn = nullptr, *d_y = nullptr;
        check(DPCT_CHECK_ERROR(d_o = (float *)sycl::malloc_device(
                                   o.size() * 4, dpct::get_in_order_queue())),
              "no");
        check(DPCT_CHECK_ERROR(d_z = (float *)sycl::malloc_device(
                                   z.size() * 4, dpct::get_in_order_queue())),
              "nz");
        check(DPCT_CHECK_ERROR(d_sn = (float *)sycl::malloc_device(
                                   sn.size() * 4, dpct::get_in_order_queue())),
              "ns");
        check(
            DPCT_CHECK_ERROR(d_y = (float *)sycl::malloc_device(
                                 y_ref.size() * 4, dpct::get_in_order_queue())),
            "ny");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_o, o.data(),
                                                                 o.size() * 4).wait()),
              "nco");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_z, z.data(),
                                                                 z.size() * 4).wait()),
              "ncz");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                  d_sn, sn.data(), sn.size() * 4).wait()),
              "ncs");
        strata::kernels::gdn_out_norm(d_o, d_z, d_sn, d_y, hv2, S2, eps, nullptr);
        std::vector<float> got(y_ref.size());
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                   .memcpy(got.data(), d_y, got.size() * 4)
                                   .wait()),
              "ncy");
        const double rel = rel_l1(y_ref, got);
        std::printf("\n  %-42s rel %.3e\n", "out norm (rms * ssm_norm * sigmoid z)", rel);
        if (!(rel <= 1e-6)) ++bad;

        // SiLU instead of sigmoid is the qwen3.5 reading; `ref/gdn.py` says this artifact does NOT use it
        std::vector<float> y_silu(y_ref.size());
        for (int h = 0; h < hv2; ++h) {
            double acc = 0;
            for (int i = 0; i < S2; ++i) acc += (double) o[(size_t) (h * S2 + i)] * (double) o[(size_t) (h * S2 + i)];
            const float inv = (float) (1.0 / std::sqrt(acc / S2 + (double) eps));
            for (int i = 0; i < S2; ++i) {
                const float zz = z[(size_t) (h * S2 + i)];
                y_silu[(size_t) (h * S2 + i)] =
                    o[(size_t) (h * S2 + i)] * inv * sn[(size_t) i] * (zz / (1.0f + std::exp(-zz)));
            }
        }
        const double rel_silu = rel_l1(y_ref, y_silu);
        std::printf("  %-42s %-4s (%.2f%% apart)\n", "sigmoid vs SiLU on the gate is observable",
                    rel_silu > 0.05 ? "yes" : "*** NO ***", rel_silu * 100);
        if (!(rel_silu > 0.05)) ++bad;
        sycl::free(d_o, dpct::get_in_order_queue());
            sycl::free(d_z, dpct::get_in_order_queue());
            sycl::free(d_sn, dpct::get_in_order_queue());
            sycl::free(d_y, dpct::get_in_order_queue());
    }

    // ================= 5. the verify-window kernels, T = 1..8, bitwise against the single-token kernels =================
    // gdn_ab_multi and gdn_step_norm_multi are documented as bitwise `fused_gdn_ab` / `fused_gdn_step_norm` per
    // token (verify_kernels.hpp). Every T from 1 to kVerifyMaxT is its own exact-T instantiation, so each is run,
    // and every output and the state are compared with memcmp (not a tolerance).
    {
        const int n_embd = 2560, qk = S * h_k, C = 2 * qk + S * h_v, vd = S * h_v;
        const float eps = 1e-6f;
        auto bf16_of = [](float f) {
            uint32_t u;
            std::memcpy(&u, &f, 4);
            return (uint16_t) (u >> 16);
        };
        std::vector<uint16_t> wa((size_t) h_v * n_embd), wb((size_t) h_v * n_embd);
        for (auto& w : wa) w = bf16_of(gauss(rng) * 0.05f);
        for (auto& w : wb) w = bf16_of(gauss(rng) * 0.05f);
        std::vector<float> dt((size_t) h_v), ssm_a((size_t) h_v), gamma((size_t) S);
        for (auto& x : dt) x = gauss(rng);
        for (auto& x : ssm_a) x = -(0.1f + (float) (rng() % 100) / 50.0f);
        for (auto& x : gamma) x = 1.0f + 0.1f * gauss(rng);
        std::vector<float> st0((size_t) S * h_v * S);
        for (auto& x : st0) x = gauss(rng) * 0.1f;

        auto dev = [&](size_t bytes) {
            try {
        void *p = nullptr;
            check(DPCT_CHECK_ERROR(p = (void *)sycl::malloc_device(
                                       bytes, dpct::get_in_order_queue())),
                  "m5");
            return p;
        }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        auto up = [&](const void *h, size_t bytes) {
            try {
        void *p = dev(bytes);
            /*
            DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
            assuming in the original code the source host memory is pageable
            memory. If the memory is not pageable, call wait() on event return
            by memcpy API to ensure synchronization behavior.
            */
            check(DPCT_CHECK_ERROR(
                      (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(p, h, bytes).wait()),
                  "u5");
            return p;
        }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        auto down = [&](std::vector<float> &h, const void *d) {
            try {
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                   .memcpy(h.data(), d, h.size() * 4)
                                   .wait()),
              "d5");
        }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        auto d_wa = (uint16_t*) up(wa.data(), wa.size() * 2);
        auto d_wb = (uint16_t*) up(wb.data(), wb.size() * 2);
        auto d_dt = (float*) up(dt.data(), dt.size() * 4);
        auto d_sa = (float*) up(ssm_a.data(), ssm_a.size() * 4);
        auto d_gm = (float*) up(gamma.data(), gamma.size() * 4);
        int ab_bad = 0, sn_bad = 0, ab_n = 0, sn_n = 0;
        for (int T = 1; T <= strata::kernels::kVerifyMaxT; ++T) {
            // --- alpha/beta
            std::vector<float> x((size_t) T * n_embd);
            for (auto& v : x) v = gauss(rng);
            auto d_x = (float*) up(x.data(), x.size() * 4);
            auto d_g = (float*) dev((size_t) T * h_v * 4), d_b = (float*) dev((size_t) T * h_v * 4);
            auto d_gr = (float*) dev((size_t) T * h_v * 4), d_br = (float*) dev((size_t) T * h_v * 4);
            strata::kernels::gdn_ab_multi(d_x, d_wa, d_wb, d_dt, d_sa, d_g, d_b, n_embd, h_v, T, nullptr);
            for (int t = 0; t < T; ++t)
                strata::kernels::fused_gdn_ab(d_x + (size_t) t * n_embd, d_wa, d_wb, d_dt, d_sa, d_gr + (size_t) t * h_v,
                                              d_br + (size_t) t * h_v, n_embd, h_v, nullptr);
            std::vector<float> g((size_t) T * h_v), b(g.size()), gr(g.size()), br(g.size());
            down(g, d_g); down(b, d_b); down(gr, d_gr); down(br, d_br);
            ++ab_n;
            if (std::memcmp(g.data(), gr.data(), g.size() * 4) != 0 || std::memcmp(b.data(), br.data(), b.size() * 4) != 0) {
                std::printf("    *** gdn_ab_multi T=%d differs from fused_gdn_ab (memcmp) ***\n", T);
                ++ab_bad;
            }
            sycl::free(d_x, dpct::get_in_order_queue());
                sycl::free(d_g, dpct::get_in_order_queue());
                sycl::free(d_b, dpct::get_in_order_queue());
                sycl::free(d_gr, dpct::get_in_order_queue());
                sycl::free(d_br, dpct::get_in_order_queue());

            // --- the recurrence + output norm
            std::vector<float> h((size_t) T * C), gate((size_t) T * h_v), beta((size_t) T * h_v), z((size_t) T * vd);
            for (auto& v : h) v = gauss(rng) * 0.1f;
            for (auto& v : gate) v = -(0.5f + 3.0f * (float) (rng() % 100) / 100.0f);
            for (auto& v : beta) v = (float) (rng() % 100) / 100.0f;
            for (auto& v : z) v = gauss(rng);
            auto d_h = (float*) up(h.data(), h.size() * 4);
            auto d_gate = (float*) up(gate.data(), gate.size() * 4);
            auto d_beta = (float*) up(beta.data(), beta.size() * 4);
            auto d_z = (float*) up(z.data(), z.size() * 4);
            auto d_y = (float*) dev((size_t) T * vd * 4);
            auto d_yr = (float*) dev((size_t) T * vd * 4);
            auto d_st = (float*) dev(st0.size() * 4);
            auto d_sr = (float*) dev(st0.size() * 4);
            int32_t* d_nk = (int32_t*) dev(4);
            // the reference chain (the single-token kernel), n tokens, into d_sr / d_yr
            auto run_ref = [&](int n) {
                /*
                DPCT1114: cudaMemcpy is migrated to asynchronization
                memcpy, assuming in the original code the source host memory is
                pageable memory. If the memory is not pageable, call wait() on
                event return by memcpy API to ensure synchronization behavior.
                */
                try {
            check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                      d_sr, st0.data(), st0.size() * 4).wait()),
                  "r0");
                for (int t = 0; t < n; ++t)
                    strata::kernels::fused_gdn_step_norm(d_sr, d_h + (size_t) t * C, d_h + (size_t) t * C + qk,
                                                         d_h + (size_t) t * C + 2 * qk, d_gate + (size_t) t * h_v,
                                                         d_beta + (size_t) t * h_v, d_z + (size_t) t * vd, d_gm, eps,
                                                         d_yr + (size_t) t * vd, h_k, h_v, nullptr);
            }
            catch (sycl::exception const &exc) {
              std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                        << ", line:" << __LINE__ << std::endl;
              std::exit(1);
            }
            };
            std::vector<float> y((size_t) T * vd), yr(y.size()), st(st0.size()), sr(st0.size());
            // (name, n_keep tokens or -1 = verify half, t_out_begin)
            struct Case { const char* name; int n_keep; int tb; };
            std::vector<Case> cases = {{"verify", -1, 0}, {"verify tb", -1, T > 1 ? T / 2 : 0}, {"commit", T, 0},
                                       {"commit tb", T, T / 2}, {"commit state only", T, T}};
            if (T > 1) cases.push_back({"commit n-1", T - 1, T - 1});
            for (const Case& c : cases) {
                /*
                DPCT1114: cudaMemcpy is migrated to asynchronization
                memcpy, assuming in the original code the source host memory is
                pageable memory. If the memory is not pageable, call wait() on
                event return by memcpy API to ensure synchronization behavior.
                */
                check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                          d_st, st0.data(), st0.size() * 4).wait()),
                      "s0");
                check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                           .memset(d_y, 0, (size_t)T * vd * 4)
                                           .wait()),
                      "y0");
                check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                           .memset(d_yr, 0, (size_t)T * vd * 4)
                                           .wait()),
                      "yr0");
                const int n = c.n_keep < 0 ? T : c.n_keep;
                if (c.n_keep >= 0)
                    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                               .memcpy(d_nk, &c.n_keep, 4)
                                               .wait()),
                          "nk");
                strata::kernels::gdn_step_norm_multi(d_st, d_h, C, d_gate, d_beta, d_z, d_gm, eps, d_y, h_k, h_v, T,
                                                     c.n_keep >= 0 ? d_nk : nullptr, nullptr, c.tb);
                run_ref(n);
                down(y, d_y); down(yr, d_yr); down(st, d_st); down(sr, d_sr);
                bool ok = true;
                size_t y_bad = 0, y_all = 0;
                for (int t = c.tb; t < n; ++t) {
                    for (int i = 0; i < vd; ++i, ++y_all)
                        if (std::memcmp(&y[(size_t) t * vd + i], &yr[(size_t) t * vd + i], 4) != 0) ++y_bad;
                }
                if (y_bad) ok = false;
                // the verify half leaves the state untouched; the commit half writes the state after n tokens
                const std::vector<float>& want_st = c.n_keep < 0 ? st0 : sr;
                if (std::memcmp(st.data(), want_st.data(), st.size() * 4) != 0) ok = false;
                ++sn_n;
                // #3948 (STRATA_QFUSE): the same call with a q8_1 destination also writes the outputs' q8_1 image, row t - tb
                // at (t - tb) * value_dim: the bytes native_quantize_q8_1 makes of y rows [tb, n), the outputs and the state
                // unchanged by asking for it, and nothing written past those rows (the engine's second group has tb > 0)
                if (false && c.tb < n) {   // SYCL port: STRATA_QFUSE is not built, the q8_1 destination is never asked for
                    if (std::getenv("STRATA_GDN_PARITY_TRACE")) std::printf("    trace: T=%d '%s' with a q8_1 destination\n", T, c.name);
                    const size_t qb = (size_t) (vd / 32) * 36, rows = (size_t) (n - c.tb);
                    void *d_q = dev((size_t) T * qb), *d_qr = dev((size_t) T * qb);
                    check(
                        DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                             .memset(d_q, 0x5a, (size_t)T * qb)
                                             .wait()),
                        "q0");
                    /*
                    DPCT1114: cudaMemcpy is migrated to asynchronization
                    memcpy, assuming in the original code the source host memory
                    is pageable memory. If the memory is not pageable, call
                    wait() on event return by memcpy API to ensure
                    synchronization behavior.
                    */
                    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                              d_st, st0.data(), st0.size() * 4).wait()),
                          "s1");
                    std::vector<float> y2(y.size()), st2(st0.size());
                    check(
                        DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                             .memset(d_y, 0, (size_t)T * vd * 4)
                                             .wait()),
                        "y1");
                    strata::kernels::gdn_step_norm_multi(d_st, d_h, C, d_gate, d_beta, d_z, d_gm, eps, d_y, h_k, h_v, T,
                                                         c.n_keep >= 0 ? d_nk : nullptr, nullptr, c.tb, d_q);
                    static dpct::queue_ptr qstream =
                        &dpct::get_in_order_queue(); // (native_quantize_q8_1
                                                     // refuses the null stream;
                                                     // a blocking stream orders
                                                     // with the legacy one)
                    if (qstream == &dpct::get_in_order_queue()) check(
                        DPCT_CHECK_ERROR(
                            qstream =
                                dpct::get_current_device().create_queue(true)),
                        "qstream");
                    strata::kernels::native_quantize_q8_1(d_y + (size_t) c.tb * vd, d_qr, vd, (int) rows, qstream);
                    check(
                        DPCT_CHECK_ERROR(
                            dpct::get_current_device().queues_wait_and_throw()),
                        "q sync");
                    std::vector<uint8_t> hq((size_t) T * qb), hr((size_t) T * qb);
                    check(
                        DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                             .memcpy(hq.data(), d_q, hq.size())
                                             .wait()),
                        "q read");
                    check(
                        DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                             .memcpy(hr.data(), d_qr, rows * qb)
                                             .wait()),
                        "qr read");
                    down(y2, d_y); down(st2, d_st);
                    bool q_ok = std::memcmp(hq.data(), hr.data(), rows * qb) == 0;
                    for (size_t i = rows * qb; q_ok && i < hq.size(); ++i) q_ok = hq[i] == 0x5a;
                    for (int t = c.tb; q_ok && t < n; ++t)
                        q_ok = std::memcmp(&y2[(size_t) t * vd], &y[(size_t) t * vd], (size_t) vd * 4) == 0;
                    q_ok = q_ok && std::memcmp(st2.data(), st.data(), st.size() * 4) == 0;
                    if (!q_ok) {
                        std::printf("    *** gdn_step_norm_multi T=%d '%s' with a q8_1 destination: bytes differ from native_quantize_q8_1 or outputs/state changed ***\n", T, c.name);
                        ++sn_bad;
                    }
                    ++sn_n;
                    sycl::free(d_q, dpct::get_in_order_queue());
                        sycl::free(d_qr, dpct::get_in_order_queue());
                }
                if (!ok) {
                    std::printf("    *** gdn_step_norm_multi T=%d '%s' differs from fused_gdn_step_norm (memcmp): %zu of %zu y words%s ***\n",
                                T, c.name, y_bad, y_all,
                                std::memcmp(st.data(), (c.n_keep < 0 ? st0 : sr).data(), st.size() * 4) != 0 ? ", state too" : "");
                    ++sn_bad;
                }
                if (std::getenv("STRATA_GDN_VERIFY_BENCH")) {
                    auto hash = [](const std::vector<float>& values) {
                        uint64_t h = 1469598103934665603ull;
                        for (float v : values) {
                            uint32_t u;
                            std::memcpy(&u, &v, sizeof(u));
                            h = (h ^ u) * 1099511628211ull;
                        }
                        return h;
                    };
                    std::printf("GDNSIGNATURE T=%d case=%s y=%016llx state=%016llx\n", T, c.name,
                                (unsigned long long) hash(y), (unsigned long long) hash(st));
                    if (c.n_keep < 0 && c.tb == 0) {
                        auto* q = &dpct::get_in_order_queue();
                        constexpr int calls = 32;
                        dpct::experimental::command_graph_ptr graph;
                        dpct::experimental::begin_recording(q);
                        for (int i = 0; i < calls; ++i)
                            strata::kernels::gdn_step_norm_multi(
                                d_st, d_h, C, d_gate, d_beta, d_z, d_gm, eps,
                                d_y, h_k, h_v, T, nullptr, q, 0);
                        dpct::experimental::end_recording(q, &graph);
                        auto executable = graph->finalize();
                        delete graph;
                        std::vector<double> times;
                        for (int i = 0; i < 20; ++i) {
                            auto begin = q->ext_oneapi_submit_barrier();
                            q->ext_oneapi_graph(executable);
                            auto end = q->ext_oneapi_submit_barrier();
                            end.wait_and_throw();
                            if (i >= 5)
                                times.push_back(double(
                                    end.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                    begin.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000 / calls);
                        }
                        std::sort(times.begin(), times.end());
                        std::printf("GDNVERIFYBENCH T=%d us=%.3f\n", T, times[times.size() / 2]);
                    }
                }
            }
            sycl::free(d_h, dpct::get_in_order_queue());
                sycl::free(d_gate, dpct::get_in_order_queue());
                sycl::free(d_beta, dpct::get_in_order_queue());
                sycl::free(d_z, dpct::get_in_order_queue());
                sycl::free(d_y, dpct::get_in_order_queue());
                sycl::free(d_yr, dpct::get_in_order_queue());
                sycl::free(d_st, dpct::get_in_order_queue());
            sycl::free(d_sr, dpct::get_in_order_queue());
                sycl::free(d_nk, dpct::get_in_order_queue());
        }
        sycl::free(d_wa, dpct::get_in_order_queue());
            sycl::free(d_wb, dpct::get_in_order_queue());
            sycl::free(d_dt, dpct::get_in_order_queue());
            sycl::free(d_sa, dpct::get_in_order_queue());
            sycl::free(d_gm, dpct::get_in_order_queue());
        std::printf("\n  %-42s %s (%d cases)\n", "gdn_ab_multi T=1..8 == fused_gdn_ab", ab_bad ? "*** NO ***" : "bitwise", ab_n);
        std::printf("  %-42s %s (%d cases)\n", "gdn_step_norm_multi T=1..8 == single-token", sn_bad ? "*** NO ***" : "bitwise", sn_n);
        bad += ab_bad + sn_bad;
    }

    // ================= 6. the layer-batched commit (engine-latency), bitwise against the per-layer calls =================
    // gdn_commit_layers runs every layer's conv-history commit and state commit in one launch each; the per-layer loop
    // (gdn_conv_commit + gdn_step_norm_multi with n_keep, no outputs) is the reference.  Several layers with their own
    // state, rows and gate/beta; n_keep 1..T; memcmp on the whole state arena (states and conv histories).
    {
        const int qk = S * h_k, C = 2 * qk + S * h_v, L = 5, MT = strata::kernels::kVerifyMaxT;
        const int64_t conv_off = (int64_t) S * h_v * S, stride = conv_off + (int64_t) C * 3;
        std::vector<float> st0((size_t) (stride * L)), hb((size_t) L * MT * C), qkv((size_t) L * MT * C),
            gt((size_t) L * MT * h_v), bt((size_t) L * MT * h_v), zdummy((size_t) MT * S * h_v), gm(S, 1.0f);
        for (auto& x : st0) x = gauss(rng) * 0.1f;
        for (auto& x : hb) x = gauss(rng) * 0.1f;
        for (auto& x : qkv) x = gauss(rng);
        for (auto& x : gt) x = -(0.5f + (float) (rng() % 100) / 50.0f);
        for (auto& x : bt) x = (float) (rng() % 100) / 100.0f;
        float *d_a = nullptr, *d_b = nullptr, *d_h = nullptr, *d_qkv = nullptr, *d_g = nullptr, *d_bt = nullptr,
              *d_z = nullptr, *d_y = nullptr, *d_gm = nullptr;
        int32_t* d_nk = nullptr;
        check(DPCT_CHECK_ERROR(d_a = (float *)sycl::malloc_device(
                                   st0.size() * 4, dpct::get_in_order_queue())),
              "la");
            check(DPCT_CHECK_ERROR(
                      d_b = (float *)sycl::malloc_device(
                          st0.size() * 4, dpct::get_in_order_queue())),
                  "lb");
        check(DPCT_CHECK_ERROR(d_h = (float *)sycl::malloc_device(
                                   hb.size() * 4, dpct::get_in_order_queue())),
              "lh");
            check(DPCT_CHECK_ERROR(
                      d_qkv = (float *)sycl::malloc_device(
                          qkv.size() * 4, dpct::get_in_order_queue())),
                  "lq");
        check(DPCT_CHECK_ERROR(d_g = (float *)sycl::malloc_device(
                                   gt.size() * 4, dpct::get_in_order_queue())),
              "lg");
            check(DPCT_CHECK_ERROR(
                      d_bt = (float *)sycl::malloc_device(
                          bt.size() * 4, dpct::get_in_order_queue())),
                  "lbt");
        check(DPCT_CHECK_ERROR(
                  d_z = (float *)sycl::malloc_device(
                      zdummy.size() * 4, dpct::get_in_order_queue())),
              "lz");
            check(DPCT_CHECK_ERROR(
                      d_y = (float *)sycl::malloc_device(
                          zdummy.size() * 4, dpct::get_in_order_queue())),
                  "ly");
        check(DPCT_CHECK_ERROR(d_gm = (float *)sycl::malloc_device(
                                   gm.size() * 4, dpct::get_in_order_queue())),
              "lgm");
            check(DPCT_CHECK_ERROR(d_nk = (int32_t *)sycl::malloc_device(
                                       4, dpct::get_in_order_queue())),
                  "lnk");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                  d_h, hb.data(), hb.size() * 4).wait()),
              "lch");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                  d_qkv, qkv.data(), qkv.size() * 4).wait()),
              "lcq");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                  d_g, gt.data(), gt.size() * 4).wait()),
              "lcg");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                  d_bt, bt.data(), bt.size() * 4).wait()),
              "lcb");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                  d_gm, gm.data(), gm.size() * 4).wait()),
              "lcm");
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                   .memset(d_z, 0, zdummy.size() * 4)
                                   .wait()),
              "lz0");
        int lb_bad = 0, lb_n = 0;
        bool launched = true;
        for (int nk = 1; nk <= MT && launched; ++nk) {
            check(DPCT_CHECK_ERROR(
                      (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_nk, &nk, 4).wait()),
                  "lnk1");
            /*
            DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
            assuming in the original code the source host memory is pageable
            memory. If the memory is not pageable, call wait() on event return
            by memcpy API to ensure synchronization behavior.
            */
            check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                      d_a, st0.data(), st0.size() * 4).wait()),
                  "lca");
            /*
            DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
            assuming in the original code the source host memory is pageable
            memory. If the memory is not pageable, call wait() on event return
            by memcpy API to ensure synchronization behavior.
            */
            check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                      d_b, st0.data(), st0.size() * 4).wait()),
                  "lcb2");
            for (int l = 0; l < L; ++l) {   // the reference: the per-layer calls
                float* stl = d_a + (size_t) l * stride;
                strata::kernels::gdn_conv_commit(stl + conv_off, d_qkv + (size_t) l * MT * C, C, d_nk, nullptr);
                strata::kernels::gdn_step_norm_multi(stl, d_h + (size_t) l * MT * C, C, d_g + (size_t) l * MT * h_v,
                                                     d_bt + (size_t) l * MT * h_v, d_z, d_gm, 1e-6f, d_y, h_k, h_v, MT, d_nk,
                                                     nullptr, MT);
            }
            launched = strata::kernels::gdn_commit_layers(d_b, stride, conv_off, d_qkv, (int64_t) MT * C, d_h, (int64_t) MT * C,
                                                          d_g, d_bt, (int64_t) MT * h_v, C, h_k, h_v, MT, L, d_nk, nullptr);
            if (!launched) break;
            check(DPCT_CHECK_ERROR(
                      dpct::get_current_device().queues_wait_and_throw()),
                  "lsync");
            std::vector<float> a(st0.size()), b(st0.size());
            check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                       .memcpy(a.data(), d_a, a.size() * 4)
                                       .wait()),
                  "lda");
            check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                       .memcpy(b.data(), d_b, b.size() * 4)
                                       .wait()),
                  "ldb");
            ++lb_n;
            if (std::memcmp(a.data(), b.data(), a.size() * 4) != 0 || std::memcmp(a.data(), st0.data(), a.size() * 4) == 0) {
                std::printf("    *** gdn_commit_layers n_keep=%d differs from the per-layer commit (or wrote nothing) ***\n", nk);
                ++lb_bad;
            }
        }
        sycl::free(d_a, dpct::get_in_order_queue());
            sycl::free(d_b, dpct::get_in_order_queue());
            sycl::free(d_h, dpct::get_in_order_queue());
            sycl::free(d_qkv, dpct::get_in_order_queue());
            sycl::free(d_g, dpct::get_in_order_queue());
            sycl::free(d_bt, dpct::get_in_order_queue());
            sycl::free(d_z, dpct::get_in_order_queue());
        sycl::free(d_y, dpct::get_in_order_queue());
            sycl::free(d_gm, dpct::get_in_order_queue());
            sycl::free(d_nk, dpct::get_in_order_queue());
        if (launched)
            std::printf("  %-42s %s (%d cases, %d layers)\n", "gdn_commit_layers == per-layer commit", lb_bad ? "*** NO ***" : "bitwise",
                        lb_n, L);
        else
            std::printf("  %-42s skipped (the per-layer commit is not the state-only kernel here)\n", "gdn_commit_layers");
        bad += lb_bad;
    }

    std::printf("\ngdn: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("gdn_parity OK\n");
    return 0;
}
