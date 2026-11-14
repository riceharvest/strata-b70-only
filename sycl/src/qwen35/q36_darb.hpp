// q36_darb.hpp - the throughput arbiter of the DFlash serve loop (STRATA_Q36_DFLASH_ARB=1, default off).  Host only, no SYCL.
//
// Two arms: 0 = a drafter step (draft + verify, a tree too), 1 = a lookup / MTP step.  Every step is a sample: tokens = accepted drafts + 1 (1 .. 8, a noisy
// reading: ~55% relative standard deviation on the measured 26K histogram) and ms = the host clock around the step (the step already waits once for its picks:
// no sync is added; it barely varies).  An arm's rate is mean(tokens) / mean(ms) over a discounted window (weight 0.995 per step, half-life ~140 steps), with the
// standard error of the mean tokens (weighted Welford).
//   * The request runs on the chosen arm; the other arm is probed.  While the comparison is INCONCLUSIVE (|rate difference| < z = 1 standard error) the arms
//     alternate (every 2nd step is a probe, bounded by a budget of 35% of the steps + 6), because one step is a poor reading.  Once the other arm is confidently
//     slower (-z standard errors, z = 1) the probe period starts at `period` (16 steps) and doubles after every probe that confirms it, up to `max_period` (256): a
//     request where one arm is clearly faster ends on it for almost every step and pays the probes.  The choice flips when the other arm is faster by z
//     standard errors AND by `margin`.  Arms that are equal stay inconclusive and alternate, which costs nothing; arms 5% apart cost at most half of that.
//   * The first look at the other arm is `init` (6) steps in a row.
//   * A step that is not a clean sample (it recorded a graph once, ~60 ms; the drafter flushed context rows it had not seen: the caller says) is `valid = false`:
//     counted as a step, not as a sample; a probe whose step was not clean is repeated (3 times at most).
// Read once from the environment: STRATA_Q36_DFLASH_ARB (0/1), _ARB_PERIOD (16), _ARB_MAX_PERIOD (256), _ARB_MARGIN (0.05), _ARB_INIT (6), _ARB_Z (1.0).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::q36da {

struct ArbCfg {
    bool on = false;
    int period = 16, max_period = 256, init = 6;
    double margin = 0.05, z = 1.0;
    static ArbCfg from_env() {
        auto num = [](const char* k, double d) { const char* v = std::getenv(k); return v != nullptr && v[0] != '\0' ? std::atof(v) : d; };
        ArbCfg c;
        c.on = num("STRATA_Q36_DFLASH_ARB", 0.0) != 0.0;
        c.period = std::max(2, (int) num("STRATA_Q36_DFLASH_ARB_PERIOD", 16));
        c.max_period = std::max(c.period, (int) num("STRATA_Q36_DFLASH_ARB_MAX_PERIOD", 256));
        c.init = std::max(1, (int) num("STRATA_Q36_DFLASH_ARB_INIT", 6));
        c.margin = std::max(0.0, num("STRATA_Q36_DFLASH_ARB_MARGIN", 0.05));
        c.z = std::max(0.0, num("STRATA_Q36_DFLASH_ARB_Z", 1.0));
        return c;
    }
};

class Arb {
public:
    static constexpr int kDf = 0, kMtp = 1;

    explicit Arb(const ArbCfg& c) : c_(c), period_(c.period) {}
    static const ArbCfg& env_cfg() {
        static const ArbCfg c = ArbCfg::from_env();
        return c;
    }
    Arb() : Arb(env_cfg()) {}

    bool on() const { return c_.on; }
    int chosen() const { return cur_; }

    /// The arm of the next step.
    int pick() const { return probing_ ? 1 - cur_ : cur_; }

    /// A step on `arm` is done: `toks` tokens in `ms` milliseconds; `valid` false = not a clean sample.
    void note(int arm, int toks, double ms, bool valid) {
        ++total_[arm];
        ++g_;
        if (!c_.on) return;
        const bool other = arm != cur_;
        if (!valid) {
            ++invalid_;
            if (probing_ && other && ++probe_bad_ >= 3) end_probe(false);   // (give up: a probe that never gets a clean step must not hold the request)
            return;
        }
        est_[arm].add(toks, ms, g_);
        if (other && !probing_) return;   // (an unplanned step of the other arm, e.g. the drafter declined: a sample, nothing else)
        if (!other) {
            ++since_probe_;
            if (!probing_) {
                const bool first = est_[1 - cur_].n(g_) < kMinN;
                if (since_probe_ >= (first ? c_.init : next_gap()) && (first || budget_ok())) {
                    probing_ = true;
                    probe_bad_ = 0;
                    probe_got_ = 0;
                }
            }
            return;
        }
        ++probes_;
        if (++probe_got_ >= (est_[1 - cur_].n(g_) < kMinN ? c_.init : 1)) end_probe(true);
    }

    double rate(int arm) const { return est_[arm].rate(); }
    double ms_step(int arm) const { return est_[arm].ms(); }
    int samples(int arm) const { return est_[arm].count; }
    long long steps(int arm) const { return total_[arm]; }
    long long probes() const { return probes_; }
    long long flips() const { return flips_; }
    long long invalid() const { return invalid_; }
    int period() const { return period_; }

private:
    static constexpr int kMinN = 3;           // samples an arm needs before its rate counts
    static constexpr double kDecay = 0.995;   // per step: half-life ~140 steps

    struct Est {
        double w = 0, mt = 0, m2 = 0, mm = 0;   // weights, weighted mean tokens, weighted sum of squares, weighted mean ms
        long long last = 0;
        int count = 0;
        void decay(long long g) {
            if (g > last) { const double f = std::pow(kDecay, (double) (g - last)); w *= f; m2 *= f; last = g; }
        }
        void add(int t, double ms, long long g) {
            decay(g);
            ++count;
            const double nw = w + 1.0, d = (double) t - mt;
            mt += d / nw;
            m2 += d * ((double) t - mt);
            mm += (ms - mm) / nw;
            w = nw;
        }
        double n(long long g) const { return w * std::pow(kDecay, (double) std::max<long long>(0, g - last)); }
        double rate() const { return count > 0 && mm > 0 ? mt / mm : 0.0; }
        double ms() const { return count > 0 ? mm : 0.0; }
        /// standard error of the rate: sd(tokens) / sqrt(n) / mean ms (a floor on the variance for a few samples)
        double se(long long g) const {
            const double nn = std::max(1.0, n(g));
            const double var = w > 1.0 ? m2 / (w - 1.0) : 4.0;
            return std::sqrt(std::max(var, 0.04)) / std::sqrt(nn) / std::max(mm, 1e-9);
        }
    };

    double diff() const { return est_[1 - cur_].rate() - est_[cur_].rate(); }
    double se_diff() const {
        const double a = est_[1 - cur_].se(g_), b = est_[cur_].se(g_);
        return std::sqrt(a * a + b * b);
    }
    bool budget_ok() const { return (double) probes_ <= 0.35 * (double) g_ + 6.0; }
    /// steps of the chosen arm between two probes: 1 while the comparison is inconclusive, the (doubling) period once the other arm is confidently slower
    int next_gap() const {
        const double z = se_diff() > 0 ? diff() / se_diff() : 0.0;
        return std::fabs(z) < c_.z ? 1 : period_;
    }
    void end_probe(bool got) {
        probing_ = false;
        since_probe_ = 0;
        if (!got || est_[1 - cur_].n(g_) < kMinN) return;
        const double d = diff(), s = se_diff();
        const double z = s > 0 ? d / s : 0.0;
        if (z > c_.z && d > est_[cur_].rate() * c_.margin) { flip(); return; }
        if (z < -c_.z) period_ = std::min(period_ * 2, c_.max_period);   // confidently slower: probe less
        else period_ = c_.period;                                         // (inconclusive: next_gap() is 1 for now)
    }
    void flip() {
        cur_ = 1 - cur_;
        ++flips_;
        since_probe_ = 0;
        period_ = c_.period;
        probing_ = false;
    }

    ArbCfg c_;
    Est est_[2];
    int cur_ = kDf, period_;
    int since_probe_ = 0, probe_bad_ = 0, probe_got_ = 0;
    bool probing_ = false;
    long long total_[2] = {0, 0}, probes_ = 0, flips_ = 0, invalid_ = 0, g_ = 0;
};

}   // namespace strata::q36da
