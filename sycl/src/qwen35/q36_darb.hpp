// q36_darb.hpp - the throughput arbiter of the DFlash serve loop (STRATA_Q36_DFLASH_ARB=1, default off).  Host only, no SYCL.
//
// Two arms: 0 = a drafter step (draft + verify, a tree too), 1 = a lookup / MTP step.  Every step is a sample (tokens = accepted drafts + 1,
// ms = host clock around the step; the step already waits once for its picks, so no sync is added).  An arm's rate is EWMA(tokens) / EWMA(ms).
// The request runs on the arm with the higher rate; the other arm is probed with one step in `period` steps (doubled after a probe that does not
// flip the choice, up to `max_period`; reset when it does), so a request where MTP is faster ends on MTP for almost every step and pays the probes.
//   * a sample of a step that recorded a graph (once, ~60 ms) is invalid: not counted, and a probe whose step was invalid is repeated (3 times at most);
//   * a flip needs the other rate above the chosen one by `margin` and `dwell` steps since the last flip;
//   * the chosen arm's samples weigh max(1/16, 1/(k + 2)) (k = steps since it became chosen: the stale value of a probe is replaced fast), a probe's weigh 0.4.
// The first look at the other arm is `init` (6) steps in a row; the choice flips only on an arm with 3+ samples; a probe that finds the other arm 15% or more slower doubles the period.
// Read once from the environment: STRATA_Q36_DFLASH_ARB (0/1), _ARB_PERIOD (16), _ARB_MAX_PERIOD (256), _ARB_MARGIN (0.05), _ARB_INIT (6), _ARB_DWELL (8).
#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::q36da {

struct ArbCfg {
    bool on = false;
    int period = 16, max_period = 256, init = 6, dwell = 8;
    double margin = 0.05;
    static ArbCfg from_env() {
        auto num = [](const char* k, double d) { const char* v = std::getenv(k); return v != nullptr && v[0] != '\0' ? std::atof(v) : d; };
        ArbCfg c;
        c.on = num("STRATA_Q36_DFLASH_ARB", 0.0) != 0.0;
        c.period = std::max(2, (int) num("STRATA_Q36_DFLASH_ARB_PERIOD", 16));
        c.max_period = std::max(c.period, (int) num("STRATA_Q36_DFLASH_ARB_MAX_PERIOD", 256));
        c.init = std::max(1, (int) num("STRATA_Q36_DFLASH_ARB_INIT", 6));
        c.dwell = std::max(1, (int) num("STRATA_Q36_DFLASH_ARB_DWELL", 8));
        c.margin = std::max(0.0, num("STRATA_Q36_DFLASH_ARB_MARGIN", 0.05));
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

    /// A step on `arm` is done: `toks` tokens in `ms` milliseconds; `valid` false = not a sample (it recorded a graph).
    void note(int arm, int toks, double ms, bool valid) {
        ++total_[arm];
        if (!c_.on) return;
        const bool other = arm != cur_;   // a step of the arm that is not the chosen one is a probe (also when the drafter declined and the step took the other path)
        if (!valid) {
            ++invalid_;
            if (probing_ && other && ++probe_bad_ >= 3) end_probe(false);   // (give up: a probe that never gets a clean step must not hold the request)
            return;
        }
        Est& e = est_[arm];
        if (!other) {
            const double a = std::max(1.0 / 16.0, 1.0 / (double) (k_ + 2));
            ++k_;
            e.add(toks, ms, a);
            ++since_flip_;
            ++since_probe_;
            // the chosen arm's rate can fall under the other arm's old value (checked with the dwell and the margin)
            maybe_flip();
            if (!probing_ && since_probe_ >= (est_[1 - cur_].n < kMinN ? c_.init : period_)) {
                probing_ = true;
                probe_bad_ = 0;
                probe_got_ = 0;
            }
            return;
        }
        e.add(toks, ms, std::max(0.15, 1.0 / (double) (e.n + 1)));   // (a running mean for the first probes, then ~6 probes' memory)
        ++probes_;
        if (++probe_got_ >= probe_len()) end_probe(true);
    }

    double rate(int arm) const { return est_[arm].rate(); }
    double ms_step(int arm) const { return est_[arm].n > 0 ? est_[arm].ms : 0.0; }
    int samples(int arm) const { return est_[arm].n; }
    long long steps(int arm) const { return total_[arm]; }
    long long probes() const { return probes_; }
    long long flips() const { return flips_; }
    long long invalid() const { return invalid_; }
    int period() const { return period_; }

private:
    struct Est {
        double tok = 0, ms = 0;
        int n = 0;
        void add(int t, double m, double a) {
            if (n == 0) { tok = (double) t; ms = m; }
            else { tok += a * ((double) t - tok); ms += a * (m - ms); }
            ++n;
        }
        double rate() const { return n > 0 && ms > 0 ? tok / ms : 0.0; }
    };

    static constexpr int kMinN = 3;   // samples an arm needs before its rate can flip the choice
    /// Steps of a probe: the first look at the other arm takes `init` steps in a row (one step of an arm is a poor reading), later ones one.
    int probe_len() const { return est_[1 - cur_].n < kMinN ? c_.init : 1; }
    void end_probe(bool got) {
        probing_ = false;
        since_probe_ = 0;
        if (!got) return;
        const int before = cur_;
        // the probe's verdict: flip when the other arm is clearly faster (the dwell does not apply to a probe); back off only when it is clearly slower
        // (a close or noisy reading keeps the period: one step of an arm is 1 .. 8 tokens)
        if (est_[1 - cur_].n >= kMinN && est_[1 - cur_].rate() > est_[cur_].rate() * (1.0 + c_.margin)) flip();
        if (cur_ == before && est_[1 - cur_].n >= kMinN && est_[1 - cur_].rate() < est_[cur_].rate() * 0.85) period_ = std::min(period_ * 2, c_.max_period);
    }
    void maybe_flip() {
        if (since_flip_ < c_.dwell || est_[1 - cur_].n < kMinN) return;
        if (est_[1 - cur_].rate() > est_[cur_].rate() * (1.0 + c_.margin)) flip();
    }
    void flip() {
        cur_ = 1 - cur_;
        ++flips_;
        k_ = 0;
        since_flip_ = 0;
        since_probe_ = 0;
        period_ = c_.period;
        probing_ = false;
    }

    ArbCfg c_;
    Est est_[2];
    int cur_ = kDf, period_;
    int k_ = 0, since_flip_ = 0, since_probe_ = 0, probe_bad_ = 0, probe_got_ = 0;
    bool probing_ = false;
    long long total_[2] = {0, 0}, probes_ = 0, flips_ = 0, invalid_ = 0;
};

}   // namespace strata::q36da
