// src/spec/draft_policy_test.cpp - DraftPolicy: when does a lookup window beat the MTP's?
//
// Simulated rounds with the costs measured on the RTX 5070 (window-cost: ~10 ms more per token) check that
//   1. with no lookup proposal the MTP window is kept;
//   2. lookup drafts that are mostly rejected stop being taken (their bucket's rate falls);
//   3. lookup drafts that are always accepted are taken, and the window grows with them;
//   4. match-length buckets learn separately (short matches failing does not stop long ones);
//   5. the policy never proposes a window beyond its cap.
#include "strata/spec/draft_policy.hpp"

#include <cstdint>
#include <cstdio>

using strata::spec::DraftPolicy;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
double cost(int t) { return 19.0 + 10.5 * (t - 1); }   // ms per round, measured shape
}  // namespace

int main() {
    std::printf("draft_policy_test\n");
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 2, 0, cost(4));   // MTP windows of 4: 3 tokens each
        const DraftPolicy::Pick k = p.choose(4, 0, 0);
        check(!k.lookup && k.t == 4, "no proposal: the MTP window");
    }
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 2, 0, cost(4));
        for (int t = 2; t <= 6; ++t) p.observe(false, t, 0, 0, cost(t));
        for (int i = 0; i < 40; ++i) p.observe(true, 6, 0, 4, cost(6));      // short matches, all rejected
        check(p.lookup_rate(4) < 0.15, "rejected short-match drafts: their rate falls below 0.15");
        check(!p.choose(4, 5, 4).lookup, "rejected short-match drafts: no longer taken");
        for (int i = 0; i < 40; ++i) p.observe(true, 6, 5, 30, cost(6));     // long matches, all accepted
        check(p.lookup_rate(30) > 0.9, "accepted long-match drafts: their rate rises above 0.9");
        const DraftPolicy::Pick k = p.choose(4, 5, 30);
        check(k.lookup && k.t == 6, "accepted long matches: the full lookup window is taken");
        check(!p.choose(4, 5, 4).lookup, "buckets are separate: short matches still not taken");
        check(p.choose(4, 20, 30).t <= 6, "never beyond the window cap");
    }
    {
        DraftPolicy p(8);
        for (int i = 0; i < 50; ++i) p.observe(false, 3, 2, 0, cost(3));      // a very good MTP: 3 of 3 tokens
        for (int t = 2; t <= 8; ++t) p.observe(false, t, t - 1, 0, cost(t));
        for (int i = 0; i < 40; ++i) p.observe(true, 4, 2, 8, cost(4));       // lookup at q ~ 0.67
        check(!p.choose(3, 7, 8).lookup, "a mediocre lookup does not replace a strong MTP window");
    }
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 3, 0, cost(4));      // a near-perfect MTP, only size 4 seen
        const DraftPolicy::Pick k = p.choose(4, 5, 40);
        check(k.lookup && k.t == 6, "an unmeasured size is probed for a confident lookup");
        for (int i = 0; i < 3; ++i) p.observe(true, 6, 5, 40, 3.0 * cost(6));   // it turns out very expensive
        check(!p.choose(4, 5, 40).lookup, "after the probes, the measured cost decides");
    }
    {
        // #1252: a wide size priced high in one slow stretch is measured again after a while
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 3, 0, cost(4));      // a strong MTP
        for (int t = 2; t <= 6; ++t) for (int i = 0; i < 4; ++i) p.observe(false, t, t - 1, 0, cost(t));
        for (int i = 0; i < 6; ++i) p.observe(true, 6, 5, 40, 4.0 * cost(6));  // the wide size, in a slow stretch
        for (int i = 0; i < 30; ++i) p.observe(true, 4, 3, 40, cost(4));       // and the short window's rate is fine
        bool saw6 = false;
        for (int i = 0; i < 400 && !saw6; ++i) {
            const DraftPolicy::Pick k = p.choose(4, 5, 40);
            if (k.lookup && k.t == 6) { saw6 = true; p.observe(true, 6, 5, 40, cost(6)); }
            else p.observe(k.lookup, k.t, k.t - 1, 40, cost(k.t));
        }
        check(saw6, "a stale wide size is tried again within a few hundred rounds");
    }
    {
        // --lookup-chain: rows cheap (UMA-like) and the chain always accepted -> chained; rows dear or the chain
        // always rejected -> not
        auto cheap = [](int t) { return 40.0 + 2.0 * (t - 1); };
        DraftPolicy p(8);
        for (int t = 2; t <= 8; ++t)
            for (int i = 0; i < 5; ++i) p.observe(false, t, 1, 0, cheap(t));
        for (int i = 0; i < 30; ++i) p.observe(false, 4, 2, 0, cheap(4));
        for (int i = 0; i < 30; ++i) p.observe_chain(4, 3, 6, 12, cheap(7));    // every chained token accepted
        check(p.chain(4, 0.9, 3, 12) == 3, "chain: cheap rows, always accepted -> all 3 chained");
        DraftPolicy q(8);
        for (int t = 2; t <= 8; ++t)
            for (int i = 0; i < 5; ++i) q.observe(false, t, 1, 0, cost(t));
        for (int i = 0; i < 30; ++i) q.observe_chain(4, 3, 3, 4, cost(7));       // reached, never accepted
        check(q.chain(4, 0.9, 3, 4) == 0, "chain: dear rows, never accepted -> none");
        check(q.chain(4, 0.9, 0, 4) == 0 && q.chain(8, 1.0, 3, 40) == 0, "chain: nothing proposed / no room -> none");
    }
    {
        // The stale-size check must also run when a shorter chain already pays. Synthetic costs isolate the
        // selection policy: even a size that stays expensive is retried, but only once per stale interval.
        DraftPolicy p(6);
        for (int i = 0; i < 3; ++i) p.observe(false, 4, 3, 0, 40.0);
        for (int i = 0; i < 3; ++i) p.observe_chain(4, 1, 4, 40, 42.0);
        for (int i = 0; i < 3; ++i) p.observe_chain(4, 2, 5, 40, 1000.0);
        check(p.chain(4, 1.0, 2, 40) == 1, "chain: the profitable shorter chain initially wins");
        int probes = 0, last_probe = -1;
        bool spaced = true, confident = true;
        for (int i = 0; i < 1000; ++i) {
            const int k = p.chain(4, 1.0, 2, 40);
            if (k == 2) {
                ++probes;
                spaced &= last_probe < 0 ? i >= 300 : i - last_probe > 300;
                last_probe = i;
                confident &= p.chain(4, 0.5, 2, 40) < 2;
            }
            p.observe_chain(4, k, 3 + k, 40, k == 2 ? 1000.0 : 42.0);
        }
        check(probes > 0 && probes <= 3 && spaced,
              "chain: stale full size is re-probed despite a profitable shorter chain");
        check(probes > 0 && confident, "chain: stale re-probes retain the MTP confidence gate");
    }
    {
        // Gated proposals: within one generation the policy copy learns from committed outcomes only. A size
        // whose cost is still guessed is probed until kProbes rounds of it have run, timed or not, not every round.
        DraftPolicy live(6);
        for (int i = 0; i < 50; ++i) live.observe(false, 4, 3, 0, cost(4));      // a strong MTP
        live.observe(true, 6, 5, 8, 3.0 * cost(6));                             // size 6 timed once, dear
        DraftPolicy frozen = live, copy = live;
        int frozen_probes = 0, copy_probes = 0;
        for (int i = 0; i < 20; ++i) {
            const DraftPolicy::Pick f = frozen.choose(4, 5, 8);
            frozen_probes += f.lookup && f.t == 6;
            const DraftPolicy::Pick c = copy.choose(4, 5, 8);
            copy_probes += c.lookup && c.t == 6;
            copy.observe_committed(c.lookup, c.lookup ? c.t : 4, c.lookup ? 5 : 3, 8);
        }
        check(frozen_probes == 20, "gated: a frozen copy probes the guessed size every round");
        check(copy_probes == 2, "gated: committed outcomes end the probes after kProbes rounds of the size");
        check(copy.cost_ms(6) == live.cost_ms(6), "gated: committed outcomes leave the measured costs alone");
    }
    {
        // Gated proposals: a round's time reaches the generation's copy only once the committed position has passed
        // every counter it depends on. Cheap costs from earlier text stop steering long lookups once the dear
        // repetitive-text times are applied; a held time is not a new probe.
        DraftPolicy live(6);
        for (int t = 2; t <= 6; ++t) for (int i = 0; i < 5; ++i) live.observe(false, t, 1, 0, cost(t) * 0.2);   // cheap stretch
        DraftPolicy copy = live;
        check(copy.choose(4, 5, 40).lookup && copy.choose(4, 5, 40).t == 6, "delayed: cheap inherited costs take the full lookup");
        const double before = copy.cost_ms(6);
        copy.observe_committed(true, 6, 5, 40);
        copy.defer_cost(110, 6, 40.0 * cost(6));
        copy.apply_costs(110);
        check(copy.cost_ms(6) == before, "delayed: a time is held while its counters can still be drawn");
        copy.apply_costs(111);
        check(copy.cost_ms(6) > before, "delayed: applied once the committed position passes its last counter");
        for (int i = 0; i < 40; ++i) {
            copy.observe_committed(true, 6, 5, 40);
            copy.defer_cost(120 + i, 6, 40.0 * cost(6));
            copy.apply_costs(121 + i);
        }
        check(copy.choose(4, 5, 40).t < 6, "delayed: dear measured times shorten the lookup window");
        DraftPolicy fresh(6);
        fresh.observe_committed(true, 6, 5, 40);
        fresh.defer_cost(5, 6, 30.0);
        fresh.apply_costs(6);
        fresh.observe_committed(true, 6, 5, 40);
        check(fresh.choose(4, 5, 40).lookup && fresh.choose(4, 5, 40).t == 6,
              "delayed: a held time does not count as another probe");
    }
    {
        // PROBE-MIN (STRATA_SPEC_PROBE_MIN=1).  One B70's measured round times (ms by rows) and drafts that hold 92% of the
        // time: 5 or 6 rows give the most tokens per ms (143-144 tok/s), 4 rows 136, 3 rows 120.  The first use of a size in a
        // context bucket is ~20 ms slower.  The policy has to end up on 5-6 rows with every price close to the real one.
        static const double real[9] = {0, 18.1, 20.6, 23.4, 26.4, 30.0, 34.3, 42.1, 49.7};
        struct Sim {
            DraftPolicy pol;
            bool used[3][9] = {};
            int bucket = 0;
            uint32_t rng = 12345;
            double extra[9] = {};          // ms added to every round of a size (a slow stretch)
            explicit Sim(int cap) : pol(cap) {}
            int round(int cap) {
                DraftPolicy::ChainState st;
                float pr[8] = {};
                for (int idx = 0; idx < cap - 1; ++idx) {
                    pr[idx] = 0.92f;
                    if (!pol.keep_draft(st, idx, pr[idx], cap)) break;
                }
                const int t = st.best_rows;
                int acc = 0;
                for (; acc < t - 1; ++acc) {
                    rng = rng * 1664525u + 1013904223u;
                    if ((rng >> 8) % 100 >= 92) break;
                }
                double ms = real[t] + extra[t];
                if (!used[bucket][t]) { used[bucket][t] = true; ms += 20.0; }
                pol.observe_probs(pr, t - 1, acc);
                pol.observe(false, t, acc, 0, ms);
                return t;
            }
        };
        auto run = [](Sim& s, int rounds, int cap) {
            double sum = 0;
            for (int i = 0; i < rounds; ++i) sum += s.round(cap);
            return sum / rounds;
        };
        for (const int cap : {4, 6, 8}) {
            Sim s(cap);
            s.pol.set_robust(true);
            s.pol.set_probe_min(true);
            s.pol.set_context(1000);
            run(s, 400, cap);
            s.pol.set_context(8000);   // a new bucket: priors from the first, and every size's first use is slow again
            s.bucket = 1;
            run(s, 400, cap);
            const double rows = run(s, 200, cap);
            const int best = cap < 5 ? cap : 5;   // 5 and 6 rows are within 1% of each other
            char what[96];
            std::snprintf(what, sizeof what, "probe-min, cap %d: settles on the best size (%.2f rows)", cap, rows);
            check(rows >= best - 0.25 && rows <= 6.25, what);
            bool close = true;
            for (int t = 2; t <= cap; ++t) close = close && s.pol.cost_ms(t) > 0.9 * real[t] && s.pol.cost_ms(t) < 1.1 * real[t];
            std::snprintf(what, sizeof what, "probe-min, cap %d: every size's price is within 10%% of the real one", cap);
            check(close, what);
        }
        {
            // a slow stretch prices 5 and 6 rows out; once it is over, the stale sizes are measured again and come back
            Sim s(6);
            s.pol.set_robust(true);
            s.pol.set_probe_min(true);
            s.pol.set_context(8000);
            s.bucket = 1;
            s.extra[5] = s.extra[6] = 15.0;
            run(s, 200, 6);
            check(s.pol.cost_ms(5) > 1.3 * real[5], "probe-min: a slow stretch raises a size's price");
            s.extra[5] = s.extra[6] = 0.0;
            run(s, 1500, 6);
            check(s.pol.cost_ms(5) < 1.1 * real[5] && s.pol.cost_ms(6) < 1.1 * real[6],
                  "probe-min: a stale price is measured again and recovers");
            check(run(s, 200, 6) >= 4.75, "probe-min: the window is back on 5-6 rows after the slow stretch");
            Sim never(6);
            never.pol.set_robust(true);
            never.pol.set_probe_min(true);
            never.pol.set_stale_rounds(0);
            never.pol.set_context(8000);
            never.bucket = 1;
            never.extra[5] = never.extra[6] = 15.0;
            run(never, 200, 6);
            never.extra[5] = never.extra[6] = 0.0;
            run(never, 1500, 6);
            check(never.pol.cost_ms(6) > 1.2 * real[6], "probe-min: stale rounds 0 never measures a priced-out size again");
        }
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
