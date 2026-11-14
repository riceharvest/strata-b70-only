// dflash_tree_test.cpp - host-only test of q36_dtree.hpp (no GPU): the tree built from the drafter's candidates, its DFS layout and the
// checkpoint slots.  The GDN verify is simulated with a hash as the "state": the state a row reads must be the state after its parent,
// with the kernel's rules (ld loads slot (ld - 1) & 1 before the row, sv saves the state after the row into slot (sv - 1) & 1; two slots).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

#include "q36_dtree.hpp"
#include "q36_darb.hpp"

using namespace strata::q36dt;

static int g_bad = 0, g_n = 0;
static void check(bool ok, const char* what, const std::string& detail = "") {
    ++g_n;
    if (!ok) { ++g_bad; std::printf("FAIL: %s %s\n", what, detail.c_str()); }
}

static uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

static uint64_t mix(uint64_t s, uint64_t t) {
    s ^= t + 0x9E3779B97F4A7C15ULL + (s << 6) + (s >> 2);
    s *= 0xff51afd7ed558ccdULL;
    s ^= s >> 33;
    return s;
}

// a random gather with the selector's walk filled in (as the device does it: pick the best candidate after the previous pick)
static void random_gather(Gather& g, std::mt19937_64& rng, float sharp) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::memset(&g, 0, sizeof g);
    g.anchor = 1000 + (int) (rng() % 1000);
    for (int l = 0; l < kDepth; ++l) {
        for (int c = 0; c < kTopK; ++c) {
            g.cand[l][c] = 5000 + l * 100 + c * 3 + (int) (rng() % 3);   // distinct within a row
            g.unary[l][c] = sharp * nd(rng);
        }
        for (int t = 0; t < kRank; ++t) g.hp[l][t] = nd(rng) * 0.25f;
    }
    for (int r = 0; r < kPredRows; ++r) for (int t = 0; t < kRank; ++t) g.pred[r][t] = to_bf16(nd(rng) * 0.25f);
    for (int l = 0; l < kDepth; ++l) for (int c = 0; c < kTopK; ++c) for (int t = 0; t < kRank; ++t) g.succ[l][c][t] = to_bf16(nd(rng) * 0.25f);
    int row = 0;
    for (int l = 0; l < kDepth; ++l) {
        float s[kTopK];
        scores(g, l, row, s);
        int b = 0;
        for (int c = 1; c < kTopK; ++c) if (s[c] > s[b]) b = c;
        g.out_tok[l] = g.cand[l][b];
        row = 1 + l * kTopK + b;
    }
}

// the state a row must read: the state after its parent along the path of tokens
static uint64_t path_state(const std::vector<Node>& n, int u) {
    if (u < 0) return 0x1234;
    return mix(path_state(n, n[(size_t) u].parent), (uint64_t) n[(size_t) u].tok);
}

static bool simulate(const std::vector<Node>& n, const Layout& L, std::string& why) {
    uint64_t fork[2] = {0, 0}, cur = 0x1234;
    for (int t = 0; t < L.T; ++t) {
        const int ld = (int) (L.tslot[t] & 3u), sv = (int) ((L.tslot[t] >> 2) & 3u);
        if (ld > 0) cur = fork[(ld - 1) & 1];
        const int u = L.src[t];
        if (cur != path_state(n, n[(size_t) u].parent)) { why = "row " + std::to_string(t) + " reads the wrong state"; return false; }
        cur = mix(cur, (uint64_t) n[(size_t) u].tok);
        if (sv > 0) fork[(sv - 1) & 1] = cur;
    }
    return true;
}

static bool structure(const std::vector<Node>& n, const Layout& L, std::string& why) {
    if (!L.ok || L.T != (int) n.size()) { why = "layout not ok"; return false; }
    std::vector<int> seen((size_t) L.T, 0);
    for (int t = 0; t < L.T; ++t) {
        const int u = L.src[t];
        if (u < 0 || u >= L.T || seen[(size_t) u]++) { why = "src is not a permutation"; return false; }
        if (L.tok[t] != n[(size_t) u].tok) { why = "token"; return false; }
        if (L.dep[t] != n[(size_t) u].depth) { why = "depth"; return false; }
        if (t == 0 && u != 0) { why = "row 0 is not the root"; return false; }
        if (t > 0 && (L.parent[t] < 0 || L.parent[t] >= t)) { why = "parent after child"; return false; }
        if (t > 0 && L.src[L.parent[t]] != n[(size_t) u].parent) { why = "parent mismatch"; return false; }
        // ancestors
        std::vector<int> anc;
        for (int r = t; r >= 0; r = L.parent[r]) anc.push_back(r);
        uint32_t mk = 0;
        for (int a : anc) mk |= 1u << a;
        if (mk != L.tmask[t]) { why = "tmask"; return false; }
        const int d = (int) anc.size() - 1;   // anc[0] = the row, anc[k] = k-th ancestor
        auto at = [&](int k) { return k <= d ? anc[(size_t) k] : -(k - d); };   // the k-th ancestor; beyond the root: history -1, -2, -3
        const int p1 = at(1), p2 = at(2), p3 = at(3);
        const uint32_t want = (uint32_t) (p3 + 3) | ((uint32_t) (p2 + 3) << 5) | ((uint32_t) (p1 + 3) << 10);
        if (want != L.tprev3[t]) { why = "tprev3 row " + std::to_string(t); return false; }
    }
    return true;
}

static void test_shapes() {
    auto mk = [](std::initializer_list<int> parents) {
        std::vector<Node> n;
        int i = 0;
        for (int p : parents) {
            Node x;
            x.tok = 100 + i;
            x.parent = p;
            x.depth = p < 0 ? 0 : n[(size_t) p].depth + 1;
            n.push_back(x);
            ++i;
        }
        return n;
    };
    check(slot_need(mk({-1, 0, 1, 2})) == 0, "chain needs 0 slots");
    check(slot_need(mk({-1, 0, 0})) == 1, "a fork of two leaves needs 1");
    check(slot_need(mk({-1, 0, 0, 0})) == 1, "a fork of three leaves needs 1");
    // root -> {A, B}; A -> {A1, A2}; A1 -> {a, b}; B -> B1 -> B2 : the big subtree goes last, so 1 slot is enough
    check(slot_need(mk({-1, 0, 0, 1, 1, 3, 3, 2, 7})) == 1, "the deepest subtree last (need 1)");
    // the perfect binary tree of 15 nodes: need 3
    auto perfect = mk({-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6});
    check(slot_need(perfect) == 3, "perfect binary tree of 15 nodes needs 3");
    check(!layout(perfect).ok, "layout refuses a tree that needs 3 slots");
    // 7 nodes binary: need 2
    auto p7 = mk({-1, 0, 0, 1, 1, 2, 2});
    check(slot_need(p7) == 2, "perfect binary tree of 7 nodes needs 2");
    Layout L = layout(p7);
    std::string why;
    check(structure(p7, L, why) && simulate(p7, L, why), "7-node binary tree layout", why);
    check(L.slots <= 2, "7-node binary tree: at most 2 slots live");
}

static void test_random_layouts() {
    std::mt19937_64 rng(12345);
    int tested = 0, refused = 0, with_fork = 0;
    for (int it = 0; it < 20000; ++it) {
        const int T = 1 + (int) (rng() % kRows);
        std::vector<Node> n;
        Node root;
        root.tok = 7;
        root.parent = -1;
        n.push_back(root);
        for (int i = 1; i < T; ++i) {
            Node x;
            // bias to deeper shapes half of the time
            x.parent = (rng() & 1) ? i - 1 : (int) (rng() % (uint64_t) i);
            x.depth = n[(size_t) x.parent].depth + 1;
            x.tok = 100 + i;
            n.push_back(x);
        }
        Layout L = layout(n);
        if (slot_need(n) > kMaxSlots) { ++refused; check(!L.ok, "layout must refuse need > 2"); continue; }
        ++tested;
        std::string why;
        const bool s1 = structure(n, L, why);
        check(s1, "structure", why);
        std::string why2;
        check(s1 && simulate(n, L, why2), "GDN state machine reads the parent's state", why2);
        check(L.slots <= kMaxSlots, "live slots <= 2");
        if (L.slots > 0) ++with_fork;
    }
    std::printf("  random layouts: %d valid (%d with a fork), %d refused (need > 2)\n", tested, with_fork, refused);
    check(tested > 1000 && with_fork > 100, "the random test saw valid and forked trees");   // (need 3 takes the perfect 15-node tree: test_shapes)
}

static void test_build() {
    std::mt19937_64 rng(777);
    int forks = 0, trees = 0;
    for (int it = 0; it < 300; ++it) {
        Gather* g = new Gather;
        random_gather(*g, rng, it % 3 == 0 ? 0.5f : 2.0f);
        Params p;
        p.max_rows = 2 + (int) (rng() % 15);
        p.max_depth = 1 + (int) (rng() % kDepth);
        p.spine = p.max_depth;
        p.min_lp = it % 2 ? -3.9f : -12.f;
        const auto n = build(*g, p);
        const int D = std::min({p.spine, p.max_depth, p.max_rows - 1});
        check((int) n.size() <= p.max_rows && (int) n.size() >= 1 + D, "node budget respected, spine kept", "rows " + std::to_string(n.size()));
        bool ok = n[0].tok == g->anchor && n[0].parent == -1;
        for (int i = 1; i <= D && ok; ++i) ok = n[(size_t) i].tok == g->out_tok[i - 1] && n[(size_t) i].parent == i - 1 && n[(size_t) i].spine;
        check(ok, "nodes 1 .. D are the selector's chain");
        float last = 1e9f;
        for (size_t i = 1; i < n.size(); ++i) {
            const Node& x = n[i];
            ok = ok && x.parent >= 0 && x.parent < (int) i && x.depth == n[(size_t) x.parent].depth + 1 && x.depth <= p.max_depth && x.cum <= n[(size_t) x.parent].cum + 1e-4f;
            if (!x.spine) { ok = ok && x.cum <= last + 1e-4f; last = x.cum; ok = ok && x.cum >= p.min_lp - 1e-4f; }
        }
        check(ok, "tree invariants: parents first, depth, cumulative log prob falls, best first, floor");
        // distinct tokens among siblings
        for (size_t i = 1; i < n.size(); ++i) for (size_t j = i + 1; j < n.size(); ++j)
            if (n[i].parent == n[j].parent && n[i].tok == n[j].tok) ok = false;
        check(ok, "siblings are distinct");
        const Layout L = layout(n);
        std::string why, why2;
        check(structure(n, L, why) && simulate(n, L, why2), "built tree: layout + GDN state machine", why + why2);
        if (L.slots > 0) ++forks;
        ++trees;
        if (p.max_rows == 1 + D) check((int) n.size() == 1 + D, "a budget of D + 1 rows is the chain");
        // greedy accept: the target picks along the chain -> all D accepted; picks that leave the tree -> 0
        int32_t pick[kRows];
        int path[kRows];
        for (int t = 0; t < L.T; ++t) pick[t] = -1;
        // find the chain's rows
        int cur = 0;
        int want_a = 0;
        for (int d = 0; d < D; ++d) {
            int hit = -1;
            for (int i = 0; i < L.nchild[cur]; ++i) if (L.tok[L.child[cur][i]] == g->out_tok[d]) hit = L.child[cur][i];
            if (hit < 0) break;
            pick[cur] = g->out_tok[d];
            cur = hit;
            ++want_a;
        }
        pick[cur] = 424242;
        const int a = walk_greedy(L, pick, 7, path);
        check(a == want_a && a == D, "greedy walk follows the picks along the chain", std::to_string(a) + " vs " + std::to_string(D));
        delete g;
    }
    std::printf("  built trees: %d (%d with checkpoint slots)\n", trees, forks);
    check(forks > 30, "the built trees use forks");
}

static void test_mismatch_ok() {
    // a pending child below the floor is not added: a very sharp selector gives the chain only
    std::mt19937_64 rng(5);
    Gather* g = new Gather;
    random_gather(*g, rng, 30.f);
    Params p;
    p.max_rows = 16;
    p.min_lp = -3.9f;
    const auto n = build(*g, p);
    check(n.size() >= 8, "sharp selector: the chain at least");
    delete g;
}

// ---- the throughput arbiter (q36_darb.hpp): a simulated request, no card.  Arm 0 / 1 give (tokens a step ~ mean m, noise) in a fixed step time.
static double geo_p(double mean) {   // E[t] = (1 - p^8) / (1 - p) for t clamped to 1..8: solve for p by bisection
    double lo = 0.0, hi = 0.9999;
    for (int i = 0; i < 60; ++i) {
        const double p = 0.5 * (lo + hi);
        const double m = (1.0 - std::pow(p, 8)) / (1.0 - p);
        (m < mean ? lo : hi) = p;
    }
    return 0.5 * (lo + hi);
}
struct SimArm { double mean_tok, ms; };
struct SimOut { long long steps[2]; double tokens, ms; strata::q36da::Arb* arb; };
static SimOut simulate(const strata::q36da::ArbCfg& cfg, SimArm a0, SimArm a1, int n, std::mt19937_64& rng, int switch_at = -1, SimArm b0 = {0, 0}, SimArm b1 = {0, 0}, double graph_ms = 0, double flush_ms = 0) {
    static strata::q36da::Arb* keep = nullptr;
    delete keep;
    keep = new strata::q36da::Arb(cfg);
    SimOut o{{0, 0}, 0, 0, keep};
    SimArm arms[2] = {a0, a1};
    std::uniform_real_distribution<double> u(0.0, 1.0);
    bool seen[2] = {false, false};
    bool mtp_since_df = false;
    for (int i = 0; i < n; ++i) {
        if (i == switch_at) { arms[0] = b0; arms[1] = b1; }
        const int arm = keep->pick();
        // tokens: 1 + geometric-ish accepted count with the arm's mean (clamped 1..8)
        const double p = geo_p(arms[arm].mean_tok);   // success probability of "one more accepted" (the mean of the clamped 1..8 count is mean_tok)
        int t = 1;
        while (t < 8 && u(rng) < p) ++t;
        double ms = arms[arm].ms * (0.97 + 0.06 * u(rng));
        bool valid = true;
        if (!seen[arm]) { seen[arm] = true; ms += graph_ms; valid = graph_ms == 0; }   // the first step of an arm records a graph
        if (arm == 0 && mtp_since_df && flush_ms > 0) { ms += flush_ms; valid = false; }   // the drafter's first step after MTP steps flushes the rows it missed (the engine says so: not a clean sample)
        mtp_since_df = arm == 1 ? true : false;
        keep->note(arm, t, ms, valid);
        ++o.steps[arm];
        o.tokens += t;
        o.ms += ms;
    }
    return o;
}

static void test_arbiter() {
    using strata::q36da::Arb;
    using strata::q36da::ArbCfg;
    std::mt19937_64 rng(7);
    ArbCfg c;
    c.on = true;
    // off: always the drafter, nothing changes
    {
        ArbCfg off;
        Arb a(off);
        for (int i = 0; i < 100; ++i) { check(a.pick() == 0, "arb off picks the drafter"); a.note(0, 1, 100.0, true); }
        check(a.flips() == 0 && a.probes() == 0, "arb off: no flips, no probes");
    }
    // 1-3. mean over 100 simulated requests of 700 steps: the throughput against the better arm alone and against staying on the drafter.  The token counts are a clamped
    //      geometric draw (1 .. 8 a step, as a verify step's 1 + accepted drafts), so a few steps of an arm are a noisy reading: single requests can pick the wrong arm,
    //      the mean is the claim.
    struct Case { const char* name; SimArm d, m; int best; double need; };
    const Case cases[] = {
        {"drafter +30%", {6.0, 38.0}, {4.5, 36.0}, 0, 0.96},
        {"drafter +15%", {5.0, 38.0}, {4.4, 36.0}, 0, 0.95},
        {"equal", {4.0, 40.0}, {4.0, 40.0}, 0, 0.97},
        {"MTP +15%", {3.3, 50.0}, {3.7, 48.0}, 1, 0.94},
        {"MTP +30%", {3.3, 50.0}, {3.8, 44.0}, 1, 0.94},
    };
    for (const Case& k : cases) {
        double frac = 0, fixed = 0;
        long long probes = 0, other = 0;
        for (int rep = 0; rep < 100; ++rep) {
            const SimOut o = simulate(c, k.d, k.m, 700, rng);
            const double best = std::max(k.d.mean_tok / k.d.ms, k.m.mean_tok / k.m.ms);
            frac += (o.tokens / o.ms) / best;
            fixed += (k.d.mean_tok / k.d.ms) / best;
            other += o.steps[1 - k.best];
            probes += o.arb->probes();
        }
        std::printf("  arbiter %-13s: %.3f of the better arm (staying on the drafter: %.3f), %.1f%% of the steps on the worse arm\n", k.name, frac / 100, fixed / 100,
                    100.0 * (double) other / 70000.0);
        check(frac / 100 >= k.need, (std::string("arbiter ") + k.name + ": mean throughput near the better arm").c_str(), std::to_string(frac / 100));
        if (std::string(k.name).find("30%") != std::string::npos) check((double) other / 70000.0 <= 0.10, (std::string("arbiter ") + k.name + ": at most 10% of the steps on the worse arm").c_str(), std::to_string((double) other / 70000.0));
        if (k.best == 1) check(frac / 100 > fixed / 100 + 0.05, (std::string("arbiter ") + k.name + ": beats staying on the drafter").c_str());
    }
    // 4. the regime changes at step 300 (the drafter's acceptance falls: code -> prose): the arbiter follows
    for (int rep = 0; rep < 20; ++rep) {
        const SimOut o = simulate(c, {6.0, 38.0}, {4.5, 36.0}, 1200, rng, 300, {2.8, 48.0}, {4.0, 36.0});
        check(o.arb->chosen() == 1, "regime change: ends on MTP");
        check(o.arb->flips() >= 1, "regime change: flipped");
    }
    // 5. graph recording steps (first step of an arm +60 ms) are not samples: the estimates are not poisoned
    for (int rep = 0; rep < 20; ++rep) {
        const SimOut o = simulate(c, {6.0, 38.0}, {4.5, 36.0}, 400, rng, -1, {0, 0}, {0, 0}, 60.0);
        check(o.arb->invalid() == 2, "graph steps are invalid samples", std::to_string(o.arb->invalid()));
        check(o.arb->chosen() == 0, "graph steps do not flip a better drafter");
        check(std::fabs(o.arb->ms_step(1) - 36.0) < 4.0, "the MTP estimate is the real step time, not the graph's", std::to_string(o.arb->ms_step(1)));
    }
    // 5b. the drafter's catch-up flush after MTP steps (20 ms on the first drafter step after a probe or a stretch of MTP steps) is not a sample: a better drafter stays chosen
    {
        double frac = 0;
        int on_df = 0;
        for (int rep = 0; rep < 100; ++rep) {
            const SimOut o = simulate(c, {6.0, 38.0}, {4.5, 36.0}, 700, rng, -1, {0, 0}, {0, 0}, 0, 20.0);
            frac += (o.tokens / o.ms) / (6.0 / 38.0);
            on_df += o.arb->chosen() == 0;
        }
        std::printf("  arbiter drafter +30%% with a 20 ms catch-up flush after MTP steps: %.3f of the better arm, ends on the drafter in %d of 100\n", frac / 100, on_df);
        check(frac / 100 > 0.93 && on_df >= 90, "arbiter: the drafter's catch-up flush does not push a better drafter off");
    }
    // 6. a probe that never gets a clean step is given up after 3 tries, the request goes on on its arm
    {
        Arb a(c);
        for (int i = 0; i < c.init; ++i) { check(a.pick() == 0, "arb: drafter first"); a.note(0, 5, 38.0, true); }
        check(a.pick() == 1, "arb: first probe after init steps");
        for (int i = 0; i < 3; ++i) a.note(1, 4, 200.0, false);
        check(a.pick() == 0, "arb: a probe of 3 invalid steps is given up");
    }
    // 7. a step that the drafter declined (the arm taken is not the one picked) is a sample of the arm it ran
    {
        Arb a(c);
        a.note(1, 3, 40.0, true);
        check(a.samples(1) == 1 && a.samples(0) == 0, "arb: a declined drafter step counts for the MTP arm");
    }
    // 8. the probe cost on a long run on the better arm falls with the back-off: steps on the other arm over 3000 steps
    {
        const SimOut o = simulate(c, {6.0, 38.0}, {3.0, 36.0}, 3000, rng);
        check(o.steps[1] < 3000 / 16 / 2, "back-off: probes well under 1 in 32 on a long run", std::to_string(o.steps[1]));
    }
}

int main() {
    test_shapes();
    test_random_layouts();
    test_build();
    test_mismatch_ok();
    test_arbiter();
    std::printf("dflash_tree_test: %s (%d checks, %d failed)\n", g_bad ? "FAIL" : "OK", g_n, g_bad);
    return g_bad ? 1 : 0;
}
