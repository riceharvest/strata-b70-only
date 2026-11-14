// tests/core/q36_spec_sampling_test.cpp - strata-q36's drafting for requests the device argmax cannot serve (a penalty, a
// temperature above 0): the host side, no GPU, no model.  Built by tools/test_q36_spec_sampling.py (g++).
//
//   1. the keys the kernels order logits by (float -> key, lowest id on a tie);
//   2. the kernels' two-stage top-M, run on the host as the kernels write it, against a plain sort;
//   3. a window row's token (raw top-M, penalties on the host) against the plain sampler on the whole row: the same token and
//      the same random draw, with the penalized tokens at the top, ties at the candidate cap, a cap of 64 and a window of 64;
//   4. a whole answer through windows against the plain one-token-a-step loop: the same tokens for the same seed, with end of
//      answer and max_new inside a window, drafts that are right, wrong and outside the candidates;
//   5. the accept / reject rule reproduces the target distribution (statistical; a fixed seed; a biased rule is caught);
//   6. the same, against the point-mass case of the Flash-Next engine's rule (core/spec_prob.hpp);
//   7. the plain sampler is what it was (the old code, copied here) except on exact ties;
//   8. which requests draft (spec_plan).
#include "q36_sampling.hpp"
#include "strata/core/spec_prob.hpp"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace strata::qwen35;

namespace {
int g_fail = 0, g_checks = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        std::printf("  FAIL %s\n", what.c_str());
    }
}

uint64_t splitmix(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
double unit(uint64_t& x) { return (double) (splitmix(x) >> 11) * (1.0 / 9007199254740992.0); }

// ------------------------------------------------------------------------------------------------ 1. the keys
void test_keys() {
    std::printf("keys\n");
    uint64_t st = 1;
    std::vector<float> v = {0.f, -0.f, 1.f, -1.f, INFINITY, -INFINITY, NAN, FLT_MIN, -FLT_MIN, FLT_MAX, -FLT_MAX, 1e-40f, -1e-40f};
    for (int i = 0; i < 4000; ++i) {
        const uint32_t u = (uint32_t) splitmix(st);
        v.push_back(__builtin_bit_cast(float, u));
    }
    for (int i = 0; i < 2000; ++i) v.push_back((float) (unit(st) * 20.0 - 10.0));
    bool order = true, back = true;
    for (size_t i = 0; i < v.size(); ++i) {
        const float a = v[i];
        const uint32_t ka = spec_key32(a);
        if (ka == 0) order = false;   // key 0 is the "no entry" mark
        const float r = spec_key_float(ka);
        const bool nan = std::isnan(a);
        if (nan ? !(std::isinf(r) && r < 0) : !(r == a && std::signbit(r) == (a == 0.f ? false : std::signbit(a)))) back = false;
        for (size_t j = 0; j < v.size(); j += 7) {
            const float b = v[j];
            const float aa = nan ? -INFINITY : a, bb = std::isnan(b) ? -INFINITY : b;
            const uint32_t kb = spec_key32(b);
            if ((aa < bb) != (ka < kb)) order = false;
            if ((aa == bb) != (ka == kb)) order = false;
        }
    }
    check(order, "the key orders like the float (NaN as -inf, -0 as +0) and is never 0");
    check(back, "the float comes back from its key");
    // the 64-bit key: the higher logit first, the lower id first on a tie, and the id comes back
    check(spec_key64(spec_key32(1.f), 5) > spec_key64(spec_key32(1.f), 6), "a tie goes to the lower id");
    check(spec_key64(spec_key32(2.f), 900000) > spec_key64(spec_key32(1.f), 0), "a higher logit goes first");
    check(spec_key_id(spec_key64(spec_key32(-3.f), 248319)) == 248319, "the id comes back from the key");
    check(spec_key_id(spec_key64(spec_key32(-3.f), 0)) == 0, "id 0 comes back from the key");
}

// ------------------------------------------------------------------------------- 2. the kernels' algorithm on the host
// sp_topm (q36_engine.dp.cpp) as written: stage 1, a chunk's M best by taking the best key and its owner re-reading; stage 2,
// the merge of the chunks' lists by their heads.  kThreads = 256.
void sim_topm(const float* row, int V, int M, int32_t* ids, int32_t* bits) {
    constexpr int kThreads = 256;
    const int nch = (V + kSpecChunk - 1) / kSpecChunk;
    std::vector<std::vector<uint64_t>> lists((size_t) nch, std::vector<uint64_t>((size_t) kSpecTopM, 0x5555555555555555ull));   // stale
    for (int ch = 0; ch < nch; ++ch) {
        const int base = ch * kSpecChunk, n = std::min(kSpecChunk, V - base);
        std::vector<uint32_t> sk((size_t) kSpecChunk);
        for (int i = 0; i < kSpecChunk; ++i) sk[(size_t) i] = i < n ? spec_key32(row[base + i]) : 0u;
        auto scan = [&](int tid) {
            uint64_t best = 0;
            for (int i = tid; i < kSpecChunk; i += kThreads) {
                const uint32_t k = sk[(size_t) i];
                if (k != 0u) best = std::max(best, spec_key64(k, (uint32_t) (base + i)));
            }
            return best;
        };
        std::vector<uint64_t> best((size_t) kThreads);
        for (int t = 0; t < kThreads; ++t) best[(size_t) t] = scan(t);
        for (int r = 0; r < M; ++r) {
            const uint64_t w = *std::max_element(best.begin(), best.end());
            lists[(size_t) ch][(size_t) r] = w;
            if (w == 0) break;
            const int e = spec_key_id(w) - base;
            sk[(size_t) e] = 0u;
            best[(size_t) (e % kThreads)] = scan(e % kThreads);
        }
    }
    std::vector<int> p((size_t) nch, 0);
    std::vector<uint64_t> head((size_t) kThreads, 0);
    for (int c = 0; c < nch; ++c) head[(size_t) c] = lists[(size_t) c][0];
    for (int r = 0; r < M; ++r) {
        const uint64_t w = *std::max_element(head.begin(), head.end());
        ids[r] = w != 0 ? spec_key_id(w) : -1;
        bits[r] = w != 0 ? __builtin_bit_cast(int32_t, spec_key_float(spec_key_hi(w))) : 0;
        if (w == 0) continue;
        for (int c = 0; c < nch; ++c)
            if (head[(size_t) c] == w) {
                ++p[(size_t) c];
                head[(size_t) c] = p[(size_t) c] < kSpecTopM && p[(size_t) c] < M ? lists[(size_t) c][(size_t) p[(size_t) c]] : 0;
            }
    }
}

std::vector<float> random_row(uint64_t& st, int V, int quant) {   // quant > 0: values on a grid of 1/quant (many ties)
    std::vector<float> r((size_t) V);
    for (auto& x : r) {
        double v = unit(st) * 12.0 - 6.0;
        if (quant > 0) v = std::floor(v * quant) / quant;
        x = (float) v;
    }
    return r;
}

void test_topm_algorithm() {
    std::printf("the kernels' top-M on the host\n");
    uint64_t st = 7;
    bool same = true;
    int cases = 0;
    for (int V : {1, 5, 130, 4095, 4096, 4097, 9000, 12288 + 17, 20000}) {
        for (int quant : {0, 4, 1}) {
            for (int M : {1, 2, 20, 64, 84, 128}) {
                auto row = random_row(st, V, quant);
                if (V > 10) {   // specials: -inf, NaN, -0 / +0, a run of equal values at the top
                    row[3] = -INFINITY;
                    row[(size_t) V - 2] = NAN;
                    row[4] = -0.f;
                    row[5] = 0.f;
                    for (int i = 0; i < std::min(V, 40); i += 3) row[(size_t) (V - 1 - i)] = 7.f;
                }
                std::vector<int32_t> a((size_t) M), b((size_t) M), c((size_t) M), d((size_t) M);
                spec_topm_ref(row.data(), V, M, a.data(), b.data());
                sim_topm(row.data(), V, M, c.data(), d.data());
                if (a != c || b != d) {
                    same = false;
                    std::printf("  differs: V %d quant %d M %d\n", V, quant, M);
                }
                ++cases;
            }
        }
    }
    check(same, "two-stage top-M == a sort by (logit, lowest id), " + std::to_string(cases) + " rows");
    // the reference itself: ranks are in order and complete
    auto row = random_row(st, 3000, 4);
    std::vector<int32_t> ids(128), bits(128);
    spec_topm_ref(row.data(), 3000, 128, ids.data(), bits.data());
    bool ok = true;
    std::set<int> seen;
    for (int i = 0; i < 128; ++i) {
        const float v = __builtin_bit_cast(float, bits[(size_t) i]);
        ok = ok && v == row[(size_t) ids[(size_t) i]] && seen.insert(ids[(size_t) i]).second;
        if (i > 0) {
            const float pv = __builtin_bit_cast(float, bits[(size_t) i - 1]);
            ok = ok && (pv > v || (pv == v && ids[(size_t) i - 1] < ids[(size_t) i]));
        }
    }
    check(ok, "the reference top-M is ordered, distinct, and holds the row's values");
}

// ------------------------------------------------------------------------------- 3. a window row against the plain sampler
std::vector<int32_t> pack_row(const std::vector<float>& row, int M) {
    std::vector<int32_t> w((size_t) kSpecRowWords, 0);
    spec_topm_ref(row.data(), (int) row.size(), M, w.data(), w.data() + kSpecTopM);
    return w;
}

struct Setting {
    Sampling s;
    const char* name;
};

std::vector<Setting> settings() {
    std::vector<Setting> v;
    auto mk = [&](const char* name, float temp, float top_p, int top_k, float min_p, float rep, float fr, float pr, int last_n) {
        Sampling s;
        s.temperature = temp; s.top_p = top_p; s.top_k = top_k; s.min_p = min_p; s.repeat = rep; s.freq = fr; s.present = pr;
        s.last_n = last_n;
        v.push_back({s, name});
    };
    mk("greedy", 0.f, 1.f, 20, 0.f, 1.f, 0.f, 0.f, 0);
    mk("greedy presence 1.5 / 64", 0.f, 1.f, 20, 0.f, 1.f, 0.f, 1.5f, 64);
    mk("greedy repeat 1.1 freq 0.3 / 5", 0.f, 1.f, 20, 0.f, 1.1f, 0.3f, 0.f, 5);
    mk("greedy everything / 1", 0.f, 1.f, 20, 0.f, 1.3f, 0.2f, 0.7f, 1);
    mk("t 0.7 p 0.8 k 20", 0.7f, 0.8f, 20, 0.f, 1.f, 0.f, 0.f, 0);
    mk("t 1.0 p 0.95 k 20", 1.0f, 0.95f, 20, 0.f, 1.f, 0.f, 0.f, 0);
    mk("t 1.0 p 1 k 64", 1.0f, 1.0f, 64, 0.f, 1.f, 0.f, 0.f, 0);
    mk("t 1.0 p 1 k 0 (cap 64)", 1.0f, 1.0f, 0, 0.f, 1.f, 0.f, 0.f, 0);
    mk("t 0.7 p 0.8 k 20 presence 1.5 / 64", 0.7f, 0.8f, 20, 0.f, 1.f, 0.f, 1.5f, 64);
    mk("t 1.0 p 1 k 64 presence 0.5 repeat 1.2 / 64", 1.0f, 1.0f, 64, 0.f, 1.2f, 0.f, 0.5f, 64);
    mk("t 0.9 min_p 0.05 k 40 freq 0.4 / 17", 0.9f, 1.0f, 40, 0.05f, 1.f, 0.4f, 0.f, 17);
    mk("t 0.3 p 0.9 k 1", 0.3f, 0.9f, 1, 0.f, 1.f, 0.f, 0.f, 0);
    mk("t 1.5 p 0.99 k 3 repeat 2 / 2", 1.5f, 0.99f, 3, 0.f, 2.f, 0.f, 0.f, 2);
    return v;
}

void test_row_vs_plain() {
    std::printf("a window row against the plain sampler\n");
    uint64_t st = 99;
    long long rows = 0, bad = 0, draws_bad = 0;
    for (const Setting& se : settings()) {
        const SpecPlan plan = spec_plan(se.s, 1000);
        check(plan.ok, std::string("planned: ") + se.name);
        for (int trial = 0; trial < 120; ++trial) {
            const int V = trial % 3 == 0 ? 300 : trial % 3 == 1 ? 1000 : 6000;
            const int quant = trial % 4 == 0 ? 0 : trial % 4 == 1 ? 8 : trial % 4 == 2 ? 2 : 1;   // 1: very many ties
            auto row = random_row(st, V, quant);
            // a history that hits the top of the row (so the penalties reorder it), with repeats, of any length
            std::vector<int32_t> top(row.size());
            for (size_t i = 0; i < top.size(); ++i) top[i] = (int32_t) i;
            std::partial_sort(top.begin(), top.begin() + 80, top.end(), [&](int a, int b) { return row[(size_t) a] > row[(size_t) b]; });
            const int len = trial % 5 == 0 ? 0 + 1 : (int) (splitmix(st) % 150) + 1;
            std::vector<int32_t> hist((size_t) len);
            const bool all_top = trial % 7 == 0;   // 64 distinct top tokens: every penalized id at the top, a window of 64
            for (int i = 0; i < len; ++i)
                hist[(size_t) i] = all_top ? top[(size_t) (i % 64)] : (splitmix(st) % 2 ? top[(size_t) (splitmix(st) % 40)] : (int32_t) (splitmix(st) % (uint64_t) V));
            const int D = (int) (splitmix(st) % 5);   // 0 .. 4 drafts; drafts often the plain pick, sometimes anything
            std::vector<std::vector<float>> rws((size_t) D + 1);
            rws[0] = row;
            for (int t = 1; t <= D; ++t) rws[(size_t) t] = random_row(st, V, quant);
            // the plain loop on those rows, drafts chosen as it goes
            const uint64_t seed = splitmix(st);
            std::vector<int32_t> drafts((size_t) D), picks_ref((size_t) D + 1), picks((size_t) D + 1);
            std::mt19937_64 rp(seed);
            std::vector<int32_t> h = hist;
            int a_ref = D;
            {
                std::mt19937_64 rd(seed ^ 0xabcdef);   // a drafter's stream: a copy of the sampler's, or noise
                std::mt19937_64 oracle(seed);
                std::vector<int32_t> ho = hist;
                for (int t = 0; t < D; ++t) {   // drafts: the plain pick with probability 0.7 (from a copy of the stream)
                    std::vector<float> l = rws[(size_t) t];
                    const int pick = sample_row(l, se.s, ho, oracle);
                    drafts[(size_t) t] = rd() % 10 < 7 ? pick : (int32_t) (rd() % (uint64_t) V);
                    ho.push_back(drafts[(size_t) t]);
                    if (drafts[(size_t) t] != pick) {   // the oracle copy must follow the plain stream: stop guessing right
                        for (int u = t + 1; u < D; ++u) drafts[(size_t) u] = (int32_t) (rd() % (uint64_t) V);
                        break;
                    }
                }
            }
            for (int t = 0; t <= D; ++t) {
                std::vector<float> l = rws[(size_t) t];
                picks_ref[(size_t) t] = sample_row(l, se.s, h, rp);
                if (t == D || picks_ref[(size_t) t] != drafts[(size_t) t]) { a_ref = t; break; }
                h.push_back(drafts[(size_t) t]);
            }
            // the window path
            std::vector<int32_t> words;
            const SpecPlan pl = spec_plan(se.s, V);
            for (int t = 0; t <= D; ++t) {
                const auto w = pack_row(rws[(size_t) t], pl.M);
                words.insert(words.end(), w.begin(), w.end());
            }
            std::mt19937_64 rw(seed);
            const int a = decide_window(se.s, pl.kk, pl.M, D, hist, drafts.data(), words.data(), rw, picks.data());
            ++rows;
            bool same = a == a_ref;
            for (int t = 0; t <= a_ref && same; ++t) same = picks[(size_t) t] == picks_ref[(size_t) t];
            if (!same) {
                ++bad;
                if (bad < 5) std::printf("  differs: %s trial %d V %d len %d D %d a %d / %d\n", se.name, trial, V, len, D, a, a_ref);
            }
            if (rw() != rp()) ++draws_bad;   // the same number of draws from the same stream
        }
    }
    check(bad == 0, "window rows == the plain sampler, " + std::to_string(rows) + " windows over the penalty and sampling mixes");
    check(draws_bad == 0, "and the same draws were taken from the stream");
}

// ------------------------------------------------------------------------------- 4. a whole answer, windows against plain
// A model whose row depends on the whole context (so a wrong window position gives other tokens), with ties and a few
// tokens that the context repeats; token 1 is the end of the answer.
std::vector<float> model_row(const std::vector<int32_t>& ctx, int V) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = ctx.size() > 6 ? ctx.size() - 6 : 0; i < ctx.size(); ++i) h = (h ^ (uint64_t) ctx[i] ^ (i << 40)) * 1099511628211ull;
    std::vector<float> r((size_t) V);
    uint64_t st = h;
    for (int i = 0; i < V; ++i) r[(size_t) i] = std::floor((float) (unit(st) * 10.0 - 5.0) * 4.f) / 4.f;
    // the context's own tokens lean the row (what a repeat-prone model does): the last tokens get a lift
    for (size_t i = ctx.size() > 8 ? ctx.size() - 8 : 0; i < ctx.size(); ++i) r[(size_t) ctx[i]] += 2.5f;
    r[1] = ctx.size() % 11 == 0 ? 9.f : r[1] - 3.f;   // the end of the answer now and then
    return r;
}

struct Result {
    std::vector<int32_t> out;
    long long windows = 0, accepted = 0, cut = 0;   // cut: the answer ended inside a window (the end token, max_new)
};

Result gen_plain(const Sampling& s, uint64_t seed, std::vector<int32_t> hist, int max_new, int V) {
    std::mt19937_64 rng(seed);
    Result res;
    std::vector<float> lg = model_row(hist, V);
    int nx = sample_row(lg, s, hist, rng);
    for (;;) {
        res.out.push_back(nx);
        hist.push_back(nx);
        if (nx == 1 || (int) res.out.size() >= max_new) break;
        lg = model_row(hist, V);
        nx = sample_row(lg, s, hist, rng);
    }
    return res;
}

// drafter: 0 the argmax chain of the model (often right), 1 random ids, 2 the chain, a wrong id now and then, 3 ids the rows never offer
Result gen_spec(const Sampling& s, uint64_t seed, std::vector<int32_t> hist, int max_new, int V, int D, int drafter) {
    const SpecPlan plan = spec_plan(s, V);
    std::mt19937_64 rng(seed), dr(seed + 1);
    Result res;
    std::vector<float> lg = model_row(hist, V);
    int nx = sample_row(lg, s, hist, rng);   // the first token: the plain sampler, as the serve loop does after the prompt
    for (;;) {
        res.out.push_back(nx);
        hist.push_back(nx);
        if (nx == 1 || (int) res.out.size() >= max_new) break;
        // the window [nx, d0 .. d(D-1)]: the drafts first
        std::vector<int32_t> drafts((size_t) D), ctx = hist;
        for (int t = 0; t < D; ++t) {
            const auto r = model_row(ctx, V);
            int d = (int) (std::max_element(r.begin(), r.end()) - r.begin());
            if (drafter == 1) d = (int) (dr() % (uint64_t) V);
            if (drafter == 2 && dr() % 4 == 0) d = (int) (dr() % (uint64_t) V);
            if (drafter == 3) d = (int) (std::min_element(r.begin(), r.end()) - r.begin());   // a token far below the candidates
            drafts[(size_t) t] = d;
            ctx.push_back(d);
        }
        std::vector<int32_t> words;
        ctx = hist;
        for (int t = 0; t <= D; ++t) {
            const auto w = pack_row(model_row(ctx, V), plan.M);
            words.insert(words.end(), w.begin(), w.end());
            if (t < D) ctx.push_back(drafts[(size_t) t]);
        }
        int32_t picks[16];
        const int a = decide_window(s, plan.kk, plan.M, D, hist, drafts.data(), words.data(), rng, picks);
        ++res.windows;
        res.accepted += a;
        bool ended = false;
        for (int i = 0; i < a && !ended; ++i) {   // each accepted draft is emitted; the answer may end inside them
            res.out.push_back(drafts[(size_t) i]);
            hist.push_back(drafts[(size_t) i]);
            if (drafts[(size_t) i] == 1 || (int) res.out.size() >= max_new) ended = true;
        }
        if (ended) {
            ++res.cut;
            break;
        }
        nx = picks[a];
    }
    return res;
}

void test_answers() {
    std::printf("a whole answer: windows against one token a step\n");
    long long runs = 0, bad = 0, windows = 0, accepted = 0, cut = 0;
    uint64_t st = 5;
    for (const Setting& se : settings()) {
        for (int trial = 0; trial < 12; ++trial) {
            const int V = 400 + (int) (splitmix(st) % 300);
            const int D = 1 + (int) (splitmix(st) % 7);
            const int drafter = trial % 4;
            const int max_new = trial % 3 == 0 ? 5 + (int) (splitmix(st) % 9) : 60 + (int) (splitmix(st) % 60);
            std::vector<int32_t> prompt((size_t) (1 + splitmix(st) % 40));
            for (auto& t : prompt) t = 2 + (int32_t) (splitmix(st) % (uint64_t) (V - 2));
            const uint64_t seed = splitmix(st);
            const Result p = gen_plain(se.s, seed, prompt, max_new, V);
            const Result q = gen_spec(se.s, seed, prompt, max_new, V, D, drafter);
            ++runs;
            windows += q.windows;
            accepted += q.accepted;
            cut += q.cut;
            if (p.out != q.out) {
                ++bad;
                if (bad < 5) std::printf("  differs: %s trial %d D %d drafter %d, %zu / %zu tokens\n", se.name, trial, D, drafter, p.out.size(), q.out.size());
            }
        }
    }
    check(bad == 0, "windows give the plain loop's answer for the same seed, " + std::to_string(runs) + " answers");
    check(accepted > 0 && windows > 0, "drafts were accepted in those runs (" + std::to_string(accepted) + " in " + std::to_string(windows) + " windows)");
    check(cut > 0, "some answers ended inside a window (the end token or max_new): " + std::to_string(cut));
    // the same seed twice: the same text; another seed: another text
    Sampling s;
    s.temperature = 1.0f; s.top_p = 0.95f; s.top_k = 20;
    std::vector<int32_t> prompt = {5, 9, 33, 7};
    const Result a1 = gen_spec(s, 77, prompt, 80, 500, 3, 0), a2 = gen_spec(s, 77, prompt, 80, 500, 3, 0), a3 = gen_spec(s, 78, prompt, 80, 500, 3, 0);
    check(a1.out == a2.out, "seed reproducibility: the same seed twice gives the same tokens");
    check(a1.out != a3.out, "another seed gives other tokens");
    const Result a4 = gen_spec(s, 77, prompt, 80, 500, 5, 2);
    check(a1.out == a4.out, "the same seed gives the same tokens at another window size and with another drafter");
}


// ------------------------------------------------------------------------------- 4b. several slots in one window (--batch-mtp)
// The host side of the engine's window_mtp for a batch: every slot has its own sampler, history and random stream; a window
// gives each slot 1 + D rows (D chosen per window and slot, the rows being shared, as fit_depths does); a slot that draws on
// the host takes its picks from the rows' candidates (batch_slot_step, the list as long as the longest any slot needs), a greedy
// slot from the argmax of its rows, a request the candidates cannot serve (spec_plan not ok) from the whole row, with no drafts.
// Every slot's answer must be the answer of the same request alone, one token a step, whatever else shares the window.
struct BSlot {
    Sampling s;
    SpecPlan plan;
    uint64_t seed = 0;
    int max_new = 0;
    std::mt19937_64 rng;
    std::mt19937_64 dr;
    std::vector<int32_t> hist, out;
    bool active = false;
    long long offered = 0, accepted = 0;
    bool drafts_ok() const { return !s.needs_logits() || plan.ok; }
};

bool b_eos(int t) { return t == 1; }

std::vector<std::vector<int32_t>> run_batch(const std::vector<BSlot>& init, int V, bool reverse_order, uint64_t plan_seed, long long* off,
                                            long long* acc, long long* mixed_windows, long long* cand_rows) {
    std::vector<BSlot> sl = init;
    uint64_t ps = plan_seed;
    for (BSlot& b : sl) {   // admission: the first token drawn as the plain path does
        b.rng.seed(b.seed);
        b.dr.seed(b.seed + 1);
        std::vector<float> lg = model_row(b.hist, V);
        int nx = sample_row(lg, b.s, b.hist, b.rng);
        b.out.push_back(nx);
        b.hist.push_back(nx);
        b.active = !(b_eos(nx) || b.max_new <= 1);
    }
    for (int guard = 0; guard < 100000; ++guard) {
        std::vector<int> act;
        for (int i = 0; i < (int) sl.size(); ++i) {
            const int k = reverse_order ? (int) sl.size() - 1 - i : i;
            if (sl[(size_t) k].active) act.push_back(k);
        }
        if (act.empty()) break;
        const int S = (int) act.size();
        // the depths: what each slot that drafts wants (0 .. 6), cut to the 16 rows the window has (the largest first)
        std::vector<int> want((size_t) S, 0);
        for (int i = 0; i < S; ++i)
            if (sl[(size_t) act[(size_t) i]].drafts_ok()) want[(size_t) i] = 1 + (int) (splitmix(ps) % 6);
        int total = S;
        for (int v : want) total += v;
        while (total > 16) {
            int best = -1;
            for (int i = 0; i < S; ++i)
                if (want[(size_t) i] > 0 && (best < 0 || want[(size_t) i] > want[(size_t) best])) best = i;
            if (best < 0) break;
            --want[(size_t) best];
            --total;
        }
        // the window's candidate list length: the largest M of the slots that draw on the host
        int Mwin = 0;
        bool has_cand = false, has_greedy = false;
        for (int i = 0; i < S; ++i) {
            const BSlot& b = sl[(size_t) act[(size_t) i]];
            if (b.s.needs_logits() && b.plan.ok) { Mwin = std::max(Mwin, b.plan.M); has_cand = true; }
            if (!b.s.needs_logits()) has_greedy = true;
        }
        if (has_cand && has_greedy) ++*mixed_windows;
        for (int i = 0; i < S; ++i) {
            BSlot& b = sl[(size_t) act[(size_t) i]];
            const int D = want[(size_t) i];
            const int drafter = (int) (splitmix(ps) % 4);
            std::vector<int32_t> drafts((size_t) D), ctx = b.hist;
            for (int t = 0; t < D; ++t) {
                const auto r = model_row(ctx, V);
                int d = (int) (std::max_element(r.begin(), r.end()) - r.begin());
                if (drafter == 1) d = (int) (b.dr() % (uint64_t) V);
                if (drafter == 2 && b.dr() % 4 == 0) d = (int) (b.dr() % (uint64_t) V);
                if (drafter == 3) d = (int) (std::min_element(r.begin(), r.end()) - r.begin());
                drafts[(size_t) t] = d;
                ctx.push_back(d);
            }
            // this slot's rows as the window hands them over
            const bool cand = b.s.needs_logits() && b.plan.ok;
            int32_t picks[16] = {};
            int y0 = 0, a = 0;
            const int64_t produced = (int64_t) b.out.size();
            if (cand) {
                std::vector<int32_t> words;
                ctx = b.hist;
                for (int t = 0; t <= D; ++t) {
                    const auto w = pack_row(model_row(ctx, V), Mwin);
                    words.insert(words.end(), w.begin(), w.end());
                    if (t < D) ctx.push_back(drafts[(size_t) t]);
                }
                a = batch_slot_step(b.s, b.plan, b.hist, drafts, words.data(), b.rng, picks, false, produced, b.max_new, 1000, 1 << 30,
                                    b_eos);
                y0 = picks[0];
                ++*cand_rows;
            } else if (b.s.needs_logits()) {   // (the whole row to the host, no drafts)
                std::vector<float> lg = model_row(b.hist, V);
                y0 = sample_row(lg, b.s, b.hist, b.rng);
                picks[0] = y0;
            } else {   // a greedy slot: the argmax of its rows, and the engine's own loop
                ctx = b.hist;
                for (int t = 0; t <= D; ++t) {
                    const auto r = model_row(ctx, V);
                    picks[t] = (int) (std::max_element(r.begin(), r.end()) - r.begin());
                    if (t < D) ctx.push_back(drafts[(size_t) t]);
                }
                y0 = picks[0];
                for (int y = y0; a < D && y == drafts[(size_t) a] && !b_eos(y) && produced + a + 2 <= b.max_new; y = picks[a]) ++a;
            }
            b.offered += D;
            b.accepted += a;
            int y = y0;
            for (int j = 0; j <= a; ++j) {
                if (j > 0) y = picks[j];
                b.out.push_back(y);
                b.hist.push_back(y);
                if (b_eos(y) || (int) b.out.size() >= b.max_new) { b.active = false; break; }
            }
        }
    }
    std::vector<std::vector<int32_t>> outs;
    for (const BSlot& b : sl) {
        outs.push_back(b.out);
        *off += b.offered;
        *acc += b.accepted;
    }
    return outs;
}

void test_batch_slots() {
    std::printf("slots of one batch window against the same request alone\n");
    std::vector<Setting> st = settings();
    {   // two requests the candidates cannot serve: they run the whole-row path, with no drafts, beside the others
        Sampling a;
        a.temperature = 0.8f; a.top_p = 0.9f; a.top_k = 100;   // a cap above 64
        st.push_back({a, "t 0.8 k 100 (not drafted)"});
        Sampling b;
        b.temperature = 0.f; b.repeat = 0.8f; b.last_n = 8;    // a penalty that raises a logit
        st.push_back({b, "greedy repeat 0.8 (not drafted)"});
    }
    uint64_t rs = 99;
    long long runs = 0, bad = 0, off = 0, acc = 0, mixed = 0, cand_rows = 0, orderbad = 0;
    for (int trial = 0; trial < 40; ++trial) {
        const int V = 400 + (int) (splitmix(rs) % 300);
        const int S = 2 + (int) (splitmix(rs) % 7);   // 2 .. 8 slots
        std::vector<BSlot> init((size_t) S);
        for (int i = 0; i < S; ++i) {
            BSlot& b = init[(size_t) i];
            const Setting& se = st[(size_t) ((trial * 3 + i * 5 + (int) (splitmix(rs) % 4)) % (int) st.size())];
            b.s = se.s;
            b.plan = spec_plan(b.s, V);
            b.seed = splitmix(rs);
            b.max_new = (i + trial) % 4 == 0 ? 3 + (int) (splitmix(rs) % 8) : 50 + (int) (splitmix(rs) % 70);
            b.hist.resize((size_t) (1 + splitmix(rs) % 40));
            for (auto& t : b.hist) t = 2 + (int32_t) (splitmix(rs) % (uint64_t) (V - 2));
        }
        const uint64_t plan_seed = splitmix(rs);
        const auto o1 = run_batch(init, V, false, plan_seed, &off, &acc, &mixed, &cand_rows);
        long long o2n = 0, a2n = 0, m2 = 0, c2 = 0;
        const auto o2 = run_batch(init, V, true, plan_seed + 1, &o2n, &a2n, &m2, &c2);   // another order and other depths
        for (int i = 0; i < S; ++i) {
            const BSlot& b = init[(size_t) i];
            const Result p = gen_plain(b.s, b.seed, b.hist, b.max_new, V);
            ++runs;
            if (o1[(size_t) i] != p.out) {
                ++bad;
                if (bad < 5) std::printf("  differs: trial %d slot %d of %d, %zu / %zu tokens\n", trial, i, S, p.out.size(), o1[(size_t) i].size());
            }
            if (o2[(size_t) i] != p.out) ++orderbad;
        }
    }
    check(bad == 0, "every slot's answer equals the same request alone, one token a step: " + std::to_string(runs) + " answers");
    check(orderbad == 0, "the same with the slots in the other order and other depths");
    check(acc > 0 && off > 0, "drafts were accepted in those windows (" + std::to_string(acc) + " of " + std::to_string(off) + ")");
    check(mixed > 0, "windows with greedy and host-drawn slots together: " + std::to_string(mixed));
    check(cand_rows > 0, "slots drew from the rows' candidates: " + std::to_string(cand_rows));
}

// ------------------------------------------------------------------------------- 5. the rule reproduces the target
// A Markov model over V = 8 tokens: the row depends on the last token only, so each row is a known distribution.
constexpr int kV = 8;
std::vector<float> markov_row(int last) {
    std::vector<float> r(kV);
    for (int i = 0; i < kV; ++i) r[(size_t) i] = (float) (1.6 * std::sin(1.3 * (last + 1) * (i + 2)) + 0.4 * ((i * 5 + last * 3) % 7));
    return r;
}
// the distribution the sampler draws from, worked out here from the definition (temperature, then the smallest prefix of the
// ranking that reaches top_p, min_p against the top one), not from draw()
std::vector<double> target_of(const std::vector<float>& lg, const Sampling& s) {
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return lg[(size_t) a] > lg[(size_t) b]; });
    const int kk = std::min<int>(s.top_k > 0 ? s.top_k : 64, (int) lg.size());
    std::vector<double> p(lg.size(), 0.0), e((size_t) kk);
    double z = 0;
    for (int i = 0; i < kk; ++i) z += (e[(size_t) i] = std::exp(((double) lg[(size_t) idx[(size_t) i]] - lg[(size_t) idx[0]]) / s.temperature));
    int keep = kk;
    double cum = 0;
    for (int i = 0; i < kk; ++i) {
        if (e[(size_t) i] / z < (double) s.min_p * (e[0] / z)) { keep = i; break; }
        cum += e[(size_t) i] / z;
        if (cum >= (double) s.top_p) { keep = i + 1; break; }
    }
    keep = std::max(1, keep);
    double tot = 0;
    for (int i = 0; i < keep; ++i) tot += e[(size_t) i];
    for (int i = 0; i < keep; ++i) p[(size_t) idx[(size_t) i]] = e[(size_t) i] / tot;
    return p;
}

enum Rule { kExact, kAcceptAll, kPickDraft };
struct Stats {
    long long cnt[kV][kV] = {};
    long long rows = 0, hits = 0;
    double expect_hits = 0, var_hits = 0;
};

// Windows of D drafts on the Markov model.  The draft is the argmax of a noisy copy of the row (a point mass, drawn from a
// stream the sampler does not see).  Rule kExact is decide_window; the others are wrong on purpose.
Stats run_markov(const Sampling& s, Rule rule, int D, long long tokens, uint64_t seed) {
    Stats S;
    std::mt19937_64 rng(seed);
    uint64_t noise = seed * 31 + 7;
    std::vector<int32_t> hist = {3};
    auto first_row = markov_row(3);
    int nx = sample_row(first_row, s, hist, rng);
    long long made = 0;
    int prev = 3;
    const SpecPlan plan = spec_plan(s, kV);
    while (made < tokens) {
        S.cnt[prev][nx]++;
        prev = nx;
        hist.push_back(nx);
        ++made;
        std::vector<int32_t> drafts((size_t) D), words;
        int last = nx;
        for (int t = 0; t < D; ++t) {   // the drafter: a noisy argmax chain
            auto r = markov_row(last);
            for (auto& x : r) x += (float) (unit(noise) * 2.4 - 1.2);
            drafts[(size_t) t] = (int) (std::max_element(r.begin(), r.end()) - r.begin());
            last = drafts[(size_t) t];
        }
        last = nx;
        for (int t = 0; t <= D; ++t) {
            const auto w = pack_row(markov_row(last), plan.M);
            words.insert(words.end(), w.begin(), w.end());
            if (t < D) last = drafts[(size_t) t];
        }
        int32_t picks[16];
        int a = 0;
        if (rule == kExact) {
            a = decide_window(s, plan.kk, plan.M, D, hist, drafts.data(), words.data(), rng, picks);
        } else {
            for (int t = 0; t <= D; ++t) {   // wrong rules: take the draft whenever it is a candidate / emit the draft's own row pick
                const int32_t* ids = words.data() + (size_t) t * kSpecRowWords;
                std::vector<int32_t> none;
                picks[t] = sample_window_row(ids, ids + kSpecTopM, plan.M, s, plan.kk, nullptr, 0, rng);
                if (t == D) { a = D; break; }
                bool cand = false;
                for (int i = 0; i < plan.M && ids[i] >= 0; ++i) cand = cand || ids[i] == drafts[(size_t) t];
                const bool take = rule == kAcceptAll ? cand : picks[t] == drafts[(size_t) t] || (rng() & 3) == 0;
                if (!take) { a = t; break; }
            }
        }
        // the target's chance of each draft that was looked at, given that it was reached
        last = nx;
        for (int t = 0; t <= std::min(a, D - 1); ++t) {
            const auto p = target_of(markov_row(last), s);
            const double pd = p[(size_t) drafts[(size_t) t]];
            S.expect_hits += pd;
            S.var_hits += pd * (1 - pd);
            ++S.rows;
            S.hits += t < a;
            last = drafts[(size_t) t];
        }
        for (int i = 0; i < a && made < tokens; ++i) {
            S.cnt[prev][drafts[(size_t) i]]++;
            prev = drafts[(size_t) i];
            hist.push_back(prev);
            ++made;
        }
        nx = picks[a];
    }
    return S;
}

// The largest |count - n p| / sqrt(n p (1 - p)) over the cells of the transition table.  The tokens are the output of
// the run; row `prev` of the table must be the target of the model's row after prev.
double worst_z(const Stats& S, const Sampling& s, int* cells_out = nullptr) {
    double worst = 0;
    int cells = 0;
    for (int i = 0; i < kV; ++i) {
        long long n = 0;
        for (int j = 0; j < kV; ++j) n += S.cnt[i][j];
        if (n < 2000) continue;
        const auto p = target_of(markov_row(i), s);
        for (int j = 0; j < kV; ++j) {
            const double mu = (double) n * p[(size_t) j], sd = std::sqrt(std::max(1e-12, (double) n * p[(size_t) j] * (1 - p[(size_t) j])));
            if (p[(size_t) j] == 0) { if (S.cnt[i][j] != 0) return 1e9; continue; }
            worst = std::max(worst, std::fabs((double) S.cnt[i][j] - mu) / sd);
            ++cells;
        }
    }
    if (cells_out) *cells_out = cells;
    return worst;
}

void test_statistics() {
    std::printf("the accept / reject rule reproduces the target (statistical)\n");
    // Each token of the output is a cell of a multinomial: its count is binomial(n_i, p) with a standard deviation of
    // sqrt(n p (1 - p)).  There are at most 64 cells, so a threshold of 5.0 standard deviations is passed by a correct rule
    // with probability 1 - 64 * 5.7e-7 (a two-sided normal tail, 5.7e-7, a cell): one run in 27,000 fails by chance.  The
    // seed is fixed, so the result does not change from run to run.  (The tokens of one window are not independent of each
    // other, but each is a draw from its own row's target given the tokens before it, which is what the cells count.)
    const double kZ = 5.0;
    struct Case { const char* name; float temp, top_p; int top_k; float min_p; };
    const Case cases[] = {{"t 1.0 p 1 k 8", 1.0f, 1.0f, 8, 0.f}, {"t 0.7 p 0.8 k 20", 0.7f, 0.8f, 20, 0.f},
                          {"t 1.0 p 0.95 k 4", 1.0f, 0.95f, 4, 0.f}, {"t 0.9 p 1 k 8 min_p 0.1", 0.9f, 1.0f, 8, 0.1f}};
    for (const Case& c : cases) {
        Sampling s;
        s.temperature = c.temp; s.top_p = c.top_p; s.top_k = c.top_k; s.min_p = c.min_p;
        for (int D : {1, 3, 7}) {
            const Stats S = run_markov(s, kExact, D, 400000, 12345 + (uint64_t) D);
            int cells = 0;
            const double z = worst_z(S, s, &cells);
            check(z < kZ, std::string("exact rule ") + c.name + " D " + std::to_string(D) + ": worst cell " + std::to_string(z) + " sd over " + std::to_string(cells) + " cells");
            // the acceptance chance of a draft is the target's probability of it: hits against the sum of p(d)
            const double sd = std::sqrt(S.var_hits);
            check(std::fabs((double) S.hits - S.expect_hits) < 5.0 * sd,
                  std::string("accepted drafts ") + std::to_string(S.hits) + " of " + std::to_string(S.rows) + " rows, expected " + std::to_string(S.expect_hits) + " +- " + std::to_string(sd));
        }
        // the wrong rules are caught
        Sampling s2 = s;
        const double z_all = worst_z(run_markov(s2, kAcceptAll, 3, 400000, 99), s2);
        const double z_dr = worst_z(run_markov(s2, kPickDraft, 3, 400000, 99), s2);
        check(z_all > 10.0 || c.top_k <= 1, std::string("a rule that takes any candidate draft is caught (") + c.name + ", " + std::to_string(z_all) + " sd)");
        check(z_dr > 10.0, std::string("a rule that takes the draft's own row is caught (") + c.name + ", " + std::to_string(z_dr) + " sd)");
    }
}

// ------------------------------------------------------------------------------- 6. the Flash-Next rule for a point mass
void test_flash_next_point_mass() {
    std::printf("the Flash-Next engine's rule for a point-mass draft\n");
    // strata::core::spec_verify_row with no q list (a point mass at the draft): it accepts with probability p(d) and, on a
    // rejection, draws from p without d.  The emitted token is then distributed as p.  Enumerated over a grid of the two
    // uniforms.  The q36 rule emits the row's own draw whether or not it equals the draft, so its emitted token is p
    // as well, and a draft is kept with probability p(d): the same chance, on the plain path's own stream.
    const double ps[3][5] = {{0.5, 0.3, 0.1, 0.07, 0.03}, {0.9, 0.05, 0.03, 0.01, 0.01}, {0.2, 0.2, 0.2, 0.2, 0.2}};
    const int32_t ids[5] = {10, 11, 12, 13, 14};
    int bad = 0;
    for (const auto& p : ps) {
        for (int d = 0; d < 5; ++d) {
            std::vector<double> out(5, 0.0);
            double acc = 0;
            const int N = 1000;
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j) {
                    const strata::core::SpecP P{ids, p, 5};
                    const strata::core::SpecQ Q{nullptr, nullptr, -1};
                    const auto r = strata::core::spec_verify_row(P, ids[d], Q, (float) ((i + 0.5) / N), (float) ((j + 0.5) / N));
                    for (int x = 0; x < 5; ++x) out[(size_t) x] += ids[x] == r.token;
                    acc += r.accepted;
                }
            for (int x = 0; x < 5; ++x) bad += std::fabs(out[(size_t) x] / ((double) N * N) - p[x]) > 3e-3;
            bad += std::fabs(acc / ((double) N * N) - p[d]) > 3e-3;
        }
    }
    check(bad == 0, "point-mass draft: the emitted token is p and the draft is kept with probability p(d) (spec_verify_row)");
}

// ------------------------------------------------------------------------------- 7. the plain sampler is what it was
// The sampler as it was before this change (copied from the engine), for rows without exact ties.
int old_sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        std::map<int, int> cnt;
        const size_t from = s.last_n < 0 || (size_t) s.last_n >= hist.size() ? 0 : hist.size() - (size_t) s.last_n;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto& [id, c] : cnt) {
            float& v = lg[(size_t) id];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) c * s.freq + s.present;
        }
    }
    if (s.temperature <= 0.f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize((size_t) kk);
    std::vector<double> p(idx.size());
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (p[i] = std::exp((lg[(size_t) idx[i]] - mx) / s.temperature));
    size_t keep = idx.size();
    double cum = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) { keep = i; break; }
        cum += p[i] / sum;
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i) {
        if ((r -= p[i]) <= 0) return idx[i];
    }
    return idx[keep - 1];
}

void test_plain_unchanged() {
    std::printf("the plain sampler against the code it replaced\n");
    uint64_t st = 2024;
    long long n = 0, bad = 0;
    for (const Setting& se : settings()) {
        for (int trial = 0; trial < 150; ++trial) {
            const int V = trial % 2 ? 777 : 2500;
            auto row = random_row(st, V, 0);   // no exact ties
            std::vector<int32_t> hist((size_t) (1 + splitmix(st) % 100));
            for (auto& t : hist) t = (int32_t) (splitmix(st) % (uint64_t) V);
            auto a = row, b = row;
            std::mt19937_64 r1(trial), r2(trial);
            const int x = old_sample(a, se.s, hist, r1), y = sample_row(b, se.s, hist, r2);
            ++n;
            bad += x != y || r1() != r2();
        }
    }
    check(bad == 0, "sample_row == the old sample() on rows without ties, " + std::to_string(n) + " rows");
    // exact ties: the lowest id first (the old order was whatever partial_sort left)
    std::vector<float> lg(50, 1.f);
    lg[7] = lg[30] = lg[3] = 5.f;
    Sampling s;
    s.temperature = 1.f; s.top_k = 2;   // the cap cuts the tie of 3 tokens: ids 3 and 7 stay, 30 does not
    int seen[50] = {};
    for (int i = 0; i < 400; ++i) {
        auto l = lg;
        std::mt19937_64 r(i);
        ++seen[sample_row(l, s, {}, r)];
    }
    check(seen[30] == 0 && seen[3] > 0 && seen[7] > 0, "a tie at the candidate cap keeps the lowest ids");
    Sampling g;
    auto l2 = lg;
    std::mt19937_64 r(1);
    check(sample_row(l2, g, {}, r) == 3, "greedy on a tie takes the lowest id");
}

// ------------------------------------------------------------------------------- 8. which requests draft
void test_plan() {
    std::printf("which requests draft\n");
    auto S = [](float t, int k, float rep, float fr, float pr, int n) {
        Sampling s;
        s.temperature = t; s.top_k = k; s.repeat = rep; s.freq = fr; s.present = pr; s.last_n = n;
        return s;
    };
    const int V = 248320;
    check(spec_plan(S(0.f, 20, 1.f, 0.f, 1.5f, 64), V).ok && !spec_plan(S(0.f, 20, 1.f, 0.f, 1.5f, 64), V).sampled, "greedy with presence 1.5 / 64");
    check(spec_plan(S(0.f, 20, 1.f, 0.f, 1.5f, 64), V).M == 65, "  returns 65 candidates a row (1 + the window)");
    check(spec_plan(S(0.7f, 20, 1.f, 0.f, 1.5f, 64), V).ok && spec_plan(S(0.7f, 20, 1.f, 0.f, 1.5f, 64), V).M == 84, "t 0.7 k 20 presence 1.5 / 64: 84 candidates");
    check(spec_plan(S(1.0f, 20, 1.f, 0.f, 0.f, 0), V).ok && spec_plan(S(1.0f, 20, 1.f, 0.f, 0.f, 0), V).M == 20, "t 1.0 k 20: 20 candidates");
    check(spec_plan(S(1.0f, 0, 1.f, 0.f, 0.f, 0), V).kk == 64 && spec_plan(S(1.0f, 0, 1.f, 0.f, 0.f, 0), V).ok, "top_k 0: the cap of 64");
    check(spec_plan(S(1.0f, 64, 1.2f, 0.f, 1.f, 64), V).M == 128 && spec_plan(S(1.0f, 64, 1.2f, 0.f, 1.f, 64), V).ok, "top_k 64 with a window of 64: 128, the most");
    check(!spec_plan(S(1.0f, 100, 1.f, 0.f, 0.f, 0), V).ok, "top_k above 64: the plain path");
    check(!spec_plan(S(1.0f, 20, 1.f, 0.f, 0.5f, 65), V).ok, "a window above 64: the plain path");
    check(!spec_plan(S(1.0f, 20, 1.f, 0.f, 0.5f, -1), V).ok, "the whole history as the window: the plain path");
    check(!spec_plan(S(1.0f, 20, 1.f, 0.f, -0.5f, 64), V).ok, "a negative presence penalty (it raises a logit): the plain path");
    check(!spec_plan(S(1.0f, 20, 1.f, -0.5f, 0.f, 64), V).ok, "a negative frequency penalty: the plain path");
    check(!spec_plan(S(1.0f, 20, 0.9f, 0.f, 0.f, 64), V).ok, "a repeat penalty below 1: the plain path");
    check(spec_plan(S(1.0f, 20, 0.9f, 0.f, 0.f, 0), V).ok, "a window of 0 turns the penalties off: they do not matter");
    check(!spec_plan(S(1.0f, 20, NAN, 0.f, 0.f, 64), V).ok, "a NaN penalty: the plain path");
    check(spec_plan(S(0.f, 20, 1.f, 0.f, 0.5f, 0), V).ok && spec_plan(S(0.f, 20, 1.f, 0.f, 0.5f, 0), V).M == 1, "a penalty with window 0 at temperature 0 is the argmax: 1 candidate");
}


// 9. LANE SOLO: the stream of a request that goes from the single-session path to a slot.  A window draws its rows in order and the
// request shows only some of them: the stream before the window advanced by the shown ones is the stream after plain draws of
// those tokens (advance_draws), whatever the window drew past them.
void test_rng_hand() {
    std::printf("random stream hand-over\n");
    Sampling s;
    s.temperature = 1.0f; s.top_k = 8; s.top_p = 0.95f;
    const int M = 8, D = 5, T = D + 1;
    std::vector<int32_t> rows((size_t) T * kSpecRowWords, 0);
    std::mt19937_64 gen(77);
    for (int t = 0; t < T; ++t) {
        int32_t* r = rows.data() + (size_t) t * kSpecRowWords;
        for (int i = 0; i < kSpecTopM; ++i) { r[i] = -1; r[kSpecTopM + i] = 0; }
        for (int i = 0; i < M; ++i) {
            r[i] = 100 * t + i;
            const float v = 10.f - 0.7f * (float) i + (float) (gen() % 100) / 400.f;
            r[kSpecTopM + i] = __builtin_bit_cast(int32_t, v);
        }
    }
    int bad = 0, cases = 0, partial = 0;
    for (unsigned long long seed = 1; seed <= 60; ++seed) {
        // drafts: the row's first candidate for a few rows, then a miss
        std::vector<int32_t> drafts((size_t) D);
        for (int d = 0; d < D; ++d) drafts[(size_t) d] = 100 * d + (d < (int) (seed % 5) ? 0 : 7);
        const std::vector<int32_t> hist(1, 5);
        std::mt19937_64 pre(seed), wr(seed);
        int32_t picks[16];
        const int a = decide_window(s, 8, M, D, hist, drafts.data(), rows.data(), wr, picks);
        for (int shown = 1; shown <= a + 1; ++shown) {
            std::mt19937_64 hand = pre;
            advance_draws(hand, shown);
            // the plain loop: one draw a row, in order, the same rows
            std::mt19937_64 plain(seed);
            for (int t = 0; t < shown; ++t) {
                const int32_t* r = rows.data() + (size_t) t * kSpecRowWords;
                const int p = sample_window_row(r, r + kSpecTopM, M, s, 8, nullptr, 0, plain);
                if (p != picks[t]) { ++bad; std::printf("pick mismatch seed %llu t %d a %d\n", seed, t, a); }
            }
            ++cases;
            if (shown < a + 1) ++partial;   // a stop inside the window: it drew more rows than were shown
            if (hand != plain) { ++bad; std::printf("stream mismatch seed %llu shown %d\n", seed, shown); }
            if (shown == a + 1 && wr != hand) { ++bad; std::printf("window stream mismatch seed %llu a %d\n", seed, a); }   // every draw shown: the window's own stream
        }
    }
    check(cases >= 60 && partial > 0 && bad == 0, "stream before the window + shown draws = the plain loop's stream, and the picks agree (some cases cut inside the window)");
}

}  // namespace

int main() {
    test_keys();
    test_topm_algorithm();
    test_row_vs_plain();
    test_answers();
    test_batch_slots();
    test_statistics();
    test_flash_next_point_mass();
    test_plain_unchanged();
    test_plan();
    test_rng_hand();
    std::printf("%d checks, %d failed\n", g_checks, g_fail);
    std::printf(g_fail == 0 ? "ALL PASSED\n" : "FAILED\n");
    return g_fail == 0 ? 0 : 1;
}
