#pragma once
// q36_sampling.hpp - strata-q36's host sampler, and the exact choice of the tokens of a verify window for a request that
// the device argmax cannot serve (a penalty, or a temperature above 0).  Header-only and STL only: the host test
// (tests/core/q36_spec_sampling_test.cpp, tools/test_q36_spec_sampling.py) runs it without a GPU.  The key helpers are
// plain integer code that the engine's kernels call too.
//
// The plain path (one token a step) and the window path use the same functions for the penalty of one logit and for the
// draw, so the same row gives the same token and the same random draw.
//
// What a window row hands the host: the M highest RAW logits of the row, ranked by (logit, then lowest token id), M up
// to kSpecTopM.  The host applies the request's penalties to the listed tokens that are in the row's penalty window,
// ranks again, keeps the sampler's candidate cap kk, and draws.  That is the plain path's candidate list when
//     M >= kk + (the penalized tokens of the row)  and  every penalty only lowers a logit
// (repeat >= 1, frequency >= 0, presence >= 0): a token outside the raw top M has at least kk raw predecessors that no
// penalty touches, so it is not in the penalized top kk.  spec_plan() refuses the requests for which that does not hold.
#include "strata/qwen35/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace strata::qwen35 {

constexpr int kSpecTopM = 128;                  // the most raw candidates one row of a window returns
constexpr int kSpecRowWords = 2 * kSpecTopM;    // a row as the engine copies it: kSpecTopM ids, then their logits' bits
constexpr int kSpecChunk = 4096;                // the logits of one row that one work-group of the first stage takes
constexpr int kSpecMaxCand = 64;                // the sampler's candidate cap (top_k above it is not drafted)
constexpr int kSpecMaxWin = 64;                 // the widest penalty window a window takes

// ---------------------------------------------------------------------------------------------------------------- keys
// A float as an unsigned key that orders like the float (NaN below everything, -0 as +0), so that a 64-bit maximum
// is "the highest logit, the lowest id on a tie".  Key 0 never belongs to a real entry.  Also used by the kernels.
inline uint32_t spec_key32(float v) {
    uint32_t u = __builtin_bit_cast(uint32_t, v);
    if ((u & 0x7fffffffu) > 0x7f800000u) u = 0xff800000u;
    else if ((u << 1) == 0u) u = 0u;
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
inline float spec_key_float(uint32_t k) {
    return __builtin_bit_cast(float, (k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}
inline uint64_t spec_key64(uint32_t key32, uint32_t id) { return ((uint64_t) key32 << 32) | (uint64_t) (0xffffffffu - id); }
inline int32_t spec_key_id(uint64_t k) { return (int32_t) (0xffffffffu - (uint32_t) (k & 0xffffffffu)); }
inline uint32_t spec_key_hi(uint64_t k) { return (uint32_t) (k >> 32); }

// ------------------------------------------------------------------------------------------------------ the sampler
struct Cand {
    int32_t id;
    float v;
};
/// The candidate order: the higher logit first, the lower id on a tie.
inline bool cand_before(const Cand& a, const Cand& b) { return a.v > b.v || (a.v == b.v && a.id < b.id); }

/// Whether the request's penalties do anything (the plain sampler's own condition).
inline bool penalty_active(const Sampling& s) {
    return s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f);
}

/// The sampler's candidate cap: top_k, or 64 for 0 or less, at most the vocabulary, at least 1.
inline int cand_cap(const Sampling& s, int vocab) { return std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, vocab)); }

/// One logit after the penalties of a token that is `c` times in the window.  The plain path and the window path both
/// call this one function (kept out of line, so both run one compiled copy of the float expression).
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
inline float penalized(float v, int c, const Sampling& s) {
    if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
    v -= (float) c * s.freq + s.present;
    return v;
}

/// The penalties on a full logit row for the tokens of the last s.last_n (all when < 0) history tokens.
inline void apply_penalties_row(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist) {
    if (!penalty_active(s)) return;
    std::vector<int32_t> w;
    const size_t from = s.last_n < 0 || (size_t) s.last_n >= hist.size() ? 0 : hist.size() - (size_t) s.last_n;
    w.assign(hist.begin() + (long) from, hist.end());
    std::sort(w.begin(), w.end());
    for (size_t i = 0; i < w.size();) {
        size_t j = i;
        while (j < w.size() && w[j] == w[i]) ++j;
        const int id = w[i];
        if (id >= 0 && (size_t) id < lg.size()) lg[(size_t) id] = penalized(lg[(size_t) id], (int) (j - i), s);
        i = j;
    }
}

/// The draw: temperature, then min_p and top_p over the ranked candidates (at most the cap), then one draw of `rng`.
/// `top` is ranked by cand_before and already cut to the cap.  One call takes exactly one draw.
inline int draw(const std::vector<Cand>& top, const Sampling& s, std::mt19937_64& rng) {
    std::vector<double> p(top.size());
    const double mx = top[0].v;
    double sum = 0;
    for (size_t i = 0; i < top.size(); ++i) sum += (p[i] = std::exp((top[i].v - mx) / s.temperature));
    size_t keep = top.size();
    double cum = 0;
    for (size_t i = 0; i < top.size(); ++i) {
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
        if ((r -= p[i]) <= 0) return top[i].id;
    }
    return top[keep - 1].id;
}

/// The plain path: one token from a full logit row (it is changed: the penalties are applied in place).  A temperature of 0 or
/// less is the argmax (the lowest id on a tie).
inline int sample_row(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    apply_penalties_row(lg, s, hist);
    if (s.temperature <= 0.f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const int kk = cand_cap(s, (int) lg.size());
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) {
        return cand_before(Cand{a, lg[(size_t) a]}, Cand{b, lg[(size_t) b]});
    });
    std::vector<Cand> top((size_t) kk);
    for (int i = 0; i < kk; ++i) top[(size_t) i] = Cand{idx[(size_t) i], lg[(size_t) idx[(size_t) i]]};
    return draw(top, s, rng);
}

// -------------------------------------------------------------------------------------------------------- the window
/// What a request needs of a verify window.
struct SpecPlan {
    bool ok = false;        // the window path serves this request exactly
    bool sampled = false;   // temperature above 0 (else: the argmax of the penalized row)
    int kk = 1;             // the candidate cap
    int M = 1;              // raw candidates a row must return
};

/// Whether a request that the device argmax cannot serve (Sampling::needs_logits()) can draft, and what its rows return.
/// Not drafted (the plain path serves them): a window that is not 1 .. kSpecMaxWin tokens, a penalty that raises a logit
/// (repeat below 1, a negative frequency or presence), and a candidate cap above kSpecMaxCand.
inline SpecPlan spec_plan(const Sampling& s, int vocab) {
    SpecPlan p;
    const bool pen = penalty_active(s);
    if (pen && !(s.last_n >= 1 && s.last_n <= kSpecMaxWin && s.repeat >= 1.f && s.freq >= 0.f && s.present >= 0.f)) return p;
    p.sampled = s.temperature > 0.f;
    p.kk = p.sampled ? cand_cap(s, vocab) : 1;
    if (p.kk > kSpecMaxCand) return p;
    p.M = p.kk + (pen ? s.last_n : 0);
    p.ok = p.M <= kSpecTopM && p.M <= vocab;
    return p;
}

/// Where row t's penalty window starts in [the last min(len, last_n) history tokens, then the drafts]: the window is the
/// last last_n tokens of (history, drafts 0 .. t - 1).  n_tail = min(len, last_n).
inline int window_start(int n_tail, int t, int last_n) { return std::max(0, n_tail + t - last_n); }

/// One row's token: `ids` / `bits` the row's M raw candidates (ids below 0 end the list), `win` the n tokens of the row's
/// penalty window.  A temperature of 0 or less gives the argmax of the penalized row, else one draw of `rng`.
inline int sample_window_row(const int32_t* ids, const int32_t* bits, int M, const Sampling& s, int kk, const int32_t* win,
                             int n, std::mt19937_64& rng) {
    std::vector<Cand> c;
    c.reserve((size_t) M);
    for (int i = 0; i < M && i < kSpecTopM; ++i) {
        if (ids[i] < 0) break;
        float v = __builtin_bit_cast(float, bits[i]);
        int cnt = 0;
        for (int j = 0; j < n; ++j) cnt += win[j] == ids[i];
        if (cnt > 0) v = penalized(v, cnt, s);
        c.push_back(Cand{ids[i], v});
    }
    if (c.empty()) return 0;
    std::sort(c.begin(), c.end(), cand_before);
    if ((int) c.size() > kk) c.resize((size_t) kk);
    if (s.temperature <= 0.f) return c[0].id;
    return draw(c, s, rng);
}

/// The tokens of a verify window T = D + 1 rows.  Row t is the model's output after window token t (token 0 is the real next
/// token, tokens 1 .. D the drafts), so its pick is the token that follows the history, token 0 .. t.  Row t is drawn only
/// after rows 0 .. t - 1 matched their drafts: row t's pick is compared with draft t (the window's token t + 1) and the first
/// row that does not match ends the window.  Each row draws once (a greedy row not at all), in order, as the plain path does.
/// `hist`: the tokens before the window including token 0; `drafts`: D ids; `rows`: T rows of kSpecRowWords words.
/// Returns the accepted drafts a and writes picks[0 .. a] (picks[a]: the token that follows the accepted ones).
/// Exact: row t's pick has the distribution the plain sampler has for that position; the draft only decides whether it
/// is kept (a point-mass proposal: it is accepted with probability p(d), which is the min(1, p / q) rule for q a point
/// mass, and a rejected row's pick is a draw from p given that it differs from d).
inline int decide_window(const Sampling& s, int kk, int M, int D, const std::vector<int32_t>& hist, const int32_t* drafts,
                         const int32_t* rows, std::mt19937_64& rng, int32_t* picks) {
    const bool pen = penalty_active(s);
    std::vector<int32_t> seq;
    int n_tail = 0;
    if (pen) {
        n_tail = (int) std::min<size_t>(hist.size(), (size_t) s.last_n);
        seq.assign(hist.end() - n_tail, hist.end());
        seq.insert(seq.end(), drafts, drafts + D);
    }
    for (int t = 0; t <= D; ++t) {
        const int32_t* ids = rows + (size_t) t * kSpecRowWords;
        const int from = pen ? window_start(n_tail, t, s.last_n) : 0;
        const int n = pen ? n_tail + t - from : 0;
        picks[t] = sample_window_row(ids, ids + kSpecTopM, M, s, kk, pen ? seq.data() + from : nullptr, n, rng);
        if (t == D || picks[t] != drafts[t]) return t;
    }
    return D;
}

// --------------------------------------------------------------------------------------------- a slot of a batch window
/// How many of a batch slot's D drafts stand, given the picks of its rows (picks[t]: the token the row t gives; picks[0 .. a]
/// are the tokens the slot emits).  Draft a stands when the pick before it is that draft and the request can still take it:
/// the pick is not an end-of-answer token, no stop was asked for, max_new and the context have room.  Same rule as the greedy
/// slot's loop in the engine's window_mtp (`produced`: tokens the slot has emitted; `pos`: the cell of its next token).
template <class IsEos>
inline int batch_accept_count(const int32_t* picks, const int32_t* drafts, int D, bool stop, int64_t produced, int64_t max_new,
                              int64_t pos, int64_t ctx, IsEos is_eos) {
    int a = 0;
    for (int y = picks[0]; a < D && y == drafts[a] && !is_eos(y) && !stop && produced + a + 2 <= max_new && pos + a + 3 <= ctx;
         y = picks[a])
        ++a;
    return a;
}

/// One batch slot's part of a verify window for a request that needs the host's draw (a temperature above 0, or a penalty):
/// the slot's own sampler, history and random stream, its rows' candidates (`rows`: the slot's first row, kSpecRowWords words
/// a row, as the engine copies them; the list may be longer than the slot's M: the first M of it are the slot's, the list is a
/// total order), and the drafts it offered.  `hist` ends with the slot's next token (the window's token 0), as decide_window
/// wants it.  Writes picks[0 .. a] and returns a, the drafts that stand.  Exactly the single-session path's decision
/// (decide_window), then the batch loop's limits.  decide_window draws one row after the last matching one, so a stop, max_new,
/// an end-of-answer token or the context can end the slot a few rows before the draws did; each of those ends the request,
/// whose random stream is not used again.
template <class IsEos>
inline int batch_slot_step(const Sampling& s, const SpecPlan& p, const std::vector<int32_t>& hist, const std::vector<int32_t>& drafts,
                           const int32_t* rows, std::mt19937_64& rng, int32_t* picks, bool stop, int64_t produced, int64_t max_new,
                           int64_t pos, int64_t ctx, IsEos is_eos) {
    const int D = (int) drafts.size();
    decide_window(s, p.kk, p.M, D, hist, drafts.data(), rows, rng, picks);
    return batch_accept_count(picks, drafts.data(), D, stop, produced, max_new, pos, ctx, is_eos);
}

/// The M highest raw logits of a row as the device returns them: ranked by (logit, lowest id), ids then bits, padded with
/// -1 and 0.  The reference for the kernels (STRATA_Q36_SPEC_CHECK compares them) and for the host test.
inline void spec_topm_ref(const float* row, int vocab, int M, int32_t* ids, int32_t* bits) {
    std::vector<uint64_t> k((size_t) vocab);
    for (int i = 0; i < vocab; ++i) k[(size_t) i] = spec_key64(spec_key32(row[i]), (uint32_t) i);
    const int m = std::min(M, vocab);
    std::partial_sort(k.begin(), k.begin() + m, k.end(), [](uint64_t a, uint64_t b) { return a > b; });
    for (int r = 0; r < M; ++r) {
        if (r < m) {
            ids[r] = spec_key_id(k[(size_t) r]);
            bits[r] = __builtin_bit_cast(int32_t, spec_key_float(spec_key_hi(k[(size_t) r])));
        } else {
            ids[r] = -1;
            bits[r] = 0;
        }
    }
}

}  // namespace strata::qwen35
