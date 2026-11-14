// q36_dtree.hpp - a verify tree built from the DFlash 2 drafter's candidates (STRATA_Q36_DFLASH_TREE).  Host only, no SYCL: the tree is
// built on the host from one small gather of the drafter's scores (Gather, ~115 KB), laid out in the order the engine's tree verify
// takes (the DFS order of Engine::spec_step_tree, GDN checkpoints in at most 2 SLM slots), and walked to accept.  tools: dflash_tree_test.
//
// The drafter's candidate selector is a bigram scorer: the score of candidate c of row l (token of position l + 1) after the previous
// token p is  s(c | p) = unary[l][c] + sum_r pred[p][r] hp[l+1][r] succ[cand[l][c]][r],  and softmax over the 16 candidates is a
// conditional distribution given p ALONE.  So a tree is grown by best-first expansion on the cumulative log probability:
//   * the selector's own walk (out_tok, the linear arm's draft) goes in first, as the spine: nodes 1 .. D.  A tree whose budget is D
//     rows is that chain (the same drafts as the linear arm);
//   * then the best pending child of any node (a candidate of row l after the node's token) is added, and expanded, until the node
//     budget, the depth limit or the probability floor stops it.  A node is only kept if the tree still needs <= 2 GDN checkpoint slots.
//
// Checkpoint slots.  k_gdn_dec<true> walks the T nodes in DFS order; a node with several children saves the state after it in a slot
// (sv) and every child but the first loads it (ld); the last child frees the slot before its subtree.  need(u) of a subtree =
// 0 for a leaf, need(child) for one child, and for children sorted by need n_1 <= ... <= n_k: max(1 + n_{k-1}, n_k).  The kernel has 2 slots.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

namespace strata::q36dt {

constexpr int kRows = 16;       ///< the most rows of a verify window (root + 15 nodes)
constexpr int kDepth = 7;       ///< the drafter drafts 7 positions
constexpr int kTopK = 16;       ///< candidates a position
constexpr int kRank = 256;      ///< the selector's rank
constexpr int kMaxSlots = 2;    ///< SLM checkpoint slots of k_gdn_dec<true>
constexpr int kPredRows = 1 + (kDepth - 1) * kTopK;   ///< pred rows of the possible previous tokens: the anchor, then candidates of rows 0 .. 5

/// The device gather (Dflash::tree_gather()): everything the host needs to score any node of the tree.  Fixed layout, no padding.
struct Gather {
    int32_t anchor;
    int32_t out_tok[8];                     ///< the selector's walk, positions 1 .. 7 (+1 pad)
    int32_t cand[kDepth][kTopK];            ///< candidate ids a position
    float unary[kDepth][kTopK];             ///< their raw head logits
    float hp[kDepth][kRank];                ///< hidden_projection rows of positions 1 .. 7
    uint16_t pred[kPredRows][kRank];        ///< bf16 pred_cb rows: [0] the anchor's, [1 + 16 l + c] cand[l][c]'s (l = 0 .. 5)
    uint16_t succ[kDepth][kTopK][kRank];    ///< bf16 succ_cb rows of every candidate
};

inline float bf2f(uint16_t b) { uint32_t u = (uint32_t) b << 16; float f; std::memcpy(&f, &u, 4); return f; }

/// s(c | previous token) of the 16 candidates of position l + 1 (l = 0 .. 6); `pred_row` indexes Gather::pred.  Same order of summation as
/// selector_walk (16 partial sums of 16 products, then added in order).
inline void scores(const Gather& g, int l, int pred_row, float s[kTopK]) {
    float w[kRank];
    for (int t = 0; t < kRank; ++t) w[t] = bf2f(g.pred[pred_row][t]) * g.hp[l][t];
    for (int c = 0; c < kTopK; ++c) {
        float part[16];
        for (int p = 0; p < 16; ++p) {
            float a = 0.f;
            for (int e = 0; e < 16; ++e) a += w[p * 16 + e] * bf2f(g.succ[l][c][p * 16 + e]);
            part[p] = a;
        }
        float v = g.unary[l][c];
        for (int e = 0; e < 16; ++e) v += part[e];
        s[c] = v;
    }
}

struct Node {
    int32_t tok = 0;
    int parent = -1, depth = 0;
    int cidx = -1;       ///< index among cand[depth - 1] (-1: the root)
    float cum = 0.f;     ///< cumulative log probability along the path (root 0)
    bool spine = false;
};

struct Params {
    int max_rows = 16;        ///< window rows incl. the root (<= kRows)
    int max_depth = kDepth;   ///< deepest node (<= the drafts the step may take)
    int spine = kDepth;       ///< nodes of the selector's chain (<= max_depth)
    float min_lp = -3.9f;     ///< ln 0.02: a pending child below this is not added
};

/// Subtree needs of the tree `n` (parents before children): need[u] as in the header comment.  Returns need of the root.
inline int slot_need(const std::vector<Node>& n, std::vector<int>* out = nullptr) {
    const int T = (int) n.size();
    std::vector<std::vector<int>> kn((size_t) T);
    std::vector<int> need((size_t) T, 0);
    for (int u = T - 1; u >= 0; --u) {
        auto& v = kn[(size_t) u];
        std::sort(v.begin(), v.end());
        const int k = (int) v.size();
        need[(size_t) u] = k == 0 ? 0 : (k == 1 ? v[0] : std::max(1 + v[(size_t) k - 2], v[(size_t) k - 1]));
        if (u > 0) kn[(size_t) n[(size_t) u].parent].push_back(need[(size_t) u]);
    }
    if (out) *out = need;
    return need[0];
}

struct Pend { int parent; int c; int32_t tok; float cum; };

/// Grows the tree from the gather.  nodes[0] is the root (the anchor); the nodes are in selection order, every parent before its children.
inline std::vector<Node> build(const Gather& g, const Params& p) {
    std::vector<Node> nodes;
    nodes.push_back(Node{g.anchor, -1, 0, -1, 0.f, true});
    const int max_depth = std::min(p.max_depth, kDepth);
    const int max_rows = std::min(p.max_rows, kRows);
    std::vector<Pend> pool;
    // expands node u: its candidates after its token become pending children
    auto expand = [&](int u, std::vector<Pend>& out) {
        const Node& nu = nodes[(size_t) u];
        if (nu.depth >= max_depth) return;
        const int l = nu.depth;
        const int row = nu.depth == 0 ? 0 : 1 + (nu.depth - 1) * kTopK + nu.cidx;
        float s[kTopK];
        scores(g, l, row, s);
        float m = s[0];
        for (int c = 1; c < kTopK; ++c) m = std::max(m, s[c]);
        float den = 0.f;
        for (int c = 0; c < kTopK; ++c) den += std::exp(s[c] - m);
        const float lse = m + std::log(den);
        for (int c = 0; c < kTopK; ++c) out.push_back(Pend{u, c, g.cand[l][c], nu.cum + (s[c] - lse)});
    };
    // the spine: the selector's own picks, in order
    const int D = std::max(0, std::min({p.spine, max_depth, max_rows - 1}));
    for (int l = 0; l < D; ++l) {
        const int u = (int) nodes.size() - 1;
        std::vector<Pend> kids;
        expand(u, kids);
        int hit = -1;
        for (int c = 0; c < kTopK; ++c) if (g.cand[l][c] == g.out_tok[l]) { hit = c; break; }
        if (hit < 0) break;   // (cannot happen: the walk picks among the candidates)
        Node nn{g.out_tok[l], u, l + 1, hit, 0.f, true};
        nn.cum = kids.empty() ? nodes[(size_t) u].cum : kids[(size_t) hit].cum;
        for (size_t i = 0; i < kids.size(); ++i) if ((int) i != hit) pool.push_back(kids[i]);
        nodes.push_back(nn);
    }
    // the expansion of the last spine node (its children are position D + 1: only when that is within the depth limit)
    expand((int) nodes.size() - 1, pool);
    // best-first
    while ((int) nodes.size() < max_rows && !pool.empty()) {
        size_t bi = 0;
        for (size_t i = 1; i < pool.size(); ++i) if (pool[i].cum > pool[bi].cum) bi = i;
        const Pend e = pool[bi];
        pool.erase(pool.begin() + (long) bi);
        if (e.cum < p.min_lp) break;
        nodes.push_back(Node{e.tok, e.parent, nodes[(size_t) e.parent].depth + 1, e.c, e.cum, false});
        if (slot_need(nodes) > kMaxSlots) { nodes.pop_back(); continue; }
        expand((int) nodes.size() - 1, pool);
    }
    return nodes;
}

/// The window in the DFS order the verify takes (tslot / tprev3 / tmask as k_gdn_dec<true> / k_conv_dec<true> / the attention read them).
struct Layout {
    int T = 0, max_depth = 0, slots = 0;
    bool ok = false;
    int32_t tok[kRows] = {};
    int src[kRows] = {};            ///< the node (selection order) of each DFS row
    int dep[kRows] = {};
    int parent[kRows] = {};         ///< DFS row of the parent (-1: the root)
    uint32_t tmask[kRows] = {};     ///< ancestors incl. the row itself (bit = DFS row)
    uint32_t tprev3[kRows] = {};    ///< (anc[L-3] + 3) | (anc[L-2] + 3) << 5 | (anc[L-1] + 3) << 10 : the conv's three inputs before the row (negative: the history)
    uint32_t tslot[kRows] = {};     ///< ld | sv << 2
    int nchild[kRows] = {};
    int child[kRows][kRows] = {};   ///< DFS rows of the children
    int spine_depth = 0;            ///< the deepest node of the selector's chain
};

inline Layout layout(const std::vector<Node>& n) {
    Layout L;
    const int T = (int) n.size();
    if (T < 1 || T > kRows) return L;
    std::vector<int> need;
    if (slot_need(n, &need) > kMaxSlots) return L;
    std::vector<std::vector<int>> kids((size_t) T);
    for (int u = 1; u < T; ++u) kids[(size_t) n[(size_t) u].parent].push_back(u);
    for (int u = 0; u < T; ++u)
        std::stable_sort(kids[(size_t) u].begin(), kids[(size_t) u].end(), [&](int a, int b) { return need[(size_t) a] < need[(size_t) b]; });   // the largest need last
    int used = 0, next = 0;
    bool bad = false;
    std::vector<int> anc;
    std::function<void(int, int, int)> walk = [&](int u, int ld_s, int pd) {
        const int did = next++;
        anc.push_back(did);
        const int dep = (int) anc.size() - 1;
        L.tok[did] = n[(size_t) u].tok;
        L.src[did] = u;
        L.dep[did] = dep;
        L.parent[did] = pd;
        L.max_depth = std::max(L.max_depth, dep);
        if (n[(size_t) u].spine) L.spine_depth = std::max(L.spine_depth, dep);
        uint32_t mk = 0;
        for (int a : anc) mk |= 1u << a;
        L.tmask[did] = mk;
        const int p1 = dep >= 1 ? anc[(size_t) dep - 1] : dep - 1;
        const int p2 = dep >= 2 ? anc[(size_t) dep - 2] : dep - 2;
        const int p3 = dep >= 3 ? anc[(size_t) dep - 3] : dep - 3;
        L.tprev3[did] = (uint32_t) (p3 + 3) | ((uint32_t) (p2 + 3) << 5) | ((uint32_t) (p1 + 3) << 10);
        if (pd >= 0) L.child[pd][L.nchild[pd]++] = did;
        const auto& ck = kids[(size_t) u];
        int sv_s = 0;
        if (ck.size() > 1) {
            const int sl = (used & 1) == 0 ? 0 : ((used & 2) == 0 ? 1 : -1);
            if (sl < 0) { bad = true; return; }
            used |= 1 << sl;
            sv_s = sl + 1;
            L.slots = std::max(L.slots, (used & 1) + ((used >> 1) & 1));
        }
        L.tslot[did] = (uint32_t) ld_s | ((uint32_t) sv_s << 2);
        for (size_t i = 0; i < ck.size() && !bad; ++i) {
            if (i + 1 == ck.size() && sv_s > 0) used &= ~(1 << (sv_s - 1));   // the last child frees the slot before its subtree
            walk(ck[i], i == 0 ? 0 : sv_s, did);
        }
        anc.pop_back();
    };
    walk(0, 0, -1);
    L.T = T;
    L.ok = !bad && next == T;
    return L;
}

/// The accepted path of a greedy window: `pick[did]` = the target's argmax at DFS row did.  path[0] = 0; returns the accepted drafts a;
/// the draft tokens are L.tok[path[1 .. a]]; the next token is pick[path[a]].
inline int walk_greedy(const Layout& L, const int32_t* pick, int max_drafts, int* path) {
    int cur = 0, a = 0;
    path[0] = 0;
    while (a < max_drafts) {
        int hit = -1;
        for (int i = 0; i < L.nchild[cur]; ++i) if (L.tok[L.child[cur][i]] == pick[cur]) { hit = L.child[cur][i]; break; }
        if (hit < 0) break;
        path[++a] = hit;
        cur = hit;
    }
    return a;
}

}   // namespace strata::q36dt
