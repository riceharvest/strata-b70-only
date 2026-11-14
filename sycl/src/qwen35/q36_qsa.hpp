// q36_qsa.hpp - Qwen Sparse Attention (QSA) for the 27B's 16 full-attention layers on the Arc Pro B70 (q36_qsa.dp.cpp).
//
// What QSA is (docs/QSA.md in REPORT.md's words): a learned indexer scores the context in blocks of 4 cells and each query row attends
// only to its best 512 blocks (2,048 cells) plus the cells of its own block.  The 48 GDN layers and the MTP block are unchanged.
// Per attention layer L the indexer has q_proj [4 heads x 128][5120], k_proj [128][5120] and three RMS-norm multipliers [128]
// (qsa_extract.py's file).  Per cell: raw key = rms(k_proj . h) * k_pre; block b (cells 4b..4b+3) is pooled when its 4th cell is
// processed: rms(mean of the 4 raw keys) * k_norm, rotated (NEOX, first 64 of 128 dims) to the position of the block's first cell.
// Per query row at position p: q_h = rope(rms(q_proj_h . h) * q_norm) for h = 0..3; the score of a COMPLETE block b (4b + 3 <= p) is
// sum_h relu(q_h . pooled_b) / sqrt(128); the row keeps the 512 best (ties: the lower block index), plus its own block's cells <= p.
// A row with at most 512 complete blocks keeps all of them: its attention is the dense one.  Selection is per ROW and shared by the
// 24 query heads.  (The reference's mask form takes the union of the selected blocks and the own block; a block is never counted
// twice - the reference's fast path counts the own block twice when p % 4 == 3 and it is selected; this code does what the mask form does.)
//
// Everything here reads the rows' positions from DEVICE memory (a replayed command graph never sees a host position) and launches over
// the whole cache capacity, returning early past the rows' blocks.  Every reduction has a fixed order that does not depend on the
// window's row count: a row of a verify window is bitwise the row of a one-row decode step.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::q36qsa {

constexpr int kEmbd = 5120;       // hidden size
constexpr int kD = 128;           // indexer key / query dimension
constexpr int kHeads = 4;         // indexer heads
constexpr int kProj = kHeads * kD + kD;   // 640 outputs of one layer's two projections: q rows 0..511, then the key
constexpr int kRatio = 4;         // cells per block
constexpr int kTopBlocks = 512;   // blocks a row keeps (top_k 2048 / 4)
constexpr int kRing = 32;         // raw keys kept for the last cells (a verify window's 16 rows + a block's 3 earlier cells)
constexpr int kMaxWin = 16;       // rows of a decode / verify window (Engine::kVerify)
constexpr int kMaxTail = 5;       // blocks of a row's own region: a tree window's 3 committed cells + 16 rows is 19 cells
constexpr int kPieceItems = 64;   // blocks (items) of one work-group of the sparse attention
constexpr int kPieces = 9;        // ceil((512 + kMaxTail) / kPieceItems)
constexpr int kHD = 256;          // main attention head dimension
constexpr int kG = 6;             // query heads per KV head
constexpr int kScoreRows = 8;     // rows of one scoring launch (the accumulators live in registers)
constexpr float kInvSqrtD = 0.08838834764831845f;   // 1 / sqrt(128) as a float constant (the same bits on the host and the card)

/// One attention layer's indexer weights (device pointers; the weights are fp16, the norm multipliers fp32).
struct LayerWeights {
    const void* wq = nullptr;        // sycl::half [512][5120]
    const void* wk = nullptr;        // sycl::half [128][5120]
    const float* qn = nullptr;       // [128]
    const float* kpre = nullptr;     // [128]
    const float* knorm = nullptr;    // [128]
};
struct Weights {
    std::vector<int> layer_ids;      // the model's layer numbers (3, 7, ..., 63)
    std::vector<LayerWeights> w;
    size_t bytes = 0;
    bool from_f32 = false;           // the file held fp32 weights (rounded to fp16 on load)
};
/// Reads a qsa_extract.py file onto the device (stream: a sycl::queue*).  false: `err` says why.
bool load_weights(const char* path, Weights& out, void* stream, std::string& err);
void free_weights(Weights& w, void* stream);

/// One layer's indexer state.  pooled: sycl::half [nb_cap][128]; ring: float [kRing][128] (raw keys by cell % kRing); traw: float
/// [kMaxWin][128] (a tree window's raw keys until the window is committed).
struct LayerState {
    void* pooled = nullptr;
    float* ring = nullptr;
    float* traw = nullptr;
};
inline int nb_cap_for(int ctx) { return (ctx + kRatio - 1) / kRatio; }
size_t state_bytes(int ctx);                       // one layer's LayerState
/// A layer's state on the device, zeroed.
bool alloc_state(LayerState& s, int ctx, void* stream);
void free_state(LayerState& s, void* stream);
void copy_state(const LayerState& src, const LayerState& dst, int cells, int ctx, void* stream);   // a session's indexer state (blocks below cells / 4 + the ring)

/// Scratch shared by every layer (the kernels of a layer run one after the other on the in-order queue).
struct Scratch {
    float* proj = nullptr;       // [tmax][kProj]
    float* qidx = nullptr;       // [tmax][4][128]
    float* raw = nullptr;        // [tmax][128]
    float* score = nullptr;      // [score_rows][nb_cap]
    int32_t* sel = nullptr;      // [score_rows][512]  the selected blocks of a row, ascending
    int32_t* nsel = nullptr;     // [score_rows]
    int32_t* nitems = nullptr;   // [score_rows]       items (blocks) a row's attention walks
    float* pacc = nullptr;       // [kMaxWin * 24][kPieces][256]
    float* pm = nullptr;         // [kMaxWin * 24][kPieces]
    float* pl = nullptr;
    int tmax = 0, score_rows = 0, nb_cap = 0;
};
size_t scratch_bytes(int tmax, int score_rows, int ctx);
bool alloc_scratch(Scratch& s, int tmax, int score_rows, int ctx, void* stream);
void free_scratch(Scratch& s, void* stream);

/// The rows of one call.  Row t is the cell *pos0 + t (+ row0 when a prompt chunk goes through in tiles); its rotary position is
/// the cell plus delta_rows[t] (a tree window) or *delta (else, null: 0).  Tree windows: tp0 = the root cell (device scalar) and
/// tmask[t] = the ancestors of row t as a bitmask over the window's cells (bit i: cell *tp0 + i).
struct Win {
    const int32_t* pos0 = nullptr;
    const int32_t* delta = nullptr;
    const int32_t* delta_rows = nullptr;
    const int32_t* tp0 = nullptr;
    const uint32_t* tmask = nullptr;
    int T = 0;                   // rows of the forward (the pooling covers them all)
    int row0 = 0;                // first row of this call's tile
    int nr = 0;                  // rows of this call's tile (<= kScoreRows for score, <= tmax for the rest)
};

struct Rope { float base = 1e7f; int n_rot = 64; float eps = 1e-6f; };

/// Step 1, every forward: the indexer's projections of h [T][5120] (the layer's normed input, fp32), the per-row norms and rotation of
/// the queries (-> sc.qidx), the raw keys (-> sc.raw, and the ring by cell, or lyr.traw for a tree window: tree = true) and the pooling
/// of every block the rows complete (not for a tree window: pool_tree() does it after the commit).
void index_rows(const LayerWeights& w, const LayerState& lyr, const Scratch& sc, const float* h, const Win& win, const Rope& rp, bool tree,
                void* stream);
/// Step 2: scores of the window's rows over their visible blocks and each row's 512 best (sc.sel, sc.nsel).  For the rows
/// [win.row0, win.row0 + win.nr) with qidx rows relative to win.row0 and the outputs (score, sel, nsel) relative to the call.
void select_rows(const LayerState& lyr, const Scratch& sc, const Win& win, void* stream);
/// Step 3: the sparse attention of the rows (q, gate, out: [nt][24][256], pointers already at row win.row0) over the 8-bit K / V
/// cache (K: codes [HKV][ctx][256] then fp16 scales [HKV][ctx][4]; V: the engine's transposed blocks of 32 cells, then fp32 scales
/// [HKV][vt]) -> out = the merged result times sigmoid(gate).
struct AttendArgs {
    const float* q = nullptr;
    const float* gate = nullptr;
    float* out = nullptr;
    const void* kc = nullptr;
    const void* vc = nullptr;
    int ctx = 0, vt = 0, H = 24, HKV = 4;
    const int32_t* sel = nullptr;
    const int32_t* nsel = nullptr;
    int32_t* nitems = nullptr;
    float* pacc = nullptr; float* pm = nullptr; float* pl = nullptr;
};
void attend_rows(const AttendArgs& a, const Win& win, void* stream);
/// After a tree window is committed along `path` (device int32 [n_keep], window row of each kept cell): the kept rows' raw keys into the
/// ring and the blocks they complete pooled.  p0 = the root cell (device scalar).
void pool_tree(const LayerWeights& w, const LayerState& lyr, const int32_t* p0, const int32_t* path, int n_keep, const Rope& rp,
               const int32_t* delta, void* stream);

}   // namespace strata::q36qsa
