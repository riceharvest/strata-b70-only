// q36_dflash.hpp - the DFlash 2 block drafter (incoai/Qwen3.8-27B-DFlash2) for the dense Qwen3.8-27B on the Arc Pro B70.
//
// One forward of the drafter gives 7 draft tokens: the block is 8 rows (the token the target just chose, then 7 mask tokens), the
// 5 layers read the target's hidden states as context, attention inside the block is NOT causal, two-tap dynamic convolutions
// run before and after attention and the MLP, and a candidate selector walks the head's top-16 of each row left to right.
// tools/qwen35/dflash_ref.py is the spec and the CPU reference (PLAN.md D1-D9 in the lane's notes).
//
//   weights   int4 g128 records of tools/qwen35/b70w4_pack.py (tools/qwen35/dflash_pack.py makes the pack): every product is
//             q36xr::mmvq_w4h (W4A16, <= 32 columns, one weight pass)
//   context   the engine's forward() calls capture() after target layers 5 19 33 47 61 (the residual stream, bf16) into a staging
//             ring of 2064 positions; before a draft the committed rows not yet in the drafter's K/V ring are flushed through
//             fc -> hidden_norm -> k/v projections -> k_norm + RoPE -> the ring (fp16), also 2064 positions
//   block     prepare() stages the positions, RoPE tables and tokens in one small device array; run() enqueues the context
//             update (n rows, n = 0 .. 8), the 5 layers on the 8 block rows, the target head on rows 1-7, the top-16 and the
//             selector walk.  run() reads nothing from the host and allocates nothing: the engine records it in a graph per n.
//   result    7 token ids on the device (out_tokens()) and the selector's probability of each pick
//
// New behaviour: the engine uses this only with STRATA_Q36_DFLASH=<pack>; unset, nothing here runs.
#pragma once
#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "q36_dtree.hpp"

namespace strata::q36df {

constexpr int kN = 5120, kFF = 17408, kLayers = 5, kHeads = 32, kKvHeads = 8, kHd = 128;
constexpr int kRows = 8, kDrafts = 7, kTaps = 5, kRank = 256, kTopK = 16;
constexpr int kWindow = 2048;   ///< a row sees positions with |q - k| < 2048
constexpr int kRing = 2064;     ///< ring positions (window + slack for rows a rejected draft wrote ahead)
constexpr int kEagerCols = 32;  ///< rows one eager context flush takes (one weight pass)
constexpr int kKpOut = 1280;    ///< kernel_projection outputs: side x tap x group = 2 x 2 x 320
constexpr int kChunks = 17;     ///< attention chunks: 16 x 128 context keys + the 8 block keys
constexpr int kMaxNd = 248320;  ///< the most logits a head row can have
constexpr int kMaxBlocks = 4;   ///< blocks one batch pass drafts for (4 x 8 rows = 32 columns, the most one weight pass takes)
constexpr size_t kStagElems = (size_t) kRing * kTaps * kN;                 ///< one slot's staging ring (bf16 elements)
constexpr size_t kRingElems = (size_t) kLayers * kKvHeads * kRing * kHd;   ///< one slot's K (or V) ring (fp16 elements)
/// The block's product classes (STRATA_Q36_DFLASH_W4C=ckp,q,kv,o,gate_up,down,hproj: the threads a tile of each, 0 = the engine's own form).
enum ProdClass { kCkp = 0, kCq, kCkv, kCo, kCgu, kCdn, kCh, kCNum };

/// The device meta's layout (int32 / float words), staged by prepare().  Per block b (up to kMaxBlocks: the batch passes draft for several slots at once): [kMetaP + b]
/// the block's first position, [kMetaAnchor + b] its anchor token, [kMetaVlo + b] the first valid context position, [kMetaBSlot + b] the slot whose rings it reads.
/// Context column j (up to 32 of a pass): [kMetaCtxPos + j] its position, [kMetaCtxSlot + j] its ring cell, [kMetaCtxId + j] its slot; its RoPE (cos[64], sin[64]) at
/// kMetaRopeCtx + 128 j.  Block row r (8 a block): [kMetaTok + r] its token, its RoPE at kMetaRopeBlk + 128 r.
namespace k_consts {
constexpr int kMetaP = 0, kMetaAnchor = 4, kMetaVlo = 8, kMetaBSlot = 12, kMetaN = 16, kMetaCtxPos = 32, kMetaCtxSlot = 64, kMetaCtxId = 96, kMetaTok = 128,
              kMetaRopeCtx = 192, kMetaRopeBlk = 4288, kMetaWords = 8384;
}

/// The kernels, public for the parity program (q36_mmvq_dense_parity --dflash).  Layouts: q36_dflash.dp.cpp's header.
namespace k {
void capture(sycl::queue& q, uint16_t* stag, int tap, const float* x, int T, const int32_t* pos0);
/// Batch rows (rows_mode): row r is cell rcell[r] of slot rslot[r]; stag = slot 0's staging.
void capture_rows(sycl::queue& q, uint16_t* stag, int tap, const float* x, int T, const int32_t* rcell, const int32_t* rslot);
void ctx_prep(sycl::queue& q, const uint16_t* stag, const int32_t* meta, int n, uint16_t* x16, float* scale, float* xf = nullptr);
void ctx_norm(sycl::queue& q, const float* y, const float* scale, const float* w, float eps, int n, uint16_t* x16, float* hf = nullptr);
void ctx_store(sycl::queue& q, const float* kv, const int32_t* meta, int n, const float* const* kn, sycl::half* kring, sycl::half* vring, float eps);
// the block kernels take `rows` = 8 x the blocks of the pass (8 .. 32); a block is its own 8 rows (the conv taps and the attention stop at its edge)
void rms_pack(sycl::queue& q, const float* x, const float* w, float eps, float* h, uint16_t* x16, int rows = kRows);
void conv_pack(sycl::queue& q, const float* x, const float* dyn, int side, const float* base, uint16_t* x16, int rows = kRows, float* yf = nullptr);
void finish_rms(sycl::queue& q, const float* o, const float* dyn, const float* base, float* x, const float* w, float eps, float* h, uint16_t* x16, int rows = kRows);
void qk_prep(sycl::queue& q, const float* qy, const float* kvy, const float* qn, const float* kn, const int32_t* meta, float eps, float* qb, float* kb, float* vb, int rows = kRows);
/// part buffers: [block][chunk][head][row] (m, l) and [..][128] (acc)
void attn_part(sycl::queue& q, const float* qb, const float* kb, const float* vb, const sycl::half* kring, const sycl::half* vring, int layer,
               const int32_t* meta, float* part_acc, float* part_m, float* part_l, int nb = 1);
void attn_merge(sycl::queue& q, const float* part_acc, const float* part_m, const float* part_l, uint16_t* x16, int nb = 1, float* yf = nullptr);
/// rows 1 .. 7 of each block of h [nb x 8][kN] -> hn [nb x 7][kN]
void gather_hn(sycl::queue& q, const float* h, float* hn, int nb);
void top1(sycl::queue& q, const float* logits, int nd, int rows, int nch, uint64_t* part);
void top2(sycl::queue& q, const uint64_t* part, int nch, int rows, const int32_t* idmap, int32_t* cand, float* unary);
/// hp [nb x 8][256], cand / unary [nb x 7][16]; block b walks from meta anchor b and writes out_tok / out_prob [b][7]
void selector_walk(sycl::queue& q, const float* hp, const int32_t* cand, const float* unary, const uint16_t* pred, const uint16_t* succ, const int32_t* meta,
                   int32_t* out_tok, float* out_prob, int nb = 1);
/// STRATA_Q36_DFLASH_TREE: the host image of everything the tree builder scores with (q36_dtree.hpp's Gather), one block: anchor, the selector's picks, candidates,
/// raw logits, hp rows 1 .. 7, the bf16 pred_cb rows of the possible previous tokens and the succ_cb rows of every candidate.
void gather_tree(sycl::queue& q, const int32_t* out_tok, const int32_t* cand, const float* unary, const float* hp, const uint16_t* pred, const uint16_t* succ,
                 const int32_t* meta, q36dt::Gather* g);
}   // namespace k

/// The target's pieces the drafter borrows (the engine fills them).
struct Hooks {
    /// fp32 embedding rows [n][kN] of n device token ids.
    std::function<void(const int32_t* tok, int n, float* out)> embed;
    /// The head over n rows hn [n][kN] (device): logits [n][nd] into `logits` (capacity n * max_nd); sets nd and the id map (device
    /// int32 [nd] giving each column's token id, or null: the column is the id).
    std::function<void(const float* hn, int n, float* logits, int& nd, const int32_t*& idmap)> head;
    int max_nd = kMaxNd;
};

/// What debug_last() returns (STRATA_Q36_DFLASH_DUMP).
struct Debug {
    std::vector<float> embed;       ///< [2][kN]: the anchor's and the mask token's embedding
    std::vector<float> hn;          ///< [8][kN]: the final-normed block rows
    std::vector<int32_t> cand;      ///< [7][16]
    std::vector<float> unary;       ///< [7][16]
    std::vector<int32_t> path;      ///< [7]
    std::vector<float> prob;        ///< [7]
};

class Dflash {
public:
    Dflash() = default;
    ~Dflash();
    Dflash(const Dflash&) = delete;
    Dflash& operator=(const Dflash&) = delete;

    /// Reads the pack (tools/qwen35/dflash_pack.py) to the device and allocates the rings and buffers (~1.4 GB).  `stage_rows`: the
    /// most rows one forward() call captures (the engine's chunk; larger calls keep their last kRing rows).
    bool load(sycl::queue& q, const std::string& pack_path, std::string& err, int n_slots = 1);
    /// A random pack's worth of weights (parity / bench: no file; ~1.2 GB of random incompressible records).
    bool load_synthetic(sycl::queue& q, uint64_t seed, std::string& err, int n_slots = 1);
    bool loaded() const { return blob_ != nullptr; }
    void set_hooks(sycl::queue& q, const Hooks& h);   ///< allocates the logits scratch
    void enable_debug(bool on) { debug_ = on; }
    int mask_token() const { return mask_token_; }
    uint64_t vram_bytes() const { return vram_; }

    // ---- target side (the engine calls these)
    /// Inside forward(): rows 0 .. T-1 of the residual stream x [T][kN] (fp32) after target layer tap_layer(tap) take positions
    /// *pos0 + r of the staging ring (bf16).  Recorded in the verify graph; pos0 is the device position the graph reads.
    void capture(sycl::queue& q, int tap, const float* x, int T, const int32_t* pos0_dev, int slot = 0);
    /// The same for batch rows (rows_mode): row r is cell rcell_dev[r] of slot rslot_dev[r] (the engine's device tables).
    void capture_rows(sycl::queue& q, int tap, const float* x, int T, const int32_t* rcell_dev, const int32_t* rslot_dev);
    static int tap_layer(int k) { return 5 + 14 * k; }   ///< 5 19 33 47 61 (0-based layer whose output is tapped)
    /// Host bookkeeping, after the rows [p0, p0 + n) of the history were forwarded and COMMITTED (after the accept decision).
    void note_forward(int64_t p0, int64_t n, int slot = 0);
    /// The history is another one (a restore, a copied session, a reset): the slot's context is dropped.
    void invalidate(int slot = 0);

    // ---- draft
    /// Plans the draft whose block starts at position P (every row below P is committed; `anchor` is the token at P): flushes
    /// all but the last <= 8 pending context rows eagerly, stages the meta of the rest.  Returns n, the context rows run() takes
    /// (0 .. 8); the engine keys its graph on it.  Async on q except for the eager flushes' small waits.
    int prepare(sycl::queue& q, int64_t P, int32_t anchor, int slot = 0);
    /// One batch pass for several slots (STRATA_Q36_DFLASH_BATCH): the block of request b starts at position P (anchor token) of its slot.  Flushes each slot's
    /// older pending rows eagerly and stages the meta of every block; `cols` gets each block's context rows for the pass (0 .. 8, 32 in all at most).  false: more
    /// than kMaxBlocks requests, or a slot that does not exist.
    struct Req { int slot; int64_t P; int32_t anchor; };
    bool prepare_batch(sycl::queue& q, const Req* reqs, int nb, int* cols);
    /// The pass of nb blocks after prepare_batch: the context update (eager, `cols` columns in all) and the blocks.  Out: out_tokens() [nb][7].
    void run_batch(sycl::queue& q, int nb, int cols_total);
    /// The block pass alone for nb blocks (recordable: the engine keys a graph on nb).
    void block_batch(sycl::queue& q, int nb);
    /// Enqueues the context update (n rows) and the block.  Reads only device memory; recordable.
    void run(sycl::queue& q, int n);
    const int32_t* out_tokens() const { return out_tok_; }   ///< device [nb][7]
    const float* out_probs() const { return out_prob_; }     ///< device [7]
    // ---- STRATA_Q36_DFLASH_TREE (the engine turns it on; off: nothing below runs)
    /// On: a single block pass also gathers the scores of its candidates (tree_gather(), device) for the host's tree builder.  Allocates once.
    bool enable_tree(sycl::queue& q, bool on);
    bool tree_on() const { return tree_ && tgath_ != nullptr; }
    const q36dt::Gather* tree_gather() const { return tgath_; }   ///< device; valid after run() of a single block with the tree on
    /// After a tree verify of `n_keep` rows whose row r (the window's DFS order) was captured at position p0 + r: the kept rows path[k] (DFS rows, path[0] = 0) move to
    /// position p0 + k of the slot's staging ring (in order, on q).  Rows the capture wrote for rejected nodes are left (never read: the host mirror ends at p0 + n_keep).
    void compact_stage(sycl::queue& q, int64_t p0, const int32_t* path, int n_keep, int slot = 0);
    // ---- STRATA_Q36_DFLASH_W4X (default off): the products of 16 .. 32 columns (batch passes, big context flushes) run q36xr::mmvq_w4x (u4 x s8 DPAS, q36_w4x.dp.cpp from b70-w4x)
    // on activations quantized to int8 (a scale a group of 128 inputs) instead of mmvq_w4h on fp16 operands.  The 8-column pass keeps mmvq_w4h.  NOT bitwise to the w4h forms.
    bool set_w4x(sycl::queue& q, bool on, int min_cols = 16);
    bool w4x_on() const { return w4x_; }
    /// Debug: the final-normed rows h [rows][kN] of the last block pass (waits).
    bool debug_h(sycl::queue& q, int rows, std::vector<float>& out);
    /// Debug: the last block's intermediates and candidates (waits).
    bool debug_last(sycl::queue& q, Debug& d);
    /// Debug: staging rows of positions [p0, p0 + n) as bf16 [n][5 x kN] (waits); false when a position is not in the ring.
    bool debug_stage_rows(sycl::queue& q, int64_t p0, int n, std::vector<uint16_t>& out);
    /// The first context position the last prepare() left valid (for the dump).
    int64_t valid_lo() const { return sl_.empty() ? 0 : sl_[0].vlo; }
    int64_t ctx_hi() const { return sl_.empty() ? 0 : sl_[0].hi; }

    // ---- STRATA_Q36_DFLASH_DUMP=<dir>: what tools/qwen35/dflash_ref.py replay reads (format: its docstring)
    void enable_dump(const std::string& dir, int max_steps, int max_rows);
    void dump_subset(const int32_t* ids, int n);                       ///< the draft head's token ids (subset.bin)
    void dump_step(sycl::queue& q, int64_t P, int32_t anchor);         ///< after a draft: the step's block, candidates and path (waits)

    // ---- products on named tensors (parity / bench)
    const void* tensor(const std::string& name) const;

    // ---- the parts, public for the parity program (q36_mmvq_dense_parity --dflash)
    void ctx_update(sycl::queue& q, int n, bool eager);
    void block(sycl::queue& q) { block_batch(q, 1); }

private:
    struct Ent { uint64_t off = 0, bytes = 0; int n_in = 0, n_out = 0; int kind = 0; };
    bool parse_index(const std::string& js, std::string& err);
    bool alloc(sycl::queue& q, std::string& err);
    void stage_meta(sycl::queue& q, int64_t P, int32_t anchor, int64_t c0, int nrows, int slot);   // fills + copies the meta (one block)
    void flush_eager(sycl::queue& q, int64_t c0, int nrows, int slot);
    int plan_flush(sycl::queue& q, int slot, int64_t P, int64_t* at);   // flushes all but the last <= 8 pending rows of a slot; returns those rows, `at` their first position
    // `xf`: the same activations as floats [ncols][n_in] (null: none); with W4X on and ncols in range they are quantized (requant) and the product is mmvq_w4x
    bool prod(sycl::queue& q, const void* w, const uint16_t* x16, float* y, int n_in, int n_out, int ncols, int cls = -1, const float* xf = nullptr, bool requant = true);
    bool xw(int ncols) const { return w4x_ && ncols >= w4x_min_ && ncols <= 32; }   // this pass's products take the int8 route
    bool xmm(sycl::queue& q, const void* w, float* y, int n_in, int n_out, int ncols);   // mmvq_w4x on the quantized activations already in w4x_act_
    int w4c_[8] = {0, 0, 0, 0, 0, 0, 0, 0};   // STRATA_Q36_DFLASH_W4C: threads a tile of the block's products by class (0: the engine's mmvq_w4h)
    int w4t_ = 1024;                          // STRATA_Q36_DFLASH_W4T: threads in all of those
    const void* w(const std::string& name) const;

    // pack
    std::vector<std::pair<std::string, Ent>> ents_;
    uint8_t* blob_ = nullptr;
    uint64_t blob_bytes_ = 0, vram_ = 0;
    int mask_token_ = 248070;
    float eps_ = 1e-6f;
    bool debug_ = false;
    Hooks hooks_;

    // typed pointers into the blob (set by parse/alloc)
    const void *fc_ = nullptr, *kv_ = nullptr, *hproj_ = nullptr;
    const void *wq_[kLayers] = {}, *wo_[kLayers] = {}, *wg_[kLayers] = {}, *wu_[kLayers] = {}, *wd_[kLayers] = {}, *akp_[kLayers] = {}, *mkp_[kLayers] = {};
    const float *in_ln_[kLayers] = {}, *post_ln_[kLayers] = {}, *qn_[kLayers] = {}, *kn_[kLayers] = {}, *abase_[kLayers] = {}, *mbase_[kLayers] = {};
    const float *hidden_norm_ = nullptr, *norm_ = nullptr;
    const uint16_t *pred_cb_ = nullptr, *succ_cb_ = nullptr;

    // device state
    uint16_t* stag_ = nullptr;                 // [slot][kRing][kTaps * kN] bf16
    sycl::half *kring_ = nullptr, *vring_ = nullptr;   // [slot][layer][kv head][kRing][128]
    int32_t* meta_ = nullptr;                  // device meta (see q36_dflash.dp.cpp), host twin below
    int32_t* hmeta_ = nullptr;
    uint16_t* ctx_x16_ = nullptr;              // fc input   [25600 / 16][32][16]
    float* ctx_y_ = nullptr;                   // fc output  [32][kN]
    float* ctx_scale_ = nullptr;               // [32]
    uint16_t* ctx_h16_ = nullptr;              // hidden_norm output, packed
    float* kv_y_ = nullptr;                    // [32][10240]
    float *x_ = nullptr, *h_ = nullptr;        // residual [8][kN], normed [8][kN]
    uint16_t *h16_ = nullptr, *h16b_ = nullptr, *att16_ = nullptr, *sw16_ = nullptr;
    float *dyn_a_ = nullptr, *dyn_m_ = nullptr, *qy_ = nullptr, *kvy_ = nullptr, *qb_ = nullptr, *kb_ = nullptr, *vb_ = nullptr;
    float *part_acc_ = nullptr, *part_m_ = nullptr, *part_l_ = nullptr;
    float *oy_ = nullptr, *gy_ = nullptr, *uy_ = nullptr, *dny_ = nullptr, *hp_y_ = nullptr;
    float* logits_ = nullptr;
    uint64_t* top_part_ = nullptr;
    float* hn_ = nullptr;                      // the head's input rows [kMaxBlocks x 7][kN]
    int n_slots_ = 1;
    bool w4x_ = false;
    int w4x_min_ = 16;
    uint8_t* w4x_act_ = nullptr;               // the int8 activations of the product in flight (w4x_act_bytes(25600, 32, 128))
    float *hcf_ = nullptr, *attf_ = nullptr, *ctxf_ = nullptr, *ctxhf_ = nullptr;   // float images of conv_pack / attn_merge / ctx_prep / ctx_norm outputs (W4X)
    bool tree_ = false;
    q36dt::Gather* tgath_ = nullptr;           // device image for the tree builder (STRATA_Q36_DFLASH_TREE)
    int32_t *cand_ = nullptr, *out_tok_ = nullptr;
    float *unary_ = nullptr, *out_prob_ = nullptr, *dbg_embed_ = nullptr;
    int nch_ = 0;

    // dump
    bool dump_ = false, dump_subset_ = false;
    std::string dump_dir_;
    std::FILE *f_ctx_ = nullptr, *f_steps_ = nullptr, *f_blocks_ = nullptr;
    int dump_max_ = 0, dump_steps_ = 0, dump_rows_max_ = 0;
    int64_t dump_records_ = 0;
    void dump_rows(sycl::queue& q, int64_t c0, int n);
    void dump_meta();

    // host mirror of each slot: the captured run and the flushed run (positions)
    struct Ctx {
        int64_t cap_lo = 0, cap_hi = 0;        // staging holds rows [cap_lo, cap_hi) of the current history (cap_lo = cap_hi: none)
        int64_t vlo = 0, hi = 0;               // the drafter's K/V ring holds rows [vlo, hi)
        bool have = false;
    };
    std::vector<Ctx> sl_;
};

}   // namespace strata::q36df
