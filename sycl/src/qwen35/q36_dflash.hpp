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

namespace strata::q36df {

constexpr int kN = 5120, kFF = 17408, kLayers = 5, kHeads = 32, kKvHeads = 8, kHd = 128;
constexpr int kRows = 8, kDrafts = 7, kTaps = 5, kRank = 256, kTopK = 16;
constexpr int kWindow = 2048;   ///< a row sees positions with |q - k| < 2048
constexpr int kRing = 2064;     ///< ring positions (window + slack for rows a rejected draft wrote ahead)
constexpr int kEagerCols = 32;  ///< rows one eager context flush takes (one weight pass)
constexpr int kKpOut = 1280;    ///< kernel_projection outputs: side x tap x group = 2 x 2 x 320
constexpr int kChunks = 17;     ///< attention chunks: 16 x 128 context keys + the 8 block keys
constexpr int kMaxNd = 248320;  ///< the most logits a head row can have

/// The device meta's layout (int32 / float words), staged by prepare().
namespace k_consts {
constexpr int kMetaP = 0, kMetaAnchor = 1, kMetaVlo = 2, kMetaN = 3, kMetaCtxPos = 16, kMetaCtxSlot = 48, kMetaTok = 80, kMetaRopeCtx = 96,
              kMetaRopeBlk = 4192, kMetaWords = 5216;
}

/// The kernels, public for the parity program (q36_mmvq_dense_parity --dflash).  Layouts: q36_dflash.dp.cpp's header.
namespace k {
void capture(sycl::queue& q, uint16_t* stag, int tap, const float* x, int T, const int32_t* pos0);
void ctx_prep(sycl::queue& q, const uint16_t* stag, const int32_t* meta, int n, uint16_t* x16, float* scale);
void ctx_norm(sycl::queue& q, const float* y, const float* scale, const float* w, float eps, int n, uint16_t* x16);
void ctx_store(sycl::queue& q, const float* kv, const int32_t* meta, int n, const float* const* kn, sycl::half* kring, sycl::half* vring, float eps);
void rms_pack(sycl::queue& q, const float* x, const float* w, float eps, float* h, uint16_t* x16);
void conv_pack(sycl::queue& q, const float* x, const float* dyn, int side, const float* base, uint16_t* x16);
void finish_rms(sycl::queue& q, const float* o, const float* dyn, const float* base, float* x, const float* w, float eps, float* h, uint16_t* x16);
void qk_prep(sycl::queue& q, const float* qy, const float* kvy, const float* qn, const float* kn, const int32_t* meta, float eps, float* qb, float* kb, float* vb);
void attn_part(sycl::queue& q, const float* qb, const float* kb, const float* vb, const sycl::half* kring, const sycl::half* vring, int layer,
               const int32_t* meta, float* part_acc, float* part_m, float* part_l);
void attn_merge(sycl::queue& q, const float* part_acc, const float* part_m, const float* part_l, uint16_t* x16);
void top1(sycl::queue& q, const float* logits, int nd, int rows, int nch, uint64_t* part);
void top2(sycl::queue& q, const uint64_t* part, int nch, int rows, const int32_t* idmap, int32_t* cand, float* unary);
void selector_walk(sycl::queue& q, const float* hp, const int32_t* cand, const float* unary, const uint16_t* pred, const uint16_t* succ, const int32_t* meta,
                   int32_t* out_tok, float* out_prob);
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
    bool load(sycl::queue& q, const std::string& pack_path, std::string& err);
    /// A random pack's worth of weights (parity / bench: no file; ~1.2 GB of random incompressible records).
    bool load_synthetic(sycl::queue& q, uint64_t seed, std::string& err);
    bool loaded() const { return blob_ != nullptr; }
    void set_hooks(sycl::queue& q, const Hooks& h);   ///< allocates the logits scratch
    void enable_debug(bool on) { debug_ = on; }
    int mask_token() const { return mask_token_; }
    uint64_t vram_bytes() const { return vram_; }

    // ---- target side (the engine calls these)
    /// Inside forward(): rows 0 .. T-1 of the residual stream x [T][kN] (fp32) after target layer tap_layer(tap) take positions
    /// *pos0 + r of the staging ring (bf16).  Recorded in the verify graph; pos0 is the device position the graph reads.
    void capture(sycl::queue& q, int tap, const float* x, int T, const int32_t* pos0_dev);
    static int tap_layer(int k) { return 5 + 14 * k; }   ///< 5 19 33 47 61 (0-based layer whose output is tapped)
    /// Host bookkeeping, after the rows [p0, p0 + n) of the history were forwarded and COMMITTED (after the accept decision).
    void note_forward(int64_t p0, int64_t n);
    /// The history is another one (a session switch, a restore): the context is dropped.
    void invalidate();

    // ---- draft
    /// Plans the draft whose block starts at position P (every row below P is committed; `anchor` is the token at P): flushes
    /// all but the last <= 8 pending context rows eagerly, stages the meta of the rest.  Returns n, the context rows run() takes
    /// (0 .. 8); the engine keys its graph on it.  Async on q except for the eager flushes' small waits.
    int prepare(sycl::queue& q, int64_t P, int32_t anchor);
    /// Enqueues the context update (n rows) and the block.  Reads only device memory; recordable.
    void run(sycl::queue& q, int n);
    const int32_t* out_tokens() const { return out_tok_; }   ///< device [7]
    const float* out_probs() const { return out_prob_; }     ///< device [7]
    /// Debug: the last block's intermediates and candidates (waits).
    bool debug_last(sycl::queue& q, Debug& d);
    /// Debug: staging rows of positions [p0, p0 + n) as bf16 [n][5 x kN] (waits); false when a position is not in the ring.
    bool debug_stage_rows(sycl::queue& q, int64_t p0, int n, std::vector<uint16_t>& out);
    /// The first context position the last prepare() left valid (for the dump).
    int64_t valid_lo() const { return ctx_vlo_; }
    int64_t ctx_hi() const { return ctx_hi_; }

    // ---- STRATA_Q36_DFLASH_DUMP=<dir>: what tools/qwen35/dflash_ref.py replay reads (format: its docstring)
    void enable_dump(const std::string& dir, int max_steps, int max_rows);
    void dump_subset(const int32_t* ids, int n);                       ///< the draft head's token ids (subset.bin)
    void dump_step(sycl::queue& q, int64_t P, int32_t anchor);         ///< after a draft: the step's block, candidates and path (waits)

    // ---- products on named tensors (parity / bench)
    const void* tensor(const std::string& name) const;

    // ---- the parts, public for the parity program (q36_mmvq_dense_parity --dflash)
    void ctx_update(sycl::queue& q, int n, bool eager);
    void block(sycl::queue& q);

private:
    struct Ent { uint64_t off = 0, bytes = 0; int n_in = 0, n_out = 0; int kind = 0; };
    bool parse_index(const std::string& js, std::string& err);
    bool alloc(sycl::queue& q, std::string& err);
    void stage_meta(sycl::queue& q, int64_t P, int32_t anchor, int64_t c0, int nrows);   // fills + copies the meta
    void flush_eager(sycl::queue& q, int64_t c0, int nrows);
    bool prod(sycl::queue& q, const void* w, const uint16_t* x16, float* y, int n_in, int n_out, int ncols);
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
    uint16_t* stag_ = nullptr;                 // [kRing][kTaps * kN] bf16
    sycl::half *kring_ = nullptr, *vring_ = nullptr;   // [layer][kv head][kRing][128]
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

    // host mirror: the captured run and the flushed run (positions)
    int64_t cap_lo_ = 0, cap_hi_ = 0;          // staging holds rows [cap_lo_, cap_hi_) of the current history (cap_lo_ = cap_hi_: none)
    int64_t ctx_vlo_ = 0, ctx_hi_ = 0;         // the drafter's K/V ring holds rows [ctx_vlo_, ctx_hi_)
    bool have_cap_ = false;
};

}   // namespace strata::q36df
