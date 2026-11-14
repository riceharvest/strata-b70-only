// q36_xatt.hpp - the decode attention of strata-q36 on the Arc Pro B70's matrix engine (q36_xatt.dp.cpp, in libq36_w4).
#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::q36xatt {

constexpr int kHD = 256;       // head dimension
constexpr int kG = 6;          // query heads per KV head (the 27B: 24 / 4)
constexpr int kMaxRows = 96;   // rows per KV head a call takes: 16 tokens x kG
constexpr int kMaxLs = 6;      // ... in sets of 16 (the lanes of a DPAS)
constexpr int kBlk = 32;       // cells per block of the transposed V and of a P operand

/// The cells a chunk's scratch must hold at a context of `ctx` cells split into at most `nchk` chunks per row.
int chunk_cap(int ctx, int nchk);
/// The mirror's padded cell count (a multiple of 64 with 64 spare cells: a tile may read past the last cell).
inline int ctx_pad(int ctx) { return (ctx + 63) / 64 * 64 + 64; }

// The mirror of one layer's K / V cache, what the kernels read:
//   k8  [HKV][ctxp][64 dwords]         a cell's K as 256 int8 codes
//   ks  [HKV][ctxp]                    its scale: K = code x ks   (max |K| / 127)
//   v8t [HKV][ctxp / 32][256][8 dw]    V transposed in blocks of 32 cells: dimension d's codes of the block's 32 cells
//   vs  [HKV][ctxp]                    a cell's V scale
inline std::size_t k8_dwords(int hkv, int ctxp) { return (std::size_t) hkv * ctxp * 64; }
inline std::size_t v8t_dwords(int hkv, int ctxp) { return (std::size_t) hkv * (ctxp / kBlk) * 256 * 8; }
inline std::size_t sc_floats(int hkv, int ctxp) { return (std::size_t) hkv * ctxp; }

// A call's scratch (one set serves every layer):
inline std::size_t qb_dwords(int hkv) { return (std::size_t) hkv * 8 * kMaxLs * 128; }
inline std::size_t qs_floats(int hkv) { return (std::size_t) hkv * kMaxRows; }
// (`ls`: the row sets of 16 the scratch is made for; a call of more is not taken)
inline std::size_t score_floats(int hkv, int nchk, int capc, int ls = kMaxLs) { return (std::size_t) hkv * nchk * capc * ls * 16; }
inline std::size_t pb_dwords(int hkv, int nchk, int capc, int ls = kMaxLs) { return (std::size_t) 2 * hkv * nchk * (capc / kBlk) * ls * 256; }
inline std::size_t srow_floats(int hkv, int nchk) { return (std::size_t) hkv * nchk * kMaxRows; }

struct Call {
    const float* q;            // the rows' queries: row (t0 + k) * H + head, 256 values
    const int32_t* pos0;       // on the device: token t sees the cells [0, *pos0 + t + 1)
    int t0, nt, H, HKV, nchk;  // tokens [t0, t0 + nt) (nt <= 16); H == kG * HKV; chunks per row (k_attn_gqa's)
    float scale;               // 1 / sqrt(head dimension)
    const uint32_t* k8;        // K codes, [HKV][kstride cells][64 dwords]
    const float* ks;           // ns == 1: a scale a cell, [HKV][kstride]
    const uint16_t* ks16 = nullptr;   // ns == 4: the engine's 8-bit cache read as it is - an FP16 scale per 64 values, [HKV][kstride][4]
    int ns = 1;
    int kstride = 0;           // cells a KV head of the K arrays (0: ctxp)
    int max_ls = kMaxLs;       // the row sets the scratch holds
    const uint32_t* v8t;
    const float* vs;
    int ctxp;                  // ctx_pad(ctx)
    uint32_t* qb;              // scratch
    float* qs;
    float* sc;
    uint32_t* pb;
    float* srow;
    int capc;                  // chunk_cap(ctx, nchk)
    float* pacc;               // out: the chunks' partial sums [row][nchk][256], max [row][nchk] and sum [row][nchk] with
    float* pm;                 //      row = (t - t0) * H + head: what k_attn_merge_gqa merges
    float* pl;
    int dbg = 0;               // (STRATA_XATT_DBG, timing only: parts of the kernels left out)
    const uint32_t* tmask = nullptr;   // optional device [nt] ancestor bitmasks (tree verify)
    const int32_t* tp0 = nullptr;      // device scalar tree root cell (defaults to *pos0 when null)
};
/// Three kernels on `stream`'s queue: the queries as int8, the scores and each chunk's softmax, the value sums.
/// false: the shape is not taken (the caller runs k_attn_gqa).
bool attend(const Call& c, void* stream);

// ---- XATT2 (STRATA_Q36_XATT2=1): single-pass online-softmax ESIMD+DPAS split-K attention + fast ESIMD merge.
constexpr int kMaxBatchGroups2 = 16;   // max slot groups in one batched rows_mode launch (total tokens <= 16)
constexpr int kMaxSegs2 = 96;          // Engine::n_chunk: the partial slots a row of a 16-token call has (the engine allocates 16 x H x 96)
constexpr int kMaxSegsHard2 = 252;     // most segments a row may have (a call of nt tokens: nseg_max * 16 / nt of them fit the same partials)

inline std::size_t qb2_dwords(int hkv) { return (std::size_t) hkv * 8 * kMaxBatchGroups2 * 128; }
inline std::size_t qs2_floats(int hkv) { return (std::size_t) hkv * kMaxBatchGroups2 * 16; }

struct BatchGroup2 {
    int32_t slot = 0;   // index into k_tab / v_tab / mir_tab
    int32_t r0 = 0;     // starting token index in q [0..nt) and out [0..nt)
    int32_t cnt = 0;    // consecutive tokens of this slot
    int32_t cell_hint = 0;   // host estimate of cell0 (optional, for sizing only)
};

struct Call2 {
    const float* q = nullptr;            // [nt, H, 256] queries (from t0=0)
    const int32_t* pos0 = nullptr;       // single-session: *pos0; batch (ngrp > 0): device [nt] d_rcell
    int t0 = 0, nt = 1;                  // tokens [0, nt), nt <= 16
    int H = 24, HKV = 4;                 // H == 6 * HKV, HD == 256
    float scale = 1.0f / 16.0f;          // 1 / sqrt(256)

    // Single-session KV pointers (used when ngrp == 0):
    const uint32_t* k8 = nullptr;        // [HKV][kstride][64 dwords]
    const float* ks = nullptr;           // ns == 1: [HKV][kstride]
    const uint16_t* ks16 = nullptr;      // ns == 4: [HKV][kstride][4]
    const uint32_t* v8t = nullptr;       // [HKV][ctxp/32][256][8 dwords]
    const float* vs = nullptr;           // [HKV][ctxp]

    // Batch-mode KV tables (used when ngrp > 0):
    // When kv_vt (ns == 4): k_tab[slot] -> k8 (+ HKV*ctx*256 -> ks16), v_tab[slot] -> v8t (+ v8t_dwords*4 -> vs).
    // When !kv_vt: mir_tab[slot * 4 + {0..3}] = {xk8, xks_or_ks16, xv8t, xvs}.
    void* const* k_tab = nullptr;
    void* const* v_tab = nullptr;
    const void* const* mir_tab = nullptr;
    int ngrp = 0;
    BatchGroup2 grp[kMaxBatchGroups2]{};

    int ns = 1;                          // 1 or 4
    int ctx = 0;                         // kstride for ns==4 via k_tab
    int kstride = 0;                     // cells per KV head of K (0 -> ctxp)
    int ctxp = 0;                        // ctx_pad(ctx)
    int pos_hint = 0;                    // host cell count estimate for auto nseg selection
    int nseg_max = kMaxSegs2;            // segments a row has room for when the call is 16 tokens (nt tokens: nseg_max * 16 / nt, at most 252)

    // Tiny query scratch (NO xsc, xpb, or xsrow required!):
    uint32_t* qb = nullptr;              // >= qb2_dwords(HKV) dwords
    float* qs = nullptr;                 // >= qs2_floats(HKV) floats

    // Split-K partials [row][nseg][256 / 1] (fits in pacc/pm/pl):
    float* pacc = nullptr;
    float* pm = nullptr;
    float* pl = nullptr;

    // Optional final output + gate (when out != nullptr, attend2 runs k_xatt2_merge directly):
    const float* gate = nullptr;         // optional [nt * H * 256]
    float* out = nullptr;                // optional [nt * H * 256]

    // Optional tree verify mask (single-session only):
    const uint32_t* tmask = nullptr;
    const int32_t* tp0 = nullptr;

    // Tuning / sweep overrides (-1 = use env/default):
    int form_override = -1;              // -1=auto/env, 0=R8 (reg/L1), 1=RS16 (16K SLM / R8_D8), 2=RS8 (8K SLM), 3=RS16K (16K Kb+Vb)
    int nseg_override = 0;               // 0=auto/env, >0=fixed segments per KV head
    int u16_override = -1;               // -1=auto/env (default 1), 0=u8 P, 1=u16 P
    int exp_override = -1;               // -1=env (STRATA_Q36_XATT2_EXP, default 0), 0=hardware exp2, 1=exp by its series
    int zero_fence = 0;                  // runtime 0 for anti-hoist fencing

    // Optional bounds-check debug mode (active under STRATA_Q36_XATT2_DBG=1):
    uint32_t* dbg_flag = nullptr;        // device uint32 (zeroed by caller; set non-zero on clamped OOB)
    size_t dbg_q_floats = 0;             // allocated capacity of `q` in floats (0 = derive from Call2)
    size_t dbg_k8_dwords = 0;            // allocated capacity of `k8` per slot in dwords (0 = derive)
    size_t dbg_ks_floats = 0;            // allocated capacity of `ks` per slot in floats (0 = derive)
    size_t dbg_ks16_u16 = 0;             // allocated capacity of `ks16` per slot in uint16s (0 = derive)
    size_t dbg_v8t_dwords = 0;           // allocated capacity of `v8t` per slot in dwords (0 = derive)
    size_t dbg_vs_floats = 0;            // allocated capacity of `vs` per slot in floats (0 = derive)
    size_t dbg_out_floats = 0;           // allocated capacity of `out`/`gate` in floats (0 = derive)
};

/// Returns the form name string ("R8", "RS16", "RS8", "RS16K").
const char* xatt2_form_name(int form);
/// Returns the planned segment count for group `group_idx` (matches attend2's dispatch).
int xatt2_plan_nseg(const Call2& c, int group_idx = 0);
/// The row stride of the partials pm / pl / pacc (segments a row) the call will use (matches attend2).
int xatt2_plan_stride(const Call2& c);
/// The form attend2 picks for the call (the forced one, else the auto policy of the rows a row set holds).
int xatt2_plan_form(const Call2& c);
/// A probe of the softmax's two exps on the device (no attention): out_hw[i] = hardware exp2(x[i] * log2e), out_poly[i] =
/// the series, for n inputs in [-87, 0] (n a multiple of 32).  The parity compares them with exp in float64.
bool xatt2_exp_probe(void* stream, const float* d_x, float* d_hw, float* d_poly, int n);
/// Single-pass online-softmax attention (+ merge when c.out != nullptr).
bool attend2(const Call2& c, void* stream);

}   // namespace strata::q36xatt
