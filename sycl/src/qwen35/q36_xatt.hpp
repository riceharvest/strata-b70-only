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
};
/// Three kernels on `stream`'s queue: the queries as int8, the scores and each chunk's softmax, the value sums.
/// false: the shape is not taken (the caller runs k_attn_gqa).
bool attend(const Call& c, void* stream);

}   // namespace strata::q36xatt
