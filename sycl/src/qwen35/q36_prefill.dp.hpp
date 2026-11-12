// q36_prefill.dp.hpp - prompt-path kernels of strata-q36 (M4b, long prompts): the causal prompt attention as one
// flash-style XMX kernel, and the prompt GDN recurrence with its state in registers.  Included once, by q36_engine.dp.cpp.
//
// 1. The prompt attention (flash_prompt_attn).  The GEMM path before it (attn_core, STRATA_Q36_PATTN=gemm) wrote every
//    score of a chunk to memory and read it back three times (FP32 scores, the softmax, FP16 probabilities), the queries
//    in slices whose scores fit 256 MB: at 118K cells that is ~22 queries a slice, ~90 slices a 2,048-token chunk, each
//    re-reading the KV head's 60 MB of K and V (one value GEMM per query head).  Prompt reads fell from 946 tok/s at 30K
//    to 261 at 118K.  Here nothing of the scores leaves the sub-group: S = Q K^T, the online softmax and O += P V run per
//    64-cell block, causal blocks only.
//    Layout (LP2, the tiling of vllm-xpu-kernels' Xe2 chunk_prefill for head 256 on the B70: 8-row sub-groups that each
//    hold all 256 dims of O, many of them on one K / V block): a work-group per (32 query tokens, KV head), 4 sub-groups
//    of 16 lanes per query head of that KV head (4 G of them, 24 = 192 rows at the 27B's G = 6; G <= 8 keeps it within
//    the 512 work-items of the 256-GRF mode).  Sub-group s takes head s / 4 and the 8 tokens (s % 4) 8 .. + 8 of the
//    work-group's 32: the 8-row tile of the M4b kernel (one work-group per 8 tokens and 6 heads), with the same rows, the
//    same block count (its own) and the same per-row arithmetic, so the bits are the M4b kernel's.
//    A sub-group holds its 8 rows (8 tokens of one head) x all 256 dims of O in joint_matrix accumulators (16 tiles of
//    8x16 FP32, 128 GRFs at 256 a thread).  Per block: the 16 Q tiles (8x16, from the FP16 queries, L1-hot) times the
//    K tiles (16 dims x 16 cells) into 4 score tiles; the score tiles through 2 KB of the sub-group's local memory to the
//    softmax (lane = row r = lane % 8, half hf = lane / 8 of the 64 cells: its 32 scores into registers, then FP16 P back
//    over the same bytes), exp2 of the scaled scores minus the running max; O rescaled when a row's max moved (rare at
//    depth); P (4 A tiles) times the V tiles (16 cells x 16 dims) into O.  Only the diagonal blocks compare cells with
//    rows.  The heaviest query tiles (the chunk's last tokens) start first.  Rows past the chunk inside its last 8-token
//    slice are computed on the padding rows of q16 and not stored; slices wholly past the chunk load nothing and compute
//    nothing (they still stage and meet every barrier).  Numerics: FP16 Q / K / P / V, FP32 accumulation, FP32 softmax
//    statistics, O / l at the end (the GEMM path: normalised FP16 probabilities); not the GEMM path's bits.
//    The K / V operand is the kernel's template argument (KvIO):
//    - FP16 cache ([kvh][ctx][256]): the B tiles straight from the cache (col_major K, row_major V), the 24 sub-groups of
//      a work-group on the same cells at the same time (L1).  No work-group barrier.
//    - 8-bit cache (STRATA_Q36_KV=q8, kv_q8.hpp's format: int8 codes, one FP16 scale per 64 values of a cell and KV head):
//      each 64-cell block's K and V are dequantized ONCE per work-group (x' = fp16(code * scale), the gather's arithmetic)
//      by all its work-items into 64 KB of local memory, already in the XMX B layout (16x16 tiles of 512 bytes, dword =
//      the two values a DPAS lane pairs: two dims of a cell for K^T, two cells of a dim for V; ext_intel_packed loads),
//      and every sub-group reads its B tiles from there.  Two work-group barriers a block (the M4b kernel: K and V in
//      turn through 33 KB per 48 rows, four barriers, 1-2 work-groups an Xe core).  The staging loads all of a
//      work-item's codes of a block first, then converts and stores (the cell, or for V the tile, fastest across lanes:
//      64-byte rows, no bank conflicts on K).
//    Needs: head_dim 256, G <= 8, ctx a multiple of 64 (a block never reads past the cache), q16 with 8 rows of padding
//    past the chunk, the STRATA_XMX_SG16 build (Xe2: joint_matrix 8x16x16 on 16-lane sub-groups), 116 KB of local
//    memory for the 8-bit kernel at G = 6.  Cells past the chunk's last position inside the last block are read (masked
//    to P = 0): the cache is zeroed at allocation and only ever holds finite K / V, so 0 x V adds nothing.
//
// 2. The prompt GDN recurrence (gdn_xe_rg): a copy of this repository's sycl/include/strata/prefill/gdn_xe.hpp variant 7
//    (gdn_xe_rg_body / gdn_xe_rg_launch, Flash-Next's prompt recurrence on Battlemage; B70, T = 16384, 48 heads: 11.1-12.0
//    ms against 21.3-21.6 ms for the column kernels), under its own kernel name.  The same h / gate / beta / state /
//    output layouts and the same per-element arithmetic as k_gdn_seq_cols (g = exp(gate), delta = (v - g k.S) beta,
//    S = g S + k delta, o = q.S / sqrt(128)); the row sums in 4 groups of 32 instead of 16 of 8 (other bits).  The
//    geometry is compiled in: 128-wide states, 16 K heads, 48 V heads, 10240 conv channels (the 27B's).
#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace q36pf {

// ------------------------------------------------------------------------------------------------ flash attention
constexpr int kFaHD = 256;                      // head_dim
constexpr int kFaRows = 8;                      // rows of a sub-group (joint_matrix M): 8 query tokens of one head
constexpr int kFaSlices = 4;                    // 8-token slices of a work-group = its sub-groups per query head
constexpr int kFaTok = kFaRows * kFaSlices;     // query tokens of a work-group
constexpr int kFaMaxSg = 32;                    // sub-groups of a work-group at most (512 work-items, 256-GRF mode)
constexpr int kFaBC = 64;                       // cells per block
constexpr int kFaNT = kFaBC / 16;               // score tiles per block
constexpr int kFaDT = kFaHD / 16;               // O tiles (16 dims each)
constexpr int kFaSLD = kFaBC + 4;               // score row stride in local memory (floats)
constexpr int kFaPLD = kFaBC + 8;               // P row stride in local memory (halves)
constexpr int kFaMaxG = kFaMaxSg / kFaSlices;   // query heads per KV head
// a sub-group's local memory: scores [8][kFaSLD] floats, P [8][kFaPLD] halves over the same bytes (every lane has its
// scores in registers before any P is written), a value per row [8] floats (16-byte multiples)
constexpr size_t kFaSBytes = (size_t) kFaRows * kFaSLD * 4;
constexpr size_t kFaPBytes = (size_t) kFaRows * kFaPLD * 2;
constexpr size_t kFaRBytes = (size_t) kFaRows * 4 < 16 ? 16 : (size_t) kFaRows * 4;
constexpr size_t kFaSgBytes = kFaSBytes + kFaRBytes;
// the 8-bit cache: a block's K^T, then its V, in FP16 packed B tiles (16x16 = 8 rows of 16 dwords, 512 bytes; a dword the
// two values one DPAS lane pairs, the even one in the low half), after the sub-groups' regions.  K^T tile kk * kFaNT + ct:
// dims [16 kk, +16) x cells [16 ct, +16), dword (p, n) = dims 16 kk + 2p, +1 of cell 16 ct + n.  V tile kt * kFaDT + dt:
// cells [16 kt, +16) x dims [16 dt, +16), dword (p, n) = cells 16 kt + 2p, +1 of dim 16 dt + n.
constexpr int kFaTileH = 16 * 16;                               // halves of a packed tile
constexpr int kFaTileW = kFaTileH / 2;                          // its dwords
constexpr size_t kFaStageKBytes = (size_t) kFaBC * kFaHD * 2;   // a block's K^T (32 KB), then as much for V
constexpr size_t kFaStageBytes = 2 * kFaStageKBytes;
constexpr int kFaStageItems = kFaBC * kFaHD / 16;               // staging steps of a block: 16 codes of K and 16 of V each
constexpr int kFaStageU = 3;                                    // steps a work-item loads before it converts any
static_assert(kFaSBytes % 16 == 0 && kFaPBytes <= kFaSBytes && kFaRBytes % 16 == 0, "16-byte aligned regions, P over S");
static_assert(kFaSgBytes % 16 == 0 && kFaStageKBytes % 512 == 0, "aligned staging tiles");
static_assert(kFaRows * 16 <= kFaRows * kFaSLD, "the tail store's tile fits in the score region");

struct FaArgs {
    const sycl::half* q;   // [T + 8][H][256] FP16 (the last slice's rows past T are read, not stored)
    const sycl::half* k;   // the layer's K cache, [HKV][ctx][256] FP16
    const sycl::half* v;   // its V cache, same layout
    const int8_t* k8 = nullptr;   // the 8-bit cache instead (k / v unused): the layer's K codes [HKV][ctx][256], its FP16
    int vt = 0;                   // > 0: the 8-bit cache's V is transposed (q36_xatt.hpp's v8t; the padded cell count)
    const int8_t* v8 = nullptr;   // scales [HKV][ctx][4] right after them (byte HKV * ctx * 256 of the allocation); V alike
    float* o;              // [T][H][256]: softmax(Q K^T / 16) V per row, causal (row t sees cells [0, p0 + t])
    int T = 0, p0 = 0, H = 0, HKV = 0, G = 0, ctx = 0, ntiles = 0;   // ntiles: of kFaTok tokens
    float scale_log2 = 0.0f;   // log2(e) / sqrt(256)
};

// The K / V operand of fa_body (KvIO).
struct FaKvF16 { static constexpr bool kQ8 = false; };   // the FP16 cache: B tiles straight from it
struct FaKvQ8 { static constexpr bool kQ8 = true; };     // the 8-bit cache: a block staged in local memory, B tiles from there

#ifdef STRATA_XMX_SG16
namespace jm = sycl::ext::oneapi::experimental::matrix;

template <typename T>
inline auto fa_gptr(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p);
}
template <typename T>
inline auto fa_lptr(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
}

using FaAcc = jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, kFaRows, 16>;
using FaA = jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, kFaRows, 16, jm::layout::row_major>;
using FaBk = jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::col_major>;
using FaBv = jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::row_major>;
using FaBp = jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::ext_intel_packed>;

// O[row][*] *= v of that row; v: lane l holds row l % 8's value (lanes l and l + 8 the same), rv: the sub-group's 8 floats
// of local memory.  An 8 x 16 FP32 accumulator on 16-lane sub-groups: work item `lane` holds column lane, its element i
// row i, and joint_matrix_apply visits the elements in order (ds4_kernels.dp.cpp's sparse_attn_xmx_kernel scales its O
// rows the same way; ds4_xmx_test checks that layout on the B70).
__attribute__((always_inline)) inline void fa_row_scale(sycl::sub_group sg, FaAcc (&O)[kFaDT], float v, float* rv) {
    const int lane = (int) sg.get_local_linear_id();
    sycl::group_barrier(sg);   // (the last scaling's reads of rv are done)
    if (lane < kFaRows) rv[lane] = v;
    sycl::group_barrier(sg);
#pragma unroll
    for (int d = 0; d < kFaDT; ++d) {
        int i = 0;
        jm::joint_matrix_apply(sg, O[d], [&](float& x) {
            x *= rv[i];
            ++i;
        });
    }
}

// Two FP16 values in a dword, x0 in the low half.
__attribute__((always_inline)) inline uint32_t fa_h2(float x0, float x1) {
    return sycl::bit_cast<uint32_t>(sycl::vec<sycl::half, 2>(sycl::half(x0), sycl::half(x1)));
}

// The 8-bit cache: cells [c0, c0 + 64) of one KV head (K codes kq / scales ksc, V codes vq / scales vsc) into the packed
// K^T tiles kst and V tiles vst (x' = fp16(code * scale), the gather's arithmetic), by the work-group's nwi work-items
// (lid: this one), kFaStageU steps loaded before any is converted.  Step x of K: cell x % 64, dims [16 (x / 64), +16) (a
// 16-byte load; a sub-group: 16 cells of one tile, 8 rows of 64 contiguous bytes stored).  Step x of V: tile x / 16, its
// row (x % 16) / 2 (a cell pair), dims 8 (x % 2) + [0, 8) of the tile's 16 (two 8-byte loads; a sub-group: one whole
// tile, 32 contiguous bytes a lane).
// vt > 0: V transposed in blocks of 32 cells (a dimension's 32 cells side by side: byte ((cell / 32) * 256 + dim) * 32 +
// cell % 32 of the KV head's part at vq) with one FP32 scale a cell (vsf): a cell's 8 values are 8 loads 32 bytes apart.
__attribute__((always_inline)) inline void fa_stage_q8(int lid, int nwi, const int8_t* kq, const uint16_t* ksc,
                                                       const int8_t* vq, const uint16_t* vsc, int c0, uint32_t* kst,
                                                       uint32_t* vst, int vt = 0, const float* vsf = nullptr) {
    constexpr int SPC = kFaHD / 64;   // scales per cell
    for (int x0 = lid; x0 < kFaStageItems; x0 += kFaStageU * nwi) {
        sycl::vec<int8_t, 16> kcode[kFaStageU];
        sycl::vec<int8_t, 8> va[kFaStageU], vb[kFaStageU];
        float ks[kFaStageU], vsa[kFaStageU], vsb[kFaStageU];
#pragma unroll
        for (int u = 0; u < kFaStageU; ++u) {
            const int x = x0 + u * nwi;
            if (x < kFaStageItems) {
                const size_t kcell = (size_t) (c0 + x % kFaBC);
                const int kd = (x / kFaBC) * 16;
                kcode[u] = *reinterpret_cast<const sycl::vec<int8_t, 16>*>(kq + kcell * kFaHD + kd);
                ks[u] = (float) sycl::bit_cast<sycl::half>(ksc[kcell * SPC + kd / 64]);
                const int tl = x / 16, l = x % 16;
                const size_t vcell = (size_t) (c0 + (tl / kFaDT) * 16 + (l / 2) * 2);
                const int vd = (tl % kFaDT) * 16 + (l % 2) * 8;
                if (vt > 0) {
                    const int8_t* const pa = vq + ((vcell >> 5) * kFaHD + vd) * 32 + (vcell & 31);
                    const int8_t* const pb2 = vq + (((vcell + 1) >> 5) * kFaHD + vd) * 32 + ((vcell + 1) & 31);
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        va[u][j] = pa[j * 32];
                        vb[u][j] = pb2[j * 32];
                    }
                    vsa[u] = vsf[vcell];
                    vsb[u] = vsf[vcell + 1];
                } else {
                    va[u] = *reinterpret_cast<const sycl::vec<int8_t, 8>*>(vq + vcell * kFaHD + vd);
                    vb[u] = *reinterpret_cast<const sycl::vec<int8_t, 8>*>(vq + (vcell + 1) * kFaHD + vd);
                    vsa[u] = (float) sycl::bit_cast<sycl::half>(vsc[vcell * SPC + vd / 64]);
                    vsb[u] = (float) sycl::bit_cast<sycl::half>(vsc[(vcell + 1) * SPC + vd / 64]);
                }
            }
        }
#pragma unroll
        for (int u = 0; u < kFaStageU; ++u) {
            const int x = x0 + u * nwi;
            if (x < kFaStageItems) {
                const int c = x % kFaBC, kk = x / kFaBC;
                uint32_t* const kdst = kst + (kk * kFaNT + c / 16) * kFaTileW + c % 16;
#pragma unroll
                for (int p = 0; p < 8; ++p)
                    kdst[p * 16] = fa_h2((float) kcode[u][2 * p] * ks[u], (float) kcode[u][2 * p + 1] * ks[u]);
                const int tl = x / 16, l = x % 16;
                sycl::vec<uint32_t, 8> o;
#pragma unroll
                for (int j = 0; j < 8; ++j) o[j] = fa_h2((float) va[u][j] * vsa[u], (float) vb[u][j] * vsb[u]);
                *reinterpret_cast<sycl::vec<uint32_t, 8>*>(vst + tl * kFaTileW + (l / 2) * 16 + (l % 2) * 8) = o;
            }
        }
    }
}

template <class KvIO>
__attribute__((always_inline)) inline void fa_body(sycl::nd_item<3> it, const FaArgs& a, uint8_t* smem) {
    constexpr bool Q8 = KvIO::kQ8;
    sycl::sub_group sg = it.get_sub_group();
    const int lane = (int) sg.get_local_linear_id();
    const int s = (int) sg.get_group_linear_id();             // head s / kFaSlices of the KV head, slice s % kFaSlices
    const int kvh = (int) it.get_group(1);
    const int tile = a.ntiles - 1 - (int) it.get_group(2);   // the chunk's last tokens (the most cells) first
    const int wtok0 = tile * kFaTok;                         // the work-group's first token
    const int tok0 = wtok0 + (s % kFaSlices) * kFaRows;      // the sub-group's
    const int h = kvh * a.G + s / kFaSlices;
    const int nval = sycl::max(0, sycl::min(kFaRows, a.T - tok0));   // rows of the slice inside the chunk (0: past it)
    float* const ss = reinterpret_cast<float*>(smem + (size_t) s * kFaSgBytes);
    sycl::half* const ps = reinterpret_cast<sycl::half*>(smem + (size_t) s * kFaSgBytes);   // over the scores
    float* const rv = reinterpret_cast<float*>(smem + (size_t) s * kFaSgBytes + kFaSBytes);
    const size_t rs = (size_t) a.H * kFaHD;                  // a token's stride in q / o
    const sycl::half* const qb = a.q + ((size_t) tok0 * a.H + h) * kFaHD;   // (read only when nval > 0)
    const sycl::half* const kb = Q8 ? nullptr : a.k + (size_t) kvh * a.ctx * kFaHD;
    const sycl::half* const vb = Q8 ? nullptr : a.v + (size_t) kvh * a.ctx * kFaHD;
    // the 8-bit cache: the KV head's codes and scales, the work-group's staging tiles
    const int8_t* const kq = Q8 ? a.k8 + (size_t) kvh * a.ctx * kFaHD : nullptr;
    const size_t vtb = (size_t) (a.vt / 32) * kFaHD * 32;   // bytes of a KV head's transposed V
    const int8_t* const vq = !Q8 ? nullptr : a.vt > 0 ? a.v8 + (size_t) kvh * vtb : a.v8 + (size_t) kvh * a.ctx * kFaHD;
    const float* const vsf = Q8 && a.vt > 0 ? reinterpret_cast<const float*>(a.v8 + (size_t) a.HKV * vtb) + (size_t) kvh * a.vt : nullptr;
    const uint16_t* const ksc =
        Q8 ? reinterpret_cast<const uint16_t*>(a.k8 + (size_t) a.HKV * a.ctx * kFaHD) + (size_t) kvh * a.ctx * (kFaHD / 64) : nullptr;
    const uint16_t* const vsc = Q8 && a.vt == 0
        ? reinterpret_cast<const uint16_t*>(a.v8 + (size_t) a.HKV * a.ctx * kFaHD) + (size_t) kvh * a.ctx * (kFaHD / 64) : nullptr;
    const int nsg = a.G * kFaSlices;
    uint32_t* const kst = reinterpret_cast<uint32_t*>(smem + (size_t) nsg * kFaSgBytes);
    uint32_t* const vst = kst + kFaStageKBytes / 4;
    sycl::half* const ksh = reinterpret_cast<sycl::half*>(kst);
    sycl::half* const vsh = reinterpret_cast<sycl::half*>(vst);
    const int nwi = nsg * 16, lid = (int) it.get_local_id(2);
    const int r = lane & (kFaRows - 1), hf = lane >> 3;       // softmax: row r, cells [32 hf, 32 hf + 32) of a block
    const int lim = a.p0 + tok0 + r;                          // row r's last cell
    const int nblk = nval > 0 ? (a.p0 + tok0 + nval + kFaBC - 1) / kFaBC : 0;   // the slice's blocks (M4b's tile's)
    const int wval = sycl::min(kFaTok, a.T - wtok0);                           // tokens of the work-group (>= 1)
    const int wblk = (a.p0 + wtok0 + wval + kFaBC - 1) / kFaBC;                // its blocks: its last slice's
    const int nloop = Q8 ? wblk : nblk;   // staged: every work-item through every block of the work-group (the barriers)

    FaAcc O[kFaDT];
#pragma unroll
    for (int d = 0; d < kFaDT; ++d) jm::joint_matrix_fill(sg, O[d], 0.0f);
    float m_run = -INFINITY, l_run = 0.0f;

    for (int b = 0; b < nloop; ++b) {
        const int c0 = b * kFaBC;
        if constexpr (Q8) {   // the block's K^T and V into the staging tiles (every sub-group read the last block's)
            it.barrier(sycl::access::fence_space::local_space);
            fa_stage_q8(lid, nwi, kq, ksc, vq, vsc, c0, kst, vst, a.vt, vsf);
            it.barrier(sycl::access::fence_space::local_space);
        }
        if (b >= nblk) continue;   // past the slice's cells (sub-group uniform; the staged kernel's next barrier follows)
        // S = Q K^T over the block's cells
        FaAcc S[kFaNT];
#pragma unroll
        for (int ct = 0; ct < kFaNT; ++ct) jm::joint_matrix_fill(sg, S[ct], 0.0f);
#pragma unroll 2
        for (int kk = 0; kk < kFaDT; ++kk) {
            FaA A;
            jm::joint_matrix_load(sg, A, fa_gptr(qb + kk * 16), rs);
#pragma unroll
            for (int ct = 0; ct < kFaNT; ++ct) {
                if constexpr (Q8) {
                    FaBp B;
                    jm::joint_matrix_load(sg, B, fa_lptr(ksh + (size_t) (kk * kFaNT + ct) * kFaTileH), (size_t) 32);
                    jm::joint_matrix_mad(sg, S[ct], A, B, S[ct]);
                } else {
                    FaBk B;
                    jm::joint_matrix_load(sg, B, fa_gptr(kb + (size_t) (c0 + ct * 16) * kFaHD + kk * 16), (size_t) kFaHD);
                    jm::joint_matrix_mad(sg, S[ct], A, B, S[ct]);
                }
            }
        }
        sycl::group_barrier(sg);   // the previous block's reads of P (over the scores) are done
#pragma unroll
        for (int ct = 0; ct < kFaNT; ++ct)
            jm::joint_matrix_store(sg, S[ct], fa_lptr(ss + ct * 16), (size_t) kFaSLD, jm::layout::row_major);
        sycl::group_barrier(sg);
        // the online softmax of row r over this lane's 32 cells (the other 32 in lane l ^ 8)
        const float* srow = ss + r * kFaSLD + hf * (kFaBC / 2);
        sycl::half* prow = ps + r * kFaPLD + hf * (kFaBC / 2);
        const int cb = c0 + hf * (kFaBC / 2);
        float xs[kFaBC / 2];   // (16-byte loads and stores below: the rows start at 16-byte multiples)
        float mx = -INFINITY;
#pragma unroll
        for (int g4 = 0; g4 < kFaBC / 8; ++g4) {
            const sycl::vec<float, 4> s4 = *reinterpret_cast<const sycl::vec<float, 4>*>(srow + 4 * g4);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int j = 4 * g4 + e;
                xs[j] = cb + j <= lim ? s4[e] * a.scale_log2 : -INFINITY;
                mx = sycl::fmax(mx, xs[j]);
            }
        }
        sycl::group_barrier(sg);   // every lane's scores are in its registers: P may go over them
        mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, 8));
        const float m_new = sycl::fmax(m_run, mx);
        const float alpha = m_new == m_run ? 1.0f : m_run == -INFINITY ? 0.0f : sycl::native::exp2(m_run - m_new);
        float sum = 0.0f;
#pragma unroll
        for (int g8 = 0; g8 < kFaBC / 16; ++g8) {
            sycl::vec<sycl::half, 8> p8;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float x = xs[8 * g8 + e];
                const float p = x == -INFINITY ? 0.0f : sycl::native::exp2(x - m_new);
                p8[e] = sycl::half(p);
                sum += p;
            }
            *reinterpret_cast<sycl::vec<sycl::half, 8>*>(prow + 8 * g8) = p8;
        }
        sum += sycl::permute_group_by_xor(sg, sum, 8);
        l_run = l_run * alpha + sum;
        m_run = m_new;
        // O was 0 before the first block: no rescale there
        if (b > 0 && sycl::any_of_group(sg, alpha != 1.0f)) fa_row_scale(sg, O, alpha, rv);
        sycl::group_barrier(sg);   // P from every lane
        FaA P[kFaNT];
#pragma unroll
        for (int kt = 0; kt < kFaNT; ++kt) jm::joint_matrix_load(sg, P[kt], fa_lptr(ps + kt * 16), (size_t) kFaPLD);
        // O += P V
#pragma unroll
        for (int d = 0; d < kFaDT; ++d) {
#pragma unroll
            for (int kt = 0; kt < kFaNT; ++kt) {
                if constexpr (Q8) {
                    FaBp B;
                    jm::joint_matrix_load(sg, B, fa_lptr(vsh + (size_t) (kt * kFaDT + d) * kFaTileH), (size_t) 32);
                    jm::joint_matrix_mad(sg, O[d], P[kt], B, O[d]);
                } else {
                    FaBv B;
                    jm::joint_matrix_load(sg, B, fa_gptr(vb + (size_t) (c0 + kt * 16) * kFaHD + d * 16), (size_t) kFaHD);
                    jm::joint_matrix_mad(sg, O[d], P[kt], B, O[d]);
                }
            }
        }
    }
    if (nval == 0) return;   // a slice past the chunk (no barrier after the loop)
    fa_row_scale(sg, O, l_run > 0.0f ? 1.0f / l_run : 0.0f, rv);
    float* const ob = a.o + ((size_t) tok0 * a.H + h) * kFaHD;
    if (nval == kFaRows) {
#pragma unroll
        for (int d = 0; d < kFaDT; ++d) jm::joint_matrix_store(sg, O[d], fa_gptr(ob + d * 16), rs, jm::layout::row_major);
    } else {   // the chunk's last slice: its rows inside the chunk, through the score region
#pragma unroll
        for (int d = 0; d < kFaDT; ++d) {
            sycl::group_barrier(sg);
            jm::joint_matrix_store(sg, O[d], fa_lptr(ss), (size_t) 16, jm::layout::row_major);
            sycl::group_barrier(sg);
            for (int rr = 0; rr < nval; ++rr) ob[(size_t) rr * rs + d * 16 + lane] = ss[rr * 16 + lane];
        }
    }
}
class q36_flash_prompt_attn_k;
class q36_flash_prompt_attn_q8_k;
#endif

/// The kernel is in this build (Xe2 AOT) and the device runs 16-lane sub-groups.
inline bool flash_built(sycl::queue& q) {
#ifdef STRATA_XMX_SG16
    const auto sz = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
    return std::find(sz.begin(), sz.end(), (size_t) 16) != sz.end();
#else
    (void) q;
    return false;
#endif
}
/// The shapes the kernel takes (the caller keeps the GEMM path for the rest): G <= 8 (4 G sub-groups a work-group).
inline bool flash_shape_ok(int H, int HKV, int HD, int ctx) {
    return HD == kFaHD && HKV > 0 && H % HKV == 0 && H / HKV >= 1 && H / HKV <= kFaMaxG && ctx > 0 && ctx % kFaBC == 0;
}
/// Local memory of a work-group: the sub-groups' regions, the 8-bit kernel's staging tiles after them.
inline size_t flash_slm_bytes(int G, bool q8) {
    return (size_t) G * kFaSlices * kFaSgBytes + (q8 ? kFaStageBytes : 0);
}
/// The 8-bit cache's kernel also fits its staging tiles in the device's work-group memory.
inline bool flash_q8_ok(sycl::queue& q, int H, int HKV, int HD, int ctx) {
    if (!flash_shape_ok(H, HKV, HD, ctx)) return false;
    return q.get_device().get_info<sycl::info::device::local_mem_size>() >= flash_slm_bytes(H / HKV, true);
}
/// out [T][H][256] = causal attention of the chunk's T > 0 queries (positions p0 .. p0 + T - 1) over the FP16 cache.
inline void flash_prompt_attn(sycl::queue& q, const sycl::half* q16, const sycl::half* kc, const sycl::half* vc, float* out,
                              int T, int p0, int H, int HKV, int ctx) {
#ifdef STRATA_XMX_SG16
    FaArgs a;
    a.q = q16; a.k = kc; a.v = vc; a.o = out;
    a.T = T; a.p0 = p0; a.H = H; a.HKV = HKV; a.G = H / HKV; a.ctx = ctx;
    a.ntiles = (T + kFaTok - 1) / kFaTok;
    a.scale_log2 = 1.4426950408889634f / std::sqrt((float) kFaHD);
    const size_t wg = (size_t) a.G * kFaSlices * 16;
    const size_t slm = flash_slm_bytes(a.G, false) / sizeof(sycl::float4);
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::float4, 1> lm(sycl::range<1>(slm), cgh);
        auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
        cgh.parallel_for<q36_flash_prompt_attn_k>(
            sycl::nd_range<3>(sycl::range<3>(1, (size_t) a.HKV, (size_t) a.ntiles * wg), sycl::range<3>(1, 1, wg)), props,
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
                fa_body<FaKvF16>(it, a, reinterpret_cast<uint8_t*>(lm.get_multi_ptr<sycl::access::decorated::no>().get()));
            });
    });
#else
    (void) q; (void) q16; (void) kc; (void) vc; (void) out; (void) T; (void) p0; (void) H; (void) HKV; (void) ctx;
#endif
}
/// The same over the 8-bit cache (kc / vc: a layer's K / V allocation, codes then scales; flash_q8_ok).
inline void flash_prompt_attn_q8(sycl::queue& q, int vt, const sycl::half* q16, const int8_t* kc, const int8_t* vc, float* out,
                                 int T, int p0, int H, int HKV, int ctx) {
#ifdef STRATA_XMX_SG16
    FaArgs a;
    a.q = q16; a.k = nullptr; a.v = nullptr; a.k8 = kc; a.v8 = vc; a.o = out; a.vt = vt;
    a.T = T; a.p0 = p0; a.H = H; a.HKV = HKV; a.G = H / HKV; a.ctx = ctx;
    a.ntiles = (T + kFaTok - 1) / kFaTok;
    a.scale_log2 = 1.4426950408889634f / std::sqrt((float) kFaHD);
    const size_t wg = (size_t) a.G * kFaSlices * 16;
    const size_t slm = flash_slm_bytes(a.G, true) / sizeof(sycl::float4);
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::float4, 1> lm(sycl::range<1>(slm), cgh);
        auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
        cgh.parallel_for<q36_flash_prompt_attn_q8_k>(
            sycl::nd_range<3>(sycl::range<3>(1, (size_t) a.HKV, (size_t) a.ntiles * wg), sycl::range<3>(1, 1, wg)), props,
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
                fa_body<FaKvQ8>(it, a, reinterpret_cast<uint8_t*>(lm.get_multi_ptr<sycl::access::decorated::no>().get()));
            });
    });
#else
    (void) q; (void) q16; (void) kc; (void) vc; (void) out; (void) T; (void) p0; (void) H; (void) HKV; (void) ctx;
#endif
}

// ------------------------------------------------------------------------------------------------ LP3: chunk_prefill
// The 8-bit cache's prompt attention on vllm-xpu-kernels' Xe2 chunk_prefill (q36_tla.hpp, libq36_tla.so; the default
// for the 8-bit cache in an STRATA_Q36_TLA build, STRATA_Q36_PATTN=lp2: flash_prompt_attn_q8 above).  Their kernel reads
// FP16 K / V, so the chunk's cells [0, p0 + T) of a layer are dequantized to FP16 (x' = fp16(code * scale), the
// operands flash_prompt_attn_q8 stages) segment by segment into one segment's buffers: a full FP16 copy of a layer's
// 262144 cells (1 GiB) does not fit beside the 27B and its cache.  The segment holding the chunk's own cells
// [K - Lc, K), K = p0 + T, Lc = min(K, seg), runs causal (their bottom-right causal mask: query t sees cells
// [0, Lc - T + t] of the segment = [0, p0 + t] of the cache); the cells before it in segments of seg cells (the last
// shorter) run without a mask.  Each segment's normalised output (FP16) and its natural-log LSE are merged into the
// running FP32 output by vllm-xpu-kernels' merge_attn_states (FP32 in, FP32 out; merge_attn_states.hpp beside this file).
// One segment (K <= seg): the causal launch alone, its output widened to FP32.  Kernels of this section: the dequant,
// the FP16 -> FP32 widening, the cumulative-length table, merge_attn_states; theirs is the attention.
#ifdef STRATA_Q36_TLA
}  // namespace q36pf
#include "q36_tla.hpp"
#include "merge_attn_states.hpp"
namespace q36pf {

constexpr int kTlaSeg = 32768;   // cells of a segment (STRATA_Q36_TLA_SEG; at least the chunk, a multiple of 64)

/// The segment buffers (one set, every attention layer in turn).  seg: cells of a segment; T: the chunk's most queries.
struct TlaBufs {
    int vt = 0;                  // > 0: the 8-bit cache's V is transposed (the padded cell count)
    sycl::half* k16 = nullptr;   // [seg][HKV][256] FP16
    sycl::half* v16 = nullptr;   // [seg][HKV][256] FP16
    sycl::half* o16 = nullptr;   // [T][H][256]: a segment's output (normalised over its cells)
    float* of = nullptr;         // [T][H][256]: o16 widened (the merge's suffix)
    float* lse[3] = {};          // [H][T] each: the running output's, the segment's, the merge's (no in-place LSE)
    int32_t* cu = nullptr;       // [8] {0, T, 0, Lc, 0, seg, 0, last}: the kernel's cumulative lengths per launch kind
    int seg = 0, T = 0, H = 0, HKV = 0;
    bool ok() const { return k16 && v16 && o16 && of && lse[0] && lse[1] && lse[2] && cu && seg > 0; }
};
/// Device bytes of the buffers (the VRAM estimate).
inline size_t tla_bytes(int seg, int T, int H, int HKV) {
    return (size_t) 2 * seg * HKV * kFaHD * 2 + (size_t) T * H * kFaHD * (2 + 4) + (size_t) 3 * H * T * 4 + 8 * 4;
}
/// The shape the kernel takes: head 256, whole query heads per KV head, a segment holding a chunk.
inline bool tla_shape_ok(int H, int HKV, int HD, int seg, int T) {
    return HD == kFaHD && HD == q36tla::kHeadDim && HKV > 0 && H % HKV == 0 && T > 0 && seg >= T && seg % kFaBC == 0;
}

class q36_tla_kv16_k;
class q36_tla_h2f_k;
class q36_tla_cu_k;
class q36_tla_merge_k;

/// Cells [c0, c0 + L) of the layer's 8-bit K and V (codes [HKV][ctx][256], FP16 scales [HKV][ctx][4] after them) into
/// k16 / v16 [L][HKV][256] FP16: x' = fp16((float) code * scale), as fa_stage_q8.  A work-item: 16 values of one cell
/// of one KV head, its K and its V.
inline void tla_kv16(sycl::queue& q, const int8_t* kc, const int8_t* vc, sycl::half* k16, sycl::half* v16, int c0, int L,
                     int HKV, int ctx, int vt = 0) {   // (vt > 0: V transposed, a scale a cell - fa_stage_q8)
    constexpr int G = kFaHD / 16;   // 16-value groups of a cell
    const size_t n = (size_t) HKV * L * G;
    const size_t wg = 256;
    q.parallel_for<q36_tla_kv16_k>(sycl::nd_range<1>((n + wg - 1) / wg * wg, wg), [=](sycl::nd_item<1> it) {
        const size_t x = it.get_global_id(0);
        if (x >= n) return;
        const int gi = (int) (x % G);
        const size_t hc = x / G;   // (KV head, cell of the segment), the cell fastest
        const int c = (int) (hc % (size_t) L), h = (int) (hc / (size_t) L);
        const size_t cell = (size_t) h * ctx + (size_t) (c0 + c);   // in [HKV][ctx]
        const size_t sidx = cell * (kFaHD / 64) + (size_t) (gi / 4);
        const uint16_t* const ks = reinterpret_cast<const uint16_t*>(kc + (size_t) HKV * ctx * kFaHD);
        const uint16_t* const vs = reinterpret_cast<const uint16_t*>(vc + (size_t) HKV * ctx * kFaHD);
        const float sk = (float) sycl::bit_cast<sycl::half>(ks[sidx]);
        float sv;
        const sycl::vec<int8_t, 16> a = *reinterpret_cast<const sycl::vec<int8_t, 16>*>(kc + cell * kFaHD + gi * 16);
        sycl::vec<int8_t, 16> b;
        if (vt > 0) {
            const size_t cc = (size_t) (c0 + c), vtb = (size_t) (vt / 32) * kFaHD * 32;
            const int8_t* const pv = vc + (size_t) h * vtb + ((cc >> 5) * kFaHD + (size_t) gi * 16) * 32 + (cc & 31);
#pragma unroll
            for (int j = 0; j < 16; ++j) b[j] = pv[j * 32];
            sv = reinterpret_cast<const float*>(vc + (size_t) HKV * vtb)[(size_t) h * vt + cc];
        } else {
            sv = (float) sycl::bit_cast<sycl::half>(vs[sidx]);
            b = *reinterpret_cast<const sycl::vec<int8_t, 16>*>(vc + cell * kFaHD + gi * 16);
        }
        sycl::vec<sycl::half, 16> ka, va;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            ka[j] = sycl::half((float) a[j] * sk);
            va[j] = sycl::half((float) b[j] * sv);
        }
        const size_t dst = ((size_t) c * HKV + h) * kFaHD + (size_t) gi * 16;
        *reinterpret_cast<sycl::vec<sycl::half, 16>*>(k16 + dst) = ka;
        *reinterpret_cast<sycl::vec<sycl::half, 16>*>(v16 + dst) = va;
    });
}
/// dst[i] = (float) src[i], n a multiple of 8.
inline void tla_h2f(sycl::queue& q, const sycl::half* src, float* dst, size_t n) {
    const size_t n8 = n / 8, wg = 256;
    q.parallel_for<q36_tla_h2f_k>(sycl::nd_range<1>((n8 + wg - 1) / wg * wg, wg), [=](sycl::nd_item<1> it) {
        const size_t x = it.get_global_id(0);
        if (x >= n8) return;
        const sycl::vec<sycl::half, 8> h = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(src + x * 8);
        sycl::vec<float, 8> f;
#pragma unroll
        for (int j = 0; j < 8; ++j) f[j] = (float) h[j];
        *reinterpret_cast<sycl::vec<float, 8>*>(dst + x * 8) = f;
    });
}
/// out [T][H][256] (in place, its LSE in *lse_run, the merged LSE into *lse_nxt): vllm-xpu-kernels'
/// merge_attn_states_launcher for an FP32 output (NUM_THREADS 128, one 16-byte pack a work-item), prefix = out,
/// suffix = of / lse_seg, every token merged.
inline void tla_merge(sycl::queue& q, float* out, float* lse_nxt, const float* lse_run, const float* of,
                      const float* lse_seg, int T, int H) {
    constexpr uint NUM_THREADS = 128;
    const uint num_tokens = (uint) T, num_heads = (uint) H, head_size = (uint) kFaHD;
    const uint pack_size = 16 / sizeof(float);
    const uint threads_per_head = head_size / pack_size;
    const uint total_threads = num_tokens * num_heads * threads_per_head;
    const sycl::range<3> block(1, 1, NUM_THREADS);
    const sycl::range<3> grid(1, 1, (total_threads + NUM_THREADS - 1) / NUM_THREADS);
    q.parallel_for<q36_tla_merge_k>(sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> it) {
        vllm::merge_attn_states_kernel<float, float, NUM_THREADS>(out, lse_nxt, out, lse_run, of, lse_seg, num_tokens,
                                                                  num_heads, head_size, head_size, head_size,
                                                                  num_tokens, it);
    });
}

/// out [T][H][256] = causal attention of the chunk's T queries (positions p0 .. p0 + T - 1, q16 [T][H][256]) over the
/// layer's 8-bit cache (kc / vc: its K / V allocations, codes then scales), on chunk_prefill segment by segment.
inline void tla_prompt_attn_q8(sycl::queue& q, const TlaBufs& b, const sycl::half* q16, const int8_t* kc, const int8_t* vc,
                               float* out, int T, int p0, int H, int HKV, int ctx) {
    const int K = p0 + T;                   // cells the chunk sees: [0, K)
    const int Lc = std::min(K, b.seg);      // the causal segment: [K - Lc, K)
    const int nrest = K - Lc;               // cells before it: segments [s0, s0 + seg), the last one `last` long
    const int last = nrest % b.seg ? nrest % b.seg : b.seg;
    const int seg = b.seg;
    int32_t* const cu = b.cu;
    q.single_task<q36_tla_cu_k>([=]() {
        cu[0] = 0; cu[1] = T;
        cu[2] = 0; cu[3] = Lc;
        cu[4] = 0; cu[5] = seg;
        cu[6] = 0; cu[7] = last;
    });
    q36tla::Fmha f;
    f.q = q16; f.k = b.k16; f.v = b.v16; f.o = b.o16;
    f.cu_q = cu;
    f.T = T; f.H = H; f.HKV = HKV;
    f.scale = 1.0f / std::sqrt((float) kFaHD);
    const size_t n = (size_t) T * H * kFaHD;
    // the chunk's own segment, causal: straight into out
    tla_kv16(q, kc, vc, b.k16, b.v16, K - Lc, Lc, HKV, ctx, b.vt);
    f.L = Lc; f.cu_k = cu + 2; f.lse = b.lse[0]; f.causal = true;
    q36tla::fmha(q, f);
    tla_h2f(q, b.o16, out, n);
    // the cells before it, unmasked, each merged into out
    const float* run = b.lse[0];
    float* nxt = b.lse[2];
    for (int s0 = 0; s0 < nrest; s0 += seg) {
        const int L = std::min(seg, nrest - s0);
        tla_kv16(q, kc, vc, b.k16, b.v16, s0, L, HKV, ctx, b.vt);
        f.L = L; f.cu_k = L == seg ? cu + 4 : cu + 6; f.lse = b.lse[1]; f.causal = false;
        q36tla::fmha(q, f);
        tla_h2f(q, b.o16, b.of, n);
        tla_merge(q, out, nxt, run, b.of, b.lse[1], T, H);
        float* const was = const_cast<float*>(run);
        run = nxt;
        nxt = was;
    }
}
#endif

// ------------------------------------------------------------------------------------------------ GDN recurrence
constexpr int kGxS = 128, kGxHK = 16, kGxHV = 48, kGxC = 10240;
constexpr int kGxRCB = 16, kGxRNCB = kGxS / kGxRCB;   // columns per block, blocks per head

/// The geometry gdn_xe_rg is compiled for, and a device with 16-lane sub-groups.
inline bool gdn_xe_ok(sycl::queue& q, int S, int HK, int HV, int C) {
    if (S != kGxS || HK != kGxHK || HV != kGxHV || C != kGxC) return false;
    const auto sz = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
    return std::find(sz.begin(), sz.end(), (size_t) 16) != sz.end();
}

// (gdn_xe.hpp's gdn_xe_rg_body, verbatim but for the names.)  A 16-lane sub-group holds 16 columns (a lane
// each) x one row group (32 rows, 32 GRFs), a work-group the four row groups of one 16-column block.  One pass over a
// thread's rows per token does the update, its q^T S partial and the next token's k^T S partial; the partials meet in
// SLM (two banks by token parity) behind one barrier a token.
inline void gdn_xe_rg_body(sycl::nd_item<1> it, float* __restrict__ state, const float* __restrict__ h,
                           const float* __restrict__ gate, const float* __restrict__ beta, float* __restrict__ out,
                           int64_t T, float* __restrict__ red) {   // red: [2 banks][2: kv, o][4 rg][16]
    const auto sg = it.get_sub_group();
    const unsigned lane = sg.get_local_linear_id(), rg = sg.get_group_linear_id();
    const unsigned w = it.get_group_linear_id();
    const unsigned head = w / kGxRNCB, col = (w % kGxRNCB) * kGxRCB + lane;
    const unsigned qh = head % kGxHK;
    constexpr unsigned RPG = kGxS / 4, NR = RPG / 16;   // rows per row group, 16-row pieces of them
    constexpr size_t rs = (size_t) kGxHV * kGxS;
    constexpr unsigned BANK = 2 * 4 * 16, KO = 4 * 16;
    float* base = state + (size_t) (rg * RPG) * rs + head * kGxS + col;
    float s[RPG];
#pragma unroll
    for (unsigned r = 0; r < RPG; ++r) s[r] = base[r * rs];
    // the inputs of tokens t (c*), t + 1 (n*: its k rows feed this token's pass) and t + 2 (f*, loading); zeros past T
    float cq[NR], ck[NR], nq[NR], nk[NR], cv = 0, cg = 0, cbt = 0, nv = 0, ng = 0, nb = 0;
    auto fetch = [&](int64_t t, float* q_, float* k_, float& v_, float& g_, float& b_) {
        if (t < T) {
            const float* ht = h + t * kGxC;
#pragma unroll
            for (unsigned u = 0; u < NR; ++u) {
                q_[u] = ht[qh * kGxS + rg * RPG + u * 16 + lane];
                k_[u] = ht[kGxHK * kGxS + qh * kGxS + rg * RPG + u * 16 + lane];
            }
            v_ = ht[2 * kGxHK * kGxS + head * kGxS + col];
            g_ = gate[t * kGxHV + head];
            b_ = beta[t * kGxHV + head];
        } else {
#pragma unroll
            for (unsigned u = 0; u < NR; ++u) q_[u] = k_[u] = 0.0f;
            v_ = g_ = b_ = 0.0f;
        }
    };
    fetch(0, cq, ck, cv, cg, cbt);
    fetch(1, nq, nk, nv, ng, nb);
    // row r of the row group in a token's k / q: lane r % 16 of piece r / 16
    auto bc = [&](const float* x, unsigned r) { return sycl::select_from_group(sg, x[r / 16], r % 16); };
    float* mine = red + rg * 16 + lane;   // this thread's slot ([bank][kv / o] added)
    const float* part = red + lane;       // the column's four row-group slots
    {   // k^T S of token 0
        float kv = 0.0f;
#pragma unroll
        for (unsigned r = 0; r < RPG; ++r) kv = sycl::fma(s[r], bc(ck, r), kv);
        mine[0] = kv;
    }
    it.barrier(sycl::access::fence_space::local_space);
    float kv_col = part[0] + part[16] + part[32] + part[48];
    for (int64_t t = 0; t < T; ++t) {
        float fq[NR], fk[NR], fv, fg, fb;
        fetch(t + 2, fq, fk, fv, fg, fb);
        const unsigned b0 = (t & 1) * BANK, b1 = BANK - b0;   // this token's bank, the next one's
        const float g = sycl::native::exp(cg);
        const float delta = (cv - g * kv_col) * cbt;
        float o = 0.0f, kv = 0.0f;
#pragma unroll
        for (unsigned r = 0; r < RPG; ++r) {
            s[r] = sycl::fma(g, s[r], bc(ck, r) * delta);
            o = sycl::fma(s[r], bc(cq, r), o);
            kv = sycl::fma(s[r], bc(nk, r), kv);
        }
        // bank b1's kv slots were last read before the previous barrier, bank b0's o slots (row group 0) before it too
        mine[b0 + KO] = o;
        mine[b1] = kv;
        it.barrier(sycl::access::fence_space::local_space);
        kv_col = part[b1] + part[b1 + 16] + part[b1 + 32] + part[b1 + 48];   // the next token's k^T S
        if (rg == 0)
            out[t * kGxHV * kGxS + head * kGxS + col] =
                (part[b0 + KO] + part[b0 + KO + 16] + part[b0 + KO + 32] + part[b0 + KO + 48]) * sycl::rsqrt(float(kGxS));
#pragma unroll
        for (unsigned u = 0; u < NR; ++u) { cq[u] = nq[u]; ck[u] = nk[u]; nq[u] = fq[u]; nk[u] = fk[u]; }
        cv = nv; cg = ng; cbt = nb; nv = fv; ng = fg; nb = fb;
    }
#pragma unroll
    for (unsigned r = 0; r < RPG; ++r) base[r * rs] = s[r];
}
class q36_gdn_xe_rg_k;

/// The prompt recurrence over T > 0 tokens: state [128 rows][48 heads][128 cols] advanced in place, out [T][48][128]
/// the raw outputs (k_gdn_norm's input); h [T][10240] the conv outputs (q | k | v), gate / beta [T][48].
inline void gdn_xe_rg(sycl::queue& q, float* state, const float* h, const float* gate, const float* beta, float* out,
                      int64_t T) {
    const auto range = sycl::nd_range<1>((size_t) kGxHV * kGxRNCB * 4 * 16, 4 * 16);
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> red(2 * 2 * 4 * 16, cgh);
        cgh.parallel_for<q36_gdn_xe_rg_k>(range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            gdn_xe_rg_body(it, state, h, gate, beta, out, T, red.template get_multi_ptr<sycl::access::decorated::no>().get());
        });
    });
}

}  // namespace q36pf
