// q36_qsa.dp.cpp - Qwen Sparse Attention for the 27B on the Arc Pro B70.  See q36_qsa.hpp for what is computed; this file is how.
//
// Plain SYCL (no ESIMD / DPAS): every kernel here is memory- or latency-bound at decode shapes (the byte arithmetic is in REPORT.md);
// the matrix engine is a follow-up for the scoring of many rows (prompt chunks).  Nothing has run on the card yet: the parity program
// (q36_qsa_parity.cpp, q36_mmvq_dense_parity --qsa) checks each kernel against a float64 reference.
//
// Determinism contract (the verify window's rows are bitwise the one-row decode step's rows):
//   project   a sub-group of 16 lanes owns one output; lane l sums k = it*128 + l*8 + j in the order it, j; the 16 lanes are added by
//             an xor butterfly 8, 4, 2, 1.  No dependence on the number of rows of the tile.
//   score     the dot of 128 dims in order d = 0..127 into one accumulator per (row, head); heads added 0..3.
//   select    integer counts only; ties go to the lower block index; the list is written in ascending order.
//   attend    one work-group per (piece of the row's own list, KV head, row); a fixed 256-lane layout; sums by butterfly then a
//             sequential sum of the 16 sub-group partials.
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "q36_qsa.hpp"

namespace strata::q36qsa {
namespace {

using sycl::half;

inline sycl::queue& Q(void* s) { return *static_cast<sycl::queue*>(s); }

struct TagProj; struct TagPrep; struct TagPool; struct TagRingTree; struct TagRingUpd; struct TagScore1; struct TagScore2;
struct TagScore4; struct TagScore8; struct TagSelect; struct TagAttend; struct TagMerge;

inline float h2f(uint16_t u) { return static_cast<float>(sycl::bit_cast<half>(u)); }
// the sort key of a score: its bits without the sign (scores are sums of relu's, so >= 0; a -0.0 must not outrank every positive score)
inline uint32_t skey(float v) { return sycl::bit_cast<uint32_t>(v) & 0x7FFFFFFFu; }

// the 16 lanes of a sub-group added by an xor butterfly (the same order in every lane and for every caller)
inline float sg_sum16(sycl::sub_group sg, float v) {
    v += sycl::permute_group_by_xor(sg, v, 8);
    v += sycl::permute_group_by_xor(sg, v, 4);
    v += sycl::permute_group_by_xor(sg, v, 2);
    v += sycl::permute_group_by_xor(sg, v, 1);
    return v;
}
inline float sg_max16(sycl::sub_group sg, float v) {
    v = sycl::fmax(v, sycl::permute_group_by_xor(sg, v, 8));
    v = sycl::fmax(v, sycl::permute_group_by_xor(sg, v, 4));
    v = sycl::fmax(v, sycl::permute_group_by_xor(sg, v, 2));
    v = sycl::fmax(v, sycl::permute_group_by_xor(sg, v, 1));
    return v;
}

// 128 work-items (8 sub-groups): the sum of one value each, in a fixed order, every work-item gets it.  sm: local float[8].
inline float wg_sum128(sycl::nd_item<1> it, float v, float* sm) {
    auto sg = it.get_sub_group();
    v = sg_sum16(sg, v);
    if (sg.get_local_id()[0] == 0) sm[sg.get_group_id()[0]] = v;
    sycl::group_barrier(it.get_group());
    float r = sm[0];
#pragma unroll
    for (int k = 1; k < 8; ++k) r += sm[k];
    sycl::group_barrier(it.get_group());
    return r;
}

// NEOX rotation of the first n_rot dims of a 128-vector held in local memory buf (work-item i owns element i), position rp.
inline float rope_elem(const float* buf, int i, float rpos, float base, int n_rot) {
    float v = buf[i];
    if (i < n_rot) {
        const int hlf = n_rot / 2;
        const int p = i < hlf ? i : i - hlf;
        const float theta = rpos * sycl::pow(base, -2.0f * (float) p / (float) n_rot);
        const float sn = sycl::sin(theta), cs = sycl::cos(theta);
        v = i < hlf ? buf[i] * cs - buf[i + hlf] * sn : buf[i - hlf] * sn + buf[i] * cs;
    }
    return v;
}

// the cell and rope position of row t of a window
inline int row_cell(const Win& w, int t) { return *w.pos0 + w.row0 + t; }
inline int row_delta(const Win& w, int t) { return w.delta_rows ? w.delta_rows[w.row0 + t] : (w.delta ? *w.delta : 0); }
// complete blocks a row can score: those whose last cell is at or before it; a tree window: the blocks wholly before the root
inline int row_vis_blocks(const Win& w, int t) { return w.tp0 ? (*w.tp0) / kRatio : (row_cell(w, t) + 1) / kRatio; }

}   // namespace

// ================================================================================================================ host side

bool load_weights(const char* path, Weights& out, void* stream, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = std::string("cannot open ") + path; return false; }
    char magic[8];
    f.read(magic, 8);
    if (!f || std::memcmp(magic, "QSAIDX\0\1", 8) != 0) { err = "not a qsa_extract.py file (magic)"; return false; }
    uint64_t jl = 0;
    f.read(reinterpret_cast<char*>(&jl), 8);
    std::string js(jl, '\0');
    f.read(&js[0], (std::streamsize) jl);
    if (!f) { err = "truncated header"; return false; }
    const size_t data0 = (16 + jl + 4095) / 4096 * 4096;
    // the JSON is ours and flat: find each "key":{...} and read off / bytes / dtype by hand
    auto num_after = [&](size_t from, const char* key, long long& v) -> bool {
        const size_t p = js.find(key, from);
        if (p == std::string::npos) return false;
        v = std::atoll(js.c_str() + p + std::strlen(key));
        return true;
    };
    // layers
    {
        const size_t p = js.find("\"layers\":[");
        if (p == std::string::npos) { err = "no layers in the index"; return false; }
        const char* c = js.c_str() + p + 10;
        while (*c && *c != ']') {
            out.layer_ids.push_back(std::atoi(c));
            while (*c && *c != ',' && *c != ']') ++c;
            if (*c == ',') ++c;
        }
    }
    if (out.layer_ids.size() != 16) { err = "expected 16 layers"; return false; }
    if (js.find("\"n_embd\":5120") == std::string::npos || js.find("\"idx_dim\":128") == std::string::npos ||
        js.find("\"idx_heads\":4") == std::string::npos || js.find("\"ratio\":4") == std::string::npos ||
        js.find("\"top_k\":2048") == std::string::npos) { err = "the file's geometry is not the 27B's QSA (5120 / 128 x 4 / ratio 4 / top_k 2048)"; return false; }
    sycl::queue& q = Q(stream);
    std::vector<unsigned char> host;
    auto upload = [&](int L, const char* name, size_t n_elems, bool wt, const void*& dev) -> bool {
        char key[64];
        std::snprintf(key, sizeof key, "\"%d.%s\":{", L, name);
        const size_t p = js.find(key);
        if (p == std::string::npos) { err = std::string("missing tensor ") + key; return false; }
        long long off = 0, bytes = 0;
        if (!num_after(p, "\"off\":", off) || !num_after(p, "\"bytes\":", bytes)) { err = "bad index entry"; return false; }
        const size_t pd = js.find("\"dtype\":\"", p);
        const bool is32 = pd != std::string::npos && js.compare(pd + 9, 3, "f32") == 0;
        if ((size_t) bytes != n_elems * (is32 ? 4 : 2)) { err = std::string("size of ") + key; return false; }
        host.resize((size_t) bytes);
        f.seekg((std::streamoff) (data0 + (size_t) off));
        f.read(reinterpret_cast<char*>(host.data()), bytes);
        if (!f) { err = "truncated data"; return false; }
        if (wt) {   // weights: fp16 on the device
            half* d = sycl::malloc_device<half>(n_elems, q);
            if (!d) { err = "device allocation failed"; return false; }
            if (is32) {
                std::vector<half> h16(n_elems);
                const float* s = reinterpret_cast<const float*>(host.data());
                for (size_t i = 0; i < n_elems; ++i) h16[i] = half(s[i]);
                q.memcpy(d, h16.data(), n_elems * 2).wait();
                out.from_f32 = true;
            } else {
                q.memcpy(d, host.data(), n_elems * 2).wait();
            }
            out.bytes += n_elems * 2;
            dev = d;
        } else {
            if (!is32) { err = std::string("norm ") + key + " is not fp32"; return false; }
            float* d = sycl::malloc_device<float>(n_elems, q);
            if (!d) { err = "device allocation failed"; return false; }
            q.memcpy(d, host.data(), n_elems * 4).wait();
            out.bytes += n_elems * 4;
            dev = d;
        }
        return true;
    };
    for (int L : out.layer_ids) {
        LayerWeights w;
        const void *wq = nullptr, *wk = nullptr, *qn = nullptr, *kp = nullptr, *kn = nullptr;
        if (!upload(L, "q_proj", (size_t) kHeads * kD * kEmbd, true, wq) || !upload(L, "k_proj", (size_t) kD * kEmbd, true, wk) ||
            !upload(L, "q_norm", kD, false, qn) || !upload(L, "k_pre", kD, false, kp) || !upload(L, "k_norm", kD, false, kn))
            return false;
        w.wq = wq; w.wk = wk; w.qn = static_cast<const float*>(qn); w.kpre = static_cast<const float*>(kp); w.knorm = static_cast<const float*>(kn);
        out.w.push_back(w);
    }
    return true;
}

void free_weights(Weights& w, void* stream) {
    sycl::queue& q = Q(stream);
    for (auto& l : w.w) {
        sycl::free(const_cast<void*>(l.wq), q); sycl::free(const_cast<void*>(l.wk), q);
        sycl::free(const_cast<float*>(l.qn), q); sycl::free(const_cast<float*>(l.kpre), q); sycl::free(const_cast<float*>(l.knorm), q);
    }
    w.w.clear(); w.layer_ids.clear(); w.bytes = 0;
}

size_t state_bytes(int ctx) {
    return (size_t) nb_cap_for(ctx) * kD * 2 + (size_t) kRing * kD * 4 + (size_t) kMaxWin * kD * 4;
}

bool alloc_state(LayerState& s, int ctx, void* stream) {
    sycl::queue& q = Q(stream);
    const size_t pb = (size_t) nb_cap_for(ctx) * kD * 2;
    s.pooled = sycl::malloc_device(pb, q);
    s.ring = sycl::malloc_device<float>((size_t) kRing * kD, q);
    s.traw = sycl::malloc_device<float>((size_t) kMaxWin * kD, q);
    if (!s.pooled || !s.ring || !s.traw) return false;
    q.memset(s.pooled, 0, pb);
    q.memset(s.ring, 0, (size_t) kRing * kD * 4);
    q.memset(s.traw, 0, (size_t) kMaxWin * kD * 4).wait();
    return true;
}
void free_state(LayerState& s, void* stream) {
    sycl::queue& q = Q(stream);
    sycl::free(s.pooled, q); sycl::free(s.ring, q); sycl::free(s.traw, q);
    s = LayerState{};
}
void copy_state(const LayerState& src, const LayerState& dst, int cells, int ctx, void* stream) {
    sycl::queue& q = Q(stream);
    const size_t nb = std::min<size_t>((size_t) cells / kRatio + 1, (size_t) nb_cap_for(ctx));
    q.memcpy(dst.pooled, src.pooled, nb * kD * 2);
    q.memcpy(dst.ring, src.ring, (size_t) kRing * kD * 4);
}

size_t scratch_bytes(int tmax, int score_rows, int ctx) {
    const size_t nbc = nb_cap_for(ctx);
    return ((size_t) tmax * (kProj + kHeads * kD + kD) + (size_t) score_rows * nbc + (size_t) kMaxWin * 24 * kPieces * (kHD + 2)) * 4 +
           (size_t) score_rows * (kTopBlocks + 2 + 16) * 4;
}
bool alloc_scratch(Scratch& s, int tmax, int score_rows, int ctx, void* stream) {
    sycl::queue& q = Q(stream);
    s.tmax = tmax; s.score_rows = score_rows; s.nb_cap = nb_cap_for(ctx);
    s.proj = sycl::malloc_device<float>((size_t) tmax * kProj, q);
    s.qidx = sycl::malloc_device<float>((size_t) tmax * kHeads * kD, q);
    s.raw = sycl::malloc_device<float>((size_t) tmax * kD, q);
    s.score = sycl::malloc_device<float>((size_t) score_rows * s.nb_cap, q);
    s.sel = sycl::malloc_device<int32_t>((size_t) score_rows * kTopBlocks, q);
    s.nsel = sycl::malloc_device<int32_t>((size_t) score_rows, q);
    s.dbg = sycl::malloc_device<int32_t>((size_t) score_rows * 16, q);
    s.nitems = sycl::malloc_device<int32_t>((size_t) score_rows, q);
    s.pacc = sycl::malloc_device<float>((size_t) kMaxWin * 24 * kPieces * kHD, q);
    s.pm = sycl::malloc_device<float>((size_t) kMaxWin * 24 * kPieces, q);
    s.pl = sycl::malloc_device<float>((size_t) kMaxWin * 24 * kPieces, q);
    return s.proj && s.qidx && s.raw && s.score && s.sel && s.nsel && s.dbg && s.nitems && s.pacc && s.pm && s.pl;
}
void free_scratch(Scratch& s, void* stream) {
    sycl::queue& q = Q(stream);
    sycl::free(s.proj, q); sycl::free(s.qidx, q); sycl::free(s.raw, q); sycl::free(s.score, q); sycl::free(s.sel, q);
    sycl::free(s.nsel, q); sycl::free(s.dbg, q); sycl::free(s.nitems, q); sycl::free(s.pacc, q); sycl::free(s.pm, q); sycl::free(s.pl, q);
    s = Scratch{};
}

// ================================================================================================================ kernels

namespace {

// ---- project: proj[t][o] = W_o . h[t], o in 0..639 (q rows 0..511 of wq, then the key row of wk), 16 rows a tile -------------------
void launch_project(sycl::queue& q, const half* wq, const half* wk, const float* h, int T, float* proj) {
    const int tiles = (T + 15) / 16;
    const int groups = tiles * (kProj / 4);   // 4 outputs (4 sub-groups of 16 lanes) a work-group
    q.parallel_for<TagProj>(sycl::nd_range<1>((size_t) groups * 64, 64), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        const int g = (int) it.get_group(0);
        const int rt = g / (kProj / 4), og = g % (kProj / 4);
        const int o = og * 4 + (int) sg.get_group_id()[0];
        const int l = (int) sg.get_local_id()[0];
        const half* w = o < kHeads * kD ? wq + (size_t) o * kEmbd : wk + (size_t) (o - kHeads * kD) * kEmbd;
        const int t0 = rt * 16;
        const int nr = sycl::min(16, T - t0);
        float acc[16];
#pragma unroll
        for (int r = 0; r < 16; ++r) acc[r] = 0.f;
#pragma unroll 4   // (4 loads of 256 B a sub-group in flight: 1 KiB, the card's rule for a stream)
        for (int itn = 0; itn < kEmbd / 128; ++itn) {
            const int k = itn * 128 + l * 8;
            const sycl::vec<uint32_t, 4> wv = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(w + k);
            float wf[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t u = wv[j];
                wf[2 * j] = h2f((uint16_t) (u & 0xFFFFu));
                wf[2 * j + 1] = h2f((uint16_t) (u >> 16));
            }
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                if (r < nr) {
                    const float* hp = h + (size_t) (t0 + r) * kEmbd + k;
                    const sycl::float4 a = *reinterpret_cast<const sycl::float4*>(hp);
                    const sycl::float4 b = *reinterpret_cast<const sycl::float4*>(hp + 4);
                    float s = acc[r];
                    s = sycl::fma(wf[0], a.x(), s); s = sycl::fma(wf[1], a.y(), s); s = sycl::fma(wf[2], a.z(), s); s = sycl::fma(wf[3], a.w(), s);
                    s = sycl::fma(wf[4], b.x(), s); s = sycl::fma(wf[5], b.y(), s); s = sycl::fma(wf[6], b.z(), s); s = sycl::fma(wf[7], b.w(), s);
                    acc[r] = s;
                }
            }
        }
#pragma unroll
        for (int r = 0; r < 16; ++r) acc[r] = sg_sum16(sg, acc[r]);
        if (l == 0) {
#pragma unroll
            for (int r = 0; r < 16; ++r)
                if (r < nr) proj[(size_t) (t0 + r) * kProj + o] = acc[r];
        }
    });
}

// ---- prep: per row, 4 query heads (rms * q_norm, rotated -> qidx) and the key (rms * k_pre -> raw, and traw for a tree window) -------
void launch_prep(sycl::queue& q, const float* proj, const float* qn, const float* kpre, Win win, Rope rp, bool tree, float* qidx, float* raw,
                 float* traw) {
    const int T = win.T;
    q.parallel_for<TagPrep>(sycl::nd_range<1>((size_t) T * 5 * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto& sm = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[8]>(it.get_group());
        auto& buf = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(it.get_group());
        const int g = (int) it.get_group(0);
        const int t = g / 5, part = g % 5;
        const int i = (int) it.get_local_id(0);
        const float x = proj[(size_t) t * kProj + part * 128 + i];
        const float s = wg_sum128(it, x * x, sm);
        const float r = sycl::rsqrt(s / 128.0f + rp.eps);
        const float v = x * r * (part < 4 ? qn[i] : kpre[i]);
        const int cell = *win.pos0 + t;
        if (part == 4) {
            raw[(size_t) t * 128 + i] = v;
            if (tree) traw[(size_t) t * 128 + i] = v;   // (the ring is written after the pooling: launch_ring_update)
            return;
        }
        buf[i] = v;
        sycl::group_barrier(it.get_group());
        const int dl = win.delta_rows ? win.delta_rows[t] : (win.delta ? *win.delta : 0);
        qidx[((size_t) t * kHeads + part) * 128 + i] = rope_elem(buf, i, (float) (cell + dl), rp.base, rp.n_rot);
    });
}

// ---- pool: every block whose last cell is among the window's rows: mean of 4 raw keys, rms * k_norm, rotated to the block's first cell ----
// rawwin: the window's raw keys (cell >= p0), null: every member from the ring (a committed tree window).  Rows of the window: [p0, p0 + n).
void launch_pool(sycl::queue& q, const float* rawwin, const float* ring, const float* knorm, const int32_t* pos0, const int32_t* delta, int n,
                 Rope rp, half* pooled) {
    const int nwg = (n + kRatio - 1) / kRatio + 1;
    q.parallel_for<TagPool>(sycl::nd_range<1>((size_t) nwg * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto& sm = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[8]>(it.get_group());
        auto& buf = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(it.get_group());
        const int p0 = *pos0;
        const int b = p0 / kRatio + (int) it.get_group(0);
        const int last = p0 + n - 1;                  // the last cell of the window
        if (b * kRatio + kRatio - 1 > last) return;   // the block is not complete (a work-group-uniform exit)
        const int i = (int) it.get_local_id(0);
        float m[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = b * kRatio + j;
            m[j] = (rawwin != nullptr && c >= p0) ? rawwin[(size_t) (c - p0) * 128 + i] : ring[(size_t) (c % kRing) * 128 + i];
        }
        const float mean = (((m[0] + m[1]) + m[2]) + m[3]) * 0.25f;
        const float s = wg_sum128(it, mean * mean, sm);
        const float r = sycl::rsqrt(s / 128.0f + rp.eps);
        buf[i] = mean * r * knorm[i];
        sycl::group_barrier(it.get_group());
        const int dl = delta ? *delta : 0;
        const float v = rope_elem(buf, i, (float) (b * kRatio + dl), rp.base, rp.n_rot);
        pooled[(size_t) b * 128 + i] = half(v);
    });
}

// ---- ring update, AFTER the pooling: the pooling reads the ring for the (at most 3) cells before the window, and a window of 29..35 rows
// (or a long chunk) writes the slot of a cell 32 later over them.  A window of up to kRing rows writes all its rows, a longer one (a prompt
// chunk, never rolled back) its last 4.
void launch_ring_update(sycl::queue& q, const float* raw, const int32_t* pos0, int T, float* ring) {
    q.parallel_for<TagRingUpd>(sycl::nd_range<1>((size_t) T * 128, 128), [=](sycl::nd_item<1> it) {
        const int t = (int) it.get_group(0), i = (int) it.get_local_id(0);
        if (T <= kRing || t >= T - 4) ring[(size_t) ((*pos0 + t) % kRing) * 128 + i] = raw[(size_t) t * 128 + i];
    });
}

// ---- score: scores[r][b] = sum_h relu(q_{r,h} . pooled_b) / sqrt(128), b below the row's visible blocks --------------------------------
template <int R, class Tag>
void launch_score(sycl::queue& q, const half* pooled, const float* qidx, Win win, int nb_cap, float* score) {
    const int nwg = (nb_cap + 255) / 256;
    q.parallel_for<Tag>(sycl::nd_range<1>((size_t) nwg * 256, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto& sq = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[R * kHeads * kD]>(it.get_group());
        const int nr = win.nr;
        for (int i = (int) it.get_local_id(0); i < nr * kHeads * kD; i += 256) sq[i] = qidx[i];
        sycl::group_barrier(it.get_group());
        const int b = (int) it.get_global_id(0);
        // the largest visible block count of the tile's rows (the last row's)
        const int cmax = row_vis_blocks(win, nr - 1);
        if (b >= cmax || b >= nb_cap) return;
        const half* kp = pooled + (size_t) b * kD;
        float acc[R][kHeads];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int hh = 0; hh < kHeads; ++hh) acc[r][hh] = 0.f;
        for (int j = 0; j < kD / 8; ++j) {
            const sycl::vec<uint32_t, 4> kv = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(kp + j * 8);
            float kf[8];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                kf[2 * e] = h2f((uint16_t) (kv[e] & 0xFFFFu));
                kf[2 * e + 1] = h2f((uint16_t) (kv[e] >> 16));
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (r < nr) {
#pragma unroll
                    for (int hh = 0; hh < kHeads; ++hh) {
                        const float* qp = sq + (r * kHeads + hh) * kD + j * 8;
                        float a = acc[r][hh];
#pragma unroll
                        for (int e = 0; e < 8; ++e) a = sycl::fma(qp[e], kf[e], a);
                        acc[r][hh] = a;
                    }
                }
            }
        }
        const float inv = kInvSqrtD;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r < nr) {
                float s = sycl::fmax(acc[r][0], 0.f);
                s += sycl::fmax(acc[r][1], 0.f);
                s += sycl::fmax(acc[r][2], 0.f);
                s += sycl::fmax(acc[r][3], 0.f);
                score[(size_t) r * nb_cap + b] = s * inv;
            }
        }
    });
}

// ---- select: one work-group (1024) a row, the 512 best blocks by 4 passes of an 8-bit radix select, written in ascending order -------
// Plain SLM only: replicated histograms with local-space atomics (the 16 lanes of a sub-group count into 16 different copies, so the
// low-entropy top digits - the scores of a row share their exponent - do not queue on one address), explicit scans in SLM (no library
// group algorithm, no sub-group shuffle).  dbg (16 ints a row) records what the kernel decided: [0] threshold key, [1] rem (scores equal
// to the threshold that are taken), [2] count above the threshold, [3] count equal to it, [4..7] candidates of pass 0..3, [8] c,
// [9..12] the bin chosen in pass 0..3, [13] number of ids written.
void launch_select(sycl::queue& q, const float* score, Win win, int nb_cap, int32_t* sel, int32_t* nsel, int32_t* dbg) {
    q.parallel_for<TagSelect>(sycl::nd_range<1>((size_t) win.nr * 1024, 1024), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto grp = it.get_group();
        auto& hist = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[16 * 256]>(grp);
        auto& hsum = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[256]>(grp);
        auto& chunk = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[16]>(grp);
        auto& sh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[16]>(grp);
        auto& sa = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[1024]>(grp);
        auto& sb = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[1024]>(grp);
        auto& ta = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[64]>(grp);
        auto& tb = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[64]>(grp);
        const int t = (int) it.get_group(0);
        const int tid = (int) it.get_local_id(0);
        const int c = sycl::min(row_vis_blocks(win, t), nb_cap);
        const float* sc = score + (size_t) t * nb_cap;
        int32_t* out = sel + (size_t) t * kTopBlocks;
        int32_t* dg = dbg + (size_t) t * 16;
        if (c <= kTopBlocks) {
            for (int i = tid; i < c; i += 1024) out[i] = i;
            if (tid == 0) {
                nsel[t] = c;
                for (int k = 0; k < 16; ++k) dg[k] = 0;
                dg[8] = c; dg[13] = c;
            }
            return;
        }
        uint32_t prefix = 0;
        int rem = kTopBlocks;
        const int rep = (tid & 15) * 256;   // this lane's copy of the histogram
        for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - 8 * pass;
            for (int i = tid; i < 16 * 256; i += 1024) hist[i] = 0;
            sycl::group_barrier(grp);
            for (int i = tid; i < c; i += 1024) {
                const uint32_t u = skey(sc[i]);
                if (pass == 0 || (u >> (shift + 8)) == prefix) {
                    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>
                        a(hist[rep + (int) ((u >> shift) & 255u)]);
                    a.fetch_add(1);
                }
            }
            sycl::group_barrier(grp);
            if (tid < 256) {
                int s = 0;
                for (int r = 0; r < 16; ++r) s += hist[r * 256 + tid];
                hsum[tid] = s;
            }
            sycl::group_barrier(grp);
            if (tid < 16) {
                int s = 0;
                for (int k = 0; k < 16; ++k) s += hsum[tid * 16 + k];
                chunk[tid] = s;
            }
            sycl::group_barrier(grp);
            if (tid == 0) {
                // the highest bin b with (count in the bins above b) + hsum[b] >= rem
                int cum = 0, ch = 15, total = 0;
                for (int k = 0; k < 16; ++k) total += chunk[k];
                for (; ch > 0; --ch) {
                    if (cum + chunk[ch] >= rem) break;
                    cum += chunk[ch];
                }
                int bin = ch * 16 + 15;
                for (; bin > ch * 16; --bin) {
                    if (cum + hsum[bin] >= rem) break;
                    cum += hsum[bin];
                }
                sh[0] = bin; sh[1] = cum;
                dg[4 + pass] = total; dg[9 + pass] = bin;
            }
            sycl::group_barrier(grp);
            prefix = (prefix << 8) | (uint32_t) sh[0];
            rem -= sh[1];
            sycl::group_barrier(grp);
        }
        // prefix is now the 512th best score's bits; rem of the scores equal to it are taken, the lowest block indices first
        const int S = (c + 1023) / 1024;
        const int lo = tid * S, hi = sycl::min(c, lo + S);
        int gt = 0, eq = 0;
        for (int i = lo; i < hi; ++i) {
            const uint32_t u = skey(sc[i]);
            gt += u > prefix;
            eq += u == prefix;
        }
        // exclusive scans of gt and eq over the 1024 work-items: 64 chunks of 16 in parallel, then the 64 chunk sums by one work-item
        sa[tid] = gt; sb[tid] = eq;
        sycl::group_barrier(grp);
        if (tid < 64) {
            int g = 0, e = 0;
            for (int k = 0; k < 16; ++k) {
                const int i = tid * 16 + k;
                const int x = sa[i], y = sb[i];
                sa[i] = g; sb[i] = e;
                g += x; e += y;
            }
            ta[tid] = g; tb[tid] = e;
        }
        sycl::group_barrier(grp);
        if (tid == 0) {
            int g = 0, e = 0;
            for (int k = 0; k < 64; ++k) {
                const int x = ta[k], y = tb[k];
                ta[k] = g; tb[k] = e;
                g += x; e += y;
            }
            sh[2] = g; sh[3] = e;
        }
        sycl::group_barrier(grp);
        const int gt_before = sa[tid] + ta[tid >> 4];
        const int eq_before = sb[tid] + tb[tid >> 4];
        // the work-items before this one take min(rem, eq_before) of the equal scores (each takes the first rem - eq_before_j of its own)
        const int take_eq = sycl::min(sycl::max(rem - eq_before, 0), eq);
        int o = gt_before + sycl::min(rem, eq_before), seen = 0;
        for (int i = lo; i < hi; ++i) {
            const uint32_t u = skey(sc[i]);
            if (u > prefix) {
                out[o++] = i;
            } else if (u == prefix) {
                if (seen < take_eq) out[o++] = i;
                ++seen;
            }
        }
        if (tid == 0) {
            nsel[t] = kTopBlocks;
            dg[0] = (int) prefix; dg[1] = rem; dg[2] = sh[2]; dg[3] = sh[3]; dg[8] = c; dg[13] = sh[2] + sycl::min(rem, sh[3]);
        }
    });
}

// ---- attend: the sparse attention of one row's piece of its list for one KV head: 6 query heads share each gathered block ----------
// The row's list: its selected blocks (ascending; the own block is dropped when selected: the tail covers it) then the tail blocks (the
// cells of the own region).  Piece s walks items [s * 64, s * 64 + 64).  Work-item j of 256 owns cell j (item j / 4, cell j % 4) for
// the scores and dimension j for the value sums.  Partials (max, sum, 256 sums) per (row, head, piece) go to pm / pl / pacc.
struct ItemGeom {
    int neff, ntail, tail_lo, tail_hi, n_items;
};
inline ItemGeom item_geom(const Win& w, int t, const int32_t* sel, const int32_t* nsel) {
    ItemGeom g;
    const int cell = row_cell(w, t);
    const int n = nsel[t];
    if (w.tp0) {
        const int root = *w.tp0;
        g.tail_lo = root / kRatio * kRatio;
        g.tail_hi = root + w.T;
        g.neff = n;
    } else {
        g.tail_lo = cell / kRatio * kRatio;
        g.tail_hi = cell + 1;
        g.neff = (n > 0 && sel[(size_t) t * kTopBlocks + n - 1] == cell / kRatio) ? n - 1 : n;
    }
    g.ntail = (g.tail_hi - g.tail_lo + kRatio - 1) / kRatio;
    g.n_items = g.neff + g.ntail;
    return g;
}

void launch_attend(sycl::queue& q, const AttendArgs a, Win win) {
    const int nr = win.nr, HKV = a.HKV, H = a.H, ctx = a.ctx, vt = a.vt;
    const float scale = 1.0f / 16.0f;   // 1 / sqrt(256)
    q.parallel_for<TagAttend>(sycl::nd_range<1>((size_t) kPieces * HKV * nr * 256, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto grp = it.get_group();
        const int gid = (int) it.get_group(0);
        const int s = gid % kPieces, kvh = (gid / kPieces) % HKV, t = gid / (kPieces * HKV);
        const ItemGeom G = item_geom(win, t, a.sel, a.nsel);
        const int i0 = s * kPieceItems;
        if (i0 >= G.n_items) return;                          // (work-group uniform)
        const int i1 = sycl::min(G.n_items, i0 + kPieceItems);
        auto& sq = *sycl::ext::oneapi::group_local_memory_for_overwrite<sycl::float4[kG * kHD / 4]>(grp);    // [6][64] float4
        auto& sw = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[kG * 256]>(grp);                // weights times the V scale
        auto& sblk = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[kPieceItems]>(grp);
        auto& rmax = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[16 * kG]>(grp);
        auto& rsum = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[16 * kG]>(grp);
        auto sg = it.get_sub_group();
        const int j = (int) it.get_local_id(0);
        const int sgi = (int) sg.get_group_id()[0], lane = (int) sg.get_local_id()[0];

        // the queries of the KV head's 6 heads (scaled once)
        for (int i = j; i < kG * kHD / 4; i += 256) {
            const int hh = i / (kHD / 4), d4 = i % (kHD / 4);
            const sycl::float4 v = *reinterpret_cast<const sycl::float4*>(a.q + ((size_t) t * H + kvh * kG + hh) * kHD + d4 * 4);
            sq[i] = v * scale;
        }
        if (j < kPieceItems) {
            const int it_i = i0 + j;
            int blk = 0;
            if (it_i < i1) blk = it_i < G.neff ? a.sel[(size_t) t * kTopBlocks + it_i] : G.tail_lo / kRatio + (it_i - G.neff);
            sblk[j] = blk;
        }
        sycl::group_barrier(grp);

        // scores of this work-item's cell for the 6 heads
        const int ii = j >> 2, cb = j & 3;
        const int item = i0 + ii;
        const bool valid = item < i1;
        bool vis = false;
        int x = 0;
        if (valid) {
            const int blk = sblk[ii];
            x = blk * kRatio + cb;
            if (item < G.neff) vis = true;
            else if (win.tp0) {
                const int root = *win.tp0;
                vis = x < root || (x < root + win.T && ((win.tmask ? win.tmask[win.row0 + t] : 0xFFFFFFFFu) >> (x - root)) & 1u);
            } else vis = x <= row_cell(win, t);
        }
        float sc[kG];
#pragma unroll
        for (int hh = 0; hh < kG; ++hh) sc[hh] = -INFINITY;
        float vsc = 0.f;
        if (vis) {
            const int8_t* kcode = reinterpret_cast<const int8_t*>(a.kc) + ((size_t) kvh * ctx + x) * kHD;
            const uint16_t* ksc = reinterpret_cast<const uint16_t*>(reinterpret_cast<const int8_t*>(a.kc) + (size_t) HKV * ctx * kHD) + ((size_t) kvh * ctx + x) * 4;
            float tot[kG];
#pragma unroll
            for (int hh = 0; hh < kG; ++hh) tot[hh] = 0.f;
#pragma unroll
            for (int gq = 0; gq < 4; ++gq) {
                float part[kG];
#pragma unroll
                for (int hh = 0; hh < kG; ++hh) part[hh] = 0.f;
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const sycl::vec<uint32_t, 4> c4 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(kcode + gq * 64 + v * 16);
#pragma unroll
                    for (int w4 = 0; w4 < 4; ++w4) {
                        const uint32_t u = c4[w4];
                        const int d4 = gq * 16 + v * 4 + w4;   // float4 index of dims 4*d4 .. 4*d4+3
                        const float c0 = (float) (int8_t) (u & 0xFFu), c1 = (float) (int8_t) ((u >> 8) & 0xFFu);
                        const float c2 = (float) (int8_t) ((u >> 16) & 0xFFu), c3 = (float) (int8_t) (u >> 24);
#pragma unroll
                        for (int hh = 0; hh < kG; ++hh) {
                            const sycl::float4 qv = sq[hh * (kHD / 4) + d4];
                            float p = part[hh];
                            p = sycl::fma(qv.x(), c0, p); p = sycl::fma(qv.y(), c1, p); p = sycl::fma(qv.z(), c2, p); p = sycl::fma(qv.w(), c3, p);
                            part[hh] = p;
                        }
                    }
                }
                const float ksg = h2f(ksc[gq]);
#pragma unroll
                for (int hh = 0; hh < kG; ++hh) tot[hh] = sycl::fma(ksg, part[hh], tot[hh]);
            }
#pragma unroll
            for (int hh = 0; hh < kG; ++hh) sc[hh] = tot[hh];
            const size_t vtb = (size_t) (vt / 32) * kHD * 32;
            vsc = reinterpret_cast<const float*>(reinterpret_cast<const int8_t*>(a.vc) + (size_t) HKV * vtb)[(size_t) kvh * vt + x];
        }
        // piece maxima (order does not matter for a max)
        float mx[kG];
#pragma unroll
        for (int hh = 0; hh < kG; ++hh) {
            mx[hh] = sg_max16(sg, sc[hh]);
            if (lane == 0) rmax[sgi * kG + hh] = mx[hh];
        }
        sycl::group_barrier(grp);
        float M[kG];
#pragma unroll
        for (int hh = 0; hh < kG; ++hh) {
            float m = rmax[hh];
            for (int k = 1; k < 16; ++k) m = sycl::fmax(m, rmax[k * kG + hh]);
            M[hh] = m;
        }
        // weights, their sums (fixed order), and the weights times the cell's V scale for the value sums
        float wsum[kG];
#pragma unroll
        for (int hh = 0; hh < kG; ++hh) {
            const float w = vis ? sycl::native::exp(sc[hh] - M[hh]) : 0.f;
            sw[hh * 256 + j] = w * vsc;
            wsum[hh] = sg_sum16(sg, w);
            if (lane == 0) rsum[sgi * kG + hh] = wsum[hh];
        }
        sycl::group_barrier(grp);

        // value sums: work-item d = j owns dimension j; the block's 4 cells are one dword of the transposed V
        float acc[kG];
#pragma unroll
        for (int hh = 0; hh < kG; ++hh) acc[hh] = 0.f;
        const size_t vtb = (size_t) (vt / 32) * kHD * 32;
        const uint8_t* vb = reinterpret_cast<const uint8_t*>(a.vc) + (size_t) kvh * vtb;
        const int nit = i1 - i0;
#pragma unroll 4
        for (int k = 0; k < nit; ++k) {
            const int blk = sblk[k];
            const uint32_t u = *reinterpret_cast<const uint32_t*>(vb + ((size_t) (blk >> 3) * kHD + j) * 32 + (blk & 7) * 4);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float v = (float) (int8_t) ((u >> (8 * c)) & 0xFFu);
#pragma unroll
                for (int hh = 0; hh < kG; ++hh) acc[hh] = sycl::fma(sw[hh * 256 + k * 4 + c], v, acc[hh]);
            }
        }
        // write the partials
#pragma unroll
        for (int hh = 0; hh < kG; ++hh) {
            const size_t row = (size_t) t * H + kvh * kG + hh;
            a.pacc[(row * kPieces + s) * kHD + j] = acc[hh];
            if (j == 0) {
                float l = rsum[hh];
                for (int k = 1; k < 16; ++k) l += rsum[k * kG + hh];
                a.pm[row * kPieces + s] = M[hh];
                a.pl[row * kPieces + s] = l;
            }
        }
        if (s == 0 && kvh == 0 && j == 0) a.nitems[t] = G.n_items;
    });
}

// ---- merge: the pieces of a row's head in order, times sigmoid(gate) -------------------------------------------------------------------
void launch_merge(sycl::queue& q, const AttendArgs a, Win win) {
    const int nr = win.nr, H = a.H;
    q.parallel_for<TagMerge>(sycl::nd_range<1>((size_t) nr * H * kHD, kHD), [=](sycl::nd_item<1> it) {
        const int row = (int) it.get_group(0);
        const int d = (int) it.get_local_id(0);
        const int t = row / H;
        const int np = (a.nitems[t] + kPieceItems - 1) / kPieceItems;
        float M = -INFINITY;
        for (int s = 0; s < np; ++s) M = sycl::fmax(M, a.pm[(size_t) row * kPieces + s]);
        float num = 0.f, den = 0.f;
        for (int s = 0; s < np; ++s) {
            const float w = sycl::native::exp(a.pm[(size_t) row * kPieces + s] - M);
            num = sycl::fma(w, a.pacc[((size_t) row * kPieces + s) * kHD + d], num);
            den = sycl::fma(w, a.pl[(size_t) row * kPieces + s], den);
        }
        const size_t gi = (size_t) row * kHD + d;
        const float gt = a.gate[gi];
        a.out[gi] = (den > 0.f ? num / den : 0.f) * (1.0f / (1.0f + sycl::native::exp(-gt)));
    });
}

// ---- tree commit: the kept rows' raw keys into the ring --------------------------------------------------------------------------
void launch_ring_tree(sycl::queue& q, const float* traw, const int32_t* path, const int32_t* pos0, int n_keep, float* ring) {
    q.parallel_for<TagRingTree>(sycl::nd_range<1>((size_t) n_keep * 128, 128), [=](sycl::nd_item<1> it) {
        const int k = (int) it.get_group(0), i = (int) it.get_local_id(0);
        ring[(size_t) ((*pos0 + k) % kRing) * 128 + i] = traw[(size_t) path[k] * 128 + i];
    });
}

}   // namespace

// ================================================================================================================ public entry points

void index_rows(const LayerWeights& w, const LayerState& lyr, const Scratch& sc, const float* h, const Win& win, const Rope& rp, bool tree, void* stream) {
    sycl::queue& q = Q(stream);
    launch_project(q, static_cast<const half*>(w.wq), static_cast<const half*>(w.wk), h, win.T, sc.proj);
    launch_prep(q, sc.proj, w.qn, w.kpre, win, rp, tree, sc.qidx, sc.raw, lyr.traw);
    if (!tree) {
        launch_pool(q, sc.raw, lyr.ring, w.knorm, win.pos0, win.delta, win.T, rp, static_cast<half*>(lyr.pooled));
        launch_ring_update(q, sc.raw, win.pos0, win.T, lyr.ring);
    }
}

void select_rows(const LayerState& lyr, const Scratch& sc, const Win& win, void* stream) {
    sycl::queue& q = Q(stream);
    const half* pooled = static_cast<const half*>(lyr.pooled);
    const float* qidx = sc.qidx + (size_t) win.row0 * kHeads * kD;
    // scoring launches of at most kScoreRows rows, each writing its rows of sc.score
    for (int r0 = 0; r0 < win.nr; r0 += kScoreRows) {
        Win w = win;
        w.row0 = win.row0 + r0;
        w.nr = std::min(kScoreRows, win.nr - r0);
        const float* qp = qidx + (size_t) r0 * kHeads * kD;
        float* sp = sc.score + (size_t) r0 * sc.nb_cap;
        if (w.nr == 1) launch_score<1, TagScore1>(q, pooled, qp, w, sc.nb_cap, sp);
        else if (w.nr == 2) launch_score<2, TagScore2>(q, pooled, qp, w, sc.nb_cap, sp);
        else if (w.nr <= 4) launch_score<4, TagScore4>(q, pooled, qp, w, sc.nb_cap, sp);
        else launch_score<8, TagScore8>(q, pooled, qp, w, sc.nb_cap, sp);
    }
    launch_select(q, sc.score, win, sc.nb_cap, sc.sel, sc.nsel, sc.dbg);
}

void attend_rows(const AttendArgs& a, const Win& win, void* stream) {
    sycl::queue& q = Q(stream);
    launch_attend(q, a, win);
    launch_merge(q, a, win);
}

void pool_tree(const LayerWeights& w, const LayerState& lyr, const int32_t* p0, const int32_t* path, int n_keep, const Rope& rp, const int32_t* delta, void* stream) {
    sycl::queue& q = Q(stream);
    launch_ring_tree(q, lyr.traw, path, p0, n_keep, lyr.ring);
    launch_pool(q, nullptr, lyr.ring, w.knorm, p0, delta, n_keep, rp, static_cast<half*>(lyr.pooled));
}

}   // namespace strata::q36qsa
