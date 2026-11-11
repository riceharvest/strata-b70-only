// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{quantize.cu,vecdotq.cuh,mmvq.cu,common.cuh}
// and ggml/src/ggml-common.h. See docs/native-mmvq.md for exact scope.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//
// strata-q36 only: the decode matrix-vector forms tuned for the dense Qwen3.8-27B on the B70 (2026-11-02 / 11-03), in a
// translation unit of their own so that the kernel library `strata` links is the one it always was.  What is here is
// copied from sycl/src/kernels/cuda/native_mmvq.dp.cpp (the block types, the KREG structs, KREG2) with the 27B's forms added:
// KREG for Q3_K and IQ4_NL, a persistent grid for every format, two-piece and K-split rows for the 17408-wide ffn_down, other
// tiles at 5-8 columns, and an SLM codebook table for the wide IQ4_NL kernel.  strata-q36 calls q36dense::mmvq first and
// native_mmvq for what it declines.  Nothing here is called by `strata`.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/dp4a.hpp"
#include "q36_mmvq_dense.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>

namespace strata::q36dense {
namespace {

constexpr int QK = 256;
constexpr int Q8K = 32;
constexpr int WARPS = 4;
constexpr int WARP = 32;
struct Q5KBlock {
    sycl::half2 dm;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
struct Q81Block {
    sycl::half2 ds;
    int8_t qs[32];
};
struct Q20Block {
    sycl::half d;
    uint8_t qs[16];
};
struct Q3KBlock {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    sycl::half d;
};
struct IQ4XSBlock {
    sycl::half d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};
struct Q4KBlock {
    sycl::half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    sycl::half d;
};
struct Q40Block {
    sycl::half d;
    uint8_t qs[16];
};
struct Q50Block {
    sycl::half d;
    uint8_t qh[4];
    uint8_t qs[16];
};
struct Q80Block {
    sycl::half d;
    int8_t qs[32];
};
struct IQ4NLBlock {
    sycl::half d;
    uint8_t qs[16];
};
static_assert(sizeof(Q5KBlock) == 176 && alignof(Q5KBlock) == 4);
static_assert(sizeof(Q81Block) == 36 && alignof(Q81Block) == 4);
static_assert(sizeof(Q20Block) == 18 && alignof(Q20Block) == 2 && offsetof(Q20Block, qs) == 2);
static_assert(sizeof(Q3KBlock) == 110 && alignof(Q3KBlock) == 2 && offsetof(Q3KBlock, qs) == 32 &&
              offsetof(Q3KBlock, scales) == 96 && offsetof(Q3KBlock, d) == 108);
static_assert(sizeof(IQ4XSBlock) == 136 && alignof(IQ4XSBlock) == 2 &&
              offsetof(IQ4XSBlock, scales_h) == 2 && offsetof(IQ4XSBlock, scales_l) == 4 &&
              offsetof(IQ4XSBlock, qs) == 8);
static_assert(offsetof(Q5KBlock, scales) == 4 && offsetof(Q5KBlock, qh) == 16 &&
              offsetof(Q5KBlock, qs) == 48 && offsetof(Q81Block, qs) == 4);
static_assert(sizeof(Q4KBlock) == 144 && alignof(Q4KBlock) == 4 &&
              offsetof(Q4KBlock, scales) == 4 && offsetof(Q4KBlock, qs) == 16);
static_assert(sizeof(Q6KBlock) == 210 && alignof(Q6KBlock) == 2 &&
              offsetof(Q6KBlock, qh) == 128 && offsetof(Q6KBlock, scales) == 192 &&
              offsetof(Q6KBlock, d) == 208);
static_assert(sizeof(Q40Block) == 18 && alignof(Q40Block) == 2 && offsetof(Q40Block, qs) == 2);
static_assert(sizeof(Q50Block) == 22 && alignof(Q50Block) == 2 &&
              offsetof(Q50Block, qh) == 2 && offsetof(Q50Block, qs) == 6);
static_assert(sizeof(Q80Block) == 34 && alignof(Q80Block) == 2 && offsetof(Q80Block, qs) == 2);
static_assert(sizeof(IQ4NLBlock) == 18 && alignof(IQ4NLBlock) == 2 && offsetof(IQ4NLBlock, qs) == 2);

__dpct_inline__ sycl::int4 ld_i4_a4(const int* p) {
    sycl::int4 v;
    v.load(0, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p));
    return v;
}
__dpct_inline__ sycl::int4 load16_a2(const void* p) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const sycl::int4* q = reinterpret_cast<const sycl::int4*>(a & ~uintptr_t(15));
    const int k = int(a >> 2) & 3;          // dword offset within the first aligned chunk
    const bool half = (a & 2) != 0;          // and a 2-byte shift (the address is always 2-byte aligned)
    // the second chunk only when the address is unaligned: then it holds needed bytes, so (16-byte chunks never
    // straddle a page) it cannot fault even at the very end of an allocation
    const sycl::int4 lo = q[0], hi = (a & 15) ? q[1] : lo;
    const uint32_t d[8] = {(uint32_t) lo.x(), (uint32_t) lo.y(), (uint32_t) lo.z(), (uint32_t) lo.w(),
                           (uint32_t) hi.x(), (uint32_t) hi.y(), (uint32_t) hi.z(), (uint32_t) hi.w()};
    uint32_t w[5];
#pragma unroll
    for (int i = 0; i < 5; ++i)              // w = d[k .. k+4] by selects (no indexed register access)
        w[i] = k == 0 ? d[i] : k == 1 ? d[i + 1] : k == 2 ? d[i + 2] : d[(i + 3) & 7];
    uint32_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = half ? (w[i] >> 16) | (w[i + 1] << 16) : w[i];
    return sycl::int4((int) o[0], (int) o[1], (int) o[2], (int) o[3]);
}
__dpct_inline__ sycl::int4 ld_q8_16(const Q81Block* b, int half) {   // Q8_1 qs is 4-byte aligned: one dword-vector load
    return ld_i4_a4(reinterpret_cast<const int*>(b->qs) + 4 * half);
}
__dpct_inline__ int dp4a4(const sycl::int4 a, const sycl::int4 b, int acc) {
    acc = strata::dp4a(a.x(), b.x(), acc); acc = strata::dp4a(a.y(), b.y(), acc);
    acc = strata::dp4a(a.z(), b.z(), acc); return strata::dp4a(a.w(), b.w(), acc);
}

static constexpr int8_t kIq4nlTable[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

constexpr int KREG_SG = 16;
// STRATA_MMVQ_KREG=1 (strata-q36 sets it for the dense model before the first call): the KREG kernels on
inline bool kreg_on() {
    static const bool v = [] { const char* e = std::getenv("STRATA_MMVQ_KREG"); return e && e[0] == '1'; }();
    return v;
}
inline int kreg_groups_cap(int ncols, int nsg, int grf, dpct::queue_ptr s) {
    static const int cu = int(s->get_device().get_info<sycl::info::device::max_compute_units>());
    static const int cores = std::max(1, cu >= 64 ? cu / 8 : cu);   // Intel reports XVEs (8 a core)
    static const std::pair<int, int> w = [] {
        int a = -1, b = -1;
        if (const char* e = std::getenv("STRATA_KREG_WAVES")) std::sscanf(e, "%d,%d", &a, &b);
        return std::make_pair(a, b);
    }();
    const int per_core = (grf == 256 ? 32 : 64) / nsg;
    int waves = ncols == 1 ? w.first : w.second;
    if (waves < 0) waves = ncols >= 3 && per_core >= 2 ? 1 : 0;
    if (waves <= 0) return 1 << 30;
    return std::max(1, waves * cores * std::max(1, per_core));
}

template <int O, int N, int P>
__dpct_inline__ void kreg_tsum(float (&v)[P], const sycl::sub_group& sg, int lane) {
    if constexpr (O > 0) {
        constexpr int H = N / 2;
        const bool up = (lane & O) != 0;
#pragma unroll
        for (int i = 0; i < H; ++i) {
            const float lo = v[i], hi = v[i + H];   // values: a ?: of two array lvalues selects ADDRESSES and
            const float send = up ? lo : hi;        // sends v[] to private (global) memory
            const float keep = up ? hi : lo;
            v[i] = keep + sycl::permute_group_by_xor(sg, send, O);
        }
        kreg_tsum<O / 2, H, P>(v, sg, lane);
    }
}
struct KregActK { sycl::int4 ua, uc; float da, dc, sa, sc; };
__dpct_inline__ KregActK kreg_act_k(const Q81Block* xb, int g) {
    const int sb = 2 * (g >> 1), half = g & 1;
    const sycl::int4 ones(0x01010101, 0x01010101, 0x01010101, 0x01010101);
    KregActK v;
    v.ua = ld_q8_16(xb + sb, half); v.uc = ld_q8_16(xb + sb + 1, half);
    v.da = (float) xb[sb].ds[0]; v.dc = (float) xb[sb + 1].ds[0];
    v.sa = v.da * (float) dp4a4(ones, v.ua, 0);
    v.sc = v.dc * (float) dp4a4(ones, v.uc, 0);
    return v;
}
struct KregWK { sycl::int4 lo, hi; float dl, dh, ml, mh; };
// dm + the 12 scale bytes = the block's first 16 bytes; ggml get_scale_min_k4 for j = sb, sb + 1, branchless
__dpct_inline__ void kreg_scales_k(const sycl::int4 hdr, int g, KregWK& r) {
    const int sb = 2 * (g >> 1), sh = 8 * (sb & 3);
    const uint32_t A = ((uint32_t) hdr.y() >> sh) & 0xFFFF, B = ((uint32_t) hdr.z() >> sh) & 0xFFFF,
                   C = ((uint32_t) hdr.w() >> sh) & 0xFFFF;
    const uint32_t ls0 = A & 63, ls1 = (A >> 8) & 63, lm0 = B & 63, lm1 = (B >> 8) & 63;
    const uint32_t hs0 = (C & 0xF) | (((A >> 6) & 3) << 4), hs1 = ((C >> 8) & 0xF) | (((A >> 14) & 3) << 4);
    const uint32_t hm0 = ((C >> 4) & 0xF) | (((B >> 6) & 3) << 4), hm1 = ((C >> 12) & 0xF) | (((B >> 14) & 3) << 4);
    const bool lo4 = sb < 4;
    const uint32_t s0 = lo4 ? ls0 : hs0, s1 = lo4 ? ls1 : hs1, m0 = lo4 ? lm0 : hm0, m1 = lo4 ? lm1 : hm1;
    const sycl::half2 dmh = sycl::bit_cast<sycl::half2>(hdr.x());
    const float d = (float) dmh.x(), dmin = (float) dmh.y();
    r.dl = d * (float) s0; r.ml = dmin * (float) m0; r.dh = d * (float) s1; r.mh = dmin * (float) m1;
}
__dpct_inline__ float kreg_dot_k(const KregWK& r, const KregActK& v) {
    return r.dl * (v.da * (float) dp4a4(r.lo, v.ua, 0)) - r.ml * v.sa +
           r.dh * (v.dc * (float) dp4a4(r.hi, v.uc, 0)) - r.mh * v.sc;
}
struct KregQ4K {
    static constexpr int kTableWords = 0;
    static uint32_t table_entry(int) { return 0; }
    using Block = Q4KBlock; using A = KregActK; using W = KregWK;
    static A act(const Q81Block* xb, int g) { return kreg_act_k(xb, g); }
    static W load(const Block* b, int g) {
        W r;
        const sycl::int4 hdr = *reinterpret_cast<const sycl::int4*>(b);
        const sycl::int4 q = reinterpret_cast<const sycl::int4*>(b->qs)[g];
        r.lo = sycl::int4(q.x() & 0x0f0f0f0f, q.y() & 0x0f0f0f0f, q.z() & 0x0f0f0f0f, q.w() & 0x0f0f0f0f);
        r.hi = sycl::int4((q.x() >> 4) & 0x0f0f0f0f, (q.y() >> 4) & 0x0f0f0f0f, (q.z() >> 4) & 0x0f0f0f0f, (q.w() >> 4) & 0x0f0f0f0f);
        kreg_scales_k(hdr, g, r);
        return r;
    }
    static float dot(const W& r, const A& v) { return kreg_dot_k(r, v); }
};
struct KregQ5K {
    static constexpr int kTableWords = 0;
    static uint32_t table_entry(int) { return 0; }
    using Block = Q5KBlock; using A = KregActK; using W = KregWK;
    static A act(const Q81Block* xb, int g) { return kreg_act_k(xb, g); }
    static W load(const Block* b, int g) {
        W r;
        const int sb = 2 * (g >> 1), half = g & 1;
        const sycl::int4 hdr = *reinterpret_cast<const sycl::int4*>(b);
        const sycl::int4 q = reinterpret_cast<const sycl::int4*>(b->qs)[g];
        const sycl::int4 h = reinterpret_cast<const sycl::int4*>(b->qh)[half];
        auto hb = [](int v, int bit) { return ((v >> bit) & 0x01010101) << 4; };
        r.lo = sycl::int4((q.x() & 0x0f0f0f0f) | hb(h.x(), sb), (q.y() & 0x0f0f0f0f) | hb(h.y(), sb),
                          (q.z() & 0x0f0f0f0f) | hb(h.z(), sb), (q.w() & 0x0f0f0f0f) | hb(h.w(), sb));
        r.hi = sycl::int4(((q.x() >> 4) & 0x0f0f0f0f) | hb(h.x(), sb + 1), ((q.y() >> 4) & 0x0f0f0f0f) | hb(h.y(), sb + 1),
                          ((q.z() >> 4) & 0x0f0f0f0f) | hb(h.z(), sb + 1), ((q.w() >> 4) & 0x0f0f0f0f) | hb(h.w(), sb + 1));
        kreg_scales_k(hdr, g, r);
        return r;
    }
    static float dot(const W& r, const A& v) { return kreg_dot_k(r, v); }
};
struct KregQ6K {   // the wide Q6_K kernel's lane mapping and dot expression; (q - 32) per byte by (q + 0x60) ^ 0x80
    static constexpr int kTableWords = 0;
    static uint32_t table_entry(int) { return 0; }
    using Block = Q6KBlock;
    struct A { sycl::int4 u0, u1; float ds0, ds1; };
    struct W { int vi0[4], vi1[4]; float dsc0, dsc1; };
    static A act(const Q81Block* xb, int g) {
        const int bq8_offset = 4 * (g >> 2) + ((g & 3) >> 1);
        A a;
        a.u0 = ld_q8_16(xb + bq8_offset, g & 1); a.u1 = ld_q8_16(xb + bq8_offset + 2, g & 1);
        a.ds0 = (float) xb[bq8_offset].ds[0]; a.ds1 = (float) xb[bq8_offset + 2].ds[0];
        return a;
    }
    static W load(const Block* b, int g) {
        W r;
        const int vh_shift = 2 * ((g & 3) >> 1);
        const int qh4_idx = 2 * (g >> 2) + (g & 1);
        const int scale_offset = 8 * (g >> 2) + (g & 3);
        const sycl::int4 ql4 = load16_a2(b->ql + 16 * g), qh4 = load16_a2(b->qh + 16 * qh4_idx);
        const float d = b->d;
        r.dsc0 = d * (float) b->scales[scale_offset]; r.dsc1 = d * (float) b->scales[scale_offset + 4];
        const int vl[4] = {ql4.x(), ql4.y(), ql4.z(), ql4.w()};
        const int vh[4] = {qh4.x() >> vh_shift, qh4.y() >> vh_shift, qh4.z() >> vh_shift, qh4.w() >> vh_shift};
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            r.vi0[p] = (int) ((uint32_t((vl[p] & 0x0f0f0f0f) | ((vh[p] << 4) & 0x30303030)) + 0x60606060u) ^ 0x80808080u);
            r.vi1[p] = (int) ((uint32_t(((vl[p] >> 4) & 0x0f0f0f0f) | (((vh[p] >> 4) << 4) & 0x30303030)) + 0x60606060u) ^ 0x80808080u);
        }
        return r;
    }
    static float dot(const W& r, const A& a) {
        int s0 = 0, s1 = 0;
        s0 = strata::dp4a(r.vi0[0], a.u0.x(), s0); s0 = strata::dp4a(r.vi0[1], a.u0.y(), s0);
        s0 = strata::dp4a(r.vi0[2], a.u0.z(), s0); s0 = strata::dp4a(r.vi0[3], a.u0.w(), s0);
        s1 = strata::dp4a(r.vi1[0], a.u1.x(), s1); s1 = strata::dp4a(r.vi1[1], a.u1.y(), s1);
        s1 = strata::dp4a(r.vi1[2], a.u1.z(), s1); s1 = strata::dp4a(r.vi1[3], a.u1.w(), s1);
        return a.ds0 * r.dsc0 * (float) s0 + a.ds1 * r.dsc1 * (float) s1;
    }
};
// IQ4_XS (lane g = sub-block g, as WideIQ4XS): the codebook through a 256-entry SLM table indexed by a qs byte,
// TL[b] = code(b & 15) | code(b >> 4) << 16, so two lookups and two shifts give four low and four high codes.
// The select-chain lookup (iq4_lut4) costs ~10 instructions per nibble and made a K-split kernel slower than WideIQ4XS.
struct KregIQ4XS {
    using Block = IQ4XSBlock;
    static constexpr int kTableWords = 256;
    static uint32_t table_entry(int i) {
        return (uint32_t) (uint8_t) kIq4nlTable[i & 15] | ((uint32_t) (uint8_t) kIq4nlTable[i >> 4] << 16);
    }
    struct A { sycl::int4 u0, u1; float ds; };
    struct W { sycl::int4 lo, hi; float d; };
    static A act(const Q81Block* xb, int g) {
        A a; a.u0 = ld_q8_16(xb + g, 0); a.u1 = ld_q8_16(xb + g, 1); a.ds = (float) xb[g].ds[0]; return a;
    }
    static W load(const Block* b, int g, const uint32_t* tl) {
        W r;
        const sycl::int2 hdr = *reinterpret_cast<const sycl::int2*>(b);              // 136 = 17 x 8: 8-byte aligned
        const sycl::int2* q2 = reinterpret_cast<const sycl::int2*>(b->qs + 16 * g);
        const sycl::int2 qa = q2[0], qb = q2[1];
        const uint32_t v[4] = {(uint32_t) qa.x(), (uint32_t) qa.y(), (uint32_t) qb.x(), (uint32_t) qb.y()};
        int lo[4], hi[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t P = tl[v[i] & 0xFF] | (tl[(v[i] >> 8) & 0xFF] << 8);
            const uint32_t Q = tl[(v[i] >> 16) & 0xFF] | (tl[v[i] >> 24] << 8);
            lo[i] = (int) ((P & 0xFFFFu) | (Q << 16));
            hi[i] = (int) ((P >> 16) | (Q & 0xFFFF0000u));
        }
        r.lo = sycl::int4(lo[0], lo[1], lo[2], lo[3]); r.hi = sycl::int4(hi[0], hi[1], hi[2], hi[3]);
        const uint32_t h0 = (uint32_t) hdr.x(), sl = (uint32_t) hdr.y();
        const int ls = (int) (((sl >> (8 * (g >> 1) + 4 * (g & 1))) & 0x0f) | (((h0 >> (16 + 2 * g)) & 0x03) << 4));
        r.d = (float) sycl::bit_cast<sycl::half>((uint16_t) (h0 & 0xFFFF)) * (float) (ls - 32);
        return r;
    }
    static float dot(const W& r, const A& v) { return r.d * v.ds * (float) dp4a4(r.hi, v.u1, dp4a4(r.lo, v.u0, 0)); }
};
// IQ4_NL (M2): eight 18-byte blocks as one 256-weight group (144 bytes; a row is n_in / 32 blocks, so a group starts every
// 144 bytes), lane g = block g: KregIQ4XS's table decode and dot with the block's own fp16 scale.  2-byte aligned qs loads.
struct IQ4NLGroup { IQ4NLBlock b[8]; };
static_assert(sizeof(IQ4NLGroup) == 144, "IQ4_NL group");
struct KregIQ4NL {
    using Block = IQ4NLGroup;
    static constexpr int kTableWords = 256;
    static uint32_t table_entry(int i) { return KregIQ4XS::table_entry(i); }
    using A = KregIQ4XS::A;
    using W = KregIQ4XS::W;
    static A act(const Q81Block* xb, int g) { return KregIQ4XS::act(xb, g); }
    static W load(const Block* b, int g, const uint32_t* tl) {
        W r;
        const IQ4NLBlock& q = b->b[g];
        const sycl::int4 qv = load16_a2(q.qs);
        const uint32_t v[4] = {(uint32_t) qv.x(), (uint32_t) qv.y(), (uint32_t) qv.z(), (uint32_t) qv.w()};
        int lo[4], hi[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t P = tl[v[i] & 0xFF] | (tl[(v[i] >> 8) & 0xFF] << 8);
            const uint32_t Q = tl[(v[i] >> 16) & 0xFF] | (tl[v[i] >> 24] << 8);
            lo[i] = (int) ((P & 0xFFFFu) | (Q << 16));
            hi[i] = (int) ((P >> 16) | (Q & 0xFFFF0000u));
        }
        r.lo = sycl::int4(lo[0], lo[1], lo[2], lo[3]); r.hi = sycl::int4(hi[0], hi[1], hi[2], hi[3]);
        r.d = (float) q.d;
        return r;
    }
    static float dot(const W& r, const A& v) { return KregIQ4XS::dot(r, v); }
};
// Q3_K (M2): lane g = the 256-block's 32 weights 32 g .. 32 g + 31 (Q8_1 block g): qs half n = g / 4 at shift 2 (g % 4),
// hmask bit g, scales 2 g and 2 g + 1 (16 weights each).  q = (q2 | h << 2) - 4: the "- 4" as 4 * sum(x) per half block
// on the activation side (once per column), so a column's dot is integer up to one conversion:
// d * dx * float(s0 * (dp0 - 4 sum0) + s1 * (dp1 - 4 sum1)).  110-byte blocks: 2-byte aligned loads.
struct KregQ3K {
    static constexpr int kTableWords = 0;
    static uint32_t table_entry(int) { return 0; }
    using Block = Q3KBlock;
    struct A { sycl::int4 u0, u1; int c0, c1; float ds; };
    struct W { sycl::int4 lo, hi; int s0, s1; float d; };
    static A act(const Q81Block* xb, int g) {
        const sycl::int4 ones(0x01010101, 0x01010101, 0x01010101, 0x01010101);
        A a;
        a.u0 = ld_q8_16(xb + g, 0); a.u1 = ld_q8_16(xb + g, 1); a.ds = (float) xb[g].ds[0];
        a.c0 = 4 * dp4a4(ones, a.u0, 0); a.c1 = 4 * dp4a4(ones, a.u1, 0);
        return a;
    }
    static W load(const Block* b, int g) {
        W r;
        const int n = g >> 2, sh = 2 * (g & 3);
        const uint8_t* bb = reinterpret_cast<const uint8_t*>(b);
        const sycl::int4 h0 = load16_a2(bb), h1 = load16_a2(bb + 16);
        const sycl::int4 q0 = load16_a2(bb + 32 + 32 * n), q1 = load16_a2(bb + 48 + 32 * n);
        const sycl::int4 sc = load16_a2(bb + 96);   // scales[12], then d
        auto v = [&](int q, int h) { return (((uint32_t) q >> sh) & 0x03030303u) | ((((uint32_t) h >> g) & 0x01010101u) << 2); };
        r.lo = sycl::int4((int) v(q0.x(), h0.x()), (int) v(q0.y(), h0.y()), (int) v(q0.z(), h0.z()), (int) v(q0.w(), h0.w()));
        r.hi = sycl::int4((int) v(q1.x(), h1.x()), (int) v(q1.y(), h1.y()), (int) v(q1.z(), h1.z()), (int) v(q1.w(), h1.w()));
        const uint32_t sw[3] = {(uint32_t) sc.x(), (uint32_t) sc.y(), (uint32_t) sc.z()};
        auto byte = [&](int i) { return (sw[i >> 2] >> (8 * (i & 3))) & 0xFFu; };
        // scale j: low 4 bits sc[j] & 15 (j < 8) or sc[j - 8] >> 4, high 2 bits (sc[8 + j % 4] >> 2 (j / 4)) & 3; minus 32
        const int j0 = 2 * g, j1 = 2 * g + 1;
        const uint32_t l0 = g < 4 ? byte(j0) & 15u : byte(j0 - 8) >> 4, l1 = g < 4 ? byte(j1) & 15u : byte(j1 - 8) >> 4;
        const uint32_t hh0 = (byte(8 + (j0 & 3)) >> (2 * (j0 >> 2))) & 3u, hh1 = (byte(8 + (j1 & 3)) >> (2 * (j1 >> 2))) & 3u;
        r.s0 = (int) (l0 | hh0 << 4) - 32; r.s1 = (int) (l1 | hh1 << 4) - 32;
        r.d = (float) sycl::bit_cast<sycl::half>((uint16_t) ((uint32_t) sc.w() & 0xFFFFu));
        return r;
    }
    static float dot(const W& r, const A& a) {
        const int i0 = dp4a4(r.lo, a.u0, 0) - a.c0, i1 = dp4a4(r.hi, a.u1, 0) - a.c1;
        return r.d * a.ds * (float) (r.s0 * i0 + r.s1 * i1);
    }
};
template <int R, int NCOLS> constexpr int kreg_p() { return (R * NCOLS + KREG_SG - 1) / KREG_SG * KREG_SG; }
// ===== the 27B's KREG forms (strata-q36 only).  The forms are the 27B tree's (sweeps on the B70, 2026-11-02 / 11-03): the two-piece (H = 2) and K-split rows for a row wider than
// 16384, a persistent grid for every format, other tile shapes at 5-8 columns, and KREG for Q3_K and IQ4_NL.
// H = 2 (M2, rows wider than 16384; STRATA_KREG_H2=0: off): a lane owns two 32-weight pieces of a row, block kb and kb + bpr / 2,
// so a 17408-wide row takes 272 lanes (17 sub-groups: three groups a core with 128 registers, where 544 lanes left one group and
// 30 idle threads a core).  The lane adds its two dots (first half, then second) before the butterfly: another summation order
// than H = 1, the same for every column count (KREG2 then stays off for those rows).  The activation pieces of both halves are
// in registers at 1-2 columns, reloaded per tile at 3-4.
template <typename F, int NCOLS, int R, int H = 1, bool AR = (H == 1 || NCOLS <= 2)>
void native_mmvq_kreg_dense_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                             float* __restrict__ y, int n_in, int n_out, float* __restrict__ slm) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const sycl::sub_group sg = it.get_sub_group();
    constexpr int P0 = R * NCOLS, P = kreg_p<R, NCOLS>(), PL = P / KREG_SG;
    constexpr bool AREG = AR;   // the activation pieces stay in registers across tiles
    const int lid = int(it.get_local_id(0)), lane = int(sg.get_local_linear_id()), sgi = lid / KREG_SG;
    const int nsg = int(it.get_local_range(0)) / KREG_SG;
    const int bpr = n_in / QK, x_stride = n_in / Q8K, bph = bpr / H;
    const bool live = (lid >> 3) < bph;
    const int kb = live ? (lid >> 3) : 0, g = lid & 7;
    typename F::A a[AREG ? H : 1][NCOLS];
    auto load_act = [&](int hh, typename F::A (&dst)[NCOLS]) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            dst[j] = F::act(x + std::size_t(j) * x_stride + (kb + hh * bph) * (QK / Q8K), g);
            if (!live) dst[j] = typename F::A{};
        }
    };
    if constexpr (AREG) {
#pragma unroll
        for (int hh = 0; hh < H; ++hh) load_act(hh, a[hh]);
    }
    const uint32_t* tbl = nullptr;
    if constexpr (F::kTableWords > 0) {   // per-group codebook table after the two reduction buffers
        uint32_t* tw = reinterpret_cast<uint32_t*>(slm + 2 * nsg * P);
        for (int i = lid; i < F::kTableWords; i += int(it.get_local_range(0))) tw[i] = F::table_entry(i);
        it.barrier(sycl::access::fence_space::local_space);
        tbl = tw;
    }
    const int ntiles = (n_out + R - 1) / R;
    int buf = 0;
    for (int t = int(it.get_group(0)); t < ntiles; t += int(it.get_group_range(0))) {
        const int row0 = t * R;
        const int rlast = n_out - 1 - row0;
        float v[P];
        // one half's weights and dots at a time (AREG: unrolled, the halves' activations indexed by a constant; else a rolled
        // loop, so the compiler does not hoist the second half's loads next to the first's: 3-4 columns spilled that way)
        auto half = [&](const int hh, const typename F::A* av) {
            const typename F::Block* wt = w + std::size_t(row0) * bpr + kb + hh * bph;
            typename F::W wv[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if constexpr (F::kTableWords > 0) wv[r] = F::load(wt + (r < rlast ? r : rlast) * bpr, g, tbl);
                else wv[r] = F::load(wt + (r < rlast ? r : rlast) * bpr, g);
            }
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) {
                    const float d = F::dot(wv[r], av[j]);
                    if (hh == 0) v[r * NCOLS + j] = d;
                    else v[r * NCOLS + j] += d;
                }
        };
        if constexpr (AREG) {
#pragma unroll
            for (int hh = 0; hh < H; ++hh) half(hh, a[hh]);
        } else {
#pragma nounroll
            for (int hh = 0; hh < H; ++hh) {
                typename F::A ar[NCOLS];
                load_act(hh, ar);
                half(hh, ar);
            }
        }
#pragma unroll
        for (int k = P0; k < P; ++k) v[k] = 0.0f;
        kreg_tsum<KREG_SG / 2, P, P>(v, sg, lane);
        float* b = slm + buf * nsg * P;
#pragma unroll
        for (int i = 0; i < PL; ++i) b[sgi * P + lane * PL + i] = v[i];
        it.barrier(sycl::access::fence_space::local_space);   // SLM-only fence (group_barrier also fences global)
        for (int k = lid; k < P0; k += int(it.get_local_range(0))) {
            float s = 0.0f;
            for (int q = 0; q < nsg; ++q) s += b[q * P + k];
            const int r = k / NCOLS, j = k % NCOLS;
            if (row0 + r < n_out) y[std::size_t(j) * n_out + row0 + r] = s;
        }
        buf ^= 1;   // the next tile writes the other buffer; this one is rewritten only after the next barrier
    }
}
template <typename F, int NCOLS, int R, int GRF, int H = 1>
void launch_kreg_dense(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, dpct::queue_ptr s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const int L = ((n_in / QK / H) * 8 + KREG_SG - 1) / KREG_SG * KREG_SG;
    const int nsg = L / KREG_SG;
    constexpr int P = kreg_p<R, NCOLS>();
    const int tiles = (n_out + R - 1) / R;
    const int groups = std::min(tiles, kreg_groups_cap(NCOLS, nsg, GRF, s));   // (a group loops over tiles: the same sums)
    s->submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>(2 * nsg * P + F::kTableWords), h);
        auto k = [=](sycl::nd_item<1>) [[sycl::reqd_sub_group_size(16)]] {
            native_mmvq_kreg_dense_kernel<F, NCOLS, R, H, (H == 1 || NCOLS <= 2 || GRF == 256)>(
                w, x, y, n_in, n_out, slm.get_multi_ptr<sycl::access::decorated::no>().get());
        };
        const sycl::nd_range<1> nr(std::size_t(groups) * L, L);
        if constexpr (GRF == 256)
            h.parallel_for<dpct_kernel_name<class native_mmvq_kreg_dense256, F, dpct_kernel_scalar<NCOLS>, dpct_kernel_scalar<R>,
                                            dpct_kernel_scalar<H>>>(
                nr, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, k);
        else
            h.parallel_for<dpct_kernel_name<class native_mmvq_kreg_dense, F, dpct_kernel_scalar<NCOLS>, dpct_kernel_scalar<R>,
                                            dpct_kernel_scalar<H>>>(nr, k);
    });
}
// KS (M2, the default for rows wider than 16384 with an even block count; STRATA_KREG_KS=0: off): the row's K split over
// two work-groups (bench, 17408 x 5120 on the B70: 4 columns Q5_K 183 -> 130 us, Q4_K 152 -> 102, IQ4_XS 157 -> 122,
// IQ4_NL 155 -> 119, Q6_K 155 -> 144; 1 column Q5_K 139 -> 128, Q4_K 114 -> 94). Group g
// takes part g % ks (ks parts of bpr / ks blocks each) of tile g / ks, ...; each part's sum is the one-piece arithmetic over
// its blocks (butterfly, then sub-group order) into a scratch plane, and a second launch adds the planes in part order
// (((p0 + p1) + p2) + p3).  Another summation order than one piece, the same at every column count.
// M4 (STRATA_KREG_KS=4, opt-in): ks = 4 where the block count allows it (the 27B's 17408-wide ffn_down: 68 blocks, 136 live
// lanes of a 144-lane group, 7 groups a core with 128 registers where the 272-lane halves fit 3), so 5-8 columns run in ONE
// launch pair (R 4 / 256 registers for the K-quants: 8 columns of their activations are 96 registers a lane), every weight
// read once per window where the halves run 5-8 columns as 4 columns and the rest.  Measured on the B70 (2026-11-03, 27B,
// MTP 5): the 256-register 5-6 column launches ran 228 us a call against ~126 us for 1-4 columns, little better than the
// split, and the 4-part sums moved two near-tie top-1 picks of the KL gate (answers 0.9883 -> 0.9824, KL unchanged): the
// default stays the M2 halves.
template <typename F, int NCOLS, int R>
void native_mmvq_kreg_ks_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                                float* __restrict__ part, int n_in, int n_out, int ks, float* __restrict__ slm) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const sycl::sub_group sg = it.get_sub_group();
    constexpr int P0 = R * NCOLS, P = kreg_p<R, NCOLS>(), PL = P / KREG_SG;
    const int lid = int(it.get_local_id(0)), lane = int(sg.get_local_linear_id()), sgi = lid / KREG_SG;
    const int nsg = int(it.get_local_range(0)) / KREG_SG;
    const int bpr = n_in / QK, x_stride = n_in / Q8K, bph = bpr / ks;
    const int hh = int(it.get_group(0)) % ks;
    const bool live = (lid >> 3) < bph;
    const int kb = (live ? (lid >> 3) : 0) + hh * bph, g = lid & 7;
    float* out = part + std::size_t(hh) * NCOLS * n_out;
    typename F::A a[NCOLS];
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        a[j] = F::act(x + std::size_t(j) * x_stride + kb * (QK / Q8K), g);
        if (!live) a[j] = typename F::A{};
    }
    const uint32_t* tbl = nullptr;
    if constexpr (F::kTableWords > 0) {
        uint32_t* tw = reinterpret_cast<uint32_t*>(slm + 2 * nsg * P);
        for (int i = lid; i < F::kTableWords; i += int(it.get_local_range(0))) tw[i] = F::table_entry(i);
        it.barrier(sycl::access::fence_space::local_space);
        tbl = tw;
    }
    const int ntiles = (n_out + R - 1) / R;
    int buf = 0;
    for (int t = int(it.get_group(0)) / ks; t < ntiles; t += int(it.get_group_range(0)) / ks) {
        const int row0 = t * R;
        const typename F::Block* wt = w + std::size_t(row0) * bpr + kb;
        const int rlast = n_out - 1 - row0;
        typename F::W wv[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if constexpr (F::kTableWords > 0) wv[r] = F::load(wt + (r < rlast ? r : rlast) * bpr, g, tbl);
            else wv[r] = F::load(wt + (r < rlast ? r : rlast) * bpr, g);
        }
        float v[P];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) v[r * NCOLS + j] = F::dot(wv[r], a[j]);
#pragma unroll
        for (int k = P0; k < P; ++k) v[k] = 0.0f;
        kreg_tsum<KREG_SG / 2, P, P>(v, sg, lane);
        float* b = slm + buf * nsg * P;
#pragma unroll
        for (int i = 0; i < PL; ++i) b[sgi * P + lane * PL + i] = v[i];
        it.barrier(sycl::access::fence_space::local_space);
        for (int k = lid; k < P0; k += int(it.get_local_range(0))) {
            float s = 0.0f;
            for (int q = 0; q < nsg; ++q) s += b[q * P + k];
            const int r = k / NCOLS, j = k % NCOLS;
            if (row0 + r < n_out) out[std::size_t(j) * n_out + row0 + r] = s;
        }
        buf ^= 1;
    }
}
constexpr int kKregKsMax = 4;   // the most parts a row's K is split into
inline float* kreg_ks_scratch(dpct::queue_ptr s) {   // 4 planes x 8 columns x 65536 rows, made once (before any graph uses it)
    static float* p = sycl::malloc_device<float>(std::size_t(kKregKsMax) * 8 * 65536, *s);
    return p;
}
/// The K split of a row (0: none): rows wider than 16384 (the default; STRATA_KREG_KS=0: H = 2 / one piece) in 2 parts;
/// STRATA_KREG_KS=4: in 4 where the row's block count allows it.  It depends on the shape alone, never on the column count,
/// so a T-column call stays bitwise T one-column calls.
inline int kreg_ks_n(int n_in, int n_out) {
    static const int e = [] { const char* v = std::getenv("STRATA_KREG_KS"); return v && v[0] ? std::atoi(v) : -1; }();
    if (e == 0 || n_in <= 16384 || n_out > 65536 || n_in % QK != 0) return 0;
    const int bpr = n_in / QK;
    if (e == 4 && bpr % 4 == 0) return 4;
    return bpr % 2 == 0 ? 2 : 0;
}
inline bool kreg_ks(int n_in, int n_out) { return kreg_ks_n(n_in, n_out) > 0; }
template <typename F, int NCOLS, int R, int GRF>
void launch_kreg_ks(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, dpct::queue_ptr s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    float* part = kreg_ks_scratch(s);
    const int ks = kreg_ks_n(n_in, n_out);   // 2 or 4 (the caller checked kreg_ks)
    const int L = ((n_in / QK / ks) * 8 + KREG_SG - 1) / KREG_SG * KREG_SG;
    const int nsg = L / KREG_SG;
    constexpr int P = kreg_p<R, NCOLS>();
    const int tiles = (n_out + R - 1) / R;
    const int groups = ks * std::min(tiles, std::max(1, kreg_groups_cap(NCOLS, nsg, GRF, s) / ks));
    s->submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>(2 * nsg * P + F::kTableWords), h);
        auto k = [=](sycl::nd_item<1>) [[sycl::reqd_sub_group_size(16)]] {
            native_mmvq_kreg_ks_kernel<F, NCOLS, R>(w, x, part, n_in, n_out, ks,
                                                     slm.get_multi_ptr<sycl::access::decorated::no>().get());
        };
        const sycl::nd_range<1> nr(std::size_t(groups) * L, L);
        if constexpr (GRF == 256)
            h.parallel_for<dpct_kernel_name<class native_mmvq_kreg_ks256, F, dpct_kernel_scalar<NCOLS>, dpct_kernel_scalar<R>>>(
                nr, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, k);
        else
            h.parallel_for<dpct_kernel_name<class native_mmvq_kreg_ks, F, dpct_kernel_scalar<NCOLS>, dpct_kernel_scalar<R>>>(nr, k);
    });
    const std::size_t n = std::size_t(NCOLS) * n_out;
    s->parallel_for<dpct_kernel_name<class native_mmvq_kreg_ks_add, F, dpct_kernel_scalar<NCOLS>>>(
        sycl::range<1>(n), [=](sycl::id<1> i) {
            float v = part[i] + part[n + i];   // the planes in part order (ks 2: the M2 sum, bit for bit)
            for (int q = 2; q < ks; ++q) v += part[std::size_t(q) * n + i];
            y[i] = v;
        });
}
// Per-(format, columns) tile shape: R rows per tile, GRF mode. R and the GRF mode change no sum (a column's sum depends on
// neither: see above), so any choice is bitwise the same output.
//   - the 2026-10-16 sweep on an Arc Pro B70 (Qwen3.8-Flash-Next shapes, n_in 2560 / 6144): kreg_default_rg;
//   - a 256-register form allows 512 lanes a work-group: a wider row (17408: 544 lanes) takes the 128-register form of
//     the same R (R 8 -> 4 there); try_kreg runs such rows at 5-8 columns as 4 columns and the rest, and with H = 2
//     (272 lanes, the default for them) kreg_dispatch picks its own tiles;
//   - STRATA_KREG_RG="R,G" (bench knob, read once): every call at 2-8 columns takes R in {2,4,8}, G in {128,256} where
//     that fits the lane rule above.
struct KregRG { int r, g; };
// M4: 7-8 columns (an MTP window of 6-7 drafts) at R 4 / 256 registers for the Q4_K-like formats (R 8 there holds 64 sums,
// 96 weight and 84-96 activation registers a lane: past 256) and Q5_K (R 2 / 128: 84-96 activation registers alone); IQ4_XS
// at R 2 (R 4 / 128: 63-72 + 36 + 32).  5-6 columns as M2 measured them (2026-11-02, waves 1: Q5_K 5120 x 17408 at 5
// columns 147 us at R 2 / 128, 146 at R 8 / 256).
template <typename F, int NCOLS>
KregRG kreg_default_rg(int n_in) {
    if constexpr (std::is_same_v<F, KregIQ4XS>) return NCOLS == 1 ? KregRG{8, 128} : NCOLS <= 6 ? KregRG{4, 128} : KregRG{2, 128};
    else if constexpr (std::is_same_v<F, KregQ6K>) return NCOLS <= 3 ? KregRG{4, 128} : KregRG{2, 128};
    else {
        if constexpr (NCOLS == 1) return {4, 128};
        else if constexpr (NCOLS == 2) return n_in <= 4096 ? KregRG{8, 256} : KregRG{4, 128};
        else if constexpr (NCOLS <= 4) return {4, 128};
        else if constexpr (NCOLS >= 7) return {4, 256};
        else if constexpr (std::is_same_v<F, KregQ5K>) return n_in <= 4096 ? KregRG{8, 256} : KregRG{2, 128};
        else return {8, 256};
    }
}
inline KregRG kreg_env_rg() {
    static const KregRG v = [] {
        KregRG k{0, 0};
        if (const char* e = std::getenv("STRATA_KREG_RG")) {
            int r = 0, g = 0;
            if (std::sscanf(e, "%d,%d", &r, &g) == 2 && (r == 2 || r == 4 || r == 8) && (g == 128 || g == 256)) k = {r, g};
        }
        return k;
    }();
    return v;
}
// M2: rows wider than 16384 in the two-piece (H = 2) form, per format (a format takes one form at every column count).  The
// default from the 17408 x 5120 bench on the B70 (2026-11-02): Q6_K, IQ4_XS, IQ4_NL and Q3_K two-piece (Q6_K 1 column 165 ->
// 129 us, 4 columns 204 -> 155; IQ4_XS 1-2 columns 117-145 -> 109); Q4_K / Q5_K one-piece (two-piece: 1-2 columns faster,
// 3-4 columns slower, Q5_K 4 columns 172 -> 220 us).  STRATA_KREG_H2=1: every format two-piece, 0: none.
template <typename F>
inline bool kreg_h2(int n_in) {
    static const int e = [] { const char* v = std::getenv("STRATA_KREG_H2"); return v ? (v[0] == '1' ? 1 : 0) : -1; }();
    const bool on = e >= 0 ? e == 1 : !(std::is_same_v<F, KregQ4K> || std::is_same_v<F, KregQ5K>);
    return on && n_in > 16384 && (n_in / QK) % 2 == 0;
}
inline bool kreg_h2g() {   // STRATA_KREG_H2G=1: the two-piece form at 3-4 columns in 256 registers (activations kept), R 4
    static const bool v = [] { const char* e = std::getenv("STRATA_KREG_H2G"); return e && e[0] == '1'; }();
    return v;
}
// M4: the K-split rows' tiles.  1-4 columns R 4 / 128 registers (M2).  5-8 columns (ks 4, STRATA_KREG_KS=4, only: try_kreg
// splits ks 2) by the
// format's register needs per lane (activations a column + weights a row, both x the count, + the sums): Q4_K / Q5_K / Q3_K
// 11-12 + 11-12 at R 4 / 256 (60-96 + 48 + 32); Q6_K R 2 / 128 as its other 5-8 column rows (10 + 10: 50-80 + 20 + 16; M2's
// 256-register Q6_K forms ran 1.3-1.4x slower); IQ4_XS / IQ4_NL 9 + 9 at R 4 / 128 to 6 columns (45-54 + 36 + 32), R 2 above.
template <typename F, int NCOLS>
constexpr KregRG kreg_ks_rg() {
    if constexpr (NCOLS <= 4) return {4, 128};
    else if constexpr (std::is_same_v<F, KregQ6K>) return {2, 128};
    else if constexpr (std::is_same_v<F, KregIQ4XS> || std::is_same_v<F, KregIQ4NL>) return NCOLS <= 6 ? KregRG{4, 128} : KregRG{2, 128};
    else return {4, 256};
}
template <typename F, int NCOLS>
void kreg_dispatch_dense(const void* w, const void* x, float* y, int n_in, int n_out, dpct::queue_ptr s) {
    if (kreg_ks(n_in, n_out)) {
        constexpr KregRG c = kreg_ks_rg<F, NCOLS>();
        launch_kreg_ks<F, NCOLS, c.r, c.g>(w, x, y, n_in, n_out, s);
        return;
    }
    if constexpr (NCOLS <= 4) {   // (R 4 at 1-2 columns, 2 at 3-4; STRATA_KREG_H2G=1: 3-4 in 256 registers at R 4)
        if (kreg_h2<F>(n_in)) {
            if constexpr (NCOLS >= 3) {
                if (kreg_h2g()) { launch_kreg_dense<F, NCOLS, 4, 256, 2>(w, x, y, n_in, n_out, s); return; }
            }
            launch_kreg_dense<F, NCOLS, NCOLS <= 2 ? 4 : 2, 128, 2>(w, x, y, n_in, n_out, s);
            return;
        }
    } else {                      // 5-8 columns (STRATA_KREG_H2W=0: off): 272 lanes allow the 256-register form; R 2
        if (kreg_h2<F>(n_in)) { launch_kreg_dense<F, NCOLS, 2, 256, 2>(w, x, y, n_in, n_out, s); return; }
    }
    KregRG rg = kreg_default_rg<F, NCOLS>(n_in);
    if constexpr (NCOLS == 1) {   // one form per format (the 1-column tiles were not re-swept)
        if (rg.r == 8) launch_kreg_dense<F, 1, 8, 128>(w, x, y, n_in, n_out, s);
        else launch_kreg_dense<F, 1, 4, 128>(w, x, y, n_in, n_out, s);
        return;
    } else {
        const KregRG ov = kreg_env_rg();
        if (ov.r > 0) rg = ov;
        const int L = ((n_in / QK) * 8 + KREG_SG - 1) / KREG_SG * KREG_SG;
        if (rg.g == 256 && L > 512) rg = {rg.r > 4 ? 4 : rg.r, 128};   // (the 17408-wide ffn_down at 5-8 columns: Q4_K's R 8 / 256)
        if (rg.g == 128 && rg.r == 8) rg.r = 4;                        // (no 8-row 128-register form)
        if (rg.g == 256 && rg.r == 2) rg.r = 4;                        // (nor a 2-row 256-register one)
        if (rg.g == 256) {
            if (rg.r == 8) launch_kreg_dense<F, NCOLS, 8, 256>(w, x, y, n_in, n_out, s);
            else launch_kreg_dense<F, NCOLS, 4, 256>(w, x, y, n_in, n_out, s);
        } else {
            if (rg.r == 2) launch_kreg_dense<F, NCOLS, 2, 128>(w, x, y, n_in, n_out, s);
            else launch_kreg_dense<F, NCOLS, 4, 128>(w, x, y, n_in, n_out, s);
        }
    }
}
// ===== end of dense mode
// KREG2 (IQ4_XS at 1-4 columns under STRATA_MMVQ_KREG=1, at 5-8 columns when STRATA_KREG2_R names them; STRATA_KREG2=0:
// off): the KREG IQ4_XS per-output arithmetic
// (a column's sum is bitwise the KREG one: the 16-lane butterfly and the sub-group-ordered SLM sum do not depend on
// the work-group, the tile or R) on another schedule:
//   - a persistent grid of 60 threads per Xe core walks the tiles; at 2-4 columns the next tile's raw block words
//     (header + 16 qs bytes per lane per row) are in flight while this tile decodes, reduces and waits (PF);
//   - the first tile's loads go out before the activation loads and the codebook fill; the codebook is computed from
//     four constant words (no loads), with a second table holding the entries << 8;
//   - no lo/hi recombination: a lookup pair gives [c(lo b0), c(lo b1), c(hi b0), c(hi b1)] and the activations are
//     permuted to match once per work-group, so the integer dot is the same sum.
// Taken where it measured faster than KREG on the Arc Pro B70 (2026-11-01): 2-4 columns when the matrix has more
// tiles (R = 4 rows) than the grid (2560 x 6144..248320 and 6144 x 2560: 1.05-1.26x), and 1 column at R = 2 (one
// work-group per tile) when the tiles fit in the grid (2560 x 512 / x 640: 1.2-1.36x; KREG's R = 8 leaves cores
// idle). Elsewhere KREG. STRATA_KREG2_R="a,b,c,d[,e,f,g,h]": R (2 or 4) per column count (2,4,2,2 measured faster at 3-4
// columns on all but the head; bitwise either way).  Entries 5-8 are 0 by default (KREG at those column counts);
// "2,4,4,4,2,2,2,2" measured faster at 5-6 columns (qkv 40.1 -> 32.7 us at 5, 45.6 -> 33.3 at 6, 2026-11-10).
struct K2Raw { sycl::int2 hdr, qa, qb; };
__dpct_inline__ uint32_t k2_code(int n) {   // kIq4nlTable[n] as a byte, from registers
    const uint32_t w0 = 0xBFAD9881u, w1 = 0xF6EADDCFu, w2 = 0x26190D01u, w3 = 0x71594535u;
    const uint32_t wd = n < 8 ? (n < 4 ? w0 : w1) : (n < 12 ? w2 : w3);
    return (wd >> (8 * (n & 3))) & 0xFFu;
}
__dpct_inline__ K2Raw k2_raw(const IQ4XSBlock* b, int g) {
    K2Raw r;
    r.hdr = *reinterpret_cast<const sycl::int2*>(b);
    const sycl::int2* q2 = reinterpret_cast<const sycl::int2*>(b->qs + 16 * g);
    r.qa = q2[0]; r.qb = q2[1];
    return r;
}
// lo / hi: the P / Q words of each qs word (tl: the table, tl + 256: the table << 8); d as KregIQ4XS::load
__dpct_inline__ KregIQ4XS::W k2_decode(const K2Raw& raw, int g, const uint32_t* tl) {
    KregIQ4XS::W r;
    const uint32_t v[4] = {(uint32_t) raw.qa.x(), (uint32_t) raw.qa.y(), (uint32_t) raw.qb.x(), (uint32_t) raw.qb.y()};
    const uint32_t* tb = tl + 256;
    int lo[4], hi[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        lo[i] = (int) (tl[v[i] & 0xFF] | tb[(v[i] >> 8) & 0xFF]);
        hi[i] = (int) (tl[(v[i] >> 16) & 0xFF] | tb[v[i] >> 24]);
    }
    r.lo = sycl::int4(lo[0], lo[1], lo[2], lo[3]); r.hi = sycl::int4(hi[0], hi[1], hi[2], hi[3]);
    const uint32_t h0 = (uint32_t) raw.hdr.x(), sl = (uint32_t) raw.hdr.y();
    const int ls = (int) (((sl >> (8 * (g >> 1) + 4 * (g & 1))) & 0x0f) | (((h0 >> (16 + 2 * g)) & 0x03) << 4));
    r.d = (float) sycl::bit_cast<sycl::half>((uint16_t) (h0 & 0xFFFF)) * (float) (ls - 32);
    return r;
}
template <int R>
__dpct_inline__ void k2_load_tile(K2Raw (&raw)[R], const IQ4XSBlock* __restrict__ w, int t, int bpr, int kb, int g,
                                  int n_out) {
    const int row0 = t * R;
    const IQ4XSBlock* wt = w + std::size_t(row0) * bpr + kb;
    const int rlast = n_out - 1 - row0;
#pragma unroll
    for (int r = 0; r < R; ++r) raw[r] = k2_raw(wt + (r < rlast ? r : rlast) * bpr, g);
}
template <int NCOLS, int R, bool PF>
void kreg2_iq4xs_kernel(const IQ4XSBlock* __restrict__ w, const Q81Block* __restrict__ x, float* __restrict__ y,
                        int n_in, int n_out, float* __restrict__ slm) {
    using F = KregIQ4XS;
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const sycl::sub_group sg = it.get_sub_group();
    constexpr int P0 = R * NCOLS, P = kreg_p<R, NCOLS>(), PL = P / KREG_SG;
    const int lid = int(it.get_local_id(0)), lane = int(sg.get_local_linear_id()), sgi = lid / KREG_SG;
    const int L = int(it.get_local_range(0)), nsg = L / KREG_SG;
    const int bpr = n_in / QK, x_stride = n_in / Q8K;
    const bool live = (lid >> 3) < bpr;
    const int kb = live ? (lid >> 3) : 0, g = lid & 7;
    const int ntiles = (n_out + R - 1) / R, G = int(it.get_group_range(0));
    int t = int(it.get_group(0));   // < ntiles: the grid never exceeds the tiles
    K2Raw cur[R];
    k2_load_tile<R>(cur, w, t, bpr, kb, g, n_out);
    typename F::A a[NCOLS];
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {   // word i: [x 4i, x 4i+1, x 16+4i, x 16+4i+1] and [x 4i+2, x 4i+3, x 16+4i+2, x 16+4i+3]
        a[j] = F::act(x + std::size_t(j) * x_stride + kb * (QK / Q8K), g);
        if (!live) a[j] = typename F::A{};
        const sycl::int4 u0 = a[j].u0, u1 = a[j].u1;
        auto pl = [](int l, int h) { return (int) (((uint32_t) l & 0xFFFFu) | ((uint32_t) h << 16)); };
        auto ph = [](int l, int h) { return (int) (((uint32_t) l >> 16) | ((uint32_t) h & 0xFFFF0000u)); };
        a[j].u0 = sycl::int4(pl(u0.x(), u1.x()), pl(u0.y(), u1.y()), pl(u0.z(), u1.z()), pl(u0.w(), u1.w()));
        a[j].u1 = sycl::int4(ph(u0.x(), u1.x()), ph(u0.y(), u1.y()), ph(u0.z(), u1.z()), ph(u0.w(), u1.w()));
    }
    uint32_t* tw = reinterpret_cast<uint32_t*>(slm + 2 * nsg * P);
    for (int i = lid; i < 256; i += L) {
        const uint32_t te = k2_code(i & 15) | (k2_code(i >> 4) << 16);
        tw[i] = te;
        tw[256 + i] = te << 8;
    }
    it.barrier(sycl::access::fence_space::local_space);
    int buf = 0;
    for (; t < ntiles; t += G) {
        const int tn = t + G;
        K2Raw nxt[R];
        if constexpr (PF) k2_load_tile<R>(nxt, w, tn < ntiles ? tn : t, bpr, kb, g, n_out);
        const int row0 = t * R;
        float v[P];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const typename F::W wv = k2_decode(cur[r], g, tw);
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) v[r * NCOLS + j] = F::dot(wv, a[j]);
        }
#pragma unroll
        for (int k = P0; k < P; ++k) v[k] = 0.0f;
        kreg_tsum<KREG_SG / 2, P, P>(v, sg, lane);
        float* b = slm + buf * nsg * P;
#pragma unroll
        for (int i = 0; i < PL; ++i) b[sgi * P + lane * PL + i] = v[i];
        it.barrier(sycl::access::fence_space::local_space);
        for (int k = lid; k < P0; k += L) {
            float s = 0.0f;
            for (int q = 0; q < nsg; ++q) s += b[q * P + k];
            const int r = k / NCOLS, j = k % NCOLS;
            if (row0 + r < n_out) y[std::size_t(j) * n_out + row0 + r] = s;
        }
        buf ^= 1;
        if constexpr (PF) {
#pragma unroll
            for (int r = 0; r < R; ++r) cur[r] = nxt[r];
        } else if (tn < ntiles) {
            k2_load_tile<R>(cur, w, tn, bpr, kb, g, n_out);
        }
    }
}
template <int NCOLS, int R>
void launch_kreg2(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int G, dpct::queue_ptr s) {
    constexpr bool PF = NCOLS > 1;
    const auto* w = static_cast<const IQ4XSBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const int L = ((n_in / QK) * 8 + KREG_SG - 1) / KREG_SG * KREG_SG;
    const int nsg = L / KREG_SG;
    constexpr int P = kreg_p<R, NCOLS>();
    s->submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>(2 * nsg * P + 2 * 256), h);
        auto k = [=](sycl::nd_item<1>) [[sycl::reqd_sub_group_size(16)]] {
            kreg2_iq4xs_kernel<NCOLS, R, PF>(w, x, y, n_in, n_out, slm.get_multi_ptr<sycl::access::decorated::no>().get());
        };
        h.parallel_for<dpct_kernel_name<class q36d_kreg2_iq4xs, dpct_kernel_scalar<NCOLS>, dpct_kernel_scalar<R>>>(
            sycl::nd_range<1>(std::size_t(G) * L, L), k);
    });
}
struct Kreg2Cfg {
    bool on = true;
    // R per column count; 0 = KREG2 not used at that count.  5-8 columns are off unless STRATA_KREG2_R names them: the kernel
    // is the same template (a column's sum does not depend on R or the tile), only never measured there before 2026-11-10.
    int r[9] = {0, 2, 4, 4, 4, 0, 0, 0, 0};
};
inline const Kreg2Cfg& kreg2_cfg() {   // latched before graph capture
    static const Kreg2Cfg c = [] {
        Kreg2Cfg k;
        const char* e = std::getenv("STRATA_KREG2");
        k.on = !(e && e[0] == '0');
        if (const char* rl = std::getenv("STRATA_KREG2_R")) {   // "a,b,c,d[,e,f,g,h]": 2 or 4; anything else keeps the default
            for (int i = 1; i <= 8 && *rl; ++i) {                 // (entries 5-8: 0 keeps KREG at that column count)
                const int v = std::atoi(rl);
                if (v == 2 || v == 4 || (v == 0 && i > 4 && (*rl == '0'))) k.r[i] = v;
                while (*rl && *rl != ',') ++rl;
                if (*rl == ',') ++rl;
            }
        }
        return k;
    }();
    return c;
}
template <int NC>
bool k2_go(const Kreg2Cfg& c, const void* w, const void* x, float* y, int n_in, int n_out, dpct::queue_ptr s) {
    static const int cores = std::max(1, int(s->get_device().get_info<sycl::info::device::max_compute_units>()) / 8);
    const int nsg = ((n_in / QK) * 8 + KREG_SG - 1) / KREG_SG;
    const int grid = std::max(1, 60 / nsg) * cores;   // 60 threads per Xe core (max_compute_units / 8 XVEs)
    const int R = c.r[NC];
    if (R == 0) return false;
    const int tiles = (n_out + R - 1) / R;
    if (NC == 1 ? tiles > grid : tiles <= grid) return false;
    const int G = tiles < grid ? tiles : grid;
    if (R == 2) launch_kreg2<NC, 2>(w, x, y, n_in, n_out, G, s);
    else launch_kreg2<NC, 4>(w, x, y, n_in, n_out, G, s);
    return true;
}
inline bool try_kreg2(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, dpct::queue_ptr s) {
    const Kreg2Cfg& c = kreg2_cfg();
    if (!c.on) return false;
    if (kreg_h2<KregIQ4XS>(n_in) || kreg_ks(n_in, n_out)) return false;   // (the H = 2 / KS rows: KREG at every column count)
    switch (ncols) {
        case 1: return k2_go<1>(c, w, x, y, n_in, n_out, s);
        case 2: return k2_go<2>(c, w, x, y, n_in, n_out, s);
        case 3: return k2_go<3>(c, w, x, y, n_in, n_out, s);
        case 4: return k2_go<4>(c, w, x, y, n_in, n_out, s);
        case 5: return k2_go<5>(c, w, x, y, n_in, n_out, s);
        case 6: return k2_go<6>(c, w, x, y, n_in, n_out, s);
        case 7: return k2_go<7>(c, w, x, y, n_in, n_out, s);
        case 8: return k2_go<8>(c, w, x, y, n_in, n_out, s);
        default: return false;
    }
}

// IQ4_NL's wide kernel with the codebook in an SLM table (what the KREG form leaves: a row of a width it does not take)
struct Wide32IQ4NL {
    static constexpr int LPB = 1;
    struct W { sycl::int4 lo, hi; float d; };
    static float apply(const W& r, const Q81Block* xb) {
        return r.d * (float) xb->ds[0] * (float) dp4a4(r.hi, ld_q8_16(xb, 1), dp4a4(r.lo, ld_q8_16(xb, 0), 0));
    }
};
__dpct_inline__ void iq4nl_tbl_decode(const sycl::int4 q, const uint32_t* tl, sycl::int4& lo4, sycl::int4& hi4) {
    const uint32_t v[4] = {(uint32_t) q.x(), (uint32_t) q.y(), (uint32_t) q.z(), (uint32_t) q.w()};
    int lo[4], hi[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t P = tl[v[i] & 0xFF] | (tl[(v[i] >> 8) & 0xFF] << 8);
        const uint32_t Q = tl[(v[i] >> 16) & 0xFF] | (tl[v[i] >> 24] << 8);
        lo[i] = (int) ((P & 0xFFFFu) | (Q << 16));
        hi[i] = (int) ((P >> 16) | (Q & 0xFFFF0000u));
    }
    lo4 = sycl::int4(lo[0], lo[1], lo[2], lo[3]); hi4 = sycl::int4(hi[0], hi[1], hi[2], hi[3]);
}
template <int NCOLS>
void native_mmvq_wide32_iq4nl_tbl_kernel(const IQ4NLBlock* __restrict__ w, const Q81Block* __restrict__ x,
                                         float* __restrict__ y, int n_in, int n_out, uint32_t* __restrict__ tl) {
    using F = Wide32IQ4NL;
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    for (int i = warp * WARP + lane; i < 256; i += WARPS * WARP)   // the table, before any row returns
        tl[i] = (uint32_t) (uint8_t) kIq4nlTable[i & 15] | ((uint32_t) (uint8_t) kIq4nlTable[i >> 4] << 16);
    item.barrier(sycl::access::fence_space::local_space);
    const int row = int(item.get_group(2)) * WARPS + warp;
    if (row >= n_out) return;
    constexpr int BPS = WARP / F::LPB;                  // blocks per sub-group step
    const int blocks_per_row = n_in / 32;
    const int sub = lane / F::LPB;
    const IQ4NLBlock* wr = w + std::size_t(row) * blocks_per_row;
    float acc[NCOLS] = {};
    for (int kb = sub; kb < blocks_per_row; kb += BPS) {
        F::W wv;
        iq4nl_tbl_decode(load16_a2(wr[kb].qs), tl, wv.lo, wv.hi);
        wv.d = (float) wr[kb].d;
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) acc[j] += F::apply(wv, x + std::size_t(j) * blocks_per_row + kb);
    }
    auto sg = item.get_sub_group();
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float v = acc[j];
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
        if (lane == 0) y[std::size_t(j) * n_out + row] = v;
    }
}
template <int NCOLS>
void launch_wide32_iq4nl_tbl(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, dpct::queue_ptr s) {
    const auto* w = static_cast<const IQ4NLBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
    s->submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> tl(sycl::range<1>(256), h);
        h.parallel_for<dpct_kernel_name<class native_mmvq_wide32_iq4nl_tbl, dpct_kernel_scalar<NCOLS>>>(
            sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, WARPS, WARP), sycl::range(1, WARPS, WARP)),
            [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
                native_mmvq_wide32_iq4nl_tbl_kernel<NCOLS>(w, x, y, n_in, n_out, tl.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
}

inline bool wide_32() {   // STRATA_MMVQ_WIDE_32=0: not this kernel
    static const bool v = std::getenv("STRATA_MMVQ_WIDE_32") == nullptr || std::atoi(std::getenv("STRATA_MMVQ_WIDE_32")) != 0;
    return v;
}
bool try_wide32_iq4nl_tbl(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (!wide_32() || ncols < 1 || ncols > 8 || n_in % 32 != 0) return false;
    const auto s = strata::q_of(stream);
    switch (ncols) {
        case 1: launch_wide32_iq4nl_tbl<1>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 2: launch_wide32_iq4nl_tbl<2>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 3: launch_wide32_iq4nl_tbl<3>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 4: launch_wide32_iq4nl_tbl<4>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 5: launch_wide32_iq4nl_tbl<5>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 6: launch_wide32_iq4nl_tbl<6>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 7: launch_wide32_iq4nl_tbl<7>(weights, x_q8_1, y, n_in, n_out, s); return true;
        default: launch_wide32_iq4nl_tbl<8>(weights, x_q8_1, y, n_in, n_out, s); return true;
    }
}

template <typename F>
bool try_kreg(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (!kreg_on()) return false;   // STRATA_MMVQ_KREG unset: nothing below runs
    if (ncols < 1 || ncols > 8 || n_out <= 0 || n_in <= 0 || n_in % QK != 0 || n_in > 32768) return false;
    if (!weights || !x_q8_1 || !y) return false;   // the caller's checks report it
    const uintptr_t wa = reinterpret_cast<uintptr_t>(weights);
    if ((std::is_same_v<F, KregQ4K> || std::is_same_v<F, KregQ5K>) && (wa & 15u) != 0) return false;
    if (std::is_same_v<F, KregIQ4XS> && (wa & 7u) != 0) return false;
    if ((wa & 1u) != 0 || (reinterpret_cast<uintptr_t>(x_q8_1) & 3u) != 0) return false;
    const auto s = strata::q_of(stream);
    // A row wider than 16384 (the 27B's ffn_down, 17408: 544 lanes, so only the 128-register form) at 5-8 columns: 4 columns,
    // then the rest.  In one launch the activations of 5-8 columns spill (Q4_K 17408 x 5120: 4 columns 154 us, 6 columns
    // 665 us, 8 columns 4.4 ms, and a lost device after it).  Each launch is bitwise its one-column calls.  (The two-piece
    // formats take 5-8 columns in one 256-register launch: 17408 x 5120 Q6_K 6 columns 373 -> 315 us, IQ4_XS 309 -> 286;
    // Q4_K / Q5_K, one-piece, keep the split: their 5-8 column launches spill.  STRATA_KREG_H2W=0: split all.  The K-split
    // rows at 5-8 columns too with the halves: 4 + the rest beat one 256-register launch of 272 lanes, Q5_K 5 columns 258
    // against 401 us.  STRATA_KREG_KS=4 runs them in one launch pair, kreg_dispatch_dense.)
    static const bool h2w = [] { const char* e = std::getenv("STRATA_KREG_H2W"); return !(e && e[0] == '0'); }();
    const int ksn = kreg_ks_n(n_in, n_out);
    if (ncols > 4 && n_in > 16384 && (ksn == 2 || (ksn == 0 && !(h2w && kreg_h2<F>(n_in))))) {
        try_kreg<F>(weights, x_q8_1, y, n_in, n_out, 4, stream);
        return try_kreg<F>(weights, static_cast<const Q81Block*>(x_q8_1) + std::size_t(4) * (n_in / Q8K), y + std::size_t(4) * n_out,
                           n_in, n_out, ncols - 4, stream);
    }
    if constexpr (std::is_same_v<F, KregIQ4XS>)
        if (try_kreg2(weights, x_q8_1, y, n_in, n_out, ncols, s)) return true;   // the default; STRATA_KREG2=0: off
    switch (ncols) {
        case 1: kreg_dispatch_dense<F, 1>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 2: kreg_dispatch_dense<F, 2>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 3: kreg_dispatch_dense<F, 3>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 4: kreg_dispatch_dense<F, 4>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 5: kreg_dispatch_dense<F, 5>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 6: kreg_dispatch_dense<F, 6>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 7: kreg_dispatch_dense<F, 7>(weights, x_q8_1, y, n_in, n_out, s); return true;
        default: kreg_dispatch_dense<F, 8>(weights, x_q8_1, y, n_in, n_out, s); return true;
    }
}

// the arguments native_mmvq's wrappers would reject (a null or 4-byte-misaligned pointer, a width the format's block does not
// divide): declined here, so native_mmvq reports them as it always did
inline bool args_ok(const void* w, const void* x, const float* y, int n_in, int n_out, int ncols, int block_elems, void* stream) {
    const auto p4 = [](const void* p) { return p != nullptr && reinterpret_cast<std::uintptr_t>(p) % 4 == 0; };
    return stream != nullptr && p4(w) && p4(x) && p4(y) && n_in > 0 && n_in % block_elems == 0 && n_out > 0 && ncols >= 1 && ncols <= 8;
}

}  // namespace

bool mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    switch (ggml_type) {
        case 11: {   // Q3_K (STRATA_KREG_Q3K=0: off)
            static const bool on = [] { const char* e = std::getenv("STRATA_KREG_Q3K"); return !(e && e[0] == '0'); }();
            return on && args_ok(weights, x_q8_1, y, n_in, n_out, ncols, 256, stream) &&
                   try_kreg<KregQ3K>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        }
        case 12:
            return args_ok(weights, x_q8_1, y, n_in, n_out, ncols, 256, stream) &&
                   try_kreg<KregQ4K>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        case 13:
            return args_ok(weights, x_q8_1, y, n_in, n_out, ncols, 256, stream) &&
                   try_kreg<KregQ5K>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        case 14:
            return args_ok(weights, x_q8_1, y, n_in, n_out, ncols, 256, stream) &&
                   try_kreg<KregQ6K>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        case 23:
            return args_ok(weights, x_q8_1, y, n_in, n_out, ncols, 256, stream) &&
                   try_kreg<KregIQ4XS>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        case 20: {   // IQ4_NL: KREG (STRATA_KREG_IQ4NL=0: off), else the wide kernel with the codebook table
            static const bool on = [] { const char* e = std::getenv("STRATA_KREG_IQ4NL"); return !(e && e[0] == '0'); }();
            if (!args_ok(weights, x_q8_1, y, n_in, n_out, ncols, 32, stream)) return false;
            if (on && try_kreg<KregIQ4NL>(weights, x_q8_1, y, n_in, n_out, ncols, stream)) return true;
            return try_wide32_iq4nl_tbl(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        }
        default: return false;
    }
}

}  // namespace strata::q36dense
