// src/kernels/native_grouped_parity.cpp - native_expert_grouped's group stride (grid_groups) and its fused SwiGLU +
// q8_1 pass, against the launches before them (native_grouped_set_v1: a block row per possible group, SwiGLU and the
// q8_1 quantization as two kernels over all cap_entries).  Synthetic blobs, no model.
//
//     build/native_grouped_parity           the checks (a ctest; needs the GPU)
//     build/native_grouped_parity --bench   the checks, then v1/new timings of 48 calls in a graph (a window's layers)
//     build/native_grouped_parity --iq      SYCL port: the STRATA_IQ_DECODE_FAST expert kernels against the current ones, bitwise
//
// Every gate/up format with every down format; calls of 0..cap groups of 1..T entries each (an empty call is what the
// verify window's PCIe call usually is), entries from 0 and from past 0 (the PCIe call's follow the VRAM call's),
// scattered destinations, stale scratch, grid_groups 0 (= cap), 1, 2, 3, 4 and above cap: the output must be BITWISE
// equal to v1's, the rows the call must not write included (both start from the same NaN pattern).
//
// Random bytes are valid codes for every format here (every grid index is in range); only the fp16 block scales are
// set, small enough that the SwiGLU outputs keep a finite fp16 q8_1 scale.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/iq_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace strata::kernels {
// SYCL port only (iq_kernels.dp.cpp): the K-quant / Q5_1 / Q8_0 lane kernels with (true, the default) or without the
// multi-entry split; STRATA_KQ_SPLIT=0 at startup is false
void native_expert_set_kq_split(bool on);
// SYCL port only: the STRATA_IQ_DECODE_FAST expert kernels on (1) or off (0) whatever the environment says, -1 the environment's
void native_expert_set_iq_decode_fast(int on);
// SYCL port only: the STRATA_XE2_IQ variants of those kernels (vector loads, a three-entry pass) on (1) or off (0), -1 the environment's
void native_expert_set_xe2_iq(int on);
}
namespace k = strata::kernels;

namespace {

int g_fail = 0;

void ck(dpct::err0 e, const char *what) {
}

template<typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(DPCT_CHECK_ERROR(p = (T *)sycl::malloc_device(
                            n * sizeof(T) + 256, dpct::get_in_order_queue())),
       "malloc");
    ck(DPCT_CHECK_ERROR(
           (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memset(p, 0, n * sizeof(T) + 256).wait()),
       "memset");
    return p;
}

const char* name_of(int t) {
    switch (t) {
        case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS";
        case 20: return "IQ4_NL";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 23: return "IQ4_XS";
        case 29: return "IQ1_M";
        case 42: return "Q2_0";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case 7: return "Q5_1";
        case 8: return "Q8_0";
        default: return "?";
    }
}
int block_values(int t) { return t == 20 || t == 7 || t == 8 ? 32 : t == 42 ? 64 : 256; }

// `rows` rows of `n` values of format t: random bytes, then a finite fp16 scale in every block
std::vector<uint8_t> random_rows(int t, int64_t rows, int64_t n, std::mt19937& rng) {
    const size_t rb = k::iq_row_bytes(t, n), bs = k::iq_row_bytes(t, block_values(t));
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255), ex(2, 8), man(0, 1023), sgn(0, 1);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {
        if (t == 29) {
            // IQ1_M: the fp16 scale is the top nibbles of its four scale words (bytes 48..55), the sign and the
            // exponent's top bits in the last: 0x1 / 0x2 there (0x9 / 0xA negative) keeps it in 2^-11 .. 2^-3
            const uint8_t nib = (uint8_t) ((sgn(rng) ? 0x8 : 0x0) | (1 + (byte(rng) & 1)));
            w[o + 55] = (uint8_t) ((w[o + 55] & 0x0F) | (nib << 4));
        } else {   // the other formats start their block with the fp16 scale: 2^-13 .. 2^-6
            // (Q4_K / Q5_K: 2^-14 .. 2^-11, their 6-bit sub-scales multiply it by up to 63)
            const int e = (t == 12 || t == 13) ? 1 + ex(rng) % 4 : ex(rng);
            const uint16_t h = (uint16_t) ((sgn(rng) ? 0x8000 : 0) | (e << 10) | man(rng));
            std::memcpy(&w[o + (t == 14 ? 208 : 0)], &h, 2);   // Q6_K keeps its fp16 scale last (ql, qh, scales, d)
            if (t == 12 || t == 13 || t == 7) {   // Q4_K / Q5_K: dmin, Q5_1: m - the second fp16, as small
                const int em = (t == 7) ? ex(rng) : 1 + ex(rng) % 4;
                const uint16_t m = (uint16_t) ((sgn(rng) ? 0x8000 : 0) | (em << 10) | man(rng));
                std::memcpy(&w[o + 2], &m, 2);
            }
        }
    }
    return w;
}

// one window's worth of expert calls on one format pair: blobs, T tokens' q8_1 activations, cap = T * K entries
struct Setup {
    k::NativeExpertLayout L;
    int T = 0, K = 0, cap = 0, n_blobs = 0;
    int active_lo = 0, active_hi = 0;
    size_t slot = 0;
    uint8_t* blobs = nullptr;
    uint8_t* xq = nullptr;
    uint8_t* scr = nullptr;
    size_t scr_bytes = 0;
    float* out = nullptr;
    size_t out_floats = 0;
    unsigned long long* ptr = nullptr;
    int32_t *start = nullptr, *n = nullptr, *dst = nullptr, *tok = nullptr;
    std::vector<std::vector<uint8_t>> hb;          // host copies: the blobs, the q8_1 activations, the last plan
    std::vector<uint8_t> h_xq;
    std::vector<int> h_blob;
    std::vector<int32_t> h_start, h_tok, h_dst;

    Setup(int gu, int dt, int64_t H, int64_t FF, int tokens, int k_per_token,
          int blobs_n, std::mt19937 &rng, dpct::queue_ptr s) {
        L = k::native_expert_layout(gu, dt, H, FF);
        T = tokens; K = k_per_token; cap = T * K; n_blobs = blobs_n;
        slot = (L.bytes + 255) & ~(size_t) 255;
        blobs = dalloc<uint8_t>(slot * (size_t) n_blobs);
        for (int b = 0; b < n_blobs; ++b) {
            // gate rows | up rows (format gu, H values each) | down rows (format dt, FF values each)
            auto blob = random_rows(gu, 2 * FF, H, rng);
            const auto down = random_rows(dt, H, FF, rng);
            blob.insert(blob.end(), down.begin(), down.end());
            hb.push_back(blob);
            if (blob.size() != L.bytes) { std::fprintf(stderr, "blob %zu != %zu bytes\n", blob.size(), L.bytes); std::exit(2); }
            /*
            DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
            assuming in the original code the source host memory is pageable
            memory. If the memory is not pageable, call wait() on event return
            by memcpy API to ensure synchronization behavior.
            */
            ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                   blobs + slot * (size_t)b, blob.data(), blob.size()).wait()),
               "blob");
        }
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) T * H);
        for (auto& v : x) v = nd(rng);
        float* dx = dalloc<float>(x.size());
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        ck(DPCT_CHECK_ERROR(
               (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(dx, x.data(), x.size() * 4).wait()),
           "x");
        xq = dalloc<uint8_t>((size_t) T * (H / 32) * 36);
        k::quantize_q8_1_rows(dx, T, H, xq, s);
        h_xq.resize((size_t) T * (H / 32) * 36);
        s->memcpy(h_xq.data(), xq, h_xq.size()).wait();
        ck(DPCT_CHECK_ERROR(s->wait()), "quantize");
        sycl::free(dx, dpct::get_in_order_queue());
        scr_bytes = k::native_expert_scratch_bytes(cap, FF);
        scr = dalloc<uint8_t>(scr_bytes);
        out_floats = (size_t) cap * H;
        out = dalloc<float>(out_floats);
        ptr = dalloc<unsigned long long>(cap);
        start = dalloc<int32_t>(cap + 1);
        n = dalloc<int32_t>(1);
        dst = dalloc<int32_t>(cap);
        tok = dalloc<int32_t>(cap);
    }
    ~Setup() {
        sycl::free(blobs, dpct::get_in_order_queue());
            sycl::free(xq, dpct::get_in_order_queue());
            sycl::free(scr, dpct::get_in_order_queue());
            sycl::free(out, dpct::get_in_order_queue());
        sycl::free(ptr, dpct::get_in_order_queue());
            sycl::free(start, dpct::get_in_order_queue());
            sycl::free(n, dpct::get_in_order_queue());
            sycl::free(dst, dpct::get_in_order_queue());
            sycl::free(tok, dpct::get_in_order_queue());
    }
    // a plan: `groups` groups of 1..T entries from entry `base` on (as many as fit in cap)
    int plan(int groups, int base, std::mt19937& rng) {
        std::uniform_int_distribution<int> size(1, T);
        std::vector<int> sizes;
        int used = base;
        for (int g = 0; g < groups; ++g) {
            const int m = std::min(size(rng), cap - used);
            if (m <= 0) break;
            sizes.push_back(m);
            used += m;
        }
        return plan_sizes(sizes, base, rng);
    }
    // groups of these sizes from entry `base` on, each its own blob (as long as there are enough: a window's groups
    // are different experts); scattered destinations, random tokens
    int plan_sizes(const std::vector<int>& sizes, int base, std::mt19937& rng) {
        std::uniform_int_distribution<int> tk(0, T - 1);
        std::vector<int> order(n_blobs);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        std::vector<unsigned long long> p;
        std::vector<int32_t> st(1, base), tk_v, ds(cap);
        h_blob.clear();
        for (const int m : sizes) {
            h_blob.push_back(order[p.size() % order.size()]);
            p.push_back((unsigned long long) (blobs + slot * (size_t) order[p.size() % order.size()]));
            st.push_back(st.back() + m);
        }
        if (st.back() > cap) { std::fprintf(stderr, "plan: %d entries > cap %d\n", st.back(), cap); std::exit(2); }
        const int ng = (int) p.size();
        active_lo = base;
        active_hi = st.back();
        std::iota(ds.begin(), ds.end(), 0);
        std::shuffle(ds.begin(), ds.end(), rng);                  // scattered rows of `out`
        for (int e = 0; e < cap; ++e) tk_v.push_back(tk(rng));
        h_start = st; h_tok = tk_v; h_dst = ds;   // host copies for the CPU reference
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        if (ng) ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                       ptr, p.data(), p.size() * 8).wait()),
                   "ptr");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(start, st.data(),
                                                              st.size() * 4).wait()),
           "start");
        ck(DPCT_CHECK_ERROR(
               (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(n, &ng, 4).wait()),
           "n");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(dst, ds.data(),
                                                              ds.size() * 4).wait()),
           "dst");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(tok, tk_v.data(),
                                                              tk_v.size() * 4).wait()),
           "tok");
        return ng;
    }
    void call(int64_t grid_groups, dpct::queue_ptr s) {
        k::native_expert_grouped(L, ptr, start, n, dst, tok, cap, cap, xq, scr, out, s, grid_groups);
    }
    // `out` after one call from the NaN pattern, the scratch holding `junk` (what earlier calls left there)
    std::vector<uint32_t> result(bool v1, int64_t grid_groups, int junk,
                                 dpct::queue_ptr s) {
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                .memset(out, 0xFF, out_floats * 4)
                                .wait()),
           "memset");
        ck(DPCT_CHECK_ERROR(
               (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memset(scr, junk, scr_bytes).wait()),
           "memset");
        k::native_grouped_set_v1(v1);
        call(grid_groups, s);
        k::native_grouped_set_v1(false);
        ck(DPCT_CHECK_ERROR(s->wait()), "sync");
        std::vector<uint32_t> o(out_floats);
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                .memcpy(o.data(), out, o.size() * 4)
                                .wait()),
           "out");
        // Fused SwiGLU does not write h or inactive Q8 rows. Compare only the
        // active Q8_1 bytes, appended after the full output (including padding).
        const size_t fa = ((size_t) cap * L.n_ff * sizeof(float) + 255) & ~(size_t) 255;
        const size_t qrow = (size_t) (L.n_ff / 32) * 36;
        const size_t qb = (size_t) (active_hi - active_lo) * qrow;
        const size_t old = o.size();
        o.resize(old + qb / sizeof(uint32_t));
        if (qb)
            ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                    .memcpy(o.data() + old,
                                            scr + 3 * fa + active_lo * qrow, qb)
                                    .wait()),
               "active hq");
        return o;
    }
};

void check(int gu, int dt, int64_t H, int64_t FF, dpct::queue_ptr s,
           std::mt19937 &rng) {
    Setup S(gu, dt, H, FF, 6, 4, 9, rng, s);   // 6 tokens x 4 experts: cap 24 entries
    int calls = 0, bad = 0;
    size_t written = 0;
    for (const int base : {0, 7}) {
        for (const int groups : {0, 1, 2, 3, 5, 8, 24}) {
            const int ng = S.plan(groups, base, rng);
            const auto ref = S.result(true, 0, 0x00, s);
            size_t w = 0;
            for (size_t i = 0; i < S.out_floats; ++i) w += ref[i] != 0xFFFFFFFFu;
            written += w;
            for (const int64_t gy : {(int64_t) 0, (int64_t) 1, (int64_t) 2, (int64_t) 3, (int64_t) 4, (int64_t) S.cap + 3}) {
                const auto got = S.result(false, gy, (calls & 1) ? 0x7F : 0x00, s);
                ++calls;
                size_t diff = 0;
                for (size_t i = 0; i < ref.size(); ++i) diff += ref[i] != got[i];
                if (diff) {
                    std::printf("  %-8s/%-7s base %d, %d groups, grid_groups %lld: %zu of %zu output/Q8 words differ  FAIL\n",
                                name_of(gu), name_of(dt), base, ng, (long long) gy, diff, ref.size());
                    ++bad;
                }
            }
        }
    }
    std::printf("%-8s/%-7s %5lld x %4lld  %d calls: %s (%zu output floats written in all)\n", name_of(gu), name_of(dt),
                (long long) H, (long long) FF, calls, bad ? "FAIL" : "bitwise equal to v1", written);
    if (written == 0) { std::printf("  nothing was written: the check checked nothing  FAIL\n"); ++bad; }
    g_fail += bad;
}

// ------------------------------------------------------------------------------------------------ --kq
// MODEL-UPDATE (Intel): the K-quant / Q5_1 / Q8_0 multi-entry split (native_expert_set_kq_split) against the per-entry
// kernels, bitwise; and both against a CPU reference in float: ggml's dequantize_row_* of the weights, the q8_1
// activations' own values (d x q), SwiGLU in float (the GPU re-quantizes h to q8_1 for the down product, the reference
// does not: that rounding is the tolerance, 3e-2 of the output's largest magnitude).
float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return (float) sycl::bit_cast<sycl::half>(h);
}
// ggml-quants.c dequantize_row_* (one block)
void dq_block(int t, const uint8_t* b, float* y) {
    switch (t) {
        case 8: {   // Q8_0: d, qs[32]
            const float d = h2f(b);
            for (int i = 0; i < 32; ++i) y[i] = d * (float) (int8_t) b[2 + i];
            break;
        }
        case 7: {   // Q5_1: d, m, qh[4], qs[16]
            const float d = h2f(b), m = h2f(b + 2);
            uint32_t qh;
            std::memcpy(&qh, b + 4, 4);
            const uint8_t* qs = b + 8;
            for (int j = 0; j < 16; ++j) {
                const uint8_t xh0 = ((qh >> j) << 4) & 0x10, xh1 = ((qh >> (j + 12))) & 0x10;
                y[j] = (float) ((qs[j] & 0x0F) | xh0) * d + m;
                y[j + 16] = (float) ((qs[j] >> 4) | xh1) * d + m;
            }
            break;
        }
        case 12: case 13: {   // Q4_K: d, dmin, scales[12], qs[128]; Q5_K: d, dmin, scales[12], qh[32], qs[128]
            const float d = h2f(b), dmin = h2f(b + 2);
            const uint8_t* sc = b + 4;
            const uint8_t* qh = b + 16;
            const uint8_t* ql = t == 12 ? b + 16 : b + 48;
            auto smk = [&](int j, uint8_t& s, uint8_t& m) {
                if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; }
                else { s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
            };
            int is = 0;
            uint8_t u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                uint8_t s, m;
                smk(is, s, m);
                const float d1 = d * s, m1 = dmin * m;
                smk(is + 1, s, m);
                const float d2 = d * s, m2 = dmin * m;
                for (int l = 0; l < 32; ++l)
                    *y++ = d1 * (float) ((ql[l] & 0xF) + (t == 13 && (qh[l] & u1) ? 16 : 0)) - m1;
                for (int l = 0; l < 32; ++l)
                    *y++ = d2 * (float) ((ql[l] >> 4) + (t == 13 && (qh[l] & u2) ? 16 : 0)) - m2;
                ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
            }
            break;
        }
        case 14: {   // Q6_K: ql[128], qh[64], scales[16], d
            const uint8_t* ql = b;
            const uint8_t* qh = b + 128;
            const int8_t* sc = (const int8_t*) (b + 192);
            const float d = h2f(b + 208);
            for (int n = 0; n < 256; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16;
                    const int q1 = (int8_t) ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const int q2 = (int8_t) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int q3 = (int8_t) ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const int q4 = (int8_t) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    y[l] = d * sc[is] * q1;
                    y[l + 32] = d * sc[is + 2] * q2;
                    y[l + 64] = d * sc[is + 4] * q3;
                    y[l + 96] = d * sc[is + 6] * q4;
                }
                y += 128; ql += 64; qh += 32; sc += 8;
            }
            break;
        }
        default: std::fprintf(stderr, "dq_block: type %d\n", t); std::exit(2);
    }
}
std::vector<float> dq_rows(int t, const uint8_t* w, int64_t rows, int64_t n) {
    const int bv = block_values(t);
    const size_t bs = k::iq_row_bytes(t, bv);
    std::vector<float> y((size_t) rows * n);
    for (size_t b = 0; b < y.size() / bv; ++b) dq_block(t, w + b * bs, &y[b * bv]);
    return y;
}
// the CPU reference of the last plan's call: out rows ent_dst[e] (n_embd floats each)
std::vector<float> reference(const Setup& S) {
    const int64_t H = S.L.n_embd, FF = S.L.n_ff;
    std::vector<float> ref(S.out_floats, std::nanf(""));
    std::vector<float> x(H);
    for (size_t g = 0; g + 1 < S.h_start.size(); ++g) {
        const auto& blob = S.hb[S.h_blob[g]];
        const auto G = dq_rows(S.L.gu_type, blob.data(), FF, H);
        const auto U = dq_rows(S.L.gu_type, blob.data() + S.L.up_off, FF, H);
        const auto D = dq_rows(S.L.d_type, blob.data() + S.L.down_off, H, FF);
        for (int e = S.h_start[g]; e < S.h_start[g + 1]; ++e) {
            const uint8_t* xb = &S.h_xq[(size_t) S.h_tok[e] * (H / 32) * 36];
            for (int64_t i = 0; i < H; ++i) x[i] = h2f(xb + (i / 32) * 36) * (float) (int8_t) xb[(i / 32) * 36 + 4 + i % 32];
            std::vector<float> h(FF);
            for (int64_t r = 0; r < FF; ++r) {
                double gs = 0, us = 0;
                for (int64_t i = 0; i < H; ++i) { gs += (double) G[r * H + i] * x[i]; us += (double) U[r * H + i] * x[i]; }
                h[r] = (float) (gs / (1.0 + std::exp(-gs)) * us);
            }
            for (int64_t r = 0; r < H; ++r) {
                double s = 0;
                for (int64_t i = 0; i < FF; ++i) s += (double) D[r * FF + i] * h[i];
                ref[(size_t) S.h_dst[e] * H + r] = (float) s;
            }
        }
    }
    return ref;
}
void check_kq(int gu, int dt, int64_t H, int64_t FF, dpct::queue_ptr s, std::mt19937& rng) {
    Setup S(gu, dt, H, FF, 6, 4, 9, rng, s);   // groups of 1..6 entries: the split's chunks of 4 and 2 and the rest
    int calls = 0, bad = 0;
    double worst = 0;
    for (const int groups : {1, 2, 3, 5, 8}) {
        const int ng = S.plan(groups, 0, rng);
        k::native_expert_set_kq_split(false);
        const auto a = S.result(false, 0, 0x00, s);
        k::native_expert_set_kq_split(true);
        const auto b = S.result(false, 0, 0x7F, s);
        ++calls;
        size_t diff = 0;
        for (size_t i = 0; i < a.size(); ++i) diff += a[i] != b[i];
        if (diff) {
            std::printf("  %-6s/%-6s %d groups: split vs per-entry: %zu of %zu floats differ  FAIL\n", name_of(gu), name_of(dt),
                        ng, diff, a.size());
            ++bad;
        }
        const auto ref = reference(S);
        double amax = 0, emax = 0;
        size_t n = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (std::isnan(ref[i])) continue;
            float g;
            std::memcpy(&g, &b[i], 4);
            amax = std::max(amax, (double) std::fabs(ref[i]));
            emax = std::max(emax, (double) std::fabs(g - ref[i]));
            ++n;
        }
        const double rel = amax > 0 ? emax / amax : 1.0;
        worst = std::max(worst, rel);
        if (n == 0 || !(rel < 3e-2)) {
            std::printf("  %-6s/%-6s %d groups: GPU vs CPU reference: max err %.3g of %.3g (%zu values)  FAIL\n", name_of(gu),
                        name_of(dt), ng, emax, amax, n);
            ++bad;
        }
    }
    std::printf("%-6s/%-6s %5lld x %4lld  %d calls: split %s, CPU reference worst %.2e of the largest output\n", name_of(gu),
                name_of(dt), (long long) H, (long long) FF, calls, bad ? "FAIL" : "bitwise equal to per-entry", worst);
    g_fail += bad;
}

// ------------------------------------------------------------------------------------------------ --iq
// SYCL port: the STRATA_IQ_DECODE_FAST expert kernels (native_expert_set_iq_decode_fast) against the current ones,
// bitwise: every IQ gate/up format with both IQ down formats, groups of 1..6 entries (the chunks of 4 and 2 and the
// single tail), entries from 0 and from past 0, scattered destinations, stale scratch; the rows the call must not
// write included (both start from the same NaN pattern).  The fast kernels take 8 lanes a row (the default split).
// STRATA_XE2_IQ (native_expert_set_xe2_iq): the fast kernels with the vector loads and the three-entry pass, a third
// run of every call, bitwise the first two (the groups of 1..6 entries include groups of exactly three: one pass
// where the kernels without the switch take two).  The first two runs have it off, so they are the kernels as they
// are without the switch.  (It needs STRATA_IQ_GRID and STRATA_IQ_DT at their defaults: with either off the third run
// is the second one's kernels.)
void check_iq(int gu, int dt, int64_t H, int64_t FF, dpct::queue_ptr s, std::mt19937& rng) {
    Setup S(gu, dt, H, FF, 6, 4, 9, rng, s);   // 6 tokens x 4 experts: cap 24 entries
    int calls = 0, bad = 0;
    size_t written = 0;
    for (const int base : {0, 7})
        for (const int groups : {1, 2, 3, 5, 8, 24}) {
            const int ng = S.plan(groups, base, rng);
            k::native_expert_set_xe2_iq(0);
            k::native_expert_set_iq_decode_fast(0);
            const auto a = S.result(false, 0, 0x00, s);
            k::native_expert_set_iq_decode_fast(1);
            const auto b = S.result(false, 0, 0x7F, s);
            k::native_expert_set_xe2_iq(1);
            const auto c = S.result(false, 0, 0x7F, s);
            k::native_expert_set_xe2_iq(-1);
            k::native_expert_set_iq_decode_fast(-1);
            ++calls;
            size_t diff = 0, diffx = 0;
            for (size_t i = 0; i < a.size(); ++i) { diff += a[i] != b[i]; diffx += a[i] != c[i]; written += a[i] != 0xFFFFFFFFu; }
            if (diff) {
                std::printf("  %-8s/%-7s base %d, %d groups: fast vs current: %zu of %zu floats differ  FAIL\n", name_of(gu),
                            name_of(dt), base, ng, diff, a.size());
                ++bad;
            }
            if (diffx) {
                std::printf("  %-8s/%-7s base %d, %d groups: STRATA_XE2_IQ vs current: %zu of %zu floats differ  FAIL\n", name_of(gu),
                            name_of(dt), base, ng, diffx, a.size());
                ++bad;
            }
        }
    std::printf("%-8s/%-7s %5lld x %4lld  %d calls: %s (%zu output floats written in all)\n", name_of(gu), name_of(dt),
                (long long) H, (long long) FF, calls,
                bad ? "FAIL" : "fast decode and STRATA_XE2_IQ bitwise equal to the current kernels", written);
    if (written == 0) { std::printf("  nothing was written: the check checked nothing  FAIL\n"); ++bad; }
    g_fail += bad;
}

// ------------------------------------------------------------------------------------------------ --bench
// A window's 48 calls, one per layer, each with ITS OWN experts as the layers have: copies of the Setup's blobs in one
// arena, so the weights stream from DRAM as in the engine.  (48 calls re-reading the same 28 experts - 73 MB, about an
// Ada L2 - would read part of them from L2; the engine's layers share none.)
struct Layers {
    uint8_t* arena = nullptr;
    std::vector<unsigned long long*> ptr;           // per layer: its groups' blobs (device)
    Layers(Setup& S, int ng) {
        const int per = std::max(ng, 1);
        arena = dalloc<uint8_t>(S.slot * (size_t) (48 * per));
        for (int l = 0; l < 48; ++l) {
            std::vector<unsigned long long> p;
            for (int j = 0; j < per; ++j) {
                uint8_t* d = arena + S.slot * (size_t) (l * per + j);
                ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                       d,
                       S.blobs + S.slot * (size_t)((l * per + j) % S.n_blobs),
                       S.L.bytes).wait()),
                   "blob copy");
                p.push_back((unsigned long long) d);
            }
            unsigned long long* dp = dalloc<unsigned long long>(p.size());
            /*
            DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
            assuming in the original code the source host memory is pageable
            memory. If the memory is not pageable, call wait() on event return
            by memcpy API to ensure synchronization behavior.
            */
            ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                   dp, p.data(), p.size() * 8).wait()),
               "ptr");
            ptr.push_back(dp);
        }
    }
    Layers(const Layers&) = delete;
    Layers& operator=(const Layers&) = delete;
    ~Layers() {
        for (auto *p : ptr)
            DPCT_CHECK_ERROR(sycl::free(p, dpct::get_in_order_queue()));
        sycl::free(arena, dpct::get_in_order_queue());
    }
};

// the 48 calls of one variant captured in one graph
dpct::experimental::command_graph_exec_ptr
make_graph(dpct::queue_ptr s, Setup &S, const Layers &Ls, bool v1, int64_t gy) {
    k::native_grouped_set_v1(v1);
    dpct::experimental::command_graph_ptr g;
    dpct::experimental::command_graph_exec_ptr ge;
    ck(DPCT_CHECK_ERROR(dpct::experimental::begin_recording(s)), "capture");
    for (int l = 0; l < 48; ++l)
        k::native_expert_grouped(S.L, Ls.ptr[l], S.start, S.n, S.dst, S.tok, S.cap, S.cap, S.xq, S.scr, S.out, s, gy);
    ck(DPCT_CHECK_ERROR(dpct::experimental::end_recording(s, &g)), "capture");
    ck(DPCT_CHECK_ERROR(
           ge = new sycl::ext::oneapi::experimental::command_graph<
               sycl::ext::oneapi::experimental::graph_state::executable>(
               g->finalize())),
       "instantiate");
    ck(DPCT_CHECK_ERROR(delete (g)), "graph");
    k::native_grouped_set_v1(false);
    return ge;
}

// one launch of the graph: microseconds per call
float launch_us(dpct::queue_ptr s,
                dpct::experimental::command_graph_exec_ptr ge,
                dpct::event_ptr e0, dpct::event_ptr e1) {
    /*
    DPCT1024: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    ck(DPCT_CHECK_ERROR(dpct::sync_barrier(e0, s)), "record");
    ck(DPCT_CHECK_ERROR(s->ext_oneapi_graph(*ge)), "launch");
    /*
    DPCT1024: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    ck(DPCT_CHECK_ERROR(dpct::sync_barrier(e1, s)), "record");
    ck(DPCT_CHECK_ERROR(e1->wait_and_throw()), "bench");
    float ms = 0;
    ck(DPCT_CHECK_ERROR(
           ms = (e1->get_profiling_info<
                     sycl::info::event_profiling::command_end>() -
                 e0->get_profiling_info<
                     sycl::info::event_profiling::command_start>()) /
                1000000.0f),
       "elapsed");
    return 1e3f * ms / 48.0f;
}

float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// The two variants' launches ALTERNATE (v1-new, new-v1, ...) after a warm-up, and each reports its median: a card
// under sustained load steps its clock down after a fraction of a second (an RTX 4080 SUPER went from 80 to 89 us
// per gate/up call within one run), and whichever variant ran before the step would look faster.
constexpr int kWarmPairs = 20, kPairs = 40;

void bench(dpct::queue_ptr s, std::mt19937 &rng) {
    std::printf("\n--bench: microseconds per call, a window's 48 calls in a graph, each layer with its own experts (the\n"
                "weights stream from DRAM as in the engine); v1 / new (grid_groups): medians of %d launches each,\n"
                "alternating, after %d warm-up pairs - idle GPU assumed\n", kPairs, kWarmPairs);
    // a verify window's shape: 4 tokens x 10 experts, 2560 x 768 IQ3_S / IQ4_XS experts
    Setup S(21, 23, 2560, 768, 4, 10, 32, rng, s);
    std::vector<int> vram(16, 1);
    vram.insert(vram.end(), 12, 2);                                   // 28 groups, 40 entries
    const struct { const char* what; std::vector<int> sizes; int64_t gy; } rows[] = {
        {"no group (the PCIe call at pcie_frac 0)", {}, 4},
        {"1 group of 2 entries (a PCIe call)", {2}, 4},
        {"28 groups, 40 entries (a VRAM call)", vram, 0},
    };
    dpct::event_ptr e0, e1;
    ck(DPCT_CHECK_ERROR(e0 = new sycl::event()), "event");
    ck(DPCT_CHECK_ERROR(e1 = new sycl::event()), "event");
    for (const auto& r : rows) {
        const int ng = S.plan_sizes(r.sizes, 0, rng);
        const Layers Ls(S, ng);                                       // 48 x 28 experts: 3.7 GB for the VRAM call
        dpct::experimental::command_graph_exec_ptr ga = make_graph(s, S, Ls,
                                                                   true, 0),
                                                   gb = make_graph(s, S, Ls,
                                                                   false, r.gy);
        std::vector<float> ta, tb;
        for (int i = 0; i < kWarmPairs + kPairs; ++i) {
            const bool a_first = i % 2 == 0;
            const float x = launch_us(s, a_first ? ga : gb, e0, e1), y = launch_us(s, a_first ? gb : ga, e0, e1);
            if (i < kWarmPairs) continue;
            ta.push_back(a_first ? x : y);
            tb.push_back(a_first ? y : x);
        }
        ck(DPCT_CHECK_ERROR(delete (ga)), "graph");
        ck(DPCT_CHECK_ERROR(delete (gb)), "graph");
        std::printf("  %-42s %8.2f / %8.2f us (grid_groups %lld)\n", r.what, median(ta), median(tb), (long long) r.gy);
    }
    dpct::destroy_event(e0);
    dpct::destroy_event(e1);
}

// ------------------------------------------------------------------------------------------------ --kqbench STRIP kernels
// STRATA_KQBENCH_STRIP (bench only): the default B70 expert kernels for IQ2_XXS / IQ2_S gate/up and Q2_0 down with work taken
// out, to tell whether the call is limited by the weight reads or by the decode.  Same nd_range / work-group size / rows per
// lane / lanes per row / sub-group size, same weight and activation load addresses and widths, in the order of the real kernels:
//   gate/up: native_gu_kernel_fast<16 or 22, 8, 2, true, false> with rows_dot_fast (launch_gu_fast), iq_kernels.dp.cpp;
//   down:    native_down_kernel_dt<2> with rows_dot_dt (launch_down_fast), iq_kernels.dp.cpp.
// That is the engine's default path: STRATA_IQ_DECODE_FAST on, STRATA_IQ_GRID on, STRATA_IQ_DT on, STRATA_XE2_IQ off, 8 lanes a
// row (kqs::check_env refuses a run where an environment switch makes the real call another kernel).  STRATA_IQ_SGNMUL is a
// compile-time macro of iq_kernels.dp.cpp: its default (1) is assumed here.  The values these kernels store are not results.
//   STRIP=1 reads only: the weight words as loaded (and the fp16 block scale), the activation words as loaded (gate/up: the
//           9 loads per entry and step of acts8; down: the staged words through SLM), xor-folded; no codebook lookups, no sign
//           masks, no dp4a, no finish.  The gate/up work-groups still stage their codebook tables into SLM and hit the barrier
//           (a fixed per-work-group cost of the real kernel; one SLM word is folded into the stored value).
//   STRIP=2 reads + decode, no rows: STRIP=1 plus the real decode of each weight part (codebook lookup, sign masks, scales,
//           block scale as float), the decoded words summed into one running value instead of dp4a with the activations.
//           The activation transpose of the down kernel's staging is not done (it is activation work, not weight decode).
//   STRIP=3 (extra) weight reads only: as 1 without any activation load (gate/up) and without the down kernel's activation
//           staging, its barriers and its SLM reads.
// A work-item's value goes through the same 8-lane xor reduction as lanes_sum and is stored where the real kernel stores its
// float (the lane with sub == 0), so every lane's loads stay live.
namespace kqs {

constexpr int kDtEnt = 8, kDtW = 10, kDtMaxHb = 32;   // as native_down_kernel_dt: staged entries, words per staged q8_1 block, n_ff / 32 at most

__dpct_inline__ sycl::int2 load8_a2(const uint8_t* p) {   // = iq_kernels.dp.cpp's load8_a2: 8 bytes at a 2-byte-aligned address, as two ints
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uint32_t* q = reinterpret_cast<const uint32_t*>(a & ~uintptr_t(3));
    const uint32_t w0 = q[0], w1 = q[1];
    if (!(a & 2)) return sycl::int2((int) w0, (int) w1);
    const uint32_t w2 = q[2];
    return sycl::int2((int) ((w0 >> 16) | (w1 << 16)), (int) ((w1 >> 16) | (w2 << 16)));
}
__dpct_inline__ int load4_a2(const uint8_t* p) {   // = load4_a2: 4 bytes at a 2-byte-aligned address
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uint32_t* q = reinterpret_cast<const uint32_t*>(a & ~uintptr_t(3));
    const uint32_t w0 = q[0];
    if (!(a & 2)) return (int) w0;
    return (int) ((w0 >> 16) | (q[1] << 16));
}
__dpct_inline__ uint32_t sgn01(uint32_t n) { return (n * 0x00204081u) & 0x01010101u; }   // = iq_fast::sgn01
__dpct_inline__ int sgn(uint32_t g, uint32_t m, uint32_t m1) { return (int) ((g ^ m) + m1); }   // = iq_fast::sgn
template <int LANES>
__dpct_inline__ uint32_t lanes_xor(uint32_t v) {   // lanes_sum's butterfly on words
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
#pragma unroll
    for (int o = LANES / 2; o > 0; o >>= 1)
        v ^= sycl::permute_group_by_xor(sg, v, o);
    return v;
}

// rows_dot_fast<TG, 8, R, E, true, false>: lane `sub` takes part iqs = 2 * (sub % 8) of block sub / 8, + 1, ... of the R rows
// (nb = 10 blocks: 10 steps), E entries.  Block bytes: IQ2_XXS 66 (d, 32 x uint16), IQ2_S 82 (d, qs 64, qh 8, scales 8).
// A q8_1 block is 36 bytes: ds (fp16 pair) at 0, 32 quants at 4.
template <int TG, bool ACT, bool DEC, int R, int E>
__dpct_inline__ void gu_rows(const uint8_t* const* rows, const uint8_t* const* xs, int nb, int sub, uint32_t (&out)[R][E], const uint64_t* tab) {
    constexpr int LANES = 8, IPB = 8, STEP = 2, KSTEP = LANES / IPB, XB = 8, BS = TG == 22 ? 82 : 66;
    const int iqs = STEP * (sub % IPB);
    const int kb0 = sub / IPB;
    uint32_t s[R][E];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int e = 0; e < E; ++e) s[r][e] = 0u;
    const uint8_t* wp[R];
#pragma unroll
    for (int r = 0; r < R; ++r) wp[r] = rows[r] + (size_t) kb0 * BS;
    const uint8_t* xp[E];
#pragma unroll
    for (int e = 0; e < E; ++e) xp[e] = xs[e] + (size_t) kb0 * XB * 36;
    for (int kbx = kb0; kbx < nb; kbx += KSTEP) {
        uint32_t af[E];
#pragma unroll
        for (int e = 0; e < E; ++e) af[e] = 0u;
        if constexpr (ACT) {   // acts8: block y[iqs / 2], 8 int loads and the fp16 scale
#pragma unroll
            for (int e = 0; e < E; ++e) {
                const uint8_t* y = xp[e] + (size_t) (iqs / 2) * 36;
                const int* q = reinterpret_cast<const int*>(y + 4);
                uint32_t a = (uint32_t) *reinterpret_cast<const uint16_t*>(y);
#pragma unroll
                for (int j = 0; j < 8; ++j) a ^= (uint32_t) q[j];
                af[e] = a;
                xp[e] += (size_t) KSTEP * XB * 36;
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t* blk = wp[r];
            wp[r] += (size_t) KSTEP * BS;
            const uint16_t dh = *reinterpret_cast<const uint16_t*>(blk);   // the block's fp16 scale
            uint32_t wf;
            if constexpr (TG == 16) {   // IqFast<16>::prep_t<true>: q2 | aux32 as one load8_a2, 4 codebook words, 4 ksigns masks
                const sycl::int2 qq = load8_a2(blk + 2 + 4 * iqs);
                if constexpr (!DEC) {
                    wf = (uint32_t) qq.x() ^ (uint32_t) qq.y() ^ (uint32_t) dh;
                } else {
                    const uint32_t q2 = (uint32_t) qq.x(), aux32 = (uint32_t) qq.y();
                    uint32_t w = 0u;
#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        const uint64_t v = tab[(q2 >> (8 * kk)) & 0xFF];
                        const sycl::uint4 sm = reinterpret_cast<const sycl::uint4*>(tab + 256)[(aux32 >> (7 * kk)) & 0x7F];
                        w += (uint32_t) sgn((uint32_t) v, sm.x(), sm.z()) + (uint32_t) sgn((uint32_t) (v >> 32), sm.y(), sm.w());
                    }
                    const uint32_t a = (aux32 >> 27) | 1u;
                    wf = w ^ a ^ sycl::bit_cast<uint32_t>(static_cast<float>(sycl::bit_cast<sycl::half>(dh)));
                }
            } else {   // IqFast<22>::prep_t<true>: qs and the sign bytes as two load4_a2, qh and scales as byte loads
                const uint32_t qs = (uint32_t) load4_a2(blk + 2 + 4 * (iqs / 2));
                const uint32_t sp = (uint32_t) load4_a2(blk + 2 + 4 * (8 + iqs / 2));
                const uint32_t qh = (uint32_t) blk[66 + iqs / 2];
                const uint32_t sc = (uint32_t) blk[74 + iqs / 2];
                if constexpr (!DEC) {
                    wf = qs ^ sp ^ (qh << 8) ^ (sc << 16) ^ (uint32_t) dh;
                } else {
                    uint32_t w = 0u;
#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        const uint32_t gi = ((qs >> (8 * kk)) & 0xFF) | ((qh << (8 - 2 * kk)) & 0x300);
                        const uint64_t v = tab[gi];
                        const uint32_t t0 = sgn01((sp >> (8 * kk)) & 0x0F), t1 = sgn01((sp >> (8 * kk + 4)) & 0x0F);
                        w += (uint32_t) sgn((uint32_t) v, t0 * 0xFFu, t0) + (uint32_t) sgn((uint32_t) (v >> 32), t1 * 0xFFu, t1);
                    }
                    wf = w ^ (sc & 0x0F) ^ (sc >> 4) ^ sycl::bit_cast<uint32_t>(static_cast<float>(sycl::bit_cast<sycl::half>(dh)));
                }
            }
#pragma unroll
            for (int e = 0; e < E; ++e) s[r][e] += wf ^ af[e];
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int e = 0; e < E; ++e) out[r][e] = lanes_xor<LANES>(s[r][e]);
}

// native_gu_kernel_fast<TG, 8, 2, true, false>: 256 work-items = 32 rows x 8 lanes, 2 rows a lane (64 rows a work-group),
// entries in passes of 4, 2, 1; the codebook tables in SLM first (tab_src: TN words, IQ2_S 1024, IQ2_XXS 256 codebook + 256 masks)
template <int TG, bool ACT, bool DEC>
__dpct_inline__ void gu_kernel(sycl::nd_item<3> item, const unsigned long long* grp_ptr, const int32_t* grp_start,
                               const int32_t* n_groups, const int32_t* ent_tok, const uint8_t* xq, int n_embd, int n_ff,
                               size_t gu_row, size_t up_off, float* gate, float* up, const uint64_t* tab_src) {
    constexpr int LN = 8, RPL = 2, RB = 256 / LN, TN = TG == 22 ? 1024 : 512;
    const int ng = *n_groups;
    if ((int) item.get_group(1) >= ng) return;   // (uniform over the work-group)
    auto& sg4 = *sycl::ext::oneapi::group_local_memory_for_overwrite<sycl::uint4[TN / 2]>(item.get_group());
    uint64_t* sg = reinterpret_cast<uint64_t*>(&sg4[0]);
    {
        const int t = (int) item.get_local_id(2), nt = (int) item.get_local_range(2);
        for (int i = t; i < TN; i += nt) sg[i] = tab_src[i];
        item.barrier(sycl::access::fence_space::local_space);
    }
    const uint64_t* tab = sg;
    const int lid = (int) item.get_local_id(2);
    const uint32_t tseed = (uint32_t) tab[lid];   // keeps the staged tables live in every variant
    const int rib = lid / LN, sub = lid % LN;
    const int row0 = (int) item.get_group(2) * RB * RPL + rib;
    const int nrows = 2 * n_ff;
    if (row0 >= nrows) return;
    const int nb = n_embd / 256, xb = n_embd / 32;
    size_t off[RPL];
    float* dst[RPL];
    bool live[RPL];
#pragma unroll
    for (int i = 0; i < RPL; ++i) {
        const int v = row0 + i * RB;
        live[i] = v < nrows;
        const int vv = live[i] ? v : row0;
        const bool is_up = vv >= n_ff;
        const int r = is_up ? vv - n_ff : vv;
        off[i] = (is_up ? up_off : (size_t) 0) + (size_t) r * gu_row;
        dst[i] = (is_up ? up : gate) + r;
    }
    for (int g = (int) item.get_group(1); g < ng; g += (int) item.get_group_range(1)) {
        const uint8_t* base = (const uint8_t*) grp_ptr[g];
        const uint8_t* rows[RPL];
#pragma unroll
        for (int i = 0; i < RPL; ++i) rows[i] = base + off[i];
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        int e = e0;
        auto pass = [&](auto ec) {
            constexpr int EC = decltype(ec)::value;
            const uint8_t* xs[EC];
#pragma unroll
            for (int j = 0; j < EC; ++j) xs[j] = xq + (size_t) ent_tok[e + j] * (size_t) xb * 36;
            uint32_t o[RPL][EC];
            gu_rows<TG, ACT, DEC, RPL, EC>(rows, xs, nb, sub, o, tab);
            if (sub == 0)
#pragma unroll
                for (int i = 0; i < RPL; ++i)
                    if (live[i])
#pragma unroll
                        for (int j = 0; j < EC; ++j) dst[i][(size_t) (e + j) * n_ff] = sycl::bit_cast<float>(o[i][j] ^ tseed);
            e += EC;
        };
        for (; e + 4 <= e1;) pass(std::integral_constant<int, 4>{});
        if (e + 2 <= e1) pass(std::integral_constant<int, 2>{});
        if (e < e1) pass(std::integral_constant<int, 1>{});
    }
}

// rows_dot_dt<R, E>: lane `sub` takes part sub % 2 of blocks sub / 2, + 4, + 8 (nb = 10: 3, 3, 2, 2 steps for sub / 2 = 0..3) of R rows;
// the activations come from the staged SLM words of E entries (kDtW words per q8_1 block: 8 quants, ds, byte sum).
// A Q2_0 block is 18 bytes: fp16 scale at 0, 16 bytes of 2-bit fields at 2.
template <bool ACT, bool DEC, int R, int E>
__dpct_inline__ void down_rows(const uint8_t* const* rows, const uint32_t* ta, int hb, int nb, int sub, uint32_t (&out)[R][E]) {
    constexpr int LANES = 8, KSTEP = 4, BS = 18;
    const int iqs = sub % 2, kb0 = sub / 2;
    uint32_t s[R][E];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int e = 0; e < E; ++e) s[r][e] = 0u;
    const uint8_t* wp[R];
#pragma unroll
    for (int r = 0; r < R; ++r) wp[r] = rows[r] + kb0 * BS;
    for (int kbx = kb0; kbx < nb; kbx += KSTEP) {
        uint32_t af[E];
#pragma unroll
        for (int e = 0; e < E; ++e) af[e] = 0u;
        if constexpr (ACT) {   // the ten SLM words of each entry's staged block
#pragma unroll
            for (int e = 0; e < E; ++e) {
                const uint32_t* t = ta + (e * hb + 2 * kbx + iqs) * kDtW;
                uint32_t a = 0u;
#pragma unroll
                for (int kk = 0; kk < 8; ++kk) a ^= t[kk];
                af[e] = a ^ (t[8] & 0xFFFFu) ^ t[9];
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t* bq = wp[r];
            wp[r] += KSTEP * BS;
            const uint16_t dh = *reinterpret_cast<const uint16_t*>(bq);
            const sycl::int2 qq = load8_a2(bq + 2 + 8 * iqs);   // the part's 8 weight bytes
            uint32_t wf;
            if constexpr (!DEC) {
                wf = (uint32_t) qq.x() ^ (uint32_t) qq.y() ^ (uint32_t) dh;
            } else {   // the dt kernel's weight side: 2-bit fields of each byte, 14 shifts / masks
                uint32_t w = 0u;
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    w += (((uint32_t) qq.x() >> (2 * kk)) & 0x03030303u) + (((uint32_t) qq.y() >> (2 * kk)) & 0x03030303u);
                }
                wf = w ^ sycl::bit_cast<uint32_t>(static_cast<float>(sycl::bit_cast<sycl::half>(dh)));
            }
#pragma unroll
            for (int e = 0; e < E; ++e) s[r][e] += wf ^ af[e];
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int e = 0; e < E; ++e) out[r][e] = lanes_xor<LANES>(s[r][e]);
}

// native_down_kernel_dt<2>: 256 work-items = 32 rows x 8 lanes, 2 rows a lane (64 rows a work-group); per chunk of up to 8
// entries the work-group stages the entries' q8_1 blocks into SLM (barrier, global loads, SLM stores, barrier), then passes of 4, 2, 1
template <bool ACT, bool DEC>
__dpct_inline__ void down_kernel(sycl::nd_item<3> item, const unsigned long long* grp_ptr, const int32_t* grp_start,
                                 const int32_t* n_groups, const int32_t* ent_dst, const uint8_t* hq, int n_embd, int n_ff,
                                 size_t down_off, size_t d_row, float* out) {
    constexpr int LN = 8, RB = 256 / LN, RPL = 2;
    const int ng = *n_groups;
    if ((int) item.get_group(1) >= ng) return;   // (uniform over the work-group)
    uint32_t* ta = nullptr;
    if constexpr (ACT) {
        auto& taa = *sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t[kDtEnt * kDtMaxHb * kDtW]>(item.get_group());
        ta = &taa[0];
    }
    const int lid = (int) item.get_local_id(2);
    const int rib = lid / LN, sub = lid % LN;
    const int r0 = (int) item.get_group(2) * RB * RPL + rib;   // n_embd % 64 == 0: every lane has its rows
    const int nb = n_ff / 64, hb = n_ff / 32;
    int rr[RPL];
#pragma unroll
    for (int i = 0; i < RPL; ++i) rr[i] = r0 + i * RB;
    for (int g = (int) item.get_group(1); g < ng; g += (int) item.get_group_range(1)) {
        const uint8_t* base = (const uint8_t*) grp_ptr[g] + down_off;
        const uint8_t* rows[RPL];
#pragma unroll
        for (int i = 0; i < RPL; ++i) rows[i] = base + (size_t) rr[i] * d_row;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int ec = e0; ec < e1; ec += kDtEnt) {
            const int n = e1 - ec < kDtEnt ? e1 - ec : kDtEnt;
            if constexpr (ACT) {
                item.barrier(sycl::access::fence_space::local_space);   // the previous chunk's readers are done
                for (int x = lid; x < n * hb; x += 256) {
                    const int j = x / hb, b = x - j * hb;
                    const uint8_t* y = hq + (size_t) ((ec + j) * hb + b) * 36;
                    const int* yq = reinterpret_cast<const int*>(y + 4);
                    uint32_t a[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) a[i] = (uint32_t) yq[i];
                    uint32_t c = 0u;
                    uint32_t* d = &ta[(j * hb + b) * kDtW];
#pragma unroll
                    for (int i = 0; i < 8; ++i) { d[i] = a[i]; c ^= a[i]; }   // (the real kernel stores the 4 x 4 transpose here)
                    d[8] = *reinterpret_cast<const uint32_t*>(y);
                    d[9] = c;
                }
                item.barrier(sycl::access::fence_space::local_space);
            }
            int e = ec;
            const int ee = ec + n;
            auto pass = [&](auto ecn) {
                constexpr int EC = decltype(ecn)::value;
                const uint32_t* tp = nullptr;
                if constexpr (ACT) tp = &ta[(e - ec) * hb * kDtW];
                uint32_t o[RPL][EC];
                down_rows<ACT, DEC, RPL, EC>(rows, tp, hb, nb, sub, o);
                if (sub == 0)
#pragma unroll
                    for (int i = 0; i < RPL; ++i)
#pragma unroll
                        for (int j = 0; j < EC; ++j) out[(size_t) ent_dst[e + j] * n_embd + rr[i]] = sycl::bit_cast<float>(o[i][j]);
                e += EC;
            };
            for (; e + 4 <= ee;) pass(std::integral_constant<int, 4>{});
            if (e + 2 <= ee) pass(std::integral_constant<int, 2>{});
            if (e < ee) pass(std::integral_constant<int, 1>{});
        }
    }
}

template <int TG, bool ACT, bool DEC>
void gu_launch(dpct::queue_ptr s, const k::NativeExpertLayout& L, int64_t gy, const unsigned long long* grp_ptr,
               const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_tok, const uint8_t* xq, float* gate,
               float* up, const uint64_t* tab_src) {
    const size_t gx = (size_t) ((2 * L.n_ff + 63) / 64);   // (2 * n_ff rows + (256 / 8) * 2 - 1) / ((256 / 8) * 2)
    const int n_embd = (int) L.n_embd, n_ff = (int) L.n_ff;
    const size_t gu_row = L.gu_row, up_off = L.up_off;
    s->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, (size_t) gy, gx * 256), sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(16)]] {
                        gu_kernel<TG, ACT, DEC>(item, grp_ptr, grp_start, n_groups, ent_tok, xq, n_embd, n_ff, gu_row, up_off, gate, up, tab_src);
                    });
}
template <bool ACT, bool DEC>
void down_launch(dpct::queue_ptr s, const k::NativeExpertLayout& L, int64_t gy, const unsigned long long* grp_ptr,
                 const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst, const uint8_t* hq, float* out) {
    const size_t gx = (size_t) ((L.n_embd + 63) / 64);
    const int n_embd = (int) L.n_embd, n_ff = (int) L.n_ff;
    const size_t down_off = L.down_off, d_row = L.d_row;
    s->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, (size_t) gy, gx * 256), sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(16)]] {
                        down_kernel<ACT, DEC>(item, grp_ptr, grp_start, n_groups, ent_dst, hq, n_embd, n_ff, down_off, d_row, out);
                    });
}

// One stripped grouped call (what native_expert_grouped does with phase 0 minus the SwiGLU / q8_1 pass): gate/up when phase is not 2,
// down when phase is 0 or 2.  The scratch layout is native_expert_grouped's: gate, up, h, then the q8_1 blocks of h.
// grid y = cap_groups as there (grid_groups 0), the work-groups past *n_groups return.
void call(int mode, int phase, const k::NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
          const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap, const uint8_t* xq, uint8_t* scr,
          float* out, const uint64_t* tab, dpct::queue_ptr s) {
    if ((L.gu_type != 16 && L.gu_type != 22) || L.d_type != 42 || L.n_embd % 256 != 0 || L.n_embd % 64 != 0 || L.n_ff % 64 != 0 || L.n_ff / 32 > kDtMaxHb) {
        std::fprintf(stderr, "--kqbench STRIP: pair %d / %d at %lld x %lld is not a shape of the stripped kernels\n", L.gu_type, L.d_type,
                     (long long) L.n_embd, (long long) L.n_ff);
        std::exit(2);
    }
    const size_t f = (size_t) cap * (size_t) L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scr;
    float* up = (float*) (scr + fa);
    const uint8_t* hq = scr + 3 * fa;
    if (phase != 2) {
        if (L.gu_type == 22) {
            if (mode == 1) gu_launch<22, true, false>(s, L, cap, grp_ptr, grp_start, n_groups, ent_tok, xq, gate, up, tab);
            else if (mode == 2) gu_launch<22, true, true>(s, L, cap, grp_ptr, grp_start, n_groups, ent_tok, xq, gate, up, tab);
            else gu_launch<22, false, false>(s, L, cap, grp_ptr, grp_start, n_groups, ent_tok, xq, gate, up, tab);
        } else {
            if (mode == 1) gu_launch<16, true, false>(s, L, cap, grp_ptr, grp_start, n_groups, ent_tok, xq, gate, up, tab);
            else if (mode == 2) gu_launch<16, true, true>(s, L, cap, grp_ptr, grp_start, n_groups, ent_tok, xq, gate, up, tab);
            else gu_launch<16, false, false>(s, L, cap, grp_ptr, grp_start, n_groups, ent_tok, xq, gate, up, tab);
        }
    }
    if (phase == 0 || phase == 2) {
        if (mode == 1) down_launch<true, false>(s, L, cap, grp_ptr, grp_start, n_groups, ent_dst, hq, out);
        else if (mode == 2) down_launch<true, true>(s, L, cap, grp_ptr, grp_start, n_groups, ent_dst, hq, out);
        else down_launch<false, false>(s, L, cap, grp_ptr, grp_start, n_groups, ent_dst, hq, out);
    }
}

// The stripped kernels mirror the default path only: refuse (exit 2) where an environment switch makes the real call another kernel.
void check_env() {
    auto on = [](const char* n) { const char* v = std::getenv(n); return v != nullptr && v[0] != '\0' && v[0] != '0'; };
    auto set = [](const char* n) { const char* v = std::getenv(n); return v != nullptr && v[0] != '\0'; };
    const char* bad = nullptr;
    if (set("STRATA_IQ_DECODE_FAST") && !on("STRATA_IQ_DECODE_FAST")) bad = "STRATA_IQ_DECODE_FAST=0 (the real call is the old lane kernels)";
    else if (set("STRATA_IQ_GRID") && !on("STRATA_IQ_GRID")) bad = "STRATA_IQ_GRID=0 (the real gate/up kernel reads global tables)";
    else if (set("STRATA_IQ_DT") && !on("STRATA_IQ_DT")) bad = "STRATA_IQ_DT=0 (the real down kernel is the plain fast one)";
    else if (on("STRATA_XE2_IQ")) bad = "STRATA_XE2_IQ (the real kernels are the vector-load variants)";
    else if (on("STRATA_IQ_XXS_SGNMUL")) bad = "STRATA_IQ_XXS_SGNMUL (the real IQ2_XXS kernel computes its sign masks; the stripped one reads the table)";
    else if (set("STRATA_GU_LANES") && std::atoi(std::getenv("STRATA_GU_LANES")) != 8) bad = "STRATA_GU_LANES != 8";
    else if (set("STRATA_DOWN_LANES") && std::atoi(std::getenv("STRATA_DOWN_LANES")) != 8) bad = "STRATA_DOWN_LANES != 8";
    else if (on("STRATA_EXPERT_SPLIT")) bad = "STRATA_EXPERT_SPLIT";
    else if (on("STRATA_GU_SWIGLU")) bad = "STRATA_GU_SWIGLU";
    else if (on("STRATA_EXPDEC")) bad = "STRATA_EXPDEC";
    else if (on("STRATA_OLD_IQ_MMVQ")) bad = "STRATA_OLD_IQ_MMVQ";
    else if (on("STRATA_GROUPED_V1")) bad = "STRATA_GROUPED_V1";
    if (bad == nullptr && set("STRATA_IQ_DECODE_FAST_FMTS")) {   // the list must hold 16, 22 and 42
        bool has[64] = {};
        for (const char* p = std::getenv("STRATA_IQ_DECODE_FAST_FMTS"); *p != '\0';) {
            char* end = nullptr;
            const long t = std::strtol(p, &end, 10);
            if (end == p) break;
            if (t >= 0 && t < 64) has[t] = true;
            p = *end == ',' ? end + 1 : end;
        }
        if (!has[16] || !has[22] || !has[42]) bad = "STRATA_IQ_DECODE_FAST_FMTS without 16, 22 and 42";
    }
    if (bad != nullptr) {
        std::fprintf(stderr, "--kqbench STRIP: not the default kernels here: %s; unset it\n", bad);
        std::exit(2);
    }
}

}  // namespace kqs

// ------------------------------------------------------------------------------------------------ --kqbench [NL [host]]
// Unsloth's formats against the IQ packs' at the model's shapes (2560 x 640): microseconds per grouped call and the
// expert bytes it reads per microsecond.  NL layers (default 16), each with its own experts, so the weights stream from
// DRAM; a call reads each of its groups' whole blobs (gate + up + down) once.  Shapes: one token (10 groups of 1), a
// window of 2 tokens (16 groups, 20 entries) and of 4 (28 groups, 40 entries; native_grouped_parity's --bench).
// STRATA_GU_LANES / STRATA_DOWN_LANES pick the lane split as in the engine.
// STRATA_KQBENCH_STRIP=1|2|3 (unset or 0: the real kernels, as always): the stripped kernels above (kqs::) in place of
// native_expert_grouped, for the IQ2_XXS / IQ2_S gate/up with Q2_0 down pairs only (the other pairs print a skip line):
// 1 reads only, 2 reads + decode without the row products, 3 weight reads only.  Compare with the real kernels at the SAME
// STRATA_KQBENCH_PHASE (3 gate/up, 2 down; 0 = both: the stripped call has no SwiGLU / q8_1 pass; 1 runs as 3).  Same
// columns; the MB column is the whole blob of each group even when one phase runs (as for the real kernels).  Checks: skipped.
void kq_bench(dpct::queue_ptr s, std::mt19937& rng, int nl, bool host) {
    const int pairs[][2] = {{22, 42}, {16, 42}, {21, 20}, {23, 20}, {18, 42}, {12, 7}, {12, 8}, {13, 8},
#ifdef STRATA_Q6K_EXPERTS
                            {14, 8},
#endif
                            {8, 8}};
    std::vector<int> w1(10, 1), w2(12, 1), w4(16, 1);
    w2.insert(w2.end(), 4, 2);                                  // 16 groups, 20 entries
    w4.insert(w4.end(), 12, 2);                                 // 28 groups, 40 entries
    const struct { const char* what; int T; const std::vector<int>* sizes; } shapes[] = {
        {"1 token, 10x1", 1, &w1}, {"2 tokens, 16 groups / 20", 2, &w2}, {"4 tokens, 28 groups / 40", 4, &w4}};
    dpct::event_ptr e0 = new sycl::event(), e1 = new sycl::event();
    std::printf("\n--kqbench: %d layers per graph, each its own experts (%s); us per call (median of 40), expert MB per call, GB/s\n",
                nl, host ? "pinned host memory" : "device memory");
    // STRATA_KQBENCH_PAIRS=n: the first n pairs only (2: Flash-Next's own, IQ2_S and IQ2_XXS gate/up with Q2_0 down)
    const int all = (int) (sizeof pairs / sizeof pairs[0]);
    const int npairs = std::getenv("STRATA_KQBENCH_PAIRS") ? std::max(1, std::min(all, std::atoi(std::getenv("STRATA_KQBENCH_PAIRS")))) : all;
    // STRATA_KQBENCH_STRIP: see above; read once here
    const int strip = std::getenv("STRATA_KQBENCH_STRIP") && *std::getenv("STRATA_KQBENCH_STRIP") ? std::atoi(std::getenv("STRATA_KQBENCH_STRIP")) : 0;
    if (strip < 0 || strip > 3) { std::fprintf(stderr, "--kqbench: STRATA_KQBENCH_STRIP must be 0, 1, 2 or 3\n"); std::exit(2); }
    if (strip != 0) kqs::check_env();
    int phase = 0;   // STRATA_KQBENCH_PHASE (the stripped kernels read it here; the real ones through native_expert_set_mode)
    // STRATA_KQBENCH_PHASE=p: part of the call only (1 gate/up + swiglu, 2 down, 3 gate/up)
    if (const char* ph = std::getenv("STRATA_KQBENCH_PHASE")) {
        phase = std::atoi(ph);
        k::native_expert_set_mode(-1, std::atoi(ph));
        std::printf("--kqbench: phase %d only\n", std::atoi(ph));
    }
    if (strip != 0) {
        const char* what[] = {"", "reads only: weight and activation words loaded and folded, no codebook / decode / dp4a",
                              "reads + decode, no rows: weight parts decoded and summed, activations loaded and folded, no dp4a / finish",
                              "weight reads only: no activation loads, no activation staging"};
        std::printf("--kqbench: STRIP=%d (%s); mirrors the default B70 kernels (IQ_DECODE_FAST + IQ_GRID + IQ_DT, XE2_IQ off, 8 lanes, 2 rows a lane)\n",
                    strip, what[strip]);
        std::printf("--kqbench: STRIP=%d: parity checks skipped (the stripped kernels' stores are not results); IQ2_XXS / IQ2_S gate/up with Q2_0 down only\n", strip);
        if (phase == 1) {
            phase = 3;
            std::printf("--kqbench: STRIP: phase 1 runs as 3 (gate/up only: no SwiGLU / q8_1 pass in the stripped call)\n");
        }
        if (phase == 0) std::printf("--kqbench: STRIP: the call is gate/up + down, without the SwiGLU / q8_1 pass; compare real and stripped at STRATA_KQBENCH_PHASE=3 and =2\n");
    }
    uint64_t* kqs_tab = nullptr;   // the stripped gate/up kernels' codebook source: 1024 words (IQ2_S codebook; IQ2_XXS codebook + ksigns masks)
    if (strip != 0) {
        std::mt19937 trng(777);   // its own generator: the experts' random layout is the same with and without the switch
        std::uniform_int_distribution<int> gb(1, 62);
        std::vector<uint64_t> th(1024);
        for (auto& v : th) {   // codebook-like words (grid bytes 1..62; the lookups' cost does not depend on the content)
            v = 0;
            for (int b = 0; b < 8; ++b) v |= (uint64_t) gb(trng) << (8 * b);
        }
        for (int c = 0; c < 128; ++c) {   // IQ2_XXS: kIqKsignMask as words 256..511: {m_lo, m_hi, m_lo & 0x01010101, m_hi & 0x01010101}
            const uint32_t par = (uint32_t) __builtin_popcount((unsigned) c) & 1u, sgb = (uint32_t) c | (par << 7);
            uint32_t lo = 0, hi = 0;
            for (int b = 0; b < 4; ++b) {
                if ((sgb >> b) & 1u) lo |= 0xFFu << (8 * b);
                if ((sgb >> (4 + b)) & 1u) hi |= 0xFFu << (8 * b);
            }
            th[(size_t) (256 + 2 * c)] = (uint64_t) lo | ((uint64_t) hi << 32);
            th[(size_t) (256 + 2 * c + 1)] = (uint64_t) (lo & 0x01010101u) | ((uint64_t) (hi & 0x01010101u) << 32);
        }
        kqs_tab = dalloc<uint64_t>(th.size());
        s->memcpy(kqs_tab, th.data(), th.size() * sizeof(uint64_t)).wait();
    }
    // The engine's layout (STRATA_KQBENCH_ARENA_MIB=n): the experts sit at random slots of one n MiB device arena (the
    // engine's expert cache: one allocation, slots in profile order, so a layer's experts are scattered over all of it);
    // STRATA_KQBENCH_MIRROR=m: m groups of every layer point into pinned host memory instead (the engine's host-mirrored
    // experts, read over PCIe inside the same launch), at STRATA_KQBENCH_MPOS 0 the end of the group list (default),
    // 1 the front, 2 random places; STRATA_KQBENCH_SORT=1: the device groups of a layer in address order.
    // STRATA_KQBENCH_SHAPES=bits: which window shapes run (1: 1 token, 2: 2 tokens, 4: 4 tokens; default all).
    auto env_i = [](const char* n, long long d) { const char* v = std::getenv(n); return v && *v ? std::atoll(v) : d; };
    const long long arena_mib = env_i("STRATA_KQBENCH_ARENA_MIB", 0);
    const int n_mirror = (int) env_i("STRATA_KQBENCH_MIRROR", 0);
    const int mpos = (int) env_i("STRATA_KQBENCH_MPOS", 0);
    const bool sort_addr = env_i("STRATA_KQBENCH_SORT", 0) != 0;
    const int shape_bits = (int) env_i("STRATA_KQBENCH_SHAPES", 7);
    // STRATA_KQBENCH_SETS=n (with the arena): n graphs over different random slots, launched in turn, so consecutive
    // launches touch different pages as the engine's windows do (one set replayed keeps the TLB warm)
    const int n_sets = arena_mib > 0 ? std::max(1, (int) env_i("STRATA_KQBENCH_SETS", 1)) : 1;
    if (arena_mib > 0 || n_mirror > 0)
        std::printf("--kqbench: engine layout: arena %lld MiB, %d mirrored group(s) per layer at %s%s\n", arena_mib, n_mirror,
                    mpos == 0 ? "the end" : mpos == 1 ? "the front" : "random places", sort_addr ? ", device groups by address" : "");
    uint8_t* big = nullptr;
    if (arena_mib > 0) {
        big = (uint8_t*) sycl::malloc_device((size_t) arena_mib << 20, dpct::get_in_order_queue());
        if (big == nullptr) { std::fprintf(stderr, "--kqbench: the %lld MiB arena does not fit\n", arena_mib); std::exit(2); }
    }
    for (int pi = 0; pi < npairs; ++pi) {
        const auto& pr = pairs[pi];
        if (strip != 0 && !((pr[0] == 16 || pr[0] == 22) && pr[1] == 42)) {
            std::printf("  %-7s/%-6s skipped in the stripped run (stripped kernels: IQ2_XXS / IQ2_S gate/up with Q2_0 down)\n", name_of(pr[0]), name_of(pr[1]));
            continue;
        }
        for (int si = 0; si < 3; ++si) {
            const auto& sh = shapes[si];
            if (!((shape_bits >> (si == 2 ? 2 : si)) & 1)) continue;
            Setup S(pr[0], pr[1], 2560, 640, sh.T, 10, 28, rng, s);
            const int ng = S.plan_sizes(*sh.sizes, 0, rng);
            // host: the arena in pinned host memory, read by the kernels over PCIe (the B70's mirrored experts)
            uint8_t* arena = host ? (uint8_t*) sycl::malloc_host(S.slot * (size_t) (nl * ng) + 256, dpct::get_in_order_queue())
                                  : big != nullptr ? nullptr : dalloc<uint8_t>(S.slot * (size_t) (nl * ng));
            uint8_t* mirror = n_mirror > 0 ? (uint8_t*) sycl::malloc_host(S.slot * (size_t) (n_sets * nl * n_mirror) + 256,
                                                                          dpct::get_in_order_queue()) : nullptr;
            std::vector<size_t> slot_pick;   // the engine layout: distinct random slots of the big arena
            if (big != nullptr) {
                const size_t n_slots = ((size_t) arena_mib << 20) / S.slot;
                if (n_slots < (size_t) (n_sets * nl * ng)) { std::fprintf(stderr, "--kqbench: arena too small\n"); std::exit(2); }
                std::vector<size_t> all(n_slots);
                std::iota(all.begin(), all.end(), (size_t) 0);
                std::shuffle(all.begin(), all.end(), rng);
                slot_pick.assign(all.begin(), all.begin() + n_sets * nl * ng);
            }
            std::vector<unsigned long long*> lp;
            for (int l = 0; l < n_sets * nl; ++l) {
                std::vector<unsigned long long> p;
                std::vector<int> mj;   // the group indices of this layer that read host memory
                if (n_mirror > 0) {
                    std::vector<int> idx(ng);
                    std::iota(idx.begin(), idx.end(), 0);
                    if (mpos == 2) std::shuffle(idx.begin(), idx.end(), rng);
                    for (int i = 0; i < std::min(n_mirror, ng); ++i) mj.push_back(mpos == 0 ? ng - 1 - i : idx[(size_t) i]);
                }
                int mi = 0;
                for (int j = 0; j < ng; ++j) {
                    const bool on_host = std::find(mj.begin(), mj.end(), j) != mj.end();
                    uint8_t* d = on_host ? mirror + S.slot * (size_t) (l * n_mirror + mi++)
                               : big != nullptr ? big + S.slot * slot_pick[(size_t) (l * ng + j)]
                                                : arena + S.slot * (size_t) (l * ng + j);
                    dpct::get_in_order_queue().memcpy(d, S.blobs + S.slot * (size_t) (j % S.n_blobs), S.L.bytes).wait();
                    p.push_back((unsigned long long) d);
                }
                if (sort_addr) {   // the device groups by address; the plan's group sizes stay where they are
                    std::vector<unsigned long long> dev;
                    for (int j = 0; j < ng; ++j)
                        if (std::find(mj.begin(), mj.end(), j) == mj.end()) dev.push_back(p[(size_t) j]);
                    std::sort(dev.begin(), dev.end());
                    size_t di = 0;
                    for (int j = 0; j < ng; ++j)
                        if (std::find(mj.begin(), mj.end(), j) == mj.end()) p[(size_t) j] = dev[di++];
                }
                unsigned long long* dp = dalloc<unsigned long long>(p.size());
                dpct::get_in_order_queue().memcpy(dp, p.data(), p.size() * 8).wait();
                lp.push_back(dp);
            }
            std::vector<sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>*> ges;
            for (int st = 0; st < n_sets; ++st) {
                dpct::experimental::command_graph_ptr g;
                dpct::experimental::begin_recording(s);
                for (int l = 0; l < nl; ++l)
                    if (strip != 0)
                        kqs::call(strip, phase, S.L, lp[(size_t) (st * nl + l)], S.start, S.n, S.dst, S.tok, S.cap, S.xq, S.scr, S.out,
                                  kqs_tab, s);
                    else
                        k::native_expert_grouped(S.L, lp[(size_t) (st * nl + l)], S.start, S.n, S.dst, S.tok, S.cap, S.cap, S.xq, S.scr,
                                                 S.out, s, 0);
                dpct::experimental::end_recording(s, &g);
                ges.push_back(new sycl::ext::oneapi::experimental::command_graph<
                              sycl::ext::oneapi::experimental::graph_state::executable>(g->finalize()));
                delete g;
            }
            std::vector<float> t;
            for (int i = 0; i < 60; ++i) {
                const float us = launch_us(s, ges[(size_t) (i % n_sets)], e0, e1) * 48.0f / (float) nl;
                if (i >= 20) t.push_back(us);
            }
            for (auto* ge : ges) delete ge;
            const double mb = (double) ng * (double) S.L.bytes / 1e6, us = median(t);
            std::printf("  %-7s/%-6s %-26s %8.2f us  %6.2f MB  %6.1f GB/s\n", name_of(pr[0]), name_of(pr[1]), sh.what, us,
                        mb, mb * 1e3 / us);
            for (auto* p : lp) sycl::free(p, dpct::get_in_order_queue());
            if (arena != nullptr) sycl::free(arena, dpct::get_in_order_queue());
            if (mirror != nullptr) sycl::free(mirror, dpct::get_in_order_queue());
        }
    }
    if (big != nullptr) sycl::free(big, dpct::get_in_order_queue());
    if (kqs_tab != nullptr) sycl::free(kqs_tab, dpct::get_in_order_queue());
    dpct::destroy_event(e0);
    dpct::destroy_event(e1);
}

}  // namespace

int main(int argc, char** argv) {
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    dpct::queue_ptr s;
    ck(DPCT_CHECK_ERROR(s = dpct::get_current_device().create_queue(true)),
       "stream");
    std::mt19937 rng(316);
    if (argc > 1 && std::string(argv[1]) == "--kqbench") {   // timings only (the checks: run without arguments)
        kq_bench(s, rng, argc > 2 ? std::atoi(argv[2]) : 16, argc > 3 && std::string(argv[3]) == "host");
        dpct::get_current_device().destroy_queue(s);
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--iq") {   // the STRATA_IQ_DECODE_FAST kernels against the current ones
        if (std::getenv("STRATA_GU_LANES") || std::getenv("STRATA_DOWN_LANES")) {
            std::fprintf(stderr, "--iq: unset STRATA_GU_LANES / STRATA_DOWN_LANES (the fast decodes take 8 lanes a row)\n");
            return 2;
        }
        for (int gu : {16, 17, 18, 21, 22})
            for (int dt : {20, 42}) check_iq(gu, dt, 512, 256, s, rng);
        const int pairs[][2] = {{16, 42}, {17, 20}, {18, 20}, {21, 42}, {22, 42}};   // Flash-Next's expert pairs
        for (const auto& pr : pairs) check_iq(pr[0], pr[1], 2560, 640, s, rng);     // at the model's shapes
        std::printf("native_grouped_parity --iq: %d failures\n", g_fail);
        dpct::get_current_device().destroy_queue(s);
        return g_fail ? 1 : 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--kq") {   // the K-quant split and the CPU reference only
        const int pairs[][2] = {{12, 7}, {12, 8}, {13, 8}, {14, 8}, {8, 8}, {13, 7}};
        for (const auto& pr : pairs) check_kq(pr[0], pr[1], 2560, 640, s, rng);   // the model's shapes
        for (const auto& pr : pairs) check_kq(pr[0], pr[1], 512, 256, s, rng);
        std::printf("native_grouped_parity --kq: %d failures\n", g_fail);
        dpct::get_current_device().destroy_queue(s);
        return g_fail ? 1 : 0;
    }
    for (int gu : {16, 17, 18, 21, 22, 23, 29, 42, 12, 13,
#ifdef STRATA_Q6K_EXPERTS
                   14,
#endif
                   8})        // STRATA_GU_FMTS
        for (int dt : {20, 23, 42, 7, 8}) check(gu, dt, 512, 256, s, rng);   // STRATA_D_FMTS; IQ4_XS: n_ff % 256
    check(42, 42, 2560, 640, s, rng);
    // the served Unsloth UD-IQ1_S pack's expert pairs at the model's 2560 x 640 (IQ2_XXS / IQ2_S gate/up, Q2_0 down; upstream c1d5e4d5)
    check(16, 42, 2560, 640, s, rng);
    check(22, 42, 2560, 640, s, rng);
    check(21, 20, 2560, 640, s, rng);                                   // a model's shapes
    check(21, 23, 2560, 768, s, rng);
    if (do_bench) bench(s, rng);
    std::printf("native_grouped_parity: %d failures\n", g_fail);
    dpct::get_current_device().destroy_queue(s);
    return g_fail ? 1 : 0;
}
