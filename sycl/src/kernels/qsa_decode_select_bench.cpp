#include <sycl/sycl.hpp>
// src/kernels/qsa_decode_select_bench.cpp - the verifier's long-context decode select (STRATA_SELECT_LONG):
// qsa_block_scores_long against qsa_block_scores (memcmp of the score bits) and qsa_block_topk_long against
// qsa_block_topk (the ids), on synthetic keys and on crafted score rows (ties at the threshold, all equal, a tail block
// of 1-3 cells, contexts up to the capacity), then each timed (run it under unitrace -d for the device times).
// Usage: qsa_decode_select_bench [context=196608] [nq=3] [reps=200] [capacity=262144]
// Exit 0: every check bitwise equal.
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/sycl_queue.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
sycl::queue& Q() { return *strata::q_of(nullptr); }
template <typename T> T* dev(size_t n) { T* p = sycl::malloc_device<T>(n + 16, Q()); Q().memset(p, 0, (n + 16) * sizeof(T)).wait(); return p; }
template <typename T> void up(T* d, const std::vector<T>& h) { Q().memcpy(d, h.data(), h.size() * sizeof(T)).wait(); }
template <typename T> std::vector<T> down(const T* d, size_t n) { std::vector<T> h(n); Q().memcpy(h.data(), d, n * sizeof(T)).wait(); return h; }

int g_fail = 0;

std::vector<int32_t> steps_for(int64_t ctx, int nq, const k::QsaShapes& s) {
    std::vector<int32_t> st((size_t) nq * k::kStepCount);
    for (int q = 0; q < nq; ++q) k::qsa_step_fill(st.data() + (size_t) q * k::kStepCount, ctx - nq + q, s);
    return st;
}

// the ids of both top-k calls on one score buffer, compared over each query's width
bool check_topk(const char* what, const float* d_sc, const int32_t* d_st, const std::vector<int32_t>& hst, int nq,
                int64_t max_blocks, int64_t cap, const k::QsaShapes& s, int32_t* d_ids_a, int32_t* d_ids_b,
                int32_t* d_scratch) {
    Q().memset(d_ids_a, 0xff, (size_t) nq * cap * 4).wait();
    Q().memset(d_ids_b, 0xff, (size_t) nq * cap * 4).wait();
    k::qsa_block_topk(d_sc, d_st, nq, max_blocks, cap, s, d_ids_a, nullptr);
    k::qsa_block_topk_long(d_sc, d_st, nq, max_blocks, cap, s, d_ids_b, nullptr, -1, d_scratch);
    Q().wait();
    auto a = down(d_ids_a, (size_t) nq * cap), b = down(d_ids_b, (size_t) nq * cap);
    int bad = 0;
    for (int q = 0; q < nq; ++q) {
        const int w = hst[(size_t) q * k::kStepCount + k::kStepWidth];
        for (int i = 0; i < w; ++i)
            if (a[(size_t) q * cap + i] != b[(size_t) q * cap + i]) { if (bad < 3) std::printf("  %s q%d i%d ref %d got %d\n", what, q, i, a[(size_t) q * cap + i], b[(size_t) q * cap + i]); ++bad; }
    }
    std::printf("topk %-28s nq %d: %s\n", what, nq, bad ? "DIFFER" : "same ids");
    if (bad) ++g_fail;
    return bad == 0;
}
}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const int64_t ctx = argc > 1 ? std::atoll(argv[1]) : 196608;
    const int nq_t = argc > 2 ? std::atoi(argv[2]) : 3;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 200;
    const int64_t capacity = argc > 4 ? std::atoll(argv[4]) : 262144;
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_blocks = capacity / s.idx_block + 2;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    constexpr int MAXQ = 8;
    std::printf("device %s, local memory %zu B\n", Q().get_device().get_info<sycl::info::device::name>().c_str(),
                (size_t) Q().get_device().get_info<sycl::info::device::local_mem_size>());
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> pooled((size_t) max_blocks * 128), dead(128), qidx((size_t) MAXQ * 4 * 128);
    for (auto& v : pooled) v = nd(rng);
    for (auto& v : dead) v = nd(rng);
    for (auto& v : qidx) v = nd(rng) * 0.1f;
    float* d_pooled = dev<float>(pooled.size());
    float* d_dead = dev<float>(dead.size());
    float* d_q = dev<float>(qidx.size());
    up(d_pooled, pooled); up(d_dead, dead); up(d_q, qidx);
    float* d_sa = dev<float>((size_t) MAXQ * max_blocks);
    float* d_sb = dev<float>((size_t) MAXQ * max_blocks);
    int32_t* d_st = dev<int32_t>((size_t) MAXQ * k::kStepCount);
    int32_t* d_ia = dev<int32_t>((size_t) MAXQ * cap);
    int32_t* d_ib = dev<int32_t>((size_t) MAXQ * cap);
    const int64_t scr = k::qsa_topk_multi_scratch_ints(MAXQ);
    int32_t* d_scr = dev<int32_t>((size_t) (scr > 0 ? scr : 1));   // zero, as the verifier's arena

    // 1. the score bits: the long scorer against the plain one, nq 1..6, several contexts (n_kv % 4 = 0..3)
    for (int nq = 1; nq <= 6; ++nq)
        for (int64_t c : {ctx, ctx + 1, ctx + 2, ctx + 3, (int64_t) 70001, capacity - 1}) {
            if (c > capacity) continue;
            auto hst = steps_for(c, nq, s);
            up(d_st, hst);
            Q().memset(d_sa, 0, (size_t) nq * max_blocks * 4).wait();
            Q().memset(d_sb, 0, (size_t) nq * max_blocks * 4).wait();
            k::qsa_block_scores(d_pooled, d_dead, d_q, d_st, nq, max_blocks, s, d_sa, nullptr);
            k::qsa_block_scores_long(d_pooled, d_dead, d_q, d_st, nq, max_blocks, s, d_sb, nullptr, -1);
            Q().wait();
            auto a = down(d_sa, (size_t) nq * max_blocks), b = down(d_sb, (size_t) nq * max_blocks);
            int64_t bad = 0;
            for (int q = 0; q < nq; ++q) {
                const int64_t nb = hst[(size_t) q * k::kStepCount + k::kStepNBid] + 1;
                bad += std::memcmp(a.data() + (size_t) q * max_blocks, b.data() + (size_t) q * max_blocks, (size_t) nb * 4) != 0;
            }
            if (bad) { std::printf("scores nq %d ctx %lld: DIFFER\n", nq, (long long) c); ++g_fail; }
            // 2. the ids on these scores
            char what[64];
            std::snprintf(what, sizeof what, "scorer ctx %lld", (long long) c);
            check_topk(what, d_sb, d_st, hst, nq, max_blocks, cap, s, d_ia, d_ib, d_scr);
        }
    std::printf("scores: long == plain bitwise for nq 1..6 (%s)\n", g_fail ? "SEE ABOVE" : "all equal");

    // 3. crafted rows: ties, equal scores, few distinct values, mostly zero, a near-full context
    std::uniform_int_distribution<int> ui(0, 1 << 30);
    for (int kind = 0; kind < 6; ++kind)
        for (int nq : {1, 3, 6}) {
            const int64_t c = kind == 5 ? capacity - 3 : ctx + kind % 4;
            auto hst = steps_for(c, nq, s);
            up(d_st, hst);
            std::vector<float> sc((size_t) nq * max_blocks, 0.f);
            for (int q = 0; q < nq; ++q)
                for (int64_t b = 0; b < max_blocks; ++b) {
                    float v = 0.f;
                    switch (kind) {
                        case 0: v = (float) (ui(rng) % 8); break;                          // 8 values: huge ties
                        case 1: v = 3.0f; break;                                           // all equal
                        case 2: v = (ui(rng) % 100 == 0) ? 1.0f + (ui(rng) % 4) : 0.0f; break;   // mostly zero
                        case 3: v = 10.0f + (float) (ui(rng) % 4096) * (1.0f / 4096.f); break;   // one coarse bin
                        case 4: v = std::fabs(nd(rng)) * 5.f; break;                       // continuous
                        default: v = (float) (ui(rng) % 1000) * 0.01f; break;              // 1000 values, full context
                    }
                    sc[(size_t) q * max_blocks + b] = v;
                }
            for (int q = 0; q < nq; ++q) {   // the tail block as the scorer leaves it
                const int32_t* st = hst.data() + (size_t) q * k::kStepCount;
                if (st[k::kStepNKv] % 4) sc[(size_t) q * max_blocks + st[k::kStepNBid]] += 1e9f;
            }
            up(d_sa, sc);
            char what[64];
            std::snprintf(what, sizeof what, "crafted kind %d ctx %lld", kind, (long long) c);
            check_topk(what, d_sa, d_st, hst, nq, max_blocks, cap, s, d_ia, d_ib, d_scr);
        }
    // short contexts (the identity and just past it)
    for (int64_t c : {(int64_t) 100, (int64_t) 2051, (int64_t) 2052, (int64_t) 4099}) {
        auto hst = steps_for(c, 3, s);
        up(d_st, hst);
        k::qsa_block_scores(d_pooled, d_dead, d_q, d_st, 3, max_blocks, s, d_sb, nullptr);
        char what[64];
        std::snprintf(what, sizeof what, "short ctx %lld", (long long) c);
        check_topk(what, d_sb, d_st, hst, 3, max_blocks, cap, s, d_ia, d_ib, d_scr);
    }

    // 3b. the INT8 KV append's host copy (KV streaming; STRATA_KV_APPEND_PACK writes it eight codes to a store) against
    // its VRAM copy of the same cells: byte for byte
    {
        const int KH = (int) s.n_head_kv, HD = (int) s.head_dim, PS = (int) s.page_size, GR = HD / k::KV_Q8_GROUP;
        const int ntok = 3, npages = 8;
        const int64_t P = 4 * 1000 + 2, nblocks = P / PS + 4;
        std::vector<int32_t> pt((size_t) nblocks, -1);
        pt[P / PS] = 5; pt[P / PS + 1] = 2;   // both blocks resident, in pages 5 and 2
        std::vector<int32_t> st((size_t) ntok * k::kStepCount);
        for (int j = 0; j < ntok; ++j) k::qsa_step_fill(st.data() + (size_t) j * k::kStepCount, P + j, s);
        std::vector<float> kc((size_t) ntok * KH * HD), vc(kc.size());
        for (auto& v : kc) v = nd(rng);
        for (auto& v : vc) v = nd(rng) * 3.f;
        int32_t* d_pt = dev<int32_t>(pt.size()); up(d_pt, pt);
        int32_t* d_st2 = dev<int32_t>(st.size()); up(d_st2, st);
        float* d_kc = dev<float>(kc.size()); up(d_kc, kc);
        float* d_vc = dev<float>(vc.size()); up(d_vc, vc);
        const size_t rows_d = (size_t) npages * KH * PS, rows_h = (size_t) nblocks * KH * PS;
        int8_t* dk = dev<int8_t>(rows_d * HD); int8_t* dv = dev<int8_t>(rows_d * HD);
        uint16_t* dks = dev<uint16_t>(rows_d * GR); uint16_t* dvs = dev<uint16_t>(rows_d * GR);
        k::KvHostPools hp;
        hp.k_q = sycl::malloc_host<int8_t>(rows_h * HD, Q()); hp.v_q = sycl::malloc_host<int8_t>(rows_h * HD, Q());
        hp.k_scale = sycl::malloc_host<uint16_t>(rows_h * GR, Q()); hp.v_scale = sycl::malloc_host<uint16_t>(rows_h * GR, Q());
        std::memset(hp.k_q, 0x55, rows_h * HD); std::memset(hp.v_q, 0x55, rows_h * HD);
        k::kv_append_q8_steps(dk, dv, dks, dvs, d_pt, d_st2, k::kStepCount, d_kc, d_vc, KH * HD, ntok, s, nullptr, &hp);
        Q().wait();
        auto hk = down(dk, rows_d * HD), hv = down(dv, rows_d * HD);
        auto hks = down(dks, rows_d * GR), hvs = down(dvs, rows_d * GR);
        int bad = 0;
        for (int j = 0; j < ntok; ++j)
            for (int h = 0; h < KH; ++h) {
                const int64_t pos = P + j, page = pt[pos / PS];
                const size_t rd = (size_t) ((page * KH + h) * PS + pos % PS), rh = (size_t) (((pos / PS) * KH + h) * PS + pos % PS);
                bad += std::memcmp(hk.data() + rd * HD, hp.k_q + rh * HD, HD) != 0;
                bad += std::memcmp(hv.data() + rd * HD, hp.v_q + rh * HD, HD) != 0;
                bad += std::memcmp(hks.data() + rd * GR, hp.k_scale + rh * GR, GR * 2) != 0;
                bad += std::memcmp(hvs.data() + rd * GR, hp.v_scale + rh * GR, GR * 2) != 0;
            }
        std::printf("kv append: host copy %s the VRAM copy (%d rows)\n", bad ? "DIFFERS FROM" : "byte-identical to", ntok * KH);
        if (bad) ++g_fail;
    }

    // 4. timing at ctx, nq_t queries (device times: unitrace -d)
    auto hst = steps_for(ctx, nq_t, s);
    up(d_st, hst);
    for (int i = 0; i < 5; ++i) k::qsa_block_scores_long(d_pooled, d_dead, d_q, d_st, nq_t, max_blocks, s, d_sb, nullptr, -1);
    Q().wait();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) k::qsa_block_scores_long(d_pooled, d_dead, d_q, d_st, nq_t, max_blocks, s, d_sb, nullptr, -1);
    Q().wait();
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) k::qsa_block_topk_long(d_sb, d_st, nq_t, max_blocks, cap, s, d_ib, nullptr, -1, d_scr);
    Q().wait();
    auto t2 = std::chrono::steady_clock::now();
    const double us_s = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    const double us_t = std::chrono::duration<double, std::micro>(t2 - t1).count() / reps;
    const double gb = (double) hst[(size_t) (nq_t - 1) * k::kStepCount + k::kStepNBid] * 512.0 / 1e9;
    std::printf("timing ctx %lld nq %d: scores_long %.1f us/call (%.0f GB/s of keys), topk_long %.1f us/call (wall, "
                "queue-bound for small kernels)\n", (long long) ctx, nq_t, us_s, gb / (us_s * 1e-6), us_t);
    check_topk("after timing", d_sb, d_st, hst, nq_t, max_blocks, cap, s, d_ia, d_ib, d_scr);
    std::printf("%s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
