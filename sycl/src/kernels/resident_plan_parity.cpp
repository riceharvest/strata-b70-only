// src/kernels/resident_plan_parity.cpp - the verify window's device-built expert plan (#783 PR-e: the parallel
// grouping and the prefix scan that replaced thread 0's serial pass) against a host replay of its contract:
// distinct experts in routing order, each with its entries ascending, `ptr[g]` = the expert's slot, counts / start /
// dst / tok / start2 in the host pool's layout.
//
// n = 1..80 entries (a window of up to kVerifyMaxT tokens x 10 experts), ids drawn from pools of 3..512 experts so
// duplicates run from "every entry the same expert" to "all distinct", with and without a slot table (`slot_off`).
// Both modes are run: with a `skip` word (ring when the plan was built, 0 when an expert was not resident - and
// then the plan is untouched) and without one (the all-resident graph: a non-resident expert gives an empty plan and
// sets *plan_err).
// Every word of the plan the host pool would read is compared with memcmp. GPU, synthetic, no model.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/resident_plan_mirror.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(dpct::err0 e, const char *w) {
}
}  // namespace

int main() {
    const int K = 10, NE = 512;
    const long long capx = (long long) k::kVerifyMaxT * K;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    const size_t plan_words = (size_t) (ptr_off + 4 * capx + capx + 16);
    std::mt19937 rng(783);
    const long long blob = 1337;
    uint8_t* const cache_base = reinterpret_cast<uint8_t*>(uintptr_t(0x40000000));   // never dereferenced

    int32_t *d_ids, *d_res, *d_plan;
    unsigned long long* d_off;
    uint32_t* d_skip;
    ck(DPCT_CHECK_ERROR(d_ids = (int32_t *)sycl::malloc_device(
                            128 * 4, dpct::get_in_order_queue())),
       "ids");
    ck(DPCT_CHECK_ERROR(d_res = (int32_t *)sycl::malloc_device(
                            NE * 4, dpct::get_in_order_queue())),
       "res");
    ck(DPCT_CHECK_ERROR(d_off = (unsigned long long *)sycl::malloc_device(
                            1024 * 8, dpct::get_in_order_queue())),
       "off");
    ck(DPCT_CHECK_ERROR(d_plan = (int32_t *)sycl::malloc_device(
                            plan_words * 4, dpct::get_in_order_queue())),
       "plan");
    ck(DPCT_CHECK_ERROR(d_skip = (uint32_t *)sycl::malloc_device(
                            4, dpct::get_in_order_queue())),
       "skip");
    uint32_t* d_err = nullptr;
    ck(DPCT_CHECK_ERROR(d_err = (uint32_t *)sycl::malloc_device(
                            4, dpct::get_in_order_queue())),
       "err");

    std::vector<int32_t> res(NE);
    for (int e = 0; e < NE; ++e) res[e] = (e * 37 + 11) % 1000;      // distinct slots, not the identity
    std::vector<unsigned long long> off(1024);
    for (int s = 0; s < 1024; ++s) off[s] = (unsigned long long) s * 4096 + (s % 7) * 64;
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming
    in the original code the source host memory is pageable memory. If the
    memory is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    ck(DPCT_CHECK_ERROR(
           (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_res, res.data(), NE * 4).wait()),
       "res");
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming
    in the original code the source host memory is pageable memory. If the
    memory is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    ck(DPCT_CHECK_ERROR(
           (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_off, off.data(), 1024 * 8).wait()),
       "off");

    int cases = 0, bad = 0;
    for (int n = 1; n <= 80; ++n) {
        for (int variant = 0; variant < 4; ++variant) {
            const int pool = variant == 0 ? 3 : (variant == 1 ? 12 : (variant == 2 ? 40 : 512));
            std::vector<int32_t> ids(n);
            for (int i = 0; i < n; ++i) ids[i] = (int32_t) ((rng() % pool) * (512 / pool));
            const bool use_off = (n + variant) % 2 == 1;
            /*
            DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
            assuming in the original code the source host memory is pageable
            memory. If the memory is not pageable, call wait() on event return
            by memcpy API to ensure synchronization behavior.
            */
            ck(DPCT_CHECK_ERROR(
                   (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_ids, ids.data(), n * 4).wait()),
               "ids");

            // host replay
            std::vector<int32_t> counts = {0, n, 0}, start, dst(n), tok(n);
            std::vector<unsigned long long> ptr;
            std::vector<int> first_seen;     // group -> expert
            for (int i = 0; i < n; ++i) {
                int g = -1;
                for (size_t q = 0; q < first_seen.size(); ++q)
                    if (first_seen[q] == ids[i]) g = (int) q;
                if (g < 0) { first_seen.push_back(ids[i]); g = (int) first_seen.size() - 1; }
            }
            const int groups = (int) first_seen.size();
            counts[0] = groups;
            start.assign(groups + 1, 0);
            int pos = 0;
            for (int g = 0; g < groups; ++g) {
                start[g] = pos;
                ptr.push_back((unsigned long long) (uintptr_t) cache_base +
                              (use_off ? off[res[first_seen[g]]] : (unsigned long long) res[first_seen[g]] * (unsigned long long) blob));
                for (int i = 0; i < n; ++i)
                    if (ids[i] == first_seen[g]) { dst[pos] = i; tok[pos] = i / K; ++pos; }
            }
            start[groups] = n;

            for (int mode = 0; mode < 2; ++mode) {   // 0: general kernel with a skip word, 1: all resident
                ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                        .memset(d_plan, 0xAB, plan_words * 4)
                                        .wait()),
                   "fill");
                ck(DPCT_CHECK_ERROR(
                       (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memset(d_skip, 0, 4).wait()),
                   "skip0");
                ck(DPCT_CHECK_ERROR(
                       (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memset(d_err, 0, 4).wait()),
                   "err0");
                k::resident_plan(d_ids, n, K, d_res, NE, cache_base, use_off ? d_off : nullptr, blob, d_plan, capx,
                                 mode == 0 ? d_skip : nullptr, 7u, nullptr, d_err);
                ck(DPCT_CHECK_ERROR(
                       dpct::get_current_device().queues_wait_and_throw()),
                   "sync");
                std::vector<int32_t> plan(plan_words);
                ck(DPCT_CHECK_ERROR(
                       (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                           .memcpy(plan.data(), d_plan, plan_words * 4)
                           .wait()),
                   "plan");
                uint32_t skip = 0;
                ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                        .memcpy(&skip, d_skip, 4)
                                        .wait()),
                   "skipr");
                bool ok = mode == 1 || skip == 7u;
                ok = ok && std::memcmp(plan.data(), counts.data(), 3 * 4) == 0;
                ok = ok && std::memcmp(plan.data() + 4, start.data(), (size_t) (groups + 1) * 4) == 0;
                const int32_t* pdst = plan.data() + 4 + capx + 1;
                ok = ok && std::memcmp(pdst, dst.data(), (size_t) n * 4) == 0;
                ok = ok && std::memcmp(pdst + capx, tok.data(), (size_t) n * 4) == 0;
                ok = ok && std::memcmp(plan.data() + ptr_off, ptr.data(), (size_t) groups * 8) == 0;
                ok = ok && plan[(size_t) (ptr_off + 4 * capx)] == n;
                ++cases;
                if (!ok) {
                    std::printf("    *** n=%d pool=%d %s %s: the plan differs from the host replay ***\n", n, pool,
                                use_off ? "slot_off" : "blob", mode ? "all_resident" : "general");
                    ++bad;
                }
            }
        }
    }

    // a routed expert that is not resident: the general kernel leaves the plan alone and zeroes *skip
    {
        std::vector<int32_t> res2 = res;
        res2[5] = -1;
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        ck(DPCT_CHECK_ERROR(
               (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_res, res2.data(), NE * 4).wait()),
           "res2");
        const int32_t ids[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        ck(DPCT_CHECK_ERROR(
               (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_ids, ids, sizeof ids).wait()),
           "ids");
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                .memset(d_plan, 0xAB, plan_words * 4)
                                .wait()),
           "fill");
        const uint32_t seven = 7;
        ck(DPCT_CHECK_ERROR(
               (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_skip, &seven, 4).wait()),
           "skip7");
        k::resident_plan(d_ids, 10, K, d_res, NE, cache_base, nullptr, blob, d_plan, capx, d_skip, 7u, nullptr, d_err);
        ck(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()),
           "sync");
        std::vector<int32_t> plan(plan_words);
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                .memcpy(plan.data(), d_plan, plan_words * 4)
                                .wait()),
           "plan");
        uint32_t skip = 99;
        ck(DPCT_CHECK_ERROR(
               (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(&skip, d_skip, 4).wait()),
           "skipr");
        bool untouched = true;
        for (int32_t w : plan) untouched = untouched && (uint32_t) w == 0xABABABABu;
        ++cases;
        if (skip != 0 || !untouched) {
            std::printf("    *** a non-resident expert: skip=%u untouched=%d (want 0 / 1) ***\n", skip, (int) untouched);
            ++bad;
        }
    }

    // STRATA_MIRROR_STAGE: a third of the experts host-mirrored; the plan lists the VRAM groups first (routing order,
    // counts[3] of them), then the mirrored ones (routing order), these also as a list of their own (ptr2 / start2 /
    // counts[2]) pointing at stage + j * stride, their mirror addresses in src[j]
    {
        std::vector<int32_t> res3(NE);
        std::vector<unsigned long long> mtab(NE, 0ull);
        for (int e = 0; e < NE; ++e) {
            const bool mirrored = (e * 7 + 3) % 3 == 0;
            res3[e] = mirrored ? -1 : (e * 37 + 11) % 1000;
            if (mirrored) mtab[e] = 0x7f0000000000ull + (unsigned long long) e * 1536 * 1024;   // never dereferenced
        }
        unsigned long long* d_tab = sycl::malloc_device<unsigned long long>(NE, dpct::get_in_order_queue());
        unsigned long long* d_src = sycl::malloc_device<unsigned long long>(capx, dpct::get_in_order_queue());
        dpct::get_in_order_queue().memcpy(d_res, res3.data(), NE * 4).wait();
        dpct::get_in_order_queue().memcpy(d_tab, mtab.data(), NE * 8).wait();
        k::resident_plan_set_mirror(d_res, d_tab, nullptr, NE);
        uint8_t* const stage = reinterpret_cast<uint8_t*>(uintptr_t(0x50000000));   // never dereferenced
        const long long stride = 1536 * 1024;
        for (int n = 1; n <= 80; ++n)
            for (int variant = 0; variant < 4; ++variant) {
                const int pool = variant == 0 ? 3 : (variant == 1 ? 12 : (variant == 2 ? 40 : 512));
                std::vector<int32_t> ids(n);
                for (int i = 0; i < n; ++i) ids[i] = (int32_t) ((rng() % pool) * (512 / pool));
                dpct::get_in_order_queue().memcpy(d_ids, ids.data(), n * 4).wait();
                std::vector<int> vexp, mexp;   // distinct experts in routing order, VRAM / mirrored
                for (int i = 0; i < n; ++i) {
                    std::vector<int>& L = res3[ids[i]] >= 0 ? vexp : mexp;
                    if (std::find(L.begin(), L.end(), ids[i]) == L.end()) L.push_back(ids[i]);
                }
                const int gv = (int) vexp.size(), gm = (int) mexp.size(), groups = gv + gm;
                std::vector<int32_t> start(groups + 1), dst(n), tok(n), start2(gm + 1);
                std::vector<unsigned long long> ptr(groups), ptr2(gm), src(gm);
                int pos = 0;
                for (int g = 0; g < groups; ++g) {
                    const int e = g < gv ? vexp[g] : mexp[g - gv];
                    start[g] = pos;
                    ptr[g] = g < gv ? (unsigned long long) (uintptr_t) cache_base + (unsigned long long) res3[e] * (unsigned long long) blob
                                    : (unsigned long long) (uintptr_t) stage + (unsigned long long) (g - gv) * (unsigned long long) stride;
                    if (g >= gv) { ptr2[g - gv] = ptr[g]; start2[g - gv] = pos; src[g - gv] = mtab[e]; }
                    for (int i = 0; i < n; ++i)
                        if (ids[i] == e) { dst[pos] = i; tok[pos] = i / K; ++pos; }
                }
                start[groups] = n;
                start2[gm] = n;
                const int32_t counts[4] = {groups, n, gm, gv};
                dpct::get_in_order_queue().memset(d_plan, 0xAB, plan_words * 4).wait();
                k::resident_plan_set_stage(stage, stride, d_src);
                k::resident_plan(d_ids, n, K, d_res, NE, cache_base, nullptr, blob, d_plan, capx, nullptr, 0u, nullptr, d_err);
                k::resident_plan_set_stage(nullptr, 0, nullptr);
                dpct::get_current_device().queues_wait_and_throw();
                std::vector<int32_t> plan(plan_words);
                std::vector<unsigned long long> got_src(gm + 1);
                dpct::get_in_order_queue().memcpy(plan.data(), d_plan, plan_words * 4).wait();
                if (gm > 0) dpct::get_in_order_queue().memcpy(got_src.data(), d_src, gm * 8).wait();
                const int32_t* pdst = plan.data() + 4 + capx + 1;
                const unsigned long long* pptr = (const unsigned long long*) (plan.data() + ptr_off);
                bool ok = std::memcmp(plan.data(), counts, 4 * 4) == 0;
                ok = ok && std::memcmp(plan.data() + 4, start.data(), (size_t) (groups + 1) * 4) == 0;
                ok = ok && std::memcmp(pdst, dst.data(), (size_t) n * 4) == 0;
                ok = ok && std::memcmp(pdst + capx, tok.data(), (size_t) n * 4) == 0;
                ok = ok && std::memcmp(pptr, ptr.data(), (size_t) groups * 8) == 0;
                ok = ok && (gm == 0 || std::memcmp(pptr + capx, ptr2.data(), (size_t) gm * 8) == 0);
                ok = ok && std::memcmp(plan.data() + ptr_off + 4 * capx, start2.data(), (size_t) (gm + 1) * 4) == 0;
                ok = ok && (gm == 0 || std::memcmp(got_src.data(), src.data(), (size_t) gm * 8) == 0);
                ++cases;
                if (!ok) {
                    std::printf("    *** stage n=%d pool=%d (%d VRAM + %d mirrored groups): the plan differs from the host replay ***\n",
                                n, pool, gv, gm);
                    ++bad;
                }
            }
        // STRATA_MIRROR_RING: 400 calls in a row through a ring of 256 slots: within a call no two mirrored groups share a
        // ring slot; a group copies (src = its mirror address) exactly when its expert was not in the ring before the call,
        // and then the ring says it is there now; a group that does not copy reads the slot the ring already held it in
        {
            std::string rerr;
            const long long rstride = 4096;
            const int R = 256;
            if (!k::mirror_ring_create(R, rstride, rerr)) { std::printf("    *** ring: %s ***\n", rerr.c_str()); ++bad; }
            std::vector<int> where(NE, -1);   // the host's view: expert -> ring slot
            unsigned long long base = 0;
            int ring_bad = 0, hits = 0, inserts = 0;
            const unsigned long long st_lo = (unsigned long long) (uintptr_t) stage, st_hi = st_lo + (unsigned long long) capx * rstride;
            for (int call = 0; call < 400 && ring_bad == 0; ++call) {
                const int n = 1 + (int) (rng() % 60);
                std::vector<int32_t> ids(n);
                for (int i = 0; i < n; ++i) ids[i] = (int32_t) (rng() % (call < 200 ? 96 : 512));
                dpct::get_in_order_queue().memcpy(d_ids, ids.data(), n * 4).wait();
                k::resident_plan_set_stage(stage, rstride, d_src, true);
                k::resident_plan(d_ids, n, K, d_res, NE, cache_base, nullptr, blob, d_plan, capx, nullptr, 0u, nullptr, d_err);
                k::resident_plan_set_stage(nullptr, 0, nullptr);
                dpct::get_current_device().queues_wait_and_throw();
                std::vector<int32_t> plan(plan_words);
                dpct::get_in_order_queue().memcpy(plan.data(), d_plan, plan_words * 4).wait();
                const int gm = plan[2], gv = plan[3];
                std::vector<unsigned long long> got_src(gm + 1);
                if (gm > 0) dpct::get_in_order_queue().memcpy(got_src.data(), d_src, gm * 8).wait();
                const unsigned long long* pptr = (const unsigned long long*) (plan.data() + ptr_off);
                const int32_t* pdst = plan.data() + 4 + capx + 1;
                const int32_t* pstart = plan.data() + 4;
                if (base == 0) {   // the ring is new: the first call that inserts takes slots from 0, the lowest is its base
                    unsigned long long lo = ~0ull;
                    for (int j = 0; j < gm; ++j)
                        if (got_src[j] != 0 && (pptr[capx + j] < st_lo || pptr[capx + j] >= st_hi)) lo = std::min(lo, pptr[capx + j]);
                    if (lo != ~0ull) base = lo;
                }
                std::vector<int> nw = where, used;
                for (int j = 0; j < gm && ring_bad == 0; ++j) {
                    const int e = ids[pdst[pstart[gv + j]]];   // the group's expert: its first entry's id
                    const unsigned long long a = pptr[capx + j];
                    if (res3[e] >= 0 || pptr[gv + j] != a) { ++ring_bad; break; }
                    int slot = -1;
                    if (got_src[j] == 0) {   // a hit: the ring held it there before this call
                        ++hits;
                        slot = where[e];
                        if (slot < 0 || a != base + (unsigned long long) slot * rstride) { ++ring_bad; break; }
                    } else {
                        if (got_src[j] != mtab[e] || where[e] >= 0) { ++ring_bad; break; }   // copies its mirror; was not in the ring
                        if (a >= st_lo && a < st_hi) continue;                                 // no ring slot left: staging
                        if (a < base || (a - base) % rstride != 0 || (a - base) / rstride >= (unsigned long long) R) { ++ring_bad; break; }
                        slot = (int) ((a - base) / rstride);
                        for (int x = 0; x < NE; ++x) if (nw[x] == slot) nw[x] = -1;   // evicted
                        nw[e] = slot;
                        ++inserts;
                    }
                    if (std::find(used.begin(), used.end(), slot) != used.end()) { ++ring_bad; break; }
                    used.push_back(slot);
                }
                where = nw;
            }
            ++cases;
            if (ring_bad || hits == 0 || inserts == 0) {
                std::printf("    *** ring: %d inconsistencies (%d hits, %d inserts) ***\n", ring_bad, hits, inserts);
                ++bad;
            } else {
                std::printf("ring: 400 plans, %d hits, %d inserts, consistent\n", hits, inserts);
            }
        }
        k::resident_plan_set_mirror(nullptr, nullptr, nullptr, 0);
        sycl::free(d_tab, dpct::get_in_order_queue());
        sycl::free(d_src, dpct::get_in_order_queue());
    }

    std::printf("resident_plan: %d cases, %d failures\n", cases, bad);
    if (bad) return 1;
    std::printf("resident_plan_parity OK\n");
    return 0;
}
