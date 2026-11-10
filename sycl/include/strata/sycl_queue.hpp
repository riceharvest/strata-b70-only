// include/strata/sycl_queue.hpp - the SYCL port's one addition to the engine's API surface.
//
// Every launcher takes `void* stream`, a cudaStream_t where null means the default stream. dpct migrates the
// cast to `(dpct::queue_ptr) stream` and dereferences it, so a null stream is a null sycl::queue* and a crash.
// q_of() is that cast with CUDA's null-stream meaning restored: the default in-order queue.
#pragma once
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdlib>

namespace strata {
inline sycl::queue* q_of(const void* stream) {
    if (stream) return (sycl::queue*) stream;
    static thread_local unsigned int tls_dev = ~0u;
    static thread_local sycl::queue* tls_q = nullptr;
    const unsigned int cur = dpct::dev_mgr::instance().current_device_id();
    if (cur != tls_dev || tls_q == nullptr) {
        tls_dev = cur;
        tls_q = &dpct::get_in_order_queue();
    }
    return tls_q;
}
}  // namespace strata

namespace strata {
// A large device fill as compute kernels of at most `chunk` bytes, each waited for.  On the Arc Pro B70 (xe driver) one
// queue.memset of many GiB runs on the blitter engine and times out ("Engine memory CAT error", GT reset); kernels do not.
// Small fills (<= chunk) and non-multiple-of-8 tails fall back to memset.
inline void big_fill_zero(sycl::queue& q, void* p, size_t bytes, size_t chunk = (size_t)256 << 20) {
    static const bool plain = [] { const char* v = std::getenv("STRATA_CHUNKED_FILL"); return v && v[0] == '0'; }();
    if (plain) { q.memset(p, 0, bytes); q.wait(); return; }
    uint8_t* b = (uint8_t*)p;
    size_t off = 0;
    while (off < bytes) {
        size_t n = bytes - off < chunk ? bytes - off : chunk;
        if (n >= (1u << 20) && ((uintptr_t)(b + off) & 7) == 0) {
            size_t w = n / 8;
            uint64_t* d = (uint64_t*)(b + off);
            q.parallel_for(sycl::range<1>(w), [=](sycl::id<1> i) { d[i] = 0; });
            if (n & 7) q.memset(b + off + w * 8, 0, n & 7);
        } else {
            q.memset(b + off, 0, n);
        }
        q.wait();
        off += n;
    }
}
}  // namespace strata

// ---- host memory a running kernel polls for the host's stores (the doorbell flags, the CPU experts' result rows).
// On an Arc A750 (Alchemist, i915) a kernel never sees a host store to ordinary host USM, with or without
// system-scope atomics: every ring wait runs to its bound and the window reads stale rows (NaN logits). Host memory
// from zeMemAllocHost with ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED does work there (sycl/probe/doorbell.cpp mode 2,
// "HANDSHAKE OK"), so those buffers come from it when the Level Zero headers are present (libze-dev), else from
// sycl::malloc_host. STRATA_HOST_UNCACHED=0 turns it off.
#if defined(__has_include)
#if __has_include(<level_zero/ze_api.h>) && __has_include(<sycl/ext/oneapi/backend/level_zero.hpp>)
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#define STRATA_HAVE_ZE 1
#endif
#endif
#include <cstdlib>
#include <mutex>
#include <unordered_set>
namespace strata {
namespace detail {
inline std::unordered_set<void*>& uncached_set() { static std::unordered_set<void*> s; return s; }
inline std::mutex& uncached_mu() { static std::mutex m; return m; }
}  // namespace detail
inline void* host_malloc_polled(size_t bytes, sycl::queue& q) {
#ifdef STRATA_HAVE_ZE
    static const bool on = [] { const char* v = std::getenv("STRATA_HOST_UNCACHED"); return !(v && v[0] == '0'); }();
    if (on && q.get_backend() == sycl::backend::ext_oneapi_level_zero) {
        auto ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        ze_host_mem_alloc_desc_t d{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED};
        void* p = nullptr;
        if (zeMemAllocHost(ctx, &d, bytes, 64, &p) == ZE_RESULT_SUCCESS && p != nullptr) {
            std::lock_guard<std::mutex> lk(detail::uncached_mu());
            detail::uncached_set().insert(p);
            return p;
        }
    }
#endif
    return sycl::malloc_host(bytes, q);
}
inline void host_free_polled(void* p, sycl::queue& q) {
    if (p == nullptr) return;
#ifdef STRATA_HAVE_ZE
    {
        std::lock_guard<std::mutex> lk(detail::uncached_mu());
        if (detail::uncached_set().erase(p)) {
            zeMemFree(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context()), p);
            return;
        }
    }
#endif
    sycl::free(p, q);
}
}  // namespace strata

// The kernel driver of the Intel GPU ("xe", "i915", or "" when unknown), from sysfs.  The driver decides what the GPU
// may read: an Arc A750 (i915) reads ordinary host memory a kernel is handed; an Arc Pro B70 (xe) page-faults on it
// (ccs0, FaultType 0, address in the CPU's mmap range) and the card times the job out.
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <utility>
namespace strata {
inline const std::string& intel_gpu_driver() {
    static const std::string drv = [] {
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator("/sys/class/drm", ec)) {
            const std::string n = e.path().filename().string();
            if (n.rfind("card", 0) != 0 || n.find('-') != std::string::npos) continue;
            std::ifstream vf(e.path() / "device" / "vendor");
            std::string vendor;
            if (!(vf >> vendor) || vendor != "0x8086") continue;
            const auto d = std::filesystem::read_symlink(e.path() / "device" / "driver", ec);
            if (!ec) return d.filename().string();
        }
        return std::string();
    }();
    return drv;
}
}  // namespace strata

namespace strata {
// Which Intel GPU generation a SYCL device is, from the device itself (so it also works on Windows / OpenCL, where
// intel_gpu_driver() has no sysfs to read): the architecture enum when the runtime knows it, else the marketing name
// ("Intel(R) Arc(TM) Pro B70", "... A750 ..."). Battlemage = Arc B-series / Pro B-series (B570 B580 B50 B60 B65 B70).
enum class IntelGpuGen { Unknown, Alchemist, Battlemage };
inline IntelGpuGen intel_gpu_gen(const sycl::device& d) {
    if (!d.is_gpu()) return IntelGpuGen::Unknown;
    try {
        {
            namespace exp = sycl::ext::oneapi::experimental;
            const auto a = d.get_info<exp::info::device::architecture>();
            if (a == exp::architecture::intel_gpu_bmg_g21 || a == exp::architecture::intel_gpu_bmg_g31)
                return IntelGpuGen::Battlemage;
            if (a == exp::architecture::intel_gpu_acm_g10 || a == exp::architecture::intel_gpu_acm_g11 ||
                a == exp::architecture::intel_gpu_acm_g12)
                return IntelGpuGen::Alchemist;
        }
    } catch (...) {
    }
    std::string n;
    try { n = d.get_info<sycl::info::device::name>(); } catch (...) { return IntelGpuGen::Unknown; }
    if (n.find("Intel") == std::string::npos) return IntelGpuGen::Unknown;
    // "Intel(R) Graphics [0xe223]": drivers that print the PCI id instead of a marketing name (the B70 on Linux)
    if (const size_t b = n.find("[0x"); b != std::string::npos) {
        const unsigned long id = std::strtoul(n.c_str() + b + 1, nullptr, 16);
        if (id >= 0xE200 && id <= 0xE2FF) return IntelGpuGen::Battlemage;                 // BMG-G21 / G31
        if ((id >= 0x5690 && id <= 0x56BF) || id == 0x56C0 || id == 0x56C1) return IntelGpuGen::Alchemist;   // DG2
    }
    if (n.find("Battlemage") != std::string::npos) return IntelGpuGen::Battlemage;
    if (n.find("Alchemist") != std::string::npos) return IntelGpuGen::Alchemist;
    // "Arc(TM) Pro B70", "Arc(TM) B580", "Arc(TM) A750": a letter and 2-3 digits, as a whole word
    for (size_t i = 0; i + 2 < n.size(); ++i) {
        if ((i > 0 && n[i - 1] != ' ') || (n[i] != 'A' && n[i] != 'B')) continue;
        size_t j = i + 1;
        while (j < n.size() && n[j] >= '0' && n[j] <= '9') ++j;
        if (j - i - 1 < 2 || j - i - 1 > 3 || (j < n.size() && n[j] != ' ' && n[j] != '(')) continue;
        return n[i] == 'B' ? IntelGpuGen::Battlemage : IntelGpuGen::Alchemist;
    }
    return IntelGpuGen::Unknown;
}

// The B70's run-time switches (this branch runs on an Arc Pro B70 under xe and nothing else): the device-built verify
// plan, no host handshakes in it (xe does not show the GPU the host's flag stores), and the host copy threads, so
// `build-sycl-aot/strata` run by hand behaves as it does under sycl/serve/strata-sycl.sh.  A variable that is set
// wins, "0" included (STRATA_VERIFY_NO_HOST=0 turns the switch off); STRATA_B70_DEFAULTS=0 skips the table.
inline void apply_b70_defaults() {
    if (const char* off = std::getenv("STRATA_B70_DEFAULTS"); off != nullptr && off[0] == '0') return;
    const unsigned hw = std::thread::hardware_concurrency();
    const std::string stager = std::to_string(hw ? std::min(12u, hw) : 12u);
    const std::pair<const char*, std::string> defaults[] = {
        {"STRATA_VERIFY_DEVICE_PLAN", "1"},
        {"STRATA_VERIFY_NO_HOST", "1"},
        {"STRATA_STAGER_THREADS", stager},
        // 0.1.54 (2026-10-30, Flash-Next IQ2_XS, a 118K prompt, ~298 tokens): the long-context decode top-k and the
        // cost-aware MTP window bring decode back to 48.6 tok/s from 42.6 (b70 before the merge: 47.4 / 50.2), the
        // greedy answers byte-identical
        {"STRATA_SELECT_LONG", "1"},
        {"STRATA_SPEC_COST", "1"},
        // `--prefill auto` may take chunks up to 16,384 tokens (118K prompt: GPU timeline 78.6 s at 4,096, 68.5 s at
        // 16,384; 32,768 is no faster and borrows twice the cache)
        {"STRATA_PREFILL_AUTO_MAX", "16384"},
        // the port's own B70 decode kernels, measured but left opt-in until now (docs/INTEL.md): IQ_DECODE_FAST is bitwise
        // the default expert kernels (+7.7% decode on 2x B70), MMVQ_KREG +19.5% (another summation order); on one B70 at a
        // 118K prompt the two together decode 63.0 / 63.3 tok/s against 60.2 / 60.4 (2026-10-30).  HC_Q8: the Q8_0
        // hyper-connection read in the decode window, -5.3% per window on 2x B70 (the BF16 copy stays for the prompt).
        {"STRATA_IQ_DECODE_FAST", "1"},
        {"STRATA_MMVQ_KREG", "1"},
        {"STRATA_HC_Q8", "1"},
        // NOT defaults: STRATA_XE2_KQ / STRATA_XE2_IQ / STRATA_HC_Q8_DOWN2 (the decode window's launches for the UD-IQ1_S
        // pack, native_mmvq.dp.cpp / iq_kernels.dp.cpp / fused_gr.dp.cpp).  Their parity programs pass, but in the engine
        // (2026-11-10, one B70, ABBA at a fixed expert cache, STRATA_SPEC_COST=0) they were no faster (reference prompts
        // 114.2 / 95.6 and 120.9 / 95.4 tok/s on, 113.0 / 95.1 and 117.7 / 98.8 off) and two runs with them on gave two
        // different greedy texts for one prompt, while the two runs with them off were identical.  Opt-in until the
        // one that is not deterministic is found.
        // a turn checkpoint (118 MB of DeltaNet state) moved to the host off the critical path: card-to-card into one VRAM
        // stage, then a background copy, into a pooled vector (two kept ready: a fresh one faulted in 59 ms of pages),
        // the 37 small pieces as one copy.  An agent's small turn at 203K: 195-236 -> 104-149 ms (the save 77.5 -> 5 ms),
        // 2026-10-31.  Not STRATA_CKPT_VRAM_SLOTS / STRATA_PARK_ASYNC: they hold VRAM down to 1.5 GB free.
        {"STRATA_CKPT_ASYNC", "1"},
        {"STRATA_CKPT_POOL", "1"},
        {"STRATA_CKPT_POOL_READY", "2"},
        {"STRATA_CKPT_GATHER", "1"},
        // sampled requests: the MTP drafter samples its own distribution q and the verifier accepts a draft with
        // probability min(1, p/q), redrawing a rejection from (p - q)+ (the `top` gate: exact rejection sampling; the
        // default pick gate needs --pcie-frac 0 and no STRATA_SPEC_COST, else it keeps an older, inexact rule).  Six
        // agent-shaped turns (thinking on, temperature 1.0, 1,000 tokens each, expert cache held at 16,000 slots):
        // 82.1 / 82.9 / 83.5 tok/s off, 88.1 / 88.2 on (+5.5%), 2026-11-01.  Greedy requests are unchanged.
        {"STRATA_SPEC_PROB", "1"},
        {"STRATA_SPEC_COUPLED", "1"},
        {"STRATA_SPEC_PROB_GATE", "top"},
    };
    std::string applied;
    for (const auto& [k, v] : defaults) {
        if (std::getenv(k) != nullptr) continue;
        setenv(k, v.c_str(), 0);
        applied += (applied.empty() ? "" : " ") + std::string(k) + "=" + v;
    }
    if (!applied.empty())
        std::fprintf(stderr, "strata: B70 defaults: %s (set one to 0 to turn it off; STRATA_B70_DEFAULTS=0: none)\n", applied.c_str());
}
// The Battlemage decode controls of PR #4185 (glorysdj): unset = `bm_default` for a Battlemage device and off for every other
// device (Arc A-series included); a set value turns the control on ("1") or off ("0") whatever the device, as before. The
// controls with bm_default=false measured neutral or slower on 2x Arc Pro B65 (IQ3_S): they stay opt-in.
inline bool bm_default_on(const char* name, const sycl::device& d, bool bm_default = true) {
    const char* v = std::getenv(name);
    if (v != nullptr && v[0] != '\0') return v[0] != '0';
    return bm_default && intel_gpu_gen(d) == IntelGpuGen::Battlemage;
}
}  // namespace strata

#include <cstdint>
namespace strata {
// Does every page of a big device allocation keep its own bytes?  On an Arc Pro B70 (xe) a 22 GiB expert-cache arena came
// back with two of its 2 MiB pages mapped onto the same memory (a write to the page at +2 MiB showed up at +1022 MiB; the
// cache slots there then held other experts' bytes and the prompt came out NaN).  The check writes a tag per 64 KiB of
// the range, then reads them back in a second kernel: it returns the number of 64 KiB blocks that do not hold their own tag
// (0 = clean) and the first such block's offset.  The range's contents are destroyed.
inline uint64_t arena_alias_check(sycl::queue& q, void* base, size_t bytes, size_t* first_bad = nullptr) {
    constexpr size_t kBlock = 64 << 10;
    const size_t n = bytes / kBlock;
    if (n == 0) return 0;
    uint8_t* b = (uint8_t*) base;
    uint64_t* res = sycl::malloc_device<uint64_t>(2, q);
    uint64_t init[2] = {0, ~0ull};
    q.memcpy(res, init, sizeof init).wait();
    // chunked so that no single launch is huge (an xe job past a few seconds is timed out)
    const size_t chunk = (size_t) 1 << 20;
    for (size_t c0 = 0; c0 < n; c0 += chunk) {
        const size_t cn = n - c0 < chunk ? n - c0 : chunk;
        q.parallel_for(sycl::range<1>(cn), [=](sycl::id<1> i) {
            const size_t blk = c0 + i[0];
            *(volatile uint64_t*) (b + blk * kBlock) = (blk + 1) * 0x9E3779B97F4A7C15ull;
        });
        q.wait();
    }
    for (size_t c0 = 0; c0 < n; c0 += chunk) {
        const size_t cn = n - c0 < chunk ? n - c0 : chunk;
        q.parallel_for(sycl::range<1>(cn), [=](sycl::id<1> i) {
            const size_t blk = c0 + i[0];
            if (*(volatile uint64_t*) (b + blk * kBlock) != (blk + 1) * 0x9E3779B97F4A7C15ull) {
                sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device> bad(res[0]);
                sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device> lo(res[1]);
                bad.fetch_add(1);
                lo.fetch_min((uint64_t) blk * kBlock);
            }
        });
        q.wait();
    }
    uint64_t out[2] = {0, 0};
    q.memcpy(out, res, sizeof out).wait();
    sycl::free(res, q);
    if (first_bad) *first_bad = (size_t) out[1];
    return out[0];
}
}  // namespace strata

namespace strata {
inline bool arena_alias_guard_enabled() {
    static const bool on = [] { const char* v = std::getenv("STRATA_ARENA_ALIAS_CHECK"); return v == nullptr || v[0] != '0'; }();
    return on;
}
}  // namespace strata

#include <vector>
#include <map>
#include <mutex>
namespace strata {
// Every big device allocation (base -> size), so device_offset_end can find the allocation that holds an address.  The
// CPU expert pool's handshake and oneMKL's GEMM both care where, within its allocation, an operand lies.
inline std::map<const void*, size_t>& device_alloc_map() {
    static std::map<const void*, size_t> m;
    return m;
}
inline std::mutex& device_alloc_mutex() {
    static std::mutex m;
    return m;
}
inline void device_alloc_register(const void* p, size_t bytes) {
    if (p == nullptr || bytes == 0) return;
    std::lock_guard<std::mutex> lk(device_alloc_mutex());
    device_alloc_map()[p] = bytes;
}
// Drop a base when its allocation is freed, so a later allocation that reuses the address is not mis-attributed by
// device_offset_end.
inline void device_alloc_unregister(const void* p) {
    if (p == nullptr) return;
    std::lock_guard<std::mutex> lk(device_alloc_mutex());
    device_alloc_map().erase(p);
}
// How far into its device allocation the byte range [p, p + bytes) ends; 0 when p is not in one we know.  oneMKL's GEMM
// reads the last, partial tile of an operand from the wrong place when it lies more than 4 GiB in (Arc A770, oneMKL
// 2026.1), so the GEMM stages any operand this reports > 4 GiB for.  An operand in an allocation we did not register
// (small cudaMalloc'd buffers) is reported 0: those can never reach 4 GiB.
inline size_t device_offset_end(const void* p, size_t bytes) {
    if (p == nullptr) return 0;
    std::lock_guard<std::mutex> lk(device_alloc_mutex());
    auto& m = device_alloc_map();
    auto it = m.upper_bound(p);
    if (it == m.begin()) return 0;
    --it;
    const uint8_t* base = (const uint8_t*) it->first;
    const uint8_t* end = (const uint8_t*) p + bytes;
    if (end > base + it->second) return 0;
    return (size_t) (end - base);
}
// sycl::malloc_device for a big allocation, checked for aliased pages (arena_alias_check) and, when it has some, allocated
// again behind a growing spacer that moves it.  Small allocations and STRATA_ARENA_ALIAS_CHECK=0 go straight to
// sycl::malloc_device.  The memory comes back uninitialised, as from sycl::malloc_device.
inline void* malloc_device_guarded(size_t bytes, sycl::queue& q, const char* what = "device buffer") {
    void* p = sycl::malloc_device(bytes, q);
    if (p == nullptr || bytes < ((size_t) 32 << 20) || !arena_alias_guard_enabled()) {
        device_alloc_register(p, bytes);
        return p;
    }
    std::vector<void*> spacers;
    bool clean = false;
    for (int attempt = 0; attempt < 8 && p != nullptr; ++attempt) {
        size_t first = 0;
        const uint64_t bad = arena_alias_check(q, p, bytes, &first);
        if (bad == 0) {
            clean = true;
            if (attempt > 0)
                std::fprintf(stderr, "strata: %s (%.2f GiB) is clean after %d move(s)\n", what, (double) bytes / 1073741824.0, attempt);
            break;
        }
        std::fprintf(stderr,
                     "strata: WARNING: %llu 64-KiB blocks of the %s (%.2f GiB) do not keep their own bytes (first at +%zu MiB): the GPU's "
                     "page table maps them onto other pages; allocating it again behind a %d MiB spacer\n",
                     (unsigned long long) bad, what, (double) bytes / 1073741824.0, first >> 20, 2 * (attempt + 1));
        sycl::free(p, q);
        spacers.push_back(sycl::malloc_device((size_t) (2 * (attempt + 1)) << 20, q));
        p = sycl::malloc_device(bytes, q);
    }
    for (void* sp : spacers) if (sp != nullptr) sycl::free(sp, q);
    if (p != nullptr && !clean)
        std::fprintf(stderr, "strata: WARNING: the %s still has aliased pages after 8 tries (STRATA_ARENA_ALIAS_CHECK=0 skips the check)\n", what);
    device_alloc_register(p, bytes);
    return p;
}
}  // namespace strata
