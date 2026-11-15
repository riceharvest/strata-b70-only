#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/core/native_dense.hpp"
#include <cstdlib>
#include "strata/core/weights.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/fused_gr_q8.hpp"
#include "strata/core/geometry.hpp"
#include "strata/platform/memory.hpp"
#if defined(STRATA_B70_LIB)
#include "strata/b70/pack.hpp"     // STRATA_B70_PACK: projection matrices from the card's own weight format
#endif

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <set>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::core {
namespace {
// `load` makes one cudaMalloc per projection matrix, and the driver backs each call with 2 MiB granules
// (measured on the RTX 5060: 300 buffers of 6.73 MiB asked 2019 MiB and cost 2400).  So a stage's real
// footprint is the granule-rounded sum, not the payload sum.  This rule - a granule for every matrix of a
// MiB or more, the payload for the rest - prices the iq3_s pack's 300 matrices at 2255 MiB where they
// really take 2238 (+0.8%).  It over-prices, which is the safe side.  (Pricing every matrix at a granule is
// far worse, 2378 MiB: the driver packs the sub-MiB matrices into the slack of the rounded ones.)
constexpr uint64_t kAllocGranule = 2ull << 20;
constexpr uint64_t kSmallAlloc = 1ull << 20;
uint64_t alloc_bytes(uint64_t bytes) {
    return bytes < kSmallAlloc ? bytes : (bytes + kAllocGranule - 1) / kAllocGranule * kAllocGranule;
}
int g_layer_lb = -1, g_layer_le = -1;   // set_layer_range; -1: every layer
bool in_range(const std::string& name) {
    if (g_layer_lb < 0 || name.rfind("blk.", 0) != 0) return true;
    const int l = std::atoi(name.c_str() + 4);
    return l >= g_layer_lb && l < g_layer_le;
}
// S23 experiment (STRATA_HC_Q8=1): the hyper-connection projections' Q8_0 bytes for the verify read
bool hc_q8_env() {
    static const bool on = [] { const char* v = std::getenv("STRATA_HC_Q8"); return v != nullptr && v[0] == '1'; }();
    return on;
}
bool hc_q8_name(const std::string& name) {
    if (name == "output_hc_down.weight" || name == "output_hc_up.weight") return true;   // S25: the final mixer too
    if (name.rfind("blk.", 0) != 0) return false;
    static const char* suffixes[] = {".hc_attn_down.weight", ".hc_attn_up.weight", ".hc_attn_inject.weight",
                                     ".hc_ffn_down.weight", ".hc_ffn_up.weight", ".hc_ffn_inject.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
// S25: the GGUF keeps the inject rows as F32 - quantized here to Q8_0 (ggml's reference rounding) so the Q8_0 read
// covers every hc projection
uint16_t half_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) {                                   // subnormal or zero
        if (e < -10) return (uint16_t) sign;
        m |= 0x800000u;
        const uint32_t shift = (uint32_t) (14 - e);
        uint32_t h = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) ++h;
        return (uint16_t) (sign | h);
    }
    if (e >= 31) return (uint16_t) (sign | 0x7c00u);
    uint32_t h = ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) (sign | h);
}
std::vector<uint8_t> q8_0_of(const float* x, uint64_t n) {
    std::vector<uint8_t> out((size_t) (n / 32 * 34));
    for (uint64_t b = 0; b < n / 32; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = (std::max)(amax, std::fabs(x[b * 32 + j]));
        const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        uint8_t* o = out.data() + b * 34;
        const uint16_t hb = half_bits(d);
        std::memcpy(o, &hb, 2);
        for (int j = 0; j < 32; ++j) o[2 + j] = (uint8_t) (int8_t) std::lround(x[b * 32 + j] * id);
    }
    return out;
}
// A bf16 matrix (the pack file's values) as Q8_0, ggml's reference rounding; the blocks are independent, so a few threads
std::vector<uint8_t> q8_0_of_bf16(const uint16_t* x, uint64_t n) {
    std::vector<uint8_t> out((size_t) (n / 32 * 34));
    const uint64_t nb = n / 32;
    auto work = [&](uint64_t b0, uint64_t b1) {
        for (uint64_t b = b0; b < b1; ++b) {
            float v[32], amax = 0.0f;
            for (int j = 0; j < 32; ++j) {
                const uint32_t u = (uint32_t) x[b * 32 + j] << 16;
                std::memcpy(&v[j], &u, 4);
                amax = (std::max)(amax, std::fabs(v[j]));
            }
            const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
            uint8_t* o = out.data() + b * 34;
            const uint16_t hb = half_bits(d);
            std::memcpy(o, &hb, 2);
            for (int j = 0; j < 32; ++j) o[2 + j] = (uint8_t) (int8_t) std::lround(v[j] * id);
        }
    };
    const unsigned nt = (std::min)(8u, (std::max)(1u, std::thread::hardware_concurrency()));
    std::vector<std::thread> pool;
    for (unsigned i = 0; i < nt; ++i) pool.emplace_back(work, nb * i / nt, nb * (i + 1) / nt);
    for (auto& th : pool) th.join();
    return out;
}
bool eligible(const strata::TensorInfo& tensor, bool include_ple_key) {
    const auto& name = tensor.name;
    if (name.rfind("blk.", 0) != 0) return false;
    // Match the native PLE kernel: Q2_0, IQ3_XXS, IQ4_XS and Q8_0 (UD-Q4_K_XL). Other keys retain the packed BF16
    // fallback.
    if (name == "blk.1.ple_key.weight")
        return include_ple_key && (tensor.type == 42 || tensor.type == 18 || tensor.type == 23 || tensor.type == 8);
    static const char* suffixes[] = {".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_output.weight",
        ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
#if defined(STRATA_B70_LIB)
// STRATA_B70_PACK=<file> (the engine's --b70-pack; Settings: "b70_pack"): a pack (strata/b70/pack.hpp, written by
// tools/b70_pack.py) that names projection matrices of this model.  A matrix the pack holds with the GGUF's shape is uploaded
// from it instead of from the GGUF and carries the pack type, so its decode products are the card library's stream product
// (native_mmvq) and its prompt products the library's dequantize + the FP16 GEMM (Gemm::native).  Only the projections that
// reach the weights through those two calls are taken: the shared expert's matrices and the PLE key have kernels of their own
// that read the GGUF's blocks.  Unset: nothing of this runs.
const strata::b70::pack::File* b70_pack(std::string& err) {
    static strata::b70::pack::File file;
    static std::string open_err;
    static const int state = [] {
        const char* path = std::getenv("STRATA_B70_PACK");
        if (path == nullptr || path[0] == 0) return 0;
        return file.open(path, open_err) ? 1 : -1;
    }();
    if (state < 0) err = "STRATA_B70_PACK: " + (open_err.empty() ? std::string("no matrices in the pack") : open_err);
    return state > 0 ? &file : nullptr;
}
bool b70_pack_name(const std::string& name) {
    static const char* suffixes[] = {".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_output.weight"};
    if (name.rfind("blk.", 0) != 0) return false;
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
// the pack's entry for a projection of this shape, or nullptr (no pack, not a pack name, another shape)
const strata::b70::pack::Entry* b70_pack_entry(const strata::b70::pack::File* pack, const std::string& name, int64_t n_in,
                                               int64_t n_out) {
    return pack != nullptr && b70_pack_name(name) ? pack->find(name, n_in, n_out) : nullptr;
}
#endif
struct DeviceFree {
    void
    operator()(void *p) const {
        if (p) sycl::free(p, dpct::get_in_order_queue());
    }
};
using DevicePtr = std::unique_ptr<void, DeviceFree>;
struct Pending {
    WeightRef* ref;
    int type;
    uint64_t bytes;
    DevicePtr data;
    DevicePtr packed;   // STRATA_Q8_PACKED=1 copy, or null
};
}

// the shared header declares the static member (b60aab1b); this port only needs the env switch
bool NativeDense::hc_q8_requested() { return hc_q8_env(); }

bool NativeDense::served_names(const std::vector<std::string>& shards, bool include_ple_key,
                               std::set<std::string>& out, std::string& err) {
    try {
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& tensor : gguf.tensors())
                if (eligible(tensor, include_ple_key) && strata::kernels::native_mmvq_supported(tensor.type) &&
                    tensor.shape.size() == 2)
                    out.insert(tensor.name);
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}

void NativeDense::set_layer_range(int lb, int le) { g_layer_lb = lb; g_layer_le = le; }
// The byte walk `load` does for the layers [lb, le), without the allocations: the same filters in the same order,
// reading the GGUF headers and the canonical table only, so it needs no device.  A later stage's projections are not
// on the card when the split search prices that stage, and `load` runs one cudaMalloc per matrix (#1238).  The two
// have to stay in step.  A repeated name is skipped rather than refused (`load` refuses it).
bool NativeDense::weight_bytes_for(const std::vector<std::string>& shards, WeightTable& table, bool include_ple_key,
                                   int64_t lb, int64_t le, uint64_t& out, std::string& err) {
    out = 0;
    try {
        std::set<std::string> seen;
        uint64_t total = 0;
#if defined(STRATA_B70_LIB)
        const strata::b70::pack::File* pack = b70_pack(err);
        if (!err.empty()) return false;
#endif
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& tensor : gguf.tensors()) {
                if (!eligible(tensor, include_ple_key)) continue;
                if (tensor.name.rfind("blk.", 0) == 0) {
                    const long l = std::strtol(tensor.name.c_str() + 4, nullptr, 10);
                    if (l < lb || l >= le) continue;
                }
                if (!seen.insert(tensor.name).second) continue;
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name;
                    return false;
                }
                const auto& ref = found->second;
                if (!strata::kernels::native_mmvq_supported(tensor.type)) continue;
                if (tensor.name == "blk.1.ple_key.weight" && !ref.quantized()) continue;
                if (!ref.quantized() || tensor.shape.size() != 2 ||
                    ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 || ref.ne1 > INT_MAX ||
                    tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name;
                    return false;
                }
#if defined(STRATA_B70_LIB)
                if (const auto* pe = b70_pack_entry(pack, tensor.name, ref.ne0, ref.ne1)) {   // as `load` takes it
                    total += alloc_bytes(pe->bytes);
                    continue;
                }
#endif
                total += alloc_bytes(strata::kernels::native_mmvq_weight_bytes(tensor.type, (int) ref.ne0, (int) ref.ne1));
            }
        }
        out = total;
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}
bool NativeDense::keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip,
                                           std::string& err) {
    const std::string key = "blk.1.ple_key.weight";
    if (!skip.count(key)) return true;
    int code_bits = -1;
    if (!WeightTable::index_code_bits(pack_dir, key, code_bits, err)) return false;
    if (code_bits == 0) skip.erase(key);
    return true;
}

// STRATA_HC_Q8 without a bf16 copy: the two dequantization scratches (one bf16 matrix each) of `hc_bf16_view`, per device,
// allocated by `load` - never on first use, so a graph capture cannot meet the allocation
static void* g_hc_scratch[64][2][2] = {};     // [device][class][slot]
static dpct::event_ptr g_hc_ev[64][2] =
    {}; // [device][slot]: the prompt path's last reader of the slot
static bool g_hc_ev_set[64][2] = {};

NativeDense::~NativeDense() {
    for (auto& dev : g_hc_scratch)
        for (auto& cl : dev)
            for (void*& b : cl)
                if (b != nullptr && std::find(weights_.begin(), weights_.end(), b) != weights_.end()) b = nullptr;
    for (const void* p : packed_keys_) strata::kernels::native_q8_0_packed_unregister(p);
    if (scratch_) sycl::free(scratch_, dpct::get_in_order_queue());
    for (void *p : weights_)
        DPCT_CHECK_ERROR(sycl::free(p, dpct::get_in_order_queue()));
}

bool NativeDense::load(const std::vector<std::string> &shards,
                       WeightTable &table, std::string &err,
                       bool include_ple_key, int64_t layer_lo,
                       int64_t layer_hi) try {
    auto outside = [&](const std::string& name) {   // a blk.<l>. tensor of another stage's layers
        if (layer_hi < 0 || name.rfind("blk.", 0) != 0) return false;
        const long l = std::strtol(name.c_str() + 4, nullptr, 10);
        return l < layer_lo || l >= layer_hi;
    };
    if (scratch_ || !weights_.empty()) { err = "native dense: already loaded"; return false; }
    if (shards.empty()) { err = "native dense: at least one GGUF shard is required"; return false; }
    try {
        std::vector<Pending> pending;
        std::set<std::string> seen;
        int max_in = 0;
        uint64_t total = 0, hc_q8_bytes = 0, allocated = 0;
        uint64_t split_count = 0, split_tensors = 0;
        std::set<uint64_t> split_numbers;
        bool have_architecture = false;
#if defined(STRATA_B70_LIB)
        const strata::b70::pack::File* pack = b70_pack(err);
        if (!err.empty()) return false;
        uint64_t pack_count = 0, pack_bytes = 0;
#endif
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            const auto* count = gguf.get("split.count");
            const auto* number = gguf.get("split.no");
            const auto* tensors = gguf.get("split.tensors.count");
            if (gguf.get("general.architecture") && (!number || number->u == 0)) {
                // splitter may copy general.architecture into every shard (Huihui abliterated re-split)
                err = strata::check_engine_architecture(gguf);
                if (!err.empty()) return false;
                have_architecture = true;
                if (count && number && tensors && number->u == 0 && count->u > 1) {
                    split_count = count->u;
                    split_tensors = tensors->u;
                }
            } else if (!have_architecture || !split_count || !count || !number || !tensors ||
                       count->u != split_count || number->u == 0 || number->u >= split_count ||
                       tensors->u != split_tensors) {
                err = "native dense: additional shard must match the architecture-validated first shard's split metadata";
                return false;
            }
            if (number && !split_numbers.insert(number->u).second) {
                err = "native dense: duplicate split shard number"; return false;
            }
            std::vector<uint64_t> offsets;
            for (const auto& tensor : gguf.tensors()) offsets.push_back(tensor.offset);
            std::sort(offsets.begin(), offsets.end());
            if (std::adjacent_find(offsets.begin(), offsets.end()) != offsets.end()) {
                err = "native dense: tensor payload offsets overlap"; return false;
            }
            // Validate every directory span, including tensors we do not upload:
            // an ignored tensor must not overlap the native matrix that follows it.
            const uint64_t payload = gguf.file_size() - gguf.data_start();
            for (const auto& tensor : gguf.tensors()) {
                int block_elements = 0, block_bytes = 0;
                uint64_t elements = 1;
                if (tensor.shape.empty() || !strata::block_geometry(tensor.type, block_elements, block_bytes) ||
                    tensor.shape[0] % (uint64_t) block_elements != 0) {
                    err = "native dense: invalid block geometry " + tensor.name; return false;
                }
                for (uint64_t dimension : tensor.shape) {
                    if (!dimension || elements > (std::numeric_limits<uint64_t>::max)() / dimension) {
                        err = "native dense: invalid tensor extent " + tensor.name; return false;
                    }
                    elements *= dimension;
                }
                const uint64_t blocks = elements / (uint64_t) block_elements;
                if (blocks > (std::numeric_limits<uint64_t>::max)() / (uint64_t) block_bytes) {
                    err = "native dense: tensor byte count overflow " + tensor.name; return false;
                }
                const uint64_t bytes = blocks * (uint64_t) block_bytes;
                if (tensor.offset > payload || bytes > payload - tensor.offset) {
                    err = "native dense: truncated payload " + tensor.name; return false;
                }
                const auto next = std::upper_bound(offsets.begin(), offsets.end(), tensor.offset);
                if (next != offsets.end() && bytes > *next - tensor.offset) {
                    err = "native dense: overlapping payload " + tensor.name; return false;
                }
                // the uploads below read these: ask for them now so the reads overlap
                if (eligible(tensor, include_ple_key) && !outside(tensor.name) &&
                    strata::kernels::native_mmvq_supported(tensor.type))
                    strata::platform::advise_willneed(gguf.tensor_data(tensor), bytes);
            }
            for (const auto& tensor : gguf.tensors()) {
                if (!eligible(tensor, include_ple_key) || outside(tensor.name)) continue;
                if (!in_range(tensor.name) && tensor.name.find("ple") == std::string::npos) continue;
                if (!seen.insert(tensor.name).second) {
                    err = "native dense: duplicate tensor " + tensor.name; return false;
                }
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name; return false;
                }
                auto& ref = found->second;
                if (ref.native_data) { err = "native dense: override already attached"; return false; }
                if (!strata::kernels::native_mmvq_supported(tensor.type)) continue;
                // #326: the pack keeps an unquantized (--compat-bf16) key, which the PLE reads from the arena
                if (tensor.name == "blk.1.ple_key.weight" && !ref.quantized()) continue;
                if (!ref.quantized() || tensor.shape.size() != 2 ||
                    ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 || ref.ne1 > INT_MAX ||
                    tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name; return false;
                }
#if defined(STRATA_B70_LIB)
                if (const auto* pe = b70_pack_entry(pack, tensor.name, ref.ne0, ref.ne1)) {
                    // this projection from the pack: its records on a 2 MiB boundary, the pack type instead of the GGUF's
                    std::vector<uint8_t> host((size_t) pe->bytes);
                    std::string read_err;
                    if (!pack->read(*pe, host.data(), read_err)) {
                        err = "native dense (STRATA_B70_PACK) " + tensor.name + ": " + read_err; return false;
                    }
                    void* pack_allocation = nullptr;
                    auto pack_status = DPCT_CHECK_ERROR(
                        pack_allocation = (void *)sycl::aligned_alloc_device(
                            strata::b70::pack::kDeviceAlign, host.size(), dpct::get_in_order_queue()));
                    DevicePtr pack_data(pack_allocation);
                    if (pack_status == 0 && pack_allocation != nullptr)
                        pack_status = DPCT_CHECK_ERROR(
                            dpct::get_in_order_queue()
                                .memcpy(pack_data.get(), host.data(), host.size())
                                .wait());
                    if (pack_status != 0 || pack_allocation == nullptr) {
                        err = "native dense (STRATA_B70_PACK) upload " + tensor.name; return false;
                    }
                    strata::device_alloc_register(pack_data.get(), host.size());
                    max_in = (std::max)(max_in, (int) ref.ne0);
                    total += pe->bytes;
                    allocated += alloc_bytes(pe->bytes);
                    ++pack_count;
                    pack_bytes += pe->bytes;
                    pending.push_back(Pending{&ref, strata::b70::kTypeW4, pe->bytes, std::move(pack_data), DevicePtr()});
                    continue;
                }
#endif
                const auto bytes = strata::kernels::native_mmvq_weight_bytes(
                    tensor.type, (int) ref.ne0, (int) ref.ne1);
                void* allocation = nullptr;
                auto status =
                    DPCT_CHECK_ERROR(allocation = (void *)strata::malloc_device_guarded(
                                         bytes, dpct::get_in_order_queue()));
                DevicePtr data(allocation);
                if (status == 0)
                    status = DPCT_CHECK_ERROR(
                        dpct::get_in_order_queue()
                            .memcpy(data.get(), gguf.tensor_data(tensor), bytes)
                            .wait());
                /*
                DPCT1000: Error handling if-stmt was detected but could not
                be rewritten.
                */
                if (status != 0) {
                    /*
                    DPCT1009: SYCL reports errors using exceptions and does
                    not use error codes. Please replace the
                    "get_error_string_dummy(...)" with a real error-handling
                    function.
                    */
                    /*
                    DPCT1001: The statement could not be removed.
                    */
                    err = "native dense upload " + tensor.name + ": " +
                          dpct::get_error_string_dummy(status);
                        return false;
                }
                max_in = (std::max)(max_in, (int) ref.ne0);
                total += bytes;
                allocated += alloc_bytes(bytes);
                // STRATA_Q8_PACKED=1: a second, packed copy of an eligible Q8_0 matrix for the decode MMVQ (the GGUF
                // copy stays: the prompt path's GEMMs read it).
                DevicePtr packed;
                if (tensor.type == 8 && tensor.name != "blk.1.ple_key.weight" &&   // the PLE has its own kernel
                    strata::kernels::native_q8_0_packed_enabled() &&
                    strata::kernels::native_q8_0_packed_eligible((int) ref.ne0, (int) ref.ne1)) {
                    std::vector<uint8_t> host(bytes);
                    strata::kernels::native_q8_0_pack_host(gguf.tensor_data(tensor), host.data(), (int) ref.ne0,
                                                           (int) ref.ne1);
                    void* packed_allocation = nullptr;
                    auto packed_status = DPCT_CHECK_ERROR(
                        packed_allocation = (void *)strata::malloc_device_guarded(
                            bytes, dpct::get_in_order_queue()));
                    packed.reset(packed_allocation);
                    if (packed_status == 0)
                        packed_status = DPCT_CHECK_ERROR(
                            dpct::get_in_order_queue()
                                .memcpy(packed.get(), host.data(), bytes)
                                .wait());
                    /*
                    DPCT1000: Error handling if-stmt was detected but could
                    not be rewritten.
                    */
                    if (packed_status != 0) {
                        /*
                        DPCT1009: SYCL reports errors using exceptions and
                        does not use error codes. Please replace the
                        "get_error_string_dummy(...)" with a real error-handling
                        function.
                        */
                        /*
                        DPCT1001: The statement could not be removed.
                        */
                        err = "native dense packed upload " + tensor.name +
                              ": " +
                              dpct::get_error_string_dummy(packed_status);
                        return false;
                    }
                }
                pending.push_back(Pending{&ref, (int) tensor.type, bytes, std::move(data), std::move(packed)});
            }
            // STRATA_HC_Q8=1: the hyper-connection down / up projections (the layers' and the final mixer's) as Q8_0 for
            // fused_gr_read_multi's Q8_0 read: the GGUF's own Q8_0 blocks (type 8, the Unsloth files), or Q8_0 of the
            // GGUF's BF16 values (type 30, the ISTA files; ggml's reference rounding, on the host).  Uploaded in the
            // read's split layout (fused_gr_q8_0_split_host).  The inject rows are not copied: the read takes the
            // pack's BF16 ones, so STRATA_HC_Q8_INJECT has no effect here.  The BF16 copies stay in the arena (the
            // prompt path's GEMMs read them): the Q8_0 copies are extra VRAM on this port, not a replacement.
            if (hc_q8_requested())
                for (const auto& tensor : gguf.tensors()) {
                    if ((tensor.type != 8 && tensor.type != 30) || !hc_q8_name(tensor.name) || tensor.shape.size() != 2 ||
                        tensor.name.ends_with("_inject.weight") || outside(tensor.name) || !in_range(tensor.name)) continue;
                    auto found = table.table_.find(tensor.name);
                    if (found == table.table_.end() || found->second.hc_q8 != nullptr) continue;
                    auto& ref = found->second;
                    if (tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1 || ref.ne0 % 32 != 0) {
                        err = "native dense (STRATA_HC_Q8): incompatible matrix " + tensor.name; return false;
                    }
                    // the read is compiled for the model geometry (hc_lr rows of hc * n_embd, or the transpose)
                    constexpr int64_t hc_dim = strata::core::kCompiledGeometry.hc_dim(), hc_lr = strata::core::kCompiledGeometry.hc_lr;
                    if (!((ref.ne0 == hc_dim && ref.ne1 == hc_lr) || (ref.ne0 == hc_lr && ref.ne1 == hc_dim))) {
                        err = "native dense (STRATA_HC_Q8): " + tensor.name + " is not a hyper-connection projection of the compiled geometry";
                        return false;
                    }
                    const uint64_t n = (uint64_t) ref.ne0 * (uint64_t) ref.ne1, bytes = n / 32 * 34;
                    std::vector<uint8_t> q8;
                    const uint8_t* blocks = (const uint8_t*) gguf.tensor_data(tensor);
                    if (tensor.type == 30) {
                        const uint16_t* src = (const uint16_t*) blocks;
                        std::vector<float> f((size_t) n);
                        for (uint64_t i = 0; i < n; ++i) { const uint32_t u = (uint32_t) src[i] << 16; std::memcpy(&f[i], &u, 4); }
                        q8 = q8_0_of(f.data(), n);
                        blocks = q8.data();
                    }
                    std::vector<uint8_t> split((size_t) bytes);
                    strata::kernels::fused_gr_q8_0_split_host(blocks, split.data(), ref.ne1, ref.ne0);
                    void* p = nullptr;
                    if (DPCT_CHECK_ERROR(
                            p = (void *)strata::malloc_device_guarded(
                                bytes, dpct::get_in_order_queue())) != 0 ||
                        DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(p, split.data(), bytes).wait()) != 0) {
                        err = "native dense (STRATA_HC_Q8) upload " + tensor.name; return false;
                    }
                    ref.hc_q8 = p;
                    weights_.push_back(p);
                    hc_q8_bytes += bytes;
                }
        }
        if (hc_q8_env()) {
            // a bf16 pack stores no Q8_0 copy: quantize the layers' down / up projections. A row the arena kept
            // (--keep-canonical, a layer split) is quantized on the GPU from the arena's bf16; a skipped row (the
            // single-GPU default: add_hc_skip) from the pack file on the host, and has no bf16 copy at all. The
            // inject rows stay bf16 (82 KB), the final mixer stays bf16 (it runs once a window).
            uint64_t made = 0;
            for (auto& [name, ref] : table.table_) {
                if (ref.hc_q8 != nullptr || name.rfind("blk.", 0) != 0 || !in_range(name) ||
                    !hc_q8_name(name) || !(name.ends_with("_down.weight") || name.ends_with("_up.weight"))) continue;
                const bool skipped = ref.data == nullptr;
                if (skipped && ref.resident) continue;
                if (ref.ne0 % 32 != 0 || ref.elements != ref.ne0 * ref.ne1 || (!skipped && ref.bytes != (uint64_t) ref.elements * 2)) {
                    err = "native dense (STRATA_HC_Q8): " + name + " is not a bf16 matrix"; return false;
                }
                const uint64_t nb = (uint64_t) ref.elements / 32, bytes = nb * 34;
                void* p = nullptr;
                if (DPCT_CHECK_ERROR(p = (void *)sycl::malloc_device(
                                         bytes, dpct::get_in_order_queue())) !=
                    0) {
                    err = "native dense (STRATA_HC_Q8) alloc " + name;
                    return false;
                }
                weights_.push_back(p);
                if (skipped) {
                    std::vector<uint16_t> host;
                    if (!table.read_bf16_host(name, host, err)) return false;
                    const std::vector<uint8_t> q8 = q8_0_of_bf16(host.data(), (uint64_t) host.size());
                    /*
                    DPCT1114: cudaMemcpy is migrated to asynchronization
                    memcpy, assuming in the original code the source host memory
                    is pageable memory. If the memory is not pageable, call
                    wait() on event return by memcpy API to ensure
                    synchronization behavior.
                    */
                    if (DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
                            p, q8.data(), bytes).wait()) != 0) {
                        err = "native dense (STRATA_HC_Q8) upload " + name; return false;
                    }
                } else {
                    strata::kernels::fused_gr_quantize_q8_0((const uint16_t*) ref.data, (uint8_t*) p, (int64_t) nb, nullptr);
                }
                ref.hc_q8 = p;
                hc_q8_bytes += bytes;
                allocated += alloc_bytes(bytes);
                ++made;
            }
            /*
            DPCT1010: SYCL uses exceptions to report errors and does not
            use the error codes. The cudaGetLastError function call was replaced
            with 0. You need to rewrite this code.
            */
            if (made && (0 != 0 ||
                         DPCT_CHECK_ERROR(dpct::get_current_device()
                                              .queues_wait_and_throw()) != 0)) {
                err = "native dense (STRATA_HC_Q8): quantizing the projections failed"; return false;
            }
        }
        if (hc_q8_env()) {   // the scratches of hc_bf16_view (rows without a bf16 copy), here and not on first use
            uint64_t elems = 0;
            for (const auto& [name, ref] : table.table_)
                if (ref.hc_q8 != nullptr && ref.data == nullptr) elems = (std::max)(elems, (uint64_t) ref.elements);
            int dev = 0;
            if (elems != 0 &&
                DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) == 0 &&
                dev >= 0 && dev < 64 && g_hc_scratch[dev][0][0] == nullptr) {
                for (int c = 0; c < 2; ++c)
                    for (int i = 0; i < 2; ++i) {
                        void* b = nullptr;
                        if (DPCT_CHECK_ERROR(b = (void *)sycl::malloc_device(
                                                 (size_t)elems * 2,
                                                 dpct::get_in_order_queue())) !=
                            0) {
                            err = "native dense (STRATA_HC_Q8): scratch alloc";
                            return false;
                        }
                        weights_.push_back(b);
                        g_hc_scratch[dev][c][i] = b;
                    }
                for (int i = 0; i < 2; ++i)
                    if (g_hc_ev[dev][i] == nullptr &&
                        DPCT_CHECK_ERROR(g_hc_ev[dev][i] = new sycl::event()) !=
                            0) {
                        err = "native dense (STRATA_HC_Q8): scratch event"; return false;
                    }
            }
        }
        if (pending.empty()) { err = "native dense: no supported GDN/QSA matrices in supplied shards"; return false; }
        void* allocation = nullptr;
        const auto status =
            DPCT_CHECK_ERROR(allocation = (void *)strata::malloc_device_guarded(
#if defined(STRATA_B70_LIB)
                                 strata::kernels::native_q8_1_bytes(max_in) + 64,   // (+64: a pack product reads 28 bytes past the last block)
#else
                                 strata::kernels::native_q8_1_bytes(max_in),
#endif
                                 dpct::get_in_order_queue()));
        DevicePtr scratch(allocation);
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        /*
        DPCT1000: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (status != 0) {
            err = std::string("native dense scratch: ") +
                  dpct::get_error_string_dummy(status);
            return false;
        }
        // All checks and allocations finish before publishing any reference.
        weights_.reserve(pending.size());
        uint64_t packed_bytes = 0;
        size_t packed_count = 0;
        for (auto& item : pending) {
            item.ref->native_data = item.data.get();
            item.ref->native_type = item.type;
            item.ref->native_q8_1 = scratch.get();
            if (item.packed) {
                strata::kernels::native_q8_0_packed_register(item.data.get(), item.packed.get(), (int) item.ref->ne0,
                                                             (int) item.ref->ne1);
                packed_keys_.push_back(item.data.get());
                packed_bytes += item.bytes;
                ++packed_count;
                weights_.push_back(item.packed.release());
            }
            weights_.push_back(item.data.release());
        }
#if defined(STRATA_B70_LIB)
        if (pack != nullptr)
            std::fprintf(stderr, "native dense: STRATA_B70_PACK=%s: %llu projection matrices from the pack (%.1f MiB; int4, the card "
                                 "library's products), %zu from the GGUF\n", pack->path.c_str(), (unsigned long long) pack_count,
                         pack_bytes / 1048576.0, pending.size() - (size_t) pack_count);
#endif
        if (packed_count)
            std::fprintf(stderr, "native dense: STRATA_Q8_PACKED=1 packed %zu Q8_0 matrices for decode (+%.1f MiB)\n",
                         packed_count, packed_bytes / 1048576.0);
        scratch_ = scratch.release();
        bytes_ = total + hc_q8_bytes;
        allocated_ = allocated;
        if (hc_q8_env())
            std::fprintf(stderr, "strata: STRATA_HC_Q8=1: %.2f GiB of Q8_0 hyper-connection projections for the verify read\n",
                         (double) hc_q8_bytes / 1073741824.0);
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool NativeDense::add_hc_skip(const std::string& pack_dir, std::set<std::string>& skip) {
    std::FILE* f = std::fopen((pack_dir + "/index.txt").c_str(), "rb");
    if (!f) return false;
    char line[1024], name[256];
    while (std::fgets(line, sizeof line, f)) {
        if (line[0] == '#' || std::sscanf(line, "%255s", name) != 1) continue;
        const std::string n = name;
        if (n.rfind("blk.", 0) == 0 && hc_q8_name(n) && (n.ends_with("_down.weight") || n.ends_with("_up.weight"))) skip.insert(n);
    }
    std::fclose(f);
    return true;
}

const uint16_t *NativeDense::hc_bf16_view(const WeightRef &w, void *stream,
                                          int slot, int cls) try {
    if (w.data != nullptr) return (const uint16_t*) w.data;
    if (w.hc_q8 == nullptr) return nullptr;
    // the device's scratch (allocated at load, one pair per device): the prompt path and the single-token read run one
    // after the other on a device, and each use is stream-ordered after the previous GEMM that read the slot
    int dev = 0;
    if (DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) != 0 || dev < 0 ||
        dev >= 64 || slot < 0 || slot > 1 || cls < 0 || cls > 1) return nullptr;
    void* buf = g_hc_scratch[dev][cls][slot];
    if (buf == nullptr) return nullptr;
    // the prompt path: the scratch may still be read by a GEMM another stream enqueued (batch serving: several slots' prompts)
    if (cls == 0 && g_hc_ev_set[dev][slot])strata::q_of(stream)
        ->ext_oneapi_submit_barrier({*g_hc_ev[dev][slot]});
    strata::kernels::fused_gr_dequant_q8_0((const uint8_t*) w.hc_q8, (uint16_t*) buf, w.elements / 32, stream);
    return (const uint16_t*) buf;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void NativeDense::hc_bf16_release(void *stream, int slot, int cls) try {
    int dev = 0;
    if (cls != 0 || slot < 0 || slot > 1 ||
        DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) != 0 || dev < 0 ||
        dev >= 64 || g_hc_ev[dev][slot] == nullptr) return;
    dpct::sync_barrier(g_hc_ev[dev][slot], strata::q_of(stream));
    g_hc_ev_set[dev][slot] = true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
} // namespace strata::core
