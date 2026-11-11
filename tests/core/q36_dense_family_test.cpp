// tests/core/q36_dense_family_test.cpp - the dense qwen35 family (Qwen3.8-27B) on synthetic GGUFs: no model, no GPU.
//
// Built twice by tools/test_q36_family.py (g++, the host headers only):
//   * with -DSTRATA_Q36_BUILD, as strata-q36 compiles them: the architecture guard accepts a qwen35 file, the geometry is the
//     27B's, and the layout table names every tensor of a dense layer (and of the MTP block) and nothing else;
//   * without it, as `strata` compiles them: the same file is refused by name, with the message b70 gives, and the family is
//     unknown.  (That the three headers are then b70's text is tools/check_q36_isolation.py's job.)
#include "gguf_fixture.hpp"

#include "strata/core/family_layout.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-q36-family-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

using fixture::Kv;
using fixture::Tensor;
const uint32_t F32 = 0;

struct Dims {
    uint64_t block_count, embd, heads, heads_kv, key_len, ff, conv, state, groups, v_heads, inner, interval, nextn;
};
// Qwen3.8-27B as its UD-Q4_K_M and UD-Q4_K_XL files say (65 blocks: 64 layers and the MTP block, GDN K 16 V 48 x 128)
const Dims k27B = {65, 5120, 24, 4, 256, 17408, 4, 128, 16, 48, 6144, 4, 1};
// a tiny dense model for the layout checks (every tensor F32)
const Dims kTiny = {5, 64, 2, 1, 16, 32, 4, 8, 2, 4, 32, 4, 1};

std::vector<Kv> metadata(const Dims& d, uint64_t vocab) {
    const std::string p = "qwen35.";
    std::vector<Kv> kv = {fixture::str("general.architecture", "qwen35"),
                          fixture::u32(p + "block_count", d.block_count),
                          fixture::u32(p + "embedding_length", d.embd),
                          fixture::u32(p + "attention.head_count", d.heads),
                          fixture::u32(p + "attention.head_count_kv", d.heads_kv),
                          fixture::u32(p + "attention.key_length", d.key_len),
                          fixture::u32(p + "attention.value_length", d.key_len),
                          fixture::u32(p + "feed_forward_length", d.ff),
                          fixture::u32(p + "ssm.conv_kernel", d.conv),
                          fixture::u32(p + "ssm.state_size", d.state),
                          fixture::u32(p + "ssm.group_count", d.groups),
                          fixture::u32(p + "ssm.time_step_rank", d.v_heads),
                          fixture::u32(p + "ssm.inner_size", d.inner),
                          fixture::u32(p + "full_attention_interval", d.interval),
                          fixture::u32(p + "nextn_predict_layers", d.nextn)};
    std::vector<std::string> tokens;
    for (uint64_t i = 0; i < vocab; ++i) tokens.push_back("t" + std::to_string(i));
    kv.push_back(fixture::str_array("tokenizer.ggml.tokens", tokens));
    return kv;
}

#if defined(STRATA_Q36_BUILD)
std::vector<Tensor> tensors_of(const strata::core::ModelGeometry& g) {
    std::vector<Tensor> ts;
    uint8_t seed = 1;
    auto add = [&](const std::string& name, const std::vector<int64_t>& shape) {
        ts.push_back({name, std::vector<uint64_t>(shape.begin(), shape.end()), F32, seed++});
    };
    for (const auto& w : strata::core::global_tensors(g)) add(w.name, w.shape);
    for (int64_t l = 0; l < g.n_layers + g.n_mtp_layers; ++l)
        for (const auto& w : strata::core::layer_tensors(g, l)) add("blk." + std::to_string(l) + "." + w.name, w.shape);
    return ts;
}

std::string layout_error(const fs::path& dir, void (*mutate)(std::vector<Tensor>&)) {
    const fs::path meta = dir / "meta.gguf";
    fixture::write(meta, metadata(kTiny, 16), {});
    const strata::GgufFile f(meta.string());
    strata::core::ModelGeometry g;
    int64_t k = 0;
    std::string err;
    if (!strata::core::geometry_from_gguf(f, g, k, err)) return "geometry: " + err;
    std::vector<Tensor> ts = tensors_of(g);
    if (mutate) mutate(ts);
    const fs::path model = dir / "dense-tiny.gguf";
    fixture::write(model, metadata(kTiny, 16), ts);
    const strata::GgufModel m({model.string()});
    size_t n = 0;
    if (!strata::core::check_gguf_layout(m, g, err, &n)) return err;
    return {};
}
#endif
}  // namespace

int main() {
    std::printf("q36_dense_family_test (%s)\n",
#if defined(STRATA_Q36_BUILD)
                "STRATA_Q36_BUILD"
#else
                "as strata compiles it"
#endif
    );
    TempDir d;
    const fs::path file = d.path / "dense.gguf";
    fixture::write(file, metadata(k27B, 8), {});
    const strata::GgufFile f(file.string());
#if defined(STRATA_Q36_BUILD)
    check(strata::model_family(f) == strata::ModelFamily::Qwen35, "model_family: qwen35 is the dense family");
    check(strata::check_architecture(f).empty(), "the architecture guard accepts a qwen35 file");
    check(std::string(strata::model_family_arch(strata::ModelFamily::Qwen35)) == "qwen35", "its arch string and key prefix");
    strata::core::ModelGeometry g;
    int64_t k = -1;
    std::string err;
    const bool ok = strata::core::geometry_from_gguf(f, g, k, err);
    check(ok, "geometry_from_gguf reads it" + (ok ? std::string() : ": " + err));
    check(g.family == strata::ModelFamily::Qwen35 && g.n_layers == 64 && g.n_mtp_layers == 1 && g.n_embd == 5120,
          "27B: 64 layers + 1 MTP, hidden 5120");
    check(g.n_head == 24 && g.n_head_kv == 4 && g.head_dim == 256 && g.qsa_interval == 4, "27B attention: 24 Q / 4 KV x 256, every 4th layer");
    check(g.ssm_k_heads == 16 && g.ssm_v_heads == 48 && g.ssm_state_size == 128 && g.ssm_conv_channels == 10240 && g.ssm_d_conv == 4,
          "27B GDN: 16 K groups, 48 V heads, state 128, conv 10240 channels");
    check(g.n_expert == 0 && k == 0 && g.n_ff == 17408 && g.n_ff_shexp == 0, "27B FFN: dense, 17408 wide, no experts");
    check(!g.has_hyper_connections() && !g.has_indexer(), "no hyper-connections, no indexer");
    {
        std::vector<Kv> kv = metadata(k27B, 8);
        for (auto it = kv.begin(); it != kv.end(); ++it)
            if (it->key == "qwen35.feed_forward_length") { kv.erase(it); break; }
        const fs::path miss = d.path / "miss.gguf";
        fixture::write(miss, kv, {});
        const strata::GgufFile m(miss.string());
        check(strata::check_architecture(m) == "missing qwen35.feed_forward_length", "a file without feed_forward_length is refused, naming the key");
        Dims bad = k27B;
        bad.nextn = bad.block_count;
        const fs::path bp = d.path / "bad.gguf";
        fixture::write(bp, metadata(bad, 8), {});
        const strata::GgufFile b(bp.string());
        check(strata::check_architecture(b).find("nextn_predict_layers") != std::string::npos, "an MTP block count not below block_count is refused");
    }
    check(layout_error(d.path, nullptr).empty(), "a complete dense file passes its layout (ffn_gate / up / down, the MTP block's too)");
    std::string e = layout_error(d.path, [](std::vector<Tensor>& ts) {
        for (auto it = ts.begin(); it != ts.end(); ++it)
            if (it->name == "blk.2.ffn_down.weight") { ts.erase(it); break; }
    });
    check(e == "missing blk.2.ffn_down.weight", "a missing dense FFN tensor is refused (" + e + ")");
    e = layout_error(d.path, [](std::vector<Tensor>& ts) { ts.push_back({"blk.0.ffn_gate_inp.weight", {64, 4}, F32, 99}); });
    check(e.find("blk.0.ffn_gate_inp.weight") != std::string::npos && e.find("not in the") != std::string::npos,
          "a router tensor, which a dense model has not, is refused");
    e = layout_error(d.path, [](std::vector<Tensor>& ts) {
        for (auto it = ts.begin(); it != ts.end(); ++it)
            if (it->name == "blk.4.ffn_up.weight") { it->shape[1] += 1; break; }
    });
    check(e.find("blk.4.ffn_up.weight") != std::string::npos && e.find("wants") != std::string::npos,
          "a wrong shape in the MTP block's FFN is refused, naming the tensor");
#else
    check(strata::model_family(f) == strata::ModelFamily::Unknown, "model_family: qwen35 is unknown to strata");
    check(strata::check_architecture(f) == "architecture is 'qwen35', this engine requires 'qwen4exp'",
          "strata refuses a qwen35 file by name, as b70 does: " + strata::check_architecture(f));
    strata::core::ModelGeometry g;
    int64_t k = 0;
    std::string err;
    check(!strata::core::geometry_from_gguf(f, g, k, err) && err == strata::check_architecture(f), "and geometry_from_gguf says the same");
#endif
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
