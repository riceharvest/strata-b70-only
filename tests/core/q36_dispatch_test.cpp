// tests/core/q36_dispatch_test.cpp - the family switch of `strata` (include/strata/qwen35/dispatch.hpp) on synthetic GGUF
// headers: no model, no GPU.  Built by tools/test_q36_family.py with g++ and -DSTRATA_Q36_BUILD (the merged binary).
#include "gguf_fixture.hpp"

#include "strata/qwen35/dispatch.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

bool dense(std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    return strata::qwen35::serves_dense((int) argv.size(), argv.data());
}
}  // namespace

int main() {
    std::printf("q36_dispatch_test\n");
    const fs::path dir = fs::temp_directory_path() / ("strata-q36-dispatch-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const std::string dense_gguf = (dir / "dense.gguf").string();
    const std::string flash_gguf = (dir / "flash.gguf").string();
    const std::string moe_gguf = (dir / "moe.gguf").string();
    const std::string junk = (dir / "junk.gguf").string();
    fixture::write(dir / "dense.gguf", {fixture::str("general.architecture", "qwen35")}, {});
    fixture::write(dir / "flash.gguf", {fixture::str("general.architecture", "qwen4exp")}, {});
    fixture::write(dir / "moe.gguf", {fixture::str("general.architecture", "qwen35moe")}, {});
    std::ofstream(junk) << "this is not a gguf file";

    check(dense({"strata", "--native", dense_gguf, "--serve"}), "--native <qwen35 file> goes to the dense family");
    check(dense({"strata", "--serve", "--model", dense_gguf, "--mtp", "3"}), "--model <qwen35 file>, flags before and after");
    check(!dense({"strata", "--native", flash_gguf, "--serve"}), "--native <qwen4exp file> stays on Flash-Next");
    check(!dense({"strata", "--model", flash_gguf}), "--model <qwen4exp file> stays on Flash-Next");
    check(!dense({"strata", "--native", moe_gguf}), "a qwen35moe file is not the dense family (the Flash-Next path refuses it by name)");
    check(!dense({"strata", "--pack", "some-pack-dir", "--serve"}), "--pack (no GGUF) stays on Flash-Next");
    check(!dense({"strata"}), "no arguments");
    check(!dense({"strata", "--native"}), "--native without a value");
    check(!dense({"strata", "--native", junk}), "a file that is not a GGUF is left to the Flash-Next path to report");
    check(!dense({"strata", "--native", (dir / "missing.gguf").string()}), "a missing file is left to the Flash-Next path to report");
    check(!dense({"strata", "--version"}), "--version");

    std::error_code ignored;
    fs::remove_all(dir, ignored);
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
