// include/strata/qwen35/dispatch.hpp - the family switch of `strata`: which engine path a command line asks for.
//
// `strata` is one binary for two model families.  Qwen3.8-Flash-Next (GGUF architecture qwen4exp, or a --pack) runs on the
// Flash-Next engine (sycl/src/program/generate.cpp).  The dense Qwen3.8-27B (architecture qwen35) runs on its own driver,
// strata::qwen35::run() (sycl/src/qwen35/q36_engine.dp.cpp).  main() asks this header first, before any engine state exists.
//
// Host only (no SYCL): tools/test_q36_family.py compiles it with g++ against synthetic GGUFs.  Present only in a build with
// STRATA_Q36_BUILD (the dense family compiled in; `Q36=OFF` leaves it out and `strata` is the Flash-Next-only binary).
#pragma once

#if defined(STRATA_Q36_BUILD)

#include "strata/artifact/gguf_reader.hpp"

#include <cstring>
#include <exception>

namespace strata::qwen35 {

/// True when the command line names a model (`--native F` or `--model F`, F the first shard) whose general.architecture is
/// the dense qwen35.  Only the file's header is read.  A file that cannot be read is not dense: the Flash-Next path reports it.
inline bool serves_dense(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--native") != 0 && std::strcmp(argv[i], "--model") != 0) continue;
        try {
            const strata::GgufFile meta(strata::gguf_split_paths(argv[i + 1]).front());
            return strata::model_family(meta) == strata::ModelFamily::Qwen35;
        } catch (const std::exception&) {
            return false;   // not readable here: the normal path reports it
        }
    }
    return false;
}

}  // namespace strata::qwen35

#endif  // STRATA_Q36_BUILD
