// include/strata/qwen35/engine.hpp - the dense Qwen3.8-27B (GGUF architecture qwen35) engine path as a library:
// `strata`'s main() hands a qwen35 file to it as its first act (the family switch, include/strata/qwen35/dispatch.hpp).
// There is no binary of its own any more.  See sycl/src/qwen35/q36_engine.dp.cpp and docs/Q36.md.
#pragma once

namespace strata::qwen35 {

/// The engine's command line (strata's own flag names where they mean the same thing: --native / --model,
/// --max-context, --prefill / --chunk, --serve, --mtp N, --pool-workers, --tokens, --max-new, --no-token-graph; the
/// family's own: --kv, --w4, --batch, --batch-mtp, --draft-vocab, ...; docs/Q36.md).  Returns the process exit code.
int run(int argc, char** argv);

}  // namespace strata::qwen35
