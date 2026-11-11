# Third-party code

What this repository carries or builds from other projects, and under which licence. The licence of the repository itself is
[LICENSE](LICENSE) (MIT).

## In the repository

| Project | What | Licence (SPDX) | Where |
| --- | --- | --- | --- |
| ggml / llama.cpp, [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | two headers, and the CPU dequantizer the build fetches at the commit pinned in `sycl/CMakeLists.txt` | MIT | [third_party/ggml/LICENSE](third_party/ggml/LICENSE) |
| ggml / llama.cpp, adapted | `sycl/src/qwen35/q36_mmvq_dense.dp.cpp`: the block types and KREG structs copied from `sycl/src/kernels/cuda/native_mmvq.dp.cpp`, which is adapted from llama.cpp's CUDA kernels; it carries the same notice | MIT | its own header (the ggml authors) |
| SYCLomatic (dpct) helper headers, [oneapi-src/SYCLomatic](https://github.com/oneapi-src/SYCLomatic) | the CUDA to SYCL helper headers the migrated sources include | Apache-2.0 WITH LLVM-exception | [sycl/include/dpct/LICENSE.TXT](sycl/include/dpct/LICENSE.TXT) |
| vllm-xpu-kernels, [vllm-project/vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels) | `sycl/src/qwen35/merge_attn_states.hpp`: a **modified** copy of one file; `sycl/third_party/q36-tla/vllm-fmha_utils-no-torch.patch`: a patch to another (both for `strata-q36`, see below) | Apache-2.0 | [third_party/licenses/vllm-xpu-kernels-LICENSE](third_party/licenses/vllm-xpu-kernels-LICENSE) |

## Not in the repository: fetched for `strata-q36` (optional)

`strata-q36`'s long-prompt attention (`libq36_tla.so`, [docs/Q36.md](docs/Q36.md)) is compiled against two projects.
`sycl/tools/get-q36-tla.sh` downloads them into `third_party/` (git-ignored) at these commits. Nothing of them is committed
here except what the table above says.

| Project | Commit | Licence (SPDX) | Notes |
| --- | --- | --- | --- |
| [intel/sycl-tla](https://github.com/intel/sycl-tla) (the CUTLASS fork for SYCL) | `87f6850680a580654b9ea2c80dbc01aeb36ad231` (2026-08-10, v0.9.2 + 7 commits, 9 before the v0.9.3 tag) | BSD-3-Clause; the 15 files under `include/cute/util/` (`compat/`) are Apache-2.0 WITH LLVM-exception | `LICENSE.txt` of the checkout; a copy is [third_party/licenses/sycl-tla-LICENSE.txt](third_party/licenses/sycl-tla-LICENSE.txt). Used unchanged: `include/`, `tools/util/include/`, `applications/flash_attention_v2/`. |
| [vllm-project/vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels) | `ddf336d86e3c8602888572a3502f951abd51df12` (2026-09-18, v0.1.15 + 4) | Apache-2.0 (the repository's `LICENSE`); the files under `csrc/xpu/attn/xe_2/collective/` and `kernel/` carry `SPDX-License-Identifier: BSD-3-Clause` (Intel, Codeplay) | Used unchanged: `csrc/xpu/attn/xe_2/`. One file is patched, see below. |

What was changed, as Apache-2.0 asks us to say:

- `csrc/xpu/attn/fmha_utils.hpp` (vllm-xpu-kernels): the torch includes and the `aten_*` dtype converters are taken out, because
  the library is built without torch. The patch is `sycl/third_party/q36-tla/vllm-fmha_utils-no-torch.patch`;
  `sycl/tools/get-q36-tla.sh` applies it to the fetched copy.
- `csrc/attention/merge_attn_states.cpp` (vllm-xpu-kernels) became `sycl/src/qwen35/merge_attn_states.hpp`: the kernel only (no torch
  launcher), the `to_float` / `from_float` helpers of `csrc/utils.h` added, the FP8-output branches left out. The merge itself is
  the released code. The file says so at its top.

If you hand someone a built `libq36_tla.so`, include the BSD-3-Clause notice of sycl-tla and the vllm-xpu-kernels licence with it.

## Data

- `tools/qwen35/q36_draft_vocab_64k.json` (and the 32k and 48k lists beside it): the token ids the MTP draft head of `strata-q36`
  picks among. The 64K list is the one the vLLM Intel B70 cookbook publishes (a list of token ids built from public text), with its
  ids checked against the Qwen3.8-27B GGUF. No licence statement came with that list when it was copied; check it before you pass
  the file on.
