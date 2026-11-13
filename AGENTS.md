# AGENTS.md

This is a fork of [Strata](https://github.com/Niko1221/Strata) for exactly one setup: one Intel Arc Pro B70 (32 GB, Battlemage
`bmg-g31`) on Linux with the `xe` driver, serving Qwen3.8-Flash-Next from the SYCL engine in `sycl/`. Work that does not help
this setup is out of scope. The NVIDIA, AMD and Apple runtimes, the desktop app and the Windows and macOS packages are not part of it.

Issues, pull requests and benchmark reports for this fork belong to `riceharvest/strata-b70-only`, never to upstream
Strata: upstream has no B70 and cannot reproduce a report made here. The maintainer answers fastest on X as
`@imdariotoo`, otherwise at dev@darioa.nl.

A second model family exists in the same binary: the dense Qwen3.8-27B (`sycl/src/qwen35`, [docs/Q36.md](docs/Q36.md)). `strata` reads the
architecture of the GGUF at start and runs the dense engine for `qwen35`; `Q36=OFF` for `sycl/tools/build.sh` builds the Flash-Next-only `strata`.

DeepSeek V4.1-Flash (`sycl/src/dsv41`) is not maintained since 2026-10-30 and its build is off by default.

## Where things are

- `sycl/`: the SYCL engine (`sycl/CMakeLists.txt`, kernels in `sycl/src`, headers in `sycl/include`). `src/` and `include/` hold the
  CPU-side code it compiles.
- `serve/`: the Python server (`serve/server.py`) with an OpenAI- and Anthropic-compatible API and a web page.
- `strata_cli/`, `strata_control/`, `setup.py`, `setup.sh`: the terminal command and the setup that writes a config.
- `docs/B70.md`: the numbers that were measured, the switches, the build and run commands. Read it first. `docs/INTEL.md` has the
  rest of the Intel engine's switches.
- B70 defaults of the engine are set in `strata::apply_b70_defaults` (`sycl/include/strata/sycl_queue.hpp`). A switch is an
  environment variable `STRATA_*`, read once.

## Build

The engine builds in the `strata-sycl-dev` image. From the repository folder (as in README.md and docs/B70.md; with docker, pass `-e PARITY=ON` for the next paragraph):

    docker build --pull -t strata-sycl-dev sycl/tools
    H=$(basename "$PWD")
    docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e AOT=bmg-g31 -e REPO=/work/$H \
        -e BUILD_DIR=/work/$H/build-sycl-aot strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"

The build ends with `BUILD EXIT 0` and `errors: 0`. `sycl/tools/build.sh` with no target builds `strata` only. The kernel parity
programs are built on request: `PARITY=ON bash sycl/tools/build.sh <target>`. The dense family is in that binary (docs/Q36.md; for its
long-prompt attention run `bash sycl/tools/get-q36-tla.sh` once first, else the build says it goes on without). Other variables of `build.sh` (`AOT`, `I8_ESIMD`,
`IQ_DECODE_FAST`, `HOST_MARCH`, `BUILD_DIR`, `JOBS`) are listed at the top of the script. Without `AOT` the first start compiles every
kernel (about 47 s).

## Serve

`serve/server.py --engine strata --config <config.json>`. The config is JSON (`exe`, `args`, `cwd`, `tokenizer`, `model_name`,
`env`). `./setup.sh --backend sycl` writes it (`sycl/setup_intel.py`, `to_sycl`) with `exe` = `sycl/serve/strata-sycl.sh`, which runs the
built binary (`build-sycl-aot/strata` by default) in the `strata-sycl-dev` container, and a `run-<tag>.sh` that starts the server; the
variables the wrapper reads are described at the top of that script. For `--family unsloth --model UD-IQ1_S`, the model this fork
serves, setup downloads the files at a pinned revision, grafts one tensor into shard 2 (`tools/graft_tensor.py`, see "The served
setup" in docs/B70.md), builds the pack and writes the served flags and `env` into the config (`--no-served-recipe` leaves them
out). `--host` and `--port` (default 8095) choose the address. The flags the engine is served with, and what each costs and gains,
are in docs/B70.md.

## Working rules for agents here

- One configuration: the one in docs/B70.md (Flash-Next, `strata`). The dense Qwen3.8-27B (docs/Q36.md) is the one
  optional second family of the same `strata` binary. Do not add a third model, another quant, or a second GPU path. What the dense
  family needs differently from a shared kernel or header goes into sources of its own (`sycl/src/qwen35`, as
  `q36_mmvq_dense.dp.cpp` does) or behind `STRATA_Q36_BUILD`; with `Q36=OFF` `strata` compiles the same sources as before the merge
  (`tools/check_q36_isolation.py` checks it); a change that moves the Flash-Next path needs a measurement on it.
- One stream. The served setup is single-stream (`parallel` unset). Two streams were measured on the B70 and did not beat one:
  102-118 tok/s solo against 81-102 together, and a prompt read and a decode take turns. `parallel` stays an opt-in setting;
  work goes into the single stream unless parallel is asked for.
- Code first. Read the change again several times (indexing, sizes, sync, edge cases, every caller) before any build. Then one
  build and one GPU test, not a build per edit.
- One GPU job at a time. Never run a benchmark while a build is running: both change the other's numbers.
- A new run-time switch is `STRATA_*`, read once, with its default in `apply_b70_defaults`. A change in a hot path must be switchable
  off, so it can be compared on one binary.
- Compare two settings with a fixed `--expert-cache <n>` (not `auto`) and `STRATA_SPEC_COST=0`, so the window sizes do not follow
  timings that differ between runs. Give the numbers with what they were measured on. Make no claim without a measurement.
- Tests that need no GPU: `python3 -m pytest serve/test_<name>.py -q` for one server test file (`serve/test_q36_dense.py` for the 27B's
  config), `python3 tools/test_setup_sycl.py`
  for setup, `python3 tools/test_graft_tensor.py` for the graft tool. The host-only C++ tests (`conversation_spill_test`, `draft_policy_test`, `gated_proposal_test`, `spec_admission_test`) are always configured;
  build one by name with `build.sh <target>`.
- Plain words and short sentences in docs and comments.

## Contributing a change or report

The rules are in [CONTRIBUTING.md](CONTRIBUTING.md). Short version: title starts with its area (`[Engine]`, `[Docs]`, `[Other]`),
one problem per change, validation stated as passed, failed and not run, a finding from reading code is marked as unconfirmed.
Reports from other hardware: [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md).
