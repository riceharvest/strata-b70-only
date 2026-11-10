# Strata on an Intel Arc: the SYCL engine

This page is for people who build the Intel Arc engine from source or run it in a container, and for those who maintain the
port. For the released Linux engine package, start with [INTEL_ARC.md](INTEL_ARC.md).

Strata's engine is CUDA. On an Intel Arc it runs as Strata's own engine ported to SYCL (`sycl/`), behind the same server:
OpenAI and Anthropic APIs, streaming, tool calls, MCP and the web app are unchanged. Everything Intel-specific lives in `sycl/`;
no shared upstream file is changed. `sycl/setup_intel.py` and `sycl/serve/server_intel.py` wrap upstream's `setup.py` and
`serve/server.py` from outside. llama.cpp's SYCL backend runs the same GGUF at about a third of the speed.

DeepSeek V4.1-Flash does not run on Intel. Pictures are on by default ([below](#pictures-on-an-arc)).

## Quick start

On Ubuntu 24.04 or newer, with Intel's GPU driver installed (`ls /dev/dri` shows a `renderD128`; see
[INTEL_ARC.md](INTEL_ARC.md#what-you-need)):

```sh
sudo apt install docker.io git
sudo usermod -aG docker,render $USER     # log out and in; `docker ps` must work without sudo
cd Strata
docker build --pull -t strata-sycl-dev sycl/tools                 # once, about 6 minutes, 13 GB
# build the engine: see "How to build it", then
./setup.sh --backend sycl [setup's options, e.g. --model IQ3_S --context 32768 --port 8085]
```

`setup.sh` makes its own Python environment in `.venv`. Do not run `python3 sycl/setup_intel.py` yourself: Ubuntu's system
Python refuses it (`externally-managed-environment`). `./setup.sh --backend sycl` is the one command for the first install and
every later start. Then `run-<model>.sh` starts the model; the first start of a 30 GB model takes about two minutes.

## Setup

`sycl/setup_intel.py` imports upstream's `setup.py` and replaces the Intel steps. The model choice, download, pack, tokenizer,
MTP draft layer and the context and KV questions are setup's own.

| Part | What changes on an Arc |
| --- | --- |
| GPU check | The Arc is found in sysfs (vendor 8086 under `xe` or `i915`) and offered through setup's AMD path (it builds locally and has no images). |
| Engine | Uses the SYCL build (`build-sycl-aot/strata` or `build-sycl/strata`, run in the `strata-sycl-dev` image by `sycl/serve/strata-sycl.sh`). |
| RAM rule | Does not apply on an `xe` card: the port streams the experts from the GGUF into VRAM (`--stream-experts`), so RAM only decides KV streaming. `--check` lists what fits by VRAM. |
| Config | Container paths and `"backend": "sycl"`. VRAM reserve 1,024 MiB up to 32K context and 2,048 MiB above, 300 MiB on a card under 12 GB. A `--vram-reserve-mib N` you give is kept. KV streaming (`--kv-resident 32768`) is on from 64K when the RAM holds the KV. |
| Fit rule | A model that does not fit the card at the default context gets a context down the ladder 32K, 16K, 8K, a smaller prompt chunk and a lower reserve until it does. Setup prints one plain line per change. |
| Prompt chunk | 4,096 tokens; 8,192 on a Battlemage card with 24 GB or more above 32K context. A `--prefill` in the config is kept. |
| Alchemist (A-series) | A different config: [Arc A750 and the other Alchemist cards](#arc-a750-and-the-other-alchemist-cards). |
| Engine settings | The config's `"env"` block reaches the engine. `strata-sycl.sh` forwards every `STRATA_`, `ONEAPI_`, `UR_`, `IGC_`, `SYCL_` and `ZES_` variable and sets `STRATA_VERIFY_DEVICE_PLAN=1`, `STRATA_VERIFY_NO_HOST=1` and `STRATA_STAGER_THREADS=12` unless the environment says otherwise (for example `"env": {"STRATA_VERIFY_NO_HOST": "0"}`). A value of `0` is not passed on, because the engine tests these switches by presence. |

The models, packs and checkout must sit under the folder `strata-sycl.sh` mounts at `/work` (the one above the checkout, or
`STRATA_SYCL_ROOT`). An earlier Strata install on the same PC keeps its data folder in `~/.config/strata/settings.json`; pass
`--data-dir <folder above the checkout>/Strata-data` to put this one under the mount. A symlink to a folder outside the mount does
not work in the container; a hard link does. If a future `setup.py` drops a step this relies on, setup stops with a message
instead of writing a wrong config.

## How to build it

The engine is built in Docker, in the `strata-sycl-dev` image (oneAPI 2026.1 compiler, oneMKL and Level Zero, from a
community llama.cpp SYCL image), so the PC needs no oneAPI. From the checkout (its folder is called `Strata` here; the
folder above it is what the container mounts at `/work`):

    cd Strata
    docker build --pull -t strata-sycl-dev sycl/tools   # once: about 6 minutes, a 13 GB image
    H=$(basename "$PWD")

**Arc B-series** (Pro B70: `bmg-g31`; B580, B570 and Pro B60: `bmg-g21`; `ocloc ids bmg-g21` in the image lists the
device ids). Ahead-of-time code, no compile at the first start:

    docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e AOT=bmg-g31 -e REPO=/work/$H \
        -e BUILD_DIR=/work/$H/build-sycl-aot strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"

**Arc A-series** (Alchemist: A750, A770, A580, A380): no `AOT=`. The code is compiled from SPIR-V at the first start
(JIT, about a minute). Nobody has run an AOT build for these cards.

    docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e REPO=/work/$H \
        -e BUILD_DIR=/work/$H/build-sycl strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"

`--pull` matters: the image starts from a community image whose tag moves, and `docker build` reuses a copy of it from an earlier
build unless it is told to fetch the current one. An old copy can hold a device compiler (ocloc from Compute Runtime
26.22.38646, IGC 2.36.3) that crashes on the Battlemage kernels ("IGC: Internal Compiler Error: Floating point exception", 77 device
images, then `gen compiler command failed with exit code 245` at the last link, #3219). `sycl/tools/build.sh` checks the
version before the compile and stops with the fix. If you cannot update the image, `-e I8_ESIMD=OFF` builds without the fused int8
kernels (stubs: the prompt's experts take the slower XMX path, the answers do not change); `-e ALLOW_BAD_OCLOC=1` builds anyway.
**Podman instead of Docker** (e.g. Fedora Silverblue, rootless): `podman build -t strata-sycl-dev sycl/tools`, then the
same `run` lines with `podman` and `--security-opt label=disable` (a data disk without SELinux labels is otherwise
unreadable in the container). To serve or bench that way, set `STRATA_SYCL_RUNNER=podman` for `strata-sycl.sh`;
`STRATA_SYCL_RUNNER=native` runs the engine directly when the server already runs inside a toolbox with oneAPI (the
data root at `/work`), and `distrobox:<name>` enters that distrobox from the host.

The build takes about 3 minutes and ends with `BUILD EXIT 0` and `errors: 0`. The binary is `build-sycl-aot/strata`
(B-series) or `build-sycl/strata` (A-series); setup finds either one. Run the `docker build` again after pulling a
Strata update that changes `sycl/tools/Dockerfile`, and the `docker run` after every update of the engine (setup
says when the engine is older than it needs).

## Things that matter on this GPU

- **`SYCL_CACHE_PERSISTENT` must be 0.** The persistent JIT cache segfaults on Xe2 during the first
  compile. The start script sets it; if you run llama-server by hand, do too.
- **The whole model goes on the card** (`--n-gpu-layers 999`), except `per_layer_token_embd.weight`,
  the single 28.8 GB tensor of shard 2. `--override-tensor per_layer_token_embd=CPU` keeps it in host
  memory, mmapped and paged from the SSD by row, which is how the model's authors serve it. Host RSS
  stays around 2 GB.
- **Thinking.** The model reasons before it answers. Strata's web app has the setting; for an API
  client, put `{"reasoning_effort": "none"}` in `strata-<model>.shared-settings.json` next to the
  config, or send `chat_template_kwargs: {"enable_thinking": false}` per request. Without one of
  those a short `max_tokens` is spent entirely inside the think block and the answer looks empty.
- **`/health` says 503 while loading**; the server polls `/props` instead.

## How much context fits

The architecture keeps this small: only every fourth layer is full attention (12 of 48); the other
36 are gated-delta-net layers with a fixed-size recurrent state that does not grow with context.
Per token, the 12 attention layers keep 2 KV heads x 256 x (K + V) at q8_0, plus the
sparse-attention indexer's keys. **Measured: about 20 KiB per token** (VRAM grows 0.6 GB per 32k
of context with everything else unchanged).

| context | VRAM in use, model loaded | headroom on 32 GB | status |
|---|---|---|---|
| 32,768 | 28.4 GB | 3.5 GB | measured, the default |
| 65,536 | 29.0 GB | 2.9 GB | measured, loads |
| 98,304 | 29.6 GB | 2.3 GB | measured, loads |
| 131,072 | 30.3 GB | 1.5 GB | **measured, the practical ceiling**: a 104,798-token prompt (79% of the window) read in 247 s at 424 tok/s, then answered |
| 163,840 | ~31.0 GB | ~0.9 GB | not attempted: under the 1.2 GB safety margin |
| 262,144 | ~32.6 GB | none | **does not fit - asking for it took the host down** |

**Do not ask for more than fits.** On this driver a GPU allocation past VRAM does not fail: the xe
driver evicts buffers into host RAM, the kernel runs out of memory, and the machine livelocks until
its hardware watchdog resets it. A 262K request did exactly that on 2026-09-29, and llama.cpp's own
"failed to fit params" check fired too late to prevent it. Compute the KV size first and leave
1.5 GB free.

Prompt reading gets faster with size, because the work batches better: ~150 tok/s on a 2.7k-token
prompt, 424 tok/s on a 105k-token one. So a full 128k window costs about five minutes to read,
not fifteen.

## The engine itself on Intel: the SYCL port (`sycl/`)

Everything above runs the model through llama.cpp. This is the other half: Strata's own engine - the 50 CUDA
kernels and the host code that drives them (streams, events, graph capture, pinned memory) - built for the
Arc with oneAPI. It is a migration of the tree, not a new backend: the engine has no backend seam to slot
into.

**Layout.** `sycl/src` and `sycl/include` mirror the tree and hold only the files the port changes; the
SYCL build (`sycl/CMakeLists.txt`) takes every other source from the original location. No upstream file
is edited. `sycl/include/dpct/` is the vendored SYCLomatic helper library, so the port builds without the
migration tool.

**How it was made, so it can be redone.**

1. `sycl/tools/Dockerfile` - the dev image: the llama.cpp SYCL image plus SYCLomatic (`dpct` 2025.3),
   ninja, and the CUDA 12.8 headers that `sycl/tools/get-cuda-headers.sh` pulls out of NVIDIA's pip wheels
   (dpct parses CUDA; it needs the headers, not the toolkit).
2. `sycl/tools/migrate.sh` - writes a compilation database for the 86 CUDA-touching translation units and
   runs dpct over them. 85 migrate; dpct reports no line it could not migrate, and about 1,400 advisory
   notes.
3. `sycl/tools/fixups.py` - what dpct got wrong or could not do, as an idempotent script with a reason per
   item. The ones that mattered:
   - CUDA's null stream means the default stream; dpct turned it into a null `sycl::queue*`. Every
     stream cast now goes through `strata::q_of()` (`sycl/include/strata/sycl_queue.hpp`).
   - `__ldg((const float*) p)` came out as `*p`, reading one byte of a float scale (two sites, s_gemv).
   - `__fadd_rn(a, b ? c : d)` lost its parentheses (two sites).
   - the ggml lookup tables were threaded through kernel parameters with the wrong table per template;
     they are plain `static const` arrays read from device code, as ggml-sycl does.
   - helper headers renamed since dpct 2025.3 (`entangle`, `chunked_partition`), graph introspection and
     `cudaGraphUpload` (no SYCL equivalents), `%globaltimer` (the stage profiler reads zeros).
4. `sycl/tools/build.sh` - configure + build with icpx inside the image. Two compiler flags are load-bearing:
   `-fp-model=precise` (icpx defaults to a fast FP model) and `-cl-fp32-correctly-rounded-divide-sqrt`
   for the device compiler - the Arc's fp32 divide is not correctly rounded by default (OpenCL allows
   2.5 ulp), and Strata's quantizers are byte-exact against ggml through `amax / 127`. Measured: without
   it `quantize_act_parity` has 303k mismatches, with it none.

**Where it stands (2026-09-29).** The whole tree builds and links: the `strata` binary and 22 kernel
parity tests. On the B70, 19 of the 22 pass byte-for-byte or within their tolerances; the other three
need model fixtures (`iq_parity`, `ple_parity`) or the tensor-core kernel below (`qsa_prompt_attn`).

| parity test | result |
|---|---|
| bf16_gemv, cvec, dequant_s2, elementwise, gdn, gr, kv_q4, kv_q8, kv_stream, qsa, quantize_act, rope, router_top10, s2_gemv, s2_gemv_q8, s_gemv, s_gemv_q8k, sampler, shared_expert | pass |
| iq_parity, ple_parity | `ctest` generates the `iq_parity` fixtures into the build tree (`iq_parity_fixtures`, needs python3 with numpy; skipped without). `ple_parity` needs the Q2_0 shard 2 GGUF and the dense pack: configure with `-DSTRATA_PLE_GGUF=<shard 2 .gguf>` (and `-DSTRATA_PLE_FIXTURE_DIR=<dir with ple_in.bin and ple_out.bin>`); a missing input is a hard error, as in the CUDA build (#1795) |
| qsa_prompt_attn_parity | needs the tensor-core kernel (below) |
| kv_hybrid_parity (0.1.27, hybrid K8V4 KV) | appends, gathers and attention pass; its last step is that same tensor-core kernel |

**The engine end to end (2026-09-29, later the same day).** It generates. Prompt "Write a Python function
that returns the n-th Fibonacci number", greedy, 64 tokens:

> `<think>` The user wants a Python function that returns the n-th Fibonacci number. Let me consider the
> options: 1. **Recursive approach** - Simple but O(2^n) time complexity, very slow for large n. 2.
> **Iterative approach** - O(n) time, O(1) ...

Measured on the same card, same 2,185-token prompt, 64 greedy tokens (AOT build, `--spec 2`,
`--no-prefill-borrow`), against llama.cpp's SYCL build serving the same GGUF:

| | llama.cpp SYCL | Strata SYCL port |
|---|---|---|
| prompt reading | 138-319 tok/s | **560 tok/s** (693 at 2,000 tokens, 180 at 300) |
| decode, suffix drafter only (`--spec 2`) | 24.2-26.0 tok/s | 20.2-20.8 tok/s |
| decode with the MTP draft layer (`--spec 4 --mtp`) | - | **42.7 tok/s** at the 2,185-token context (81% accepted, 2.9 tokens/round); 45.3 on a short prompt |

Prompt reading is where Strata's design pays (oneMKL GEMM over dequantised experts, the whole model on
the card). Decode is at ~80%: a speculative round costs ~53 ms whatever its size and the suffix drafter
is accepted 9% of the time, so nearly every round yields one token; `--spec 4` (16.9 tok/s) was worse
than `--spec 2` (20.8) for that reason. The kernels themselves run at 130-160 GB/s of weights on a card
that streams 600 GB/s (`sycl/probe/bw.cpp`, incompressible data) - the same class llama.cpp reaches -
and five variants of the hot Q6_K matvec (lanes per row, rows per warp, unroll, 16-byte loads, software
pipelining; `mmvq_bench`) all landed in that band. The draft source was the lever: the base
Qwen3.8-Flash-Next checkpoint's MTP layer (`tools/mtp_fetch.py fetch`, `mtp_pack.py --experts q2_0`,
`mtp_rt.py`; 4.9 GB downloaded, 809 MiB of VRAM) drafts for the Coder fine-tune at 87% acceptance:
3.4 tokens per 53 ms round, 45 tok/s, the same greedy tokens. The `mtp.cpp` path ran on the first try
under the same flags.

Without AOT the runtime JIT-compiles every kernel on first use, ~47 s the first time a process runs;
`SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=<dir>` keeps that across runs (15 MB).

How to run it by hand (a greedy test run, the way every number in this section was measured; inside the
`strata-sycl-dev` image, AOT build in `build-sycl-aot/`):

```sh
cd Strata
docker build --pull -t strata-sycl-dev sycl/tools     # once: about 6 minutes, a 13 GB image
H=$(basename "$PWD")
```

**Arc B-series** (Pro B70: `bmg-g31`; B580, B570 and Pro B60: `bmg-g21`; `ocloc ids bmg-g21` in the image lists device ids).
Ahead-of-time code, no compile at the first start:

```sh
docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e AOT=bmg-g31 -e REPO=/work/$H \
    -e BUILD_DIR=/work/$H/build-sycl-aot strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"
```

**Arc A-series** (Alchemist: A750, A770, A580, A380): no `AOT=`. The code is compiled from SPIR-V at the first start (JIT, about
a minute). Nobody has run an AOT build for these cards.

```sh
docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e REPO=/work/$H \
    -e BUILD_DIR=/work/$H/build-sycl strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"
```

The build takes about 3 minutes and ends with `BUILD EXIT 0` and `errors: 0`. Setup finds `build-sycl-aot/strata` or
`build-sycl/strata`. Run the `docker build` again after pulling an update that changes `sycl/tools/Dockerfile`, and the
`docker run` after every engine update (setup says when the engine is older than it needs).

| `build.sh` variable | Effect |
| --- | --- |
| `AOT=bmg-g31` or `bmg-g21` | Ahead-of-time device code. Empty: SPIR-V, JIT at first start. |
| `I8_ESIMD=OFF` | Build without the fused int8 ESIMD expert kernels (stubs; the prompt's experts take the slower XMX path, answers do not change). |
| `ALLOW_BAD_OCLOC=1` | Build although this `ocloc` is known to crash on the Battlemage kernels |
| `SPIN_MAX=N` | The bound on a device spin (`STRATA_SYCL_SPIN_MAX`); see [Things that matter](#things-that-matter-on-this-gpu) |

`--pull` matters because the image starts from a community image whose tag moves, and `docker build` reuses an old copy unless
told to fetch the current one. An old copy can hold a device compiler (ocloc from Compute Runtime 26.22.38646, IGC 2.36.3) that
crashes on the Battlemage kernels (`IGC: Internal Compiler Error: Floating point exception`, then `gen compiler command failed
with exit code 245`, #3219). `sycl/tools/build.sh` checks the version and stops with the fix.

## How to run it by hand

A greedy test run, inside the `strata-sycl-dev` image, AOT build in `build-sycl-aot/`:

```sh
STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1 \
build-sycl/strata --pack <iq pack> --native <shard1> --ple-gguf <shard2> \
    --expert-profile data/expert-profile-coder.bin --expert-cache auto --stream-experts \
    --prefill auto --spec 4 --spec-min-p 0.5 --max-context 8192 --tokens <ids> --max-new 64 --greedy
```

| Flag or variable | Meaning |
| --- | --- |
| `--stream-experts` | No resident host copy of the experts: every one goes from the GGUF into the VRAM cache through a small staging ring. The engine runs on 23 GiB of RAM; upstream needs 32 GB for this model. |
| `STRATA_VERIFY_DEVICE_PLAN=1` | The GPU plans each layer itself |
| `STRATA_VERIFY_NO_HOST=1` | The host waits for the whole window graph instead of per-layer rings. Required on `xe` whenever part of the experts is not in VRAM. |
| `--no-prefill-borrow` | The prompt path does not lend expert slots. The port borrows by default only above a 32K context (`--prefill-borrow` / `--no-prefill-borrow` decide explicitly). |

With `--tokens-file` instead of `--tokens` for long prompts (80K ids exceed Linux's single-argument limit). With a native pack
`--prefill-until N` skips the token loop, so compare output tokens between paths, never only timings.

Serve the port with `serve/server.py --engine strata` and `sycl/serve/strata-sycl.sh` as the `exe`, which starts the binary inside
the oneAPI runtime image with the serve pipes attached. Paths in the config's `args` are the container's. A config:

```json
{"engine": "strata", "exe": "<repo>/sycl/serve/strata-sycl.sh",
 "args": ["--pack", "/work/pack", "--native", "<shard 1>", "--ple-gguf", "<shard 2>",
          "--expert-profile", "data/expert-profile-coder.bin", "--expert-cache", "auto", "--stream-experts",
          "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", "/work/mtp/rt",
          "--max-context", "32768", "--kv", "int8", "--vram-reserve-mib", "1024"],
 "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.05}}
```

A native (IQ) pack takes `--spec` 2 at the smallest: the engine refuses `--spec 0` and `--spec 1` on one (it needs `--native SHARD1`, `--spec T`
with T >= 2 and `--prefill CHUNK`), so speculative decoding cannot be turned off for these packs.

With `--stream-experts` under `STRATA_VERIFY_NO_HOST=1` (the `strata-sycl.sh` default) there is no SSD expert tier: every expert
must be in VRAM or in the pinned RAM mirror (`STRATA_MIRROR_MIB`), and the engine refuses to start when the two do not cover the model
("REFUSED: N experts are neither in VRAM nor mirrored"). Raise the mirror or lower `--max-context`. So the reserve matters: on the B70
with Coder IQ1_M 1,024 MiB fits all 12,288 experts with 2 GB of VRAM free.

## Things that matter on this GPU

- **`SYCL_CACHE_PERSISTENT` must be 0.** The persistent JIT cache segfaults on Xe2 during the first compile. The start script
  sets it.
- **Do not ask for more VRAM than the card has.** On the `xe` driver an allocation past VRAM does not fail: the driver evicts
  buffers into host RAM, the kernel runs out of memory, and the machine can livelock until its hardware watchdog resets it.
  Compute the KV size first and leave about 1.5 GB free.
- **Thinking.** The model reasons before it answers. Put `{"reasoning_effort": "none"}` in
  `strata-<model>.shared-settings.json` next to the config, or send `chat_template_kwargs: {"enable_thinking": false}` per
  request. Otherwise a short `max_tokens` is spent inside the think block and the answer looks empty.
- **`/health` says 503 while loading;** the server polls `/props` instead.
- **Bounded spins.** A device spin that never sees its flag makes the `xe` driver time out the queue and reset the GT, and the
  card can stay wedged until a reboot. Every spin is capped. The cap is the build option `STRATA_SYCL_SPIN_MAX` (2,000,000
  reads, a few seconds, unless the build is an AOT build for a `bmg` card, which keeps 20,000).
- **JIT builds** add `-ze-opt-greater-than-4GB-buffer-required` so kernels keep 64-bit addresses past 4 GiB into an allocation.
  oneMKL's own precompiled kernels do not get it; the GEMM stages an operand past 4 GiB (`STRATA_SYCL_GEMM_STAGE=1` forces
  that on Battlemage).
- **Aliased pages in big allocations (xe).** The driver can return a large arena with two 2 MiB pages mapped onto the same
  memory. Every allocation of 32 MiB or more goes through `strata::malloc_device_guarded`, which checks the range and allocates
  again behind a spacer until it is clean (a warning names each retry; under 0.5 s at startup; `STRATA_ARENA_ALIAS_CHECK=0`
  skips it). `STRATA_VERIFY_ALL_SLOTS=1` reads every filled cache slot back and compares it with the GGUF.
- **Monitor tab.** `server_intel.py` plugs `sycl/serve/xe_telemetry.py` into the Monitor (temperature, power, PCIe link from
  sysfs). Load and VRAM come from `/run/gpustat.json`, written by a small root sampler (`sycl/tools/gpustat.py`, installed with
  `gpustat.service`), because `xe` reports them per client in `/proc/*/fdinfo`, which only root can read. A run config may name
  a `model_switcher`; the web app's header then gets a Model menu.
- **Logprobs.** `/v1/chat/completions` takes `"logprobs": true` and `"top_logprobs": 0..20` through `server_intel.py`, streamed
  or not. Not on `/v1/completions` or the Anthropic endpoint.

## How much context fits

Only 12 of the 48 layers are full attention, so the KV grows slowly. KV streaming (`--kv-resident`) keeps the whole KV in pinned
host memory and only the attended window in VRAM; setup turns it on from 64K and keeps INT8 KV. `--kv k8v4` does not support
streaming: setup keeps its KV in VRAM, which pushes more experts into the host mirror.

> **Measured on:** Arc Pro B70 32 GB, Coder IQ1_M, engine 0.1.32-sycl, setup's flags (`--kv-resident 32768
> --vram-reserve-mib 2048 --prefill 4096`, the prompt borrowing cache slots), the same text repeated to length, 256 greedy tokens.

| Context | KV | Prompt | Decode after | Peak VRAM |
| --- | --- | ---: | ---: | ---: |
| 128K | int8 | 888 tok/s (144 s) | 66.6 tok/s | 29.9 GB |
| 128K | q4_0 | 848 tok/s | 62.8 tok/s | 29.7 GB |
| 128K | k8v4, KV in VRAM | 1,006 tok/s | 66.8 tok/s | 30.7 GB |
| 256K | int8 | 757 tok/s (338 s) | 55.9 tok/s | 30.2 GB |
| 256K | q4_0 | 737 tok/s | 51.7 tok/s | 30.0 GB |
| 256K | k8v4, KV in VRAM | 823 tok/s | 52.2 tok/s | 30.9 GB |

Borrowing keeps every expert in VRAM (the prompt buffers live in lent cache slots, refilled after the prompt). Without it
(`--no-prefill-borrow`) about 1,000-2,300 experts move to the pinned host mirror: the prompt is 2-5% faster and decode after it
10-23% slower.

## Arc Pro B70 with a model that does not fit: Flash-Next IQ3_S

IQ3_S has 24,576 experts of 1.97 MB (46 GB). The 32 GB card holds about 11.5k of them (22 GiB of cache) and the rest sit in a
pinned host mirror that the GPU reads over PCIe. Every prompt chunk streams the experts the cache does not hold, so a 4K prompt
moves about 26 GB; bigger chunks read faster.

- `setup_intel.py` writes `--prefill 4096` for a card with 24 GB or more (the chunk's buffers take about 700 cache slots).
  `--prefill 8192 --prefill-borrow` lends cache slots instead and keeps all of them for decode, at about 1 s of refill per
  prompt. Setup recommends 8,192 for a Battlemage card with 24 GB or more above 32K context.
- `STRATA_PREFILL_FIRST` (a short first chunk that starts the GPU while PLE rows are read) defaults to 256 tokens only when the
  cache holds every expert, else 0.
- `STRATA_VERIFY_NO_HOST=1` is required whenever part of the experts is not in VRAM (`strata-sycl.sh` sets it). Without it the
  host/GPU handshake is not visible across the bus on `xe` and the logits turn NaN a few tokens in.
- On `xe`, setup writes `data/expert-profile-sycl.bin`, a ranking built from the routing of 45 varied prompts, so the cache
  holds the experts a chat, code or non-English prompt uses (`tools/make_profile_mix.py`, `tools/collect_profile_traces.py`).
  An `i915` card keeps the shipped profile.
- The prompt's expert copies run on the card's copy engine on Battlemage (`STRATA_COPY_ENGINE=1` forces it elsewhere, `=0`
  turns it off). The experts in the last 10% of the cache slots are also mirrored so that lent slots refill by DMA
  (`STRATA_LEND_MIRROR=0` off).

> **Measured on:** Arc Pro B70 32 GB (xe, kernel 7.0, oneAPI 2026.1, PCIe 13.3 GB/s host to device), 60 GB RAM, Flash-Next IQ3_S,
> 11.5k experts resident, greedy, `--spec 4 --spec-min-p 0.5 --mtp`, INT8 KV, `--stream-experts --vram-reserve-mib 1024
> --ple-io ram`, engine 0.1.40-sycl to 0.1.43. The 10-prompt `mg_norepeat` gate passes 10 of 10.

| Workload | Result |
| --- | --- |
| 4,095-token prompt, `--prefill 4096` | 980-1,002 tok/s |
| 8,169-token prompt, `--prefill 8192` lending cache slots | 1,117 tok/s (9.4k slots) |
| Decode, 200-token story (66% of drafts accepted) | 30.5 tok/s |
| Decode, 200-token code answer (90% accepted) | 40.8 tok/s |
| 128K config, 4K prompt incl. refill (0.1.43) | 3.3 s (was 7.2 s); 16K 13.5 s (was 25.5); 49K 44 s (was 78) |
| `--prefill 8192` against 4096, 16K / 32K prompt (IQ3_S, setup's 128K flags) | 1,347-1,362 against 1,222 tok/s (+10-11%); 1,361-1,383 against 1,187 (+15-16%); decode after unchanged |
| Fused int8 expert prompt path (`auto`) against FP16 chain, 1,024 / 4,095 tokens | 783.5 against 645.8 (+21%); 1,407.7 against 1,257.0 (+12%) |

Draft length (200 tokens, `--spec-min-p 0.5`): `--spec 6` against `--spec 4` is +6.5% on code and -1.5% on prose, so the setup
default (4, 0.5) stays. A `--prefill 8192` chunk is not bit-identical to 4096 (first-token KL 1.1e-2 at 16K, top-1 the same).

**Fused int8 expert path (Battlemage default).** On a Battlemage GPU (B70, B580, B570, detected by device architecture) the
engine computes the prompt's routed experts with ESIMD kernels that decode the quantized weights in registers into the s8 DPAS B
operand and multiply them by q8_1 activations. Covered: gate/up IQ3_S, IQ3_XXS, IQ2_S and down IQ4_NL, Q2_0. Other formats keep
the FP16 path, as does every other GPU. `STRATA_PREFILL_MMQ` unset is `auto`, which times both chains at the first layer of each
format and takes int8 below the measured crossover (130-250 rows per expert); `=0` turns it off, `=1` forces it. Activations are
quantized, so bits differ from the FP16 path (KL 2e-4 nats mean over 6,002 positions, top-1 99.7%). Details:
[INTEL_INT8_EXPERTS.md](INTEL_INT8_EXPERTS.md).

## Arc A750 and the other Alchemist cards

An Alchemist card (A750, A770 and so on) has no FP64 hardware, and the A750 runs on the `i915` driver. `./setup.sh --backend sycl`
writes a different config for an `i915` card, and a 300 MiB VRAM reserve and `--draft-vocab en` for any card under 12 GB.

| | Arc Pro B70 (`xe`, 32 GB) | Arc A750 (`i915`, 8 GB) |
| --- | --- | --- |
| Experts | `--stream-experts`: from the GGUF into VRAM, the rest in a pinned RAM mirror | No `--stream-experts`: all of them in a RAM arena (about 40 GiB for IQ3_XXS), a few hundred in VRAM, the CPU computes the others |
| `STRATA_VERIFY_NO_HOST` | 1 (the wrapper's default) | 0, written to the config's `"env"` |
| `--vram-reserve-mib` | 1024 / 2048 | 300 |
| `--ple-io` | `ram` | `ram` |
| FP64 | Native | Emulated: `IGC_EnableDPEmulation=1`, `OverrideDefaultFP64Settings=1`, `NEOReadDebugKeys=1` in `"env"` |
| Build | AOT `bmg-g31` | JIT (no `AOT=`) |

- **Why no `--stream-experts` on the A750.** The mirror is one pinned host allocation, and a single `sycl::malloc_host` above a few
  GB fails on the A750. Without the flag the engine loads the experts into an ordinary RAM arena and the CPU computes the experts
  that are not in the 0.67 GiB cache. That needs about 43 GB of RAM for IQ3_XXS plus the 29 GB n-gram table with `--ple-io ram`.
  Setup warns when the RAM is less than the model's.
- **Why `STRATA_VERIFY_NO_HOST=0`.** With it set the GPU plans each layer itself and the CPU's experts are never asked for; the
  window hangs (`Fence expiration time out` in the i915 log).
- **FP64.** One kernel on the sampled path declared `double`; the sampler's tail now accumulates in float
  (`-DSTRATA_SYCL_SAMPLER_FP64=1` keeps double). Setup still writes the three emulation keys for an A-series card, because other
  kernels that use double were not exercised without them on long contexts or with vision. On the A750 a sampled decode ran at
  12.0 tok/s without the emulation keys against 9.8 with them (median of 4 interleaved pairs).
- **Prompt GDN recurrence** runs a sub-group kernel on Alchemist (variant 6, `STRATA_GDN_XE`): on by default there, `=0` off, `=1`
  on any Intel GPU with 16-lane sub-groups. On Battlemage it stays off (-3.5%).
- **Grouped XMX expert launch** (`STRATA_PF_XMX=2`) is the default on Alchemist: the prompt path decodes and multiplies a group of
  experts in one fused launch on the XMX units. `=0` turns it off; `=1` dequantizes first and runs one grouped GEMM per product.
  Battlemage keeps the older path (the same kernels are 7-11% slower there).
- **Streaming copies.** Only experts in USM host memory (the `--stream-experts` mirror) are copied directly; every other expert
  goes through the stager's pinned buffers, because SYCL has no `cudaHostRegister` and a direct copy out of pageable memory could
  hang a prompt. `STRATA_PF_DMA_PAGEABLE=1` restores the direct copy.
- The engine segfaults in a worker thread of the CPU pool when it exits (dmesg only, after the server printed "stopped"). It is
  not a GPU event.

> **Measured on:** Arc A750 8 GB (i915, PCIe 4.0), Ryzen 5 5600X (AVX2, no AVX-512), 64 GB RAM, Flash-Next IQ3_XXS, default
> flags, engine 0.1.45.3.

| Workload | Result |
| --- | --- |
| Prompt, 1K / 2K / 4K / 6K tokens, grouped XMX launch | 46.4 / 83.1 / 110.4 / 114.8 tok/s (before: 35.1 / 61.7 / 79.3 / 66.3) |
| Decode | 18.7 tok/s (unchanged by the grouped launch); 10-15 tok/s in the first report |
| PCIe probe | 10.6 GB/s |
| Quality of the grouped launch against the default | teacher-forced KL 0.0002 over 6,006 tokens (default against itself 0.0001), first-token KL 0.0017, top-1 99.7%; HumanEval (every 2nd, 82) 73, 69, 74; needles at 2K-6K 9/9; 8K-context tool-call agent 6/6 |

**Gyro on an Arc.** Gyro-S and Gyro-M (trellis-coded experts, [GYRO.md](GYRO.md#intel-arc-sycl)) run on the SYCL engine. They were
tested on the A750 only. Setup writes the usual A750 config (32K context, 300 MiB reserve) without `--ple-io ram`, because the
28 GB n-gram table stays on the SSD while the expert arena takes 24 GB of RAM (Gyro-S). Gyro-M needs `--max-context 8192
--vram-reserve-mib 100 --prefill 1024` on 8 GB. A second Arc card does not take Gyro's experts (the engine refuses and says so).
The Battlemage AOT build compiles with the trellis kernels and was not run.

> **Measured on:** Arc A750 (i915, 8 GB), Ryzen 5 5600X, 62 GB RAM, Gyro-S, engine version not recorded.

| Workload | Result |
| --- | ---: |
| Prompt, 1,406 tokens | 82 tok/s (40 with dequantize + oneMKL) |
| Prompt, 2,377 / 20,000 tokens | 102 / 195 tok/s |
| Decode, MTP draft of 4 | 15-16 tok/s |
| Experts in the card's cache | 410 of 24,576 |
| Gyro-M, 8K context | decode 14 tok/s, prompt 62 tok/s |

Decode is bound by the CPU pool and the PCIe link (10.4 GB/s), not by the card's kernels. `--prefill 1024` gave +13% decode and
-24% prompt speed; the default stays `--prefill auto`.

## Pictures on an Arc

<a id="pictures-on-an-arc-2026-10-19-branch-f-021-arcvis"></a>
Pictures follow the rule of every other card ([MODELS.md](MODELS.md#the-versions)): the picture reader runs on the CPU under 24 GB
of VRAM and on the card from 24 GB. `strata_control/vision_reader.py` is the one place for that rule. On an Arc, pictures are on by
default with the CPU reader, a B-series card included. An Arc of 24 GB or more also keeps the CPU reader; `--vision gpu` asks for
the GPU reader.

- **CPU reader:** `strata-vision` (llama.cpp's mtmd). Setup compiles it into `engine-sycl/`, the server runs it on the host, and
  the engine in the container reads the embeddings from the host's `/dev/shm`, which `strata-sycl.sh` shares
  (`STRATA_SYCL_SHM=0` keeps a private one).
- **SYCL reader:** `strata-vision-sycl`, the same program on llama.cpp's SYCL backend. `setup --backend sycl --vision gpu`
  compiles it on the host when the oneAPI compiler (`icx`, `icpx`, `intel-oneapi-mkl-devel`) is installed; otherwise setup says so
  and keeps the CPU reader. It needs the PC's oneAPI runtime, not the container. If it does not start, the server falls back to
  the CPU reader with one log line. It takes VRAM the experts would use, which is why the rule gives it to 24 GB cards only. Nobody
  has run it on a card of 24 GB or more.
- **Known risk on Battlemage.** The first picture on the Arc Pro B70 (engine 0.1.45, IQ3_S, CPU reader) logged an xe engine
  reset (ccs + bcs) and `Fault response: Unsuccessful -EINVAL`, and the request ended 503. It has not been looked at since.
  Setup and `strata launch` print one plain line that the card is new to Strata and how to report a problem. `--vision no`
  (setup) or `--vision off` (`strata launch`) turns pictures off.

> **Measured on:** Arc A750 (8 GB), IQ3_XXS, 8K context, JIT build, 448x448 picture.

| Reader | Time per picture |
| --- | ---: |
| CPU (`strata-vision`) | 1.73 s |
| SYCL (`strata-vision-sycl`), after a 30 s first start that compiles the kernels | 0.16 s |

The embeddings match the CPU reader's (cosine 0.9959 to 0.9999 over 8 pictures). A test picture (red circle, blue square, the words
"STRATA 42") was answered correctly by both readers 10 of 10; two pictures in one request, a picture kept in a second turn's history,
and text requests after the pictures all answered, with no GPU fault in the kernel log.

## The SYCL port (`sycl/`)

**Layout.** `sycl/src` and `sycl/include` mirror the tree and hold only the files the port changes; `sycl/CMakeLists.txt` takes every
other source from the original location. `sycl/include/dpct/` is the vendored SYCLomatic helper library.

**How it is made, so it can be redone.**

1. `sycl/tools/Dockerfile` is the dev image: the llama.cpp SYCL image plus SYCLomatic (`dpct` 2025.3), ninja, and the CUDA 12.8
   headers that `sycl/tools/get-cuda-headers.sh` pulls from NVIDIA's pip wheels (dpct parses CUDA; it needs headers, not the toolkit).
2. `sycl/tools/migrate.sh` writes a compilation database for the CUDA-touching translation units and runs dpct over them.
3. `sycl/tools/fixups.py` fixes what dpct got wrong, as an idempotent script with a reason per item. Examples: CUDA's null stream
   becomes the default stream (every stream cast goes through `strata::q_of()`), `__ldg` and `__fadd_rn` translations that lost
   their meaning, ggml lookup tables as plain `static const` arrays.
4. `sycl/tools/build.sh` configures and builds with icpx inside the image. Two compiler flags are load-bearing:
   `-fp-model=precise` (icpx defaults to a fast model) and `-cl-fp32-correctly-rounded-divide-sqrt` for the device compiler. The
   Arc's fp32 divide is not correctly rounded by default, and Strata's quantizers are byte-exact against ggml through `amax / 127`.

**What the port had to get right.**

- Host-mapped flags: `volatile` device loads do not bypass the caches on Intel; system-scope atomics do. Host-to-device visibility
  during a kernel stays unreliable, which is why the all-resident path waits for the window instead.
- 32-lane sub-groups (`-fsycl-default-sub-group-size=32`): the kernels are written for warps.
- Synchronous copies: `cudaMemcpy` blocks, dpct's default-queue `memcpy` did not wait.
- Stager and PLE waits: no host thread waits on a queue's event under the Level Zero v2 adapter; the copy queue writes a sequence
  number into page-locked host memory and the stager polls it.
- Profiling: `sycl/tools/Dockerfile.unitrace` builds the dev image with Intel's unitrace; `unitrace -d` around the engine plus
  `sycl/rank_kernels.py` gives device time per kernel.

**Keeping up with upstream.** A merge of upstream `main` leaves the copies in `sycl/` behind wherever upstream touched a file they
mirror. They are refreshed by re-migration:

1. Merge upstream. No shared file should conflict. Check that `sycl/setup_intel.py --check` and
   `sycl/serve/server_intel.py --engine mock` still run against the new `setup.py` and `server.py`.
2. Migrate the old upstream tree with `migrate.sh` into `sycl-base` and the merged tree into `sycl-new`.
3. Run `sycl/tools/normalize.sh <dir> <commit>` on both (dpct's file names, unchanged files, message serials, then `fixups.py`).
4. For every file present in both: `git merge-file sycl/F sycl-base/F sycl-new/F`. Copies dpct never produced (`verify.cpp`,
   `mtp.cpp`, `remote_experts.cpp`) take upstream's diff by hand; new parity tests are copied in and listed in `sycl/CMakeLists.txt`.
5. A fixup whose pattern upstream changed shows up as a compile error; extend the fixup, re-run it, rebuild. Run a symbol audit
   (count the port's feature identifiers before and after).

`sycl/tools/merge_upstream.py` ports open upstream PRs ahead of upstream the same way.

## Arc Pro B70: cheaper IQ decodes in the grouped expert kernels (`STRATA_IQ_DECODE_FAST=1`, opt-in, 2026-10-16)

The decode kernels that read the routed experts (`native_gu_kernel` / `native_down_kernel` in `sycl/src/kernels/cuda/iq_kernels.dp.cpp`,
8 lanes a row) are ALU-bound on this card, not bandwidth-bound: an ISA dump (IGC, AOT `bmg-g31`) of the IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S,
IQ2_S gate/up loops and the Q2_0 / IQ4_NL down loops shows 280-480 instructions per 32-weight part for the part's 8 dp4a, with the kernels
reading experts at 19-44% of DRAM bandwidth. Most of it is the dpct per-byte emulation of `byte_level_permute` in the Q2_0 dot (476 of the
instructions, 157 of them `mov`), the compare/select chain of the IQ4_NL codebook lookup, the popcount/multiply ksigns unpacking and the
64-bit `kbx * sizeof(block)` address products each iteration.

`STRATA_IQ_DECODE_FAST=1` switches the seven formats to kernels that produce the same bits with less of that: the ksigns masks from a
2 KB table (IQ2_XXS, IQ2_XS), the signed grid byte as `(g ^ m) + (m & 0x01010101)` (exact because every byte of the IQ grids is 1..62),
the Q2_0 fields spread by shifts and masks and biased to {-1, 0, 1, 2} in one add and xor, the IQ4_NL codebook through four 256-entry
position tables, a loop that walks block pointers, two rows per lane in 16-lane sub-groups (the one-row SIMD32 form spilled registers and
measured slower), and a multi-entry form for IQ2_XXS (the current kernel decodes it once per entry). Every part yields the same integers to
the same `finish` in the same float order as the current `Multi<TY>` / `vec_dot_*` dots and keeps each entry's lane-strided k order, so the
outputs are bit for bit the current kernels': `native_grouped_parity --iq` (ctest `native_grouped_parity_iq_decode`) compares the two on
random blobs, every IQ gate/up format with both IQ down formats, groups of 1..6 entries, at 512 x 256 and at Flash-Next's 2560 x 640.
`STRATA_IQ_DECODE_FAST_FMTS=16,42` (ggml type ids) restricts it to some formats for an A/B. Off by default; nothing changes without the flag.

Measured 2026-10-16 on a B70 (card 1 of two, xe, oneAPI 2026.1 AOT) with a grouped-call microbenchmark (48 layers per graph, each its
own experts streaming from VRAM, us per call, median of 5 x 3 replays; the shapes are the per-card expert calls of a 2-card layer-split
serve at 1, 2 and 4 concurrent streams, 10 to 32 groups of 1 to 4 entries), new kernels against the current ones, bitwise equal in every row:

| gate/up / down | 1 stream, card 0 / card 1 | 2 streams | 4 streams |
|---|---|---|---|
| IQ2_XXS / Q2_0 | 114.8 -> 59.8 / 135.1 -> 83.8 | 158.7 -> 85.8 / 192.3 -> 121.5 | 213.1 -> 112.3 / 248.7 -> 147.3 |
| IQ3_S / Q2_0 | 105.1 -> 70.3 / 147.9 -> 100.1 | 147.4 -> 99.3 / 215.2 -> 147.2 | 194.8 -> 130.8 / 256.9 -> 175.1 |
| IQ2_S / Q2_0 | 97.0 -> 67.2 / 135.5 -> 91.5 | 136.4 -> 94.5 / 195.5 -> 133.1 | 181.1 -> 123.1 / 237.3 -> 161.9 |
| IQ2_XS / IQ4_NL | (not run) / 135.6 -> 119.6 | 138.1 -> 117.7 / 196.7 -> 169.9 | (not run) / 242.6 -> 205.4 |
| IQ3_XXS / IQ4_NL | (not run) / 146.6 -> 127.9 | 142.7 -> 126.0 / 208.2 -> 183.9 | (not run) / 253.4 -> 221.3 |

The Q2_0 down product alone went 63.7 -> 26.4 us (card 0) and 92.3 -> 35.2 (card 1) at 2 streams, IQ4_NL down 64.9 -> 55.8 and
95.9 -> 80.3. Those numbers were taken on a development tree with other decode changes stacked on it.

End to end on this tree, 2026-10-16: two Arc Pro B70 (32 GB, kernel 7.1, Compute Runtime 26.22.38646.4, oneAPI 2026.1, AOT `bmg-g31`),
Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, `--layer-split 30 --prefill 8192 --max-context 65536 --kv int8 --spec 4 --mtp <BF16 draft layer>
--mtp-max-t 4 --spec-min-p 0 --stream-experts --expert-cache auto`, one slot, greedy, a fixed prompt with a 512-token coding reply
(`--max-context 200000` does not fit all 24,576 experts on a plain two-card layer split, so 65536). Decode: main 81.8 / 79.5 / 78.7 tok/s
(3 runs, median 79.5); this branch with the flag unset 81.7 / 79.3; with `STRATA_IQ_DECODE_FAST=1` 88.1 / 85.6 / 84.7 (median 85.6,
+7.7%). Output tokens and draft acceptance counts identical in every run, flag on or off, so the gain is kernel time alone. A turn at a
40,796-token context: 76.0 -> 81.4 tok/s (+7.1%), identical output. Prompt reading unchanged (4K 1189 -> 1192 tok/s, 32K 1956-2004 ->
1964-2009). `native_grouped_parity --iq`: 0 failures (15 cases bitwise equal), plain `native_grouped_parity`: 0 failures; per-card health
after every stop passed. Two streams were not tested (main does not take `--batch-mtp` on a layer split).
## Arc Pro B70: KREG decode kernels for Q4_K, Q5_K, Q6_K, IQ4_XS (opt-in `STRATA_MMVQ_KREG=1`, 2026-10-16)

`STRATA_MMVQ_KREG=1` runs `native_mmvq` for Q4_K, Q5_K, Q6_K and IQ4_XS at 1-8 columns (the dense projections, the
verify window and the output head) through KREG kernels. One work-group covers a row's whole K: each lane loads its
32-value slice of every column's activations once, into registers, then the group streams tiles of 2-8 rows (one or
two 16-byte weight loads per lane and row), a sub-group butterfly and a sum across sub-groups in local memory. IQ4_XS
looks its codebook up in a 256-entry local-memory table. The wide kernels it replaces re-read every column's
activations for every row and are bound by instruction issue, not by memory: Q4_K 2560 x 10240 at 6 columns took 82 us
with the matrix held in L2 and 94 us from DRAM.

Not bitwise equal to the default kernels (another summation order). A T-column call is bitwise equal to T one-column
calls, as with the default kernels (`native_mmvq_set_multi_exact`). Unset, nothing on the default path changes: the
flag is the first check of the new branch. Matrices that are not 16-byte (Q4_K, Q5_K) or 8-byte (IQ4_XS) aligned and
rows over 32,768 values keep the default kernels. `mmvq_kreg_parity --selftest` checks it against a float64 reference,
the default kernels and itself one column at a time: on the B70 all 160 cases pass, KREG and the default kernels both
within 5e-8 of sum |w x| of the reference, every multi-column output bitwise equal to the one-column calls.
`mmvq_kreg_parity --bench` times it.

Kernel time, us per call, default -> KREG (`mmvq_kreg_parity --bench`: B70, random weights, distinct copies per graph
so L2 never holds a matrix, median of 7-11 graph replays):

| format, shape (n_in x n_out) | 1 column | 2 | 4 | 6 |
|---|---|---|---|---|
| Q6_K 2560 x 10240 | 55.7 -> 41.0 | 63.0 -> 41.5 | 73.4 -> 46.0 | 88.9 -> 57.7 |
| Q6_K 6144 x 2560 | 47.3 -> 28.1 | 50.3 -> 27.8 | 56.1 -> 30.7 | 66.7 -> 37.6 |
| Q5_K 2560 x 10240 | 49.2 -> 34.7 | 61.3 -> 35.9 | 85.3 -> 40.1 | 113.2 -> 52.0 |
| Q5_K head 2560 x 248320 | 1063 -> 759 | 1374 -> 775 | 2013 -> 893 | 2691 -> 1175 |
| Q4_K 2560 x 10240 | 44.5 -> 28.8 | 55.6 -> 29.6 | 78.8 -> 31.9 | 107.9 -> 40.5 |
| IQ4_XS 2560 x 10240 | 42.9 -> 33.5 | 47.9 -> 37.8 | 55.9 -> 41.2 | 67.6 -> 49.3 |

Over the large shapes (2560 x 6144 / 10240 / 12288, 6144 x 2560, the 248,320-row head) the K-quant kernels stream
74-97% of the card's 608 GB/s at 1-2 columns and 47-66% at 6 (default kernels 31-72% and 18-41%); IQ4_XS 53-81% and
40-50% (default 41-54% and 30-34%), ALU-bound in the codebook decode. The small matrices (2560 x 512 / 640) sit near
the 4-6 us launch floor: K-quants 1.2-2.2x faster, IQ4_XS 0.90-0.98x at 1 column and 1.14-1.34x at 2-6 (KREG takes
every column count so that a multi-column call stays bitwise equal to one-column calls).

Engine, upstream main 296b59d2 against this change: two B70s, kernel 7.1, Compute Runtime 26.22.38646.4, oneAPI
2026.1, AOT bmg-g31; Qwen3.8-Flash-Next ISTA GSQ-RCO IQ3_XXS, `--layer-split 30 --prefill 8192 --max-context 65536
--kv int8 --spec 4 --mtp <BF16 draft layer> --mtp-max-t 4 --spec-min-p 0 --stream-experts --expert-cache auto`, one
slot, greedy, a fixed prompt with a 512-token coding reply, 3 runs each:

| | decode tok/s | tokens per verify window | ms per window |
|---|---|---|---|
| main | 81.8 / 79.5 / 78.7 | 3.22 / 3.14 / 3.12 | 39.4 / 39.5 / 39.7 |
| `STRATA_MMVQ_KREG` unset | 81.9 / 79.4 | the same as main | - |
| `STRATA_MMVQ_KREG=1` | 95.0 / 93.1 / 96.3 | 3.35 / 3.20 / 3.33 | 35.2 / 34.4 / 34.5 |

Median decode 79.5 -> 95.0 tok/s (+19.5%). A verify window costs about 12% less (39.5 -> 34.5 ms), the part that does
not depend on the drafts; the rest is the changed output drafting a little better on this prompt. With the flag unset
the output and the draft counts are identical to main (both runs, and a turn at a 40,796-token context); with it the
output differs from main. At a 40,796-token context: 76.0 -> 83.4 tok/s. Prompt reading is unchanged (4K 1,189 ->
1,191 tok/s, 32K 1,956-2,004 -> 1,961-2,008). Not measured: two streams (main refuses `--batch-mtp` on a layer split),
`--max-context 200000` (a plain two-card layer split of main does not fit all 24,576 experts there: 710 refused).
Measured on the B70 only; the A-series cards are not tested.
## `STRATA_HC_Q8=1` on SYCL: the Q8_0 hyper-connection read (2026-10-16)

`STRATA_HC_Q8=1` (docs/QUANTS.md: the hyper-connection down / up projections read as Q8_0 instead of BF16, an opt-in
on CUDA, the default on AMD gfx12) did nothing on the SYCL port: the loader kept a Q8_0 copy only when the GGUF
stored the projections as Q8_0 (the Unsloth files; the ISTA files store them as BF16, so nothing was copied and the
log said `0.00 GiB`), and the port's `fused_gr_read_multi` never had the Q8_0 read - it ignored `q8_down` / `q8_up`
and read the BF16 copies either way. The flag cost VRAM on the Unsloth files and changed nothing on any file.

Now, with the flag, the loader quantizes BF16 projections to Q8_0 at load (ggml's rounding, `d = amax / 127` as fp16,
on the host; a GGUF's own Q8_0 blocks are taken as they are) and the read uses them: two kernels, one row per
16-lane sub-group lane, the per-stream norm and the partial sums folded into the down kernel (the last work-group of
a row group to arrive finishes it, so no co-residency is assumed), K split over the sub-groups in the up kernel.
The inject rows stay BF16 (`STRATA_HC_Q8_INJECT` has no effect here). The default (flag unset) is the same code as
before: the Q8_0 pointers are null and the hook returns to the BF16 read; outputs and draft counts were identical to
`main` on the fixed prompts below.

**VRAM.** The BF16 copies stay in the arena (the prompt path's GEMMs read them), so on this port the flag ADDS the
Q8_0 bytes instead of halving them as on a single CUDA GPU: 0.63 GiB for Flash-Next (48 layers x 4 projections + the
final mixer, 3.32 MiB each; on the two-card layer split below 0.40 + 0.24 GiB), which comes out of the expert cache.
On 2x B70 with `--layer-split 30` the model that ran at `--max-context 65536` without the flag refused 310 experts
with it (132 at 32768 with the default 1,024 MiB reserve); the comparison below ran at `--max-context 32768
--vram-reserve-mib 512`. Dropping the BF16 copies (CUDA's `add_hc_skip` / `hc_bf16_view`) is not ported.

**What changes in the output.** The Q8_0 read is NOT bitwise the BF16 read - it reads Q8_0-rounded weights and sums in
another order, as `STRATA_HC_Q8=1` does on CUDA (`fused_gr.cu`, "an output-changing step"). `gr_q8_parity
--selftest` checks it against an FP64 reference computed from the very Q8_0 bytes it reads (T = 1..8, with and
without the pending write and the inject rows): lo / rs / inject / mixed within FP32 summation rounding (1e-4 of the
largest value), R_out bitwise the BF16 read's, a captured graph replayed twice bitwise the direct call, arguments not
at the callers' fixed strides fall back to the BF16 read bitwise; 33 of 33 cases pass on the B70. On the model, the
teacher-forced NLL of 4,762 tokens of three texts through the decode windows (same targets, same binary) was 3.1974
with the flag unset and 3.1699 with it set: no degradation measured on this gate (the gate's own spread between
rounding-only variants is of the order of 0.01-0.1 nats per token, so this is not a measured improvement either).

**Speed** (2x Arc Pro B70 32 GB, kernel 7.1, Compute Runtime 26.22.38646.4, oneAPI 2026.1, AOT bmg-g31; Flash-Next
ISTA GSQ-RCO IQ3_XXS, `--layer-split 30 --prefill 8192 --kv int8 --spec 4 --mtp <BF16 draft layer> --mtp-max-t 4
--spec-min-p 0 --stream-experts --expert-cache auto --max-context 32768 --vram-reserve-mib 512`, one slot, greedy,
a fixed prompt with a 512-token coding reply, 3 runs, same binary, flag unset vs set): the verify window 39.4 / 39.6 /
39.8 -> 37.4 / 37.5 / 37.4 ms (-5.3%, independent of draft acceptance); decode 81.8 / 79.4 / 78.5 -> 85.0 / 88.0 /
86.0 tok/s (median 79.4 -> 86.0, +8.3%, part of it a different acceptance of the changed output: drafts accepted
353/474, 349/489, 348/492 -> 351/489, 357/471, 353/483); prompt reading unchanged (4K 1,093 / 1,190 -> 1,093 / 1,188
tok/s, 16K 1,774 / 1,749 -> 1,769 / 1,750). Kernel level (one card, graph replay, random weights not L2-resident, us per
read, BF16 -> Q8_0): T = 1 38.7 -> 24.7, T = 4 50.8 -> 29.3, T = 6 62.0 -> 34.4. The down kernel's shape (4 rows per
lane x 32 columns per sub-group) was the fastest of (1..8 rows) x (32..160 columns) tried.

Switches: `STRATA_HC_Q8=1` on (unset or `0`: off, the BF16 read as before).

## `STRATA_GR_FUSED=1` on SYCL: the two-kernel BF16 hyper-connection read (2026-10-16, opt-in)

The port's default hyper-connection read is four launches per layer half (the norm per token and stream, the sliced
down projection, its reduce, the up projection); at T = 6 rows they take 5.8 + 22.7 + 2.7 + 30.2 us on the B70, and
the down / up kernels grow ~1.7 / 3 us per extra row - latency and issue, not DRAM (the weights are 13.2 MB a read).
`STRATA_GR_FUSED=1` runs the BF16 weights through the structure of the Q8_0 read above: one row per 16-lane
sub-group lane with the activations uniform across the sub-group, the norm and the partial reduction folded into the
down kernel (the last work-group of a row group to arrive finishes it; nothing waits), K split over the sub-groups in
the up kernel. Two launches instead of four. The weights are the same BF16 bytes; no extra VRAM. With `STRATA_HC_Q8=1`
the Q8_0 read runs instead (it has the pointers; this flag then changes nothing). Unset, the default read runs
unchanged: outputs and draft counts identical to `main` on the fixed prompts below.

**What changes in the output.** NOT bitwise the default read. The per-element arithmetic is the default read's (the
same fma chains, the same epilogue: R_out, the pending write, IS bitwise), but the down projection and the up
projection are summed in another order (slabs of 1,280 columns, 16 sub-group partials; K quarters), so lo, rs and
mixed differ at FP32 rounding level - the same kind of change as upstream's opt-in `STRATA_GR_V3` read.
`gr_fused_parity --selftest` checks it (T = 1..8, with and without the pending write and the inject rows): both reads
within 1e-4 of the largest value of an FP64 reference and of each other, R_out bitwise, a captured graph replayed
twice bitwise the direct call, arguments not at the callers' fixed strides fall back to the default read bitwise; 33
of 33 cases pass on the B70. Greedy outputs on the model differ from the default read's after a near-tie, as every
summation-order change does.

**Speed** (2x Arc Pro B70 32 GB, kernel 7.1, Compute Runtime 26.22.38646.4, oneAPI 2026.1, AOT bmg-g31; Flash-Next
ISTA GSQ-RCO IQ3_XXS, `--layer-split 30 --prefill 8192 --kv int8 --spec 4 --mtp <BF16 draft layer> --mtp-max-t 4
--spec-min-p 0 --stream-experts --expert-cache auto --max-context 65536`, one slot, greedy, a fixed prompt with a
512-token coding reply, 3 runs, `main` vs this read): the verify window 39.4 / 39.5 / 39.7 -> 38.8 / 38.1 / 38.7 ms
(-2.5%, independent of draft acceptance - the effect of the read); decode 81.8 / 79.5 / 78.7 -> 86.8 / 79.0 / 88.3
tok/s, whose spread is mostly a different draft acceptance of the changed output (drafts accepted 360/462, 342/510,
362/456); a turn at a 40,796-token context 76.0 -> 79.4 tok/s (output differs); prompt reading unchanged (4K 993 ->
986 tok/s, first run). Kernel level (one card, graph replay, random weights not L2-resident, us per read, default ->
two-kernel): T = 1 38.7 -> 34.8 (-10%), T = 4 50.8 -> 41.5 (-18%), T = 6 62.0 -> 47.1 (-24%). Of the down kernel's
shapes (rows per lane x columns per sub-group) at T = 6, (1, 80) 30.6, (2, 40) 26.9, (1, 40) 30.2, (2, 80) 23.6 us:
(2, 80) is the one built.

Switches: `STRATA_GR_FUSED=1` on (unset or `0`: off, the default read, byte-identical).

## Measured, 2026-10-01, Arc Pro B70, Coder IQ1_M, 32K context, INT8 KV: the SYCL port (engine 0.1.31-sycl)

## Switch reference

All are environment variables for the engine. Put them in the config's `"env"` block. Defaults are for the current engine.

| Switch | Default | Effect |
| --- | --- | --- |
| `STRATA_VERIFY_NO_HOST` | 1 (`strata-sycl.sh`); 0 on `i915` | See above |
| `STRATA_MIRROR_MIB` | Free RAM less 4 GiB | Size of the pinned host mirror for experts without a VRAM slot. Set `0` on a layer split where the mirror would not be used. |
| `STRATA_PF_XMX` | 2 on Alchemist | Grouped XMX expert launch; `0` off, `1` dequantize first |
| `STRATA_GDN_XE` | On for Alchemist | Sub-group GDN recurrence |
| `STRATA_COPY_ENGINE` | On for Battlemage | Expert copies on the copy engine |
| `STRATA_PREFILL_MMQ` | `auto` on Battlemage | Fused int8 prompt experts |
| `STRATA_PREFILL_STREAM_ALL` | On when VRAM holds more than 90% of the (layer, expert) pairs | Copy every expert the cache does not hold, layer by layer; `=0` / `=1` force it |
| `STRATA_PREFILL_FIRST` | 256 if the cache holds every expert, else 0 | Tokens in the short first prompt chunk |
| `STRATA_LEND_MIRROR` | On | Mirror the experts in the last 10% of cache slots; `=0` off |
| `STRATA_VERIFY_LEAN` | On | Decode graph without unanswered nodes; `=0` off |
| `STRATA_FILL_SERIAL=1` | Off | The old serial expert-cache fill (the pipelined fill is the default) |
| `STRATA_MMVQ_A2=0`, `STRATA_MMVQ_WIDE_32=0` | On | Restore the older matvec kernels |
| `STRATA_MMVQ_LARGE_GRF` | On for Arc A-series | 256 registers per thread for the 2-8 column wide kernels |
| `STRATA_Q6_PLANAR` | On | Q6_K output head and MTP head as planes (same bytes) |
| `STRATA_ATTN_RS` | 2 | QSA prompt attention probability layout (bit-exact variants) |
| `STRATA_IQ3S_WIDE=1` | Off | Wide kernel for dense IQ3_S matrices: neutral on one B70 (decode -0.3%), the author reports gains on two cards |
| `STRATA_EXPERT_SPLIT=1` | Off | Upstream's i-quant multi kernels (-11 to -16% decode here) |
| `STRATA_PROMPT_ATTN_XMX` | 3 (int8 KV, Battlemage) | 3: ESIMD prompt attention, the kernel alone 4.4x (118K context, 2048 queries) to 4.8x (the engine's own selections, 16,384 queries) the per-cell kernel in `qsa_prompt_attn_parity`; `=1` the joint_matrix one (-24% to -27% prompt); `=0` the per-cell kernel |
| `STRATA_SYCL_GEMM_STAGE=1` | Off on Battlemage | Stage GEMM operands past 4 GiB |
| `STRATA_I8_NO_ESIMD` | n/a | Compile-time: stubs for the int8 ESIMD kernels (set for a device without them) |
| `STRATA_DBG_NAN=1` | Off | Report non-finite values, including the experts' FP16 inputs |

## History / older measurements

Dated results from earlier engines. They show how the numbers moved and are not instructions. Narrative from the original work
log (debugging stories, per-merge notes, planned-item lists) was dropped; the numbers and their labels are kept.

> **Measured on:** Arc Pro B70 32 GB (xe), PCIe 3.0 x8, Ubuntu 24.04, Coder IQ1_M, 32K context, INT8 KV, the SYCL port,
> engine 0.1.31-sycl (2026-10-01), against llama.cpp SYCL (2026-09-29, q8_0 KV).

| | NVIDIA (RTX 5070, Strata) | Intel Arc, SYCL port | Intel Arc, llama.cpp |
| --- | --- | --- | --- |
| Decode, Coder IQ1_M | 44-51 tok/s | 78.2 tok/s (19-token prompt), 75.7 (2,184-token prompt); 48-72 through the API | 23-25 tok/s |
| Prompt | about 1,870 tok/s | 790 tok/s at 2,184 tokens, 1,117 at 40K, 888 at 128K, 757 at 256K | about 150 tok/s (424 at 105K) |
| Experts | in RAM, hot ones on the card | all 12,288 in VRAM (`--stream-experts`), about 1.9 GB VRAM free | all of shard 1 on the card |
| Load time | | about 26 s to first token (was 82 s before the pipelined fill) | about 100 s |

**Decode, Coder IQ1_M on the B70 (19-token prompt, 256 greedy tokens), by change.** Each step measured alone, output tokens
identical unless noted.

| Change | Decode |
| --- | ---: |
| First working engine (suffix drafter only, `--spec 2`) | 20.2-20.8 tok/s |
| With the base checkpoint's MTP draft layer (`--spec 4 --mtp`, 81-87% accepted) | 42.7-45.3 tok/s |
| After the 0.1.25-0.1.27 merge | 46.5 |
| Commit graph left running while the drafter runs on its own queue | 49.8 |
| Q6_K alignment fix (a Q6_K block is 210 bytes; two aligned 16-byte loads) | 53.9-54 |
| `resident_plan` grouping in parallel | 57.1 |
| GR down kernel reads activations directly | 58.7 |
| Wide 16-byte-load kernels for Q4_K, Q5_K, IQ4_XS | 62.6 |
| Wide kernels for Q8_0 and IQ4_NL; Q6_K activation loads aligned | 65.6 |
| GR down projection sliced by columns | 70.6 |
| IQ4_NL expert dots with codebook in registers | 76.1 |
| GR norm per (token, stream); aligned weight loads in IQ3 gate/up dots | 77.9 (output differs after token 146) |

Prompt, same card, 2,184-token prompt: 560 tok/s (first engine) to 792 tok/s after dequant vector stores (0.085 to 0.030 ms per
expert), a short first prompt chunk (711 to 765 tok/s), one device module per kernel and the PLE reader at 64 threads
(466 to 330 ms for that prompt's rows). Draft sweep at 2,184 tokens: `--spec 4` 41.6-43.5 tok/s, `--spec 2` 33.6, `--spec 6`
35-37. Decode depends on the text: 45-51 tok/s on code (77-85% of drafts accepted), 30-39 on prose (55-72%).

**Flash-Next IQ2_XS on the B70 (original model, 18,329 of 24,576 experts in VRAM, 6,247 in the pinned mirror).** Decode 50.8 tok/s
on a 19-token prompt and 60.6 after 2,184 tokens, prompt 549 tok/s (0.1.31); after the 0.1.31 merge 58.6 / 64.2. A forced test
(`--expert-cache 8000`, 1,879 experts out of VRAM) took decode from 2.6 to 40.9 tok/s (full cache 43.3) once the host mirror existed.

**Long prompts.** 80,000 tokens at `--max-context 81920 --kv int8 --vram-reserve-mib 2048 --prefill 4096` with slot borrowing:
75 s (1,062 tok/s), decode 35-40 tok/s after it; without borrowing, decode after the prompt ran at 2 tok/s. 128K int8 with
KV streaming: 897 tok/s routed-only against 920 with the stream-all walk; 40K 1,106 against 1,160; 128K k8v4 1,006 against 1,041;
256K 760 against 780 (0.1.38). Prompt time at 40K (GPU timeline): expert dequant 15.7%, expert GEMMs 23.4%, prompt attention
19.2%, host grouping 6.5%, QSA select 6.8%, DeltaNet 5.8%.

**Context sizes on llama.cpp (2026-09-29, Coder IQ1_M, q8_0 KV, 32 GB card).** About 20 KiB of KV per token.

| Context | VRAM in use | Status |
| --- | ---: | --- |
| 32,768 | 28.4 GB | Measured, the default |
| 65,536 | 29.0 GB | Loads |
| 131,072 | 30.3 GB | Practical ceiling: a 104,798-token prompt read in 247 s at 424 tok/s |
| 262,144 | about 32.6 GB | Does not fit; asking for it took the host down |

**Load time.** The expert-cache fill became a pipeline (reads in file order by up to 8 threads into page-locked batches of 64,
copies overlapped): a cold fill of the Coder's 23.4 GiB went from 76.2 s to 18.8 s (0.33 to 1.34 GB/s), a whole start from
82 s to 26 s, and the IQ2_XS from 120 s to 41 s.

The B70's int8 expert kernels also build and pass the parity test as a SPIR-V JIT build and as an AOT build for `bmg-g21` (compile only: no
G21 card here); an AOT build for another device (Alchemist, Xe-HPC) gets stubs (`STRATA_I8_NO_ESIMD`).

## Arc Pro B70: the `--stream-experts` mirror and the page cache (2026-10-22, #3681)

The pinned host mirror (`STRATA_MIRROR_MIB`, above) is 2 GiB `sycl::malloc_host` chunks, each one system-memory BO that
the `xe` driver backs as it likes. Chunks it hands out while the kernel reclaims page cache are slower for the GPU:
13 x 2 GiB chunks allocated with 11 GB free read at 14-20 GB/s (kernel zero-copy) from the fourth chunk on, against 24 GB/s
for the first three and for every chunk allocated with 60 GB free; CPU reads of the same chunks were 30.5 GB/s in both
cases. The engine's own start-up reads leave its GGUFs in the page cache, so on a 64 GB host MemFree at the mirror's
allocation was often under 1 GiB (the mirror needs 25.7 GiB for Flash-Next IQ3_S here), and decode after a restart
depended on what the page cache held: 37.5-41.4 tok/s against 42.5-43.0 from a clean start.

`STRATA_MIRROR_DROP_CACHE=before` (opt-in, Linux) drops the GGUF files' page cache right before the chunks are allocated
and prints MemFree before and after. Same box, PCIe 4.0 x16, 2K/32K needle prompts, 512 tokens, 4 engine starts per block
with a 22 GB unrelated file read between starts, blocks off/before/before/off (32 runs, outputs byte-identical across
the arms): restarts with `before` decoded at 42.4-44.5 tok/s, restarts without it at 37.5-41.4 (first starts 44.1-44.5
and 42.5-43.0); the mirror fill took 14.2-14.4 s instead of 13.2-13.4 s (it reads the disk).
It cannot free page cache other processes hold. Dropping the mirrored ranges *after* the fill instead was slower than
off on restarts (27.6-30.3 tok/s), so the variable takes `before` only.
## Arc Pro B70: QSA prompt attention per cell and block scores as GEMM tiles (2026-10-19)

Two prompt-path kernels that grow with the context. On a 40K prompt (Coder IQ1_M, 256K config) the QSA attention was 27.6%
of the GPU timeline (7.3 s) and the block selection 8.0% (2.1 s).

- **Attention scores one work-item per cell** (`qsa_prompt_attn_percell`, prompt path only; `STRATA_ATTN_PERCELL=0` off): a
  work-item holds one cell's key row and all 12 heads' dot products instead of a sub-group splitting the 256 dimensions with a
  shuffle tree. 2.0-2.2x the warp kernel in `qsa_prompt_attn_parity` (int8 / FP16 pools; 1.2-1.4x on Q4_0).
- **Block scores as oneMKL FP32 GEMM tiles** (pooled keys x the batch's indexer queries, then a relu-sum kernel;
  `qsa_block_scores_reduce`). The tile lives in the attention scratch, so it costs no VRAM. It loses to the warp kernel below
  ~8K cells (oneMKL's launch: -2% at 2K, even at 8K), so it runs only for a batch reaching 2,048 blocks; `STRATA_SELECT_GEMM=0`
  never, `=1` always. Selection at 40K 2.1 -> 0.7 s.

Both are FP32 in another summation order, not bitwise. First-token logits against the old kernels (whole vocabulary): KL
3.9e-10 nats at 8K and 1.3e-11 at 40K, top-1 equal (upstream's int8 against FP16 prompt experts, the same prompts: 8.4e-10 and
5.1e-12); the 256 greedy tokens after a 128K prompt are identical.

`--pcie-frac 0 --adapt-every 0 STRATA_PREFILL_MMQ=1`, 256 greedy tokens, interleaved runs, prefill tok/s (old -> new):

| model | 2,185 | 40,000 | 128,000 |
|---|---|---|---|
| Coder IQ1_M, 256K config | 1,329 -> 1,468 (+10%) | 1,562 -> 1,885 (+21%) | 1,336 -> 1,798 (+35%) |
| Flash-Next IQ2_XS, 128K config | | 1,373 -> 1,613 (+18%) | |

### Mixed-IQ3 decode kernels for Battlemage (PR #4185)

Packed Q2 / IQ4 lookup / IQ3_S sign arithmetic (the build option) and fourteen decode controls. Measured on two Arc Pro B65
(Qwen3.8-Flash-Next GSQ-RCO IQ3_S, all experts resident, layer split, `--spec 2 --pipeline-windows 2`, int8 KV, AOT bmg-g31): decode
+9.7% on a 4K prompt and +11.1% on a free-form answer against the build without them, same generated tokens, same logprobs to six
decimals. Prompt speed is unchanged.

* `STRATA_SYCL_IQ_DECODE_FAST` (CMake, default ON; `IQ_DECODE_FAST=OFF` for `sycl/tools/build.sh`):
  integer arithmetic with the same results as the helpers it replaces. It is the largest part of the gain (+5.9% alone).
* Runtime controls. Unset: on for a Battlemage device, off for every other device (Arc A-series keeps the earlier kernels).
  `STRATA_<name>=0` turns a control off, `=1` turns it on whatever the device. Restart after changing one.

| Runtime control | On by default (Battlemage) | Selected work |
|---|---|---|
| `STRATA_MMVQ_SHAPE_RPW` | yes | Row reuse for bounded Q4_K/Q5_K projection shapes |
| `STRATA_IQ3S_GU_ROW_REUSE` | yes | Reuse IQ3_S gate/up row inputs across two rows |
| `STRATA_IQ2S_GU_ROW_REUSE` | yes | The same for IQ2_S |
| `STRATA_IQ3XXS_GU_ROW_REUSE` | yes | The same for IQ3_XXS |
| `STRATA_GR_LO_REUSE` | yes | Reuse the hyper-connection low-rank SiLU output on supported layouts |
| `STRATA_ATTN_MERGE_ACTIVE` | yes | Merge only the active chunks in bounded batched QSA decode |
| `STRATA_IQ3S_GU_SLM` | no | Stage native IQ3_S gate/up activations (unreachable while the IQ3_S row reuse is on) |
| `STRATA_IQ4NL_DOWN_SLM` | no | Stage native IQ4_NL down activations |
| `STRATA_EXPERT_ROW_E3` | no | Three-entry tails in the IQ2_S / IQ3_XXS row-reuse path |
| `STRATA_RESIDENT_PLAN_COMPACT` | not on this tree | Size the resident routing work-group for the entry count |
| `STRATA_GR_CACHED_NORM` | no | Reuse the hyper-connection normalized input on supported layouts |
| `STRATA_GR_REDUCE_ILP` | yes on this tree | Keep partial loads in flight in the cached-lo hyper-connection reduce |
| `STRATA_Q6_SHAPE_GRF` | no | Large GRF for bounded Q6_K projection shapes |
| `STRATA_GDN_QK_REG` | no | Register-broadcast QK for supported Battlemage GDN geometry |

The eight "no" controls measured neutral or slightly slower on that configuration (single-control table in
`phaseA-tests/pr-priorities/RESULTS-INTEL.md`); they stay available for other models and cards. Unsupported layouts retain their
existing paths. Explicit projection overrides take precedence over shape selection.

**On this tree (the B70 fork).** Two different switches have nearly the same name. `STRATA_SYCL_IQ_DECODE_FAST` is the CMake
option above, fixed at build time. `STRATA_IQ_DECODE_FAST` is this tree's run-time switch, on by default on the B70. It selects
its own grouped expert kernels: two rows a lane at 8 lanes and SIMD16, cheaper decodes, and the codebook in shared memory. Where
this tree already had a kernel for the same call, that kernel stays first, because it was measured on the B70. The #4185 kernel
takes the call where it is switched off:

| #4185 control | Runs ahead of it on the B70 | The #4185 path runs with |
|---|---|---|
| `STRATA_IQ2S_GU_ROW_REUSE`, `STRATA_IQ3XXS_GU_ROW_REUSE`, `STRATA_IQ3S_GU_ROW_REUSE`, `STRATA_IQ3S_GU_SLM`, `STRATA_IQ4NL_DOWN_SLM` | `STRATA_IQ_DECODE_FAST` kernels (same two-row reuse) | `STRATA_IQ_DECODE_FAST=0` |
| `STRATA_MMVQ_SHAPE_RPW`, `STRATA_Q6_SHAPE_GRF` | `STRATA_MMVQ_KREG` (1-8 columns) | `STRATA_MMVQ_KREG=0` |
| `STRATA_ATTN_MERGE_ACTIVE` | `STRATA_ATTN_MERGE_V2` (the same bound and batched loads) | `STRATA_ATTN_MERGE_V2=0` |
| `STRATA_GR_LO_REUSE` | the Q8_0 hyper-connection read (`STRATA_HC_Q8`) | the BF16 read: the MTP drafter, or `STRATA_HC_Q8=0` |

`STRATA_GR_REDUCE_ILP` keeps the 8 loads in flight of this tree's plain reduce by default (`=0` off); the bits are the same
either way. `STRATA_RESIDENT_PLAN_COMPACT` is not wired: this tree's resident plan is built for 128 or 256 threads and its
mirror-ring clock walks that many candidates. IQ4_XS experts already take the multi-entry path here (`Multi<23>`), so the
port adds no IQ4_XS split.
