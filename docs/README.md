# Strata guides

**This fork (`b70`, tag `b70-2026.11.11`, upstream `v0.2.8` base; CLI `0.2.8`, engine `0.1.54`) builds one card on Linux: an Intel Arc Pro B70 (`bmg-g31`, 32 GB) in Docker (`sycl/`).** There is no desktop app in this fork, and no prebuilt `install.sh` / `install.ps1` release installer — start from the [README](../README.md) or [Install Strata](INSTALL.md).

## Use Strata

| I want… | Guide |
| --- | --- |
| Build and run on one Arc Pro B70 (Docker + `./setup.sh`) | [README](../README.md), [B70 branch notes](B70.md) |
| Install or update (Terminal / upstream installer reference) | [Install Strata](INSTALL.md) |
| Choose a model and check memory | [Models and quantizations](MODELS.md), [Quantizations](QUANTS.md) |
| Start from the command line (`strata launch`) | [Terminal launch](TERMINAL_LAUNCH.md) |
| Connect a coding tool | [Tool connections](TERMINAL_LAUNCH.md#connect-your-apps-strata-connect) |
| Fix an install or startup problem | [Troubleshooting](TROUBLESHOOTING.md) |
| Check privacy, downloads and network access | [Safety and privacy](SAFETY.md) |

## Hardware

This fork builds only the Intel Arc (`sycl/`) engine on Linux for one Arc Pro B70 (`bmg-g31`). Guides for NVIDIA, AMD Radeon/HIP, Strix Halo Linux, Apple Silicon Mac, multi-GPU layer splits, and older GPUs were removed from this fork.

| Hardware | Guide |
| --- | --- |
| Intel Arc Pro B70 on Linux (source container build, `bmg-g31`) | [B70 branch](B70.md), [SYCL port](INTEL.md) |
| Intel Arc on Linux (setup and driver prerequisites) | [Intel Arc](INTEL_ARC.md) |
| Two cards: one reads the prompt, one makes the answer (opt-in, upstream path) | [Split serving](SPLIT_SERVE.md) |

## Fork, model, and serving guides

| Guide | What it covers |
| --- | --- |
| [The `b70` branch](B70.md) | What the `b70` branch changes over upstream `main`, defaults, compiler pins, and B70 measurements |
| [The B70 served setup](B70.md#the-served-setup) | What `./setup.sh --backend sycl --family unsloth --model UD-IQ1_S` configures on one B70 (layer-18 graft, MTP, KV/experts, `--no-served-recipe`) |
| [Intel Arc SYCL engine](INTEL.md) | Building and running the `sycl/` engine inside `strata-sycl-dev` and maintaining the port |
| [Qwen3.8-27B (`qwen35`) on B70](Q36.md) | The dense Qwen3.8-27B family built into the same `strata` binary (`Q36=OFF` excludes it) |
| [Batch slots (`BATCHING.md`)](BATCHING.md) | Serving multiple requests together with `"parallel": N` / `--batch N` to share dense/expert reads |
| [KV prefetch (`KV_PREFETCH.md`)](KV_PREFETCH.md) | Opt-in `STRATA_KV_PREFETCH=1` to overlap streamed KV uploads from RAM with prefill compute |
| [Prefill head alias (`PREFILL_HEAD_ALIAS.md`)](PREFILL_HEAD_ALIAS.md) | Opt-in `--prefill-head-alias` to share output-head VRAM with prompt buffers during prefill |
| [Prompt cache tail (`PROMPT_CACHE_TAIL.md`)](PROMPT_CACHE_TAIL.md) | Opt-in `--prompt-cache-tail` to save an extra evict-first checkpoint near a prompt's end |
| [Message boundary cache (`MESSAGE_BOUNDARY_CACHE.md`)](MESSAGE_BOUNDARY_CACHE.md) | Opt-in `STRATA_CACHE_MESSAGE_BOUNDARY=1` to checkpoint the previous turn when editing the last message |
| [Disjoint expert cache (`DISJOINT_EXPERT_CACHE.md`)](DISJOINT_EXPERT_CACHE.md) | Opt-in `STRATA_DISJOINT_ADAPT=1` so the primary GPU avoids promoting experts held by a helper GPU |
| [Exchange rotation (`EXCHANGE_ROTATION.md`)](EXCHANGE_ROTATION.md) | Opt-in `STRATA_EXCHANGE_ROTATE=1` to swap a temporary buffer with resident RAM instead of copying evicted experts |
| [Gyro trellis quants (`GYRO.md`)](GYRO.md) | Running agentionai's Gyro-S and Gyro-M (`TQ2_T`, `TQK6`, `TQK7`) via `--family gyro` |
| [KVA (`KVA.md`)](KVA.md) | Opt-in `STRATA_KVA=1` for approximate late-layer evaluation on long cold prompts (off by default) |
| [Elastic VRAM (`VRAM_ELASTIC.md`)](VRAM_ELASTIC.md) | Opt-in `--vram-elastic` and `POST /v1/vram` dynamic expert-cache segment mapping (one NVIDIA GPU only) |
| [OrcaRouter Uncensored (`ORCA.md`)](ORCA.md) | Manual `tools/iq_pack.py --compat-bf16` conversion for `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF` (IQ3_XXS) |
| [Monitoring templates (`monitoring/`)](monitoring/) | Starter Prometheus scrape, Kubernetes ServiceMonitor, and Grafana dashboard definitions (`grafana-strata.json`, `prometheus.yml`, `servicemonitor.yaml`) |
| [Terminal telemetry fields (`TERMINAL_TELEMETRY_FIELDS.md`)](TERMINAL_TELEMETRY_FIELDS.md) | Schema v1 field contract (`GET /v1/telemetry`) backing `strata status`, `strata monitor`, and `strata logs` |

## Upstream / out-of-scope files kept in tree

Kept for history and provenance; not part of the active single-B70 Linux setup:

| File | Why it is out of scope here |
| --- | --- |
| [DEEPSEEK41.md](DEEPSEEK41.md) | Upstream/unmaintained: DeepSeek-V4.1-Flash (`strata-dsv41`) has not been maintained since 2026-10-30 after 2-bit + pruning quality loss on one B70; off by default (`DSV41=ON` enables the build) |
| [DEEPSEEK_MAC.md](DEEPSEEK_MAC.md) | Upstream/out-of-scope: DeepSeek-V4.1-Flash on Apple Silicon Metal (`metal/` is removed in this fork) |
| [STRIX_HALO_WINDOWS.md](STRIX_HALO_WINDOWS.md) | Upstream/out-of-scope: AMD Ryzen AI Max / Strix Halo (`gfx1151`) on Windows |
| [A770_SPEED_SPEC.md](A770_SPEED_SPEC.md) | Upstream/out-of-scope: Intel Arc A770 Coder IQ1_M optimization campaign spec (not the Arc Pro B70 target) |

## Developers and contributors

- [Build or run from a source checkout](SOURCE_INSTALL.md)
- [Contribution rules](../CONTRIBUTING.md)
- [Community benchmarks](COMMUNITY_BENCHMARKS.md) and the [standard benchmark protocol](BENCHMARK_PROTOCOL.md)
- [How the engine works](HOW_IT_WORKS.md), [technical reference](DETAILS.md) and [Terminal reference](TERMINAL.md)

## AI tools and media

- [Install for a user](AI_SETUP.md)
- [Verified facts, model availability and API reference](AI_REFERENCE.md)
- [Articles and videos](MEDIA_KIT.md)
- [Repository instructions for agents](../AGENTS.md)

## Which instructions apply?

`strata launch` belongs to the installed Terminal. `START-HERE.bat`, `SETUP.bat`, `UPDATE.bat` and `setup.sh`
belong to a **source checkout**. Raw engine commands are a third interface. Their flags are not interchangeable:
Terminal launch uses `--gpu` and `--vision on|off`; source setup uses `--gpus` and `--vision yes|no`.

Release notes, design proposals and benchmark reports describe their named versions and machines. They remain
available as historical evidence; use the guides above for a new installation. A published result on one GPU
does not establish support or speed on every GPU.

## Writing a guide page

- Installation steps live in [INSTALL.md](INSTALL.md), source steps in [SOURCE_INSTALL.md](SOURCE_INSTALL.md). The public READMEs stay short.
- Check command examples against the parser and filenames against the release assets.
- A guide describes the current release. Version history, fixed bugs and superseded settings go in a closing
  **History / older measurements** section, in `bench/results`, or in the release notes. Do not mix them into instructions.

Every guide page has this shape:

| Part | Content |
| --- | --- |
| Title | `# Name of the task or topic` |
| Purpose | One to three lines: what the page is for and who it is for |
| Quick start or Steps | The shortest working path, commands in code blocks |
| Details | `##` sections; option and flag lists as tables; short paragraphs |
| History / older measurements | Optional, last. Dated numbers only |

Label every measurement the same way, directly above the number or table:

> **Measured on:** hardware (GPU, CPU, RAM), model or set, Strata version, workload.
