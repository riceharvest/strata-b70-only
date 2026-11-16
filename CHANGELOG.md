# Changelog

All notable changes of this fork (`riceharvest/strata-b70-only`, branch `b70`) are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This list is written from `git log --no-merges 870aedd9e..origin/b70`. `870aedd9e` is the merge base with upstream
`Niko1221/Strata` `main`. The commits are dated 2026-10-24 to 2026-11-16. The fork's one release tag is `b70-2026.11.11`
(commit 3a0c44a40, dated 2026-11-11, notes in `docs/release-notes-b70-2026.11.11.md`); the `v0.2.x` tags in the repository
are upstream's. Entries are sorted into the tag's section or Unreleased by whether the commit is in the tag's history. It is a
selection, not every commit: a reader who needs the rest runs `git log`. The short hash after an entry is the commit it comes
from. Numbers are the ones the commit message states, measured on one Intel Arc Pro B70 unless it says otherwise.

## [Unreleased]

### Added

#### Dense Qwen3.8-27B (Q36 family)

- The dense `qwen35` model family (Qwen3.8-27B) in the architecture guard, geometry and layout table (931038f4b), first as
  `strata-q36` (3cab4c2cb), then as one `strata` binary with two model families (34d05903b). Docs: `docs/Q36.md`.
- Example serve configs for the 27B, with a test of what the server hands the engine (29a1286d5); parity, KL and launcher
  tools in `tools/qwen35` (b0c5f55be).
- Matrix-engine decode attention (`STRATA_Q36_XATT`, first version 99f7dd34f; on by default with the int4 pack when its
  mirror fits, ede869b0d; also for the MTP block, 7dc27002e), and the single-pass `XATT2` form behind `STRATA_Q36_XATT2`
  (03cee3e55, rewritten 848de7129).
- B70W4, a GPTQ int4 checkpoint packed for the matrix engine (`--w4`, c32771ee7); native u4 x s8 `W4X` decode kernels
  (6149092a5); W4A16, the int4 pack against FP16 activations, on for batch engines (ffe9cdc6f).
- XMX row forms for the decode products (`STRATA_Q36_XR=1`, 95a53537a); verify windows of up to 16 rows, up to 15 drafts a
  step with the int4 pack (08c299455); the W4 form takes up to 32 columns in one read of the weights (dcac50075).
- Drafting: draft depth follows acceptance (9bd899c2d); lookup drafts on top of the MTP drafts (3d50d928b); sampled requests
  and requests with penalties draft with the plain path's draws (1fcb27ef0, c85960887); batch windows verify a chain of MTP
  drafts per request (`--batch-mtp`, 5ae0aa54f).
- DFlash 2 block drafter for the 27B (`STRATA_Q36_DFLASH`, f4e2046e9), its tree verify (fed5830d5, committed as not built yet), a throughput arbiter
  (`STRATA_Q36_DFLASH_ARB`, 2f4c8e1f6, committed as not built yet) and a drafter pack tool and CPU reference.
- QSA: indexer, block selection and sparse attention kernels behind `STRATA_Q36_QSA` (f258ddaec), sparse prefill tiles
  (1e9e060ad), `qsa_extract.py` for the 80 QSA indexer tensors (e50b44c70), `docs/QSA.md` (d78e062bd).
- Prefill: Xe2 int8 XMX prefill attention (61ac705e8), chunkwise C=64 WY XMX GDN prefill (e32223df6), W4 to INT8 XMX prefill
  and fused RMS/AB16 kernels (e076f3f7c), `q36_pp_i8` int8 DPAS and fused W4 to FP16 prefill GEMMs (296c691b4).
- Multi-checkpoint GDN and KV prefix cache (bd6ffb3ac); adaptive MTP/Lookup with SLM-checkpointed tree verify (cd98d352d).
- Batch serving: per-slot decode attention with `--batch N` and `--kv int8` (726c514ad); batch windows draft for sampled and
  penalised requests (d5fd9bf0c); single-stream windows for a lone request on a batch engine (6931e975d); `ignore_eos`
  (4c00609dd); agent threads keep each thread's cache in a tiered conversation cache (237a48dec); a 32-row batch window on the
  int4 pack (5306efb7d).
- Agent benchmarks against vLLM with real agents (bf7ed6963, c375dc92e) and a profile of a run of real agents (9216057ed).
- Docs: the 27B measured against vLLM (2219b4247), marked experimental while sampled requests ran without the draft
  (3f908a4aa); the 27B built with the pinned device compiler: 81.0 / 65.2 tok/s at short context (c842f9dcb).
#### The B70 card library and packs

- The B70 card library: the matrix-engine kernels and the pack, for any model family (d63316f69); Flash-Next on it, measured
  (4e0fb9a7d); calibrated packs where the engine captures each projection's inputs and the pack tool quantizes with GPTQ
  (b00d6162e). Docs: `docs/B70_LIB.md`.
#### Flash-Next engine (SYCL)

- The image pins the device compiler (IGC 2.41.5, compute runtime 26.35) and `build.sh` reports it (c1b16af25).
#### Server and web

- Web: engine settings grouped, page-only settings split, cost presets (75e100019).
#### Docs

- A B70 Linux quickstart (03ab88dbc); a model-fit VRAM and RAM guide (a6bd643d5); `b70-2026.11.11` release notes (ed98173ee);
  a TROUBLESHOOTING section for one B70 on Linux (84dc62fac).

### Changed

- Contact info and issue routing point at this fork, not upstream (86f64dc27).
- 27B: a 4,096-token prompt chunk when VRAM allows (4c15ee307); a request alone verifies a tree of 8 rows, not 12, from an
  in-run comparison on 2 real agents (141dc78f3).
- Setup: on the Intel route the n-gram table stays on the SSD unless the real RAM holds it (6918ac59a).

### Removed

- The DeepSeek V4.1 engine path and non-B70 build targets (5b6775385), the family in the server (31b801f8c), in setup, launch,
  control and tools (6e840aa01) and in the docs (db3571742).
- Tests of removed pages and features (1f5e201e0, ef0998194).

### Fixed

- QSA: the ring is written after the pooling, since a window of 29-35 rows overwrote the earlier cells' slots (916106b3f).
- Setup: a shard that is a link keeps its own name in the container path (6e39b0f91).
- The dense bench no longer reads past its weight pool (4fd5eca60).

## [b70-2026.11.11] - 2026-11-11

### Added

#### Flash-Next engine (SYCL)

- Decode kernels from upstream PR #4185, ported and defaulted on for Battlemage where they measured a gain (8d164c8c7,
  e14ddfa4b, b9c02b85f).
- Prompt path: FP16 expert products above the int8 crossover (`STRATA_PREFILL_F16X`, 1bd556c64) and for every expert format of
  Unsloth's files (95fe15b19); XMX block scores with the relu-sum fused and a candidate top-k (`STRATA_SELECT_XMX`, b31ab93d0);
  ESIMD prompt attention with int8 DPAS scores (77e12affd); XMX block scores past about 201K tokens (6214770c3); expert copies
  overlap the batched read (e3be567df).
- Decode: a lean exchange, an adaptive VRAM expert tier for the lean verify window, on by default (4804e774e); host-mirrored
  experts staged into VRAM with 16-byte loads (f3247653a); the cheaper IQ expert decodes on by default on the B70 (1cdfd16ad).
- Speculation: exact sampled-draft acceptance on by default on the B70 (7389a10f2); `STRATA_SPEC_PROBE_MIN` (c2f1bd2b8).
- Conversation cache: an SSD tier where evicted parked conversations spill to a folder and are read back (1bfd25771);
  `--conversation-cache-park-ssd` (7a3244e4b); the system prompt's end checkpoint is kept through the rotation (992759c62).
- A coding-agent expert profile and a serve-mode routing dump (61f5dc0cc).
- `STRATA_HOST_MARCH` for the engine's host code, and `HOST_MARCH` in `build.sh` (ee7a6a63a).
- `strata-sycl.sh` runs the engine under Podman, in a toolbox or a distrobox (`STRATA_SYCL_RUNNER`, dc0c2fdf2).
- The Arc Pro B65 (PCI id e222) by name in `setup_intel.py` (d205e888b); `--check` sizes a model across every Intel card
  (9aa8e6002).
#### Setup

- Unsloth UD-IQ1_S as a model, pinned at its own revision, with the layer 18 graft before the pack (cced8faf8); the served B70
  recipe for it, preselected on one B70 with 64 GB (c58442184); `tools/graft_tensor.py` takes one tensor from another GGUF
  file into a shard in place (59eaeb90d), with `--compact` (00e6fa207).
- `STRATA_VISION_DIR`, the picture reader's folder mounted into the engine's container (2cb778788).
- `strata launch` finds the source-built SYCL engine and sets up through `./setup.sh --backend sycl` (562b0f75f).
#### Server and web

- Tool calls: `tool_arg_coercion` (declared schema types) and `tool_skipped_closer`, opt-in at first (15e0b454e);
  `tool_arg_coercion` was then switched on by default, `tool_skipped_closer` stayed off (184b16342); a misspelt opener before a declared tool's call opens the call (0b79d1341); Anthropic tool results follow
  `tool_use_id` (6f65c6eed).
- `repeat_cycle_tokens` guard for short token cycles, with repeat stops counted in `/metrics` (44ff57276).
- `POST /tokenize` returns the token count and ids of the prompt a chat request would read (f0994c762); the answer is
  `{count, tokens}` like llama.cpp and vLLM (fb9c6a3bc).
- Stream order: the stream served last goes on first (759529424); threads can be pinned (5a6692506).
- Settings keys for the engine's options and the server's fixed numbers (597019bff); `stream_write_timeout_s` (c77e6b35f);
  the Settings view lists parked conversations and parallel requests (4b6bd2208).
- Web: switch models from the Settings view (4146c2e17); a dashboard with one bar per running parallel request (99c7322f8);
  cost-saved metrics against a Qwen-Flash baseline (f49dc56b7).
#### Docs

- AGENTS.md rewritten for the B70 fork (6660f00a7); `docs/B70.md`, what the b70 branch changes and what it measured against
  upstream (e5f078a49).

### Changed

- B70 run-time defaults live in the engine (`apply_b70_defaults`) with working opt-outs (583f6658d).
- The README is the B70-only fork's, without upstream's translations (0e8538b31).
- `build.sh` builds `strata` when no target is named; parity programs are off by default (558a5d4de).
- `STRATA_XE2_KQ`, `STRATA_XE2_IQ` and `STRATA_HC_Q8_DOWN2` were switched on (bfb9479f6) and then taken out of the B70
  defaults: no faster in the engine (bdf317721).
- The penalties count the tokens the reply has written, not the last tokens of the prompt; `STRATA_PENALTY_PROMPT=1` brings
  the old window back (462fab63f, 4e977839c).
- Prompt speed rounds on one B70, as the commits state them: 118K prompt 1,581 -> 2,360 tok/s (9cc2181e7), then 1,801 ->
  2,923 tok/s in a clean A/B (baf82d682); the localmaxxing prompts 688 -> 903 and 634 -> 823 tok/s (243e413f8).
- Setup: the served recipe is for exactly one B70 and an env entry the person set survives a rerun (a54b0a28e).
- AGENTS.md: one stream is the served setup, parallel stays opt-in (13d15ccc9).

### Removed

- The CUDA and HIP engine sources, the top-level CUDA/HIP build and the CUDA Dockerfile and Nix flake (31663d004, bc1374052,
  c887d1a41); the NVIDIA, AMD and Apple GPU readers, probes and setup paths (9076740db, 3525ebab9, 267499068); the NVFP4 family
  and tools (9a2d21f79, 036107a55); the Qwen3.6-35B-A3B family and harness (9bceebd97, f5f7b6b27).
- The Apple Metal engine and the Mac packaging (8799ccb06); the Windows `.bat` launchers (b7ee62cf9); the Metal build workflow
  (4a072b530).
- `metal/`, `apps/` and `packaging/` (the Metal engine, the desktop app, the Windows and macOS packages) with their tests and
  release workflows (454b36fb2); the upstream release image (`docker/`), the DGX Spark path and `README.pl.md` (3a381d34c).
- Pages that only cover NVIDIA, AMD or Apple (50699582e), and the NVIDIA/AMD benchmark results (17d88442d).
- DeepSeek V4.1-Flash, which had been ported to SYCL (5a470aa50), was marked not maintained for its quality loss on one B70
  (b54d87a2e).
- Checks of README translations and of the upstream README text (cab60f1d1).

### Fixed

- A streaming client that stops reading no longer holds the engine's turn (e51fae2c0, from upstream c08fd2ec); the drain and
  slot-release limits count silence, never progress (c29208a66, from upstream daf4ac08); a failed engine start no longer crashes
  when another thread closed the process (01f9acc6b, from upstream 0d922698); the STOP of a request that ends early never
  raises (c41b80c69).
- An evicted parked conversation gives its RAM back to the system (d1489a013, from upstream 8c71c492).
- `STRATA_FUSE_HEAD_GR=1` with the Q8_0 final mixer no longer drops the last layer's write (9c8f9e212).
- `--mtp-q4` no longer overflows the draft head buffer on an IQ4_XS head (442cee341); `STRATA_HC_Q8=1` reads the Q8_0
  hyper-connection projections it loaded and never read (faaa8a859).
- Prefill: the quiet timing records every mark again, since one per layer let a cold 250K prompt stop (cb62472e8); a stager hang
  fixed (8dd45a8ca).
- The Arc's copy engine hangs on pinned-host <-> pageable copies, so host-only copies go through the CPU (458b70ba8).
- A finished one-shot run on the B70 no longer ends with exit 139: the select kernels' four global tables are never freed, since
  their static destructors ran after the SYCL runtime was down (eed9e5196).
- Setup: `graft_tensor` refuses links and symlinks and turns a cut body or a bad target into a clean refusal (43a0665dd).
- Tool-call parsing: a call's closing tags end a value only when no prose and later closing tags follow, so a file that lists
  them survives (4c8836434); a fence is closed only by one as long as it opened with (2a75f39cf).
