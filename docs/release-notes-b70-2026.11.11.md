# Strata B70 (`b70-2026.11.11` and follow-up PRs `#27`–`#36`)

This release targets one card only: the **Intel Arc Pro B70** (`bmg-g31`, 32 GB) on Linux with the `xe` driver, built in Docker (`sycl/`). There is no desktop app, no Windows/macOS binary, and no CUDA or HIP runtime in this fork.

> **Important note on the downloadable `.gz` asset:** Git tag `b70-2026.11.11` (`3a0c44a4`, PR `#26`) carries a compiled engine asset built with **oneAPI 2026.1.0**, **IGC `2.41.5`**, and **Compute Runtime / `ocloc` `26.35.39758.10`** on and for one desktop (Ryzen 9 5950X, 64 GB RAM, Arc Pro B70 on Linux). It is not a general-purpose portable Linux binary; for other machines or newer commits, build from source in Docker ([QUICKSTART.md](QUICKSTART.md), [README.md](../README.md#install)).

---

## Compiler pin (`IGC 2.41.5` / `ocloc 26.35.39758.10`)

The AOT device compiler determines runtime decode throughput ([B70.md](B70.md#build-and-run)):

| Device compiler | Binary size | Flash-Next `UD-IQ1_S` (`reasoning` / `code`) | 27B `UD-Q4_K_M` (`reasoning` / `code`) |
| --- | ---: | ---: | ---: |
| **IGC `2.41.5`**, `ocloc 26.35.39758.10` (pinned) | `162,115,384` bytes | `127-133` / `108-110` tok/s | `81.0` / `65.2` tok/s (`Q36.md`) |
| **IGC `2.42.6`**, `ocloc 26.40.40196.8` | `161,967,928` bytes | `120-125` / `103-106` tok/s (**3-5% slower**) | `74.6` / `60.0` tok/s (**8% slower**) |

`sycl/tools/Dockerfile` downloads and holds the `2.41.5+22716` / `26.35.39758.10-0` packages, and `sycl/tools/build.sh` warns if a different compiler is active. An engine built with `2.41.5` runs at full speed on newer `26.40` host/container Level Zero runtimes.

---

## What shipped (`#23`, `#26`, `#27`, `#28`–`#36`)

### Flash-Next (`UD-IQ1_S` served setup, pictures, and setup fixes)

- **Served B70 recipe (`#23`, `0fc9f618`):** `./setup.sh --backend sycl --family unsloth --model UD-IQ1_S --context 262144 --vision no` now pins Unsloth `UD-IQ1_S` (`084c631`, `66.4 GB` across 3 shards), patches `blk.18.ffn_gate_exps.weight` from `UD-IQ1_M` (`216,268,800 B`; optional `--compact` shrinks shard 2 to `37,515,666,848 B` for llama.cpp), sets `--spec 6`, `--prefill 16384`, `--kv int8 --kv-resident 32768`, `--draft-vocab code --mtp-window 8192`, `expert-profile-pi.bin`, `STRATA_MIRROR_MIB=25000`, `STRATA_PLE_AUTO_RAM=0`, and defaults `--vision no` on 64 GB single-B70 setups ([B70.md#the-served-setup](B70.md#the-served-setup)).
- **Measured Flash-Next speed (`#23`, `#27`, `d8f231a1`):** On one B70 (Ryzen 9 5950X, 64 GB RAM, 3 GB desktop reserve, 2026-11-11): `3,475 / 3,356 tok/s` prompt and `85.6 / 86.5 tok/s` answer at `118,227` tokens; `3,325 / 3,329 tok/s` prompt and `84.7 / 87.5 tok/s` answer at `249,650` tokens; greedy fixed-text `localmaxxing` `129.6-131.2 / 106.6-109.4 tok/s` (`129-134 / 105-110 tok/s` with 700 MiB reserve). Served kernels match exact fallback kernels in `36 of 36` prompt top-1 predictions and `99.1%` of `2,867` decode steps.
- **CPU picture reader (`#23`):** `UD-IQ1_S` passed 4/4 test image requests via host `strata-vision` (`2.6 s` cold, `0.5 s` repeat); added `STRATA_VISION_DIR` container mount for per-request image embeddings (`2cb77878`).
- **Serve & memory behavior (`#23`, `#29`):** Evicted parked conversations release host RAM (`d1489a01`); `/tokenize` returns `{count, tokens}` (`fb9c6a3b`); `stream_write_timeout_s` (default 60 s), `tool_arg_coercion`, fenced/indented tool-call parsing updates, and chat UI scroll/math toggle landed (`7aef20e5`).
- **FP16 expert prefill default (`#26`, `3a0c44a4` — tag `b70-2026.11.11`):** `STRATA_F16X_ANY` enabled by default (`=0` disables), lifting `UD-IQ2_XS` 118K warm prompt reads from `2,671-2,844 tok/s` to `2,980-3,076 tok/s`.
- **64 GB RAM safety and symlinks (`#28`, `d08c5990`):** Fixed RAM detection in `sycl/setup_intel.py` so `--ple-io ram` (locking the `28.8 GB` table, `84.8 GB` total requirement) is only chosen when real RAM is `>= 85 GB`; preserved symlink filenames in container mounts and warned clearly when targets sit outside `/work`.

### Experimental: Qwen3.8-27B (`qwen35`) on one B70 (`#30`–`#36`)

- **Device compiler pin (`#30`, `c13ffd57`):** Added `IGC 2.41.5` / `26.35.39758.10` pin and `STRATA_Q36_BUILD` isolation.
- **Dense 27B path ported (`#31`, `a8306bdb`):** Brought the `qwen35` dense engine and `libq36_tla.so` long-prompt attention (`Q36.md`). Fits `UD-Q4_K_M` (`16.5 GB`) at `262,144` context with 8-bit KV on one B70 (`26.68 GiB` peak on `2026-11-03`; `5.46 GiB` free on `2026-11-11`). Marked **experimental**: beats llama.cpp (`44-48 tok/s`), behind vLLM (`109.9 / 87.5 tok/s`).
- **Penalty & sampled drafting fixes (`#32` `a9c8af8c`, `#33` `a1efe06b`, `#34` `7c25c21e`):** Fixed `presence_penalty` blocking greedy thinking-off drafts (`21.5 -> 31.0 tok/s` after 118K prompt) and added exact top-M verify kernels so temperature-sampled and penalized client requests draft (`27.6 -> 74.9 tok/s` default thinking-on; `27.6 -> 82.6 tok/s` thinking-off).
- **Parity bench fix & XMX rows (`#35` `120be1eb`, `#36` `2b26d75e`):** Fixed out-of-bounds read in `q36_mmvq_dense_parity --bench`, enabled XMX rows decode forms, `Q4_K` draft weights, and adaptive `--mtp 5` default: greedy `localmaxxing` reached **`85.9 / 66.8 tok/s`** (`124.7 tok/s` short question, `82.5 tok/s` default client sampled request, `1,097 tok/s` on a `118,226`-token prompt, `Q36.md:205-214`, `2026-11-12`).

---

## Experimental / known limits

- **`Qwen3.8-27B` is experimental (`#33`, [README.md](../README.md#what-runs)):** Runs ahead of llama.cpp (`44-48 tok/s`) and behind vLLM (`109.9 / 87.5 tok/s`); 118K prompts meet the fork's `>= 1,000 tok/s` bar (`1,097 tok/s`), while 256K prompts were only measured earlier at `721 tok/s` (`2026-11-03`, `Q36.md:17`). Follow-up PRs after `#36` (`#37`–`#41` and `34d05903`–`4c15ee30` on `origin/b70`) later merged `strata-q36` into the single `strata` binary and added `--w4` pack / 256K XATT support.
- **DeepSeek V4.1-Flash (`sycl/src/dsv41/`):** Unmaintained since `2026-10-30`; disabled by default (`DSV41=ON` to build).
- **Upstream `0.2.8` base ([B70.md](B70.md)):** Multi-GPU split serving, desktop UI features, and unmeasured opt-in kernel variables are inherited from upstream `870aedd9` and untested unless noted in `docs/B70.md`.
