# Which model fits one B70: VRAM and RAM arithmetic

This page shows the memory budget on one **Intel Arc Pro B70** (`32 GB` card on Linux with the `xe` driver) and the one quantization each supported model picks under that budget. See [the served setup](B70.md#the-served-setup) for the exact Flash-Next flags and [the 27B guide](Q36.md) for the dense runtime.

## Hardware formats and VRAM budget

- **Hardware compute and cache formats:** Battlemage (`bmg-g31`, Xe2) matrix hardware (`dpas` / `joint_matrix`) runs **`int8`**, **`fp16`**, and **`bf16`** natively (`sycl/src/qwen35/q36_xr_dense.dp.cpp`, `sycl/src/prefill/xmx_moe.dp.cpp`, `sycl/src/qwen35/q36_w4h.dp.cpp`). There is no FP8 matrix path in this fork (`sycl/src/qwen35/merge_attn_states.hpp` strips FP8 output paths), so weight packs and KV caches stay in GGML/native formats, `int8`, `fp16`, or `bf16`.
- **Upfront card budget:**
  - `32 GB` decimal is `29.80 GiB` binary; Level Zero reports **`31.89 GiB`** total device memory on the B70 (`sycl/src/qwen35/q36_engine.dp.cpp:8858`).
  - Subtract **`3.0 GiB`** kept free for a Linux desktop (`--vram-reserve-mib 3072`, [README](../README.md)): that gives a **`28.9 GiB`** hard cap in the 27B dense chunk rule (`31.89 - 3.0 = 28.89 GiB`, `q36_engine.dp.cpp:8858,8873`; or `~26.8 GiB` starting from `29.80 GiB` decimal). Staying below 32 GB avoids `xe` driver wedges ([INTEL.md](INTEL.md#how-much-context-fits)).
  - Subtract **engine buffers** (`1.85 GiB` at chunk 512 to **`2.56 GiB`** measured at chunk 2048 with `--mtp 3`, plus `0.4 GiB` headroom in the auto-chunk rule = `~2.96 GiB`, `q36_engine.dp.cpp:8856,8873`; and **`0.77 GiB`** shared call scratch at 262,144 cells when matrix-engine decode attention `STRATA_Q36_XATT` is enabled, `Q36.md:172,196`).
  - Subtract the **KV cache** (plus recurrent state) at your chosen context.

## Summary: one quant per model

| Model | Chosen quant | Context | Why this one fits | Guide |
| --- | --- | ---: | --- | --- |
| Qwen3.8-Flash-Next (MoE) | **UD-IQ1_S** (`66.4 GB` download) | 262,144 (`int8` KV, 32K resident) | `15,784 of 24,576` experts fit in VRAM; **`18.2 GiB`** stay in pinned RAM, leaving **`11 GiB`** free on 64 GB RAM | [B70.md#the-served-setup](B70.md#the-served-setup) |
| Qwen3.8-27B (dense) | **UD-Q4_K_M** (`16.5 GB` GGUF) | 262,144 (`int8` KV) | Whole model (`15.37 GiB`) + `8.91 GiB` session cache + `2.56 GiB` buffers = **`26.84 GiB`** (`26.68 GiB` measured peak) | [Q36.md](Q36.md) |

---

## Qwen3.8-27B (dense): fits in VRAM at 256K

> **Architecture (`Q36.md`, `q36_engine.dp.cpp`):** 64 layers (16 attention + 48 GDN) plus 1 MTP attention block = **17 attention layers**, 4 KV heads of dim 256. Every weight stays in VRAM.

- **KV cache per cell (`q36_kv_cell_bytes` / `q36_sess_bytes`, `q36_engine.dp.cpp:1007-1011,3033-3039`):**
  - **FP16 KV:** `2 (K+V) * 4 heads * 256 * 2 B = 4,096 B` (`4 KiB`) per layer per cell -> `17 * 4,096 = 69,632 B` (`68.0 KiB` / **`69.6 KB`**, `Q36.md:94`; `64 KiB` across the 16 main layers). At `262,144` cells, FP16 KV alone is `69,632 * 262,144 =` **`17.00 GiB`** (`16.00 GiB` main + `1.00 GiB` MTP), plus `0.14 GiB` GDN recurrent state = **`17.14 GiB`** — **does not fit** alongside model weights on 32 GB.
  - **8-bit KV (`--kv int8`, automatic above `131,072` cells):** `int8` codes plus one FP16 (`2 B`) scale per 64 values (`q36_engine.dp.cpp:1008`): `1,024 + (1,024/64) * 2 = 1,056 B` per K or V -> `2,112 B` per layer per cell -> `17 * 2,112 = 35,904 B`. At `262,144` cells: `35,904 * 262,144 = 9,412,018,176 B =` **`8.77 GiB`** (`Q36.md:95`), plus **`0.14 GiB`** GDN recurrent state = **`8.91 GiB`** (`Q36.md:237`, `q36_engine.dp.cpp:8859`).

- **Weight fit arithmetic at `--max-context 262144` (`UD-Q4_K_M` vs `UD-Q4_K_XL`):**
  - Available before weights on `31.89 GiB` minus `3.0 GiB` desktop reserve (`28.9 GiB` cap) minus `8.91 GiB` 8-bit session minus `2.56 GiB` chunk-2048 buffers (`--mtp 3`): `28.89 - 8.91 - 2.56 =` **`17.42 GiB`** (`~15.9 GiB` if starting from `29.80 GiB` decimal minus `3.0 GiB` minus `~2.9 GiB` engine extra minus `~8 GiB` KV).
  - **`UD-Q4_K_M`** (`16.5 GB` decimal = `15.37 GiB`, `Q36.md:46`):
    - Sum: `15.37 GiB` weights + `8.91 GiB` cache + `2.56 GiB` chunk-2048 buffers = **`26.84 GiB`** (`26.13 GiB` at chunk 512).
    - **Measured on:** one Arc Pro B70 (Ryzen 9 5950X, 64 GB RAM), `UD-Q4_K_M`, `--mtp 3`, `--max-context 262144`, `strata-q36` branch (2026-11-03): **`26.68 GiB`** peak VRAM (`26.49 GiB` before LP3, `Q36.md:271`).
    - **Measured on:** one Arc Pro B70 (Ryzen 9 5950X, 64 GB RAM), `UD-Q4_K_M`, `--mtp 3`, `--max-context 262144`, `b70` tree (2026-11-11): **`5.46 GiB`** free VRAM at start out of `31.89 GiB` (`Q36.md:237`).
    - **Measured on:** one Arc Pro B70 (Ryzen 9 5950X, 64 GB RAM), `UD-Q4_K_M`, `--max-context 262144`, `b70` tree (2026-11-13): **`5.51 GiB`** free after start (`STRATA_Q36_XATT` off), **`4.80 GiB`** free with `STRATA_Q36_XATT=1` (`0.77 GiB` shared attention scratch, `Q36.md:195-196`).
  - **`UD-Q4_K_XL`** (`17.6 GB` decimal = `16.39 GiB`, `Q36.md:47`): `16.39 + 8.91 + 2.56 =` **`27.86 GiB`** before LP3 and before `0.77 GiB` XATT (`28.63 GiB`), leaving no reliable headroom under the `28.9 GiB` cap and exceeding a `~26.8 GiB` budget. **`UD-Q4_K_M` is the single quant picked for 256K.**

## Qwen3.8-Flash-Next (MoE): VRAM expert cache + host RAM

Flash-Next (`UD-IQ1_S`, `66.4 GB` download: shards `10,946,618 + 37,568,095,648 + 28,800,138,432 = 66,379,180,698 B` in `setup.py:305-311`, plus `~5 GB` draft layer in `README.md:101`) puts active context, dense weights, and as many routed experts as fit into VRAM, and keeps the remaining experts in pinned host RAM (`STRATA_MIRROR_MIB=25000` in the served recipe, [B70.md#the-served-setup](B70.md#the-served-setup)).

> **Measured on:** one Arc Pro B70, Ryzen 9 5950X, 64 GB RAM, `--spec 6`, `--vram-reserve-mib 3072` (3 GB free), `STRATA_MIRROR_MIB=25000`, `--max-context 262144`, `--kv int8`, `--kv-resident 32768`, `--prefill 16384` (2026-11-11, [B70.md](B70.md), [README.md](../README.md#things-that-matter-on-this-card)):

- **`UD-IQ1_S` (the served pick):** `33.3 GB` expert arena (`setup.py:379`); **`15,784 of 24,576`** experts (`64.2%`) reside in VRAM and **`18.2 GiB`** are pinned in host RAM (`B70.md:602`). With a desktop, browser, and Android emulator open on 64 GB RAM, the 250K run kept **`11 GiB` of host RAM free** (`README.md:214`, `B70.md:623-624`).
- **`UD-IQ2_XS` (why it is the edge on 64 GB, not the default):** `75.8 GB` pack; only **`11,832 of 24,576`** experts fit in VRAM, requiring **`24.4 GiB`** pinned in host RAM (`B70.md:603`). On the same 64 GB desktop at 250K, free RAM dropped to **`7 GiB`** with `STRATA_F16X_ANY` products enabled, and without them the run hit the 6 GiB free-RAM guard and stopped (`B70.md:622-623`).

## What does not fit on one B70 (and 64 GB RAM)

- **`Qwen3.8-27B` with `fp16` KV at 262,144 context:** `15.37 GiB` weights + `17.14 GiB` FP16 session (`17.00` KV + `0.14` GDN) + `>= 1.85 GiB` buffers = `>= 34.36 GiB > 31.89 GiB`.
- **A second loaded model on the same card:** at 256K, Flash-Next `UD-IQ1_S` fills VRAM down to the 3 GB desktop reserve, and 27B `UD-Q4_K_M` leaves `4.80–5.51 GiB` free (`Q36.md:195-196,237`), which cannot hold another model (`serve/server.py` also stops leftover engines unless `STRATA_KEEP_LEFTOVERS=1`, `Q36.md:86-87`).
- **`--ple-io ram` on a 64 GB host:** Shard 3 (`per_layer_token_embd`, the n-gram table) is **`28,800,138,432 B`** (`28.8 GB` decimal / `26.82 GiB`, `setup.py:311,392`). `sycl/setup_intel.py:398` computes the RAM requirement as `ram_gb (40) + PLE_TABLE_GB (28.8) + 16 = 84.8 GB` (~**`85 GB` real RAM**, no swap). With 64 GB RAM (or anything under 85 GB), leave `--ple-io` at its **`direct`** SSD default (`README.md:159,212`).
