# The B70 card library

Kernels written from the Arc Pro B70's measured limits (how fast it streams memory, what its matrix engine multiplies natively,
how many activation columns a weight pass carries for free) instead of for one model, and the card's own weight format, the
**pack**. Any model family of `strata` can call them: every entry takes its shape as arguments and declines a call it does not
take.

They were written for the dense Qwen3.8-27B ([Q36.md](Q36.md)) and lived in that family's folder until 2026-11-15. The move
changed no kernel: the 27B runs the same code under the same switches. What is new is that the shared decode dispatch, the
shared prompt GEMM and the Flash-Next loader know the pack, so a second family can load one.

## What is in it

| Header (`sycl/include/strata/b70/`) | Source (`sycl/src/b70/`) | What | Shapes it takes |
| --- | --- | --- | --- |
| `pack.hpp` | (host only) | the pack: record layout, sizes, the file and its index | inputs a multiple of 256, rows a multiple of 16 |
| `stream.hpp` | `stream_w4.dp.cpp`, `stream_w4x.dp.cpp`, `stream_w4h.dp.cpp` | stream products: a pack matrix read once against 1-32 activation columns (a decode step, a verify window, a batch's rows) | any pack matrix, 1-32 columns |
| `fat.hpp` | `fat_i8.dp.cpp`, `fat_w4i8.dp.cpp` | fat products: a prompt chunk's GEMMs on the matrix engine, pack and GGUF weights, fused normalise / quantize producers | inputs a multiple of 256; rows a multiple of 16 for the pack |
| `attn.hpp` | `attn_cache.dp.cpp` | decode attention over an 8-bit cache (K as int8 with a scale, V transposed in blocks of 32 cells) | head dimension 256, 6 query heads per KV head, up to 16 tokens a call |
| `attn_prompt.hpp` | `attn_prompt_i8.dp.cpp` | causal prompt attention over the same cache | head dimension 256, 6 query heads per KV head |
| `fat_bench.hpp` | `fat_bench.dp.cpp` | the pure-integer fat-product reference and a layout benchmark (`PARITY=ON`) | |
| `env.hpp` | | how a kernel reads a tuning switch | |

The stream products have three activation forms, by what the caller already has:

- `mmvq_w4`: the q8_1 image the shared quantizer writes (`native_quantize_q8_1`). A drop-in for `native_mmvq`.
- `mmvq_w4x`: int8 activations with one scale per 128 inputs; the 4-bit weights go to the matrix engine as they are.
- `mmvq_w4h`: FP16 activations, for many columns (no float work per column and 32 weights).

Measured on the 27B, the only model that runs them so far ([Q36.md](Q36.md) has the tables): the W4X stream product at 1.03 to
1.17 times the memory floor for 1 to 16 columns, the one-pass decode attention at 87 to 91% of the memory floor for 3 to 5 rows
at 118K to 256K cells, the prompt GEMM at 207 to 222 TOPS on most shapes (186 to 189 on the 17,408-wide one, 184 to 188 for the
fused FFN pair; `q36_mmvq_dense_parity --pp3-bench`, 2026-11-15).

## The pack

4-bit weights with a scale and a zero per output row and 128 inputs (`weight = scale x (q - zero)`), laid out for the matrix
engine instead of for a CPU decoder. A matrix is records of 16 output rows (a tile) x 256 inputs (a chunk), tile after tile,
chunk after chunk; a record is 2,176 bytes (4.25 bits a weight):

    4 x 128 dwords   the weights: pair p of the chunk's 32-weight sub-blocks (2p in the low nibbles, 2p + 1 in the high
                     ones), dword d (4 of the 32 bytes) of row i at [128 p + 16 d + i]
    2 x 16 dwords    group g of the chunk (128 inputs), row i at [512 + 16 g + i]: the FP16 scale in the low half, the
                     zero (1 .. 16) in the high half

A hardware thread owns a tile and reads its records in order with plain block loads (no gather, no table); a sub-block's 32
nibbles are one DPAS operand as they are. The file: `"B70W4\0\0\1"`, a u64 length, a JSON index
`{"tensors": {name: {off, bytes, n_in, n_out}}}`, then the blobs at 4096-byte aligned offsets. A matrix carries the name the
model's GGUF gives it, so an engine that loads matrices by name can take them from a pack and everything else from the GGUF.

### Making one

    python3 tools/b70_pack.py --gguf model.gguf [more shards] --out model.b70w4      # any model
    python3 tools/b70_pack.py --gguf model.gguf --out x --list                         # what would be packed, nothing written

`tools/b70_pack.py` takes every 2-D tensor whose name matches `--match` (default: the attention, recurrence and FFN
projections) and whose shape the pack takes, dequantizes it, and quantizes it again to 4 bits. It reads the GGUF's directory
with this repository's reader, so a tensor type the installed gguf package does not know does not stop it. `--match` picks
the kinds: `--match '\.(attn_gate|attn_q)\.weight$'` packs only those two.

**With calibration** (GPTQ: the error each input's rounding leaves is moved onto the inputs not yet rounded, weighted by how
the model's real inputs vary together, so the products change least instead of the weights). The engine supplies the
inputs:

1. Start the model as it is served, with `STRATA_B70_CALIB_DIR=<folder>` in the engine's environment and no prompt reuse
   (`--prompt-cache 0 --conversation-cache-mib 0`), and send it text of the kind it will see: about 20,000 tokens in
   prompts of a few thousand each (`max_tokens` 1), then one more prompt that is held out. The prompt path writes the FP16
   inputs of every projection a pack can hold into the folder (about 0.84 MB a token for Flash-Next: 20 GB for that run).
   Nothing else changes in that start; it is slow on purpose.
2. `python3 tools/b70_pack.py --gguf ... --out model.b70w4 --calib <folder>` (36 minutes on a 32-thread CPU for
   Flash-Next's 156 matrices). Each matrix's line gives the error of its products on the held-out prompt, calibrated and
   not: for Flash-Next 0.35 -> 0.08 on `attn_output`, 0.14 -> 0.07 on `ssm_out`, about 1.4 times lower on the rest.
3. Delete the folder.

Without `--calib` the tool rounds to nearest with a clip search per group of 128 inputs (`--search`). A pack made from a
GGUF is a second quantization of the GGUF's values either way: it is as good as its source at best. A checkpoint that was
quantized to int4 with calibration from the original weights (GPTQ, AutoRound, AWQ) keeps its own codes instead:
`tools/qwen35/b70w4_pack.py` packs one for the dense family and stays the way the 27B's pack is made.

**Check a pack on the card before serving it** (the table under "Flash-Next with a pack" is such a check).

### Loading one

| Family | Flag | Also |
| --- | --- | --- |
| Flash-Next (`qwen4exp`) | `--b70-pack FILE` | `STRATA_B70_PACK=FILE`; Settings key `b70_pack` |
| dense (`qwen35`) | `--w4 FILE` | `--b70-pack FILE` is the same flag |

In the Flash-Next family the loader (`sycl/src/core/native_dense.cpp`) takes from the pack the projections that reach their
weights through the two shared calls, when the pack holds them with the GGUF's shape: `attn_qkv`, `attn_gate`, `ssm_out`,
`attn_q`, `attn_k`, `attn_v`, `attn_output`. Their decode products then run `mmvq_w4` (through `native_mmvq`, type
`strata::b70::kTypeW4`), their prompt products the library's dequantize and the FP16 GEMM (through `Gemm::native`). It never
takes the shared expert's matrices, the PLE key or the routed experts: those have kernels of their own that read the GGUF's
blocks. Without the flag nothing of this runs. The start prints how many matrices came from the pack.

## Checked on the card (2026-11-15)

- **The move changed nothing.** The 27B with its pack, greedy, two prompts of 256 tokens: the binary built before the move
  and this one give the same tokens, at the same speed (74.9 and 75.0 tok/s; 135.7 and 135.6), with the same drafts accepted
  (162 of 400; 205 of 271) and the same peak VRAM (22.77 GiB).
- **The kernels against a float64 reference** (`q36_mmvq_dense_parity --w4 1,4,8,16,32 27`, built with `PARITY=ON`): the
  q8_1 and FP16 stream products within 3.6e-7 on seven shapes; the dequantize to FP16 bit for bit.
- **The shared dispatch** (new checks in the same run): `native_mmvq` with the pack type is `mmvq_w4` bit for bit at 1, 4, 8,
  16 and 32 columns on every shape; `Gemm::native` with the pack type is the FP16 GEMM on the dequantized matrix bit for bit
  when the scratch holds the matrix, and within 3.4e-7 when it runs in slices of whole tiles.
- Host side: `python3 -m pytest tools/test_b70_pack.py -q` (the record layout value by value, the quantizer, the file, the
  C++ reader on a file the tool wrote, a small GGUF through the command line).

## Flash-Next with a pack (measured 2026-11-15)

The first second model on the library: the served Flash-Next setup (UD-IQ1_S, the config of [B70.md](B70.md)) with its
attention and recurrence projections from a pack. In the served GGUF those 156 matrices are Q5_K and Q6_K (5.5 and 6.6 bits
a weight), so a pack is fewer bytes as well as a cheaper product, and it is a coarser quantization. Same binary in every
row; "GGUF" is the served setup.

**Speed, like for like** (`STRATA_DECODE_TIMING=1`: the verify time of a draft window by its size, which does not depend on
the text an arm happens to write; two requests a start):

| Window | GGUF | Calibrated pack, all 156 |
| --- | --- | --- |
| 2 rows | 16.4-16.7 ms | 14.9 |
| 3 rows | 18.3 | 16.7-17.0 |
| 4 rows | 20.3-20.4 | 18.5 |
| 5 rows | 23.4 | 20.2-20.4 |
| 6 rows | 25.8-26.0 | 22.4-22.6 |
| a whole window, average | 22.9-23.3 | 20.6-20.8 |

A window is 10.5% shorter. On generated text (greedy, 512 tokens, three runs a start) that shows as 124-131 -> 134-144
tok/s on the reasoning leaderboard prompt in every pack start; the code prompt moves between 102 and 125 tok/s from one
pack to the next (GGUF 104-108), because each pack writes another text and the drafts hold differently on it, so read the
window table. Projection weights in VRAM: 2,784 -> 2,238 MiB. The prompt read was not changed measurably: a 19.6K-token
prompt at 2,161 tok/s with a pack against 1,958-2,312 for the GGUF at the same place in a session (the first start of a
session read it at 1,128-1,492 with or without a pack).

**What it costs** (the KL kit of the 2026-11-11 check: a GGUF start captures greedy streams, each pack start replays the
same tokens; every start repeats itself bit for bit, and six short questions with one right answer were answered correctly
in every start):

| Pack | Matrices | VRAM saved | First token: KL, same top token (36 prompts) | Decode: KL, same top token (2,870 rows) | Perplexity |
| --- | --- | --- | --- | --- | --- |
| round to nearest, all | 156 | 546 MiB | 0.0017, 36 of 36 | 0.037, 94.8% | +3.47% |
| calibrated, all | 156 | 546 MiB | 0.00067, 36 of 36 | 0.0103, 97.1% | +1.42% |
| calibrated, without `ssm_out` and `attn_output` | 108 | 409 MiB | 0.00055, 36 of 36 | 0.0076, 97.3% | +1.04% |
| calibrated, `attn_qkv` only | 36 | 141 MiB | 0.00026, 36 of 36 | 0.0042, 97.9% | +0.57% |
| calibrated, `attn_gate` and `attn_q` only | 48 | 260 MiB | 0.00023, 36 of 36 | 0.0027, 98.5% | +0.24% |

For scale: the served kernels against exact ones measured KL 0.0009, 99.1% and +0.07% on this kit; two backends running
the same file are at KL 0.038 in the repository's notes.

**Reading.** Calibration (GPTQ on the projections' own inputs, captured from the engine) takes the loss from 3.5% to 1.4%
and keeps all of the speed. It does not reach the level of the served kernels: 4 bits a weight is the limit here, not the
calibration (half the calibration tokens moved a typical matrix's held-out error from 0.055 to 0.058; damping from 0.001 to
0.2 moved it by less). The loss is spread over the projections, so the choice is a trade:

- `attn_gate` and `attn_q` only: +0.24% perplexity, a window 1 to 4% shorter (two requests: 22.0 and 24.1 ms a window
  against 22.9-23.3; its 3-row windows 17.7-18.1 ms against 18.3), 260 MiB of VRAM.
- all 156: +1.42%, a window 10.5% shorter, 546 MiB.

**Nothing is on by default.** Both packs are choices for whoever serves the model (`--b70-pack`, Settings "b70_pack").

## Calling it from a model family

1. Load: `strata::b70::pack::File` (open, `find(name, n_in, n_out)`, `read`), the matrix into a device buffer aligned to
   `pack::kDeviceAlign`. Give it the type `strata::b70::kTypeW4` where the family keeps a ggml type id.
2. Decode and verify: `native_mmvq(kTypeW4, ...)` with the q8_1 image (leave 28 readable bytes after its last block), or
   call `stream::mmvq_w4x` / `mmvq_w4h` with their own activation forms.
3. Prompt chunks: `Gemm::native(X, kTypeW4, ...)`, or `fat::gemm_w4_i8_pp3` with `fat::quantize_x128` activations.
4. Attention: `attn::attend2` when the cache has the layout in `attn.hpp` and the head shape it takes.

## Build

`STRATA_BUILD_B70LIB` (build.sh: `B70LIB`, default ON) builds `libstrata_b70.so` and `libstrata_b70_fat.so` beside the binary
and defines `STRATA_B70_LIB` for every target; the shared sources reach the library only behind that macro. `B70LIB=OFF`
builds `strata` without it (and needs `Q36=OFF`: the dense family is built on it). `STRATA_B70_GRF=128` at configure time
builds `libstrata_b70.so` with 128 registers a thread instead of 256. `python3 tools/check_q36_isolation.py` checks the
layout (the library's section, its sources, the macro's users).

## Switches

Tuning switches of the kernels. Each still answers to the name it had in the dense family (second column).

| Variable | Earlier name | Does |
| --- | --- | --- |
| `STRATA_B70_W4_C`, `STRATA_B70_W4H_C` | `STRATA_Q36_W4_C`, `_W4H_C` | threads a 16-row tile of the q8_1 / FP16 stream products (1, 2, 4) |
| `STRATA_B70_W4X_KS` | `STRATA_Q36_W4X_KS` | the W4X product's split of a tile's inputs between threads |
| `STRATA_B70_XATT2_FORM`, `_SEGS`, `_U16`, `_EXP`, `_PFD`, `_THREADS`, `_DBG` | `STRATA_Q36_XATT2_*` | the one-pass decode attention's form, segments, operand width, exp, prefetch distance, threads, bounds check |
| `STRATA_B70_XATT_PARTS`, `_XATT_DBG` | `STRATA_XATT_PARTS`, `_XATT_DBG` | timing aids of the three-kernel decode attention |
| `STRATA_B70_PP3_PANEL`, `_PP3_THR`, `_PP2_WGS`, `_PP2_FUSE`, `_W4F16_CFG` | `STRATA_Q36_*` | the fat products' column panel, threads, work-group size, fused producers |

`STRATA_Q36_PP2`, `STRATA_Q36_PP3` and `STRATA_Q36_XR_THREADS` keep their names: the dense family's driver reads them too.

## Not done

- **A pack of all of Flash-Next's projections at the served kernels' quality.** Calibrated it costs 1.4% perplexity; at 4 bits
  a weight that looks like the limit. A finer scale (one per 32 inputs, 4.5 bits a weight) or 5 bits for the matrices that
  cost most are the formats that could close it; neither exists in the kernels.
- Calibration on the engine's inputs uses the unquantized model's activations for every matrix; the usual refinement (each
  layer calibrated on the inputs the already quantized layers produce) is not built.
- A pack's prompt product is the dequantize and the FP16 GEMM, not the integer fat product the 27B uses.
- The attention kernels are fixed at a head dimension of 256 and 6 query heads per KV head.
- The routed experts stay in their GGUF formats (about 2 bits a weight). 2-bit integers are a native matrix-engine operand
  on this card, but a 2-bit pack is another quantization of those weights and its quality is not known.
- The fat product is at 78 to 80% of a pure-integer product on the same shapes and the same records (207-222 against 262-282
  TOPS, 72% on the 17,408-wide shape; measured 2026-11-15). The difference is the scale and zero of every group of 128 inputs,
  which the format needs: the room left is about a quarter, not the 1.7 times the microbenchmark's 308-325 TOPS suggested
  (that figure was another shape, 8,192 inputs, without scales). A layout with the weights contiguous measured no faster.
- The GGUF decode forms tuned for the 27B's shapes (`q36_mmvq_dense`, `q36_xr_dense`) are still that family's sources.
