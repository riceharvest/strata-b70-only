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
at 118K to 256K cells, the prompt GEMM at 184 to 186 TOPS (the card's microbenchmark kernel reaches 308 to 325: open).

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
projections) and whose shape the pack takes, dequantizes it, and quantizes it again to 4 bits with a clip search per group
of 128 inputs (`--search`, default 4 narrower ranges tried). It prints each matrix's relative error and lowest row
correlation with its source.

A pack made this way is a second quantization of the GGUF's values: it is as good as its source at best, and the lowest
error comes from a higher-precision GGUF (F16, BF16, Q8_0). A checkpoint that was quantized to int4 with calibration (GPTQ,
AutoRound, AWQ) keeps its own codes instead: `tools/qwen35/b70w4_pack.py` packs one for the dense family and stays the way
the 27B's pack is made.

**Check a pack made from a GGUF on the card before serving it.** The one that has been run (Flash-Next, below) was faster and
measurably worse.

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

The first second model on the library: the served Flash-Next setup (UD-IQ1_S, the config of [B70.md](B70.md)) with its 156
attention and recurrence projections from a pack that `tools/b70_pack.py` made from the served GGUF. In that file those
matrices are Q5_K and Q6_K (5.5 and 6.6 bits a weight), so the pack is fewer bytes as well as a cheaper product, and it is
a coarser quantization: the tool reports a relative error of 11 to 14% a matrix.

Same binary, three starts in the order GGUF, pack, GGUF; greedy, the two leaderboard prompts on a fixed text, 512 tokens,
three runs a start:

| | GGUF (first start) | Pack | GGUF (third start) |
| --- | --- | --- | --- |
| reasoning text, tok/s | 124.2 / 131.0 / 127.3 | 141.2 / 142.5 / 143.7 | 125.6 / 127.5 / 129.0 |
| code text, tok/s | 103.8 / 108.3 / 106.4 | 110.5 / 112.8 / 111.4 | 106.7 / 107.5 / 106.3 |
| answer after a 19.6K-token prompt, tok/s | 113.3 | 125.6 | 117.5 |
| projection weights in VRAM | 2,784 MiB | 2,238 MiB | 2,784 MiB |

That is about 12% faster on the reasoning text and 5% on the code text. The 19.6K prompt was read at 1,492 / 2,161 / 2,312
tok/s in that order: the two GGUF starts differ more than the pack differs from either, so the prompt read is not decided.
Six short questions with one right answer were answered correctly in every start; the texts differ from the first words on.

What it costs, with the KL kit of the 2026-11-11 check (the GGUF start captures greedy streams, the pack start replays the
same tokens; both starts repeat themselves bit for bit):

| | Pack against GGUF | For scale: the served kernels against exact ones (2026-11-11) |
| --- | --- | --- |
| first token after a prompt, 36 prompts | KL mean 0.0017, highest 0.012, same top token 36 of 36 | KL mean 0.00006, 36 of 36 |
| decode, 2,871 forced rows | KL mean 0.037 (median 0.0006), same top token 94.8% | KL 0.0009, 99.1% |
| perplexity of the forced stream | +3.5% | +0.07% |

So a pack made from this GGUF is a real loss of quality for the speed: about the distance between two backends running the
same file (the repository's yardstick for that is KL 0.038, 94.5% same top token). **It stays off, and this pack is not a
recommendation.** What the run shows is that the hooks work on a second model and what the lever is worth. The loss is the
quantizer's (round to nearest on values that were already quantized, no calibration), not the format's: the 27B's pack keeps
a calibrated checkpoint's own codes. A calibrated 4-bit quantization of these 156 matrices from the original weights is the
way to keep the speed without this cost; it has not been made.

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

- **A good pack for Flash-Next.** The one measured above costs 3.5% perplexity. A calibrated 4-bit quantization of its
  projections from the original weights has not been made.
- The prompt read with a pack is not decided (see the table's note), and the pack's prompt product is still the dequantize
  and the FP16 GEMM, not the integer fat product the 27B uses.
- The attention kernels are fixed at a head dimension of 256 and 6 query heads per KV head.
- The routed experts stay in their GGUF formats (about 2 bits a weight). 2-bit integers are a native matrix-engine operand
  on this card, but a 2-bit pack is another quantization of those weights and its quality is not known.
- The fat product is at 184 to 186 TOPS against the 308 to 325 the card's microbenchmark kernel reaches.
- The GGUF decode forms tuned for the 27B's shapes (`q36_mmvq_dense`, `q36_xr_dense`) are still that family's sources.
