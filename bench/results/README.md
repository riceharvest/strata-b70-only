# bench/results: what is in here

Six entries: three community benchmark folders on Intel Arc cards, the two files of one CUDA receipt for an opt-in feature, and
one older index. None of them was measured on an Arc Pro B70. The B70 numbers of this fork are in [docs/B70.md](../../docs/B70.md),
[docs/Q36.md](../../docs/Q36.md) and the README, each with its own conditions.

The NVIDIA and AMD folders that used to be here were removed with their engines (`17d88442d`, 2026-10-24). A few old
documents still name them; those links go to upstream.

"Measured" is the date the file itself states. "In repo since" is `git log --diff-filter=A` for the path: every entry came in
with the merge `a539e12c9` on 2026-10-22. "Conditions not recorded" means the file does not say.

| Entry | What it measured | Measured | In repo since | Hardware | Model and quant | Conditions as the file states them |
| --- | --- | --- | --- | --- | --- | --- |
| [2026-10-04-community-2x-arc-pro-b60/](2026-10-04-community-2x-arc-pro-b60/) | Speed of the SYCL port on two cards, one request at a time. A short test: two prompt sizes, 2 to 3 requests each; one run on a single card. Also a correctness spot check. Reported by LocalXPU. | 2026-10-04 | 2026-10-22 | 2x Intel Arc Pro B60 24 GB, PCIe 3.0 x8 each, Ryzen 5 5600, 64 GB DDR4; Ubuntu 24.04, kernel 6.17 (`xe`), oneAPI DPC++ 2026.1.1 | Qwen3.8-Flash-Next GSQ-RCO Coder IQ1_M (`--layer-split 24`) and IQ2_XS (`--layer-split 25`) | Strata `6f32ec0` plus two `sycl/` fixes, engine `0.1.39-sycl`, source build. Context 8,192, KV int8, `--spec 4 --spec-min-p 0.5`, greedy, thinking off, no vision. Storage type not recorded. Time to first token not measured |
| [2026-10-05-community-arc-b65/](2026-10-05-community-arc-b65/) | Speed and output stability of one Arc Pro B65: five fixed synthetic tasks at 512 and 7,000 exact input tokens, 640 output tokens each, three fresh engine launches. Not an answer-quality test. Reported by timnevits. | 2026-10-06 (the folder name says 10-05) | 2026-10-22 | 1x Intel Arc Pro B65 32 GB (`8086:e222`), PCIe Gen4 x16, 200 W cap, Core i5-12600K, 128 GB DDR4-3200, Samsung 980 NVMe; Ubuntu 26.04.1, kernel 7.0.0-38, oneAPI 2026.1.0 | Qwen3.8-Flash-Next GSQ-RCO IQ2_XS, full 512 experts, MTP draft packed to Q2_0 with upstream's tools | Strata v0.1.40 source with six local patches ([BUILD.md](2026-10-05-community-arc-b65/BUILD.md)). Context 8,192, KV int8, prefill 512, `--spec 4 --mtp-max-t 4 --spec-min-p 0.5`, greedy seed 17, thinking off, pinned 16 GiB host mirror, `STRATA_VERIFY_NO_HOST=1`. Page cache not flushed |
| [2026-10-07-community-arc-b65-v01402/](2026-10-07-community-arc-b65-v01402/) | The same card on the official v0.1.40.2 release with no local engine patches: three memory profiles (8K context with prefill 512 and 4096, 262K context with prefill 512), matched short and 7K inputs. Reported by timnevits. | 2026-10-07 | 2026-10-22 | 1x Intel Arc Pro B65 32 GiB, Gen4 x16, 200 W cap, i5-12600K, 128 GB DDR4-3200, Samsung 980 NVMe; Ubuntu 26.04.1, kernel 7.0.0-38 | Qwen3.8-Flash-Next GSQ-RCO IQ2_XS (full 512 experts), original Q2_0 MTP pack | Commit `e8ca9af`, native JIT build (the engine labels itself `0.1.40-sycl`). One request, KV int8, MTP window 4, min-p 0.5, thinking off, greedy seed 17, caches off. `results.json` was trimmed from the folder ([TRIMMED.md](2026-10-07-community-arc-b65-v01402/TRIMMED.md)) |
| [exchange-rotation-ab.json](exchange-rotation-ab.json) | A same-binary off/on test of `STRATA_EXCHANGE_ROTATE=1` ([docs/EXCHANGE_ROTATION.md](../../docs/EXCHANGE_ROTATION.md)): 1,024 prompt tokens, 1,024 generated, one pair. Output tokens were identical. | 2026-10-04 | 2026-10-22 | **NVIDIA RTX PRO 6000 Blackwell**, Ryzen 9 7950X, MemTotal 130,975,976 kB, kernel 7.0.0-34, CUDA 13.2, arch 120 | Qwen3.8-Flash-Next GSQ-RCO Q2_0 (512 experts, 48 layers) | Source commit `18a30ad`, FP16 KV, context 40,960, `--expert-cache 12000` (capped on purpose), target-only decoding, greedy. The file's own limits: one off/on pair, no general speedup claim, AMD and Windows untested. It is a CUDA run, not a SYCL one |
| [exchange-rotation-token-ids.json](exchange-rotation-token-ids.json) | The 1,024 input token IDs and the output IDs of the two arms above, kept so the hash in the receipt can be checked | 2026-10-04 (same run) | 2026-10-22 | as the receipt | as the receipt | Conditions not recorded in the file itself; see the receipt |
| [COMMUNITY.md](COMMUNITY.md) | An index of the community folders ported for 0.1.40.2 and 0.1.41. After the 2026-10-24 removal it lists only the two B65 folders and does not list the B60 folder | not applicable | 2026-10-22 | Arc Pro B65 rows only | | Its numbers are copied from each folder's README, as it says |

## What the folders contain

- Each community folder has a `README.md` with the full conditions and method, the exact launch arguments (`profile*.json`,
  `data/*.json`), the per-run results (`results.csv`, `summary.json`) and the scripts that made them.
- Raw dumps over 100 KB were left out when the folders were ported; `TRIMMED.md` in a folder says which.
- Paths and names were made generic by their authors. Nothing here was re-run for this fork.

## Notes and doubts

- The two B65 rows were measured with upstream's `0.1.40` source on a B65, not a B70. They show that the SYCL port runs on
  that card and what one slow desktop reached then. They are not a baseline for this fork's engine, which has changed since.
- `exchange-rotation-ab.json` is a CUDA run (the file says CUDA 13.2). The page that cites it
  ([EXCHANGE_ROTATION.md](../../docs/EXCHANGE_ROTATION.md)) is kept as upstream history. The switch is in the SYCL sources too, but no B70 run of it is recorded here.
- For how to add a result of your own, see [docs/COMMUNITY_BENCHMARKS.md](../../docs/COMMUNITY_BENCHMARKS.md).
