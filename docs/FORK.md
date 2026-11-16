# What is different from upstream Strata

This is a fork of [Strata](https://github.com/Niko1221/Strata) by Niko1221 and its contributors. It lives at
[riceharvest/strata-b70-only](https://github.com/riceharvest/strata-b70-only), branch `b70`. It supports one setup: one
Intel Arc Pro B70 (32 GB, Battlemage `bmg-g31`) on Linux with the `xe` driver, using the SYCL engine in `sycl/`. If you
have another GPU, use upstream.

This page and [CLI.md](CLI.md) describe the tree at `1b121d6db` plus the documentation commits of 2026-11-16. If other work removes more
code later (for example the two-card paths), the "not removed" list below and CLI.md need checking and regenerating.

Every claim below can be checked with git or in the tree. The commands are given where a number comes from.

## Where the fork starts

- The merge base with upstream's `main` is `870aedd9e` (2026-10-29): `git merge-base HEAD upstream/main`.
- Since then this branch has 437 commits that are not merges: `git rev-list --count --no-merges 870aedd9e..HEAD`
  (counted on 2026-11-16, before this page was added).
- The fork has not merged upstream since then. Upstream's `main` is 5,525 commits ahead: `git rev-list --count HEAD..upstream/main`
  (after `git fetch upstream`, 2026-11-16). Fixes and features that landed upstream after 2026-10-29 are not here unless a
  commit here says it ported them.
- Against that merge base the tree differs in 4,273 files, 127,896 lines added and 895,118 removed:
  `git diff --shortstat 870aedd9e HEAD`. Most of the removed lines are the engines and apps listed below.

### Share of the code that is upstream's

`scripts/upstream_share.py` does not exist in this tree, so there is no official number. A rough one, from `git blame -w`
on 2026-11-16: of 465,653 lines in code files (`.cpp .hpp .h .cu .py .sh .cmake .js .html .css`, `CMakeLists.txt`; not
`docs/`, `third_party/` or `ref/`; 1,210 files), 73.7% were last changed by a commit that is in `upstream/main`'s history
and 26.3% by a commit that only this fork has. By language: Python is mostly upstream's (145,435 lines against 13,859);
C++ is split (131,389 against 94,782 in `.cpp`, 50,700 against 11,379 in `.hpp`). A commit that was cherry-picked from an
upstream pull request has a new hash here, so its lines count as the fork's; the community pull requests listed in
[B70.md](B70.md) are such commits (for example `e14ddfa4b` is not an ancestor of `upstream/main`, its upstream twin is
`8fffe81d4`). The fork's share is therefore an upper bound. The script that made the number was not kept in the repository.

## What was removed

Checked against the tree (`ls`) and the commits named here:

| Removed | Commit |
| --- | --- |
| The CUDA and HIP engine sources in `src/` and `include/`, and the top-level CUDA/HIP build | `31663d004`, `bc1374052` |
| The Apple Metal engine and the Mac packaging (`metal/`) | `8799ccb06` |
| The desktop app (`apps/`), the Windows and macOS packages (`packaging/`), their tests and release workflows | `454b36fb2` |
| The Windows `.bat` launchers | `b7ee62cf9` |
| The CUDA/HIP/Mac/Windows engine packaging | `a09d0b820`, `8757cbb21` |
| The upstream release image (`docker/`), the DGX Spark path, the translated READMEs | `3a381d34c`, `0e8538b31` |
| NVIDIA and AMD benchmark results in `bench/results/` | `17d88442d` |
| Doc pages that covered only NVIDIA, AMD or Apple: `AMD_HIP.md`, `MAC.md`, `MULTI_GPU.md`, `OLDER_GPUS.md`, `STRIX_HALO.md`, `NVIDIA_V100.md`, `NVFP4.md` and others | `50699582e` |
| The Qwen3.6-35B-A3B family (its engine path was CUDA and HIP only) and its test harness | `9bceebd97`, `f5f7b6b27` |
| The NVFP4 family and its conversion tools | `9a2d21f79`, `036107a55` |
| DeepSeek V4.1-Flash: engine path, setup, launch, server and docs. One B70 cannot run it at usable quality | `5b6775385`, `31b801f8c`, `6e840aa01`, `db3571742` |
| The Strix Halo Windows page and the media kit (this documentation pass) | `06b93046f` |

Things that are not removed, so you do not trust the list too far:

- **Two-card code is still in the tree.** The guides for layer splits (`MULTI_GPU.md`, `SECOND_GPU.md`) are gone, but code such as
  `sycl/src/core/peer_experts.cpp`, the `--layer-split` and `--gpus` options of setup, and split serving ([SPLIT_SERVE.md](SPLIT_SERVE.md))
  were not deleted. This fork is tested on one card. also `serve/test_split.py`, `sycl/src/core/pipeline_auto_test.cpp`,
  `tests/sycl/test_intel_two_card_spec.py` and the Arc Pro B60 report in `bench/results/`.
- **Other Intel cards are not removed either.** [A770_SPEED_SPEC.md](A770_SPEED_SPEC.md) and the Arc B60 and B65 reports are kept as
  history. The fork is built for and measured on the B70.
- **Some pages still describe the removed engines.** [DETAILS.md](DETAILS.md) is upstream's reference page and still has CUDA and
  HIP rows. The `usage()` text of the engine still has CUDA wording ([CLI.md](CLI.md#engine-flags)). Treat a page that
  is not about the B70 as upstream text until it says otherwise.

## Which model families are live

| Family | Where | Notes |
| --- | --- | --- |
| Qwen3.8-Flash-Next (GGUF architecture `qwen4exp`, or a pack) | the `strata` binary, [B70.md](B70.md) | The served setup is Unsloth's `UD-IQ1_S`: `./setup.sh --backend sycl --family unsloth --model UD-IQ1_S` |
| Dense Qwen3.8-27B (GGUF architecture `qwen35`; called Q36 in the sources) | the same `strata` binary, [Q36.md](Q36.md) | Read from the GGUF at start. `Q36=OFF` for `sycl/tools/build.sh` builds `strata` without it |

Both are chosen by `strata` from the architecture of the GGUF it is given. There is no second binary any more.

Not live: Qwen3.6-35B-A3B, DeepSeek V4.1-Flash and the NVFP4 packs (see the table above).

## Things that exist only on the fork

- **The B70 pack and `--w4`.** A GPTQ int4 checkpoint packed for the card's matrix engine, loaded with `--w4 FILE` (the dense
  family) or `--b70-pack FILE` (Flash-Next, also `STRATA_B70_PACK`). The kernels and the pack are the B70 card library,
  [B70_LIB.md](B70_LIB.md): `sycl/src/b70/`, `tools/b70_pack.py`, `tools/qwen35/b70w4_pack.py`. First commit of the 27B's pack
  was `c32771ee7` (2026-11-12). Nothing in `upstream/main` has these files.
- **The DFlash2 drafter for the 27B.** `STRATA_Q36_DFLASH=<pack>` swaps the MTP draft chain for the DFlash2 block drafter. It is
  opt-in. The pack is made by `tools/qwen35/dflash_pack.py` (`f4e2046e9`, 2026-11-14). The measurements and the arbiter that
  moves a request back to MTP are in [Q36.md](Q36.md).
- **B70 defaults.** `strata::apply_b70_defaults` in `sycl/include/strata/sycl_queue.hpp` sets the B70 switches unless you set
  them. `STRATA_B70_DEFAULTS=0` turns all of them off; a switch set to `0` turns one off. `git grep apply_b70_defaults upstream/main` finds nothing.
- **The served setup.** `./setup.sh --backend sycl --family unsloth --model UD-IQ1_S` downloads at a pinned revision, grafts one
  tensor into shard 2 (`tools/graft_tensor.py`), builds the pack and writes the served flags into the config.
  `--no-served-recipe` leaves them out. Neither it nor `tools/graft_tensor.py` exists in `upstream/main`. See [B70.md](B70.md#the-served-setup).
- **Many `STRATA_*` switches.** Counted with `git grep -h -o -E "STRATA_[A-Z0-9_]+"` over the tree outside `docs/`: 334 names
  appear in the tree now that did not appear at the merge base. That count includes names in comments and tests, so it is
  an upper bound on switches. The ones that matter are in [B70.md](B70.md), [Q36.md](Q36.md), [B70_LIB.md](B70_LIB.md) and
  [INTEL.md](INTEL.md). A switch is `STRATA_*`, read once; a hot path change has an off switch ([AGENTS.md](../AGENTS.md)).
- **A pinned device compiler.** The `strata-sycl-dev` image and `sycl/tools/build.sh` are upstream's; the pin of the compiler
  (IGC 2.41.5) was added here (`c1b16af25`, 2026-11-11, `git log -S"2.41.5" -- sycl/tools`).
  [release-notes-b70-2026.11.11.md](release-notes-b70-2026.11.11.md) says why.

## Upstream features the fork still carries

- **DFlash2 in the terminal.** `strata launch --drafting dflash2` and `strata drafting plan|fetch`
  were added upstream in 0.2.3 ([release-notes-0.2.3.md](release-notes-0.2.3.md)) for Flash-Next. They are still in the
  command line ([CLI.md](CLI.md)). No B70 measurement of that path is recorded in [B70.md](B70.md).

## What stays upstream on purpose

- `LICENSE` is upstream's MIT text, with "Niko1221 and the Strata contributors". The fork keeps it.
- [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and the credit to the engine's authors in the README.
- "Local port of ... pull/NNNN" comments in the source keep the upstream pull request numbers they came from.
- `DEFAULT_FEED` in `strata_cli/update_cmd.py` still points to `https://github.com/Niko1221/Strata/releases/latest/...`, and the
  install one-liners in [INSTALL.md](INSTALL.md) fetch upstream's release. Both are upstream's installer for upstream's builds,
  not for this fork's. This fork is built from source.
- Upstream issue and pull request numbers quoted in messages and release notes are upstream's, not this repository's.

## Where issues and reports go

To `riceharvest/strata-b70-only`, never to upstream: upstream has no B70 and cannot reproduce a report made on one. See
[CONTRIBUTING.md](../CONTRIBUTING.md) and [COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md).
