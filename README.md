# Strata for the Intel Arc Pro B70

A fork of [Strata](https://github.com/Niko1221/Strata) that runs on one GPU only: the **Intel Arc Pro B70** (32 GB,
Battlemage `bmg-g31`) on Linux with the `xe` driver.

The NVIDIA (CUDA), AMD (HIP/ROCm) and Apple (Metal) runtimes are removed. What's left is Strata's engine ported to SYCL
(`sycl/`), tuned for this one card, plus Strata's server, setup and APIs. If you have a different GPU, use
[upstream Strata](https://github.com/Niko1221/Strata).

Not affiliated with Intel or with upstream Strata.

## What runs

| Model | Status on one B70 |
| --- | --- |
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), Unsloth's [UD-IQ1_S](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (the file set this fork serves, setup's choice on one B70 with 64 GB of RAM) | Usable at 128K-256K context, chats and coding agents included. The numbers under [Speed](#speed) were measured on the earlier IQ2_XS file ([ISTA-DASLab GSQ-RCO GGUFs](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)); the UD-IQ1_S ones are in [docs/B70.md](docs/B70.md) |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (dense), Unsloth's [UD-Q4_K_M and UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), on a second engine, `strata-q36` (optional, off by default: `Q36=ON`) | **Experimental: ahead of llama.cpp for this model, not yet level with vLLM.** A request as a client sends it (sampled, thinking on) answers at about 82 tok/s, a greedy one at 67-86; llama.cpp's best runs on this card are 44-48 tok/s and vLLM 88-110. The whole model is in VRAM. It fits 262,144 tokens of context on the one card (8-bit K/V cache) and reads a 118K-token prompt at 1,097 tok/s. With a GPTQ int4 checkpoint packed for the card (`--w4`, the weights vLLM serves): 121 and 85 tok/s on the two greedy test prompts (vLLM on this card with that checkpoint: 110 and 88), a whole-file edit at 287, prose after a 25K-token prompt at 67, the answer after a 118K-token prompt at 49 (256K context); code generation (114) and several requests at once (163 tok/s over 4) are behind the fastest vLLM setups. Measured 2026-11-12. [docs/Q36.md](docs/Q36.md) |

## What counts as usable here

A model and setting is called usable when it does all of this on one B70:

- 128K to 256K tokens of context
- at least 1,000 tokens/s reading the prompt
- at least 10 tokens/s writing the answer

Anything that misses one of these is labelled "works, not usable".

**DeepSeek V4.1-Flash is not maintained** (since 2026-10-30). The code stays in `sycl/src/dsv41`, and its build is off by default
(`DSV41=ON` for `sycl/tools/build.sh`). On one B70 the 2-bit pack plus expert pruning cost too much answer quality: a pruned model
was fluent but had lost knowledge (in our tests it got simple facts wrong). Unpruned, most experts come off the disk on every token.

## Speed

Qwen3.8-Flash-Next UD-IQ1_S (the served recipe setup writes) on one Arc Pro B70, Ryzen 9 5950X, 64 GB RAM, measured
2026-11-11. Settings:

- MTP draft, `--spec 6`, int8 KV, 32K of the KV in VRAM, `--prefill 16384`, `--max-context 262144`
- 3 GB of VRAM kept free for the desktop; `STRATA_MIRROR_MIB=25000`
- one start; every prompt begins with a new line of its own, so nothing is reused from an earlier request
- long prompts: this repo's docs with a fact hidden in the middle (every answer found it), thinking off, temperature 0,
  an answer of up to 900 tokens, two runs each

| Context | Prompt (tok/s) | Answer (tok/s) |
| --- | ---: | ---: |
| 118K (118,227 tokens) | 3,475 / 3,356 | 85.6 / 86.5 |
| 250K (249,650 tokens) | 3,325 / 3,329 | 84.7 / 87.5 |

Short cold prompts (one user message, nothing cached, same settings, the median of three), prompt read time and time to
the first token:

| Prompt | Read (tok/s) | First token |
| --- | ---: | ---: |
| 139 tokens | 320 ms (435) | 356 ms |
| 384 tokens | 377 ms (1,018) | 412 ms |
| 1,378 tokens | 673 ms (2,047) | 726 ms |
| 5,303 tokens | 1,863 ms (2,846) | 1,967 ms |

A short prompt cannot reach the long prompts' speed: about a third of the experts live in RAM, and a few hundred tokens
already route to most of them, so every read has to bring most of those across PCIe whatever its length.

The two [localmaxxing](https://localmaxxing.com) prompts (greedy, 512 tokens, the same text three times after a
warm-up): reasoning-v1 129.6 / 130.2 / 131.2 tok/s, code-v1 106.6 / 108.2 / 109.4 tok/s. With 700 MiB of VRAM kept free
on a card that drives no desktop: 129-134 and 105-110 tok/s. A single run with another closing line in the prompt reads
up to 10% lower or higher, because the text the model writes changes with it.

Does the tuning change what the model says? Measured 2026-11-11: the served kernels against their exact alternatives give
the same top token in 36 of 36 prompt reads and in 99.1% of 2,867 decoded positions, and llama.cpp on the CPU scores the
same as this engine on the same file on an instruction-following probe
([docs/B70.md](docs/B70.md#instruction-following-the-served-kernels-against-exact-ones-and-llamacpp-2026-11-11)).

[docs/B70.md](docs/B70.md) has what changed since upstream, who wrote it, and every earlier measurement (the earlier
GSQ-RCO IQ2_XS files read 3,343-3,402 tok/s at 118K and answered at about 77 tok/s on 2026-11-02; upstream Strata on this
card was 12-17% slower on prompts on 2026-10-24, before this work).

**A coding agent on one B70, up to 256K** (measured 2026-10-31 to 2026-11-02 on the earlier IQ2_XS files). The [pi](https://github.com/earendil-works/pi) agent with thinking `xhigh`,
`--max-context 262144` and 3 GB of VRAM kept for the desktop. A session with a 170K-token system prompt (a 156K-token
`AGENTS.md`) ran eleven coding tasks: 82 requests, the context growing from 170K to 245K tokens, where pi compacted
the conversation, then on from 195K to 225K. Every task finished. (Rerun on 2026-11-01 with that day's work: the cold
170K-token first prompt 60.0 -> 50.6 s, small turns 418 -> 319 ms, turns of 100-2,000 new tokens 709 ms, answers at
87 tok/s; with the later VRAM ring and sampled-draft acceptance, small turns 288 ms and answers 92.7 tok/s. On 2026-11-02, after a night on pi's turns: answers 102.7 tok/s, small turns 246 ms, turns of 100-1,000 new tokens 573 ms, the cold prompt 50.4 s, every task finished; [docs/B70.md](docs/B70.md#pi-round-2026-11-02-night).) Each turn reused the history it had already read and
read only the new tokens (64 of 79 turns under 1.5 s); answers ran at 73-103 tok/s. After the compaction the next
prompt reused the 170K-token system prompt and read only the summary and recent turns (26K tokens, 11 s). An earlier
session from 64K to 118K ran the same way. Turns that continue a conversation used to hang on this card; the engine
now works around it (2% prompt speed), see [docs/B70.md](docs/B70.md#agent-use-and-the-second-turn-hang-2026-10-31).

## What you need

- An Intel Arc Pro B70, Linux with the `xe` kernel driver and the Intel compute runtime (`ls /dev/dri` shows
  `renderD128`)
- Docker (or Podman). The engine builds and runs in a container, so the host needs no oneAPI
- 64 GB of RAM (what it's measured on)
- Disk: about 75 GB for UD-IQ1_S (66 GB of model files and 5 GB of draft layer; IQ2_XS takes about 80 GB)

### The rest of your PC matters too

Every number in this repo comes from one older desktop: a Ryzen 9 5950X (Zen 3, AVX2, no AVX-512) with dual-channel DDR4 on a
PCIe 4.0 x16 slot. Treat them as a floor; on a newer platform expect more, but nobody has measured that here yet.

- Qwen3.8-Flash-Next keeps the experts that don't fit in VRAM in RAM and reads them over PCIe on every token; that link is the
  biggest cost in decode. The B70 is a PCIe 5.0 card, so a PCIe 5.0 x16 slot can carry up to twice what this box does.
  Faster RAM on its own changes little here: dual-channel DDR4 already outruns a PCIe 4.0 link.
- Run the card in an x16 slot wired to the CPU. An x8 or chipset slot halves the link or worse
  (`cat /sys/bus/pci/devices/*/current_link_width` and `current_link_speed` show what you got).

## Install

1. Docker, once:

       sudo apt install docker.io git        # or your distro's equivalent
       sudo usermod -aG docker,render $USER  # then log out and in again

2. Get the code and build the engine. The image takes about 6 minutes once (13 GB); the engine takes about 11 minutes on
   a 16-core desktop:

       git clone -b b70 https://github.com/riceharvest/strata-b70-only && cd strata-b70-only
       docker build --pull -t strata-sycl-dev sycl/tools
       H=$(basename "$PWD")
       docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e AOT=bmg-g31 -e REPO=/work/$H \
           -e BUILD_DIR=/work/$H/build-sycl-aot strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"

   The build ends with `BUILD EXIT 0` and `errors: 0`.
   The image pins the device compiler (IGC 2.41.5, ocloc 26.35.39758.10): a newer one made answers 3-5% slower (docs/B70.md, "Build and run").

3. Set up the model and serve it:

       ./setup.sh --backend sycl --family unsloth --model UD-IQ1_S --context 262144 --vision no

   On one B70 with 64 GB of RAM, a first `./setup.sh --backend sycl` with no `--family` or `--model` preselects the same
   model and context in its menu (`--yes` takes them). That run keeps pictures on, which the served setup does not
   have: use the command above for the served one.
   Setup downloads Unsloth's UD-IQ1_S files at a pinned revision, checks them against pinned sizes and SHA-256 sums, and
   takes one tensor (layer 18's gate experts) from the UD-IQ1_M file of the same repository into shard 2, about 216 MB read
   (the file as published cannot be packed: `docs/QUANTS.md` says why; nothing is quantized again). It then builds the pack,
   fetches the draft layer (about 5 GB of tensors from the Qwen checkpoint, read by range requests), writes a config
   (`strata-unsloth-ud-iq1_s.json`, its `exe` is `sycl/serve/strata-sycl.sh`) and a `run-unsloth-ud-iq1_s.sh` that starts
   the server, and starts it. The config carries the settings this fork serves with: `--spec 6`, the coding agent's expert
   profile and draft vocabulary, `--prefill 16384`, int8 KV with 32K of it in VRAM, and three `STRATA_*` switches. The list
   is printed when setup writes it and kept in [docs/B70.md](docs/B70.md#the-served-setup). `--no-served-recipe` leaves it
   out. Another model: `--model IQ2_XS` (the earlier default), or any size of `--family`.

   Useful with it: `--gguf-dir <folder>` when the model files are already on the PC (setup checks them, downloads nothing
   it finds there and leaves a small `.done` mark beside each file), `--data-dir <folder>` for where the pack and the draft
   layer go (default: next to the repository), `--yes` to take the defaults without questions, `--no-start` to stop after
   the config and the run script are written. The model and the repository must be under the same folder the container
   mounts (the repository's parent); a model elsewhere needs its folder in `STRATA_SYCL_MOUNTS`.
   On a 64 GB PC setup leaves the n-gram table on the SSD; with 85 GB of RAM or more it keeps it in RAM (`--ple-io ram`).
   A fresh clone took 20 minutes here from `git clone` to a config that starts, with the model files already on disk.

Details, every engine switch and what was measured: [docs/INTEL.md](docs/INTEL.md).

## Using it

The server speaks the OpenAI, Anthropic and OpenAI Responses APIs on the port setup prints:

| Your app speaks | Base URL |
| --- | --- |
| OpenAI API (most apps and agents) | `http://127.0.0.1:PORT/v1` |
| Anthropic API (Claude Code) | `http://127.0.0.1:PORT` (no `/v1`) |
| OpenAI Responses API (Codex CLI) | `http://127.0.0.1:PORT/v1/responses` |

### For a coding agent

For one long agent session (a system prompt of 40K-60K tokens, the context up to the model's 262,144 tokens):

    ./setup.sh --backend sycl --family unsloth --model UD-IQ1_S --context 262144 --vram-reserve-mib 3072 --vision no

- **3 GB of VRAM for the desktop** was enough on the test box (KDE and a browser). Everything else goes to the expert
  cache: about 16,000 experts in VRAM instead of 13,000 with 7 GB kept free, so less is read over PCIe.
- **Drafts for code.** With a context above 64K setup drafts from an English + code token subset (`--draft-vocab code`)
  over the last 8,192 cells (`--mtp-window 8192`): ~6% faster answers for a coding agent, the same tokens. For UD-IQ1_S
  the config also has `--expert-profile data/expert-profile-pi.bin`, which keeps the experts pi sessions use in VRAM (made
  from pi sessions on this repo, on the IQ2_XS pack; not measured on UD-IQ1_S yet). For another model, add it to the
  config's args by hand.
- **Thinking** is on at the model's highest level (`xhigh`) unless the client asks for another (`reasoning_effort`:
  `low`, `medium`, `high` or `xhigh`, `none` turns it off).
- **Earlier turns' thinking stays in the prompt** when the client sends it back (`reasoning_content`, as pi does). The
  history then matches what the engine already read, and each turn reads only its new tokens.
- **Pictures** stay off with `--vision no`. Setup's default reads them on the CPU, from RAM, and takes no VRAM. A
  four-request test on 2026-11-11 answered every picture right (2.6 s for the first, 0.5 s for the same one again);
  what it needs and what was not run is in [docs/B70.md](docs/B70.md#pictures-on-one-b70-by-hand).

For [pi](https://github.com/earendil-works/pi), add the server to `~/.pi/agent/models.json`:

```json
{"providers": {"strata": {"baseUrl": "http://127.0.0.1:8080/v1", "api": "openai-completions", "apiKey": "local",
  "models": [{"id": "qwen38-flash-next", "contextWindow": 262144, "maxTokens": 32768, "reasoning": true,
              "compat": {"supportsReasoningEffort": true}}]}}}
```

and start it with `pi --provider strata --model qwen38-flash-next --thinking xhigh`. What one session measured is
under [Speed](#speed); sessions past 118K tokens are not measured yet.

## Things that matter on this card

- **Keep VRAM headroom.** Near 32 GB the `xe` driver can wedge the card, and only a reboot or driver rebind brings it
  back. Leave a few GB free (`--vram-reserve-mib`).
- **Pictures run on the CPU by default** (setup's choice for this card). See the note under
  [For a coding agent](#for-a-coding-agent) before turning them on.
- **Keep RAM headroom on a desktop.** On a 64 GB machine leave `--ple-io` at its default (`direct`); `ram` locks a
  28.8 GB table into memory. Size `STRATA_MIRROR_MIB` so a few GB stay free: 25000 for UD-IQ1_S through the 250K runs above
  left 11 GB free on this 64 GB desktop, with a browser and an Android emulator open (15.9 GB for the earlier IQ2_XS files).
- **B70 defaults are on.** The engine sets its B70 settings (`STRATA_VERIFY_DEVICE_PLAN`, `STRATA_SELECT_LONG`,
  `STRATA_HC_Q8` and the others listed in [docs/B70.md](docs/B70.md)) unless you set them yourself; `0` turns one off,
  `STRATA_B70_DEFAULTS=0` turns all of them off. The B70 prompt kernels (attention, selection, GDN, hyper-connections,
  experts) are on by default on this card, each with its own switch to turn it off, also in [docs/B70.md](docs/B70.md).

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

- The engine and server are [Strata](https://github.com/Niko1221/Strata) by Niko1221 and its contributors. Several B70
  changes here are community pull requests to upstream, credited in [docs/B70.md](docs/B70.md).
- The model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team. The quantized sets
  are by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab), who also made the DeepSeek V4.1-Flash sets.
- Strata uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). The optional 27B engine's long-prompt attention is
  built on Intel's [sycl-tla](https://github.com/intel/sycl-tla) and [vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels);
  licences and what was changed: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

MIT License ([LICENSE](LICENSE)). The models have their own licenses.
