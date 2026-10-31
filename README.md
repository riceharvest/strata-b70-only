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
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [ISTA-DASLab GSQ-RCO GGUFs](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | Usable at 128K-256K context, chats and coding agents included (measured at 118K and 250K). Numbers below |
| DeepSeek V4.1-Flash, [ISTA-DASLab sets](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF), **pruned** (`--prune-experts`) | Runs, not recommended, no further work planned: the 2-bit set plus pruning loses too much of what the model knows (in our tests it got simple facts wrong) |
| DeepSeek V4.1-Flash, unpruned | Not recommended. On one card with 64 GB of RAM most experts come off the disk on every token: about 1 tok/s from a SATA SSD, about 5 tok/s from NVMe |

## What counts as usable here

A model and setting is called usable when it does all of this on one B70:

- 128K to 256K tokens of context
- at least 1,000 tokens/s reading the prompt
- at least 10 tokens/s writing the answer

Anything that misses one of these is labelled "works, not usable".

## Speed

Qwen3.8-Flash-Next IQ2_XS on one Arc Pro B70, Ryzen 9 5950X, 64 GB RAM, measured 2026-10-31. Settings:

- MTP draft, `--spec 4`, int8 KV, 32K of the KV in VRAM, `--prefill 16384`
- 7 GB of VRAM kept free, because the card also drives a desktop
- page cache dropped before every start
- one 118,196-token prompt (this repo's docs with a fact hidden in the middle; every answer found it), a ~730-token
  answer, temperature 0, two runs each

| Context | Prompt (tok/s) | Answer (tok/s) |
| --- | ---: | ---: |
| 118K | 2,861 / 2,857 | 65.4 / 66.0 |
| 250K | 2,750 / 2,832 | 64.6 / 62.4 |

The 250K row is the same test with a 249,618-token prompt (more of the same docs, the fact in the middle, found) and
`--max-context 262144`: one run before and one after the fix for the prompt selection past ~201K tokens.

The same test on this branch before the latest prompt work read 1,786 / 1,817 tok/s, so the prompt is read 59% faster.
The answer speed did not change. These numbers include the fix for the second-turn hang below, which costs about 2%.
[docs/B70.md](docs/B70.md) has what changed, every change against upstream and who wrote it, and the earlier
comparisons with upstream Strata on this card (+12% to +17% prompt speed at 4K and 16K on 2026-10-24, before this work).

**A coding agent on one B70, up to 256K.** The [pi](https://github.com/earendil-works/pi) agent with thinking `xhigh`,
`--max-context 262144` and 3 GB of VRAM kept for the desktop. A session with a 170K-token system prompt (a 156K-token
`AGENTS.md`) ran eleven coding tasks: 82 requests, the context growing from 170K to 245K tokens, where pi compacted
the conversation, then on from 195K to 225K. Every task finished. Each turn reused the history it had already read and
read only the new tokens (64 of 79 turns under 1.5 s); answers ran at 73-103 tok/s. After the compaction the next
prompt reused the 170K-token system prompt and read only the summary and recent turns (26K tokens, 11 s). An earlier
session from 64K to 118K ran the same way. Turns that continue a conversation used to hang on this card; the engine
now works around it (2% prompt speed), see [docs/B70.md](docs/B70.md#agent-use-and-the-second-turn-hang-2026-10-31).

## What you need

- An Intel Arc Pro B70, Linux with the `xe` kernel driver and the Intel compute runtime (`ls /dev/dri` shows
  `renderD128`)
- Docker (or Podman). The engine builds and runs in a container, so the host needs no oneAPI
- 64 GB of RAM (what it's measured on)
- Disk: about 80 GB for Qwen3.8-Flash-Next IQ2_XS, about 260 GB for a DeepSeek V4.1-Flash set

### The rest of your PC matters too

Every number in this repo comes from one older desktop: a Ryzen 9 5950X (Zen 3, AVX2, no AVX-512) with dual-channel DDR4 on a
PCIe 4.0 x16 slot. Treat them as a floor; on a newer platform expect more, but nobody has measured that here yet.

- **Qwen** keeps the experts that don't fit in VRAM in RAM and reads them over PCIe on every token; that link is the
  biggest cost in decode. The B70 is a PCIe 5.0 card, so a PCIe 5.0 x16 slot can carry up to twice what this box does.
  Faster RAM on its own changes little here: dual-channel DDR4 already outruns a PCIe 4.0 link.
- **DeepSeek V4.1 pruned** runs the experts that aren't in VRAM on the CPU. That is bound by memory bandwidth and by the
  CPU's vector units, so DDR5 and a CPU with AVX-512 (Zen 4/5, recent Xeons) should help it directly.
- Run the card in an x16 slot wired to the CPU. An x8 or chipset slot halves the link or worse
  (`cat /sys/bus/pci/devices/*/current_link_width` and `current_link_speed` show what you got).

## Install

1. Docker, once:

       sudo apt install docker.io git        # or your distro's equivalent
       sudo usermod -aG docker,render $USER  # then log out and in again

2. Get the code and build the engine. The image takes about 6 minutes once; the engine takes a few minutes:

       git clone -b b70 https://github.com/riceharvest/strata-b70-only && cd strata-b70-only
       docker build --pull -t strata-sycl-dev sycl/tools
       H=$(basename "$PWD")
       docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e AOT=bmg-g31 -e REPO=/work/$H \
           -e BUILD_DIR=/work/$H/build-sycl-aot strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"

   The build ends with `BUILD EXIT 0` and `errors: 0`. Build `strata-dsv41` the same way for DeepSeek.

3. Set up a model and serve it:

       ./setup.sh --backend sycl --model IQ2_XS

   Setup downloads the model, builds its pack, writes a config and a `run-<model>.sh` that starts the server.

Details, every engine switch and what was measured: [docs/INTEL.md](docs/INTEL.md). DeepSeek: [docs/DEEPSEEK41.md](docs/DEEPSEEK41.md).

## Using it

The server speaks the OpenAI, Anthropic and OpenAI Responses APIs on the port setup prints:

| Your app speaks | Base URL |
| --- | --- |
| OpenAI API (most apps and agents) | `http://127.0.0.1:PORT/v1` |
| Anthropic API (Claude Code) | `http://127.0.0.1:PORT` (no `/v1`) |
| OpenAI Responses API (Codex CLI) | `http://127.0.0.1:PORT/v1/responses` |

### For a coding agent

For one long agent session (a system prompt of 40K-60K tokens, the context up to the model's 262,144 tokens):

    ./setup.sh --backend sycl --model IQ2_XS --context 262144 --vram-reserve-mib 3072 --vision no

- **3 GB of VRAM for the desktop** was enough on the test box (KDE and a browser). Everything else goes to the expert
  cache: about 16,000 experts in VRAM instead of 13,000 with 7 GB kept free, so less is read over PCIe.
- **Thinking** is on at the model's highest level (`xhigh`) unless the client asks for another (`reasoning_effort`:
  `low`, `medium`, `high` or `xhigh`, `none` turns it off).
- **Earlier turns' thinking stays in the prompt** when the client sends it back (`reasoning_content`, as pi does). The
  history then matches what the engine already read, and each turn reads only its new tokens.
- **Pictures** stay off with `--vision no`. Setup's default reads them on the CPU, from RAM; the first picture sent on a
  B70 once reset the card ([docs/INTEL.md](docs/INTEL.md#pictures-on-an-arc)), and nobody has looked at it since.

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
  28.8 GB table into memory. Size `STRATA_MIRROR_MIB` so a few GB stay free.
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
- Strata uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp).

MIT License ([LICENSE](LICENSE)). The models have their own licenses.
