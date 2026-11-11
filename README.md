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

Qwen3.8-Flash-Next IQ2_XS on one Arc Pro B70, Ryzen 9 5950X, 64 GB RAM, measured 2026-11-02 (by day). Settings:

- MTP draft, `--spec 4`, int8 KV, 32K of the KV in VRAM, `--prefill 16384`
- 7 GB of VRAM kept free, because the card also drives a desktop; `STRATA_MIRROR_MIB=25000` (24 GiB of experts pinned in RAM)
- page cache dropped before every start
- one 118,196-token prompt (this repo's docs with a fact hidden in the middle; every answer found it), a ~730-token
  answer, temperature 0, two runs each

| Context | Prompt (tok/s) | Answer (tok/s) |
| --- | ---: | ---: |
| 118K | 3,343 / 3,402 | 76.8 / 76.6 |
| 250K | 3,333 / 3,339 / 3,335 | 75.4 / 76.4 / 75.6 |

Short cold prompts (one user message, nothing cached, same settings but 3 GB kept free), prompt read time and time to the
first token:

| Prompt | Read (tok/s) | First token |
| --- | ---: | ---: |
| 135 tokens | 340 ms (397) | 377 ms |
| 393 tokens | 416 ms (944) | 454 ms |
| 1,178 tokens | 678 ms (1,738) | 728 ms |
| 5,382 tokens | 2,008 ms (2,680) | 2,121 ms |

A short prompt cannot reach the long prompts' speed: about 40% of the experts live in RAM, and a few hundred tokens
already route to most of them, so every read has to bring most of those across PCIe whatever its length.

The 250K row is the same test with a 249,618-token prompt (more of the same docs, the fact in the middle, found every
time) and `--max-context 262144`, three runs. On 2026-11-01 the same tests read 3,228 / 3,213 and 3,171-3,174 tok/s with answers at 71.7 / 71.5 and 69.5-70.5 tok/s; the 2026-11-02 work (decode graphs without the shared-expert fork, cheaper drafts, the prompt's expert copies started a layer ahead, smaller tiles for experts of few rows) made the prompts 4-6% and the answers 7% faster here, and short prompts 20-30% faster to their first token ([docs/B70.md](docs/B70.md#short-prompts-2026-11-02)). Before the 2026-11-01 work the same tests read 2,861 / 2,857 tok/s at 118K
and 2,750 / 2,832 at 250K: the prompt is now read 12-15% faster in these runs (14-18% in the night runs), mostly from mirroring the experts a prompt chunk
borrows (the default now; it needs the `STRATA_MIRROR_MIB` above) and, past ~201K tokens, from keeping the prompt's
selection on XMX. The answers are 6-8% faster than that day's first runs (64.6-67.8 tok/s): the experts that are not in
VRAM are copied in by one wide-load kernel and kept in a VRAM ring that the next windows read again (from the cache's own
budget, so it needs the larger `STRATA_MIRROR_MIB` above). The prompt numbers are lower than that night's runs of the
previous build (3,261 / 3,289 and 3,246-3,248 tok/s), but that build rerun the same day read 3,190 / 3,108 (118K) and
3,094 tok/s (250K), and no switch of the new work, turned off, read faster: the machine reads slower by day.
Sampled answers (an agent's) also gain ~6% from exact sampled-draft acceptance, on by default. These numbers include the
fix for the second-turn hang below, which costs about 2%.
[docs/B70.md](docs/B70.md) has what changed, every change against upstream and who wrote it, and the earlier
comparisons with upstream Strata on this card (+12% to +17% prompt speed at 4K and 16K on 2026-10-24, before this work).

**A coding agent on one B70, up to 256K.** The [pi](https://github.com/earendil-works/pi) agent with thinking `xhigh`,
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

2. Get the code and build the engine. The image takes about 6 minutes once; the engine takes a few minutes:

       git clone -b b70 https://github.com/riceharvest/strata-b70-only && cd strata-b70-only
       docker build --pull -t strata-sycl-dev sycl/tools
       H=$(basename "$PWD")
       docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e AOT=bmg-g31 -e REPO=/work/$H \
           -e BUILD_DIR=/work/$H/build-sycl-aot strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"

   The build ends with `BUILD EXIT 0` and `errors: 0`.

3. Set up the model and serve it:

       ./setup.sh --backend sycl --family unsloth --model UD-IQ1_S --context 262144 --vision no

   On one B70 with 64 GB of RAM, a first `./setup.sh --backend sycl` with no `--family` or `--model` preselects the same
   model and context in its menu (`--yes` takes them).
   Setup downloads Unsloth's UD-IQ1_S files at a pinned revision, checks them against pinned sizes and SHA-256 sums, and
   takes one tensor (layer 18's gate experts) from the UD-IQ1_M file of the same repository into shard 2, about 216 MB read
   (the file as published cannot be packed: `docs/QUANTS.md` says why; nothing is quantized again). It then builds the pack,
   fetches the draft layer (about 5 GB of tensors from the Qwen checkpoint, read by range requests), writes a config
   (`strata-unsloth-ud-iq1_s.json`, its `exe` is `sycl/serve/strata-sycl.sh`) and a `run-unsloth-ud-iq1_s.sh` that starts
   the server, and starts it. The config carries the settings this fork serves with: `--spec 6`, the coding agent's expert
   profile and draft vocabulary, `--prefill 16384`, int8 KV with 32K of it in VRAM, and three `STRATA_*` switches. The list
   is printed when setup writes it and kept in [docs/B70.md](docs/B70.md#the-served-setup). `--no-served-recipe` leaves it
   out. Another model: `--model IQ2_XS` (the earlier default), or any size of `--family`.

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
  28.8 GB table into memory. Size `STRATA_MIRROR_MIB` so a few GB stay free: 25000 for Flash-Next IQ2_XS at 256K left
  15.9 GB free on 64 GB (not measured for UD-IQ1_S).
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
