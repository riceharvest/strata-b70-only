# Community benchmark results

Share measurements from your own hardware so other users can judge how Strata
might run on a similar machine. This guide suggests a report format and a pull
request workflow; a single model, GPU, or context length is enough to contribute.
There are two ways to measure. For results other people can compare with, run the **standard benchmark**: `strata bench run`
follows the draft protocol in [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) (fixed prompts, thinking on, the model card's
sampling, twelve sanity checks, about 2 to 4 minutes after the model has loaded on a fast card, more on a slow one) and writes `results.json` with
the hardware, versions and every request. For a quick look at your own machine, [`BENCH.bat` / `./bench.sh`](#run-the-benchmark)
takes 3 to 10 minutes and writes a report in the format below.

The protocol is still a draft until a pilot has run on two kinds of hardware. The reports listed below were made before
it existed and are not standard results; they stay valid under this guide.

Existing [benchmark reports](../bench/results/) provide examples. The
[2x Arc Pro B60 report](../bench/results/2026-10-04-community-2x-arc-pro-b60/README.md)
includes hardware, settings, and per-run JSON. The
[technical details](DETAILS.md#speed-measured) explain the published measurements
and their limits. Report what you actually measured and label estimates separately.

## Community reports

- [2026-10-04: 2x Intel Arc Pro B60 24 GB, Ryzen 5 5600, 64 GB RAM](../bench/results/2026-10-04-community-2x-arc-pro-b60/README.md):
  the SYCL port at `6f32ec0` with two fixes, Coder IQ1_M and Flash-Next IQ2_XS, 8,192-token context, layer split across both cards;
  short prompts and 2K-token prompts, plus one run on a single B60. Intel's SYCL engine, no 4,096/32,768/128,000-token sweep.

## Run the standard benchmark

```sh
strata bench run --profile NAME --track default --track-notes "release 0.2.1, setup defaults"
strata bench run --url http://127.0.0.1:8080          # a server that is already running
```

With `--profile` the command starts that profile's engine, waits for the model, runs the protocol and stops the engine. With `--url`
it uses the server that is running. Start from a freshly loaded engine and close other programs that use the GPU: the
protocol counts a second run against the same engine as reuse of the cache and marks it `nonconforming`. The command prints a
summary when it ends and writes a folder `<date>-standard-<gpu>-<run id>` with `README.md`, `results.json`, `outputs.jsonl`,
the settings, the chat template and, when it started the engine, the engine log. `strata bench report FOLDER` shows it again.

A run is `complete`, `partial` (something was skipped or cancelled) or `nonconforming` (for example a remote server, a missing
commit or model hash, or reused prompt text). A complete run can still fail sanity checks; low scores and slow hardware are valid
results. The runner does not know the quantization repository's revision, the original weights' revision or what else ran on the
machine: pass `--model-revision`, `--source-revision` and `--background`, or the result says `null` with a reason. To share
a result, follow [Submit and review](BENCHMARK_PROTOCOL.md#submit-and-review) in the protocol, not the pull request steps below.

## Run the benchmark

The simplest report is one command after setup. Windows: double-click `BENCH.bat`. Linux: `./bench.sh`. Or
`python tools/strata_bench.py` from the Strata folder. Setup also offers it at the end of an interactive install
(default no). The Terminal's `strata bench run` runs the standard benchmark (next section); its `--suite serving-smoke` and
other serving suites ([TERMINAL.md](TERMINAL.md)) write JSON evidence for your own comparisons, not a report in this format.

It starts the model with the settings you already have (the newest `strata-*.json`; `--config FILE` picks another one),
or uses a Strata server that is already running on that port (`--url http://127.0.0.1:8080` for another address). It then
sends a fixed suite through the HTTP API, three runs each, and reports medians with ranges. The default takes about
3 to 10 minutes once the model has loaded:

| Test | What it sends | What it reports |
| --- | --- | --- |
| chat | three short questions, up to 100 new tokens | decode tok/s, time to first token |
| 4K and 16K prompt | a fresh word-list prompt of about 4,096 and 16,384 tokens (skipped when your context is smaller) | prompt tok/s, time to first token |
| agent session | a system prompt and four tools, then five turns that each add about 1,500 words of a project file and continue from the model's own earlier answers | tokens the conversation cache reused, tokens read again, time to first token per turn |
| `--clients N` | N requests at the same moment (the server needs `--batch` to run them together) | total tok/s, p50 and p95 first-token time and latency |
| `--long` | the prompt test at about 32,768 tokens | prompt tok/s, time to first token |

Speeds are the engine's own timings (`prompt_per_second`, `predicted_per_second` in the answer); first-token time and
latency are measured by the client. Thinking is off for the suite so the answers are short and the runs comparable
(`--think` leaves it on). Every prompt carries a unique tag, so the fresh-prompt tests never get help from the cache.
Close other programs first: a browser or a game changes the numbers.

The result goes to `bench/results/<date>-community-<gpu>-<model>/` (or `--out DIR`):

- `README.md` in the format below, with the GPU, VRAM, CPU, RAM, OS, driver, Strata and engine versions, model and
  quantization, context and your settings filled in from the machine and the server
- `results.json` with every request (tokens, timings, errors) and the machine facts
- `engine-log-excerpt.txt` with the engine's start-up lines (when the benchmark started the server)

Home folders, user and host names, local network addresses and API keys are replaced by placeholders before anything is
written. Read the report once before you share it: the benchmark cannot know what else you consider private.

To share it, open a pull request that adds the folder (see [Submit a report](#submit-a-report)), or paste
`README.md` into a GitHub issue. Add what the tool does not know: background load, power limits, storage.

To compare with an earlier run on the same machine (before and after an update or a setting change):

```bash
python tools/strata_bench.py --compare bench/results/2026-10-04-community-2x-arc-pro-b60
python tools/strata_bench.py --compare OLD_FOLDER --compare-only NEW_FOLDER   # two saved runs, nothing is started
```

The tool is a speed check, not a quality test. For recall and other checks, see [Optional correctness
checks](#optional-correctness-checks); for A/B tests of two builds use `tools/ab_engine.py`.

## What to record

Include enough information for someone else to repeat your run:

- **Hardware:** GPU model, VRAM, selected GPUs for a multi-GPU run, CPU, installed
  RAM, and storage type. Include PCIe link speed and width, GPU power limits, and
  other workloads when known or relevant.
- **Software:** OS, Strata commit and engine version, `xe` kernel driver and oneAPI
  version, and source build or container image. Include changed build options.
- **Model:** exact repository and revision when available, quantization, GGUF
  filenames, and vision encoder if enabled. Identify custom packs, expert
  profiles, or draft vocabularies, with hashes or reproducible preparation steps.
- **Settings:** launch command and relevant run configuration, context limit, KV
  type and streaming window, expert cache, low-RAM mode, prefill size, CPU workers,
  MTP, reasoning effort, sampling, and vision. State whether calibration or
  experimental speed projection was enabled; attach the resulting settings.
- **Workload:** benchmark command or script, shareable prompts or instructions to
  build them, actual prompt and generated token counts, output cap, repetitions,
  and cache state for each run.

Use GB or GiB consistently and name the unit. Mark unavailable values explicitly
instead of guessing. Remove credentials and private content from configurations,
prompts, and logs before publishing them.

## How to measure

Start with a configuration that already works on your machine. You do not need to
download every model or test the largest context. A useful initial speed report
tests one short prompt and one longer prompt that fits, with a fixed output cap
(for example, 256 tokens). Record the actual output length if the model ends early.

For each configuration, try at least three measured runs and publish all of them
with a median and range. If you only have one run, say so. Explain any warm-up and
whether model loading is included in the timing.

Strata reuses conversation prefixes and adapts its expert cache. Separate prompts
that are fully processed from follow-ups that reuse tokens, and report the number
of reused and freshly read tokens from the engine log. Describe how you reset or
retained state between runs. A warmed expert cache and a reused prompt prefix are
different conditions; record both when possible.

Keep these measurements separate:

- **Prompt throughput:** freshly processed prompt tokens per second, from the
  engine's timing output. Keep the corresponding token count and duration.
- **Decode throughput:** generated tokens per second, from the engine's timing
  output. Count reasoning tokens too, or explain any different convention.
- **Time to first token:** seconds from sending the request to its first generated
  token. For a streaming API measurement, ignore keep-alives and empty deltas and
  state whether the first token is reasoning or answer text.
- **Total latency:** elapsed seconds for the whole request. State whether measured
  at the engine or client, and whether loading, vision encoding, or queueing is included.
- **Memory:** observed RAM and VRAM usage, including whether the value is a startup
  snapshot or a peak during inference. Report paging or out-of-memory failures.

Attach the engine timing lines and your measurement script if you used one. Avoid
calculating decode throughput by dividing generated tokens by total request time,
which also includes prompt processing. Cancelled or failed requests belong in the
report as failures, outside the successful throughput summary.

When comparing versions or settings, run both on the same machine and workload
and list every changed setting. Preserve the per-run results: output text and
draft acceptance can change speed even when the input is identical.

## Optional correctness checks

Speed measurements alone do not establish answer quality. Include a small check
relevant to your workload, such as a completed coding task with test results, a
tool call followed by its result, or an image question with an expected answer.

For long-context recall, the repository includes
[`tools/needle_bench.py`](../tools/needle_bench.py). With a running local server,
from the repository root:

```bash
python tools/needle_bench.py --help
python tools/needle_bench.py --url http://127.0.0.1:8080 --lengths 32k,128k --depths 10,50,90 --out needles.json
```

Choose lengths your configured context can hold. The script supports `--api-key`
for an authenticated server; keep that key out of published commands. Save the
reported actual prompt lengths, misses, errors, and skipped cases as well as
successes. A needle test measures recall on those inputs, not overall model quality.

## Submit a report

1. Fork Strata and create a branch for your report.
2. Add a folder such as `bench/results/YYYY-MM-DD-community-arc-pro-b70/`. Use the
   measurement date and a short hardware label; add a suffix if the name exists.
3. Put a `README.md` based on the template below in that folder, together with
   per-run JSON or CSV and small scripts or prompts needed to repeat it. Use your
   own data format and document its fields and units. Keep large model files and
   generated packs out of the PR.
4. Open a pull request against `riceharvest/strata-b70-only:b70`. B70 numbers
   belong to this fork: upstream `Niko1221/Strata` has no B70, cannot run the
   SYCL engine and cannot take a report it cannot reproduce. Summarize the
   hardware, model, configurations tested, and limitations. Keep a results-only
   submission separate from engine changes so reviewers can assess the
   measurements directly.

Maintainers can review the report and decide where to include it. Adding a report
does not require changing the README's headline performance claims. For a failed
run or a suspected bug, a GitHub issue with the configuration and relevant logs
may be more useful than a benchmark PR.

## Report template

Copy this into your report's `README.md`, replace the placeholders, and remove
sections that do not apply. Use `not measured` for missing measurements; the table
contains no example performance numbers.

````markdown
# Community benchmark on GPU name

Measured on DATE by HANDLE. State what was tested and the main limitation.

## Hardware and software

- GPU and VRAM; CPU; installed RAM; storage; PCIe link if known:
- OS; kernel driver (`xe`); oneAPI version:
- Strata commit; engine version; release binary or source build:
- Background workloads and any power limits:

## Model and configuration

- Model repository and revision; quantization; GGUF filenames:
- Vision encoder; custom packs or profiles:
- Context; KV type and streaming; cache; prefill; low-RAM mode:
- MTP; reasoning; sampling; calibration; experimental speed projection:

```text
Exact launch command and relevant configuration, with credentials removed.
```

## Method

Link prompts and scripts. Describe output cap, repetitions, warm-up, loading,
prompt reuse, expert-cache state, timing boundaries, and memory measurement.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Fill in measured values | | | | | | | |

Link per-run data and logs. Include total latency and memory observations when
measured, with units and timing boundaries. List failures and skipped cases.

## Correctness and limitations

Describe checks, expected answers, observed results, and untested behavior.
````
