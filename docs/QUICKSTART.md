# Quickstart: fresh B70 Linux box to a serving model

One page from a clean Linux install to a running Qwen3.8-Flash-Next server on one **Intel Arc Pro B70** (`bmg-g31`, 32 GB).

## 1. What you need

- One **Intel Arc Pro B70** (32 GB) on Linux with the `xe` kernel driver and the Intel compute runtime (`ls /dev/dri` shows `renderD128`, [README](../README.md)).
- Docker (the engine builds and runs in a container, so the host needs no oneAPI).
- Disk: about 75 GB free for Unsloth `UD-IQ1_S` (`66.4 GB` of model shards and `~5 GB` of draft layer, plus ~13 GB for the `strata-sycl-dev` Docker image).
- RAM (tested with no swap):
  - **64 GB** works with the served recipe (`--ple-io direct`, the default).
  - **85 GB+** lets `--ple-io ram` lock the `28.8 GB` n-gram table in RAM (`84.8 GB` formula in `sycl/setup_intel.py`).

## 2. Docker and group access (once)

```sh
sudo apt install docker.io git        # or your distro's equivalent
sudo usermod -aG docker,render $USER  # then log out and in again
```

## 3. Clone and build

The container image takes about **6 minutes once** (**13 GB**); the engine takes about **11 minutes** on a 16-core desktop and ends with `BUILD EXIT 0` and `errors: 0` ([README](../README.md)):

```sh
git clone -b b70 https://github.com/riceharvest/strata-b70-only && cd strata-b70-only
docker build --pull -t strata-sycl-dev sycl/tools
H=$(basename "$PWD")
docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e AOT=bmg-g31 -e REPO=/work/$H \
    -e BUILD_DIR=/work/$H/build-sycl-aot strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"
```

## 4. Download the model and start the server

```sh
./setup.sh --backend sycl --family unsloth --model UD-IQ1_S --context 262144 --vision no
```

`setup.sh` creates `.venv` if needed and runs `setup.py`. With those flags it downloads Unsloth `UD-IQ1_S` at the pinned revision (`084c631`), verifies sizes and SHA-256 sums, grafts layer 18's gate tensor (`blk.18.ffn_gate_exps.weight`) from `UD-IQ1_M`, packs the weights, fetches the ~5 GB draft layer, writes `strata-unsloth-ud-iq1_s.json` and `run-unsloth-ud-iq1_s.sh`, and starts the server. On a fresh clone with the model files already on disk, going from `git clone` to a starting config took about **20 minutes** ([README](../README.md)).

## 5. First-hour gotchas and common flags

- **Models outside the repository's parent folder (`STRATA_SYCL_MOUNTS`):** the Docker runtime mounts the checkout's parent directory (`/work`). A model stored elsewhere (or reached through a symlink pointing outside that directory) is invisible in the container or fails with `duplicate split shard number`. Keep models under the checkout's parent directory or add their directory path in `STRATA_SYCL_MOUNTS` (`sycl/serve/strata-sycl.sh`, [README](../README.md)).
- **Port `8080` already in use:** setup defaults to `--port 8080`; pass another port such as `--port 8081` or `--port 8085` ([INTEL.md](INTEL.md), [TROUBLESHOOTING.md](TROUBLESHOOTING.md)).

Useful `./setup.sh` options ([setup.py](../setup.py), [README](../README.md)):

- `--gguf-dir DIR` — use GGUF shard files you already have on disk instead of re-downloading (verifies them and writes a `.done` mark).
- `--data-dir DIR` — directory where prepared packs and the draft layer are stored (default: `Strata-data` next to the checkout).
- `--yes` — accept the recommended defaults without interactive prompts.
- `--no-start` — write the config and `run-unsloth-ud-iq1_s.sh`, then exit without starting the server.

## 6. Check it and stop it

Check the running server on port `8080` (or the `--port` you passed):

```sh
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "strata", "messages": [{"role": "user", "content": "Write a haiku about GPUs."}], "max_tokens": 512}'
```

Inspect a background service managed by Strata (`STRATA_PYTHON="$PWD/.venv/bin/python3" bin/strata ...` if `strata` is not on `PATH`, [SOURCE_INSTALL.md](SOURCE_INSTALL.md)):

```sh
strata runtime status --profile NAME
```

Closing rules ([CLOSING.md](CLOSING.md), `strata runtime --help`): what started the server owns it, and when that owner exits the server drains requests and stops the engine.

- **Foreground** (`./setup.sh`, `strata serve --profile NAME`, `python serve/server.py`): press `Ctrl+C` or close the terminal.
- **Background service** (`strata runtime start --profile NAME`): stop it with:

```sh
strata runtime stop --profile NAME
```

- **Leftover processes** after an owner crash (running instances are never touched):

```sh
strata runtime cleanup
```

## 7. Where to go next

- [The B70 served setup](B70.md#the-served-setup) — exact flags in the `UD-IQ1_S` served config and how to inspect or opt out (`--no-served-recipe`)
- [INTEL.md](INTEL.md) — SYCL engine switches, driver prerequisites, and context limits
- [MODELS.md](MODELS.md) and [B70_MODELS.md](B70_MODELS.md) — upstream model options and exact B70 VRAM/RAM fit arithmetic (including `Qwen3.8-27B`)
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — install, port, container, and runtime errors
