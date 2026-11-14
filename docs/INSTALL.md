# Install Strata

This page is for anyone installing or updating Strata (version 0.2.5). Use the **desktop app** for a graphical
interface, or the **Terminal** for `strata launch` and coding tools. Both use released engine packages. A normal
install does not need a source checkout or a startup `.bat` file.

## Quick start

| I want | Do this |
| --- | --- |
| A window with chat | This fork ships no desktop app. Use the [Terminal route](#the-terminal), or the README's Docker build for a B70 |
| The terminal, or my coding tool | Run the one-line install, then `strata launch` ([steps](#the-terminal)) |
| DeepSeek V4.1-Flash | Pick it in the app, or run `strata launch --family deepseek41` ([model page](MODELS.md#deepseek-v41-flash)) |

[Requirements](#what-you-need) · [Desktop app](#the-app) · [Terminal](#the-terminal) ·
[Update](#updating) · [Files and folders](#where-things-are-stored) · [Build from source](SOURCE_INSTALL.md)

## What you need

| Computer | Guide |
| --- | --- |
| Windows 10/11 or Linux x86-64 with an NVIDIA RTX GPU | Upstream app/Terminal only (removed in this fork); older NVIDIA cards used upstream's CUDA 12 guide. DeepSeek V4.1-Flash runs on RTX 30, 40 and 50 cards upstream. |
| Windows or Linux x86-64 with a supported AMD Radeon | Upstream AMD/HIP guide (removed in this fork). DeepSeek V4.1-Flash runs on the Radeon AI PRO R9700 under Linux upstream. |
| Linux x86-64 with Intel Arc | [Intel Arc guide](INTEL_ARC.md), [SYCL build notes](INTEL.md), and [B70 branch notes](B70.md); this fork targets one Arc Pro B70 on Linux |
| Apple Silicon Mac, macOS 15 or newer | Upstream Mac/Metal guide (removed in this fork). Intel Macs were not supported. |
| AMD Ryzen AI Max / Strix Halo | Upstream Strix Halo Linux guide (removed in this fork; see [Windows notes](STRIX_HALO_WINDOWS.md) for upstream history) |

Memory and disk needs depend on the **model, quantization, context and enabled features**. Check
[which model fits](MODELS.md#pick-by-memory); download size is not the same as RAM use. Unified memory counts once.
Use an SSD and leave room for helper files and model preparation. The Terminal's `--plan` prints what is missing.

Install a graphics driver that supports the package you choose. CUDA 13 packages require NVIDIA driver 580 or newer;
older NVIDIA cards use CUDA 12. AMD and Intel requirements are in their guides. Packaged GPU runtimes do not replace
hardware drivers. Models and updates need an internet connection to download.

## The app

**This fork has no desktop app** - the fork deleted it. Everything in this section is upstream's, for other cards.
On one B70, install through the Docker build in the [README](../README.md) or the source route below.

1. Open the [latest release](https://github.com/Niko1221/Strata/releases/latest).
2. Download the one installer for your system. There is one per system, not one per graphics card:
   - **Windows:** run `Strata-0.2.8-windows-x64-setup.exe`. It installs for your user and needs no administrator rights.
   - **Linux:** make `Strata-0.2.8-linux-x86_64.AppImage` executable, then open it. If FUSE is unavailable, run the
     downloaded file with `--appimage-extract-and-run`. [Linux help](TROUBLESHOOTING.md).
   - **Mac:** open `Strata-0.2.8-macos-arm64.dmg`, drag Strata into Applications, then open it (upstream Mac guide, removed in this fork).
3. On Windows and Linux the installer has no engine inside. On first start the app shows the engine for your
   graphics card with its size and one **Download** button, with progress and resume. If you already downloaded the
   engine file, choose **I already have the file** and it installs without internet. The Mac file keeps its small
   Metal engine inside.
4. Choose a model and its storage folder, let it download, then chat. DeepSeek V4.1-Flash is in the model list on
   the supported cards ([model page](MODELS.md#deepseek-v41-flash)); it has three sizes, an 8K context and a **Speed**
   choice, Faster or Full quality.

The `strata-engine-…` and `strata-control-…` archives on the release page are components, not installers. The
installer is the same file for a first install and for an update. There is no Intel desktop installer yet; Intel Arc
runs on Linux only, through the [Terminal](INTEL_ARC.md).

The first launch can take time to check and prepare large model files. Watch its progress and logs; there is no fixed
startup time for every model or computer. Closing the desktop app stops its managed engine.

The provider order recommended here is **ISTA-DASLab, Unsloth, Gyro, then others**, when a compatible pack fits.
Current app recommendations can differ; choose the model explicitly when you want a particular provider.

## The Terminal

Install once, then use `strata launch`. The Terminal includes Python and downloads the engine for your computer.

**These are upstream's one-line installers.** This fork publishes no `install.sh` or `install.ps1` - its releases
carry a built engine binary for one machine. On a B70 use the Docker build in the [README](../README.md), or
[build from git](#build-from-git-on-linux-to-try-a-pull-request-or-change-the-code) below.

**Linux / Mac**

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

On Windows, open a **new terminal** afterwards. On Linux and Mac, if `strata` is not found, use the full path printed
by the installer (normally `~/.local/bin/strata`) or add that folder to your shell's PATH.

```sh
strata launch
```

Follow the screens to choose a model, context, storage folder and access settings. The monitor opens after startup.
Use the address printed by Strata; the default local endpoint is `http://127.0.0.1:8080`.
The Terminal opens a browser only when you add `--open`.

For a script or AI agent, inspect a plan before downloading. This example selects ISTA-DASLab Q2_0 explicitly:

```sh
strata launch --family qwen --model q2_0 --plan
strata launch --family qwen --model q2_0 --yes --no-tui
```

`--no-tui` leaves the server running in the background. Check it, connect a tool, or stop it:

### Build from git on Linux (to try a pull request or change the code)

You do not need the app or the AppImage for this: the source install builds the engine on your PC and keeps
everything in the checkout, so you can edit, rebuild and test.

```bash
git clone -b b70 https://github.com/riceharvest/strata-b70-only && cd strata-b70-only
./setup.sh              # first run: .venv, the engine build (build/, or build-hip/ on a Radeon), the model, then it serves
```

On one B70 the engine is built inside the oneAPI container the [README](../README.md) describes (the SYCL route),
not by a plain `./setup.sh` host build; `./setup.sh --backend sycl` then fetches the model and writes the run
config. Every flag and what was measured: [docs/INTEL.md](INTEL.md).

- **What it needs:** `git`, a C++ compiler and `cmake` (setup installs the build tools through `sudo apt` on
  Debian and Ubuntu; elsewhere install them first), and the CUDA Toolkit for NVIDIA or ROCm with the HIP compiler for
  a Radeon card (upstream AMD/HIP guide, removed in this fork). Python goes into `.venv` of the checkout.
- **After you change the code:** run `./setup.sh` again. It compiles only what changed and starts the model. To
  update without starting: `git pull`, then `./update.sh`. By hand, the build folder is an ordinary CMake build:
  `cmake --build build --target strata -j8` (`build-hip` for a Radeon card upstream; AMD/HIP guide removed in this
  fork).
- **To try a pull request:** `git fetch origin pull/<number>/head:pr-<number> && git checkout pr-<number>`, then
  `./setup.sh`. `git checkout main` goes back; the next `./setup.sh` rebuilds.
- **Settings for the engine** (for example `STRATA_HIPBLASLT_TUNING=/path/to/table.txt`) go into the `"env"` object of
  `strata-<model>.json`, which `./setup.sh` writes next to the checkout. `./run-<model>.sh` starts the model with it.
- **The Terminal monitor from a checkout:** the monitor needs one extra package, which the source install does not
  put into `.venv`:

  ```bash
  .venv/bin/pip install -r requirements-monitor.txt
  STRATA_PYTHON="$PWD/.venv/bin/python3" bin/strata monitor --url http://127.0.0.1:8080
  ```

  (`bin/strata` is the same `strata` command as the Terminal's: `bin/strata --help` lists the rest.)
- **Only compile, without the installer:** the engine is a normal CMake project. For NVIDIA:
  `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native &&
  cmake --build build --target strata -j8`. The options, and the flags setup passes, are in `setup.py` (search for
  `-DSTRATA_`).

## From source: Mac

On an Apple Silicon Mac with macOS 15 or newer, install Apple's Command Line Tools once (`xcode-select --install`),
then run `./setup.sh` in the Strata folder (upstream Metal path; removed in this fork together with `metal/` and the
Mac guide).

## AMD cards

The app and the Terminal pick the AMD engine for a Radeon card by themselves. From source the steps are the same as with NVIDIA: `START-HERE.bat` on Windows, `./setup.sh` on Linux. Setup finds the Radeon
card and chooses the AMD (HIP) engine by itself on a PC with no NVIDIA card Strata can use; `--backend hip` chooses
it on a PC that has both. Supported cards: RX 7900 XT / XTX (gfx1100), RX 7800 XT / 7700 XT (gfx1101), RX 9060 XT
(gfx1200), RX 9070 / 9070 XT and Radeon AI PRO R9700 (gfx1201), and the RX 6800 / 6900 series (gfx1030). Integrated
Radeon GPUs are listed as not supported.

On Linux setup uses a system ROCm 7 when there is one, or installs ROCm into `.venv` from AMD's wheels (~10 GB, no
sudo), and compiles the engine on your PC for the card (10-20 minutes, once; it needs a C++ compiler and git:
`sudo apt install build-essential git`). Several AMD cards share the model with `--gpus`, as on NVIDIA.

What differs from NVIDIA for now: `--calibrate` is NVIDIA-only, and Unsloth's 4-bit model needs an NVIDIA card. (The
upstream AMD/HIP guide is not in this fork.)

## Two or three cards

**Two or three NVIDIA cards?** Just run `START-HERE.bat`: it lists your cards, says which ones Strata can use, and
asks whether to share the model across them (recommended when two can). An install made on one card asks once at
its next start. Or choose yourself: `START-HERE.bat --gpus 0,2` (both, remembered), `--gpus all`, or `--gpu 0` (one
card, this start only). Each card keeps the experts of its own layers, and prompts flow through the cards in a
pipeline: on an RTX 5080 + RTX 3090 prompts were read 18-20% faster than on the 5080 alone, decoding on par.
Every card must be an RTX 20 series or newer with 8 GB or more (upstream multi-GPU guide, removed in this fork).

## Docker (Linux)

There is no ready-made image. Build the engine from source: `docker build --pull -t strata-sycl-dev sycl/tools`, then
`sycl/tools/build.sh` in that image ([B70.md](B70.md)).

## Older CPUs (experimental)

Strata's ready-made engine needs AVX2 (Intel Haswell 2013, AMD Zen 2017 or newer). Since 0.1.39 an older CPU - AVX
only (Sandy Bridge / Ivy Bridge, Xeon E5 v1/v2, AMD Bulldozer) or SSE4.2 only (Nehalem / Westmere, Xeon X5600) - runs
as an **experimental** build that setup compiles on that PC. Run setup as usual: it says the CPU has no AVX2, warns
that this is experimental and slow, and compiles the engine (10-20 minutes, once; Windows needs the Visual Studio Build
Tools and the CUDA Toolkit, which setup offers to install). It does not stop. The ready-made engine still refuses such
a CPU, and says so.

What differs in that build (`STRATA_ISA_FLOOR=avx` or `none`, chosen from the CPU):

- ggml-cpu is compiled for AVX or SSE4.2 instead of AVX2, and the CPU's share of the experts runs on its kernels.
  Strata's own AVX2 / AVX-512 kernels stay in the program but are used only on a CPU that has them.
- Only the i-quant models' packs run (every model setup installs on such a CPU); the AVX-512 Q2_0 pack does not.
- Speed: the GPU part is unchanged and the CPU part is slower, so a GPU with more VRAM (more experts cached there)
  helps most. With the older paths forced on a Ryzen 5 7600 + RTX 5070 and the expert cache held at 1500 slots, greedy
  decode went from 26 to 11-17 tok/s with the AVX build and to 3.5-3.8 tok/s with the SSE4.2 build (Coder and IQ3_XXS,
  10 prompts each, one run). An old Xeon with DDR3 will be slower than these.
- Untested on a real old CPU by us. It was checked here by forcing the older paths on a Ryzen 5 7600
  (`STRATA_FORCE_ISA=avx` or `sse`): the answers were sane and passed the engine's checks, but not the same token for
  token as the normal build (ggml-cpu rounds the CPU experts differently, as the AVX2 kernel does from the AVX-512
  one). Contributors ran earlier versions of the same approach on a Xeon E5-2680 (2.87 tok/s with a GTX 1080 Ti)
  and on Xeon E5-2687W / X5690 machines.
- AMD cards: Linux only (setup compiles the AMD engine there anyway); NVIDIA on Windows and Linux.
- Two-socket Xeons: memory on the other socket is slow; keep the RAM on one CPU if you can.

By hand: `cmake ... -DSTRATA_ISA_FLOOR=avx` (or `none`) builds it; `STRATA_ISA_FLOOR=avx` set for setup builds it on
any PC (to try it). Problems and results on real hardware are welcome as GitHub issues.

## Updating

**Desktop app:** open Settings and find the update controls, check for updates, and use the offered app or engine update.
Unload running models before replacing the engine. Your model files and settings are kept.

**Terminal:**

```sh
strata update --check
strata update
strata --version
```

Stop models first. The next Terminal command uses the updated Control component. The app, Control and engine have
separate version numbers, so they can differ.

Use `strata components rollback` to return to the previous engine.
Source checkouts have a [separate update workflow](SOURCE_INSTALL.md#updating).

## Where things are stored

| Item | Default location |
| --- | --- |
| Desktop chats and host state | `~/.strata` (or `STRATA_HOME`) |
| Engine components, Windows | `%LOCALAPPDATA%\Strata\components` |
| Engine components, Linux / Mac | `$XDG_DATA_HOME/strata/components`, normally `~/.local/share/strata/components` |
| Terminal files | The `terminal` folder beside `components`; the installer prints the exact path |
| Models | The folder chosen in the app or launch screens; an existing model library is reused |

The desktop app normally proposes `Documents/Strata/Models` for a new library. The Terminal discovers saved
folders from earlier installs. Use the path shown by the app or `strata models list`, not an assumed folder name.
`--data-dir` changes where new downloads go; it does not move existing models unless you add `--move-data`.
See [model discovery](TERMINAL_LAUNCH.md#models-you-already-have).

## More options

- [Build or run from a source checkout](SOURCE_INSTALL.md)
- [B70 branch notes](B70.md)
- [Troubleshooting and logs](TROUBLESHOOTING.md)
- [Model compatibility](MODELS.md)
