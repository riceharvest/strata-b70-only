#!/usr/bin/env bash
# Configure + build the SYCL port inside strata-sycl-dev.  sycl/tools/build.sh [target...]
#   No target: builds `strata` only. The kernel parity programs, tests and benches are not part of it:
#   configure with PARITY=ON (a new build folder, or an existing one: the switch is applied to it) and name them, e.g.
#   PARITY=ON sycl/tools/build.sh gdn_parity   (or PARITY=ON sycl/tools/build.sh all  for every one of them)
#   The dense Qwen3.8-27B (GGUF architecture qwen35) is a model family of the same `strata` binary (docs/Q36.md) and is built in:
#   Q36=OFF builds the Flash-Next-only strata. With AOT=bmg-g31 the family's long-prompt attention (LP3, libq36_tla.so) needs the
#   two projects bash sycl/tools/get-q36-tla.sh fetches into third_party/: without them the build goes on without LP3 and says so
#   (Q36_TLA=ON stops instead, Q36_TLA=OFF never builds it).
#   BUILD_DIR=...       build folder (default <repo>/build-sycl)
#   HOST_MARCH=x86-64-v3  -march of the engine's host code (empty: baseline x86-64; native: this PC only). Gain not measured.
#   AOT=bmg-g31         ahead-of-time device code (empty: SPIR-V, JIT at the first start)
#   IQ_DECODE_FAST=OFF  the earlier IQ/Q2 decode helpers (default ON: bit-exact packed arithmetic, PR #4185)
#   I8_ESIMD=OFF        no fused int8 ESIMD expert kernels (stubs; the prompt's experts take the XMX path). AUTO (default) or ON otherwise.
#   ALLOW_BAD_OCLOC=1   build although this ocloc is known to crash on the Battlemage kernels (see below)
set -uo pipefail
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true; set -u

# #3219: ocloc from Compute Runtime 26.22.38646.x (IGC 2.36.3) ends with "IGC: Internal Compiler Error: Floating point exception" on the
# fused int8 expert kernels when it builds for a Battlemage device (bmg-*): 77 device images, then "gen compiler command failed with exit
# code 245" at the last link, after the long compile.  A newer ocloc builds them.  The strata-sycl-dev image keeps whatever ocloc its base image
# had when it was built (the base tag moves; `docker build` reuses a cached base unless it gets --pull).  Say it before the compile, not after.
# Other versions are not judged: the list holds what has been seen to fail.
known_bad_ocloc() {
  case "$1" in
    26.22.38646|26.22.38646.*) return 0 ;;
  esac
  return 1
}
ocloc_version() {
  command -v ocloc >/dev/null 2>&1 || return 1
  ocloc --version 2>/dev/null | grep -Eo '[0-9]+\.[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1
}
if [[ "${AOT:-}" =~ ^(bmg|lnl) ]] && [[ "${I8_ESIMD:-AUTO}" != "OFF" ]] && [[ "${ALLOW_BAD_OCLOC:-}" != "1" ]]; then
  v=$(ocloc_version || true)
  if [[ -n "$v" ]] && known_bad_ocloc "$v"; then
    cat >&2 <<EOF
build.sh: this image's ocloc is $v (IGC 2.36.3). It is known to crash while it compiles the int8 ESIMD kernels for AOT=${AOT} (#3219).
  Fix: rebuild the image with a fresh base, then build again:   docker build --pull -t strata-sycl-dev sycl/tools
  Or build without those kernels (the prompt's experts take the slower XMX path):   I8_ESIMD=OFF
  Or build anyway:   ALLOW_BAD_OCLOC=1
EOF
    exit 2
  fi
fi

# Which device compiler builds the engine decides its speed: measured 3-5% slower answers with IGC 2.42.6 than with 2.41.5 (docs/B70.md).
# One line says what was found; a second one warns when it is not the pinned pair. Never fails the build; silent without dpkg.
PINNED_IGC=2.41.5 PINNED_OCLOC=26.35.39758.10
if command -v dpkg-query >/dev/null 2>&1; then
  igc_v=$(dpkg-query -W -f='${Version}' intel-igc-core-2 2>/dev/null || true)
  ocl_v=$(ocloc_version || true)
  if [[ -n "$igc_v" || -n "$ocl_v" ]]; then
    echo "build.sh: device compiler: ocloc ${ocl_v:-?}, IGC ${igc_v:-?}"
    if [[ "$igc_v" != "$PINNED_IGC"* || "$ocl_v" != "$PINNED_OCLOC"* ]]; then
      echo "build.sh: WARNING: not the pinned compiler (IGC $PINNED_IGC, ocloc $PINNED_OCLOC): answers measured 3-5% slower with IGC 2.42.6; see docs/B70.md. Rebuild the image: docker build -t strata-sycl-dev sycl/tools" >&2
    fi
  fi
fi

repo=${REPO:-$(git rev-parse --show-toplevel)}
b=${BUILD_DIR:-$repo/build-sycl}
# the dense family's long-prompt attention (LP3: an AOT build for Battlemage / Lunar Lake) compiles against two projects fetched into
# third_party/.  Without them: Q36_TLA=ON stops before cmake is touched (so a build folder is not left configured for a build that
# cannot start); otherwise the build goes on without LP3 (CMake says so too) and the 27B's 8-bit-cache prompts are slower.
if [[ ! "${Q36:-ON}" =~ ^(OFF|off|0|FALSE|false|NO|no)$ ]] && [[ "${Q36_TLA:-AUTO}" != OFF ]] && [[ "${AOT:-}" =~ ^(bmg|lnl) ]] &&
   { [ ! -d "$repo/third_party/sycl-tla/include/cutlass" ] || [ ! -f "$repo/third_party/vllm-xpu-kernels/csrc/xpu/attn/xe_2/chunk_prefill.hpp" ]; }; then
  if [[ "${Q36_TLA:-AUTO}" =~ ^(ON|on|1|TRUE|true|YES|yes)$ ]]; then
    echo "build.sh: Q36_TLA=ON with AOT=${AOT} compiles against third_party/sycl-tla and third_party/vllm-xpu-kernels, which are not there." >&2
    echo "  Fetch them once (git and a network, about 40 MB), then build again:   bash sycl/tools/get-q36-tla.sh" >&2
    exit 2
  fi
  echo "build.sh: NOTE: third_party/sycl-tla and third_party/vllm-xpu-kernels are not fetched: strata is built WITHOUT LP3 (the Qwen3.8-27B's long-prompt attention; its 8-bit-cache prompts are slower)." >&2
  echo "  For LP3, fetch them once (git and a network, about 40 MB) and build again:   bash sycl/tools/get-q36-tla.sh" >&2
fi
[ -f $b/build.ninja ] || cmake -S $repo/sycl -B $b -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DSTRATA_SYCL_AOT="${AOT:-}" -DSTRATA_SYCL_SPIN_MAX="${SPIN_MAX:-}" -DSTRATA_SYCL_I8_ESIMD="${I8_ESIMD:-AUTO}" -DSTRATA_SYCL_IQ_DECODE_FAST="${IQ_DECODE_FAST:-ON}" \
    -DSTRATA_HOST_MARCH="${HOST_MARCH:-}" -DSTRATA_SYCL_PARITY="${PARITY:-OFF}" -DSTRATA_BUILD_Q36="${Q36:-ON}" -DSTRATA_Q36_TLA="${Q36_TLA:-AUTO}" 2>&1 | tail -15
# an existing build folder keeps its cache: a switch given now reaches it (only this one, so AOT and the rest stay as configured)
[ -z "${I8_ESIMD:-}" ] || cmake -S $repo/sycl -B $b -DSTRATA_SYCL_I8_ESIMD="$I8_ESIMD" >/dev/null 2>&1
[ -z "${HOST_MARCH+x}" ] || cmake -S $repo/sycl -B $b -DSTRATA_HOST_MARCH="$HOST_MARCH" >/dev/null 2>&1
[ -z "${PARITY:-}" ] || cmake -S $repo/sycl -B $b -DSTRATA_SYCL_PARITY="$PARITY" >/dev/null 2>&1
[ -z "${Q36:-}" ] || cmake -S $repo/sycl -B $b -DSTRATA_BUILD_Q36="$Q36" 2>&1 | grep -E 'CMake Error|needs third_party|get-q36-tla' | head -5
[ -z "${Q36_TLA:-}" ] || cmake -S $repo/sycl -B $b -DSTRATA_Q36_TLA="$Q36_TLA" 2>&1 | grep -E 'CMake Error|needs third_party|get-q36-tla' | head -5
# which families this build holds (a folder configured before the dense family was built in keeps its cache: Q36=ON reaches it)
echo "build.sh: dense Qwen3.8-27B family in strata: $(sed -n 's/^STRATA_BUILD_Q36:BOOL=//p' $b/CMakeCache.txt 2>/dev/null | head -1)"
cmake --build $b -j ${JOBS:-12} --target ${@:-strata} -- -k 0 2>&1 | tee $b/build.log | grep -E '^FAILED|error:|^ninja: build stopped|Linking|^\[[0-9]+/[0-9]+\] Linking' | tail -40
rc=${PIPESTATUS[0]}                       # cmake's status, kept: the script ends with this, not with the status of the greps below (#2990)
echo "BUILD EXIT $rc"
grep -c 'error:' $b/build.log | sed 's/^/errors: /'
grep -E '^FAILED' $b/build.log | sed 's#.*/##' | sort | uniq | head -60
if grep -q 'IGC: Internal Compiler Error' $b/build.log 2>/dev/null; then
  echo "build.sh: the device compiler (ocloc $(ocloc_version || echo '?')) crashed: try  docker build --pull -t strata-sycl-dev sycl/tools  or  I8_ESIMD=OFF  (#3219)"
fi
exit "$rc"
