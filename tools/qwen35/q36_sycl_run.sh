#!/usr/bin/env bash
# q36_sycl_run.sh <strata args...> - one run of this checkout's strata (build-sycl-aot/strata; a --native qwen35 file takes the dense
# Qwen3.8-27B family, docs/Q36.md) in the strata-sycl-dev
# image, for the parity and KL tools (tools/qwen35/parity.py --q36, kl.py) and for a run by hand.  The checkout is mounted at
# /work/<its folder name>, the model's folder and the output folder at their host paths, so the arguments name host paths.
# The peak VRAM the process holds (xe's drm-total-vram0 in /proc/<pid>/fdinfo, read from the host: the container runs as your user)
# and the lowest free RAM are printed at the end.
#   Q36_MODELS      the folder of the GGUF (required)
#   Q36_OUT         a folder for output files (default: <checkout>/q36-out)
#   Q36_BIN         the binary, relative to the checkout (default: build-sycl-aot/strata)
#   Q36_IMAGE       the image (default: strata-sycl-dev)
#   Q36_ENV         "K=V K2=V2": variables set in the container
#   Q36_PREFIX      a command put before the engine in the container (e.g. "unitrace -d", in an image that has it)
#   Q36_STDIN=1     pass stdin on (--serve)
#   Q36_LOCK        a command put before `docker run` (e.g. "flock /var/lock/gpu.lock", when other jobs share the card)
#   Q36_MIN_RAM_GB  end the run when MemAvailable falls below this many GiB (default 0: no watchdog)
set -uo pipefail
: "${Q36_MODELS:?set Q36_MODELS to the folder of the GGUF}"
R=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${Q36_OUT:-$R/q36-out}
mkdir -p "$OUT"
N=q36-run-$$
IMG=${Q36_IMAGE:-strata-sycl-dev}
envs=(-e ONEAPI_DEVICE_SELECTOR=level_zero:0 -e ZES_ENABLE_SYSMAN=1)
for kv in ${Q36_ENV:-}; do envs+=(-e "$kv"); done
args=$(printf '%q ' "$@")
peak=$OUT/.vram-$$
( up=; mx=0; minram=999
  while sleep 1; do
    if docker inspect $N >/dev/null 2>&1; then
      up=1
      a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
      [ "$a" -lt "$minram" ] && minram=$a
      if [ "$a" -lt "${Q36_MIN_RAM_GB:-0}" ]; then echo "WATCHDOG: MemAvailable ${a} GiB, ending $N" >&2; docker rm -f $N >/dev/null 2>&1; fi
      for c in /proc/[0-9]*/comm; do
        [ "$(cat $c 2>/dev/null)" = "strata" ] || continue   # (the one binary of both families: run one GPU job at a time)
        p=${c%/comm}
        v=$(grep -h "drm-total-vram0" $p/fdinfo/* 2>/dev/null | awk '{print $2}' | sort -n | tail -1)
        [ -n "$v" ] && [ "$v" -gt "$mx" ] && mx=$v
      done
      echo "$mx $minram" > $peak
    elif [ -n "$up" ]; then break; fi
  done ) &
wd=$!
# (Q36_LOCK is split into words on purpose)
${Q36_LOCK:-} docker run --rm ${Q36_STDIN:+-i} --name $N -u $(id -u):$(id -g) --device /dev/dri \
  -v "$R:/work/$(basename "$R")" -v "$Q36_MODELS:$Q36_MODELS:ro" -v "$OUT:$OUT" "${envs[@]}" $IMG \
  "cd /work/$(basename "$R") && ${Q36_PREFIX:-} ${Q36_BIN:-build-sycl-aot/strata} $args"
rc=$?
kill $wd 2>/dev/null
if [ -s $peak ]; then read mx minram < $peak; echo "q36_sycl_run: peak VRAM held by strata ${mx} KiB ($(awk -v k=$mx 'BEGIN{printf "%.2f", k/1048576}') GiB), min MemAvailable ${minram} GiB" >&2; fi
rm -f $peak
exit $rc
