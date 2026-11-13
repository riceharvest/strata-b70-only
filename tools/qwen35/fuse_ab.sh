#!/bin/bash
# STRATA_Q36_FUSE old / new / old on one checkout: for each setting, start the 27B server (GGUF path), run tools/qwen35/fuse_ab.py
# (greedy answers: md5 and tok/s per request), print the engine's per-phase window times (STRATA_Q36_BATCH_TIME=1), stop.
# Run under the GPU lock (~/b70_prep/gpu.sh); one GPU job at a time, no build running.
#   usage: fuse_ab.sh "<engine args>" "<N list>" ["<FUSE list>"]         e.g.
#          fuse_ab.sh "--kv int8 --max-context 32768 --batch 4 --batch-mtp" "1 4"        (default FUSE list: 0 1 0)
#          fuse_ab.sh "--kv int8 --max-context 32768 --batch 4" "1 4"                    (plain batch windows, no drafts)
#          fuse_ab.sh "--max-context 32768" "1"                                           (single stream: the control, no change expected)
# Env: MODEL, TOKENIZER (the model's tokenizer folder), OUT (logs), PORT, MT (max tokens, default 300), EXTRA (more K=V for the engine).
CO=$(cd "$(dirname "$0")/../.." && pwd); ARGS=$1; NS=${2:-"1 4"}; FUSES=${3:-"0 1 0"}
MODEL=${MODEL:-/mnt/ssd/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf}
TOKENIZER=${TOKENIZER:-/home/dario/Strata-review/2026-11-11/q36/tokenizer}
OUT=${OUT:-/mnt/ssd/hermes/cache/scratch/strata-runs/lc}; PORT=${PORT:-8131}; MT=${MT:-300}
mkdir -p "$OUT"; cd "$CO" || exit 1
n=0
for F in $FUSES; do
  n=$((n + 1)); TAG=fuse$F-$n; CFG=$OUT/cfg-$TAG.json
  python3 - "$CO" "$CFG" "$TAG" "$MODEL" "$TOKENIZER" "$OUT" "$F" "$ARGS" $EXTRA <<'EOF'
import json, shlex, sys
co, cfg, tag, model, tok, out, fuse, args, *kv = sys.argv[1:]
env = {"STRATA_SYCL_MOUNTS": model.rsplit("/", 1)[0], "STRATA_SYCL_NAME": "strata-" + tag, "STRATA_SYCL_USER": "1000:1000",
       "STRATA_Q36_BATCH_TIME": "1", "STRATA_Q36_FUSE": fuse}
for x in kv:
    k, v = x.split("=", 1); env[k] = v
json.dump({"engine": "strata", "arch": "qwen35", "model_name": "qwen3.8-27b", "exe": co + "/sycl/serve/strata-sycl.sh",
           "args": ["--native", model] + shlex.split(args), "env": env, "tokenizer": tok, "log": "%s/engine-%s.log" % (out, tag)},
          open(cfg, "w"))
EOF
  STRATA_KEEP_LEFTOVERS=1 setsid python3 serve/server.py --engine strata --config "$CFG" --port "$PORT" --no-api-key > "$OUT/server-$TAG.out" 2>&1 &
  SRV=$!; t0=$(date +%s)
  until curl -s 127.0.0.1:$PORT/health | grep -q '"loaded": *true'; do
    kill -0 $SRV 2>/dev/null || { echo "START-FAIL ($TAG)"; tail -12 "$OUT/server-$TAG.out" | cut -c1-300; exit 1; }
    [ $(( $(date +%s) - t0 )) -gt 420 ] && { echo "START-FAIL timeout ($TAG)"; kill -TERM -$SRV; exit 1; }
    sleep 3
  done
  echo "######## STRATA_Q36_FUSE=$F (run $n), args: $ARGS"
  for N in $NS; do echo "== N=$N"; python3 tools/qwen35/fuse_ab.py "$PORT" "$N" "$MT" 2>&1 | cut -c1-240; done
  kill -TERM -$SRV; wait $SRV 2>/dev/null
  docker rm -f "strata-$TAG" > /dev/null 2>&1
  grep -h "windows:" "$OUT/engine-$TAG.log" | tail -2 | cut -c1-260
done
