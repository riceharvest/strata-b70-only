#!/usr/bin/env bash
# i24_arm.sh: Run one Flash-Next issue-#24 diagnostic arm through serve/server.py using ~/.config/strata/flash-next.json.
# Sends the 36-pair protocol (28 probe prompts + 8 long prompts, repeated twice = 72 requests),
# records STRATA_LAYER_HASH and optionally STRATA_DUMP_FIRST_LOGITS, and checks rep0 vs rep1.
#
# Never uses port 8012 and always uses a unique container name (strata-i24-<arm>-<pid>).
# Usage:
#   tools/flashnext/i24_arm.sh [--arm default|margin1e9|fix] [--port 8195] [--config ~/.config/strata/flash-next.json] [--out-dir <dir>]
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ARM="default"
PORT=8195
HOST="127.0.0.1"
CONFIG="${HOME}/.config/strata/flash-next.json"
OUT_DIR=""
EXTRA_ENV=()

while [ $# -gt 0 ]; do
    case "$1" in
        --arm) ARM="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --env) EXTRA_ENV+=("$2"); shift 2 ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0 ;;
        *) echo "i24_arm.sh: unknown option: $1" >&2; exit 2 ;;
    esac
done

if [ "$PORT" = "8012" ]; then
    echo "i24_arm.sh: refusing port 8012 (reserved for the user's inference server)" >&2
    exit 2
fi

if [ ! -f "$CONFIG" ]; then
    echo "i24_arm.sh: missing served config: $CONFIG" >&2
    exit 2
fi

TAG="${ARM}-$(date +%Y%m%d-%H%M%S)-$$"
if [ -z "$OUT_DIR" ]; then
    # Default under parent of repo (mounted at /work in strata-sycl.sh) so engine writes are visible on host
    OUT_DIR="$(dirname "$REPO")/i24-logs/${TAG}"
fi
mkdir -p "$OUT_DIR/pf"

CTR_NAME="strata-i24-${ARM}-$$"
SYCL_ROOT="${STRATA_SYCL_ROOT:-$(dirname "$REPO")}"
DERIVED_CFG="$OUT_DIR/flash-next-arm.json"
SERVER_LOG="$OUT_DIR/server.log"
ENGINE_LOG="$OUT_DIR/engine.log"
LAYER_HASH_HOST="$OUT_DIR/layer_hash.txt"
READY_FILE="$OUT_DIR/ready.json"

python3 - "$CONFIG" "$DERIVED_CFG" "$REPO" "$SYCL_ROOT" "$OUT_DIR" "$CTR_NAME" "$ARM" "$ENGINE_LOG" "$LAYER_HASH_HOST" "${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"}" <<'PY'
import json, os, sys

src_cfg, dst_cfg, repo, sycl_root, out_dir, ctr_name, arm, eng_log, lhash_host = sys.argv[1:10]
extra_kv = sys.argv[10:]

def to_container_path(host_path):
    rel = os.path.relpath(os.path.abspath(host_path), os.path.abspath(sycl_root))
    if not rel.startswith(".."):
        return "/work/" + rel.replace(os.sep, "/")
    return os.path.abspath(host_path)

with open(src_cfg, "r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg["exe"] = os.path.join(repo, "sycl/serve/strata-sycl.sh")
cfg["log"] = eng_log
cfg.pop("expert_profile_save", None)
cfg.pop("gpu_locks", None)
env = dict(cfg.get("env") or {})
env.pop("STRATA_USAGE_DUMP", None)
env["STRATA_SYCL_ROOT"] = sycl_root
env["STRATA_SYCL_NAME"] = ctr_name
env["STRATA_SYCL_BIN"] = "build-sycl-aot/strata"
env["STRATA_LAYER_HASH"] = to_container_path(lhash_host)
env["STRATA_DUMP_FIRST_LOGITS"] = to_container_path(os.path.join(out_dir, "pf/pf"))

mounts = [m for m in env.get("STRATA_SYCL_MOUNTS", "").split() if m]
if os.path.relpath(os.path.abspath(out_dir), os.path.abspath(sycl_root)).startswith(".."):
    mounts.append(os.path.abspath(out_dir))
if mounts:
    env["STRATA_SYCL_MOUNTS"] = " ".join(dict.fromkeys(mounts))

if arm == "default":
    pass
elif arm == "margin1e9":
    env["STRATA_LEAN_EXCHANGE_MARGIN"] = "1000000000"
elif arm == "fix":
    env["STRATA_I24_FIX"] = "1"
else:
    raise SystemExit(f"unknown --arm {arm!r} (expected default, margin1e9, or fix)")

for item in extra_kv:
    if "=" not in item:
        raise SystemExit(f"invalid --env {item!r}, expected KEY=VALUE")
    k, v = item.split("=", 1)
    env[k] = v

cfg["env"] = env
with open(dst_cfg, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
PY

SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    docker rm -f "$CTR_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[i24_arm] arm=$ARM port=$PORT container=$CTR_NAME out=$OUT_DIR"
python3 -m serve.server \
    --engine strata \
    --config "$DERIVED_CFG" \
    --host "$HOST" \
    --port "$PORT" \
    --ready-file "$READY_FILE" \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Wait for API + engine readiness (model load takes ~30-60 s on B70)
python3 - "$HOST" "$PORT" "$SERVER_PID" "$SERVER_LOG" <<'PY'
import json, os, sys, time, urllib.error, urllib.request

host, port, pid, slog = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
url = f"http://{host}:{port}/v1/models"
deadline = time.time() + 600
while time.time() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        raise SystemExit(f"server exited early; see {slog}")
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            if r.status == 200:
                print(f"[i24_arm] server ready at {host}:{port}", flush=True)
                sys.exit(0)
    except Exception:
        time.sleep(1.0)
raise SystemExit(f"timed out waiting for server at {host}:{port}; see {slog}")
PY

# Send the 36-pair protocol (28 probe prompts + 8 long prompts, twice)
python3 - "$HOST" "$PORT" "$REPO" "$OUT_DIR" <<'PY'
import glob, hashlib, json, os, sys, time, urllib.error, urllib.request

host, port, repo, out_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
base = f"http://{host}:{port}"

PROBE_PATHS = [
    "/home/dario/agent-test/probe-28-systematic.jsonl",
    os.path.join(repo, "tools/flashnext/probe-28-systematic.jsonl"),
]
DOC_PATHS = [
    ["/home/dario/Strata-b70/docs/B70.md", "/home/dario/Strata-b70/docs/INTEL.md", "/home/dario/Strata-b70/docs/QUANTS.md"],
    [os.path.join(repo, "docs/B70.md"), os.path.join(repo, "docs/INTEL.md")],
]
LONG_CHARS = [1200, 2400, 4000, 6000, 9000, 10500, 12000, 14000]

probe_list = None
for p in PROBE_PATHS:
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            probe_list = [json.loads(l) for l in f if l.strip()]
        print(f"[i24_arm] loaded {len(probe_list)} probe prompts from {p}", flush=True)
        break
if probe_list is None:
    print("[i24_arm] WARNING: probe-28-systematic.jsonl not found; using deterministic synthetic 28 prompts", flush=True)
    probe_list = [
        {"key": f"syn_{i+1}", "prompt": ("Explain step by step why deterministic ring indexing prevents overwrites. " * ((i % 7) + 2)).strip()}
        for i in range(28)
    ]

doc_txt = ""
for group in DOC_PATHS:
    parts = []
    for p in group:
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                parts.append(f.read())
    if parts:
        doc_txt = "\n".join(parts) + "\n"
        print(f"[i24_arm] loaded KEYS_DOCS ({len(doc_txt)} chars) from {group}", flush=True)
        break
if len(doc_txt) < 40000:
    doc_txt = (doc_txt + "The quick brown fox jumps over the lazy dog. ") * 2000

long_list = []
for i, n in enumerate(LONG_CHARS):
    off = (i * 3777) % max(1, len(doc_txt) - n)
    long_list.append({
        "key": f"long{i+1}",
        "prompt": "Read the following text and answer with one word, OK.\n\n" + doc_txt[off:off + n],
    })

prompts = probe_list + long_list
assert len(prompts) == 36, f"expected 36 prompts, got {len(prompts)}"

manifest = []
for rnd in (0, 1):
    for idx, item in enumerate(prompts):
        req_id = rnd * len(prompts) + idx
        key = item.get("key", item.get("id", f"p{idx}"))
        body = {
            "model": "x",
            "messages": [{"role": "user", "content": item["prompt"]}],
            "temperature": 0.0,
            "max_tokens": 2,
            "reasoning_effort": "medium",
            "stream": False,
        }
        req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as r:
            resp = json.loads(r.read())
        dt = time.time() - t0
        pf_path = os.path.join(out_dir, "pf", f"pf.{req_id}")
        pf_sha = ""
        if os.path.isfile(pf_path):
            with open(pf_path, "rb") as f:
                pf_sha = hashlib.sha256(f.read()).hexdigest()[:16]
        ch = (resp.get("choices") or [{}])[0].get("message") or {}
        manifest.append({
            "req_id": req_id,
            "round": rnd,
            "key": key,
            "s": round(dt, 3),
            "pf_sha16": pf_sha,
            "text": (ch.get("content") or "")[:40],
        })
        print(f"  req={req_id:2d} r={rnd} key={key:18s} {dt:5.2f}s pf_sha16={pf_sha or '-'}", flush=True)

with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

# Check first-window logit sha mismatches across rep 0 vs rep 1
pair_mismatches = []
for i in range(len(prompts)):
    a, b = manifest[i], manifest[i + len(prompts)]
    if a["pf_sha16"] != b["pf_sha16"]:
        pair_mismatches.append((i, a["key"], a["pf_sha16"], b["pf_sha16"]))
if pair_mismatches:
    print(f"[i24_arm] FIRST-LOGITS PAIR MISMATCHES ({len(pair_mismatches)}): {pair_mismatches}", flush=True)
else:
    print("[i24_arm] FIRST-LOGITS: all 36 pairs bit-identical!", flush=True)
PY

if [ -f "$LAYER_HASH_HOST" ]; then
    echo "[i24_arm] layer hashes saved to $LAYER_HASH_HOST"
    python3 "$REPO/tools/flashnext/hashdiff.py" --half-split --pairs 36 "$LAYER_HASH_HOST" || true
fi
