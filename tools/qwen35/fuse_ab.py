#!/usr/bin/env python3
"""fuse_ab.py PORT [N] [MAX_TOKENS]: N concurrent greedy requests (thinking off, temperature 0, fixed prompts) against a running
serve/server.py of the 27B, then one line per request: tokens, decode tok/s as the server reports it, and the md5 of the
answer text.  For the STRATA_Q36_FUSE old / new / old comparison (tools/qwen35/fuse_ab.sh): the md5 of request i must be the
same in every run.  On an engine started with --batch and --batch-mtp a single request (N=1) already runs through the batch
window and commit (its window schedule depends only on its own text), so N=1 is the clean text test; N=4 texts are comparable
only if the two old runs agree with each other (otherwise inconclusive, not a failure).  On an engine started without --batch,
N=1 is the single-stream control: STRATA_Q36_FUSE does not touch that path."""
import hashlib, json, sys, threading, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8131
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1
MT = int(sys.argv[3]) if len(sys.argv) > 3 else 300
PROMPTS = [
    "Explain, step by step, how a hash map handles collisions, and write a short Python class that implements it with chaining.",
    "Write a Python function that parses a CSV line with quoted fields and escaped quotes, then explain the edge cases it handles.",
    "Give a careful comparison of B-trees and LSM-trees for a write-heavy key-value store, with the trade-offs in a table.",
    "Describe how TCP congestion control works (slow start, congestion avoidance, fast retransmit) and why each part exists.",
]
out = [None] * N


def run(i):
    body = json.dumps({"model": "qwen3.8-27b", "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
                       "temperature": 0, "max_tokens": MT, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t0 = time.time()
    try:
        r = json.loads(urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:%d/v1/chat/completions" % PORT, body, {"Content-Type": "application/json"}), timeout=900).read())
        txt = r["choices"][0]["message"].get("content") or ""
        t = r.get("timings") or {}
        out[i] = dict(i=i, n=t.get("predicted_n"), tps=t.get("predicted_per_second"), wall=time.time() - t0,
                      md5=hashlib.md5(txt.encode()).hexdigest(), chars=len(txt))
    except Exception as e:  # noqa: BLE001
        out[i] = dict(i=i, err=repr(e)[:120])


t0 = time.time()
th = [threading.Thread(target=run, args=(i,)) for i in range(N)]
[t.start() for t in th]
[t.join() for t in th]
for o in out:
    print(o)
ok = [o for o in out if "err" not in o]
print("N=%d wall %.1f s, tokens %d, sum of per-request decode rates %.1f tok/s" % (
    N, time.time() - t0, sum((o["n"] or 0) for o in ok), sum((o["tps"] or 0.0) for o in ok)))
