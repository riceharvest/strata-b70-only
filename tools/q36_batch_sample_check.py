#!/usr/bin/env python3
"""Runs N sampled chat requests at once (or one after the other) against a running strata server and saves each request's text,
so that two runs can be compared token for token.  Standard library only.  It does not start the server and does not touch the
card by itself: it only sends the requests you point it at.

  # text check with a decoy in a slot (use --n 3 on a --batch 4 server)
  python3 tools/q36_batch_sample_check.py run --n 3 --decoy 1500 --out new_text.json
  # 4 at once, a fixed seed each (seed = --seed + i), temperature 0.7, 256 tokens, thinking as the server's default
  python3 tools/q36_batch_sample_check.py run --url http://127.0.0.1:8095 --n 4 --out new.json
  # the same four requests one at a time (the reference: one request alone)
  python3 tools/q36_batch_sample_check.py run --url http://127.0.0.1:8095 --n 4 --alone --out alone.json
  # greedy and sampled together: --temps 0,0.7,0,0.7 (one temperature a request)
  python3 tools/q36_batch_sample_check.py run --n 4 --temps 0,0.7,0,0.7 --out mixed.json
  # whether two runs gave the same text, request by request
  python3 tools/q36_batch_sample_check.py compare old.json new.json

The text is the streamed reasoning plus content.  The wall time is from the first request sent to the last token received.

Why --decoy: the server sends a request that arrives alone down the solo path (GEN) and, when another request arrives, stops it
and continues it in a batch slot with a new random stream (the same seed again); a request left alone goes back to the solo
path.  A sampled text that went through such a hand-over is not comparable with the same request alone.  `--decoy 1500` first
starts a long greedy request, waits --decoy-wait seconds, and then sends the test requests, which all go straight into slots;
the decoy outlives them and is closed at the end (the wall does not include it).  Use it for the text checks, not for the
timing runs (the decoy takes a slot and rows).
"""
import argparse
import json
import sys
import threading
import time
import urllib.request

PROMPTS = [
    "Write a short Python function that merges two sorted lists, then explain its running time.",
    "Explain in plain words why the sky is blue and why sunsets are red.",
    "Give me a three-step plan to debug a program that crashes only on the second run.",
    "Describe how a hash map handles collisions, with a small example.",
    "Summarise the main ideas of the actor model of concurrency.",
    "What are the trade-offs between a B-tree and an LSM tree for a write-heavy store?",
    "Write a haiku about a quiet harbour, then translate it into plain prose.",
    "How does speculative decoding speed up a language model? Keep it short.",
]


def one(url, body, res, i, t_start):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    text, n_chunks, first = [], 0, None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except ValueError:
                continue
            for ch in ev.get("choices", []):
                d = ch.get("delta") or {}
                for k in ("reasoning_content", "reasoning", "content"):
                    if d.get(k):
                        text.append(d[k])
                        n_chunks += 1
                        if first is None:
                            first = time.time() - t_start
    res[i] = {"text": "".join(text), "chunks": n_chunks, "first_s": first, "done_s": time.time() - t_start}


def decoy(url, model, n, stop, started):
    body = {"model": model, "stream": True, "max_tokens": n, "temperature": 0,
            "messages": [{"role": "user", "content": "Count from one upwards in words, one number a line, without stopping."}]}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for _ in r:
                started.set()
                if stop.is_set():
                    break
    except Exception:
        pass
    started.set()


def cmd_run(a):
    temps = [float(x) for x in a.temps.split(",")] if a.temps else [a.temperature] * a.n
    if len(temps) != a.n:
        sys.exit("--temps needs one value per request")
    bodies = []
    for i in range(a.n):
        b = {"model": a.model, "stream": True, "max_tokens": a.max_tokens, "temperature": temps[i],
             "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}]}
        if temps[i] > 0:
            b.update({"top_p": a.top_p, "top_k": a.top_k, "seed": a.seed + i})
        if a.presence_penalty:
            b["presence_penalty"] = a.presence_penalty
        bodies.append(b)
    res = [None] * a.n
    dstop, dstart = threading.Event(), threading.Event()
    dthread = None
    if a.decoy:
        dthread = threading.Thread(target=decoy, args=(a.url, a.model, a.decoy, dstop, dstart))
        dthread.start()
        dstart.wait(60)
        time.sleep(a.decoy_wait)
    t0 = time.time()
    if a.alone:
        for i in range(a.n):
            one(a.url, bodies[i], res, i, time.time())
    else:
        ts = [threading.Thread(target=one, args=(a.url, bodies[i], res, i, t0)) for i in range(a.n)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    wall = time.time() - t0
    if dthread is not None:
        dstop.set()
        dthread.join(30)
    out = {"alone": a.alone, "wall_s": wall, "temps": temps, "bodies": bodies, "results": res}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{a.n} requests, {'one at a time' if a.alone else 'at once'}: wall {wall:.2f} s")
    for i, r in enumerate(res):
        print(f"  request {i}: temperature {temps[i]}, {r['chunks']} chunks, done at {r['done_s']:.2f} s, {len(r['text'])} characters")


def cmd_compare(a):
    x, y = json.load(open(a.a)), json.load(open(a.b))
    same = 0
    for i, (p, q) in enumerate(zip(x["results"], y["results"])):
        eq = p["text"] == q["text"]
        same += eq
        if eq:
            print(f"  request {i}: the same text ({len(p['text'])} characters)")
        else:
            k = next((j for j, (c, d) in enumerate(zip(p["text"], q["text"])) if c != d), min(len(p["text"]), len(q["text"])))
            print(f"  request {i}: DIFFERENT from character {k} (lengths {len(p['text'])} and {len(q['text'])})")
    print(f"{same} of {len(x['results'])} the same; wall {x['wall_s']:.2f} s and {y['wall_s']:.2f} s")
    sys.exit(0 if same == len(x["results"]) else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", default="http://127.0.0.1:8095")
    r.add_argument("--model", default="strata")
    r.add_argument("--n", type=int, default=4)
    r.add_argument("--alone", action="store_true", help="one request at a time")
    r.add_argument("--temperature", type=float, default=0.7)
    r.add_argument("--temps", default="", help="a temperature per request, comma separated (overrides --temperature)")
    r.add_argument("--top-p", dest="top_p", type=float, default=0.8)
    r.add_argument("--top-k", dest="top_k", type=int, default=20)
    r.add_argument("--presence-penalty", dest="presence_penalty", type=float, default=0.0)
    r.add_argument("--max-tokens", dest="max_tokens", type=int, default=256)
    r.add_argument("--seed", type=int, default=1000)
    r.add_argument("--decoy", type=int, default=0, help="first start a greedy request of this many tokens (see above); 0: none")
    r.add_argument("--decoy-wait", dest="decoy_wait", type=float, default=2.0)
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(fn=cmd_compare)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
