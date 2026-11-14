#!/usr/bin/env python3
"""Text and speed checks for a lone request on a --batch N server (lane SOLO).  Standard library only.

  run: one streamed request (chat or /v1/completions), written to a JSON file: its text, its sha256, the usage and the decode
       speed from the arrival of its text pieces.  `--second-after S` starts a second request (seed + 1) S seconds after the
       first, so the server hands the first one over from the solo path to a batch slot.
  cmp: two of those files: equal or not, and where the texts first differ.

  python3 tools/solo_text_check.py run http://127.0.0.1:8095 --endpoint completions --seed 1234 --out a.json
  python3 tools/solo_text_check.py cmp a.json b.json
"""
import argparse
import hashlib
import json
import sys
import threading
import time
import urllib.request

PROMPT = "Write a short Python function that computes the Fibonacci sequence and explain it."


def one(base, endpoint, prompt, tokens, temperature, seed, top_k, top_p, ignore_eos, model, thinking_off=True):
    body = {"model": model, "max_tokens": tokens, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": ignore_eos}
    if seed:
        body["seed"] = seed
    if temperature > 0:
        body.update(top_k=top_k, top_p=top_p)
    if endpoint == "chat":
        url = base + "/v1/chat/completions"
        body["messages"] = [{"role": "user", "content": prompt}]
        if thinking_off:
            body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        url = base + "/v1/completions"
        body["prompt"] = prompt
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    text, usage, pieces = [], {}, []
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            usage = chunk.get("usage") or usage
            for c in chunk.get("choices", []):
                piece = c.get("text") or (c.get("delta") or {}).get("content") or ""
                if piece:
                    text.append(piece)
                    pieces.append(time.perf_counter())
    full = "".join(text)
    n = usage.get("completion_tokens")
    tps = (n - 1) / (pieces[-1] - pieces[0]) if n and n > 1 and len(pieces) > 3 and pieces[-1] > pieces[0] else None
    return {"text": full, "sha256": hashlib.sha256(full.encode()).hexdigest(), "usage": usage,
            "decode_tok_s": round(tps, 1) if tps else None}


def cmd_run(a):
    out = {}
    second = None
    if a.second_after > 0:
        def later():
            time.sleep(a.second_after)
            out["second"] = one(a.base, a.endpoint, a.prompt, a.tokens, a.temperature, a.seed + 1, a.top_k, a.top_p,
                                a.ignore_eos, a.model)
        second = threading.Thread(target=later)
        second.start()
    out.update(one(a.base, a.endpoint, a.prompt, a.tokens, a.temperature, a.seed, a.top_k, a.top_p, a.ignore_eos, a.model))
    if second is not None:
        second.join()
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{a.endpoint} seed={a.seed} T={a.temperature}: {out['usage'].get('completion_tokens')} tokens, "
          f"{out['decode_tok_s']} tok/s, sha256 {out['sha256'][:16]}" +
          (f"; second request {out['second']['decode_tok_s']} tok/s" if "second" in out else ""))


def cmd_cmp(a):
    x, y = (json.load(open(f)) for f in (a.a, a.b))
    if x["text"] == y["text"]:
        print(f"IDENTICAL ({len(x['text'])} characters, sha256 {x['sha256'][:16]})")
        return 0
    k = next((i for i, (p, q) in enumerate(zip(x["text"], y["text"])) if p != q), min(len(x["text"]), len(y["text"])))
    print(f"DIFFERENT: the first {k} characters agree (of {len(x['text'])} and {len(y['text'])}); "
          f"a: ...{x['text'][max(0, k - 20):k + 20]!r}  b: ...{y['text'][max(0, k - 20):k + 20]!r}")
    return 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("base")
    r.add_argument("--endpoint", choices=("chat", "completions"), default="completions")
    r.add_argument("--model", default="qwen")
    r.add_argument("--prompt", default=PROMPT)
    r.add_argument("--tokens", type=int, default=256)
    r.add_argument("--temperature", type=float, default=1.0, help="0: greedy")
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--top-k", type=int, default=20)
    r.add_argument("--top-p", type=float, default=0.95)
    r.add_argument("--ignore-eos", action="store_true")
    r.add_argument("--second-after", type=float, default=0.0, help="start a second request after this many seconds")
    r.add_argument("--out", required=True)
    c = sub.add_parser("cmp")
    c.add_argument("a")
    c.add_argument("b")
    a = p.parse_args()
    if a.cmd == "run":
        cmd_run(a)
        return 0
    return cmd_cmp(a)


if __name__ == "__main__":
    sys.exit(main())
