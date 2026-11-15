#!/usr/bin/env python3
"""agent_replay.py URL MODEL THREADS TURNS sessions=<folder> [requests=<glob>] [key=value ...]

Replays recorded agent threads against an OpenAI-compatible server, to measure a server on what agents send instead of
on one prompt at a time.

A thread is one recorded agent session: a JSONL file under `sessions`, one message a line (role user / assistant / tool,
content, an assistant's tool_calls, a tool's tool_call_id), as an agent framework logs it.  Each thread runs under a real
system prompt and tool list, taken from recorded request bodies (`requests`: JSON files that are an OpenAI chat request,
or hold one at request.body); the bodies with the most common tool list are used, one system prompt a thread.  Without
`requests` the threads run with no system prompt and no tools.

Turn k of a thread sends the recorded history before the session's k-th assistant message and lets the model write that
message; the next turn sends the RECORDED history again (as a client does: the answer without its reasoning, the tool's
result, the next message).  So what the server reads each turn is what an agent's turn brings: the last answer, a tool
result, sometimes a user message.  THREADS threads run at once; each waits `gap` seconds between its turns (the tool
running).  Nothing leaves the machine but the requests to URL.

keys: gap=1.0, max_tokens=768, start=0 (first assistant turn replayed), min_ctx=0 (a thread starts at its first turn whose
prompt has at least this many tokens: long conversations), ctx=60000 (a thread ends before its prompt passes this many
tokens; both estimated as characters / 3), seed=1 (which sessions), tag=<name> and out=<folder> (the turns are
written to <out>/replay-<tag>.json; no out: not written), stagger=2.0 (seconds between thread starts), same_system=0
(1: every thread gets the same system prompt).

It prints the output tokens a second over the whole run, and for first and later turns: prompt tokens, the share the
server had cached (usage.prompt_tokens_details.cached_tokens), tokens read a turn, time to the first token and the turn's
time (p50, p90), and the later turns that read more than half their prompt again (a thread that lost its cache).
"""
import glob, json, os, random, sys, threading, time, urllib.request

URL, MODEL, NTHREADS, NTURNS = sys.argv[1].rstrip("/"), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
kv = dict(a.split("=", 1) for a in sys.argv[5:])
GAP, MAXTOK, START = float(kv.get("gap", 1.0)), int(kv.get("max_tokens", 768)), int(kv.get("start", 0))
CTX, SEED, TAG, STAGGER = int(kv.get("ctx", 60000)), int(kv.get("seed", 1)), kv.get("tag", "run"), float(kv.get("stagger", 2.0))
MIN_CTX = int(kv.get("min_ctx", 0))
SAME_SYSTEM = kv.get("same_system", "0") == "1"
SESS = kv.get("sessions", "")
REQS = kv.get("requests", "")
OUT = kv.get("out", "")
if not SESS:
    sys.exit("agent_replay.py: sessions=<folder of recorded sessions (.jsonl)> is required")


def est_tokens(x):   # (the server's usage gives the true count; this only decides where a thread stops)
    return len(x) // 3


def load_prefixes():
    """Real (system prompt, tools) pairs: the recorded requests that share the most common tool list."""
    by = {}
    for f in sorted(glob.glob(REQS), key=os.path.getmtime) if REQS else []:
        try:
            b = json.load(open(f))
            b = (b.get("request") or {}).get("body") or b
        except Exception:
            continue
        m = b.get("messages") or []
        if not m or m[0].get("role") != "system" or not isinstance(m[0].get("content"), str) or not b.get("tools"):
            continue
        by.setdefault(json.dumps(b["tools"], sort_keys=True), {})[m[0]["content"]] = b["tools"]
    if not by:
        return [(None, None)]
    tools_key = max(by, key=lambda k: len(by[k]))
    return [(s, t) for s, t in by[tools_key].items()]


def clean(m):
    """A recorded message as a client sends it back: no reasoning, no bookkeeping fields."""
    out = {"role": m["role"]}
    c = m.get("content")
    out["content"] = c if isinstance(c, str) else ("" if c is None else json.dumps(c))
    if m["role"] == "assistant" and m.get("tool_calls"):
        out["tool_calls"] = [{"id": t.get("id") or "call_%d" % i, "type": "function",
                              "function": {"name": (t.get("function") or {}).get("name") or t.get("name") or "tool",
                                           "arguments": (t.get("function") or {}).get("arguments") if isinstance((t.get("function") or {}).get("arguments"), str)
                                           else json.dumps((t.get("function") or {}).get("arguments") or t.get("arguments") or {})}}
                             for i, t in enumerate(m["tool_calls"])]
    if m["role"] == "tool":
        out["tool_call_id"] = m.get("tool_call_id") or "call_0"
    return out


def prompt_est(rows, upto, prefix):
    """The estimated tokens of the prompt that asks for the assistant message at rows[upto]."""
    system, tools = prefix
    return est_tokens(system or "") + est_tokens(json.dumps(tools or [])) + sum(est_tokens(json.dumps(clean(r))) for r in rows[:upto])


def first_turn(rows, prefix):
    """The first assistant turn a thread replays: `start`, or with min_ctx the first one whose prompt is that long."""
    a_idx = [k for k, r in enumerate(rows) if r["role"] == "assistant"]
    for t in range(START, len(a_idx)):
        if prompt_est(rows, a_idx[t], prefix) >= MIN_CTX:
            return t
    return None


def load_sessions(n, prefix):
    fs = sorted(glob.glob(os.path.join(SESS, "*.jsonl")), key=os.path.getmtime)
    good = []
    for f in fs:
        try:
            rows = [json.loads(l) for l in open(f) if l.strip()]
        except Exception:
            continue
        rows = [r for r in rows if r.get("role") in ("user", "assistant", "tool")]
        if not rows or rows[0]["role"] != "user":
            continue
        a_idx = [k for k, r in enumerate(rows) if r["role"] == "assistant"]
        t0 = first_turn(rows, prefix)
        # (a thread must have its turns, and with min_ctx its first one must fit the context)
        if t0 is None or t0 + NTURNS > len(a_idx) or (MIN_CTX > 0 and prompt_est(rows, a_idx[t0], prefix) + MAXTOK > CTX):
            continue
        good.append(rows)
    random.Random(SEED).shuffle(good)
    return good[:n]


def call(body):
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); tf = None; usage = None
    with urllib.request.urlopen(req, timeout=3000) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            j = json.loads(line[5:])
            if j.get("usage"):
                usage = j["usage"]
            ch = j.get("choices") or []
            d = ch[0].get("delta", {}) if ch else {}
            if tf is None and (d.get("content") or d.get("reasoning_content") or d.get("reasoning") or d.get("tool_calls")):
                tf = time.time()
    te = time.time()
    u = usage or {}
    return {"prompt": u.get("prompt_tokens", 0), "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "out": u.get("completion_tokens", 0), "ttft": (tf or te) - t0, "wall": te - t0, "t0": t0, "te": te}


def thread(i, rows, prefix, res):
    system, tools = prefix
    a_idx = [k for k, r in enumerate(rows) if r["role"] == "assistant"]
    t0 = first_turn(rows, prefix)
    for turn in range(t0, t0 + NTURNS):
        hist = [clean(r) for r in rows[:a_idx[turn]]]
        if est_tokens(system or "") + est_tokens(json.dumps(tools or [])) + sum(est_tokens(json.dumps(m)) for m in hist) + MAXTOK > CTX:
            break
        body = {"model": MODEL, "messages": ([{"role": "system", "content": system}] if system else []) + hist, "max_tokens": MAXTOK,
                "stream": True, "stream_options": {"include_usage": True}}
        if tools:
            body["tools"] = tools
        try:
            r = call(body)
        except Exception as e:
            res[i].append({"err": repr(e)[:160], "turn": turn})
            break
        r["turn"] = turn
        res[i].append(r)
        time.sleep(GAP)


def main():
    prefixes = load_prefixes()
    sessions = load_sessions(NTHREADS, prefixes[0])
    if len(sessions) < NTHREADS:
        print("only %d sessions long enough" % len(sessions))
    res = [[] for _ in sessions]
    th = [threading.Thread(target=thread, args=(i, rows, prefixes[0] if SAME_SYSTEM else prefixes[i % len(prefixes)], res))
          for i, rows in enumerate(sessions)]
    T0 = time.time()
    for t in th:
        t.start()
        time.sleep(STAGGER)
    for t in th:
        t.join()
    wall = time.time() - T0
    ok = [r for rs in res for r in rs if "err" not in r]
    errs = [r for rs in res for r in rs if "err" in r]
    first = [r for rs in res for r in rs[:1] if "err" not in r]
    later = [r for rs in res for r in rs[1:] if "err" not in r]
    def pct(v, p):
        v = sorted(v)
        return v[min(len(v) - 1, int(p / 100 * len(v)))] if v else 0.0
    out = sum(r["out"] for r in ok)
    print("[%s] %d threads x up to %d turns (%d system prompts, gap %.1f s): %d turns done, %d errors, wall %.1f s" % (
        TAG, len(sessions), NTURNS, 1 if SAME_SYSTEM else min(len(prefixes), len(sessions)), GAP, len(ok), len(errs), wall))
    print("[%s] output %d tokens = %.1f tok/s over the wall; turns a minute %.1f" % (TAG, out, out / wall, 60.0 * len(ok) / wall))
    for name, rs in (("first turn", first), ("later turns", later)):
        if not rs:
            continue
        p = sum(r["prompt"] for r in rs); c = sum(r["cached"] for r in rs)
        print("[%s] %-11s n=%3d: prompt %6.0f tokens (mean), cached %5.1f%%, read %6.0f a turn; out %4.0f; time to first token p50 %5.1f s p90 %5.1f s; turn p50 %5.1f s p90 %5.1f s" % (
            TAG, name, len(rs), p / len(rs), 100.0 * c / max(1, p), (p - c) / len(rs), sum(r["out"] for r in rs) / len(rs),
            pct([r["ttft"] for r in rs], 50), pct([r["ttft"] for r in rs], 90), pct([r["wall"] for r in rs], 50), pct([r["wall"] for r in rs], 90)))
    if later:
        worst = sorted(later, key=lambda r: r["prompt"] - r["cached"])[-5:]
        print("[%s] the five largest reads of a later turn (tokens read of prompt): %s" % (
            TAG, ", ".join("%d of %d" % (r["prompt"] - r["cached"], r["prompt"]) for r in worst)))
        miss = sum(1 for r in later if r["prompt"] - r["cached"] > 0.5 * r["prompt"])
        print("[%s] later turns that read more than half their prompt again: %d of %d" % (TAG, miss, len(later)))
    for e in errs[:3]:
        print("[%s] error:" % TAG, e)
    if OUT:
        json.dump(res, open(os.path.join(OUT, "replay-%s.json" % TAG), "w"))


main()
