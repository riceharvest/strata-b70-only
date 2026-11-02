#!/usr/bin/env python3
"""tools/pi_replay.py SESSION.jsonl [SESSION2.jsonl ...]: replays pi coding-agent sessions against a running server
(PI_REPLAY_URL, default http://127.0.0.1:8190), to collect the decode routing of agent traffic (tools/usage_profile.py)
or to time it.

Every assistant turn of a session (before its compaction) becomes one request: the system prompt (the session's
sections), the tools, and the recorded history up to that turn; the model answers it fresh (sampled like pi,
REPLAY_MAXTOK tokens, default 200), and the recorded turn - not the fresh answer - continues the history.  The engine's
STRATA_USAGE_DUMP then holds the decode routing of all those answers.  REPLAY_EVERY=N replays every Nth turn.  One
JSON line per request (prompt / reused tokens, read ms, decode tok/s); three failed requests in a row end it."""
import json, os, sys, time, urllib.request

URL = os.environ.get("PI_REPLAY_URL", "http://127.0.0.1:8190").rstrip("/") + "/v1/chat/completions"
MAXTOK = int(os.environ.get("REPLAY_MAXTOK", "200"))
EVERY = int(os.environ.get("REPLAY_EVERY", "1"))      # replay every Nth turn


def text_of(content):
    if isinstance(content, str):
        return content
    return "".join(c.get("text", "") for c in content if c.get("type") == "text")


def load(path):
    rows = [json.loads(l) for l in open(path)]
    sysm, tools, msgs, turns = None, [], [], []
    for r in rows:
        if r["type"] == "compaction":
            break
        if r["type"] != "message":
            continue
        m = r["message"]
        role = m["role"]
        if role == "system":
            secs = m.get("sections") or {}
            sysm = "\n\n".join(v for v in secs.values() if v) or text_of(m.get("content", ""))
            tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                        "parameters": t.get("parameters", {})}} for t in m.get("toolsAdded") or []]
        elif role == "user":
            msgs.append({"role": "user", "content": text_of(m["content"])})
        elif role == "assistant":
            turns.append(len(msgs))
            a = {"role": "assistant", "content": ""}
            th, tx, tc = [], [], []
            for c in m["content"]:
                if c["type"] == "thinking":
                    th.append(c["thinking"])
                elif c["type"] == "text":
                    tx.append(c["text"])
                elif c["type"] == "toolCall":
                    tc.append({"id": c["id"], "type": "function",
                               "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments", {}))}})
            a["content"] = "".join(tx)
            if th:
                a["reasoning_content"] = "".join(th)
            if tc:
                a["tool_calls"] = tc
            msgs.append(a)
        elif role == "toolResult":
            msgs.append({"role": "tool", "tool_call_id": m.get("toolCallId", ""), "content": text_of(m["content"])})
    return [{"role": "system", "content": sysm}] + msgs, tools, [t + 1 for t in turns]


def req(msgs, tools, seed):
    body = {"model": "x", "messages": msgs, "tools": tools, "max_tokens": MAXTOK, "temperature": 1.0, "top_p": 0.95,
            "top_k": 20, "seed": seed, "stream": False}
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}),
                               timeout=1800)
    j = json.loads(r.read())
    return j, time.time() - t0


n = 0
fails = 0
for path in sys.argv[1:]:
    msgs, tools, turns = load(path)
    for i, t in enumerate(turns):
        if i % EVERY:
            continue
        try:
            j, wall = req(msgs[:t], tools, 4321 + n)
        except Exception as e:   # one failed request does not end the collection
            print(json.dumps({"session": os.path.basename(path)[:24], "turn": i, "error": str(e)[:200]}), flush=True)
            fails += 1
            if fails >= 3:
                print("REPLAY-ABORT 3 failures in a row", flush=True)
                sys.exit(1)
            continue
        fails = 0
        u = j.get("usage", {}); tm = j.get("timings", {})
        print(json.dumps({"session": os.path.basename(path)[:24], "turn": i, "prompt": u.get("prompt_tokens"),
                          "gen": u.get("completion_tokens"), "cache_n": tm.get("cache_n"), "read_ms": tm.get("prompt_ms"),
                          "tps": tm.get("predicted_per_second"), "wall": round(wall, 1)}), flush=True)
        n += 1
print(f"REPLAY-DONE {n} requests", flush=True)
