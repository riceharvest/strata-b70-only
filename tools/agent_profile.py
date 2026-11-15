#!/usr/bin/env python3
"""agent_profile.py <run folder | engine log> [--json out.json]: where a run's time went.

Reads what a run left behind (tools/agent_bench.py's folder, or just an engine log):
  engine.log     the dense engine's `PROF {json}` lines (STRATA_Q36_PROFILE=1): a record a request, a batch slot's end,
                 every 64 batch windows
  metrics.json   the server's GET /metrics (its history of finished requests: waits, first token, duration)
  run.json       the agents and their tasks (time, exit code)
and prints, with every part optional when its file is missing:
  1. the run: agents, tasks, requests, output tokens and tokens a second over the run
  2. the engine's time by part: prompt reads, decoding alone, batch windows by part, session work (parking, bringing
     back, copies), and what is left (idle: no request, or waiting on the harness)
  3. the cache: how each request got its cells, the share of the prompt that was kept, tokens read a request
  4. decoding alone: tokens a second, a step's time, tokens a step, the steps by kind, drafts accepted (lookups apart),
     by context length
  5. batch windows: slots and rows a window, a window's parts, tokens a window, drafts accepted; and the decode by
     the number of streams decoding at once (1: a request alone), which shows where the summed decode peaks
  6. the server's view: the wait before the engine began, time to the first token, a request's time
  7. the slowest requests with what they spent their time on
"""
import json, os, re, statistics as st, sys, glob


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p / 100.0 * len(v)))] if v else 0.0


def load(path):
    d = path if os.path.isdir(path) else None
    log = os.path.join(d, "engine.log") if d else path
    prof = []
    if os.path.exists(log):
        for line in open(log, errors="replace"):
            i = line.find("PROF {")
            if i >= 0:
                try:
                    prof.append(json.loads(line[i + 5:]))
                except ValueError:
                    pass
    metrics = run = None
    if d:
        if os.path.exists(os.path.join(d, "metrics.json")):
            try:
                metrics = json.load(open(os.path.join(d, "metrics.json")))
            except ValueError:
                pass
        if os.path.exists(os.path.join(d, "run.json")):
            run = json.load(open(os.path.join(d, "run.json")))
    return prof, metrics, run


def history(metrics):
    """The finished requests in the server's /metrics, wherever that build keeps them."""
    if not isinstance(metrics, dict):
        return []
    for k in ("history", "requests_history", "recent"):
        if isinstance(metrics.get(k), list):
            return metrics[k]
    for v in metrics.values():
        if isinstance(v, dict):
            h = history(v)
            if h:
                return h
        if isinstance(v, list) and v and isinstance(v[0], dict) and "prompt_tokens" in v[0] and "duration_s" in v[0]:
            return v
    return []


BUCKETS = ((0, 16000, "under 16K"), (16000, 32000, "16K-32K"), (32000, 64000, "32K-64K"), (64000, 128000, "64K-128K"), (128000, 1 << 40, "128K and up"))


def bucket(cells):
    for lo, hi, name in BUCKETS:
        if lo <= cells < hi:
            return name
    return "?"


def report(prof, metrics, run):
    out = []
    P = out.append
    reqs = [r for r in prof if r.get("ev") == "req"]
    slots = [r for r in prof if r.get("ev") == "slot"]
    wins = [r for r in prof if r.get("ev") == "win"]
    res = {}
    # ---- 1. the run
    t_lo = min([r["t"] - r.get("total_ms", 0) for r in reqs] + [w["t"] for w in wins] or [0.0])
    t_hi = max([r["t"] for r in prof] or [0.0])
    span = max(1e-9, (t_hi - t_lo) / 1000.0)
    wall = run["wall"] if run else span
    tok_solo = sum(r["out"] for r in reqs if r["kind"] == "solo")
    tok_adm = sum(r["out"] for r in reqs if r["kind"] == "adm")
    tok_slot = sum(s["out"] for s in slots)
    out_tokens = tok_solo + tok_slot   # (a slot's count has its admission's first token in it)
    P("1. The run")
    if run:
        tasks = run.get("tasks", [])
        P("   %d agents (%s), %d tasks: %d ended with exit 0, %d timed out; %.0f s" % (
            run["agents"], run["harness"], len(tasks), sum(1 for t in tasks if t["exit"] == 0), sum(1 for t in tasks if t.get("timed_out")), run["wall"]))
        for t in sorted(tasks, key=lambda x: (x["agent"], x["start"])):
            P("     agent %d  %-18s %-22s %6.0f s  exit %s%s" % (t["agent"], t["task"], t.get("kind") or "", t["seconds"], t["exit"], "  TIMED OUT" if t.get("timed_out") else ""))
    P("   %d requests reached the engine in %.0f s; %d tokens written = %.1f tok/s over the run" % (len(reqs), span, out_tokens, out_tokens / wall))
    res["run"] = {"requests": len(reqs), "seconds": round(span, 1), "wall": wall, "output_tokens": out_tokens, "tok_s": round(out_tokens / wall, 2)}
    if not reqs:
        P("   (no PROF records: start the engine with STRATA_Q36_PROFILE=1)")
        return "\n".join(out), res
    # ---- 2. the engine's time
    read_solo = sum(r["read_ms"] for r in reqs if r["kind"] == "solo")
    read_adm = sum(r["read_ms"] for r in reqs if r["kind"] == "adm")
    dec_solo = sum(r["dec_ms"] for r in reqs if r["kind"] == "solo")
    sess = sum(r["sess_ms"] for r in reqs)
    park = sum(r["park_ms"] for r in reqs)
    unpark = sum(r["unpark_ms"] for r in reqs)
    wparts = {k: sum(w["n"] * w["ms"][k] for w in wins) for k in ("draft0", "window", "picks", "commit", "drafts")}
    wtotal = sum(wparts.values())
    n_win = sum(w["n"] for w in wins)
    used = read_solo + read_adm + dec_solo + wtotal + sess
    total = span * 1000.0
    P("")
    P("2. The engine's time (%.0f s)" % span)
    rows = [("reading prompts, a request alone", read_solo), ("reading prompts, beside decoding slots", read_adm), ("decoding alone", dec_solo),
            ("batch windows", wtotal), ("  the window (verify rows)", wparts["window"]), ("  next drafts", wparts["drafts"]), ("  first drafts", wparts["draft0"]),
            ("  commit", wparts["commit"]), ("  picks", wparts["picks"]), ("session work before a read", sess), ("  parking a cache", park),
            ("  bringing one back", unpark), ("not in any of these (idle, the harness's turn)", max(0.0, total - used))]
    for name, ms in rows:
        P("   %-44s %8.1f s  %5.1f%%" % (name, ms / 1000.0, 100.0 * ms / total))
    P("   (a read beside decoding slots lets their windows run between its chunks: those windows are counted in both lines)")
    res["time_s"] = {name.strip(): round(ms / 1000.0, 2) for name, ms in rows}
    # ---- 3. the cache
    P("")
    P("3. The cache")
    hows = {}
    for r in reqs:
        hows.setdefault(r["how"], []).append(r)
    for h, rs in sorted(hows.items(), key=lambda kv: -len(kv[1])):
        P("   %-10s %4d requests: prompt %7.0f tokens, kept %5.1f%%, read %6.0f a request" % (
            h, len(rs), st.mean(r["prompt"] for r in rs), 100.0 * sum(r["kept"] for r in rs) / max(1, sum(r["prompt"] for r in rs)), st.mean(r["read"] for r in rs)))
    tot_p, tot_k, tot_r = sum(r["prompt"] for r in reqs), sum(r["kept"] for r in reqs), sum(r["read"] for r in reqs)
    rd = [r for r in reqs if r["read"] >= 256 and r["read_ms"] > 0]
    P("   all: %d prompt tokens, %.1f%% kept, %d read; tokens read a request: median %d, nine in ten under %d" % (
        tot_p, 100.0 * tot_k / max(1, tot_p), tot_r, pct([r["read"] for r in reqs], 50), pct([r["read"] for r in reqs], 90)))
    if rd:
        P("   read speed (reads of 256 tokens and more): %.0f tok/s alone, %.0f beside decoding slots" % (
            1000.0 * sum(r["read"] for r in rd if r["kind"] == "solo") / max(1e-9, sum(r["read_ms"] for r in rd if r["kind"] == "solo")),
            1000.0 * sum(r["read"] for r in rd if r["kind"] == "adm") / max(1e-9, sum(r["read_ms"] for r in rd if r["kind"] == "adm"))))
    lost = [r for r in reqs if r["prompt"] >= 4096 and r["read"] > 0.5 * r["prompt"]]
    P("   requests of 4,096 tokens and more that read over half their prompt: %d of %d" % (len(lost), sum(1 for r in reqs if r["prompt"] >= 4096)))
    P("   parked %d times (%.1f s), brought back %d times (%.1f s), to the spill folder %d, from it %d" % (
        sum(1 for r in reqs if r["park_ms"] > 0), park / 1000.0, sum(1 for r in reqs if r["unpark_ms"] > 0), unpark / 1000.0,
        sum(r["to_drive"] for r in reqs), sum(r["from_drive"] for r in reqs)))
    res["cache"] = {"kept_pct": round(100.0 * tot_k / max(1, tot_p), 1), "read_tokens": tot_r, "lost": len(lost),
                    "how": {h: len(rs) for h, rs in hows.items()}}
    # ---- 4. decoding alone
    solo = [r for r in reqs if r["kind"] == "solo" and r["out"] >= 8 and r["dec_ms"] > 0]
    P("")
    P("4. Decoding alone (%d requests of 8 tokens and more, %d tokens)" % (len(solo), sum(r["out"] for r in solo)))
    if solo:
        def line(name, rs):
            toks = sum(r["out"] for r in rs)
            ms = sum(r["dec_ms"] for r in rs)
            steps = sum(sum(r["steps"].values()) for r in rs)
            off, acc = sum(r["offered"] for r in rs), sum(r["accepted"] for r in rs)
            lo, la = sum(r["look_offered"] for r in rs), sum(r["look_accepted"] for r in rs)
            kinds = {k: sum(r["steps"][k] for r in rs) for k in ("tree", "chain", "lookup", "drafter")}
            P("   %-12s %4d req %7d tok  %6.1f tok/s  %5.1f ms a step  %4.2f tok a step  drafts %4.1f%% of %d (lookups %4.1f%% of %d)  steps: %s" % (
                name, len(rs), toks, 1000.0 * toks / ms, ms / max(1, steps), toks / max(1, steps), 100.0 * (acc - la) / max(1, off - lo), off - lo,
                100.0 * la / max(1, lo), lo, " ".join("%s %d" % (k, v) for k, v in kinds.items() if v)))
        line("all", solo)
        for _, _, name in BUCKETS:
            rs = [r for r in solo if bucket(r["cells"]) == name]
            if rs:
                line(name, rs)
        res["solo"] = {"tok_s": round(1000.0 * sum(r["out"] for r in solo) / sum(r["dec_ms"] for r in solo), 1)}
    # ---- 5. batch windows
    P("")
    P("5. Batch windows (%d windows, %d slots ended, %d tokens)" % (n_win, len(slots), tok_slot))
    if n_win:
        cyc = wtotal / n_win
        P("   a window: %.2f slots, %.1f rows; %.1f ms = window %.1f + next drafts %.1f + commit %.1f + first drafts %.1f + picks %.1f; between two windows %.1f ms" % (
            sum(w["n"] * w["slots"] for w in wins) / n_win, sum(w["n"] * w["rows"] for w in wins) / n_win, cyc, wparts["window"] / n_win, wparts["drafts"] / n_win,
            wparts["commit"] / n_win, wparts["draft0"] / n_win, wparts["picks"] / n_win, sum(w["n"] * w["ms"]["between"] for w in wins) / n_win))
        P("   %.2f tokens a window = %.1f tok/s while windows run; drafts accepted %.1f%% of %d" % (
            tok_slot / n_win, 1000.0 * tok_slot / max(1e-9, wtotal), 100.0 * sum(s["accepted"] for s in slots) / max(1, sum(s["offered"] for s in slots)),
            sum(s["offered"] for s in slots)))
        lo, la = sum(x.get("look_offered", 0) for x in slots), sum(x.get("look_accepted", 0) for x in slots)
        if lo:
            P("   of the slots' drafts, lookup copies: %.1f%% stood of %d (%.1f%% of the slots' tokens); the MTP chain's: %.1f%% of %d" % (
                100.0 * la / lo, lo, 100.0 * la / max(1, tok_slot), 100.0 * (sum(x["accepted"] for x in slots) - la) / max(1, sum(x["offered"] for x in slots) - lo),
                sum(x["offered"] for x in slots) - lo))
        gw = [w for w in wins if "graphs" in w]
        if gw:
            fr = [w["vram_free_mib"] for w in gw]
            P("   window graphs: %d kept at the end, %d recorded, %d dropped; the card's free memory %.0f MiB at first, %.0f at the end, %.0f at its lowest" % (
                gw[-1]["graphs"], gw[-1]["graphs_recorded"], gw[-1]["graphs_dropped"], fr[0], fr[-1], min(fr)))
            res["graphs"] = {"kept": gw[-1]["graphs"], "recorded": gw[-1]["graphs_recorded"], "dropped": gw[-1]["graphs_dropped"],
                             "vram_free_mib": [round(fr[0]), round(fr[-1]), round(min(fr))]}
        by = {}
        for w in wins:
            by.setdefault(bucket(w["cells_max"]), []).append(w)
        for _, _, name in BUCKETS:
            ws = by.get(name)
            if ws:
                n = sum(w["n"] for w in ws)
                P("   deepest session %-12s %6d windows: %.2f slots, %.1f rows, window %.1f ms, next drafts %.1f ms" % (
                    name, n, sum(w["n"] * w["slots"] for w in ws) / n, sum(w["n"] * w["rows"] for w in ws) / n, sum(w["n"] * w["ms"]["window"] for w in ws) / n,
                    sum(w["n"] * w["ms"]["drafts"] for w in ws) / n))
        res["batch"] = {"windows": n_win, "cycle_ms": round(cyc, 2), "tok_per_window": round(tok_slot / n_win, 2)}
        if lo:
            res["batch"]["lookup"] = {"offered": lo, "accepted": la}
    # ---- decode by the streams decoding at once: a request alone, then the windows by their slots
    by = {}
    for w in wins:
        for k, v in (w.get("by_slots") or {}).items():
            a = by.setdefault(int(k), [0, 0.0, 0, 0])
            a[0] += v[0]; a[1] += v[1]; a[2] += v[2]; a[3] += v[3]
    if by or solo:
        P("")
        P("   Decode by the streams decoding at once (a window's own time: drafts, verify rows, commit)")
        P("   streams   windows   time   rows a window   ms a window   tokens a window   tok/s in all   tok/s a stream")
        curve = {}
        if solo:
            toks, ms = sum(r["out"] for r in solo), sum(r["dec_ms"] for r in solo)
            steps = sum(sum(r["steps"].values()) for r in solo)
            curve[1] = 1000.0 * toks / ms
            P("   %2d alone %8d %5.0f s   %13s   %11.1f   %15.2f   %12.1f   %14.1f" % (1, steps, ms / 1000.0, "-", ms / max(1, steps), toks / max(1, steps), curve[1], curve[1]))
        for k in sorted(by):
            n, ms, tok, rows = by[k]
            if n == 0 or ms <= 0:
                continue
            rate = 1000.0 * tok / ms
            if k > 1 or 1 not in curve:
                curve[k] = rate
            P("   %2d       %8d %5.0f s   %13.1f   %11.1f   %15.2f   %12.1f   %14.1f" % (k, n, ms / 1000.0, rows / n, ms / n, tok / n, rate, rate / k))
        if curve:
            # (a count seen in a handful of windows is not a measurement: the peak is taken over those with 200 and more)
            solid = {k: v for k, v in curve.items() if k == 1 or by.get(k, [0])[0] >= 200} or curve
            peak = max(solid, key=lambda k: solid[k])
            P("   the summed decode peaks at %d streams: %.1f tok/s (%.1fx a request alone)" % (peak, solid[peak], solid[peak] / curve[1] if 1 in curve else 0.0))
            res["decode_by_streams"] = {str(k): round(v, 1) for k, v in sorted(curve.items())}
            res["decode_peak"] = {"streams": peak, "tok_s": round(solid[peak], 1)}
    # ---- 6. the server's view
    hist = history(metrics)
    if hist:
        P("")
        P("6. The server's view (%d finished requests)" % len(hist))
        wait = [max(0.0, h["first_engine_token_s"] - (h.get("prompt_ms") or 0.0) / 1000.0) for h in hist if h.get("first_engine_token_s") is not None]
        ft = [h["first_token_s"] for h in hist if h.get("first_token_s") is not None]
        du = [h["duration_s"] for h in hist if h.get("duration_s") is not None]
        if wait:
            P("   before the engine began on a request (queue, a free slot, tokenizing): median %.2f s, nine in ten under %.2f s, in all %.0f s" % (pct(wait, 50), pct(wait, 90), sum(wait)))
        if ft:
            P("   time to the first token: median %.2f s, nine in ten under %.2f s" % (pct(ft, 50), pct(ft, 90)))
        if du:
            P("   a request: median %.1f s, nine in ten under %.1f s; output %d tokens a request (median)" % (pct(du, 50), pct(du, 90), pct([h.get("output_tokens") or 0 for h in hist], 50)))
        fin = {}
        for h in hist:
            fin[h.get("finish")] = fin.get(h.get("finish"), 0) + 1
        P("   how they ended: " + ", ".join("%s %d" % (k, v) for k, v in sorted(fin.items(), key=lambda kv: -kv[1])))
        res["server"] = {"wait_p50": round(pct(wait, 50), 2) if wait else None, "first_token_p50": round(pct(ft, 50), 2) if ft else None}
    # ---- 7. the slowest requests
    P("")
    P("7. The slowest requests on the control lines")
    for r in sorted(reqs, key=lambda r: -r["total_ms"])[:6]:
        P("   #%d %-4s %6.1f s: prompt %d, kept %d (%s), read %d in %.1f s, session work %.1f s (parked %.1f, back %.1f), wrote %d in %.1f s, %d cells, %d slots busy" % (
            r["id"], r["kind"], r["total_ms"] / 1000.0, r["prompt"], r["kept"], r["how"], r["read"], r["read_ms"] / 1000.0, r["sess_ms"] / 1000.0,
            r["park_ms"] / 1000.0, r["unpark_ms"] / 1000.0, r["out"], r["dec_ms"] / 1000.0, r["cells"], r["slots_busy"]))
    return "\n".join(out), res


def harness_turns(d):
    """The model turns the harness recorded (agent-<i>/events-<task>.jsonl, pi's --mode json): when the request went out, its
    first and last streamed piece, the characters and the time by kind (thinking, toolcall, text), its usage, its tool's time."""
    res = []
    for f in sorted(glob.glob(os.path.join(d, "agent-*", "events-*.jsonl"))):
        agent = os.path.basename(os.path.dirname(f))
        task = os.path.basename(f)[7:-6]
        cur = None
        last_t = sent = None

        def close():
            nonlocal cur
            if cur is not None:
                cur["end"] = last_t
                res.append(cur)
                cur = None
        for l in open(f, errors="replace"):
            try:
                j = json.loads(l)
            except ValueError:
                continue
            t, line = j.get("t", 0.0), j.get("line", "")
            try:
                e = json.loads(line)
            except ValueError:   # (a line cut by an older recorder: its type is still there)
                e = {"type": line[9:line.index('"', 9)]} if line.startswith('{"type":"') and '"' in line[9:] else {}
            ty = e.get("type")
            if ty == "turn_start":
                close()
                sent = t
            elif ty == "message_update":
                ame = e.get("assistantMessageEvent") or {}
                k = ame.get("type", "")
                if cur is None:
                    cur = {"agent": agent, "task": task, "sent": sent if sent is not None else t, "first": t, "chars": {}, "time": {}, "tool_s": 0.0, "usage": {}}
                    last_t = t
                kind = k.split("_")[0]
                if k.endswith("_delta"):
                    cur["chars"][kind] = cur["chars"].get(kind, 0) + len(ame.get("delta") or "")
                    cur["time"][kind] = cur["time"].get(kind, 0.0) + (t - last_t)
                last_t = t
                if k.endswith("_end") and e.get("usage"):
                    cur["usage"] = e["usage"]
            elif ty == "tool_execution_end" and cur is not None:
                cur["tool_s"] = t - last_t
            elif ty in ("agent_settled", "agent_end"):
                close()
        close()
    return res


def threads(d, prof):
    """Section 8: the harness's record joined to the engine's, a turn to its requests (by time and prompt size)."""
    T = sorted(harness_turns(d), key=lambda t: t["sent"])
    reqs = sorted((dict(r) for r in prof if r.get("ev") == "req" and "ts" in r), key=lambda r: r["ts"] - r["total_ms"] / 1000.0)
    if not T or not reqs:
        return "", {}
    for r in reqs:
        r["start"] = r["ts"] - r["total_ms"] / 1000.0
        r["used"] = False
    for t in T:
        u = t["usage"]
        t["P"], t["out"], t["reqs"] = u.get("input", 0) + u.get("cacheRead", 0), u.get("output", 0), []
        for r in reqs:
            if r["used"] or r["start"] < t["sent"] - 0.3 or r["start"] > t["end"] + 0.3:
                continue
            if abs(r["prompt"] - t["P"]) <= 2:
                r["used"] = True
                t["reqs"].append(r)
                if r["finish"] != "cancel":   # (a request alone that the server moved into a slot: the admission follows)
                    break
    J = [t for t in T if t["reqs"]]
    out = []
    P = out.append
    P("")
    P("8. The agents' threads (the harness's record joined to the engine's: %d of %d turns)" % (len(J), len(T)))
    if not J:
        return "\n".join(out), {}
    dur = sum(t["end"] - t["sent"] for t in J)
    ttft = sum(t["first"] - t["sent"] for t in J)
    wait = sum(max(0.0, t["reqs"][0]["start"] - t["sent"]) for t in J)
    read = sum(sum(r["read_ms"] for r in t["reqs"]) for t in J) / 1000.0
    sess = sum(sum(r["sess_ms"] for r in t["reqs"]) for t in J) / 1000.0
    P("   a turn's time: %.1f%% before its first piece (%.1f%% waiting for a session, %.1f%% its prompt read, %.1f%% session work), %.1f%% in the agent's tool" % (
        100.0 * ttft / dur, 100.0 * wait / dur, 100.0 * read / dur, 100.0 * sess / dur, 100.0 * sum(t["tool_s"] for t in J) / dur))
    ch, tm = {}, {}
    for t in J:
        for k, v in t["chars"].items():
            ch[k] = ch.get(k, 0) + v
        for k, v in t["time"].items():
            tm[k] = tm.get(k, 0.0) + v
    if sum(ch.values()) > 0:
        P("   what the model wrote: " + "; ".join("%s %.1f%% of the characters, %.1f%% of the time (%.0f chars/s)" % (
            k, 100.0 * ch[k] / sum(ch.values()), 100.0 * tm.get(k, 0.0) / max(1e-9, sum(tm.values())), ch[k] / max(1e-9, tm.get(k, 0.0)))
            for k in sorted(ch, key=lambda k: -ch[k])))
    # the cells a thread held when its last turn ended (its prompt and what it wrote) that its next turn read again
    last, hows = {}, {}
    for t in J:
        key, r0 = (t["agent"], t["task"]), t["reqs"][0]
        if key in last:
            had = last[key]["P"] + last[key]["out"]
            again = max(0, min(had, r0["prompt"]) - r0["kept"])
            h = hows.setdefault(r0["how"], [0, 0, 0, 0.0])
            h[0] += 1
            h[1] += again
            h[2] += r0["read"]
            h[3] += r0["read_ms"] * again / max(1, r0["read"]) / 1000.0
        last[key] = t
    all_read = sum(r["read_ms"] for r in reqs) / 1000.0
    again_s = sum(h[3] for h in hows.values())
    P("   later turns, the cells their thread held that they read again: %d tokens, %.1f s = %.1f%% of all the time reading prompts" % (
        sum(h[1] for h in hows.values()), again_s, 100.0 * again_s / max(1e-9, all_read)))
    for k in sorted(hows, key=lambda k: -hows[k][1]):
        h = hows[k]
        P("     %-10s %4d turns: %7d tokens read again (%.0f%% of what they read)" % (k, h[0], h[1], 100.0 * h[1] / max(1, h[2])))
    div = [r for r in reqs if r.get("div", 0) > 0]
    if div:
        P("   checkpoints left where a prompt parts from another thread's: %d (at cell %s)" % (len(div), ", ".join(str(r["div"]) for r in div[:6])))
    return "\n".join(out), {"turns": len(T), "joined": len(J), "before_first_pct": round(100.0 * ttft / dur, 1), "waiting_pct": round(100.0 * wait / dur, 1),
                             "read_again_tokens": sum(h[1] for h in hows.values()), "read_again_s": round(again_s, 1),
                             "read_again_pct_of_reads": round(100.0 * again_s / max(1e-9, all_read), 1),
                             "read_again_by_how": {k: h[1] for k, h in hows.items()}, "div_checkpoints": len(div)}


def in_run(path, prof):
    """Section 9: the figures of switches that alternate inside a run (STRATA_Q36_MTP_CHAIN_GRAPH=ab, STRATA_Q36_PARK_PIN=ab),
    and the draft block's passes by kind."""
    out, res = [], {}
    dr = {"first": [0, 0.0, 0], "chain": [0, 0.0, 0], "chain_g": [0, 0.0, 0]}
    for w in prof:
        if w.get("ev") == "win":
            for k, v in (w.get("dr") or {}).items():
                if k in dr:
                    for i in range(3):
                        dr[k][i] += v[i]
    if any(v[0] for v in dr.values()):
        out.append("   the draft block's passes: " + "; ".join("%s %d calls, %.2f ms each, %.1f rows" % (
            {"first": "a window's kept rows", "chain": "a chained level, kernel by kernel", "chain_g": "a chained level from a graph"}[k], v[0], v[1] / v[0], v[2] / v[0])
            for k, v in dr.items() if v[0]))
        res["draft_passes"] = {k: {"calls": v[0], "ms": round(v[1] / v[0], 3), "rows": round(v[2] / v[0], 2)} for k, v in dr.items() if v[0]}
    # the two kinds of 64-window blocks (a switch that alternates: STRATA_Q36_BATCH_BASE_AB, ..._CHAIN_GRAPH=ab)
    ab = {"0": [0.0] * 8, "1": [0.0] * 8}
    for w in prof:
        if w.get("ev") == "win":
            for k, v in (w.get("ab") or {}).items():
                if k in ab:
                    for i in range(8):
                        ab[k][i] += v[i]
    if ab["1"][0] > 0:
        for k, name in (("0", "blocks as set"), ("1", "the other blocks")):
            v = ab[k]
            if v[0] > 0:
                out.append("   %-16s %5d windows: %.1f rows, %.2f tokens in %.1f ms = %.1f tok/s; with every slot decoding %d windows: %.1f rows, %.2f tokens in %.1f ms = %.1f tok/s" % (
                    name, v[0], v[3] / v[0], v[2] / v[0], v[1] / v[0], 1000.0 * v[2] / max(1e-9, v[1]),
                    v[4], v[7] / max(1, v[4]), v[6] / max(1, v[4]), v[5] / max(1, v[4]), 1000.0 * v[6] / max(1e-9, v[5])))
        res["ab"] = {k: {"windows": int(v[0]), "tok_s": round(1000.0 * v[2] / max(1e-9, v[1]), 1), "full_windows": int(v[4]),
                         "full_tok_s": round(1000.0 * v[6] / max(1e-9, v[5]), 1), "full_rows": round(v[7] / max(1, v[4]), 1),
                         "full_tokens": round(v[6] / max(1, v[4]), 2), "full_ms": round(v[5] / max(1, v[4]), 1)} for k, v in ab.items() if v[0] > 0}
    # a sampled window's parts (STRATA_Q36_PROFILE_PHASES=1)
    wp = [w for w in prof if w.get("ev") == "wphase"]
    if wp:
        tot = {}
        for w in wp:
            for k, v in w["ms"].items():
                tot[k] = tot.get(k, 0.0) + v
        allms = sum(tot.values())
        out.append("   a window's parts (%d sampled windows run with a wait after each part, %.1f rows, %.1f ms each): %s" % (
            len(wp), st.mean(w["rows"] for w in wp), allms / len(wp),
            ", ".join("%s %.1f%% (%.1f ms)" % (k, 100.0 * v / allms, v / len(wp)) for k, v in sorted(tot.items(), key=lambda kv: -kv[1]))))
        res["window_parts_ms"] = {k: round(v / len(wp), 2) for k, v in tot.items()}
    log = os.path.join(path, "engine.log") if os.path.isdir(path) else path
    park = {}
    if os.path.exists(log):
        for line in open(log, errors="replace"):
            m = re.search(r"the cells left the card at ([\d.]+) GiB/s \((pinned staging|direct), ([\d.]+) ms\)", line)
            if m:
                e = park.setdefault(m.group(2), [0, 0.0, 0.0])
                e[0] += 1
                e[1] += float(m.group(3))
                e[2] += float(m.group(1)) * float(m.group(3)) / 1000.0   # GiB
    if park:
        out.append("   parked cells leaving the card: " + "; ".join("%s %d parks, %.2f GiB/s (%.0f ms each)" % (k, v[0], v[2] / max(1e-9, v[1] / 1000.0), v[1] / v[0])
                                                              for k, v in sorted(park.items())))
        res["park_copy"] = {k: {"parks": v[0], "gib_s": round(v[2] / max(1e-9, v[1] / 1000.0), 2), "ms": round(v[1] / v[0], 1)} for k, v in park.items()}
    if not out:
        return "", {}
    return "\n".join(["", "9. Inside the run (the draft block's passes; switches that alternate)"] + out), res


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    prof, metrics, run = load(sys.argv[1])
    text, res = report(prof, metrics, run)
    t9, r9 = in_run(sys.argv[1], prof)
    if os.path.isdir(sys.argv[1]):
        t8, r8 = threads(sys.argv[1], prof)
        text += ("\n" + t8) if t8 else ""
        if r8:
            res["threads"] = r8
    text += ("\n" + t9) if t9 else ""
    if r9:
        res["in_run"] = r9
    print(text)
    if "--json" in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
