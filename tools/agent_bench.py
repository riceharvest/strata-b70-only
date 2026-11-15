#!/usr/bin/env python3
"""agent_bench.py: real coding agents on real tasks against an OpenAI-compatible server, for tools/agent_profile.py.

Nothing here is synthetic: each agent is a harness process (pi by default) that reads files, runs commands and edits a
copy of a small project until its task is done, and the server sees what that harness sends: its system prompt, its
tools, the tool results, the turns.

  agent_bench.py --url http://127.0.0.1:8095 --model <id> --out <run folder> [--agents 4] [--rounds 1]
                 [--tasks tools/agent_tasks/mixed.json] [--project-files a,b,...] [--timeout 900] [--stagger 2]
                 [--harness pi | --harness-cmd "<command with {task}>"] [--thinking medium] [--sandbox bwrap|none]
                 [--engine-log <file>]

--agents N      N harness processes at once; agent i takes tasks i, i + N, ... of the list, --rounds of them, one after
                another (so the load is continuous while agents come and go between their tool calls).  Past the end
                of the list it starts over, each time in a fresh working copy.
--project-files the files (relative to the repository) each agent's working copy starts with.
--harness pi    `pi -p --mode json` with its own config folder (a provider that points at --url), no session, no
                extensions: the user's pi configuration is not read or changed.
--harness-cmd   any other harness: a shell command, `{task}` is the task text (quoted), run in the working copy with
                OPENAI_BASE_URL / OPENAI_API_BASE and OPENAI_MODEL set.
--sandbox bwrap the harness sees the machine read-only and can write only its own folder (bubblewrap; the network stays,
                for the server).  none: no sandbox.
--engine-log    copied into the run folder at the end (its `PROF` lines: STRATA_Q36_PROFILE=1).

The run folder gets: run.json (settings, per agent and task: time, exit code, tool calls seen), agent-<i>/ (working copy,
events-<task>.jsonl: the harness's output, a time stamp a line), metrics.json (the server's GET /metrics), engine.log.
"""
import argparse, json, os, shlex, shutil, subprocess, sys, threading, time, urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FILES = "tools/b70_pack.py,tools/test_b70_pack.py,tools/gguf_reader.py,docs/B70_LIB.md"


def parse():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--url", required=True)
    a.add_argument("--model", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--agents", type=int, default=4)
    a.add_argument("--rounds", type=int, default=1)
    a.add_argument("--tasks", default=os.path.join(REPO, "tools", "agent_tasks", "mixed.json"))
    a.add_argument("--project-files", default=DEFAULT_FILES)
    a.add_argument("--timeout", type=float, default=900.0)
    a.add_argument("--stagger", type=float, default=2.0)
    a.add_argument("--harness", default="pi")
    a.add_argument("--harness-cmd", default="")
    a.add_argument("--thinking", default="")
    a.add_argument("--context", type=int, default=131072)
    a.add_argument("--max-tokens", type=int, default=8192)
    a.add_argument("--sandbox", default="bwrap", choices=("bwrap", "none"))
    a.add_argument("--engine-log", default="")
    return a.parse_args()


def make_copy(root, files):
    work = os.path.join(root, "work")
    os.makedirs(work, exist_ok=True)
    for f in files:
        dst = os.path.join(work, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(REPO, f), dst)
    return work


def pi_command(args, root, task):
    cfg = os.path.join(root, "pi")
    os.makedirs(cfg, exist_ok=True)
    with open(os.path.join(cfg, "models.json"), "w") as f:
        json.dump({"providers": {"bench": {"baseUrl": args.url.rstrip("/") + "/v1", "api": "openai-completions", "apiKey": "none",
                                           "models": [{"id": args.model, "name": args.model, "contextWindow": args.context,
                                                       "maxTokens": args.max_tokens, "reasoning": True}]}}}, f)
    cmd = ["pi", "-p", "--mode", "json", "--no-session", "--offline", "--no-extensions", "--no-skills", "--no-mcp", "--no-context-files",
           "--no-prompt-templates", "--no-themes", "--provider", "bench", "--model", args.model]
    if args.thinking:
        cmd += ["--thinking", args.thinking]
    return cmd + [task], {"PI_CODING_AGENT_DIR": cfg, "PI_OFFLINE": "1"}


def wrap(args, root, work, cmd, env):
    if args.sandbox != "bwrap":
        return cmd
    b = ["bwrap", "--die-with-parent", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--bind", root, root, "--chdir", work]
    for k, v in env.items():
        b += ["--setenv", k, v]
    return b + cmd


def run_task(args, i, root, work, task, rec):
    if args.harness_cmd:
        cmd = ["bash", "-c", args.harness_cmd.replace("{task}", shlex.quote(task["prompt"]))]
        env = {"OPENAI_BASE_URL": args.url.rstrip("/") + "/v1", "OPENAI_API_BASE": args.url.rstrip("/") + "/v1", "OPENAI_MODEL": args.model,
               "OPENAI_API_KEY": "none"}
    else:
        cmd, env = pi_command(args, root, task["prompt"])
    full = wrap(args, root, work, cmd, env)
    ev_path = os.path.join(root, "events-%s.jsonl" % task["id"])
    t0 = time.time()
    tools = 0
    lines = 0
    code = None
    try:
        # (no standard input: a harness that is not on a terminal would wait for a task on it)
        p = subprocess.Popen(full, cwd=work, env=dict(os.environ, **env), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, errors="replace")
        timer = threading.Timer(args.timeout, p.kill)
        timer.start()
        with open(ev_path, "w") as ev:
            for line in p.stdout:
                lines += 1
                ev.write(json.dumps({"t": round(time.time(), 3), "line": line.rstrip("\n")[:20000]}) + "\n")
                if '"tool' in line and ('"toolCall"' in line or '"tool_execution_start"' in line or '"tool_call"' in line or '"type":"tool_use"' in line):
                    tools += 1
        code = p.wait()
        timer.cancel()
    except FileNotFoundError as e:
        code = "not found: %s" % e
    rec.append({"agent": i, "task": task["id"], "kind": task.get("kind"), "start": round(t0, 3), "seconds": round(time.time() - t0, 1), "exit": code,
                "timed_out": time.time() - t0 >= args.timeout - 0.5, "lines": lines, "tool_lines": tools})


def agent(args, i, tasks, files, rec):
    root = os.path.join(os.path.abspath(args.out), "agent-%d" % i)
    os.makedirs(root, exist_ok=True)
    work = make_copy(root, files)
    for r in range(args.rounds):
        k = i + r * args.agents
        t = dict(tasks[k % len(tasks)])   # (more agents than tasks: the list starts over, each turn of it in a fresh working copy)
        if k >= len(tasks):
            t["id"] = "%s-%d" % (t["id"], k // len(tasks) + 1)
            work = make_copy(os.path.join(root, "again-%d" % k), files)
        run_task(args, i, root, work, t, rec)


def main():
    args = parse()
    tasks = json.load(open(args.tasks))
    files = [f for f in args.project_files.split(",") if f]
    os.makedirs(args.out, exist_ok=True)
    rec = []
    th = [threading.Thread(target=agent, args=(args, i, tasks, files, rec)) for i in range(args.agents)]
    T0 = time.time()
    for t in th:
        t.start()
        time.sleep(args.stagger)
    for t in th:
        t.join()
    wall = time.time() - T0
    try:
        with urllib.request.urlopen(args.url.rstrip("/") + "/metrics?requests=all", timeout=30) as r:   # (every finished request the server keeps)
            open(os.path.join(args.out, "metrics.json"), "wb").write(r.read())
    except Exception as e:
        print("agent_bench: no /metrics from the server (%s)" % e)
    if args.engine_log and os.path.exists(args.engine_log):
        shutil.copy2(args.engine_log, os.path.join(args.out, "engine.log"))
    json.dump({"url": args.url, "model": args.model, "harness": args.harness_cmd or args.harness, "agents": args.agents, "rounds": args.rounds,
               "start": round(T0, 3), "wall": round(wall, 1), "thinking": args.thinking, "sandbox": args.sandbox, "tasks": rec},
              open(os.path.join(args.out, "run.json"), "w"), indent=1)
    ok = sum(1 for r in rec if r["exit"] == 0)
    print("agent_bench: %d agents, %d tasks (%d ended with exit 0, %d timed out), %.1f s; the run is in %s" % (
        args.agents, len(rec), ok, sum(1 for r in rec if r["timed_out"]), wall, args.out))


if __name__ == "__main__":
    main()
