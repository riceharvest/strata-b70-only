"""agent_profile.py on a hand-made engine log, and agent_bench.py with a stand-in harness: the report adds the records up as
written, and a run folder has what the report reads."""
import http.server, json, os, subprocess, sys, tempfile, threading, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import agent_profile


def req(i, kind, how, prompt, kept, read, read_ms, out, dec_ms, steps, offered, accepted, lo=0, la=0, park=0.0, unpark=0.0, cells=None, t=None):
    return {"ev": "req", "id": i, "t": t if t is not None else 1000.0 * i, "ts": 1.0, "kind": kind, "sess": 0, "prompt": prompt, "kept": kept, "how": how,
            "read": read, "read_ms": read_ms, "sess_ms": park + unpark + 1.0, "park_ms": park, "unpark_ms": unpark, "to_drive": 0, "from_drive": 0, "out": out,
            "dec_ms": dec_ms, "steps": steps, "offered": offered, "accepted": accepted, "look_offered": lo, "look_accepted": la, "finish": "stop",
            "cells": cells if cells is not None else prompt + out, "slots_busy": 0, "total_ms": read_ms + dec_ms + park + unpark + 1.0}


class Profile(unittest.TestCase):
    def test_report_adds_the_records_up(self):
        recs = [
            req(1, "solo", "none", 40000, 0, 40000, 20000.0, 300, 3000.0, {"tree": 80, "chain": 10, "lookup": 10, "drafter": 0}, 400, 200, lo=40, la=38),
            req(2, "solo", "ckpt", 41000, 40000, 1000, 500.0, 100, 1000.0, {"tree": 40, "chain": 0, "lookup": 0, "drafter": 0}, 150, 60),
            req(3, "adm", "unpark", 42000, 41000, 1000, 600.0, 1, 30.0, {"tree": 0, "chain": 0, "lookup": 0, "drafter": 0}, 0, 0, park=700.0, unpark=100.0),
            {"ev": "win", "t": 9000.0, "ts": 1.0, "n": 64, "slots": 2.0, "rows": 8.0, "cells_max": 42000,
             "ms": {"draft0": 0.1, "window": 40.0, "picks": 0.1, "commit": 2.0, "drafts": 3.0, "between": 5.0},
             "graphs": 40, "graphs_recorded": 55, "graphs_dropped": 15, "vram_free_mib": 2500.0,
             "by_slots": {"2": [300, 13500.0, 1500, 2400], "4": [250, 12500.0, 2500, 4000], "3": [10, 400.0, 100, 120]}},
            {"ev": "slot", "t": 9500.0, "ts": 1.0, "slot": 1, "out": 320, "ms": 2900.0, "offered": 300, "accepted": 150, "look_offered": 100, "look_accepted": 90,
             "finish": "stop", "cells": 42320},
        ]
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "engine.log"), "w") as f:
                f.write("strata-q36: something else\n")
                for r in recs:
                    f.write("PROF " + json.dumps(r) + "\n")
                f.write("PROF {broken\n")
            json.dump({"history": [{"first_engine_token_s": 21.0, "prompt_ms": 20000.0, "first_token_s": 21.2, "duration_s": 24.0, "output_tokens": 300, "finish": "stop"}]},
                      open(os.path.join(d, "metrics.json"), "w"))
            prof, metrics, run = agent_profile.load(d)
            text, res = agent_profile.report(prof, metrics, run)
        self.assertEqual(len(prof), 5)                                   # (the broken line is skipped)
        self.assertEqual(res["run"]["requests"], 3)
        self.assertEqual(res["run"]["output_tokens"], 300 + 100 + 320)   # the tokens written alone and the slot's
        self.assertEqual(res["cache"]["how"], {"none": 1, "ckpt": 1, "unpark": 1})
        self.assertEqual(res["cache"]["kept_pct"], round(100.0 * 81000 / 123000, 1))
        self.assertEqual(res["cache"]["lost"], 1)                        # the first request read all of its 40,000 tokens
        self.assertEqual(res["time_s"]["decoding alone"], 4.0)
        self.assertEqual(res["time_s"]["batch windows"], round(64 * 45.2 / 1000.0, 2))
        self.assertEqual(res["time_s"]["parking a cache"], 0.7)
        self.assertEqual(res["solo"]["tok_s"], 100.0)                    # 400 tokens in 4 s
        self.assertEqual(res["batch"], {"windows": 64, "cycle_ms": 45.2, "tok_per_window": 5.0, "lookup": {"offered": 100, "accepted": 90}})
        self.assertIn("lookup copies: 90.0% stood of 100 (28.1% of the slots' tokens); the MTP chain's: 30.0% of 200", text)
        self.assertIn("drafts 43.5% of 510 (lookups 95.0% of 40)", text)   # the lookups are counted apart from the model's drafts
        # the decode by streams: alone 100 tok/s, two streams 111.1, four 200; three were seen in 10 windows only, so no peak there
        self.assertEqual(res["decode_by_streams"], {"1": 100.0, "2": 111.1, "3": 250.0, "4": 200.0})
        self.assertEqual(res["decode_peak"], {"streams": 4, "tok_s": 200.0})
        self.assertIn("the summed decode peaks at 4 streams: 200.0 tok/s (2.0x a request alone)", text)
        self.assertEqual(res["graphs"], {"kept": 40, "recorded": 55, "dropped": 15, "vram_free_mib": [2500, 2500, 2500]})
        self.assertIn("before the engine began on a request", text)
        self.assertIn("unpark", text)

    def test_an_engine_log_alone(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "e.log")
            open(p, "w").write("PROF " + json.dumps(req(1, "solo", "live", 100, 99, 1, 0.0, 50, 500.0, {"tree": 10, "chain": 0, "lookup": 0, "drafter": 0}, 40, 30)) + "\n")
            text, res = agent_profile.report(*agent_profile.load(p))
        self.assertEqual(res["run"]["output_tokens"], 50)
        self.assertNotIn("The server's view", text)


    def test_the_harness_record_is_joined_to_the_engines(self):
        # one agent, two turns: the second finds its 100-token prompt and 40-token answer gone but for 100 cells (a checkpoint)
        def ev(t, e):
            return json.dumps({"t": t, "line": json.dumps(e)}) + "\n"

        def upd(t, kind, delta=None, usage=None):
            a = {"type": kind}
            if delta is not None:
                a["delta"] = delta
            e = {"type": "message_update", "assistantMessageEvent": a}
            if usage:
                e["usage"] = usage
            return ev(t, e)
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "agent-0"))
            with open(os.path.join(d, "agent-0", "events-a.jsonl"), "w") as f:
                f.write(ev(10.0, {"type": "turn_start"}))
                f.write(upd(11.0, "thinking_delta", "abcd"))
                f.write(upd(13.0, "thinking_delta", "efgh"))
                f.write(upd(14.0, "toolcall_delta", "xy"))
                f.write(upd(14.0, "toolcall_end", usage={"input": 100, "cacheRead": 0, "output": 40}))
                f.write('{"t": 14.5, "line": "{\\"type\\":\\"tool_execution_end\\",\\"cut"}\n')   # (an older recorder cut long lines)
                f.write(ev(15.0, {"type": "turn_start"}))
                f.write(upd(17.0, "text_delta", "done"))
                f.write(upd(18.0, "text_end", usage={"input": 60, "cacheRead": 100, "output": 5}))
                f.write(ev(18.0, {"type": "agent_settled"}))
            r1 = dict(req(1, "solo", "none", 100, 0, 100, 500.0, 40, 3000.0, {"tree": 4, "chain": 0, "lookup": 0, "drafter": 0}, 10, 8), ts=14.0)
            r2 = dict(req(2, "solo", "ckpt", 160, 100, 60, 1000.0, 5, 1000.0, {"tree": 1, "chain": 0, "lookup": 0, "drafter": 0}, 4, 4), ts=18.0, div=64)
            open(os.path.join(d, "engine.log"), "w").write("".join("PROF " + json.dumps(r) + "\n" for r in (r1, r2)))
            text, res = agent_profile.threads(d, agent_profile.load(d)[0])
        self.assertEqual((res["turns"], res["joined"]), (2, 2))
        self.assertEqual(res["read_again_tokens"], 40)            # (140 cells held, 100 kept)
        self.assertEqual(res["read_again_by_how"], {"ckpt": 40})
        self.assertAlmostEqual(res["read_again_s"], 1.0 * 40 / 60, places=1)
        self.assertEqual(res["div_checkpoints"], 1)
        self.assertIn("thinking 57.1% of the characters", text)   # (8 of 14 characters)
        self.assertIn("in the agent's tool", text)

    def test_the_figures_of_switches_that_alternate_inside_a_run(self):
        with tempfile.TemporaryDirectory() as d:
            w = {"ev": "win", "dr": {"first": [2, 9.0, 40], "chain": [4, 10.0, 32], "chain_g": [4, 6.0, 32]},
                 "ab": {"0": [10, 800.0, 200, 320, 5, 400.0, 110, 160], "1": [10, 700.0, 190, 280, 5, 350.0, 100, 140]}}
            ph = {"ev": "wphase", "slots": 8, "rows": 32, "cells_max": 20000, "ms": {"proj": 30.0, "gdn": 10.0, "attn": 20.0, "norm": 5.0, "ffn": 25.0, "embed": 1.0, "head": 9.0}}
            open(os.path.join(d, "engine.log"), "w").write(
                "PROF " + json.dumps(w) + "\n" + "PROF " + json.dumps(ph) + "\n"
                "strata serve: the cells left the card at 3.00 GiB/s (direct, 100 ms)\n"
                "strata serve: the cells left the card at 6.00 GiB/s (pinned staging, 50 ms)\n"
                "strata serve: the cells left the card at 1.00 GiB/s (direct, 100 ms)\n")
            text, res = agent_profile.in_run(d, agent_profile.load(d)[0])
        self.assertEqual(res["draft_passes"]["chain"], {"calls": 4, "ms": 2.5, "rows": 8.0})
        self.assertEqual(res["draft_passes"]["chain_g"]["ms"], 1.5)
        self.assertEqual(res["park_copy"]["direct"], {"parks": 2, "gib_s": 2.0, "ms": 100.0})   # (0.4 GiB in 0.2 s)
        self.assertEqual(res["park_copy"]["pinned staging"]["gib_s"], 6.0)
        self.assertIn("a chained level from a graph 4 calls, 1.50 ms each, 8.0 rows", text)
        self.assertEqual(res["ab"]["0"]["tok_s"], 250.0)
        self.assertEqual(res["ab"]["1"]["full_tok_s"], 285.7)
        self.assertEqual(res["window_parts_ms"]["proj"], 30.0)
        self.assertIn("proj 30.0% (30.0 ms)", text)

    def test_a_long_harness_line_stays_a_record(self):
        import agent_bench
        line = json.dumps({"type": "turn_end", "message": {"content": [{"type": "text", "text": "x" * 50000}]}, "n": 3})
        kept = json.loads(agent_bench.keep_line(line))
        self.assertEqual((kept["type"], kept["n"]), ("turn_end", 3))
        self.assertTrue(kept["message"]["content"][0]["text"].endswith("...[+48000 chars]"))
        self.assertEqual(agent_bench.keep_line("short"), "short")
        self.assertEqual(len(agent_bench.keep_line("y" * 30000)), 20000)   # (not JSON: cut as before)


class Bench(unittest.TestCase):
    def test_a_run_folder(self):
        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"history": []}).encode())
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        with tempfile.TemporaryDirectory() as d:
            tasks = os.path.join(d, "tasks.json")
            json.dump([{"id": "a", "kind": "k", "prompt": "say it's done"}, {"id": "b", "kind": "k", "prompt": "two"}, {"id": "c", "kind": "k", "prompt": "three"}], open(tasks, "w"))
            log = os.path.join(d, "engine-src.log")
            open(log, "w").write("PROF {}\n")
            out = os.path.join(d, "run")
            r = subprocess.run([sys.executable, os.path.join(HERE, "agent_bench.py"), "--url", "http://127.0.0.1:%d" % srv.server_address[1], "--model", "m", "--out", out,
                                "--agents", "2", "--rounds", "2", "--tasks", tasks, "--stagger", "0", "--sandbox", "none", "--engine-log", log,
                                "--harness-cmd", "echo task={task} url=$OPENAI_BASE_URL; ls tools; echo '{\"type\":\"tool_execution_start\"}'"],
                               capture_output=True, text=True, timeout=120)
            srv.shutdown()
            self.assertEqual(r.returncode, 0, r.stderr)
            run = json.load(open(os.path.join(out, "run.json")))
            # agent 0 takes tasks a and c, agent 1 task b and then the list again from its start, in a fresh working copy
            self.assertEqual(sorted((t["agent"], t["task"]) for t in run["tasks"]), [(0, "a"), (0, "c"), (1, "a-2"), (1, "b")])
            self.assertTrue(os.path.exists(os.path.join(out, "agent-1", "again-3", "work", "tools", "b70_pack.py")))
            self.assertTrue(all(t["exit"] == 0 and not t["timed_out"] and t["tool_lines"] == 1 for t in run["tasks"]))
            ev = [json.loads(l)["line"] for l in open(os.path.join(out, "agent-0", "events-a.jsonl"))]
            self.assertIn("task=say it's done url=http://127.0.0.1:%d/v1" % srv.server_address[1], ev[0])
            self.assertIn("b70_pack.py", "\n".join(ev))                 # the working copy has the project's files
            self.assertTrue(os.path.exists(os.path.join(out, "metrics.json")) and os.path.exists(os.path.join(out, "engine.log")))


if __name__ == "__main__":
    unittest.main()
