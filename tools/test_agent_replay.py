"""agent_replay.py against a fake OpenAI-compatible server: the turns it sends are the recorded history up to each assistant
message, under the recorded system prompt and tools, and its report counts them."""
import http.server, json, os, subprocess, sys, tempfile, threading, unittest

HERE = os.path.dirname(os.path.abspath(__file__))


class Fake(http.server.BaseHTTPRequestHandler):
    seen = []
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with Fake.lock:
            Fake.seen.append(body)
        n = sum(len(json.dumps(m)) for m in body["messages"])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in ("ok", " done"):
            self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}) + "\n\n").encode())
        usage = {"prompt_tokens": n, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": n // 2}}
        self.wfile.write(("data: " + json.dumps({"choices": [], "usage": usage}) + "\n\ndata: [DONE]\n\n").encode())


def session(n_turns, tag):
    rows = [{"role": "user", "content": "task " + tag}]
    for k in range(n_turns):
        rows.append({"role": "assistant", "content": "", "reasoning": "hidden", "tool_calls": [
            {"id": "c%d" % k, "type": "function", "function": {"name": "read", "arguments": json.dumps({"path": "f%d" % k})}}]})
        rows.append({"role": "tool", "tool_call_id": "c%d" % k, "content": "result %d of %s" % (k, tag)})
    return rows


class Replay(unittest.TestCase):
    def test_threads_turns_and_report(self):
        Fake.seen = []
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "s"))
            for tag in ("a", "b"):
                with open(os.path.join(d, "s", tag + ".jsonl"), "w") as f:
                    for r in session(4, tag):
                        f.write(json.dumps(r) + "\n")
            tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
            for i, sysmsg in enumerate(("You are agent one.", "You are agent two.")):
                json.dump({"request": {"body": {"messages": [{"role": "system", "content": sysmsg}], "tools": tools}}},
                          open(os.path.join(d, "req%d.json" % i), "w"))
            out = subprocess.run([sys.executable, os.path.join(HERE, "agent_replay.py"), "http://127.0.0.1:%d" % srv.server_address[1], "m", "2", "3",
                                  "sessions=" + os.path.join(d, "s"), "requests=" + os.path.join(d, "req*.json"), "gap=0", "stagger=0", "out=" + d, "tag=t"],
                                 capture_output=True, text=True, timeout=60)
            srv.shutdown()
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("2 threads x up to 3 turns (2 system prompts, gap 0.0 s): 6 turns done, 0 errors", out.stdout)
            self.assertIn("output 12 tokens", out.stdout)
            self.assertIn("later turns n=  4", out.stdout)
            self.assertIn("cached  50.0%", out.stdout)
            res = json.load(open(os.path.join(d, "replay-t.json")))
            self.assertEqual([len(r) for r in res], [3, 3])
        self.assertEqual(len(Fake.seen), 6)
        for body in Fake.seen:
            m = body["messages"]
            self.assertEqual(m[0]["role"], "system")
            self.assertEqual(body["tools"], tools)
            self.assertEqual(m[-1]["role"], "user" if len(m) == 2 else "tool")   # a turn ends with what the agent got back
            self.assertTrue(all("reasoning" not in x and "timestamp" not in x for x in m))   # as a client sends a history back
        # turn k of a thread carries k earlier assistant messages (the recorded ones, with their tool calls)
        self.assertEqual(sorted(sum(1 for x in b["messages"] if x["role"] == "assistant") for b in Fake.seen), [0, 0, 1, 1, 2, 2])
        self.assertEqual({b["messages"][0]["content"] for b in Fake.seen}, {"You are agent one.", "You are agent two."})


if __name__ == "__main__":
    unittest.main()
