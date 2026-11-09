"""serve/test_runconfig.py - #564: the web page's Settings view of the run config (GET / POST /config), against the
mock engine: only the listed keys change, every other key of the file stays, a bad value changes nothing, and only
Strata's own page may write it (JSON, no foreign Origin, the API key when one is set).

    python -m unittest serve.test_runconfig
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve import runconfig
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]
CFG = {"exe": "engine/strata.exe", "args": ["--pack", "data/packs/q2_0", "--kv", "int8", "--spec-min-p", "0.5"],
       "model_name": "qwen3.8-flash-next-q2_0", "port": 8080, "host": "0.0.0.0", "api_key": "",
       "sampling": {"temperature": 0.7, "presence_penalty": 1.5},
       "mcp_servers": {"files": {"command": "npx", "args": ["-y", "server-filesystem", "."]}},
       "cors_origins": ["http://localhost:3000"]}


class Apply(unittest.TestCase):
    def test_only_the_named_keys_change(self):
        new, changed = runconfig.apply(CFG, {"sampling.top_p": 0.95, "reasoning_budget_tokens": 4096,
                                             "vram_reserve_mib": 2048, "aliases": "qwen, local", "lazy_load": True})
        self.assertEqual(changed, ["sampling.top_p", "reasoning_budget_tokens", "vram_reserve_mib", "aliases",
                                   "lazy_load"])
        self.assertEqual(new["sampling"], {"temperature": 0.7, "presence_penalty": 1.5, "top_p": 0.95})
        self.assertEqual(new["args"][-2:], ["--vram-reserve-mib", "2048"])
        self.assertEqual(new["aliases"], ["qwen", "local"])
        for k in ("exe", "model_name", "port", "host", "api_key", "mcp_servers", "cors_origins"):
            self.assertEqual(new[k], CFG[k], k)
        self.assertEqual(CFG["sampling"], {"temperature": 0.7, "presence_penalty": 1.5})   # the input is left alone

    def test_parked_conversations_and_parallel(self):
        new, changed = runconfig.apply(CFG, {"conversation_cache_slots": 3, "conversation_cache_mib": 7000, "parallel": 2})
        self.assertEqual(changed, ["conversation_cache_slots", "conversation_cache_mib", "parallel"])
        self.assertEqual(new["args"][-4:], ["--conversation-cache-slots", "3", "--conversation-cache-mib", "7000"])
        self.assertEqual(new["parallel"], 2)
        back, changed = runconfig.apply(new, {"parallel": 1, "conversation_cache_slots": None})   # 1 = one at a time: the key goes
        self.assertEqual(changed, ["parallel", "conversation_cache_slots"])
        self.assertNotIn("parallel", back)
        self.assertEqual(back["args"][-2:], ["--conversation-cache-mib", "7000"])
        for bad in ({"parallel": 9}, {"parallel": -1}, {"parallel": 2.5}, {"conversation_cache_slots": -1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                runconfig.apply(CFG, bad)

    def test_ssd_tier_of_the_conversation_cache(self):
        new, changed = runconfig.apply(CFG, {"conversation_cache_spill_dir": "/work/strata-spill",
                                             "conversation_cache_spill_mib": 30000,
                                             "conversation_cache_spill_min_free_mib": 50000})
        self.assertEqual(changed, ["conversation_cache_spill_dir", "conversation_cache_spill_mib",
                                   "conversation_cache_spill_min_free_mib"])
        self.assertEqual(new["args"][-6:], ["--conversation-cache-spill-dir", "/work/strata-spill",
                                            "--conversation-cache-spill-mib", "30000",
                                            "--conversation-cache-spill-min-free-mib", "50000"])
        self.assertEqual(runconfig.value_of(new, "conversation_cache_spill_dir"), "/work/strata-spill")
        moved, changed = runconfig.apply(new, {"conversation_cache_spill_mib": 1000, "conversation_cache_spill_dir": "/work/x"})
        self.assertEqual(sorted(changed), ["conversation_cache_spill_dir", "conversation_cache_spill_mib"])
        self.assertEqual(runconfig.value_of(moved, "conversation_cache_spill_mib"), 1000)
        for empty in ("", "  ", None):                  # an empty folder removes all three flags
            gone, changed = runconfig.apply(new, {"conversation_cache_spill_dir": empty})
            self.assertEqual(len(changed), 3, empty)
            self.assertEqual(gone["args"], CFG["args"])
        only, _ = runconfig.apply(new, {"conversation_cache_spill_mib": None})
        self.assertNotIn("--conversation-cache-spill-mib", only["args"])
        self.assertIn("--conversation-cache-spill-dir", only["args"])
        for bad in ({"conversation_cache_spill_dir": "relative/dir"}, {"conversation_cache_spill_dir": 5},
                    {"conversation_cache_spill_dir": "/a\nb"}, {"conversation_cache_spill_mib": 100},   # no folder yet
                    {"conversation_cache_spill_min_free_mib": 100},
                    {"conversation_cache_spill_dir": "/w", "conversation_cache_spill_mib": -1},
                    {"conversation_cache_spill_dir": "/w", "conversation_cache_spill_min_free_mib": 1.5}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                runconfig.apply(CFG, bad)

    def test_batch_mtp_needs_parallel(self):
        with self.assertRaises(ValueError):
            runconfig.apply(CFG, {"batch_mtp": True})
        with self.assertRaises(ValueError):
            runconfig.apply({**CFG, "parallel": 1}, {"batch_mtp": True})
        with self.assertRaises(ValueError):
            runconfig.apply(CFG, {"batch_mtp": "yes"})
        new, changed = runconfig.apply(CFG, {"parallel": 2, "batch_mtp": True})
        self.assertEqual(changed, ["parallel", "batch_mtp"])
        self.assertEqual(new["args"][-1], "--batch-mtp")
        self.assertEqual(runconfig.value_of(new, "batch_mtp"), True)
        same, changed = runconfig.apply(new, {"batch_mtp": True})
        self.assertEqual(changed, [])
        off, changed = runconfig.apply(new, {"batch_mtp": False})
        self.assertEqual(changed, ["batch_mtp"])
        self.assertNotIn("--batch-mtp", off["args"])
        self.assertEqual(off["args"], CFG["args"])
        none, _ = runconfig.apply(new, {"batch_mtp": None})
        self.assertNotIn("--batch-mtp", none["args"])
        down, changed = runconfig.apply(new, {"parallel": 1})   # parallel off takes the flag with it
        self.assertEqual(changed, ["parallel", "batch_mtp"])
        self.assertNotIn("--batch-mtp", down["args"])
        self.assertNotIn("parallel", down)
        with self.assertRaises(ValueError):
            runconfig.apply(new, {"parallel": 1, "batch_mtp": True})
        got = {k["key"]: k for k in runconfig.view(new, "x.json")["keys"]}
        self.assertEqual(got["batch_mtp"]["kind"], "bool")
        self.assertEqual(got["conversation_cache_spill_dir"]["kind"], "text")

    def test_null_is_the_default_again(self):
        cfg, _ = runconfig.apply(CFG, {"vram_reserve_mib": 2048, "fit_max_tokens": True})
        new, changed = runconfig.apply(cfg, {"vram_reserve_mib": None, "fit_max_tokens": None,
                                             "sampling.temperature": None, "idle_unload_s": None})
        self.assertEqual(changed, ["vram_reserve_mib", "fit_max_tokens", "sampling.temperature"])
        self.assertEqual(new["args"], CFG["args"])
        self.assertNotIn("fit_max_tokens", new)
        self.assertEqual(new["sampling"], {"presence_penalty": 1.5})
        new, _ = runconfig.apply({"sampling": {"top_k": 20}}, {"sampling.top_k": None})
        self.assertNotIn("sampling", new)                                 # an empty block goes

    def test_refused(self):
        for changes in ({"api_key": "x"}, {"host": "0.0.0.0"}, {"mcp_servers": {}}, {"before_load": "calc.exe"},
                        {"args": []}, {"exe": "evil.exe"}, {"trusted_origins": ["*"]}, {"allowed_hosts": ["*"]},
                        {"sampling.temperature": -1}, {"sampling.top_p": 0}, {"sampling.top_k": 65},
                        {"sampling.min_p": 2}, {"reasoning_budget_tokens": 1.5}, {"fit_max_tokens": "yes"},
                        {"effort_position": "middle"}, {"vram_reserve_mib": -5}, {"idle_unload_s": True},
                        {"aliases": [1]}, {}, None, ["x"]):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                runconfig.apply(CFG, changes)
        with self.assertRaises(ValueError):                              # one bad value: nothing is applied
            runconfig.apply(CFG, {"fit_max_tokens": True, "sampling.top_k": 0})
        with self.assertRaises(ValueError):
            runconfig.apply({**CFG, "vision": {"gpu": True}}, {"lazy_load": True})

    def test_view(self):
        v = runconfig.view({**CFG, "args": CFG["args"] + ["--vram-reserve-mib", "1500"]}, "x/strata-q2_0.json")
        self.assertEqual(v["file"], "strata-q2_0.json")
        got = {k["key"]: k for k in v["keys"]}
        self.assertEqual(got["sampling.temperature"]["value"], 0.7)
        self.assertEqual(got["vram_reserve_mib"]["value"], 1500)
        self.assertIsNone(got["fit_max_tokens"]["value"])
        self.assertEqual(got["anthropic_thinking"]["choices"], ["model", "on_request"])
        self.assertNotIn("api_key", got)
        self.assertNotIn("mcp_servers", got)


class Http(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "strata-q2_0.json"
        self.path.write_text(json.dumps(CFG, indent=1), encoding="utf-8")
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.config_path = str(self.path)
        self.httpd = serve(self.svc, port=0)
        self.host = f"127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.dir.cleanup()

    def call(self, method, body=None, headers=None):
        h = {"Content-Type": "application/json", **(headers or {})}
        req = urllib.request.Request(f"http://{self.host}/config", method=method, headers=h,
                                     data=None if body is None else (body if isinstance(body, bytes)
                                                                     else json.dumps(body).encode()))
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def saved(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_read_and_write_from_the_page(self):
        status, v = self.call("GET")
        self.assertEqual(status, 200, v)
        self.assertEqual({k["key"]: k["value"] for k in v["keys"]}["sampling.temperature"], 0.7)
        with contextlib.redirect_stdout(io.StringIO()):
            status, v = self.call("POST", {"set": {"sampling.temperature": 1.0, "open_browser": False}},
                                  {"Origin": f"http://{self.host}"})
        self.assertEqual(status, 200, v)
        self.assertEqual(v["changed"], ["sampling.temperature", "open_browser"])
        cfg = self.saved()
        self.assertEqual(cfg["sampling"]["temperature"], 1.0)
        self.assertIs(cfg["open_browser"], False)
        for k in ("mcp_servers", "cors_origins", "host", "args", "exe"):
            self.assertEqual(cfg[k], CFG[k], k)
        self.assertEqual(json.loads((self.path.parent / "strata-q2_0.json.bak").read_text(encoding="utf-8")), CFG)
        status, v = self.call("POST", {"set": {"open_browser": False}})   # no Origin (a script on this PC): allowed
        self.assertEqual((status, v["changed"]), (200, []))

    def test_only_from_strata_s_own_page(self):
        before = self.path.read_bytes()
        status, _ = self.call("POST", {"set": {"fit_max_tokens": True}}, {"Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        status, _ = self.call("POST", b'{"set": {"fit_max_tokens": true}}', {"Content-Type": "text/plain"})
        self.assertEqual(status, 415)
        for body in ({"set": {"api_key": ""}}, {"set": {"mcp_servers": {"x": {"command": "calc"}}}},
                     {"set": {"sampling.top_k": 0}}, {"nothing": 1}, b"not json"):
            status, b = self.call("POST", body, {"Origin": f"http://{self.host}"})
            self.assertEqual(status, 400, (body, b))
        self.assertEqual(self.path.read_bytes(), before)                 # nothing was written
        self.assertFalse((self.path.parent / "strata-q2_0.json.bak").exists())

    def test_the_api_key(self):
        self.svc.api_key = "secret"
        self.assertEqual(self.call("GET")[0], 401)
        self.assertEqual(self.call("POST", {"set": {"fit_max_tokens": True}})[0], 401)
        with contextlib.redirect_stdout(io.StringIO()):
            status, _ = self.call("POST", {"set": {"fit_max_tokens": True}}, {"Authorization": "Bearer secret"})
        self.assertEqual(status, 200)
        self.assertIs(self.saved()["fit_max_tokens"], True)

    def test_without_a_run_config(self):
        self.svc.config_path = None
        self.assertEqual(self.call("GET")[0], 404)
        self.assertEqual(self.call("POST", {"set": {"fit_max_tokens": True}})[0], 404)

    def test_the_page_has_the_view(self):
        web = ROOT / "serve" / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        for el in ("cfg-card", "cfg-form", "cfg-save", "cfg-msg"):
            self.assertIn(f'id="{el}"', html)
        self.assertIn('fetch("config", {method: "POST", headers: headers(true)', js)


if __name__ == "__main__":
    unittest.main()
