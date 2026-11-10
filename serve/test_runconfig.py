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
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve import runconfig
from serve import server
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
        for bad in ({"parallel": 9}, {"parallel": -1}, {"parallel": 2.5}, {"conversation_cache_slots": -1},
                    {"conversation_cache_slots": 2147483648},                   # the engine exits above INT32_MAX
                    {"conversation_cache_park_ssd_keep_pct": 150},              # a percentage
                    {"conversation_cache_spill_probe_s": 0},                    # the engine's least is 1 s
                    {"queue_beat_s": float("nan")}, {"queue_beat_s": float("inf")},   # json reads both as numbers
                    {"conversation_cache_slots": float("inf")}):
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


def keys_of_kind(*names):
    return [k for k, kind, _ in runconfig.EDITABLE if (kind[0] if isinstance(kind, tuple) else kind) in names]


class EngineFlags(unittest.TestCase):
    """The engine flags the page can set: a whole number in a range, a number, one of some words, auto or a number."""

    WHOLE = {"max_context": (1024, 2147483639), "kv_resident": (0, 2147483647), "spec": (0, 8),
             "mtp_window": (0, 2147483647), "conversation_cache_min_free_mib": (0, 4194304),
             "conversation_cache_min_tokens": (0, 2147483647), "conversation_cache_park_tail": (0, 2147483647),
             "conversation_cache_park_gain": (0, 2147483647), "conversation_cache_spill_min_tokens": (0, 2147483647),
             "conversation_cache_spill_mbps": (1, 100000), "conversation_cache_spill_prefill_us": (1, 1000000)}
    FLAG = {"max_context": "--max-context", "kv_resident": "--kv-resident", "spec": "--spec", "mtp_window": "--mtp-window",
            "conversation_cache_min_free_mib": "--conversation-cache-min-free-mib",
            "conversation_cache_min_tokens": "--conversation-cache-min-tokens",
            "conversation_cache_park_tail": "--conversation-cache-park-tail",
            "conversation_cache_park_gain": "--conversation-cache-park-gain",
            "conversation_cache_spill_min_tokens": "--conversation-cache-spill-min-tokens",
            "conversation_cache_spill_mbps": "--conversation-cache-spill-mbps",
            "conversation_cache_spill_prefill_us": "--conversation-cache-spill-prefill-us"}
    SPILL = {"conversation_cache_spill_dir": "/work/strata-spill"}

    def base(self, key):
        """A config the key can be set in: the spill keys need their folder."""
        return {**CFG, "args": CFG["args"] + ["--conversation-cache-spill-dir", "/work/strata-spill"]} \
            if key in runconfig.SPILL_DEPENDENTS else CFG

    def test_a_whole_number_in_its_range(self):
        for key, (lo, hi) in self.WHOLE.items():
            flag = self.FLAG[key]
            with self.subTest(key=key):
                base = self.base(key)
                mid = lo + 7 if lo + 7 <= hi else hi
                for good in (lo, mid, hi):
                    new, changed = runconfig.apply(base, {key: good})
                    self.assertEqual(changed, [key])
                    self.assertEqual(new["args"][new["args"].index(flag) + 1], str(good))
                    self.assertEqual(runconfig.value_of(new, key), good)                          # the round trip
                    self.assertEqual(runconfig.check(key, runconfig.value_of(new, key), new), good)
                    self.assertEqual(new["user_args"][-2:], [flag, str(good)])                    # kept as the person's
                for bad in (lo - 1, hi + 1, mid + 0.5, "12", True, [3]):
                    with self.assertRaises(ValueError, msg=repr(bad)):
                        runconfig.apply(base, {key: bad})
                new, _ = runconfig.apply(base, {key: mid})
                gone, changed = runconfig.apply(new, {key: None})                                 # empty: the engine's default
                self.assertEqual(changed, [key])
                self.assertEqual(gone["args"], base["args"])
                self.assertNotIn("user_args", gone)
                self.assertIsNone(runconfig.value_of(gone, key))

    def test_a_number_between_0_and_1(self):
        base = {**CFG, "args": ["--pack", "p"]}
        for good, text in ((0.25, "0.25"), (1, "1"), (0, "0"), (0.5, "0.5"), (1.0, "1")):
            new, changed = runconfig.apply(base, {"spec_min_p": good})
            self.assertEqual(changed, ["spec_min_p"])
            self.assertEqual(new["args"][-2:], ["--spec-min-p", text], good)
            self.assertEqual(runconfig.value_of(new, "spec_min_p"), float(good))
            self.assertEqual(new["user_args"], ["--spec-min-p", text])
        for bad in (-0.01, 1.01, "0.5", True, [0.5]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                runconfig.apply(base, {"spec_min_p": bad})
        same, changed = runconfig.apply(CFG, {"spec_min_p": 0.5})                                 # CFG holds --spec-min-p 0.5
        self.assertEqual(changed, [])
        moved, _ = runconfig.apply(CFG, {"spec_min_p": 0.3})
        self.assertEqual(moved["args"].count("--spec-min-p"), 1)
        self.assertEqual(moved["args"][moved["args"].index("--spec-min-p") + 1], "0.3")
        off, changed = runconfig.apply(CFG, {"spec_min_p": None})
        self.assertEqual((changed, "--spec-min-p" in off["args"]), (["spec_min_p"], False))

    def test_one_of_some_words(self):
        for word in ("fp16", "int8", "q4_0", "k8v4"):
            new, changed = runconfig.apply({**CFG, "args": ["--pack", "p"]}, {"kv_format": word})
            self.assertEqual(new["args"][-2:], ["--kv", word])
            self.assertEqual(runconfig.value_of(new, "kv_format"), word)
            self.assertEqual(new["user_args"], ["--kv", word])
        new, _ = runconfig.apply(CFG, {"kv_format": "q4"})                                         # the engine's other spelling
        self.assertEqual(runconfig.value_of(new, "kv_format"), "q4_0")
        self.assertEqual(new["args"].count("--kv"), 1)
        held = {**CFG, "args": ["--kv", "q4"]}                                                     # a hand edit: q4 stays, not "default"
        self.assertEqual(runconfig.value_of(held, "kv_format"), "q4_0")
        self.assertEqual(runconfig.apply(held, {"kv_format": runconfig.value_of(held, "kv_format")})[1], [])
        for bad in ("int4", "INT8", "", 8, True, ["int8"]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                runconfig.apply(CFG, {"kv_format": bad})
        gone, _ = runconfig.apply(CFG, {"kv_format": None})
        self.assertNotIn("--kv", gone["args"])
        got = {k["key"]: k for k in runconfig.view(CFG, "x.json")["keys"]}["kv_format"]
        self.assertEqual((got["kind"], got["choices"], got["value"]), ("enum", ["fp16", "int8", "q4_0", "k8v4"], "int8"))

    def test_auto_or_a_whole_number(self):
        for key, flag, words in (("expert_cache", "--expert-cache", ("auto",)),
                                 ("prefill_chunk", "--prefill", ("auto", "auto:16384", "auto:32768"))):
            for good in (*words, 0, 4096, 16384, "4096"):
                with self.subTest(key=key, good=good):
                    new, changed = runconfig.apply({**CFG, "args": ["--pack", "p"]}, {key: good})
                    stored = int(good) if isinstance(good, str) and good.isdigit() else good
                    self.assertEqual(new["args"][-2:], [flag, str(stored)])
                    self.assertEqual(runconfig.value_of(new, key), stored)
                    self.assertEqual(runconfig.check(key, runconfig.value_of(new, key), new), stored)
                    self.assertEqual(new["user_args"], [flag, str(stored)])
            for bad in ("fast", "auto:1", "auto:", -1, 1.5, 2147483648, True, [1]):
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    runconfig.apply(CFG, {key: bad})
            for empty in (None, "", "  "):
                held = {**CFG, "args": ["--pack", "p", flag, "auto"]}
                gone, changed = runconfig.apply(held, {key: empty})
                self.assertEqual((changed, flag in gone["args"]), ([key], False), empty)
        with self.assertRaises(ValueError):                                                        # auto:16384 is the prefill's only
            runconfig.apply(CFG, {"expert_cache": "auto:16384"})
        # what the live config holds comes back as it is: a Save that touches nothing else must not remove it
        held = {**CFG, "args": ["--expert-cache", "auto", "--prefill", "auto:16384"]}
        for key in ("expert_cache", "prefill_chunk"):
            self.assertEqual(runconfig.apply(held, {key: runconfig.value_of(held, key)})[1], [])
        got = {k["key"]: k for k in runconfig.view(held, "x.json")["keys"]}
        self.assertEqual((got["expert_cache"]["kind"], got["expert_cache"]["value"]), ("auto", "auto"))
        self.assertEqual(got["prefill_chunk"]["value"], "auto:16384")

    def test_a_value_only_in_user_args_is_seen_and_can_be_removed(self):
        cfg = {**CFG, "user_args": ["--conversation-cache-min-free-mib", "7680"]}                 # as the live config holds it
        self.assertEqual(runconfig.value_of(cfg, "conversation_cache_min_free_mib"), 7680)
        gone, changed = runconfig.apply(cfg, {"conversation_cache_min_free_mib": None})
        self.assertEqual(changed, ["conversation_cache_min_free_mib"])
        self.assertNotIn("user_args", gone)
        new, changed = runconfig.apply(cfg, {"conversation_cache_min_free_mib": 4096})
        self.assertEqual(new["user_args"], ["--conversation-cache-min-free-mib", "4096"])
        self.assertEqual(runconfig.apply(cfg, {"conversation_cache_min_free_mib": 7680})[1], [])

    def test_the_view_gives_ranges_for_the_page(self):
        got = {k["key"]: k for k in runconfig.view(CFG, "x.json")["keys"]}
        self.assertEqual((got["spec"]["kind"], got["spec"]["step"], got["spec"]["min"], got["spec"]["max"]), ("number", 1, 0, 8))
        self.assertEqual((got["spec_min_p"]["kind"], got["spec_min_p"]["step"], got["spec_min_p"]["max"]), ("number", "any", 1))
        self.assertEqual(got["spec_min_p"]["value"], 0.5)
        self.assertEqual(got["conversation_cache_spill_verify"]["kind"], "bool")
        self.assertEqual(got["expert_copies_loan_slots"]["kind"], "number")
        self.assertEqual(got["reasoning_loop_recovery"]["choices"], ["stop", "recover"])

    def test_spill_keys_need_the_folder(self):
        new_keys = {"conversation_cache_spill_min_tokens": 2048, "conversation_cache_spill_mbps": 3000,
                    "conversation_cache_spill_prefill_us": 250, "conversation_cache_spill_verify": True}
        for dep in new_keys:
            self.assertIn(dep, runconfig.SPILL_DEPENDENTS)
            with self.subTest(dep=dep), self.assertRaises(ValueError) as cm:
                runconfig.apply(CFG, {dep: new_keys[dep]})
            self.assertIn("conversation_cache_spill_dir", str(cm.exception))
        new, changed = runconfig.apply(CFG, {"conversation_cache_spill_dir": "/work/s", **new_keys})   # with the folder: fine
        self.assertEqual(sorted(changed), sorted(["conversation_cache_spill_dir", *new_keys]))
        self.assertIn("--conversation-cache-spill-verify", new["args"])
        self.assertEqual(runconfig.value_of(new, "conversation_cache_spill_verify"), True)
        gone, changed = runconfig.apply(new, {"conversation_cache_spill_dir": None})              # no folder: they all go
        self.assertEqual(len(changed), 5)
        self.assertEqual(gone["args"], CFG["args"])
        self.assertNotIn("user_args", gone)
        off, _ = runconfig.apply(new, {"conversation_cache_spill_verify": False})                 # off needs no folder
        self.assertNotIn("--conversation-cache-spill-verify", off["args"])
        self.assertEqual(runconfig.apply(CFG, {"conversation_cache_spill_verify": False})[1], [])


class EnvSwitches(unittest.TestCase):
    ENV = {"STRATA_SYCL_MOUNTS": "/home/x/models", "STRATA_LEND_MIRROR": "1536"}

    def test_stored_as_text_under_env(self):
        new, changed = runconfig.apply(CFG, {"expert_copies_loan_slots": 1536, "expert_copies_max_mib": 23500})
        self.assertEqual(changed, ["expert_copies_loan_slots", "expert_copies_max_mib"])
        self.assertEqual(new["env"], {"STRATA_LEND_MIRROR": "1536", "STRATA_MIRROR_MIB": "23500"})
        self.assertEqual(new["args"], CFG["args"])                                                  # no flag, no user_args
        self.assertNotIn("env", CFG)                                                                # the input is left alone
        self.assertEqual(runconfig.value_of(new, "expert_copies_loan_slots"), 1536)
        self.assertEqual(runconfig.check("expert_copies_max_mib", runconfig.value_of(new, "expert_copies_max_mib"), new), 23500)

    def test_other_entries_stay_and_empty_removes(self):
        cfg = {**CFG, "env": dict(self.ENV)}
        new, changed = runconfig.apply(cfg, {"expert_copies_max_mib": 17000})
        self.assertEqual(new["env"], {**self.ENV, "STRATA_MIRROR_MIB": "17000"})
        gone, changed = runconfig.apply(cfg, {"expert_copies_loan_slots": None})
        self.assertEqual((changed, gone["env"]), (["expert_copies_loan_slots"], {"STRATA_SYCL_MOUNTS": "/home/x/models"}))
        self.assertEqual(runconfig.apply(cfg, {"expert_copies_max_mib": None})[1], [])              # nothing to remove: no change
        only, _ = runconfig.apply({**CFG, "env": {"STRATA_LEND_MIRROR": "0"}}, {"expert_copies_loan_slots": None})
        self.assertNotIn("env", only)                                                               # an empty block goes
        zero, changed = runconfig.apply(CFG, {"expert_copies_loan_slots": 0})                       # 0 is a value (no copies), not empty
        self.assertEqual((changed, zero["env"]), (["expert_copies_loan_slots"], {"STRATA_LEND_MIRROR": "0"}))
        self.assertEqual(runconfig.value_of(zero, "expert_copies_loan_slots"), 0)

    def test_refused(self):
        for bad in (-1, 1.5, True, [1], "x", "-3", 4194305):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                runconfig.apply(CFG, {"expert_copies_max_mib": bad})
        new, _ = runconfig.apply(CFG, {"expert_copies_loan_slots": "2048"})                         # a script may send digits as text
        self.assertEqual(new["env"], {"STRATA_LEND_MIRROR": "2048"})

    def test_a_hand_written_value(self):
        for text, want in (("1536", 1536), (1536, 1536), (" 1536 ", 1536)):
            cfg = {**CFG, "env": {"STRATA_LEND_MIRROR": text}}
            self.assertEqual(runconfig.value_of(cfg, "expert_copies_loan_slots"), want)
            self.assertEqual(runconfig.apply(cfg, {"expert_copies_loan_slots": want})[1], [], text)


class Groups(unittest.TestCase):
    def test_every_key_has_one_group(self):
        keys = [k for k, _, _ in runconfig.EDITABLE]
        self.assertEqual(len(keys), len(set(keys)))
        grouped = [k for ks in runconfig.GROUPS.values() for k in ks]
        self.assertEqual(sorted(grouped), sorted(keys))                      # none missing, none twice, none unknown
        self.assertEqual(set(runconfig.GROUPS), {"Model and context", "Speed", "Parked conversations", "SSD tier",
                                                 "Thinking and answers", "Server", "Monitor"})

    def test_the_view_carries_the_groups(self):
        v = runconfig.view(CFG, "x.json")
        self.assertEqual(v["groups"], list(runconfig.GROUPS))
        by_key = {k["key"]: k["group"] for k in v["keys"]}
        self.assertEqual(by_key["max_context"], "Model and context")
        self.assertEqual(by_key["conversation_cache_spill_verify"], "SSD tier")
        self.assertEqual(by_key["queue_beat_s"], "Server")
        self.assertEqual(by_key["api_monitor_requests"], "Monitor")
        self.assertEqual({k["group"] for k in v["keys"]}, set(v["groups"]))
        order = [k["key"] for k in v["keys"]]                                # inside a group the keys keep the list's order
        for keys in runconfig.GROUPS.values():
            self.assertEqual([k for k in order if k in keys], sorted(keys, key=order.index))

    def test_a_save_that_changes_nothing_changes_nothing(self):
        """What the view shows, sent back as the page sends it, is no change - for the config that runs now too."""
        cfg = json.loads(json.dumps({**CFG, "args": CFG["args"] + ["--expert-cache", "auto", "--prefill", "16384", "--kv-resident",
                                                                    "32768", "--max-context", "262144", "--spec", "4"],
                                     "user_args": ["--conversation-cache-min-free-mib", "7680"],
                                     "env": {"STRATA_LEND_MIRROR": "1536"}, "reasoning_loop_recovery": True}))
        shown = {k["key"]: k["value"] for k in runconfig.view(cfg, "x.json")["keys"] if k["value"] is not None}
        self.assertGreaterEqual(len(shown), 10)
        self.assertEqual(shown["reasoning_loop_recovery"], "recover")                              # true is recover
        new, changed = runconfig.apply(cfg, shown)
        self.assertEqual(changed, [])
        self.assertEqual(new, cfg)


class ServerNumbers(unittest.TestCase):
    """The server's own numbers (the Settings page's server keys): a key changes what the server does, an absent key does not."""

    def setUp(self):
        self.addCleanup(server.apply_server_numbers, {})                     # every test leaves the defaults as it found them
        self.before = {where: getattr(*server._number_home(where)) for where in server.SERVER_NUMBERS.values()}

    def run_reply(self, svc, max_new):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            list(svc.run(svc.tok.encode("hi"), True, None, max_new, {}, threading.Event()))
        return out.getvalue()

    def service(self):
        tok = ByteTokenizer()
        return Service(MockEngine(tok, "Thinking.</think>\n\nok", max_context=8192), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))

    def test_absent_keys_leave_the_defaults(self):
        self.assertEqual(server.apply_server_numbers({}), [])
        self.assertEqual(server.apply_server_numbers({"sampling": {"top_k": 20}, "args": []}), [])
        for where, v in self.before.items():
            self.assertEqual(getattr(*server._number_home(where)), v, where)
        self.assertEqual((server.LOOP_CHECK_EVERY, server.LOOP_COVERAGE, server.LOOP_HISTORY_WORDS, server.LOOP_TAIL_CHARS,
                          server.LOOP_TAIL_PERIOD), (512, 0.25, 30000, 2048, 64))
        self.assertEqual((server.ANSWER_ROOM_MIN_CAP, server.ANSWER_RESERVE_MIN, server.QUEUE_BEAT_S, server.KEEPALIVE_S,
                          server.METRICS_REQUESTS, server.API_MONITOR_REQUESTS), (1024, 512, 10.0, 10.0, 12, 100))
        self.assertEqual((server.StrataEngine.YIELDS_MAX, server.StrataEngine.RESTART_RETRY_S, server.Service.UNLOAD_WAIT_S),
                         (2, 15.0, 300.0))
        self.assertEqual((server.LOOP_TAIL_CHARS, server.LOOP_TAIL_PERIOD), runconfig.LOOP_TAIL_DEFAULTS)
        self.assertEqual(self.service().api_requests.maxlen, 100)

    def test_every_server_number_is_a_settings_key(self):
        listed = {k for k, _, _ in runconfig.EDITABLE}
        self.assertLessEqual(set(server.SERVER_NUMBERS), listed)

    def test_a_key_sets_its_number_and_all_of_them_at_once(self):
        cfg = {"loop_check_every": 256, "loop_coverage": 0.5, "loop_history_words": 20000, "loop_tail_chars": 4096,
               "loop_tail_period": 128, "answer_room_min_cap": 2048, "answer_reserve_min": 800, "yields_max": 4,
               "admit_short_first": 8192, "unload_wait_s": 60, "restart_retry_s": 5, "queue_beat_s": 2.5, "keepalive_s": 3,
               "stream_linger_s": 1.5, "monitor_requests": 30, "api_monitor_requests": 250}
        self.assertEqual(server.apply_server_numbers(cfg), list(cfg))
        got = {k: getattr(*server._number_home(w)) for k, w in server.SERVER_NUMBERS.items()}
        self.assertEqual(got, {k: (float(v) if isinstance(self.before[server.SERVER_NUMBERS[k]], float) else v)
                               for k, v in cfg.items()})
        self.assertEqual(self.service().api_requests.maxlen, 250)
        server.apply_server_numbers({})                                                            # a start without them: defaults again
        self.assertEqual({where: getattr(*server._number_home(where)) for where in self.before}, self.before)

    def test_answer_room_follows_its_numbers(self):
        svc = self.service()
        self.assertIn("thinking limited to 1488 of max_tokens 2000", self.run_reply(svc, 2000))    # 2000 - max(512, 500)
        server.apply_server_numbers({"answer_reserve_min": 800})
        self.assertIn("thinking limited to 1200 of max_tokens 2000", self.run_reply(svc, 2000))
        server.apply_server_numbers({"answer_room_min_cap": 4096})                                  # 2000 is under the new floor
        self.assertNotIn("thinking limited", self.run_reply(svc, 2000))
        server.apply_server_numbers({})
        self.assertIn("thinking limited to 1488", self.run_reply(svc, 2000))

    def test_the_monitor_keeps_and_lists_what_it_is_told(self):
        svc = self.service()
        svc.history.extend({"id": str(i)} for i in range(40))
        self.assertEqual(len(svc.metrics()["requests"]), 12)
        server.apply_server_numbers({"monitor_requests": 5})
        m = svc.metrics()
        self.assertEqual([r["id"] for r in m["requests"]], ["39", "38", "37", "36", "35"])         # the newest first
        self.assertEqual(len(svc.metrics(all_requests=True)["requests"]), 40)                      # the full list is not cut
        self.assertEqual(m["requests_kept"], 40)

    def test_the_loop_detector_reads_its_numbers(self):
        cycle = "ab" * 5000                                                                         # one long word, period 2
        self.assertEqual(server.reasoning_repeat_coverage(cycle), 1.0)
        server.apply_server_numbers({"loop_tail_period": 1})                                        # a period of 1 does not see it
        self.assertEqual(server.reasoning_repeat_coverage(cycle), 0.0)
        server.apply_server_numbers({"loop_tail_chars": 256, "loop_tail_period": 2})
        self.assertEqual(server.reasoning_repeat_coverage(cycle[:300]), 1.0)                        # a shorter tail is judged too
        server.apply_server_numbers({})
        self.assertEqual(server.reasoning_repeat_coverage(cycle[:300]), 0.0)

    def test_a_bad_value_changes_nothing(self):
        for bad in ({"queue_beat_s": 0}, {"keepalive_s": -1}, {"loop_check_every": 0}, {"loop_coverage": 0}, {"loop_coverage": 1.5},
                    {"loop_history_words": 1999}, {"loop_tail_period": 2000}, {"loop_tail_chars": 300, "loop_tail_period": 200},
                    {"yields_max": -1}, {"restart_retry_s": "15"}, {"monitor_requests": 501}, {"api_monitor_requests": 0},
                    {"unload_wait_s": True}, {"admit_short_first": 1.5}, {"answer_room_min_cap": 0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError) as cm:
                server.apply_server_numbers({**bad, "answer_reserve_min": 999})
            self.assertEqual(server.ANSWER_RESERVE_MIN, 512)                                         # not even the good key was set
            if len(bad) == 1:
                self.assertIn(next(iter(bad)), str(cm.exception))                                    # the message names the key

    def test_session_wait_stays_the_engines(self):
        self.assertNotIn("SESSION_WAIT_MAX_S", server.SERVER_NUMBERS.values())
        self.assertNotIn("session_wait_max_s", {k for k, _, _ in runconfig.EDITABLE})

    def test_the_settings_page_refuses_what_the_server_would(self):
        for bad in ({"loop_tail_period": 2000}, {"loop_tail_chars": 300, "loop_tail_period": 200}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                runconfig.apply(CFG, bad)
        new, _ = runconfig.apply({**CFG, "loop_tail_chars": 256}, {"loop_tail_period": 128})
        self.assertEqual(new["loop_tail_period"], 128)
        with self.assertRaises(ValueError):                                                         # the other half of the pair
            runconfig.apply({**CFG, "loop_tail_period": 128}, {"loop_tail_chars": 255})


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

    def test_the_new_keys_over_http(self):
        status, v = self.call("GET")
        self.assertEqual(status, 200, v)
        self.assertIn("SSD tier", v["groups"])
        self.assertEqual({k["key"]: k["group"] for k in v["keys"]}["spec"], "Speed")
        with contextlib.redirect_stdout(io.StringIO()):
            status, v = self.call("POST", {"set": {"spec": 6, "kv_format": "q4", "expert_copies_loan_slots": 1536,
                                                   "expert_cache": "auto", "loop_check_every": 256}},
                                  {"Origin": f"http://{self.host}"})
        self.assertEqual(status, 200, v)
        cfg = self.saved()
        self.assertEqual(cfg["env"], {"STRATA_LEND_MIRROR": "1536"})
        self.assertEqual(cfg["loop_check_every"], 256)
        self.assertEqual(cfg["args"][cfg["args"].index("--kv") + 1], "q4_0")
        self.assertEqual(cfg["args"][cfg["args"].index("--spec") + 1], "6")
        self.assertEqual(cfg["args"][cfg["args"].index("--expert-cache") + 1], "auto")
        status, b = self.call("POST", {"set": {"conversation_cache_spill_mbps": 100}}, {"Origin": f"http://{self.host}"})
        self.assertEqual(status, 400, b)                                                           # no folder: nothing written
        self.assertNotIn("--conversation-cache-spill-mbps", self.saved()["args"])

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
        self.assertIn('class="cfg-group"', js)                              # a heading per group
        self.assertIn("c.groups", js)
        self.assertIn('k.kind === "auto"', js)                              # auto or a whole number: a text field, a number back


if __name__ == "__main__":
    unittest.main()
