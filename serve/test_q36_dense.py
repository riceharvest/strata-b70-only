"""serve/test_q36_dense.py - the dense Qwen3.8-27B on strata-q36 (arch qwen35), no engine and no GPU: what the server
hands the engine and what it reads back.

  * the example configs (tools/qwen35/serve-27b*.json) name the SYCL launcher and strata-q36, give the family's API name,
    and carry placeholders, not paths of any machine;
  * engine_args() passes the config's arguments through (strata-q36 reads --native, --max-context and --mtp);
  * every sampling key the server puts on a GEN line is `key=value` (strata-q36's request parser takes the leading
    `key=value` words and stops at the first word without a `=`: the prompt's ids);
  * a DONE line in strata-q36's form (figures after the finish reason, no expert tiers) is read by the server.

    python -m unittest serve.test_q36_dense
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

from serve.server import StrataEngine, default_model_name, engine_args, model_card_sampling

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "tools" / "qwen35").glob("serve-27b*.json"))


def strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield from strings(k)
            yield from strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from strings(v)


class ExampleConfigs(unittest.TestCase):
    def test_there_are_examples(self):
        self.assertEqual([p.name for p in CONFIGS], ["serve-27b-128k.json", "serve-27b-256k.json", "serve-27b.json"])

    def test_each_one_serves_the_27b_on_strata_q36(self):
        for path in CONFIGS:
            cfg = json.loads(path.read_text(encoding="utf-8"))
            with self.subTest(path.name):
                self.assertEqual(cfg["arch"], "qwen35")
                self.assertEqual(default_model_name(cfg), "qwen3.8-27b")
                self.assertEqual(cfg["exe"], "sycl/serve/strata-sycl.sh")
                self.assertEqual(cfg["env"]["STRATA_SYCL_BIN"], "build-sycl-aot/strata-q36")
                # the launcher removes a container of this name before it starts one: not the default name that the
                # Flash-Next server's container has
                self.assertNotIn(cfg["env"]["STRATA_SYCL_NAME"], ("", "strata-sycl-serve"))
                a = cfg["args"]
                self.assertEqual(a[a.index("--mtp") + 1], "3")
                self.assertTrue(a[a.index("--native") + 1].endswith(".gguf"))
                self.assertGreaterEqual(int(a[a.index("--max-context") + 1]), 32768)

    def test_the_context_of_each(self):
        want = {"serve-27b.json": 32768, "serve-27b-128k.json": 131072, "serve-27b-256k.json": 262144}
        for path in CONFIGS:
            a = json.loads(path.read_text(encoding="utf-8"))["args"]
            self.assertEqual(int(a[a.index("--max-context") + 1]), want[path.name], path.name)

    def test_placeholders_not_paths_of_a_machine(self):
        for path in CONFIGS:
            for s in strings(json.loads(path.read_text(encoding="utf-8"))):
                with self.subTest(f"{path.name}: {s}"):
                    if s.startswith("/"):
                        self.assertTrue(s.startswith("/path/to/"), "an absolute path must be a /path/to/ placeholder")
                    self.assertNotRegex(s, r"/home/|/mnt/|[A-Za-z]:\\|\bq27\b")

    def test_engine_args_are_the_configs(self):
        for path in CONFIGS:
            cfg = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(engine_args(cfg), cfg["args"], path.name)

    def test_parallel_is_a_batch_flag_the_engine_also_reads(self):
        cfg = json.loads(CONFIGS[0].read_text(encoding="utf-8"))
        a = engine_args({**cfg, "parallel": 2})
        self.assertEqual(a[a.index("--batch") + 1], "2")


class Sampling(unittest.TestCase):
    def test_model_card_defaults(self):
        self.assertEqual(model_card_sampling("qwen35", True), {"temperature": 1.0, "top_p": 0.95, "top_k": 20})
        self.assertEqual(model_card_sampling("qwen35", False)["presence_penalty"], 1.5)

    def test_gen_keys_are_all_key_value(self):
        # the most the server sends for a chat request: sampling, penalties, a seed, pictures' keys, logprobs, a bias
        for sampling in ({}, {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "seed": 7},
                         {"temperature": 1.0, "repetition_penalty": 1.1, "min_p": 0.05, "logprobs": True, "top_logprobs": 3,
                          "strata_img_keys": ["0a1b2c"], "strata_checkpoint": False, "logit_bias": {"5": -2.0}}):
            keys = StrataEngine.sampling_keys(sampling)
            with self.subTest(sampling):
                for word in keys.split():
                    self.assertRegex(word, r"^[a-z_]+=\S+$", f"{word!r} in {keys!r}")


class Done(unittest.TestCase):
    def test_strata_q36_done_line(self):
        # q36_engine.dp.cpp: "DONE <produced> <prompt tokens> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts
        # offered> <reused> 0 0 0 0 0.0 <prompt tokens read> 0"
        e = SimpleNamespace()
        StrataEngine._parse_done(e, "DONE 64 2048 1193.4 1212.0 length 90 120 0 0 0 0 0 0.0 2048 0")
        self.assertEqual((e.last["generated"], e.last["prompt_tokens"], e.last["finish"]), (64, 2048, "length"))
        self.assertEqual((e.last["drafts_accepted"], e.last["drafts_offered"], e.last["reused"]), (90, 120, 0))
        self.assertEqual(e.last["prompt_read"], 2048)
        self.assertEqual(e.last["offloaded"], 0)

    def test_the_cancel_line(self):
        e = SimpleNamespace()
        StrataEngine._parse_done(e, "DONE 0 512 80.0 0.0 cancel 0 0 0 0 0 0 0 0.0 17 0")
        self.assertEqual(e.last["finish"], "cancel")


if __name__ == "__main__":
    unittest.main()
