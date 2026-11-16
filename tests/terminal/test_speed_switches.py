"""The 0.2.7 speed features and their switches: the contract names, the defaults (Faster long prompts OFF, the other two ON), where each applies, and how the Terminal carries them.

The app (apps/host/server/native-settings.mjs SPEED_SWITCHES) and Control (strata_control/speed_switches.py) say the same thing; the last test reads the
app's table with node and fails when the two differ.  `strata launch` shows the environment entries as rows of its Advanced table and writes them into
the run config's "env" (`--set STRATA_KVA=1` turns Faster long prompts on); a default launch has none of them.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from strata_cli.launch import params as P
from strata_cli.launch import settings as S
from strata_cli.launch.runner import derive_config
from strata_control import speed_switches as W

ROOT = Path(__file__).resolve().parents[2]

ENV = {      # what each switch writes outside its default
    "faster_long_prompts": {"STRATA_KVA": "1"},
    "faster_12gb_nvidia": {"STRATA_ROUTE_PRIOR": "1.5"},
    "faster_radeon_9000": {"STRATA_PCIE_FRAC_DEFAULT": "old"},
    "qwen_defaults_026": {"STRATA_DEFAULTS": "026"},         # the Terminal's troubleshooting switch: not in the app
}
DEFAULT = {"faster_long_prompts": "off", "faster_12gb_nvidia": "off", "faster_radeon_9000": "on", "qwen_defaults_026": "off"}
RTX3060 = {"name": "NVIDIA GeForce RTX 3060", "vendor": "nvidia", "arch": "86", "vram_gb": 12.0}
RTX4070S = {"name": "NVIDIA GeForce RTX 4070 SUPER", "vendor": "nvidia", "arch": "89", "vram_gb": 12.0}
RTX4060TI16 = {"name": "NVIDIA GeForce RTX 4060 Ti 16GB", "vendor": "nvidia", "arch": "89", "vram_gb": 16.0}
R9700 = {"name": "AMD Radeon AI PRO R9700", "vendor": "amd", "arch": "gfx1201"}
RX7900 = {"name": "AMD Radeon RX 7900 XTX", "vendor": "amd", "arch": "gfx1100"}
STRIX = {"name": "AMD Radeon 8060S", "vendor": "amd", "arch": "gfx1151", "uma": True}
RTX = {"name": "NVIDIA GeForce RTX 4090", "vendor": "nvidia", "arch": "89"}
# A Qwen3.8-Flash-Next run config (the original, Swift, Coder, Unsloth, Gyro, NVFP4): setup writes no "arch" for it.  Qwen3.6-35B-A3B has an engine path of its own: "arch".
QWEN = {"model_name": "qwen3.8-flash-next-iq3_s", "backend": "cuda", "args": ["--native", "m.gguf"], "env": {"PATH": "/bin"}}
QWEN36 = {"arch": "qwen35moe", "model_name": "qwen3.6-35b-a3b-ud-q4_k_xl", "backend": "cuda", "args": ["--native", "m.gguf"], "env": {"PATH": "/bin"}}


def ids(cfg, gpus, backend=None):
    """The switches the app would show (the Terminal-only one is tested apart)."""
    return [s.id for s in W.applicable(cfg, gpus, backend) if not s.terminal_only]


class Contract(unittest.TestCase):
    def test_four_switches_with_the_contract_names_and_their_defaults(self):
        self.assertEqual([s.id for s in W.SWITCHES], list(ENV))
        for s in W.SWITCHES:
            self.assertEqual(dict(s.env), ENV[s.id], s.id)
            self.assertEqual(s.default, DEFAULT[s.id], s.id)

    def test_an_effect_text_says_what_the_feature_does_in_plain_words(self):
        for s in W.SWITCHES:
            if not s.terminal_only:
                self.assertRegex(s.effect, r"Off by default" if s.default == "off" else r"[Tt]urn off", s.id)
            self.assertNotRegex(s.effect, r"STRATA_|--|env", s.id)
        self.assertEqual(W.SWITCHES[0].effect, "Reads very long prompts faster by approximating part of the work. Off by default: answers about long documents can miss or mix up details.")

    def test_where_each_switch_applies(self):
        self.assertEqual(ids(QWEN, [RTX]), ["faster_long_prompts"])
        self.assertEqual(ids({**QWEN, "backend": "hip"}, [RX7900]), ["faster_long_prompts"])
        self.assertEqual(ids({**QWEN, "backend": "hip"}, [STRIX]), ["faster_long_prompts"], "Strix Halo is gfx1151, not gfx12")
        self.assertEqual(ids({**QWEN, "backend": "hip"}, [R9700]), ["faster_long_prompts", "faster_radeon_9000"])
        self.assertEqual(ids({"model_name": "qwen3.8-flash-next-iq3_s"}, [R9700], "hip"), ["faster_long_prompts", "faster_radeon_9000"], "the installed engine's backend when the config has none")
        self.assertEqual(ids({**QWEN, "backend": "hip"}, [{"name": "AMD Radeon RX 9070 XT", "vendor": "amd"}]), ["faster_long_prompts", "faster_radeon_9000"])
        self.assertEqual(ids({**QWEN, "backend": "cuda"}, [RTX, R9700]), ["faster_long_prompts"], "the Radeon switch needs the HIP engine")
        for backend in ("sycl", "metal", "cpu", None):
            self.assertEqual(ids({**QWEN, "backend": backend}, [R9700]), [], backend)

    def test_faster_answers_on_12gb_nvidia_cards_only_where_the_engine_routing_default_applies(self):
        """The engine turns residency-biased routing on by default on a DISCRETE NVIDIA card of 13 GiB or less in a CUDA build (the Flash-Next shape)."""
        self.assertEqual(ids(QWEN, [RTX3060]), ["faster_long_prompts", "faster_12gb_nvidia"])
        self.assertEqual(ids(QWEN, [RTX4070S]), ["faster_long_prompts", "faster_12gb_nvidia"])
        self.assertEqual(ids(QWEN, [{**RTX3060, "vram_gb": 13.0}]), ["faster_long_prompts", "faster_12gb_nvidia"], "13 GiB counts")
        self.assertEqual(ids(QWEN, [{**RTX3060, "vram_gb": 13.5}]), ["faster_long_prompts"])
        self.assertEqual(ids(QWEN, [RTX4060TI16]), ["faster_long_prompts"], "16 GB is not a 12 GB card")
        self.assertEqual(ids(QWEN, [RTX]), ["faster_long_prompts"])
        self.assertEqual(ids(QWEN, [RTX3060, RTX]), ["faster_long_prompts", "faster_12gb_nvidia"], "one small card among others")
        self.assertEqual(ids(QWEN, [{**RTX3060, "uma": True}]), ["faster_long_prompts"], "not a chip that shares system memory")
        self.assertEqual(ids(QWEN, [{**RTX3060, "integrated": True}]), ["faster_long_prompts"])
        self.assertEqual(ids(QWEN, [{**RTX3060, "usable": False}]), ["faster_long_prompts"])
        self.assertEqual(ids({**QWEN, "backend": "hip"}, [{"name": "AMD Radeon RX 7700 XT", "vendor": "amd", "arch": "gfx1101", "vram_gb": 12.0}]), ["faster_long_prompts"], "not a Radeon")
        self.assertEqual(ids({**QWEN, "backend": "sycl"}, [RTX3060]), [], "not a CUDA build")
        self.assertEqual(ids(QWEN36, [RTX3060]), [], "Qwen3.6-35B-A3B: the routing needs 512 experts")
        self.assertEqual(ids(QWEN, [{"name": "GeForce RTX 3060", "vram_gb": 12.0}]), ["faster_long_prompts", "faster_12gb_nvidia"], "no vendor reported: the name decides")
        self.assertEqual(ids(QWEN, [{"name": "Mystery card", "vram_gb": 12.0}]), ["faster_long_prompts"])

    def test_the_12gb_switch_is_off_by_default_and_on_writes_route_prior_1_5(self):
        # 0.2.8 (owner, 2026-10-29, #4402): off by default; a default launch carries nothing, so the engine runs as 0.2.6 did
        env = {"PATH": "/bin"}
        self.assertFalse(W.is_on({}, "faster_12gb_nvidia"))
        self.assertEqual(W.set_switch(env, "faster_12gb_nvidia", False), env, "Off is the default: it writes nothing")
        on = W.set_switch(env, "faster_12gb_nvidia", True)
        self.assertEqual(on, {"PATH": "/bin", "STRATA_ROUTE_PRIOR": "1.5"})
        self.assertTrue(W.is_on(on, "faster_12gb_nvidia"))
        self.assertEqual(W.set_switch(on, "faster_12gb_nvidia", False), env, "back to Off removes the entry")
        # the 0.2.7 Off wrote STRATA_ROUTE_PRIOR=0: that file is in the default state now, and On replaces it with 1.5
        self.assertFalse(W.is_on({"STRATA_ROUTE_PRIOR": "0"}, "faster_12gb_nvidia"))
        self.assertEqual(W.set_switch({"STRATA_ROUTE_PRIOR": "0"}, "faster_12gb_nvidia", True), {"STRATA_ROUTE_PRIOR": "1.5"})
        self.assertEqual(W.set_switch({"STRATA_ROUTE_PRIOR": "0"}, "faster_12gb_nvidia", False), {})
        sw = W.BY_ID["faster_12gb_nvidia"]
        self.assertEqual(sw.default, "off")
        self.assertIn("faster on 12 GB cards", sw.effect)
        self.assertIn("Off by default", sw.effect)
        self.assertIn("long agent sessions it can repeat itself", sw.effect)

    def test_the_terminal_only_switch_is_for_qwen_and_never_in_the_app(self):
        sw = W.BY_ID["qwen_defaults_026"]
        self.assertTrue(sw.terminal_only)
        self.assertEqual(dict(sw.env), {"STRATA_DEFAULTS": "026"})
        self.assertNotIn("qwen_defaults_026", [s.id for s in W.app_switches()])
        for cfg, gpus in ((QWEN, [RTX]), (QWEN36, [RTX]), ({**QWEN, "backend": "sycl"}, []), ({**QWEN, "backend": "cpu"}, [])):
            self.assertIn("qwen_defaults_026", [s.id for s in W.applicable(cfg, gpus)])
        self.assertEqual(W.set_switch({"PATH": "/bin"}, sw, True), {"PATH": "/bin", "STRATA_DEFAULTS": "026"})
        self.assertFalse(W.is_on({}, sw))
        self.assertEqual(W.set_switch({"STRATA_DEFAULTS": "026"}, sw, False), {})

    def test_faster_long_prompts_is_for_qwen38_flash_next_only(self):
        """KVA was measured on the 48-layer / 512-expert Flash-Next shape and the engine declines other shapes: no row for Qwen3.6-35B-A3B (arch qwen35moe)."""
        self.assertEqual(ids(QWEN36, [RTX]), [], "Qwen3.6-35B-A3B on CUDA: no KVA row")
        self.assertEqual(ids({**QWEN36, "backend": "hip"}, [RX7900]), [], "on HIP too")
        self.assertEqual(ids({**QWEN36, "backend": "hip"}, [R9700]), ["faster_radeon_9000"], "the Radeon switch is not tied to the model's shape")
        for family in ("qwen3.8-flash-next-iq3_s", "swift-1.5-iq3_s", "qwen3.8-flash-next-coder-iq3_s", "qwen3.8-flash-next-unsloth-ud-q4_k_xl",
                       "qwen3.8-flash-next-gyro-gyro-s", "qwen3.8-flash-next-nvfp4-nvfp4"):
            self.assertEqual(ids({**QWEN, "model_name": family}, [RTX]), ["faster_long_prompts"], family)

    def test_outside_the_default_and_back_is_the_identical_environment(self):
        env = {"PATH": "/bin", "STRATA_PREFILL": "2"}
        for s in W.SWITCHES:
            other = W.set_switch(env, s, s.default != "on")           # KVA on, the other two off
            self.assertEqual(other, {**env, **ENV[s.id]})
            self.assertTrue(W.is_set(other, s))
            self.assertEqual(W.is_on(other, s), s.default != "on")
            back = W.set_switch(other, s, s.default == "on")
            self.assertEqual(json.dumps(back), json.dumps(env), s.id)

    def test_kva_is_off_by_default_and_an_earlier_builds_zero_shows_as_off(self):
        self.assertFalse(W.is_on({}, "faster_long_prompts"))
        self.assertTrue(W.is_on({"STRATA_KVA": "1"}, "faster_long_prompts"))
        self.assertFalse(W.is_on({"STRATA_KVA": "0"}, "faster_long_prompts"), "stored by the old off switch: harmless, and Off is what it is now")
        self.assertFalse(W.is_on({"STRATA_KVA": "yes"}, "faster_long_prompts"), "only 1 turns it on")
        self.assertEqual(W.set_switch({"PATH": "/bin", "STRATA_KVA": "0"}, "faster_long_prompts", False), {"PATH": "/bin"}, "choosing Off cleans it up")


class Terminal(unittest.TestCase):
    """`strata launch`: the run config a launch writes."""

    def config(self, base, extra=None):
        return derive_config(base, S.LaunchSettings(extra=extra or {}), bool_flags=frozenset())

    def test_a_default_launch_has_none_of_the_new_entries(self):
        env = self.config(QWEN).get("env") or {}
        for name in W.env_names():
            self.assertNotIn(name, env, name)
        self.assertEqual(env, QWEN["env"], "the environment is exactly the installed one")
        self.assertEqual(json.dumps(self.config(QWEN), sort_keys=True), json.dumps(self.config(QWEN, {}), sort_keys=True))

    def test_each_switch_outside_its_default_by_set_gives_exactly_its_entries(self):
        for sid, entries in ENV.items():          # --set STRATA_KVA=1 turns KVA on; the other two are turned off by their entries
            cfg = self.config(QWEN, S.settings_from_pairs([f"{k}={v}" for k, v in entries.items()]))
            self.assertEqual(cfg["env"], {**QWEN["env"], **entries}, sid)
            self.assertEqual(cfg["args"], QWEN["args"], "no engine flag changes")

    def test_off_again_puts_the_default_back(self):
        cfg = self.config(QWEN, S.settings_from_pairs(["STRATA_KVA=off"]))
        self.assertEqual(cfg["env"], QWEN["env"])
        turned_on = {**QWEN, "env": {**QWEN["env"], "STRATA_KVA": "1"}}
        self.assertEqual(self.config(turned_on, S.settings_from_pairs(["STRATA_KVA=off"]))["env"], QWEN["env"])
        old = {**QWEN, "env": {**QWEN["env"], "STRATA_KVA": "0"}}
        self.assertEqual(self.config(old, S.settings_from_pairs(["STRATA_KVA=off"]))["env"], QWEN["env"])

    def test_the_advanced_table_lists_only_the_switches_that_apply(self):
        def names(cfg, gpus, extra=None):
            rows = P.build_rows(None, cfg["args"], None, cfg["env"], None, extra or {}, P.switch_params(W.applicable(cfg, gpus)))
            return [r.name for r in rows if r.name in W.env_names()]
        # (STRATA_DEFAULTS, the Qwen "0.2.6 behaviour" row, closes every Qwen table)
        self.assertEqual(names(QWEN, [RTX]), ["STRATA_KVA", "STRATA_DEFAULTS"])
        self.assertEqual(names(QWEN, [RTX3060]), ["STRATA_KVA", "STRATA_ROUTE_PRIOR", "STRATA_DEFAULTS"])
        self.assertEqual(names({**QWEN, "backend": "hip"}, [R9700]), ["STRATA_KVA", "STRATA_PCIE_FRAC_DEFAULT", "STRATA_DEFAULTS"])
        self.assertEqual(names({**QWEN, "backend": "sycl"}, [R9700]), ["STRATA_DEFAULTS"])
        self.assertEqual(names(QWEN36, [RTX]), ["STRATA_DEFAULTS"])

    def test_the_qwen_0_2_6_row_in_the_advanced_table(self):
        rows = P.build_rows(None, QWEN["args"], None, QWEN["env"], None, {}, P.switch_params(W.applicable(QWEN, [RTX])))
        row = next(r for r in rows if r.name == "STRATA_DEFAULTS")
        self.assertEqual((row.default, row.yours, row.known), ("off", "", True))
        self.assertIn("Qwen: 0.2.6 behaviour", row.description)
        self.assertIn("off by default; set 026 to turn on", row.description)
        cfg = derive_config(QWEN, S.LaunchSettings(extra=S.settings_from_pairs(["STRATA_DEFAULTS=026"])), bool_flags=frozenset())
        self.assertEqual(cfg["env"], {**QWEN["env"], "STRATA_DEFAULTS": "026"})
        self.assertEqual(derive_config(QWEN, S.LaunchSettings(), bool_flags=frozenset())["env"], QWEN["env"], "a default launch carries nothing")

    def test_a_row_shows_its_default_until_it_is_changed_and_explains_itself(self):
        rows = P.build_rows(None, QWEN["args"], None, QWEN["env"], None, {}, P.switch_params(W.applicable(QWEN, [RTX])))
        row = next(r for r in rows if r.name == "STRATA_KVA")
        self.assertEqual((row.name, row.default, row.tuned, row.yours, row.known), ("STRATA_KVA", "off", "off", "", True))
        self.assertIn("Faster long prompts", row.description)
        self.assertIn("off by default; set 1 to turn on", row.description)
        self.assertIn("answers about long documents can miss or mix up details", row.description)
        on = P.build_rows(None, QWEN["args"], None, QWEN["env"], None, {"STRATA_KVA": "1"}, P.switch_params(W.applicable(QWEN, [RTX])))
        self.assertEqual(next(r for r in on if r.name == "STRATA_KVA").yours, "1")
        hip = {**QWEN, "backend": "hip"}
        radeon = next(r for r in P.build_rows(None, hip["args"], None, hip["env"], None, {}, P.switch_params(W.applicable(hip, [R9700]))) if r.name == "STRATA_PCIE_FRAC_DEFAULT")
        self.assertEqual(radeon.default, "on")
        self.assertIn("on by default; set old to turn off", radeon.description)


if __name__ == "__main__":
    unittest.main()
