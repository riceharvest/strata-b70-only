"""Tests for setup.py --backend sycl, the experimental Intel Arc engine (sycl/, PR #423): it warns, stops with a clear
message on Windows, and on Linux hands the run to sycl/setup_intel.py without its --backend flag. The CUDA / HIP
paths never see it. No GPU, no network.

    python -m unittest tools.test_setup_sycl
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def run(argv, win):
    out = io.StringIO()
    with mock.patch.object(setup, "WIN", win), mock.patch.object(setup.subprocess, "call", return_value=0) as call, \
            contextlib.redirect_stdout(out):
        try:
            rc = setup.sycl_setup(argv)
        except SystemExit as e:
            rc = ("exit", e.code)
    return rc, out.getvalue(), call


class SyclBackend(unittest.TestCase):
    def test_windows_stops_with_the_docs_pointer(self):
        rc, out, call = run(["--backend", "sycl"], win=True)
        self.assertEqual(rc, ("exit", 1))
        self.assertIn("supported since 0.1.40.2", out)
        self.assertIn("docs/INTEL_ARC.md", out)
        call.assert_not_called()

    def test_linux_hands_over_to_setup_intel_without_the_backend_flag(self):
        rc, out, call = run(["--model", "IQ2_XS", "--backend", "sycl", "--port", "8085"], win=False)
        self.assertEqual(rc, 0)
        self.assertIn("supported since 0.1.40.2", out)
        self.assertIn("built from source", out)
        cmd = call.call_args[0][0]
        self.assertTrue(cmd[1].endswith(str(Path("sycl") / "setup_intel.py")))
        self.assertEqual(cmd[2:], ["--model", "IQ2_XS", "--port", "8085"])

    def test_backend_equals_form_is_dropped_too(self):
        _, _, call = run(["--backend=sycl", "--check"], win=False)
        self.assertEqual(call.call_args[0][0][2:], ["--check"])

    def test_every_run_goes_to_the_intel_setup(self):
        """No --backend and --backend sycl both hand over to sycl/setup_intel.py (sycl_setup); --backend hip is the
        setup_intel's own re-entry and goes on in main() (no loop)."""
        for argv in (["--check"], ["--backend", "sycl", "--check"], ["--backend=sycl", "--model", "IQ3_S"]):
            with self.subTest(argv=argv), mock.patch.object(setup, "sycl_setup", return_value=7) as sy, \
                    mock.patch.object(sys, "argv", ["setup.py", *argv]):
                self.assertEqual(setup.main(), 7)
                sy.assert_called_once_with(argv)
        with mock.patch.object(setup, "sycl_setup", side_effect=AssertionError("looped")), \
                mock.patch.object(setup, "installed_configs", side_effect=RuntimeError("main went on")), \
                mock.patch.object(sys, "argv", ["setup.py", "--backend", "hip", "--check"]), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "main went on"):
                setup.main()

    def test_the_intel_sizes_are_offered(self):
        for m in ("IQ2_XS", "IQ3_S"):
            self.assertIn(m, setup.MODELS)
            self.assertIn("qwen", setup.MODELS[m].get("families", ("qwen", "swift")))

    def test_setup_intel_finds_the_setup_functions_it_replaces(self):
        """sycl/setup_intel.py swaps these steps; setup.py must still have them (it stops with a message otherwise)."""
        src = (ROOT / "sycl" / "setup_intel.py").read_text(encoding="utf-8")
        names = re.search(r'for name in \(([^)]*)\):', src).group(1)
        for name in re.findall(r'"(\w+)"', names):
            self.assertTrue(callable(getattr(setup, name, None)), name)

    def test_setup_intel_replacements_take_every_argument_of_the_function_they_replace(self):
        """#870: setup.py grew write_run_script's fourth argument and the port's replacement still took three, so
        setup died with a TypeError after the download.  Every `S.<name> = <replacement>` in setup_intel.py must accept
        all the parameters of setup.<name> (or take *args / **kwargs)."""
        import ast
        import inspect
        tree = ast.parse((ROOT / "sycl" / "setup_intel.py").read_text(encoding="utf-8"))
        defs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        checked = 0
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Attribute)
                    and isinstance(n.targets[0].value, ast.Name) and n.targets[0].value.id == "S"):
                continue
            name = n.targets[0].attr
            fn = n.value if isinstance(n.value, ast.Lambda) else defs.get(getattr(n.value, "id", ""))
            orig = getattr(setup, name, None)
            if fn is None or not callable(orig):
                continue
            a = fn.args
            if a.vararg is not None and a.kwarg is not None:
                checked += 1
                continue
            have = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
            for param in inspect.signature(orig).parameters.values():
                if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    continue
                self.assertIn(param.name, have, f"setup_intel's {name} lacks setup.{name}'s `{param.name}`")
            checked += 1
        self.assertGreaterEqual(checked, 8)

    def test_arc_pro_b60_is_known_under_both_pci_ids(self):
        src = (ROOT / "sycl" / "setup_intel.py").read_text(encoding="utf-8")
        self.assertIn('"e211": ("Arc Pro B60", 24.0)', src)
        self.assertIn('"e221": ("Arc Pro B60", 24.0)', src)


class SyclVersion(unittest.TestCase):
    def setUp(self):
        from sycl import setup_intel
        self.mod = setup_intel

    def test_version_matches_the_source_and_satisfies_native_models(self):
        version = self.mod.sycl_version()
        self.assertNotEqual(version, "0")
        self.assertEqual(version, setup.source_version())
        numbers = tuple(int(x) for x in version.split("."))
        for model in ("UD-IQ4_XS", "UD-Q4_K_XL"):
            with self.subTest(model=model):
                self.assertGreaterEqual(numbers, setup.MODELS[model].get("engine", setup.UNSLOTH_ENGINE))

    def test_version_follows_a_source_hotfix_with_a_variable_sycl_project(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "CMakeLists.txt").write_text("project(strata VERSION 0.1.41.2 LANGUAGES CXX)\n", encoding="utf-8")
            (root / "sycl").mkdir()
            (root / "sycl" / "CMakeLists.txt").write_text(
                "set(STRATA_SYCL_VERSION 0.1.40.2)\n"
                "project(strata_sycl VERSION ${STRATA_SYCL_VERSION} LANGUAGES C CXX)\n", encoding="utf-8")
            with mock.patch.object(self.mod, "ROOT", root), mock.patch.object(setup, "ROOT", root):
                self.assertEqual(self.mod.sycl_version(), "0.1.41.2")


class IntelIds(unittest.TestCase):
    """sycl/setup_intel.py names a card by its PCI device id. The Arc Pro B60s measured on 2x B60 (`lspci -nn` 8086:e211,
    bmg-g21) must not fall through to "Intel GPU e211 (xe)" with a guessed VRAM size."""

    @staticmethod
    def table():
        import importlib.util
        spec = importlib.util.spec_from_file_location("setup_intel", ROOT / "sycl" / "setup_intel.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.INTEL_ARC

    def test_b60_e211_is_named_with_its_vram(self):
        self.assertEqual(self.table()["e211"], ("Arc Pro B60", 24.0))

    def test_the_ids_that_were_listed_before_are_kept(self):
        t = self.table()
        self.assertEqual(t["e223"][0], "Arc Pro B70")
        self.assertEqual(t["e221"][0], "Arc Pro B60")


class ToSycl(unittest.TestCase):
    """The config sycl/setup_intel.py writes: an explicit --vram-reserve-mib is kept, the B-series streams its experts,
    the A-series (i915) cannot (a single pinned host allocation above a few GB fails there) and loads them into RAM."""

    @staticmethod
    def mod():
        import importlib.util
        spec = importlib.util.spec_from_file_location("setup_intel", ROOT / "sycl" / "setup_intel.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def cfg(self, m, extra=()):
        base = m.MOUNT
        return {"exe": "x", "cwd": "y", "args": ["--pack", str(base / "d" / "pack"), "--native", str(base / "d" / "a.gguf"),
                                                 "--expert-cache", "auto", "--prefill", "auto", "--max-context", "32768",
                                                 "--kv", "int8", *extra]}

    def test_b_series_streams_and_defaults_the_reserve(self):
        m = self.mod()
        c = m.to_sycl(self.cfg(m), m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe")
        self.assertIn("--stream-experts", c["args"])
        self.assertEqual(c["args"][c["args"].index("--vram-reserve-mib") + 1], "1024")
        self.assertNotIn("STRATA_VERIFY_NO_HOST", c.get("env", {}))

    def test_xe_ranks_the_experts_with_the_sycl_profile_and_i915_keeps_the_shipped_one(self):
        m = self.mod()
        shipped = str(m.MOUNT / "data" / "expert-profile.bin")
        other = str(m.MOUNT / "mine" / "learned.bin")
        c = m.to_sycl(self.cfg(m, ["--expert-profile", shipped]), m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe")
        self.assertTrue(c["args"][c["args"].index("--expert-profile") + 1].endswith("expert-profile-sycl.bin"))
        c = m.to_sycl(self.cfg(m, ["--expert-profile", other]), m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe")
        self.assertTrue(c["args"][c["args"].index("--expert-profile") + 1].endswith("learned.bin"))   # the user's own file stays
        c = m.to_sycl(self.cfg(m, ["--expert-profile", shipped]), m.ROOT / "build-sycl" / "strata", 64.0, {}, 8.0, "i915")
        self.assertTrue(c["args"][c["args"].index("--expert-profile") + 1].endswith("expert-profile.bin"))   # the CPU computes the misses there

    def test_a_long_context_drafts_over_the_last_8192_cells_and_asks_for_the_code_subset(self):
        m = self.mod()
        mtp = ["--mtp", str(m.MOUNT / "d" / "mtp" / "rt")]
        long = self.cfg(m, mtp)
        long["args"][long["args"].index("--max-context") + 1] = "262144"
        c = m.to_sycl(long, m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe", fit=False)
        self.assertEqual(c["args"][c["args"].index("--mtp-window") + 1], "8192")
        c = m.to_sycl(self.cfg(m, mtp), m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe", fit=False)
        self.assertNotIn("--mtp-window", c["args"])                      # 32K: the engine's own window
        mine = self.cfg(m, [*mtp, "--mtp-window", "4096"])
        mine["args"][mine["args"].index("--max-context") + 1] = "262144"
        c = m.to_sycl(mine, m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe", fit=False)
        self.assertEqual(c["args"].count("--mtp-window"), 1)
        self.assertEqual(c["args"][c["args"].index("--mtp-window") + 1], "4096")
        self.assertEqual(m.asked_context(["--model", "IQ2_XS", "--context", "262144"]), 262144)
        self.assertEqual(m.asked_context(["--context=131072"]), 131072)
        self.assertEqual(m.asked_context(["--model", "IQ2_XS"]), 0)

    def test_an_explicit_reserve_is_kept_on_both_drivers(self):
        m = self.mod()
        for vram, drv in ((32.0, "xe"), (8.0, "i915")):
            c = m.to_sycl(self.cfg(m, ["--vram-reserve-mib", "500"]), m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, vram, drv)
            self.assertEqual(c["args"].count("--vram-reserve-mib"), 1)
            self.assertEqual(c["args"][c["args"].index("--vram-reserve-mib") + 1], "500")

    def test_the_ngram_table_stays_on_the_ssd_unless_the_real_ram_holds_it(self):
        # setup's rule (16 GB to spare beyond the model and the table) saw this route's made-up RAM size and put --ple-io ram
        # into every config: 29 GB pinned on a 64 GB PC beside the experts' mirror
        m = self.mod()
        exe = m.ROOT / "build-sycl-aot" / "strata"
        notes = []
        c = m.to_sycl(self.cfg(m, ["--ple-io", "ram"]), exe, 63.0, {}, 32.0, "xe", model="unsloth-UD-IQ1_S", fit_notes=notes)
        self.assertNotIn("--ple-io", c["args"])
        self.assertIn("--stream-experts", c["args"])
        self.assertTrue(any("n-gram table is read from the SSD" in n for n in notes), notes)
        c = m.to_sycl(self.cfg(m, ["--ple-io", "ram"]), exe, 128.0, {}, 32.0, "xe", model="unsloth-UD-IQ1_S")
        self.assertEqual(c["args"][c["args"].index("--ple-io") + 1], "ram")       # 40 + 28.8 + 16 GB fits
        mine = {**self.cfg(m, ["--ple-io", "ram"]), "user_args": ["--ple-io", "ram"]}
        c = m.to_sycl(mine, exe, 63.0, {}, 32.0, "xe", model="unsloth-UD-IQ1_S")
        self.assertEqual(c["args"][c["args"].index("--ple-io") + 1], "ram")       # the person's own choice is kept
        self.assertEqual(m.ple_ram_need("UD-IQ1_S"), m.ple_ram_need("unsloth-UD-IQ1_S"))

    def test_a_series_loads_experts_into_ram_and_leaves_no_host_off(self):
        m = self.mod()
        c = m.to_sycl(self.cfg(m), m.ROOT / "build-sycl" / "strata", 64.0, {}, 8.0, "i915")
        self.assertNotIn("--stream-experts", c["args"])
        self.assertEqual(c["args"][c["args"].index("--ple-io") + 1], "ram")
        self.assertEqual(c["args"][c["args"].index("--vram-reserve-mib") + 1], "300")
        self.assertEqual(c["env"]["STRATA_VERIFY_NO_HOST"], "0")
        self.assertEqual(c["env"]["STRATA_SYCL_BIN"], "build-sycl/strata")

    def test_gyro_memory_warning_reads_the_real_ram(self):
        # setup's RAM rule is fed 1024 GB on the Intel path; Gyro counts VRAM + RAM, so it must see the real RAM
        m = self.mod()
        import importlib.util
        spec = importlib.util.spec_from_file_location("setup_for_test", ROOT / "setup.py")
        s = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(s)
        have = m.real_mem_have(s.mem_have, 16.0, 1024.0)
        self.assertEqual(have("Gyro-S", 1024.0, 8.0), 24.0)              # 16 GB RAM + 8 GB VRAM: below Gyro-S's 27.6, setup warns
        self.assertLess(have("Gyro-S", 1024.0, 8.0), s.MODELS["Gyro-S"]["ram_gb"])
        self.assertEqual(have("Gyro-S", 40.0, 8.0), 48.0)                # a real RAM figure passes through
        self.assertEqual(have("Q2_0", 1024.0, 8.0), 16.0)                # the other sizes count RAM only, as before

    def test_a_series_keeps_gyros_ngram_table_on_the_ssd(self):
        # Gyro's table is half of the 58-92 GB file: --ple-io ram would not fit beside the experts' arena on a 64 GB PC
        m = self.mod()
        cfg = self.cfg(m)
        cfg["args"][cfg["args"].index("--native") + 1] = str(m.MOUNT / "d" / "Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf")
        c = m.to_sycl(cfg, m.ROOT / "build-sycl" / "strata", 64.0, {}, 8.0, "i915")
        self.assertNotIn("--ple-io", c["args"])
        self.assertEqual(c["env"]["STRATA_VERIFY_NO_HOST"], "0")

    def test_split_short_rule(self):
        # #1980: 2x B60 24 GB - IQ3_XXS (42.9 GB) was refused under NO_HOST=1, IQ2_XS (35.5 GB) loads fully
        m = self.mod()
        self.assertTrue(m.split_likely_short(42.9, [24.0, 24.0]))
        self.assertFalse(m.split_likely_short(35.5, [24.0, 24.0]))
        self.assertFalse(m.split_likely_short(50.3, [32.0, 32.0]))   # IQ3_S on 2x B70: 79%
        self.assertTrue(m.split_likely_short(77.0, [32.0, 32.0]))
        self.assertFalse(m.split_likely_short(0, [24.0]))
        self.assertFalse(m.split_likely_short(40.0, []))

    def test_small_card_rule(self):
        m = self.mod()
        self.assertTrue(m.small_card(8.0, "i915"))
        self.assertTrue(m.small_card(24.0, "i915"))
        self.assertTrue(m.small_card(10.0, "xe"))
        self.assertFalse(m.small_card(32.0, "xe"))

    def test_pictures_follow_the_one_reader_rule_and_the_vision_section_survives(self):
        """Intel Arc: pictures on gives the CPU reader (strata-vision) on the host under 24 GB and, until a 24 GB+ Arc has run the SYCL reader,
        also from 24 GB; only --vision gpu asks for the SYCL reader (strata_control/vision_reader.py is the one place for the rule)."""
        m = self.mod()
        for asked in (None, "yes", "cpu", "on"):
            self.assertEqual(m.intel_vision(asked, 8.0), "cpu")
            self.assertEqual(m.intel_vision(asked, 32.0), "cpu")
        self.assertEqual(m.intel_vision("gpu", 8.0), "gpu")
        self.assertEqual(m.intel_vision("gpu", 32.0), "gpu")
        for asked in ("no", "none", "off"):
            self.assertEqual(m.intel_vision(asked, 32.0), "none")
        section = {"exe": "/h/engine-sycl/strata-vision", "mmproj": "/h/models/mmproj.gguf", "model": "/h/models/m.gguf", "gpu": False,
                   "max_tokens": 300}
        cfg = self.cfg(m, ["--vision"])
        cfg["vision"] = section
        c = m.to_sycl(cfg, m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe")
        self.assertEqual(c["vision"], section)                              # host paths: the reader runs on the host, not in the container
        self.assertIn("--vision", c["args"])

    def test_pictures_are_on_by_default_on_every_arc_and_a_battlemage_is_told_it_is_new(self):
        """Owner decision 2026-10-22: an Arc B-series card runs like any Arc. The first picture on the Arc Pro B70 reset the GPU (xe, 2026-10-19) is a
        known risk, not a refusal: pictures are on (the CPU reader) and setup prints one plain line about it."""
        m = self.mod()
        for driver, devid in (("i915", "56a1"), ("i915", ""), ("xe", "56a0"), ("xe", "e223"), ("xe", "e20b"), ("xe", "e211"), ("xe", "")):
            self.assertTrue(m.pictures_on_by_default(driver, devid), (driver, devid))
        self.assertTrue(m.is_b_series("xe", "e223"))                        # B70
        self.assertTrue(m.is_b_series("xe", "e20b"))                        # B580
        self.assertFalse(m.is_b_series("i915", "56a1"))                     # A750
        self.assertFalse(m.is_b_series("xe", "56a0"))                       # A770 on xe
        self.assertIn("new to Strata", m.B_SERIES_LINE)
        self.assertIn("issues", m.B_SERIES_LINE)
        src = (ROOT / "sycl" / "setup_intel.py").read_text(encoding="utf-8")
        self.assertIn("B_SERIES_LINE", src)
        self.assertNotIn('sys.argv += ["--vision", "none"]', src)          # no B-series card has pictures switched off for it
        self.assertIn("S.hip_vision = intel_vision", src)
        self.assertIn("S.build_vision_cpu(", src)                           # the CPU reader is compiled by setup's own step
        self.assertIn("build_vision_sycl(", src)                            # --vision gpu: the SYCL reader, when the oneAPI compiler is there

    def test_the_sycl_reader_is_built_only_when_asked_and_the_cpu_reader_stays_without_the_compiler(self):
        from unittest import mock
        m = self.mod()
        with mock.patch.object(m, "oneapi_compiler", return_value=None), mock.patch.object(m.S, "warn") as warn:
            self.assertFalse(m.build_vision_sycl_reader(Path("/nonexistent"), "/llama"))
        self.assertIn("oneAPI compiler", warn.call_args[0][0])
        self.assertIn("pictures are read on the CPU", warn.call_args[0][0])

    def test_the_container_shares_dev_shm_with_the_server_for_the_embeddings(self):
        sh = (ROOT / "sycl" / "serve" / "strata-sycl.sh").read_text(encoding="utf-8")
        self.assertIn("-v /dev/shm:/dev/shm", sh)
        self.assertIn("STRATA_SYCL_SHM", sh)

    def test_a_picture_that_cannot_reach_the_container_is_said_once(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("server_intel_t", ROOT / "sycl" / "serve" / "server_intel.py")
        src = (ROOT / "sycl" / "serve" / "server_intel.py").read_text(encoding="utf-8")
        self.assertIn("install_shared_embeddings()", src.split('if __name__ == "__main__":')[1])
        sys.path.insert(0, str(ROOT / "sycl" / "serve"))
        try:
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
        except ImportError as e:                                           # no oneAPI / xe telemetry import here: the source check above stands
            self.skipTest(str(e))
        finally:
            sys.path.pop(0)
        out = io.StringIO()
        with mock.patch.object(m.S, "combined_embeddings_path", lambda d, n: Path("/tmp/x/req.sve")), contextlib.redirect_stdout(out):
            m.install_shared_embeddings()
            m.S.combined_embeddings_path(Path("/tmp/x"), 10)
            m.S.combined_embeddings_path(Path("/tmp/x"), 10)
        self.assertEqual(out.getvalue().count("is not where the engine's container can read it"), 1)


def load_setup_intel():
    import importlib.util
    spec = importlib.util.spec_from_file_location("setup_intel_t", ROOT / "sycl" / "setup_intel.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class ServedModel(unittest.TestCase):
    """Unsloth's UD-IQ1_S: the model this fork serves on a B70 (pins, the revision of its own, the layer 18 graft)."""

    def test_the_files_are_pinned_at_their_own_revision(self):
        fam = setup.FAMILIES["unsloth"]
        self.assertEqual(setup.model_shards(fam, "UD-IQ1_S"), 3)
        names = [setup.model_file(fam, "UD-IQ1_S", i) for i in (1, 2, 3)]
        self.assertEqual(names[1], "Qwen3.8-Flash-Next-UD-IQ1_S-00002-of-00003.gguf")
        pins = setup.REPO_PINS["unsloth/Qwen3.8-Flash-Next-GGUF"]
        for n in names:
            self.assertIn(n, pins)
            self.assertIn(n, fam["sha256"])
            self.assertRegex(pins[n][1], r"^[0-9a-f]{64}$")
        self.assertIn("/resolve/" + setup.UNSLOTH_IQ1_S_REVISION + "/UD-IQ1_S/", setup.model_hf(fam, "UD-IQ1_S"))
        # the other sizes stay at the family's revision
        self.assertIn("/resolve/" + setup.HF_REVISIONS["unsloth/Qwen3.8-Flash-Next-GGUF"] + "/UD-IQ4_XS/", setup.model_hf(fam, "UD-IQ4_XS"))
        self.assertIn("/resolve/" + setup.HF_REVISIONS["ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"] + "/", setup.model_hf(setup.FAMILIES["qwen"], "IQ2_XS"))

    def test_the_graft_names_a_shard_of_the_model_and_keeps_its_size(self):
        g = setup.MODELS["UD-IQ1_S"]["graft"]
        fam = setup.FAMILIES["unsloth"]
        self.assertEqual(g["result"][0], setup.REPO_PINS["unsloth/Qwen3.8-Flash-Next-GGUF"][g["shard"]][0])
        self.assertEqual(g["compacted_size"], g["result"][0] - 52428800)      # the old IQ2_S slot minus the IQ2_XXS tensor
        self.assertNotEqual(g["result"][1], setup.REPO_PINS["unsloth/Qwen3.8-Flash-Next-GGUF"][g["shard"]][1])
        self.assertEqual(setup.shard_pin("UD-IQ1_S", fam, Path("/x") / g["shard"]), tuple(g["result"]))
        other = setup.model_file(fam, "UD-IQ1_S", 3)
        self.assertEqual(setup.shard_pin("UD-IQ1_S", fam, Path("/x") / other), setup.REPO_PINS["unsloth/Qwen3.8-Flash-Next-GGUF"][other])
        self.assertIsNone(setup.MODELS["UD-IQ4_XS"].get("graft"))

    def test_the_model_is_a_choice_of_the_unsloth_family_with_the_served_context(self):
        d = setup.MODELS["UD-IQ1_S"]
        self.assertEqual(d["families"], ("unsloth",))
        self.assertEqual(d["rec_ctx"], 262144)
        self.assertTrue(d["ple_gguf"])
        self.assertFalse(d.get("budget"))


class GraftStep(unittest.TestCase):
    """setup.graft_step on tiny files: the graft, the check against the pinned result, the finish mark, a rerun, a file of the person's own."""

    def setUp(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import graft_tensor
        import test_graft_tensor as T
        self.G, self.T = graft_tensor, T
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        (self.dir / "src").mkdir()
        (self.dir / "src" / "source.gguf").write_bytes(T.source_bytes(lead=True))
        self.pristine = T.target_bytes()
        done = self.dir / "expected.gguf"
        done.write_bytes(self.pristine)
        graft_tensor.graft(done, T.NAME, self.dir / "src" / "source.gguf", log=lambda *_: None)
        import hashlib
        self.expected = (done.stat().st_size, hashlib.sha256(done.read_bytes()).hexdigest())
        compacted = self.dir / "compacted.gguf"
        compacted.write_bytes(done.read_bytes())
        graft_tensor.compact(compacted, log=lambda *_: None)
        self.compacted_size = compacted.stat().st_size
        self.shard = self.dir / "shard-2.gguf"
        self.shard.write_bytes(self.pristine)
        self.graft = {"shard": "shard-2.gguf", "tensor": T.NAME, "type": "IQ2_XXS", "source_model": "UD-IQ1_M", "source_file": "source.gguf",
                      "source_sha256": __import__("hashlib").sha256(T.NEW).hexdigest(), "result": self.expected,
                      "compacted_size": self.compacted_size}
        self.fam = setup.FAMILIES["unsloth"]

    def step(self, own=False, graft=None):
        out = io.StringIO()
        with mock.patch.dict(setup.MODELS["UD-IQ1_S"], {"graft": graft or self.graft}), \
                mock.patch.object(setup, "hf_folder", lambda fam, q, rev=None: str(self.dir / "src") + "/"), \
                contextlib.redirect_stdout(out):
            setup.graft_step("UD-IQ1_S", self.fam, [self.dir / "shard-1.gguf", self.shard], own=own)
        return out.getvalue()

    def test_the_shard_is_grafted_checked_and_marked(self):
        out = self.step()
        self.assertEqual(self.shard.read_bytes(), (self.dir / "expected.gguf").read_bytes())
        self.assertIn(self.expected[1], self.shard.with_name("shard-2.gguf.done").read_text())
        self.assertIn("has blk.1.ffn_gate_exps.weight from UD-IQ1_M", out)

    def test_a_second_run_does_nothing(self):
        self.step()
        with mock.patch.object(self.G, "graft", side_effect=AssertionError("grafted again")):
            self.assertIn("already has", self.step())

    def test_a_run_that_stopped_after_the_graft_only_checks_and_marks(self):
        self.shard.write_bytes((self.dir / "expected.gguf").read_bytes())
        with mock.patch.object(self.G, "graft", side_effect=AssertionError("grafted again")):
            self.step()
        self.assertIn(self.expected[1], self.shard.with_name("shard-2.gguf.done").read_text())

    def test_a_wrong_source_leaves_the_shard_as_it_was(self):
        bad = {**self.graft, "source_sha256": "0" * 64}
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
            self.step(graft=bad)
        self.assertEqual(self.shard.read_bytes(), self.pristine)

    def test_a_wrong_result_is_said_and_the_file_is_not_deleted(self):
        bad = {**self.graft, "result": (self.expected[0], "0" * 64)}
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
            self.step(graft=bad)
        self.assertTrue(self.shard.exists())

    def test_a_file_of_the_persons_own_is_never_changed(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
            self.step(own=True)
        self.assertEqual(self.shard.read_bytes(), self.pristine)

    def test_a_hard_linked_shard_is_copied_first_and_the_other_link_is_untouched(self):
        other = self.dir / "other.gguf"
        other.hardlink_to(self.shard)
        self.step()
        self.assertEqual(other.read_bytes(), self.pristine)
        self.assertEqual(self.shard.read_bytes(), (self.dir / "expected.gguf").read_bytes())
        self.assertEqual(self.shard.stat().st_nlink, 1)

    def test_a_linked_shard_with_no_room_for_a_copy_stops_with_the_command(self):
        (self.dir / "other.gguf").hardlink_to(self.shard)
        with mock.patch.object(setup.shutil, "disk_usage", return_value=mock.Mock(free=10)), \
                self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()) as out:
            self.step()
        self.assertEqual(self.shard.read_bytes(), self.pristine)

    def test_a_cut_short_shard_is_left_for_check_shards_to_name(self):
        self.shard.write_bytes(self.pristine[:100])
        self.assertEqual(self.step(), "")                          # no traceback, no failure here
        self.assertEqual(self.shard.read_bytes(), self.pristine[:100])

    def test_a_compacted_shard_is_recognised_by_size_and_header_and_not_hashed(self):
        self.shard.write_bytes((self.dir / "compacted.gguf").read_bytes())
        self.shard.with_name("shard-2.gguf.done").write_text("sha256 " + self.expected[1])     # the mark of the graft run
        out = self.step()
        self.assertIn("is compacted", out)
        with mock.patch.dict(setup.MODELS["UD-IQ1_S"], {"graft": self.graft}):
            self.assertIsNone(setup.shard_pin("UD-IQ1_S", self.fam, self.shard))
            self.shard.write_bytes((self.dir / "expected.gguf").read_bytes())
            self.assertEqual(setup.shard_pin("UD-IQ1_S", self.fam, self.shard), tuple(self.expected))

    def test_a_file_of_the_right_size_with_holes_is_not_taken_for_a_compacted_one(self):
        self.shard.write_bytes(b"\0" * self.compacted_size)
        self.assertEqual(self.step(), "")

    def test_an_interrupted_compact_says_to_run_it_again_not_to_download(self):
        self.shard.write_bytes((self.dir / "expected.gguf").read_bytes())
        self.shard.with_name("shard-2.gguf.compact").write_text("{}")
        said = []
        with self.assertRaises(SystemExit), mock.patch.object(setup, "say", said.append):
            self.step()
        text = "\n".join(said)
        self.assertIn("--compact", text)
        self.assertNotIn("download it again", text)
        self.assertTrue(self.shard.exists())

    def test_a_model_without_a_graft_is_left_alone(self):
        setup.graft_step("UD-IQ4_XS", self.fam, [self.shard])
        self.assertEqual(self.shard.read_bytes(), self.pristine)


class ServedRecipe(unittest.TestCase):
    """sycl/setup_intel.py: the config for UD-IQ1_S on one B70 is the served one; every other choice is as it was."""

    @classmethod
    def setUpClass(cls):
        cls.si = load_setup_intel()

    def cfg(self, **over):
        si = self.si
        args = ["--pack", str(si.ROOT.parent / "d" / "pack"), "--native", str(si.ROOT.parent / "d" / "a.gguf"),
                "--ple-gguf", str(si.ROOT.parent / "d" / "c.gguf"), "--expert-profile", str(si.ROOT / "data" / "expert-profile.bin"),
                "--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
                "--mtp", str(si.ROOT.parent / "d" / "mtp" / "rt"), "--max-context", "262144", "--kv", "int8"]
        return {"exe": "x", "args": args, **over}

    def make(self, model="UD-IQ1_S", served=True, cfg=None, **kw):
        notes = []
        out = self.si.to_sycl(cfg or self.cfg(), self.si.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe", "e223",
                              model=model, served=served, served_notes=notes, **kw)
        return out, notes

    def flag(self, out, name):
        return self.si.flag(out["args"], name)

    def test_the_served_settings(self):
        out, notes = self.make()
        self.assertEqual(self.flag(out, "--spec"), "6")
        self.assertEqual(self.flag(out, "--spec-min-p"), "0.5")
        self.assertEqual(self.flag(out, "--max-context"), "262144")
        self.assertEqual(self.flag(out, "--kv"), "int8")
        self.assertEqual(self.flag(out, "--kv-resident"), "32768")
        self.assertEqual(self.flag(out, "--prefill"), "16384")
        self.assertEqual(self.flag(out, "--mtp-window"), "8192")
        self.assertEqual(self.flag(out, "--expert-cache"), "auto")
        self.assertIn("--stream-experts", out["args"])
        self.assertTrue(self.flag(out, "--expert-profile").endswith("/data/expert-profile-pi.bin"))
        self.assertTrue(self.flag(out, "--mtp-draft-vocab").endswith("/data/draft_vocab_code.bin"))
        self.assertTrue(self.flag(out, "--mtp-draft-vocab").startswith("/work/"))
        self.assertEqual(out["env"], {"STRATA_SPEC_PROBE_MIN": "1", "STRATA_IQ_XXS_SGNMUL": "1", "STRATA_KREG2_R": "2,4,4,4,2,2,2,2"})
        self.assertEqual(out["exe"], str(self.si.SYCL_WRAPPER))
        self.assertTrue(notes)

    def test_setup_hands_the_config_tag_not_the_size_name(self):
        """write_run_script is called with the tag (unsloth-UD-IQ1_S), as the config and the start script are named."""
        out, notes = self.make(model="unsloth-UD-IQ1_S")
        self.assertEqual(self.flag(out, "--spec"), "6")
        self.assertTrue(notes)
        other, notes = self.make(model="unsloth-UD-IQ4_XS")
        self.assertEqual(self.flag(other, "--spec"), "4")
        self.assertEqual(notes, [])

    def test_no_machine_specific_setting_is_written(self):
        out, _ = self.make()
        text = repr({"args": out["args"], "env": out["env"]})
        for word in ("gpu_locks", "LEND_MIRROR", "USAGE_DUMP", "spill", "/home/"):
            self.assertNotIn(word, text)

    def test_the_recipe_is_off_for_another_model_and_when_not_asked_for(self):
        plain, notes = self.make(served=False)
        self.assertEqual(self.flag(plain, "--spec"), "4")
        self.assertNotIn("--mtp-draft-vocab", plain["args"])
        self.assertIsNone(plain.get("env"))
        self.assertEqual(notes, [])
        other, notes = self.make(model="UD-IQ4_XS")
        self.assertEqual(self.flag(other, "--spec"), "4")
        self.assertNotIn("--mtp-draft-vocab", other["args"])
        self.assertIsNone(other.get("env"))
        self.assertEqual(notes, [])

    def test_what_the_person_chose_is_kept(self):
        cfg = self.cfg()
        cfg["args"][cfg["args"].index("--spec") + 1] = "8"
        cfg["args"][cfg["args"].index("--expert-profile") + 1] = str(self.si.ROOT.parent / "mine.bin")
        cfg["args"] += ["--mtp-draft-vocab", str(self.si.ROOT.parent / "v.bin")]
        out, _ = self.make(cfg=cfg)
        self.assertEqual(self.flag(out, "--spec"), "8")
        self.assertEqual(self.flag(out, "--expert-profile"), "/work/mine.bin")
        self.assertEqual(self.flag(out, "--mtp-draft-vocab"), str(self.si.ROOT.parent / "v.bin"))
        asked, _ = self.make(draft_asked=True)
        self.assertNotIn("--mtp-draft-vocab", asked["args"])
        env = {"STRATA_KREG2_R": "1"}
        self.si.apply_served_recipe(self.cfg()["args"], env)
        self.assertEqual(env["STRATA_KREG2_R"], "1")

    def test_the_b70_on_a_64_gb_pc_gets_the_served_model_preselected(self):
        si = self.si
        b70 = [{"devid": "e223", "vram_gb": 32.0, "driver": "xe"}]
        with mock.patch.object(setup, "tier_pick", setup.tier_pick), mock.patch.object(setup, "RECOMMENDED_SIZES", dict(setup.RECOMMENDED_SIZES)):
            self.assertEqual(setup.tier_pick(1024.0, 32.0), ("qwen", "IQ3_S"))
            self.assertEqual(si.prefer_served(b70, 62.0, ["--check"]), ["--check"])
            self.assertEqual(setup.tier_pick(1024.0, 32.0), (si.SERVED_FAMILY, si.SERVED_MODEL))
            names = [m for m in setup.MODELS if "unsloth" in setup.MODELS[m].get("families", ("qwen", "swift"))]
            self.assertEqual(setup.recommended_size("unsloth", names, 1024.0, 32.0), "UD-IQ1_S")
            self.assertEqual(si.prefer_served(b70, 62.0, ["--model", "UD-IQ1_S"]), ["--model", "UD-IQ1_S", "--vision", "no"])
            self.assertEqual(si.prefer_served(b70, 62.0, ["--model", "UD-IQ1_S", "--vision", "yes"]), ["--model", "UD-IQ1_S", "--vision", "yes"])

    def test_another_card_or_less_ram_keeps_the_old_choice(self):
        si = self.si
        for intel, ram in (([{"devid": "e221", "vram_gb": 24.0, "driver": "xe"}], 62.0),
                           ([{"devid": "e223", "vram_gb": 32.0, "driver": "xe"}], 32.0),
                           ([{"devid": "e223", "vram_gb": 32.0}, {"devid": "e223", "vram_gb": 32.0}], 62.0)):
            with self.subTest(ram=ram, cards=len(intel)), mock.patch.object(setup, "tier_pick", setup.tier_pick), \
                    mock.patch.object(setup, "RECOMMENDED_SIZES", dict(setup.RECOMMENDED_SIZES)):
                self.assertEqual(si.prefer_served(intel, ram, ["--model", "UD-IQ1_S"]), ["--model", "UD-IQ1_S"])
                self.assertEqual(setup.tier_pick(1024.0, 32.0), ("qwen", "IQ3_S"))

    def test_a_rerun_does_not_read_the_recipe_flags_as_the_persons_own(self):
        from strata_control import user_args
        cfg = self.cfg()
        cfg["user_args"] = ["--vram-reserve-mib", "700"]
        user_args.stamp(cfg)                                          # what setup does before to_sycl rewrites the args
        out, _ = self.make(cfg=cfg)
        self.assertEqual(out["args_written"], out["args"])
        self.assertEqual(user_args.capture(out, setup.SETUP_CHOICE_FLAGS), ["--vram-reserve-mib", "700"])

    def test_an_env_entry_the_person_set_survives_a_rerun_and_a_recipe_one_does_not_stay_without_the_recipe(self):
        old = {"STRATA_KREG2_R": "1", "STRATA_MY_SWITCH": "x", "STRATA_SPEC_PROBE_MIN": "1", "STRATA_SYCL_BIN": "stale",
               "IGC_EnableDPEmulation": "1", "STRATA_ROUTE_TAIL_SKIP": "0"}
        cfg = self.cfg(env=old, model_name="qwen3.8-flash-next-unsloth-ud-iq1_s")
        out, _ = self.make(cfg=cfg)
        self.assertEqual(out["env"]["STRATA_KREG2_R"], "1")                       # the person's value beats the recipe's
        self.assertEqual(out["env"]["STRATA_MY_SWITCH"], "x")
        self.assertEqual(out["env"]["STRATA_SPEC_PROBE_MIN"], "1")
        self.assertNotIn("STRATA_SYCL_BIN", out["env"])                           # what setup writes itself is regenerated
        self.assertNotIn("IGC_EnableDPEmulation", out["env"])
        self.assertNotIn("STRATA_ROUTE_TAIL_SKIP", out["env"])                    # setup's own entry of an Unsloth config
        off, _ = self.make(served=False, cfg=self.cfg(env={**self.si.SERVED_ENV, "STRATA_MY_SWITCH": "x"}))
        self.assertEqual(off["env"], {"STRATA_MY_SWITCH": "x"})                   # the recipe's values of an earlier run are gone
        gone, _ = self.make(served=False, cfg=self.cfg(env=dict(self.si.SERVED_ENV)))
        self.assertIsNone(gone.get("env"))

    def test_the_recipe_is_for_exactly_one_b70(self):
        si = self.si
        b70 = {"devid": "e223"}
        self.assertTrue(si.served_applies([b70], "e223", True))
        self.assertFalse(si.served_applies([b70, b70], "e223", True))             # a layer split keeps its two-card --spec 2 rule
        self.assertFalse(si.served_applies([{"devid": "e221"}], "e221", True))
        self.assertFalse(si.served_applies([b70], "e223", False))

    def test_the_model_and_vision_options_in_the_equals_form_are_seen(self):
        si = self.si
        b70 = [{"devid": "e223", "vram_gb": 32.0, "driver": "xe"}]
        with mock.patch.object(setup, "tier_pick", setup.tier_pick), mock.patch.object(setup, "RECOMMENDED_SIZES", dict(setup.RECOMMENDED_SIZES)):
            self.assertEqual(si.prefer_served(b70, 62.0, ["--model=UD-IQ1_S"]), ["--model=UD-IQ1_S", "--vision", "no"])
            self.assertEqual(si.prefer_served(b70, 62.0, ["--model", "UD-IQ1_S", "--vision=yes"]), ["--model", "UD-IQ1_S", "--vision=yes"])

    def test_the_off_switch_is_taken_out_before_setup_sees_it(self):
        src = (ROOT / "sycl" / "setup_intel.py").read_text(encoding="utf-8")
        self.assertIn("NO_RECIPE_FLAG not in argv", src)
        self.assertIn("a != NO_RECIPE_FLAG", src)


if __name__ == "__main__":
    unittest.main()
