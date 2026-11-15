"""`strata launch` on a SYCL checkout: the engine built from source (sycl/tools/build.sh -> build-sycl-aot/strata, run by sycl/serve/strata-sycl.sh)
is the engine, with no package and no `strata update`; setup runs as `./setup.sh --backend sycl`; the runtime host starts sycl/serve/server_intel.py
for a SYCL run config, as run-<model>.sh does. Nothing here starts a server, a container or the GPU: every process start is mocked."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.runner import Runner, files_present
from strata_control import discovery, runtime, sycl_build
from strata_control import engine_component as ec

ROOT = Path(__file__).resolve().parents[2]


def checkout(base: Path, built: bool = True) -> Path:
    """A Strata folder with the SYCL launcher and server, and (built) the engine binary sycl/tools/build.sh leaves."""
    src = base / "Strata"
    (src / "sycl" / "serve").mkdir(parents=True)
    (src / "sycl" / "serve" / "strata-sycl.sh").write_text("#!/bin/sh\nexit 99\n")      # never run by these tests
    (src / "sycl" / "serve" / "server_intel.py").write_text("raise SystemExit(99)\n")
    (src / "setup.sh").write_text("#!/bin/sh\nexit 99\n")
    (src / "CMakeLists.txt").write_text("project(strata VERSION 0.1.49.2 LANGUAGES CXX)\n")
    if built:
        (src / "build-sycl-aot").mkdir()
        (src / "build-sycl-aot" / "strata").write_text("")
    return src


class FakeProc:
    def __init__(self, lines=(), code: int = 0):
        self.stdout = iter(lines)
        self._code = code

    def wait(self) -> int:
        return self._code

    def poll(self):
        return self._code


def runner(src: Path, seen: list) -> Runner:
    cat = mock.Mock(src=src, root=src)
    cat.tag_of.return_value = "Q3_S"
    cat.installed_config.return_value = None
    cat.mcp.install_progress.side_effect = lambda lines: {}
    cat.mcp.combined_progress.side_effect = lambda prog, job: {}
    r = Runner(S.LaunchSettings(family="qwen", model="Q3_S", context=32768, vision="off"), catalog=cat, emit=seen.append)
    r.check_cancel = lambda: None
    return r


class SourceBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def test_find_needs_the_binary_the_launcher_and_the_server(self):
        self.assertIsNone(sycl_build.find(checkout(self.base / "a", built=False)))
        src = checkout(self.base / "b")
        found = sycl_build.find(src)
        self.assertEqual(found["backend"], "sycl")
        self.assertEqual(found["version"], "0.1.49.2")
        self.assertEqual(found["exe"], str(src / "build-sycl-aot" / "strata"))
        self.assertEqual(found["launcher"], str(src / "sycl" / "serve" / "strata-sycl.sh"))
        (src / "sycl" / "serve" / "server_intel.py").unlink()
        self.assertIsNone(sycl_build.find(src))

    def test_the_jit_build_counts_too(self):
        src = checkout(self.base, built=False)
        (src / "build-sycl").mkdir()
        (src / "build-sycl" / "strata").write_text("")
        self.assertEqual(sycl_build.find(src)["exe"], str(src / "build-sycl" / "strata"))

    def test_setup_intel_uses_the_same_paths(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("setup_intel_paths", ROOT / "sycl" / "setup_intel.py")
        si = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(si)
        self.assertEqual(si.SERVER, sycl_build.server(ROOT))
        self.assertEqual(si.SYCL_WRAPPER, sycl_build.wrapper(ROOT))

    def test_host_paths_undo_what_setup_intel_writes(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("setup_intel_roundtrip", ROOT / "sycl" / "setup_intel.py")
        si = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(si)
        data = si.MOUNT / "strata-data"                      # the models live under the folder strata-sycl.sh mounts (nothing is created)
        host = [str(data / "packs" / "q3_s"), str(data / "models" / "Q3_S" / "m-00001-of-00002.gguf")]
        cfg = si.to_sycl({"args": ["--pack", host[0], "--native", host[1], "--max-context", "32768", "--kv", "int8"]},
                         ROOT / "build-sycl-aot" / "strata", 64.0, {}, fit=False)
        self.assertTrue(si.flag(cfg["args"], "--pack").startswith("/work/"))
        args = sycl_build.host_args(cfg)
        self.assertEqual([args[args.index("--pack") + 1], args[args.index("--native") + 1]], host)
        self.assertEqual(sycl_build.server_command(cfg), [str(ROOT / "sycl" / "serve" / "server_intel.py")])

    def test_container_paths_are_read_where_they_are_on_this_computer(self):
        cfg = {"backend": "sycl", "sycl_root": str(self.base), "args": ["--pack", "/work/data/packs/q3_s", "--native", "/work/data/models/Q3_S/m-00001-of-00002.gguf",
                                                                         "--kv", "int8"]}
        self.assertEqual(sycl_build.host_args(cfg), ["--pack", str(self.base / "data/packs/q3_s"), "--native",
                                                     str(self.base / "data/models/Q3_S/m-00001-of-00002.gguf"), "--kv", "int8"])
        self.assertEqual(sycl_build.host_path("/work/x", {"args": []}), "/work/x")          # not a SYCL config: unchanged
        self.assertEqual(len(files_present(cfg)), 2)
        (self.base / "data/packs/q3_s").mkdir(parents=True)
        (self.base / "data/models/Q3_S").mkdir(parents=True)
        (self.base / "data/models/Q3_S/m-00001-of-00002.gguf").write_text("")
        self.assertEqual(files_present(cfg), [])                 # an installed model is not set up again on every launch
        self.assertEqual(discovery.footprint(cfg).data_dir, self.base / "data")

    def test_the_engine_step_takes_the_source_build_and_fetches_nothing(self):
        seen: list = []
        r = runner(checkout(self.base), seen)
        with mock.patch.object(ec, "status", side_effect=AssertionError("no package lookup")), \
                mock.patch.object(Runner, "_spawn", side_effect=AssertionError("no strata update")):
            r.ensure_engine()
        self.assertIsNone(r.engine_problem)
        self.assertIn("Engine 0.1.49.2 (sycl) is built from source", seen[-1].text)
        self.assertEqual(seen[-1].pct, 100)

    def test_a_mocked_catalog_finds_no_source_build(self):
        r = Runner(S.LaunchSettings(family="qwen", model="Q3_S"), catalog=mock.Mock(), emit=lambda e: None)
        self.assertIsNone(r.source_engine())

    def test_setup_runs_as_setup_sh_backend_sycl_not_the_app_installer(self):
        seen: list = []
        src = checkout(self.base)
        r = runner(src, seen)
        cfg = src / "strata-q3_s.json"
        spawned: list = []

        def spawn(argv, env=None, cwd=None):
            spawned.append((argv, env, cwd))
            cfg.write_text(json.dumps({"backend": "sycl", "args": []}))      # what setup_intel writes into the Strata folder
            return FakeProc()

        env = dict(os.environ, STRATA_CONTROL_INSTALL="1", STRATA_INSTALL_DIR="/elsewhere", STRATA_NEED_ENGINE="1", STRATA_ALLOW_NO_ENGINE="1")
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(Runner, "setup_args", return_value=["--yes", "--no-start", "--family", "qwen", "--model", "Q3_S"]), \
                mock.patch.object(Runner, "progress_job", return_value={}), \
                mock.patch.object(Runner, "_spawn", side_effect=spawn):
            r.catalog.strata.describe_config.return_value = {"images": "off"}
            path = r.ensure_model()
        self.assertEqual(path, cfg)
        argv, env, cwd = spawned[0]
        self.assertEqual(argv[:2], ["/bin/sh", str(src / "setup.sh")])
        self.assertEqual(argv[-2:], ["--backend", "sycl"])
        self.assertIn("--no-start", argv)
        for k in ("STRATA_CONTROL_INSTALL", "STRATA_INSTALL_DIR", "STRATA_NEED_ENGINE", "STRATA_ALLOW_NO_ENGINE"):
            self.assertNotIn(k, env)
        self.assertEqual(cwd, str(src))

    def test_without_a_source_build_setup_keeps_the_app_installer_mode(self):
        src = checkout(self.base, built=False)
        r = runner(src, [])
        spawned: list = []
        with mock.patch.object(Runner, "setup_args", return_value=["--yes"]), \
                mock.patch.object(Runner, "progress_job", return_value={}), \
                mock.patch.object(Runner, "_spawn", side_effect=lambda argv, env=None, cwd=None: spawned.append((argv, env)) or FakeProc(code=1)), \
                mock.patch.object(Runner, "adopt_prepared_files", return_value=None):
            with self.assertRaises(Exception):
                r.ensure_model()
        argv, env = spawned[0]
        self.assertTrue(argv[1].endswith("setup.py"))
        self.assertEqual(env["STRATA_CONTROL_INSTALL"], "1")
        self.assertEqual(env["STRATA_NEED_ENGINE"], "1")

    def test_the_run_serves_with_the_source_build(self):
        seen: list = []
        r = runner(checkout(self.base), seen)
        with mock.patch.object(ec, "status", side_effect=AssertionError("no package lookup")), \
                mock.patch.object(Runner, "check_own_files"), mock.patch.object(Runner, "check_serving", return_value=None), \
                mock.patch.object(Runner, "ensure_model", return_value=self.base / "strata-q3_s.json"), \
                mock.patch.object(Runner, "ensure_drafter"), \
                mock.patch.object(Runner, "build_config", return_value=self.base / "launch.json"), \
                mock.patch.object(Runner, "ensure_profile"), mock.patch.object(Runner, "tune"), \
                mock.patch.object(Runner, "start") as start, mock.patch.object(Runner, "save_choices"), \
                mock.patch.object(Runner, "open_access", return_value=None):
            result = r.run()
        start.assert_called_once()
        self.assertTrue(result.serving)
        self.assertIsNone(result.engine_problem)

    def test_the_plan_does_not_promise_an_engine_download(self):
        r = runner(checkout(self.base), [])
        st = mock.Mock(remaining_gb=0.0, own_files=False, installed=True, on_disk_gb=0.0, parts=[])
        with mock.patch.object(Runner, "model_state", return_value=st), \
                mock.patch.object(Runner, "row", return_value=None), \
                mock.patch.object(ec, "status", side_effect=AssertionError("no package lookup")):
            plan = r.download_plan()
        self.assertTrue(plan["engine_usable"])
        self.assertNotIn("engine for your graphics card is downloaded", plan["text"])


class RuntimeServer(unittest.TestCase):
    def test_a_sycl_config_starts_server_intel_like_run_sh(self):
        cmd = runtime._server_command(json.dumps({"backend": "sycl", "exe": "/x/sycl/serve/strata-sycl.sh"}).encode())
        self.assertEqual(cmd, [str(ROOT / "sycl" / "serve" / "server_intel.py")])
        self.assertEqual(runtime._server_command(json.dumps({"exe": "component:strata-engine"}).encode()), ["-m", "serve.server"])
        self.assertEqual(runtime._server_command(b"not json"), ["-m", "serve.server"])

    @unittest.skipIf(os.name == "nt", "the POSIX host path")
    def test_the_host_runs_it_with_the_same_server_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            host = object.__new__(runtime.Host)
            host.name, host.token, host.shutdown_timeout, host.probe_timeout = "launch-q3", "t" * 64, 30, 300
            host.state_path = Path(tmp) / "launch-q3.json"
            host.owner_write_fd = None
            source = json.dumps({"backend": "sycl", "args": []}).encode()
            args = ["--engine", "strata", "--config", "/x/launch-q3.json", "--host", "127.0.0.1", "--port", "8080"]
            child = mock.Mock(pid=4242)
            with mock.patch.object(runtime.subprocess, "Popen", return_value=child) as popen, \
                    mock.patch.object(runtime.procs, "write_record"), mock.patch.object(runtime.Host, "_mark_child"):
                host._launch(args, Path("/x/launch-q3.json"), source, None)
            try:
                argv = popen.call_args.args[0]
                self.assertEqual(argv[1], str(ROOT / "sycl" / "serve" / "server_intel.py"))
                self.assertEqual(argv[2:2 + len(args)], args)
                self.assertIn("--ready-file", argv)
                self.assertIn("--owner-fd", argv)
                self.assertEqual(popen.call_args.kwargs["cwd"], runtime.ROOT)
            finally:
                if host.owner_write_fd is not None:
                    os.close(host.owner_write_fd)


if __name__ == "__main__":
    unittest.main()
