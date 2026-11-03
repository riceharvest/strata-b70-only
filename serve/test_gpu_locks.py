"""serve/test_gpu_locks.py - the config's "gpu_locks" are held exactly while the model is loaded, and POST /v1/restart
unloads, then asks main() for its clean stop (after which the process starts itself again).

    python -m unittest serve.test_gpu_locks -v
"""
from __future__ import annotations

import fcntl
import os
import tempfile
import threading
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, GpuBusy, GpuLocks, Service, serve, with_config
from serve.test_lifecycle import Lifecycle, ResidentEngine


def held_elsewhere(path: str) -> bool:
    """True when another descriptor holds the flock (as `flock -n <file>` in a job script would see it)."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        os.close(fd)


class GpuLocksFollowTheModel(unittest.TestCase):
    request = Lifecycle.request

    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.paths = [str(Path(d.name) / "b70.lock"), str(Path(d.name) / "quiet.lock")]
        tok = ByteTokenizer()
        self.engine = ResidentEngine(tok)
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc.gpu_locks = GpuLocks(self.paths)
        self.addCleanup(self.svc.gpu_locks.release)
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def test_load_takes_both_and_unload_releases_them(self):
        self.assertFalse(any(map(held_elsewhere, self.paths)))
        self.assertEqual(self.request("/v1/load", {})[0], 200)
        self.assertTrue(all(map(held_elsewhere, self.paths)))
        self.assertEqual(self.request("/v1/load", {})[0], 200)          # loaded already: nothing taken twice
        self.assertEqual(len(self.svc.gpu_locks.fds), 2)
        self.assertEqual(self.request("/v1/unload", {})[0], 200)
        self.assertFalse(any(map(held_elsewhere, self.paths)))
        self.assertEqual(self.request("/v1/load", {})[0], 200)          # a reload takes them again
        self.assertTrue(all(map(held_elsewhere, self.paths)))

    def test_idle_unload_releases_them(self):
        self.svc.load()
        self.svc.last_request_at = 1.0                                   # long ago
        self.assertEqual(self.svc.unload(idle_for=1), "unloaded")
        self.assertFalse(any(map(held_elsewhere, self.paths)))

    def test_a_held_lock_is_gpu_busy_and_the_model_stays_unloaded(self):
        fd = os.open(self.paths[1], os.O_RDWR | os.O_CREAT)
        self.addCleanup(os.close, fd)
        fcntl.flock(fd, fcntl.LOCK_EX)                                  # a job holds quiet.lock
        with self.assertRaisesRegex(GpuBusy, "quiet.lock is held"):
            self.svc.load()
        self.assertFalse(self.engine.loaded)
        self.assertEqual(self.engine.starts, 0)
        self.assertFalse(held_elsewhere(self.paths[0]))                # b70.lock was given back too
        code, body = self.request("/v1/load", {})
        self.assertEqual(code, 503)
        self.assertIn("held", body["error"]["message"])

    def test_unload_after_the_engine_died_gives_them_back(self):
        self.svc.load()
        self.engine.loaded = False                                      # the engine died on its own
        self.assertEqual(self.svc.unload(), "not loaded")
        self.assertFalse(any(map(held_elsewhere, self.paths)))

    def test_a_failed_start_gives_them_back(self):
        def boom():
            raise RuntimeError("no start")
        self.engine.restart = boom
        with self.assertRaises(RuntimeError):
            self.svc.load()
        self.assertFalse(any(map(held_elsewhere, self.paths)))

    def test_restart_unloads_then_asks_for_the_clean_stop(self):
        stopped = threading.Event()
        self.svc.request_stop = lambda reason: (self.assertEqual(reason, "restart"), stopped.set())
        self.svc.load()
        code, body = self.request("/v1/restart", {})
        self.assertEqual((code, body["status"]), (200, "restarting"))
        self.assertTrue(stopped.wait(5))
        self.assertTrue(self.svc.restart_requested)
        self.assertFalse(self.engine.loaded)
        self.assertFalse(any(map(held_elsewhere, self.paths)))

    def test_force_restart_goes_ahead_while_a_turn_runs(self):
        stopped = threading.Event()
        self.svc.request_stop = lambda reason: stopped.set()
        self.svc.load()
        with self.svc.status_lock:
            self.svc.status["busy"] = True                              # a turn the engine hangs on
        self.addCleanup(lambda: self.svc.status.update(busy=False))
        code, body = self.request("/v1/restart?force=1", {})
        self.assertEqual((code, body["status"]), (200, "restarting"))
        self.assertTrue(stopped.wait(5))
        self.assertTrue(self.svc.restart_requested)

    def test_restart_refuses_while_a_turn_runs_and_from_another_site(self):
        self.svc.request_stop = lambda reason: self.fail("must not stop")
        self.svc.load()
        with self.svc.status_lock:
            self.svc.status["busy"] = True
        self.assertEqual(self.request("/v1/restart", {})[0], 409)
        with self.svc.status_lock:
            self.svc.status["busy"] = False
        self.assertEqual(self.request("/v1/restart", {}, {"Origin": "https://evil.example"})[0], 403)
        self.assertFalse(self.svc.restart_requested)
        self.assertTrue(self.engine.loaded)


class ModelSwitch(GpuLocksFollowTheModel):
    def setUp(self):
        super().setUp()
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.dir = Path(d.name)
        (self.dir / "flash-next.json").write_text('{"engine": "strata", "args": [], "model_name": "qwen38-flash-next"}')
        (self.dir / "qwen38-27b.json").write_text('{"engine": "strata", "arch": "qwen35", "args": ["--kv", "int8"]}')
        (self.dir / "flash-next.lifetime.json").write_text('{"tokens": 1}')          # not a run config
        self.svc.config_path = str(self.dir / "flash-next.json")

    def test_the_settings_view_lists_the_run_configs(self):
        code, body = self.request("/config")
        self.assertEqual(code, 200)
        self.assertEqual(body["models"], [
            {"file": "flash-next.json", "model": "qwen38-flash-next", "current": True},
            {"file": "qwen38-27b.json", "model": "qwen3.8-27b", "current": False}])

    def test_restart_with_another_config_switches_the_model(self):
        stopped = threading.Event()
        self.svc.request_stop = lambda reason: stopped.set()
        self.assertEqual(self.request("/v1/restart", {"config": "../etc.json"})[0], 404)
        self.assertEqual(self.request("/v1/restart", {"config": "flash-next.lifetime.json"})[0], 404)
        self.assertFalse(stopped.is_set())
        code, body = self.request("/v1/restart", {"config": "qwen38-27b.json"})
        self.assertEqual((code, body["status"]), (200, "restarting"))
        self.assertTrue(stopped.wait(5))
        self.assertEqual(self.svc.restart_config, str((self.dir / "qwen38-27b.json").resolve()))

    def test_with_config_rewrites_argv(self):
        self.assertEqual(with_config(["s.py", "--config", "a.json", "--port", "1"], "b.json"),
                         ["s.py", "--config", "b.json", "--port", "1"])
        self.assertEqual(with_config(["s.py", "--config=a.json"], "b.json"), ["s.py", "--config=b.json"])
        self.assertEqual(with_config(["s.py"], "b.json"), ["s.py", "--config", "b.json"])


if __name__ == "__main__":
    unittest.main()
