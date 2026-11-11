"""Serve-side robustness fixes taken from upstream: #4792 a failed engine start when another thread closed the process,
and (below) a streaming client that stops reading.  The fake engine is built inline."""
import unittest
from unittest import mock

from serve.server import StrataEngine


class FailedStart(unittest.TestCase):
    """#4792 (PR by a community author; the fix is written in our tree): a start that ends before READY must not crash
    when another thread closed the process (set self.proc to None) while the failed start cleaned up."""

    def test_close_from_another_thread_during_a_failed_start(self):
        import serve.server as server
        holder = {}

        class Pipe:
            def close(self):
                pass

            def __iter__(self):
                return iter(())                      # the engine's output ends: no READY

        class Proc:
            returncode = 1
            stdin, stdout = Pipe(), Pipe()
            pid = 0

            def wait(self, timeout=None):
                holder["engine"].proc = None         # restart()'s close() on another thread
                return 1

            def poll(self):
                return 1

            def kill(self):
                pass

        engine = StrataEngine.__new__(StrataEngine)
        holder["engine"] = engine
        with mock.patch.object(server, "popen", lambda *a, **k: Proc()), \
                mock.patch.object(server, "end_leftover_engines", lambda exe: None), \
                mock.patch.object(server, "transient_start_failure", lambda *a, **k: False):
            with self.assertRaises(server.EngineDied):
                engine.__init__("strata", [])



if __name__ == "__main__":
    unittest.main()
