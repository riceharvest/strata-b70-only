"""Serve-side robustness fixes taken from upstream: #4792 a failed engine start when another thread closed the process,
and (below) a streaming client that stops reading.  The fake engine is built inline."""
import unittest
from pathlib import Path
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



class StalledClient(unittest.TestCase):
    """#4271: a streaming client that stops reading, without closing its connection, must not hold the engine's turn
    for ever: the write that makes no progress ends the request as a disconnect and the next request is served."""

    def test_a_client_that_stops_reading_gives_the_turn_back(self):
        import json
        import socket
        import time
        import urllib.request
        import serve.server as server
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Service, serve
        tok = ByteTokenizer()
        engine = MockEngine(tok, " ".join(str(i) for i in range(80000)), max_context=2_000_000)   # (not one repeated token)
        svc = Service(engine, tok, ChatTemplate(Path(server.__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        port = httpd.server_address[1]
        stuck = socket.socket()
        try:
            with mock.patch.object(server, "STREAM_WRITE_TIMEOUT_S", 0.5, create=True):
                stuck.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
                stuck.connect(("127.0.0.1", port))
                body = json.dumps({"messages": [{"role": "user", "content": "go"}], "stream": True,
                                   "max_tokens": 400000, "reasoning_effort": "none"}).encode()
                stuck.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                              b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
                # ... and it never reads.  Wait until the server is writing into a full buffer, then until it lets go.
                end = time.monotonic() + 3
                while time.monotonic() < end and not svc.fifo.locked():
                    time.sleep(0.01)
                self.assertTrue(svc.fifo.locked(), "the stuck client's request holds the engine's turn")
                end = time.monotonic() + 20
                while time.monotonic() < end and svc.fifo.locked():
                    time.sleep(0.05)
                self.assertFalse(svc.fifo.locked(), "the turn is still held by a client that does not read")
        finally:
            stuck.close()
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
