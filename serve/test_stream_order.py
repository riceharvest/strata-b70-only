"""serve/test_stream_order.py - the stream served last goes on first (serve/stream_order.py; no process, no GPU).

    python -m unittest serve.test_stream_order -v
"""
from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import stream_order as so  # noqa: E402


def stream(tag: int, n: int, system: int = 3000) -> list[int]:
    """A prompt: a system prompt every stream shares, then `n` tokens of this stream."""
    return list(range(system)) + [10_000 * tag + i for i in range(n)]


def serve_one(order: so.StreamOrder, ids, take: float = 0.0, open_: bool = True) -> so.Entry:
    """A request that takes the engine, 'reads' for `take` s and ends (its reply ended in a tool call: `open_`)."""
    e = order.enter(ids)
    assert order.wait_turn(e, 5)
    e.served, e.open = True, open_
    time.sleep(take)
    order.finish(e)
    return e


class ContinuesTest(unittest.TestCase):
    def test_prefix_and_tail(self):
        a = stream(1, 20_000)
        self.assertTrue(so.continues(a + [7] * 500, a))                       # the next turn
        self.assertTrue(so.continues(a[:-1000] + [9] * 800, a))               # the template dropped the last reply's tail
        self.assertFalse(so.continues(stream(2, 20_000), a))                  # same system prompt, another stream
        self.assertFalse(so.continues(a[:5000] + [9] * 20_000, a))            # diverges early
        short = stream(1, 500)
        self.assertFalse(so.continues(stream(2, 500), short))                 # a short prompt: only the system prompt shared
        self.assertTrue(so.continues(short + [3] * 40, short))
        self.assertFalse(so.continues(a, None))
        self.assertEqual(so.common_prefix(a, a + [1]), len(a))
        self.assertEqual(so.common_prefix([], a), 0)


class OrderTest(unittest.TestCase):
    def setUp(self):
        self.patches = [mock.patch.object(so, "HOLD_S", 0.0), mock.patch.object(so, "MAX_WAIT_S", 120.0)]
        for p in self.patches:
            p.start()
        self.o = so.StreamOrder(enabled=True)
        self.A, self.B = stream(1, 20_000), stream(2, 20_000)

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def waiters_go(self, *entries):
        """The entries wait for the turn at once; returns the order they took it in."""
        order, lock = [], threading.Lock()

        def go(name, e):
            self.assertTrue(self.o.wait_turn(e, 10))
            with lock:
                order.append(name)
            e.served = True
            time.sleep(0.05)
            self.o.finish(e)

        ts = [threading.Thread(target=go, args=x) for x in entries]
        for t in ts:
            t.start()
        for t in ts:
            t.join(15)
        return order

    def test_continuing_waiter_beats_an_older_cold_one(self):
        serve_one(self.o, self.A)                                   # A is the stream served last
        cold = self.o.enter(self.B)                                 # B waits first ...
        time.sleep(0.02)
        nxt = self.o.enter(self.A + [5] * 300)                      # ... A's next turn comes second
        self.assertEqual(self.waiters_go(("B", cold), ("A", nxt)), ["A", "B"])

    def test_arrival_order_when_nobody_continues(self):
        serve_one(self.o, self.A)
        c, d = self.o.enter(stream(3, 20_000)), None
        time.sleep(0.02)
        d = self.o.enter(stream(4, 20_000))
        self.assertEqual(self.waiters_go(("c", c), ("d", d)), ["c", "d"])

    def test_an_open_thread_holds_the_others_for_its_next_request(self):
        with mock.patch.object(so, "HOLD_S", 5.0):
            serve_one(self.o, self.A)                               # A's reply ended in a tool call
            cold = self.o.enter(self.B)
            got = []
            t = threading.Thread(target=lambda: got.append(self.o.wait_turn(cold, 8)))
            t.start()
            time.sleep(0.5)                                         # the tool runs ... (far longer than a few seconds in life)
            self.assertEqual(got, [], "B went while A's thread was open")
            nxt = self.o.enter(self.A + [5] * 100)                  # A's tool call finished: its next request
            self.assertTrue(self.o.wait_turn(nxt, 2))
            nxt.served, nxt.open = True, True
            time.sleep(0.2)
            self.assertEqual(got, [], "B went before A's thread ended")
            self.o.finish(nxt)                                      # open again: B still waits
            time.sleep(0.3)
            self.assertEqual(got, [])
            last = self.o.enter(self.A + [5] * 100 + [6] * 50)
            self.assertTrue(self.o.wait_turn(last, 2))
            last.served, last.open = True, False                    # the final answer, no tool call: the thread is closed
            self.o.finish(last)
            t.join(5)
            self.assertEqual(got, [True])

    def test_a_closed_thread_lets_the_next_one_go_at_once(self):
        with mock.patch.object(so, "HOLD_S", 5.0):
            serve_one(self.o, self.A, open_=False)
            cold = self.o.enter(self.B)
            t0 = time.monotonic()
            self.assertTrue(self.o.wait_turn(cold, 5))
            self.assertLess(time.monotonic() - t0, 0.5)

    def test_the_hold_ends_when_the_thread_never_goes_on(self):
        with mock.patch.object(so, "HOLD_S", 0.3):
            serve_one(self.o, self.A)
            cold = self.o.enter(self.B)
            t0 = time.monotonic()
            self.assertTrue(self.o.wait_turn(cold, 5))
            self.assertGreaterEqual(time.monotonic() - t0, 0.25)
            self.assertIn("did not go on", cold.why)

    def test_max_wait_lets_a_waiter_pass_an_open_thread(self):
        with mock.patch.object(so, "MAX_WAIT_S", 0.2), mock.patch.object(so, "HOLD_S", 5.0):
            serve_one(self.o, self.A)
            cold = self.o.enter(self.B)
            t0 = time.monotonic()
            self.assertTrue(self.o.wait_turn(cold, 5))
            self.assertLess(time.monotonic() - t0, 2)
            self.assertIn("waited", cold.why)

    def test_a_waiter_that_leaves_gives_its_place_up(self):
        e1 = self.o.enter(self.A)
        self.assertTrue(self.o.wait_turn(e1, 1))                    # e1 is the active one
        e2 = self.o.enter(self.B)
        self.assertFalse(self.o.wait_turn(e2, 0.2))                 # e2 waits behind it
        self.o.finish(e2)                                           # e2's client left
        self.assertNotIn(e2, self.o.waiters)
        self.o.finish(e1)                                           # e1 cancelled before the engine read it
        self.assertIsNone(self.o.last)
        e3 = self.o.enter(self.B)
        self.assertTrue(self.o.wait_turn(e3, 1))
        self.o.finish(e3)
        self.o.finish(e3)                                           # idempotent

    def test_off_never_waits(self):
        o = so.StreamOrder(enabled=False)
        a, b = o.enter(self.A), o.enter(self.B)
        self.assertTrue(o.wait_turn(a, 0.1))
        self.assertTrue(o.wait_turn(b, 0.1))


class ThroughService(unittest.TestCase):
    """The real Service.run (one request at a time, no batch hooks): an agent thread runs to its final answer first."""

    CALL = "<tool_call><function=read><parameter=p>x</parameter></function></tool_call>"
    ANSWER = "all done."

    def test_a_thread_with_tool_calls_is_served_to_its_end_before_a_waiting_stream(self):
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Service, IM_END
        tok = ByteTokenizer()
        end = tok.encode(IM_END, parse_special=True)
        call, answer = tok.encode(self.CALL, parse_special=True) + end, tok.encode(self.ANSWER, parse_special=True) + end
        sys_ids = list(range(100, 3100))
        order, turns = [], {}

        class Eng(MockEngine):                  # (no queue_started call: that is the --batch path's)
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                name = "AB"[(ids[3000] // 10_000) - 1]
                turns[name] = turns.get(name, 0) + 1
                order.append(name)
                for t in (call if name == "A" and turns[name] < 3 else answer):
                    time.sleep(0.004)
                    yield t

        svc = Service(Eng(tok, "x"), tok, ChatTemplate(Path(__file__).resolve().parents[1] / "serve/chat_template.jinja"))
        tools = [{"name": "read", "description": "r",
                  "parameters": {"type": "object", "properties": {"p": {"type": "string"}}}}]

        def request(ids):
            return [x.kind for k, x in svc.run(ids, False, tools, 64, {}, threading.Event()) if k == "event"]

        a_kinds = []

        def agent():
            ids = sys_ids + [10_000 + i for i in range(20_000)]
            for _ in range(3):
                a_kinds.append(request(ids))
                time.sleep(0.4)                                # the tool runs, then the next request
                ids = ids + [777] * 40 + [10_999]

        with mock.patch.object(so, "HOLD_S", 5.0), mock.patch.object(so, "MAX_WAIT_S", 60.0):
            ta = threading.Thread(target=agent)
            tb = threading.Thread(target=lambda: (time.sleep(0.1), request(sys_ids + [20_000 + i for i in range(20_000)])))
            ta.start()
            tb.start()
            ta.join(30)
            tb.join(30)
        self.assertTrue(any(k in ("tool_start", "tool_call") for k in a_kinds[0]), a_kinds[0])
        self.assertFalse(any(k in ("tool_start", "tool_call") for k in a_kinds[2]), a_kinds[2])
        self.assertEqual(order, ["A", "A", "A", "B"], "B took the engine in the middle of A's thread")


if __name__ == "__main__":
    unittest.main()
