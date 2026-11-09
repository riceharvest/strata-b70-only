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


def serve_one(order: so.StreamOrder, ids, take: float = 0.0) -> so.Entry:
    """A request that takes the engine, 'reads' for `take` s and ends."""
    e = order.enter(ids)
    assert order.wait_turn(e, 5)
    e.served = True
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
        self.assertFalse(so.continues(a, None))
        self.assertEqual(so.common_prefix(a, a + [1]), len(a))
        self.assertEqual(so.common_prefix([], a), 0)


class OrderTest(unittest.TestCase):
    def setUp(self):
        self.patches = [mock.patch.object(so, "LINGER_FIXED_S", 0.0), mock.patch.object(so, "MAX_WAIT_S", 120.0)]
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

    def test_linger_holds_for_the_next_turn_of_the_stream(self):
        with mock.patch.object(so, "LINGER_FIXED_S", 0.6):
            serve_one(self.o, self.A)
            cold = self.o.enter(self.B)
            got = []
            t = threading.Thread(target=lambda: got.append(self.o.wait_turn(cold, 5)))
            t0 = time.monotonic()
            t.start()
            time.sleep(0.3)
            self.assertEqual(got, [], "B went before the linger was over")
            nxt = self.o.enter(self.A + [5] * 100)                  # A's tool call finished within the linger
            self.assertTrue(self.o.wait_turn(nxt, 2))
            nxt.served = True
            self.assertEqual(got, [], "B went before A's request")
            self.o.finish(nxt)
            t.join(5)
            self.assertEqual(got, [True])
            self.assertEqual(len(self.o.gaps), 1)
            self.assertLess(time.monotonic() - t0, 3)

    def test_linger_ends_and_the_cold_waiter_goes(self):
        with mock.patch.object(so, "LINGER_FIXED_S", 0.3):
            serve_one(self.o, self.A)
            cold = self.o.enter(self.B)
            t0 = time.monotonic()
            self.assertTrue(self.o.wait_turn(cold, 5))
            self.assertGreaterEqual(time.monotonic() - t0, 0.25)
            self.assertIn("linger", cold.why)

    def test_max_wait_lets_a_waiter_pass_a_busy_stream(self):
        with mock.patch.object(so, "MAX_WAIT_S", 0.2), mock.patch.object(so, "LINGER_FIXED_S", 5.0):
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

    def test_measured_linger_follows_the_gaps(self):
        with mock.patch.object(so, "LINGER_FIXED_S", None):
            o = so.StreamOrder(enabled=True)
            self.assertEqual(o.linger_s(), so.LINGER_FIRST_S)
            o.gaps.extend([0.2, 0.3, 0.4, 0.5])
            self.assertEqual(o.linger_s(), so.LINGER_MIN_S)          # 1.25 x p75 is under the floor
            o.gaps.extend([30.0] * 12)
            self.assertEqual(o.linger_s(), so.LINGER_MAX_S)


if __name__ == "__main__":
    unittest.main()
