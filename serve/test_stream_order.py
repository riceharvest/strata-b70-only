"""serve/test_stream_order.py - the thread the engine holds goes on first, the engine never idles (serve/stream_order.py; no process, no GPU).

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


def serve_one(order: so.StreamOrder, ids, open_: bool = True, key=None) -> so.Entry:
    """A request that takes the engine and ends (`open_`: its reply ended in a tool call)."""
    e = order.enter(ids, key)
    assert order.wait_turn(e, 5)
    e.served, e.open = True, open_
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

    def test_clean_key(self):
        self.assertEqual(so.clean_key("  gwk_abc "), "gwk_abc")
        self.assertEqual(len(so.clean_key("x" * 500)), so.KEY_MAX)
        self.assertIsNone(so.clean_key(""))
        self.assertIsNone(so.clean_key(None))
        self.assertIsNone(so.clean_key(5))


class OrderTest(unittest.TestCase):
    def setUp(self):
        self.o = so.StreamOrder(enabled=True, capacity=3)
        self.A, self.B = stream(1, 20_000), stream(2, 20_000)

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
        serve_one(self.o, self.A)                                   # A is the thread served last
        cold = self.o.enter(self.B)                                 # B waits first ...
        time.sleep(0.02)
        nxt = self.o.enter(self.A + [5] * 300)                      # ... A's next turn comes second
        self.assertEqual(self.waiters_go(("B", cold), ("A", nxt)), ["A", "B"])

    def test_arrival_order_when_nobody_continues(self):
        serve_one(self.o, self.A)
        c = self.o.enter(stream(3, 20_000))
        time.sleep(0.02)
        d = self.o.enter(stream(4, 20_000))
        self.assertEqual(self.waiters_go(("c", c), ("d", d)), ["c", "d"])

    def test_the_engine_never_idles_with_a_request_queued(self):
        serve_one(self.o, self.A)                                   # A's reply ended in a tool call: its thread is open
        cold = self.o.enter(self.B)
        t0 = time.monotonic()
        self.assertTrue(self.o.wait_turn(cold, 5))                  # B takes the gap, no hold
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertIn("free engine", cold.why)

    def test_the_open_thread_goes_next_when_its_request_arrives(self):
        serve_one(self.o, self.A)
        b = serve_one(self.o, self.B, open_=False)                  # B filled A's gap and ended with an answer
        self.assertEqual(b.why, "took the free engine for a new thread (1 open)") if b.why else None
        cold = self.o.enter(stream(3, 20_000))
        time.sleep(0.02)
        nxt = self.o.enter(self.A + [5] * 100)
        self.assertEqual(self.waiters_go(("C", cold), ("A", nxt)), ["A", "C"])

    def test_a_closed_thread_gives_its_place_up(self):
        serve_one(self.o, self.A)
        self.assertEqual(len(self.o.recent), 1)
        serve_one(self.o, self.A + [5] * 100, open_=False)          # the final answer
        self.assertEqual(self.o.recent, [])
        cold = self.o.enter(self.B)
        later = self.o.enter(self.A + [5] * 100 + [6] * 50)         # nobody continues a closed thread: arrival order
        self.assertEqual(self.waiters_go(("B", cold), ("A", later)), ["B", "A"])

    def test_a_dead_thread_blocks_nobody(self):
        serve_one(self.o, self.A)                                   # A never sends its next request
        for tag in (2, 3, 4):
            e = self.o.enter(stream(tag, 20_000))
            self.assertTrue(self.o.wait_turn(e, 2))
            e.served = True
            self.o.finish(e)

    def test_the_thread_key_survives_a_rewritten_history(self):
        serve_one(self.o, self.A, key="gwk_a")
        cold = self.o.enter(self.B, "gwk_b")
        time.sleep(0.02)
        rewritten = self.o.enter(stream(7, 9_000), "gwk_a")         # a compaction: the prompt no longer continues A's
        self.assertFalse(so.continues(rewritten.ids, self.A))
        self.assertEqual(self.waiters_go(("B", cold), ("A", rewritten)), ["A", "B"])

    def test_a_side_request_of_a_thread_leaves_it_open(self):
        serve_one(self.o, self.A, key="gwk_a")
        serve_one(self.o, stream(9, 800), open_=False, key="gwk_a")  # a title: another prompt, a plain answer, the same key
        self.assertEqual(len(self.o.recent), 1)
        self.assertEqual(self.o.recent[0].ids, self.A)
        cold = self.o.enter(self.B)
        time.sleep(0.02)
        nxt = self.o.enter(self.A + [5] * 100, "gwk_a")
        self.assertEqual(self.waiters_go(("B", cold), ("A", nxt)), ["A", "B"])

    def test_without_a_key_the_prompt_decides(self):
        serve_one(self.o, self.A, key=None)
        cold = self.o.enter(self.B)
        time.sleep(0.02)
        nxt = self.o.enter(self.A + [5] * 100)                      # no key on either: shares A's whole prompt
        self.assertEqual(self.waiters_go(("B", cold), ("A", nxt)), ["A", "B"])

    def test_the_least_recent_thread_leaves_when_the_cache_is_full(self):
        o = so.StreamOrder(enabled=True, capacity=2)
        a, b, c = stream(1, 20_000), stream(2, 20_000), stream(3, 20_000)
        for ids in (a, b, c):
            serve_one(o, ids)
        self.assertEqual([t.ids for t in o.recent], [b, c])         # A is gone (the cache dropped it)
        old = o.enter(stream(4, 20_000))
        time.sleep(0.02)
        nxt = o.enter(a + [5] * 100)                                # A's thread is cold: it takes its turn in order
        self.assertTrue(o.wait_turn(old, 2))
        o.finish(old)
        self.assertTrue(o.wait_turn(nxt, 2))
        o.evicted()
        self.assertEqual(len(o.recent), 1)
        o.evicted(5)
        self.assertEqual(o.recent, [])

    def test_a_waiter_that_leaves_gives_its_place_up(self):
        e1 = self.o.enter(self.A)
        self.assertTrue(self.o.wait_turn(e1, 1))                    # e1 is the active one
        e2 = self.o.enter(self.B)
        self.assertFalse(self.o.wait_turn(e2, 0.2))                 # e2 waits behind it
        self.o.finish(e2)                                           # e2's client left
        self.assertNotIn(e2, self.o.waiters)
        self.o.finish(e1)                                           # e1 cancelled before the engine read it
        self.assertEqual(self.o.recent, [])
        e3 = self.o.enter(self.B)
        self.assertTrue(self.o.wait_turn(e3, 1))
        self.o.finish(e3)
        self.o.finish(e3)                                           # idempotent

    def test_max_wait_lets_a_waiter_go_first_when_asked_for(self):
        with mock.patch.object(so, "MAX_WAIT_S", 0.2):
            serve_one(self.o, self.A)
            cold = self.o.enter(self.B)
            time.sleep(0.25)
            nxt = self.o.enter(self.A + [5] * 100)
            self.assertEqual(self.waiters_go(("A", nxt), ("B", cold))[0], "A")   # the thread served last still goes first
        serve_one(self.o, stream(5, 20_000), open_=False)
        with mock.patch.object(so, "MAX_WAIT_S", 0.2):
            cold = self.o.enter(self.B)
            time.sleep(0.25)
            self.assertTrue(self.o.wait_turn(cold, 2))
            self.assertIn("waited", cold.why)

    def test_off_never_waits(self):
        o = so.StreamOrder(enabled=False)
        a, b = o.enter(self.A), o.enter(self.B)
        self.assertTrue(o.wait_turn(a, 0.1))
        self.assertTrue(o.wait_turn(b, 0.1))


class ThroughService(unittest.TestCase):
    """The real Service.run (one request at a time, no batch hooks): B and C fill A's tool gaps, A's next request goes
    ahead of the older C, and the engine is never idle with a request waiting."""

    CALL = "<tool_call><function=read><parameter=p>x</parameter></function></tool_call>"
    ANSWER = "all done."

    def test_a_waiting_thread_takes_the_gap_and_the_open_thread_goes_ahead_of_older_ones(self):
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Service, IM_END
        tok = ByteTokenizer()
        end = tok.encode(IM_END, parse_special=True)
        call, answer = tok.encode(self.CALL, parse_special=True) + end, tok.encode(self.ANSWER, parse_special=True) + end
        sys_ids = list(range(100, 3100))
        order, turns = [], {}

        class Eng(MockEngine):                  # (no queue_started call: that is the --batch path's)
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                name = "ABC"[(ids[3000] // 10_000) - 1]
                turns[name] = turns.get(name, 0) + 1
                order.append(name)
                for t in (call if name == "A" and turns[name] < 3 else answer):
                    time.sleep(0.12 if name == "B" else 0.004)         # B's turn is long: A's next request arrives during it
                    yield t

        svc = Service(Eng(tok, "x"), tok, ChatTemplate(Path(__file__).resolve().parents[1] / "serve/chat_template.jinja"))
        tools = [{"name": "read", "description": "r",
                  "parameters": {"type": "object", "properties": {"p": {"type": "string"}}}}]

        def request(ids, key=None):
            cancel = threading.Event()
            if key:
                setattr(cancel, "_strata_thread", key)
            return [x.kind for k, x in svc.run(ids, False, tools, 64, {}, cancel) if k == "event"]

        a_kinds = []

        def agent():
            ids = sys_ids + [10_000 + i for i in range(20_000)]
            for _ in range(3):
                a_kinds.append(request(ids, "gwk_a"))
                time.sleep(0.6)                                # the tool runs, then the next request
                ids = ids + [777] * 40 + [10_999]

        def later(delay, tag):
            time.sleep(delay)
            request(sys_ids + [10_000 * tag + i for i in range(20_000)], f"gwk_{tag}")

        ta = threading.Thread(target=agent)
        tb = threading.Thread(target=later, args=(0.45, 2))
        tc = threading.Thread(target=later, args=(0.5, 3))
        for t in (ta, tb, tc):
            t.start()
        for t in (ta, tb, tc):
            t.join(30)
        self.assertTrue(any(k in ("tool_start", "tool_call") for k in a_kinds[0]), a_kinds[0])
        self.assertFalse(any(k in ("tool_start", "tool_call") for k in a_kinds[2]), a_kinds[2])
        # B took A's first gap; A's second request (waiting while B read) went ahead of the older C; C took the next gap
        self.assertEqual(order, ["A", "B", "A", "C", "A"])



class Pin(unittest.TestCase):
    """The web page's pin: one thread runs to its end, every other request waits (also while the engine is free)."""

    def test_only_the_pinned_thread_takes_the_engine(self):
        o = so.StreamOrder(enabled=True, capacity=3)
        a, b = stream(1, 500), stream(2, 500)
        serve_one(o, a, key="A")                      # A's turn ended in a tool call: its agent is running the tool
        self.assertTrue(o.pin("A"))
        wb = o.enter(b + [1], "B")
        self.assertFalse(o.wait_turn(wb, 0.05))       # the engine is free, B still waits behind the pin
        wa = o.enter(a + [7], "A")                    # A's next request arrives after B's
        self.assertFalse(o.wait_turn(wb, 0.05))
        self.assertTrue(o.wait_turn(wa, 1))
        self.assertEqual(wa.why, "the pinned thread")
        wa.served, wa.open = True, True
        o.finish(wa)
        self.assertEqual(o.pinned, "A")               # a tool call: the thread goes on, so does the pin
        self.assertFalse(o.wait_turn(wb, 0.05))
        o.finish(wb)

    def test_the_pin_ends_with_the_thread_and_not_on_a_cancelled_turn(self):
        o = so.StreamOrder(enabled=True, capacity=3)
        a, b = stream(1, 500), stream(2, 500)
        serve_one(o, a, key="A")
        o.pin("A")
        wb = o.enter(b, "B")
        e = o.enter(a + [7], "A")                     # this turn is cancelled (the client left): the agent sends it again
        self.assertTrue(o.wait_turn(e, 1))
        e.served, e.open, e.finish = True, False, "cancel"
        o.finish(e)
        self.assertEqual(o.pinned, "A")
        self.assertFalse(o.wait_turn(wb, 0.05))
        e = o.enter(a + [7], "A")                     # the same turn again, now a finished answer with no tool call
        self.assertTrue(o.wait_turn(e, 1))
        e.served, e.open, e.finish = True, False, "stop"
        o.finish(e)
        self.assertIsNone(o.pinned)                   # the thread is complete: the pin is gone
        self.assertTrue(o.wait_turn(wb, 1))           # and the request that waited goes on
        o.finish(wb)

    def test_unpin_lets_the_waiting_requests_go(self):
        o = so.StreamOrder(enabled=True, capacity=3)
        o.pin("A")
        wb = o.enter(stream(2, 500), "B")
        self.assertFalse(o.wait_turn(wb, 0.05))
        snap = o.snapshot()
        self.assertEqual((snap["pinned"], snap["held_back"]), ("A", 1))
        self.assertEqual({r["key"]: r["state"] for r in snap["threads"]}, {"B": "waiting", "A": "not seen yet"})
        self.assertTrue(next(r for r in snap["threads"] if r["key"] == "A")["pinned"])
        o.unpin()
        self.assertTrue(o.wait_turn(wb, 1))
        self.assertEqual(o.snapshot()["threads"][0]["state"], "running")
        o.finish(wb)
        self.assertFalse(o.pin("   "))                # no usable name: nothing is pinned
        self.assertIsNone(o.pinned)

    def test_a_set_max_wait_also_ends_the_wait_behind_a_pin(self):
        o = so.StreamOrder(enabled=True, capacity=3)
        o.pin("never-comes")                          # a thread that never sends a request (a typo, a crashed agent)
        wb = o.enter(stream(2, 500), "B")
        self.assertFalse(o.wait_turn(wb, 0.3))        # no max wait (the default): B waits until the pin is taken off
        with mock.patch.object(so, "MAX_WAIT_S", 0.2):
            self.assertTrue(o.wait_turn(wb, 2))       # it has waited longer than the max wait: it goes
            self.assertIn("behind the pin", wb.why)
        self.assertEqual(o.pinned, "never-comes")     # the pin itself stays
        o.finish(wb)



class Linger(unittest.TestCase):
    """The run config's stream_linger_s: when the thread served last has a tool running, another thread's request is held
    back that long for its next request.  Off (0) nothing waits; the return gaps are counted either way."""

    def setUp(self):
        self.before = so.LINGER_S

    def tearDown(self):
        so.LINGER_S = self.before

    @staticmethod
    def turn(o, ids, key, timeout=1.0):
        e = o.enter(ids, key)
        if not o.wait_turn(e, timeout):
            return e, False
        e.served, e.open = True, True
        o.finish(e)
        return e, True

    def test_off_the_other_thread_goes_at_once(self):
        so.LINGER_S = 0.0
        o = so.StreamOrder(capacity=3)
        self.assertTrue(self.turn(o, list(range(5000)), "A")[1])
        t = time.monotonic()
        self.assertTrue(self.turn(o, list(range(9000, 14000)), "B")[1])
        self.assertLess(time.monotonic() - t, 0.2)
        self.assertEqual(o.snapshot()["lingered"], 0)

    def test_the_thread_that_comes_back_in_time_keeps_the_engine(self):
        so.LINGER_S = 0.6
        o = so.StreamOrder(capacity=3)
        self.assertTrue(self.turn(o, list(range(5000)), "A")[1])
        b = o.enter(list(range(9000, 14000)), "B")
        self.assertFalse(o.wait_turn(b, 0.2))                       # held: A's tool is running
        a2 = o.enter(list(range(5000)) + [7, 8, 9], "A")
        self.assertTrue(o.wait_turn(a2, 0.2))                       # A is back in time and goes first
        self.assertIn("within the linger", a2.why)
        a2.served, a2.open = True, True
        o.finish(a2)
        t = time.monotonic()
        self.assertTrue(o.wait_turn(b, 2.0))                        # A does not come back again: B goes when the linger ends
        self.assertTrue(0.3 < time.monotonic() - t < 1.2)
        snap = o.snapshot()
        self.assertEqual((snap["linger_s"], snap["lingered"], snap["linger_kept"]), (0.6, 2, 1))
        self.assertEqual(snap["return_gap"]["n"], 1)
        self.assertLessEqual(snap["return_gap"]["p50_s"], 1.0)

    def test_a_closed_thread_is_not_waited_for(self):
        so.LINGER_S = 0.6
        o = so.StreamOrder(capacity=3)
        a = o.enter(list(range(5000)), "A")
        self.assertTrue(o.wait_turn(a, 1.0))
        a.served, a.open = True, False                              # a plain answer: nothing comes after it
        o.finish(a)
        t = time.monotonic()
        self.assertTrue(self.turn(o, list(range(9000, 14000)), "B")[1])
        self.assertLess(time.monotonic() - t, 0.2)

if __name__ == "__main__":
    unittest.main()
