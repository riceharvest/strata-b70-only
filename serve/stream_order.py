"""serve/stream_order.py - who gets the engine's single turn next: an agent thread runs to its end, the others wait.

One request at a time, several agent streams: each switch parks the live conversation and restores another.  The cache
holds 2-3 big conversations (RAM budget), so with more streams than that the returning one is often evicted and its
whole prompt is read again (50-100K tokens, 15-30 s).  A bare threading.Lock hands the turn to an arbitrary waiter.

StreamOrder sits in front of that lock.  A thread is OPEN while its last reply ended in a tool call (the agent runs the
tool, then sends the next request) and CLOSED when it ended in a plain answer.  When the engine is free it picks:
  1. a waiter that continues the stream served last (its prompt shares the last prompt, to within the tail the
     conversation cache also tolerates: the chat template drops the last reply's thinking);
  2. a waiter that has waited STRATA_STREAM_MAX_WAIT_S (default 1 h; 0 = never: a thread is waited for to its end);
  3. while the last thread is open: nobody, for up to STRATA_STREAM_HOLD_S (default 600) after its turn ended - the
     tool call and the agent's next request are on their way; the others stay parked;
  4. the oldest waiter (the last thread is closed, or its next request never came).
STRATA_STREAM_ORDER=0 turns it off.  Only the order changes, never what is read."""
from __future__ import annotations

import os
import threading
import time


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return max(0.0, float(v))
    except ValueError:
        return default


ENABLED = os.environ.get("STRATA_STREAM_ORDER", "1") != "0"
MAX_WAIT_S = _env_float("STRATA_STREAM_MAX_WAIT_S", 3600.0)   # 0: no escape
HOLD_S = _env_float("STRATA_STREAM_HOLD_S", 600.0)            # an open thread is waited for this long after its turn
TAIL = 4096                 # as --conversation-cache-park-tail
MIN_SHARED = 2048           # sharing less than this (a system prompt) never makes a stream


def common_prefix(a, b) -> int:
    """Length of the longest common prefix of two token lists (bisection on slice equality: C speed)."""
    n = min(len(a), len(b))
    if n == 0 or a[0] != b[0]:
        return 0
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n                     # a[:lo] == b[:lo]; a[:hi] != b[:hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid
    return lo


def continues(ids, last) -> bool:
    """Is `ids` the next request of the stream whose prompt was `last`?  It shares all of it but the last TAIL tokens
    (at most 1/16 of a short one), and not just a system prompt."""
    if not last:
        return False
    return common_prefix(ids, last) >= max(len(last) - min(TAIL, len(last) // 16 + 1), min(len(last), MIN_SHARED))


class Entry:
    __slots__ = ("ids", "t0", "cont_for", "cont", "why", "served", "open")

    def __init__(self, ids):
        self.ids = ids
        self.t0 = time.monotonic()
        self.cont_for = None          # the `last` list this entry's `cont` was computed against
        self.cont = False
        self.why = ""
        self.served = False           # the engine began reading its prompt (so it is the stream served last)
        self.open = False             # its reply ended in a tool call: the thread goes on with the tool's result


class StreamOrder:
    def __init__(self, enabled: bool | None = None):
        self.enabled = ENABLED if enabled is None else enabled
        self.cv = threading.Condition()
        self.waiters: list[Entry] = []
        self.active: Entry | None = None      # picked, until its turn is over
        self.last = None                      # the prompt of the stream served last
        self.last_open = False                # ... and whether its thread is open
        self.last_done = 0.0                  # when that turn ended

    # -- the caller's side -------------------------------------------------------------------------------------------
    def enter(self, ids) -> Entry:
        e = Entry(ids if isinstance(ids, list) else list(ids))
        with self.cv:
            self.waiters.append(e)
            self.cv.notify_all()          # a held pick may be waiting for exactly this request
        return e

    def wait_turn(self, e: Entry, timeout: float) -> bool:
        """True when `e` may take the engine now (and stays so until finish()); False after `timeout`."""
        end = time.monotonic() + timeout
        with self.cv:
            while True:
                now = time.monotonic()
                if not self.enabled or self.active is e:
                    return True
                hold = None
                if self.active is None:
                    pick, why, hold = self._pick(now)
                    if pick is e:
                        self.active, e.why = e, why
                        return True
                left = end - now
                if left <= 0:
                    return False
                self.cv.wait(min(left, hold) if hold else left)

    def finish(self, e: Entry) -> None:
        """`e` is over (done, cancelled or expired).  Idempotent."""
        with self.cv:
            if e in self.waiters:
                self.waiters.remove(e)
            if self.active is e:
                self.active = None
                if e.served:
                    self.last, self.last_open, self.last_done = e.ids, e.open, time.monotonic()
            self.cv.notify_all()

    # -- the choice (cv held) ----------------------------------------------------------------------------------------
    def _continues(self, e: Entry) -> bool:
        if e.cont_for is not self.last:
            e.cont_for = self.last
            e.cont = continues(e.ids, self.last)
        return e.cont

    def _pick(self, now: float):
        """(entry, why, hold): the waiter to go next; or None and how long to hold for the last thread's next request."""
        if not self.waiters:
            return None, "", None
        others = len(self.waiters) > 1
        if self.last is not None:
            for w in self.waiters:                    # (in arrival order)
                if self._continues(w):
                    return w, "kept the thread" if others else "", None
        if MAX_WAIT_S > 0:
            for w in self.waiters:
                if now - w.t0 >= MAX_WAIT_S:
                    return w, f"waited {now - w.t0:.0f} s", None
        if self.last is not None and self.last_open and HOLD_S > 0:
            hold = self.last_done + HOLD_S - now
            if hold > 0:
                if MAX_WAIT_S > 0:                    # wake up when the longest waiter's escape is due
                    hold = min(hold, max(0.01, min(w.t0 for w in self.waiters) + MAX_WAIT_S - now))
                return None, "", hold
            return self.waiters[0], f"its thread did not go on for {HOLD_S:.0f} s", None
        return self.waiters[0], "switched, the thread had ended" if self.last is not None and others else "", None
