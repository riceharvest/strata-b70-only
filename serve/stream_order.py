"""serve/stream_order.py - who gets the engine's single turn next: the stream that was just served goes on first.

One request at a time, several agent streams: each switch parks the live conversation and restores another.  The cache
holds 2-3 big conversations (RAM budget), so with more streams than that the returning one is often evicted and its
whole prompt is read again (50-100K tokens, 15-30 s).  A bare threading.Lock hands the turn to an arbitrary waiter.

StreamOrder sits in front of that lock.  When the engine is free it picks, in this order:
  1. a waiter that continues the stream served last (its prompt shares the last prompt, to within the tail the
     conversation cache also tolerates: the chat template drops the last reply's thinking);
  2. a waiter that has waited STRATA_STREAM_MAX_WAIT_S (nobody is held back for good);
  3. when others wait but none continues the last stream: nobody yet, for a short LINGER after the last turn ended
     (an agent's tool call, then its next request) - the linger follows the gaps measured on this server;
  4. the oldest waiter.
STRATA_STREAM_ORDER=0 turns it off.  Only the order changes, never what is read."""
from __future__ import annotations

import collections
import os
import threading
import time


def _env_float(name: str, default: float | None) -> float | None:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return max(0.0, float(v))
    except ValueError:
        return default


ENABLED = os.environ.get("STRATA_STREAM_ORDER", "1") != "0"
MAX_WAIT_S = _env_float("STRATA_STREAM_MAX_WAIT_S", 120.0) or 0.0   # 0: no escape
LINGER_FIXED_S = _env_float("STRATA_STREAM_LINGER_S", None)         # None: follow the measured gaps; 0: no linger
LINGER_FIRST_S = 2.0        # until GAPS_MIN gaps are known
LINGER_MIN_S, LINGER_MAX_S = 1.0, 8.0
GAPS_MIN = 3
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
    (or 15/16 of it), and not just a system prompt."""
    if not last:
        return False
    shared = common_prefix(ids, last)
    return shared >= max(len(last) - TAIL, min(len(last), MIN_SHARED)) or \
        (shared >= MIN_SHARED and shared * 16 >= len(last) * 15)


class Entry:
    __slots__ = ("ids", "t0", "cont_for", "cont", "why", "served", "waited_for_others")

    def __init__(self, ids):
        self.ids = ids
        self.t0 = time.monotonic()
        self.cont_for = None          # the `last` list this entry's `cont` was computed against
        self.cont = False
        self.why = ""
        self.served = False           # the engine began reading its prompt (so it is the stream served last)
        self.waited_for_others = False


class StreamOrder:
    def __init__(self, enabled: bool | None = None):
        self.enabled = ENABLED if enabled is None else enabled
        self.cv = threading.Condition()
        self.waiters: list[Entry] = []
        self.active: Entry | None = None      # picked, until its turn is over
        self.last = None                      # the prompt of the stream served last
        self.last_done = 0.0                  # when that turn ended
        self.gaps: collections.deque = collections.deque(maxlen=16)   # end of a turn -> its stream's next request

    # -- the caller's side -------------------------------------------------------------------------------------------
    def enter(self, ids) -> Entry:
        e = Entry(ids if isinstance(ids, list) else list(ids))
        with self.cv:
            self.waiters.append(e)
            self.cv.notify_all()          # a lingering pick may be waiting for exactly this request
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
                        if e.cont and e.cont_for is self.last and e.t0 >= self.last_done:
                            self.gaps.append(e.t0 - self.last_done)       # the tool-call gap this stream just showed
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
                    self.last, self.last_done = e.ids, time.monotonic()
            self.cv.notify_all()

    # -- the choice (cv held) ----------------------------------------------------------------------------------------
    def linger_s(self) -> float:
        if LINGER_FIXED_S is not None:
            return LINGER_FIXED_S
        if len(self.gaps) < GAPS_MIN:
            return LINGER_FIRST_S
        g = sorted(self.gaps)
        return min(LINGER_MAX_S, max(LINGER_MIN_S, 1.25 * g[int(0.75 * (len(g) - 1))]))

    def _continues(self, e: Entry) -> bool:
        if e.cont_for is not self.last:
            e.cont_for = self.last
            e.cont = continues(e.ids, self.last)
        return e.cont

    def _pick(self, now: float):
        """(entry, why, hold): the waiter to go next; or None and how long to hold for the last stream's next request."""
        if not self.waiters:
            return None, "", None
        others = len(self.waiters) > 1
        for w in self.waiters:
            w.waited_for_others = w.waited_for_others or others
        if self.last is not None:
            for w in self.waiters:                    # (in arrival order)
                if self._continues(w):
                    return w, "kept the stream" if others else "", None
        if MAX_WAIT_S > 0:
            for w in self.waiters:
                if now - w.t0 >= MAX_WAIT_S:
                    return w, f"waited {now - w.t0:.0f} s", None
        if self.last is not None:
            hold = self.last_done + self.linger_s() - now
            if hold > 0:
                if MAX_WAIT_S > 0:                    # wake up when the longest waiter's escape is due
                    hold = min(hold, max(0.01, min(w.t0 for w in self.waiters) + MAX_WAIT_S - now))
                return None, "", hold
            return self.waiters[0], f"switched after a {self.linger_s():.1f} s linger", None
        return self.waiters[0], "", None
