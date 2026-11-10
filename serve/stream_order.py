"""serve/stream_order.py - who gets the engine's single turn next: a thread the engine already holds goes on first.

One request at a time, several agent threads: each switch parks the live conversation and restores another.  The cache
holds 2-3 big conversations (RAM budget), so a thread that comes back after more conversations than that have parked
is read again from its first token (50-100K tokens, 15-70 s).  A bare threading.Lock hands the turn to an arbitrary waiter.

StreamOrder sits in front of that lock and keeps the threads the cache still holds - `recent`, least recent first, as
many as the cache holds (`capacity`; an eviction the engine reports drops the least recent).  Nothing here is a timer.
When the engine is free it picks:
  1. a waiter that continues a thread in `recent`, the most recently served thread first: it restores from the cache
     (or goes on in the live session) and reads only what is new;
  2. else the oldest waiter, at once - the engine never sits idle with a request queued.  A thread whose next request is
     still on its way (its reply ended in a tool call) is not waited for: when that request arrives it goes next, behind
     the one turn that took the gap.  An agent that died sends no next request and just ages out of `recent`.
A thread is recognised by the header the client sends per conversation (Hermes: `session_affinity_header: x-strata-thread`,
exact and stable through compaction), else by prompt: it shares all of the thread's last prompt but a tail.
A reply with a plain answer (no tool call) closes its thread: it leaves `recent` and frees its place.
A PIN (the web page's "Agent threads" card, POST /v1/threads/pin) names one thread by its key: while it is set only that
thread's requests take the engine and every other request waits, also while the pinned agent runs a tool and the engine
is free.  The pin ends when the thread ends (a finished reply with no tool call) or when it is taken off by hand.
STRATA_STREAM_ORDER=0 turns it off; STRATA_STREAM_MAX_WAIT_S=N (default 0 = never) lets a waiter that long go first."""
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
MAX_WAIT_S = _env_float("STRATA_STREAM_MAX_WAIT_S", 0.0)      # 0: no escape; a thread is waited for to its end
# The run config's "stream_linger_s" (0 = off, the default): when the engine falls free and the thread it just served has a
# tool running, a waiter of ANOTHER thread is held back this long for that thread's next request.  If it arrives in time
# the engine goes on with the live conversation (no park, no restore); if not, the waiter goes after the wait.
LINGER_S = 0.0
GAPS_KEPT = 256             # the last gaps (a thread's reply end to its next request) kept for the snapshot's statistics
TAIL = 4096                 # as --conversation-cache-park-tail
MIN_SHARED = 2048           # sharing less than this (a system prompt) never makes a thread
KEY_MAX = 128               # a thread key is cut to this
THREAD_HEADER = "X-Strata-Thread"      # the request header that names a conversation (Hermes: session_affinity_header)


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
    """Is `ids` the next request of the thread whose prompt was `last`?  It shares all of it but the last TAIL tokens
    (at most 1/16 of a short one), and not just a system prompt."""
    if not last:
        return False
    return common_prefix(ids, last) >= max(len(last) - min(TAIL, len(last) // 16 + 1), min(len(last), MIN_SHARED))


def clean_key(key) -> str | None:
    """A thread key from a request header: text up to KEY_MAX characters, or None (empty, not text)."""
    if not isinstance(key, str):
        return None
    key = key.strip()[:KEY_MAX]
    return key or None


class Thread:
    """A conversation the engine's cache may still hold: its last prompt, its key (if the client names threads)."""
    __slots__ = ("key", "ids", "ver", "turns", "last")

    def __init__(self, key, ids):
        self.key, self.ids, self.ver = key, ids, 0
        self.turns, self.last = 0, time.monotonic()   # turns served; when the last one ended


class Entry:
    __slots__ = ("ids", "key", "t0", "cache", "why", "served", "open", "finish")

    def __init__(self, ids, key=None):
        self.ids = ids
        self.key = clean_key(key)
        self.t0 = time.monotonic()
        self.cache = {}               # id(thread) -> (thread.ver, does this request continue it by prompt)
        self.why = ""
        self.served = False           # the engine began reading its prompt (so it is the live conversation)
        self.open = False             # its reply ended in a tool call: the thread goes on with the tool's result
        self.finish = None            # how its reply ended ("stop", "length", "cancel", "error", ...), when the server said


class StreamOrder:
    def __init__(self, enabled: bool | None = None, capacity: int = 3):
        self.enabled = ENABLED if enabled is None else enabled
        self.capacity = max(1, capacity)      # conversations the cache holds, the live one included
        self.cv = threading.Condition()
        self.waiters: list[Entry] = []
        self.active: Entry | None = None      # picked, until its turn is over
        self.recent: list[Thread] = []        # open threads the cache holds, least recently served first
        self.pinned: str | None = None        # the key of the one thread that may take the engine (None: no pin)
        self.pinned_at = 0.0
        self.gaps: list[float] = []           # seconds from a thread's reply end (a tool call) to its next request
        self.lingered = 0                     # waits LINGER_S began; `linger_kept`: the thread came back within one
        self.linger_kept = 0
        self._hold_until = 0.0                # _pick is holding the waiters back until then (0: it is not)
        self._held_for: Thread | None = None

    # -- the caller's side -------------------------------------------------------------------------------------------
    def enter(self, ids, key=None) -> Entry:
        e = Entry(ids if isinstance(ids, list) else list(ids), key)
        with self.cv:
            self.waiters.append(e)
            self.cv.notify_all()
        return e

    def wait_turn(self, e: Entry, timeout: float) -> bool:
        """True when `e` may take the engine now (and stays so until finish()); False after `timeout`."""
        end = time.monotonic() + timeout
        with self.cv:
            while True:
                now = time.monotonic()
                if not self.enabled or self.active is e:
                    return True
                if self.active is None:
                    pick, why = self._pick(now)
                    if pick is e:
                        self.active, e.why = e, why
                        return True
                left = end - now
                if left <= 0:
                    return False
                # a linger ends by the clock, not by a notify: wake then
                self.cv.wait(min(left, max(0.01, self._hold_until - now)) if self._hold_until > now else left)

    def finish(self, e: Entry) -> None:
        """`e` is over (done, cancelled or expired).  Idempotent."""
        with self.cv:
            if e in self.waiters:
                self.waiters.remove(e)
            if self.active is e:
                self.active = None
                if e.served:
                    self._served(e)
            self.cv.notify_all()

    def evicted(self, n: int = 1) -> None:
        """The engine's cache dropped `n` parked conversations (its oldest first): the least recent threads are gone."""
        with self.cv:
            del self.recent[:max(0, n)]

    def pin(self, key) -> bool:
        """Only the thread named `key` takes the engine from now on, until it ends or unpin() (a request that has waited
        STRATA_STREAM_MAX_WAIT_S, when that is set, still goes).  False: no usable key."""
        key = clean_key(key)
        if key is None:
            return False
        with self.cv:
            self.pinned, self.pinned_at = key, time.monotonic()
            self.cv.notify_all()
        return True

    def unpin(self) -> None:
        with self.cv:
            self.pinned = None
            self.cv.notify_all()

    def snapshot(self, tail: int = 48) -> dict:
        """What the web page shows: the pin, and each thread seen now - running, waiting or between two turns (its agent
        runs a tool).  `tail_ids`: the last tokens of its newest prompt, for a preview the server decodes."""
        now = time.monotonic()
        with self.cv:
            rows: dict = {}

            def put(key, state, ids, waited=None, turns=None):
                k = key if key is not None else f"\0{len(rows)}"        # a thread without a name stays its own row
                if k in rows and rows[k]["state"] != "between turns":
                    return
                old = rows.get(k) or {}
                rows[k] = {"key": key, "state": state, "tokens": len(ids), "tail_ids": list(ids[-tail:]) if tail > 0 else [],
                           "waited_s": round(waited, 1) if waited is not None else None,
                           "turns": turns if turns is not None else old.get("turns")}

            for t in self.recent:
                put(t.key, "between turns", t.ids, now - t.last, t.turns)
            if self.active is not None:
                put(self.active.key, "running", self.active.ids)
            for w in self.waiters:
                if w is not self.active:
                    put(w.key, "waiting", w.ids, now - w.t0)
            if self.pinned is not None and self.pinned not in rows:
                rows[self.pinned] = {"key": self.pinned, "state": "not seen yet", "tokens": 0, "tail_ids": [],
                                     "waited_s": None, "turns": None}
            held = sum(1 for w in self.waiters if w is not self.active and w.key != self.pinned) if self.pinned else 0
            g = sorted(self.gaps)
            gap = {"n": len(g), "p50_s": round(g[len(g) // 2], 2) if g else None,
                   "p90_s": round(g[min(len(g) - 1, len(g) * 9 // 10)], 2) if g else None,
                   "within_1s": sum(1 for x in g if x <= 1.0), "within_3s": sum(1 for x in g if x <= 3.0)}
            return {"enabled": self.enabled, "linger_s": LINGER_S, "lingered": self.lingered, "linger_kept": self.linger_kept,
                    "return_gap": gap, "pinned": self.pinned,
                    "pinned_for_s": round(now - self.pinned_at, 1) if self.pinned else None, "held_back": held,
                    "threads": [dict(r, pinned=r["key"] is not None and r["key"] == self.pinned) for r in rows.values()]}

    def set_capacity(self, n: int) -> None:
        with self.cv:
            self.capacity = max(1, int(n))
            del self.recent[:max(0, len(self.recent) - self.capacity)]
            self.cv.notify_all()

    # -- the choice (cv held) ----------------------------------------------------------------------------------------
    def _by_prompt(self, e: Entry, t: Thread) -> bool:
        c = e.cache.get(id(t))
        if c is None or c[0] != t.ver:
            c = e.cache[id(t)] = (t.ver, continues(e.ids, t.ids))
        return c[1]

    def _belongs(self, e: Entry, t: Thread) -> bool:
        return (e.key is not None and e.key == t.key) or self._by_prompt(e, t)

    def _pick(self, now: float):
        """(entry, why): the waiter to go next."""
        if not self.waiters:
            return None, ""
        if self.pinned is not None:                             # a pin: its thread or nobody (the others wait)
            for w in self.waiters:
                if w.key == self.pinned:
                    return w, "the pinned thread" if len(self.waiters) > 1 else ""
            if MAX_WAIT_S > 0:                                  # ... but not longer than the max wait, when one is set:
                for w in self.waiters:                          # a pin on a thread that never comes back holds nobody for good
                    if now - w.t0 >= MAX_WAIT_S:
                        return w, f"waited {now - w.t0:.0f} s behind the pin"
            return None, ""
        others = len(self.waiters) > 1
        for i in range(len(self.recent) - 1, -1, -1):          # the thread served last first
            t = self.recent[i]
            for w in self.waiters:                              # (in arrival order)
                if self._belongs(w, t):
                    why = ("kept the thread" if i == len(self.recent) - 1 else "back to a parked thread") if others else ""
                    if i == len(self.recent) - 1 and self._held_for is t:
                        self.linger_kept += 1                   # it came back within the linger: no switch
                        why = "kept the thread (it came back within the linger)"
                    self._hold_until, self._held_for = 0.0, None
                    return w, why
            if i == len(self.recent) - 1 and LINGER_S > 0 and now - t.last < LINGER_S:
                # stream_linger_s: the thread served last has a tool running and none of the waiters is its next request;
                # nobody else goes before the linger is over
                if self._held_for is not t:
                    self._held_for = t
                    self.lingered += 1
                self._hold_until = t.last + LINGER_S
                return None, ""
        self._hold_until, self._held_for = 0.0, None
        if MAX_WAIT_S > 0:
            for w in self.waiters:
                if now - w.t0 >= MAX_WAIT_S:
                    return w, f"waited {now - w.t0:.0f} s"
        why = f"took the free engine for a new thread ({len(self.recent)} open)" if others or self.recent else ""
        return self.waiters[0], why

    def _served(self, e: Entry) -> None:
        """`e` ran its turn: its thread is the most recent; a plain answer closes it (cv held)."""
        t = next((t for t in self.recent if self._by_prompt(e, t)), None)
        if t is None and e.key is not None:
            k = next((k for k in self.recent if k.key == e.key), None)
            if k is not None:
                if not e.open:
                    return            # a side request of a thread (a title, a summary): the thread is as it was
                t = k                 # a tool-calling turn that does not continue its prompt (the history was rewritten)
        if t is not None:
            self.recent.remove(t)
            if e.t0 >= t.last:                 # how long its agent took to come back (a tool's run time)
                self.gaps.append(e.t0 - t.last)
                del self.gaps[:max(0, len(self.gaps) - GAPS_KEPT)]
        if not e.open:
            # no tool call: nothing comes after it, so no place is kept for it.  A pinned thread that ended this way
            # (a finished reply, not a cancelled or failed one, which its agent sends again) is complete: the pin ends.
            if self.pinned is not None and e.key == self.pinned and e.finish == "stop":
                self.pinned = None
                print(f"[strata] stream order: the pinned thread {e.key[:24]} ended; the others go on", flush=True)
            return
        if t is None:
            t = Thread(e.key, e.ids)
        else:
            t.key, t.ids, t.ver = e.key or t.key, e.ids, t.ver + 1
        t.turns, t.last = t.turns + 1, time.monotonic()
        self.recent.append(t)
        del self.recent[:max(0, len(self.recent) - self.capacity)]
