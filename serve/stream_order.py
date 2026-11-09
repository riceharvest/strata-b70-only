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
    __slots__ = ("key", "ids", "ver")

    def __init__(self, key, ids):
        self.key, self.ids, self.ver = key, ids, 0


class Entry:
    __slots__ = ("ids", "key", "t0", "cache", "why", "served", "open")

    def __init__(self, ids, key=None):
        self.ids = ids
        self.key = clean_key(key)
        self.t0 = time.monotonic()
        self.cache = {}               # id(thread) -> (thread.ver, does this request continue it by prompt)
        self.why = ""
        self.served = False           # the engine began reading its prompt (so it is the live conversation)
        self.open = False             # its reply ended in a tool call: the thread goes on with the tool's result


class StreamOrder:
    def __init__(self, enabled: bool | None = None, capacity: int = 3):
        self.enabled = ENABLED if enabled is None else enabled
        self.capacity = max(1, capacity)      # conversations the cache holds, the live one included
        self.cv = threading.Condition()
        self.waiters: list[Entry] = []
        self.active: Entry | None = None      # picked, until its turn is over
        self.recent: list[Thread] = []        # open threads the cache holds, least recently served first

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
                self.cv.wait(left)

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
        others = len(self.waiters) > 1
        for i in range(len(self.recent) - 1, -1, -1):          # the thread served last first
            t = self.recent[i]
            for w in self.waiters:                              # (in arrival order)
                if self._belongs(w, t):
                    why = ("kept the thread" if i == len(self.recent) - 1 else "back to a parked thread") if others else ""
                    return w, why
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
        if not e.open:
            return                    # no tool call: nothing comes after it, so no place is kept for it
        if t is None:
            t = Thread(e.key, e.ids)
        else:
            t.key, t.ids, t.ver = e.key or t.key, e.ids, t.ver + 1
        self.recent.append(t)
        del self.recent[:max(0, len(self.recent) - self.capacity)]
