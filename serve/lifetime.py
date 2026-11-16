"""Lifetime token counters for the web dashboard (GET /metrics -> "lifetime").

Service.totals only lives as long as the server process. This keeps the sums that matter across restarts in a
small JSON file next to the run config (strata-<model>.lifetime.json): input / output / cached prompt tokens,
request counts, engine time, plus one bucket per day for the last DAYS days (the dashboard's daily chart).
Writes are throttled and atomic; a missing or damaged file just starts the counters again.
"""
import json
import os
import threading
import time
from datetime import date
from pathlib import Path

DAYS = 60
SAVE_EVERY_S = 5.0
_KEYS = ("requests", "prompt_tokens", "reused", "output_tokens", "prompt_ms", "decode_ms")


def _zero() -> dict:
    return {k: 0 for k in _KEYS}


class Lifetime:
    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path else None
        self.lock = threading.Lock()
        self.data = {"since": time.time(), "total": _zero(), "days": {}}
        self.last_save = 0.0
        self.dirty = False
        if self.path:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self.data["since"] = float(raw.get("since") or self.data["since"])
                for k in _KEYS:
                    self.data["total"][k] = raw.get("total", {}).get(k, 0) or 0
                self.data["days"] = {d: {k: v.get(k, 0) or 0 for k in _KEYS}
                                     for d, v in (raw.get("days") or {}).items() if isinstance(v, dict)}
            except (OSError, ValueError, AttributeError, TypeError):
                pass

    def record(self, prompt_tokens: int | None = 0, reused: int | None = 0, output_tokens: int | None = 0,
               prompt_ms: float | None = 0.0, decode_ms: float | None = 0.0, when: float | None = None) -> None:
        day = date.fromtimestamp(when or time.time()).isoformat()
        with self.lock:
            bucket = self.data["days"].setdefault(day, _zero())
            for target in (self.data["total"], bucket):
                target["requests"] += 1
                target["prompt_tokens"] += prompt_tokens or 0
                target["reused"] += reused or 0
                target["output_tokens"] += output_tokens or 0
                target["prompt_ms"] += prompt_ms or 0.0
                target["decode_ms"] += decode_ms or 0.0
            for old in sorted(self.data["days"])[:-DAYS]:
                del self.data["days"][old]
            self.dirty = True
        self.save()

    def save(self, force: bool = False) -> None:
        if not self.path or not self.dirty:
            return
        now = time.time()
        if not force and now - self.last_save < SAVE_EVERY_S:
            return
        with self.lock:
            text = json.dumps(self.data, indent=1)
            self.dirty = False
        self.last_save = now
        try:
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            self.dirty = True

    def view(self) -> dict:
        with self.lock:
            return {"since": self.data["since"], "persisted": self.path is not None,
                    **self.data["total"], "days": json.loads(json.dumps(self.data["days"]))}
