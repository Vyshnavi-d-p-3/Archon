"""Shared helpers for dashboard production reliability (rate limits, safe JSONL appends)."""

from __future__ import annotations

import fcntl
import json
import time
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock
from typing import Any


class SlidingWindowRateLimiter:
    """
    In-process sliding window: at most `max_events` per `window_sec` per key.
    Thread-safe. For multi-node deployments, enforce limits at a reverse proxy or API gateway.
    """

    def __init__(self, max_events: int, window_sec: float) -> None:
        self._max = max(1, max_events)
        self._window = max(0.1, float(window_sec))
        self._lock = Lock()
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._t = time.monotonic

    def allow(self, key: str) -> bool:
        now = self._t()
        cutoff = now - self._window
        with self._lock:
            q = self._events[key]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self._max:
                return False
            q.append(now)
            return True


def append_jsonl_line(path: Path, record: dict[str, Any], *, use_flock: bool = True) -> None:
    """Append one JSON line; optional POSIX flock around the write for safer concurrent appends."""
    line = json.dumps(record, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        if use_flock and hasattr(fcntl, "flock"):
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
        finally:
            if use_flock and hasattr(fcntl, "flock"):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
