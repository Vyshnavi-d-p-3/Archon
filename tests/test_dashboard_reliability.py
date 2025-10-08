"""Tests for dashboard reliability helpers (rate limit, JSONL append)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from scripts.dashboard_reliability import SlidingWindowRateLimiter, append_jsonl_line


def test_sliding_window_rate_limit_allows_up_to_max() -> None:
    limiter = SlidingWindowRateLimiter(max_events=3, window_sec=1.0)
    assert limiter.allow("a")
    assert limiter.allow("a")
    assert limiter.allow("a")
    assert not limiter.allow("a")


def test_sliding_window_independent_keys() -> None:
    limiter = SlidingWindowRateLimiter(max_events=1, window_sec=10.0)
    assert limiter.allow("u1")
    assert not limiter.allow("u1")
    assert limiter.allow("u2")


def test_sliding_window_recovers_after_window() -> None:
    limiter = SlidingWindowRateLimiter(max_events=1, window_sec=0.1)
    assert limiter.allow("k")
    assert not limiter.allow("k")
    time.sleep(0.2)
    assert limiter.allow("k")


def test_append_jsonl_line_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    append_jsonl_line(p, {"x": 1, "e": "ingest"}, use_flock=True)
    append_jsonl_line(p, {"x": 2, "e": "ingest"}, use_flock=True)
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"x": 1, "e": "ingest"}
    assert json.loads(lines[1]) == {"x": 2, "e": "ingest"}
