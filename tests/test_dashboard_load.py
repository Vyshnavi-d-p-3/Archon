"""Dashboard data loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.serve_dashboard import _load_dashboard_data, _normalize_trace, _summarize


def test_load_dashboard_respects_list_limit_aggregates_use_all_in_metrics_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Table size follows `list_limit`; KPIs are computed from all files in the metrics cap (all 5 here)."""
    monkeypatch.delenv("ARCHON_DASHBOARD_METRICS_MAX", raising=False)
    for i in range(5):
        p = tmp_path / f"trace_{i}.json"
        p.write_text(
            json.dumps(
                {
                    "trace_id": f"id{i}",
                    "task_description": f"t{i}",
                    "model_name": "m",
                    "success": True,
                    "wall_time_seconds": 1,
                    "plans": [],
                }
            ),
            encoding="utf-8",
        )
    data = _load_dashboard_data(tmp_path, list_limit=3)
    assert len(data.traces) == 3
    assert data.meta["traces_on_disk"] == 5
    assert data.meta["traces_loaded"] == 3
    assert data.meta["traces_in_response"] == 3
    assert data.meta["limit"] == 3
    assert data.meta["metrics_files_read"] == 5
    assert data.summary["trace_count"] == 5
    assert data.meta["metrics_omit_older"] is False


def test_load_dashboard_metrics_cap_omits_older_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARCHON_DASHBOARD_METRICS_MAX", "4")
    for i in range(6):
        p = tmp_path / f"trace_{i}.json"
        p.write_text(
            json.dumps(
                {
                    "trace_id": f"id{i}",
                    "task_description": f"t{i}",
                    "model_name": "m",
                    "success": i % 2 == 0,
                    "wall_time_seconds": 1,
                    "plans": [],
                }
            ),
            encoding="utf-8",
        )
    data = _load_dashboard_data(tmp_path, list_limit=2)
    assert data.meta["traces_on_disk"] == 6
    assert data.meta["metrics_omit_older"] is True
    assert data.meta["metrics_files_read"] == 4
    assert data.summary["trace_count"] == 4
    assert len(data.traces) == 2
    assert "ARCHON_DASHBOARD_METRICS_MAX" in (data.meta.get("metrics_note") or "")


def test_summarize_prioritizes_safety_in_failure_taxonomy() -> None:
    """Safety-related failure keys appear before operational keys when both exist."""
    raw = {
        "trace_id": "t1",
        "task_description": "task",
        "model_name": "m1",
        "success": False,
        "wall_time_seconds": 1.0,
        "plans": [
            {
                "steps": [
                    {
                        "step_id": "a",
                        "description": "s1",
                        "tool_call": {"tool_name": "calculator", "arguments": {}},
                        "tool_result": {"success": True, "latency_ms": 10, "output": 1},
                        "reflection": {
                            "verdict": "continue",
                            "failure_category": "unsafe_output",
                        },
                        "status": "failed",
                    },
                    {
                        "step_id": "b",
                        "description": "s2",
                        "tool_call": {"tool_name": "calculator", "arguments": {}},
                        "tool_result": {"success": True, "latency_ms": 10, "output": 1},
                        "reflection": {
                            "verdict": "retry",
                            "failure_category": "tool_arg_schema_violation",
                        },
                        "status": "failed",
                    },
                ]
            }
        ],
    }
    t = _normalize_trace(raw)
    s = _summarize([t])
    labels = [x[0] for x in s["failure_taxonomy"]]
    assert labels[0] == "unsafe_output"
    assert s["safety_failure_steps"] == 1
    assert s["total_step_count"] == 2
    assert s["safety_tagged_step_rate"] == 50.0
