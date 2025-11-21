"""Dashboard data loading."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.serve_dashboard import _load_dashboard_data


def test_load_dashboard_respects_limit(tmp_path: Path) -> None:
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
    data = _load_dashboard_data(tmp_path, limit=3)
    assert len(data.traces) == 3
    assert data.meta["traces_on_disk"] == 5
    assert data.meta["traces_loaded"] == 3
    assert data.meta["limit"] == 3
