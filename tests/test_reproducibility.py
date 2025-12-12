"""Reproducibility helpers and run manifest contracts."""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path

import numpy as np
import pytest

from config.version import (
    TRACE_SCHEMA_VERSION as TRACE_SCHEMA_VERSION_CFG,
    package_version,
)
from evaluation.benchmarks import load_benchmark_tasks
from evaluation.reproducibility import (
    TRACE_SCHEMA_VERSION,
    build_config_fingerprint,
    env_eval_seed,
    make_eval_manifest,
    set_global_seeds,
    write_run_manifest,
)


def test_trace_schema_version_matches_config() -> None:
    assert TRACE_SCHEMA_VERSION == TRACE_SCHEMA_VERSION_CFG


def test_config_fingerprint_stable() -> None:
    fp1 = build_config_fingerprint(
        task_ids=["a", "b"],
        models=[{"provider": "openai", "model": "m"}],
        num_trials=2,
        use_mock=True,
        eval_seed=7,
    )
    fp2 = build_config_fingerprint(
        task_ids=["b", "a"],
        models=[{"provider": "openai", "model": "m"}],
        num_trials=2,
        use_mock=True,
        eval_seed=7,
    )
    assert fp1 == fp2
    assert len(fp1) == 64


def test_set_global_seeds_deterministic_numpy() -> None:
    set_global_seeds(12345)
    a = np.random.random(3).tolist()
    set_global_seeds(12345)
    b = np.random.random(3).tolist()
    assert a == b


def test_write_run_manifest_roundtrip(tmp_path: Path) -> None:
    from datetime import datetime

    m = make_eval_manifest(
        task_ids=["t1"],
        models=[{"provider": "x", "model": "y"}],
        num_trials=1,
        use_mock=True,
        eval_seed=99,
        started=datetime(2026, 1, 1, tzinfo=UTC),
        completed=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
    )
    p = tmp_path / "m.json"
    write_run_manifest(str(p), m)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["eval_seed"] == 99
    assert data["archon_version"] == package_version()
    assert data["trace_schema_version"] == TRACE_SCHEMA_VERSION


def test_load_benchmark_default_smaller_than_extended() -> None:
    core = load_benchmark_tasks(include_extended=False)
    all_t = load_benchmark_tasks(include_extended=True)
    assert len(all_t) > len(core)


def test_env_eval_seed_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARCHON_EVAL_SEED", raising=False)
    assert env_eval_seed() == 42
