"""
Reproducibility utilities for research-grade evaluation runs.

- Centralized random seeding for *library* randomness (Python `random`, NumPy). Remote LLM
  APIs are not guaranteed to be bit-reproducible; manifests record temperature and model IDs
  so reported experiments remain auditable and comparable, not bit-identical, unless using
  a deterministic local backend.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from config.version import TRACE_SCHEMA_VERSION, package_version


def set_global_seeds(seed: int) -> None:
    """Set Python and NumPy RNG seeds (call at start of an eval run or statistical script)."""
    random.seed(seed)
    np.random.seed(seed)


def env_eval_seed() -> int:
    """Read `ARCHON_EVAL_SEED` or default 42 (documented in RESEARCH.md)."""
    raw = os.getenv("ARCHON_EVAL_SEED", "42")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 42


@dataclass(frozen=True)
class RunManifest:
    """Structured metadata for a single evaluation or comparison run (serializable to JSON)."""

    archon_version: str
    trace_schema_version: str
    started_at_utc: str
    completed_at_utc: str
    eval_seed: int
    use_mock_tools: bool
    num_trials: int
    task_ids: list[str]
    models: list[dict[str, str]]
    config_fingerprint_sha256: str
    non_determinism_note: str = (
        "LLM and cloud APIs may not repeat outputs even at temperature=0; "
        "use mock tools and deterministic fakes for bit-level replay where needed."
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def build_config_fingerprint(
    task_ids: list[str],
    models: list[dict[str, str]],
    num_trials: int,
    use_mock: bool,
    eval_seed: int,
) -> str:
    """Stable SHA-256 of primary experimental knobs (for manifest + paper supplemental)."""
    payload = {
        "task_ids": sorted(task_ids),
        "models": models,
        "num_trials": num_trials,
        "use_mock_tools": use_mock,
        "eval_seed": eval_seed,
        "trace_schema": TRACE_SCHEMA_VERSION,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_run_manifest(
    path: str,
    manifest: RunManifest,
) -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8")


def make_eval_manifest(
    *,
    task_ids: list[str],
    models: list[dict[str, str]],
    num_trials: int,
    use_mock: bool,
    eval_seed: int,
    started: datetime,
    completed: datetime,
) -> RunManifest:
    return RunManifest(
        archon_version=package_version(),
        trace_schema_version=TRACE_SCHEMA_VERSION,
        started_at_utc=started.astimezone(UTC).isoformat(),
        completed_at_utc=completed.astimezone(UTC).isoformat(),
        eval_seed=eval_seed,
        use_mock_tools=use_mock,
        num_trials=num_trials,
        task_ids=task_ids,
        models=list(models),
        config_fingerprint_sha256=build_config_fingerprint(
            task_ids, models, num_trials, use_mock, eval_seed
        ),
    )
