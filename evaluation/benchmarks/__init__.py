"""
Benchmark task suites: core (published baseline) and optional extended (harder / longer).
"""

from __future__ import annotations

from evaluation.benchmarks.extended_tasks import EXTENDED_TASKS
from evaluation.benchmarks.tasks import BENCHMARK_TASKS, BenchmarkTask


def load_benchmark_tasks(*, include_extended: bool = False) -> list[BenchmarkTask]:
    """
    Return tasks for evaluation. Core suite only by default; extended adds RAG-heavy and
    long-horizon tasks (higher cost / runtime—enable explicitly for ablations).
    """
    base = list(BENCHMARK_TASKS)
    if not include_extended:
        return base
    return base + list(EXTENDED_TASKS)
