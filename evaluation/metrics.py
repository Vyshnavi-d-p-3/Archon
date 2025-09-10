"""
Evaluation Metrics Module.

Computes per-step and per-task metrics for comparing model performance:
  - Tool-call accuracy: Did the agent select the correct tool?
  - JSON schema adherence: Were tool arguments valid?
  - Error-recovery success rate: Did the agent recover from failures?
  - Step efficiency: How many steps vs. expected?
  - Final answer quality: Does the answer contain expected content?

Also builds a failure-mode taxonomy distribution for analysis.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

from agent.state import (
    AgentTrace,
    FailureCategory,
    PlanStep,
    ReflectionVerdict,
    StepStatus,
)
from evaluation.benchmarks.tasks import BenchmarkTask, ExpectedStep

logger = structlog.get_logger(__name__)


# ── Per-step metrics ─────────────────────────────────────────────────

@dataclass
class StepMetrics:
    """Metrics for a single executed step against its expected step."""
    step_id: str
    expected_tool: str
    actual_tool: str
    tool_correct: bool
    schema_valid: bool
    schema_errors: list[str] = field(default_factory=list)
    required_args_present: bool = True
    arg_pattern_matches: dict[str, bool] = field(default_factory=dict)
    execution_success: bool = False
    retries_needed: int = 0
    recovered_after_retry: bool = False
    latency_ms: float = 0.0
    failure_category: Optional[FailureCategory] = None


@dataclass
class TaskMetrics:
    """Aggregate metrics for a single task execution."""
    task_id: str
    model_name: str
    # Core metrics
    tool_call_accuracy: float = 0.0       # % of steps with correct tool
    schema_adherence_rate: float = 0.0     # % of steps with valid schema
    error_recovery_rate: float = 0.0       # % of failed steps that recovered
    step_efficiency: float = 0.0           # expected_steps / actual_steps
    final_answer_score: float = 0.0        # % of expected keywords found
    # Execution stats
    total_steps_executed: int = 0
    total_expected_steps: int = 0
    total_retries: int = 0
    total_replans: int = 0
    wall_time_seconds: float = 0.0
    overall_success: bool = False
    # Per-step detail
    step_metrics: list[StepMetrics] = field(default_factory=list)
    # Failure distribution
    failure_distribution: dict[str, int] = field(default_factory=dict)


# ── Scoring Engine ──────────────────────────────────────────────────

class MetricsScorer:
    """
    Scores an AgentTrace against a BenchmarkTask to produce TaskMetrics.
    """

    def score(self, trace: AgentTrace, task: BenchmarkTask) -> TaskMetrics:
        """
        Score a trace against its benchmark task.
        """
        metrics = TaskMetrics(
            task_id=task.task_id,
            model_name=trace.model_name,
            total_steps_executed=trace.total_steps_executed,
            total_expected_steps=len(task.expected_steps),
            total_retries=trace.total_retries,
            total_replans=trace.total_replans,
            wall_time_seconds=trace.wall_time_seconds,
            overall_success=trace.success,
        )

        actual_steps = trace.all_steps()

        # ── 1. Per-step scoring ──────────────────────────────────
        step_scores = self._score_steps(actual_steps, task.expected_steps)
        metrics.step_metrics = step_scores

        # ── 2. Tool-call accuracy ────────────────────────────────
        if step_scores:
            correct = sum(1 for s in step_scores if s.tool_correct)
            metrics.tool_call_accuracy = correct / len(step_scores)

        # ── 3. Schema adherence ──────────────────────────────────
        schema_steps = [s for s in step_scores if s.actual_tool != "unknown"]
        if schema_steps:
            valid = sum(1 for s in schema_steps if s.schema_valid)
            metrics.schema_adherence_rate = valid / len(schema_steps)

        # ── 4. Error recovery rate ───────────────────────────────
        failed_then_retried = [
            s for s in step_scores if s.retries_needed > 0
        ]
        if failed_then_retried:
            recovered = sum(1 for s in failed_then_retried if s.recovered_after_retry)
            metrics.error_recovery_rate = recovered / len(failed_then_retried)

        # ── 5. Step efficiency ───────────────────────────────────
        if metrics.total_steps_executed > 0:
            metrics.step_efficiency = min(
                1.0,
                metrics.total_expected_steps / metrics.total_steps_executed,
            )

        # ── 6. Final answer quality ──────────────────────────────
        if task.expected_final_answer_contains and trace.final_answer:
            answer_lower = trace.final_answer.lower()
            matches = sum(
                1 for kw in task.expected_final_answer_contains
                if kw.lower() in answer_lower
            )
            metrics.final_answer_score = matches / len(
                task.expected_final_answer_contains
            )

        # ── 7. Failure distribution ──────────────────────────────
        failure_counts: Counter[str] = Counter()
        for step in actual_steps:
            if step.reflection and step.reflection.failure_category:
                failure_counts[step.reflection.failure_category.value] += 1
        metrics.failure_distribution = dict(failure_counts)

        return metrics

    def _score_steps(
        self,
        actual: list[PlanStep],
        expected: list[ExpectedStep],
    ) -> list[StepMetrics]:
        """
        Align actual steps to expected steps and score each.
        Uses greedy sequential matching with flexible-order support.
        """
        scores: list[StepMetrics] = []
        used_expected: set[int] = set()

        for act_step in actual:
            if act_step.status == StepStatus.SKIPPED:
                continue

            actual_tool = (
                act_step.tool_call.tool_name if act_step.tool_call else "unknown"
            )

            # Find best matching expected step
            best_match: Optional[ExpectedStep] = None
            best_idx = -1
            for i, exp in enumerate(expected):
                if i in used_expected:
                    continue
                if exp.tool == actual_tool:
                    best_match = exp
                    best_idx = i
                    break
            if best_match is None:
                # Try flexible-order steps
                for i, exp in enumerate(expected):
                    if i in used_expected and not exp.order_flexible:
                        continue
                    if exp.tool == actual_tool and exp.order_flexible:
                        best_match = exp
                        best_idx = i
                        break

            sm = StepMetrics(
                step_id=act_step.step_id,
                expected_tool=best_match.tool if best_match else "N/A",
                actual_tool=actual_tool,
                tool_correct=best_match is not None,
                schema_valid=(
                    act_step.tool_call.schema_valid
                    if act_step.tool_call else False
                ),
                schema_errors=(
                    act_step.tool_call.schema_errors
                    if act_step.tool_call else []
                ),
                execution_success=(
                    act_step.tool_result.success
                    if act_step.tool_result else False
                ),
                retries_needed=act_step.retries,
                recovered_after_retry=(
                    act_step.retries > 0
                    and act_step.status == StepStatus.COMPLETED
                ),
                latency_ms=(
                    act_step.tool_result.latency_ms
                    if act_step.tool_result else 0.0
                ),
                failure_category=(
                    act_step.reflection.failure_category
                    if act_step.reflection else None
                ),
            )

            # Check required args
            if best_match and act_step.tool_call:
                actual_args = set(act_step.tool_call.arguments.keys())
                required = set(best_match.required_args)
                sm.required_args_present = required.issubset(actual_args)

                # Check arg patterns
                for arg_name, pattern in best_match.expected_arg_patterns.items():
                    arg_val = str(act_step.tool_call.arguments.get(arg_name, ""))
                    sm.arg_pattern_matches[arg_name] = bool(
                        re.search(pattern, arg_val, re.IGNORECASE)
                    )

            if best_idx >= 0:
                used_expected.add(best_idx)

            scores.append(sm)

        return scores


# ── Aggregation across multiple runs ─────────────────────────────────

@dataclass
class ModelSummary:
    """Aggregated metrics for a model across all tasks and trials."""
    model_name: str
    num_tasks: int = 0
    num_trials: int = 0

    # Mean metrics
    mean_tool_accuracy: float = 0.0
    mean_schema_adherence: float = 0.0
    mean_error_recovery: float = 0.0
    mean_step_efficiency: float = 0.0
    mean_final_answer_score: float = 0.0
    mean_wall_time: float = 0.0
    overall_success_rate: float = 0.0

    # Std dev (for confidence intervals)
    std_tool_accuracy: float = 0.0
    std_schema_adherence: float = 0.0
    std_error_recovery: float = 0.0

    # Failure taxonomy
    failure_taxonomy: dict[str, int] = field(default_factory=dict)

    # Per-task detail
    task_metrics: list[TaskMetrics] = field(default_factory=list)


def aggregate_metrics(
    all_metrics: list[TaskMetrics],
    model_name: str,
) -> ModelSummary:
    """Aggregate TaskMetrics across tasks/trials into a ModelSummary."""
    import numpy as np

    summary = ModelSummary(
        model_name=model_name,
        num_tasks=len(set(m.task_id for m in all_metrics)),
        num_trials=len(all_metrics),
        task_metrics=all_metrics,
    )

    if not all_metrics:
        return summary

    tool_accs = [m.tool_call_accuracy for m in all_metrics]
    schema_rates = [m.schema_adherence_rate for m in all_metrics]
    recovery_rates = [m.error_recovery_rate for m in all_metrics]
    efficiencies = [m.step_efficiency for m in all_metrics]
    answer_scores = [m.final_answer_score for m in all_metrics]
    wall_times = [m.wall_time_seconds for m in all_metrics]
    successes = [1.0 if m.overall_success else 0.0 for m in all_metrics]

    summary.mean_tool_accuracy = float(np.mean(tool_accs))
    summary.mean_schema_adherence = float(np.mean(schema_rates))
    summary.mean_error_recovery = float(np.mean(recovery_rates))
    summary.mean_step_efficiency = float(np.mean(efficiencies))
    summary.mean_final_answer_score = float(np.mean(answer_scores))
    summary.mean_wall_time = float(np.mean(wall_times))
    summary.overall_success_rate = float(np.mean(successes))

    summary.std_tool_accuracy = float(np.std(tool_accs))
    summary.std_schema_adherence = float(np.std(schema_rates))
    summary.std_error_recovery = float(np.std(recovery_rates))

    # Merge failure taxonomies
    merged: Counter[str] = Counter()
    for m in all_metrics:
        merged.update(m.failure_distribution)
    summary.failure_taxonomy = dict(merged)

    return summary
