"""Tests for state models and evaluation metrics."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.state import (
    AgentTrace,
    FailureCategory,
    Plan,
    PlanStep,
    ReflectionVerdict,
    StepReflection,
    StepStatus,
    ToolCall,
    ToolResult,
    WorkingMemory,
)
from evaluation.benchmarks.tasks import (
    BENCHMARK_TASKS,
    BenchmarkTask,
    Difficulty,
    ExpectedStep,
    TaskCategory,
    get_tasks_by_category,
    get_tasks_by_difficulty,
)
from evaluation.metrics import MetricsScorer, TaskMetrics, aggregate_metrics


# ── State Model Tests ────────────────────────────────────────────────

class TestPlanStep:
    def test_default_status(self):
        step = PlanStep(description="Test step")
        assert step.status == StepStatus.PENDING
        assert step.retries == 0
        assert step.step_id  # auto-generated

    def test_serialization(self):
        step = PlanStep(
            description="Search for data",
            expected_tool="web_search",
            tool_call=ToolCall(
                tool_name="web_search",
                arguments={"query": "test"},
            ),
            tool_result=ToolResult(success=True, output=["result"]),
        )
        data = step.model_dump()
        restored = PlanStep.model_validate(data)
        assert restored.description == step.description
        assert restored.tool_call.tool_name == "web_search"


class TestWorkingMemory:
    def test_record_and_retrieve(self):
        mem = WorkingMemory()
        mem.record_step_output("step1", {"data": [1, 2, 3]})
        mem.add_fact("city", "Tokyo")
        assert "step1" in mem.step_outputs
        assert mem.facts["city"] == "Tokyo"

    def test_context_summary(self):
        mem = WorkingMemory()
        mem.record_step_output("s1", "first result")
        mem.record_step_output("s2", "second result")
        mem.add_fact("key", "value")
        summary = mem.get_context_summary()
        assert "s1" in summary
        assert "s2" in summary
        assert "key" in summary

    def test_empty_context_summary(self):
        mem = WorkingMemory()
        assert mem.get_context_summary() == "(no prior context)"


class TestAgentTrace:
    def _make_trace(self) -> AgentTrace:
        return AgentTrace(
            task_description="Test task",
            plans=[
                Plan(
                    task_description="Test",
                    steps=[
                        PlanStep(
                            description="Step 1",
                            status=StepStatus.COMPLETED,
                        ),
                        PlanStep(
                            description="Step 2",
                            status=StepStatus.FAILED,
                        ),
                        PlanStep(
                            description="Step 3",
                            status=StepStatus.COMPLETED,
                        ),
                    ],
                )
            ],
        )

    def test_all_steps(self):
        trace = self._make_trace()
        assert len(trace.all_steps()) == 3

    def test_completed_steps(self):
        trace = self._make_trace()
        assert len(trace.completed_steps()) == 2

    def test_failed_steps(self):
        trace = self._make_trace()
        assert len(trace.failed_steps()) == 1

    def test_json_roundtrip(self):
        trace = self._make_trace()
        json_str = trace.model_dump_json()
        restored = AgentTrace.model_validate_json(json_str)
        assert len(restored.all_steps()) == 3


# ── Benchmark Task Tests ─────────────────────────────────────────────

class TestBenchmarkTasks:
    def test_all_tasks_have_ids(self):
        for task in BENCHMARK_TASKS:
            assert task.task_id
            assert task.description
            assert len(task.expected_steps) > 0

    def test_filter_by_category(self):
        reasoning = get_tasks_by_category(TaskCategory.MULTI_STEP_REASONING)
        assert len(reasoning) > 0
        for t in reasoning:
            assert t.category == TaskCategory.MULTI_STEP_REASONING

    def test_filter_by_difficulty(self):
        easy = get_tasks_by_difficulty(Difficulty.EASY)
        assert len(easy) > 0

    def test_unique_ids(self):
        ids = [t.task_id for t in BENCHMARK_TASKS]
        assert len(ids) == len(set(ids)), "Duplicate task IDs found"


# ── Metrics Scorer Tests ─────────────────────────────────────────────

class TestMetricsScorer:
    def setup_method(self):
        self.scorer = MetricsScorer()

    def _make_task(self) -> BenchmarkTask:
        return BenchmarkTask(
            task_id="test_001",
            description="Test task",
            category=TaskCategory.CALCULATION,
            difficulty=Difficulty.EASY,
            expected_steps=[
                ExpectedStep(
                    tool="calculator",
                    description="Calculate something",
                    required_args=["expression"],
                ),
            ],
            success_criteria="Correct result",
            expected_final_answer_contains=["42"],
        )

    def _make_trace(
        self,
        tool_name: str = "calculator",
        success: bool = True,
        answer: str = "The answer is 42.",
    ) -> AgentTrace:
        return AgentTrace(
            task_description="Test task",
            model_name="test-model",
            plans=[
                Plan(
                    task_description="Test",
                    steps=[
                        PlanStep(
                            description="Calculate",
                            expected_tool="calculator",
                            status=StepStatus.COMPLETED if success else StepStatus.FAILED,
                            tool_call=ToolCall(
                                tool_name=tool_name,
                                arguments={"expression": "6 * 7"},
                                schema_valid=True,
                            ),
                            tool_result=ToolResult(
                                success=success,
                                output=42 if success else None,
                                latency_ms=10.0,
                            ),
                        ),
                    ],
                )
            ],
            final_answer=answer,
            success=success,
            total_steps_executed=1,
        )

    def test_perfect_score(self):
        task = self._make_task()
        trace = self._make_trace()
        metrics = self.scorer.score(trace, task)
        assert metrics.tool_call_accuracy == 1.0
        assert metrics.schema_adherence_rate == 1.0
        assert metrics.final_answer_score == 1.0

    def test_wrong_tool(self):
        task = self._make_task()
        trace = self._make_trace(tool_name="web_search")
        metrics = self.scorer.score(trace, task)
        assert metrics.tool_call_accuracy == 0.0

    def test_missing_answer_keyword(self):
        task = self._make_task()
        trace = self._make_trace(answer="No relevant info found.")
        metrics = self.scorer.score(trace, task)
        assert metrics.final_answer_score == 0.0

    def test_error_recovery_tracking(self):
        task = self._make_task()
        trace = AgentTrace(
            task_description="Test",
            model_name="test-model",
            plans=[
                Plan(
                    task_description="Test",
                    steps=[
                        PlanStep(
                            description="Calculate",
                            status=StepStatus.COMPLETED,
                            retries=2,  # Failed twice, succeeded third time
                            tool_call=ToolCall(
                                tool_name="calculator",
                                arguments={"expression": "6*7"},
                                schema_valid=True,
                            ),
                            tool_result=ToolResult(
                                success=True, output=42, latency_ms=5.0,
                            ),
                        ),
                    ],
                )
            ],
            success=True,
            total_steps_executed=1,
            total_retries=2,
            final_answer="42",
        )
        metrics = self.scorer.score(trace, task)
        assert metrics.error_recovery_rate == 1.0  # Recovered after retries


class TestAggregation:
    def test_aggregate_empty(self):
        summary = aggregate_metrics([], "test-model")
        assert summary.num_tasks == 0
        assert summary.mean_tool_accuracy == 0.0

    def test_aggregate_multiple(self):
        metrics_list = [
            TaskMetrics(
                task_id="t1",
                model_name="m1",
                tool_call_accuracy=1.0,
                schema_adherence_rate=0.8,
                error_recovery_rate=0.5,
                step_efficiency=1.0,
                final_answer_score=1.0,
                overall_success=True,
                wall_time_seconds=2.0,
            ),
            TaskMetrics(
                task_id="t2",
                model_name="m1",
                tool_call_accuracy=0.5,
                schema_adherence_rate=1.0,
                error_recovery_rate=0.0,
                step_efficiency=0.8,
                final_answer_score=0.5,
                overall_success=False,
                wall_time_seconds=4.0,
            ),
        ]
        summary = aggregate_metrics(metrics_list, "m1")
        assert summary.num_tasks == 2
        assert summary.mean_tool_accuracy == 0.75
        assert summary.overall_success_rate == 0.5
        assert summary.mean_wall_time == 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
