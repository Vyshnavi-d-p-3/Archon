"""Tests for the prompt optimization engine."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.prompt_optimizer import (
    FailureAnalyzer,
    FailurePattern,
    OptimizationReport,
    PromptMutator,
    PromptOptimizer,
    PromptVariant,
)
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
)


# ── Fixtures ─────────────────────────────────────────────────────────

def _make_trace(
    steps_config: list[tuple[str, str, bool, FailureCategory | None]],
    success: bool = True,
) -> AgentTrace:
    """
    Build a trace from step configs.
    Each tuple: (description, tool_name, succeeded, failure_category)
    """
    steps = []
    for desc, tool, succeeded, failure in steps_config:
        reflection = None
        if failure:
            reflection = StepReflection(
                verdict=ReflectionVerdict.RETRY,
                reasoning=f"Failed: {failure.value}",
                failure_category=failure,
                confidence=0.8,
            )
        elif succeeded:
            reflection = StepReflection(
                verdict=ReflectionVerdict.CONTINUE,
                reasoning="OK",
                confidence=0.9,
            )

        steps.append(PlanStep(
            description=desc,
            expected_tool=tool,
            status=StepStatus.COMPLETED if succeeded else StepStatus.FAILED,
            retries=1 if failure else 0,
            tool_call=ToolCall(tool_name=tool, arguments={"q": "test"}, schema_valid=succeeded),
            tool_result=ToolResult(
                success=succeeded,
                output="ok" if succeeded else None,
                error=f"{failure.value} error" if failure else None,
                latency_ms=100,
            ),
            reflection=reflection,
        ))

    return AgentTrace(
        task_description="Test task",
        model_name="test-model",
        plans=[Plan(task_description="Test", steps=steps)],
        success=success,
        total_steps_executed=len(steps),
        total_retries=sum(1 for _, _, s, _ in steps_config if not s),
    )


def _make_sample_traces() -> list[AgentTrace]:
    """Build a set of traces with various failure patterns."""
    return [
        _make_trace([
            ("Search data", "web_search", True, None),
            ("Calculate result", "calculator", False, FailureCategory.TOOL_ARG_SCHEMA_VIOLATION),
            ("Calculate again", "calculator", True, None),
        ]),
        _make_trace([
            ("Search info", "web_search", True, None),
            ("Use quantum tool", "quantum_tool", False, FailureCategory.HALLUCINATED_TOOL),
            ("Fallback search", "web_search", True, None),
        ]),
        _make_trace([
            ("Fetch page", "web_fetch", False, FailureCategory.TOOL_EXECUTION_FAILURE),
            ("Search instead", "web_search", True, None),
            ("Analyze text", "text_analysis", False, FailureCategory.OUTPUT_PARSE_ERROR),
        ]),
        _make_trace([
            ("Search A", "web_search", True, None),
            ("Search B", "web_search", True, None),
            ("Calculate", "calculator", False, FailureCategory.TOOL_ARG_SCHEMA_VIOLATION),
            ("Write file", "file_write", True, None),
        ]),
        _make_trace([
            ("Ingest doc", "rag_ingest", True, None),
            ("Search KB", "rag_search", False, FailureCategory.TOOL_EXECUTION_FAILURE),
            ("Search KB retry", "rag_search", True, None),
        ]),
    ]


# ── Failure Analyzer Tests ──────────────────────────────────────────

class TestFailureAnalyzer:
    def test_basic_analysis(self):
        analyzer = FailureAnalyzer()
        traces = _make_sample_traces()
        analysis = analyzer.analyze(traces)

        assert analysis.total_traces == 5
        assert analysis.total_steps > 0
        assert analysis.total_failures > 0
        assert 0 < analysis.failure_rate < 1
        assert len(analysis.patterns) > 0

    def test_patterns_sorted_by_impact(self):
        analyzer = FailureAnalyzer()
        traces = _make_sample_traces()
        analysis = analyzer.analyze(traces)

        # First pattern should have highest impact
        for i in range(len(analysis.patterns) - 1):
            assert analysis.patterns[i].impact_score >= analysis.patterns[i + 1].impact_score

    def test_schema_violations_detected(self):
        analyzer = FailureAnalyzer()
        traces = _make_sample_traces()
        analysis = analyzer.analyze(traces)

        schema_patterns = [
            p for p in analysis.patterns
            if p.category == FailureCategory.TOOL_ARG_SCHEMA_VIOLATION
        ]
        assert len(schema_patterns) == 1
        assert schema_patterns[0].frequency == 2  # Two schema violations in sample

    def test_hallucinated_tool_detected(self):
        analyzer = FailureAnalyzer()
        traces = _make_sample_traces()
        analysis = analyzer.analyze(traces)

        hallucinated = [
            p for p in analysis.patterns
            if p.category == FailureCategory.HALLUCINATED_TOOL
        ]
        assert len(hallucinated) == 1

    def test_recommendations_generated(self):
        analyzer = FailureAnalyzer()
        traces = _make_sample_traces()
        analysis = analyzer.analyze(traces)

        assert len(analysis.recommendations) > 0
        # Should mention few-shot for schema violations
        rec_text = " ".join(analysis.recommendations).lower()
        assert "few-shot" in rec_text or "schema" in rec_text or "tool" in rec_text

    def test_top_failing_tools(self):
        analyzer = FailureAnalyzer()
        traces = _make_sample_traces()
        analysis = analyzer.analyze(traces)

        assert len(analysis.top_failing_tools) > 0
        tool_names = [t for t, _ in analysis.top_failing_tools]
        assert "calculator" in tool_names or "web_fetch" in tool_names

    def test_empty_traces(self):
        analyzer = FailureAnalyzer()
        analysis = analyzer.analyze([])
        assert analysis.total_traces == 0
        assert analysis.failure_rate == 0
        assert len(analysis.patterns) == 0

    def test_no_failures(self):
        analyzer = FailureAnalyzer()
        traces = [_make_trace([
            ("Step 1", "web_search", True, None),
            ("Step 2", "calculator", True, None),
        ])]
        analysis = analyzer.analyze(traces)
        assert analysis.total_failures == 0
        assert analysis.failure_rate == 0


# ── Prompt Mutator Tests ────────────────────────────────────────────

class TestPromptMutator:
    def test_generates_variants(self):
        analyzer = FailureAnalyzer()
        traces = _make_sample_traces()
        analysis = analyzer.analyze(traces)

        mutator = PromptMutator(tool_names=["web_search", "calculator", "rag_search"])
        variants = mutator.generate_variants(analysis, base_executor_prompt="You are an executor.")

        assert len(variants) > 0
        assert len(variants) <= 3

    def test_variant_has_mutations(self):
        analyzer = FailureAnalyzer()
        traces = _make_sample_traces()
        analysis = analyzer.analyze(traces)

        mutator = PromptMutator(tool_names=["web_search", "calculator"])
        variants = mutator.generate_variants(analysis)

        for variant in variants:
            assert len(variant.mutations) > 0
            assert variant.variant_id
            assert variant.name

    def test_schema_mutation_adds_few_shot(self):
        analyzer = FailureAnalyzer()
        # Trace with only schema violations
        traces = [_make_trace([
            ("Calc", "calculator", False, FailureCategory.TOOL_ARG_SCHEMA_VIOLATION),
            ("Calc2", "calculator", False, FailureCategory.TOOL_ARG_SCHEMA_VIOLATION),
        ], success=False)]
        analysis = analyzer.analyze(traces)

        mutator = PromptMutator()
        variants = mutator.generate_variants(analysis, base_executor_prompt="Base prompt.")

        # The first variant should target schema violations with few-shot
        assert any(
            "few_shot" in m.mutation_type or "few-shot" in m.description.lower()
            for v in variants
            for m in v.mutations
        )

    def test_hallucination_mutation_emphasizes_tools(self):
        traces = [_make_trace([
            ("Bad tool", "fake_tool", False, FailureCategory.HALLUCINATED_TOOL),
        ], success=False)]
        analysis = FailureAnalyzer().analyze(traces)

        mutator = PromptMutator(tool_names=["web_search", "calculator"])
        variants = mutator.generate_variants(analysis, base_executor_prompt="Base.")

        # Should include tool list emphasis
        combined = " ".join(v.executor_prompt for v in variants)
        assert "AVAILABLE TOOLS" in combined or "tool" in combined.lower()

    def test_mutated_prompt_is_longer(self):
        traces = _make_sample_traces()
        analysis = FailureAnalyzer().analyze(traces)

        base = "You are an executor."
        mutator = PromptMutator(tool_names=["web_search"])
        variants = mutator.generate_variants(analysis, base_executor_prompt=base)

        for variant in variants:
            # Mutated prompt should be longer than base
            assert len(variant.executor_prompt) >= len(base)

    def test_no_failures_no_variants(self):
        traces = [_make_trace([("OK", "web_search", True, None)])]
        analysis = FailureAnalyzer().analyze(traces)
        mutator = PromptMutator()
        variants = mutator.generate_variants(analysis)
        assert len(variants) == 0


# ── Prompt Optimizer Integration ─────────────────────────────────────

class TestPromptOptimizer:
    def test_analyze_and_propose(self):
        optimizer = PromptOptimizer(tool_names=["web_search", "calculator", "rag_search"])
        traces = _make_sample_traces()

        analysis, variants = optimizer.analyze_and_propose(
            traces,
            base_executor_prompt="You are an executor module.",
            base_planner_prompt="You are a planner.",
        )

        assert analysis.total_traces == 5
        assert len(variants) > 0

    def test_generate_report_estimated(self):
        optimizer = PromptOptimizer(tool_names=["web_search", "calculator"])
        traces = _make_sample_traces()

        analysis, variants = optimizer.analyze_and_propose(traces)
        report = optimizer.generate_report(analysis, variants)

        assert report.variants_tested > 0
        assert len(report.results) > 0
        assert report.summary

    def test_generate_report_with_scores(self):
        optimizer = PromptOptimizer()
        traces = _make_sample_traces()
        analysis, variants = optimizer.analyze_and_propose(traces)

        baseline = {"tool_accuracy": 0.75, "schema_adherence": 0.70, "error_recovery": 0.40}
        variant_scores = [
            {"tool_accuracy": 0.82, "schema_adherence": 0.85, "error_recovery": 0.55},
        ]

        report = optimizer.generate_report(analysis, variants, baseline, variant_scores)

        assert report.results[0].overall_improvement > 0
        assert report.results[0].recommendation in ("adopt", "test_further", "reject")
        assert report.best_variant is not None

    def test_format_report(self):
        optimizer = PromptOptimizer()
        traces = _make_sample_traces()
        analysis, variants = optimizer.analyze_and_propose(traces)
        report = optimizer.generate_report(analysis, variants)

        formatted = report.format_report()
        assert "ARCHON PROMPT OPTIMIZATION REPORT" in formatted
        assert "Failure Analysis" in formatted
        assert "Recommendations" in formatted

    def test_report_rejects_bad_variant(self):
        optimizer = PromptOptimizer()
        traces = _make_sample_traces()
        analysis, variants = optimizer.analyze_and_propose(traces)

        baseline = {"tool_accuracy": 0.90, "schema_adherence": 0.95}
        variant_scores = [
            {"tool_accuracy": 0.85, "schema_adherence": 0.88},  # Worse
        ]

        report = optimizer.generate_report(analysis, variants, baseline, variant_scores)
        assert report.results[0].recommendation == "reject"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
