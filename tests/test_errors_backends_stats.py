"""Tests for exception hierarchy, LLM backends, and statistical analysis."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.errors import (
    AgentError,
    BudgetExceededError,
    CircuitBreakerOpenError,
    FatalAgentError,
    LLMConnectionError,
    LLMOutputParseError,
    LLMRateLimitError,
    MaxRetriesExceededError,
    RetryableError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolSchemaError,
)
from agent.llm_backends import CompletionResponse, DeterministicFakeBackend
from agent.state import FailureCategory


# ═══════════════════════════════════════════════════════════════════════
# Exception Hierarchy Tests
# ═══════════════════════════════════════════════════════════════════════

class TestExceptionHierarchy:
    """Verify the exception type hierarchy enables correct catch patterns."""

    def test_retryable_is_agent_error(self):
        exc = ToolNotFoundError("bad_tool", ["web_search", "calculator"])
        assert isinstance(exc, AgentError)
        assert isinstance(exc, RetryableError)
        assert not isinstance(exc, FatalAgentError)

    def test_fatal_is_agent_error(self):
        exc = BudgetExceededError("token", limit=1000, consumed=1500)
        assert isinstance(exc, AgentError)
        assert isinstance(exc, FatalAgentError)
        assert not isinstance(exc, RetryableError)

    def test_llm_connection_is_retryable(self):
        exc = LLMConnectionError("openai", "gpt-4o", "timeout")
        assert isinstance(exc, RetryableError)
        assert exc.failure_category == FailureCategory.TIMEOUT

    def test_llm_rate_limit_is_retryable(self):
        exc = LLMRateLimitError("openai", retry_after=30.0)
        assert isinstance(exc, RetryableError)
        assert exc.retry_after == 30.0

    def test_tool_not_found_category(self):
        exc = ToolNotFoundError("quantum_tool", ["web_search"])
        assert exc.failure_category == FailureCategory.HALLUCINATED_TOOL
        assert exc.tool_name == "quantum_tool"
        assert exc.available_tools == ["web_search"]

    def test_tool_schema_error_category(self):
        exc = ToolSchemaError(
            "calculator",
            schema_errors=["missing 'expression'"],
            provided_args={"wrong": "args"},
        )
        assert exc.failure_category == FailureCategory.TOOL_ARG_SCHEMA_VIOLATION

    def test_circuit_breaker_is_fatal(self):
        exc = CircuitBreakerOpenError(consecutive_failures=5, threshold=3)
        assert isinstance(exc, FatalAgentError)
        assert exc.failure_category == FailureCategory.INFINITE_LOOP

    def test_context_method(self):
        exc = ToolNotFoundError("bad", ["good1", "good2"])
        ctx = exc.context()
        assert ctx["error_type"] == "ToolNotFoundError"
        assert ctx["failure_category"] == "hallucinated_tool"
        assert ctx["tool_name"] == "bad"
        assert "available_tools" in ctx

    def test_catch_all_retryable(self):
        """Verify single except RetryableError catches all retryable subtypes."""
        retryable_errors = [
            ToolNotFoundError("x", []),
            ToolSchemaError("x", [], {}),
            ToolExecutionError("x", ValueError("boom")),
            LLMConnectionError("p", "m", "c"),
            LLMRateLimitError("p"),
            LLMOutputParseError("json", "raw"),
        ]
        for exc in retryable_errors:
            try:
                raise exc
            except RetryableError:
                pass  # Should always catch
            except Exception:
                pytest.fail(f"{type(exc).__name__} not caught by RetryableError")

    def test_catch_all_fatal(self):
        """Verify single except FatalAgentError catches all fatal subtypes."""
        fatal_errors = [
            BudgetExceededError("token", 100, 200),
            MaxRetriesExceededError("step", 3),
            CircuitBreakerOpenError(5, 3),
        ]
        for exc in fatal_errors:
            try:
                raise exc
            except FatalAgentError:
                pass
            except Exception:
                pytest.fail(f"{type(exc).__name__} not caught by FatalAgentError")


# ═══════════════════════════════════════════════════════════════════════
# LLM Backend Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCompletionResponse:
    def test_frozen_dataclass(self):
        resp = CompletionResponse(
            content="hello",
            input_tokens=10,
            output_tokens=5,
            model="test",
        )
        assert resp.content == "hello"
        assert resp.input_tokens == 10
        with pytest.raises(AttributeError):
            resp.content = "modified"  # type: ignore[misc]


class TestDeterministicFakeBackend:
    @pytest.mark.asyncio
    async def test_returns_default(self):
        fake = DeterministicFakeBackend(default_response='{"result": "default"}')
        resp = await fake.generate("sys", "anything")
        assert resp.content == '{"result": "default"}'
        assert resp.model == "deterministic-fake"

    @pytest.mark.asyncio
    async def test_matches_user_message(self):
        fake = DeterministicFakeBackend(
            responses={
                "tokyo": '{"city": "Tokyo"}',
                "paris": '{"city": "Paris"}',
            }
        )
        resp = await fake.generate("sys", "Find tokyo population")
        assert "Tokyo" in resp.content

        resp2 = await fake.generate("sys", "Weather in paris")
        assert "Paris" in resp2.content

    @pytest.mark.asyncio
    async def test_logs_calls(self):
        fake = DeterministicFakeBackend()
        await fake.generate("system prompt", "user message")
        await fake.generate("another system", "another user")
        assert len(fake.call_log) == 2
        assert "system prompt" in fake.call_log[0]["system"]

    @pytest.mark.asyncio
    async def test_token_estimation(self):
        fake = DeterministicFakeBackend(default_response="short")
        resp = await fake.generate("a" * 100, "b" * 100)
        assert resp.input_tokens > 0
        assert resp.output_tokens > 0

    @pytest.mark.asyncio
    async def test_model_id(self):
        fake = DeterministicFakeBackend()
        assert fake.model_id == "deterministic-fake"


# ═══════════════════════════════════════════════════════════════════════
# Statistical Analysis Tests
# ═══════════════════════════════════════════════════════════════════════

class TestBootstrapCI:
    def test_basic_ci(self):
        from evaluation.statistics import bootstrap_ci
        data = [0.8, 0.85, 0.9, 0.82, 0.88, 0.91, 0.87, 0.83]
        ci = bootstrap_ci(data, confidence=0.95)
        assert 0.80 <= ci.mean <= 0.92
        assert ci.ci_lower < ci.mean < ci.ci_upper
        assert ci.n_samples == 8

    def test_empty_data(self):
        from evaluation.statistics import bootstrap_ci
        ci = bootstrap_ci([])
        assert ci.mean == 0.0
        assert ci.ci_lower == 0.0

    def test_single_value(self):
        from evaluation.statistics import bootstrap_ci
        ci = bootstrap_ci([0.5])
        assert ci.mean == 0.5
        assert ci.ci_lower == 0.5
        assert ci.ci_upper == 0.5

    def test_identical_values(self):
        from evaluation.statistics import bootstrap_ci
        ci = bootstrap_ci([1.0, 1.0, 1.0, 1.0])
        assert ci.mean == 1.0
        assert ci.ci_width < 0.001

    def test_wide_spread(self):
        from evaluation.statistics import bootstrap_ci
        data = [0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0]
        ci = bootstrap_ci(data)
        assert ci.ci_width > 0.1  # Should have wide CI

    def test_str_format(self):
        from evaluation.statistics import bootstrap_ci
        ci = bootstrap_ci([0.8, 0.9, 0.85])
        s = str(ci)
        assert "[" in s and "]" in s


class TestCohensD:
    def test_identical_groups(self):
        from evaluation.statistics import cohens_d
        d = cohens_d([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        assert abs(d) < 0.01

    def test_large_effect(self):
        from evaluation.statistics import cohens_d
        a = [8.0, 9.0, 10.0, 8.5, 9.5]
        b = [2.0, 3.0, 4.0, 2.5, 3.5]
        d = cohens_d(a, b)
        assert d > 0.8  # Large positive effect

    def test_negative_effect(self):
        from evaluation.statistics import cohens_d
        d = cohens_d([1, 2, 3], [8, 9, 10])
        assert d < -0.8

    def test_small_samples(self):
        from evaluation.statistics import cohens_d
        d = cohens_d([1.0], [2.0])
        assert d == 0.0  # Returns 0 for n < 2


class TestCliffsDelta:
    def test_identical(self):
        from evaluation.statistics import cliffs_delta
        d = cliffs_delta([1, 2, 3], [1, 2, 3])
        assert abs(d) < 0.01

    def test_complete_dominance(self):
        from evaluation.statistics import cliffs_delta
        d = cliffs_delta([10, 20, 30], [1, 2, 3])
        assert d == 1.0  # A completely dominates B

    def test_empty(self):
        from evaluation.statistics import cliffs_delta
        d = cliffs_delta([], [1, 2, 3])
        assert d == 0.0


class TestMannWhitneyU:
    def test_identical_distributions(self):
        from evaluation.statistics import mann_whitney_u
        result = mann_whitney_u(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )
        assert not result.significant_at_005
        assert result.p_value > 0.05

    def test_different_distributions(self):
        from evaluation.statistics import mann_whitney_u
        result = mann_whitney_u(
            [80, 85, 90, 88, 92, 87, 91, 89, 86, 84],
            [20, 25, 30, 22, 28, 18, 24, 26, 21, 23],
        )
        assert result.significant_at_005
        assert result.p_value < 0.05

    def test_small_samples(self):
        from evaluation.statistics import mann_whitney_u
        result = mann_whitney_u([1], [2])
        assert not result.significant_at_005


class TestModelComparison:
    def test_full_comparison(self):
        from evaluation.statistics import compare_models
        comp = compare_models(
            metric_name="tool_accuracy",
            model_a_name="llama-3",
            model_a_scores=[0.9, 0.85, 0.88, 0.92, 0.87, 0.91, 0.86, 0.89],
            model_b_name="mistral",
            model_b_scores=[0.7, 0.65, 0.72, 0.68, 0.71, 0.66, 0.69, 0.73],
        )
        assert comp.metric_name == "tool_accuracy"
        assert comp.ci_a.mean > comp.ci_b.mean
        assert comp.effect_size.cohens_d > 0  # A is better
        assert comp.effect_size.interpretation in ("large", "medium")
        assert "llama" in comp.effect_size.favors.lower()

    def test_summary_string(self):
        from evaluation.statistics import compare_models
        comp = compare_models(
            "acc",
            "A", [0.9, 0.8, 0.85],
            "B", [0.6, 0.5, 0.55],
        )
        s = comp.summary()
        assert "acc" in s
        assert "A" in s
        assert "B" in s
