"""Tests for the middleware chain and interceptors."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.errors import BudgetExceededError
from agent.middleware import (
    AgentTelemetry,
    MiddlewareChain,
    RateLimitInterceptor,
    TelemetryInterceptor,
    TokenBudget,
    TokenBudgetInterceptor,
    TracingInterceptor,
    build_default_middleware,
)
from agent.state import (
    PlanStep,
    ReflectionVerdict,
    StepReflection,
    StepStatus,
    ToolCall,
    ToolResult,
    WorkingMemory,
)


# ── Token Budget ─────────────────────────────────────────────────────

class TestTokenBudget:
    def test_initial_state(self):
        b = TokenBudget(max_input_tokens=1000, max_output_tokens=500)
        assert b.input_remaining == 1000
        assert b.output_remaining == 500
        assert not b.is_exhausted()

    def test_record_usage(self):
        b = TokenBudget(max_input_tokens=1000, max_output_tokens=500)
        b.record_usage(input_tokens=400, output_tokens=200)
        assert b.input_remaining == 600
        assert b.output_remaining == 300
        assert b.llm_calls == 1

    def test_exhaustion(self):
        b = TokenBudget(max_input_tokens=100, max_output_tokens=50)
        b.record_usage(input_tokens=100, output_tokens=50)
        assert b.is_exhausted()
        assert b.input_remaining == 0

    def test_cost_tracking(self):
        b = TokenBudget(
            max_total_cost_usd=1.00,
            input_cost_per_million=0.15,
            output_cost_per_million=0.60,
        )
        b.record_usage(input_tokens=1_000_000, output_tokens=100_000)
        # 1M * 0.15/1M + 100K * 0.60/1M = 0.15 + 0.06 = 0.21
        assert abs(b.total_cost_usd - 0.21) < 0.001

    def test_cost_exhaustion(self):
        b = TokenBudget(max_total_cost_usd=0.01)
        b.record_usage(input_tokens=1_000_000, output_tokens=1_000_000)
        assert b.is_exhausted()

    def test_summary(self):
        b = TokenBudget()
        b.record_usage(100, 50)
        s = b.summary()
        assert "input_tokens" in s
        assert "cost_usd" in s
        assert s["llm_calls"] == 1


# ── Tracing Interceptor ─────────────────────────────────────────────

class TestTracingInterceptor:
    @pytest.mark.asyncio
    async def test_records_spans(self):
        tracer = TracingInterceptor(trace_id="test-123")
        step = PlanStep(description="Test step", status=StepStatus.RUNNING)
        memory = WorkingMemory()

        result = await tracer.before_step(step, memory)
        assert result is not None
        assert hasattr(step, "_span_id")

        reflection = StepReflection(
            verdict=ReflectionVerdict.CONTINUE,
            reasoning="OK",
            confidence=0.9,
        )
        await tracer.after_step(step, reflection, memory)

        assert len(tracer.spans) == 1
        span = tracer.spans[0]
        assert span["trace_id"] == "test-123"
        assert span["verdict"] == "continue"
        assert span["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_multiple_spans(self):
        tracer = TracingInterceptor()
        memory = WorkingMemory()
        reflection = StepReflection(verdict=ReflectionVerdict.CONTINUE, reasoning="OK", confidence=0.9)

        for i in range(3):
            step = PlanStep(description=f"Step {i}")
            await tracer.before_step(step, memory)
            await tracer.after_step(step, reflection, memory)

        assert len(tracer.spans) == 3


# ── Token Budget Interceptor ────────────────────────────────────────

class TestTokenBudgetInterceptor:
    @pytest.mark.asyncio
    async def test_allows_within_budget(self):
        budget = TokenBudget(max_input_tokens=10000)
        interceptor = TokenBudgetInterceptor(budget=budget)
        step = PlanStep(description="Test")
        memory = WorkingMemory()

        result = await interceptor.before_step(step, memory)
        assert result is not None

    @pytest.mark.asyncio
    async def test_raises_on_exhaustion(self):
        budget = TokenBudget(max_input_tokens=0)  # Already exhausted
        interceptor = TokenBudgetInterceptor(budget=budget)
        step = PlanStep(description="Test")
        memory = WorkingMemory()

        with pytest.raises(BudgetExceededError):
            await interceptor.before_step(step, memory)


# ── Telemetry Interceptor ───────────────────────────────────────────

class TestTelemetryInterceptor:
    @pytest.mark.asyncio
    async def test_counts_steps(self):
        telemetry = TelemetryInterceptor()
        memory = WorkingMemory()

        step = PlanStep(description="Test", status=StepStatus.COMPLETED)
        step.tool_call = ToolCall(tool_name="calculator", arguments={"expression": "2+2"})
        step.tool_result = ToolResult(success=True, output=4, latency_ms=5.0)

        await telemetry.before_step(step, memory)
        reflection = StepReflection(
            verdict=ReflectionVerdict.CONTINUE, reasoning="OK", confidence=0.9
        )
        await telemetry.after_step(step, reflection, memory)

        t = telemetry.telemetry
        assert t.steps_attempted == 1
        assert t.steps_succeeded == 1
        assert "calculator" in t.tool_latencies_ms
        assert t.verdict_counts["continue"] == 1

    @pytest.mark.asyncio
    async def test_tracks_failures(self):
        telemetry = TelemetryInterceptor()
        memory = WorkingMemory()
        step = PlanStep(description="Fail", status=StepStatus.FAILED)

        await telemetry.before_step(step, memory)
        from agent.state import FailureCategory
        reflection = StepReflection(
            verdict=ReflectionVerdict.RETRY,
            reasoning="Schema error",
            failure_category=FailureCategory.TOOL_ARG_SCHEMA_VIOLATION,
            confidence=0.8,
        )
        await telemetry.after_step(step, reflection, memory)

        t = telemetry.telemetry
        assert t.steps_failed == 1
        assert t.failure_counts["tool_arg_schema_violation"] == 1


# ── Middleware Chain ─────────────────────────────────────────────────

class TestMiddlewareChain:
    @pytest.mark.asyncio
    async def test_chain_executes_all(self):
        tracer = TracingInterceptor()
        telemetry = TelemetryInterceptor()
        chain = MiddlewareChain([tracer, telemetry])

        step = PlanStep(description="Test", status=StepStatus.COMPLETED)
        memory = WorkingMemory()

        result = await chain.run_before(step, memory)
        assert result is not None

        reflection = StepReflection(
            verdict=ReflectionVerdict.CONTINUE, reasoning="OK", confidence=0.9
        )
        await chain.run_after(step, reflection, memory)

        assert len(tracer.spans) == 1
        assert telemetry.telemetry.steps_attempted == 1

    @pytest.mark.asyncio
    async def test_chain_short_circuits(self):
        """Budget interceptor should short-circuit the chain."""
        budget = TokenBudget(max_input_tokens=0)
        budget_int = TokenBudgetInterceptor(budget=budget)
        telemetry = TelemetryInterceptor()
        chain = MiddlewareChain([budget_int, telemetry])

        step = PlanStep(description="Test")
        memory = WorkingMemory()

        with pytest.raises(BudgetExceededError):
            await chain.run_before(step, memory)

    @pytest.mark.asyncio
    async def test_default_middleware_builds(self):
        chain, tracing, budget, telemetry = build_default_middleware()
        assert tracing is not None
        assert budget is not None
        assert telemetry is not None

    @pytest.mark.asyncio
    async def test_fluent_add(self):
        chain = MiddlewareChain()
        result = chain.add(TracingInterceptor()).add(TelemetryInterceptor())
        assert result is chain  # Returns self
