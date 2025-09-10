"""
Step-level middleware chain.

Interceptors wrap every step execution, providing cross-cutting concerns
without polluting core agent logic. Inspired by gRPC interceptors and
Java servlet filters.

Chain ordering matters:
  1. TracingInterceptor (outermost — captures total wall time)
  2. TokenBudgetInterceptor (short-circuits before LLM call if budget blown)
  3. RateLimitInterceptor (enforces per-minute call limits)
  4. TelemetryInterceptor (innermost — records fine-grained metrics)

Adding a new concern (e.g., PII redaction) means writing one class,
not modifying the orchestrator.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

from agent.errors import BudgetExceededError
from agent.state import (
    FailureCategory,
    PlanStep,
    ReflectionVerdict,
    StepReflection,
    StepStatus,
    WorkingMemory,
)

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1. Tracing Interceptor — Correlation IDs + structured span logging
# ═══════════════════════════════════════════════════════════════════════

class TracingInterceptor:
    """
    Assigns correlation IDs and logs structured spans for every step.
    Produces OpenTelemetry-compatible span data without requiring
    the OTel SDK (keeps deps light; swap in real OTel in production).
    """

    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self._spans: list[dict[str, Any]] = []

    async def before_step(
        self,
        step: PlanStep,
        memory: WorkingMemory,
    ) -> PlanStep | None:
        span_id = str(uuid.uuid4())[:16]
        step._span_id = span_id  # type: ignore[attr-defined]
        step._span_start = time.perf_counter()  # type: ignore[attr-defined]
        logger.info(
            "step.start",
            trace_id=self.trace_id,
            span_id=span_id,
            step_id=step.step_id,
            description=step.description[:80],
        )
        return step

    async def after_step(
        self,
        step: PlanStep,
        reflection: StepReflection,
        memory: WorkingMemory,
    ) -> StepReflection:
        span_start = getattr(step, "_span_start", time.perf_counter())
        duration_ms = (time.perf_counter() - span_start) * 1000
        span_id = getattr(step, "_span_id", "unknown")

        span = {
            "trace_id": self.trace_id,
            "span_id": span_id,
            "step_id": step.step_id,
            "tool": step.tool_call.tool_name if step.tool_call else None,
            "status": step.status.value,
            "verdict": reflection.verdict.value,
            "duration_ms": round(duration_ms, 2),
            "retries": step.retries,
            "failure_category": (
                reflection.failure_category.value
                if reflection.failure_category else None
            ),
        }
        self._spans.append(span)

        logger.info(
            "step.end",
            **span,
        )
        return reflection

    @property
    def spans(self) -> list[dict[str, Any]]:
        return list(self._spans)


# ═══════════════════════════════════════════════════════════════════════
# 2. Token Budget Interceptor — Cost control and resource tracking
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TokenBudget:
    """Tracks token consumption against a budget."""
    max_input_tokens: int = 500_000
    max_output_tokens: int = 100_000
    max_total_cost_usd: float = 5.00

    input_tokens_consumed: int = 0
    output_tokens_consumed: int = 0
    total_cost_usd: float = 0.0
    llm_calls: int = 0

    # Cost per 1M tokens (configurable per model)
    input_cost_per_million: float = 0.15  # gpt-4o-mini default
    output_cost_per_million: float = 0.60

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens_consumed += input_tokens
        self.output_tokens_consumed += output_tokens
        self.total_cost_usd += (
            input_tokens * self.input_cost_per_million / 1_000_000
            + output_tokens * self.output_cost_per_million / 1_000_000
        )
        self.llm_calls += 1

    @property
    def input_remaining(self) -> int:
        return max(0, self.max_input_tokens - self.input_tokens_consumed)

    @property
    def output_remaining(self) -> int:
        return max(0, self.max_output_tokens - self.output_tokens_consumed)

    @property
    def cost_remaining(self) -> float:
        return max(0.0, self.max_total_cost_usd - self.total_cost_usd)

    def is_exhausted(self) -> bool:
        return (
            self.input_remaining <= 0
            or self.output_remaining <= 0
            or self.cost_remaining <= 0.0
        )

    def summary(self) -> dict[str, Any]:
        return {
            "input_tokens": f"{self.input_tokens_consumed}/{self.max_input_tokens}",
            "output_tokens": f"{self.output_tokens_consumed}/{self.max_output_tokens}",
            "cost_usd": f"${self.total_cost_usd:.4f}/${self.max_total_cost_usd:.2f}",
            "llm_calls": self.llm_calls,
        }


class TokenBudgetInterceptor:
    """
    Enforces token/cost budget. Short-circuits step execution
    if the budget is exhausted, preventing runaway costs.
    """

    def __init__(self, budget: TokenBudget | None = None):
        self.budget = budget or TokenBudget()

    async def before_step(
        self,
        step: PlanStep,
        memory: WorkingMemory,
    ) -> PlanStep | None:
        if self.budget.is_exhausted():
            logger.warning(
                "budget_exhausted",
                **self.budget.summary(),
            )
            raise BudgetExceededError(
                budget_type="token/cost",
                limit=self.budget.max_total_cost_usd,
                consumed=self.budget.total_cost_usd,
            )
        return step

    async def after_step(
        self,
        step: PlanStep,
        reflection: StepReflection,
        memory: WorkingMemory,
    ) -> StepReflection:
        # Token recording happens in the LLM adapter; here we just
        # check post-step and warn if approaching limits.
        if self.budget.cost_remaining < self.budget.max_total_cost_usd * 0.1:
            logger.warning(
                "budget_low",
                remaining=self.budget.cost_remaining,
                **self.budget.summary(),
            )
        return reflection


# ═══════════════════════════════════════════════════════════════════════
# 3. Rate Limit Interceptor — Sliding window rate limiter
# ═══════════════════════════════════════════════════════════════════════

class RateLimitInterceptor:
    """
    Sliding-window rate limiter for LLM/tool calls.
    Prevents burst overload on external APIs.
    """

    def __init__(
        self,
        max_calls_per_minute: int = 60,
        max_calls_per_step: int = 5,
    ):
        self._max_per_minute = max_calls_per_minute
        self._max_per_step = max_calls_per_step
        self._call_timestamps: list[float] = []
        self._step_call_count: int = 0

    async def before_step(
        self,
        step: PlanStep,
        memory: WorkingMemory,
    ) -> PlanStep | None:
        self._step_call_count = 0
        now = time.time()

        # Prune timestamps outside the window
        self._call_timestamps = [
            t for t in self._call_timestamps if now - t < 60
        ]

        if len(self._call_timestamps) >= self._max_per_minute:
            wait_time = 60 - (now - self._call_timestamps[0])
            logger.info("rate_limit_wait", wait_seconds=round(wait_time, 1))
            await asyncio.sleep(max(0, wait_time))

        self._call_timestamps.append(now)
        return step

    async def after_step(
        self,
        step: PlanStep,
        reflection: StepReflection,
        memory: WorkingMemory,
    ) -> StepReflection:
        return reflection


# ═══════════════════════════════════════════════════════════════════════
# 4. Telemetry Interceptor — Fine-grained metrics collection
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AgentTelemetry:
    """Accumulated telemetry for the entire agent run."""
    steps_attempted: int = 0
    steps_succeeded: int = 0
    steps_failed: int = 0
    steps_skipped: int = 0
    total_retries: int = 0
    total_replans: int = 0
    tool_latencies_ms: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    failure_counts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    verdict_counts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def summary(self) -> dict[str, Any]:
        avg_latencies = {
            tool: round(sum(lats) / len(lats), 2) if lats else 0
            for tool, lats in self.tool_latencies_ms.items()
        }
        return {
            "steps": {
                "attempted": self.steps_attempted,
                "succeeded": self.steps_succeeded,
                "failed": self.steps_failed,
                "skipped": self.steps_skipped,
            },
            "retries": self.total_retries,
            "replans": self.total_replans,
            "avg_tool_latency_ms": avg_latencies,
            "failure_distribution": dict(self.failure_counts),
            "verdict_distribution": dict(self.verdict_counts),
        }


class TelemetryInterceptor:
    """Collects fine-grained execution metrics."""

    def __init__(self):
        self.telemetry = AgentTelemetry()

    async def before_step(
        self,
        step: PlanStep,
        memory: WorkingMemory,
    ) -> PlanStep | None:
        self.telemetry.steps_attempted += 1
        return step

    async def after_step(
        self,
        step: PlanStep,
        reflection: StepReflection,
        memory: WorkingMemory,
    ) -> StepReflection:
        # Record outcome
        if step.status == StepStatus.COMPLETED:
            self.telemetry.steps_succeeded += 1
        elif step.status == StepStatus.FAILED:
            self.telemetry.steps_failed += 1
        elif step.status == StepStatus.SKIPPED:
            self.telemetry.steps_skipped += 1

        # Record tool latency
        if step.tool_call and step.tool_result:
            self.telemetry.tool_latencies_ms[step.tool_call.tool_name].append(
                step.tool_result.latency_ms
            )

        # Record failure category
        if reflection.failure_category:
            self.telemetry.failure_counts[reflection.failure_category.value] += 1

        # Record verdict
        self.telemetry.verdict_counts[reflection.verdict.value] += 1

        # Record retries
        self.telemetry.total_retries += step.retries

        return reflection


# ═══════════════════════════════════════════════════════════════════════
# Middleware Chain — Composable execution pipeline
# ═══════════════════════════════════════════════════════════════════════

class MiddlewareChain:
    """
    Composes interceptors into an ordered chain.

    Execution order:
      before_step: interceptor[0] → interceptor[1] → ... → interceptor[N]
      after_step:  interceptor[N] → ... → interceptor[1] → interceptor[0]
    (Onion model — outermost interceptor sees full lifecycle.)
    """

    def __init__(self, interceptors: list[Any] | None = None):
        self._interceptors = interceptors or []

    def add(self, interceptor: Any) -> "MiddlewareChain":
        """Add an interceptor. Returns self for fluent chaining."""
        self._interceptors.append(interceptor)
        return self

    async def run_before(
        self,
        step: PlanStep,
        memory: WorkingMemory,
    ) -> PlanStep | None:
        """Run before_step on all interceptors in order."""
        current = step
        for interceptor in self._interceptors:
            result = await interceptor.before_step(current, memory)
            if result is None:
                return None  # Short-circuit
            current = result
        return current

    async def run_after(
        self,
        step: PlanStep,
        reflection: StepReflection,
        memory: WorkingMemory,
    ) -> StepReflection:
        """Run after_step on all interceptors in reverse order."""
        current = reflection
        for interceptor in reversed(self._interceptors):
            current = await interceptor.after_step(step, current, memory)
        return current


def build_default_middleware(
    token_budget: TokenBudget | None = None,
    trace_id: str | None = None,
) -> tuple[MiddlewareChain, TracingInterceptor, TokenBudgetInterceptor, TelemetryInterceptor]:
    """
    Build the standard middleware stack.
    Returns the chain and individual interceptors for result access.
    """
    tracing = TracingInterceptor(trace_id=trace_id)
    budget = TokenBudgetInterceptor(budget=token_budget)
    rate_limit = RateLimitInterceptor()
    telemetry = TelemetryInterceptor()

    chain = MiddlewareChain([tracing, budget, rate_limit, telemetry])
    return chain, tracing, budget, telemetry
