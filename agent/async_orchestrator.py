"""
Async Orchestrator — the main agent control loop.

This is the central coordination point. It owns no business logic itself;
it delegates to Planner, Executor, and Reflector through protocol
interfaces, with cross-cutting concerns handled by the middleware chain.

Architecture decisions for reviewers:
  1. All I/O is async — no sync wrappers in the hot path.
  2. Middleware chain handles tracing, budgets, rate limits, telemetry.
  3. Error handling uses typed exceptions, not string matching.
  4. The orchestrator is stateless between runs — all state flows
     through WorkingMemory and AgentTrace.
  5. Each component is injected via constructor (DI), not created internally.

                 ┌──────────────┐
                 │  User Task   │
                 └──────┬───────┘
                        │
                 ┌──────▼───────┐
                 │   Planner    │ ◄── Protocol: agent.protocols.Planner
                 └──────┬───────┘
                        │
          ┌─────────────▼─────────────┐
          │    Middleware Chain        │
          │  ┌──────────────────────┐ │
          │  │ TracingInterceptor   │ │
          │  │ TokenBudgetIntercept │ │  ← before_step / after_step
          │  │ RateLimitInterceptor │ │
          │  │ TelemetryInterceptor │ │
          │  └──────────────────────┘ │
          └─────────────┬─────────────┘
                        │
                 ┌──────▼───────┐
                 │   Executor   │ ◄── Protocol: agent.protocols.Executor
                 │  (per step)  │
                 └──────┬───────┘
                        │
                 ┌──────▼───────┐
                 │  Reflector   │ ◄── Protocol: agent.protocols.Reflector
                 │              │
                 │  CONTINUE ───┼──→ next step
                 │  RETRY ──────┼──→ re-execute with correction
                 │  REPLAN ─────┼──→ new plan from Planner
                 │  ABORT ──────┼──→ halt
                 │  SKIP ───────┼──→ next step
                 └──────────────┘
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import structlog

from agent.errors import (
    BudgetExceededError,
    CircuitBreakerOpenError,
    FatalAgentError,
    MaxRetriesExceededError,
    RetryableError,
)
from agent.middleware import (
    AgentTelemetry,
    MiddlewareChain,
    TelemetryInterceptor,
    TokenBudget,
    TokenBudgetInterceptor,
    TracingInterceptor,
    build_default_middleware,
)
from agent.state import (
    AgentTrace,
    FailureCategory,
    Plan,
    PlanStep,
    ReflectionVerdict,
    StepStatus,
    WorkingMemory,
)
from config.settings import AgentConfig

logger = structlog.get_logger(__name__)


SYNTHESIS_PROMPT = """You are synthesizing the final answer for a completed multi-step task.

Task: {task}

Results from each step:
{step_results}

Based on these results, provide a clear, comprehensive answer to the original task.
Be specific and reference the actual data gathered. Do not fabricate information
not present in the step results."""


class AsyncOrchestrator:
    """
    Main agent loop — async, middleware-aware, protocol-driven.

    All dependencies are injected. The orchestrator has no knowledge of
    specific LLM providers, tool implementations, or persistence backends.
    """

    def __init__(
        self,
        llm_backend: Any,          # Satisfies LLMBackend protocol
        planner: Any,              # Satisfies Planner protocol
        executor: Any,             # Satisfies Executor protocol
        reflector: Any,            # Satisfies Reflector protocol
        config: AgentConfig,
        middleware: MiddlewareChain | None = None,
        token_budget: TokenBudget | None = None,
    ):
        self._llm = llm_backend
        self._planner = planner
        self._executor = executor
        self._reflector = reflector
        self._config = config

        # Build default middleware if not injected
        if middleware is not None:
            self._middleware = middleware
            self._tracing = None
            self._budget_interceptor = None
            self._telemetry = None
        else:
            (
                self._middleware,
                self._tracing,
                self._budget_interceptor,
                self._telemetry,
            ) = build_default_middleware(token_budget=token_budget)

    async def run(self, task: str) -> AgentTrace:
        """
        Execute a task end-to-end.

        Returns a complete AgentTrace regardless of success or failure.
        The trace is the primary artifact — it feeds evaluation, debugging,
        and the failure-mode taxonomy.
        """
        trace = AgentTrace(
            task_description=task,
            model_name=getattr(self._llm, "model_id", str(type(self._llm).__name__)),
        )
        memory = WorkingMemory()
        start_time = time.perf_counter()

        logger.info("agent.run.start", task=task[:120])

        try:
            # Phase 1: Plan
            plan = await self._planner.create_plan(task, memory)
            trace.plans.append(plan)

            # Phase 2: Execute with reflect/retry/replan loop
            await self._execute_plan(plan, trace, memory, task)

            # Phase 3: Synthesize
            trace.final_answer = await self._synthesize(task, trace, memory)
            trace.success = self._assess_success(trace)

        except BudgetExceededError as exc:
            logger.warning("agent.budget_exceeded", **exc.context())
            trace.final_answer = f"Stopped: {exc}"
            trace.success = False

        except FatalAgentError as exc:
            logger.error("agent.fatal", **exc.context())
            trace.final_answer = f"Fatal error: {exc}"
            trace.success = False

        except Exception as exc:
            logger.error("agent.unexpected_error", error=str(exc), error_type=type(exc).__name__)
            trace.final_answer = f"Unexpected error: {type(exc).__name__}: {exc}"
            trace.success = False

        # Finalize
        trace.wall_time_seconds = round(time.perf_counter() - start_time, 3)
        trace.completed_at = datetime.now(timezone.utc)
        trace.total_steps_executed = len(trace.all_steps())
        trace.total_retries = sum(s.retries for s in trace.all_steps())
        trace.total_replans = sum(1 for p in trace.plans if p.is_replanned)
        trace.failure_categories = [
            s.reflection.failure_category
            for s in trace.all_steps()
            if s.reflection and s.reflection.failure_category
        ]

        # Attach telemetry
        if self._telemetry:
            trace._telemetry = self._telemetry.telemetry.summary()  # type: ignore[attr-defined]
        if self._tracing:
            trace._spans = self._tracing.spans  # type: ignore[attr-defined]

        logger.info(
            "agent.run.end",
            success=trace.success,
            steps=trace.total_steps_executed,
            retries=trace.total_retries,
            replans=trace.total_replans,
            wall_time=trace.wall_time_seconds,
        )

        from config.version import TRACE_SCHEMA_VERSION, package_version

        trace.archon_version = package_version()
        trace.trace_schema_version = TRACE_SCHEMA_VERSION

        # Persist
        if self._config.trace_dir:
            self._save_trace(trace)

        return trace

    async def _execute_plan(
        self,
        plan: Plan,
        trace: AgentTrace,
        memory: WorkingMemory,
        original_task: str,
        depth: int = 0,
    ) -> None:
        """Execute all steps in a plan with middleware + reflect loop."""
        if depth > 3:
            logger.error("agent.max_replan_depth")
            return

        consecutive_failures = 0

        for step_idx, step in enumerate(plan.steps):
            # ── Dependency check ─────────────────────────────────
            if not self._deps_met(step, plan):
                step.status = StepStatus.SKIPPED
                logger.info("step.skipped.deps", step_id=step.step_id)
                continue

            # ── Middleware: before_step ───────────────────────────
            processed = await self._middleware.run_before(step, memory)
            if processed is None:
                step.status = StepStatus.SKIPPED
                logger.info("step.skipped.middleware", step_id=step.step_id)
                continue

            # ── Execute → Reflect → Retry loop ──────────────────
            correction_hint = ""
            while True:
                # Execute
                await self._executor.execute_step(step, memory, correction_hint)

                # Reflect
                reflection = await self._reflector.reflect(step, memory)
                step.reflection = reflection

                # Middleware: after_step (can modify verdict)
                reflection = await self._middleware.run_after(step, reflection, memory)
                step.reflection = reflection

                # ── Act on verdict ───────────────────────────────
                match reflection.verdict:
                    case ReflectionVerdict.CONTINUE:
                        consecutive_failures = 0
                        break

                    case ReflectionVerdict.SKIP:
                        step.status = StepStatus.SKIPPED
                        consecutive_failures = 0
                        break

                    case ReflectionVerdict.RETRY:
                        step.retries += 1
                        step.status = StepStatus.RETRYING
                        correction_hint = reflection.suggested_correction or ""
                        consecutive_failures += 1

                        if step.retries >= self._config.executor.max_retries_per_step:
                            logger.warning(
                                "step.max_retries",
                                step_id=step.step_id,
                                retries=step.retries,
                            )
                            step.status = StepStatus.FAILED
                            break

                        logger.info(
                            "step.retry",
                            step_id=step.step_id,
                            attempt=step.retries + 1,
                        )

                    case ReflectionVerdict.REPLAN:
                        if not self._config.planner.allow_replanning:
                            step.status = StepStatus.FAILED
                            break

                        logger.info("agent.replan", step_id=step.step_id)
                        completed = [
                            s for s in plan.steps[:step_idx]
                            if s.status == StepStatus.COMPLETED
                        ]
                        new_plan = await self._planner.replan(
                            task=original_task,
                            completed_steps=completed,
                            failure_info=reflection.reasoning,
                            memory=memory,
                        )
                        new_plan.parent_plan_id = plan.plan_id
                        trace.plans.append(new_plan)
                        await self._execute_plan(
                            new_plan, trace, memory, original_task, depth + 1
                        )
                        return

                    case ReflectionVerdict.ABORT:
                        step.status = StepStatus.FAILED
                        logger.error(
                            "step.abort",
                            step_id=step.step_id,
                            reason=reflection.reasoning,
                        )
                        return

            # ── Circuit breaker ──────────────────────────────────
            threshold = self._config.reflector.max_consecutive_failures
            if consecutive_failures >= threshold:
                raise CircuitBreakerOpenError(consecutive_failures, threshold)

    def _deps_met(self, step: PlanStep, plan: Plan) -> bool:
        if not step.depends_on:
            return True
        return all(
            any(
                s.step_id == dep_id and s.status == StepStatus.COMPLETED
                for s in plan.steps
            )
            for dep_id in step.depends_on
        )

    async def _synthesize(
        self,
        task: str,
        trace: AgentTrace,
        memory: WorkingMemory,
    ) -> str:
        """Synthesize final answer from completed step results."""
        results = []
        for step in trace.all_steps():
            if step.status == StepStatus.COMPLETED and step.tool_result:
                output = str(step.tool_result.output)[:500]
                results.append(
                    f"Step: {step.description}\n"
                    f"Tool: {step.tool_call.tool_name if step.tool_call else 'N/A'}\n"
                    f"Result: {output}"
                )

        if not results:
            return "No steps completed successfully. Unable to produce an answer."

        prompt = SYNTHESIS_PROMPT.format(
            task=task,
            step_results="\n\n".join(results),
        )

        try:
            response = await self._llm.generate(
                system_prompt="You are a helpful assistant synthesizing research results.",
                user_message=prompt,
            )
            return response.content
        except Exception as exc:
            logger.error("synthesis.failed", error=str(exc))
            return f"Synthesis failed: {exc}\n\nRaw results:\n" + "\n---\n".join(results)

    def _assess_success(self, trace: AgentTrace) -> bool:
        completed = len(trace.completed_steps())
        total = len(trace.all_steps())
        if total == 0:
            return False
        return completed / total >= 0.5

    def _save_trace(self, trace: AgentTrace) -> None:
        trace_dir = Path(self._config.trace_dir)  # type: ignore
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = trace_dir / f"trace_{trace.trace_id[:8]}.json"
        with open(path, "w") as f:
            f.write(trace.model_dump_json(indent=2))
        logger.info("trace.saved", path=str(path))


# ═══════════════════════════════════════════════════════════════════════
# Convenience builder — assembles all components from config
# ═══════════════════════════════════════════════════════════════════════

def build_agent(
    provider: str = "openai",
    model_name: str = "gpt-4o-mini",
    api_key: str | None = None,
    config: AgentConfig | None = None,
    use_mock_tools: bool = False,
    token_budget: TokenBudget | None = None,
) -> AsyncOrchestrator:
    """
    High-level factory: wire up all components from minimal config.

    Usage:
        agent = build_agent(provider="openai", model_name="gpt-4o-mini")
        trace = await agent.run("Find the GDP of Japan and calculate per capita")
    """
    from agent.llm_backends import DeterministicFakeBackend, LLMFactory
    from agent.async_executor import AsyncExecutor
    from agent.async_planner import AsyncPlanner
    from agent.async_reflector import AsyncReflector
    from tools.implementations import build_default_registry

    cfg = config or AgentConfig.from_env()
    budget = token_budget or TokenBudget()

    # For local smoke runs, --mock should be fully no-key: fake LLM + mock tools.
    if use_mock_tools:
        llm = DeterministicFakeBackend(
            responses={
                # Synthesis output
                "results from each step": (
                    "Based on gathered results, Tokyo has an estimated population of about 14 million."
                ),
                # Planner outputs
                "find the population": (
                    '{"reasoning":"Search then summarize.","steps":['
                    '{"description":"Search for population of Tokyo","tool":"web_search","args":{"query":"Tokyo population"},"depends_on":[]}'
                    "]}"),
                "ingest this text": (
                    '{"reasoning":"Ingest then retrieve answer.","steps":['
                    '{"description":"Ingest provided text into knowledge base","tool":"rag_ingest","args":{"text":"Python was created by Guido van Rossum and first released in 1991.","source":"inline_text"},"depends_on":[]},'
                    '{"description":"Query knowledge base for Python creator and release year","tool":"rag_search","args":{"query":"who created Python and when was it released?","top_k":3},"depends_on":[0]}'
                    "]}"),
                "replan for:": (
                    '{"reasoning":"Fallback replan.","steps":['
                    '{"description":"Search for population of Tokyo","tool":"web_search","args":{"query":"Tokyo population"},"depends_on":[]}'
                    "]}"),
                # Executor outputs
                "search for population of tokyo": (
                    '{"tool":"web_search","arguments":{"query":"Tokyo population"}}'
                ),
                "ingest provided text into knowledge base": (
                    '{"tool":"rag_ingest","arguments":{"text":"Python was created by Guido van Rossum and first released in 1991.","source":"inline_text"}}'
                ),
                "query knowledge base for python creator and release year": (
                    '{"tool":"rag_search","arguments":{"query":"who created Python and when was it released?","top_k":3}}'
                ),
            },
            default_response='{"tool":"web_search","arguments":{"query":"test"}}',
        )
    else:
        llm = LLMFactory.create(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            token_budget=budget,
        )

    registry = build_default_registry(use_mock=use_mock_tools)

    planner = AsyncPlanner(llm_backend=llm, registry=registry, config=cfg)
    executor = AsyncExecutor(llm_backend=llm, registry=registry, config=cfg)
    reflector = AsyncReflector(llm_backend=llm, registry=registry, config=cfg)

    middleware, tracing, budget_int, telemetry = build_default_middleware(
        token_budget=budget
    )

    agent = AsyncOrchestrator(
        llm_backend=llm,
        planner=planner,
        executor=executor,
        reflector=reflector,
        config=cfg,
        middleware=middleware,
        token_budget=budget,
    )
    # Expose interceptors for result access
    agent._tracing = tracing
    agent._budget_interceptor = budget_int
    agent._telemetry = telemetry

    return agent
