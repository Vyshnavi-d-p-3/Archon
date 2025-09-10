"""
Orchestrator: The main agent loop.

Coordinates Planner → Executor → Reflector in a loop:
1. Planner decomposes the task into steps
2. For each step:
   a. Executor runs the step (LLM → tool call → tool execution)
   b. Reflector analyzes the result
   c. Based on verdict: continue / retry / replan / abort / skip
3. After all steps complete, synthesize a final answer

Produces a full AgentTrace for evaluation and debugging.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from agent.executor import Executor
from agent.planner import Planner
from agent.reflector import Reflector
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
from tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)


SYNTHESIS_PROMPT = """You are synthesizing the final answer for a completed multi-step task.

Task: {task}

Results from each step:
{step_results}

Based on these results, provide a clear, comprehensive answer to the original task.
Be specific and reference the actual data gathered during execution."""


class AgentOrchestrator:
    """
    Main agent that runs the full plan→execute→reflect loop.
    
    Architecture:
    ┌──────────────┐
    │   User Task  │
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │   Planner    │ ◄─── Re-plan on REPLAN verdict
    └──────┬───────┘
           │ Plan (list of steps)
    ┌──────▼───────┐
    │  Executor    │ ◄─── Retry with correction on RETRY
    │  (per step)  │
    └──────┬───────┘
           │ Step result
    ┌──────▼───────┐
    │  Reflector   │──── CONTINUE → next step
    │              │──── RETRY → re-execute with hint
    │              │──── REPLAN → back to Planner
    │              │──── ABORT → stop
    │              │──── SKIP → next step
    └──────────────┘
    """

    def __init__(
        self,
        llm: Any,
        registry: ToolRegistry,
        config: Optional[AgentConfig] = None,
    ):
        self._config = config or AgentConfig()
        self._llm = llm
        self._registry = registry
        self._planner = Planner(llm, registry, self._config)
        self._executor = Executor(llm, registry, self._config)
        self._reflector = Reflector(llm, registry, self._config)

    def run(self, task: str) -> AgentTrace:
        """
        Execute a task end-to-end and return the full trace.
        """
        trace = AgentTrace(
            task_description=task,
            model_name=getattr(self._llm, "model_name", str(type(self._llm))),
        )
        memory = WorkingMemory()
        start_time = time.perf_counter()

        logger.info("agent_started", task=task[:100])

        try:
            # Phase 1: Create initial plan
            plan = self._planner.create_plan(task, memory)
            trace.plans.append(plan)

            # Phase 2: Execute steps with reflection loop
            self._execute_plan(plan, trace, memory, task)

            # Phase 3: Synthesize final answer
            trace.final_answer = self._synthesize_answer(task, trace, memory)
            trace.success = self._assess_success(trace)

        except Exception as exc:
            logger.error("agent_fatal_error", error=str(exc))
            trace.final_answer = f"Agent encountered a fatal error: {exc}"
            trace.success = False

        # Finalize trace
        trace.wall_time_seconds = round(time.perf_counter() - start_time, 3)
        trace.completed_at = datetime.now(timezone.utc)
        trace.total_steps_executed = len(trace.all_steps())
        trace.total_retries = sum(s.retries for s in trace.all_steps())
        trace.total_replans = sum(1 for p in trace.plans if p.is_replanned)

        # Collect failure categories
        for step in trace.all_steps():
            if step.reflection and step.reflection.failure_category:
                trace.failure_categories.append(step.reflection.failure_category)

        logger.info(
            "agent_completed",
            success=trace.success,
            steps=trace.total_steps_executed,
            retries=trace.total_retries,
            replans=trace.total_replans,
            wall_time=trace.wall_time_seconds,
        )

        # Persist trace if configured
        if self._config.trace_dir:
            self._save_trace(trace)

        return trace

    def _execute_plan(
        self,
        plan: Plan,
        trace: AgentTrace,
        memory: WorkingMemory,
        original_task: str,
    ) -> None:
        """Execute all steps in a plan with the reflect/retry/replan loop."""
        consecutive_failures = 0

        for step_idx, step in enumerate(plan.steps):
            logger.info(
                "executing_step",
                step_num=step_idx + 1,
                total=len(plan.steps),
                description=step.description[:60],
            )

            # Check dependencies
            if not self._dependencies_met(step, plan):
                step.status = StepStatus.SKIPPED
                step.reflection = None
                logger.info("step_skipped_deps", step_id=step.step_id)
                continue

            # Execute → Reflect → (Retry?) loop
            correction_hint = ""
            while True:
                # Execute step
                self._executor.execute_step(step, memory, correction_hint)

                # Reflect on result
                reflection = self._reflector.reflect(step, memory)
                step.reflection = reflection

                # Act on verdict
                if reflection.verdict == ReflectionVerdict.CONTINUE:
                    consecutive_failures = 0
                    break

                elif reflection.verdict == ReflectionVerdict.SKIP:
                    step.status = StepStatus.SKIPPED
                    consecutive_failures = 0
                    break

                elif reflection.verdict == ReflectionVerdict.RETRY:
                    step.retries += 1
                    step.status = StepStatus.RETRYING
                    correction_hint = reflection.suggested_correction or ""
                    consecutive_failures += 1

                    if step.retries >= self._config.executor.max_retries_per_step:
                        logger.warning(
                            "step_max_retries",
                            step_id=step.step_id,
                            retries=step.retries,
                        )
                        step.status = StepStatus.FAILED
                        break

                    logger.info(
                        "step_retrying",
                        step_id=step.step_id,
                        attempt=step.retries + 1,
                        correction=correction_hint[:100],
                    )
                    continue

                elif reflection.verdict == ReflectionVerdict.REPLAN:
                    if not self._config.planner.allow_replanning:
                        step.status = StepStatus.FAILED
                        break

                    logger.info("replanning_triggered", step_id=step.step_id)
                    completed = [
                        s for s in plan.steps[:step_idx]
                        if s.status == StepStatus.COMPLETED
                    ]
                    new_plan = self._planner.replan(
                        task=original_task,
                        completed_steps=completed,
                        failure_info=reflection.reasoning,
                        memory=memory,
                    )
                    new_plan.parent_plan_id = plan.plan_id
                    trace.plans.append(new_plan)
                    # Recurse into the new plan
                    self._execute_plan(new_plan, trace, memory, original_task)
                    return

                elif reflection.verdict == ReflectionVerdict.ABORT:
                    step.status = StepStatus.FAILED
                    logger.error(
                        "step_aborted",
                        step_id=step.step_id,
                        reason=reflection.reasoning,
                    )
                    return

            # Circuit breaker: too many consecutive failures
            if consecutive_failures >= self._config.reflector.max_consecutive_failures:
                logger.error(
                    "circuit_breaker_triggered",
                    consecutive_failures=consecutive_failures,
                )
                return

    def _dependencies_met(self, step: PlanStep, plan: Plan) -> bool:
        """Check that all dependency steps completed successfully."""
        if not step.depends_on:
            return True
        for dep_id in step.depends_on:
            dep_step = next(
                (s for s in plan.steps if s.step_id == dep_id), None
            )
            if dep_step is None or dep_step.status != StepStatus.COMPLETED:
                return False
        return True

    def _synthesize_answer(
        self,
        task: str,
        trace: AgentTrace,
        memory: WorkingMemory,
    ) -> str:
        """Use the LLM to produce a final answer from all step results."""
        step_results = []
        for step in trace.all_steps():
            if step.status == StepStatus.COMPLETED and step.tool_result:
                output_str = str(step.tool_result.output)[:500]
                step_results.append(
                    f"Step: {step.description}\n"
                    f"Tool: {step.tool_call.tool_name if step.tool_call else 'N/A'}\n"
                    f"Result: {output_str}"
                )

        if not step_results:
            return "No steps completed successfully. Unable to answer the task."

        prompt = SYNTHESIS_PROMPT.format(
            task=task,
            step_results="\n\n".join(step_results),
        )

        try:
            messages = [
                SystemMessage(content="You are a helpful assistant synthesizing results."),
                HumanMessage(content=prompt),
            ]
            response = self._llm.invoke(messages)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.error("synthesis_failed", error=str(exc))
            return f"Synthesis failed: {exc}. Raw results:\n" + "\n".join(step_results)

    def _assess_success(self, trace: AgentTrace) -> bool:
        """Determine if the overall task succeeded."""
        completed = trace.completed_steps()
        total = trace.all_steps()
        if not total:
            return False
        success_rate = len(completed) / len(total)
        return success_rate >= 0.5  # At least half the steps succeeded

    def _save_trace(self, trace: AgentTrace) -> None:
        """Persist trace to disk for evaluation."""
        trace_dir = Path(self._config.trace_dir)  # type: ignore
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = trace_dir / f"trace_{trace.trace_id[:8]}.json"
        with open(path, "w") as f:
            f.write(trace.model_dump_json(indent=2))
        logger.info("trace_saved", path=str(path))
