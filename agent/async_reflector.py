"""
Async Reflector — two-phase step analysis (heuristic + LLM).

Phase 1 (heuristic): Fast checks that never need an LLM call.
Phase 2 (LLM): Nuanced analysis for ambiguous cases.

The heuristic phase catches ~60% of cases in practice,
saving LLM calls and latency on the hot path.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

import structlog

from agent.errors import LLMOutputParseError
from agent.state import (
    FailureCategory,
    PlanStep,
    ReflectionVerdict,
    StepReflection,
    StepStatus,
    WorkingMemory,
)
from config.settings import AgentConfig
from tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)


REFLECTION_SYSTEM = """You are a reflection module. Analyze the step result and decide what to do next.

Failure categories: tool_selection_error, tool_arg_schema_violation, tool_execution_failure,
output_parse_error, hallucinated_tool, wrong_step_order, context_loss, infinite_loop, timeout

Verdicts: continue (success), retry (recoverable), replan (new plan needed), abort (fatal), skip (not needed)

Available tools: {tools}

Respond with ONLY valid JSON:
{{"verdict": "...", "reasoning": "...", "failure_category": "... or null", "suggested_correction": "... or null", "confidence": 0.0-1.0}}"""


REFLECTION_USER = """Step: {desc}
Tool: {tool} | Args: {args}
Success: {success} | Output: {output}
Error: {error} | Retries: {retries}
Context: {context}"""


class AsyncReflector:
    """Async reflector with heuristic fast-path + LLM fallback."""

    def __init__(self, llm_backend: Any, registry: ToolRegistry, config: AgentConfig):
        self._llm = llm_backend
        self._registry = registry
        self._config = config

    async def reflect(self, step: PlanStep, memory: WorkingMemory) -> StepReflection:
        if not self._config.reflector.enabled:
            if step.tool_result and step.tool_result.success:
                return StepReflection(
                    verdict=ReflectionVerdict.CONTINUE,
                    reasoning="Reflection disabled; step succeeded.",
                    confidence=1.0,
                )
            return StepReflection(
                verdict=ReflectionVerdict.ABORT,
                reasoning="Reflection disabled; step failed.",
                failure_category=FailureCategory.UNKNOWN,
                confidence=1.0,
            )

        # Phase 1: Heuristic
        h = self._heuristic(step)
        if h is not None:
            logger.info("reflector.heuristic", step_id=step.step_id, verdict=h.verdict)
            return h

        # Phase 2: LLM
        return await self._llm_reflect(step, memory)

    def _heuristic(self, step: PlanStep) -> StepReflection | None:
        """Fast checks — no LLM call needed."""

        # Hallucinated tool
        if step.tool_call and step.tool_call.tool_name not in self._registry:
            return StepReflection(
                verdict=ReflectionVerdict.RETRY,
                reasoning=f"Tool '{step.tool_call.tool_name}' does not exist.",
                failure_category=FailureCategory.HALLUCINATED_TOOL,
                suggested_correction=f"Use one of: {self._registry.list_tools()}",
                confidence=1.0,
            )

        # Schema violation
        if step.tool_call and not step.tool_call.schema_valid:
            can_retry = step.retries < self._config.executor.max_retries_per_step
            return StepReflection(
                verdict=ReflectionVerdict.RETRY if can_retry else ReflectionVerdict.REPLAN,
                reasoning=f"Schema errors: {step.tool_call.schema_errors}",
                failure_category=FailureCategory.TOOL_ARG_SCHEMA_VIOLATION,
                suggested_correction=f"Fix: {step.tool_call.schema_errors}",
                confidence=0.95,
            )

        # Clear success
        if step.tool_result and step.tool_result.success and step.tool_result.output:
            output_str = str(step.tool_result.output)
            if output_str and output_str not in ("None", "null", ""):
                return StepReflection(
                    verdict=ReflectionVerdict.CONTINUE,
                    reasoning="Step succeeded with non-empty output.",
                    confidence=0.9,
                )

        # Retries exhausted
        if step.retries >= self._config.executor.max_retries_per_step:
            replan_ok = self._config.planner.replan_on_failure
            return StepReflection(
                verdict=ReflectionVerdict.REPLAN if replan_ok else ReflectionVerdict.ABORT,
                reasoning=f"Exhausted {step.retries} retries.",
                failure_category=FailureCategory.TOOL_EXECUTION_FAILURE,
                confidence=0.85,
            )

        # Timeout
        if (
            step.tool_result
            and step.tool_result.latency_ms > self._config.executor.step_timeout_seconds * 1000
        ):
            return StepReflection(
                verdict=ReflectionVerdict.RETRY,
                reasoning="Step timed out.",
                failure_category=FailureCategory.TIMEOUT,
                suggested_correction="Simplify the request.",
                confidence=0.9,
            )

        return None

    async def _llm_reflect(self, step: PlanStep, memory: WorkingMemory) -> StepReflection:
        system = REFLECTION_SYSTEM.format(tools=self._registry.list_tools())
        tc = step.tool_call
        tr = step.tool_result

        user = REFLECTION_USER.format(
            desc=step.description,
            tool=tc.tool_name if tc else "none",
            args=json.dumps(tc.arguments, default=str)[:500] if tc else "{}",
            success=tr.success if tr else "N/A",
            output=_trunc(str(tr.output), 800) if tr else "N/A",
            error=tr.error if tr else "N/A",
            retries=step.retries,
            context=memory.get_context_summary(max_entries=5),
        )

        try:
            response = await self._llm.generate(system_prompt=system, user_message=user)
            return self._parse(response.content)
        except Exception as exc:
            logger.error("reflector.llm_failed", error=str(exc))
            if tr and tr.success:
                return StepReflection(
                    verdict=ReflectionVerdict.CONTINUE,
                    reasoning=f"Reflection LLM failed ({exc}), but tool succeeded.",
                    confidence=0.5,
                )
            return StepReflection(
                verdict=ReflectionVerdict.RETRY,
                reasoning=f"Reflection LLM failed: {exc}",
                failure_category=FailureCategory.UNKNOWN,
                confidence=0.3,
            )

    def _parse(self, raw: str) -> StepReflection:
        cleaned = re.sub(r"```json\s*", "", raw)
        cleaned = re.sub(r"```\s*", "", cleaned).strip()
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            cleaned = match.group(0)
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

        try:
            data = json.loads(cleaned)
            fc = None
            if data.get("failure_category") and data["failure_category"] != "null":
                try:
                    fc = FailureCategory(data["failure_category"])
                except ValueError:
                    fc = FailureCategory.UNKNOWN

            return StepReflection(
                verdict=ReflectionVerdict(data.get("verdict", "continue")),
                reasoning=data.get("reasoning", ""),
                failure_category=fc,
                suggested_correction=data.get("suggested_correction"),
                confidence=float(data.get("confidence", 0.5)),
            )
        except Exception as exc:
            return StepReflection(
                verdict=ReflectionVerdict.CONTINUE,
                reasoning=f"Parse failed: {exc}. Defaulting to continue.",
                confidence=0.3,
            )


def _trunc(s: str, n: int) -> str:
    return s[:n] + "..." if len(s) > n else s
