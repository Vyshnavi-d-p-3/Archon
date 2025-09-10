"""
Async Planner — LLM-backed task decomposition.

Uses the LLMBackend protocol (not LangChain directly) so any
backend can drive planning. Produces structured JSON plans with
dependency tracking and replan support.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Sequence

import structlog
from pydantic import BaseModel, Field

from agent.errors import EmptyPlanError, LLMOutputParseError, PlanParseError
from agent.state import Plan, PlanStep, WorkingMemory
from config.settings import AgentConfig
from tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)


class PlannedStep(BaseModel):
    description: str = Field(description="What this step accomplishes")
    tool: str = Field(description="Name of the tool to use")
    args: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[int] = Field(default_factory=list)


class PlanOutput(BaseModel):
    reasoning: str = Field(description="Chain-of-thought analysis")
    steps: list[PlannedStep] = Field(description="Ordered steps")


PLANNER_SYSTEM = """You are a task planner for an Archon. Decompose the task
into a sequence of concrete steps, each using exactly one tool.

{tool_schemas}

RULES:
1. Each step uses exactly one available tool.
2. Steps execute sequentially. Use depends_on for data dependencies.
3. Be specific — use concrete values, not placeholders.
4. Minimize steps: fewest that reliably solve the task.
5. Search first if you need information, then process/analyze.

{context}

Respond with ONLY valid JSON:
{{"reasoning": "...", "steps": [{{"description": "...", "tool": "...", "args": {{}}, "depends_on": []}}]}}"""


REPLAN_SYSTEM = """You are re-planning because the original plan failed.

Original task: {task}
Completed steps:
{completed}
Failure:
{failure}

{tool_schemas}

Create a NEW plan picking up from completed work. Respond with ONLY valid JSON."""


class AsyncPlanner:
    """Async planner using LLMBackend protocol."""

    def __init__(self, llm_backend: Any, registry: ToolRegistry, config: AgentConfig):
        self._llm = llm_backend
        self._registry = registry
        self._config = config

    async def create_plan(self, task: str, memory: WorkingMemory) -> Plan:
        logger.info("planner.create", task=task[:100])
        context = ""
        if memory.step_outputs:
            context = f"Context from prior steps:\n{memory.get_context_summary()}"

        prompt = PLANNER_SYSTEM.format(
            tool_schemas=self._registry.get_schemas_as_prompt(),
            context=context,
        )

        response = await self._llm.generate(system_prompt=prompt, user_message=task)
        plan_output = self._parse(response.content)

        if not plan_output.steps:
            raise EmptyPlanError(task)

        steps = []
        for i, ps in enumerate(plan_output.steps):
            deps = [steps[d].step_id for d in ps.depends_on if d < len(steps)]
            steps.append(PlanStep(
                description=ps.description,
                expected_tool=ps.tool,
                depends_on=deps,
            ))

        plan = Plan(
            task_description=task,
            steps=steps[:self._config.planner.max_plan_steps],
        )
        logger.info("planner.done", plan_id=plan.plan_id, num_steps=len(plan.steps))
        return plan

    async def replan(
        self,
        task: str,
        completed_steps: Sequence[PlanStep],
        failure_info: str,
        memory: WorkingMemory,
    ) -> Plan:
        logger.info("planner.replan", task=task[:100])
        completed_summary = "\n".join(
            f"  {i+1}. {s.description} → {'OK' if s.tool_result and s.tool_result.success else 'FAIL'}"
            for i, s in enumerate(completed_steps)
        ) or "(none)"

        prompt = REPLAN_SYSTEM.format(
            task=task,
            completed=completed_summary,
            failure=failure_info,
            tool_schemas=self._registry.get_schemas_as_prompt(),
        )
        response = await self._llm.generate(system_prompt=prompt, user_message=f"Replan: {task}")
        plan_output = self._parse(response.content)

        steps = []
        for i, ps in enumerate(plan_output.steps):
            deps = [steps[d].step_id for d in ps.depends_on if d < len(steps)]
            steps.append(PlanStep(
                description=ps.description,
                expected_tool=ps.tool,
                depends_on=deps,
            ))

        return Plan(
            task_description=task,
            steps=steps[:self._config.planner.max_plan_steps],
            is_replanned=True,
        )

    def _parse(self, raw: str) -> PlanOutput:
        cleaned = re.sub(r"```json\s*", "", raw)
        cleaned = re.sub(r"```\s*", "", cleaned)
        cleaned = cleaned.strip()
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            cleaned = match.group(0)
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

        try:
            return PlanOutput.model_validate(json.loads(cleaned))
        except Exception as exc:
            logger.error("planner.parse_failed", error=str(exc), raw=raw[:300])
            raise PlanParseError(raw_output=raw, parse_error=str(exc)) from exc
