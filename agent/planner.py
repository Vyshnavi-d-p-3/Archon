"""
Planner: Decomposes a high-level task into an ordered sequence of
concrete steps, each mapped to an expected tool.

Uses structured output (JSON schema) to guarantee parseable plans.
Supports re-planning when the reflector signals REPLAN.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

import structlog
from pydantic import BaseModel, Field

from agent.state import Plan, PlanStep, WorkingMemory
from config.settings import AgentConfig
from tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)


# ── Structured output schema for the LLM ────────────────────────────

class PlannedStep(BaseModel):
    """Schema the LLM must produce for each step."""
    description: str = Field(description="What this step accomplishes")
    tool: str = Field(description="Name of the tool to use")
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the tool",
    )
    depends_on: list[int] = Field(
        default_factory=list,
        description="Indices of steps this depends on (0-indexed)",
    )


class PlanOutput(BaseModel):
    """Full structured plan output from the LLM."""
    reasoning: str = Field(description="Chain-of-thought about how to solve the task")
    steps: list[PlannedStep] = Field(description="Ordered list of steps")


# ── System prompt template ───────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = """You are a task planner for an Archon. Your job is to decompose
a user's task into a concrete sequence of steps, each using exactly one tool.

{tool_schemas}

RULES:
1. Each step must use exactly one tool from the available tools list.
2. Steps execute sequentially. Use depends_on to mark data dependencies.
3. Be specific in tool arguments — use concrete values, not placeholders.
4. Keep plans minimal: use the fewest steps that reliably solve the task.
5. If a task requires information gathering first, plan search steps before analysis steps.

{context_block}

Respond with ONLY valid JSON matching this schema:
{{
  "reasoning": "your chain-of-thought analysis of the task",
  "steps": [
    {{
      "description": "what this step does",
      "tool": "tool_name",
      "args": {{"arg1": "value1"}},
      "depends_on": []
    }}
  ]
}}"""


REPLAN_SYSTEM_PROMPT = """You are re-planning because the original plan encountered issues.

Original task: {task}

Steps completed so far:
{completed_steps}

Failure that triggered re-planning:
{failure_info}

{tool_schemas}

Create a NEW plan that picks up from where the previous plan left off.
Account for what was already accomplished and what went wrong.

Respond with ONLY valid JSON matching the plan schema:
{{
  "reasoning": "your analysis of what went wrong and the new approach",
  "steps": [...]
}}"""


class Planner:
    """
    LLM-backed planner that produces structured, validated plans.
    """

    def __init__(
        self,
        llm: Any,  # LangChain LLM instance
        registry: ToolRegistry,
        config: AgentConfig,
    ):
        self._llm = llm
        self._registry = registry
        self._config = config

    def create_plan(
        self,
        task: str,
        memory: Optional[WorkingMemory] = None,
    ) -> Plan:
        """Generate a plan for the given task."""
        logger.info("planning_started", task=task[:100])

        context_block = ""
        if memory and memory.step_outputs:
            context_block = (
                f"Context from prior steps:\n{memory.get_context_summary()}"
            )

        prompt = PLANNER_SYSTEM_PROMPT.format(
            tool_schemas=self._registry.get_schemas_as_prompt(),
            context_block=context_block,
        )

        raw_output = self._call_llm(prompt, task)
        plan_output = self._parse_plan_output(raw_output)

        steps = []
        for i, ps in enumerate(plan_output.steps):
            deps = [steps[d].step_id for d in ps.depends_on if d < len(steps)]
            steps.append(
                PlanStep(
                    description=ps.description,
                    expected_tool=ps.tool,
                    depends_on=deps,
                    tool_call=None,
                )
            )

        plan = Plan(
            task_description=task,
            steps=steps[:self._config.planner.max_plan_steps],
        )
        logger.info(
            "planning_completed",
            plan_id=plan.plan_id,
            num_steps=len(plan.steps),
        )
        return plan

    def replan(
        self,
        task: str,
        completed_steps: list[PlanStep],
        failure_info: str,
        memory: Optional[WorkingMemory] = None,
    ) -> Plan:
        """Generate a new plan after a failure, accounting for prior progress."""
        logger.info("replanning_started", task=task[:100])

        completed_summary = "\n".join(
            f"  Step {i+1}: {s.description} → "
            f"{'SUCCESS' if s.tool_result and s.tool_result.success else 'FAILED'}"
            for i, s in enumerate(completed_steps)
        )

        prompt = REPLAN_SYSTEM_PROMPT.format(
            task=task,
            completed_steps=completed_summary or "(none)",
            failure_info=failure_info,
            tool_schemas=self._registry.get_schemas_as_prompt(),
        )

        raw_output = self._call_llm(prompt, f"Replan for: {task}")
        plan_output = self._parse_plan_output(raw_output)

        steps = []
        for i, ps in enumerate(plan_output.steps):
            deps = [steps[d].step_id for d in ps.depends_on if d < len(steps)]
            steps.append(
                PlanStep(
                    description=ps.description,
                    expected_tool=ps.tool,
                    depends_on=deps,
                )
            )

        plan = Plan(
            task_description=task,
            steps=steps[:self._config.planner.max_plan_steps],
            is_replanned=True,
        )
        logger.info(
            "replanning_completed",
            plan_id=plan.plan_id,
            num_steps=len(plan.steps),
        )
        return plan

    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """Invoke the LLM with system + user messages."""
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        response = self._llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    def _parse_plan_output(self, raw: str) -> PlanOutput:
        """
        Parse LLM output into a PlanOutput, with fallback
        for common formatting issues (markdown fences, trailing commas).
        """
        # Strip markdown code fences
        cleaned = re.sub(r"```json\s*", "", raw)
        cleaned = re.sub(r"```\s*", "", cleaned)
        cleaned = cleaned.strip()

        # Try to find JSON object
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            cleaned = match.group(0)

        # Fix trailing commas (common LLM mistake)
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

        try:
            data = json.loads(cleaned)
            return PlanOutput.model_validate(data)
        except (json.JSONDecodeError, Exception) as exc:
            logger.error(
                "plan_parse_failed",
                error=str(exc),
                raw_output=raw[:500],
            )
            # Fallback: create a single-step plan
            return PlanOutput(
                reasoning=f"Failed to parse plan, creating fallback. Error: {exc}",
                steps=[
                    PlannedStep(
                        description=f"Attempt task directly",
                        tool="web_search",
                        args={"query": raw[:100]},
                    )
                ],
            )
