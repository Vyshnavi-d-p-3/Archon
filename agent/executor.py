"""
Executor: Takes a plan and runs each step sequentially.

For each step:
1. Constructs a prompt with step context + tool schemas
2. Calls the LLM to decide tool + arguments
3. Validates the tool call against the registry schema
4. Executes the tool
5. Hands off to the Reflector for verdict
6. Applies retry / replan / abort logic based on verdict
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import (
    FailureCategory,
    PlanStep,
    ReflectionVerdict,
    StepStatus,
    ToolCall,
    ToolResult,
    WorkingMemory,
)
from config.settings import AgentConfig
from tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)


EXECUTOR_SYSTEM_PROMPT = """You are the executor module of an autonomous agent. Your job is to execute
a single step by choosing the right tool and providing the correct arguments.

{tool_schemas}

Current step to execute:
  Description: {step_description}
  Expected tool: {expected_tool}

{context_block}

{correction_hint}

You MUST respond with ONLY valid JSON:
{{
  "tool": "tool_name",
  "arguments": {{"arg1": "value1", "arg2": "value2"}}
}}

Choose the tool and arguments that best accomplish this step.
Use concrete values, not placeholders."""


class Executor:
    """
    Executes plan steps one at a time with full tracing.
    Coordinates with the Reflector for step-level feedback.
    """

    def __init__(
        self,
        llm: Any,
        registry: ToolRegistry,
        config: AgentConfig,
    ):
        self._llm = llm
        self._registry = registry
        self._config = config

    def execute_step(
        self,
        step: PlanStep,
        memory: WorkingMemory,
        correction_hint: str = "",
    ) -> PlanStep:
        """
        Execute a single step:
        1. Ask LLM for tool call
        2. Validate against schema
        3. Execute the tool
        4. Return updated step with results
        """
        step.status = StepStatus.RUNNING
        logger.info(
            "step_executing",
            step_id=step.step_id,
            description=step.description[:80],
            attempt=step.retries + 1,
        )

        # Build context
        context_block = ""
        if memory.step_outputs:
            context_block = (
                f"Context from prior steps:\n{memory.get_context_summary()}"
            )

        hint_block = ""
        if correction_hint:
            hint_block = (
                f"IMPORTANT CORRECTION from previous attempt:\n{correction_hint}\n"
                "Apply this correction in your tool call."
            )

        # Phase 1: Ask LLM for tool + arguments
        prompt = EXECUTOR_SYSTEM_PROMPT.format(
            tool_schemas=self._registry.get_schemas_as_prompt(),
            step_description=step.description,
            expected_tool=step.expected_tool or "any appropriate tool",
            context_block=context_block,
            correction_hint=hint_block,
        )

        try:
            raw_output = self._call_llm(prompt, step.description)
            tool_call = self._parse_tool_call(raw_output)
        except Exception as exc:
            logger.error("step_llm_failed", error=str(exc))
            step.tool_call = ToolCall(
                tool_name="unknown",
                arguments={},
                raw_llm_output=str(exc),
                schema_valid=False,
                schema_errors=[f"LLM call failed: {exc}"],
            )
            step.tool_result = ToolResult(
                success=False,
                error=f"LLM invocation failed: {exc}",
            )
            step.status = StepStatus.FAILED
            return step

        # Phase 2: Validate schema
        is_valid, errors = self._registry.validate_tool_call(tool_call)
        tool_call.schema_valid = is_valid
        tool_call.schema_errors = errors
        step.tool_call = tool_call

        if not is_valid:
            logger.warning(
                "step_schema_invalid",
                step_id=step.step_id,
                tool=tool_call.tool_name,
                errors=errors,
            )
            step.tool_result = ToolResult(
                success=False,
                error=f"Schema validation failed: {errors}",
            )
            step.status = StepStatus.FAILED
            return step

        # Phase 3: Execute the tool
        result = self._registry.execute_tool_call(tool_call)
        step.tool_result = result

        if result.success:
            step.status = StepStatus.COMPLETED
            step.completed_at = datetime.now(timezone.utc)
            memory.record_step_output(step.step_id, result.output)
            logger.info(
                "step_completed",
                step_id=step.step_id,
                latency_ms=result.latency_ms,
            )
        else:
            step.status = StepStatus.FAILED
            logger.warning(
                "step_failed",
                step_id=step.step_id,
                error=result.error,
            )

        return step

    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """Invoke the LLM."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        response = self._llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    def _parse_tool_call(self, raw: str) -> ToolCall:
        """Parse LLM output into a ToolCall."""
        cleaned = re.sub(r"```json\s*", "", raw)
        cleaned = re.sub(r"```\s*", "", cleaned)
        cleaned = cleaned.strip()

        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            cleaned = match.group(0)

        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

        try:
            data = json.loads(cleaned)
            return ToolCall(
                tool_name=data.get("tool", "unknown"),
                arguments=data.get("arguments", data.get("args", {})),
                raw_llm_output=raw,
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("tool_call_parse_failed", error=str(exc), raw=raw[:300])
            return ToolCall(
                tool_name="unknown",
                arguments={},
                raw_llm_output=raw,
                schema_valid=False,
                schema_errors=[f"Failed to parse LLM output: {exc}"],
            )
