"""
Async Executor — runs individual steps through LLM → validate → execute.

Uses typed exceptions from agent.errors instead of string matching.
All tool interactions go through the ToolRegistry's schema validation.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

import structlog

from agent.errors import LLMOutputParseError, ToolNotFoundError, ToolSchemaError
from agent.state import PlanStep, StepStatus, ToolCall, ToolResult, WorkingMemory
from config.settings import AgentConfig
from tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)

EXECUTOR_SYSTEM = """You are the executor of an autonomous agent. Execute the given step
by choosing the right tool and providing correct arguments.

{tool_schemas}

Step: {step_description}
Expected tool: {expected_tool}

{context}
{correction}

Respond with ONLY valid JSON: {{"tool": "tool_name", "arguments": {{...}}}}"""


class AsyncExecutor:
    """Async executor using LLMBackend protocol."""

    def __init__(self, llm_backend: Any, registry: ToolRegistry, config: AgentConfig):
        self._llm = llm_backend
        self._registry = registry
        self._config = config

    async def execute_step(
        self,
        step: PlanStep,
        memory: WorkingMemory,
        correction_hint: str = "",
    ) -> PlanStep:
        step.status = StepStatus.RUNNING
        logger.info(
            "executor.step",
            step_id=step.step_id,
            description=step.description[:80],
            attempt=step.retries + 1,
        )

        context = ""
        if memory.step_outputs:
            context = f"Prior context:\n{memory.get_context_summary()}"

        correction = ""
        if correction_hint:
            correction = f"CORRECTION from prior attempt:\n{correction_hint}"

        prompt = EXECUTOR_SYSTEM.format(
            tool_schemas=self._registry.get_schemas_as_prompt(),
            step_description=step.description,
            expected_tool=step.expected_tool or "any",
            context=context,
            correction=correction,
        )

        # Phase 1: LLM call for tool selection
        try:
            response = await self._llm.generate(
                system_prompt=prompt,
                user_message=step.description,
            )
            tool_call = self._parse_tool_call(response.content)
        except LLMOutputParseError as exc:
            step.tool_call = ToolCall(
                tool_name="unknown", arguments={},
                raw_llm_output=exc.raw_output,
                schema_valid=False,
                schema_errors=[str(exc)],
            )
            step.tool_result = ToolResult(success=False, error=str(exc))
            step.status = StepStatus.FAILED
            return step
        except Exception as exc:
            step.tool_call = ToolCall(
                tool_name="unknown", arguments={},
                raw_llm_output=str(exc),
                schema_valid=False,
                schema_errors=[f"LLM call failed: {exc}"],
            )
            step.tool_result = ToolResult(success=False, error=str(exc))
            step.status = StepStatus.FAILED
            return step

        # Phase 2: Schema validation
        is_valid, errors = self._registry.validate_tool_call(tool_call)
        tool_call.schema_valid = is_valid
        tool_call.schema_errors = errors
        step.tool_call = tool_call

        if not is_valid:
            step.tool_result = ToolResult(
                success=False,
                error=f"Schema validation failed: {errors}",
            )
            step.status = StepStatus.FAILED
            return step

        # Phase 3: Execute
        result = self._registry.execute_tool_call(tool_call)
        step.tool_result = result

        if result.success:
            step.status = StepStatus.COMPLETED
            step.completed_at = datetime.now(timezone.utc)
            memory.record_step_output(step.step_id, result.output)
        else:
            step.status = StepStatus.FAILED

        return step

    def _parse_tool_call(self, raw: str) -> ToolCall:
        cleaned = re.sub(r"```json\s*", "", raw)
        cleaned = re.sub(r"```\s*", "", cleaned).strip()
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
            raise LLMOutputParseError(
                expected_format="tool call JSON",
                raw_output=raw,
            ) from exc
