"""
Immutable state models for the agent system.

Every state transition produces a new snapshot, enabling full
step-level tracing and deterministic replay for evaluation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class ReflectionVerdict(str, Enum):
    CONTINUE = "continue"          # Step succeeded, move to next
    RETRY = "retry"                # Retry same step with adjustments
    REPLAN = "replan"              # Discard remaining plan, replan from here
    ABORT = "abort"                # Unrecoverable failure, stop execution
    SKIP = "skip"                  # Skip this step, move forward


class FailureCategory(str, Enum):
    """Failure-mode taxonomy for evaluation analysis."""
    TOOL_SELECTION_ERROR = "tool_selection_error"
    TOOL_ARG_SCHEMA_VIOLATION = "tool_arg_schema_violation"
    TOOL_EXECUTION_FAILURE = "tool_execution_failure"
    OUTPUT_PARSE_ERROR = "output_parse_error"
    HALLUCINATED_TOOL = "hallucinated_tool"
    WRONG_STEP_ORDER = "wrong_step_order"
    CONTEXT_LOSS = "context_loss"
    INFINITE_LOOP = "infinite_loop"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


# ── Core Data Models ─────────────────────────────────────────────────

class ToolCall(BaseModel):
    """Record of a single tool invocation."""
    tool_name: str
    arguments: dict[str, Any]
    raw_llm_output: str = ""
    schema_valid: bool = True
    schema_errors: list[str] = Field(default_factory=list)


class ToolResult(BaseModel):
    """Result returned by a tool execution."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0


class StepReflection(BaseModel):
    """Structured reflection after a step executes."""
    verdict: ReflectionVerdict
    reasoning: str
    failure_category: Optional[FailureCategory] = None
    suggested_correction: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class PlanStep(BaseModel):
    """A single step in the agent's plan."""
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str
    expected_tool: Optional[str] = None
    depends_on: list[str] = Field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None
    reflection: Optional[StepReflection] = None
    retries: int = 0
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: Optional[datetime] = None


class Plan(BaseModel):
    """An ordered sequence of steps produced by the planner."""
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_description: str
    steps: list[PlanStep]
    is_replanned: bool = False
    parent_plan_id: Optional[str] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class AgentTrace(BaseModel):
    """
    Full execution trace for a single task.
    This is the primary artifact for evaluation and debugging.
    """
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_description: str
    model_name: str = ""
    plans: list[Plan] = Field(default_factory=list)
    final_answer: Optional[str] = None
    success: bool = False
    total_steps_executed: int = 0
    total_retries: int = 0
    total_replans: int = 0
    failure_categories: list[FailureCategory] = Field(default_factory=list)
    wall_time_seconds: float = 0.0
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: Optional[datetime] = None

    def all_steps(self) -> list[PlanStep]:
        """Flatten all steps across all plans."""
        return [step for plan in self.plans for step in plan.steps]

    def completed_steps(self) -> list[PlanStep]:
        return [s for s in self.all_steps() if s.status == StepStatus.COMPLETED]

    def failed_steps(self) -> list[PlanStep]:
        return [s for s in self.all_steps() if s.status == StepStatus.FAILED]


# ── Working Memory ──────────────────────────────────────────────────

class WorkingMemory(BaseModel):
    """
    Accumulates context across steps so the executor/planner
    can reference prior results without re-reading the full trace.
    """
    facts: dict[str, Any] = Field(default_factory=dict)
    step_outputs: dict[str, Any] = Field(default_factory=dict)  # step_id → output
    scratchpad: list[str] = Field(default_factory=list)

    def record_step_output(self, step_id: str, output: Any) -> None:
        self.step_outputs[step_id] = output

    def add_fact(self, key: str, value: Any) -> None:
        self.facts[key] = value

    def get_context_summary(self, max_entries: int = 10) -> str:
        """Render a concise context block for LLM prompts."""
        lines = []
        for sid, out in list(self.step_outputs.items())[-max_entries:]:
            lines.append(f"[Step {sid}] → {_truncate(str(out), 200)}")
        if self.facts:
            lines.append(f"Known facts: {self.facts}")
        return "\n".join(lines) if lines else "(no prior context)"


def _truncate(text: str, max_len: int) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text
