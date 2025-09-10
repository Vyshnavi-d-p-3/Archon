"""
Protocol definitions for the agent system.

This module defines the contracts between components using typing.Protocol.
No component should depend on a concrete implementation — only on these
protocols. This enables:
  - Swapping LLM backends without touching agent logic
  - Testing with deterministic fakes (not fragile mocks)
  - Adding new tools without modifying the executor
  - Composable middleware via the interceptor protocol

Design notes for reviewers:
  - Protocols are runtime-checkable where feasible for fail-fast DI
  - Generic types are used for tool result variance
  - All I/O-bound methods are async — sync wrappers live in adapters only
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from agent.state import (
    AgentTrace,
    Plan,
    PlanStep,
    StepReflection,
    ToolCall,
    ToolResult,
    WorkingMemory,
)


# ═══════════════════════════════════════════════════════════════════════
# LLM Backend Protocol
# ═══════════════════════════════════════════════════════════════════════

@runtime_checkable
class LLMBackend(Protocol):
    """
    Abstraction over any LLM provider.

    Why a Protocol and not an ABC?
    - Structural subtyping: any class with these methods satisfies
      the contract — no inheritance required.
    - Enables wrapping third-party classes (LangChain, litellm, raw HTTP)
      without modification.
    """

    @property
    def model_id(self) -> str:
        """Canonical model identifier (e.g., 'gpt-4o', 'llama-3-8b')."""
        ...

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """
        Generate a completion.

        Args:
            system_prompt: System-level instructions.
            user_message: User turn content.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.
            response_format: Optional JSON schema constraint.

        Returns:
            LLMResponse with content, token counts, and metadata.
        """
        ...


class LLMResponse(Protocol):
    """Structured response from an LLM call."""

    @property
    def content(self) -> str:
        """The generated text."""
        ...

    @property
    def input_tokens(self) -> int:
        """Tokens consumed by the prompt."""
        ...

    @property
    def output_tokens(self) -> int:
        """Tokens generated in the response."""
        ...

    @property
    def model(self) -> str:
        """Model that produced this response."""
        ...


# ═══════════════════════════════════════════════════════════════════════
# Tool Provider Protocol
# ═══════════════════════════════════════════════════════════════════════

@runtime_checkable
class ToolProvider(Protocol):
    """
    Contract for the tool subsystem.

    Separates tool management (registration, schema export) from
    tool execution, enabling different execution strategies
    (local, remote, sandboxed) behind the same interface.
    """

    def list_tools(self) -> list[str]:
        """Return names of all registered tools."""
        ...

    def get_schemas_for_prompt(self) -> str:
        """Render tool schemas as text for LLM prompt injection."""
        ...

    def validate(self, tool_call: ToolCall) -> tuple[bool, list[str]]:
        """Validate a tool call without executing. Returns (valid, errors)."""
        ...

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a validated tool call and return the result."""
        ...


# ═══════════════════════════════════════════════════════════════════════
# Planner / Executor / Reflector Protocols
# ═══════════════════════════════════════════════════════════════════════

@runtime_checkable
class Planner(Protocol):
    """Decomposes a task into an executable plan."""

    async def create_plan(
        self,
        task: str,
        memory: WorkingMemory,
    ) -> Plan:
        ...

    async def replan(
        self,
        task: str,
        completed_steps: Sequence[PlanStep],
        failure_info: str,
        memory: WorkingMemory,
    ) -> Plan:
        ...


@runtime_checkable
class Executor(Protocol):
    """Executes a single plan step."""

    async def execute_step(
        self,
        step: PlanStep,
        memory: WorkingMemory,
        correction_hint: str = "",
    ) -> PlanStep:
        ...


@runtime_checkable
class Reflector(Protocol):
    """Analyzes step results and produces a verdict."""

    async def reflect(
        self,
        step: PlanStep,
        memory: WorkingMemory,
    ) -> StepReflection:
        ...


# ═══════════════════════════════════════════════════════════════════════
# Middleware / Interceptor Protocol
# ═══════════════════════════════════════════════════════════════════════

@runtime_checkable
class StepInterceptor(Protocol):
    """
    Middleware that wraps step execution.

    Interceptors form a chain (like servlet filters or gRPC interceptors).
    Each can:
      - Inspect/modify the step before execution
      - Inspect/modify the result after execution
      - Short-circuit execution (e.g., budget exceeded)
      - Record telemetry
    """

    async def before_step(
        self,
        step: PlanStep,
        memory: WorkingMemory,
    ) -> PlanStep | None:
        """
        Called before step execution.
        Return None to short-circuit (step will be marked SKIPPED).
        Return the (possibly modified) step to continue.
        """
        ...

    async def after_step(
        self,
        step: PlanStep,
        reflection: StepReflection,
        memory: WorkingMemory,
    ) -> StepReflection:
        """
        Called after step execution and reflection.
        Can modify the reflection verdict.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════
# Event Bus Protocol (for decoupled observability)
# ═══════════════════════════════════════════════════════════════════════

@runtime_checkable
class EventBus(Protocol):
    """
    Publish-subscribe event bus for decoupling agent internals
    from observability/persistence concerns.
    """

    def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        """Fire-and-forget event publication."""
        ...

    def subscribe(
        self,
        event_type: str,
        handler: Any,  # Callable[[dict[str, Any]], None]
    ) -> None:
        """Register a handler for an event type."""
        ...
