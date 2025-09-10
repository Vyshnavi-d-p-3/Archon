"""
Exception hierarchy for the agent system.

Design principles:
  1. Every exception carries structured context (not just a message string)
  2. Exceptions map to failure categories for automatic taxonomy classification
  3. Retryable vs. fatal is encoded in the type, not guessed by the caller
  4. All exceptions are serializable for trace persistence

Usage:
    try:
        result = await tool_provider.execute(call)
    except ToolSchemaError as e:
        # e.tool_name, e.schema_errors available for structured handling
        logger.warning("schema_violation", **e.context())
    except RetryableError:
        # Any retryable subclass — safe to retry
        pass
    except FatalAgentError:
        # Unrecoverable — abort the run
        pass
"""

from __future__ import annotations

from typing import Any, Optional

from agent.state import FailureCategory


class AgentError(Exception):
    """
    Base exception for all agent errors.
    Carries structured context for logging and trace persistence.
    """

    failure_category: FailureCategory = FailureCategory.UNKNOWN

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(message)
        self._context = kwargs

    def context(self) -> dict[str, Any]:
        """Structured context for logging/serialization."""
        return {
            "error_type": type(self).__name__,
            "message": str(self),
            "failure_category": self.failure_category.value,
            **self._context,
        }


# ── Retryable vs. Fatal base classes ─────────────────────────────────

class RetryableError(AgentError):
    """
    Base for errors where retrying (possibly with correction) may succeed.
    The reflector should issue RETRY, not ABORT.
    """
    pass


class FatalAgentError(AgentError):
    """
    Base for unrecoverable errors. The reflector should ABORT.
    """
    pass


# ── LLM Errors ───────────────────────────────────────────────────────

class LLMError(AgentError):
    """Base for LLM-related errors."""
    pass


class LLMConnectionError(RetryableError, LLMError):
    """LLM API unreachable — transient, safe to retry."""
    failure_category = FailureCategory.TIMEOUT

    def __init__(self, provider: str, model: str, cause: str):
        super().__init__(
            f"LLM connection failed: {provider}/{model}: {cause}",
            provider=provider,
            model=model,
            cause=cause,
        )


class LLMRateLimitError(RetryableError, LLMError):
    """Rate limited — retry after backoff."""
    failure_category = FailureCategory.TIMEOUT

    def __init__(self, provider: str, retry_after: float | None = None):
        super().__init__(
            f"Rate limited by {provider}"
            + (f", retry after {retry_after}s" if retry_after else ""),
            provider=provider,
            retry_after=retry_after,
        )
        self.retry_after = retry_after


class LLMOutputParseError(RetryableError, LLMError):
    """LLM output couldn't be parsed into expected structure."""
    failure_category = FailureCategory.OUTPUT_PARSE_ERROR

    def __init__(self, expected_format: str, raw_output: str):
        super().__init__(
            f"Failed to parse LLM output as {expected_format}",
            expected_format=expected_format,
            raw_output=raw_output[:500],
        )
        self.raw_output = raw_output


# ── Tool Errors ──────────────────────────────────────────────────────

class ToolError(AgentError):
    """Base for tool-related errors."""

    def __init__(self, tool_name: str, message: str, **kwargs: Any):
        super().__init__(message, tool_name=tool_name, **kwargs)
        self.tool_name = tool_name


class ToolNotFoundError(RetryableError, ToolError):
    """Agent tried to use a tool that doesn't exist in the registry."""
    failure_category = FailureCategory.HALLUCINATED_TOOL

    def __init__(self, tool_name: str, available_tools: list[str]):
        super().__init__(
            tool_name,
            f"Tool '{tool_name}' not found. Available: {available_tools}",
            available_tools=available_tools,
        )
        self.available_tools = available_tools


class ToolSchemaError(RetryableError, ToolError):
    """Tool arguments failed schema validation."""
    failure_category = FailureCategory.TOOL_ARG_SCHEMA_VIOLATION

    def __init__(
        self,
        tool_name: str,
        schema_errors: list[str],
        provided_args: dict[str, Any],
    ):
        super().__init__(
            tool_name,
            f"Schema validation failed for '{tool_name}': {schema_errors}",
            schema_errors=schema_errors,
            provided_args=provided_args,
        )
        self.schema_errors = schema_errors
        self.provided_args = provided_args


class ToolExecutionError(RetryableError, ToolError):
    """Tool executed but raised an error."""
    failure_category = FailureCategory.TOOL_EXECUTION_FAILURE

    def __init__(self, tool_name: str, cause: Exception):
        super().__init__(
            tool_name,
            f"Tool '{tool_name}' execution failed: {cause}",
            cause_type=type(cause).__name__,
            cause_message=str(cause),
        )
        self.__cause__ = cause


class ToolTimeoutError(RetryableError, ToolError):
    """Tool execution exceeded time budget."""
    failure_category = FailureCategory.TIMEOUT

    def __init__(self, tool_name: str, timeout_seconds: float):
        super().__init__(
            tool_name,
            f"Tool '{tool_name}' timed out after {timeout_seconds}s",
            timeout_seconds=timeout_seconds,
        )


# ── Planning Errors ──────────────────────────────────────────────────

class PlanningError(AgentError):
    """Base for planning failures."""
    pass


class EmptyPlanError(RetryableError, PlanningError):
    """Planner produced an empty plan — retry with different prompt."""
    failure_category = FailureCategory.OUTPUT_PARSE_ERROR

    def __init__(self, task: str):
        super().__init__(
            f"Planner produced empty plan for task: {task[:100]}",
            task=task[:200],
        )


class PlanParseError(RetryableError, PlanningError):
    """Planner output couldn't be parsed into a Plan."""
    failure_category = FailureCategory.OUTPUT_PARSE_ERROR

    def __init__(self, raw_output: str, parse_error: str):
        super().__init__(
            f"Failed to parse plan: {parse_error}",
            raw_output=raw_output[:500],
            parse_error=parse_error,
        )


# ── Budget / Resource Errors ─────────────────────────────────────────

class BudgetExceededError(FatalAgentError):
    """Token or cost budget exhausted."""
    failure_category = FailureCategory.UNKNOWN

    def __init__(
        self,
        budget_type: str,  # "token" | "cost" | "time"
        limit: float,
        consumed: float,
    ):
        super().__init__(
            f"{budget_type.title()} budget exceeded: {consumed:.0f} / {limit:.0f}",
            budget_type=budget_type,
            limit=limit,
            consumed=consumed,
        )


class MaxRetriesExceededError(FatalAgentError):
    """Step exhausted all retry attempts."""
    failure_category = FailureCategory.TOOL_EXECUTION_FAILURE

    def __init__(self, step_description: str, max_retries: int):
        super().__init__(
            f"Exhausted {max_retries} retries for step: {step_description[:100]}",
            step_description=step_description,
            max_retries=max_retries,
        )


class CircuitBreakerOpenError(FatalAgentError):
    """Too many consecutive failures — circuit breaker tripped."""
    failure_category = FailureCategory.INFINITE_LOOP

    def __init__(self, consecutive_failures: int, threshold: int):
        super().__init__(
            f"Circuit breaker: {consecutive_failures} consecutive failures "
            f"(threshold: {threshold})",
            consecutive_failures=consecutive_failures,
            threshold=threshold,
        )
