"""
Structured Tool Registry with JSON Schema validation.

Tools self-register via @register_tool decorator. The registry
provides schema introspection for the planner and argument
validation before execution.
"""

from __future__ import annotations

import inspect
import json
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Type

import structlog
from pydantic import BaseModel, ValidationError, create_model

from agent.state import ToolCall, ToolResult

logger = structlog.get_logger(__name__)


# ── Base Tool ────────────────────────────────────────────────────────

class BaseTool(ABC):
    """
    Abstract base for all tools. Each tool declares:
      - name: unique identifier
      - description: what the tool does (injected into LLM prompts)
      - args_schema: Pydantic model for argument validation
    """
    name: str
    description: str
    args_schema: Type[BaseModel]

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Run the tool with validated arguments. Return result or raise."""
        ...

    def get_schema_dict(self) -> dict[str, Any]:
        """JSON Schema representation for LLM function-calling prompts."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_schema.model_json_schema(),
        }

    def validate_and_run(self, raw_args: dict[str, Any]) -> ToolResult:
        """
        Validate arguments against schema, execute, and return a
        structured ToolResult. This is the primary entry point
        called by the executor.
        """
        start = time.perf_counter()
        try:
            validated = self.args_schema.model_validate(raw_args)
            result = self.execute(**validated.model_dump())
            latency = (time.perf_counter() - start) * 1000
            logger.info(
                "tool_executed",
                tool=self.name,
                latency_ms=round(latency, 2),
            )
            return ToolResult(
                success=True,
                output=result,
                latency_ms=round(latency, 2),
            )
        except ValidationError as ve:
            latency = (time.perf_counter() - start) * 1000
            logger.warning(
                "tool_schema_violation",
                tool=self.name,
                errors=ve.errors(),
            )
            return ToolResult(
                success=False,
                error=f"Schema validation failed: {ve.errors()}",
                latency_ms=round(latency, 2),
            )
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            logger.error(
                "tool_execution_error",
                tool=self.name,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            return ToolResult(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=round(latency, 2),
            )


# ── Tool Registry ────────────────────────────────────────────────────

class ToolRegistry:
    """
    Central registry for all available tools.
    Supports registration, lookup, schema export, and
    fuzzy matching for tool selection.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        logger.info("tool_registered", tool=tool.name)

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_all_schemas(self) -> list[dict[str, Any]]:
        """Export all tool schemas for injection into LLM prompts."""
        return [t.get_schema_dict() for t in self._tools.values()]

    def get_schemas_as_prompt(self) -> str:
        """
        Render tool schemas as a formatted block for LLM system prompts.
        """
        lines = ["Available tools:\n"]
        for tool in self._tools.values():
            schema = tool.args_schema.model_json_schema()
            props = schema.get("properties", {})
            required = schema.get("required", [])
            param_strs = []
            for pname, pinfo in props.items():
                req = "(required)" if pname in required else "(optional)"
                ptype = pinfo.get("type", "any")
                desc = pinfo.get("description", "")
                param_strs.append(f"    - {pname} ({ptype}, {req}): {desc}")
            params_block = "\n".join(param_strs) if param_strs else "    (no parameters)"
            lines.append(
                f"Tool: {tool.name}\n"
                f"  Description: {tool.description}\n"
                f"  Parameters:\n{params_block}\n"
            )
        return "\n".join(lines)

    def validate_tool_call(self, tool_call: ToolCall) -> tuple[bool, list[str]]:
        """
        Validate a ToolCall without executing.
        Returns (is_valid, list_of_errors).
        Used by the evaluation harness for schema-adherence scoring.
        """
        tool = self.get(tool_call.tool_name)
        if tool is None:
            return False, [f"Unknown tool: '{tool_call.tool_name}'"]
        try:
            tool.args_schema.model_validate(tool_call.arguments)
            return True, []
        except ValidationError as ve:
            return False, [str(e) for e in ve.errors()]

    def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Look up tool by name and execute with validation."""
        tool = self.get(tool_call.tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_call.tool_name}' not found in registry. "
                      f"Available: {self.list_tools()}",
            )
        return tool.validate_and_run(tool_call.arguments)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ── Decorator for quick tool creation ────────────────────────────────

def tool(
    name: str,
    description: str,
    args_schema: Optional[Type[BaseModel]] = None,
):
    """
    Decorator to register a plain function as a tool.

    Usage:
        class SearchArgs(BaseModel):
            query: str = Field(description="Search query")

        @tool("web_search", "Search the web", SearchArgs)
        def web_search(query: str) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        schema = args_schema
        if schema is None:
            # Auto-generate schema from function signature
            sig = inspect.signature(func)
            fields = {}
            for pname, param in sig.parameters.items():
                annotation = param.annotation if param.annotation != inspect.Parameter.empty else Any
                default = param.default if param.default != inspect.Parameter.empty else ...
                fields[pname] = (annotation, default)
            schema = create_model(f"{name}_Args", **fields)  # type: ignore

        class FuncTool(BaseTool):
            pass

        instance = FuncTool()
        instance.name = name
        instance.description = description
        instance.args_schema = schema
        instance.execute = lambda **kw: func(**kw)  # type: ignore

        # Stash for later registry.register()
        func._tool_instance = instance  # type: ignore
        return func

    return decorator
