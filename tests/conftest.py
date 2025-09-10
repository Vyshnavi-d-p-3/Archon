"""
Shared test fixtures.

Uses deterministic fakes instead of fragile mocks. A fake implements
the same Protocol as the real component but with predictable behavior.
This means tests verify actual behavior, not mock configuration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.llm_backends import CompletionResponse, DeterministicFakeBackend
from agent.middleware import MiddlewareChain, TokenBudget, build_default_middleware
from agent.state import (
    AgentTrace,
    Plan,
    PlanStep,
    ReflectionVerdict,
    StepReflection,
    StepStatus,
    ToolCall,
    ToolResult,
    WorkingMemory,
)
from config.settings import AgentConfig
from tools.implementations import build_default_registry
from tools.registry import ToolRegistry


@pytest.fixture
def mock_registry() -> ToolRegistry:
    """Registry with mock tools (no network calls)."""
    return build_default_registry(use_mock=True)


@pytest.fixture
def config() -> AgentConfig:
    """Standard test configuration."""
    return AgentConfig.from_env()


@pytest.fixture
def memory() -> WorkingMemory:
    """Fresh working memory."""
    return WorkingMemory()


@pytest.fixture
def token_budget() -> TokenBudget:
    """Token budget for testing."""
    return TokenBudget(
        max_input_tokens=10_000,
        max_output_tokens=5_000,
        max_total_cost_usd=1.00,
    )


@pytest.fixture
def fake_llm() -> DeterministicFakeBackend:
    """
    Deterministic LLM fake with pre-programmed responses.
    Covers the common plan/execute/reflect patterns.
    """
    return DeterministicFakeBackend(
        responses={
            # Planner responses
            "Find the population": json.dumps({
                "reasoning": "Need to search for population data then calculate density",
                "steps": [
                    {
                        "description": "Search for Tokyo population",
                        "tool": "web_search",
                        "args": {"query": "Tokyo population 2024"},
                        "depends_on": [],
                    },
                    {
                        "description": "Calculate density",
                        "tool": "calculator",
                        "args": {"expression": "14000000 / 2194"},
                        "depends_on": [0],
                    },
                ],
            }),
            # Executor responses
            "Search for": json.dumps({
                "tool": "web_search",
                "arguments": {"query": "Tokyo population 2024"},
            }),
            "Calculate": json.dumps({
                "tool": "calculator",
                "arguments": {"expression": "14000000 / 2194"},
            }),
            # Reflector responses
            "Step:": json.dumps({
                "verdict": "continue",
                "reasoning": "Step completed successfully",
                "failure_category": None,
                "suggested_correction": None,
                "confidence": 0.9,
            }),
        },
        default_response=json.dumps({
            "tool": "web_search",
            "arguments": {"query": "test"},
        }),
    )


@pytest.fixture
def sample_trace() -> AgentTrace:
    """Pre-built trace for metrics testing."""
    return AgentTrace(
        task_description="Test task",
        model_name="test-model",
        plans=[
            Plan(
                task_description="Test",
                steps=[
                    PlanStep(
                        description="Step 1: Search",
                        expected_tool="web_search",
                        status=StepStatus.COMPLETED,
                        tool_call=ToolCall(
                            tool_name="web_search",
                            arguments={"query": "test"},
                            schema_valid=True,
                        ),
                        tool_result=ToolResult(
                            success=True,
                            output=[{"title": "Result"}],
                            latency_ms=50.0,
                        ),
                    ),
                    PlanStep(
                        description="Step 2: Calculate",
                        expected_tool="calculator",
                        status=StepStatus.COMPLETED,
                        retries=1,
                        tool_call=ToolCall(
                            tool_name="calculator",
                            arguments={"expression": "2+2"},
                            schema_valid=True,
                        ),
                        tool_result=ToolResult(
                            success=True,
                            output=4,
                            latency_ms=5.0,
                        ),
                        reflection=StepReflection(
                            verdict=ReflectionVerdict.CONTINUE,
                            reasoning="Succeeded on retry",
                            confidence=0.9,
                        ),
                    ),
                    PlanStep(
                        description="Step 3: Failed",
                        expected_tool="web_fetch",
                        status=StepStatus.FAILED,
                        tool_call=ToolCall(
                            tool_name="web_fetch",
                            arguments={"url": "http://bad.url"},
                            schema_valid=True,
                        ),
                        tool_result=ToolResult(
                            success=False,
                            error="Connection refused",
                            latency_ms=1000.0,
                        ),
                    ),
                ],
            )
        ],
        success=True,
        total_steps_executed=3,
        total_retries=1,
        final_answer="The answer is 4.",
    )
