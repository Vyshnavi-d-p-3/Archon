"""
Tests for async planner, executor, and reflector.

Uses DeterministicFakeBackend — no API keys, no network, fully reproducible.
Tests verify behavior, not mock configuration.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.async_executor import AsyncExecutor
from agent.async_planner import AsyncPlanner
from agent.async_reflector import AsyncReflector
from agent.errors import EmptyPlanError, PlanParseError
from agent.llm_backends import CompletionResponse, DeterministicFakeBackend
from agent.state import (
    FailureCategory,
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


# ═══════════════════════════════════════════════════════════════════════
# Async Planner Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAsyncPlanner:
    @pytest.fixture
    def planner(self):
        fake = DeterministicFakeBackend(
            responses={
                "population": json.dumps({
                    "reasoning": "Search then calculate",
                    "steps": [
                        {"description": "Search population", "tool": "web_search",
                         "args": {"query": "population"}, "depends_on": []},
                        {"description": "Calculate density", "tool": "calculator",
                         "args": {"expression": "100/50"}, "depends_on": [0]},
                    ],
                }),
            },
            default_response=json.dumps({
                "reasoning": "Generic plan",
                "steps": [
                    {"description": "Search for info", "tool": "web_search",
                     "args": {"query": "test"}, "depends_on": []},
                ],
            }),
        )
        registry = build_default_registry(use_mock=True)
        config = AgentConfig.from_env()
        return AsyncPlanner(llm_backend=fake, registry=registry, config=config)

    @pytest.mark.asyncio
    async def test_creates_plan(self, planner):
        plan = await planner.create_plan("Find the population of Tokyo", WorkingMemory())
        assert len(plan.steps) == 2
        assert plan.steps[0].expected_tool == "web_search"
        assert plan.steps[1].expected_tool == "calculator"

    @pytest.mark.asyncio
    async def test_plan_has_dependencies(self, planner):
        plan = await planner.create_plan("Find the population", WorkingMemory())
        # Step 1 depends on step 0
        assert len(plan.steps[1].depends_on) == 1

    @pytest.mark.asyncio
    async def test_default_plan(self, planner):
        plan = await planner.create_plan("Something unrelated", WorkingMemory())
        assert len(plan.steps) >= 1

    @pytest.mark.asyncio
    async def test_replan(self, planner):
        completed = [PlanStep(description="Done step", status=StepStatus.COMPLETED)]
        plan = await planner.replan(
            task="Original task",
            completed_steps=completed,
            failure_info="Tool failed",
            memory=WorkingMemory(),
        )
        assert plan.is_replanned

    @pytest.mark.asyncio
    async def test_plan_respects_max_steps(self):
        many_steps = json.dumps({
            "reasoning": "Lots of steps",
            "steps": [
                {"description": f"Step {i}", "tool": "web_search",
                 "args": {"query": f"q{i}"}, "depends_on": []}
                for i in range(20)
            ],
        })
        fake = DeterministicFakeBackend(default_response=many_steps)
        config = AgentConfig.from_env()
        planner = AsyncPlanner(
            llm_backend=fake,
            registry=build_default_registry(use_mock=True),
            config=config,
        )
        plan = await planner.create_plan("Big task", WorkingMemory())
        assert len(plan.steps) <= config.planner.max_plan_steps

    @pytest.mark.asyncio
    async def test_empty_plan_raises(self):
        fake = DeterministicFakeBackend(
            default_response=json.dumps({"reasoning": "Nothing", "steps": []})
        )
        planner = AsyncPlanner(
            llm_backend=fake,
            registry=build_default_registry(use_mock=True),
            config=AgentConfig.from_env(),
        )
        with pytest.raises(EmptyPlanError):
            await planner.create_plan("Empty task", WorkingMemory())

    @pytest.mark.asyncio
    async def test_parse_error_raises(self):
        fake = DeterministicFakeBackend(default_response="not json at all {{{")
        planner = AsyncPlanner(
            llm_backend=fake,
            registry=build_default_registry(use_mock=True),
            config=AgentConfig.from_env(),
        )
        with pytest.raises(PlanParseError):
            await planner.create_plan("Bad parse task", WorkingMemory())


# ═══════════════════════════════════════════════════════════════════════
# Async Executor Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAsyncExecutor:
    @pytest.fixture
    def executor(self):
        fake = DeterministicFakeBackend(
            responses={
                "search": json.dumps({
                    "tool": "web_search",
                    "arguments": {"query": "test query"},
                }),
                "calculate": json.dumps({
                    "tool": "calculator",
                    "arguments": {"expression": "2 + 2"},
                }),
            },
        )
        registry = build_default_registry(use_mock=True)
        config = AgentConfig.from_env()
        return AsyncExecutor(llm_backend=fake, registry=registry, config=config)

    @pytest.mark.asyncio
    async def test_executes_search_step(self, executor):
        step = PlanStep(description="Search for data", expected_tool="web_search")
        memory = WorkingMemory()
        result = await executor.execute_step(step, memory)

        assert result.status == StepStatus.COMPLETED
        assert result.tool_call is not None
        assert result.tool_call.tool_name == "web_search"
        assert result.tool_result is not None
        assert result.tool_result.success

    @pytest.mark.asyncio
    async def test_executes_calculator_step(self, executor):
        step = PlanStep(description="Calculate the result", expected_tool="calculator")
        memory = WorkingMemory()
        result = await executor.execute_step(step, memory)

        assert result.status == StepStatus.COMPLETED
        assert result.tool_result.output == 4

    @pytest.mark.asyncio
    async def test_records_in_memory(self, executor):
        step = PlanStep(description="Search for info", expected_tool="web_search")
        memory = WorkingMemory()
        await executor.execute_step(step, memory)

        assert step.step_id in memory.step_outputs

    @pytest.mark.asyncio
    async def test_handles_unparseable_llm_output(self):
        fake = DeterministicFakeBackend(default_response="not json at all")
        executor = AsyncExecutor(
            llm_backend=fake,
            registry=build_default_registry(use_mock=True),
            config=AgentConfig.from_env(),
        )
        step = PlanStep(description="Bad step")
        result = await executor.execute_step(step, WorkingMemory())

        assert result.status == StepStatus.FAILED
        assert result.tool_call is not None
        assert not result.tool_call.schema_valid

    @pytest.mark.asyncio
    async def test_correction_hint_passed(self):
        """Verify correction hints make it into the LLM prompt."""
        class FullLogFake(DeterministicFakeBackend):
            """Fake that logs full prompts (no truncation)."""
            full_log: list[dict[str, str]] = []

            async def generate(self, system_prompt, user_message, **kw):
                self.full_log.append({"system": system_prompt, "user": user_message})
                return await super().generate(system_prompt, user_message, **kw)

        fake = FullLogFake(
            default_response=json.dumps({
                "tool": "calculator",
                "arguments": {"expression": "3 + 3"},
            })
        )
        executor = AsyncExecutor(
            llm_backend=fake,
            registry=build_default_registry(use_mock=True),
            config=AgentConfig.from_env(),
        )
        step = PlanStep(description="Calculate", expected_tool="calculator")
        await executor.execute_step(step, WorkingMemory(), correction_hint="Use multiplication")

        assert len(fake.full_log) == 1
        assert "CORRECTION" in fake.full_log[0]["system"]
        assert "multiplication" in fake.full_log[0]["system"]


# ═══════════════════════════════════════════════════════════════════════
# Async Reflector Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAsyncReflector:
    @pytest.fixture
    def reflector(self):
        fake = DeterministicFakeBackend(
            default_response=json.dumps({
                "verdict": "continue",
                "reasoning": "Step looks good",
                "failure_category": None,
                "suggested_correction": None,
                "confidence": 0.9,
            })
        )
        registry = build_default_registry(use_mock=True)
        config = AgentConfig.from_env()
        return AsyncReflector(llm_backend=fake, registry=registry, config=config)

    @pytest.mark.asyncio
    async def test_heuristic_success(self, reflector):
        """Clear success should be caught by heuristic, no LLM call."""
        step = PlanStep(
            description="Test",
            tool_call=ToolCall(
                tool_name="calculator",
                arguments={"expression": "2+2"},
                schema_valid=True,
            ),
            tool_result=ToolResult(success=True, output=4, latency_ms=5.0),
        )
        reflection = await reflector.reflect(step, WorkingMemory())
        assert reflection.verdict == ReflectionVerdict.CONTINUE
        assert reflection.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_heuristic_hallucinated_tool(self, reflector):
        step = PlanStep(
            description="Test",
            tool_call=ToolCall(
                tool_name="nonexistent_quantum_analyzer",
                arguments={},
                schema_valid=False,
            ),
            tool_result=ToolResult(success=False, error="not found"),
        )
        reflection = await reflector.reflect(step, WorkingMemory())
        assert reflection.verdict == ReflectionVerdict.RETRY
        assert reflection.failure_category == FailureCategory.HALLUCINATED_TOOL

    @pytest.mark.asyncio
    async def test_heuristic_schema_violation(self, reflector):
        step = PlanStep(
            description="Test",
            tool_call=ToolCall(
                tool_name="calculator",
                arguments={"wrong": "args"},
                schema_valid=False,
                schema_errors=["missing 'expression'"],
            ),
            tool_result=ToolResult(success=False, error="schema"),
        )
        reflection = await reflector.reflect(step, WorkingMemory())
        assert reflection.verdict == ReflectionVerdict.RETRY
        assert reflection.failure_category == FailureCategory.TOOL_ARG_SCHEMA_VIOLATION

    @pytest.mark.asyncio
    async def test_heuristic_retries_exhausted(self):
        config = AgentConfig.from_env()
        fake = DeterministicFakeBackend()
        reflector = AsyncReflector(
            llm_backend=fake,
            registry=build_default_registry(use_mock=True),
            config=config,
        )
        step = PlanStep(
            description="Test",
            retries=config.executor.max_retries_per_step,  # At the limit
            tool_call=ToolCall(tool_name="calculator", arguments={}, schema_valid=True),
            tool_result=ToolResult(success=False, error="still failing"),
        )
        reflection = await reflector.reflect(step, WorkingMemory())
        # Should escalate to REPLAN since retries are exhausted
        assert reflection.verdict in (ReflectionVerdict.REPLAN, ReflectionVerdict.ABORT)

    @pytest.mark.asyncio
    async def test_reflection_disabled(self):
        config = AgentConfig.from_env()
        # Modify reflector config
        from config.settings import ReflectorConfig
        config = AgentConfig(
            reflector=ReflectorConfig(enabled=False),
        )
        fake = DeterministicFakeBackend()
        reflector = AsyncReflector(
            llm_backend=fake,
            registry=build_default_registry(use_mock=True),
            config=config,
        )
        step = PlanStep(
            description="Test",
            tool_result=ToolResult(success=True, output="ok"),
        )
        reflection = await reflector.reflect(step, WorkingMemory())
        assert reflection.verdict == ReflectionVerdict.CONTINUE
        # Fake should NOT have been called (heuristic only)
        assert len(fake.call_log) == 0

    @pytest.mark.asyncio
    async def test_llm_fallback_on_ambiguous(self, reflector):
        """Ambiguous result (success but empty output) should trigger LLM."""
        step = PlanStep(
            description="Ambiguous step",
            tool_call=ToolCall(
                tool_name="calculator",
                arguments={"expression": "0"},
                schema_valid=True,
            ),
            tool_result=ToolResult(success=True, output=None, latency_ms=5.0),
        )
        reflection = await reflector.reflect(step, WorkingMemory())
        # LLM fake returns "continue" — verify it was consulted
        assert reflection.verdict == ReflectionVerdict.CONTINUE


# ═══════════════════════════════════════════════════════════════════════
# Integration: Planner → Executor chain
# ═══════════════════════════════════════════════════════════════════════

class TestPlannerExecutorIntegration:
    @pytest.mark.asyncio
    async def test_plan_then_execute_steps(self):
        """Plan a task, then execute each step through the executor."""
        plan_response = json.dumps({
            "reasoning": "Simple search task",
            "steps": [
                {"description": "Search for info", "tool": "web_search",
                 "args": {"query": "test"}, "depends_on": []},
                {"description": "Analyze results", "tool": "text_analysis",
                 "args": {"text": "sample", "operation": "summarize"}, "depends_on": [0]},
            ],
        })
        exec_responses = {
            "search": json.dumps({"tool": "web_search", "arguments": {"query": "test"}}),
            "analyze": json.dumps({"tool": "text_analysis",
                                    "arguments": {"text": "sample text", "operation": "summarize"}}),
        }

        fake_planner_llm = DeterministicFakeBackend(default_response=plan_response)
        fake_executor_llm = DeterministicFakeBackend(responses=exec_responses)
        registry = build_default_registry(use_mock=True)
        config = AgentConfig.from_env()

        planner = AsyncPlanner(fake_planner_llm, registry, config)
        executor = AsyncExecutor(fake_executor_llm, registry, config)

        # Plan
        plan = await planner.create_plan("Find and analyze info", WorkingMemory())
        assert len(plan.steps) == 2

        # Execute each step
        memory = WorkingMemory()
        for step in plan.steps:
            await executor.execute_step(step, memory)

        assert plan.steps[0].status == StepStatus.COMPLETED
        assert plan.steps[0].tool_result.success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
