"""Tests for tool registry and implementations."""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.state import ToolCall, ToolResult
from tools.implementations import (
    CalculatorTool,
    TextAnalysisTool,
    WebSearchTool,
    build_default_registry,
)
from tools.registry import BaseTool, ToolRegistry


# ── Registry Tests ───────────────────────────────────────────────────

class TestToolRegistry:
    def test_register_and_lookup(self):
        registry = ToolRegistry()
        tool = CalculatorTool()
        registry.register(tool)
        assert "calculator" in registry
        assert registry.get("calculator") is tool

    def test_duplicate_registration_raises(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(CalculatorTool())

    def test_get_nonexistent_returns_none(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_list_tools(self):
        registry = build_default_registry(use_mock=True)
        tools = registry.list_tools()
        assert "calculator" in tools
        assert "web_search" in tools
        assert "web_fetch" in tools
        assert "text_analysis" in tools
        assert len(tools) >= 4

    def test_schema_export(self):
        registry = build_default_registry(use_mock=True)
        schemas = registry.get_all_schemas()
        assert len(schemas) >= 4
        for schema in schemas:
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema

    def test_validate_tool_call_valid(self):
        registry = build_default_registry(use_mock=True)
        call = ToolCall(
            tool_name="calculator",
            arguments={"expression": "2 + 2"},
        )
        is_valid, errors = registry.validate_tool_call(call)
        assert is_valid
        assert errors == []

    def test_validate_tool_call_invalid_tool(self):
        registry = build_default_registry(use_mock=True)
        call = ToolCall(
            tool_name="nonexistent_tool",
            arguments={},
        )
        is_valid, errors = registry.validate_tool_call(call)
        assert not is_valid
        assert "Unknown tool" in errors[0]

    def test_validate_tool_call_missing_required_args(self):
        registry = build_default_registry(use_mock=True)
        call = ToolCall(
            tool_name="calculator",
            arguments={},  # Missing 'expression'
        )
        is_valid, errors = registry.validate_tool_call(call)
        assert not is_valid


# ── Calculator Tests ─────────────────────────────────────────────────

class TestCalculatorTool:
    def setup_method(self):
        self.calc = CalculatorTool()

    def test_basic_arithmetic(self):
        assert self.calc.execute(expression="2 + 3") == 5
        assert self.calc.execute(expression="10 * 5") == 50
        assert self.calc.execute(expression="100 / 4") == 25.0

    def test_functions(self):
        result = self.calc.execute(expression="sqrt(144)")
        assert result == 12.0

    def test_constants(self):
        import math
        result = self.calc.execute(expression="pi")
        assert abs(result - math.pi) < 0.001

    def test_complex_expression(self):
        result = self.calc.execute(expression="sqrt(144) + 10 * 2")
        assert result == 32.0

    def test_injection_prevention(self):
        with pytest.raises(ValueError, match="Forbidden"):
            self.calc.execute(expression="__import__('os').system('ls')")

    def test_division_by_zero(self):
        with pytest.raises(ValueError):
            self.calc.execute(expression="1/0")

    def test_validate_and_run(self):
        result = self.calc.validate_and_run({"expression": "2 + 2"})
        assert result.success
        assert result.output == 4

    def test_validate_and_run_bad_schema(self):
        result = self.calc.validate_and_run({"wrong_key": "2 + 2"})
        assert not result.success
        assert "Schema validation failed" in result.error


# ── Web Search Tests ─────────────────────────────────────────────────

class TestWebSearchTool:
    def test_mock_search(self):
        tool = WebSearchTool(use_mock=True)
        results = tool.execute(query="test query", max_results=3)
        assert len(results) == 3
        assert all("title" in r for r in results)
        assert all("url" in r for r in results)

    def test_mock_search_validate_and_run(self):
        tool = WebSearchTool(use_mock=True)
        result = tool.validate_and_run(
            {"query": "autonomous agents", "max_results": 5}
        )
        assert result.success
        assert len(result.output) == 5


# ── Text Analysis Tests ──────────────────────────────────────────────

class TestTextAnalysisTool:
    def setup_method(self):
        self.tool = TextAnalysisTool()

    def test_summarize(self):
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "AI systems are becoming more capable. "
            "Machine learning drives innovation. "
            "The future looks bright."
        )
        result = self.tool.execute(text=text, operation="summarize")
        assert "summary" in result
        assert result["sentence_count"] == 4

    def test_sentiment_positive(self):
        result = self.tool.execute(
            text="This is great and excellent work",
            operation="sentiment",
        )
        assert result["sentiment"] == "positive"

    def test_sentiment_negative(self):
        result = self.tool.execute(
            text="This is bad and terrible",
            operation="sentiment",
        )
        assert result["sentiment"] == "negative"

    def test_extract_entities(self):
        result = self.tool.execute(
            text="John Smith works at Google in San Francisco",
            operation="extract_entities",
        )
        assert "entities" in result
        entities = result["entities"]
        assert "John Smith" in entities
        assert "San Francisco" in entities

    def test_key_facts(self):
        result = self.tool.execute(
            text="The population is 37 million. The area is 2194 km2.",
            operation="key_facts",
        )
        assert len(result["key_facts"]) > 0

    def test_unknown_operation(self):
        with pytest.raises(ValueError, match="Unknown operation"):
            self.tool.execute(text="hello", operation="dance")


# ── Integration Test ─────────────────────────────────────────────────

class TestRegistryIntegration:
    def test_execute_through_registry(self):
        registry = build_default_registry(use_mock=True)
        call = ToolCall(
            tool_name="calculator",
            arguments={"expression": "2 ** 10"},
        )
        result = registry.execute_tool_call(call)
        assert result.success
        assert result.output == 1024

    def test_execute_nonexistent_tool(self):
        registry = build_default_registry(use_mock=True)
        call = ToolCall(
            tool_name="quantum_analyzer",
            arguments={"state": "superposition"},
        )
        result = registry.execute_tool_call(call)
        assert not result.success
        assert "not found" in result.error

    def test_schemas_prompt_generation(self):
        registry = build_default_registry(use_mock=True)
        prompt = registry.get_schemas_as_prompt()
        assert "calculator" in prompt
        assert "web_search" in prompt
        assert "expression" in prompt
        assert "query" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
