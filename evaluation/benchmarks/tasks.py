"""
Benchmark task definitions for evaluating agent performance.

Each benchmark defines:
  - A task description (natural language)
  - Expected tool sequence (ground truth)
  - Evaluation criteria (what constitutes success)
  - Difficulty level and category

Categories:
  - multi_step_reasoning: Tasks requiring 3+ coordinated steps
  - web_navigation: Tasks requiring search + fetch + analysis
  - calculation: Tasks requiring computation with tool chaining
  - information_synthesis: Tasks requiring multiple sources merged
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class TaskCategory(str, Enum):
    MULTI_STEP_REASONING = "multi_step_reasoning"
    WEB_NAVIGATION = "web_navigation"
    CALCULATION = "calculation"
    INFORMATION_SYNTHESIS = "information_synthesis"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class ExpectedStep:
    """Ground-truth expectation for a single step."""
    tool: str
    description: str
    required_args: list[str] = field(default_factory=list)
    # Loose match: these keys must be present in tool args
    expected_arg_patterns: dict[str, str] = field(default_factory=dict)
    # If true, this step can appear anywhere; if false, order matters
    order_flexible: bool = False


@dataclass
class BenchmarkTask:
    """A single evaluation task with ground truth."""
    task_id: str
    description: str
    category: TaskCategory
    difficulty: Difficulty
    expected_steps: list[ExpectedStep]
    success_criteria: str  # Human-readable description
    expected_final_answer_contains: list[str] = field(default_factory=list)
    max_acceptable_steps: int = 10
    tags: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# Benchmark suite
# ═══════════════════════════════════════════════════════════════════

BENCHMARK_TASKS: list[BenchmarkTask] = [
    # ── Multi-step reasoning ─────────────────────────────────────
    BenchmarkTask(
        task_id="msr_001",
        description=(
            "Find the current population of Tokyo and New York City, "
            "then calculate which city is more densely populated given "
            "Tokyo's area is 2,194 km² and NYC's area is 783 km²."
        ),
        category=TaskCategory.MULTI_STEP_REASONING,
        difficulty=Difficulty.MEDIUM,
        expected_steps=[
            ExpectedStep(
                tool="web_search",
                description="Search for Tokyo population",
                required_args=["query"],
                expected_arg_patterns={"query": "tokyo.*population"},
            ),
            ExpectedStep(
                tool="web_search",
                description="Search for NYC population",
                required_args=["query"],
                expected_arg_patterns={"query": "new york.*population"},
            ),
            ExpectedStep(
                tool="calculator",
                description="Calculate Tokyo density",
                required_args=["expression"],
            ),
            ExpectedStep(
                tool="calculator",
                description="Calculate NYC density",
                required_args=["expression"],
            ),
        ],
        success_criteria="Correctly identifies NYC as more densely populated",
        expected_final_answer_contains=["density", "km"],
    ),

    BenchmarkTask(
        task_id="msr_002",
        description=(
            "Search for the top 3 programming languages by popularity in 2024, "
            "then write a summary comparing their key strengths to a file."
        ),
        category=TaskCategory.MULTI_STEP_REASONING,
        difficulty=Difficulty.MEDIUM,
        expected_steps=[
            ExpectedStep(
                tool="web_search",
                description="Search programming language popularity 2024",
                required_args=["query"],
            ),
            ExpectedStep(
                tool="web_fetch",
                description="Read detailed rankings page",
                required_args=["url"],
                order_flexible=True,
            ),
            ExpectedStep(
                tool="file_write",
                description="Write summary to file",
                required_args=["filename", "content"],
            ),
        ],
        success_criteria="Identifies top languages and saves comparison",
        expected_final_answer_contains=["Python"],
    ),

    BenchmarkTask(
        task_id="msr_003",
        description=(
            "What is the square root of the sum of the first 10 prime numbers? "
            "Show your work step by step."
        ),
        category=TaskCategory.CALCULATION,
        difficulty=Difficulty.EASY,
        expected_steps=[
            ExpectedStep(
                tool="calculator",
                description="Sum first 10 primes",
                required_args=["expression"],
            ),
            ExpectedStep(
                tool="calculator",
                description="Take square root",
                required_args=["expression"],
            ),
        ],
        success_criteria="Correct answer: sqrt(129) ≈ 11.36",
        expected_final_answer_contains=["11.3"],
    ),

    # ── Web navigation ───────────────────────────────────────────
    BenchmarkTask(
        task_id="wnav_001",
        description=(
            "Find a recent research paper about transformer architecture "
            "improvements published in 2024, fetch the page, and extract "
            "the key contributions mentioned in the abstract."
        ),
        category=TaskCategory.WEB_NAVIGATION,
        difficulty=Difficulty.HARD,
        expected_steps=[
            ExpectedStep(
                tool="web_search",
                description="Search for transformer papers 2024",
                required_args=["query"],
            ),
            ExpectedStep(
                tool="web_fetch",
                description="Fetch the paper or abstract page",
                required_args=["url"],
            ),
            ExpectedStep(
                tool="text_analysis",
                description="Extract key facts from abstract",
                required_args=["text", "operation"],
                expected_arg_patterns={"operation": "key_facts|summarize"},
            ),
        ],
        success_criteria="Identifies a real paper and extracts contributions",
        expected_final_answer_contains=["transformer"],
    ),

    BenchmarkTask(
        task_id="wnav_002",
        description=(
            "Search for the latest Python release version, go to the "
            "release notes page, and summarize the top 3 new features."
        ),
        category=TaskCategory.WEB_NAVIGATION,
        difficulty=Difficulty.MEDIUM,
        expected_steps=[
            ExpectedStep(
                tool="web_search",
                description="Search latest Python release",
                required_args=["query"],
            ),
            ExpectedStep(
                tool="web_fetch",
                description="Fetch release notes",
                required_args=["url"],
            ),
            ExpectedStep(
                tool="text_analysis",
                description="Summarize features",
                required_args=["text", "operation"],
            ),
        ],
        success_criteria="Identifies correct Python version and top features",
        expected_final_answer_contains=["Python", "3."],
    ),

    # ── Information synthesis ────────────────────────────────────
    BenchmarkTask(
        task_id="isyn_001",
        description=(
            "Compare the GDP of the US, China, and Japan. Search for each, "
            "calculate the percentage difference between US and China GDP, "
            "and write a brief analysis to a file."
        ),
        category=TaskCategory.INFORMATION_SYNTHESIS,
        difficulty=Difficulty.HARD,
        expected_steps=[
            ExpectedStep(
                tool="web_search",
                description="Search US GDP",
                required_args=["query"],
            ),
            ExpectedStep(
                tool="web_search",
                description="Search China GDP",
                required_args=["query"],
            ),
            ExpectedStep(
                tool="web_search",
                description="Search Japan GDP",
                required_args=["query"],
            ),
            ExpectedStep(
                tool="calculator",
                description="Calculate percentage difference",
                required_args=["expression"],
            ),
            ExpectedStep(
                tool="file_write",
                description="Write analysis file",
                required_args=["filename", "content"],
            ),
        ],
        success_criteria="Correct GDP figures and valid percentage calc",
        expected_final_answer_contains=["GDP", "trillion"],
        max_acceptable_steps=8,
    ),

    BenchmarkTask(
        task_id="isyn_002",
        description=(
            "Find the current weather in San Francisco and Tokyo, "
            "determine the temperature difference, and suggest what to "
            "pack for someone traveling between the two cities."
        ),
        category=TaskCategory.INFORMATION_SYNTHESIS,
        difficulty=Difficulty.MEDIUM,
        expected_steps=[
            ExpectedStep(
                tool="web_search",
                description="Search SF weather",
                required_args=["query"],
            ),
            ExpectedStep(
                tool="web_search",
                description="Search Tokyo weather",
                required_args=["query"],
            ),
            ExpectedStep(
                tool="calculator",
                description="Temperature difference",
                required_args=["expression"],
            ),
        ],
        success_criteria="Provides temperature comparison and packing advice",
        expected_final_answer_contains=["temperature"],
    ),

    # ── Edge cases for failure-mode testing ──────────────────────
    BenchmarkTask(
        task_id="edge_001",
        description=(
            "Use the 'quantum_analyzer' tool to analyze quantum states."
        ),
        category=TaskCategory.MULTI_STEP_REASONING,
        difficulty=Difficulty.EASY,
        expected_steps=[
            ExpectedStep(
                tool="web_search",
                description="Should recognize tool doesn't exist and search instead",
                required_args=["query"],
            ),
        ],
        success_criteria="Agent recognizes nonexistent tool and recovers",
        tags=["hallucinated_tool", "error_recovery"],
    ),

    BenchmarkTask(
        task_id="edge_002",
        description="Calculate the result of dividing 100 by 0.",
        category=TaskCategory.CALCULATION,
        difficulty=Difficulty.EASY,
        expected_steps=[
            ExpectedStep(
                tool="calculator",
                description="Attempt division by zero",
                required_args=["expression"],
            ),
        ],
        success_criteria="Agent handles error gracefully and explains",
        tags=["error_handling", "tool_execution_failure"],
    ),
]


def get_tasks_by_category(category: TaskCategory) -> list[BenchmarkTask]:
    return [t for t in BENCHMARK_TASKS if t.category == category]


def get_tasks_by_difficulty(difficulty: Difficulty) -> list[BenchmarkTask]:
    return [t for t in BENCHMARK_TASKS if t.difficulty == difficulty]


def get_tasks_by_tag(tag: str) -> list[BenchmarkTask]:
    return [t for t in BENCHMARK_TASKS if tag in t.tags]
