"""
Centralized configuration for the agent system.
Supports env-var overrides for all settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class LLMProvider(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for an LLM backend."""
    provider: LLMProvider = LLMProvider.OPENAI
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 4096
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 120

    def resolve_api_key(self) -> Optional[str]:
        if self.api_key:
            return self.api_key
        if self.provider == LLMProvider.OPENAI:
            return os.getenv("OPENAI_API_KEY")
        if self.provider == LLMProvider.HUGGINGFACE:
            return os.getenv("HF_API_TOKEN")
        return None


@dataclass(frozen=True)
class PlannerConfig:
    """Planner-specific tuning."""
    max_plan_steps: int = 10
    allow_replanning: bool = True
    replan_on_failure: bool = True
    structured_output: bool = True  # Force JSON schema output


@dataclass(frozen=True)
class ExecutorConfig:
    """Executor-specific tuning."""
    max_retries_per_step: int = 3
    step_timeout_seconds: int = 60
    parallel_tool_calls: bool = False


@dataclass(frozen=True)
class ReflectorConfig:
    """Reflector-specific tuning."""
    enabled: bool = True
    reflection_depth: str = "standard"  # "shallow" | "standard" | "deep"
    fail_fast_on_critical: bool = True
    max_consecutive_failures: int = 3


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation harness configuration."""
    benchmark_dir: str = "evaluation/benchmarks"
    output_dir: str = "evaluation/results"
    models_to_compare: list[str] = field(
        default_factory=lambda: [
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ]
    )
    num_trials: int = 3
    compute_confidence_intervals: bool = True


@dataclass(frozen=True)
class AgentConfig:
    """Top-level agent configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    reflector: ReflectorConfig = field(default_factory=ReflectorConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    verbose: bool = True
    trace_dir: Optional[str] = "traces"

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Build config from environment variables with sensible defaults."""
        return cls(
            llm=LLMConfig(
                provider=LLMProvider(os.getenv("AGENT_LLM_PROVIDER", "openai")),
                model_name=os.getenv("AGENT_LLM_MODEL", "gpt-4o-mini"),
                temperature=float(os.getenv("AGENT_LLM_TEMP", "0.0")),
            ),
            verbose=os.getenv("AGENT_VERBOSE", "true").lower() == "true",
        )
