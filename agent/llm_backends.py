"""
LLM Backend Adapters.

Concrete implementations of the LLMBackend protocol.
Each adapter wraps a specific provider's SDK and normalizes
the interface for the agent system.

The adapter is responsible for:
  - Connection management and retry logic
  - Token counting and cost tracking
  - Response normalization into LLMResponse
  - Provider-specific error mapping to agent exceptions

Adding a new provider:
  1. Implement LLMBackend protocol
  2. Register in the LLMFactory
  3. Done — no changes to planner/executor/reflector needed
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import structlog

from agent.errors import LLMConnectionError, LLMOutputParseError, LLMRateLimitError
from agent.middleware import TokenBudget

logger = structlog.get_logger(__name__)


# ── Normalized Response ──────────────────────────────────────────────

@dataclass(frozen=True)
class CompletionResponse:
    """Concrete LLMResponse implementation."""
    content: str
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: float = 0.0
    raw_response: Any = None


# ── OpenAI / OpenAI-Compatible Adapter ───────────────────────────────

class OpenAIAdapter:
    """
    Adapter for OpenAI and OpenAI-compatible APIs (Azure, Together, Anyscale).

    Uses the openai SDK directly (no LangChain dependency in the adapter).
    Async-native via openai.AsyncOpenAI.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
        max_retries: int = 3,
        token_budget: TokenBudget | None = None,
    ):
        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._budget = token_budget
        self._client: Any = None

    @property
    def model_id(self) -> str:
        return self._model_name

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                kwargs: dict[str, Any] = {"timeout": self._timeout}
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                self._client = AsyncOpenAI(**kwargs)
            except ImportError:
                raise ImportError(
                    "openai package required for OpenAIAdapter. "
                    "Install with: pip install openai"
                )
        return self._client

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        client = self._ensure_client()
        start = time.perf_counter()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = await client.chat.completions.create(**kwargs)
                latency = (time.perf_counter() - start) * 1000

                usage = resp.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0

                # Track budget
                if self._budget:
                    self._budget.record_usage(input_tokens, output_tokens)

                return CompletionResponse(
                    content=resp.choices[0].message.content or "",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=resp.model or self._model_name,
                    latency_ms=round(latency, 2),
                    raw_response=resp,
                )
            except Exception as exc:
                last_error = exc
                exc_name = type(exc).__name__
                if "RateLimitError" in exc_name:
                    wait = 2 ** attempt
                    logger.warning(
                        "llm_rate_limited",
                        attempt=attempt + 1,
                        wait_seconds=wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                elif "APIConnectionError" in exc_name or "Timeout" in exc_name:
                    wait = 2 ** attempt
                    logger.warning(
                        "llm_connection_error",
                        attempt=attempt + 1,
                        error=str(exc),
                    )
                    await asyncio.sleep(wait)
                    continue
                else:
                    raise LLMConnectionError(
                        provider="openai",
                        model=self._model_name,
                        cause=str(exc),
                    ) from exc

        raise LLMConnectionError(
            provider="openai",
            model=self._model_name,
            cause=f"Exhausted {self._max_retries} retries. Last error: {last_error}",
        )


# ── HuggingFace Inference API Adapter ────────────────────────────────

class HuggingFaceAdapter:
    """
    Adapter for HuggingFace Inference API.
    Wraps huggingface_hub InferenceClient for async usage.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        api_token: str | None = None,
        token_budget: TokenBudget | None = None,
    ):
        self._model_name = model_name
        self._api_token = api_token
        self._budget = token_budget

    @property
    def model_id(self) -> str:
        return self._model_name

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError(
                "huggingface_hub required. Install with: pip install huggingface_hub"
            )

        client = InferenceClient(
            model=self._model_name,
            token=self._api_token,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        start = time.perf_counter()

        # Run sync client in executor to not block event loop
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                partial(
                    client.chat_completion,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=max(temperature, 0.01),  # HF requires > 0
                ),
            )
        except Exception as exc:
            raise LLMConnectionError(
                provider="huggingface",
                model=self._model_name,
                cause=str(exc),
            ) from exc

        latency = (time.perf_counter() - start) * 1000
        content = response.choices[0].message.content or ""

        # Estimate tokens (HF doesn't always return usage)
        input_tokens = len(system_prompt + user_message) // 4
        output_tokens = len(content) // 4

        if self._budget:
            self._budget.record_usage(input_tokens, output_tokens)

        return CompletionResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._model_name,
            latency_ms=round(latency, 2),
        )


# ── LangChain Bridge Adapter ────────────────────────────────────────

class LangChainAdapter:
    """
    Wraps any existing LangChain LLM/ChatModel to satisfy the
    LLMBackend protocol. Useful for gradual migration.
    """

    def __init__(
        self,
        langchain_llm: Any,
        model_name: str = "langchain-wrapped",
        token_budget: TokenBudget | None = None,
    ):
        self._llm = langchain_llm
        self._model_name = model_name
        self._budget = token_budget

    @property
    def model_id(self) -> str:
        return getattr(self._llm, "model_name", self._model_name)

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        start = time.perf_counter()

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                partial(self._llm.invoke, messages),
            )
        except Exception as exc:
            raise LLMConnectionError(
                provider="langchain",
                model=self._model_name,
                cause=str(exc),
            ) from exc

        latency = (time.perf_counter() - start) * 1000
        content = response.content if hasattr(response, "content") else str(response)

        # Extract usage if available
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "input_tokens", len(system_prompt) // 4)
        output_tokens = getattr(usage, "output_tokens", len(content) // 4)

        if self._budget:
            self._budget.record_usage(input_tokens, output_tokens)

        return CompletionResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model_id,
            latency_ms=round(latency, 2),
        )


# ── Deterministic Fake (for testing) ─────────────────────────────────

class DeterministicFakeBackend:
    """
    Deterministic LLM fake for unit tests.
    Returns pre-programmed responses keyed by message content.
    No network calls, no randomness, fully reproducible.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default_response: str = '{"tool": "web_search", "arguments": {"query": "test"}}',
    ):
        self._responses = responses or {}
        self._default = default_response
        self.call_log: list[dict[str, str]] = []

    @property
    def model_id(self) -> str:
        return "deterministic-fake"

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        self.call_log.append({
            "system": system_prompt[:200],
            "user": user_message[:200],
        })

        # Match on user message substring
        content = self._default
        for key, response in self._responses.items():
            if key.lower() in user_message.lower():
                content = response
                break

        return CompletionResponse(
            content=content,
            input_tokens=len(system_prompt + user_message) // 4,
            output_tokens=len(content) // 4,
            model="deterministic-fake",
        )


# ── Factory ──────────────────────────────────────────────────────────

class LLMFactory:
    """
    Factory for constructing LLM backends from configuration.
    Centralizes provider-specific setup.
    """

    @staticmethod
    def create(
        provider: str,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        token_budget: TokenBudget | None = None,
        **kwargs: Any,
    ) -> OpenAIAdapter | HuggingFaceAdapter:
        """
        Create an LLM backend by provider name.

        Args:
            provider: "openai", "huggingface", "ollama", "langchain"
            model_name: Provider-specific model identifier
            api_key: API key (falls back to env vars)
            base_url: Custom API endpoint
            token_budget: Shared budget for cost tracking
        """
        if provider == "openai":
            return OpenAIAdapter(
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                token_budget=token_budget,
                **kwargs,
            )
        elif provider == "huggingface":
            return HuggingFaceAdapter(
                model_name=model_name,
                api_token=api_key,
                token_budget=token_budget,
            )
        elif provider == "ollama":
            return OpenAIAdapter(
                model_name=model_name,
                base_url=base_url or "http://localhost:11434/v1",
                api_key="ollama",  # Ollama doesn't need a real key
                token_budget=token_budget,
            )
        else:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Supported: openai, huggingface, ollama"
            )
