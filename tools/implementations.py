"""
Concrete tool implementations.

Each tool is a self-contained class with:
  - Pydantic args schema for validation
  - execute() method with error handling
  - Schema export for LLM prompt injection
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Optional
from urllib.parse import quote_plus

import httpx
import structlog
from pydantic import BaseModel, Field

from tools.registry import BaseTool

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 1. Web Search
# ═══════════════════════════════════════════════════════════════════

class WebSearchArgs(BaseModel):
    query: str = Field(description="Search query string")
    max_results: int = Field(
        default=5, ge=1, le=10,
        description="Maximum number of results to return",
    )


class WebSearchTool(BaseTool):
    """
    Search the web using DuckDuckGo and return structured results.
    Falls back to a mock when running without network.
    """
    name = "web_search"
    description = (
        "Search the web for current information. Returns a list of results "
        "with title, URL, and snippet. Use for fact-finding, research, and "
        "answering questions that require up-to-date information."
    )
    args_schema = WebSearchArgs

    def __init__(self, use_mock: bool = False):
        self._use_mock = use_mock

    def execute(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        if self._use_mock:
            return self._mock_search(query, max_results)
        return self._live_search(query, max_results)

    def _live_search(self, query: str, max_results: int) -> list[dict[str, str]]:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=max_results))
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in raw
            ]
        except ImportError:
            logger.warning("duckduckgo_search not installed, using mock")
            return self._mock_search(query, max_results)
        except Exception as exc:
            logger.error("web_search_failed", error=str(exc))
            raise RuntimeError(f"Web search failed: {exc}") from exc

    def _mock_search(self, query: str, max_results: int) -> list[dict[str, str]]:
        return [
            {
                "title": f"Result {i+1} for: {query}",
                "url": f"https://example.com/search?q={quote_plus(query)}&r={i+1}",
                "snippet": f"This is a mock search result #{i+1} for the query '{query}'.",
            }
            for i in range(max_results)
        ]


# ═══════════════════════════════════════════════════════════════════
# 2. Web Page Reader
# ═══════════════════════════════════════════════════════════════════

class WebFetchArgs(BaseModel):
    url: str = Field(description="Full URL of the page to fetch")
    extract_text: bool = Field(
        default=True,
        description="If true, extract visible text. If false, return raw HTML.",
    )
    max_length: int = Field(
        default=4000, ge=100, le=50000,
        description="Maximum characters of content to return",
    )


class WebFetchTool(BaseTool):
    """
    Fetch and extract content from a web page.
    Strips HTML to return clean text for the agent.
    """
    name = "web_fetch"
    description = (
        "Fetch the content of a web page. Returns the page text or HTML. "
        "Use after web_search to read full article content, or to navigate "
        "to a known URL and extract information."
    )
    args_schema = WebFetchArgs

    def __init__(self, use_mock: bool = False):
        self._use_mock = use_mock

    def execute(
        self, url: str, extract_text: bool = True, max_length: int = 4000
    ) -> str:
        if self._use_mock:
            return self._mock_fetch(url, max_length)
        return self._live_fetch(url, extract_text, max_length)

    def _live_fetch(
        self, url: str, extract_text: bool, max_length: int
    ) -> str:
        try:
            resp = httpx.get(
                url, follow_redirects=True, timeout=30,
                headers={"User-Agent": "Archon/1.0"},
            )
            resp.raise_for_status()
            if not extract_text:
                return resp.text[:max_length]
            return self._extract_text(resp.text)[:max_length]
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc

    def _extract_text(self, html: str) -> str:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            # Collapse whitespace
            return re.sub(r"\n{3,}", "\n\n", text)
        except ImportError:
            # Crude fallback: strip tags with regex
            text = re.sub(r"<[^>]+>", " ", html)
            return re.sub(r"\s+", " ", text).strip()

    def _mock_fetch(self, url: str, max_length: int) -> str:
        return (
            f"[Mock page content from {url}]\n\n"
            "This is simulated page content for testing. "
            "In production, this would contain the actual extracted text "
            "from the web page."
        )[:max_length]


# ═══════════════════════════════════════════════════════════════════
# 3. Calculator
# ═══════════════════════════════════════════════════════════════════

class CalculatorArgs(BaseModel):
    expression: str = Field(
        description="Mathematical expression to evaluate (e.g., '2 + 3 * 4', 'sqrt(144)', 'log(100, 10)')"
    )


class CalculatorTool(BaseTool):
    """
    Evaluate mathematical expressions safely.
    Supports standard arithmetic, exponents, trig, log, sqrt.
    """
    name = "calculator"
    description = (
        "Evaluate a mathematical expression and return the numeric result. "
        "Supports arithmetic (+, -, *, /, **), functions (sqrt, log, sin, cos, tan, abs), "
        "and constants (pi, e). Use for any computation the agent needs."
    )
    args_schema = CalculatorArgs

    # Whitelist of safe names
    _SAFE_GLOBALS: dict[str, Any] = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "log": math.log, "log2": math.log2,
        "log10": math.log10, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "pi": math.pi, "e": math.e,
        "pow": pow, "ceil": math.ceil, "floor": math.floor,
        "__builtins__": {},  # Block all builtins
    }

    def execute(self, expression: str) -> float | int:
        # Sanitize: only allow digits, operators, parens, dots, spaces, function names
        sanitized = expression.strip()
        allowed_pattern = re.compile(
            r"^[0-9+\-*/().,%\s]+"
            r"|sqrt|log|log2|log10|sin|cos|tan|pi|abs|round|min|max|pow|ceil|floor|e"
        )
        # Basic injection guard
        forbidden = {"import", "exec", "eval", "open", "__", "os", "sys"}
        tokens = set(re.findall(r"[a-zA-Z_]+", sanitized))
        if tokens & forbidden:
            raise ValueError(f"Forbidden tokens in expression: {tokens & forbidden}")

        try:
            result = eval(sanitized, self._SAFE_GLOBALS, {})
            if isinstance(result, (int, float)):
                return result
            raise ValueError(f"Expression did not return a number: {type(result)}")
        except Exception as exc:
            raise ValueError(f"Cannot evaluate '{expression}': {exc}") from exc


# ═══════════════════════════════════════════════════════════════════
# 4. Text Analysis
# ═══════════════════════════════════════════════════════════════════

class TextAnalysisArgs(BaseModel):
    text: str = Field(description="Text to analyze")
    operation: str = Field(
        description="Analysis operation: 'summarize', 'extract_entities', 'sentiment', 'key_facts'"
    )


class TextAnalysisTool(BaseTool):
    """
    Perform text analysis operations: summarization, entity
    extraction, sentiment analysis, key fact extraction.
    Uses heuristic methods (no LLM call) for deterministic eval.
    """
    name = "text_analysis"
    description = (
        "Analyze text content. Operations: 'summarize' (compress text), "
        "'extract_entities' (find names, places, orgs), 'sentiment' (positive/negative/neutral), "
        "'key_facts' (extract factual statements). "
        "Use to process text retrieved from web pages or documents."
    )
    args_schema = TextAnalysisArgs

    def execute(self, text: str, operation: str) -> dict[str, Any]:
        ops = {
            "summarize": self._summarize,
            "extract_entities": self._extract_entities,
            "sentiment": self._sentiment,
            "key_facts": self._key_facts,
        }
        if operation not in ops:
            raise ValueError(
                f"Unknown operation '{operation}'. Choose from: {list(ops.keys())}"
            )
        return ops[operation](text)

    def _summarize(self, text: str) -> dict[str, Any]:
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        # Heuristic: pick first, middle, last sentence
        if len(sentences) <= 3:
            summary = ". ".join(sentences) + "."
        else:
            mid = len(sentences) // 2
            summary = ". ".join([sentences[0], sentences[mid], sentences[-1]]) + "."
        return {
            "summary": summary,
            "original_length": len(text),
            "sentence_count": len(sentences),
        }

    def _extract_entities(self, text: str) -> dict[str, Any]:
        # Heuristic NER: capitalized multi-word sequences
        pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
        candidates = re.findall(pattern, text)
        # Deduplicate preserving order
        seen = set()
        entities = []
        for c in candidates:
            if c not in seen and len(c) > 2:
                seen.add(c)
                entities.append(c)
        return {"entities": entities[:20]}

    def _sentiment(self, text: str) -> dict[str, Any]:
        positive = {"good", "great", "excellent", "amazing", "best", "love",
                     "wonderful", "fantastic", "happy", "success", "positive"}
        negative = {"bad", "worst", "terrible", "awful", "hate", "fail",
                     "horrible", "poor", "negative", "sad", "wrong"}
        words = set(text.lower().split())
        pos_count = len(words & positive)
        neg_count = len(words & negative)
        if pos_count > neg_count:
            label = "positive"
        elif neg_count > pos_count:
            label = "negative"
        else:
            label = "neutral"
        return {
            "sentiment": label,
            "positive_signals": pos_count,
            "negative_signals": neg_count,
        }

    def _key_facts(self, text: str) -> dict[str, Any]:
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        # Heuristic: sentences with numbers or dates are likely factual
        facts = [
            s for s in sentences
            if re.search(r"\d", s) or any(
                kw in s.lower()
                for kw in ["is", "was", "are", "were", "has", "have", "percent", "million", "billion"]
            )
        ]
        return {"key_facts": facts[:10]}


# ═══════════════════════════════════════════════════════════════════
# 5. File Operations (for multi-step task support)
# ═══════════════════════════════════════════════════════════════════

class FileWriteArgs(BaseModel):
    filename: str = Field(description="Name of the file to write")
    content: str = Field(description="Content to write to the file")


class FileWriteTool(BaseTool):
    """Write content to a file. Used for saving intermediate results."""
    name = "file_write"
    description = (
        "Write content to a file. Use to save intermediate results, "
        "create reports, or store data for later steps."
    )
    args_schema = FileWriteArgs

    def __init__(self, output_dir: str = "/tmp/agent_outputs"):
        import os
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def execute(self, filename: str, content: str) -> dict[str, str]:
        import os
        # Sanitize filename
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
        path = os.path.join(self._output_dir, safe_name)
        with open(path, "w") as f:
            f.write(content)
        return {"status": "written", "path": path, "bytes": len(content)}


class FileReadArgs(BaseModel):
    filename: str = Field(description="Name of the file to read")


class FileReadTool(BaseTool):
    """Read content from a previously written file."""
    name = "file_read"
    description = (
        "Read content from a file. Use to retrieve previously saved "
        "results or data from earlier steps."
    )
    args_schema = FileReadArgs

    def __init__(self, output_dir: str = "/tmp/agent_outputs"):
        self._output_dir = output_dir

    def execute(self, filename: str) -> str:
        import os
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
        path = os.path.join(self._output_dir, safe_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path) as f:
            return f.read()


# ═══════════════════════════════════════════════════════════════════
# Factory: Build a default registry with all tools
# ═══════════════════════════════════════════════════════════════════

def build_default_registry(use_mock: bool = False, include_rag: bool = True) -> "ToolRegistry":
    """Create a registry pre-loaded with all standard tools."""
    from tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(WebSearchTool(use_mock=use_mock))
    registry.register(WebFetchTool(use_mock=use_mock))
    registry.register(CalculatorTool())
    registry.register(TextAnalysisTool())
    registry.register(FileWriteTool())
    registry.register(FileReadTool())

    if include_rag:
        from tools.rag_pipeline import register_rag_tools
        register_rag_tools(registry)

    return registry
