# Archon — Autonomous Agent System

An autonomous agent built in Python with a **planner–executor–reflector** architecture, Protocol-based dependency injection, composable middleware chain, and a statistically rigorous evaluation harness for comparing LLM backends.

## Architecture

```
                  ┌──────────────┐
                  │  User Task   │
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │   Planner    │ ◄── Protocol: agent.protocols.Planner
                  └──────┬───────┘
                         │
           ┌─────────────▼─────────────┐
           │    Middleware Chain        │
           │  ┌──────────────────────┐ │
           │  │ TracingInterceptor   │ │
           │  │ TokenBudgetIntercept │ │  ← onion model: L→R / R→L
           │  │ RateLimitInterceptor │ │
           │  │ TelemetryInterceptor │ │
           │  └──────────────────────┘ │
           └─────────────┬─────────────┘
                         │
                  ┌──────▼───────┐
                  │   Executor   │ ◄── Protocol: agent.protocols.Executor
                  │  (per step)  │      Schema-validated tool calls
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │  Reflector   │ ◄── 2-phase: heuristic + LLM
                  │              │
                  │  CONTINUE ───┼──→ next step
                  │  RETRY ──────┼──→ re-execute with correction hint
                  │  REPLAN ─────┼──→ new plan from Planner
                  │  ABORT ──────┼──→ halt (FatalAgentError)
                  │  SKIP ───────┼──→ skip to next
                  └──────────────┘
```

## Design Decisions (Staff/Principal Level)

**Protocol-Based Abstractions** — All component contracts defined via `typing.Protocol` (structural subtyping). No component depends on a concrete implementation. Enables swapping LLM backends, testing with deterministic fakes, and adding tools without modifying the executor.

**Typed Exception Hierarchy** — Every exception carries structured context and maps to a failure category. `RetryableError` vs `FatalAgentError` is encoded in the type, not guessed by callers. `except RetryableError` catches all retryable subtypes in one branch.

**Composable Middleware** — Cross-cutting concerns (tracing, token budgets, rate limiting, telemetry) are interceptors in a chain, not code mixed into the orchestrator. Adding PII redaction or audit logging means writing one class, not modifying core logic. Onion execution model (before: L→R, after: R→L).

**Token Budget Tracking** — Cost control baked into the middleware layer. Tracks input/output tokens, computes running cost, and short-circuits execution when budget is exhausted.

**Deterministic Fakes Over Mocks** — Tests use `DeterministicFakeBackend` (implements `LLMBackend` protocol with pre-programmed responses) instead of `unittest.mock`. Tests verify actual behavior, not mock configuration. Fully reproducible, zero network calls.

**Statistically Rigorous Evaluation** — Bootstrap confidence intervals (non-parametric, assumption-free), Cohen's d effect size, Cliff's delta (ordinal, robust to non-normality), Mann-Whitney U test. LLM scores are NOT normally distributed — parametric tests would be wrong.

**Async Throughout** — All I/O-bound methods are `async`. Sync wrappers only exist in LLM adapters for wrapping third-party sync SDKs via `run_in_executor`.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Architecture demo (no API keys)
python main.py demo

# Run tests
pytest tests/ -v

# Run agent (requires API key)
export OPENAI_API_KEY="sk-..."
python main.py run "Find the GDP of Japan and calculate per capita income"

# Evaluation (mock tools, no keys)
python main.py eval --mock --trials 1
```

## Operations dashboard and RAG

Run the local dashboard (trace metrics, filters, and **RAG Studio** for ingest + questions):

```bash
python main.py dashboard --host 127.0.0.1 --port 8787 --traces-dir traces
```

- **Health check** (load balancers, k8s probes): `GET /api/health` — returns service status, `version`, and whether RAG bearer auth is enabled.
- **RAG auth probe** (UI uses this for the status badge): `GET /api/rag/auth-check` — optional `Authorization: Bearer <token>`.
- **Securing RAG** — set `ARCHON_DASHBOARD_TOKEN` on the server, then paste the same value in the dashboard’s **RAG API Access** field (stored in the browser as `archon_rag_api_token`).

Optional environment variables:

| Variable | Purpose |
|----------|---------|
| `ARCHON_DASHBOARD_TOKEN` | If set, all `/api/rag/*` requests must send `Authorization: Bearer <token>`. |
| `ARCHON_RAG_MAX_REQUEST_BYTES` | Max JSON body size for RAG POSTs (default `200000`). |
| `ARCHON_RAG_MAX_INGEST_CHARS` | Max characters per ingest `text` field (default `50000`). |
| `ARCHON_RAG_RATE_MAX` | Max RAG API calls per client IP per sliding window (default `120`). |
| `ARCHON_RAG_RATE_WINDOW_SEC` | Sliding window in seconds (default `60`). |
| `ARCHON_LOG_LEVEL` | Python log level for the dashboard process (default `INFO`). |
| `ARCHON_AUDIT_JSON` | If `1` (default), emit one JSON line per HTTP request to the `archon.audit` logger (no bodies or secrets). Set to `0` to disable. |

RAG session data is persisted under `<traces-dir>/rag_store/ingests.jsonl` (see `.gitignore`).

For **TLS, reverse proxy, HA, and remaining production gaps**, see [docs/PRODUCTION.md](docs/PRODUCTION.md).

## Project Structure

```
archon/
├── agent/
│   ├── protocols.py          # Protocol definitions (LLMBackend, ToolProvider, etc.)
│   ├── errors.py             # Typed exception hierarchy with failure categories
│   ├── state.py              # Pydantic models (Plan, Step, Trace, WorkingMemory)
│   ├── middleware.py          # Interceptor chain (tracing, budgets, telemetry)
│   ├── llm_backends.py       # LLM adapters (OpenAI, HuggingFace, Fake)
│   ├── async_orchestrator.py # Main loop with middleware + DI
│   ├── async_planner.py      # Task decomposition → structured JSON plans
│   ├── async_executor.py     # Step execution with schema validation
│   └── async_reflector.py    # 2-phase reflection (heuristic + LLM)
├── tools/
│   ├── registry.py           # Tool base class, registry, schema export
│   └── implementations.py    # Concrete tools (search, fetch, calc, etc.)
├── evaluation/
│   ├── benchmarks/tasks.py   # 9 benchmark tasks, 4 categories
│   ├── metrics.py            # Per-step + aggregate scoring
│   ├── statistics.py         # Bootstrap CI, Cohen's d, Mann-Whitney U
│   └── harness.py            # Multi-model evaluation runner
├── tests/
│   ├── conftest.py           # Shared fixtures + deterministic fakes
│   ├── test_tools.py         # Registry + tool implementation tests
│   ├── test_state_and_metrics.py  # State models + metrics tests
│   ├── test_middleware.py    # Middleware chain + interceptor tests
│   ├── test_errors_backends_stats.py  # Exceptions + backends + stats
│   └── test_async_components.py  # Async planner/executor/reflector
├── config/settings.py        # Centralized config with env-var support
├── pyproject.toml            # Python packaging + tool config
├── Makefile                  # Build automation
├── Dockerfile                # Container support
└── main.py                   # CLI entry point
```

## Test Coverage (115 tests)

| Module | Tests | What's Verified |
|--------|-------|-----------------|
| Tool Registry | 16 | Registration, schema validation, execution |
| State Models | 13 | Serialization, trace queries, working memory |
| Middleware | 14 | Token budgets, tracing, telemetry, chain composition |
| Exceptions | 10 | Type hierarchy, catch patterns, structured context |
| LLM Backends | 5 | Deterministic fakes, response normalization |
| Statistics | 14 | Bootstrap CI, Cohen's d, Cliff's delta, Mann-Whitney |
| Async Planner | 7 | Plan creation, replanning, parsing, edge cases |
| Async Executor | 5 | Step execution, memory recording, error handling |
| Async Reflector | 6 | Heuristic fast-path, LLM fallback, verdicts |
| Integration | 1 | Planner → Executor end-to-end chain |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Tool-call accuracy | % steps with correct tool selected |
| Schema adherence | % tool calls with valid JSON arguments |
| Error recovery rate | % failed steps that recovered via retry/replan |
| Step efficiency | Ratio of expected steps to actual |
| Final answer score | % expected keywords in answer |

## Statistical Methods

| Method | Purpose | Why Not Parametric? |
|--------|---------|---------------------|
| Bootstrap CI | Confidence intervals | Non-parametric, works with n≥10 |
| Cohen's d | Effect size magnitude | Standard, but supplemented by Cliff's |
| Cliff's delta | Ordinal effect size | Robust to non-normality |
| Mann-Whitney U | Significance test | LLM scores are NOT normally distributed |
