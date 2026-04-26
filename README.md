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
# Install (tests / demo: dev only; agent + eval also need LangChain)
pip install -e ".[dev]"
# For `python main.py run` and `python main.py eval` add extras, e.g.:
# pip install -e ".[langchain,eval]"   # or pip install -e ".[all]"

# Architecture demo (no API keys)
python main.py demo

# Run tests
pytest tests/ -v

# Run agent (requires API key)
export OPENAI_API_KEY="sk-..."
python main.py run "Find the GDP of Japan and calculate per capita income"

# Evaluation (mock tools, no keys)
python main.py eval --mock --trials 1
# Reproducible library RNG (also set ARCHON_EVAL_SEED=42)
python main.py eval --mock --trials 1 --seed 42
```

## Research, reproducibility, and citation

- **[docs/BRANCHES.md](docs/BRANCHES.md)** — `main`, `develop`, `production`, and `feature/*` workflow, plus how GitHub’s activity graph / sparkline relates to the default branch and commit email.
- **[docs/RESEARCH.md](docs/RESEARCH.md)** — definitions of metrics, statistical methods, limitations, and what is (and is not) controlled by the evaluation seed. Use it as a template for a paper **Methods** / **Reproducibility** section.
- **Run manifest** — each `eval` run writes `evaluation/results/run_manifest.json` (archon version, `trace_schema_version`, `eval_seed`, task IDs, model list, SHA-256 **config fingerprint**). Aggregated `results.json` may embed the same under `__archon_run__`.
- **Traces** — completed runs attach **`archon_version`** and **`trace_schema_version`** to `AgentTrace` JSON for interchange and long-term analysis.
- **Citation** — see [CITATION.cff](CITATION.cff); add a `repository-code` when you host the project publicly.

| Variable / CLI | Purpose |
|----------------|---------|
| `ARCHON_EVAL_SEED` | Default RNG seed for Python/NumPy in the eval harness (default `42`). |
| `ARCHON_EVAL_MODELS` | Comma-separated Hugging Face model ids for `main.py eval` (overrides the default list in `config.settings.EvalConfig`). |
| `--seed` | Per-invocation override for the eval harness. |
| `--include-extended-benchmarks` | Use core + [extended](evaluation/benchmarks/extended_tasks.py) tasks (longer / heavier than the default baseline). |

**Note:** Library RNG is reproducible for a given seed; **remote LLM APIs are not** bit-reproducible in general, even at temperature 0. Use mock tools + `DeterministicFakeBackend` for deterministic integration tests.

## Operations dashboard and RAG

Run the local dashboard (trace metrics, filters, and **RAG Studio** for ingest + questions):

```bash
python main.py dashboard --host 127.0.0.1 --port 8787 --traces-dir traces
```

- **Discovery**: `GET /api/info` — version, resolved `traces_dir`, and a short list of HTTP entry points.
- **Liveness** (process up): `GET /api/health` — includes `checks` (filesystem readiness) and rate-limit metadata.
- **Readiness** (can write traces / RAG store): `GET /api/ready` — returns **200** when ready, **503** when the traces directory or `rag_store` is not usable (use for Kubernetes readiness probes).
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
| `ARCHON_CORS_ORIGIN` | If set (e.g. `https://app.example.com` or `*`), add CORS headers and handle `OPTIONS` preflight for `/api/*`. Prefer a specific origin over `*`. |
| `ARCHON_CORS_MAX_AGE` | Preflight cache seconds (default `86400`). |
| `ARCHON_DASHBOARD_METRICS_MAX` | Max **newest** `*.json` files under the traces directory used to compute `summary` KPIs (default `10000`, max `100000`). The `?limit=` query still caps the `traces` list only. If more files exist on disk than this cap, `meta.metrics_omit_older` is true. |
| `ARCHON_EVAL_SEED` | See [Research, reproducibility, and citation](#research-reproducibility-and-citation). |

RAG session data is persisted under `<traces-dir>/rag_store/ingests.jsonl` (see `.gitignore`). The dashboard handles **SIGTERM** for graceful shutdown in container environments.

`GET /api/dashboard?limit=500` returns the **newest** `traces` for the table (capped by `limit`, max `2000`), a `summary` whose KPIs are computed from the **newest** `ARCHON_DASHBOARD_METRICS_MAX` files (unless there are fewer on disk), and `meta` (for example `version`, `traces_on_disk`, `traces_in_response`, `metrics_files_read`, `metrics_omit_older`, `server_time`).

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
│   ├── benchmarks/          # Core + optional extended tasks; load via load_benchmark_tasks()
│   ├── metrics.py            # Per-step + aggregate scoring
│   ├── statistics.py         # Bootstrap CI, Cohen's d, Mann-Whitney U
│   ├── reproducibility.py  # Run manifests, eval seed, trace schema version
│   └── harness.py            # Multi-model evaluation runner
├── tests/                    # See pytest (180+); repro + dashboard + RAG + stats
├── config/
│   ├── settings.py           # Centralized config with env-var support
│   └── version.py            # package_version(), TRACE_SCHEMA_VERSION
├── docs/
│   ├── RESEARCH.md           # Paper-style methodology and limitations
│   └── PRODUCTION.md         # Ops / deployment
├── CITATION.cff              # Software citation metadata
├── pyproject.toml
├── Makefile
├── Dockerfile
└── main.py
```

## Test coverage

Run `pytest tests/ -q` — the suite is **180+** tests (registry, state, metrics, config, async agent, statistics, RAG, dashboard, reproducibility). Prefer **deterministic fakes** over mocks; integration tests with live APIs are opt-in.

## CI/CD

- **CI** — [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs tests on pushes/PRs to `main`, `develop`, and `production`.
- **CD** — [`.github/workflows/cd.yml`](.github/workflows/cd.yml) runs on pushes to `production`:
  - builds and pushes Docker image tags to GHCR: `ghcr.io/<owner>/<repo>:production` and `:sha-<commit>`
  - optionally triggers Render deployment if repository secret `RENDER_DEPLOY_HOOK_URL` is set

## Deploy to Render (2 minutes)

1. In Render, create a **Web Service** from this repository with **Docker** environment.
   - Tip: Render can import [`render.yaml`](render.yaml) to prefill most settings.
2. Set start command:
   - `python main.py dashboard --host 0.0.0.0 --port $PORT --traces-dir /tmp/traces`
3. Set health check path:
   - `/api/health`
4. In GitHub repo secrets, add:
   - `RENDER_DEPLOY_HOOK_URL` = your Render Deploy Hook URL.
5. Push or merge to `production`:
   - GitHub Actions CD builds/pushes the image and triggers Render deploy.

Import Blueprint in Render (click path):

1. Render Dashboard → **New** → **Blueprint**
2. Select this GitHub repository (`Archon`)
3. Confirm Render detected `render.yaml`
4. Review service settings (`archon-dashboard`, free)
5. Click **Apply**
6. Open created service → **Settings** → copy **Deploy Hook**
7. GitHub → repo **Settings** → **Secrets and variables** → **Actions** → add `RENDER_DEPLOY_HOOK_URL`
8. Push to `production` branch and watch GitHub Actions `CD` + Render deploy logs

Recommended Render environment variables:

- `ARCHON_LOG_LEVEL=INFO`
- `ARCHON_AUDIT_JSON=1`
- `ARCHON_RAG_RATE_MAX=120`
- `ARCHON_RAG_RATE_WINDOW_SEC=60`
- `ARCHON_RAG_MAX_REQUEST_BYTES=200000`
- `ARCHON_RAG_MAX_INGEST_CHARS=50000`
- `ARCHON_DASHBOARD_METRICS_MAX=10000`

Optional hardening:

- `ARCHON_DASHBOARD_TOKEN=<long-random-secret>` (protect `/api/rag/*`)
- `ARCHON_CORS_ORIGIN=https://your-frontend-domain.com` (if frontend is on another origin)

Free-tier note: `/tmp/traces` is ephemeral; trace files reset on restarts/redeploys.

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
