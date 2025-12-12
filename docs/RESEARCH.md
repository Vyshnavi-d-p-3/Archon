# Archon: research methodology and reporting standards

This document supports **reproducible** experiments and **paper-quality** reporting. It states what the system measures, which statistical tools apply, and what is *not* guaranteed (so limitations are explicit rather than implied).

## 1. System summary (for a methods section)

Archon is a **planner–executor–reflector** agent with **schema-validated tool calls**, a **composable middleware** layer (tracing, token budgets, rate limits), and a **typed exception → failure category** mapping. The **async** path (`AsyncAgentOrchestrator`) is the primary execution engine for evaluation. Each run materializes a structured **`AgentTrace`** (JSON) that records plans, tool calls, reflections (including a failure-mode taxonomy), and timing.

**Trace interchange:** each trace may include `archon_version` and `trace_schema_version` (see `config.version` / `evaluation/reproducibility.py`). Bumping `TRACE_SCHEMA_VERSION` in `config/version.py` signals incompatible JSON changes for external tooling.

## 2. Default benchmark scope

- **Core suite** — `evaluation/benchmarks/tasks.py` (`BENCHMARK_TASKS`). This is the **default** baseline for `python main.py eval` when you do not supply a custom task list.
- **Extended suite** — `evaluation/benchmarks/extended_tasks.py`, loaded only with `--include-extended-benchmarks` (or by passing the task list in code). These tasks are **longer** and may stress tool chains; use them for ablations, not for minimal baseline tables unless explicitly justified.

## 3. Metric definitions (ground truth alignment)

`evaluation/metrics.MetricsScorer` scores a trace against a benchmark’s **expected step list**. Report these definitions verbatim when writing a paper’s metrics subsection:

| Metric | Definition in code |
|--------|--------------------|
| **Tool-call accuracy** | Among matched steps, fraction where the executed tool name matched an expected step for that point in the plan (greedy / flexible-order match; see `MetricsScorer._score_steps`). |
| **Schema adherence** | Among steps with a non-`unknown` tool, fraction with `schema_valid` on the tool call. |
| **Error recovery** | Among steps with `retries > 0`, fraction that ended `COMPLETED` (success after retry / reflect loop). *If no such steps, the rate is over an empty set—interpret with care at small n.* |
| **Step efficiency** | `min(1, expected_steps / actual_executed_steps)` (capped; rewards not overshooting; see `TaskMetrics.step_efficiency`). |
| **Final answer score** | If the benchmark lists `expected_final_answer_contains`, the fraction of those keywords present in the final answer (case-insensitive). |
| **Failure taxonomy** | Counts of `reflection.failure_category` on steps (and failure categories in exceptions where applicable), aggregated in `ModelSummary.failure_taxonomy`. |

**Safety categories** in `agent/state.FailureCategory` are first-class: policy, unsafe output, prompt injection, PII/secret risk, ungrounded claim—**operational** categories are separate. Reflectors are prompted to prefer safety categories when in doubt; reported rates depend on the reflector LLM, not on an external human label set.

## 4. Statistical methods

Implemented in `evaluation/statistics.py` and summarized in the main [README](../README.md):

- **Bootstrap** confidence intervals (non-parametric; default seed wired through `set_global_seeds` / `ARCHON_EVAL_SEED`).
- **Cohen’s d** and **Cliff’s delta** (effect size; robust to non-normality).
- **Mann–Whitney U** (non-parametric two-sample comparison).

**Assumption:** LLM-derived scores are **not** Gaussian; avoid parametric t-tests on raw per-task means unless you transform or justify normality. Prefer bootstrap CIs and non-parametric tests for headline comparisons.

## 5. Reproducibility and what “seed” does

- **`ARCHON_EVAL_SEED` / `--seed`** — Sets **Python `random`** and **NumPy** seeds at the start of a harness run so bootstrap resampling and any numpy-backed stats are **repeatable** for a fixed codebase version.
- **Remote LLM APIs** (OpenAI, Hugging Face Inference, etc.) are **not** bit-reproducible at temperature 0 in the general case; the run manifest’s `non_determinism_note` records this. For **deterministic** integration tests, use **mock tools** and **`DeterministicFakeBackend`**.
- **Artifacts:** each eval run writes `evaluation/results/run_manifest.json` (or under `EvalConfig.output_dir`) with `archon_version`, `trace_schema_version`, `eval_seed`, `task_ids`, `models`, and a **config fingerprint** (SHA-256 of canonicalized knobs). `results.json` may embed the same data under the key `__archon_run__`.

## 6. Honest limitations (recommended “Limitations” paragraph)

- **Ground truth is tool-level**: semantic equivalence (two valid tool paths) is only partially captured by flexible order and keyword checks.
- **Reflector-supervised safety labels** are not a substitute for human red-teaming.
- **Small trial counts** per cell produce wide bootstrap intervals; do not over-claim on `n=1` per task.
- **Dashboard** aggregates are **observable rates** (step completion, safety-tagged step share) from stored traces, not an opaque composite index.

## 7. Suggested paper outline (empirical track)

1. System description (architecture; cite this repo + version).
2. Benchmark protocol (core vs extended; number of trials; mock vs live tools).
3. Metrics (use §3).
4. Statistics (use §4; report effect sizes, not only p-values).
5. Ablations (e.g. reflector off, mock vs live tools) as separate **pre-registered** runs with distinct `run_manifest` fingerprints.
6. Ethics / safety: failure taxonomy + operational deployment notes (`docs/PRODUCTION.md`).

## 8. Citation

Use `CITATION.cff` at the repository root, or cite the package name **archon** and the version from `config.version.package_version()` (also stored on traces and run manifests).
