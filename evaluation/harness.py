"""
Evaluation Harness: Runs benchmark tasks across multiple models
and produces comparison reports with per-step traces.

This is the main entry point for the evaluation pipeline:
1. Load benchmark tasks
2. For each model × task × trial:
   a. Initialize agent with the model
   b. Run the task
   c. Score the trace against ground truth
3. Aggregate metrics per model
4. Generate comparison report with failure-mode taxonomy
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from tabulate import tabulate

from agent.orchestrator import AgentOrchestrator
from agent.state import AgentTrace
from config.settings import AgentConfig, LLMConfig, LLMProvider
from evaluation.benchmarks import load_benchmark_tasks
from evaluation.benchmarks.tasks import BenchmarkTask
from evaluation.metrics import (
    MetricsScorer,
    ModelSummary,
    TaskMetrics,
    aggregate_metrics,
)
from evaluation.reproducibility import (
    RunManifest,
    env_eval_seed,
    make_eval_manifest,
    set_global_seeds,
    write_run_manifest,
)
from tools.implementations import build_default_registry

logger = structlog.get_logger(__name__)


# ── LLM Factory ──────────────────────────────────────────────────────

def _build_llm(llm_config: LLMConfig) -> Any:
    """
    Construct a LangChain LLM from config.
    Supports OpenAI, HuggingFace, and Ollama backends.
    """
    if llm_config.provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=llm_config.model_name,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            api_key=llm_config.resolve_api_key(),
            timeout=llm_config.timeout,
        )

    elif llm_config.provider == LLMProvider.HUGGINGFACE:
        from langchain_huggingface import HuggingFaceEndpoint
        return HuggingFaceEndpoint(
            repo_id=llm_config.model_name,
            temperature=llm_config.temperature,
            max_new_tokens=llm_config.max_tokens,
            huggingfacehub_api_token=llm_config.resolve_api_key(),
        )

    elif llm_config.provider == LLMProvider.OLLAMA:
        from langchain_community.llms import Ollama
        return Ollama(
            model=llm_config.model_name,
            temperature=llm_config.temperature,
            base_url=llm_config.base_url or "http://localhost:11434",
        )

    raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")


# ── Evaluation Runner ────────────────────────────────────────────────

class EvaluationHarness:
    """
    Runs benchmark evaluations across models and produces
    structured comparison reports.
    """

    def __init__(self, config: AgentConfig | None = None):
        self._config = config or AgentConfig()
        self._scorer = MetricsScorer()
        self._results: dict[str, ModelSummary] = {}
        self._last_run_manifest: RunManifest | None = None

    def run_evaluation(
        self,
        models: list[dict[str, str]] | None = None,
        tasks: list[BenchmarkTask] | None = None,
        num_trials: int = 3,
        use_mock_tools: bool = False,
        *,
        include_extended_tasks: bool = False,
        run_seed: int | None = None,
    ) -> dict[str, ModelSummary]:
        """
        Run full evaluation across models and tasks.

        Args:
            models: List of {"provider": "...", "model": "..."} dicts.
                    Defaults to config.evaluation.models_to_compare.
            tasks:  Subset of benchmark tasks. Defaults to core (or core+extended).
            num_trials: Number of runs per model×task for variance analysis.
            use_mock_tools: If True, use mock tool implementations (no network).
            include_extended_tasks: If True and ``tasks`` is None, add extended task suite
                (longer, tool-heavy; not part of the default published baseline).
            run_seed: RNG seed for Python/NumPy (bootstrap CIs, etc.); defaults to
                ``ARCHON_EVAL_SEED`` or 42. Does not make remote LLMs bit-reproducible.
        """
        if models is None:
            models = [
                {"provider": "huggingface", "model": m}
                for m in self._config.evaluation.models_to_compare
            ]

        if tasks is None:
            tasks = load_benchmark_tasks(include_extended=include_extended_tasks)
        else:
            tasks = list(tasks)

        eval_seed = int(run_seed) if run_seed is not None else env_eval_seed()
        set_global_seeds(eval_seed)
        started_utc = datetime.now(UTC)
        self._last_run_manifest = None

        registry = build_default_registry(use_mock=use_mock_tools)

        logger.info(
            "evaluation_started",
            num_models=len(models),
            num_tasks=len(tasks),
            num_trials=num_trials,
        )

        for model_spec in models:
            model_name = model_spec["model"]
            provider = model_spec.get("provider", "openai")
            logger.info("evaluating_model", model=model_name)

            llm_config = LLMConfig(
                provider=LLMProvider(provider),
                model_name=model_name,
                temperature=0.0,  # Deterministic for eval
            )

            all_task_metrics: list[TaskMetrics] = []

            for task in tasks:
                for trial in range(num_trials):
                    logger.info(
                        "running_trial",
                        model=model_name,
                        task=task.task_id,
                        trial=trial + 1,
                    )

                    try:
                        llm = _build_llm(llm_config)
                        agent = AgentOrchestrator(
                            llm=llm,
                            registry=registry,
                            config=self._config,
                        )

                        trace = agent.run(task.description)
                        metrics = self._scorer.score(trace, task)
                        all_task_metrics.append(metrics)

                        # Save per-trial trace
                        self._save_trial_trace(
                            trace, task, model_name, trial
                        )

                    except Exception as exc:
                        logger.error(
                            "trial_failed",
                            model=model_name,
                            task=task.task_id,
                            trial=trial + 1,
                            error=str(exc),
                        )
                        # Record a zero-score metric for failed trials
                        all_task_metrics.append(
                            TaskMetrics(
                                task_id=task.task_id,
                                model_name=model_name,
                                overall_success=False,
                            )
                        )

            summary = aggregate_metrics(all_task_metrics, model_name)
            self._results[model_name] = summary

        completed_utc = datetime.now(UTC)
        self._last_run_manifest = make_eval_manifest(
            task_ids=[t.task_id for t in tasks],
            models=list(models),
            num_trials=num_trials,
            use_mock=use_mock_tools,
            eval_seed=eval_seed,
            started=started_utc,
            completed=completed_utc,
        )
        out_dir = Path(self._config.evaluation.output_dir)
        write_run_manifest(str(out_dir / "run_manifest.json"), self._last_run_manifest)
        logger.info(
            "evaluation_completed",
            models_evaluated=len(self._results),
            eval_seed=eval_seed,
        )

        return self._results

    def generate_report(self, output_path: str | None = None) -> str:
        """
        Generate a formatted comparison report.
        """
        if not self._results:
            return "No evaluation results available. Run evaluation first."

        sections = []
        sections.append("=" * 72)
        sections.append("AGENT EVALUATION REPORT")
        sections.append(f"Generated: {datetime.now(UTC).isoformat()}")
        sections.append("=" * 72)

        if self._last_run_manifest:
            m = self._last_run_manifest
            sections.append("\n## Reproducibility\n")
            sections.append(
                f"- archon {m.archon_version} | trace schema {m.trace_schema_version} | "
                f"eval_seed={m.eval_seed} | config SHA256: {m.config_fingerprint_sha256[:16]}…"
            )
            sections.append(f"- {m.non_determinism_note}\n")

        # ── 1. Summary comparison table ──────────────────────────
        sections.append("\n## Model Comparison Summary\n")
        table_data = []
        for name, summary in self._results.items():
            table_data.append([
                _short_name(name),
                f"{summary.mean_tool_accuracy:.1%}",
                f"{summary.mean_schema_adherence:.1%}",
                f"{summary.mean_error_recovery:.1%}",
                f"{summary.mean_step_efficiency:.1%}",
                f"{summary.mean_final_answer_score:.1%}",
                f"{summary.overall_success_rate:.1%}",
                f"{summary.mean_wall_time:.1f}s",
            ])

        sections.append(tabulate(
            table_data,
            headers=[
                "Model", "Tool Acc", "Schema", "Recovery",
                "Efficiency", "Answer", "Success", "Avg Time",
            ],
            tablefmt="grid",
        ))

        # ── 2. Variance analysis ─────────────────────────────────
        sections.append("\n## Variance Analysis (±1 std dev)\n")
        var_data = []
        for name, summary in self._results.items():
            var_data.append([
                _short_name(name),
                f"{summary.mean_tool_accuracy:.1%} ± {summary.std_tool_accuracy:.1%}",
                f"{summary.mean_schema_adherence:.1%} ± {summary.std_schema_adherence:.1%}",
                f"{summary.mean_error_recovery:.1%} ± {summary.std_error_recovery:.1%}",
            ])

        sections.append(tabulate(
            var_data,
            headers=["Model", "Tool Accuracy", "Schema Adherence", "Error Recovery"],
            tablefmt="grid",
        ))

        # ── 3. Failure-mode taxonomy ─────────────────────────────
        sections.append("\n## Failure-Mode Taxonomy Distribution\n")
        for name, summary in self._results.items():
            sections.append(f"\n### {_short_name(name)}")
            if summary.failure_taxonomy:
                tax_data = sorted(
                    summary.failure_taxonomy.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                sections.append(tabulate(
                    tax_data,
                    headers=["Failure Category", "Count"],
                    tablefmt="simple",
                ))
            else:
                sections.append("  No failures recorded.")

        # ── 4. Per-task breakdown ────────────────────────────────
        sections.append("\n## Per-Task Results\n")
        for name, summary in self._results.items():
            sections.append(f"\n### {_short_name(name)}")
            task_groups: dict[str, list[TaskMetrics]] = {}
            for tm in summary.task_metrics:
                task_groups.setdefault(tm.task_id, []).append(tm)

            for task_id, trials in task_groups.items():
                avg_acc = sum(t.tool_call_accuracy for t in trials) / len(trials)
                avg_schema = sum(t.schema_adherence_rate for t in trials) / len(trials)
                successes = sum(1 for t in trials if t.overall_success)
                sections.append(
                    f"  {task_id}: "
                    f"tool_acc={avg_acc:.0%} "
                    f"schema={avg_schema:.0%} "
                    f"success={successes}/{len(trials)} "
                    f"retries={sum(t.total_retries for t in trials)} "
                    f"replans={sum(t.total_replans for t in trials)}"
                )

        # ── 5. Recommendations ───────────────────────────────────
        sections.append("\n## Recommendations\n")
        sections.extend(self._generate_recommendations())

        report = "\n".join(sections)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            logger.info("report_saved", path=output_path)

        return report

    def _generate_recommendations(self) -> list[str]:
        """Produce actionable recommendations from results."""
        recs = []
        for name, summary in self._results.items():
            model = _short_name(name)
            if summary.mean_schema_adherence < 0.8:
                recs.append(
                    f"- {model}: Low schema adherence ({summary.mean_schema_adherence:.0%}). "
                    "Consider adding few-shot examples of valid tool calls to prompts."
                )
            if summary.mean_error_recovery < 0.5:
                recs.append(
                    f"- {model}: Low error recovery ({summary.mean_error_recovery:.0%}). "
                    "Improve reflection prompts with explicit correction instructions."
                )
            top_failure = max(
                summary.failure_taxonomy.items(),
                key=lambda x: x[1],
                default=None,
            )
            if top_failure and top_failure[1] > 2:
                recs.append(
                    f"- {model}: Most common failure: '{top_failure[0]}' "
                    f"({top_failure[1]} occurrences). "
                    "Redesign tool interface or prompt for this failure mode."
                )
        return recs or ["No specific recommendations — results look good."]

    def _save_trial_trace(
        self,
        trace: AgentTrace,
        task: BenchmarkTask,
        model_name: str,
        trial: int,
    ) -> None:
        """Save individual trial trace for debugging."""
        output_dir = Path(self._config.evaluation.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_model = model_name.replace("/", "_")
        path = output_dir / f"{task.task_id}_{safe_model}_trial{trial}.json"
        with open(path, "w") as f:
            f.write(trace.model_dump_json(indent=2))

    def export_results_json(
        self,
        output_path: str,
        *,
        manifest: RunManifest | None = None,
    ) -> None:
        """Export raw results as JSON for further analysis. Embeds run manifest when available."""
        data: dict[str, Any] = {}
        m = manifest or self._last_run_manifest
        if m is not None:
            data["__archon_run__"] = m.to_dict()
        for name, summary in self._results.items():
            data[name] = {
                "mean_tool_accuracy": summary.mean_tool_accuracy,
                "mean_schema_adherence": summary.mean_schema_adherence,
                "mean_error_recovery": summary.mean_error_recovery,
                "mean_step_efficiency": summary.mean_step_efficiency,
                "mean_final_answer_score": summary.mean_final_answer_score,
                "overall_success_rate": summary.overall_success_rate,
                "mean_wall_time": summary.mean_wall_time,
                "std_tool_accuracy": summary.std_tool_accuracy,
                "std_schema_adherence": summary.std_schema_adherence,
                "failure_taxonomy": summary.failure_taxonomy,
            }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("results_exported", path=output_path)


def _short_name(model: str) -> str:
    """Shorten model names for display."""
    parts = model.split("/")
    return parts[-1] if len(parts) > 1 else model
