"""
CLI entry point for the Archon.

Usage:
    python main.py demo                     # Architecture demo (no keys)
    python main.py run "your task"           # Run a task (needs API key)
    python main.py eval --mock --trials 1    # Mock evaluation
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger(__name__)


def cmd_run(args: argparse.Namespace) -> None:
    """Run the agent on a single task (async)."""
    from agent.async_orchestrator import build_agent

    agent = build_agent(
        provider=args.provider,
        model_name=args.model,
        use_mock_tools=args.mock,
    )

    trace = asyncio.run(agent.run(args.task))

    print("\n" + "=" * 68)
    print(f" TASK: {trace.task_description}")
    print("=" * 68)
    print(f" Success:    {trace.success}")
    print(f" Steps:      {trace.total_steps_executed}")
    print(f" Retries:    {trace.total_retries}")
    print(f" Replans:    {trace.total_replans}")
    print(f" Wall time:  {trace.wall_time_seconds:.2f}s")
    print("-" * 68)
    print(" ANSWER:")
    print(trace.final_answer)
    print("=" * 68)

    if args.trace_output:
        with open(args.trace_output, "w") as f:
            f.write(trace.model_dump_json(indent=2))
        print(f"\nTrace saved to: {args.trace_output}")


def cmd_eval(args: argparse.Namespace) -> None:
    """Run the evaluation harness."""
    from config.settings import AgentConfig
    from evaluation.harness import EvaluationHarness

    config = AgentConfig.from_env()
    models = []
    if args.models:
        for m in args.models:
            models.append({"provider": args.provider, "model": m})
    else:
        models = [
            {"provider": "huggingface", "model": "meta-llama/Meta-Llama-3-8B-Instruct"},
            {"provider": "huggingface", "model": "mistralai/Mistral-7B-Instruct-v0.3"},
        ]

    harness = EvaluationHarness(config=config)
    harness.run_evaluation(models=models, num_trials=args.trials, use_mock_tools=args.mock)
    report = harness.generate_report(
        output_path=args.report_output or "evaluation/results/report.txt"
    )
    print(report)
    harness.export_results_json(args.json_output or "evaluation/results/results.json")


def cmd_demo(args: argparse.Namespace) -> None:
    """Architecture demo — shows all components without API keys."""
    from agent.errors import (
        BudgetExceededError,
        RetryableError,
        ToolNotFoundError,
    )
    from agent.llm_backends import DeterministicFakeBackend
    from agent.middleware import TokenBudget, build_default_middleware
    from agent.state import (
        ToolCall,
        WorkingMemory,
    )
    from evaluation.benchmarks.tasks import BENCHMARK_TASKS
    from evaluation.statistics import compare_models
    from tools.implementations import build_default_registry

    print("=" * 68)
    print("  ARCHON — ARCHITECTURE DEMO")
    print("=" * 68)

    # 1. Tool Registry
    registry = build_default_registry(use_mock=True)
    print(f"\n📦 Tool Registry ({len(registry)} tools):")
    for name in registry.list_tools():
        print(f"   • {name}")

    # 2. Tool Execution with Schema Validation
    print("\n🔧 Tool Execution (schema-validated):")
    good = ToolCall(tool_name="calculator", arguments={"expression": "sqrt(144) + 10 * 2"})
    r = registry.execute_tool_call(good)
    print(f"   ✅ calculator('sqrt(144)+10*2') → {r.output} ({r.latency_ms:.2f}ms)")

    bad = ToolCall(tool_name="calculator", arguments={"wrong_arg": "2+2"})
    valid, errors = registry.validate_tool_call(bad)
    print(f"   ❌ Invalid call: valid={valid}")

    ghost = ToolCall(tool_name="quantum_analyzer", arguments={})
    gr = registry.execute_tool_call(ghost)
    print(f"   👻 Hallucinated: '{ghost.tool_name}' → {gr.error[:60]}...")

    # 3. Exception Hierarchy
    print("\n🛡️  Typed Exception Hierarchy:")
    retryable = ToolNotFoundError("quantum_analyzer", registry.list_tools())
    fatal = BudgetExceededError("token", limit=1000, consumed=1500)
    print(f"   Retryable: {type(retryable).__name__} (category={retryable.failure_category.value})")
    print(f"   Fatal:     {type(fatal).__name__} (auto-aborts execution)")
    try:
        raise retryable
    except RetryableError:
        print(f"   ✅ 'except RetryableError' catches all retryable subtypes")

    # 4. Token Budget
    print("\n💰 Token Budget Tracking:")
    budget = TokenBudget(max_input_tokens=100_000, max_output_tokens=50_000, max_total_cost_usd=2.00)
    budget.record_usage(45_000, 12_000)
    budget.record_usage(30_000, 8_000)
    s = budget.summary()
    print(f"   Input:  {s['input_tokens']}")
    print(f"   Output: {s['output_tokens']}")
    print(f"   Cost:   {s['cost_usd']}")

    # 5. Middleware Chain
    print("\n⛓️  Middleware Chain (onion model):")
    print("   Tracing → TokenBudget → RateLimit → Telemetry")
    print("   before: L→R | after: R→L | any can short-circuit")

    # 6. Deterministic Fake
    print("\n🤖 Deterministic Fake LLM Backend:")
    fake = DeterministicFakeBackend(responses={"tokyo": '{"population": 14000000}'})
    resp = asyncio.run(fake.generate("sys", "Find tokyo data"))
    print(f"   'Find tokyo data' → {resp.content}")
    print(f"   {resp.input_tokens}in / {resp.output_tokens}out tokens")

    # 7. Statistical Analysis
    print("\n📊 Statistical Comparison (Bootstrap CI + Cohen's d + Mann-Whitney):")
    comp = compare_models(
        "tool_call_accuracy",
        "LLaMA-3-8B", [0.92, 0.88, 0.91, 0.85, 0.90, 0.87, 0.93, 0.89],
        "Mistral-7B", [0.71, 0.68, 0.73, 0.65, 0.70, 0.66, 0.72, 0.69],
    )
    print(f"   LLaMA-3: {comp.ci_a}")
    print(f"   Mistral: {comp.ci_b}")
    print(f"   Effect:  {comp.effect_size}")
    print(f"   Test:    {comp.significance}")

    # 8. Benchmarks
    print(f"\n📋 Benchmark Suite ({len(BENCHMARK_TASKS)} tasks):")
    for t in BENCHMARK_TASKS:
        print(f"   [{t.task_id}] {t.difficulty.value:6s} │ {t.category.value:25s} │ {t.description[:48]}...")

    print("\n" + "=" * 68)
    print("  Ready. Run 'pytest tests/ -v' or 'make check'.")
    print("=" * 68)


def main() -> None:
    parser = argparse.ArgumentParser(description="Archon")
    subs = parser.add_subparsers(dest="command")

    run_p = subs.add_parser("run")
    run_p.add_argument("task", type=str)
    run_p.add_argument("--provider", default="openai")
    run_p.add_argument("--model", default="gpt-4o-mini")
    run_p.add_argument("--mock", action="store_true")
    run_p.add_argument("--trace-output", type=str)

    eval_p = subs.add_parser("eval")
    eval_p.add_argument("--provider", default="huggingface")
    eval_p.add_argument("--models", nargs="+")
    eval_p.add_argument("--trials", type=int, default=3)
    eval_p.add_argument("--mock", action="store_true")
    eval_p.add_argument("--report-output", type=str)
    eval_p.add_argument("--json-output", type=str)

    subs.add_parser("demo")
    dash_p = subs.add_parser("dashboard")
    dash_p.add_argument("--host", default="127.0.0.1")
    dash_p.add_argument("--port", type=int, default=8787)
    dash_p.add_argument("--traces-dir", type=str, default="traces")

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "dashboard":
        from scripts.serve_dashboard import serve_dashboard
        serve_dashboard(args.host, args.port, Path(args.traces_dir))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
