#!/usr/bin/env python3
"""Inspect an Archon trace JSON and print a compact triage report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _iter_steps(trace: dict[str, Any]) -> list[dict[str, Any]]:
    plans = trace.get("plans", [])
    all_steps: list[dict[str, Any]] = []
    for plan in plans:
        all_steps.extend(plan.get("steps", []))
    return all_steps


def _status_counts(steps: list[dict[str, Any]]) -> Counter[str]:
    return Counter(step.get("status", "unknown") for step in steps)


def _failure_counts(steps: list[dict[str, Any]]) -> Counter[str]:
    categories: list[str] = []
    for step in steps:
        reflection = step.get("reflection") or {}
        category = reflection.get("failure_category")
        if category:
            categories.append(str(category))
    return Counter(categories)


def _root_cause_hints(steps: list[dict[str, Any]]) -> list[str]:
    hints: list[str] = []
    schema_invalid = 0
    tool_runtime_errors = 0
    repeated_failed_signatures: Counter[tuple[str, str, str]] = Counter()

    for step in steps:
        tool_call = step.get("tool_call") or {}
        tool_result = step.get("tool_result") or {}
        reflection = step.get("reflection") or {}
        status = step.get("status", "unknown")
        retries = int(step.get("retries", 0))

        if tool_call and not tool_call.get("schema_valid", True):
            schema_invalid += 1

        if tool_result and not tool_result.get("success", True):
            tool_runtime_errors += 1

        if status == "failed":
            signature = (
                step.get("description", ""),
                str(reflection.get("failure_category", "unknown")),
                str(tool_result.get("error", "unknown")),
            )
            repeated_failed_signatures[signature] += 1

        if retries >= 2:
            hints.append(
                f"Step '{step.get('description', 'unknown')}' retried {retries} times; check reflection corrections."
            )

    if schema_invalid:
        hints.append(
            f"{schema_invalid} step(s) had schema-invalid tool calls; likely executor prompt/schema alignment issue."
        )
    if tool_runtime_errors:
        hints.append(
            f"{tool_runtime_errors} step(s) failed after valid tool invocation; likely tool/runtime/external dependency issue."
        )

    most_repeated = repeated_failed_signatures.most_common(1)
    if most_repeated and most_repeated[0][1] >= 2:
        desc, category, _ = most_repeated[0][0]
        count = most_repeated[0][1]
        hints.append(
            f"Repeated failed pattern ({count}x): '{desc}' [{category}] suggests looping or insufficient correction quality."
        )

    if not hints:
        hints.append("No obvious anti-patterns detected from summary signals.")
    return hints


def _print_report(trace: dict[str, Any], show_steps: bool) -> None:
    steps = _iter_steps(trace)
    status_counts = _status_counts(steps)
    failure_counts = _failure_counts(steps)

    print("== Archon Trace Summary ==")
    print(f"trace_id: {trace.get('trace_id', 'unknown')}")
    print(f"task: {trace.get('task_description', '')}")
    print(f"model: {trace.get('model_name', '')}")
    print(f"success: {trace.get('success', False)}")
    print(f"final_answer_present: {bool(trace.get('final_answer'))}")
    print(f"wall_time_seconds: {trace.get('wall_time_seconds', 0.0)}")
    print()

    plans = trace.get("plans", [])
    replans = sum(1 for p in plans if p.get("is_replanned"))
    print("== Plan Stats ==")
    print(f"plans_total: {len(plans)}")
    print(f"replans: {replans}")
    print(f"steps_total: {len(steps)}")
    print(f"total_retries: {trace.get('total_retries', 0)}")
    print()

    print("== Step Status Counts ==")
    for key in ("completed", "failed", "skipped", "retrying", "running", "pending", "unknown"):
        if key in status_counts:
            print(f"{key}: {status_counts[key]}")
    print()

    print("== Top Failure Categories ==")
    if not failure_counts:
        print("none")
    else:
        for category, count in failure_counts.most_common(5):
            print(f"{category}: {count}")
    print()

    print("== Root Cause Hints ==")
    for hint in _root_cause_hints(steps):
        print(f"- {hint}")
    print()

    if show_steps:
        print("== Steps ==")
        for idx, step in enumerate(steps, start=1):
            tool_call = step.get("tool_call") or {}
            tool_result = step.get("tool_result") or {}
            reflection = step.get("reflection") or {}
            print(
                f"{idx:02d}. [{step.get('status', 'unknown')}] "
                f"{step.get('description', '')} "
                f"(tool={tool_call.get('tool_name', 'n/a')}, retries={step.get('retries', 0)}, "
                f"verdict={reflection.get('verdict', 'n/a')})"
            )
            if tool_result and not tool_result.get("success", True):
                print(f"    error: {tool_result.get('error', 'unknown')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an Archon trace JSON.")
    parser.add_argument("trace_path", help="Path to trace JSON file")
    parser.add_argument(
        "--steps",
        action="store_true",
        help="Print per-step details",
    )
    args = parser.parse_args()

    trace_file = Path(args.trace_path)
    if not trace_file.exists():
        raise SystemExit(f"Trace file not found: {trace_file}")

    try:
        trace = json.loads(trace_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in trace file: {exc}") from exc

    _print_report(trace, show_steps=args.steps)


if __name__ == "__main__":
    main()
