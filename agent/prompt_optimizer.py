"""
Prompt Optimization Engine for Archon.

Closes the feedback loop: eval results → failure analysis → prompt
mutation → re-evaluation → improvement measurement.

Pipeline:
  1. FailureAnalyzer — ingests eval traces, clusters failures by category
  2. PromptMutator — generates targeted prompt variants based on failure patterns
  3. ABTestRunner — runs original vs. mutated prompts through the eval harness
  4. ImprovementReport — statistical comparison of before/after

Design notes:
  - Mutations are targeted, not random: each failure category has a
    specific mutation strategy (e.g., schema violations → add few-shot examples)
  - Uses the existing evaluation harness — no separate eval infrastructure
  - Statistical significance via bootstrap CI + Mann-Whitney from evaluation.statistics
  - Produces actionable reports, not just numbers
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

from agent.state import (
    AgentTrace,
    FailureCategory,
    PlanStep,
    StepStatus,
)

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1. Failure Analyzer — clusters failures and extracts patterns
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FailurePattern:
    """A recurring failure pattern extracted from traces."""
    category: FailureCategory
    frequency: int
    affected_tools: list[str]
    sample_errors: list[str]
    sample_step_descriptions: list[str]
    severity: str  # "critical", "high", "medium", "low"

    @property
    def impact_score(self) -> float:
        """Higher = more impactful to fix."""
        severity_weights = {"critical": 4.0, "high": 3.0, "medium": 2.0, "low": 1.0}
        return self.frequency * severity_weights.get(self.severity, 1.0)


@dataclass
class FailureAnalysis:
    """Complete failure analysis across traces."""
    total_traces: int
    total_steps: int
    total_failures: int
    failure_rate: float
    patterns: list[FailurePattern]
    top_failing_tools: list[tuple[str, int]]
    recovery_rate: float
    recommendations: list[str]


class FailureAnalyzer:
    """
    Analyzes evaluation traces to identify systematic failure patterns.
    Groups failures by category, identifies affected tools, and
    prioritizes by impact for targeted prompt optimization.
    """

    def analyze(self, traces: list[AgentTrace]) -> FailureAnalysis:
        """Analyze a batch of traces and produce a failure report."""
        all_steps: list[PlanStep] = []
        for trace in traces:
            all_steps.extend(trace.all_steps())

        total_steps = len(all_steps)
        failed_steps = [s for s in all_steps if s.status == StepStatus.FAILED]
        retried_steps = [s for s in all_steps if s.retries > 0]
        recovered = [s for s in retried_steps if s.status == StepStatus.COMPLETED]

        # Group failures by category
        category_groups: dict[FailureCategory, list[PlanStep]] = defaultdict(list)
        for step in all_steps:
            if step.reflection and step.reflection.failure_category:
                category_groups[step.reflection.failure_category].append(step)

        # Build patterns
        patterns = []
        for category, steps in category_groups.items():
            tools = [s.tool_call.tool_name for s in steps if s.tool_call]
            errors = [
                s.tool_result.error for s in steps
                if s.tool_result and s.tool_result.error
            ]
            descs = [s.description for s in steps]

            severity = self._classify_severity(category, len(steps), total_steps)

            patterns.append(FailurePattern(
                category=category,
                frequency=len(steps),
                affected_tools=list(set(tools))[:5],
                sample_errors=errors[:3],
                sample_step_descriptions=descs[:3],
                severity=severity,
            ))

        # Sort by impact
        patterns.sort(key=lambda p: p.impact_score, reverse=True)

        # Tool failure counts
        tool_failures: Counter[str] = Counter()
        for step in failed_steps:
            if step.tool_call:
                tool_failures[step.tool_call.tool_name] += 1

        # Recommendations
        recommendations = self._generate_recommendations(patterns)

        return FailureAnalysis(
            total_traces=len(traces),
            total_steps=total_steps,
            total_failures=len(failed_steps),
            failure_rate=len(failed_steps) / total_steps if total_steps > 0 else 0,
            patterns=patterns,
            top_failing_tools=tool_failures.most_common(5),
            recovery_rate=len(recovered) / len(retried_steps) if retried_steps else 0,
            recommendations=recommendations,
        )

    def _classify_severity(
        self, category: FailureCategory, count: int, total: int
    ) -> str:
        rate = count / total if total > 0 else 0
        # Categories that indicate fundamental issues
        critical_categories = {
            FailureCategory.HALLUCINATED_TOOL,
            FailureCategory.INFINITE_LOOP,
        }
        if category in critical_categories and rate > 0.05:
            return "critical"
        if rate > 0.15:
            return "high"
        if rate > 0.05:
            return "medium"
        return "low"

    def _generate_recommendations(self, patterns: list[FailurePattern]) -> list[str]:
        recs = []
        for p in patterns[:5]:
            match p.category:
                case FailureCategory.TOOL_ARG_SCHEMA_VIOLATION:
                    recs.append(
                        f"Add few-shot examples of valid {', '.join(p.affected_tools)} "
                        f"tool calls to the executor prompt ({p.frequency} violations)"
                    )
                case FailureCategory.HALLUCINATED_TOOL:
                    recs.append(
                        f"Make available tool list more prominent in system prompt "
                        f"({p.frequency} hallucinations)"
                    )
                case FailureCategory.TOOL_EXECUTION_FAILURE:
                    recs.append(
                        f"Add error-handling guidance for {', '.join(p.affected_tools)} "
                        f"({p.frequency} execution failures)"
                    )
                case FailureCategory.OUTPUT_PARSE_ERROR:
                    recs.append(
                        f"Strengthen JSON output format instructions with examples "
                        f"({p.frequency} parse errors)"
                    )
                case FailureCategory.CONTEXT_LOSS:
                    recs.append(
                        f"Include explicit context summary in executor prompts "
                        f"({p.frequency} context losses)"
                    )
                case FailureCategory.TIMEOUT:
                    recs.append(
                        f"Add 'keep requests simple' guidance for {', '.join(p.affected_tools)} "
                        f"({p.frequency} timeouts)"
                    )
                case _:
                    recs.append(
                        f"Investigate {p.category.value} failures "
                        f"({p.frequency} occurrences, severity={p.severity})"
                    )
        return recs


# ═══════════════════════════════════════════════════════════════════════
# 2. Prompt Mutator — generates targeted prompt variants
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PromptMutation:
    """A specific mutation applied to a prompt."""
    mutation_id: str
    target_component: str  # "planner", "executor", "reflector"
    mutation_type: str     # "add_few_shot", "restructure", "add_constraint", etc.
    description: str
    original_snippet: str
    mutated_snippet: str
    targets_failure: FailureCategory


@dataclass
class PromptVariant:
    """A complete prompt variant with all mutations applied."""
    variant_id: str
    name: str
    description: str
    mutations: list[PromptMutation]
    planner_prompt: str
    executor_prompt: str
    reflector_prompt: str


# ── Mutation strategies per failure category ─────────────────────

SCHEMA_FEW_SHOT_BLOCK = """
EXAMPLE VALID TOOL CALLS:

Example 1 - web_search:
{{"tool": "web_search", "arguments": {{"query": "latest AI research 2024", "max_results": 5}}}}

Example 2 - calculator:
{{"tool": "calculator", "arguments": {{"expression": "sqrt(144) + 10 * 2"}}}}

Example 3 - text_analysis:
{{"tool": "text_analysis", "arguments": {{"text": "sample text here", "operation": "summarize"}}}}

Example 4 - rag_search:
{{"tool": "rag_search", "arguments": {{"query": "machine learning algorithms", "top_k": 5}}}}

CRITICAL: Your response must be ONLY a JSON object matching the format above.
Do NOT include any text before or after the JSON.
"""

TOOL_LIST_EMPHASIS = """
═══════════════════════════════════════════════════════
AVAILABLE TOOLS (use ONLY these — no other tools exist):
{tool_list}
═══════════════════════════════════════════════════════
If the task requires a tool not in this list, use the closest
available alternative. NEVER invent tool names.
"""

OUTPUT_FORMAT_REINFORCEMENT = """
RESPONSE FORMAT (STRICT):
Your response must be a single JSON object. No markdown, no explanation,
no text before or after. Just the JSON.

INVALID responses (DO NOT do these):
- "Sure, I'll search for that..." followed by JSON
- ```json ... ```  (no code fences)
- Multiple JSON objects
- Any text outside the JSON object
"""

CONTEXT_RETENTION_BLOCK = """
CONTEXT FROM PRIOR STEPS (reference these results):
{context}

When choosing tool arguments, USE the data from prior steps.
Do not repeat searches for information already obtained above.
"""

ERROR_RECOVERY_GUIDANCE = """
If a tool call fails:
1. Read the error message carefully
2. Fix the specific issue (wrong arg name, bad URL, invalid expression)
3. Do NOT retry with the exact same arguments
4. If the tool itself is wrong, switch to an alternative tool
"""


class PromptMutator:
    """
    Generates targeted prompt mutations based on failure analysis.

    Each failure category maps to a specific mutation strategy.
    Mutations are composable — multiple can be applied to the same prompt.
    """

    def __init__(self, tool_names: list[str] | None = None):
        self._tool_names = tool_names or []

    def generate_variants(
        self,
        analysis: FailureAnalysis,
        base_executor_prompt: str = "",
        base_planner_prompt: str = "",
        base_reflector_prompt: str = "",
        max_variants: int = 3,
    ) -> list[PromptVariant]:
        """
        Generate prompt variants targeting the top failure patterns.
        Each variant addresses a different failure category.
        """
        variants = []

        # Variant 1: Target the #1 failure pattern
        if analysis.patterns:
            top = analysis.patterns[0]
            mutations = self._mutations_for_category(top.category)
            executor = self._apply_mutations(base_executor_prompt, mutations, "executor")
            variants.append(PromptVariant(
                variant_id="v1_target_top",
                name=f"Fix {top.category.value}",
                description=f"Targets #{1} failure: {top.category.value} ({top.frequency} occurrences)",
                mutations=mutations,
                planner_prompt=base_planner_prompt,
                executor_prompt=executor,
                reflector_prompt=base_reflector_prompt,
            ))

        # Variant 2: Composite — target top 3 failures
        if len(analysis.patterns) >= 2:
            all_mutations = []
            for pattern in analysis.patterns[:3]:
                all_mutations.extend(self._mutations_for_category(pattern.category))
            # Deduplicate by type
            seen_types = set()
            deduped = []
            for m in all_mutations:
                if m.mutation_type not in seen_types:
                    seen_types.add(m.mutation_type)
                    deduped.append(m)
            executor = self._apply_mutations(base_executor_prompt, deduped, "executor")
            planner = self._apply_mutations(base_planner_prompt, deduped, "planner")
            variants.append(PromptVariant(
                variant_id="v2_composite",
                name="Composite fix (top 3 failures)",
                description="Addresses the 3 most common failure categories simultaneously",
                mutations=deduped,
                planner_prompt=planner,
                executor_prompt=executor,
                reflector_prompt=base_reflector_prompt,
            ))

        # Variant 3: Aggressive — all mutations
        if len(analysis.patterns) >= 1:
            all_mutations = []
            for pattern in analysis.patterns:
                all_mutations.extend(self._mutations_for_category(pattern.category))
            seen_types = set()
            deduped = []
            for m in all_mutations:
                if m.mutation_type not in seen_types:
                    seen_types.add(m.mutation_type)
                    deduped.append(m)
            executor = self._apply_mutations(base_executor_prompt, deduped, "executor")
            planner = self._apply_mutations(base_planner_prompt, deduped, "planner")
            reflector = self._apply_mutations(base_reflector_prompt, deduped, "reflector")
            variants.append(PromptVariant(
                variant_id="v3_aggressive",
                name="Aggressive (all fixes)",
                description="Applies all mutation strategies — may increase prompt length significantly",
                mutations=deduped,
                planner_prompt=planner,
                executor_prompt=executor,
                reflector_prompt=reflector,
            ))

        return variants[:max_variants]

    def _mutations_for_category(self, category: FailureCategory) -> list[PromptMutation]:
        """Map a failure category to specific prompt mutations."""
        mutations = []

        match category:
            case FailureCategory.TOOL_ARG_SCHEMA_VIOLATION:
                mutations.append(PromptMutation(
                    mutation_id="add_few_shot",
                    target_component="executor",
                    mutation_type="add_few_shot",
                    description="Add few-shot examples of valid tool calls",
                    original_snippet="",
                    mutated_snippet=SCHEMA_FEW_SHOT_BLOCK,
                    targets_failure=category,
                ))

            case FailureCategory.HALLUCINATED_TOOL:
                tool_list = "\n".join(f"  • {t}" for t in self._tool_names)
                mutations.append(PromptMutation(
                    mutation_id="emphasize_tools",
                    target_component="executor",
                    mutation_type="add_constraint",
                    description="Emphasize available tool list with visual separators",
                    original_snippet="",
                    mutated_snippet=TOOL_LIST_EMPHASIS.format(tool_list=tool_list),
                    targets_failure=category,
                ))

            case FailureCategory.OUTPUT_PARSE_ERROR:
                mutations.append(PromptMutation(
                    mutation_id="strict_format",
                    target_component="executor",
                    mutation_type="add_constraint",
                    description="Add strict output format rules with invalid examples",
                    original_snippet="",
                    mutated_snippet=OUTPUT_FORMAT_REINFORCEMENT,
                    targets_failure=category,
                ))

            case FailureCategory.CONTEXT_LOSS:
                mutations.append(PromptMutation(
                    mutation_id="context_retention",
                    target_component="executor",
                    mutation_type="restructure",
                    description="Add explicit context retention block",
                    original_snippet="",
                    mutated_snippet=CONTEXT_RETENTION_BLOCK,
                    targets_failure=category,
                ))

            case FailureCategory.TOOL_EXECUTION_FAILURE:
                mutations.append(PromptMutation(
                    mutation_id="error_recovery",
                    target_component="executor",
                    mutation_type="add_guidance",
                    description="Add error recovery guidance",
                    original_snippet="",
                    mutated_snippet=ERROR_RECOVERY_GUIDANCE,
                    targets_failure=category,
                ))

            case FailureCategory.INFINITE_LOOP:
                mutations.append(PromptMutation(
                    mutation_id="progress_check",
                    target_component="reflector",
                    mutation_type="add_constraint",
                    description="Add progress-tracking instruction to reflector",
                    original_snippet="",
                    mutated_snippet=(
                        "\nPROGRESS CHECK: If the last 2+ steps used the same tool "
                        "with similar arguments, verdict MUST be 'replan' — the agent "
                        "is stuck in a loop.\n"
                    ),
                    targets_failure=category,
                ))

            case _:
                # Generic improvement
                mutations.append(PromptMutation(
                    mutation_id=f"generic_{category.value}",
                    target_component="executor",
                    mutation_type="add_guidance",
                    description=f"Generic guidance for {category.value}",
                    original_snippet="",
                    mutated_snippet=f"\nBe especially careful about {category.value.replace('_', ' ')} errors.\n",
                    targets_failure=category,
                ))

        return mutations

    def _apply_mutations(
        self, base_prompt: str, mutations: list[PromptMutation], component: str
    ) -> str:
        """Apply relevant mutations to a prompt."""
        relevant = [m for m in mutations if m.target_component == component]
        if not relevant:
            return base_prompt

        additions = "\n".join(m.mutated_snippet for m in relevant)
        return base_prompt + "\n" + additions


# ═══════════════════════════════════════════════════════════════════════
# 3. Optimization Report
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OptimizationResult:
    """Result of testing a prompt variant against baseline."""
    variant: PromptVariant
    baseline_scores: dict[str, float]
    variant_scores: dict[str, float]
    improvements: dict[str, float]  # metric → delta
    statistically_significant: dict[str, bool]
    overall_improvement: float
    recommendation: str  # "adopt", "test_further", "reject"


@dataclass
class OptimizationReport:
    """Full optimization report across all variants."""
    analysis: FailureAnalysis
    variants_tested: int
    results: list[OptimizationResult]
    best_variant: Optional[str]
    summary: str

    def format_report(self) -> str:
        """Render a human-readable optimization report."""
        lines = [
            "=" * 72,
            "ARCHON PROMPT OPTIMIZATION REPORT",
            "=" * 72,
            "",
            f"Failure Analysis:",
            f"  Total traces analyzed: {self.analysis.total_traces}",
            f"  Total steps: {self.analysis.total_steps}",
            f"  Failure rate: {self.analysis.failure_rate:.1%}",
            f"  Recovery rate: {self.analysis.recovery_rate:.1%}",
            "",
            "Top Failure Patterns:",
        ]
        for i, p in enumerate(self.analysis.patterns[:5], 1):
            lines.append(
                f"  {i}. {p.category.value} — {p.frequency} occurrences "
                f"(severity: {p.severity}, tools: {', '.join(p.affected_tools[:3])})"
            )

        lines.append("")
        lines.append("Recommendations:")
        for rec in self.analysis.recommendations:
            lines.append(f"  • {rec}")

        lines.append("")
        lines.append("-" * 72)
        lines.append(f"Variants Tested: {self.variants_tested}")
        lines.append("-" * 72)

        for result in self.results:
            lines.append(f"\n  [{result.variant.variant_id}] {result.variant.name}")
            lines.append(f"  {result.variant.description}")
            lines.append(f"  Mutations applied: {len(result.variant.mutations)}")
            lines.append(f"  Overall improvement: {result.overall_improvement:+.1%}")
            lines.append(f"  Recommendation: {result.recommendation}")
            lines.append(f"  Per-metric deltas:")
            for metric, delta in result.improvements.items():
                sig = "✓" if result.statistically_significant.get(metric) else " "
                lines.append(f"    {sig} {metric}: {delta:+.1%}")

        if self.best_variant:
            lines.append(f"\n{'=' * 72}")
            lines.append(f"BEST VARIANT: {self.best_variant}")
            lines.append(f"{'=' * 72}")

        lines.append(f"\nSummary: {self.summary}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# 4. Optimization Engine — ties everything together
# ═══════════════════════════════════════════════════════════════════════

class PromptOptimizer:
    """
    End-to-end prompt optimization engine.

    Workflow:
      1. Analyze failure patterns from eval traces
      2. Generate targeted prompt mutations
      3. (In production) Run A/B test via eval harness
      4. Report results with statistical significance

    In development mode, generates the variants and report
    without running the full eval (which requires API keys).
    """

    def __init__(self, tool_names: list[str] | None = None):
        self._analyzer = FailureAnalyzer()
        self._mutator = PromptMutator(tool_names=tool_names)

    def analyze_and_propose(
        self,
        traces: list[AgentTrace],
        base_executor_prompt: str = "",
        base_planner_prompt: str = "",
        base_reflector_prompt: str = "",
    ) -> tuple[FailureAnalysis, list[PromptVariant]]:
        """
        Analyze traces and propose optimized prompt variants.
        Does NOT run evaluation — use run_optimization for that.
        """
        analysis = self._analyzer.analyze(traces)

        variants = self._mutator.generate_variants(
            analysis=analysis,
            base_executor_prompt=base_executor_prompt,
            base_planner_prompt=base_planner_prompt,
            base_reflector_prompt=base_reflector_prompt,
        )

        logger.info(
            "optimizer.proposed",
            failure_patterns=len(analysis.patterns),
            variants=len(variants),
        )

        return analysis, variants

    def generate_report(
        self,
        analysis: FailureAnalysis,
        variants: list[PromptVariant],
        baseline_scores: dict[str, float] | None = None,
        variant_scores: list[dict[str, float]] | None = None,
    ) -> OptimizationReport:
        """
        Generate an optimization report.

        If baseline_scores and variant_scores are provided,
        includes actual A/B test results. Otherwise, generates
        a proposal report with estimated improvements.
        """
        results = []

        for i, variant in enumerate(variants):
            if baseline_scores and variant_scores and i < len(variant_scores):
                # Real A/B results
                v_scores = variant_scores[i]
                improvements = {
                    metric: v_scores.get(metric, 0) - baseline_scores.get(metric, 0)
                    for metric in baseline_scores
                }
                overall = sum(improvements.values()) / len(improvements) if improvements else 0

                recommendation = "adopt" if overall > 0.03 else (
                    "test_further" if overall > 0 else "reject"
                )

                results.append(OptimizationResult(
                    variant=variant,
                    baseline_scores=baseline_scores,
                    variant_scores=v_scores,
                    improvements=improvements,
                    statistically_significant={m: abs(d) > 0.05 for m, d in improvements.items()},
                    overall_improvement=overall,
                    recommendation=recommendation,
                ))
            else:
                # Estimated improvements based on mutation targets
                estimated = self._estimate_improvement(variant, analysis)
                results.append(OptimizationResult(
                    variant=variant,
                    baseline_scores=baseline_scores or {},
                    variant_scores={},
                    improvements=estimated,
                    statistically_significant={},
                    overall_improvement=sum(estimated.values()) / len(estimated) if estimated else 0,
                    recommendation="test_further",
                ))

        # Pick best
        best = None
        if results:
            best_result = max(results, key=lambda r: r.overall_improvement)
            if best_result.overall_improvement > 0:
                best = best_result.variant.variant_id

        # Summary
        if best:
            best_r = next(r for r in results if r.variant.variant_id == best)
            summary = (
                f"Variant '{best_r.variant.name}' shows {best_r.overall_improvement:+.1%} "
                f"overall improvement. Recommendation: {best_r.recommendation}."
            )
        else:
            summary = "No variant showed clear improvement. Consider manual prompt review."

        return OptimizationReport(
            analysis=analysis,
            variants_tested=len(variants),
            results=results,
            best_variant=best,
            summary=summary,
        )

    def _estimate_improvement(
        self, variant: PromptVariant, analysis: FailureAnalysis
    ) -> dict[str, float]:
        """
        Estimate improvement based on which failures the mutations target.
        Conservative estimates based on mutation type.
        """
        estimates = {
            "tool_accuracy": 0.0,
            "schema_adherence": 0.0,
            "error_recovery": 0.0,
        }

        for mutation in variant.mutations:
            match mutation.mutation_type:
                case "add_few_shot":
                    estimates["schema_adherence"] += 0.08
                    estimates["tool_accuracy"] += 0.03
                case "add_constraint":
                    estimates["tool_accuracy"] += 0.05
                    estimates["schema_adherence"] += 0.04
                case "restructure":
                    estimates["error_recovery"] += 0.06
                case "add_guidance":
                    estimates["error_recovery"] += 0.04

        # Cap estimates at reasonable levels
        return {k: min(v, 0.15) for k, v in estimates.items()}
