"""
Statistical analysis for model comparison.

Provides rigorous statistical methods beyond simple mean/std:
  - Bootstrap confidence intervals (non-parametric, assumption-free)
  - Cohen's d effect size (practical significance, not just p-values)
  - Mann-Whitney U test (non-parametric comparison, works with small n)
  - Cliff's delta (ordinal effect size, robust to non-normality)

Why these methods?
  - LLM evaluation scores are NOT normally distributed (ceiling/floor effects,
    bimodal failures). Parametric tests (t-test) assume normality.
  - Bootstrap CI is assumption-free and works with n as small as 10.
  - Effect size tells you HOW MUCH better, not just whether there's
    a statistically significant difference.
  - Mann-Whitney U is the non-parametric alternative to the independent t-test.

Usage in the evaluation report:
  - Bootstrap 95% CI on each metric → "Model A: 0.85 [0.78, 0.91]"
  - Cohen's d between models → "Large effect (d=1.2) in favor of Model A"
  - Mann-Whitney p-value → "Difference is significant (p=0.003)"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BootstrapCI:
    """Bootstrap confidence interval result."""
    mean: float
    ci_lower: float
    ci_upper: float
    std: float
    n_samples: int
    n_bootstrap: int

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.ci_lower:.3f}, {self.ci_upper:.3f}]"

    @property
    def ci_width(self) -> float:
        return self.ci_upper - self.ci_lower


@dataclass(frozen=True)
class EffectSize:
    """Effect size comparison between two groups."""
    cohens_d: float
    cliffs_delta: float
    interpretation: str  # "negligible", "small", "medium", "large"
    favors: str  # "A", "B", or "neither"

    def __str__(self) -> str:
        return f"d={self.cohens_d:.2f} ({self.interpretation}, favors {self.favors})"


@dataclass(frozen=True)
class SignificanceTest:
    """Mann-Whitney U test result."""
    u_statistic: float
    p_value: float
    significant_at_005: bool
    significant_at_001: bool
    n_a: int
    n_b: int

    def __str__(self) -> str:
        sig = "***" if self.significant_at_001 else ("*" if self.significant_at_005 else "n.s.")
        return f"U={self.u_statistic:.1f}, p={self.p_value:.4f} {sig}"


@dataclass(frozen=True)
class ModelComparison:
    """Full statistical comparison between two models on one metric."""
    metric_name: str
    model_a: str
    model_b: str
    ci_a: BootstrapCI
    ci_b: BootstrapCI
    effect_size: EffectSize
    significance: SignificanceTest

    def summary(self) -> str:
        return (
            f"{self.metric_name}:\n"
            f"  {self.model_a}: {self.ci_a}\n"
            f"  {self.model_b}: {self.ci_b}\n"
            f"  Effect: {self.effect_size}\n"
            f"  Significance: {self.significance}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Core statistical functions
# ═══════════════════════════════════════════════════════════════════════

def bootstrap_ci(
    data: list[float] | NDArray,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> BootstrapCI:
    """
    Compute bootstrap confidence interval.

    Non-parametric: makes no assumptions about the distribution.
    Uses the percentile method (simplest, sufficient for our use case).

    Args:
        data: Sample observations.
        confidence: Confidence level (default 95%).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.
    """
    arr = np.asarray(data, dtype=float)
    n = len(arr)

    if n == 0:
        return BootstrapCI(
            mean=0.0, ci_lower=0.0, ci_upper=0.0,
            std=0.0, n_samples=0, n_bootstrap=n_bootstrap,
        )

    if n == 1:
        val = float(arr[0])
        return BootstrapCI(
            mean=val, ci_lower=val, ci_upper=val,
            std=0.0, n_samples=1, n_bootstrap=n_bootstrap,
        )

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return BootstrapCI(
        mean=float(np.mean(arr)),
        ci_lower=lower,
        ci_upper=upper,
        std=float(np.std(arr, ddof=1)) if n > 1 else 0.0,
        n_samples=n,
        n_bootstrap=n_bootstrap,
    )


def cohens_d(a: list[float] | NDArray, b: list[float] | NDArray) -> float:
    """
    Compute Cohen's d effect size (pooled standard deviation).

    d ≈ 0.2 → small, 0.5 → medium, 0.8 → large (Cohen's benchmarks).
    Positive d means A > B.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    n_a, n_b = len(a_arr), len(b_arr)

    if n_a < 2 or n_b < 2:
        return 0.0

    mean_diff = np.mean(a_arr) - np.mean(b_arr)
    # Pooled std (Hedges' correction for small samples)
    var_a = np.var(a_arr, ddof=1)
    var_b = np.var(b_arr, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_std == 0:
        return 0.0

    return float(mean_diff / pooled_std)


def cliffs_delta(a: list[float] | NDArray, b: list[float] | NDArray) -> float:
    """
    Compute Cliff's delta (ordinal effect size).

    Range: [-1, 1]. |δ| < 0.147 → negligible, < 0.33 → small,
    < 0.474 → medium, else large (Romano et al. 2006).

    More robust than Cohen's d for non-normal distributions.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)

    if len(a_arr) == 0 or len(b_arr) == 0:
        return 0.0

    # Count dominance
    more = 0
    less = 0
    for ai in a_arr:
        for bi in b_arr:
            if ai > bi:
                more += 1
            elif ai < bi:
                less += 1

    n = len(a_arr) * len(b_arr)
    return float((more - less) / n) if n > 0 else 0.0


def mann_whitney_u(
    a: list[float] | NDArray,
    b: list[float] | NDArray,
) -> SignificanceTest:
    """
    Mann-Whitney U test (non-parametric).

    Tests whether the distribution of A is stochastically greater than B.
    Does NOT assume normality, equal variances, or equal sample sizes.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    n_a, n_b = len(a_arr), len(b_arr)

    if n_a < 2 or n_b < 2:
        return SignificanceTest(
            u_statistic=0.0, p_value=1.0,
            significant_at_005=False, significant_at_001=False,
            n_a=n_a, n_b=n_b,
        )

    try:
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(a_arr, b_arr, alternative="two-sided")
    except ImportError:
        # Fallback: approximate U-test without scipy
        stat, p = _approximate_mann_whitney(a_arr, b_arr)

    return SignificanceTest(
        u_statistic=float(stat),
        p_value=float(p),
        significant_at_005=p < 0.05,
        significant_at_001=p < 0.01,
        n_a=n_a,
        n_b=n_b,
    )


def _approximate_mann_whitney(
    a: NDArray, b: NDArray
) -> tuple[float, float]:
    """
    Approximate Mann-Whitney U without scipy.
    Uses normal approximation (valid for n > 20).
    """
    n_a, n_b = len(a), len(b)
    combined = np.concatenate([a, b])
    ranks = np.argsort(np.argsort(combined)) + 1  # Rank from 1

    rank_sum_a = np.sum(ranks[:n_a])
    u_a = rank_sum_a - n_a * (n_a + 1) / 2
    u_b = n_a * n_b - u_a
    u = min(u_a, u_b)

    # Normal approximation
    mu = n_a * n_b / 2
    sigma = np.sqrt(n_a * n_b * (n_a + n_b + 1) / 12)
    if sigma == 0:
        return float(u), 1.0

    z = (u - mu) / sigma
    # Two-tailed p-value (approximate with standard normal CDF)
    p = 2 * (1 - _norm_cdf(abs(z)))

    return float(u), float(p)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    import math
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ═══════════════════════════════════════════════════════════════════════
# High-level comparison
# ═══════════════════════════════════════════════════════════════════════

def _interpret_effect(d: float, delta: float) -> tuple[str, str]:
    """Interpret effect size magnitude and direction."""
    abs_d = abs(d)
    if abs_d < 0.2:
        interp = "negligible"
    elif abs_d < 0.5:
        interp = "small"
    elif abs_d < 0.8:
        interp = "medium"
    else:
        interp = "large"

    if abs(delta) < 0.05:
        favors = "neither"
    elif d > 0:
        favors = "A"
    else:
        favors = "B"

    return interp, favors


def compare_models(
    metric_name: str,
    model_a_name: str,
    model_a_scores: list[float],
    model_b_name: str,
    model_b_scores: list[float],
    confidence: float = 0.95,
) -> ModelComparison:
    """
    Full statistical comparison between two models on a single metric.
    Returns bootstrap CIs, effect size, and significance test.
    """
    ci_a = bootstrap_ci(model_a_scores, confidence=confidence)
    ci_b = bootstrap_ci(model_b_scores, confidence=confidence)

    d = cohens_d(model_a_scores, model_b_scores)
    delta = cliffs_delta(model_a_scores, model_b_scores)
    interp, favors = _interpret_effect(d, delta)

    effect = EffectSize(
        cohens_d=round(d, 3),
        cliffs_delta=round(delta, 3),
        interpretation=interp,
        favors=f"{model_a_name}" if favors == "A" else (
            f"{model_b_name}" if favors == "B" else "neither"
        ),
    )

    sig = mann_whitney_u(model_a_scores, model_b_scores)

    return ModelComparison(
        metric_name=metric_name,
        model_a=model_a_name,
        model_b=model_b_name,
        ci_a=ci_a,
        ci_b=ci_b,
        effect_size=effect,
        significance=sig,
    )
