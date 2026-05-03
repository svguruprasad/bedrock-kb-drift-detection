"""Statistical significance testing for drift detection."""

import math
from dataclasses import dataclass


@dataclass
class TestResult:
    query: str
    mean_score: float
    std_dev: float
    t_statistic: float
    p_value: float
    passed: bool
    tier: str
    sample_size: int


def t_test_one_sample(
    scores: list[float], threshold: float, alpha: float = 0.05
) -> tuple[float, float, bool]:
    """One-sample t-test: H0: mu >= threshold vs H1: mu < threshold.

    Returns (t_statistic, p_value, passed).
    passed=True means we do NOT reject H0 (safe to migrate).
    """
    k = len(scores)
    if k < 2:
        raise ValueError("Need at least 2 scores for t-test")

    mean = sum(scores) / k
    variance = sum((s - mean) ** 2 for s in scores) / (k - 1)
    std_dev = math.sqrt(variance)

    if std_dev == 0:
        return (float("inf") if mean >= threshold else float("-inf"), 0.0, mean >= threshold)

    t_stat = (mean - threshold) / (std_dev / math.sqrt(k))
    p_value = _t_cdf(t_stat, k - 1)

    passed = p_value > alpha
    return (t_stat, p_value, passed)


def bonferroni_correction(alpha: float, n_tests: int) -> float:
    """Apply Bonferroni correction for multiple testing."""
    return alpha / n_tests


def classify_tier(results: list[TestResult]) -> str:
    """Classify overall migration risk tier.

    Green: all queries pass
    Yellow: 1-5% of queries fail
    Red: more than 5% fail
    """
    total = len(results)
    if total == 0:
        return "Green"

    failed = sum(1 for r in results if not r.passed)
    failure_rate = failed / total

    if failure_rate == 0:
        return "Green"
    elif failure_rate <= 0.05:
        return "Yellow"
    else:
        return "Red"


def run_drift_tests(
    query_scores: dict[str, list[float]],
    threshold: float,
    alpha: float = 0.05,
) -> list[TestResult]:
    """Run statistical drift tests for all queries.

    Args:
        query_scores: mapping of query -> list of composite drift scores
        threshold: minimum acceptable drift score
        alpha: significance level (before Bonferroni correction)

    Returns:
        list of TestResult for each query
    """
    n_tests = len(query_scores)
    adjusted_alpha = bonferroni_correction(alpha, n_tests)

    results = []
    for query, scores in query_scores.items():
        k = len(scores)
        mean = sum(scores) / k
        variance = sum((s - mean) ** 2 for s in scores) / (k - 1) if k > 1 else 0.0
        std_dev = math.sqrt(variance)

        t_stat, p_value, passed = t_test_one_sample(scores, threshold, adjusted_alpha)

        tier = "Green" if passed else "Red"

        results.append(
            TestResult(
                query=query,
                mean_score=round(mean, 4),
                std_dev=round(std_dev, 4),
                t_statistic=round(t_stat, 4),
                p_value=round(p_value, 4),
                passed=passed,
                tier=tier,
                sample_size=k,
            )
        )

    return results


def _t_cdf(t: float, df: int) -> float:
    """Approximate one-tailed p-value for t-distribution.

    Uses the regularized incomplete beta function approximation.
    For production use, replace with scipy.stats.t.cdf.
    """
    x = df / (df + t * t)
    if t >= 0:
        return 1.0 - 0.5 * _regularized_beta(df / 2.0, 0.5, x)
    else:
        return 0.5 * _regularized_beta(df / 2.0, 0.5, x)


def _regularized_beta(a: float, b: float, x: float) -> float:
    """Approximate regularized incomplete beta function via continued fraction."""
    if x < 0 or x > 1:
        return 0.0
    if x == 0 or x == 1:
        return x

    lbeta = _log_beta(a, b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a

    # Lentz's algorithm for continued fraction
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d

    for m in range(1, 200):
        # Even step
        numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        f *= d * c

        # Odd step
        numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        f *= delta

        if abs(delta - 1.0) < 1e-10:
            break

    return front * f


def _log_beta(a: float, b: float) -> float:
    """Log of beta function: log(B(a,b)) = log(Gamma(a)) + log(Gamma(b)) - log(Gamma(a+b))."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
