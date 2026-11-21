"""Statistical utilities for CNCM evaluation and remediation experiments.

Implements:
  - Wilson score confidence intervals for binomial pass rates (Items 3.2, 5.3).
  - Student-t and percentile bootstrap confidence intervals across seeds (Items 1.x, 3.3, 5.x).
  - Paired t-test and non-parametric paired Wilcoxon / exact sign test.
  - Expected self-speculative decoding speedup formula (Item 5.1).

All claims in CLAIMS.md must be reported as mean ± 95% CI across >= 3 seeds
(or Wilson 95% CI for pass rates).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence

import numpy as np

try:
    from scipy import stats as _scipy_stats  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional fallback
    _scipy_stats = None


# Critical two-sided t-values for 95% CI (alpha = 0.05) for df = 1..30 as fallback
_T_CRIT_95_TABLE: Dict[int, float] = {
    1: 12.706205,
    2: 4.302653,
    3: 3.182446,
    4: 2.776445,
    5: 2.570582,
    6: 2.446912,
    7: 2.364624,
    8: 2.306004,
    9: 2.262157,
    10: 2.228139,
    12: 2.178813,
    15: 2.131450,
    20: 2.085963,
    25: 2.059539,
    30: 2.042272,
}


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF."""
    if _scipy_stats is not None:
        return float(_scipy_stats.norm.ppf(p))
    # High-precision Acklam rational approximation fallback
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p}")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def _t_ppf(confidence: float, df: int) -> float:
    """Two-sided Student-t critical value for `confidence` and `df`."""
    alpha = 1.0 - confidence
    if _scipy_stats is not None:
        return float(_scipy_stats.t.ppf(1.0 - alpha / 2.0, df=df))
    if abs(confidence - 0.95) < 1e-6 and df in _T_CRIT_95_TABLE:
        return _T_CRIT_95_TABLE[df]
    # Cornish-Fisher approximation fallback
    z = _norm_ppf(1.0 - alpha / 2.0)
    return z + (z**3 + z) / (4.0 * max(1, df))


def wilson_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Compute exact Wilson score confidence interval for a binomial proportion."""
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    if not (0 <= successes <= total):
        raise ValueError(f"successes ({successes}) must be in [0, {total}]")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    if total == 0:
        return {
            "successes": 0,
            "total": 0,
            "rate": 0.0,
            "low": 0.0,
            "high": 1.0,
            "confidence": confidence,
            "formatted": "0.00% [0.00%, 100.00%] (0/0)",
        }

    p_hat = successes / float(total)
    z = _norm_ppf(1.0 - (1.0 - confidence) / 2.0)
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denom
    half = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat)) / total + z2 / (4.0 * total * total))
    low = max(0.0, center - half)
    high = min(1.0, center + half)

    return {
        "successes": int(successes),
        "total": int(total),
        "rate": round(p_hat, 6),
        "low": round(low, 6),
        "high": round(high, 6),
        "confidence": confidence,
        "formatted": (
            f"{p_hat * 100:.2f}% [{low * 100:.2f}%, {high * 100:.2f}%] "
            f"({successes}/{total})"
        ),
    }


def t_interval(
    values: Sequence[float],
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Compute Student-t mean ± confidence interval across replicates/seeds."""
    arr = np.asarray(list(values), dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        raise ValueError("values must be non-empty")
    mean = float(np.mean(arr))
    if n == 1:
        return {
            "n": 1,
            "mean": round(mean, 6),
            "std": 0.0,
            "se": 0.0,
            "half_width": float("nan"),
            "low": float("nan"),
            "high": float("nan"),
            "confidence": confidence,
            "formatted": f"{mean:.4f} (n=1, CI undefined)",
        }

    std = float(np.std(arr, ddof=1))
    se = std / math.sqrt(n)
    t_crit = _t_ppf(confidence, df=n - 1)
    hw = t_crit * se
    low = mean - hw
    high = mean + hw
    return {
        "n": n,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "se": round(se, 6),
        "half_width": round(hw, 6),
        "low": round(low, 6),
        "high": round(high, 6),
        "confidence": confidence,
        "formatted": f"{mean:.4f} ± {hw:.4f} (95% t-CI, n={n})",
    }


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
    statistic: str = "mean",
) -> Dict[str, Any]:
    """Compute deterministic percentile bootstrap confidence interval."""
    arr = np.asarray(list(values), dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        raise ValueError("values must be non-empty")
    if statistic not in ("mean", "median"):
        raise ValueError(f"Unsupported statistic '{statistic}'; expected 'mean' or 'median'")

    fn = np.mean if statistic == "mean" else np.median
    point = float(fn(arr))
    if n == 1:
        return {
            "n": 1,
            "statistic": statistic,
            "point": round(point, 6),
            "low": round(point, 6),
            "high": round(point, 6),
            "confidence": confidence,
            "n_resamples": n_resamples,
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    samples = fn(arr[idx], axis=1)
    alpha = 1.0 - confidence
    low = float(np.percentile(samples, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0)))
    return {
        "n": n,
        "statistic": statistic,
        "point": round(point, 6),
        "low": round(low, 6),
        "high": round(high, 6),
        "confidence": confidence,
        "n_resamples": n_resamples,
    }


def paired_t_test(
    a: Sequence[float],
    b: Sequence[float],
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Two-sided paired t-test on `diff = a - b` with 95% CI on the mean difference."""
    arr_a = np.asarray(list(a), dtype=np.float64)
    arr_b = np.asarray(list(b), dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"Shape mismatch: {arr_a.shape} vs {arr_b.shape}")
    n = int(arr_a.size)
    if n < 2:
        raise ValueError(f"paired_t_test requires at least 2 pairs, got {n}")

    diff = arr_a - arr_b
    mean_d = float(np.mean(diff))
    std_d = float(np.std(diff, ddof=1))
    se_d = std_d / math.sqrt(n)
    t_crit = _t_ppf(confidence, df=n - 1)
    hw = t_crit * se_d

    if se_d <= 1e-15:
        t_stat = 0.0 if abs(mean_d) <= 1e-15 else math.copysign(float("inf"), mean_d)
        p_val = 1.0 if abs(mean_d) <= 1e-15 else 0.0
    else:
        t_stat = mean_d / se_d
        if _scipy_stats is not None:
            p_val = float(2.0 * _scipy_stats.t.sf(abs(t_stat), df=n - 1))
        else:  # pragma: no cover
            p_val = float(math.erfc(abs(t_stat) / math.sqrt(2.0)))

    return {
        "n": n,
        "mean_diff": round(mean_d, 6),
        "std_diff": round(std_d, 6),
        "t_stat": round(t_stat, 6) if math.isfinite(t_stat) else t_stat,
        "p_value": round(p_val, 6),
        "ci_low": round(mean_d - hw, 6),
        "ci_high": round(mean_d + hw, 6),
        "significant_05": bool(p_val < 0.05),
        "confidence": confidence,
    }


def paired_nonparametric_test(
    a: Sequence[float],
    b: Sequence[float],
) -> Dict[str, Any]:
    """Paired Wilcoxon signed-rank test (or exact two-sided sign test for small n)."""
    arr_a = np.asarray(list(a), dtype=np.float64)
    arr_b = np.asarray(list(b), dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"Shape mismatch: {arr_a.shape} vs {arr_b.shape}")
    diff = arr_a - arr_b
    nz = diff[np.abs(diff) > 1e-12]
    n_nz = int(nz.size)
    n_pos = int(np.sum(nz > 0))
    n_neg = int(np.sum(nz < 0))

    if n_nz == 0:
        return {
            "test": "exact_sign",
            "n_nonzero": 0,
            "n_pos": 0,
            "n_neg": 0,
            "p_value": 1.0,
            "significant_05": False,
        }

    if _scipy_stats is not None and n_nz >= 6:
        res = _scipy_stats.wilcoxon(nz, zero_method="wilcox", alternative="two-sided")
        p_val = float(res.pvalue)
        return {
            "test": "wilcoxon_signed_rank",
            "n_nonzero": n_nz,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "statistic": round(float(res.statistic), 6),
            "p_value": round(p_val, 6),
            "significant_05": bool(p_val < 0.05),
        }

    # Exact two-sided binomial sign test for small n (e.g. 3-5 seeds)
    k = min(n_pos, n_neg)
    tail = sum(math.comb(n_nz, i) for i in range(k + 1)) * (0.5**n_nz)
    p_val = min(1.0, 2.0 * tail)
    return {
        "test": "exact_sign",
        "n_nonzero": n_nz,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "p_value": round(p_val, 6),
        "significant_05": bool(p_val < 0.05),
    }


def expected_speculative_speedup(
    alpha: float,
    gamma: int,
    draft_cycle_cost_ratio: float,
    verify_cycle_cost_ratio: float = 1.0,
) -> Dict[str, float]:
    """Compute theoretical self-speculative speedup vs Exact-AR (Item 5.1).

    In block-draft self-speculative decoding:
      - Expected accepted tokens per draft-verify cycle (including the bonus token):
          E[tokens] = (1 - alpha^(gamma + 1)) / (1 - alpha)   if alpha < 1 else gamma + 1
      - Cost of one draft-verify cycle normalized to 1 Exact-AR token step:
          C_cycle = draft_cycle_cost_ratio + verify_cycle_cost_ratio
      - Theoretical speedup over Pure Exact-AR:
          S_expected = E[tokens] / C_cycle
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if gamma < 1:
        raise ValueError(f"gamma must be >= 1, got {gamma}")
    if draft_cycle_cost_ratio < 0.0 or verify_cycle_cost_ratio <= 0.0:
        raise ValueError("Cost ratios must be non-negative (verify > 0)")

    if abs(1.0 - alpha) < 1e-9:
        expected_tokens = float(gamma + 1)
    else:
        expected_tokens = (1.0 - (alpha ** (gamma + 1))) / (1.0 - alpha)

    cycle_cost = draft_cycle_cost_ratio + verify_cycle_cost_ratio
    speedup = expected_tokens / cycle_cost
    breakeven_tokens = cycle_cost
    return {
        "alpha": round(alpha, 6),
        "gamma": int(gamma),
        "expected_tokens_per_cycle": round(expected_tokens, 4),
        "cycle_cost_in_ar_steps": round(cycle_cost, 4),
        "expected_speedup_vs_ar": round(speedup, 4),
        "breakeven_tokens_needed": round(breakeven_tokens, 4),
    }


def summarize_seeded_metric(
    values: Sequence[float],
    confidence: float = 0.95,
    require_min_seeds: int = 1,
    bootstrap_seed: int = 42,
) -> Dict[str, Any]:
    """Summarize a metric across seeds with both Student-t and bootstrap CIs."""
    t_res = t_interval(values, confidence=confidence)
    b_res = bootstrap_ci(values, confidence=confidence, seed=bootstrap_seed)
    return {
        "values": [round(float(v), 6) for v in values],
        "n_seeds": len(values),
        "meets_min_seed_policy": len(values) >= require_min_seeds,
        "t_ci": t_res,
        "bootstrap_ci": b_res,
        "summary": t_res["formatted"],
    }
