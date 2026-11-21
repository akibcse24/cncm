"""Statistical utilities for rigorous CNCM evaluation and multi-seed reporting.

Provides:
- Wilson score confidence intervals for proportions (parse/compile/pass@1 rates).
- Student-t confidence intervals across random seeds (mean +/- half-width).
- Paired t-test and non-parametric paired bootstrap CI for A/B comparisons.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Sequence, Tuple


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> Dict[str, float]:
    """Compute exact 95% (default z=1.96) Wilson score interval for k/n."""
    if total <= 0:
        return {"rate": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0, "k": 0}
    p = float(successes) / float(total)
    n = float(total)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return {
        "k": int(successes),
        "n": int(total),
        "rate": round(p, 5),
        "ci_low": round(lo, 5),
        "ci_high": round(hi, 5),
    }


def _t_critical_95(df: int) -> float:
    """Two-sided 95% Student-t critical value (uses scipy if available, else table)."""
    if df <= 0:
        return float("nan")
    try:
        from scipy import stats as sp_stats  # type: ignore[import-not-found]

        return float(sp_stats.t.ppf(0.975, df))
    except Exception:
        table = {
            1: 12.7062,
            2: 4.3027,
            3: 3.1824,
            4: 2.7764,
            5: 2.5706,
            6: 2.4469,
            7: 2.3646,
            8: 2.3060,
            9: 2.2622,
            10: 2.2281,
        }
        return table.get(df, 1.96)


def seed_summary(values: Sequence[float]) -> Dict[str, Any]:
    """Summarize a metric across seeds with mean, std (ddof=1), and 95% t-CI."""
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "ci95_half": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "values": []}
    mean = sum(vals) / n
    if n == 1:
        return {
            "n": 1,
            "mean": round(mean, 5),
            "std": 0.0,
            "ci95_half": 0.0,
            "ci95_low": round(mean, 5),
            "ci95_high": round(mean, 5),
            "values": [round(v, 5) for v in vals],
        }
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    std = math.sqrt(max(0.0, var))
    se = std / math.sqrt(n)
    t_crit = _t_critical_95(n - 1)
    half = t_crit * se
    return {
        "n": n,
        "mean": round(mean, 5),
        "std": round(std, 5),
        "ci95_half": round(half, 5),
        "ci95_low": round(mean - half, 5),
        "ci95_high": round(mean + half, 5),
        "values": [round(v, 5) for v in vals],
    }


def paired_comparison(a_vals: Sequence[float], b_vals: Sequence[float], n_boot: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """Compute paired difference (a - b), 95% t-CI, paired t-test p-value, and bootstrap CI."""
    if len(a_vals) != len(b_vals) or len(a_vals) == 0:
        raise ValueError("a_vals and b_vals must have equal non-zero length")
    diffs = [float(a) - float(b) for a, b in zip(a_vals, b_vals)]
    summary = seed_summary(diffs)
    p_value: float = float("nan")
    if len(diffs) >= 2:
        try:
            from scipy import stats as sp_stats  # type: ignore[import-not-found]

            _, p_val = sp_stats.ttest_rel(a_vals, b_vals)
            p_value = float(p_val)
        except Exception:
            pass
    rng = random.Random(seed)
    n = len(diffs)
    boot_means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo_b = boot_means[int(0.025 * n_boot)]
    hi_b = boot_means[int(0.975 * n_boot)]
    return {
        "delta_mean": summary["mean"],
        "delta_std": summary["std"],
        "t_ci95_low": summary["ci95_low"],
        "t_ci95_high": summary["ci95_high"],
        "boot_ci95_low": round(lo_b, 5),
        "boot_ci95_high": round(hi_b, 5),
        "paired_t_pvalue": round(p_value, 6) if not math.isnan(p_value) else None,
    }


def format_mean_ci(summary: Dict[str, Any], fmt: str = ".4f") -> str:
    """Format seed_summary as 'mean +/- ci95'."""
    m = format(summary["mean"], fmt)
    h = format(summary["ci95_half"], fmt)
    return f"{m} ± {h} (n={summary['n']})"


def format_wilson(w: Dict[str, float]) -> str:
    """Format wilson_ci as 'XX.X% [lo%, hi%] (k/n)'."""
    return f"{w['rate']*100:.1f}% [{w['ci_low']*100:.1f}%, {w['ci_high']*100:.1f}%] ({w['k']}/{w['n']})"
