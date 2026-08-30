"""Paired seed-level tests (no scipy)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _wilcoxon_signed_rank(d: np.ndarray) -> dict[str, float]:
    """Two-sided Wilcoxon signed-rank on paired differences.

    Zeros dropped. Ties in |d| get average ranks. p-value uses the
    normal approximation with tie correction; NaN if n < 6.
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    n = int(d.size)
    empty = {
        "n_nonzero": n,
        "W_plus": float("nan"),
        "W_minus": float("nan"),
        "z": float("nan"),
        "p_value": float("nan"),
    }
    if n == 0:
        return empty
    abs_d = np.abs(d)
    order = np.argsort(abs_d, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_d[order[j + 1]] == abs_d[order[i]]:
            j += 1
        avg = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i : j + 1]] = avg
        i = j + 1
    signs = np.sign(d)
    w_plus = float(ranks[signs > 0].sum())
    w_minus = float(ranks[signs < 0].sum())
    # Tie correction: var = n(n+1)(2n+1)/24 - sum(t^3-t)/48
    var = n * (n + 1) * (2 * n + 1) / 24.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_d[order[j + 1]] == abs_d[order[i]]:
            j += 1
        t = j - i + 1
        if t > 1:
            var -= (t ** 3 - t) / 48.0
        i = j + 1
    mean = n * (n + 1) / 4.0
    if var <= 0:
        return {**empty, "W_plus": w_plus, "W_minus": w_minus}
    # Continuity correction toward the mean.
    w = min(w_plus, w_minus)
    z = (w - mean + 0.5 * np.sign(mean - w)) / math.sqrt(var)
    p = 2.0 * min(_norm_cdf(float(z)), 1.0 - _norm_cdf(float(z)))
    p = float(min(1.0, max(0.0, p)))
    if n < 6:
        p = float("nan")
    return {
        "n_nonzero": n,
        "W_plus": w_plus,
        "W_minus": w_minus,
        "z": float(z),
        "p_value": p,
    }


def bootstrap_mean_ci(
    d: np.ndarray,
    *,
    n_iter: int = 4000,
    seed: int = 20260822,
    alpha: float = 0.05,
) -> tuple[float, float]:
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return (float("nan"), float("nan"))
    if d.size == 1:
        v = float(d[0])
        return (v, v)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, d.size, size=(int(n_iter), d.size))
    means = d[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return (lo, hi)


def paired_delta_test(
    off: np.ndarray,
    on: np.ndarray,
    *,
    n_iter: int = 4000,
    seed: int = 20260822,
) -> dict[str, Any]:
    """on − off, one value per seed (aligned arrays)."""
    a = np.asarray(off, dtype=float)
    b = np.asarray(on, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must match, got {a.shape} vs {b.shape}")
    d = b - a
    m = np.isfinite(d)
    d_ok = d[m]
    n = int(d_ok.size)
    mean = float(np.mean(d_ok)) if n else float("nan")
    median = float(np.median(d_ok)) if n else float("nan")
    n_pos = int(np.sum(d_ok > 0)) if n else 0
    n_neg = int(np.sum(d_ok < 0)) if n else 0
    n_zero = int(np.sum(d_ok == 0)) if n else 0
    lo, hi = bootstrap_mean_ci(d_ok, n_iter=n_iter, seed=seed)
    w = _wilcoxon_signed_rank(d_ok)
    return {
        "n": n,
        "mean_delta": mean,
        "median_delta": median,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": n_zero,
        "ci95_lo": lo,
        "ci95_hi": hi,
        **w,
    }


def format_paired(stat: dict[str, Any], *, digits: int = 4) -> str:
    """One markdown line for a paired test dict."""
    if not stat or stat.get("n", 0) == 0:
        return "_no paired seeds_"
    p = stat.get("p_value", float("nan"))
    p_s = "n/a" if not np.isfinite(p) else f"{p:.3f}"
    return (
        f"Δ = {stat['mean_delta']:+.{digits}f} "
        f"(median {stat['median_delta']:+.{digits}f}, "
        f"95% CI [{stat['ci95_lo']:+.{digits}f}, {stat['ci95_hi']:+.{digits}f}], "
        f"n={stat['n']}, +/−/0 = {stat['n_pos']}/{stat['n_neg']}/{stat['n_zero']}, "
        f"Wilcoxon p={p_s})"
    )
