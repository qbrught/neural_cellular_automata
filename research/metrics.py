"""Quantitative metrics for paper figures and tables.

All metrics are pure functions of recorded time series (and optional
per-step vote diagnostics). No simulation code here.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() < 1e-12 or b[m].std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).any():
        return float("nan")
    return float(np.nanmean(x))


def summarize_run(
    series: dict[str, np.ndarray],
    *,
    goal_frac_repro: float,
    early_frac: float = 0.1,
    late_frac: float = 0.1,
) -> dict[str, Any]:
    """Compute scalar summary stats for one run.

    Expected series keys (from runner):
      alive, reproducer_alive, eliminator_alive, loss_r, loss_e
      V_kin_mean, V_foe_mean
      vote_R_help_kin, vote_R_harm_foe, vote_E_help_kin, vote_E_harm_foe
      same_goal_edge_frac  (optional)
      death_rate_same_edge, death_rate_cross_edge, death_rate_cross_minus_same
      goal_frac_repro, alive_goal_frac_repro  (optional; Step C)
    """
    ra = np.asarray(series["reproducer_alive"], dtype=float)
    ea = np.asarray(series["eliminator_alive"], dtype=float)
    alive = np.asarray(series["alive"], dtype=float)
    lr = np.asarray(series["loss_r"], dtype=float)
    le = np.asarray(series["loss_e"], dtype=float)
    T = len(alive)
    total = ra + ea
    # Density residual uses *initial* goal frac (goal_frac_repro arg): under C
    # this residual should grow as type fractions leave the init labels.
    pred_ra = goal_frac_repro * total

    early_n = max(1, int(T * early_frac))
    late_n = max(1, int(T * late_frac))
    early = slice(0, early_n)
    late = slice(T - late_n, T)

    # Extinction: first step both types (or total) hit 0 and stay near 0.
    extinct_step = None
    zeros = np.where(alive <= 0)[0]
    if len(zeros):
        # first time total dies
        extinct_step = int(zeros[0])

    # Vote specialization (reproducer: help_kin - harm_foe)
    r_hk = series.get("vote_R_help_kin")
    r_hf = series.get("vote_R_harm_foe")
    e_hk = series.get("vote_E_help_kin")
    e_hf = series.get("vote_E_harm_foe")
    if r_hk is not None and r_hf is not None:
        r_disc = np.asarray(r_hk, float) - np.asarray(r_hf, float)
        e_disc = np.asarray(e_hf, float) - np.asarray(e_hk, float)
    else:
        r_disc = np.full(T, np.nan)
        e_disc = np.full(T, np.nan)

    m_loss = np.isfinite(lr) & np.isfinite(le)

    out: dict[str, Any] = {
        "T": T,
        "goal_frac_repro": float(goal_frac_repro),  # initial (all cells)
        "goal_frac_repro_initial": float(goal_frac_repro),
        # Coupling
        "corr_ra_ea": _safe_corr(ra, ea),
        "corr_ra_density": _safe_corr(ra, pred_ra),
        "mean_abs_residual_ra": float(np.mean(np.abs(ra - pred_ra))),
        "ra_std": float(ra.std()),
        # Population
        "mean_alive": float(alive.mean()),
        "mean_ra": float(ra.mean()),
        "mean_ea": float(ea.mean()),
        "ratio_ra_ea_early": float(
            ra[early].mean() / (ea[early].mean() + 1e-9)
        ),
        "ratio_ra_ea_late": float(
            ra[late].mean() / (ea[late].mean() + 1e-9)
        ),
        "mean_alive_early": float(alive[early].mean()),
        "mean_alive_late": float(alive[late].mean()),
        "extinct_step": extinct_step,
        # Loss structure
        "corr_loss_r_e": _safe_corr(lr[m_loss], le[m_loss]) if m_loss.any() else float("nan"),
        "mean_loss_r": _nanmean(lr),
        "mean_loss_e": _nanmean(le),
        "mean_le_minus_lr": _nanmean(le - lr),
        "std_le_minus_lr": float(np.nanstd(le - lr)) if m_loss.any() else float("nan"),
        # Votes
        "mean_V_kin": _nanmean(series.get("V_kin_mean", np.array([np.nan]))),
        "mean_V_foe": _nanmean(series.get("V_foe_mean", np.array([np.nan]))),
        "mean_R_vote_disc": _nanmean(r_disc),
        "mean_E_vote_disc": _nanmean(e_disc),
        "late_R_vote_disc": _nanmean(r_disc[late]),
        "late_E_vote_disc": _nanmean(e_disc[late]),
        "early_R_vote_disc": _nanmean(r_disc[early]),
        "early_E_vote_disc": _nanmean(e_disc[early]),
    }
    if "same_goal_edge_frac" in series:
        sg = np.asarray(series["same_goal_edge_frac"], float)
        out["mean_segregation"] = _nanmean(sg)
        out["late_segregation"] = _nanmean(sg[late])

    # Typed edge death rates (same-goal vs cross-goal directed edges).
    if "death_rate_same_edge" in series and "death_rate_cross_edge" in series:
        d_same = np.asarray(series["death_rate_same_edge"], float)
        d_cross = np.asarray(series["death_rate_cross_edge"], float)
        d_gap = np.asarray(
            series.get(
                "death_rate_cross_minus_same",
                d_cross - d_same,
            ),
            float,
        )
        out["mean_death_rate_same_edge"] = _nanmean(d_same)
        out["mean_death_rate_cross_edge"] = _nanmean(d_cross)
        out["mean_death_rate_cross_minus_same"] = _nanmean(d_gap)
        out["late_death_rate_same_edge"] = _nanmean(d_same[late])
        out["late_death_rate_cross_edge"] = _nanmean(d_cross[late])
        out["late_death_rate_cross_minus_same"] = _nanmean(d_gap[late])
        out["early_death_rate_cross_minus_same"] = _nanmean(d_gap[early])

    # Step C: time-varying goal composition.
    # goal_frac_repro series = fraction of all cells with goal==REPRO (post-step).
    # alive_goal_frac_repro = same among alive cells only.
    if "goal_frac_repro" in series:
        gfr = np.asarray(series["goal_frac_repro"], float)
        out["goal_frac_repro_final"] = float(gfr[-1]) if len(gfr) else float("nan")
        out["goal_frac_repro_drift"] = (
            float(gfr[-1] - goal_frac_repro) if len(gfr) else float("nan")
        )
        out["mean_goal_frac_repro"] = _nanmean(gfr)
    else:
        out["goal_frac_repro_final"] = float(goal_frac_repro)
        out["goal_frac_repro_drift"] = 0.0
        out["mean_goal_frac_repro"] = float(goal_frac_repro)

    if "alive_goal_frac_repro" in series:
        agf = np.asarray(series["alive_goal_frac_repro"], float)
        out["alive_goal_frac_repro_final"] = (
            float(agf[-1]) if len(agf) and np.isfinite(agf[-1]) else float("nan")
        )
        if len(agf) > 1:
            d = np.abs(np.diff(agf))
            out["mean_abs_goal_frac_delta"] = _nanmean(d)
        else:
            out["mean_abs_goal_frac_delta"] = 0.0
    else:
        out["alive_goal_frac_repro_final"] = float("nan")
        out["mean_abs_goal_frac_delta"] = float("nan")

    return out


def summary_to_row(version_id: str, seed: int, summary: dict[str, Any]) -> dict[str, Any]:
    """Flat row for CSV / markdown tables."""
    return {
        "version": version_id,
        "seed": seed,
        **summary,
    }


# Metrics highlighted in the paper comparison table (order matters).
TABLE_COLUMNS: list[tuple[str, str, str]] = [
    # key, header, format
    ("corr_ra_ea", "corr(ra,ea)", ".3f"),
    ("corr_ra_density", "corr(ra, dens)", ".3f"),
    ("mean_abs_residual_ra", "|ra residual|", ".2f"),
    ("ratio_ra_ea_early", "ra/ea early", ".2f"),
    ("ratio_ra_ea_late", "ra/ea late", ".2f"),
    ("goal_frac_repro_drift", "g_frac drift", ".3f"),
    ("mean_abs_goal_frac_delta", "mean|Δg_alive|", ".4f"),
    ("corr_loss_r_e", "corr(Lr,Le)", ".3f"),
    ("late_R_vote_disc", "R vote disc late", ".3f"),
    ("late_E_vote_disc", "E vote disc late", ".3f"),
    ("late_death_rate_same_edge", "death same late", ".3f"),
    ("late_death_rate_cross_edge", "death cross late", ".3f"),
    ("late_death_rate_cross_minus_same", "death gap late", ".3f"),
    ("mean_alive_late", "alive late", ".1f"),
    ("extinct_step", "extinct@", "s"),
]
