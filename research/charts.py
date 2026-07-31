"""Chart generation for the research suite (paper figures)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Headless-safe backend
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def plot_run_panel(result: dict[str, Any], out_path: Path) -> Path:
    """4-panel figure for a single version×seed run."""
    _style()
    s = result["series"]
    t = np.arange(len(s["alive"]))
    vid = result["version_id"]
    seed = result["seed"]
    cfg_id = result.get("config_id")
    title = f"Version {vid} · seed {seed}"
    if cfg_id:
        title = f"Config `{cfg_id}` · {title}"

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(t, s["reproducer_alive"], label="repro", color="#2ca02c", lw=1.4)
    ax.plot(t, s["eliminator_alive"], label="elim", color="#d62728", lw=1.4)
    ax.plot(t, s["alive"], label="total", color="#1f77b4", lw=1.0, alpha=0.7)
    ax.set_ylabel("alive count")
    ax.set_title("Population by type")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(t, s["loss_r"], label="loss repro", color="#2ca02c", lw=1.2)
    ax.plot(t, s["loss_e"], label="loss elim", color="#d62728", lw=1.2)
    ax.set_ylabel("mean loss")
    ax.set_title("Loss trajectories")
    ax.legend()

    ax = axes[1, 0]
    r_disc = s["vote_R_help_kin"] - s["vote_R_harm_foe"]
    e_disc = s["vote_E_harm_foe"] - s["vote_E_help_kin"]
    ax.plot(t, r_disc, label="R: help_kin − harm_foe", color="#2ca02c", lw=1.2)
    ax.plot(t, e_disc, label="E: harm_foe − help_kin", color="#d62728", lw=1.2)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("vote discrimination")
    ax.set_xlabel("step")
    ax.set_title("Vote specialization")
    ax.legend()

    ax = axes[1, 1]
    if "death_rate_same_edge" in s and "death_rate_cross_edge" in s:
        ax.plot(
            t, s["death_rate_same_edge"],
            label="same-type edges", color="#2ca02c", lw=1.2,
        )
        ax.plot(
            t, s["death_rate_cross_edge"],
            label="cross-type edges", color="#d62728", lw=1.2,
        )
        ax.set_ylabel("sender death rate")
        ax.set_xlabel("step")
        ax.set_title("Typed edge death rates")
        ax.legend()
    else:
        ax.plot(t, s["V_kin_mean"], label="mean V_kin", color="#9467bd", lw=1.2)
        ax.plot(t, s["V_foe_mean"], label="mean V_foe", color="#8c564b", lw=1.2)
        ax.set_ylabel("vote aggregate")
        ax.set_xlabel("step")
        ax.set_title("Survival vote channels")
        ax.legend()

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_version_overlay(
    results: list[dict[str, Any]],
    out_path: Path,
    *,
    metric_key: str,
    ylabel: str,
    title: str,
    series_fn=None,
    title_prefix: str | None = None,
) -> Path:
    """Overlay a metric across versions (mean over seeds if multiple).

    results: list of run result dicts (may include multiple seeds per version).
    series_fn: optional callable(series) -> 1d array; else use series[metric_key].
    """
    _style()
    by_v: dict[str, list[np.ndarray]] = {}
    for r in results:
        s = r["series"]
        arr = series_fn(s) if series_fn else np.asarray(s[metric_key], float)
        by_v.setdefault(r["version_id"], []).append(arr)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = plt.cm.tab10.colors
    for i, (vid, curves) in enumerate(sorted(by_v.items())):
        # Align to min length
        L = min(len(c) for c in curves)
        stack = np.stack([c[:L] for c in curves], axis=0)
        mean = stack.mean(axis=0)
        t = np.arange(L)
        ax.plot(t, mean, label=vid, color=colors[i % len(colors)], lw=1.6)
        if stack.shape[0] > 1:
            lo = stack.min(axis=0)
            hi = stack.max(axis=0)
            ax.fill_between(t, lo, hi, color=colors[i % len(colors)], alpha=0.15)

    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    full_title = f"{title_prefix}: {title}" if title_prefix else title
    ax.set_title(full_title)
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_comparison_dashboard(
    results: list[dict[str, Any]],
    out_dir: Path,
    *,
    title_prefix: str | None = None,
) -> dict[str, Path]:
    """Write the standard comparison chart set. Returns path map.

    title_prefix: e.g. config id ``disc_0001`` prepended to every chart title.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    # Infer prefix from results if not given
    if title_prefix is None:
        cids = {r.get("config_id") for r in results if r.get("config_id")}
        if len(cids) == 1:
            title_prefix = f"Config {next(iter(cids))}"

    def _ov(name, fname, metric_key, ylabel, title, series_fn=None):
        paths[name] = plot_version_overlay(
            results,
            out_dir / fname,
            metric_key=metric_key,
            ylabel=ylabel,
            title=title,
            series_fn=series_fn,
            title_prefix=title_prefix,
        )

    _ov("alive_total", "01_alive_total.png", "alive", "total alive",
        "Total population over time")
    _ov("alive_repro", "02_alive_repro.png", "reproducer_alive", "reproducer alive",
        "Reproducer population")
    _ov("alive_elim", "03_alive_elim.png", "eliminator_alive", "eliminator alive",
        "Eliminator population")
    _ov(
        "ratio", "04_type_ratio.png", "alive", "ra / (ea+ε)",
        "Type ratio (repro / elim)",
        series_fn=lambda s: s["reproducer_alive"] / (s["eliminator_alive"] + 1e-9),
    )
    _ov("loss_r", "05_loss_repro.png", "loss_r", "loss", "Reproducer mean loss")
    _ov("loss_e", "06_loss_elim.png", "loss_e", "loss", "Eliminator mean loss")
    _ov(
        "vote_R", "07_vote_disc_repro.png", "vote_R_help_kin",
        "help_kin − harm_foe", "Reproducer vote specialization",
        series_fn=lambda s: s["vote_R_help_kin"] - s["vote_R_harm_foe"],
    )
    _ov(
        "vote_E", "08_vote_disc_elim.png", "vote_E_harm_foe",
        "harm_foe − help_kin", "Eliminator vote specialization",
        series_fn=lambda s: s["vote_E_harm_foe"] - s["vote_E_help_kin"],
    )
    _ov("V_kin", "09_V_kin.png", "V_kin_mean", "mean V_kin",
        "Kin vote channel (mean over grid)")
    _ov("V_foe", "10_V_foe.png", "V_foe_mean", "mean V_foe",
        "Foe vote channel (mean over grid)")
    _ov("segregation", "11_segregation.png", "same_goal_edge_frac",
        "same-goal edge frac", "Spatial segregation (alive–alive 4-edges)")
    _ov("death_same", "12_death_rate_same_edge.png", "death_rate_same_edge",
        "sender death rate", "Death rate on same-type directed edges")
    _ov("death_cross", "13_death_rate_cross_edge.png", "death_rate_cross_edge",
        "sender death rate", "Death rate on cross-type directed edges")
    _ov("death_gap", "14_death_rate_gap.png", "death_rate_cross_minus_same",
        "cross − same", "Typed death gap (cross − same; >0 ⇒ cross more lethal)")

    # Step C: goal fraction over time (all cells).
    has_goal_frac = any(
        "goal_frac_repro" in r.get("series", {}) for r in results
    )
    if has_goal_frac:
        _ov("goal_frac", "17_goal_frac_repro.png", "goal_frac_repro",
            "frac goal=REPRO (all cells)",
            "Goal composition over time (Step C colonization)")
        _ov("alive_goal_frac", "18_alive_goal_frac_repro.png",
            "alive_goal_frac_repro", "frac goal=REPRO (alive)",
            "Alive type fraction over time")

    # Scalar bar chart: corr(ra,ea) by version
    _style()
    by_v: dict[str, list[float]] = {}
    for r in results:
        by_v.setdefault(r["version_id"], []).append(r["summary"]["corr_ra_ea"])
    fig, ax = plt.subplots(figsize=(6, 4))
    vids = sorted(by_v)
    means = [np.nanmean(by_v[v]) for v in vids]
    stds = [np.nanstd(by_v[v]) if len(by_v[v]) > 1 else 0.0 for v in vids]
    ax.bar(vids, means, yerr=stds, color="#4c72b0", alpha=0.85, capsize=4)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("corr(repro_alive, elim_alive)")
    bar_title = "Alive-count coupling (lower = more type-specific dynamics)"
    if title_prefix:
        bar_title = f"{title_prefix}: {bar_title}"
    ax.set_title(bar_title)
    fig.tight_layout()
    paths["coupling_bar"] = out_dir / "15_coupling_bar.png"
    fig.savefig(paths["coupling_bar"])
    plt.close(fig)

    # Bar: late typed death gap by version
    by_gap: dict[str, list[float]] = {}
    for r in results:
        g = r["summary"].get("late_death_rate_cross_minus_same", float("nan"))
        by_gap.setdefault(r["version_id"], []).append(g)
    if by_gap:
        fig, ax = plt.subplots(figsize=(6, 4))
        vids = sorted(by_gap)
        means = [np.nanmean(by_gap[v]) for v in vids]
        stds = [np.nanstd(by_gap[v]) if len(by_gap[v]) > 1 else 0.0 for v in vids]
        ax.bar(vids, means, yerr=stds, color="#c44e52", alpha=0.85, capsize=4)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("late death gap (cross − same)")
        gap_title = "Cross-type lethality vs same-type (late window)"
        if title_prefix:
            gap_title = f"{title_prefix}: {gap_title}"
        ax.set_title(gap_title)
        fig.tight_layout()
        paths["death_gap_bar"] = out_dir / "16_death_gap_bar.png"
        fig.savefig(paths["death_gap_bar"])
        plt.close(fig)

    return paths
