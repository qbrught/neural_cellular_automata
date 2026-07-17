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

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f"Version {vid} · seed {seed}", fontsize=12, fontweight="bold")

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
    ax.set_title(title)
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
) -> dict[str, Path]:
    """Write the standard comparison chart set. Returns path map."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["alive_total"] = plot_version_overlay(
        results,
        out_dir / "01_alive_total.png",
        metric_key="alive",
        ylabel="total alive",
        title="Total population over time",
    )
    paths["alive_repro"] = plot_version_overlay(
        results,
        out_dir / "02_alive_repro.png",
        metric_key="reproducer_alive",
        ylabel="reproducer alive",
        title="Reproducer population",
    )
    paths["alive_elim"] = plot_version_overlay(
        results,
        out_dir / "03_alive_elim.png",
        metric_key="eliminator_alive",
        ylabel="eliminator alive",
        title="Eliminator population",
    )
    paths["ratio"] = plot_version_overlay(
        results,
        out_dir / "04_type_ratio.png",
        metric_key="alive",
        ylabel="ra / (ea+ε)",
        title="Type ratio (repro / elim)",
        series_fn=lambda s: s["reproducer_alive"]
        / (s["eliminator_alive"] + 1e-9),
    )
    paths["loss_r"] = plot_version_overlay(
        results,
        out_dir / "05_loss_repro.png",
        metric_key="loss_r",
        ylabel="loss",
        title="Reproducer mean loss",
    )
    paths["loss_e"] = plot_version_overlay(
        results,
        out_dir / "06_loss_elim.png",
        metric_key="loss_e",
        ylabel="loss",
        title="Eliminator mean loss",
    )
    paths["vote_R"] = plot_version_overlay(
        results,
        out_dir / "07_vote_disc_repro.png",
        metric_key="vote_R_help_kin",
        ylabel="help_kin − harm_foe",
        title="Reproducer vote specialization",
        series_fn=lambda s: s["vote_R_help_kin"] - s["vote_R_harm_foe"],
    )
    paths["vote_E"] = plot_version_overlay(
        results,
        out_dir / "08_vote_disc_elim.png",
        metric_key="vote_E_harm_foe",
        ylabel="harm_foe − help_kin",
        title="Eliminator vote specialization",
        series_fn=lambda s: s["vote_E_harm_foe"] - s["vote_E_help_kin"],
    )
    paths["V_kin"] = plot_version_overlay(
        results,
        out_dir / "09_V_kin.png",
        metric_key="V_kin_mean",
        ylabel="mean V_kin",
        title="Kin vote channel (mean over grid)",
    )
    paths["V_foe"] = plot_version_overlay(
        results,
        out_dir / "10_V_foe.png",
        metric_key="V_foe_mean",
        ylabel="mean V_foe",
        title="Foe vote channel (mean over grid)",
    )
    paths["segregation"] = plot_version_overlay(
        results,
        out_dir / "11_segregation.png",
        metric_key="same_goal_edge_frac",
        ylabel="same-goal edge frac",
        title="Spatial segregation (alive–alive 4-edges)",
    )

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
    ax.set_title("Alive-count coupling (lower = more type-specific dynamics)")
    fig.tight_layout()
    paths["coupling_bar"] = out_dir / "12_coupling_bar.png"
    fig.savefig(paths["coupling_bar"])
    plt.close(fig)

    return paths
