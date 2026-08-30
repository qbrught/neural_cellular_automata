"""Run / analyse cell-level functional class divergence.

Usage:
  python -m research.response_analysis run --versions original,A --n-steps 1000
  python -m research.response_analysis analyze research_results/<run_name>
  python -m research.response_analysis run --quick

See research/FUNCTIONAL_DIVERGENCE.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from parameters import Parameters
from research.charts import _style
from research.versions import ladder_sort_key
from research.functional import (
    FAMILY_COMMON,
    FAMILY_REALIZED,
    FAMILY_WEIGHTS_ONLY,
    SnapshotFunctional,
    compare_init_late,
    evaluate_snapshot,
)
from research.runner import run_experiment
from research.versions import VERSIONS, get_version, parse_version_list
from simulate import build_grid
from state import GOAL_ELIMINATE, GOAL_REPRODUCE, State

REPRO_C = "#2ca02c"
ELIM_C = "#d62728"
ATTRITION_MIN_TYPE = 0.05
PHI_THRESHOLD = 0.2


def _ordered_vids(vids) -> list[str]:
    return sorted(vids, key=ladder_sort_key)


def _fmt(x: float, spec: str = ".3f") -> str:
    if x is None or not np.isfinite(float(x) if x is not None else np.nan):
        return "nan"
    return format(float(x), spec)


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _load_snapshot_tensors(run_dir: Path) -> tuple[Parameters, State, torch.Tensor, Config]:
    cfg = Config.load(run_dir / "config.json")
    params = Parameters.load(run_dir / "params_final.pt")
    blob = torch.load(run_dir / "state_final.pt", weights_only=True, map_location="cpu")
    state = State(
        x=blob["x"],
        s=blob["s"],
        h=blob["h"],
        goals=blob["goals"],
        rho=blob["rho"],
    )
    return params, state, blob["u"], cfg


def _grid_side(n_cells: int) -> int:
    n = int(round(np.sqrt(n_cells)))
    if n * n != n_cells:
        raise ValueError(f"n_cells={n_cells} is not a square lattice")
    return n


def analyze_run_dir(run_dir: Path, *, embed: bool = True) -> dict[str, Any]:
    """Evaluate init + late snapshots for one version×seed folder."""
    run_dir = Path(run_dir)
    params, state, u, cfg = _load_snapshot_tensors(run_dir)
    with (run_dir / "summary.json").open() as f:
        summary = json.load(f)
    meta: dict[str, Any] = {}
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        with meta_path.open() as f:
            meta = json.load(f)
    version_id = str(meta.get("arm_id") or meta.get("version_id") or run_dir.parent.name)
    seed = int(meta.get("seed", str(run_dir.name).split("_")[-1]))
    config_id = str(meta.get("config_id") or "")
    if not config_id:
        parts = run_dir.resolve().parts
        if "cache" in parts:
            i = parts.index("cache")
            if i + 1 < len(parts):
                config_id = parts[i + 1]

    late = evaluate_snapshot(
        params,
        state,
        u,
        typed_votes=bool(cfg.typed_votes),
        goal_in_f=bool(cfg.goal_in_f),
        embed=embed,
    )
    grid = build_grid(cfg)
    init = evaluate_snapshot(
        grid.params,
        grid.state,
        grid.u,
        typed_votes=bool(cfg.typed_votes),
        goal_in_f=bool(cfg.goal_in_f),
        embed=embed,
    )
    learned = compare_init_late(late, init)
    init_s = init.scalars()
    late_s = late.scalars()
    scalars = {
        "version": version_id,
        "config_id": config_id,
        "seed": seed,
        "typed_votes": int(bool(cfg.typed_votes)),
        "phi_class": float(summary.get("phi_class", np.nan)),
        "phi_class_late": float(summary.get("phi_class_late", np.nan)),
        "late_min_type_frac": float(summary.get("late_min_type_frac", np.nan)),
        "corr_ra_ea": float(summary.get("corr_ra_ea", np.nan)),
        "mean_alive_late": float(summary.get("mean_alive_late", np.nan)),
        **{f"init_{k}": v for k, v in init_s.items()},
        **{f"late_{k}": v for k, v in late_s.items()},
        **learned,
    }
    return {
        "version_id": version_id,
        "config_id": config_id,
        "seed": seed,
        "summary": summary,
        "meta": meta,
        "config": cfg.to_dict(),
        "late": late,
        "init": init,
        "learned": learned,
        "scalars": scalars,
        "run_dir": str(run_dir),
    }


def _seed_dirs_with_snapshots(parent: Path) -> list[Path]:
    out: list[Path] = []
    if not parent.is_dir():
        return out
    for sdir in sorted(parent.iterdir()):
        if not sdir.is_dir() or not sdir.name.startswith("seed_"):
            continue
        if (sdir / "params_final.pt").exists() and (sdir / "state_final.pt").exists():
            out.append(sdir)
    return out


def discover_run_dirs(root: Path) -> list[Path]:
    """Find snapshot dirs under a pipeline cache/ or a suite versions/ tree."""
    root = Path(root)
    out: list[Path] = []
    cache = root / "cache"
    if cache.is_dir():
        for cfg_dir in sorted(cache.iterdir()):
            if not cfg_dir.is_dir():
                continue
            for arm_dir in sorted(cfg_dir.iterdir()):
                if not arm_dir.is_dir():
                    continue
                out.extend(_seed_dirs_with_snapshots(arm_dir))
    versions_dir = root / "versions"
    if versions_dir.is_dir():
        for vdir in sorted(versions_dir.iterdir()):
            if not vdir.is_dir():
                continue
            out.extend(_seed_dirs_with_snapshots(vdir))
    # Dedup while preserving order.
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)
    if not uniq:
        raise FileNotFoundError(
            f"No seed dirs with params_final.pt + state_final.pt under {root} "
            "(pipeline cache/ or suite versions/). Re-run sims so the runner "
            "writes late snapshots."
        )
    return uniq


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _color_goals(goals: np.ndarray) -> list[str]:
    return [REPRO_C if int(g) == GOAL_REPRODUCE else ELIM_C for g in goals]


def plot_embedding_grid(
    records: list[dict[str, Any]],
    path: Path,
    *,
    family: str,
    kind: str,
    when: str,
    title: str,
) -> Path | None:
    """kind: 'pca' | 'umap'. when: 'late' | 'init'."""
    _style()
    vids = _ordered_vids({r["version_id"] for r in records})
    seeds = sorted({r["seed"] for r in records})
    fig, axes = plt.subplots(
        len(vids),
        len(seeds),
        figsize=(3.4 * len(seeds), 3.2 * len(vids)),
        squeeze=False,
    )
    any_ok = False
    for i, vid in enumerate(vids):
        for j, seed in enumerate(seeds):
            ax = axes[i][j]
            rec = next(
                (r for r in records if r["version_id"] == vid and r["seed"] == seed),
                None,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if rec is None:
                ax.set_axis_off()
                continue
            snap: SnapshotFunctional = rec[when]
            fam = snap.families[family]
            xy = fam.pca_xy if kind == "pca" else fam.umap_xy
            if xy is None:
                ax.text(0.5, 0.5, f"{kind} n/a", ha="center", va="center", transform=ax.transAxes)
                continue
            any_ok = True
            c = _color_goals(snap.goals)
            ax.scatter(xy[:, 0], xy[:, 1], c=c, s=8, alpha=0.75, linewidths=0)
            ax.set_title(f"{vid} · seed {seed}", fontsize=9)
            if j == 0:
                ax.set_ylabel(vid)
    if not any_ok:
        plt.close(fig)
        return None
    fig.suptitle(title, fontweight="bold")
    fig.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=REPRO_C, markersize=8, label="repro"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ELIM_C, markersize=8, label="elim"),
        ],
        loc="upper right",
        frameon=True,
    )
    return _save(fig, path)


def plot_init_vs_late_pca(records: list[dict[str, Any]], path: Path, *, family: str) -> Path:
    _style()
    vids = _ordered_vids({r["version_id"] for r in records})
    seeds = sorted({r["seed"] for r in records})
    nrows = len(vids) * 2
    fig, axes = plt.subplots(
        nrows,
        len(seeds),
        figsize=(3.3 * len(seeds), 2.6 * nrows),
        squeeze=False,
    )
    for vi, vid in enumerate(vids):
        for j, seed in enumerate(seeds):
            rec = next(
                (r for r in records if r["version_id"] == vid and r["seed"] == seed),
                None,
            )
            for ti, when in enumerate(("init", "late")):
                ax = axes[vi * 2 + ti][j]
                ax.set_xticks([])
                ax.set_yticks([])
                if rec is None:
                    ax.set_axis_off()
                    continue
                snap: SnapshotFunctional = rec[when]
                xy = snap.families[family].pca_xy
                ax.scatter(xy[:, 0], xy[:, 1], c=_color_goals(snap.goals), s=7, alpha=0.75, linewidths=0)
                ax.set_title(f"{vid} {when} · {seed}", fontsize=8)
    fig.suptitle(f"PCA of `{family}` responses: init vs late (green=R, red=E)", fontweight="bold")
    return _save(fig, path)


def plot_delta_bars(records: list[dict[str, Any]], path: Path) -> Path:
    _style()
    vids = _ordered_vids({r["version_id"] for r in records})
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), sharey=False)
    x = np.arange(len(vids))
    width = 0.35
    for ax, fam, title in (
        (axes[0], FAMILY_REALIZED, "Δ agent (own-goal votes)"),
        (axes[1], FAMILY_COMMON, "Δ maps (shared questions)"),
        (axes[2], FAMILY_WEIGHTS_ONLY, "Δ weights-only (goal bits off)"),
    ):
        init_m, init_s, late_m, late_s = [], [], [], []
        for vid in vids:
            rows = [r for r in records if r["version_id"] == vid]
            iv = [float(r["learned"][f"delta_{fam}_all_init"]) for r in rows]
            lv = [float(r["learned"][f"delta_{fam}_all_late"]) for r in rows]
            init_m.append(float(np.nanmean(iv)))
            init_s.append(float(np.nanstd(iv)))
            late_m.append(float(np.nanmean(lv)))
            late_s.append(float(np.nanstd(lv)))
        ax.bar(x - width / 2, init_m, width, yerr=init_s, label="init", color="#9aa0a6", capsize=3)
        ax.bar(x + width / 2, late_m, width, yerr=late_s, label="late", color="#4c72b0", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(vids)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(title)
        ax.set_ylabel("Δ = ℰ/2 (between − equal-pool within)")
        ax.legend()
    fig.suptitle("Functional class divergence Δ (mean ± std over seeds)", fontweight="bold")
    return _save(fig, path)


def plot_phi_vs_delta(records: list[dict[str, Any]], path: Path) -> Path:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharex=True)
    vids = _ordered_vids({r["version_id"] for r in records})
    colors = plt.cm.tab10.colors
    for ax, ykey, ylab, title in (
        (
            axes[0],
            "delta_realized_all_late",
            "Δ agent (own-goal, late)",
            "Φ vs typed-agent output",
        ),
        (
            axes[1],
            "delta_common_all_late",
            "Δ maps (shared questions, late)",
            "Φ vs shared-question maps",
        ),
    ):
        for i, vid in enumerate(vids):
            rows = [r for r in records if r["version_id"] == vid]
            xs = [float(r["scalars"]["phi_class_late"]) for r in rows]
            ys = [float(r["learned"][ykey]) for r in rows]
            ax.scatter(xs, ys, s=70, color=colors[i % 10], label=vid, zorder=3)
            for r, x, y in zip(rows, xs, ys):
                ax.annotate(
                    str(r["seed"]), (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7
                )
        ax.axvline(PHI_THRESHOLD, color="k", ls="--", lw=0.7)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("Φ_late")
        ax.set_ylabel(ylab)
        ax.set_title(title)
    axes[0].legend()
    fig.suptitle("Count-level Φ vs functional Δ", fontweight="bold")
    return _save(fig, path)


def plot_distance_hists(records: list[dict[str, Any]], path: Path) -> Path:
    _style()
    vids = _ordered_vids({r["version_id"] for r in records})
    seeds = sorted({r["seed"] for r in records})
    fig, axes = plt.subplots(
        len(vids),
        len(seeds),
        figsize=(3.6 * len(seeds), 2.8 * len(vids)),
        squeeze=False,
        sharex=True,
    )
    for i, vid in enumerate(vids):
        for j, seed in enumerate(seeds):
            ax = axes[i][j]
            rec = next(
                (r for r in records if r["version_id"] == vid and r["seed"] == seed),
                None,
            )
            if rec is None:
                ax.set_axis_off()
                continue
            snap: SnapshotFunctional = rec["late"]
            fam = snap.families[FAMILY_REALIZED]
            D = fam.D
            g = snap.goals
            n = g.size
            iu, ju = np.triu_indices(n, k=1)
            same = g[iu] == g[ju]
            ax.hist(D[iu, ju][same], bins=30, alpha=0.65, label="within", color="#4c72b0", density=True)
            ax.hist(D[iu, ju][~same], bins=30, alpha=0.65, label="between", color="#dd8452", density=True)
            ax.set_title(f"{vid} · {seed}", fontsize=9)
            if i == len(vids) - 1:
                ax.set_xlabel("z-scored Euclidean")
            if j == 0:
                ax.set_ylabel("density")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Pairwise realized-vote distances (late)", fontweight="bold")
    return _save(fig, path)


def plot_grids(records: list[dict[str, Any]], path: Path) -> Path:
    _style()
    vids = _ordered_vids({r["version_id"] for r in records})
    seeds = sorted({r["seed"] for r in records})
    # 3 rows per version: goal map, agent cluster, shared-question cluster
    fig, axes = plt.subplots(
        3 * len(vids),
        len(seeds),
        figsize=(3.0 * len(seeds), 2.6 * 3 * len(vids)),
        squeeze=False,
    )
    for vi, vid in enumerate(vids):
        for j, seed in enumerate(seeds):
            rec = next(
                (r for r in records if r["version_id"] == vid and r["seed"] == seed),
                None,
            )
            ax_g = axes[vi * 3][j]
            ax_c = axes[vi * 3 + 1][j]
            ax_m = axes[vi * 3 + 2][j]
            for ax in (ax_g, ax_c, ax_m):
                ax.set_xticks([])
                ax.set_yticks([])
            if rec is None:
                ax_g.set_axis_off()
                ax_c.set_axis_off()
                ax_m.set_axis_off()
                continue
            snap: SnapshotFunctional = rec["late"]
            n = _grid_side(snap.n)
            goal = snap.goals.reshape(n, n)
            alive = snap.alive.reshape(n, n)
            cl = snap.families[FAMILY_REALIZED].cluster.reshape(n, n)
            cm = snap.families[FAMILY_COMMON].cluster.reshape(n, n)
            rgb = np.zeros((n, n, 3))
            rgb[goal == GOAL_REPRODUCE] = (0.17, 0.63, 0.17)
            rgb[goal == GOAL_ELIMINATE] = (0.84, 0.15, 0.16)
            rgb[~alive] *= 0.25
            ax_g.imshow(rgb, interpolation="nearest")
            ax_g.set_title(f"{vid} {seed} goal (dim=dead)", fontsize=8)
            ax_c.imshow(cl, interpolation="nearest", cmap="coolwarm", vmin=0, vmax=1)
            ax_c.set_title("k-means k=2 on agent output", fontsize=8)
            ax_m.imshow(cm, interpolation="nearest", cmap="coolwarm", vmin=0, vmax=1)
            ax_m.set_title("k-means k=2 on shared questions", fontsize=8)
    fig.suptitle("Late lattice: planted goal vs response clusters", fontweight="bold")
    return _save(fig, path)


def plot_kin_foe(records: list[dict[str, Any]], path: Path) -> Path:
    _style()
    vids = _ordered_vids({r["version_id"] for r in records})
    ncols = min(3, max(1, len(vids)))
    nrows = int(np.ceil(len(vids) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.2 * ncols, 3.8 * nrows),
        sharey=True,
        squeeze=False,
    )
    rng = np.random.default_rng(0)
    for k, vid in enumerate(vids):
        ax = axes[k // ncols][k % ncols]
        rows = [r for r in records if r["version_id"] == vid]
        vals_r, vals_e = [], []
        for r in rows:
            snap: SnapshotFunctional = r["late"]
            gap = snap.kin_foe_gap
            vals_r.append(gap[snap.goals == GOAL_REPRODUCE])
            vals_e.append(gap[snap.goals == GOAL_ELIMINATE])
        rcat = np.concatenate(vals_r) if vals_r else np.array([])
        ecat = np.concatenate(vals_e) if vals_e else np.array([])
        for x, arr, c, lab in (
            (0, rcat, REPRO_C, "R"),
            (1, ecat, ELIM_C, "E"),
        ):
            if arr.size == 0:
                continue
            jitter = x + rng.uniform(-0.18, 0.18, size=arr.size)
            ax.scatter(jitter, arr, s=6, alpha=0.25, c=c, linewidths=0)
            ax.scatter([x], [float(np.mean(arr))], s=50, c="k", marker="D", zorder=4, label=f"{lab} mean")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["repro", "elim"])
        ax.set_title(vid)
        ax.set_ylabel("realized kin − foe (blank-alive probe)")
    for k in range(len(vids), nrows * ncols):
        axes[k // ncols][k % ncols].set_axis_off()
    fig.suptitle("Probe kin−foe gap by planted class (late; all cells, all seeds)", fontweight="bold")
    return _save(fig, path)


def plot_ari_bars(records: list[dict[str, Any]], path: Path) -> Path:
    _style()
    vids = _ordered_vids({r["version_id"] for r in records})
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(vids))
    width = 0.2
    series = [
        ("ari_realized_all_init", "agent init", "#9aa0a6"),
        ("ari_realized_all_late", "agent late", "#4c72b0"),
        ("ari_common_all_init", "maps init", "#d4a017"),
        ("ari_common_all_late", "maps late", "#dd8452"),
    ]
    for i, (key, lab, col) in enumerate(series):
        means, stds = [], []
        for vid in vids:
            rows = [r for r in records if r["version_id"] == vid]
            vals = [float(r["learned"][key]) for r in rows]
            means.append(float(np.nanmean(vals)))
            stds.append(float(np.nanstd(vals)))
        ax.bar(x + (i - 1.5) * width, means, width, yerr=stds, label=lab, color=col, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(vids)
    ax.set_ylabel("ARI (k-means k=2 vs goals)")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(ncol=2, fontsize=8)
    ax.set_title("Do two clusters recover planted goal labels?")
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _read_joint(rec: dict[str, Any]) -> str:
    phi = float(rec["scalars"]["phi_class_late"])
    min_tf = float(rec["scalars"]["late_min_type_frac"])
    d_map = float(rec["learned"].get("delta_common_all_late", float("nan")))
    ari_map = float(rec["learned"].get("ari_common_all_late", float("nan")))
    d_agent = float(rec["learned"]["delta_realized_all_late"])
    ari_agent = float(rec["learned"]["ari_realized_all_late"])
    high_phi = np.isfinite(phi) and phi > PHI_THRESHOLD
    attrition = np.isfinite(min_tf) and min_tf < ATTRITION_MIN_TYPE
    # Shared-question Δ is the map split (init ≈ 0). Own-goal ARI at init is
    # trivially high; use late ARI only as a cluster-preservation diagnostic.
    map_split = (
        np.isfinite(d_map)
        and d_map > 0.4
        and np.isfinite(ari_map)
        and ari_map > 0.25
    )
    agent_clusters = np.isfinite(ari_agent) and ari_agent > 0.3
    if attrition:
        return "attrition (minority gone)"
    if high_phi and map_split:
        return "mix moved and maps split"
    if (not high_phi) and map_split:
        return "maps split, mix held (Φ miss)"
    if high_phi and not map_split:
        return "Φ without map split"
    if agent_clusters and not map_split:
        return "typed outputs still cluster; maps do not"
    if np.isfinite(d_agent) and d_agent > 0.5 and not map_split:
        return "agent outputs differ; maps do not"
    return "null on both"


def write_report(
    records: list[dict[str, Any]],
    out_dir: Path,
    chart_rel: dict[str, str],
    *,
    n_steps: int | None = None,
) -> Path:
    vids = _ordered_vids({r["version_id"] for r in records})
    seeds = sorted({r["seed"] for r in records})
    cfg0 = records[0].get("config") or {}
    lines: list[str] = []
    lines.append("# Functional class divergence")
    lines.append("")
    lines.append(
        "Each cell is a response vector (ψ on a frozen probe bank). "
        "We z-score those vectors, take pairwise Euclidean distances, and "
        r"summarise type geometry by the energy-distance contrast "
        r"$\Delta=\mathcal{E}/2$ (between-type mean minus the equal average of "
        "the two within-type means). Δ is undefined if a type has fewer than "
        "two cells. PCA/UMAP and k-means ARI vs goals are the clustering "
        "picture: do types organise behaviour-space?"
    )
    lines.append("")
    lines.append(
        "**Agent output (`realized`)** uses the cell's own goal as $g_s$ — "
        "what this typed agent emits. Init Δ here is large even with identical "
        "weights (the label is an input). "
        "**Maps (`common`)** ask every cell the same questions, including both "
        "sender goals: init Δ ≈ 0; a late split is learned. "
        "**`weights_only`** zeros goal bits. Spec: `research/FUNCTIONAL_DIVERGENCE.md`."
    )
    lines.append("")
    lines.append(f"- Seeds: `{seeds}`")
    lines.append(f"- Versions: {', '.join(f'`{v}`' for v in vids)}")
    if n_steps is not None:
        lines.append(f"- Steps: `{n_steps}`")
    if cfg0:
        lines.append(
            f"- Base: N={cfg0.get('N')}, d={cfg0.get('d')}, "
            f"w2={cfg0.get('w2')}, w3={cfg0.get('w3')}"
        )
    lines.append("")
    lines.append("## Seed-level table")
    lines.append("")
    lines.append(
        "| version | seed | Φ_late | min type late | "
        "Δ agent late | ARI agent late | "
        "Δ maps late | Δ maps learned | ARI maps late | "
        "Δ weights late | kin−foe R | kin−foe E | read |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    v_order = {v: i for i, v in enumerate(vids)}
    for rec in sorted(records, key=lambda r: (v_order.get(r["version_id"], 999), r["seed"])):
        s = rec["scalars"]
        L = rec["learned"]
        lines.append(
            f"| `{rec['version_id']}` | {rec['seed']} | "
            f"{_fmt(s['phi_class_late'])} | {_fmt(s['late_min_type_frac'])} | "
            f"{_fmt(L['delta_realized_all_late'])} | "
            f"{_fmt(L['ari_realized_all_late'])} | "
            f"{_fmt(L['delta_common_all_late'])} | "
            f"{_fmt(L['delta_common_all_learned'], '+.3f')} | "
            f"{_fmt(L['ari_common_all_late'])} | "
            f"{_fmt(L['delta_weights_only_all_late'])} | "
            f"{_fmt(s.get('late_mean_kin_foe_gap_R', float('nan')))} | "
            f"{_fmt(s.get('late_mean_kin_foe_gap_E', float('nan')))} | "
            f"{_read_joint(rec)} |"
        )
    lines.append("")
    lines.append("## Means over seeds")
    lines.append("")
    lines.append(
        "| version | Φ_late | Δ agent late | ARI agent late | "
        "Δ maps late | ARI maps late | Δ weights late | kin−foe R | kin−foe E |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    rank_phi: list[tuple[float, str]] = []
    rank_d: list[tuple[float, str]] = []
    for vid in vids:
        rows = [r for r in records if r["version_id"] == vid]
        def _m(key, src="learned"):
            vals = [float(r[src][key] if src == "learned" else r["scalars"][key]) for r in rows]
            return float(np.nanmean(vals)), float(np.nanstd(vals))
        phi_m, phi_s = _m("phi_class_late", "scalars")
        dl_m, dl_s = _m("delta_realized_all_late")
        ar_m, ar_s = _m("ari_realized_all_late")
        dc_m, dc_s = _m("delta_common_all_late")
        ac_m, ac_s = _m("ari_common_all_late")
        dw_m, dw_s = _m("delta_weights_only_all_late")
        kr_m, kr_s = _m("late_mean_kin_foe_gap_R", "scalars")
        ke_m, ke_s = _m("late_mean_kin_foe_gap_E", "scalars")
        rank_phi.append((phi_m, vid))
        rank_d.append((dc_m, vid))
        lines.append(
            f"| `{vid}` | {_fmt(phi_m)} ± {_fmt(phi_s)} | "
            f"{_fmt(dl_m)} ± {_fmt(dl_s)} | {_fmt(ar_m)} ± {_fmt(ar_s)} | "
            f"{_fmt(dc_m)} ± {_fmt(dc_s)} | {_fmt(ac_m)} ± {_fmt(ac_s)} | "
            f"{_fmt(dw_m)} ± {_fmt(dw_s)} | "
            f"{_fmt(kr_m)} ± {_fmt(kr_s)} | {_fmt(ke_m)} ± {_fmt(ke_s)} |"
        )
    lines.append("")
    lines.append("## Ranking: Φ_late vs Δ maps (shared questions, mean over seeds)")
    lines.append("")
    lines.append("| rank | by Φ_late | by Δ maps late |")
    lines.append("| --- | --- | --- |")
    rp = [v for _, v in sorted(rank_phi, key=lambda x: -x[0])]
    rd = [v for _, v in sorted(rank_d, key=lambda x: -x[0] if np.isfinite(x[0]) else -np.inf)]
    phi_map = {v: p for p, v in rank_phi}
    d_map = {v: d for d, v in rank_d}
    for i in range(len(vids)):
        a, b = rp[i], rd[i]
        lines.append(
            f"| {i+1} | `{a}` ({_fmt(phi_map[a])}) | `{b}` ({_fmt(d_map[b])}) |"
        )
    lines.append("")
    if "original" in vids:
        lines.append("## Mean paired difference vs `original`")
        lines.append("")
        lines.append(
            "| version | ΔΦ_late | Δ(Δ agent late) | Δ(Δ maps late) | "
            "Δ(Δ weights late) |"
        )
        lines.append("| --- | --- | --- | --- | --- |")
        base = {r["seed"]: r for r in records if r["version_id"] == "original"}
        for vid in vids:
            if vid == "original":
                continue
            dphi, dd, dc, dw = [], [], [], []
            for rec in [r for r in records if r["version_id"] == vid]:
                b = base.get(rec["seed"])
                if b is None:
                    continue
                dphi.append(rec["scalars"]["phi_class_late"] - b["scalars"]["phi_class_late"])
                dd.append(
                    rec["learned"]["delta_realized_all_late"]
                    - b["learned"]["delta_realized_all_late"]
                )
                dc.append(
                    rec["learned"]["delta_common_all_late"]
                    - b["learned"]["delta_common_all_late"]
                )
                dw.append(
                    rec["learned"]["delta_weights_only_all_late"]
                    - b["learned"]["delta_weights_only_all_late"]
                )
            lines.append(
                f"| `{vid}` | {_fmt(float(np.nanmean(dphi)), '+.3f')} | "
                f"{_fmt(float(np.nanmean(dd)), '+.3f')} | "
                f"{_fmt(float(np.nanmean(dc)), '+.3f')} | "
                f"{_fmt(float(np.nanmean(dw)), '+.3f')} |"
            )
        lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append("| Pattern | Meaning |")
    lines.append("| --- | --- |")
    lines.append("| Δ maps late ≫ 0, ARI maps high, both types alive | learned maps split by type |")
    lines.append("| ARI agent late high, Δ maps ~0 | typed *outputs* still cluster; the maps did not specialise |")
    lines.append("| Φ high, Δ maps low | mix drift / attrition without two maps |")
    lines.append("| Δ maps high, Φ low | two maps, balanced counts — Φ miss |")
    lines.append("| Δ maps nan / minority gone | do not read as two processes |")
    lines.append("| original ≈ A on Δ maps | typed routing is not what split the functions |")
    lines.append("")
    lines.append(
        "`realized` for `original` is the help head toward every receiver; "
        "for `A` it is help→kin and harm→foe. Agent-output Δ at init is large "
        "because own-goal probes put different $(g_s,g_r)$ into an untrained ψ; "
        "that is expected, not two policies. Shared-question Δ at init is the "
        "noise floor. Compare late to init; look at the embeddings."
    )
    lines.append("")
    if chart_rel:
        lines.append("## Charts")
        lines.append("")
        for name, rel in chart_rel.items():
            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"![{name}]({rel})")
            lines.append("")
    out_path = Path(out_dir) / "FUNCTIONAL_REPORT.md"
    out_path.write_text("\n".join(lines))
    return out_path


def write_csv(records: list[dict[str, Any]], path: Path) -> Path:
    path = Path(path)
    rows = [r["scalars"] for r in records]
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return path


def run_analysis(root: Path, *, embed: bool = True) -> list[dict[str, Any]]:
    root = Path(root)
    run_dirs = discover_run_dirs(root)
    print(f"Analysing {len(run_dirs)} snapshots under {root}")
    records: list[dict[str, Any]] = []
    for d in run_dirs:
        t0 = time.time()
        rec = analyze_run_dir(d, embed=embed)
        dt = time.time() - t0
        L = rec["learned"]
        print(
            f"  {rec['version_id']} seed={rec['seed']}  "
            f"Φ_late={rec['scalars']['phi_class_late']:.3f}  "
            f"Δ_agent={L['delta_realized_all_late']:.3f}  "
            f"Δ_maps={L['delta_common_all_late']:.3f} "
            f"(learned {L['delta_common_all_learned']:+.3f})  "
            f"ARI_maps={L['ari_common_all_late']:.3f}  "
            f"[{dt:.1f}s]"
        )
        records.append(rec)

    chart_dir = root / "functional_compare"
    chart_dir.mkdir(parents=True, exist_ok=True)
    chart_rel: dict[str, str] = {}

    def _add(key: str, p: Path | None) -> None:
        if p is not None:
            chart_rel[key] = str(p.relative_to(root))

    _add(
        "PCA agent output late",
        plot_embedding_grid(
            records,
            chart_dir / "01_pca_realized_late.png",
            family=FAMILY_REALIZED,
            kind="pca",
            when="late",
            title="PCA of typed-agent outputs (late, own-goal probes), coloured by goal",
        ),
    )
    umap_p = plot_embedding_grid(
        records,
        chart_dir / "02_umap_realized_late.png",
        family=FAMILY_REALIZED,
        kind="umap",
        when="late",
        title="UMAP of typed-agent outputs (late, own-goal probes), coloured by goal",
    )
    _add("UMAP agent output late", umap_p)
    n_vid = len({r["version_id"] for r in records})
    _add(
        "PCA agent output init vs late",
        plot_init_vs_late_pca(
            records, chart_dir / "03_pca_realized_init_late.png", family=FAMILY_REALIZED
        ),
    )
    _add(
        "PCA maps late",
        plot_embedding_grid(
            records,
            chart_dir / "04_pca_common_late.png",
            family=FAMILY_COMMON,
            kind="pca",
            when="late",
            title="PCA of shared-question maps (late), coloured by goal",
        ),
    )
    umap_c = plot_embedding_grid(
        records,
        chart_dir / "05_umap_common_late.png",
        family=FAMILY_COMMON,
        kind="umap",
        when="late",
        title="UMAP of shared-question maps (late), coloured by goal",
    )
    _add("UMAP maps late", umap_c)
    _add(
        "PCA maps init vs late",
        plot_init_vs_late_pca(
            records, chart_dir / "06_pca_common_init_late.png", family=FAMILY_COMMON
        ),
    )
    _add(
        "PCA weights-only late",
        plot_embedding_grid(
            records,
            chart_dir / "07_pca_weights_late.png",
            family=FAMILY_WEIGHTS_ONLY,
            kind="pca",
            when="late",
            title="PCA of weights-only responses (late; goal bits off)",
        ),
    )
    _add("Δ bars", plot_delta_bars(records, chart_dir / "08_delta_bars.png"))
    _add("ARI bars", plot_ari_bars(records, chart_dir / "09_ari_bars.png"))
    _add("Φ vs Δ", plot_phi_vs_delta(records, chart_dir / "10_phi_vs_delta.png"))
    if n_vid <= 6:
        _add("distance histograms", plot_distance_hists(records, chart_dir / "11_distance_hists.png"))
        _add("grid goal vs cluster", plot_grids(records, chart_dir / "12_grid_goal_cluster.png"))
    _add("kin−foe gap", plot_kin_foe(records, chart_dir / "13_kin_foe_gap.png"))

    n_steps = None
    if records and records[0].get("config"):
        n_steps = records[0]["config"].get("n_steps")
    write_csv(records, root / "functional_summary.csv")
    report = write_report(records, root, chart_rel, n_steps=n_steps)
    print(f"Report: {report}")
    print(f"CSV:    {root / 'functional_summary.csv'}")
    print(f"Charts: {chart_dir}")
    return records


# ---------------------------------------------------------------------------
# Run original vs A then analyse
# ---------------------------------------------------------------------------

def _worker(payload: dict) -> dict:
    version = get_version(payload["version_id"])
    cfg = Config.load(payload["cfg_path"])
    run_dir = Path(payload["run_dir"])
    result = run_experiment(
        version,
        cfg,
        int(payload["seed"]),
        n_steps=int(payload["n_steps"]),
        out_dir=run_dir,
        log_every=int(payload["log_every"]),
    )
    return {
        "version_id": result["version_id"],
        "seed": result["seed"],
        "summary": result["summary"],
    }


def cmd_run(args: argparse.Namespace) -> None:
    from research.config_sources import DEFAULT_BENCHMARK, resolve_suite_configs
    from research.charts import plot_run_panel, plot_comparison_dashboard
    from research.report import write_report as write_suite_report

    if args.quick:
        versions = parse_version_list("original,A")
        seeds = [1096812628]
        n_steps = 80
        print("Quick mode: original,A · 1 seed · 80 steps")
    else:
        versions = parse_version_list(args.versions)
        seeds = (
            [int(s) for s in args.seeds.split(",")]
            if args.seeds
            else [1096812628, 42, 7]
        )
        n_steps = int(args.n_steps)

    sources = resolve_suite_configs(config=args.config, configs=None, discoveries=None)
    source = sources[0]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.name or f"functional_{'_'.join(v.id for v in versions)}_{stamp}"
    out_root = Path(args.output_dir) / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    cfg_path = out_root / "_base_config.json"
    source.cfg.save(cfg_path)

    print(f"Output → {out_root.resolve()}")
    print(f"Versions: {[v.id for v in versions]}  seeds={seeds}  steps={n_steps}")

    payloads = []
    for version in versions:
        for seed in seeds:
            payloads.append(
                {
                    "version_id": version.id,
                    "seed": int(seed),
                    "n_steps": int(n_steps),
                    "log_every": int(args.log_every),
                    "cfg_path": str(cfg_path),
                    "run_dir": str(out_root / "versions" / version.id / f"seed_{seed}"),
                }
            )

    results: list[dict] = []
    jobs = max(1, int(args.jobs))
    t_all = time.time()
    if jobs == 1:
        for p in payloads:
            print(f"=== {p['version_id']} · seed={p['seed']} ===")
            t0 = time.time()
            r = _worker(p)
            print(f"  done in {time.time() - t0:.1f}s  Φ={r['summary'].get('phi_class', float('nan')):.4f}")
            results.append(r)
    else:
        print(f"Launching {len(payloads)} runs with {jobs} workers...")
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(_worker, p): p for p in payloads}
            for fut in as_completed(futs):
                r = fut.result()
                print(
                    f"  {r['version_id']} seed={r['seed']}  "
                    f"Φ_late={r['summary'].get('phi_class_late', float('nan')):.4f}"
                )
                results.append(r)
        order = {v.id: i for i, v in enumerate(versions)}
        results.sort(key=lambda r: (order.get(r["version_id"], 999), r["seed"]))

    # Suite-style report so Φ tables sit next to the functional report.
    full_results = []
    for r in results:
        sdir = out_root / "versions" / r["version_id"] / f"seed_{r['seed']}"
        series = {k: np.asarray(v) for k, v in np.load(sdir / "series.npz").items()}
        with (sdir / "meta.json").open() as f:
            meta = json.load(f)
        item = {
            "version_id": r["version_id"],
            "version_title": VERSIONS[r["version_id"]].title,
            "seed": r["seed"],
            "series": series,
            "summary": r["summary"],
            "config": meta.get("config", {}),
            "config_id": source.id,
            "config_title": source.title,
            "goal_frac_repro": meta.get("goal_frac_repro", r["summary"].get("goal_frac_repro")),
        }
        plot_run_panel(item, sdir / "panel.png")
        full_results.append(item)
    charts = plot_comparison_dashboard(full_results, out_root / "comparison")
    write_suite_report(
        full_results,
        versions,
        out_root,
        chart_paths=charts,
        base_config_note=str(source.path),
        n_steps=int(n_steps),
        seeds=seeds,
        config_id=source.id,
        config_title=source.title,
        suite_name=run_name,
    )
    print(f"Sims done in {time.time() - t_all:.1f}s. Running functional analysis...")
    run_analysis(out_root, embed=not args.no_embed)


def cmd_analyze(args: argparse.Namespace) -> None:
    from research.functional_analysis import run_compare, run_embed

    run_compare(Path(args.root))
    if not args.no_embed:
        run_embed(Path(args.root))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m research.response_analysis",
        description="Functional class-divergence (response-function) analysis.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("run", help="Run original vs A (or given versions) then analyse")
    sp.add_argument("--versions", default="original,A")
    sp.add_argument("--seeds", default=None, help="Comma-separated. Default: 1096812628,42,7")
    sp.add_argument("--n-steps", type=int, default=1000)
    sp.add_argument("--config", type=Path, default=None, help="Base Config JSON")
    sp.add_argument("--output-dir", type=Path, default=Path("research_results"))
    sp.add_argument("--name", type=str, default=None)
    sp.add_argument("--log-every", type=int, default=0)
    sp.add_argument("--jobs", type=int, default=6)
    sp.add_argument("--quick", action="store_true")
    sp.add_argument("--no-embed", action="store_true", help="Skip PCA/UMAP")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("analyze", help="Analyse an existing suite folder")
    sp.add_argument("root", type=Path)
    sp.add_argument("--no-embed", action="store_true")
    sp.set_defaults(func=cmd_analyze)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
