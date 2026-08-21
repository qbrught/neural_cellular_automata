"""Post-suite analysis: class divergence vs the symmetric (E / w2=w3) baseline.

Reads a research.suite output folder and writes extra figures plus INSIGHTS.md.

Usage:
    python -m research.analyze_class_div research_results/<suite_name>
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from research.versions import LADDER_ORDER, ladder_sort_key


OLD_SEEDS = {1096812628, 42, 7}
PHI_LATE_HIT = 0.2
COEXIST_FLOOR = 0.10


def _style() -> None:
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


def _f(x: Any) -> float:
    if x is None or x == "":
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_summary_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        for raw in csv.DictReader(f):
            row = dict(raw)
            row["seed"] = int(float(row["seed"]))
            for k, v in list(row.items()):
                if k in ("version", "config_id", "config_title", "seed"):
                    continue
                row[k] = _f(v)
            rows.append(row)
    return rows


def _seed_group(seed: int) -> str:
    return "old" if seed in OLD_SEEDS else "new"


def _ordered_versions(rows: list[dict[str, Any]]) -> list[str]:
    present = {r["version"] for r in rows}
    return sorted(present, key=ladder_sort_key)


def _by_version(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[r["version"]].append(r)
    return out


def _paired(rows: list[dict[str, Any]], key: str, baseline: str) -> dict[str, list[float]]:
    """Per-seed (version − baseline) for ``key``."""
    by_vs: dict[tuple[str, int], float] = {
        (r["version"], r["seed"]): _f(r.get(key)) for r in rows
    }
    seeds = sorted({r["seed"] for r in rows})
    versions = _ordered_versions(rows)
    out: dict[str, list[float]] = {}
    for vid in versions:
        if vid == baseline:
            continue
        diffs = []
        for seed in seeds:
            a = by_vs.get((vid, seed), float("nan"))
            b = by_vs.get((baseline, seed), float("nan"))
            diffs.append(a - b)
        out[vid] = diffs
    return out


def _sem(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return 0.0
    return float(vals.std(ddof=1) / np.sqrt(len(vals)))


def plot_phi_bars(
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    key: str,
    ylabel: str,
    title: str,
) -> Path:
    _style()
    versions = _ordered_versions(rows)
    by_v = _by_version(rows)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    means, sems, xs = [], [], []
    rng = np.random.default_rng(0)
    for i, vid in enumerate(versions):
        vals = np.array([_f(r.get(key)) for r in by_v[vid]], float)
        means.append(np.nanmean(vals))
        sems.append(_sem(vals))
        xs.append(i)
        jitter = rng.normal(0, 0.06, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=18,
            alpha=0.45,
            color="#4c72b0",
            zorder=3,
            linewidths=0,
        )
    ax.bar(xs, means, yerr=sems, color="#4c72b0", alpha=0.35, capsize=4, zorder=2)
    ax.set_xticks(xs, versions)
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_delta_vs_baseline(
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    key: str,
    baseline: str,
    title: str,
    ylabel: str,
) -> Path | None:
    paired = _paired(rows, key, baseline)
    if not paired:
        return None
    _style()
    vids = sorted(paired, key=ladder_sort_key)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    means = [np.nanmean(paired[v]) for v in vids]
    sems = [_sem(np.array(paired[v])) for v in vids]
    colors = ["#c44e52" if m < 0 else "#4c72b0" for m in means]
    ax.bar(vids, means, yerr=sems, color=colors, alpha=0.85, capsize=4)
    ax.axhline(0, color="k", lw=0.7)
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_heatmap(
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    key: str,
    title: str,
    old_seeds: set[int],
) -> Path:
    _style()
    versions = _ordered_versions(rows)
    seeds = sorted({r["seed"] for r in rows}, key=lambda s: (s not in old_seeds, s))
    grid = np.full((len(seeds), len(versions)), np.nan)
    lookup = {(r["version"], r["seed"]): _f(r.get(key)) for r in rows}
    for i, seed in enumerate(seeds):
        for j, vid in enumerate(versions):
            grid[i, j] = lookup.get((vid, seed), float("nan"))
    fig, ax = plt.subplots(figsize=(10, max(5.5, 0.28 * len(seeds) + 1.5)))
    im = ax.imshow(grid, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xticks(range(len(versions)), versions, rotation=30, ha="right")
    labels = [f"{s}{' ·old' if s in old_seeds else ''}" for s in seeds]
    ax.set_yticks(range(len(seeds)), labels, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=key)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_old_vs_new(
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    key: str,
    title: str,
) -> Path:
    _style()
    versions = _ordered_versions(rows)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(versions))
    width = 0.38
    old_m, new_m, old_s, new_s = [], [], [], []
    for vid in versions:
        old = np.array(
            [_f(r.get(key)) for r in rows if r["version"] == vid and _seed_group(r["seed"]) == "old"],
            float,
        )
        new = np.array(
            [_f(r.get(key)) for r in rows if r["version"] == vid and _seed_group(r["seed"]) == "new"],
            float,
        )
        old_m.append(np.nanmean(old) if old.size else np.nan)
        new_m.append(np.nanmean(new) if new.size else np.nan)
        old_s.append(_sem(old))
        new_s.append(_sem(new))
    ax.bar(x - width / 2, old_m, width, yerr=old_s, label="old (unselected)", capsize=3, color="#8c8c8c")
    ax.bar(x + width / 2, new_m, width, yerr=new_s, label="new (E-discovered)", capsize=3, color="#4c72b0")
    ax.set_xticks(x, versions, rotation=30)
    ax.set_ylabel(key)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def load_residual_series(suite_dir: Path, versions: list[str], seeds: list[int]) -> dict[str, np.ndarray]:
    """Mean residual-frac curve per version, aligned to min length."""
    curves: dict[str, list[np.ndarray]] = defaultdict(list)
    for vid in versions:
        for seed in seeds:
            path = suite_dir / "versions" / vid / f"seed_{seed}" / "series.npz"
            if not path.is_file():
                continue
            s = np.load(path)
            ra = s["reproducer_alive"].astype(float)
            ea = s["eliminator_alive"].astype(float)
            alive = s["alive"].astype(float)
            g0 = float(s["goal_frac_repro"][0]) if "goal_frac_repro" in s.files else 0.5
            pred = g0 * (ra + ea)
            curves[vid].append(np.abs(ra - pred) / (alive + 1e-9))
    out: dict[str, np.ndarray] = {}
    for vid, arrs in curves.items():
        L = min(len(a) for a in arrs)
        out[vid] = np.stack([a[:L] for a in arrs], axis=0).mean(axis=0)
    return out


def plot_residual_vs_e(
    suite_dir: Path,
    rows: list[dict[str, Any]],
    out_path: Path,
) -> Path | None:
    versions = _ordered_versions(rows)
    if "E" not in versions:
        return None
    seeds = sorted({r["seed"] for r in rows})
    means = load_residual_series(suite_dir, versions, seeds)
    if "E" not in means:
        return None
    _style()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    t_e = np.arange(len(means["E"]))
    ax.plot(t_e, means["E"], color="k", lw=2.0, label="E (symmetric baseline)", zorder=5)
    colors = plt.cm.tab10.colors
    for i, vid in enumerate(versions):
        if vid == "E" or vid not in means:
            continue
        y = means[vid]
        L = min(len(y), len(means["E"]))
        ax.plot(np.arange(L), y[:L], lw=1.5, color=colors[i % 10], label=vid)
    ax.set_xlabel("step")
    ax.set_ylabel(r"mean $|r_t-f_0 a_t|/(a_t+\varepsilon)$")
    ax.set_title("Class-divergence residual over time (mean over 22 seeds)")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def write_insights(
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    n_steps: int | None,
    figure_rel: dict[str, str],
) -> Path:
    versions = _ordered_versions(rows)
    by_v = _by_version(rows)
    seeds = sorted({r["seed"] for r in rows})
    n_old = sum(1 for s in seeds if s in OLD_SEEDS)
    n_new = len(seeds) - n_old

    def mean_std(vid: str, key: str) -> tuple[float, float]:
        vals = np.array([_f(r.get(key)) for r in by_v[vid]], float)
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    def rate(vid: str, pred) -> float:
        xs = by_v[vid]
        if not xs:
            return float("nan")
        return float(sum(1 for r in xs if pred(r)) / len(xs))

    lines: list[str] = []
    lines.append("# Class divergence vs the symmetric baseline")
    lines.append("")
    lines.append(
        "Base physics is **type-symmetric** (`w2=w3=1.4974`, "
        "`research/configs/benchmark_sym_w.json`). "
        "Version **E** is typed votes on that null; on this config **A and E "
        "are the same mechanism** (E only re-averages already-equal weights). "
        "Φ is the mean fractional residual of reproducer count vs the "
        r"density-tracking prediction \(r_t \approx f_0 a_t\)."
    )
    lines.append("")
    lines.append(
        f"- Seeds: **{len(seeds)}** "
        f"({n_old} historical / unselected, {n_new} discovered under E). "
        f"Discovered seeds were chosen because they looked non-null **under E**, "
        f"so A/E Φ on the new bank is upward-biased; old seeds are the fairer null check."
    )
    if n_steps:
        lines.append(f"- Horizon: **T={n_steps}**")
    lines.append(f"- Versions: {', '.join(f'`{v}`' for v in versions)}")
    lines.append("")

    # Headline table
    headers = [
        "version",
        "Φ",
        "Φ_late",
        "|ra res|",
        "corr(r,fa)",
        "corr(r,e)",
        "min(r,e)/a late",
        f"P(Φ_late>{PHI_LATE_HIT})",
        "P(two-process)",
        "alive late",
    ]
    body = []
    for vid in versions:
        phi, phi_s = mean_std(vid, "phi_class")
        late, late_s = mean_std(vid, "phi_class_late")
        res, _ = mean_std(vid, "mean_abs_residual_ra")
        crd, _ = mean_std(vid, "corr_ra_density")
        cre, _ = mean_std(vid, "corr_ra_ea")
        mtf, _ = mean_std(vid, "late_min_type_frac")
        alive, _ = mean_std(vid, "mean_alive_late")
        hit = rate(vid, lambda r: _f(r.get("phi_class_late")) > PHI_LATE_HIT)
        two = rate(
            vid,
            lambda r: (
                _f(r.get("phi_class_late")) > PHI_LATE_HIT
                and _f(r.get("late_min_type_frac")) > COEXIST_FLOOR
            ),
        )
        body.append(
            [
                f"`{vid}`",
                f"{phi:.4f}±{phi_s:.3f}",
                f"{late:.4f}±{late_s:.3f}",
                f"{res:.1f}",
                f"{crd:.3f}",
                f"{cre:.3f}",
                f"{mtf:.3f}",
                f"{hit:.2f}",
                f"{two:.2f}",
                f"{alive:.0f}",
            ]
        )
    lines.append("## Version ladder (mean ± std over seeds)")
    lines.append("")
    lines.append(
        "Two-process = Φ_late "
        f"> {PHI_LATE_HIT} **and** late min(r,e)/a > {COEXIST_FLOOR} "
        "(divergence with both types still present, not pure attrition)."
    )
    lines.append("")
    lines.append(_md_table(headers, body))
    lines.append("")

    # Deltas vs E
    if "E" in by_v:
        lines.append("## Residuals relative to E (paired, same seeds)")
        lines.append("")
        lines.append(
            "Each cell is mean over seeds of (version − E) on that seed. "
            "Positive ΔΦ means *more* class divergence than typed votes on "
            "symmetric weights."
        )
        lines.append("")
        d_headers = ["version − E", "ΔΦ", "ΔΦ_late", "Δ|ra res|", "Δ corr(r,fa)", "Δ min-type late", "frac seeds Φ_late > E"]
        d_body = []
        phi_d = _paired(rows, "phi_class", "E")
        late_d = _paired(rows, "phi_class_late", "E")
        res_d = _paired(rows, "mean_abs_residual_ra", "E")
        crd_d = _paired(rows, "corr_ra_density", "E")
        mtf_d = _paired(rows, "late_min_type_frac", "E")
        for vid in versions:
            if vid == "E":
                continue
            pd = np.array(phi_d.get(vid, []), float)
            ld = np.array(late_d.get(vid, []), float)
            rd = np.array(res_d.get(vid, []), float)
            cd = np.array(crd_d.get(vid, []), float)
            md = np.array(mtf_d.get(vid, []), float)
            frac = float(np.nanmean(ld > 0)) if ld.size else float("nan")
            d_body.append(
                [
                    f"`{vid}` − `E`",
                    f"{np.nanmean(pd):+.4f}",
                    f"{np.nanmean(ld):+.4f}",
                    f"{np.nanmean(rd):+.1f}",
                    f"{np.nanmean(cd):+.3f}",
                    f"{np.nanmean(md):+.3f}",
                    f"{frac:.2f}",
                ]
            )
        lines.append(_md_table(d_headers, d_body))
        lines.append("")

        # A vs E identity
        if "A" in by_v:
            a_minus_e = np.array(phi_d.get("A", []), float)
            lines.append("### A vs E identity check")
            lines.append("")
            lines.append(
                f"On already-symmetric weights, E only copies w2=w3. "
                f"Mean ΔΦ (A−E) = **{np.nanmean(a_minus_e):+.6f}** "
                f"(should be ~0; nonzero is seed-noise in apply() rounding only if any)."
            )
            lines.append("")

    # Old vs new
    lines.append("## Old vs new seeds (selection bias)")
    lines.append("")
    lines.append(
        "New seeds were VLM-saved because they showed learning-driven "
        "R-holds / E-declines **under E**. If A/E Φ_late is large on new "
        "seeds but small on old seeds, the seed bank is doing the work, "
        "not a new mechanism."
    )
    lines.append("")
    on_headers = ["version", "Φ_late old", "Φ_late new", "Δ (new−old)", "two-process old", "two-process new"]
    on_body = []
    for vid in versions:
        old = np.array(
            [_f(r.get("phi_class_late")) for r in by_v[vid] if r["seed"] in OLD_SEEDS],
            float,
        )
        new = np.array(
            [_f(r.get("phi_class_late")) for r in by_v[vid] if r["seed"] not in OLD_SEEDS],
            float,
        )
        old_two = rate(
            vid,
            lambda r: r["seed"] in OLD_SEEDS
            and _f(r.get("phi_class_late")) > PHI_LATE_HIT
            and _f(r.get("late_min_type_frac")) > COEXIST_FLOOR,
        )
        new_two = [
            r
            for r in by_v[vid]
            if r["seed"] not in OLD_SEEDS
        ]
        n_new_two = (
            sum(
                1
                for r in new_two
                if _f(r.get("phi_class_late")) > PHI_LATE_HIT
                and _f(r.get("late_min_type_frac")) > COEXIST_FLOOR
            )
            / len(new_two)
            if new_two
            else float("nan")
        )
        on_body.append(
            [
                f"`{vid}`",
                f"{np.nanmean(old):.3f}" if old.size else "n/a",
                f"{np.nanmean(new):.3f}" if new.size else "n/a",
                f"{(np.nanmean(new) - np.nanmean(old)):+.3f}" if old.size and new.size else "n/a",
                f"{old_two:.2f}" if old.size else "n/a",
                f"{n_new_two:.2f}" if new_two else "n/a",
            ]
        )
    lines.append(_md_table(on_headers, on_body))
    lines.append("")

    # Attrition vs two-process
    lines.append("## Is Φ_late two processes or one type dying?")
    lines.append("")
    lines.append(
        "High Φ_late with late min(r,e)/a ≈ 0 is **attrition** "
        "(competitive exclusion), not stable coexistence of two dynamical roles."
    )
    lines.append("")
    at_headers = ["version", "mean Φ_late", "mean min-type late", "P(type extinct)", "P(Φ_late hit | both alive)"]
    at_body = []
    for vid in versions:
        late, _ = mean_std(vid, "phi_class_late")
        mtf, _ = mean_std(vid, "late_min_type_frac")
        ext = rate(
            vid,
            lambda r: np.isfinite(_f(r.get("type_extinct_step")))
            and _f(r.get("type_extinct_step")) >= 0,
        )
        both = [
            r
            for r in by_v[vid]
            if _f(r.get("late_min_type_frac")) > COEXIST_FLOOR
        ]
        hit_both = (
            sum(1 for r in both if _f(r.get("phi_class_late")) > PHI_LATE_HIT) / len(both)
            if both
            else float("nan")
        )
        at_body.append(
            [
                f"`{vid}`",
                f"{late:.3f}",
                f"{mtf:.3f}",
                f"{ext:.2f}",
                f"{hit_both:.2f}" if both else "n/a",
            ]
        )
    lines.append(_md_table(at_headers, at_body))
    lines.append("")

    # Inheritance / colonization
    if "C" in by_v or "C_only" in by_v or "D" in by_v:
        lines.append("## Goal inheritance (C / C_only / D)")
        lines.append("")
        g_headers = ["version", "g_frac drift", "mean|Δg_alive|", "corr(r,fa)", "alive late"]
        g_body = []
        for vid in versions:
            if vid not in ("C", "C_only", "D", "D_fixed"):
                continue
            drift, ds = mean_std(vid, "goal_frac_repro_drift")
            churn, _ = mean_std(vid, "mean_abs_goal_frac_delta")
            crd, _ = mean_std(vid, "corr_ra_density")
            alive, _ = mean_std(vid, "mean_alive_late")
            g_body.append(
                [
                    f"`{vid}`",
                    f"{drift:+.3f}±{ds:.3f}",
                    f"{churn:.4f}",
                    f"{crd:.3f}",
                    f"{alive:.0f}",
                ]
            )
        if g_body:
            lines.append(_md_table(g_headers, g_body))
            lines.append("")
            lines.append(
                "If corr(r, fa) returns to ~1 while g_frac drifts, colonization "
                "rewrites the type map but counts still track *current* density "
                "— Φ vs *initial* f_0 then inflates without two live processes."
            )
            lines.append("")

    # Data-driven takeaways
    lines.append("## Takeaways (computed, not hypothesized)")
    lines.append("")
    if "original" in by_v and "E" in by_v:
        o_late, _ = mean_std("original", "phi_class_late")
        e_late, _ = mean_std("E", "phi_class_late")
        lines.append(
            f"1. **Null vs typed votes on symmetric physics:** "
            f"original Φ_late = {o_late:.3f}, E Φ_late = {e_late:.3f} "
            f"(Δ = {e_late - o_late:+.3f}). "
            + (
                "Typed votes still move counts off the density-tracking null even with w2=w3."
                if e_late - o_late > 0.05
                else "Typed votes do **not** clearly beat the density null on this seed bank."
            )
        )
        lines.append("")
    if "E" in by_v and "F" in by_v:
        e_two = rate(
            "E",
            lambda r: _f(r.get("phi_class_late")) > PHI_LATE_HIT
            and _f(r.get("late_min_type_frac")) > COEXIST_FLOOR,
        )
        f_two = rate(
            "F",
            lambda r: _f(r.get("phi_class_late")) > PHI_LATE_HIT
            and _f(r.get("late_min_type_frac")) > COEXIST_FLOOR,
        )
        lines.append(
            f"2. **Soft coexistence (F) vs E:** two-process rate "
            f"E={e_two:.2f}, F={f_two:.2f}. "
            + (
                "F increases the fraction of seeds that keep both types while remaining divergent."
                if f_two > e_two + 0.05
                else "F does not clearly convert attrition-Φ into two-process coexistence."
            )
        )
        lines.append("")
    if "B" in by_v and "E" in by_v:
        b_d = np.nanmean(_paired(rows, "phi_class_late", "E").get("B", [np.nan]))
        lines.append(
            f"3. **Predator–prey loss (B) vs E:** ΔΦ_late = {b_d:+.3f}. "
            + (
                "B *reduces* late class divergence relative to typed votes alone."
                if b_d < -0.02
                else (
                    "B *raises* late class divergence vs typed votes alone."
                    if b_d > 0.02
                    else "B is within noise of E on Φ_late."
                )
            )
        )
        lines.append("")
    if "C" in by_v and "E" in by_v:
        c_d = np.nanmean(_paired(rows, "phi_class_late", "E").get("C", [np.nan]))
        c_corr, _ = mean_std("C", "corr_ra_density")
        lines.append(
            f"4. **Inheritance stack (C) vs E:** ΔΦ_late = {c_d:+.3f}, "
            f"corr(r, fa) = {c_corr:.3f}. "
            + (
                "High corr(r,fa) under C means Φ vs *initial* frac is mostly colonization rewriting f, not two concurrent density processes."
                if c_corr > 0.9
                else "C keeps type counts from tracking the init-frac density prediction."
            )
        )
        lines.append("")

    if figure_rel:
        lines.append("## Extra figures")
        lines.append("")
        for key, rel in figure_rel.items():
            lines.append(f"### {key}")
            lines.append("")
            lines.append(f"![{key}]({rel})")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def analyze_suite(suite_dir: Path) -> Path:
    suite_dir = Path(suite_dir)
    csv_path = suite_dir / "summary.csv"
    if not csv_path.is_file():
        alt = suite_dir / "summary_all.csv"
        if alt.is_file():
            csv_path = alt
        else:
            raise FileNotFoundError(f"No summary.csv in {suite_dir}")
    rows = load_summary_csv(csv_path)
    if not rows:
        raise ValueError(f"Empty summary: {csv_path}")

    n_steps = None
    manifest = suite_dir / "manifest.json"
    if manifest.is_file():
        n_steps = json.loads(manifest.read_text()).get("n_steps")

    fig_dir = suite_dir / "analysis"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figure_rel: dict[str, str] = {}

    def _add(key: str, path: Path | None) -> None:
        if path is not None:
            figure_rel[key] = path.relative_to(suite_dir).as_posix()

    _add(
        "phi_swarm",
        plot_phi_bars(
            rows,
            fig_dir / "phi_swarm.png",
            key="phi_class",
            ylabel=r"$\Phi$",
            title="Class divergence Φ (bars = mean±SEM, dots = seeds)",
        ),
    )
    _add(
        "phi_late_swarm",
        plot_phi_bars(
            rows,
            fig_dir / "phi_late_swarm.png",
            key="phi_class_late",
            ylabel=r"$\Phi_{\mathrm{late}}$",
            title="Late class divergence Φ_late (bars = mean±SEM, dots = seeds)",
        ),
    )
    _add(
        "delta_phi_late_vs_E",
        plot_delta_vs_baseline(
            rows,
            fig_dir / "delta_phi_late_vs_E.png",
            key="phi_class_late",
            baseline="E",
            title="ΔΦ_late vs E (symmetric typed-vote baseline)",
            ylabel=r"$\Phi_{\mathrm{late}}(v)-\Phi_{\mathrm{late}}(E)$",
        ),
    )
    _add(
        "delta_phi_late_vs_original",
        plot_delta_vs_baseline(
            rows,
            fig_dir / "delta_phi_late_vs_original.png",
            key="phi_class_late",
            baseline="original",
            title="ΔΦ_late vs original (indiscriminate-vote null)",
            ylabel=r"$\Phi_{\mathrm{late}}(v)-\Phi_{\mathrm{late}}(\mathrm{original})$",
        ),
    )
    _add(
        "heatmap_phi_late",
        plot_heatmap(
            rows,
            fig_dir / "heatmap_phi_late.png",
            key="phi_class_late",
            title="Φ_late by seed × version (old seeds at top)",
            old_seeds=OLD_SEEDS,
        ),
    )
    _add(
        "old_vs_new_phi_late",
        plot_old_vs_new(
            rows,
            fig_dir / "old_vs_new_phi_late.png",
            key="phi_class_late",
            title="Φ_late: historical seeds vs E-discovered seed bank",
        ),
    )
    _add(
        "residual_vs_E_timeseries",
        plot_residual_vs_e(suite_dir, rows, fig_dir / "residual_vs_E_timeseries.png"),
    )

    insights = write_insights(
        rows,
        suite_dir / "INSIGHTS.md",
        n_steps=n_steps,
        figure_rel=figure_rel,
    )
    print(f"Wrote {insights}")
    print(f"Figures in {fig_dir}")
    return insights


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Class-divergence analysis vs E / w2=w3")
    p.add_argument("suite_dir", type=Path, help="research_results/<suite_name>")
    args = p.parse_args(argv)
    analyze_suite(args.suite_dir)


if __name__ == "__main__":
    main()
