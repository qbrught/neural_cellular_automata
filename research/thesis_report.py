"""Four-chunk thesis reports for one pipeline comparison."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from research.charts import plot_comparison_dashboard
from research.compare import markdown_comparison_table, write_summary_csv
from research.comparisons import Comparison
from research.protocol import (
    BOOTSTRAP_ITERS,
    BOOTSTRAP_SEED,
    COEXIST_FLOOR,
    PHI_LATE_HIT,
    SEEDS_HISTORICAL,
)
from research.spatial import render_comparison_frames, render_run_frames
from research.stats import format_paired, paired_delta_test
from research.versions import get_version

_METRIC_LABELS: dict[str, str] = {
    "phi_class": "Φ",
    "phi_class_late": "Φ_late",
    "corr_ra_ea": "corr(ra,ea)",
    "corr_ra_density": "corr(ra, dens)",
    "mean_abs_residual_ra": "|ra residual|",
    "late_min_type_frac": "min(r,e)/a late",
    "mean_alive_late": "alive late",
    "late_R_vote_disc": "R vote disc late",
    "late_E_vote_disc": "E vote disc late",
    "late_death_rate_cross_minus_same": "death gap late",
    "late_death_rate_E_cross_minus_same": "E death gap late",
    "goal_frac_repro_drift": "g_frac drift",
    "mean_abs_goal_frac_delta": "mean|Δg_alive|",
    "late_f_signal_type_gap": "f-signal gap late",
    "late_s_norm_type_gap": "||s|| gap late",
    "late_frac_alive_low_kappa": "frac in low-κ late",
    "late_kappa_edge_mean": "κ_edge late",
    "late_eta_mean_alive": "η alive late",
    "late_min_soft_rho": "min soft ρ late",
    "mean_coexistence_barrier": "mean B",
}


def _f(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _two_process(summary: dict[str, Any]) -> bool:
    return (
        np.isfinite(_f(summary.get("phi_class_late")))
        and _f(summary.get("phi_class_late")) > PHI_LATE_HIT
        and np.isfinite(_f(summary.get("late_min_type_frac")))
        and _f(summary.get("late_min_type_frac")) > COEXIST_FLOOR
    )


def _by_arm_seed(results: list[dict[str, Any]]) -> dict[tuple[str, int], dict]:
    return {(r["version_id"], int(r["seed"])): r for r in results}


def _aligned(results: list[dict[str, Any]], off: str, on: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    by = _by_arm_seed(results)
    seeds = sorted({int(r["seed"]) for r in results})
    a, b = [], []
    for s in seeds:
        ro = by.get((off, s))
        rn = by.get((on, s))
        if ro is None or rn is None:
            continue
        a.append(_f(ro["summary"].get(key)))
        b.append(_f(rn["summary"].get(key)))
    return np.asarray(a, float), np.asarray(b, float)


def _mean_sem(results: list[dict[str, Any]], arm: str, key: str) -> tuple[float, float]:
    vals = np.array(
        [_f(r["summary"].get(key)) for r in results if r["version_id"] == arm],
        float,
    )
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    mean = float(vals.mean())
    sem = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return mean, sem


def _rate(results: list[dict[str, Any]], arm: str, pred) -> float:
    rows = [r for r in results if r["version_id"] == arm]
    if not rows:
        return float("nan")
    return float(sum(1 for r in rows if pred(r)) / len(rows))


def _insights(comp: Comparison, results: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    off, on = comp.off, comp.on
    if off and on:
        a, b = _aligned(results, off, on, "phi_class_late")
        st = paired_delta_test(a, b, n_iter=BOOTSTRAP_ITERS, seed=BOOTSTRAP_SEED)
        lines.append(
            f"- **Φ_late ({on} − {off}):** {format_paired(st)}. "
            + (
                "On raises late class divergence vs off."
                if np.isfinite(st["mean_delta"]) and st["mean_delta"] > 0.02
                else (
                    "On *lowers* late class divergence vs off."
                    if np.isfinite(st["mean_delta"]) and st["mean_delta"] < -0.02
                    else "ΔΦ_late is within a small band of off (see CI)."
                )
            )
        )
        r_off = _rate(results, off, lambda r: _two_process(r["summary"]))
        r_on = _rate(results, on, lambda r: _two_process(r["summary"]))
        lines.append(
            f"- **Two-process rate** (Φ_late > {PHI_LATE_HIT} and "
            f"min(r,e)/a late > {COEXIST_FLOOR}): "
            f"{off}={r_off:.2f}, {on}={r_on:.2f}."
        )
        m_off, _ = _mean_sem(results, off, "late_min_type_frac")
        m_on, _ = _mean_sem(results, on, "late_min_type_frac")
        if np.isfinite(m_on) and m_on < COEXIST_FLOOR:
            lines.append(
                f"- **Attrition caveat:** mean min-type late on `{on}` is "
                f"{m_on:.3f} < {COEXIST_FLOOR}. Large Φ may be one type dying."
            )
        if "corr_ra_density" in (comp.primary_metrics + ("corr_ra_density",)):
            c_on, _ = _mean_sem(results, on, "corr_ra_density")
            if np.isfinite(c_on) and c_on > 0.95:
                lines.append(
                    f"- **Density-tracking caveat:** corr(ra, dens) on `{on}` "
                    f"is {c_on:.3f}. Counts still track f0 × density; Φ vs "
                    f"initial f0 can inflate under colonization rewrite."
                )
        for key in comp.primary_metrics:
            aa, bb = _aligned(results, off, on, key)
            pst = paired_delta_test(aa, bb, n_iter=BOOTSTRAP_ITERS, seed=BOOTSTRAP_SEED)
            lab = _METRIC_LABELS.get(key, key)
            lines.append(f"- **{lab} ({on} − {off}):** {format_paired(pst)}.")
        if comp.kind == "identity" and np.isfinite(st["mean_delta"]):
            lines.append(
                f"- **Identity:** |ΔΦ_late| = {abs(st['mean_delta']):.6f} "
                "(expect ~0 on already-symmetric weights)."
            )
    elif off:
        # Sweep: each non-off arm vs off on Φ_late
        arms = [a.id for a in comp.arms if a.id != off]
        lines.append(f"Sweep vs `{off}` on Φ_late:")
        for arm in arms:
            a, b = _aligned(results, off, arm, "phi_class_late")
            st = paired_delta_test(a, b, n_iter=BOOTSTRAP_ITERS, seed=BOOTSTRAP_SEED)
            lines.append(f"- `{arm}`: {format_paired(st)}")
    else:
        by: dict[str, list[float]] = defaultdict(list)
        for r in results:
            by[r["version_id"]].append(_f(r["summary"].get("phi_class_late")))
        ranked = sorted(
            ((vid, float(np.nanmean(v))) for vid, v in by.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )
        if ranked:
            lines.append(
                "Ladder Φ_late means (high → low): "
                + ", ".join(f"`{v}`={m:.3f}" for v, m in ranked)
            )
    lines.append("")
    lines.append(f"**Registered hypothesis:** {comp.hypothesis}")
    lines.append("")
    lines.append("_Edit this section with what the frames showed; the bullets above are computed._")
    return lines


def write_notes(comp: Comparison, out_dir: Path, *, seeds: list[int]) -> Path:
    hist = [s for s in seeds if s in SEEDS_HISTORICAL]
    lines = [
        f"# Notes — {comp.id}",
        "",
        "Fill this while looking at `frames/` and the live UI (`python server.py`) "
        "loaded with the matching `cache/.../config.json` and seed.",
        "",
        f"Visual prompt: {comp.visual_prompt}",
        "",
        f"Suggested seeds: historical {hist or list(seeds[:3])}",
        "",
        "## Checklist",
        "",
        f"- [ ] {comp.off or 'baseline'} visual dynamics",
        f"- [ ] {comp.on or 'each arm'} visual dynamics",
        "- [ ] What the metrics miss (texture, fronts, oscillations)",
        "- [ ] Attrition vs two living colours",
        "",
        "## Free notes",
        "",
        "_Write here._",
        "",
    ]
    path = Path(out_dir) / "NOTES.md"
    path.write_text("\n".join(lines))
    return path


def write_comparison_report(
    comp: Comparison,
    results: list[dict[str, Any]],
    out_dir: Path,
    *,
    n_steps: int,
    seeds: list[int],
    visual_seeds: list[int],
    suite_name: str,
    chart_paths: dict[str, Path],
    frame_paths: list[Path],
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(results, out_dir / "summary.csv")
    write_notes(comp, out_dir, seeds=seeds)

    off, on = comp.off, comp.on
    rel = out_dir.resolve()

    def _rel(p: Path) -> str:
        try:
            return Path(p).resolve().relative_to(rel).as_posix()
        except ValueError:
            return Path(p).as_posix()

    lines: list[str] = []
    lines.append(f"# {comp.title}")
    lines.append("")
    lines.append(f"- Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- Pipeline run: **{suite_name}**")
    lines.append(f"- Kind: `{comp.kind}` · config `{comp.config_id}`")
    lines.append(f"- Steps: **{n_steps}** · seeds: `{seeds}`")
    lines.append(f"- Arms: {', '.join(f'`{a.id}`' for a in comp.arms)}")
    if off and on:
        lines.append(f"- Isolation: `{off}` → `{on}`")
    lines.append("")
    lines.append("Artifacts: `NOTES.md` (manual), `summary.csv`, `comparison/`, `frames/`.")
    lines.append("")

    # ---- 1. Motivation
    lines.append("## 1. Motivation and explanation of changes")
    lines.append("")
    lines.append(comp.motivation)
    lines.append("")
    for a in comp.arms:
        try:
            spec = a.spec()
        except Exception:
            continue
        extra = f" overrides={a.spec_overrides}" if a.spec_overrides else ""
        lines.append(
            f"- `{a.id}` ← version `{a.version}`{extra}: {spec.description}"
        )
        lines.append("")
    if on:
        try:
            lines.append(f"**Hypothesis ({get_version(comp.arms[-1].version).id}):** {comp.hypothesis}")
        except Exception:
            lines.append(f"**Hypothesis:** {comp.hypothesis}")
        lines.append("")

    # ---- 2. Visual
    lines.append("## 2. Visual / manual observations")
    lines.append("")
    lines.append(comp.visual_prompt)
    lines.append("")
    lines.append("Fill [`NOTES.md`](NOTES.md) from these frames (and the UI on the same config+seed).")
    lines.append("")
    if frame_paths:
        for p in frame_paths:
            lines.append(f"![{p.stem}]({_rel(p)})")
            lines.append("")
    else:
        lines.append("_No paired frames for the visual seed set (check cache `frames.npz`)._")
        lines.append("")

    # ---- 3. Objective
    lines.append("## 3. Objective / mathematical assessment")
    lines.append("")
    lines.append(
        f"Two-process = Φ_late > {PHI_LATE_HIT} **and** "
        f"late min(r,e)/a > {COEXIST_FLOOR}."
    )
    lines.append("")
    lines.append("### Mean metrics")
    lines.append("")
    lines.append(markdown_comparison_table(results))
    lines.append("")

    headline = [
        "phi_class",
        "phi_class_late",
        "late_min_type_frac",
        "mean_alive_late",
        "corr_ra_density",
        *comp.primary_metrics,
    ]
    seen: set[str] = set()
    keys = []
    for k in headline:
        if k not in seen:
            seen.add(k)
            keys.append(k)

    if off and on:
        lines.append("### Paired deltas (same seeds, on − off)")
        lines.append("")
        lines.append("| metric | result |")
        lines.append("| --- | --- |")
        for key in keys:
            a, b = _aligned(results, off, on, key)
            st = paired_delta_test(a, b, n_iter=BOOTSTRAP_ITERS, seed=BOOTSTRAP_SEED)
            lab = _METRIC_LABELS.get(key, key)
            lines.append(f"| {lab} | {format_paired(st)} |")
        lines.append("")
        r_off = _rate(results, off, lambda r: _two_process(r["summary"]))
        r_on = _rate(results, on, lambda r: _two_process(r["summary"]))
        lines.append(f"Two-process rate: `{off}` = {r_off:.2f}, `{on}` = {r_on:.2f}.")
        lines.append("")
    elif off:
        lines.append("### Paired ΔΦ_late vs off")
        lines.append("")
        lines.append("| arm | Φ_late vs off | two-process |")
        lines.append("| --- | --- | --- |")
        for arm in [a.id for a in comp.arms if a.id != off]:
            a, b = _aligned(results, off, arm, "phi_class_late")
            st = paired_delta_test(a, b, n_iter=BOOTSTRAP_ITERS, seed=BOOTSTRAP_SEED)
            rp = _rate(results, arm, lambda r: _two_process(r["summary"]))
            lines.append(f"| `{arm}` | {format_paired(st)} | {rp:.2f} |")
        lines.append("")

    if chart_paths:
        lines.append("### Charts")
        lines.append("")
        for key, p in chart_paths.items():
            lines.append(f"#### {key}")
            lines.append("")
            lines.append(f"![{key}]({_rel(p)})")
            lines.append("")

    # ---- 4. Insights
    lines.append("## 4. Conclusions / insights")
    lines.append("")
    lines.extend(_insights(comp, results))
    lines.append("")

    path = out_dir / "REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_index(
    out_root: Path,
    *,
    suite_name: str,
    n_steps: int,
    seeds: list[int],
    comparisons: list[Comparison],
    reports: dict[str, Path],
) -> Path:
    lines = [
        f"# Thesis pipeline — {suite_name}",
        "",
        f"- Steps: **{n_steps}**",
        f"- Seeds ({len(seeds)}): `{seeds}`",
        f"- Comparisons: {', '.join(f'`{c.id}`' for c in comparisons)}",
        "",
        "Protocol: [`protocol.json`](protocol.json). "
        "Per-letter reports use the four-chunk template "
        "(motivation, visual, objective, insights).",
        "",
        "| letter | title | report |",
        "| --- | --- | --- |",
    ]
    for c in comparisons:
        rel = reports[c.id].relative_to(out_root).as_posix() if c.id in reports else ""
        lines.append(f"| `{c.id}` | {c.title} | [{rel}]({rel}) |")
    lines.append("")
    lines.append("## How to write")
    lines.append("")
    lines.append("1. Open `comparisons/<id>/REPORT.md` section 1 (already filled).")
    lines.append("2. Look at `frames/` + UI; fill `NOTES.md`; paste intuition into section 2.")
    lines.append("3. Section 3 is computed (paired tests + charts). Quote it, don't re-run by hand.")
    lines.append("4. Edit section 4 with the call: did the *intended* quantity move?")
    lines.append("")
    path = Path(out_root) / "INDEX.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def build_comparison_artifacts(
    comp: Comparison,
    results: list[dict[str, Any]],
    out_dir: Path,
    *,
    n_steps: int,
    seeds: list[int],
    visual_seeds: list[int],
    suite_name: str,
) -> Path:
    out_dir = Path(out_dir)
    chart_dir = out_dir / "comparison"
    chart_paths = plot_comparison_dashboard(
        results,
        chart_dir,
        title_prefix=comp.id,
    )
    for r in results:
        rd = r.get("run_dir")
        if rd:
            render_run_frames(Path(rd))
    frame_paths: list[Path] = []
    if comp.off and comp.on:
        frame_paths = render_comparison_frames(
            results,
            out_dir / "frames",
            off=comp.off,
            on=comp.on,
            visual_seeds=tuple(visual_seeds),
        )
    elif len(comp.arms) >= 2:
        # Montage first two arms so sweeps/ladders still have a picture.
        frame_paths = render_comparison_frames(
            results,
            out_dir / "frames",
            off=comp.arms[0].id,
            on=comp.arms[1].id,
            visual_seeds=tuple(visual_seeds),
        )
    return write_comparison_report(
        comp,
        results,
        out_dir,
        n_steps=n_steps,
        seeds=seeds,
        visual_seeds=visual_seeds,
        suite_name=suite_name,
        chart_paths=chart_paths,
        frame_paths=frame_paths,
    )
