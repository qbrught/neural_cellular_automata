"""Functional divergence on a mix-divergence suite / thesis pipeline cache.

Two passes, matching the lettered mix pipeline:

  python -m research.functional_analysis compare research_results/<run>
      scores / paired tests / letter reports (no PCA/UMAP)

  python -m research.functional_analysis embed research_results/<run>
      PCA/UMAP and cluster grids (visual seeds only if protocol.json)

  python -m research.functional_analysis analyze research_results/<run>
      compare then embed

Also: python -m research.pipeline functional|embed <run>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research.comparisons import COMPARISONS, Comparison, get_comparison
from research.protocol import (
    BOOTSTRAP_ITERS,
    BOOTSTRAP_SEED,
    COEXIST_FLOOR,
    PHI_LATE_HIT,
    SEEDS_VISUAL,
)
from research.response_analysis import (
    analyze_run_dir,
    discover_run_dirs,
    write_csv,
    write_report as write_suite_functional_report,
)
from research.stats import format_paired, paired_delta_test

MAP_DELTA_HIT = 0.4
MAP_ARI_HIT = 0.25

_FUNC_LABELS: dict[str, str] = {
    "phi_class_late": "Φ_late (mix)",
    "late_min_type_frac": "min type late",
    "delta_common_all_late": "Δ maps late",
    "ari_common_all_late": "ARI maps late",
    "delta_common_all_learned": "Δ maps learned",
    "delta_realized_all_late": "Δ agent late",
    "ari_realized_all_late": "ARI agent late",
    "delta_weights_only_all_late": "Δ weights late",
}

_HEADLINE = tuple(_FUNC_LABELS.keys())


def _f(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _learned_or_scalar(rec: dict[str, Any], key: str) -> float:
    if key in rec.get("learned", {}):
        return _f(rec["learned"][key])
    return _f(rec.get("scalars", {}).get(key))


def mix_two_process(rec: dict[str, Any]) -> bool:
    phi = _f(rec["scalars"].get("phi_class_late"))
    mn = _f(rec["scalars"].get("late_min_type_frac"))
    return np.isfinite(phi) and phi > PHI_LATE_HIT and np.isfinite(mn) and mn > COEXIST_FLOOR


def map_split(rec: dict[str, Any]) -> bool:
    d = _learned_or_scalar(rec, "delta_common_all_late")
    ari = _learned_or_scalar(rec, "ari_common_all_late")
    mn = _f(rec["scalars"].get("late_min_type_frac"))
    return (
        np.isfinite(d)
        and d > MAP_DELTA_HIT
        and np.isfinite(ari)
        and ari > MAP_ARI_HIT
        and np.isfinite(mn)
        and mn > COEXIST_FLOOR
    )


def load_protocol(root: Path) -> dict[str, Any] | None:
    p = Path(root) / "protocol.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def comparisons_for_root(root: Path, records: list[dict[str, Any]]) -> list[Comparison]:
    proto = load_protocol(root)
    if proto and proto.get("comparisons"):
        out: list[Comparison] = []
        for c in proto["comparisons"]:
            try:
                out.append(get_comparison(c["id"]))
            except KeyError:
                continue
        if out:
            return out
    present = {r["version_id"] for r in records}
    matched = [
        c
        for c in COMPARISONS
        if all(a.id in present for a in c.arms)
    ]
    if matched:
        return matched
    # Fallback: one ladder-like block of whatever arms exist.
    from research.comparisons import Arm

    arms = tuple(Arm(id=v, version=v) for v in sorted(present))
    off = "original" if "original" in present else None
    return [
        Comparison(
            id="observed",
            title="Observed arms",
            arms=arms,
            off=off,
            on=None,
            kind="ladder",
            default=False,
        )
    ]


def evaluate_records(root: Path, *, embed: bool) -> list[dict[str, Any]]:
    dirs = discover_run_dirs(root)
    print(f"Evaluating {len(dirs)} snapshots under {root} (embed={embed})")
    records: list[dict[str, Any]] = []
    for d in dirs:
        t0 = time.time()
        rec = analyze_run_dir(d, embed=embed)
        L = rec["learned"]
        print(
            f"  {rec.get('config_id') or '-'} {rec['version_id']} "
            f"seed={rec['seed']}  "
            f"Φ_late={rec['scalars']['phi_class_late']:.3f}  "
            f"Δ_maps={L['delta_common_all_late']:.3f}  "
            f"ARI_maps={L['ari_common_all_late']:.3f}  "
            f"[{time.time() - t0:.1f}s]"
        )
        records.append(rec)
    return records


def _aligned(
    records: list[dict[str, Any]],
    off: str,
    on: str,
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    by = {(r["version_id"], int(r["seed"])): r for r in records}
    seeds = sorted({int(r["seed"]) for r in records})
    a, b = [], []
    for s in seeds:
        ro, rn = by.get((off, s)), by.get((on, s))
        if ro is None or rn is None:
            continue
        a.append(_learned_or_scalar(ro, key))
        b.append(_learned_or_scalar(rn, key))
    return np.asarray(a, float), np.asarray(b, float)


def _rate(records: list[dict[str, Any]], arm: str, pred) -> float:
    rows = [r for r in records if r["version_id"] == arm]
    if not rows:
        return float("nan")
    return float(np.mean([float(bool(pred(r))) for r in rows]))


def _mean_sem(records: list[dict[str, Any]], arm: str, key: str) -> tuple[float, float]:
    vals = np.array(
        [_learned_or_scalar(r, key) for r in records if r["version_id"] == arm],
        float,
    )
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    mean = float(vals.mean())
    sem = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return mean, sem


def _fmt(x: float, spec: str = ".3f") -> str:
    if x is None or not np.isfinite(float(x) if x is not None else np.nan):
        return "nan"
    return format(float(x), spec)


def write_letter_report(
    comp: Comparison,
    records: list[dict[str, Any]],
    out_dir: Path,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = [a.id for a in comp.arms]
    rows = [r for r in records if r["version_id"] in arms]
    lines: list[str] = []
    lines.append(f"# Functional divergence — {comp.title}")
    lines.append("")
    lines.append(
        "Mix Φ is the census. Functional Δ is cell-level response geometry "
        r"($\Delta=\mathcal{E}/2$). **Maps** (`common`) are shared-question "
        "vectors (learned split). **Agent** (`realized`) is own-goal output "
        "(init already separates). Map-split = "
        f"Δ_maps > {MAP_DELTA_HIT} **and** ARI_maps > {MAP_ARI_HIT} "
        f"**and** min-type > {COEXIST_FLOOR}."
    )
    lines.append("")
    lines.append(f"- Kind: `{comp.kind}` · config `{comp.config_id}`")
    lines.append(f"- Arms: {', '.join(f'`{a}`' for a in arms)}")
    if comp.off and comp.on:
        lines.append(f"- Isolation: `{comp.off}` → `{comp.on}`")
    lines.append(f"- Seeds: `{sorted({int(r['seed']) for r in rows})}`")
    lines.append("")

    lines.append("## Seed-level scores")
    lines.append("")
    lines.append(
        "| arm | seed | Φ_late | min type | Δ maps | ARI maps | Δ maps learned | "
        "Δ agent | ARI agent | Δ weights | mix 2-proc | map split |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for rec in sorted(rows, key=lambda r: (arms.index(r["version_id"]) if r["version_id"] in arms else 99, r["seed"])):
        L = rec["learned"]
        s = rec["scalars"]
        lines.append(
            f"| `{rec['version_id']}` | {rec['seed']} | "
            f"{_fmt(s['phi_class_late'])} | {_fmt(s['late_min_type_frac'])} | "
            f"{_fmt(L['delta_common_all_late'])} | {_fmt(L['ari_common_all_late'])} | "
            f"{_fmt(L['delta_common_all_learned'], '+.3f')} | "
            f"{_fmt(L['delta_realized_all_late'])} | {_fmt(L['ari_realized_all_late'])} | "
            f"{_fmt(L['delta_weights_only_all_late'])} | "
            f"{int(mix_two_process(rec))} | {int(map_split(rec))} |"
        )
    lines.append("")

    lines.append("## Mean ± SEM")
    lines.append("")
    lines.append("| arm | Φ_late | Δ maps | ARI maps | Δ agent | ARI agent | mix 2-proc | map split |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for arm in arms:
        phi, phi_s = _mean_sem(rows, arm, "phi_class_late")
        dm, dm_s = _mean_sem(rows, arm, "delta_common_all_late")
        am, am_s = _mean_sem(rows, arm, "ari_common_all_late")
        da, da_s = _mean_sem(rows, arm, "delta_realized_all_late")
        aa, aa_s = _mean_sem(rows, arm, "ari_realized_all_late")
        lines.append(
            f"| `{arm}` | {_fmt(phi)} ± {_fmt(phi_s)} | "
            f"{_fmt(dm)} ± {_fmt(dm_s)} | {_fmt(am)} ± {_fmt(am_s)} | "
            f"{_fmt(da)} ± {_fmt(da_s)} | {_fmt(aa)} ± {_fmt(aa_s)} | "
            f"{_fmt(_rate(rows, arm, mix_two_process))} | "
            f"{_fmt(_rate(rows, arm, map_split))} |"
        )
    lines.append("")

    off, on = comp.off, comp.on
    if off and on:
        lines.append("## Paired deltas (same seeds, on − off)")
        lines.append("")
        lines.append("| metric | result |")
        lines.append("| --- | --- |")
        for key in _HEADLINE:
            a, b = _aligned(rows, off, on, key)
            st = paired_delta_test(a, b, n_iter=BOOTSTRAP_ITERS, seed=BOOTSTRAP_SEED)
            lines.append(f"| {_FUNC_LABELS[key]} | {format_paired(st)} |")
        lines.append("")
        lines.append(
            f"Mix two-process rate: `{off}` = {_rate(rows, off, mix_two_process):.2f}, "
            f"`{on}` = {_rate(rows, on, mix_two_process):.2f}."
        )
        lines.append("")
        lines.append(
            f"Map-split rate: `{off}` = {_rate(rows, off, map_split):.2f}, "
            f"`{on}` = {_rate(rows, on, map_split):.2f}."
        )
        lines.append("")
    elif off:
        lines.append("## Paired Δ maps late vs off")
        lines.append("")
        lines.append("| arm | Δ maps vs off | map-split rate |")
        lines.append("| --- | --- | --- |")
        for arm in [a for a in arms if a != off]:
            a, b = _aligned(rows, off, arm, "delta_common_all_late")
            st = paired_delta_test(a, b, n_iter=BOOTSTRAP_ITERS, seed=BOOTSTRAP_SEED)
            lines.append(
                f"| `{arm}` | {format_paired(st)} | {_rate(rows, arm, map_split):.2f} |"
            )
        lines.append("")

    lines.append("## How to read")
    lines.append("")
    lines.append("| Pattern | Meaning |")
    lines.append("| --- | --- |")
    lines.append("| Mix two-process, map-split low | counts moved; maps did not specialise |")
    lines.append("| Map-split, mix Φ low | two maps, balanced mix — Φ miss |")
    lines.append("| Both high | mix moved and maps split |")
    lines.append("| Agent ARI high, map ARI ~0 | typed outputs cluster via the label; maps do not |")
    lines.append("| min-type below floor | do not call two processes (attrition) |")
    lines.append("")

    path = out_dir / "FUNCTIONAL_REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_functional_index(
    root: Path,
    comparisons: list[Comparison],
    reports: dict[str, Path],
) -> Path:
    lines = [
        "# Functional divergence — lettered comparisons",
        "",
        "Same arms and seeds as the mix-Φ pipeline. Scores: "
        "`FUNCTIONAL_REPORT.md` per letter; embeddings under `functional_compare/`.",
        "",
        f"- Map-split: Δ_maps > {MAP_DELTA_HIT} and ARI_maps > {MAP_ARI_HIT} "
        f"and min-type > {COEXIST_FLOOR}",
        f"- Mix two-process: Φ_late > {PHI_LATE_HIT} and min-type > {COEXIST_FLOOR}",
        "",
        "| letter | title | report |",
        "| --- | --- | --- |",
    ]
    for c in comparisons:
        p = reports.get(c.id)
        rel = p.relative_to(root).as_posix() if p is not None else ""
        link = f"[{rel}]({rel})" if rel else ""
        lines.append(f"| `{c.id}` | {c.title} | {link} |")
    lines.append("")
    path = Path(root) / "FUNCTIONAL_INDEX.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def run_compare(root: Path) -> list[dict[str, Any]]:
    root = Path(root)
    records = evaluate_records(root, embed=False)
    write_csv(records, root / "functional_summary.csv")
    comparisons = comparisons_for_root(root, records)
    reports: dict[str, Path] = {}
    for comp in comparisons:
        dest = root / "functional_comparisons" / comp.id
        reports[comp.id] = write_letter_report(comp, records, dest)
        print(f"  letter {comp.id} → {reports[comp.id]}")
    idx = write_functional_index(root, comparisons, reports)
    # Overall table (all arms together) — useful next to mix REPORT.md.
    write_suite_functional_report(records, root, {}, n_steps=None)
    print(f"CSV:   {root / 'functional_summary.csv'}")
    print(f"Index: {idx}")
    print(f"Table: {root / 'FUNCTIONAL_REPORT.md'}")
    return records


def run_embed(root: Path) -> None:
    from research.response_analysis import run_analysis as plot_all

    root = Path(root)
    proto = load_protocol(root)
    # Plotting walks discover_run_dirs; restrict by evaluating only visual seeds
    # would require a filter. We pass through run_analysis which plots all
    # discovered snapshots. If protocol lists visual seeds, drop others' plots
    # by temporarily filtering — implement via env of discovered dirs.
    visual = None
    if proto:
        visual = {int(s) for s in proto.get("visual_seeds", SEEDS_VISUAL)}
    if visual:
        orig = discover_run_dirs
        from research import response_analysis as ra

        def _filtered(root_path: Path) -> list[Path]:
            dirs = orig(root_path)
            keep = []
            for d in dirs:
                try:
                    seed = int(d.name.split("_", 1)[1])
                except (IndexError, ValueError):
                    keep.append(d)
                    continue
                if seed in visual:
                    keep.append(d)
            if not keep:
                return dirs
            print(f"Embed: visual seeds {sorted(visual)} ({len(keep)} snapshots)")
            return keep

        ra.discover_run_dirs = _filtered  # type: ignore[assignment]
        try:
            plot_all(root, embed=True)
        finally:
            ra.discover_run_dirs = orig
    else:
        plot_all(root, embed=True)


def cmd_compare(args: argparse.Namespace) -> None:
    run_compare(Path(args.root))


def cmd_embed(args: argparse.Namespace) -> None:
    run_embed(Path(args.root))


def cmd_analyze(args: argparse.Namespace) -> None:
    run_compare(Path(args.root))
    if not args.no_embed:
        run_embed(Path(args.root))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m research.functional_analysis",
        description="Functional divergence scores (lettered) and embeddings.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("compare", help="Scores / paired tests / letter reports")
    sp.add_argument("root", type=Path)
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("embed", help="PCA/UMAP visualizations")
    sp.add_argument("root", type=Path)
    sp.set_defaults(func=cmd_embed)

    sp = sub.add_parser("analyze", help="compare then embed")
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
