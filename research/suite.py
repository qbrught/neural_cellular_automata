"""CLI for the centralized research experiment suite.

Examples:
  python -m research.suite list
  python -m research.suite run --versions original,A --n-steps 400
  python -m research.suite run --versions A,B,C --discoveries disc_0001,disc_0003,disc_0005
  python -m research.suite run --versions A,B,C --configs discoveries/disc_0001,discoveries/disc_0004
  python -m research.suite run --discoveries all --versions A,B,C --n-steps 400
  python -m research.suite run --quick   # fast smoke: 2 versions × 1 seed × 150 steps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on path when run as module
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research.charts import plot_comparison_dashboard, plot_run_panel
from research.config_sources import (
    DEFAULT_BENCHMARK,
    ConfigSource,
    resolve_suite_configs,
)
from research.report import write_multi_config_index, write_report
from research.runner import run_experiment
from research.versions import VERSIONS, VersionSpec, parse_version_list


DEFAULT_SEEDS = [1096812628, 42, 7]


def cmd_list(_: argparse.Namespace) -> None:
    print("Registered paper versions:\n")
    for vid, v in VERSIONS.items():
        flag = "OK " if v.implemented else "TODO"
        print(f"  [{flag}] {vid:10s}  {v.title}")
        print(f"           {v.description[:90]}...")
        print()


def _annotate_result(result: dict, source: ConfigSource) -> dict:
    """Attach config metadata used by reports, CSV, and chart titles."""
    result["config_id"] = source.id
    result["config_title"] = source.title
    result["config_description"] = source.description
    result["config_path"] = str(source.path)
    return result


def _run_one_config(
    source: ConfigSource,
    versions: list[VersionSpec],
    seeds: list[int],
    n_steps: int,
    *,
    out_dir: Path,
    log_every: int,
    suite_name: str,
) -> list[dict]:
    """Run all versions × seeds for one base config; write per-config artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 72)
    print(f"CONFIG  {source.id}")
    if source.description:
        print(f"        {source.description}")
    print(f"        path={source.path}")
    print(f"        versions={[v.id for v in versions]}  seeds={seeds}  steps={n_steps}")
    print("=" * 72)

    results: list[dict] = []
    for version in versions:
        for seed in seeds:
            run_dir = out_dir / "versions" / version.id / f"seed_{seed}"
            print(f"=== [{source.id}] {version.id} · seed={seed} ===")
            t0 = time.time()
            result = run_experiment(
                version,
                source.cfg,
                seed,
                n_steps=n_steps,
                out_dir=run_dir,
                log_every=log_every,
            )
            result = _annotate_result(result, source)
            plot_run_panel(result, run_dir / "panel.png")
            dt = time.time() - t0
            s = result["summary"]
            g_drift = s.get("goal_frac_repro_drift", float("nan"))
            phi = s.get("phi_class", float("nan"))
            phi_late = s.get("phi_class_late", float("nan"))
            min_tf = s.get("late_min_type_frac", float("nan"))
            print(
                f"  done in {dt:.1f}s  "
                f"Φ={phi:.4f}  Φ_late={phi_late:.4f}  "
                f"corr(ra,ea)={s['corr_ra_ea']:.3f}  "
                f"min_type late={min_tf:.3f}  "
                f"g_drift={g_drift:+.3f}  "
                f"alive late={s['mean_alive_late']:.1f}"
            )
            results.append(result)

    chart_dir = out_dir / "comparison"
    chart_paths = plot_comparison_dashboard(
        results,
        chart_dir,
        title_prefix=f"Config {source.id}",
    )
    chart_paths = {k: Path(p).resolve() for k, p in chart_paths.items()}

    write_report(
        results,
        versions,
        out_dir,
        chart_paths=chart_paths,
        base_config_note=str(source.path),
        n_steps=int(n_steps),
        seeds=seeds,
        config_id=source.id,
        config_title=source.title,
        config_description=source.description,
        config_path=str(source.path),
        suite_name=suite_name,
    )
    return results


def cmd_run(args: argparse.Namespace) -> None:
    if args.quick:
        versions = parse_version_list("original,A")
        seeds = [1096812628]
        n_steps = 150
        print("Quick mode: original,A · 1 seed · 150 steps")
    else:
        versions = parse_version_list(args.versions)
        seeds = (
            [int(s) for s in args.seeds.split(",")]
            if args.seeds
            else list(DEFAULT_SEEDS)
        )
        n_steps = args.n_steps

    sources = resolve_suite_configs(
        config=args.config,
        configs=args.configs,
        discoveries=args.discoveries,
    )

    # Default step count: first config's n_steps, else 500
    if n_steps is None:
        n_steps = sources[0].cfg.n_steps or 500

    stamp = time.strftime("%Y%m%d_%H%M%S")
    if args.name:
        run_name = args.name
    elif len(sources) == 1:
        run_name = f"suite_{sources[0].id}_{stamp}"
    else:
        run_name = f"suite_multi_{stamp}"
    out_root = Path(args.output_dir) / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Output → {out_root.resolve()}")
    print(f"Suite:    {run_name}")
    print(f"Versions: {[v.id for v in versions]}")
    print(f"Seeds:    {seeds}")
    print(f"Steps:    {n_steps}")
    print(f"Configs:  {[s.id for s in sources]}")
    for s in sources:
        desc = f" — {s.description}" if s.description else ""
        print(f"  · {s.id}{desc}")
        print(f"    {s.path}")

    all_results: list[dict] = []
    config_summaries: list[dict] = []

    multi = len(sources) > 1
    for source in sources:
        cfg_out = (
            out_root / "configs" / source.id
            if multi
            else out_root
        )
        results = _run_one_config(
            source,
            versions,
            seeds,
            int(n_steps),
            out_dir=cfg_out,
            log_every=args.log_every,
            suite_name=run_name,
        )
        all_results.extend(results)
        rel_report = (
            f"configs/{source.id}/REPORT.md"
            if multi
            else "REPORT.md"
        )
        config_summaries.append(
            {
                "config_id": source.id,
                "config_title": source.title,
                "config_description": source.description,
                "config_path": str(source.path),
                "rel_report": rel_report,
                "n_results": len(results),
            }
        )

    # Machine-readable suite manifest
    manifest = {
        "run_name": run_name,
        "n_steps": n_steps,
        "seeds": seeds,
        "versions": [v.id for v in versions],
        "configs": [
            {
                "id": s.id,
                "path": str(s.path),
                "title": s.title,
                "description": s.description,
            }
            for s in sources
        ],
        "results": [
            {
                "config_id": r.get("config_id"),
                "version": r["version_id"],
                "seed": r["seed"],
                "summary": r["summary"],
            }
            for r in all_results
        ],
    }
    with (out_root / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    if multi:
        index_path = write_multi_config_index(
            out_root,
            suite_name=run_name,
            versions=versions,
            seeds=seeds,
            n_steps=int(n_steps),
            config_summaries=config_summaries,
            all_results=all_results,
        )
        print()
        print(f"Multi-config INDEX: {index_path}")
        print(f"Top REPORT:         {out_root / 'REPORT.md'}")
        print(f"Combined CSV:       {out_root / 'summary_all.csv'}")
        for cs in config_summaries:
            print(f"  · {cs['config_id']}: {out_root / cs['rel_report']}")
    else:
        print()
        print(f"Report: {out_root / 'REPORT.md'}")
        print(f"Notes:  {out_root / 'NOTES.md'}")
        print(f"CSV:    {out_root / 'summary.csv'}")
        print(f"Charts: {out_root / 'comparison'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m research.suite",
        description="Centralized research experiment suite for paper version comparisons.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="List registered paper versions")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("run", help="Run version comparison suite")
    sp.add_argument(
        "--versions",
        default="original,A",
        help="Comma-separated version ids, or 'all' (implemented only). Default: original,A",
    )
    sp.add_argument(
        "--seeds",
        default=None,
        help=f"Comma-separated seeds. Default: {','.join(map(str, DEFAULT_SEEDS))}",
    )
    sp.add_argument("--n-steps", type=int, default=None, help="Override step count")
    sp.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Single base Config JSON (default: {DEFAULT_BENCHMARK})",
    )
    sp.add_argument(
        "--configs",
        type=str,
        default=None,
        help=(
            "Comma-separated config paths (JSON files or dirs with config.json). "
            "Example: discoveries/disc_0001,discoveries/disc_0003"
        ),
    )
    sp.add_argument(
        "--discoveries",
        type=str,
        default=None,
        help=(
            "Comma-separated discovery ids under discoveries/, or 'all'. "
            "Example: disc_0001,disc_0003,disc_0005  or  1,3,5  or  all"
        ),
    )
    sp.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research_results"),
        help="Parent directory for suite runs",
    )
    sp.add_argument("--name", type=str, default=None, help="Run folder name")
    sp.add_argument("--log-every", type=int, default=0, help="Print every N steps (0=off)")
    sp.add_argument(
        "--quick",
        action="store_true",
        help="Smoke run: original,A · 1 seed · 150 steps",
    )
    sp.set_defaults(func=cmd_run)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
