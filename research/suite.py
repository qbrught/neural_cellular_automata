"""CLI for the centralized research experiment suite.

Examples:
  python -m research.suite list
  python -m research.suite run --versions original,A --n-steps 400
  python -m research.suite run --versions original,A --seeds 1096812628,42,7
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

from config import Config
from research.charts import plot_comparison_dashboard, plot_run_panel
from research.report import write_report
from research.runner import run_experiment
from research.versions import VERSIONS, parse_version_list


DEFAULT_SEEDS = [1096812628, 42, 7]
DEFAULT_CONFIG = Path(__file__).parent / "configs" / "benchmark.json"


def cmd_list(_: argparse.Namespace) -> None:
    print("Registered paper versions:\n")
    for vid, v in VERSIONS.items():
        flag = "OK " if v.implemented else "TODO"
        print(f"  [{flag}] {vid:10s}  {v.title}")
        print(f"           {v.description[:90]}...")
        print()


def cmd_run(args: argparse.Namespace) -> None:
    if args.quick:
        versions = parse_version_list("original,A")
        seeds = [1096812628]
        n_steps = 150
        print("Quick mode: original,A · 1 seed · 150 steps")
    else:
        versions = parse_version_list(args.versions)
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(DEFAULT_SEEDS)
        n_steps = args.n_steps

    base = Config.load(args.config) if args.config else Config.load(DEFAULT_CONFIG)
    if n_steps is None:
        n_steps = base.n_steps or 500

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.name or f"suite_{stamp}"
    out_root = Path(args.output_dir) / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Output → {out_root.resolve()}")
    print(f"Versions: {[v.id for v in versions]}")
    print(f"Seeds: {seeds}")
    print(f"Steps: {n_steps}")
    print()

    results = []
    for version in versions:
        for seed in seeds:
            run_dir = out_root / "versions" / version.id / f"seed_{seed}"
            print(f"=== {version.id} · seed={seed} ===")
            t0 = time.time()
            result = run_experiment(
                version,
                base,
                seed,
                n_steps=n_steps,
                out_dir=run_dir,
                log_every=args.log_every,
            )
            plot_run_panel(result, run_dir / "panel.png")
            dt = time.time() - t0
            s = result["summary"]
            print(
                f"  done in {dt:.1f}s  "
                f"corr(ra,ea)={s['corr_ra_ea']:.3f}  "
                f"ra/ea late={s['ratio_ra_ea_late']:.2f}  "
                f"R_disc late={s['late_R_vote_disc']:.3f}  "
                f"alive late={s['mean_alive_late']:.1f}"
            )
            results.append(result)

    # Comparison charts + report
    chart_dir = out_root / "comparison"
    chart_paths = plot_comparison_dashboard(results, chart_dir)
    # Make paths absolute for report relativization
    chart_paths = {k: Path(p).resolve() for k, p in chart_paths.items()}

    report_path = write_report(
        results,
        versions,
        out_root,
        chart_paths=chart_paths,
        base_config_note=str(args.config or DEFAULT_CONFIG),
        n_steps=int(n_steps),
        seeds=seeds,
    )

    # Machine-readable suite manifest
    manifest = {
        "run_name": run_name,
        "n_steps": n_steps,
        "seeds": seeds,
        "versions": [v.id for v in versions],
        "base_config": (args.config or str(DEFAULT_CONFIG)),
        "results": [
            {
                "version": r["version_id"],
                "seed": r["seed"],
                "summary": r["summary"],
            }
            for r in results
        ],
    }
    with (out_root / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print()
    print(f"Report: {report_path}")
    print(f"Notes:  {out_root / 'NOTES.md'}")
    print(f"CSV:    {out_root / 'summary.csv'}")
    print(f"Charts: {chart_dir}")


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
        help=f"Base Config JSON (default: {DEFAULT_CONFIG})",
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
