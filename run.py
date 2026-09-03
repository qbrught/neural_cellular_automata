"""Command-line entry point for running a NCSA simulation.

Usage:
    python run.py                              # run with defaults
    python run.py --config configs/foo.json    # run with a saved Config
    python run.py --seed 7 --n-steps 500       # override individual fields
    python run.py --visualise                  # also render summary.png + animation.gif

By default writes to runs/<timestamped_name>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import Config
from simulate import run as run_simulation


def parse_args() -> argparse.Namespace:
    """CLI flags: config path, common overrides, and visualisation toggles."""
    p = argparse.ArgumentParser(description="Run an NCSA simulation.")
    p.add_argument("--config", type=Path, default=None,
                   help="Path to a Config JSON file. If omitted, uses defaults.")
    # Direct overrides (only the most commonly-changed knobs).
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--eta", type=float, default=None)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Explicit output directory. Default: timestamped subdir of cfg.output_dir.")
    p.add_argument("--run-name", type=str, default=None,
                   help="Subdirectory name under cfg.output_dir (timestamp used if omitted).")
    p.add_argument("--visualise", action="store_true",
                   help="Render summary.png and animation.gif after the run.")
    p.add_argument("--final-grid", action="store_true",
                   help="Render final_grid.png (just the final grid, square, no axes).")
    p.add_argument("--no-animation", action="store_true",
                   help="With --visualise, skip the animation (faster).")
    p.add_argument("--alive-count", action="store_true",
                   help="Render alive_count.svg (wide time-series, poster fonts).")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    """Load config, run the simulation, optionally render plots."""
    args = parse_args()
    cfg = Config.load(args.config) if args.config else Config()

    # Apply overrides.
    if args.seed is not None: cfg.seed = args.seed
    if args.n_steps is not None: cfg.n_steps = args.n_steps
    if args.N is not None: cfg.N = args.N
    if args.eta is not None: cfg.eta = args.eta
    if args.run_name is not None: cfg.run_name = args.run_name
    cfg.__post_init__()  # re-validate after overrides

    out = run_simulation(cfg, output_dir=args.output_dir, verbose=not args.quiet)

    if args.visualise or args.final_grid or args.alive_count:
            from visualise import (render_summary, render_animation,
                               render_final_grid, render_alive_count)
            if args.final_grid:
                grid_path = render_final_grid(out / "trajectory.npz")
                if not args.quiet:
                    print(f"  Wrote final grid: {grid_path}")
            if args.alive_count:
                print(f"  Wrote alive count: {render_alive_count(out / 'trajectory.npz')}")

            if args.visualise:
                summary_path = render_summary(out / "trajectory.npz")
                if not args.quiet:
                    print(f"  Wrote summary: {summary_path}")
                if not args.no_animation:
                    anim_path = render_animation(
                        out / "trajectory.npz", out / "animation.gif",
                        fps=15, stride=max(1, cfg.n_steps // 200),
                    )
                    if not args.quiet:
                        print(f"  Wrote animation: {anim_path}")


if __name__ == "__main__":
    main()
