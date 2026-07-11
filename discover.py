"""CLI: automated discovery of interesting NCSA configurations.

Usage:
    export GEMINI_API_KEY=...
    python discover.py --max-cycles 30 --target-discoveries 5
    python discover.py --max-cycles 10 --dry-run          # heuristic-guided, no API
    python discover.py --no-guided --max-cycles 20        # pure random search
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from discovery.loop import LoopConfig, run_discovery


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Discover interesting NCSA configs via guided VLM search "
            "(Gemini Flash) or pure random sampling."
        ),
    )
    p.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Stop after this many sample→run attempts.",
    )
    p.add_argument(
        "--target-discoveries",
        type=int,
        default=None,
        help="Stop after this many saved discoveries (includes existing catalog).",
    )
    p.add_argument("--n-steps", type=int, default=1000,
                   help="Simulation steps per trial (default: 1000).")
    p.add_argument("--N", type=int, default=20, help="Grid side length (default: 20).")
    p.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"),
                   help="Torch device (default: cpu).")
    p.add_argument(
        "--model",
        type=str,
        default="gemini-3.5-flash",
        help="Gemini model id (default: gemini-3.5-flash).",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("discoveries"),
        help="Directory for catalog + disc_* folders (default: discoveries/).",
    )
    p.add_argument(
        "--sampler-seed",
        type=int,
        default=None,
        help=(
            "RNG seed for sampling / explore jumps. Default: fresh each process. "
            "Pass an int to reproduce a previous search path."
        ),
    )
    p.add_argument(
        "--mutate-prob",
        type=float,
        default=0.0,
        help="When exploring/random, chance to mutate a catalog config (default: 0).",
    )
    p.add_argument(
        "--guided",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="VLM-guided next configs (default: on). Use --no-guided for pure random.",
    )
    p.add_argument(
        "--explore-prob",
        type=float,
        default=0.15,
        help=(
            "With --guided, probability of ignoring the VLM proposal and taking a "
            "pure random jump (default: 0.15)."
        ),
    )
    p.add_argument(
        "--keep-rejects",
        action="store_true",
        help="Keep trial dirs under discoveries/trials/ even when rejected.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Sims + prefilter only; heuristic guidance if --guided; no VLM/saves.",
    )
    p.add_argument(
        "--verbose-sim",
        action="store_true",
        help="Print per-step simulation logs (noisy).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.max_cycles is None and args.target_discoveries is None:
        print(
            "error: set at least one of --max-cycles or --target-discoveries",
            file=sys.stderr,
        )
        return 2
    if args.max_cycles is not None and args.max_cycles <= 0:
        print("error: --max-cycles must be positive", file=sys.stderr)
        return 2
    if args.target_discoveries is not None and args.target_discoveries <= 0:
        print("error: --target-discoveries must be positive", file=sys.stderr)
        return 2
    if not 0.0 <= args.mutate_prob <= 1.0:
        print("error: --mutate-prob must be in [0, 1]", file=sys.stderr)
        return 2
    if not 0.0 <= args.explore_prob <= 1.0:
        print("error: --explore-prob must be in [0, 1]", file=sys.stderr)
        return 2
    if args.n_steps <= 0 or args.N <= 0:
        print("error: --n-steps and --N must be positive", file=sys.stderr)
        return 2

    loop_cfg = LoopConfig(
        max_cycles=args.max_cycles,
        target_discoveries=args.target_discoveries,
        n_steps=args.n_steps,
        N=args.N,
        device=args.device,
        model=args.model,
        output_root=args.output_root,
        sampler_seed=args.sampler_seed,
        mutate_prob=args.mutate_prob,
        keep_rejects=args.keep_rejects,
        dry_run=args.dry_run,
        verbose_sim=args.verbose_sim,
        guided=args.guided,
        explore_prob=args.explore_prob,
    )
    run_discovery(loop_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
