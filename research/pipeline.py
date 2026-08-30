"""Thesis results pipeline: one cache, many isolations, four-chunk reports.

    python -m research.pipeline list
    python -m research.pipeline run --quick
    python -m research.pipeline run --letters A,F --include-ladder
    python -m research.pipeline report research_results/<name>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from research.charts import plot_run_panel
from research.compare import write_summary_csv
from research.comparisons import (
    Arm,
    COMPARISONS,
    Comparison,
    parse_letter_list,
)
from research.config_sources import config_source_from_path
from research.protocol import (
    CONFIG_IDS,
    N_STEPS,
    QUICK_N_STEPS,
    QUICK_SEEDS,
    SEEDS_THESIS,
    SEEDS_VISUAL,
)
from research.runner import run_experiment
from research.thesis_report import build_comparison_artifacts, write_index
from research.versions import VersionSpec


@dataclass(frozen=True)
class Job:
    config_id: str
    config_path: Path
    arm: Arm
    seed: int

    @property
    def cache_rel(self) -> str:
        return f"cache/{self.config_id}/{self.arm.id}/seed_{self.seed}"


def _load_cached(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "summary.json"
    series_path = run_dir / "series.npz"
    meta_path = run_dir / "meta.json"
    if not (summary_path.is_file() and series_path.is_file() and meta_path.is_file()):
        return None
    summary = json.loads(summary_path.read_text())
    meta = json.loads(meta_path.read_text())
    series = {k: np.asarray(v) for k, v in np.load(series_path).items()}
    frames = run_dir / "frames.npz"
    return {
        "version_id": meta.get("arm_id") or meta.get("version_id"),
        "spec_id": meta.get("spec_id"),
        "version_title": meta.get("version_title", ""),
        "seed": int(meta["seed"]),
        "goal_frac_repro": meta.get("goal_frac_repro"),
        "config": meta.get("config", {}),
        "series": series,
        "summary": summary,
        "run_dir": str(run_dir),
        "frames_path": str(frames) if frames.is_file() else None,
    }


def select_comparisons(args: argparse.Namespace) -> list[Comparison]:
    extra: list[str] = []
    if getattr(args, "include_ladder", False):
        extra.append("ladder")
    if getattr(args, "lambda_sweep", False):
        extra.append("F_lambda")
    if getattr(args, "identity", False):
        extra.append("E_identity")
    letters = args.letters if getattr(args, "letters", None) else None
    chosen = parse_letter_list(letters, include_optional=False)
    have = {c.id for c in chosen}
    for cid in extra:
        if cid not in have:
            from research.comparisons import get_comparison

            chosen.append(get_comparison(cid))
            have.add(cid)
    return chosen


def unique_jobs(
    comparisons: list[Comparison],
    seeds: list[int],
) -> list[Job]:
    seen: set[tuple[str, str, int]] = set()
    jobs: list[Job] = []
    for comp in comparisons:
        path = CONFIG_IDS[comp.config_id]
        for arm in comp.arms:
            for seed in seeds:
                key = (comp.config_id, arm.id, int(seed))
                if key in seen:
                    continue
                seen.add(key)
                jobs.append(
                    Job(
                        config_id=comp.config_id,
                        config_path=path,
                        arm=arm,
                        seed=int(seed),
                    )
                )
    return jobs


def _base_cfg(path: Path) -> Config:
    return config_source_from_path(path).cfg


def _simulate_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Subprocess entry: one (config, arm, seed). Returns a small status dict."""
    out_root = Path(payload["out_root"])
    job_dir = out_root / payload["cache_rel"]
    key = (payload["config_id"], payload["arm_id"], int(payload["seed"]))
    if payload["skip_existing"]:
        hit = _load_cached(job_dir)
        if hit is not None:
            return {
                "key": list(key),
                "skipped": True,
                "ok": True,
                "phi_late": hit["summary"].get("phi_class_late"),
                "dt": 0.0,
                "cache_rel": payload["cache_rel"],
            }
    arm = Arm(
        id=payload["arm_id"],
        version=payload["version"],
        spec_overrides=payload.get("spec_overrides") or {},
    )
    spec: VersionSpec = arm.spec()
    cfg = _base_cfg(Path(payload["config_path"]))
    t0 = time.time()
    result = run_experiment(
        spec,
        cfg,
        int(payload["seed"]),
        n_steps=int(payload["n_steps"]),
        out_dir=job_dir,
        log_every=int(payload["log_every"]),
        arm_id=arm.id,
        save_frames=True,
    )
    plot_run_panel(result, job_dir / "panel.png")
    dt = time.time() - t0
    return {
        "key": list(key),
        "skipped": False,
        "ok": True,
        "phi_late": result["summary"].get("phi_class_late"),
        "alive_late": result["summary"].get("mean_alive_late"),
        "dt": dt,
        "cache_rel": payload["cache_rel"],
    }


def _job_payload(
    job: Job,
    out_root: Path,
    *,
    n_steps: int,
    skip_existing: bool,
    log_every: int,
) -> dict[str, Any]:
    return {
        "out_root": str(out_root),
        "cache_rel": job.cache_rel,
        "config_id": job.config_id,
        "config_path": str(job.config_path),
        "arm_id": job.arm.id,
        "version": job.arm.version,
        "spec_overrides": dict(job.arm.spec_overrides),
        "seed": job.seed,
        "n_steps": n_steps,
        "skip_existing": skip_existing,
        "log_every": log_every,
    }


def _reload_cache(jobs: list[Job], out_root: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    cache: dict[tuple[str, str, int], dict[str, Any]] = {}
    missing = 0
    for job in jobs:
        hit = _load_cached(out_root / job.cache_rel)
        if hit is None:
            missing += 1
            continue
        hit["config_id"] = job.config_id
        cache[(job.config_id, job.arm.id, job.seed)] = hit
    if missing:
        print(f"warning: {missing}/{len(jobs)} cache entries missing after simulate")
    return cache


def run_jobs(
    jobs: list[Job],
    out_root: Path,
    *,
    n_steps: int,
    skip_existing: bool,
    log_every: int,
    workers: int = 1,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Simulate unique jobs into cache/. Returns (config_id, arm_id, seed) → result."""
    n = len(jobs)
    payloads = [
        _job_payload(
            job,
            out_root,
            n_steps=n_steps,
            skip_existing=skip_existing,
            log_every=log_every,
        )
        for job in jobs
    ]
    workers = max(1, int(workers))
    print(f"Simulating {n} jobs with {workers} worker(s)")
    done = 0
    if workers == 1:
        for payload in payloads:
            info = _simulate_job(payload)
            done += 1
            skip = " skip" if info.get("skipped") else ""
            print(
                f"[{done}/{n}]{skip} {info['cache_rel']}  "
                f"Φ_late={info.get('phi_late')}  {info.get('dt', 0):.1f}s"
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_simulate_job, p): p for p in payloads}
            for fut in as_completed(futs):
                info = fut.result()
                done += 1
                skip = " skip" if info.get("skipped") else ""
                print(
                    f"[{done}/{n}]{skip} {info['cache_rel']}  "
                    f"Φ_late={info.get('phi_late')}  {info.get('dt', 0):.1f}s",
                    flush=True,
                )
    return _reload_cache(jobs, out_root)


def gather_comparison(
    comp: Comparison,
    cache: dict[tuple[str, str, int], dict[str, Any]],
    seeds: list[int],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for arm in comp.arms:
        for seed in seeds:
            r = cache.get((comp.config_id, arm.id, int(seed)))
            if r is None:
                continue
            row = dict(r)
            row["config_id"] = comp.config_id
            row["version_id"] = arm.id
            out.append(row)
    return out


def write_protocol_snapshot(
    out_root: Path,
    *,
    suite_name: str,
    n_steps: int,
    seeds: list[int],
    visual_seeds: list[int],
    comparisons: list[Comparison],
    quick: bool,
) -> Path:
    payload = {
        "suite_name": suite_name,
        "n_steps": n_steps,
        "seeds": seeds,
        "visual_seeds": visual_seeds,
        "quick": quick,
        "comparisons": [
            {
                "id": c.id,
                "kind": c.kind,
                "config_id": c.config_id,
                "off": c.off,
                "on": c.on,
                "arms": [a.id for a in c.arms],
            }
            for c in comparisons
        ],
        "configs": {k: str(v) for k, v in CONFIG_IDS.items()},
    }
    path = out_root / "protocol.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def assemble_reports(
    out_root: Path,
    comparisons: list[Comparison],
    cache: dict[tuple[str, str, int], dict[str, Any]],
    *,
    suite_name: str,
    n_steps: int,
    seeds: list[int],
    visual_seeds: list[int],
) -> dict[str, Path]:
    reports: dict[str, Path] = {}
    all_rows: list[dict[str, Any]] = []
    for comp in comparisons:
        results = gather_comparison(comp, cache, seeds)
        if not results:
            print(f"  skip report {comp.id}: no cached results")
            continue
        dest = out_root / "comparisons" / comp.id
        print(f"  report {comp.id} → {dest}")
        reports[comp.id] = build_comparison_artifacts(
            comp,
            results,
            dest,
            n_steps=n_steps,
            seeds=seeds,
            visual_seeds=visual_seeds,
            suite_name=suite_name,
        )
        for r in results:
            all_rows.append(
                {
                    **r,
                    "config_id": comp.config_id,
                    "config_title": comp.id,
                }
            )
    if all_rows:
        write_summary_csv(all_rows, out_root / "summary_all.csv")
    write_index(
        out_root,
        suite_name=suite_name,
        n_steps=n_steps,
        seeds=seeds,
        comparisons=comparisons,
        reports=reports,
    )
    return reports


def cmd_list(_: argparse.Namespace) -> None:
    print("Thesis comparisons (default = isolation chapters):\n")
    for c in COMPARISONS:
        flag = "default" if c.default else "optional"
        arms = ", ".join(a.id for a in c.arms)
        pair = f"{c.off} → {c.on}" if c.off and c.on else c.kind
        print(f"  [{flag:8s}] {c.id:12s}  {pair:20s}  config={c.config_id}")
        print(f"               arms: {arms}")
        print(f"               {c.title}")
        print()
    print("Add a letter in research/comparisons.py (Comparison + Arm).")
    print("See research/THESIS_PIPELINE.md.")


def cmd_run(args: argparse.Namespace) -> None:
    quick = bool(args.quick)
    if quick and not args.letters:
        args.letters = "A"
    comparisons = select_comparisons(args)
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = list(QUICK_SEEDS if quick else SEEDS_THESIS)
    n_steps = int(args.n_steps) if args.n_steps else (QUICK_N_STEPS if quick else N_STEPS)
    visual_seeds = [s for s in SEEDS_VISUAL if s in seeds] or seeds[:3]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    if args.name:
        run_name = args.name
    elif quick:
        run_name = f"thesis_quick_{stamp}"
    else:
        run_name = f"thesis_{stamp}"
    out_root = Path(args.output_dir) / run_name

    jobs = unique_jobs(comparisons, seeds)
    print(f"Output → {out_root.resolve()}")
    print(f"Comparisons: {[c.id for c in comparisons]}")
    print(f"Unique jobs: {len(jobs)}  ({n_steps} steps, seeds={seeds})")
    if args.dry_run:
        for j in jobs:
            print(f"  would run {j.cache_rel}")
        return
    out_root.mkdir(parents=True, exist_ok=True)
    write_protocol_snapshot(
        out_root,
        suite_name=run_name,
        n_steps=n_steps,
        seeds=seeds,
        visual_seeds=visual_seeds,
        comparisons=comparisons,
        quick=quick,
    )

    n_workers = int(args.workers) if args.workers else max(1, min(8, (os.cpu_count() or 2) - 1))
    cache = run_jobs(
        jobs,
        out_root,
        n_steps=n_steps,
        skip_existing=bool(args.skip_existing),
        log_every=int(args.log_every),
        workers=n_workers,
    )
    print("Assembling reports…")
    assemble_reports(
        out_root,
        comparisons,
        cache,
        suite_name=run_name,
        n_steps=n_steps,
        seeds=seeds,
        visual_seeds=visual_seeds,
    )
    print()
    print(f"INDEX: {out_root / 'INDEX.md'}")


def cmd_report(args: argparse.Namespace) -> None:
    out_root = Path(args.suite_dir)
    proto_path = out_root / "protocol.json"
    if not proto_path.is_file():
        raise FileNotFoundError(f"No protocol.json in {out_root}")
    proto = json.loads(proto_path.read_text())
    from research.comparisons import get_comparison

    ids = [c["id"] for c in proto["comparisons"]]
    comparisons = [get_comparison(i) for i in ids]
    seeds = [int(s) for s in proto["seeds"]]
    n_steps = int(proto["n_steps"])
    visual_seeds = [int(s) for s in proto.get("visual_seeds", seeds[:3])]
    jobs = unique_jobs(comparisons, seeds)
    cache: dict[tuple[str, str, int], dict[str, Any]] = {}
    missing = 0
    for job in jobs:
        hit = _load_cached(out_root / job.cache_rel)
        if hit is None:
            missing += 1
            continue
        hit["config_id"] = job.config_id
        cache[(job.config_id, job.arm.id, job.seed)] = hit
    if missing:
        print(f"warning: {missing} cache entries missing; reports use what exists")
    assemble_reports(
        out_root,
        comparisons,
        cache,
        suite_name=str(proto.get("suite_name") or out_root.name),
        n_steps=n_steps,
        seeds=seeds,
        visual_seeds=visual_seeds,
    )
    print(f"INDEX: {out_root / 'INDEX.md'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m research.pipeline",
        description="Thesis feature-letter pipeline (isolations, frames, paired tests).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="List registered comparisons")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("run", help="Simulate + report")
    sp.add_argument(
        "--letters",
        default=None,
        help="Comma-separated comparison ids, or 'all' / 'default'. Default: all default isolations.",
    )
    sp.add_argument("--include-ladder", action="store_true", help="Also run the full version ladder")
    sp.add_argument("--lambda-sweep", action="store_true", help="Also run F λ sweep")
    sp.add_argument("--identity", action="store_true", help="Also run A vs E identity on sym weights")
    sp.add_argument("--seeds", default=None, help="Comma-separated seeds (overrides protocol)")
    sp.add_argument("--n-steps", type=int, default=None, help="Override step count")
    sp.add_argument("--quick", action="store_true", help="1 seed, short T, default letters")
    sp.add_argument("--skip-existing", action="store_true", help="Reuse cache hits")
    sp.add_argument("--dry-run", action="store_true", help="Print unique jobs and exit")
    sp.add_argument("--name", type=str, default=None, help="Run folder name")
    sp.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research_results"),
        help="Parent directory",
    )
    sp.add_argument("--log-every", type=int, default=0)
    sp.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel simulate processes (default: min(8, ncpu-1))",
    )
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("report", help="Rebuild reports from an existing cache")
    sp.add_argument("suite_dir", type=Path)
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser(
        "functional",
        help="Functional-divergence scores on a pipeline/suite folder (no PCA/UMAP)",
    )
    sp.add_argument("suite_dir", type=Path)
    sp.set_defaults(func=_cmd_functional)

    sp = sub.add_parser(
        "embed",
        help="PCA/UMAP of functional response vectors (visual seeds if protocol.json)",
    )
    sp.add_argument("suite_dir", type=Path)
    sp.set_defaults(func=_cmd_embed)
    return p


def _cmd_functional(args: argparse.Namespace) -> None:
    from research.functional_analysis import run_compare

    run_compare(Path(args.suite_dir))


def _cmd_embed(args: argparse.Namespace) -> None:
    from research.functional_analysis import run_embed

    run_embed(Path(args.suite_dir))


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
