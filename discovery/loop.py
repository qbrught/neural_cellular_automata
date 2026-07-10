"""Main discovery loop: sample → run → prefilter → judge → save."""

from __future__ import annotations

import json
import random
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path

from config import Config
from discovery.catalog import Catalog
from discovery.evidence import run_trial
from discovery.judge import JudgeResult, judge_trial
from discovery.prefilter import metrics_from_trajectory, prefilter
from discovery.sample import sample_config


@dataclass
class LoopStats:
    cycles: int = 0
    prefilter_rejects: int = 0
    vlm_calls: int = 0
    vlm_errors: int = 0
    discoveries: int = 0


@dataclass
class LoopConfig:
    max_cycles: int | None
    target_discoveries: int | None
    n_steps: int
    N: int
    device: str
    model: str
    output_root: Path
    sampler_seed: int | None  # None → fresh entropy each process
    mutate_prob: float
    keep_rejects: bool
    dry_run: bool
    verbose_sim: bool = False


def _should_stop(cfg: LoopConfig, stats: LoopStats, catalog: Catalog) -> str | None:
    if cfg.max_cycles is not None and stats.cycles >= cfg.max_cycles:
        return f"reached max_cycles={cfg.max_cycles}"
    if cfg.target_discoveries is not None and catalog.count >= cfg.target_discoveries:
        return f"reached target_discoveries={cfg.target_discoveries}"
    return None


def _promote_trial(
    trial_dir: Path,
    dest: Path,
    result: JudgeResult,
    *,
    cycle: int,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "summary.png", "trajectory.npz", "params_final.pt"):
        src = trial_dir / name
        if src.exists():
            shutil.copy2(src, dest / name)

    note = dest / "note.txt"
    note.write_text(result.one_liner + "\n", encoding="utf-8")

    meta = {
        "cycle": cycle,
        "one_liner": result.one_liner,
        "judge": result.to_dict(),
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _sample_for_cycle(
    rng: random.Random,
    catalog: Catalog,
    loop_cfg: LoopConfig,
) -> Config:
    base: Config | None = None
    if (
        loop_cfg.mutate_prob > 0
        and catalog.count > 0
        and rng.random() < loop_cfg.mutate_prob
    ):
        path = catalog.random_base_config_path(rng)
        if path is not None:
            base = Config.load(path)

    return sample_config(
        rng,
        base=base,
        n_steps=loop_cfg.n_steps,
        N=loop_cfg.N,
        device=loop_cfg.device,
    )


def run_discovery(loop_cfg: LoopConfig) -> LoopStats:
    root = Path(loop_cfg.output_root)
    root.mkdir(parents=True, exist_ok=True)
    trials_root = root / "trials"
    trials_root.mkdir(parents=True, exist_ok=True)

    catalog = Catalog(root)
    # Fresh seed each process by default so re-running explore new configs.
    # Pass --sampler-seed N to replay the same search path for debugging.
    sampler_seed = (
        loop_cfg.sampler_seed
        if loop_cfg.sampler_seed is not None
        else secrets.randbits(63)
    )
    rng = random.Random(sampler_seed)
    stats = LoopStats(discoveries=catalog.count)

    print(f"Discovery root: {root.resolve()}")
    print(f"  existing discoveries: {catalog.count}")
    print(f"  max_cycles={loop_cfg.max_cycles}  "
          f"target_discoveries={loop_cfg.target_discoveries}")
    print(f"  n_steps={loop_cfg.n_steps}  N={loop_cfg.N}  device={loop_cfg.device}")
    print(f"  model={loop_cfg.model}  dry_run={loop_cfg.dry_run}")
    print(f"  sampler_seed={sampler_seed}"
          f"{'' if loop_cfg.sampler_seed is not None else ' (auto)'}")
    print()

    while True:
        stop = _should_stop(loop_cfg, stats, catalog)
        if stop:
            print(f"Stop: {stop}")
            break

        stats.cycles += 1
        cycle = stats.cycles
        cfg = _sample_for_cycle(rng, catalog, loop_cfg)
        assert cfg.learn is True, "learning must be ON for discovery trials"

        trial_dir = trials_root / f"trial_{cycle:06d}"
        if trial_dir.exists():
            shutil.rmtree(trial_dir)

        print(f"[cycle {cycle}] running seed={cfg.seed} "
              f"w0={cfg.w0} w1={cfg.w1} w2={cfg.w2} w3={cfg.w3} "
              f"w4={cfg.w4} w5={cfg.w5} p0={cfg.init_alive_prob} ...")

        run_trial(cfg, trial_dir, verbose=loop_cfg.verbose_sim)

        metrics = metrics_from_trajectory(trial_dir / "trajectory.npz")
        pf = prefilter(metrics, N=cfg.N)
        if not pf.passed:
            stats.prefilter_rejects += 1
            print(f"[cycle {cycle}] prefilter reject: {pf.reason}")
            if not loop_cfg.keep_rejects:
                shutil.rmtree(trial_dir, ignore_errors=True)
            continue

        if loop_cfg.dry_run:
            print(f"[cycle {cycle}] prefilter PASS (dry-run, skipping VLM)")
            if not loop_cfg.keep_rejects:
                shutil.rmtree(trial_dir, ignore_errors=True)
            continue

        stats.vlm_calls += 1
        result = judge_trial(
            trial_dir / "summary.png",
            catalog.one_liners(),
            model=loop_cfg.model,
        )

        if result.error:
            stats.vlm_errors += 1
            print(f"[cycle {cycle}] VLM error: {result.error}")
            # Keep trial for debugging API issues.
            (trial_dir / "judge_error.txt").write_text(
                result.error + "\n", encoding="utf-8"
            )
            continue

        if result.worth_saving:
            disc_id = catalog.next_id()
            dest = root / disc_id
            if dest.exists():
                shutil.rmtree(dest)
            _promote_trial(trial_dir, dest, result, cycle=cycle)
            catalog.append(
                disc_id,
                result.one_liner,
                cycle=cycle,
                judge=result.to_dict(),
            )
            stats.discoveries = catalog.count
            print(f"[cycle {cycle}] SAVED {disc_id} — {result.one_liner}")
            if not loop_cfg.keep_rejects:
                shutil.rmtree(trial_dir, ignore_errors=True)
        else:
            why = result.boring_reason or result.similarity_note or "not worth saving"
            print(f"[cycle {cycle}] reject: {why}")
            if not loop_cfg.keep_rejects:
                shutil.rmtree(trial_dir, ignore_errors=True)

    print()
    print("=== Discovery summary ===")
    print(f"  cycles:             {stats.cycles}")
    print(f"  prefilter rejects:  {stats.prefilter_rejects}")
    print(f"  VLM calls:          {stats.vlm_calls}")
    print(f"  VLM errors:         {stats.vlm_errors}")
    print(f"  discoveries total:  {catalog.count}")
    print(f"  catalog:            {catalog.md_path}")
    return stats
