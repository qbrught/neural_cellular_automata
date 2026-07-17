"""Main discovery loop: sample → run → (prefilter) → VLM judge/guide → save."""

from __future__ import annotations

import json
import random
import secrets
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from config import Config
from discovery.catalog import Catalog
from discovery.evidence import run_trial
from discovery.judge import JudgeResult, judge_trial
from discovery.prefilter import metrics_from_trajectory, prefilter
from discovery.sample import (
    apply_proposal,
    config_knobs,
    heuristic_next_config,
    sample_config,
)


@dataclass
class LoopStats:
    cycles: int = 0
    prefilter_rejects: int = 0
    vlm_calls: int = 0
    vlm_errors: int = 0
    discoveries: int = 0
    guided_steps: int = 0
    explore_steps: int = 0
    random_steps: int = 0


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
    guided: bool = True
    explore_prob: float = 0.15  # chance to ignore guidance and pure-random
    history_len: int = 6


@dataclass
class TrialRecord:
    cycle: int
    knobs: dict[str, Any]
    outcome: str
    final_alive: int
    analysis: str = ""
    strategy: str = ""

    def history_line(self) -> str:
        k = self.knobs
        core = (
            f"w0={k.get('w0')} w1={k.get('w1')} w2={k.get('w2')} "
            f"w3={k.get('w3')} w4h={k.get('w4_help')} w4m={k.get('w4_harm')} w5={k.get('w5')} "
            f"p0={k.get('init_alive_prob')}"
        )
        extra = f" | {self.analysis}" if self.analysis else ""
        strat = f" | strategy={self.strategy}" if self.strategy else ""
        return (
            f"- cycle {self.cycle}: outcome={self.outcome} | final_alive={self.final_alive} "
            f"| {core}{strat}{extra}"
        )


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
    note_body = result.one_liner
    if result.analysis:
        note_body = f"{result.one_liner}\n\n{result.analysis}\n"
    note.write_text(note_body if note_body.endswith("\n") else note_body + "\n",
                    encoding="utf-8")

    meta = {
        "cycle": cycle,
        "one_liner": result.one_liner,
        "analysis": result.analysis,
        "strategy": result.strategy,
        "rationale": result.rationale,
        "next_config": result.next_config,
        "judge": result.to_dict(),
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _save_discovery(
    catalog: Catalog,
    root: Path,
    trial_dir: Path,
    result: JudgeResult,
    *,
    cycle: int,
) -> str:
    """Promote trial artifacts and append to the catalog immediately."""
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
    return disc_id


def _metrics_summary(metrics: dict[str, np.ndarray], N: int) -> str:
    alive = metrics["alive"]
    repro = metrics["repro"]
    elim = metrics["elim"]
    t = len(alive)
    n_cells = N * N
    mid = t // 2
    return (
        f"steps={t}, n_cells={n_cells}\n"
        f"alive: start={int(alive[0])}, mid={int(alive[mid])}, "
        f"final={int(alive[-1])}, peak={int(alive.max())}, min={int(alive.min())}\n"
        f"repro final={int(repro[-1])}, elim final={int(elim[-1])}\n"
        f"alive std (full)={float(alive.std()):.2f}, "
        f"alive range last quarter="
        f"{int(alive[3*t//4:].max()) - int(alive[3*t//4:].min())}"
    )


def _random_or_mutate(
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


def _choose_config(
    rng: random.Random,
    catalog: Catalog,
    loop_cfg: LoopConfig,
    *,
    pending: Config | None,
    stats: LoopStats,
) -> tuple[Config, str]:
    """Return (config, source) where source is guided|explore|random."""
    # Occasional pure explore even when we have guidance.
    if pending is not None and loop_cfg.guided and rng.random() < loop_cfg.explore_prob:
        stats.explore_steps += 1
        return _random_or_mutate(rng, catalog, loop_cfg), "explore"

    if pending is not None and loop_cfg.guided:
        stats.guided_steps += 1
        return pending, "guided"

    stats.random_steps += 1
    return _random_or_mutate(rng, catalog, loop_cfg), "random"


def run_discovery(loop_cfg: LoopConfig) -> LoopStats:
    root = Path(loop_cfg.output_root)
    root.mkdir(parents=True, exist_ok=True)
    trials_root = root / "trials"
    trials_root.mkdir(parents=True, exist_ok=True)

    catalog = Catalog(root)
    sampler_seed = (
        loop_cfg.sampler_seed
        if loop_cfg.sampler_seed is not None
        else secrets.randbits(63)
    )
    rng = random.Random(sampler_seed)
    stats = LoopStats(discoveries=catalog.count)

    history: list[TrialRecord] = []
    pending_next: Config | None = None

    print(f"Discovery root: {root.resolve()}")
    print(f"  existing discoveries: {catalog.count}")
    print(f"  max_cycles={loop_cfg.max_cycles}  "
          f"target_discoveries={loop_cfg.target_discoveries}")
    print(f"  n_steps={loop_cfg.n_steps}  N={loop_cfg.N}  device={loop_cfg.device}")
    print(f"  model={loop_cfg.model}  dry_run={loop_cfg.dry_run}")
    print(f"  guided={loop_cfg.guided}  explore_prob={loop_cfg.explore_prob}")
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
        cfg, source = _choose_config(
            rng, catalog, loop_cfg, pending=pending_next, stats=stats
        )
        pending_next = None  # consume proposal
        assert cfg.learn is True, "learning must be ON for discovery trials"

        trial_dir = trials_root / f"trial_{cycle:06d}"
        if trial_dir.exists():
            shutil.rmtree(trial_dir)

        knobs = config_knobs(cfg)
        print(
            f"[cycle {cycle}|{source}] seed={cfg.seed} "
            f"w0={cfg.w0} w1={cfg.w1} w2={cfg.w2} w3={cfg.w3} "
            f"w4h={cfg.w4_help} w4m={cfg.w4_harm} w5={cfg.w5} p0={cfg.init_alive_prob} eta={cfg.eta}"
        )

        run_trial(cfg, trial_dir, verbose=loop_cfg.verbose_sim)

        metrics = metrics_from_trajectory(trial_dir / "trajectory.npz")
        pf = prefilter(metrics, N=cfg.N)
        metrics_text = _metrics_summary(metrics, cfg.N)
        final_alive = int(metrics["alive"][-1])
        hist_lines = [h.history_line() for h in history[-loop_cfg.history_len :]]

        # --- dry-run: no VLM; optional heuristic guidance ---
        if loop_cfg.dry_run:
            if not pf.passed:
                stats.prefilter_rejects += 1
                outcome = f"prefilter:{pf.reason}"
                print(f"[cycle {cycle}] prefilter reject: {pf.reason}")
            else:
                outcome = "prefilter_pass_dry_run"
                print(f"[cycle {cycle}] prefilter PASS (dry-run, skipping VLM)")

            if loop_cfg.guided:
                pending_next = heuristic_next_config(
                    cfg,
                    outcome=outcome,
                    rng=rng,
                    n_steps=loop_cfg.n_steps,
                    N=loop_cfg.N,
                    device=loop_cfg.device,
                )
                print(f"[cycle {cycle}] heuristic next ready (dry-run guided)")

            history.append(
                TrialRecord(
                    cycle=cycle,
                    knobs=knobs,
                    outcome=outcome,
                    final_alive=final_alive,
                    analysis="dry-run",
                )
            )
            if not loop_cfg.keep_rejects:
                shutil.rmtree(trial_dir, ignore_errors=True)
            continue

        # --- guided mode: VLM even on prefilter fail (steering needs feedback) ---
        # --- unguided: only VLM when prefilter passes ---
        call_vlm = loop_cfg.guided or pf.passed

        if not pf.passed:
            stats.prefilter_rejects += 1
            print(f"[cycle {cycle}] prefilter reject: {pf.reason}")
            if not call_vlm:
                history.append(
                    TrialRecord(
                        cycle=cycle,
                        knobs=knobs,
                        outcome=f"prefilter:{pf.reason}",
                        final_alive=final_alive,
                    )
                )
                if not loop_cfg.keep_rejects:
                    shutil.rmtree(trial_dir, ignore_errors=True)
                continue

        if not call_vlm:
            # Shouldn't happen, but keep safe.
            if not loop_cfg.keep_rejects:
                shutil.rmtree(trial_dir, ignore_errors=True)
            continue

        stats.vlm_calls += 1
        result = judge_trial(
            trial_dir / "summary.png",
            catalog.one_liners(),
            config_knobs=knobs,
            metrics_summary=metrics_text,
            prefilter_reason=None if pf.passed else pf.reason,
            history_lines=hist_lines,
            model=loop_cfg.model,
        )

        if result.error:
            stats.vlm_errors += 1
            print(f"[cycle {cycle}] VLM error: {result.error}")
            (trial_dir / "judge_error.txt").write_text(
                result.error + "\n", encoding="utf-8"
            )
            # Fallback heuristic so the chain doesn't die.
            if loop_cfg.guided:
                outcome = f"vlm_error;prefilter={pf.reason if not pf.passed else 'pass'}"
                pending_next = heuristic_next_config(
                    cfg,
                    outcome=outcome,
                    rng=rng,
                    n_steps=loop_cfg.n_steps,
                    N=loop_cfg.N,
                    device=loop_cfg.device,
                )
                print(f"[cycle {cycle}] fallback heuristic next (VLM error)")
            history.append(
                TrialRecord(
                    cycle=cycle,
                    knobs=knobs,
                    outcome="vlm_error",
                    final_alive=final_alive,
                    analysis=result.error or "",
                )
            )
            continue

        if result.analysis:
            print(f"[cycle {cycle}] analysis: {result.analysis}")
        if result.strategy or result.rationale:
            print(
                f"[cycle {cycle}] guide: strategy={result.strategy or '?'} "
                f"— {result.rationale or ''}"
            )

        # Build next config from VLM proposal (clamped).
        if loop_cfg.guided:
            proposed = apply_proposal(
                cfg,
                result.next_config,
                rng,
                n_steps=loop_cfg.n_steps,
                N=loop_cfg.N,
                device=loop_cfg.device,
                jitter=0.02,
            )
            if proposed is None:
                proposed = heuristic_next_config(
                    cfg,
                    outcome=(
                        "saved" if result.worth_saving
                        else (pf.reason if not pf.passed else "reject")
                    ),
                    rng=rng,
                    n_steps=loop_cfg.n_steps,
                    N=loop_cfg.N,
                    device=loop_cfg.device,
                )
                print(f"[cycle {cycle}] VLM next_config missing/invalid → heuristic")
            else:
                changed = [
                    k for k in result.next_config
                    if k in config_knobs(proposed)
                    and k not in ("seed", "u_seed")
                ]
                print(f"[cycle {cycle}] next_config knobs: {changed or result.next_config}")
            pending_next = proposed

        # Only save when prefilter passed AND VLM says worth_saving.
        if pf.passed and result.worth_saving:
            disc_id = _save_discovery(
                catalog, root, trial_dir, result, cycle=cycle
            )
            stats.discoveries = catalog.count
            print(
                f"[cycle {cycle}] SAVED {disc_id} — {result.one_liner} "
                f"(catalog updated: {catalog.count} total)"
            )
            outcome = f"saved:{disc_id}"
            if not loop_cfg.keep_rejects:
                shutil.rmtree(trial_dir, ignore_errors=True)
        else:
            if not pf.passed:
                outcome = f"prefilter:{pf.reason}"
            else:
                why = result.boring_reason or result.similarity_note or "not worth saving"
                print(f"[cycle {cycle}] reject: {why}")
                outcome = f"reject:{why}"
            if not loop_cfg.keep_rejects:
                shutil.rmtree(trial_dir, ignore_errors=True)

        history.append(
            TrialRecord(
                cycle=cycle,
                knobs=knobs,
                outcome=outcome,
                final_alive=final_alive,
                analysis=result.analysis,
                strategy=result.strategy,
            )
        )

    print()
    print("=== Discovery summary ===")
    print(f"  cycles:             {stats.cycles}")
    print(f"  prefilter rejects:  {stats.prefilter_rejects}")
    print(f"  VLM calls:          {stats.vlm_calls}")
    print(f"  VLM errors:         {stats.vlm_errors}")
    print(f"  guided steps:       {stats.guided_steps}")
    print(f"  explore steps:      {stats.explore_steps}")
    print(f"  random steps:       {stats.random_steps}")
    print(f"  discoveries total:  {catalog.count}")
    print(f"  catalog:            {catalog.md_path}")
    return stats
