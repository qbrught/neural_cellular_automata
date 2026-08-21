"""Shared helpers for experiment scripts.

Why this exists: simulate.run() writes a trajectory to disk, which is great
for the UI and for archiving, but for parameter sweeps we usually just want
the per-step metrics back in memory as numpy arrays. run_metrics() gives us
that, with optional hooks for freezing learning and perturbing state mid-run.

All experiments import from here so the run loop lives in exactly one place.
If forward_step / gradient_step signatures change, only this file updates.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np
import torch

from config import Config
from dynamics import forward_step
from learning import gradient_step
from simulate import build_grid
from state import State


def _cfg_has_learn_flag() -> bool:
    """Whether Config carries a `learn` bool (added in the local build)."""
    return "learn" in Config.__dataclass_fields__


def run_metrics(
    cfg: Config,
    learn: bool = True,
    steps: int | None = None,
    perturb_at: int | None = None,
    perturb_fn: Callable[[State, torch.Generator], None] | None = None,
    perturb_seed: int = 999,
    record_grids: bool = False,
) -> dict:
    """Run a simulation and return per-step metrics in memory.

    Args:
        cfg: Config to run. Its n_steps is used unless `steps` overrides.
        learn: if False, gradient_step is skipped (frozen MLPs).
        steps: override cfg.n_steps for this run.
        perturb_at: step index at which to call perturb_fn (before the step).
        perturb_fn: fn(state, generator) mutating state in place.
        perturb_seed: seed for the perturbation generator (independent of run).
        record_grids: if True, also store the (N,N) display grid each step
                      (0 dead / 1 repro-alive / 2 elim-alive). Memory-heavy.

    Returns dict of numpy arrays:
        alive, reproducer_alive, eliminator_alive, loss_r, loss_e
        (and 'grids' if record_grids).
    """
    from state import GOAL_REPRODUCE, GOAL_ELIMINATE

    n = steps if steps is not None else cfg.n_steps
    if n is None:
        raise ValueError("run_metrics needs a finite step count")

    grid = build_grid(cfg)
    state, params, u = grid.state, grid.params, grid.u
    env = grid.env

    # If the local build has a `learn` flag, respect the argument by cloning cfg.
    if _cfg_has_learn_flag():
        cfg = replace(cfg, learn=learn)

    pgen = torch.Generator().manual_seed(perturb_seed)

    alive, ra, ea, lr, le = [], [], [], [], []
    grids = []

    for t in range(n):
        if perturb_at is not None and t == perturb_at and perturb_fn is not None:
            with torch.no_grad():
                perturb_fn(state, pgen)

        step_out = forward_step(state, params, u, cfg, env=env)
        if learn:
            stats = gradient_step(state, step_out, params, cfg, env=env)
        else:
            stats = {
                "loss_reproduce_mean": float("nan"),
                "loss_eliminate_mean": float("nan"),
                "n_alive": int(state.x.sum().item()),
                "loss_total": 0.0,
            }
        state = step_out.next_state

        x = state.x
        g = state.goals
        alive.append(int(x.sum().item()))
        ra.append(int(((g == GOAL_REPRODUCE) & (x > 0)).sum().item()))
        ea.append(int(((g == GOAL_ELIMINATE) & (x > 0)).sum().item()))
        lr.append(stats["loss_reproduce_mean"])
        le.append(stats["loss_eliminate_mean"])

        if record_grids:
            xn = x.cpu().numpy().astype(np.uint8)
            gn = g.cpu().numpy().astype(np.uint8)
            disp = np.zeros_like(xn)
            al = xn.astype(bool)
            disp[al & (gn == GOAL_REPRODUCE)] = 1
            disp[al & (gn == GOAL_ELIMINATE)] = 2
            grids.append(disp)

    out = {
        "alive": np.array(alive),
        "reproducer_alive": np.array(ra),
        "eliminator_alive": np.array(ea),
        "loss_r": np.array(lr, dtype=float),
        "loss_e": np.array(le, dtype=float),
    }
    if record_grids:
        out["grids"] = np.stack(grids)
    return out


def summarise_tail(metrics: dict, tail_frac: float = 0.2) -> dict:
    """Mean of each metric over the last tail_frac of steps (the 'settled' regime)."""
    out = {}
    for k, v in metrics.items():
        if k == "grids":
            continue
        arr = np.asarray(v, dtype=float)
        tail = arr[int(len(arr) * (1 - tail_frac)):]
        if len(tail) == 0 or np.all(np.isnan(tail)):
            out[k] = float("nan")
        else:
            out[k] = float(np.nanmean(tail))
    return out
