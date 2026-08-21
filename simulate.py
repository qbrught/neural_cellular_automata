"""Run a full NCSA simulation: init -> n_steps of (forward + learn + snapshot) -> save.

The trajectory is written as a single .npz with per-snapshot arrays stacked
along the leading axis. The Config is saved as JSON alongside, and the
final parameters are saved as a torch checkpoint. This trio
(trajectory.npz + config.json + params_final.pt) is everything needed to
re-analyse or resume a run.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from config import Config
from dynamics import forward_step, make_u
from environment import Environment, generate_environment
from grid import Grid
from learning import gradient_step
from parameters import init_parameters
from state import State, init_state


def set_global_seed(seed: int) -> torch.Generator:
    """Seed every RNG and return a torch.Generator for explicit use."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    gen = torch.Generator().manual_seed(seed)
    return gen


def build_grid(cfg: Config) -> Grid:
    """Construct an initial Grid from a Config, using only its seed."""
    gen = set_global_seed(cfg.seed)
    state = init_state(cfg.N, cfg.d, cfg.init_alive_prob, gen, device=cfg.device)
    # Give state and memory a tiny amount of initial noise so ψ has signal
    # to act on from step 0. Without this, alive cells all have s=h=0 at the
    # start, and ψ sees identical inputs everywhere, which is a degenerate
    # symmetry the per-cell noise in the weights then has to break.
    state.s.normal_(mean=0.0, std=cfg.init_noise_std, generator=gen)
    state.h.normal_(mean=0.0, std=cfg.init_noise_std, generator=gen)
    params = init_parameters(
        cfg.N, cfg.d, cfg.hidden, cfg.init_noise_std, gen, device=cfg.device,
    ).requires_grad_(cfg.learn)
    # u is sampled from its OWN seed, independent of the run seed, so different
    # runs with different seeds share the same projection direction.
    u = make_u(cfg.d, cfg.u_seed, device=cfg.device)
    env = generate_environment(cfg)  # identity + no Generator if flag off
    if cfg.environment_heterogeneous:
        state.x = state.x * env.occupancy  # do not consume gen; do not zero s/h
    return Grid(state=state, params=params, u=u, env=env)


@dataclass
class Snapshot:
    """One step's worth of recorded state + summary stats."""
    step: int
    x: np.ndarray            # (N, N) uint8
    s: np.ndarray | None     # (N, N, d) float32 or None
    h: np.ndarray | None     # (N, N, d) float32 or None
    goals: np.ndarray        # (N, N) uint8 — may change under goal_inheritance
    alive_count: int
    reproducer_alive: int
    eliminator_alive: int
    loss_reproduce_mean: float
    loss_eliminate_mean: float
    loss_total: float


def make_snapshot(
    step_idx: int,
    state: State,
    stats: dict,
    include_vectors: bool,
) -> Snapshot:
    repro_alive = int((state.x * state.reproduce_mask().float()).sum().item())
    elim_alive = int((state.x * state.eliminate_mask().float()).sum().item())
    return Snapshot(
        step=step_idx,
        x=state.x.detach().cpu().numpy().astype(np.uint8),
        s=state.s.detach().cpu().numpy().astype(np.float32) if include_vectors else None,
        h=state.h.detach().cpu().numpy().astype(np.float32) if include_vectors else None,
        goals=state.goals.detach().cpu().numpy().astype(np.uint8),
        alive_count=stats["n_alive"],
        reproducer_alive=repro_alive,
        eliminator_alive=elim_alive,
        loss_reproduce_mean=stats["loss_reproduce_mean"],
        loss_eliminate_mean=stats["loss_eliminate_mean"],
        loss_total=stats["loss_total"],
    )


def save_trajectory(
    snapshots: list[Snapshot],
    rho: np.ndarray,
    out_path: Path,
    env: Environment | None = None,
) -> None:
    """Stack all snapshots into one .npz file.

    Goals are saved per step as (T, N, N) so Step C (goal inheritance) is
    analysable. Older loaders that expect 2D goals should treat ndim==2 as
    constant over time. rho remains constant (N, N). Experiment G maps are
    frozen (N, N) and always written when ``env`` is provided.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "step": np.array([s.step for s in snapshots], dtype=np.int32),
        "x": np.stack([s.x for s in snapshots]),  # (T, N, N) uint8
        "alive_count": np.array([s.alive_count for s in snapshots], dtype=np.int32),
        "reproducer_alive": np.array(
            [s.reproducer_alive for s in snapshots], dtype=np.int32
        ),
        "eliminator_alive": np.array(
            [s.eliminator_alive for s in snapshots], dtype=np.int32
        ),
        "loss_reproduce_mean": np.array(
            [s.loss_reproduce_mean for s in snapshots], dtype=np.float32
        ),
        "loss_eliminate_mean": np.array(
            [s.loss_eliminate_mean for s in snapshots], dtype=np.float32
        ),
        "loss_total": np.array(
            [s.loss_total for s in snapshots], dtype=np.float32
        ),
        "goals": np.stack([s.goals for s in snapshots]),  # (T, N, N) uint8
        "rho": rho,
    }
    if env is not None:
        arrays["occupancy"] = env.occupancy.detach().cpu().numpy().astype(np.float32)
        arrays["kappa_R"] = env.kappa_R.detach().cpu().numpy().astype(np.float32)
        arrays["kappa_E"] = env.kappa_E.detach().cpu().numpy().astype(np.float32)
        arrays["eta_scale_R"] = env.eta_scale_R.detach().cpu().numpy().astype(np.float32)
        arrays["eta_scale_E"] = env.eta_scale_E.detach().cpu().numpy().astype(np.float32)
    if snapshots[0].s is not None:
        arrays["s"] = np.stack([s.s for s in snapshots])  # (T, N, N, d)
        arrays["h"] = np.stack([s.h for s in snapshots])
    np.savez_compressed(out_path, **arrays)


def run(cfg: Config, output_dir: str | Path | None = None,
        verbose: bool = True, log_every: int = 50) -> Path:
    """Run a simulation end-to-end. Returns the path to the output directory.

    If output_dir is None, derives a timestamped path under cfg.output_dir.
    """
    if output_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_name = cfg.run_name or f"run_{ts}"
        output_dir = Path(cfg.output_dir) / run_name
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = build_grid(cfg)
    state, params, u = grid.state, grid.params, grid.u
    env = grid.env

    # rho is immutable; goals may change under goal_inheritance (saved per step).
    rho_np = state.rho.detach().cpu().numpy().astype(np.float32)

    snapshots: list[Snapshot] = []

    # t=0 snapshot is the INITIAL state, before any step is taken. Stats are
    # filled with zeros / NaNs since no learning has happened yet.
    initial_stats = {
        "n_alive": int(state.x.sum().item()),
        "loss_reproduce_mean": float("nan"),
        "loss_eliminate_mean": float("nan"),
        "loss_total": 0.0,
    }
    snapshots.append(
        make_snapshot(0, state, initial_stats, cfg.save_state_vectors)
    )

    if verbose:
        print(f"Run: {output_dir}")
        print(f"  N={cfg.N}, d={cfg.d}, hidden={cfg.hidden}, "
              f"eta={cfg.eta}, n_steps={cfg.n_steps}, seed={cfg.seed}")
        print(f"  Initial alive: {initial_stats['n_alive']} / {cfg.N*cfg.N}")
        if cfg.environment_heterogeneous:
            occ_frac = float(env.occupancy.mean().item()) if env is not None else 1.0
            print(
                f"  G env: preset={cfg.env_preset} env_seed={cfg.env_seed} "
                f"occ_frac={occ_frac:.3f}"
            )

    for t in range(1, cfg.n_steps + 1):
        step_out = forward_step(state, params, u, cfg, env=env)
        if cfg.learn:
            stats = gradient_step(state, step_out, params, cfg, env=env)
        else:
            stats = {
                "loss_total": 0.0,
                "loss_reproduce_mean": float("nan"),
                "loss_eliminate_mean": float("nan"),
                "n_alive": int(state.x.sum().item()),
            }
        state = step_out.next_state
        snapshots.append(make_snapshot(t, state, stats, cfg.save_state_vectors))

        if verbose and (t % log_every == 0 or t == cfg.n_steps):
            print(
                f"  step {t:5d} | alive {stats['n_alive']:4d} | "
                f"loss_r {stats['loss_reproduce_mean']:+.3f} | "
                f"loss_e {stats['loss_eliminate_mean']:+.3f}"
            )

    # Write outputs.
    cfg.save(output_dir / "config.json")
    save_trajectory(snapshots, rho_np, output_dir / "trajectory.npz", env=env)
    params.detach_clone().save(output_dir / "params_final.pt")

    if verbose:
        print(f"  Saved: {output_dir}")
    return output_dir
