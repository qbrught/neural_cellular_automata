"""Run one (version × seed) experiment and record rich time series."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import Config
from dynamics import forward_step
from grid import gather_neighbours
from learning import gradient_step
from research.metrics import summarize_run
from research.versions import VersionSpec
from simulate import build_grid
from state import GOAL_ELIMINATE, GOAL_REPRODUCE


def _channel_mean(
    mask_i: torch.Tensor,
    channel: torch.Tensor,
    edge_mask: torch.Tensor,
) -> float:
    if not mask_i.any():
        return float("nan")
    v = channel[mask_i]
    em = edge_mask[mask_i]
    if em.sum() == 0:
        return float("nan")
    return float(v[em].detach().mean().item())


def _same_goal_edge_frac(state) -> float:
    """Fraction of alive–alive Moore edges that are same-goal (toroidal)."""
    alive = state.x > 0
    goals = state.goals
    # 4-neighbour for a cheap segregation proxy (stable, easy to interpret).
    same = 0.0
    tot = 0.0
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        n_alive = torch.roll(torch.roll(alive, dy, 0), dx, 1)
        n_goal = torch.roll(torch.roll(goals, dy, 0), dx, 1)
        edge = alive & n_alive
        tot += float(edge.sum().item())
        same += float((edge & (goals == n_goal)).sum().item())
    if tot <= 0:
        return float("nan")
    return same / tot


@torch.no_grad()
def _typed_edge_death_rates(state, x_next: torch.Tensor) -> dict[str, float]:
    """Death rates of alive senders on same-type vs cross-type directed edges.

    For every directed Moore edge i → j where both are alive at t:
      same-type  if goals match
      cross-type if goals differ

    death_rate_* = fraction of such edges for which sender i is dead at t+1.

    Interpretation: how often does a cell die when its neighbour is kin vs foe?
    gap = cross − same  (positive ⇒ cross-type contact is more lethal).
    """
    alive = state.x > 0
    dies = alive & (x_next <= 0)
    nb_alive = gather_neighbours(state.x) > 0                          # (N,N,8)
    nb_goal = gather_neighbours(state.goals.float())                   # (N,N,8)
    my_g = state.goals.float().unsqueeze(-1)                           # (N,N,1)
    sender_alive = alive.unsqueeze(-1)                                 # (N,N,1)
    edge = sender_alive & nb_alive                                     # both alive at t
    same_edge = edge & (my_g == nb_goal)
    cross_edge = edge & (my_g != nb_goal)
    dies_exp = dies.unsqueeze(-1).expand_as(edge)

    def _rate(mask: torch.Tensor) -> float:
        n = int(mask.sum().item())
        if n == 0:
            return float("nan")
        return float(dies_exp[mask].float().mean().item())

    same_r = _rate(same_edge)
    cross_r = _rate(cross_edge)
    gap = (
        cross_r - same_r
        if np.isfinite(same_r) and np.isfinite(cross_r)
        else float("nan")
    )
    return {
        "death_rate_same_edge": same_r,
        "death_rate_cross_edge": cross_r,
        "death_rate_cross_minus_same": gap,
        "n_same_edges": float(same_edge.sum().item()),
        "n_cross_edges": float(cross_edge.sum().item()),
    }


@torch.no_grad()
def _vote_diagnostics(state, step_out) -> dict[str, float]:
    votes = step_out.outgoing_votes  # (N,N,8,2)
    help_v = votes[..., 0]
    harm_v = votes[..., 1]
    nb_goal = gather_neighbours(state.goals.float())
    nb_alive = gather_neighbours(state.x)
    alive = state.x > 0
    i_repro = (state.goals == GOAL_REPRODUCE) & alive
    i_elim = (state.goals == GOAL_ELIMINATE) & alive
    my_g = state.goals.float().unsqueeze(-1)
    same = (my_g == nb_goal) & (nb_alive > 0)
    diff = (my_g != nb_goal) & (nb_alive > 0)
    return {
        "vote_R_help_kin": _channel_mean(i_repro, help_v, same),
        "vote_R_harm_foe": _channel_mean(i_repro, harm_v, diff),
        "vote_E_help_kin": _channel_mean(i_elim, help_v, same),
        "vote_E_harm_foe": _channel_mean(i_elim, harm_v, diff),
        "V_kin_mean": float(step_out.survival_inputs.V_kin.mean().item()),
        "V_foe_mean": float(step_out.survival_inputs.V_foe.mean().item()),
        "same_goal_edge_frac": _same_goal_edge_frac(state),
    }


def run_experiment(
    version: VersionSpec,
    base_cfg: Config,
    seed: int,
    *,
    n_steps: int | None = None,
    out_dir: Path | None = None,
    log_every: int = 0,
) -> dict[str, Any]:
    """Run one version on one seed; optionally write artifacts under out_dir.

    Returns dict with keys:
      series (dict of np arrays), summary (dict), config (dict), version_id, seed
    """
    cfg = version.apply(base_cfg)
    cfg = Config(**{**asdict(cfg), "seed": seed})
    if n_steps is not None:
        cfg.n_steps = n_steps
    if cfg.n_steps is None:
        raise ValueError("n_steps must be finite for the research suite")
    cfg.__post_init__()

    grid = build_grid(cfg)
    state, params, u = grid.state, grid.params, grid.u
    # Initial goal fraction over all cells (latent map); used for density residual
    # and as the baseline for goal_frac drift under Step C.
    goal_frac_initial = float(
        (state.goals == GOAL_REPRODUCE).float().mean().item()
    )

    buckets: dict[str, list] = {
        "alive": [],
        "reproducer_alive": [],
        "eliminator_alive": [],
        "loss_r": [],
        "loss_e": [],
        "loss_total": [],
        "V_kin_mean": [],
        "V_foe_mean": [],
        "vote_R_help_kin": [],
        "vote_R_harm_foe": [],
        "vote_E_help_kin": [],
        "vote_E_harm_foe": [],
        "same_goal_edge_frac": [],
        "death_rate_same_edge": [],
        "death_rate_cross_edge": [],
        "death_rate_cross_minus_same": [],
        "n_same_edges": [],
        "n_cross_edges": [],
        # Step C: goal composition (all cells + among alive)
        "goal_frac_repro": [],
        "alive_goal_frac_repro": [],
    }

    n = int(cfg.n_steps)
    for t in range(n):
        # Need grad for learning; diagnostics use no_grad snapshots after step_out.
        step_out = forward_step(state, params, u, cfg)
        if cfg.learn:
            stats = gradient_step(state, step_out, params, cfg)
        else:
            stats = {
                "loss_reproduce_mean": float("nan"),
                "loss_eliminate_mean": float("nan"),
                "loss_total": 0.0,
                "n_alive": int(state.x.sum().item()),
            }

        x = state.x
        g = state.goals
        buckets["alive"].append(int(x.sum().item()))
        buckets["reproducer_alive"].append(
            int(((g == GOAL_REPRODUCE) & (x > 0)).sum().item())
        )
        buckets["eliminator_alive"].append(
            int(((g == GOAL_ELIMINATE) & (x > 0)).sum().item())
        )
        buckets["loss_r"].append(stats["loss_reproduce_mean"])
        buckets["loss_e"].append(stats["loss_eliminate_mean"])
        buckets["loss_total"].append(stats["loss_total"])

        diag = _vote_diagnostics(state, step_out)
        for k, v in diag.items():
            buckets[k].append(v)

        # Typed edge death rates use pre-step state vs next-step alive.
        death = _typed_edge_death_rates(state, step_out.next_state.x)
        for k, v in death.items():
            buckets[k].append(v)

        # Goal fractions from post-step state so inheritance this step is visible.
        g_next = step_out.next_state.goals
        x_next = step_out.next_state.x
        buckets["goal_frac_repro"].append(
            float((g_next == GOAL_REPRODUCE).float().mean().item())
        )
        alive_next = x_next > 0
        if alive_next.any():
            buckets["alive_goal_frac_repro"].append(
                float((g_next[alive_next] == GOAL_REPRODUCE).float().mean().item())
            )
        else:
            buckets["alive_goal_frac_repro"].append(float("nan"))

        if log_every and (t % log_every == 0 or t == n - 1):
            print(
                f"  [{version.id} seed={seed}] t={t:4d} "
                f"alive={buckets['alive'][-1]:3d} "
                f"ra={buckets['reproducer_alive'][-1]:3d} "
                f"ea={buckets['eliminator_alive'][-1]:3d} "
                f"g_repro={buckets['goal_frac_repro'][-1]:.3f}"
            )

        state = step_out.next_state

    series = {k: np.asarray(v, dtype=float) for k, v in buckets.items()}
    summary = summarize_run(series, goal_frac_repro=goal_frac_initial)

    result = {
        "version_id": version.id,
        "version_title": version.title,
        "seed": seed,
        "goal_frac_repro": goal_frac_initial,
        "config": cfg.to_dict(),
        "series": series,
        "summary": summary,
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "series.npz", **series)
        with (out_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        with (out_dir / "meta.json").open("w") as f:
            json.dump(
                {
                    "version_id": version.id,
                    "version_title": version.title,
                    "version_description": version.description,
                    "hypothesis": version.hypothesis,
                    "seed": seed,
                    "goal_frac_repro": goal_frac_initial,
                    "config": cfg.to_dict(),
                },
                f,
                indent=2,
            )
        cfg.save(out_dir / "config.json")

    return result
