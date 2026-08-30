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
from environment import Environment, edge_kappa_product, eta_map
from grid import gather_neighbours
from learning import gradient_step
from research.metrics import summarize_run
from research.protocol import FRAME_FRACS, frame_steps
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
    sender_elim = (state.goals == GOAL_ELIMINATE).unsqueeze(-1)
    sender_repro = (state.goals == GOAL_REPRODUCE).unsqueeze(-1)
    e_same = _rate(same_edge & sender_elim)
    e_cross = _rate(cross_edge & sender_elim)
    r_same = _rate(same_edge & sender_repro)
    r_cross = _rate(cross_edge & sender_repro)

    def _gap(a: float, b: float) -> float:
        if np.isfinite(a) and np.isfinite(b):
            return b - a
        return float("nan")

    return {
        "death_rate_same_edge": same_r,
        "death_rate_cross_edge": cross_r,
        "death_rate_cross_minus_same": gap,
        "n_same_edges": float(same_edge.sum().item()),
        "n_cross_edges": float(cross_edge.sum().item()),
        "death_rate_E_same": e_same,
        "death_rate_E_cross": e_cross,
        "death_rate_E_cross_minus_same": _gap(e_same, e_cross),
        "death_rate_R_same": r_same,
        "death_rate_R_cross": r_cross,
        "death_rate_R_cross_minus_same": _gap(r_same, r_cross),
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


@torch.no_grad()
def _type_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not mask.any():
        return float("nan")
    return float(values[mask].mean().item())


@torch.no_grad()
def _f_and_state_by_type(state, step_out) -> dict[str, float]:
    """D diagnostics: f-signal and ||s|| split by alive type."""
    fs = step_out.survival_inputs.f_signal
    alive = state.x > 0
    is_r = (state.goals == GOAL_REPRODUCE) & alive
    is_e = (state.goals == GOAL_ELIMINATE) & alive
    s_norm = torch.linalg.vector_norm(state.s, dim=-1)
    r_fs = _type_mean(fs, is_r)
    e_fs = _type_mean(fs, is_e)
    r_sn = _type_mean(s_norm, is_r)
    e_sn = _type_mean(s_norm, is_e)
    gap = (
        abs(r_fs - e_fs)
        if np.isfinite(r_fs) and np.isfinite(e_fs)
        else float("nan")
    )
    sn_gap = (
        abs(r_sn - e_sn)
        if np.isfinite(r_sn) and np.isfinite(e_sn)
        else float("nan")
    )
    return {
        "f_signal_R_mean": r_fs,
        "f_signal_E_mean": e_fs,
        "f_signal_type_gap": gap,
        "s_norm_R": r_sn,
        "s_norm_E": e_sn,
        "s_norm_type_gap": sn_gap,
    }


@torch.no_grad()
def _env_spatial(state, cfg: Config, env: Environment | None) -> dict[str, float]:
    """G diagnostics: occupancy of low-κ cells and edge conductivity."""
    n_alive = int((state.x > 0).sum().item())
    out = {
        "alive_low_kappa": float("nan"),
        "alive_high_kappa": float("nan"),
        "frac_alive_low_kappa": float("nan"),
        "ra_low_kappa": float("nan"),
        "ea_low_kappa": float("nan"),
        "kappa_edge_mean": float("nan"),
        "eta_mean_alive": float("nan"),
    }
    if env is None:
        emap = eta_map(state, cfg, None)
        alive = state.x > 0
        out["eta_mean_alive"] = (
            float(emap[alive].mean().item()) if alive.any() else float("nan")
        )
        out["alive_low_kappa"] = 0.0
        out["alive_high_kappa"] = float(n_alive)
        out["frac_alive_low_kappa"] = 0.0
        out["ra_low_kappa"] = 0.0
        out["ea_low_kappa"] = 0.0
        return out
    kbar = 0.5 * (env.kappa_R + env.kappa_E)
    low = kbar < 0.5
    alive = state.x > 0
    is_r = (state.goals == GOAL_REPRODUCE) & alive
    is_e = (state.goals == GOAL_ELIMINATE) & alive
    n_low = int((alive & low).sum().item())
    n_high = int((alive & ~low).sum().item())
    out["alive_low_kappa"] = float(n_low)
    out["alive_high_kappa"] = float(n_high)
    out["frac_alive_low_kappa"] = (
        n_low / n_alive if n_alive > 0 else float("nan")
    )
    out["ra_low_kappa"] = float((is_r & low).sum().item())
    out["ea_low_kappa"] = float((is_e & low).sum().item())
    kprod = edge_kappa_product(state, env)
    edge = (state.x.unsqueeze(-1) > 0) & (gather_neighbours(state.x) > 0)
    if edge.any():
        out["kappa_edge_mean"] = float(kprod[edge].mean().item())
    emap = eta_map(state, cfg, env)
    out["eta_mean_alive"] = (
        float(emap[alive].mean().item()) if alive.any() else float("nan")
    )
    return out


def _extra_diagnostics(
    state, step_out, cfg: Config, env: Environment | None
) -> dict[str, float]:
    """Hook for letter-specific series. Add collectors here, not in the loop."""
    out: dict[str, float] = {}
    out.update(_f_and_state_by_type(state, step_out))
    out.update(_env_spatial(state, cfg, env))
    return out


def _env_maps(env: Environment | None) -> dict[str, np.ndarray]:
    if env is None:
        return {}
    return {
        "occupancy": env.occupancy.detach().cpu().numpy().astype(np.float32),
        "kappa_R": env.kappa_R.detach().cpu().numpy().astype(np.float32),
        "kappa_E": env.kappa_E.detach().cpu().numpy().astype(np.float32),
        "eta_scale_R": env.eta_scale_R.detach().cpu().numpy().astype(np.float32),
        "eta_scale_E": env.eta_scale_E.detach().cpu().numpy().astype(np.float32),
    }


def run_experiment(
    version: VersionSpec,
    base_cfg: Config,
    seed: int,
    *,
    n_steps: int | None = None,
    out_dir: Path | None = None,
    log_every: int = 0,
    arm_id: str | None = None,
    save_frames: bool = False,
    frame_fracs: tuple[float, ...] = FRAME_FRACS,
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
    env = grid.env
    env_use = env if cfg.environment_heterogeneous else None
    label = arm_id or version.id
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
        # Experiment F: soft coexistence diagnostics
        "coexistence_barrier": [],
        "soft_rho_R": [],
        "soft_rho_E": [],
        "min_type_frac": [],
        # D: type-split f-signal / state norm
        "f_signal_R_mean": [],
        "f_signal_E_mean": [],
        "f_signal_type_gap": [],
        "s_norm_R": [],
        "s_norm_E": [],
        "s_norm_type_gap": [],
        # G: low-κ occupancy and scales
        "alive_low_kappa": [],
        "alive_high_kappa": [],
        "frac_alive_low_kappa": [],
        "ra_low_kappa": [],
        "ea_low_kappa": [],
        "kappa_edge_mean": [],
        "eta_mean_alive": [],
        # B: sender-type death rates (filled by _typed_edge_death_rates)
        "death_rate_E_same": [],
        "death_rate_E_cross": [],
        "death_rate_E_cross_minus_same": [],
        "death_rate_R_same": [],
        "death_rate_R_cross": [],
        "death_rate_R_cross_minus_same": [],
    }

    n = int(cfg.n_steps)
    dump_at = set(frame_steps(n, frame_fracs)) if save_frames else set()
    frame_x: list[np.ndarray] = []
    frame_g: list[np.ndarray] = []
    frame_t: list[int] = []

    def _snap(step_idx: int, st) -> None:
        if step_idx not in dump_at or step_idx in frame_t:
            return
        frame_t.append(step_idx)
        frame_x.append(st.x.detach().cpu().numpy().astype(np.uint8))
        frame_g.append(st.goals.detach().cpu().numpy().astype(np.uint8))

    _snap(0, state)
    for t in range(n):
        # Need grad for learning; diagnostics use no_grad snapshots after step_out.
        step_out = forward_step(state, params, u, cfg, env=env)
        if cfg.learn:
            stats = gradient_step(state, step_out, params, cfg, env=env)
        else:
            stats = {
                "loss_reproduce_mean": float("nan"),
                "loss_eliminate_mean": float("nan"),
                "loss_total": 0.0,
                "n_alive": int(state.x.sum().item()),
                "coexistence_barrier": 0.0,
                "soft_rho_R": float("nan"),
                "soft_rho_E": float("nan"),
            }

        x = state.x
        g = state.goals
        n_alive = int(x.sum().item())
        n_ra = int(((g == GOAL_REPRODUCE) & (x > 0)).sum().item())
        n_ea = int(((g == GOAL_ELIMINATE) & (x > 0)).sum().item())
        buckets["alive"].append(n_alive)
        buckets["reproducer_alive"].append(n_ra)
        buckets["eliminator_alive"].append(n_ea)
        buckets["loss_r"].append(stats["loss_reproduce_mean"])
        buckets["loss_e"].append(stats["loss_eliminate_mean"])
        buckets["loss_total"].append(stats["loss_total"])
        buckets["coexistence_barrier"].append(
            float(stats.get("coexistence_barrier", 0.0))
        )
        buckets["soft_rho_R"].append(float(stats.get("soft_rho_R", float("nan"))))
        buckets["soft_rho_E"].append(float(stats.get("soft_rho_E", float("nan"))))
        if n_alive > 0:
            buckets["min_type_frac"].append(min(n_ra, n_ea) / n_alive)
        else:
            buckets["min_type_frac"].append(float("nan"))

        diag = _vote_diagnostics(state, step_out)
        for k, v in diag.items():
            buckets[k].append(v)

        # Typed edge death rates use pre-step state vs next-step alive.
        death = _typed_edge_death_rates(state, step_out.next_state.x)
        for k, v in death.items():
            buckets[k].append(v)

        extra = _extra_diagnostics(state, step_out, cfg, env_use)
        for k, v in extra.items():
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
                f"  [{label} seed={seed}] t={t:4d} "
                f"alive={buckets['alive'][-1]:3d} "
                f"ra={buckets['reproducer_alive'][-1]:3d} "
                f"ea={buckets['eliminator_alive'][-1]:3d} "
                f"g_repro={buckets['goal_frac_repro'][-1]:.3f}"
            )

        state = step_out.next_state
        _snap(t + 1, state)

    series = {k: np.asarray(v, dtype=float) for k, v in buckets.items()}
    summary = summarize_run(series, goal_frac_repro=goal_frac_initial)

    result = {
        "version_id": label,
        "spec_id": version.id,
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
                    "version_id": label,
                    "spec_id": version.id,
                    "arm_id": label,
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
        if save_frames and frame_x:
            maps = _env_maps(env)
            np.savez_compressed(
                out_dir / "frames.npz",
                steps=np.asarray(frame_t, dtype=np.int32),
                x=np.stack(frame_x),
                goals=np.stack(frame_g),
                **maps,
            )
        # Late snapshot so functional analysis can probe ψ without re-simulating.
        params.save(out_dir / "params_final.pt")
        torch.save(
            {
                "x": state.x.detach().cpu().clone(),
                "s": state.s.detach().cpu().clone(),
                "h": state.h.detach().cpu().clone(),
                "goals": state.goals.detach().cpu().clone(),
                "rho": state.rho.detach().cpu().clone(),
                "u": u.detach().cpu().clone(),
            },
            out_dir / "state_final.pt",
        )

    return result
