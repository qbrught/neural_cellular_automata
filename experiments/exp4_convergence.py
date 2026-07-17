"""Experiment 4 — Convergence probe.

Question: at the end of a run, has learning converged (params stable) or is
the system still drifting? Loss curves can look flat while parameters keep
moving, so we measure parameter change directly.

Method: run a single simulation, snapshot the psi/f weight tensors every K
steps, and plot the per-step L2 change. A curve decaying toward zero means
convergence; a plateau above zero means persistent drift (the ecosystem
never settles its policies).

This one does NOT use _common.run_metrics because it needs access to the
live params tensor between steps.

Run:  python -m experiments.exp4_convergence
Out:  runs/exp4_convergence/summary.png + printed drift stats
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from dynamics import forward_step
from learning import gradient_step
from simulate import build_grid

OUT = Path("runs/exp4_convergence")
STEPS = 600
GRID_N = 30
SNAP_EVERY = 5


def param_vector(params):
    """Flatten all learnable tensors into one vector (detached copy)."""
    return torch.cat([t.detach().reshape(-1) for t in params.tensors()]).clone()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config(N=GRID_N, n_steps=STEPS, seed=7,
                 w0=-0.05, w1=0.4, w2=0.4, w3=-0.4, w4_help=0.6, w4_harm=0.6, w5=0.3, eta=0.08)
    if "learn" in Config.__dataclass_fields__:
        cfg = replace(cfg, learn=True)

    grid = build_grid(cfg)
    state, params, u = grid.state, grid.params, grid.u

    prev = param_vector(params)
    steps_axis, drift, alive_axis = [], [], []

    for t in range(STEPS):
        step_out = forward_step(state, params, u, cfg)
        gradient_step(state, step_out, params, cfg)
        state = step_out.next_state

        if t % SNAP_EVERY == 0:
            cur = param_vector(params)
            delta = (cur - prev).norm().item() / SNAP_EVERY  # avg per-step L2
            prev = cur
            steps_axis.append(t)
            drift.append(delta)
            alive_axis.append(int(state.x.sum().item()))

    steps_axis = np.array(steps_axis)
    drift = np.array(drift)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(steps_axis, drift, color="#38bdf8", linewidth=1.5,
             label="param drift (L2/step)")
    ax1.set_xlabel("step"); ax1.set_ylabel("parameter drift", color="#38bdf8")
    ax1.tick_params(axis="y", labelcolor="#38bdf8")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(steps_axis, alive_axis, color="#94a3b8", alpha=0.6,
             linewidth=1.0, label="alive count")
    ax2.set_ylabel("alive count", color="#94a3b8")
    ax2.tick_params(axis="y", labelcolor="#94a3b8")

    ax1.set_title("Parameter drift over time (convergence probe)")
    fig.tight_layout()
    fig.savefig(OUT / "summary.png", dpi=120)

    early = drift[: len(drift) // 4].mean()
    late = drift[-len(drift) // 4:].mean()
    print(f"early drift (first quarter): {early:.4f}")
    print(f"late  drift (last quarter):  {late:.4f}")
    print(f"ratio late/early: {late / early:.3f}  "
          f"(<<1 => converging, ~1 => persistent drift)")
    print(f"Saved {OUT/'summary.png'}")


if __name__ == "__main__":
    main()
