"""Experiment 5 — Seed growth.

Question: your system was designed as a death-biased persistent ecosystem,
not a grow-from-seed system. Before investing in a growing/infinite grid,
check the prerequisite: can the population expand from a small alive seed in
an otherwise-dead grid, or does it just collapse?

Method: build a grid, force everything dead except a small central square,
run, and track alive count and the bounding-box radius of the living region.
If the radius grows, growth is viable and a growing grid is worth building.
If it collapses, the current rule/weights won't support unbounded growth
without changes.

This overrides the initial state after build_grid, so it manages its own loop.

Run:  python -m experiments.exp5_seed_growth
Out:  runs/exp5_seed_growth/summary.png + final grid image + printed verdict
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

OUT = Path("runs/exp5_seed_growth")
STEPS = 300
GRID_N = 60           # big grid so the pattern has room before hitting edges
SEED_HALF = 3         # seed square is (2*SEED_HALF)^2 cells at the centre


def living_radius(x: torch.Tensor) -> float:
    """Approx radius of the living region: max distance of any alive cell from centre."""
    idx = torch.nonzero(x > 0)
    if idx.numel() == 0:
        return 0.0
    c = (x.shape[0] - 1) / 2.0
    d = torch.sqrt(((idx.float() - c) ** 2).sum(dim=1))
    return float(d.max().item())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config(N=GRID_N, n_steps=STEPS, seed=7,
                 w0=-0.05, w1=0.4, w2=0.4, w3=-0.4, w4_help=0.6, w4_harm=0.6, w5=0.3, eta=0.08,
                 init_alive_prob=0.0)  # start all-dead, we set the seed manually
    if "learn" in Config.__dataclass_fields__:
        cfg = replace(cfg, learn=True)

    grid = build_grid(cfg)
    state, params, u = grid.state, grid.params, grid.u

    # Force a small central alive square.
    c = GRID_N // 2
    state.x.zero_()
    state.x[c - SEED_HALF:c + SEED_HALF, c - SEED_HALF:c + SEED_HALF] = 1.0
    # Give the seed cells some initial state so psi/f have signal.
    with torch.no_grad():
        state.s.normal_(std=cfg.init_noise_std)
        state.h.normal_(std=cfg.init_noise_std)
        m = state.x.unsqueeze(-1)
        state.s.mul_(m); state.h.mul_(m)

    alive_hist, radius_hist = [], []
    for t in range(STEPS):
        step_out = forward_step(state, params, u, cfg)
        gradient_step(state, step_out, params, cfg)
        state = step_out.next_state
        alive_hist.append(int(state.x.sum().item()))
        radius_hist.append(living_radius(state.x))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(alive_hist, color="#38bdf8"); axes[0].set_title("Alive count")
    axes[0].set_xlabel("step"); axes[0].grid(alpha=0.3)
    axes[1].plot(radius_hist, color="#3ec96b"); axes[1].set_title("Living-region radius")
    axes[1].set_xlabel("step"); axes[1].grid(alpha=0.3)
    axes[1].axhline(GRID_N / 2, color="gray", linestyle="--", alpha=0.5,
                    label="grid edge")
    axes[1].legend()
    axes[2].imshow(state.x.cpu().numpy(), cmap="Greens", interpolation="nearest")
    axes[2].set_title(f"Final living region (step {STEPS})")
    axes[2].set_xticks([]); axes[2].set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "summary.png", dpi=120)

    seed_cells = (2 * SEED_HALF) ** 2
    final = alive_hist[-1]
    r0, r1 = radius_hist[0], radius_hist[-1]
    print(f"seed cells: {seed_cells}, final alive: {final}")
    print(f"radius: {r0:.1f} -> {r1:.1f}")
    if final <= seed_cells * 0.5:
        verdict = "COLLAPSE — growth not viable without rule changes"
    elif r1 > r0 * 1.5:
        verdict = "GROWTH — expanding; a growing grid is worth building"
    else:
        verdict = "STABLE BLOB — persists but doesn't expand much"
    print("verdict:", verdict)
    print(f"Saved {OUT/'summary.png'}")


if __name__ == "__main__":
    main()
