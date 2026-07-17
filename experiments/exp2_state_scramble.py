"""Experiment 2 — State scramble (state-awareness test).

Question: do cells actually USE their internal state s,h, or have they learned
a state-blind policy? This directly tests the 'state-aware' design goal.

Method: run to a chosen step, then either (a) replace s,h with fresh noise or
(b) permute s,h across cells (same distribution, spatial correspondence
destroyed). Compare the resulting alive-count trajectory to an unperturbed
baseline that is bit-identical up to the scramble step.

Interpretation:
  - big lasting divergence  -> cells genuinely use state
  - brief divergence then reconverge -> state matters transiently; attractor
  - little divergence -> state-blind policy

Run:  python -m experiments.exp2_state_scramble
Out:  runs/exp2_state_scramble/summary.svg + printed divergence stats
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from experiments._common import run_metrics

OUT = Path("runs/exp2_state_scramble")
STEPS = 400
SCRAMBLE_AT = 200
GRID_N = 30

# Poster styling (A1 = 23.4 × 33.1 in)
POSTER_RC = {
    "font.size": 22,
    "axes.titlesize": 30,
    "axes.labelsize": 26,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 22,
    "lines.linewidth": 3.0,
    "axes.linewidth": 1.5,
    "grid.linewidth": 1.2,
}


def scramble_noise(state, gen):
    """Replace s,h with noise matched to current state magnitude, keep dead cells zero."""
    scale_s = state.s[state.x > 0].abs().mean().clamp(min=1e-3).item() if (state.x > 0).any() else 1.0
    scale_h = state.h[state.x > 0].abs().mean().clamp(min=1e-3).item() if (state.x > 0).any() else 1.0
    state.s.normal_(generator=gen).mul_(scale_s)
    state.h.normal_(generator=gen).mul_(scale_h)
    m = state.x.unsqueeze(-1)
    state.s.mul_(m); state.h.mul_(m)


def scramble_permute(state, gen):
    """Permute s,h vectors across cells; same distribution, spatial link destroyed."""
    N, d = state.N, state.d
    idx = torch.randperm(N * N, generator=gen)
    state.s = state.s.reshape(N * N, d)[idx].reshape(N, N, d).contiguous()
    state.h = state.h.reshape(N * N, d)[idx].reshape(N, N, d).contiguous()
    m = state.x.unsqueeze(-1)
    state.s = state.s * m
    state.h = state.h * m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config(N=GRID_N, n_steps=STEPS, seed=7,
                 w0=-0.05, w1=0.4, w2=0.4, w3=-0.4, w4_help=0.6, w4_harm=0.6, w5=0.3, eta=0.08)

    base = run_metrics(cfg, learn=True, steps=STEPS)
    noise = run_metrics(cfg, learn=True, steps=STEPS,
                        perturb_at=SCRAMBLE_AT, perturb_fn=scramble_noise)
    perm = run_metrics(cfg, learn=True, steps=STEPS,
                       perturb_at=SCRAMBLE_AT, perturb_fn=scramble_permute)

    x = np.arange(STEPS)

    with plt.rc_context(POSTER_RC):
        fig, ax = plt.subplots(figsize=(18.0, 6.0))
        ax.plot(x, base["alive"], label="baseline", color="#38bdf8")
        ax.plot(x, noise["alive"], label="noise scramble", color="#e0492f")
        ax.plot(x, perm["alive"], label="permute scramble", color="#3ec96b")
        ax.axvline(SCRAMBLE_AT, color="gray", linestyle="--", alpha=0.6, linewidth=2)
        ax.set_xlabel("step")
        ax.set_ylabel("alive count")
        ax.set_title("State scramble at step %d" % SCRAMBLE_AT)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "summary.svg", format="svg")
    plt.close(fig)

    def divergence(a):
        d = np.abs(base["alive"][SCRAMBLE_AT:] - a["alive"][SCRAMBLE_AT:])
        return d.max(), d.mean()

    for name, m in [("noise", noise), ("permute", perm)]:
        mx, mn = divergence(m)
        print(f"{name:8s}: max divergence={mx:4.0f}  mean divergence={mn:5.1f}")
    print(f"Saved {OUT / 'summary.svg'}")
    print("Read: large divergence => cells use state; near-zero => state-blind.")


if __name__ == "__main__":
    main()