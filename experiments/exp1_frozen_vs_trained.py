"""Experiment 1 — Frozen vs Trained.

Question: does learning actually contribute to the dynamics, or would frozen
random-init MLPs produce the same patterns?

Method: run N seeds with learning on and N with learning off, all else equal.
Compare mean alive-count trajectories (with spread across seeds) and final
population. If the two groups are indistinguishable, learning is decorative.

Run:  python -m experiments.exp1_frozen_vs_trained
Out:  runs/exp1_frozen_vs_trained/summary.png  + printed stats
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import Config
from experiments._common import run_metrics, summarise_tail

OUT = Path("runs/exp1_frozen_vs_trained")
N_SEEDS = 5
STEPS = 400
GRID_N = 30


def run_group(learn: bool) -> list[dict]:
    results = []
    for seed in range(N_SEEDS):
        cfg = Config(N=GRID_N, n_steps=STEPS, seed=seed,
                     w0=-0.05, w1=0.4, w2=0.4, w3=-0.4, w4_help=0.6, w4_harm=0.6, w5=0.3, eta=0.08)
        m = run_metrics(cfg, learn=learn, steps=STEPS)
        results.append(m)
        tail = summarise_tail(m)
        print(f"  learn={learn} seed={seed}: "
              f"final_alive={m['alive'][-1]:4d}  tail_alive={tail['alive']:.1f}")
    return results


def stack(results, key):
    return np.stack([r[key] for r in results])  # (seeds, steps)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Trained group:")
    trained = run_group(learn=True)
    print("Frozen group:")
    frozen = run_group(learn=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for label, group, color in [("trained", trained, "#38bdf8"),
                                ("frozen", frozen, "#e0492f")]:
        a = stack(group, "alive")
        mean, std = a.mean(0), a.std(0)
        x = np.arange(a.shape[1])
        axes[0].plot(x, mean, color=color, label=label, linewidth=1.5)
        axes[0].fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    axes[0].set_title("Alive count (mean ± std across seeds)")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("alive")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Final-alive distribution.
    t_final = [r["alive"][-1] for r in trained]
    f_final = [r["alive"][-1] for r in frozen]
    axes[1].boxplot([t_final, f_final], labels=["trained", "frozen"])
    axes[1].set_title("Final alive count distribution")
    axes[1].set_ylabel("alive"); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "summary.png", dpi=120)
    print(f"\nTrained final: {np.mean(t_final):.1f} ± {np.std(t_final):.1f}")
    print(f"Frozen  final: {np.mean(f_final):.1f} ± {np.std(f_final):.1f}")
    print(f"Saved {OUT/'summary.png'}")


if __name__ == "__main__":
    main()
