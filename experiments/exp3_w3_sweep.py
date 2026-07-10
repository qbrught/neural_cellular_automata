"""Experiment 3 — w3 sweep (eliminator penalty).

Question: reproducers consistently outnumber eliminators. Is that driven by
w3 (the eliminator's negative contribution to neighbour survival)? How does
the population balance respond as w3 varies from harsh to neutral?

Method: sweep w3 across a range, several seeds each, measure settled
reproducer-alive vs eliminator-alive and total alive.

Run:  python -m experiments.exp3_w3_sweep
Out:  runs/exp3_w3_sweep/summary.png + printed table
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import Config
from experiments._common import run_metrics, summarise_tail

OUT = Path("runs/exp3_w3_sweep")
W3_VALUES = [-1.0, -0.75, -0.5, -0.25, 0.0]
N_SEEDS = 3
STEPS = 300
GRID_N = 30


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for w3 in W3_VALUES:
        repro, elim, total = [], [], []
        for seed in range(N_SEEDS):
            cfg = Config(N=GRID_N, n_steps=STEPS, seed=seed,
                         w0=-0.05, w1=0.4, w2=0.4, w3=w3, w4=0.6, w5=0.3, eta=0.08)
            m = run_metrics(cfg, learn=True, steps=STEPS)
            t = summarise_tail(m)
            repro.append(t["reproducer_alive"])
            elim.append(t["eliminator_alive"])
            total.append(t["alive"])
        rows.append((w3, np.mean(repro), np.mean(elim), np.mean(total)))
        print(f"w3={w3:+.2f}: repro={np.mean(repro):5.1f}  "
              f"elim={np.mean(elim):5.1f}  total={np.mean(total):5.1f}")

    rows = np.array(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rows[:, 0], rows[:, 1], "-o", color="#3ec96b", label="reproducers alive")
    ax.plot(rows[:, 0], rows[:, 2], "-o", color="#e0492f", label="eliminators alive")
    ax.plot(rows[:, 0], rows[:, 3], "-o", color="#94a3b8", label="total alive")
    ax.set_xlabel("w3 (eliminator penalty)")
    ax.set_ylabel("settled alive count (tail mean)")
    ax.set_title("Population balance vs eliminator penalty w3")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "summary.png", dpi=120)
    print(f"Saved {OUT/'summary.png'}")


if __name__ == "__main__":
    main()
