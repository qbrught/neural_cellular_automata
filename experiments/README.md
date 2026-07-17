# Experiments

Standalone scripts for probing the NCSA system. Each runs from the project
root, saves output under `runs/exp*/`, and prints a text summary.

**For paper version comparisons (original → A → B → C → D)** use the
centralized suite instead: `python -m research.suite run` — see
[`../research/README.md`](../research/README.md).

Run any with:  `python -m experiments.exp1_frozen_vs_trained`

Most import `_common.run_metrics`, which runs a sim and returns per-step
metrics as numpy arrays in memory (no file round-trip). If `forward_step` /
`gradient_step` signatures change, only `_common.py` needs updating.

Note: exp1, exp2 and the `learn=False` paths assume the local `learn` flag on
Config. If it's absent, `run_metrics` still runs but the learn toggle is a
no-op (everything trains). Add the flag (see chat notes) for the frozen
comparisons to mean anything.

## The five

**exp1_frozen_vs_trained** — Does learning contribute, or would frozen
random-init MLPs produce the same patterns? Compares N seeds each way.
*The foundational baseline — run this first.*

**exp2_state_scramble** — Do cells actually use their state s,h? Perturbs
state mid-run (noise + permutation variants) and measures trajectory
divergence from a bit-identical baseline. Directly tests the state-aware goal.

**exp3_w3_sweep** — Sweeps the eliminator penalty w3 and measures how
population balance and total alive respond. Tests which fixed-rule parameter
drives reproducer dominance.

**exp4_convergence** — Tracks per-step parameter drift to see whether learning
converges or keeps drifting at the end of a run (loss curves can look flat
while params still move).

**exp5_seed_growth** — Starts from a small central alive seed in a dead grid
and measures whether the population expands, holds, or collapses. Gates
whether the growing/infinite-grid direction is worth building.

## Suggested order

1. exp1 — establish that learning matters at all
2. exp2 — establish that state matters (validates the core design goal)
3. exp4 — understand whether the system settles
4. exp3 — understand what the fixed weights control
5. exp5 — decide whether growing-grid work is worth it

Don't run all at once. Pick the question you most want answered, run it,
write down what you found before moving on.
