# Experiments

Standalone probes of the NCSA system. Each runs from the project root, saves
output under `runs/exp*/`, and prints a text summary.

These are **not** the paper version comparisons. For original / A–F isolations
use the thesis pipeline:

```bash
python -m research.pipeline list
python -m research.pipeline run --quick
```

See [`../research/README.md`](../research/README.md) and
[`../research/THESIS_PIPELINE.md`](../research/THESIS_PIPELINE.md).

Run any probe with: `python -m experiments.exp1_frozen_vs_trained`

Most import `_common.run_metrics`, which runs a sim and returns per-step
metrics as numpy arrays in memory (no file round-trip). If `forward_step` /
`gradient_step` signatures change, only `_common.py` needs updating.
`Config.learn` (default `True`) is the freeze switch used by exp1 / exp2.

## The six

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

**exp6_message_channel_dead** — Proves the ψ **message head is not learned**
under Path-1 detaches (exact zero grad + bit-identical weights after SGD).
Documents the claim mathematically; includes a live-`M` counterfactual.
*Paper-critical for any “learned signalling” language.*
Also: `python tests/test_message_head_dead.py`.

## Suggested order

1. exp1 — establish that learning matters at all
2. exp2 — establish that state matters (validates the core design goal)
3. exp6 — establish which channels actually train (messages vs votes)
4. exp4 — understand whether the system settles
5. exp3 — understand what the fixed weights control
6. exp5 — decide whether growing-grid work is worth it

Don't run all at once. Pick the question you most want answered, run it,
write down what you found before moving on.
