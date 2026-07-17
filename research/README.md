# Research experiment suite

Central place to **run, measure, chart, and compare** system versions for the paper.

Versions form an ablation path:

| ID | Name | Status |
| --- | --- | --- |
| `original` | Indiscriminate single vote channel | implemented |
| `A` | Typed help/harm votes (kin/foe routing) | implemented |
| `B` | Predator–prey loss | planned |
| `C` | Goal inheritance / colonization | planned |
| `D` | Goal-conditioned local update `f` | planned |

Every suite run produces the same artifact layout so you can drop results into the paper and fill `NOTES.md` with manual UI observations.

## Quick start

From the project root:

```bash
# List versions
python -m research.suite list

# Full comparison: original vs A (3 seeds × 500 steps, benchmark config)
python -m research.suite run --versions original,A

# Faster paper draft (still multi-seed)
python -m research.suite run --versions original,A --n-steps 400 --seeds 1096812628,42,7

# Smoke test
python -m research.suite run --quick
```

Results land in:

```
research_results/<run_name>/
  REPORT.md              ← start here (tables + embedded charts)
  NOTES.md               ← your manual observations template
  summary.csv            ← all scalars, one row per version×seed
  manifest.json
  comparison/            ← overlay charts across versions
  versions/
    original/seed_*/     ← per-run panel.png, series.npz, config.json
    A/seed_*/
```

Open `REPORT.md` after a run.

## What is measured

| Metric | Paper interpretation |
| --- | --- |
| `corr(ra,ea)` | Are the two types the same density curve? |
| `corr(ra, dens)` | Is repro count just goal_frac × total? |
| `\|ra residual\|` | Type-specific deviation from density sampling |
| `ra/ea early → late` | Role divergence over time |
| `corr(Lr,Le)` | Loss coupling (mirror vs independent) |
| vote discrimination | Did help/harm specialize by receiver type? |
| segregation | Spatial clustering of goals among alive cells |
| extinction | Viability cost of the change |

## Interactive UI (manual observations)

The server still runs the live system. Version flags are exposed as toggles:

```bash
python server.py
# open http://127.0.0.1:8765
```

| Flag | UI label | Version |
| --- | --- | --- |
| `typed_votes` | Typed votes A | original=0, A=1 |
| `predator_prey_loss` | Pred-prey loss B | (reserved) |
| `goal_inheritance` | Goal inherit C | (reserved) |
| `goal_in_f` | Goal in f D | (reserved) |

**Workflow for writing:**

1. Run the suite → get quantitative `REPORT.md`.
2. In the UI, set the same seed as a suite seed, toggle `typed_votes` 0 vs 1.
3. Write qualitative notes into that run’s `NOTES.md`.

## Adding step B / C / D

1. Implement the mechanism behind a Config flag (already reserved in `config.py`).
2. Set `implemented=True` on that version in `research/versions.py` and set the flags.
3. Re-run:

```bash
python -m research.suite run --versions original,A,B
```

No chart/report code changes needed — new versions plug into the same suite.

## Fair comparison design

- Shared **base config**: `research/configs/benchmark.json` (historically long-lived hyperparams).
- Shared **seeds** across versions.
- Only **version flags** differ between arms.
- `typed_votes=False` keeps the dual ψ heads for parameter-count parity but routes only the help head to all receivers (`V_foe=0`) — isolating the *routing* mechanism of step A.

## Relation to `experiments/`

| Folder | Purpose |
| --- | --- |
| `experiments/` | One-off scientific questions (frozen vs trained, w3 sweep, …) |
| `research/` | **Version ablation suite** for the paper narrative (original→A→B→C→D) |
