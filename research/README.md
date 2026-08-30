# Research experiment suite

**Thesis method (isolations + frames + paired tests):**

```bash
python -m research.pipeline list
python -m research.pipeline run --quick                 # A, 1 seed, short T
python -m research.pipeline run                         # frozen protocol (slow)
python -m research.pipeline run --letters A,B,C --include-ladder
```

See [`THESIS_PIPELINE.md`](THESIS_PIPELINE.md). Add a letter in [`comparisons.py`](comparisons.py).

---

Central place to **run, measure, chart, and compare** system versions for the paper.

Versions form an ablation path:

| ID | Name | Status |
| --- | --- | --- |
| `original` | Indiscriminate single vote channel | implemented |
| `A` | Typed help/harm votes (kin/foe routing) | implemented |
| `B` | Predator–prey loss (on top of A) | implemented |
| `C_only` | Goal inheritance **only** (no A/B) | implemented |
| `C` | Goal inheritance on A+B (full stack) | implemented |
| `D_fixed` | Goal-conditioned `f` on A+B (goals fixed) | implemented |
| `D` | Goal-conditioned `f` on A+B+C | implemented |
| `E` | Typed votes + symmetric \(w_2=w_3\) (env ablation) | implemented |
| `F` | Typed votes + soft coexistence pressure (λ barrier) | implemented |
| `G` | Typed votes + frozen transfer blobs (`A_env`) | implemented |
| `G_learn` | Typed votes + learning hotspot (`A_env_learn`) | implemented |

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

# Isolated inheritance vs original (pure C mechanism)
python -m research.suite run --versions original,C_only --n-steps 400

# Full stack C vs B (inheritance given A+B)
python -m research.suite run --versions B,C --n-steps 400

# Multi-config: A/B/C on selected discoveries (titles + one-liners from catalog)
python -m research.suite run --versions A,B,C \
  --discoveries disc_0001,disc_0003,disc_0005 \
  --n-steps 400 --seeds 1096812628,42,7 \
  --name suite_ABC_discoveries

# Same thing with explicit paths
python -m research.suite run --versions A,B,C \
  --configs discoveries/disc_0001,discoveries/disc_0003

# All catalog discoveries
python -m research.suite run --versions A,B,C --discoveries all --n-steps 400
```

**Single-config** results land in:

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

**Multi-config** results land in:

```
research_results/<run_name>/
  INDEX.md               ← start here (links + cross-config table)
  REPORT.md              ← short pointer to INDEX + per-config list
  summary_all.csv        ← all (config, version, seed) rows
  manifest.json
  configs/
    disc_0001/
      REPORT.md          ← titled "A,B,C on disc_0001" + catalog one-liner
      comparison/        ← charts titled "Config disc_0001: ..."
      versions/A|B|C/seed_*/
    disc_0003/
      ...
```

Open `INDEX.md` (multi) or `REPORT.md` (single) after a run.

`--versions all` includes **G** and **G_learn** (two extra arms). `--quick` stays `original,A`. G is transfer-only blobs; G_learn is a center learning hotspot. `env_seed` is not pinned, so suite `--seeds` share terrain. Overlay lives on `simulate.run` + `visualise.render_summary`, not on suite `panel.png`.

Hand-authored terrain: put `env_preset: "custom"` and an `env_regions` list on the base JSON (see `research/configs/g_custom_inland.json`). `apply(G)` keeps those regions instead of pinning `blobs`. A still runs homogeneous (flag off).

## What is measured

| Metric | Paper interpretation |
| --- | --- |
| `corr(ra,ea)` | Are the two types the same density curve? |
| `corr(ra, dens)` | Is repro count just goal_frac × total? |
| `\|ra residual\|` | Type-specific deviation from density sampling |
| `ra/ea early → late` | Role divergence over time |
| `corr(Lr,Le)` | Loss coupling (mirror vs independent) |
| vote discrimination | Did help/harm specialize by receiver type? |
| typed edge death rates | Sender death rate on same-type vs cross-type edges; gap = cross−same |
| segregation | Spatial clustering of goals among alive cells |
| `g_frac drift` | Final − initial goal=REPRO fraction (all cells); ~0 without C |
| `mean\|Δg_alive\|` | Mean step-to-step \|Δ\| of alive type fraction (colonization churn) |
| extinction | Viability cost of the change |

## Interactive UI (manual observations)

The server still runs the live system. Version flags are exposed as toggles:

```bash
python server.py
# open http://127.0.0.1:8765
```

| Flag | UI label | Version |
| --- | --- | --- |
| `typed_votes` | Typed votes A | original=0, A/B=1 |
| `predator_prey_loss` | Pred-prey loss B | A=0, B=1 |
| `goal_inheritance` | Goal inherit C | B=0, C=1 (births adopt majority neighbour goal) |
| `goal_in_f` | Goal in f D | (reserved) |
| `learn_messages` | Learn messages | off=Path-1 (message head dead); on=one-hop live M |
| `environment_heterogeneous` | Heterogeneous env G | overlay: transfer blobs / learning hotspot / occupancy |

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

- Shared **base config** per arm: default `research/configs/benchmark.json`, or any discovery / JSON via `--config` / `--configs` / `--discoveries`.
- Shared **seeds** across versions (and across configs in a multi-config suite).
- Only **version flags** differ between arms; survival weights / η / init density come from the base config.
- Catalog one-liners (from `discoveries/catalog.jsonl`) are attached as config titles in reports and chart headings.
- `typed_votes=False` keeps the dual ψ heads for parameter-count parity but routes only the help head to all receivers (`V_foe=0`) — isolating the *routing* mechanism of step A.

## Relation to `experiments/`

| Folder | Purpose |
| --- | --- |
| `experiments/` | One-off scientific questions (frozen vs trained, w3 sweep, …) |
| `research/` | **Version ablation suite** for the paper narrative (original→A→B→C→D) |

**Channel audit (not a version ablation):** `experiments/exp6_message_channel_dead`
proves mathematically and empirically that ψ’s message head $W_2^{(m)}$ is
gradient-dead under Path-1 (`stopgrad(M)`), which is the **default**
(`Config.learn_messages=False`). Set `learn_messages=True` (UI: “Learn
messages”) to train the message head via one-hop leakage into senders’ ψ.
Use exp6’s `REPORT.md` whenever the paper discusses learned signalling.
Unit tests: `tests/test_message_head_dead.py`.
