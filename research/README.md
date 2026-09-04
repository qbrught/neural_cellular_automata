# Research comparisons

The thesis method is `python -m research.pipeline`: lettered isolations, spatial frames, paired tests, and four-chunk reports. Protocol, letters, and artifact layout: [`THESIS_PIPELINE.md`](THESIS_PIPELINE.md). Add a letter in [`comparisons.py`](comparisons.py).

`python -m research.suite` is an older ad-hoc runner (arbitrary version lists, discovery catalogs). Use it only when you want that; it is **not** the report protocol.

```bash
python -m research.pipeline list
python -m research.pipeline run --quick                 # smoke: letter A, 1 seed, short T
python -m research.pipeline run                         # frozen protocol (slow)
python -m research.pipeline run --letters A,B,C --include-ladder
```

Default `pipeline run` (no `--letters`) includes **G** / **G_learn**. Those arms are implemented but were **not in the report** — pass `--letters A,B,C,…` to skip them. Full protocol is \(T=4000\) × 20 seeds; `--quick` only checks that the machinery runs. Snapshot numbers for the report live in `research/results_snapshots/`.

Results: `research_results/<run>/INDEX.md`.

## Versions

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
| `G` | Typed votes + frozen transfer blobs (`A_env`) | implemented (not in the report) |
| `G_learn` | Typed votes + learning hotspot (`A_env_learn`) | implemented (not in the report) |

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
| functional Δ / ARI | Cell-level response geometry (not a count residual) |

## Functional divergence

Same lettered arms and paired tests as mix Φ, on probe-response vectors. PCA/UMAP is a second pass (`scikit-learn` / `umap-learn` in `requirements.txt`).

```bash
python -m research.pipeline run --letters A,B,C --include-ladder --quick
python -m research.pipeline functional research_results/<run_name>
python -m research.pipeline embed research_results/<run_name>
```

Late snapshots (`params_final.pt`, `state_final.pt`) are written by the runner. Re-run sims if an older cache lacks them. Design: [`FUNCTIONAL_DIVERGENCE.md`](FUNCTIONAL_DIVERGENCE.md).

## Interactive UI

```bash
python server.py
# open http://127.0.0.1:8765
```

| Flag | UI label | Version |
| --- | --- | --- |
| `typed_votes` | Typed votes A | original=off, A+=on |
| `predator_prey_loss` | Pred-prey loss B | |
| `goal_inheritance` | Goal inherit C | births adopt majority neighbour goal |
| `goal_in_f` | Goal in f D | own goal is an input to *f* |
| `coexistence_pressure` | Coexist pressure F | soft barrier on both types' living mass |
| `learn_messages` | Learn messages | off=Path-1 (message head dead); on=one-hop live M |

Experiment G controls are hidden in the UI (not in the report). Enable G from a Config JSON via `python run.py --config …`.

Workflow: run the pipeline → `INDEX.md` / per-letter `REPORT.md`; in the UI, match a pipeline seed and toggle the same flags; write notes into that comparison’s `NOTES.md`.

## Adding a letter

1. Implement the Config flag.
2. Add a `VersionSpec` in [`versions.py`](versions.py) and a `Comparison` in [`comparisons.py`](comparisons.py).
3. `python -m research.pipeline list` then `python -m research.pipeline run --letters <id> --quick`.

## Ad-hoc suite (older)

Arbitrary version lists and discovery catalogs, **not** the frozen thesis protocol (suite default horizon is the JSON `n_steps`, usually 500; pipeline is 4000 × 20 seeds).

```bash
python -m research.suite list
python -m research.suite run --versions original,A
python -m research.suite run --quick
python -m research.suite run --versions A,B,C \
  --discoveries disc_0001,disc_0003,disc_0005 \
  --n-steps 400 --seeds 1096812628,42,7 \
  --name suite_ABC_discoveries
```

Single-config results: `research_results/<run>/REPORT.md`. Multi-config: `INDEX.md`. `--versions all` includes G and G_learn.

Fair-comparison rules still apply: shared base config and seeds; only version flags differ. `typed_votes=False` keeps dual ψ heads for parameter-count parity but routes only the help head (`V_foe=0`).

## Relation to `experiments/`

| Folder | Purpose |
| --- | --- |
| `experiments/` | One-off probes (frozen vs trained, w3 sweep, …) |
| `research/` | Lettered isolations for the report (`pipeline`; suite is leftover) |

**Channel audit (not a version ablation):** `experiments/exp6_message_channel_dead` proves ψ’s message head is gradient-dead under Path-1 (`Config.learn_messages=False`). Unit tests: `tests/test_message_head_dead.py`.
