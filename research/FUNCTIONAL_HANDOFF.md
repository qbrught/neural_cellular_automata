# Handoff: functional class divergence

For an agent continuing this work. Mix-level Φ (count residual) is already in the thesis pipeline. Functional divergence is the **cell-level** follow-up: each cell is a response vector, distances between those vectors, then “do the two planted types still organise that space?”

Repo: `qbrught/neural_cellular_automata`, branch `main` (commits `976c449` … `a7ad74a`).

Do **not** treat high own-goal Δ as “two learned policies.” That was the original bug.

---

## Why it exists

Φ averages the lattice to two counts:

```
Φ(t) = |r_t − f0 · a_t| / (a_t + ε)
```

It cannot see whether R and E cells *do different things*. Functional divergence fingerprints each cell’s ψ on a frozen probe bank, then summarises type geometry with energy distance Δ = ℰ/2, k-means ARI vs goals, and PCA/UMAP.

Scientific ask (from the supervisor): give every cell a response vector; distance = how much two cells agree; cluster; do clusters preserve the two families?

---

## Two objects (do not mix them up)

| Name | Code family | Inputs | Init | Use for |
|---|---|---|---|---|
| **Agent output** | `realized` (6-d) | cell’s **own** goal as `g_s` | Δ large, ARI ~1 even with identical weights (label is an input) | “Do typed *agents* still cluster after learning?” Compare **late vs init**. |
| **Maps** | `common` (12-d) | same `(g_s, g_r)` list for every cell: six situations × `g_s ∈ {0,1}` | Δ ≈ 0 | “Did the *maps* specialise?” Late rise is learned. **Headline for ‘two policies’.** |
| **Goal-ablated** | `weights_only` (6-d) | all goal bits zero | Δ ≈ 0 | Ablation: split cannot be only the goal-input columns. Misses kin/foe that lives in those columns. |

Routing is version-aware: original = help on every probe; A+ = help on kin, harm on foe.

Δ formula (equal pool of the two within-type means = sample energy distance / 2):

```
Δ = mean{d_ij : g_i ≠ g_j} − ½ (mean within-R + mean within-E)
```

Undefined (`nan`) if a type has fewer than two cells. Distances are Euclidean after **per-coordinate z-score**. Alive-only Δ **re-z-scores the living cells**; it is not a subset of the all-cell matrix.

**Map-split** (code): `Δ_maps > 0.4` AND `ARI_maps > 0.25` AND `min-type > 0.10` (`COEXIST_FLOOR` from `research/protocol.py`).

**Mix two-process**: `Φ_late > 0.2` AND `min-type > 0.10`.

Joint read: Φ high + map-split low = mix moved without two maps. Map-split + Φ low = two maps, balanced mix (Φ miss). High agent ARI + map ARI ~0 = label-as-input, not map split. High Δ with a dying type is attrition, not two processes.

---

## Files (keep)

| Path | Role |
|---|---|
| `research/functional.py` | Probe banks, Δ, ARI, evaluate_snapshot |
| `research/functional_analysis.py` | Lettered **scores** (`compare`) and **PCA/UMAP** (`embed`) |
| `research/response_analysis.py` | Per-run eval, plots, small `run` CLI |
| `research/runner.py` | Writes `params_final.pt` + `state_final.pt` next to `series.npz` |
| `research/pipeline.py` | Subcommands `functional` and `embed` |
| `tests/test_functional_divergence.py` | Metric: layout, confound, energy pooling, alive-only |
| `tests/test_functional_analysis.py` | Cache discovery + map-split predicate |
| `research/FUNCTIONAL_DIVERGENCE.md` | Spec (v1 probe table, scalars) |

Untracked leftovers (not required): `cell_level_divergence_section.tex`, `POLICY_DIVERGENCE_CITATIONS.md`, `research/phi_analysis.py` (old mix Φ post-hoc; superseded by `analyze_class_div.py`).

Mix pipeline (already on main): `research/pipeline.py`, `research/comparisons.py`, `research/thesis_report.py`, `research/protocol.py`, `research/stats.py`.

---

## How to run

From repo root. Mix Φ first (writes `cache/` + `protocol.json` + letter reports). Functional second (needs late snapshots).

```bash
# Tests
python tests/test_functional_divergence.py
python tests/test_functional_analysis.py
python tests/test_pipeline.py

# Mix-Φ lettered pipeline (same arms/seeds functional will reuse)
python -m research.pipeline list
python -m research.pipeline run --quick                          # A only, 1 seed, 60 steps
python -m research.pipeline run --letters A,B,C --include-ladder --name thesis_letters
python -m research.pipeline report research_results/<run>

# Functional scores (paired tests, letter reports) — no PCA/UMAP
python -m research.pipeline functional research_results/<run>
# same: python -m research.functional_analysis compare research_results/<run>

# PCA/UMAP (visual seeds from protocol.json if present)
python -m research.pipeline embed research_results/<run>
# same: python -m research.functional_analysis embed research_results/<run>

# Both
python -m research.functional_analysis analyze research_results/<run>
python -m research.functional_analysis analyze research_results/<run> --no-embed
```

Small standalone sim+analyse (not the lettered pipeline):

```bash
python -m research.response_analysis run --quick
python -m research.response_analysis run --versions original,A --n-steps 1000 --name functional_demo
python -m research.response_analysis analyze research_results/<folder>
```

Also works on a **suite** tree (`versions/<arm>/seed_*`), not only pipeline `cache/<config>/<arm>/seed_*`.

Older caches without `params_final.pt` / `state_final.pt` cannot be scored. Re-run those sims (runner now always writes snapshots). `skip-existing` will **not** backfill snapshots on a cache that already has `series.npz`.

---

## Outputs

| Path | Contents |
|---|---|
| `FUNCTIONAL_INDEX.md` | Letter index |
| `functional_comparisons/<letter>/FUNCTIONAL_REPORT.md` | Seed table, mean±SEM, **paired on−off tests** (same as mix) |
| `FUNCTIONAL_REPORT.md` | All-arm table |
| `functional_summary.csv` | One row per arm×seed |
| `functional_compare/` | PCA/UMAP, Δ bars, ARI bars, Φ vs Δ, grids (embed pass) |

Letter reports live next to mix `comparisons/<letter>/REPORT.md`. Quote **Δ maps** + ARI maps for “did policies specialise.” Use agent PCA/ARI vs **init** for “do types still cluster.”

---

## Pitfalls (already fixed in code; do not reintroduce)

1. Own-goal `realized` at init saturates z-scored 6-d (~`2√6 ≈ 4.9`). `Δ_late − Δ_init` on realized is usually **negative** (smear). Learned increment is **`common`**.
2. `weights_only` zeros goals, so specialisation that lives only in goal-input columns is invisible. `common` still includes goal bits, shared across cells.
3. Pair-count pooling of within is **not** energy distance unless classes are balanced. Code uses equal pool; Δ is `nan` if a class has `<2` cells.
4. Attrition manufactures huge own-goal Δ (leftover cells many majority-stds away). Ranking by agent Δ puts dying-type versions first. Rank / test **maps**.
5. k-means ARI is blunt on filaments (not two balls). Embeddings are the clustering picture; ARI is a diagnostic.
6. Probe bank is frozen. Changing it is a new metric version.
7. Sender `s=h=0` on all banks: specialisation in using own accumulated state is out of v1.

---

## Suggested next work

- Run `pipeline functional` + `embed` on a real lettered cache (not `--quick`) and fill results vs mix Φ letter-by-letter.
- If a cache predates snapshot saving, re-simulate; do not try to reconstruct ψ from `series.npz`.
- Paper methods: two banks (agent vs maps), `Δ = ℰ/2`, init vs late, ARI/PCA. Do not claim own-goal vectors are \(L^2(μ)\) on a shared measure.
- Optional: permutation test for Δ (not implemented; we report the raw contrast + paired bootstrap/Wilcoxon across seeds).
- Optional: common-domain bank as the **only** headline if the paper wants a pure map distance; keep `realized` as the agent-output diagnostic.

---

## Commands recap for the next agent

```bash
git pull
python tests/test_functional_divergence.py && python tests/test_functional_analysis.py
python -m research.pipeline run --letters A --quick --name fd_smoke
python -m research.pipeline functional research_results/fd_smoke
python -m research.pipeline embed research_results/fd_smoke
# then open research_results/fd_smoke/FUNCTIONAL_INDEX.md
```
