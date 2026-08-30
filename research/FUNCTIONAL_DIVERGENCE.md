# Functional class divergence — design and implementation

**Status:** implemented (v1)  
**Code:** `research/functional.py` (metric), `research/response_analysis.py` (run + plots + report)  
**CLI:** `python -m research.response_analysis`  
**Does not replace Φ.** It is the cell-level object Φ was standing in for.

---

## 1. Why Φ is not enough

Count-level class divergence is

\[
\Phi(t)=\frac{|r_t-f_0 a_t|}{a_t+\varepsilon}
=\bigl|\text{alive type fraction}-f_0\bigr|,
\qquad
\Phi=\operatorname{mean}_t\Phi(t).
\]

That is two averages stacked: a spatial mean of type indicators, then a time mean of the residual. Every cell of a type is interchangeable. The metric cannot see whether those cells *do different things*.

Failure modes we already hit in the suite:

| Actual dynamics | Φ |
|---|---|
| One type dies (attrition) | Large — not two processes |
| Both types persist 50/50 with different policies | ~0 if the mix stays near \(f_0\) |
| Mix drifts, policies identical | Large — no functional split |
| Subtypes inside a goal class | Invisible |

Vote-discrimination series (`vote_R_help_kin`, …) have the same shape: they are **class means of occupancy-conditioned outputs**. A bimodal R population can match a uniform mild helper.

The scientific claim is *two coexisting dynamical processes*. That is a statement about response functions, not headcount.

---

## 2. Response function of a cell

Cell \(i\) already is a map: its own \(\psi_{\theta_i}\) and \(f_{\theta_i}\).

- **ψ input** (sender \(i\), receiver \(r\)): \((s_i,h_i,x_i,g_i,s_r,h_r,x_r,g_r)\in\mathbb{R}^{4d+4}\)
- **ψ output:** \((m,\,v_{\mathrm{help}},\,v_{\mathrm{harm}})\in\mathbb{R}^{d+2}\)
- **f input:** \((s,h,x,M,\text{goal slot})\in\mathbb{R}^{3d+2}\)
- **f output:** \((\tilde s,\tilde h)\); f-signal \(u\cdot\tilde s\)

The **response function** is this map. A finite **response vector** \(v_i\) is the concatenation of outputs on a frozen probe bank. Distance \(d(i,j)=\|v_i-v_j\|_2\) after per-coordinate z-scoring across cells in the same snapshot.

Do **not** fingerprint on the neighbourhood the cell happened to occupy. That confounds policy with location.

There are two headline banks. **Agent output (`realized`)** uses the cell’s own goal as \(g_s\): that is what this typed agent emits. **Maps (`common`)** use the same \((g_s,g_r)\) list for every cell, so distances are distances between maps. Neighbourhood is still frozen on both.

### 2.1 Frozen vote-probe bank

Sender is always a canonical blank living agent, \(s=h=0\), \(x=1\).

On **`realized`**, \(g_s\) is the cell’s own goal. On **`common`**, the six receiver situations below are run twice, once with \(g_s=0\) and once with \(g_s=1\) (12-d). Kin/foe still sets \(g_r\) relative to that probe’s \(g_s\).

Receivers (v1, frozen):

| Name | Relation | \(x_r\) | \(s_r\) |
|---|---|---|---|
| `kin_alive_blank` | same goal | 1 | \(0\) |
| `foe_alive_blank` | opposite goal | 1 | \(0\) |
| `kin_dead_blank` | same goal | 0 | \(0\) |
| `foe_dead_blank` | opposite goal | 0 | \(0\) |
| `kin_alive_exc` | same goal | 1 | \(0.5\cdot\mathbf{1}\) |
| `foe_alive_exc` | opposite goal | 1 | \(0.5\cdot\mathbf{1}\) |

\(h_r=0\) on all six. Changing this table is a new metric version; do not tune it per suite.

Raw MLP outputs are used (no \(\rho\) / alive gate). Gating is occupancy, not policy.

### 2.2 Realized vote vs both heads

ψ always emits both heads. The CA does not always use both:

- **original:** help head on every edge; harm is unused (\(V_{\mathrm{foe}}=0\)).
- **A and up:** help → kin, harm → foe.

**Agent output (`realized`, 6-d):** the vote that actually reaches the receiver under that version’s routing, with the cell’s own goal as \(g_s\). This is “what would this typed agent send?” Init \(\Delta\) is large even with identical weights — the label is an input. That is expected. The question is whether **clusters still recover types after learning**, relative to init.

**Maps (`common`, 12-d):** the same routing, but every cell is asked both “if you were labelled R” and “if you were labelled E” on the six receiver situations. Identical weights \(\Rightarrow \Delta\approx 0\). A late split is a split of the maps, including how they *use* the goal bits.

**Secondary (`both_heads`, 12-d):** \((v_{\mathrm{help}},v_{\mathrm{harm}})\) on the own-goal six probes. Harm in `original` is an untrained head; it is logged, not used as the headline.

### 2.3 Goal-in-input and `weights_only`

ψ sees \(g_i\) and \(g_r\). Own-goal `realized` therefore mixes map differences with the name tag. That is the right object for “do typed *agents* output differently.” It is the wrong object for “did the *maps* specialise.”

Controls:

1. **Init snapshot.** Re-build the grid from the same config+seed (no learning). Always compare late to init. On `realized`, init already clusters by type; a late ARI collapse is smear, not a new split. On `common`, init \(\Delta\) is the noise floor; \(\Delta_{\mathrm{late}}-\Delta_{\mathrm{init}}\) is learned map structure.
2. **`common`.** Shared \((g_s,g_r)\) list. Goal columns still fire, so kin/foe specialisation is visible.
3. **`weights_only`.** All goal bits zeroed. Misses specialisation that lives only in the goal-input columns; catches splits in biases and non-goal weights. At init this \(\Delta\) is ~0.

`weights_only` probes (v1): both-alive blank, both-alive excited (\(s_r=0.5\)), sender-alive / receiver-dead. Output: both vote heads (6-d).

### 2.4 Optional `full` vector

`both_heads` plus ψ messages on the six vote probes plus three f-signal probes (`alive, M=0`, `alive, M=0.5`, `dead, M=0`). Logged, not headline. Messages live in a different scale; z-scoring is mandatory here.

---

## 3. Scalars

Let \(X\in\mathbb{R}^{n\times p}\) be z-scored response rows (one cell per row). \(D_{ij}=\|X_i-X_j\|_2\). Within-type means are the **equal** average of the two class-conditional pairwise means (not pair-count weighted), so \(\Delta=\mathcal{E}/2\) of Székely–Rizzo. \(\Delta\) is nan unless both types have at least two cells.

\[
\Delta
=\underbrace{\operatorname{mean}\{D_{ij}: g_i\neq g_j\}}_{\text{between types}}
-
\tfrac12\Bigl(
\operatorname{mean}\{D_{ij}: g_i=g_j=R\}
+
\operatorname{mean}\{D_{ij}: g_i=g_j=E\}
\Bigr).
\]

\(\Delta>0\): types occupy different regions of response space relative to how much each type varies internally.

Alive-only \(\Delta\) **re-z-scores and rebuilds \(D\)** on the living cells (the acting population), it is not a subset of the all-cell matrix.

Also report:

| Scalar | Role |
|---|---|
| \(\Delta\) all cells / alive only | dead sites still hold a policy; alive-only is the acting population |
| within-R, within-E | one class tighter? |
| ARI of \(k\)-means (\(k=2\)) vs goal labels | do two clusters recover the planted types? |
| Silhouette of the **goal** labels in \(X\) | geometry of the labelled partition (not of \(k\)-means) |
| \(\Delta_{\mathrm{late}}-\Delta_{\mathrm{init}}\) | on `common`: learned map increase. On `realized`: smear vs init blobs |
| mean kin−foe realized vote by class | 1-d probe analogue of vote discrimination |
| \(\Phi\), \(\Phi_{\mathrm{late}}\), late min-type fraction | read jointly with \(\Delta\) |

Joint reading (use **`common`** \(\Delta\) as the map column; `realized` ARI as the cluster-preservation column):

| \(\Phi_{\mathrm{late}}\) | \(\Delta_{\mathrm{common}}\) | Read as |
|---|---|---|
| high | low | mix drift / attrition without two maps |
| low | high | two maps, balanced counts — **Φ miss** |
| high | high | mix moved *and* maps split; check min-type so it is not attrition |
| low | low | null on both |

Do not declare “two processes” from high \(\Phi\) alone, from high `realized` \(\Delta\) at init, or from \(\Delta=\mathrm{nan}\) / one type gone among the living. PCA/UMAP of the same vectors are the clustering picture the scalar summarises.

---

## 4. Clustering and visualisation

- \(k\)-means, \(k=2\), on z-scored vectors, seed 0. ARI vs goals. Primary clustering diagnostic for whether types are preserved in behaviour space.
- Optional extra \(k\in\{3,4\}\) only as a diagnostic of subtypes (not in the v1 headline table).
- **PCA (always)** and **UMAP (if `umap-learn` imports)** of the same vectors, coloured by goal / by \(k\)-means / by alive.
- Grid plots: goal map vs cluster-id map. Spatial blobs of a cluster that ignore goals would mean specialisation along space, not type.

Fit embeddings **per snapshot**. Do not jointly embed independently trained runs.

---

## 5. Implementation

### Artifacts from a suite / analysis run

`research/runner.py` writes, next to `series.npz`:

- `params_final.pt` — per-cell ψ and f
- `state_final.pt` — `x, s, h, goals, rho, u`

Init baselines are reconstructed with `simulate.build_grid(cfg)` from `config.json` (same seed). No extra init checkpoint.

### Files

| Path | Role |
|---|---|
| `research/FUNCTIONAL_DIVERGENCE.md` | this spec |
| `research/functional.py` | probes (`realized`, `common`, `weights_only`), \(\Delta=\mathcal{E}/2\), ARI, embeddings |
| `research/response_analysis.py` | CLI, plots, `FUNCTIONAL_REPORT.md` |
| `tests/test_functional_divergence.py` | vectorisation, confound controls, synthetic split |

### CLI

```bash
# Small original vs A demo (writes a suite folder, then analyses it)
python -m research.response_analysis run --versions original,A --n-steps 1000

# Post-hoc on any suite that has params_final.pt / state_final.pt
python -m research.response_analysis analyze research_results/<run_name>

# Smoke
python -m research.response_analysis run --quick
```

`analyze` is the hook for repeating a full paper suite later: run `research.suite` as usual (runner now stores snapshots), then point this module at the folder.

### Reproducing a full-paper comparison

Use the same base config, seeds, and horizon as the Φ tables (typically `research/configs/benchmark.json`, seeds `1096812628,42,7`, \(T=4000\)):

```bash
python -m research.suite run --versions original,A --n-steps 4000 --name class_div_functional_4000
python -m research.response_analysis analyze research_results/class_div_functional_4000
```

Then compare `FUNCTIONAL_REPORT.md` to `PHI_REPORT.md` / `REPORT.md` seed-by-seed.

---

## 6. Caveats (v1)

- Probe bank is a degree of freedom. It is frozen here; a new bank is a new metric version.
- `realized` is version-aware by design (original’s unused harm head is excluded). `common` uses the same routing on a shared \((g_s,g_r)\) list. `both_heads` / `weights_only` are not version-routed.
- Goal inheritance (C): colour and \(\Delta\) use **snapshot** goals, not init goals.
- Parameters live on lattice sites; they do not copy on birth. \(v_i\) is “this site’s learned map.”
- UMAP hyperparameters (`n_neighbors=15`, `min_dist=0.1`) are frozen. PCA is the dependency-free picture; UMAP is the illustration.
- Alive-only \(\Delta\) is nan if a living class has fewer than two cells. Prefer all-cells \(\Delta\) plus `late_min_type_frac`.
- Sender \(s=h=0\) on every bank: specialisation in how a cell uses its own accumulated state is outside v1.

---

## 7. What v1 is not

- Not a replacement for Φ, segregation, or occupancy-conditioned vote means.
- Not a causal test of typed routing by itself; it is a representation of cell behaviour. Original vs A on the same seeds is the test.
- Not parameter-space distance (\(\|\theta_i-\theta_j\|\)). Two weight vectors can implement similar maps; probes measure the map.
