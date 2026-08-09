# Soft coexistence pressure — research & implementation brief

**Status:** proposal (not implemented)  
**Working name:** Experiment **F** / flag `coexistence_pressure`  
**Goal:** Keep both goal-classes present long enough that class divergence can mean *two coexisting dynamical processes*, not “one type won by attrition” or “grid saturated, Φ → 0.”  
**Context:** Under \(w_2=w_3\) and long horizon, typed votes (A) still produce large \(\Phi_{\mathrm{late}}\) on some seeds but **collapse to null on others** (e.g. full lattice, density tracking). Soft coexistence is a candidate stabilizer—not a substitute for typed interaction or timescale.

---

## 1. Motivation

### What we have now

Local loss for cell \(i\) (schematic):

\[
\ell_i
=
-p_i
+
\sum_{k\in\mathcal{N}(i)} c(g_i,g_k)\, p_k
\]

with \(p\) from the fixed survival logit. Nothing in \(\ell_i\) or the survival rule cares whether **global** (or regional) counts of both types stay positive. Consequences we have already seen:

| Failure mode | Effect on \(\Phi\) / narrative |
|--------------|--------------------------------|
| One type nearly dies | Residual vs \(f a_t\) can look large for uninteresting reasons; “two processes” story fails |
| Lattice fills, types well mixed | \(\Phi\to 0\), density null holds; A looks like a null on that seed |
| Seed-to-seed fragility under \(w_2=w_3\) | Same mechanism, opposite macroscopic outcomes |

### What soft coexistence is for

Add a **weak, smooth penalty** when either type becomes rare, so learning and survival pressure gently **preserve two living classes** while still allowing:

- unequal type ratios,
- anti-correlated \((r_t,e_t)\),
- large \(\Phi\) / \(\Phi_{\mathrm{late}}\).

It is an **experimental control / regularizer**, not a claim that nature must work this way. In writing: “viability regularizer so both roles remain observable.”

---

## 2. Mathematical formulation

### 2.1 Type masses

Let \(N_{\mathrm{cells}}=N^2\). At step \(t\):

\[
r_t=\sum_i x_i^t\mathbf{1}_{\{g_i=R\}},
\qquad
e_t=\sum_i x_i^t\mathbf{1}_{\{g_i=E\}},
\qquad
a_t=r_t+e_t.
\]

Normalized living fractions among alive cells (or among all cells—pick one and stick to it):

\[
\rho_t^R=\frac{r_t}{a_t+\varepsilon},
\qquad
\rho_t^E=\frac{e_t}{a_t+\varepsilon}.
\]

Global fractions of the grid (useful if \(a_t\) is large):

\[
\bar\rho_t^R=\frac{r_t}{N_{\mathrm{cells}}},
\qquad
\bar\rho_t^E=\frac{e_t}{N_{\mathrm{cells}}}.
\]

### 2.2 Recommended primary form: barrier on type densities (loss path)

Add a **global** soft barrier to every live cell’s loss (broadcast), or only to alive cells:

\[
\boxed{
\ell_i^{\mathrm{tot}}
=
\ell_i
+
\lambda\,
\Big(
-\log(\bar\rho_t^R+\delta)
-\log(\bar\rho_t^E+\delta)
\Big)
\cdot x_i
}
\]

with:

- \(\lambda\ge 0\) — strength (soft ⇒ small, e.g. \(10^{-3}\)–\(10^{-1}\) range to sweep),
- \(\delta>0\) — floor (e.g. \(1/N_{\mathrm{cells}}\) or \(10^{-4}\)) so the log is defined at zero,
- multiply by \(x_i\) so dead cells do not train (matches existing alive-masked SGD).

**Interpretation:** as either type’s global density \(\to 0\), \(-\log(\bar\rho+\delta)\to+\infty\) (softly), so every living cell pays more loss. Gradient descent on parameters then prefers configurations that **keep both \(\bar\rho^R\) and \(\bar\rho^E\) away from zero**.

Equivalent “product form” (same idea):

\[
\lambda\big(-\log(\bar\rho^R\bar\rho^E+\delta)\big)
\]

or a soft minimum barrier:

\[
\lambda\,\mathrm{softplus}\big(-\alpha\min(\bar\rho^R,\bar\rho^E)\big).
\]

**Start with the two-log form** — simple, standard (like entropy/barrier methods).

### 2.3 Differentiability and locality

\(\bar\rho^R,\bar\rho^E\) depend on **hard** \(x_i\) and discrete goals. For Path-1 locality we must not invent fake gradients through hard death.

**Recommended implementation (v1):**

1. Compute \(\bar\rho^R,\bar\rho^E\) from **current** `state.x` and `state.goals` with **`detach()`** (treat as constants for this step’s backward).
2. Add the scalar barrier \(B_t=\lambda(-\log\bar\rho^R_\perp-\log\bar\rho^E_\perp)\) to each alive cell’s loss as a **constant shift** for that step:

\[
\ell_i^{\mathrm{tot}}=\ell_i + B_t\cdot x_i.
\]

Then \(B_t\) **does not** create new gradient paths (same for all cells). It only changes the **scalar training pressure** if we later make \(\bar\rho\) depend on soft \(p\)—see v2.

**Problem with pure detach:** if \(B_t\) is constant w.r.t. parameters, it **does not train anything**. It only changes reported loss, not dynamics.

So for the regularizer to actually **change learning**, \(\bar\rho\) (or a surrogate) must depend on something parameters control.

### 2.4 Making it bite: soft density surrogates (necessary for learning effect)

Use **soft survival probabilities** already computed for learning:

\[
\tilde r_t=\sum_i p_i\,\mathbf{1}_{\{g_i=R\}},
\qquad
\tilde e_t=\sum_i p_i\,\mathbf{1}_{\{g_i=E\}},
\]

with the **same detach rules as in `compute_local_losses`** so that:

- cell \(i\)’s contribution to \(\tilde r\) or \(\tilde e\) only keeps gradient through **paths we already allow** (own \(f\)-signal for \(p_i\); own votes into neighbours’ \(p_j\) for the neighbour sum—optional).

**Simplest v1 that actually trains (recommended):**

Define soft global masses from **self** soft probs only (already local in Path-1 for \(p_i\) via \(f\)):

\[
\tilde r=\sum_i p_i^{\mathrm{self}}\mathbf{1}_{g_i=R},
\qquad
\tilde e=\sum_i p_i^{\mathrm{self}}\mathbf{1}_{g_i=E},
\]

\[
B=\lambda\Big(-\log\frac{\tilde r}{N_{\mathrm{cells}}}+\delta-\log\frac{\tilde e}{N_{\mathrm{cells}}}+\delta\Big).
\]

Add \(B\) once to the **sum of losses** (or \(B/N_{\mathrm{alive}}\) per alive cell). Then:

\[
\frac{\partial B}{\partial \theta_i}
\propto
-\lambda\cdot\frac{\mathbf{1}_{g_i=R}}{\tilde r+\cdots}\frac{\partial p_i}{\partial\theta_i}
\quad\text{(and analog for \(E\))}
\]

so a reproducer cell is pushed to **raise \(p_i\)** when \(\tilde r\) is small (via \(f\) / self-survival channel), and likewise for eliminators.

**Optional stronger v2:** also route through vote channels so cells can help rare-type neighbours’ \(p_j\) — more powerful, more entangled with typed votes; implement later if v1 is too weak.

### 2.5 Alternative: survival-logit bias (physics path, no loss change)

Instead of loss, add a bias to the survival logit of rare types:

\[
\mathrm{logit}_i
\;+\=
\;
\beta\cdot\mathbf{1}_{\{g_i=R\}}\cdot\sigma\big(-\gamma(\bar\rho^R-\tau)\big)
\;+\;
\beta\cdot\mathbf{1}_{\{g_i=E\}}\cdot\sigma\big(-\gamma(\bar\rho^E-\tau)\big)
\]

with threshold \(\tau\) (e.g. 0.05–0.15 of grid) and strength \(\beta\).  
When type \(R\) is rare, all \(R\) cells get a survival boost.

- **Pros:** acts even with `learn=False`; easy to see in dynamics.  
- **Cons:** changes the CA law (environment), not “learning pressure”; more of a hard ecological crutch.  
- **Use as backup** if loss-based soft barrier is too weak or unstable.

**Prefer loss-based soft barrier first** so the story stays “local learning under a viability regularizer.”

---

## 3. Expected results (hypotheses)

Assume stack **A + long horizon (T=4000) + optionally \(w_2=w_3\)** (harder setting).

| Hypothesis | Expected observation if F works |
|------------|----------------------------------|
| **H1 (viability)** | Fewer seeds with \(r\to 0\) or \(e\to 0\) or trivial full-grid mix with \(\Phi=0\) |
| **H2 (divergence retained)** | \(\Phi_{\mathrm{late}}\) stays **large** (or increases) on average vs A alone under \(w_2=w_3\) |
| **H3 (not forced 50/50)** | Final ratios not pinned at \(1/2\); \(\mathrm{corr}(r,e)\) can stay negative |
| **H4 (λ too large)** | Both types live but \(\Phi\) **drops** (forced coexistence kills specialization) |
| **H5 (λ too small)** | Same fragility as now |

**Success criterion (worthwhile):**  
Under \(w_2=w_3\), A+F raises **fraction of seeds with \(\Phi_{\mathrm{late}}>\tau\)** (e.g. \(\tau=0.2\)) and mean \(\Phi_{\mathrm{late}}\) vs A, without driving extinction rate up.

**Failure (not worthwhile as a divergence tool):**  
Only increases alive counts / balance but **lowers** \(\Phi\), or requires huge \(\lambda\) that freezes the ecology.

---

## 4. Implementation sketch (for an agent / later PR)

### 4.1 Config

```python
# config.py
coexistence_pressure: bool = False
coexistence_lambda: float = 0.01   # λ
coexistence_delta: float = 1e-4    # δ in logs
# optional: coexistence_mode: "loss_soft_mass" | "survival_bias"
```

No need for a deep new “mechanism layer” beyond a flag + λ—unlike typed votes. Suite version **F** = A (or B) + `coexistence_pressure=True` with a documented default λ.

### 4.2 Code touch points

| File | Change |
|------|--------|
| `config.py` | flags above |
| `learning.py` → `compute_local_losses` | after `p_self` is built, compute \(\tilde r,\tilde e\), \(B\), add \(B * state.x` to per-cell loss (or add \(B\) to `losses.sum()` once—equivalent if carefully scaled) |
| `research/versions.py` | version `F` / `A_coexist`: typed_votes=True, coexistence on, document λ |
| `research/metrics.py` / runner | log `rho_R`, `rho_E`, min type fraction, barrier value \(B_t\) over time |
| tests | barrier → +∞ as one soft mass → 0; with λ=0 bit-identical to baseline; gradient only on self \(f\) path for rare type |

### 4.3 Scaling

Adding \(B\) to **each** of \(N_{\mathrm{alive}}\) cells multiplies the barrier by \(N_{\mathrm{alive}}\) in `losses.sum()`. Prefer:

```text
total_loss = (losses * alive).sum() + B
```

with \(B=\lambda(-\log\tilde\rho^R-\log\tilde\rho^E)\), **once per step**, so λ is interpretable and independent of grid size in a controlled way. (Still scales with how \(p\) is summed—document \(\tilde\rho=\tilde r/N^2\).)

### 4.4 Gradient locality checklist

- Do **not** let \(B\) depend on undetached neighbour vote sums in v1.  
- \(p_i^{\mathrm{self}}\) uses existing Path-1 detaches.  
- Goals discrete: masks \(\mathbf{1}_{g_i=R}\) are non-diff (fine).  
- Unit test: perturb only cell \(i\)’s \(f\) weights → only that cell’s contribution to \(\tilde r\) or \(\tilde e\) moves \(B\).

### 4.5 Evaluation protocol

1. Base: benchmark with **\(w_2=w_3\)** (hard setting) and **asymmetric** (reference).  
2. Versions: `A` vs `A+F` (and maybe `original+F` as negative control).  
3. Seeds: at least the three standard + a few more if possible.  
4. \(T=4000\).  
5. Report: \(\Phi\), \(\Phi_{\mathrm{late}}\), min\((r,e)/a\), extinction, seed-wise “success” rate \(\Phi_{\mathrm{late}}>\tau\).  
6. Sweep \(\lambda\in\{0, 10^{-3}, 10^{-2}, 10^{-1}, 1\}\) on one seed first.

---

## 5. Risks and interactions

| Risk | Detail |
|------|--------|
| **Kills divergence** | Strong λ forces both types alive and similar roles → low \(\Phi\) |
| **Conflicts with eliminators** | Eliminators want fewer neighbours; barrier wants both types present → interesting tension, not a bug |
| **Inheritance (C)** | Coexistence vs colonization/takeover — may fight C; test **fixed goals first** |
| **Global signal** | Soft masses are global ⇒ slight departure from pure locality of *objectives* (params still local). Disclose in text |
| **Not free lunch** | Extra design choice; must be framed as regularizer |

---

## 6. Is it worthwhile? (recommendation)

**Yes, worth a short implementation + λ sweep**, for these reasons:

1. It directly targets the **fragility** you care about under \(w_2=w_3\) (null seeds / extinction of a class).  
2. Implementation is **small** (loss term + flag), no new MLP I/O.  
3. Even a **negative result** is useful: “viability regularizer stabilizes coexistence but does not raise \(\Phi\)” sharpens the claim that divergence is from typed interaction + time, not from keeping both types alive.  
4. Positive result (higher seed success rate for large \(\Phi_{\mathrm{late}}\)) strengthens the **two dynamical processes** narrative.

**Priority:** after documenting A + horizon + \(w_2,w_3\) control; **before** dead regions (your last spatial section).  
**Do not** expect coexistence alone to create divergence from original—pair it with **A**.

**Skip or deprioritize if:** you are happy treating seed failure under symmetric \(w\) as part of the story and do not need higher success rate for thesis plots.

---

## 7. One-paragraph abstract (for notes / professor)

> Soft coexistence pressure augments the per-step objective with a weak barrier on the soft living mass of each goal-class, \(-\lambda\log\tilde\rho^R-\lambda\log\tilde\rho^E\), so local gradient updates are discouraged from extinguishing either type. The aim is not to force equal populations but to keep both classes observable under long-horizon learning, especially when type-symmetric survival weights make class divergence seed-dependent. We will implement this as an optional loss regularizer, sweep \(\lambda\), and test whether the fraction of runs with large late class divergence \(\Phi_{\mathrm{late}}\) increases under typed votes without collapsing specialization.

---

## 8. Minimal decision checklist before coding

- [ ] v1 = loss barrier on soft self-masses only (recommended)  
- [ ] Add once to `total_loss`, not per-cell broadcast of \(B\) without scaling  
- [ ] Fixed goals + A first; \(w_2=w_3\) and asymmetric both  
- [ ] Success = seed success rate + mean \(\Phi_{\mathrm{late}}\), not only mean Φ  
- [ ] Report as regularizer, not emergent ecology  

---

## 9. Suggested suite version (when implemented)

```text
F / A_coexist:
  typed_votes=True
  predator_prey_loss=False
  goal_inheritance=False
  goal_in_f=False
  coexistence_pressure=True
  coexistence_lambda=<default from sweep>
```

Compare: `original`, `A`, `A_coexist` at T=4000 under \(w_2=w_3\) and under benchmark weights.
