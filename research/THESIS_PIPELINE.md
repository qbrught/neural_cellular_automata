# Thesis results pipeline

One command generates the evidence for every feature letter: isolations, charts, spatial frames, paired tests, and a four-part report (motivation, visual, objective, insights). Add a letter by registering a comparison — do not fork a new suite.

```bash
python -m research.pipeline list
python -m research.pipeline run --quick              # smoke: letter A, 1 seed, short T
python -m research.pipeline run                      # frozen protocol (slow)
python -m research.pipeline run --letters A,F --include-ladder
python -m research.pipeline run --lambda-sweep
python -m research.pipeline report research_results/<name>   # rebuild markdown/figures from cache
```

Existing `python -m research.suite` still works for ad-hoc version lists. This pipeline is the **thesis method**.

---

## What it produces

```
research_results/<name>/
  protocol.json              frozen knobs for this run
  INDEX.md                   start here
  summary_all.csv
  cache/<config>/<arm>/seed_*/
    series.npz, summary.json, frames.npz, panel.png, config.json
  comparisons/<letter>/
    REPORT.md                four thesis chunks
    NOTES.md                 visual checklist (fill while looking at frames)
    summary.csv
    comparison/              overlay charts
    frames/                  off vs on grids at t=0 / mid / late
```

Each unique `(config, arm, seed)` is simulated **once** and reused across comparisons (A vs original and A vs B share the same A runs).

---

## Frozen protocol (`research/protocol.py`)

| Knob | Default | Why |
| --- | --- | --- |
| Primary config | `configs/benchmark_sym_w.json` | \(w_2=w_3\); typed-vote effects are not confounded by asymmetric neighbour physics |
| Asymmetric config | `configs/benchmark.json` | used only for letter E (does symmetrizing \(w_2,w_3\) change \(\Phi\)?) |
| \(T\) | 4000 | \(\Phi_{\mathrm{late}}\) is a late-time claim |
| Seeds | `discoveries_original_sym_w` 20 (`disc_original_0001`–`0020`) | same frozen knobs as `benchmark_sym_w.json`; seed-search under **original**, not under E |
| Frame times | \(t=0\), mid, last | spatial archive for the visual chunk |
| \(\Phi_{\mathrm{late}}\) hit | \(>0.2\) | same threshold as `analyze_class_div` |
| Two-process floor | \(\min(r,e)/a > 0.10\) late | attrition vs two living classes |

`--quick` overrides \(T\) and seeds only. `--seeds` / `--n-steps` override for a custom run; the values used are written to `protocol.json`.

---

## Comparison registry (`research/comparisons.py`)

Default isolations (feature **on** vs the right **off**, nothing else changed):

| id | Off | On | Config | Primary extras besides \(\Phi\) |
| --- | --- | --- | --- | --- |
| `A` | original | A | sym | vote disc, death gap, `corr(ra,dens)` |
| `B` | A | B | sym | E-on-E vs E-on-R death |
| `C_only` | original | C_only | sym | `g_frac drift` |
| `C` | B | C | sym | `g_frac drift`, `corr(ra,dens)` |
| `D_fixed` | B | D_fixed | sym | \(f\)-signal gap by type |
| `D` | C | D | sym | \(f\)-signal gap, `g_frac drift` |
| `E` | A | E | **asym** | \(\Delta\Phi\) from \(w_2=w_3\) |
| `F` | A | F | sym | two-process rate, `min(r,e)/a`, \(\tilde\rho\) |
| `G` | A | G | sym | alive fraction in low-\(\kappa\) |
| `G_learn` | A | G_learn | sym | \(\eta\) among alive, \(f\)-signal |

Optional (`--include-ladder`, `--lambda-sweep`, `--identity`):

| id | What |
| --- | --- |
| `ladder` | full version ladder on sym (appendix) |
| `F_lambda` | A vs F at \(\lambda\in\{0.01,0.1,1\}\) |
| `E_identity` | A vs E on **sym** (expect \(\Delta\Phi\approx 0\)) |

---

## How to add a letter

1. Implement the mechanism + `VersionSpec` in `research/versions.py` (existing pattern).
2. Append a `Comparison(...)` in `research/comparisons.py` (off/on arms, primary metrics, four-chunk prose).
3. If you need a new time series, add a collector in `research/runner.py` (`_extra_diagnostics`) and a summary line in `research/metrics.py`. Charts pick up extra keys automatically when present.
4. Re-run: `python -m research.pipeline run --letters <new> --skip-existing` (shared arms are reused).

No new CLI, no copy of the suite.

---

## Per-letter writing method (what the REPORT is built for)

1. **Motivation** — copied from the comparison registry (edit there, not in the output).
2. **Visual** — open `comparisons/<id>/frames/` and the live UI on the same `config.json` + seed; fill `NOTES.md`.
3. **Objective** — paired \(\Delta\) on the same seeds (mean, bootstrap 95% CI, Wilcoxon), \(\Phi\) / \(\Phi_{\mathrm{late}}\) / two-process rate, plus that letter’s primary extras.
4. **Insights** — auto draft from the numbers (hypothesis hold / fail, attrition vs two-process); you edit the last paragraph.

---

## Stages (all skippable)

`run` = simulate missing cache entries → charts → spatial montages → reports.

`report` = charts + montages + reports from an existing cache (no sims).

`--skip-existing` leaves cache hits untouched.
