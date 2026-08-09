# Class divergence ladder (after Step D)

Regenerate full suite with series under `research_results/class_div_ladder_with_D/` (gitignored).

**Primary measure:** \(\Phi = \frac1T\sum_t |r_t - f_{\mathrm{init}} a_t|/(a_t+\varepsilon)\)

**Setup:** benchmark.json, 400 steps, seeds 1096812628, 42, 7.

| version | mean Φ | note |
|--------|--------|------|
| original | 0.032 | near density null |
| A | 0.049 | slight lift |
| B | 0.016 | lower than A |
| C | 0.270 | **strong** via goal inheritance / composition drift |
| D_fixed | 0.022 | goal_in_f without C; no strong count Φ |
| D | 0.333 | C + goal_in_f; strongest Φ, still inheritance-driven |

**Interpretation:** goal-conditioned `f` alone (D_fixed) does not create strong count-level class divergence on this base. Large Φ requires dynamical goals (C/D). D modestly increases Φ vs C.
