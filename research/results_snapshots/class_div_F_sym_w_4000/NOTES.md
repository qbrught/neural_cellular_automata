# Soft coexistence (F) under symmetric w2=w3 (T=4000)

Base: `research/configs/benchmark_sym_w.json` with w2=w3=1.4974.
Seeds: [1096812628, 42, 7]. Versions: original, A, F (default λ=0.01, δ=1e-4).
F = A + soft coexistence barrier on soft self-masses (Experiment F / v1).

Primary suite artifacts: `research_results/class_div_F_sym_w_4000/`.
λ sweep: `research_results/class_div_F_lambda_sweep_sym_w_4000/`.

## Mean Φ / Φ_late (main suite, λ_F=0.01)

- **original**: Φ=0.0024, Φ_late=0.0185, success(Φ_late>0.2)=0/3, corr(r,fa)=0.999, corr(r,e)=0.996, min_type_late=0.479, alive_late=342.9
- **A**: Φ=0.1220, Φ_late=0.3325, success=2/3, corr(r,fa)=0.531, corr(r,e)=0.500, min_type_late=0.170, alive_late=264.4
- **F**: Φ=0.1101, Φ_late=0.3311, success=2/3, corr(r,fa)=0.568, corr(r,e)=0.537, min_type_late=0.171, alive_late=264.9

Δ(F−A) Φ=-0.0119, Φ_late=-0.0014, min_type_late=+0.0014

## Seed-level (A vs F, λ=0.01)

| seed | A Φ_late | F Φ_late | A min_type late | F min_type late | note |
| --- | --- | --- | --- | --- | --- |
| 1096812628 | 0.507 | 0.506 | 0.001 | 0.002 | E nearly extinct; large Φ via attrition |
| 42 | 0.491 | 0.487 | 0.019 | 0.023 | same pattern |
| 7 | 0.000 | 0.000 | 0.490 | 0.490 | full lattice density null |

## λ sweep (F under w2=w3, same seeds, T=4000)

| setting | mean Φ | mean Φ_late | success | min_type late | alive late | corr(r,e) |
| --- | --- | --- | --- | --- | --- | --- |
| A (λ=0) | 0.1220 | 0.3325 | 2/3 | 0.170 | 264.4 | 0.500 |
| F λ=0.01 | 0.1101 | 0.3311 | 2/3 | 0.171 | 264.9 | 0.537 |
| F λ=0.1 | 0.1190 | 0.3324 | 2/3 | 0.170 | 264.7 | 0.467 |
| F λ=1.0 | 0.1074 | 0.3319 | 2/3 | 0.171 | 264.7 | 0.477 |

Seed 7 remains a null at every λ. High-Φ seeds still end with near-extinct eliminators.

## Interpretation

**Negative result for F as a divergence stabilizer (v1, loss barrier on soft self-masses).**

1. **H2 fails:** mean Φ_late does not rise vs A; success rate stays 2/3.
2. **H1 fails at practical level:** does not prevent type attrition on high-Φ seeds or the full-grid null on seed 7.
3. **H5 / structural limit:** even λ=1 leaves dynamics nearly identical to A. Likely causes:
   - Barrier only trains through the weak f-survival channel (Path-1 self soft mass).
   - Alive-masked SGD: dead rare-type cells do not update even if they contribute soft birth mass.
   - Full lattice with balanced types (seed 7): ρ̃^R, ρ̃^E stay large → barrier is flat and small.
4. **Not H4 (over-regularization):** Φ does not collapse from forced 50/50; specialization pattern of A is preserved but unimproved.

**Thesis takeaway:** under w2=w3, large Φ_late from typed votes is still seed-fragile and often *attrition-driven* (one type nearly dies). Soft viability regularization via local self-p does not convert that into stable two-process coexistence. Divergence claims should not rely on F; optional backup paths (survival-logit bias, or v2 vote-routed barrier) remain untested.

## Hypothesis checklist

| H | Result |
| --- | --- |
| H1 viability | Fail (same null + attrition) |
| H2 Φ_late retained/up | Fail (flat / slightly down) |
| H3 not forced 50/50 | Pass (ratios stay unequal on high-Φ seeds) |
| H4 λ too large kills Φ | Not observed up to λ=1 |
| H5 λ too small | Compatible; also structural channel weakness |
