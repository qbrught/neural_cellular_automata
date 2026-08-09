# Class divergence ladder with Step D

Primary measure: Φ = mean_t |r_t - f_init a_t| / (a_t+ε)

Seeds: [1096812628, 42, 7], steps: 400, base: research/configs/benchmark.json

## Versions
- original: indiscriminate votes
- A: typed votes
- B: A + predator–prey loss
- C: A+B + goal inheritance
- D_fixed: A+B + goal_in_f (goals fixed)
- D: A+B+C + goal_in_f

## Mean Φ
- original: 0.0320
- A: 0.0490
- B: 0.0157
- C: 0.2697
- D_fixed: 0.0218
- D: 0.3333
