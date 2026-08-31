# Neural State-Aware Cellular Automaton

**Version 1.0**

A 2D cellular automaton where each cell has a fixed goal (reproduce or eliminate) and learns per-cell MLP parameters to pursue that goal. Survival is governed by a fixed global CA rule that cells cannot change but learn to exploit.

![animation](animation.gif)

*20×20 grid, 300 steps. Green = alive reproducer, red = alive eliminator, dark = dead.*

## New changes from initial POC

The headline change is **Path 1**: the local update MLP `f` now actually learns.

In v0.0.1, the survival rule only listened to ψ's vote output, so only ψ received gradient. `f` ran every step but its parameters never moved. v0.0.2 fixes that by adding a second learnable channel into the survival rule:

$$p^{t+1}_i = \sigma(w_0 + w_1 A_i + w_2 R_i + w_3 E_i + w_4 V_i + w_5 \tanh(\mathbf{u} \cdot \tilde{s}^t_i))$$

Where $\tilde{s}^t_i$ is `f`'s proposed next-state output and $\mathbf{u}$ is a hand-set fixed projection vector. The tanh bounds the f-signal contribution so it can't trivially saturate the rule (without it, every cell learns "set $\mathbf{u} \cdot s$ very large, live forever" and the ecosystem freezes).

The loss now also includes a self-survival term:

$$\ell_i = -p^{t+1}_i + \text{sign}(g_i) \cdot \sum_{j \in \mathcal{N}(i)} p^{t+1}_j$$

Both goal types want themselves alive (so they can act next step); they differ on what they want for neighbours.

Together these give `f` a real gradient channel without breaking the locality property: cell $i$'s parameters still only update from cell $i$'s loss.

A new learn config flag toggles this channel. With learn=True (default), f learns as described above. With learn=False, f's parameters stay frozen at init — recovering the v0.0.1 behaviour where f runs every step but never moves. This makes it cheap to A/B the two regimes from a shared seed and watch directly how a learning f changes the ecosystem's trajectory versus the ψ-only baseline. (ψ continues to learn either way; the flag only gates f.)
## What this demonstrates

- A 2D grid of cells, each with its own ψ (message function) and `f` (local update) MLPs.
- A fixed survival rule that combines: neighbour alive count, reproducer count, eliminator count, weighted vote sum, and a bounded f-signal channel.
- Per-cell, per-step gradient descent on a local loss.
- **Both ψ and `f` learn**. Two separate channels — votes for influencing neighbours, f-signal for influencing self-survival.
- Gradient locality: cell *i*'s update depends only on cell *i*'s parameters. Verified by per-cell test across both ψ and `f` weights.
- Bit-exact reproducibility from a single seed.

## Key design choices

**The vote channel.** The spec's survival rule as originally written had no path from learnable parameters to the loss. We extended ψ to emit a vote scalar per outgoing edge, with the survival rule reading the rho-weighted sum of incoming votes. This gives ψ a way to influence neighbours' survival.

**The f-signal channel.** Reading `f`'s proposed next-state via a fixed projection $\mathbf{u}$ and a tanh-bounded weight. This gives `f` a way to influence self-survival, which is what gets it any gradient at all.

**Gradient locality via detach.** Every place where a neighbour's contribution to a survival probability appears, we detach everything except the current cell's own contribution. The result is that `loss.sum().backward()` produces per-cell gradients that respect locality without any per-cell loops.

The hand-tuned weights, per-cell goals, per-cell communication rates, and the projection vector $\mathbf{u}$ all remain fixed at init.


## Project structure

```
ncsa/
  config.py        # hyperparameters + seed + research version flags
  parameters.py    # per-cell ψ and f MLP weights, batched forward
  state.py         # per-cell state (x, s, h, goals, rho)
  grid.py          # Grid container, toroidal Moore-neighbourhood gather
  dynamics.py      # message pass, local update, survival rule (typed votes)
  learning.py      # per-cell loss, locality SGD step
  simulate.py      # full loop, trajectory writing
  visualise.py     # render trajectory.npz to summary.png / animation.gif
  run.py           # CLI
  server.py        # interactive UI
  research/        # paper version suite (original → A → B → C → D)
  experiments/     # one-off scientific probes
  tests/           # unit tests
```

## Research suite (paper comparisons)

Compare versions under fixed seeds with metrics, charts, and a markdown report.

**Thesis pipeline** (pairwise isolations, spatial frames, paired tests, four-chunk reports):

```bash
python -m research.pipeline list
python -m research.pipeline run --quick              # smoke
python -m research.pipeline run                      # frozen protocol (slow)
```

See [`research/THESIS_PIPELINE.md`](research/THESIS_PIPELINE.md).

Ad-hoc version lists still use the suite:

```bash
python -m research.suite list
python -m research.suite run --versions original,A          # full
python -m research.suite run --quick                        # smoke
```

Open `research_results/<run>/INDEX.md` (pipeline) or `REPORT.md` (suite). See [`research/README.md`](research/README.md).

Version flags (also in the UI): `typed_votes` (A), `predator_prey_loss` (B), `goal_inheritance` (C), `goal_in_f` (D), `coexistence_pressure` (F), `environment_heterogeneous` (G).

## Run simulation stand-alone

```bash
pip install torch numpy matplotlib
python run.py --visualise                        # defaults
python run.py --seed 7 --n-steps 500 --visualise # custom
```

Outputs land in `runs/<timestamp>/` as `config.json`, `trajectory.npz`, `params_final.pt`, and (with `--visualise`) `summary.png` and `animation.gif`.

## Automated config discovery

Search configs with **learning on**, prefilter crashes, and use **Gemini Flash** both to judge dynamics and to **propose the next config** (guided search). Pure random is still available via `--no-guided`.

```bash
pip install -r requirements-discovery.txt   # google-genai
export GEMINI_API_KEY=...                   # or GOOGLE_API_KEY

# Guided (default): VLM sees config + summary + recent history, returns
# analysis + next_config (small directed steps after extinction/static, etc.)
python discover.py --max-cycles 30 --target-discoveries 5

# Occasional random jumps while guided (default explore-prob=0.15)
python discover.py --max-cycles 40 --target-discoveries 5 --explore-prob 0.2

# Pure random (old behaviour)
python discover.py --no-guided --max-cycles 30 --target-discoveries 5

# Dry run: sims + prefilter + heuristic steering, no API
python discover.py --max-cycles 10 --dry-run --n-steps 500
```

Each guided cycle logs `analysis`, `strategy`, and which knobs change next. Saved finds land under `discoveries/` with a one-liner catalog.

Re-run a discovery:

```bash
python run.py --config discoveries/disc_0001/config.json --visualise
```

See `DISCOVERY_PLAN.md` for design details. Catalog: `discoveries/catalog.md`.

## Interactive UI

```bash
pip install fastapi uvicorn websockets
python server.py
# open http://127.0.0.1:8765
```

Sidebar sliders for every Config field; start/pause/reset buttons; live grid + alive/dead/loss counters; speed slider; download-current-config button. Setting `n_steps` blank runs indefinitely. The UI auto-renders new Config fields without code changes — only `UI_FIELDS` in `ui_config.py` needs editing.

### Experiment G terrain

Presets (`blobs`, `vertical_band`, …) are generated from knobs + `env_seed`. To place islands by hand, set `env_preset` to `custom` and list regions. Example: [`research/configs/g_custom_inland.json`](research/configs/g_custom_inland.json).

```json
"environment_heterogeneous": true,
"env_preset": "custom",
"env_regions": [
  {"shape": "disk", "cy": 5, "cx": 7, "radius": 3, "kappa_R": 0, "kappa_E": 0},
  {"shape": "disk", "cy": 10, "cx": 13, "radius": 3, "kappa_R": 0, "kappa_E": 0},
  {"shape": "rect", "r0": 14, "c0": 3, "r1": 17, "c1": 7, "kappa_R": 0, "kappa_E": 0}
]
```

Shapes: `disk` (`cy`, `cx`, `radius`), `rect` (`r0,c0,r1,c1`, inclusive, wraps), `band` (`axis` `h`|`v`, `center`, `width`). Optional channels: `kappa_R`, `kappa_E`, `eta_R`, `eta_E`, `occupancy` (omit = leave unchanged). Occupancy off + `kappa_*=0` is a transfer-dead island cells can still live in.

In the UI: Experiment G → Manual regions (JSON), or **click the grid** to drop a κ=0 disk. Download .json keeps the regions. `python run.py --config path.json --visualise` uses the same file.
