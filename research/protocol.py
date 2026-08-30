"""Frozen thesis comparison protocol.

Edit this file to change the default pipeline (seeds, horizon, configs).
Every pipeline run writes a copy of the resolved knobs to protocol.json
so a results folder stays interpretable after you change defaults.
"""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve().parent

CONFIG_SYM = _HERE / "configs" / "benchmark_sym_w.json"
CONFIG_ASYM = _HERE / "configs" / "benchmark.json"

# Early-snapshot trio (suite DEFAULT_SEEDS). Not in the original-sym 20.
SEEDS_HISTORICAL: tuple[int, ...] = (1096812628, 42, 7)

# Seeds from discoveries_original_sym_w/disc_original_0001..0020 (catalog order).
# Frozen survival knobs + u_seed=731007425, w2=w3=1.4974; only `seed` varies.
# Saved because original-on-sym looked interesting (not the E-divergence bank).
SEEDS_ORIGINAL_SYM: tuple[int, ...] = (
    459903122,   # disc_original_0001
    609805632,   # disc_original_0002
    1156131200,  # disc_original_0003
    985660793,   # disc_original_0004
    78966860,    # disc_original_0005
    1940754639,  # disc_original_0006
    671156278,   # disc_original_0007
    1496598298,  # disc_original_0008
    1596966266,  # disc_original_0009
    1410963482,  # disc_original_0010
    499949037,   # disc_original_0011
    592525896,   # disc_original_0012
    795737329,   # disc_original_0013
    150865030,   # disc_original_0014
    964612928,   # disc_original_0015
    672602559,   # disc_original_0016
    2129935660,  # disc_original_0017
    288842927,   # disc_original_0018
    550842555,   # disc_original_0019
    1345866039,  # disc_original_0020
)

SEEDS_THESIS: tuple[int, ...] = SEEDS_ORIGINAL_SYM
SEEDS_VISUAL: tuple[int, ...] = SEEDS_ORIGINAL_SYM[:3]

N_STEPS: int = 4000

# Smoke: enough steps that early/late 10% windows are at least 1 step.
QUICK_N_STEPS: int = 60
QUICK_SEEDS: tuple[int, ...] = (SEEDS_ORIGINAL_SYM[0],)

# Sparse spatial dumps as fractions of T (inclusive of 0 and 1).
FRAME_FRACS: tuple[float, ...] = (0.0, 0.5, 1.0)

PHI_LATE_HIT: float = 0.2
COEXIST_FLOOR: float = 0.10

F_LAMBDAS: tuple[float, ...] = (0.01, 0.1, 1.0)

BOOTSTRAP_ITERS: int = 4000
BOOTSTRAP_SEED: int = 20260822

CONFIG_IDS: dict[str, Path] = {
    "sym": CONFIG_SYM,
    "asym": CONFIG_ASYM,
}


def frame_steps(n_steps: int, fracs: tuple[float, ...] = FRAME_FRACS) -> tuple[int, ...]:
    """Snapshot times as number of updates applied, in [0, n_steps].

    0 is the initial grid; ``n_steps`` is the final grid.
    """
    n = int(n_steps)
    if n <= 0:
        return ()
    out: list[int] = []
    for f in fracs:
        t = int(round(float(f) * n))
        t = max(0, min(n, t))
        if t not in out:
            out.append(t)
    return tuple(out)
