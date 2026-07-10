"""Random (and optional mutation) sampling of NCSA Configs for discovery.

Learning is always forced ON. Architecture knobs (d, hidden) stay at defaults
unless the caller overrides N / n_steps / device for the session.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace

from config import Config

# Search ranges around known-good hand-tuned defaults.
W0_RANGE = (-4.0, -0.5)
W1_RANGE = (-4.0, 1.0)
W2_RANGE = (0.0, 5.0)
W3_RANGE = (-5.0, 0.5)
W4_RANGE = (0.0, 2.0)
W5_RANGE = (-3.0, 1.0)
INIT_ALIVE_RANGE = (0.05, 0.45)
ETA_LOG_RANGE = (math.log(1e-3), math.log(5e-2))
NOISE_LOG_RANGE = (math.log(1e-3), math.log(5e-2))

# Mutation jitter (absolute, then clipped to range).
MUTATE_SIGMA = {
    "w0": 0.35,
    "w1": 0.35,
    "w2": 0.35,
    "w3": 0.35,
    "w4": 0.2,
    "w5": 0.3,
    "init_alive_prob": 0.05,
}


def _uniform(rng: random.Random, lo: float, hi: float) -> float:
    return round(rng.uniform(lo, hi), 4)


def _log_uniform(rng: random.Random, log_lo: float, log_hi: float) -> float:
    return round(math.exp(rng.uniform(log_lo, log_hi)), 6)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _rand_seed(rng: random.Random) -> int:
    return rng.randint(0, 2**31 - 1)


def sample_config(
    rng: random.Random,
    *,
    base: Config | None = None,
    n_steps: int = 1000,
    N: int = 20,
    device: str = "cpu",
) -> Config:
    """Draw a discovery Config. If ``base`` is set, mutate it instead of pure random."""
    if base is not None:
        return _mutate_config(rng, base, n_steps=n_steps, N=N, device=device)

    cfg = Config(
        N=N,
        d=8,
        hidden=16,
        eta=_log_uniform(rng, *ETA_LOG_RANGE),
        n_steps=n_steps,
        learn=True,
        require_alive_neighbour=True,
        w0=_uniform(rng, *W0_RANGE),
        w1=_uniform(rng, *W1_RANGE),
        w2=_uniform(rng, *W2_RANGE),
        w3=_uniform(rng, *W3_RANGE),
        w4=_uniform(rng, *W4_RANGE),
        w5=_uniform(rng, *W5_RANGE),
        seed=_rand_seed(rng),
        init_noise_std=_log_uniform(rng, *NOISE_LOG_RANGE),
        init_alive_prob=_uniform(rng, *INIT_ALIVE_RANGE),
        u_seed=_rand_seed(rng),
        output_dir="runs",
        save_state_vectors=False,
        device=device,
        run_name="",
    )
    cfg.learn = True
    cfg.__post_init__()
    return cfg


def _mutate_config(
    rng: random.Random,
    base: Config,
    *,
    n_steps: int,
    N: int,
    device: str,
) -> Config:
    def jitter(name: str, value: float, lo: float, hi: float) -> float:
        sigma = MUTATE_SIGMA[name]
        return round(_clip(value + rng.gauss(0.0, sigma), lo, hi), 4)

    cfg = replace(
        base,
        N=N,
        n_steps=n_steps,
        device=device,
        learn=True,
        save_state_vectors=False,
        run_name="",
        w0=jitter("w0", base.w0, *W0_RANGE),
        w1=jitter("w1", base.w1, *W1_RANGE),
        w2=jitter("w2", base.w2, *W2_RANGE),
        w3=jitter("w3", base.w3, *W3_RANGE),
        w4=jitter("w4", base.w4, *W4_RANGE),
        w5=jitter("w5", base.w5, *W5_RANGE),
        init_alive_prob=jitter(
            "init_alive_prob", base.init_alive_prob, *INIT_ALIVE_RANGE
        ),
        eta=round(
            _clip(
                base.eta * math.exp(rng.gauss(0.0, 0.3)),
                math.exp(ETA_LOG_RANGE[0]),
                math.exp(ETA_LOG_RANGE[1]),
            ),
            6,
        ),
        init_noise_std=round(
            _clip(
                base.init_noise_std * math.exp(rng.gauss(0.0, 0.3)),
                math.exp(NOISE_LOG_RANGE[0]),
                math.exp(NOISE_LOG_RANGE[1]),
            ),
            6,
        ),
        seed=_rand_seed(rng),
        u_seed=_rand_seed(rng),
    )
    cfg.learn = True
    cfg.__post_init__()
    return cfg
