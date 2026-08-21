"""Config sampling for discovery: random, mutate, and VLM-guided proposals."""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Any

from config import Config

# Search ranges around known-good hand-tuned defaults.
W0_RANGE = (-4.0, -0.5)
W1_RANGE = (-4.0, 1.0)
W2_RANGE = (0.0, 5.0)
W3_RANGE = (-5.0, 0.5)
W4_HELP_RANGE = (0.0, 2.0)
W4_HARM_RANGE = (0.0, 2.0)
W5_RANGE = (-3.0, 1.0)
INIT_ALIVE_RANGE = (0.05, 0.45)
ETA_RANGE = (1e-3, 5e-2)
NOISE_RANGE = (1e-3, 5e-2)

# Absolute clamp ranges for VLM / heuristic proposals.
FIELD_RANGES: dict[str, tuple[float, float]] = {
    "w0": W0_RANGE,
    "w1": W1_RANGE,
    "w2": W2_RANGE,
    "w3": W3_RANGE,
    "w4_help": W4_HELP_RANGE,
    "w4_harm": W4_HARM_RANGE,
    "w5": W5_RANGE,
    "init_alive_prob": INIT_ALIVE_RANGE,
    "eta": ETA_RANGE,
    "init_noise_std": NOISE_RANGE,
}

# Fields the VLM is allowed to set (absolute values, then clamped).
GUIDABLE_FIELDS = tuple(FIELD_RANGES.keys())

# Mutation jitter (absolute, then clipped to range).
MUTATE_SIGMA = {
    "w0": 0.35,
    "w1": 0.35,
    "w2": 0.35,
    "w3": 0.35,
    "w4_help": 0.2,
    "w4_harm": 0.2,
    "w5": 0.3,
    "init_alive_prob": 0.05,
}


def _uniform(rng: random.Random, lo: float, hi: float) -> float:
    return round(rng.uniform(lo, hi), 4)


def _log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return round(math.exp(rng.uniform(math.log(lo), math.log(hi))), 6)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _rand_seed(rng: random.Random) -> int:
    return rng.randint(0, 2**31 - 1)


def config_knobs(cfg: Config) -> dict[str, float]:
    """Serializable knobs that control dynamics (for prompts / logging)."""
    return {
        "w0": cfg.w0,
        "w1": cfg.w1,
        "w2": cfg.w2,
        "w3": cfg.w3,
        "w4_help": cfg.w4_help,
        "w4_harm": cfg.w4_harm,
        "w5": cfg.w5,
        "init_alive_prob": cfg.init_alive_prob,
        "eta": cfg.eta,
        "init_noise_std": cfg.init_noise_std,
        "seed": float(cfg.seed),
        "u_seed": float(cfg.u_seed),
    }


def apply_version_flags(cfg: Config, version: str | None) -> Config:
    """Force paper-version flags onto a sampled config (A/B/C/original).

    Weights and init knobs are left unchanged; only mechanism flags are set.
    Version E also symmetrizes w2=w3 to their mean.
    """
    if not version:
        return cfg
    key = version.strip()
    # Local import avoids hard coupling at module load for unit tests.
    from research.versions import get_version

    return get_version(key).apply(cfg)


def _resample_env_seed_if_g(cfg: Config, rng: random.Random, version: str | None) -> Config:
    """Under G / G_learn, draw a fresh env_seed. Do not add env knobs to GUIDABLE_FIELDS."""
    if not version:
        return cfg
    from research.versions import get_version

    spec = get_version(version)
    if spec.environment_heterogeneous:
        return replace(cfg, env_seed=_rand_seed(rng))
    return cfg


def resample_seed(
    base: Config,
    rng: random.Random,
    *,
    n_steps: int,
    N: int,
    device: str,
    version: str | None = None,
    resample_u_seed: bool = False,
) -> Config:
    """Clone ``base`` physics and draw a new IC seed (optionally a new ``u_seed``).

    Survival weights, learning rate, init density, and version flags stay
    frozen. Used for seed search in a fixed null (e.g. baseline w2=w3).
    """
    cfg = replace(
        base,
        N=N,
        n_steps=n_steps,
        device=device,
        learn=True,
        save_state_vectors=False,
        run_name="",
        seed=_rand_seed(rng),
        u_seed=_rand_seed(rng) if resample_u_seed else base.u_seed,
    )
    cfg.learn = True
    cfg.__post_init__()
    cfg = apply_version_flags(cfg, version)
    return _resample_env_seed_if_g(cfg, rng, version)


def sample_config(
    rng: random.Random,
    *,
    base: Config | None = None,
    n_steps: int = 1000,
    N: int = 20,
    device: str = "cpu",
    version: str | None = None,
) -> Config:
    """Draw a discovery Config. If ``base`` is set, mutate it instead of pure random."""
    if base is not None:
        cfg = _mutate_config(rng, base, n_steps=n_steps, N=N, device=device)
        cfg = apply_version_flags(cfg, version)
        return _resample_env_seed_if_g(cfg, rng, version)

    cfg = Config(
        N=N,
        d=8,
        hidden=16,
        eta=_log_uniform(rng, *ETA_RANGE),
        n_steps=n_steps,
        learn=True,
        require_alive_neighbour=True,
        w0=_uniform(rng, *W0_RANGE),
        w1=_uniform(rng, *W1_RANGE),
        w2=_uniform(rng, *W2_RANGE),
        w3=_uniform(rng, *W3_RANGE),
        w4_help=_uniform(rng, *W4_HELP_RANGE),
        w4_harm=_uniform(rng, *W4_HARM_RANGE),
        w5=_uniform(rng, *W5_RANGE),
        seed=_rand_seed(rng),
        init_noise_std=_log_uniform(rng, *NOISE_RANGE),
        init_alive_prob=_uniform(rng, *INIT_ALIVE_RANGE),
        u_seed=_rand_seed(rng),
        output_dir="runs",
        save_state_vectors=False,
        device=device,
        run_name="",
    )
    cfg.learn = True
    cfg.__post_init__()
    cfg = apply_version_flags(cfg, version)
    return _resample_env_seed_if_g(cfg, rng, version)


def apply_proposal(
    base: Config,
    proposal: dict[str, Any] | None,
    rng: random.Random,
    *,
    n_steps: int,
    N: int,
    device: str,
    jitter: float = 0.0,
    version: str | None = None,
) -> Config | None:
    """Build next Config from a partial absolute proposal on top of ``base``.

    Returns None if proposal is empty/invalid. Always resamples seed/u_seed.
    Values are clamped to FIELD_RANGES. Unknown keys ignored.
    """
    if not proposal or not isinstance(proposal, dict):
        return None

    updates: dict[str, float] = {}
    for key in GUIDABLE_FIELDS:
        if key not in proposal or proposal[key] is None:
            continue
        try:
            val = float(proposal[key])
        except (TypeError, ValueError):
            continue
        lo, hi = FIELD_RANGES[key]
        if jitter > 0:
            # Tiny noise so successive guided steps aren't bit-identical.
            span = hi - lo
            val = val + rng.gauss(0.0, jitter * span)
        updates[key] = round(_clip(val, lo, hi), 6 if key in ("eta", "init_noise_std") else 4)

    if not updates:
        return None

    cfg = replace(
        base,
        N=N,
        n_steps=n_steps,
        device=device,
        learn=True,
        save_state_vectors=False,
        run_name="",
        seed=_rand_seed(rng),
        u_seed=_rand_seed(rng),
        **updates,
    )
    cfg.learn = True
    try:
        cfg.__post_init__()
    except ValueError:
        return None
    cfg = apply_version_flags(cfg, version)
    return _resample_env_seed_if_g(cfg, rng, version)


def heuristic_next_config(
    base: Config,
    *,
    outcome: str,
    rng: random.Random,
    n_steps: int,
    N: int,
    device: str,
    version: str | None = None,
) -> Config:
    """Cheap non-VLM steering for dry-run / VLM failure (small directed steps)."""
    prop: dict[str, float] = {}
    # Default: mild jitter around current.
    for k in ("w0", "w1", "w2", "w3", "w4_help", "w4_harm", "w5", "init_alive_prob"):
        lo, hi = FIELD_RANGES[k]
        cur = getattr(base, k)
        prop[k] = cur + rng.gauss(0.0, 0.15 * (hi - lo))

    reason = outcome.lower()
    if "extinct" in reason or "extinction" in reason:
        # Toward life: less death bias, more reproducer support, denser start.
        prop["w0"] = base.w0 + 0.35
        prop["w2"] = base.w2 + 0.45
        prop["w3"] = base.w3 + 0.2  # less harsh elim (w3 often negative)
        prop["init_alive_prob"] = base.init_alive_prob + 0.05
    elif "static" in reason or "freeze" in reason or "saturated" in reason:
        # Break freeze: more vote/f-signal influence, slightly more death pressure.
        prop["w0"] = base.w0 - 0.25
        prop["w4_help"] = base.w4_help + 0.25
        prop["w4_harm"] = base.w4_harm + 0.25
        prop["w5"] = base.w5 + (0.3 if base.w5 < 0 else -0.2)
        prop["eta"] = base.eta * 1.3
    elif "boring" in reason or "reject" in reason:
        prop["w4_help"] = base.w4_help + rng.choice([-0.3, 0.3])
        prop["w4_harm"] = base.w4_harm + rng.choice([-0.3, 0.3])
        prop["w5"] = base.w5 + rng.choice([-0.35, 0.35])
        prop["w2"] = base.w2 + rng.gauss(0.0, 0.4)
    elif "saved" in reason or "interesting" in reason:
        # Local refine around a good basin.
        for k in ("w0", "w1", "w2", "w3", "w4_help", "w4_harm", "w5"):
            lo, hi = FIELD_RANGES[k]
            prop[k] = getattr(base, k) + rng.gauss(0.0, 0.08 * (hi - lo))

    cfg = apply_proposal(
        base, prop, rng, n_steps=n_steps, N=N, device=device, jitter=0.0,
        version=version,
    )
    if cfg is None:
        return sample_config(
            rng, base=base, n_steps=n_steps, N=N, device=device, version=version,
        )
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
        w4_help=jitter("w4_help", base.w4_help, *W4_HELP_RANGE),
        w4_harm=jitter("w4_harm", base.w4_harm, *W4_HARM_RANGE),
        w5=jitter("w5", base.w5, *W5_RANGE),
        init_alive_prob=jitter(
            "init_alive_prob", base.init_alive_prob, *INIT_ALIVE_RANGE
        ),
        eta=round(
            _clip(
                base.eta * math.exp(rng.gauss(0.0, 0.3)),
                ETA_RANGE[0],
                ETA_RANGE[1],
            ),
            6,
        ),
        init_noise_std=round(
            _clip(
                base.init_noise_std * math.exp(rng.gauss(0.0, 0.3)),
                NOISE_RANGE[0],
                NOISE_RANGE[1],
            ),
            6,
        ),
        seed=_rand_seed(rng),
        u_seed=_rand_seed(rng),
    )
    cfg.learn = True
    cfg.__post_init__()
    return cfg
