"""Seed-only resampling keeps frozen physics (baseline w2=w3 null)."""

from __future__ import annotations

import random
from pathlib import Path

from config import Config
from discovery.sample import resample_seed
from research.versions import get_version


_ROOT = Path(__file__).resolve().parent.parent
_SYM_BASE = _ROOT / "research" / "configs" / "benchmark_sym_w.json"

FROZEN = (
    "w0", "w1", "w2", "w3", "w4_help", "w4_harm", "w5",
    "eta", "init_alive_prob", "init_noise_std", "u_seed",
)


def test_resample_seed_freezes_symmetric_baseline():
    base = Config.load(_SYM_BASE)
    spec = get_version("E")
    base = spec.apply(base)
    assert base.w2 == base.w3

    rng = random.Random(0)
    seeds = []
    for _ in range(20):
        cfg = resample_seed(
            base, rng, n_steps=200, N=20, device="cpu", version="E",
        )
        assert cfg.learn is True
        assert cfg.w2 == cfg.w3
        assert abs(cfg.w2 - 1.4974) < 1e-6
        for name in FROZEN:
            assert getattr(cfg, name) == getattr(base, name), name
        assert cfg.typed_votes is True
        assert cfg.predator_prey_loss is False
        assert cfg.goal_inheritance is False
        seeds.append(cfg.seed)

    assert len(set(seeds)) == 20


def test_resample_u_seed_optional():
    base = Config.load(_SYM_BASE)
    rng = random.Random(1)
    cfg = resample_seed(
        base, rng, n_steps=100, N=20, device="cpu", resample_u_seed=True,
    )
    assert cfg.u_seed != base.u_seed
    assert cfg.w2 == base.w2 and cfg.w3 == base.w3
