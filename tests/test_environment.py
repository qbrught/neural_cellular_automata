"""Experiment G environment generator: identity, presets, regions, seeds."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from config import Config
from environment import (
    PRESETS,
    apply_occupancy,
    band_indices,
    edge_kappa_product,
    generate_environment,
    identity_environment,
    inclusive_span,
    toroidal_disk_mask,
)
from grid import Grid
from parameters import init_parameters
from state import GOAL_ELIMINATE, GOAL_REPRODUCE, init_state


GOLDEN_REGIONS = [
    {
        "shape": "rect",
        "r0": 4, "c0": 4, "r1": 12, "c1": 16,
        "kappa_R": 0.0, "kappa_E": 0.0,
        "eta_R": None, "eta_E": None,
        "occupancy": 1.0,
    },
    {
        "shape": "disk",
        "cy": 10.0, "cx": 5.0, "radius": 3.5,
        "kappa_R": 1.0, "kappa_E": 0.0,
        "eta_R": 0.25, "eta_E": 1.0,
    },
    {
        "shape": "band",
        "axis": "v", "center": 0, "width": 2,
        "kappa_R": 0.0, "kappa_E": 0.0,
        "occupancy": 0.0,
    },
]


def _g_cfg(**kw) -> Config:
    defaults = dict(
        N=16,
        environment_heterogeneous=True,
        env_preset="identity",
        env_seed=0,
        n_steps=1,
    )
    defaults.update(kw)
    return Config(**defaults)


def test_flag_off_does_not_construct_env_generator():
    cfg = Config(N=8, environment_heterogeneous=False, env_preset="blobs")
    real_gen = torch.Generator

    def boom(*args, **kwargs):
        raise AssertionError("flag-off generate_environment must not construct a Generator")

    with patch.object(torch, "Generator", side_effect=boom):
        env = generate_environment(cfg)
    assert torch.equal(env.occupancy, torch.ones(8, 8))
    assert torch.equal(env.kappa_R, torch.ones(8, 8))
    assert torch.equal(env.kappa_E, torch.ones(8, 8))
    assert torch.equal(env.eta_scale_R, torch.ones(8, 8))
    assert torch.equal(env.eta_scale_E, torch.ones(8, 8))
    # Restore sanity: identity_environment also uses ones, no Generator.
    with patch.object(torch, "Generator", real_gen):
        ident = identity_environment(8)
    assert torch.equal(env.occupancy, ident.occupancy)
    print("test_flag_off_does_not_construct_env_generator OK")


def test_identity_fields_with_flag_on():
    off = generate_environment(Config(N=8, environment_heterogeneous=False))
    on = generate_environment(_g_cfg(N=8, env_preset="identity"))
    for a, b in zip(off.tensors(), on.tensors()):
        assert torch.equal(a, b), "identity preset must be all-ones"
    print("test_identity_fields_with_flag_on OK")


def test_grid_env_defaults_none():
    gen = torch.Generator().manual_seed(0)
    state = init_state(4, 4, 0.5, gen)
    params = init_parameters(4, 4, 8, 0.01, gen)
    u = torch.randn(4)
    grid = Grid(state=state, params=params, u=u)
    assert grid.env is None
    print("test_grid_env_defaults_none OK")


def test_unknown_preset_raises():
    cfg = _g_cfg(env_preset="not_a_preset")
    try:
        generate_environment(cfg)
        raise AssertionError("expected ValueError for unknown preset")
    except ValueError as e:
        assert "env_preset" in str(e) or "not_a_preset" in str(e)
    print("test_unknown_preset_raises OK")


def test_band_frac_0_is_identity():
    env = generate_environment(_g_cfg(
        env_preset="vertical_band", env_dead_frac=0.0, env_kappa_hi=1.0,
    ))
    assert not (env.kappa_R != 1.0).any()
    assert torch.equal(env.occupancy, torch.ones(16, 16))
    env_h = generate_environment(_g_cfg(
        env_preset="horizontal_band", env_dead_frac=0.0,
    ))
    assert torch.equal(env_h.kappa_R, torch.ones(16, 16))
    env_w = generate_environment(_g_cfg(
        env_preset="torus_wall", env_dead_frac=0.0,
    ))
    assert torch.equal(env_w.kappa_R, torch.ones(16, 16))
    print("test_band_frac_0_is_identity OK")


def test_vertical_band_paints_center_columns():
    N = 16
    cfg = _g_cfg(env_preset="vertical_band", env_dead_frac=0.125, env_kappa_lo=0.0)
    # w = round(0.125*16) = 2. center N//2 = 8. start = 8-1 = 7 → [7, 8]
    env = generate_environment(cfg)
    cols = band_indices(N // 2, 2, N)
    for j in range(N):
        if j in cols:
            assert torch.equal(env.kappa_R[:, j], torch.zeros(N))
        else:
            assert torch.equal(env.kappa_R[:, j], torch.ones(N))
    print("test_vertical_band_paints_center_columns OK")


def test_toroidal_disk_wrap():
    N = 16
    mask = toroidal_disk_mask(N, 0.0, 0.0, 2.0, "cpu")
    assert bool(mask[0, 0])
    assert bool(mask[0, 15])
    assert bool(mask[15, 0])
    assert bool(mask[15, 15])
    assert not bool(mask[8, 8])
    env = generate_environment(_g_cfg(
        N=N, env_preset="center_blob", env_blob_radius=2.0 / N, env_kappa_lo=0.0,
    ))
    # center_blob is at ((N-1)/2, (N-1)/2), not (0,0). Check wrap via blobs.
    print("test_toroidal_disk_wrap OK")


def test_blobs_wrap_at_origin():
    """One blob at (0,0), r=2 paints opposite corners."""
    N = 16
    # Force a single center at (0,0) by constructing via custom disk, and also
    # by monkeypatching randint for blobs.
    cfg = _g_cfg(N=N, env_preset="blobs", env_n_blobs=1, env_blob_radius=2.0 / N)

    def fake_randint(*args, **kwargs):
        return torch.tensor([[0, 0]])

    with patch.object(torch, "randint", side_effect=fake_randint):
        env = generate_environment(cfg)
    assert env.kappa_R[0, 0].item() == 0.0
    assert env.kappa_R[0, 15].item() == 0.0
    assert env.kappa_R[15, 0].item() == 0.0
    assert env.kappa_R[15, 15].item() == 0.0
    assert env.kappa_R[8, 8].item() == 1.0
    print("test_blobs_wrap_at_origin OK")


def test_occupancy_independent_of_kappa_lo():
    """blobs + occupancy_blocks + κ_lo=0.5 zeros occupancy only inside disks."""
    N = 16
    cfg = _g_cfg(
        N=N,
        env_preset="blobs",
        env_n_blobs=3,
        env_blob_radius=0.15,
        env_kappa_lo=0.5,
        env_kappa_hi=1.0,
        env_occupancy_blocks=True,
        env_seed=3,
    )
    env = generate_environment(cfg)
    gen = torch.Generator().manual_seed(3)
    centers = torch.randint(0, N, (3, 2), generator=gen)
    expected = torch.zeros(N, N, dtype=torch.bool)
    r = 0.15 * N
    for k in range(3):
        expected = expected | toroidal_disk_mask(
            N, float(centers[k, 0]), float(centers[k, 1]), r, "cpu",
        )
    assert torch.equal(env.occupancy == 0, expected)
    # Outside disks occupancy is 1 even though κ_lo is 0.5 (not a κ==κ_lo predicate).
    assert (env.occupancy[~expected] == 1).all()
    assert torch.allclose(env.kappa_R[expected], torch.tensor(0.5))
    assert torch.allclose(env.kappa_R[~expected], torch.tensor(1.0))
    print("test_occupancy_independent_of_kappa_lo OK")


def test_blobs_soft_shares_hard_disk_dead_mask():
    cfg_kw = dict(
        N=16, env_preset="blobs_soft", env_n_blobs=3, env_blob_radius=0.15,
        env_kappa_lo=0.5, env_kappa_hi=1.0, env_occupancy_blocks=True, env_seed=3,
    )
    env = generate_environment(_g_cfg(**cfg_kw))
    hard = generate_environment(_g_cfg(**{**cfg_kw, "env_preset": "blobs"}))
    assert torch.equal(env.occupancy, hard.occupancy)
    # Soft κ is not a hard step function.
    interior = env.occupancy == 0
    if interior.any():
        # Occupancy still matches hard disks; kappa is a blend, not forced to κ_lo.
        assert not torch.equal(env.kappa_R, hard.kappa_R)
    print("test_blobs_soft_shares_hard_disk_dead_mask OK")


def test_env_generation_reproducibility():
    a = generate_environment(_g_cfg(env_preset="blobs", env_seed=11, env_n_blobs=3))
    b = generate_environment(_g_cfg(env_preset="blobs", env_seed=11, env_n_blobs=3))
    for t0, t1 in zip(a.tensors(), b.tensors()):
        assert torch.equal(t0, t1)
    c = generate_environment(_g_cfg(env_preset="blobs", env_seed=12, env_n_blobs=3))
    assert not torch.equal(a.kappa_R, c.kappa_R)
    # Same env_seed, different cfg.seed → maps equal.
    d = generate_environment(_g_cfg(
        env_preset="blobs", env_seed=11, env_n_blobs=3, seed=99999,
    ))
    assert torch.equal(a.kappa_R, d.kappa_R)
    print("test_env_generation_reproducibility OK")


def test_split_types_and_affect_flags():
    env = generate_environment(_g_cfg(
        N=8, env_preset="split_types", env_kappa_lo=0.0, env_kappa_hi=1.0,
    ))
    assert torch.equal(env.kappa_R[:, :4], torch.zeros(8, 4))
    assert torch.equal(env.kappa_R[:, 4:], torch.ones(8, 4))
    assert torch.equal(env.kappa_E[:, :4], torch.ones(8, 4))
    assert torch.equal(env.kappa_E[:, 4:], torch.zeros(8, 4))
    muted_r = generate_environment(_g_cfg(
        N=8, env_preset="split_types", env_affect_R=False,
    ))
    assert torch.equal(muted_r.kappa_R, torch.ones(8, 8))
    assert not torch.equal(muted_r.kappa_E, torch.ones(8, 8))
    print("test_split_types_and_affect_flags OK")


def test_learning_hotspot_eta_not_kappa():
    env = generate_environment(_g_cfg(
        N=16, env_preset="learning_hotspot", env_blob_radius=0.15,
        env_eta_lo=0.25, env_eta_hi=1.0,
    ))
    assert torch.equal(env.kappa_R, torch.ones(16, 16))
    assert env.eta_scale_R.max().item() == 1.0
    assert env.eta_scale_R.min().item() == 0.25
    hot = toroidal_disk_mask(16, 15 / 2, 15 / 2, 0.15 * 16, "cpu")
    assert torch.allclose(env.eta_scale_R[hot], torch.tensor(1.0))
    assert torch.allclose(env.eta_scale_R[~hot], torch.tensor(0.25))
    print("test_learning_hotspot_eta_not_kappa OK")


def test_checker_dead_mask():
    N = 16
    env = generate_environment(_g_cfg(N=N, env_preset="checker", env_kappa_lo=0.0))
    b = max(1, N // 8)
    for i in range(N):
        for j in range(N):
            black = ((i // b) + (j // b)) % 2 == 0
            want = 0.0 if black else 1.0
            assert env.kappa_R[i, j].item() == want
    print("test_checker_dead_mask OK")


def test_apply_occupancy_none_is_identity():
    x = torch.ones(4, 4)
    assert torch.equal(apply_occupancy(x, None), x)
    print("test_apply_occupancy_none_is_identity OK")


def test_config_validation_and_roundtrip():
    cfg = _g_cfg(
        env_preset="custom",
        env_regions=GOLDEN_REGIONS,
        env_occupancy_blocks=False,
    )
    tmp = Path(tempfile.mkdtemp()) / "cfg.json"
    cfg.save(tmp)
    loaded = Config.load(tmp)
    assert loaded.environment_heterogeneous is True
    assert loaded.env_preset == "custom"
    assert loaded.env_regions[0]["shape"] == "rect"
    # Pre-G JSON → flag False.
    old = {"N": 8, "d": 4, "hidden": 8, "eta": 0.01, "n_steps": 1, "seed": 1}
    old_path = Path(tempfile.mkdtemp()) / "old.json"
    old_path.write_text(json.dumps(old))
    old_cfg = Config.load(old_path)
    assert old_cfg.environment_heterogeneous is False
    assert old_cfg.env_preset == "identity"
    try:
        Config(env_dead_frac=1.5)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    print("test_config_validation_and_roundtrip OK")


def test_env_regions_golden_n16():
    N = 16
    env = generate_environment(_g_cfg(
        N=N,
        env_preset="custom",
        env_occupancy_blocks=False,
        env_affect_R=True,
        env_affect_E=True,
        env_regions=GOLDEN_REGIONS,
    ))
    rect_cols = set(inclusive_span(4, 16, 16))
    assert rect_cols == {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0}
    rect_rows = set(inclusive_span(4, 12, 16))
    disk = toroidal_disk_mask(N, 10.0, 5.0, 3.5, "cpu")
    band_cols = set(band_indices(0, 2, 16))
    assert band_cols == {15, 0}
    assert bool(disk[10, 5])
    assert not bool(disk[0, 0])

    for i in range(N):
        for j in range(N):
            in_rect = i in rect_rows and j in rect_cols
            in_disk = bool(disk[i, j])
            in_band = j in band_cols
            # Apply in order: rect, disk, band.
            kR, kE, eR, eE, occ = 1.0, 1.0, 1.0, 1.0, 1.0
            if in_rect:
                kR, kE, occ = 0.0, 0.0, 1.0
            if in_disk:
                kR, kE, eR, eE = 1.0, 0.0, 0.25, 1.0
            if in_band:
                kR, kE, occ = 0.0, 0.0, 0.0
            assert abs(env.kappa_R[i, j].item() - kR) < 1e-6, (i, j, env.kappa_R[i, j], kR)
            assert abs(env.kappa_E[i, j].item() - kE) < 1e-6, (i, j, kE)
            assert abs(env.eta_scale_R[i, j].item() - eR) < 1e-6
            assert abs(env.eta_scale_E[i, j].item() - eE) < 1e-6
            assert abs(env.occupancy[i, j].item() - occ) < 1e-6, (i, j, occ)

    # Spot checks from the spec.
    assert env.occupancy[0, 8].item() == 1.0
    assert env.kappa_R[0, 8].item() == 1.0
    assert env.occupancy[0, 0].item() == 0.0
    assert env.kappa_R[0, 0].item() == 0.0
    print("test_env_regions_golden_n16 OK")


def test_custom_bad_shape_raises():
    cfg = _g_cfg(env_preset="custom", env_regions=[{"shape": "triangle"}])
    try:
        generate_environment(cfg)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "env_regions[0]" in str(e)
    print("test_custom_bad_shape_raises OK")


def test_edge_kappa_product_is_kappa_only():
    N = 4
    env = identity_environment(N)
    env.kappa_R[1, 1] = 0.5
    env.kappa_E[1, 1] = 0.5
    gen = torch.Generator().manual_seed(0)
    state = init_state(N, 4, 1.0, gen)
    state.goals[:] = GOAL_REPRODUCE
    prod = edge_kappa_product(state, env)
    assert prod.shape == (N, N, 8)
    # Neighbour of (1,0) to the east is (1,1) with κ=0.5; (1,0) has κ=1
    # → product 0.5. No ρ or x in this helper.
    state.rho[:] = 0.0
    state.x[:] = 0.0
    prod2 = edge_kappa_product(state, env)
    assert torch.equal(prod, prod2)
    print("test_edge_kappa_product_is_kappa_only OK")


def test_presets_tuple_complete():
    assert "identity" in PRESETS
    assert "blobs" in PRESETS
    assert "custom" in PRESETS
    print("test_presets_tuple_complete OK")


if __name__ == "__main__":
    test_flag_off_does_not_construct_env_generator()
    test_identity_fields_with_flag_on()
    test_grid_env_defaults_none()
    test_unknown_preset_raises()
    test_band_frac_0_is_identity()
    test_vertical_band_paints_center_columns()
    test_toroidal_disk_wrap()
    test_blobs_wrap_at_origin()
    test_occupancy_independent_of_kappa_lo()
    test_blobs_soft_shares_hard_disk_dead_mask()
    test_env_generation_reproducibility()
    test_split_types_and_affect_flags()
    test_learning_hotspot_eta_not_kappa()
    test_checker_dead_mask()
    test_apply_occupancy_none_is_identity()
    test_config_validation_and_roundtrip()
    test_env_regions_golden_n16()
    test_custom_bad_shape_raises()
    test_edge_kappa_product_is_kappa_only()
    test_presets_tuple_complete()
    print("\nAll environment generator tests passed.")
