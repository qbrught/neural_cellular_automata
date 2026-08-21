"""Experiment G suite wiring: VersionSpec G / G_learn pins and apply(A) clears."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from research.versions import get_version, parse_version_list


def test_aliases():
    assert get_version("G") is get_version("A_env")
    assert get_version("A_hetero").id == "G"
    assert get_version("A+G").id == "G"
    assert get_version("A_env_learn").id == "G_learn"
    print("test_aliases OK")


def test_apply_g_pins_blobs_knobs():
    base = Config(env_n_blobs=0, env_preset="identity", env_eta_lo=0.0)
    out = get_version("G").apply(base)
    assert out.environment_heterogeneous is True
    assert out.env_preset == "blobs"
    assert out.env_n_blobs == 3
    assert out.env_blob_radius == 0.15
    assert out.env_kappa_lo == 0.0
    assert out.env_kappa_hi == 1.0
    assert out.env_occupancy_blocks is False
    assert out.env_affect_R is True
    assert out.env_affect_E is True
    assert out.typed_votes is True
    assert out.coexistence_pressure is False
    assert out.env_eta_lo == 1.0
    assert out.env_eta_hi == 1.0
    # env_seed is not pinned
    assert out.env_seed == base.env_seed
    print("test_apply_g_pins_blobs_knobs OK")


def test_apply_a_clears_g():
    g_on = Config(
        environment_heterogeneous=True,
        env_preset="blobs",
        env_n_blobs=7,
    )
    out = get_version("A").apply(g_on)
    assert out.environment_heterogeneous is False
    assert out.env_preset == "identity"
    print("test_apply_a_clears_g OK")


def test_apply_g_learn_pins_eta():
    base = Config(env_eta_lo=0.0, env_eta_hi=4.0, env_preset="blobs")
    out = get_version("G_learn").apply(base)
    assert out.environment_heterogeneous is True
    assert out.env_preset == "learning_hotspot"
    assert out.env_eta_lo == 0.25
    assert out.env_eta_hi == 1.0
    print("test_apply_g_learn_pins_eta OK")


def test_all_includes_g():
    ids = [v.id for v in parse_version_list("all")]
    assert "G" in ids
    assert "G_learn" in ids
    print("test_all_includes_g OK")


def test_visualise_overlay_on_g_run():
    import tempfile
    from simulate import run
    from visualise import render_summary

    cfg = Config(
        N=8, d=4, hidden=8, n_steps=2, seed=1, learn=False,
        save_state_vectors=False, output_dir=tempfile.mkdtemp(),
        environment_heterogeneous=True, env_preset="vertical_band",
        env_dead_frac=0.25, env_kappa_lo=0.0,
    )
    out = run(cfg, verbose=False)
    svg = render_summary(out / "trajectory.npz")
    assert svg.exists()
    text = svg.read_text()
    assert "low κ" in text or "3b82f6" in text
    print("test_visualise_overlay_on_g_run OK")


def test_discovery_resamples_env_seed_under_g():
    import random
    from discovery.sample import sample_config

    rng = random.Random(0)
    a = sample_config(rng, n_steps=10, N=8, version="G")
    b = sample_config(rng, n_steps=10, N=8, version="G")
    assert a.environment_heterogeneous is True
    assert a.env_preset == "blobs"
    assert a.env_seed != b.env_seed
    rng = random.Random(1)
    c = sample_config(rng, n_steps=10, N=8, version="A")
    assert c.environment_heterogeneous is False
    print("test_discovery_resamples_env_seed_under_g OK")


if __name__ == "__main__":
    test_aliases()
    test_apply_g_pins_blobs_knobs()
    test_apply_a_clears_g()
    test_apply_g_learn_pins_eta()
    test_all_includes_g()
    test_visualise_overlay_on_g_run()
    test_discovery_resamples_env_seed_under_g()
    print("\nAll environment suite tests passed.")
