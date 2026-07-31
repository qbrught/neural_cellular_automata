"""Test reproducibility: same seed -> same trajectory, different seed -> different."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from config import Config
from simulate import run


def _short_cfg(seed: int) -> Config:
    return Config(
        N=8, d=4, hidden=8, n_steps=30, seed=seed, eta=0.01,
        output_dir=tempfile.mkdtemp(),
    )


def test_same_seed_same_trajectory():
    cfg_a = _short_cfg(seed=123)
    cfg_b = _short_cfg(seed=123)

    out_a = run(cfg_a, verbose=False)
    out_b = run(cfg_b, verbose=False)

    traj_a = np.load(out_a / "trajectory.npz")
    traj_b = np.load(out_b / "trajectory.npz")

    # All array fields should match exactly.
    for key in ["x", "step", "alive_count", "reproducer_alive", "eliminator_alive",
                "goals", "rho"]:
        assert np.array_equal(traj_a[key], traj_b[key]), (
            f"Reproducibility broken on '{key}': arrays differ"
        )

    # Float arrays compared with strict equality (same seed -> identical float ops).
    # NaNs make this awkward (NaN != NaN) — guard with isnan handling.
    for key in ["s", "h", "loss_reproduce_mean", "loss_eliminate_mean", "loss_total"]:
        a = traj_a[key]
        b = traj_b[key]
        nan_a = np.isnan(a)
        nan_b = np.isnan(b)
        assert np.array_equal(nan_a, nan_b), f"NaN pattern differs on '{key}'"
        assert np.array_equal(a[~nan_a], b[~nan_b]), (
            f"Reproducibility broken on '{key}': float values differ"
        )

    print(f"  Trajectory T={traj_a['x'].shape[0]} steps, identical bit-for-bit")
    print("test_same_seed_same_trajectory OK")


def test_different_seed_different_trajectory():
    cfg_a = _short_cfg(seed=1)
    cfg_b = _short_cfg(seed=2)

    out_a = run(cfg_a, verbose=False)
    out_b = run(cfg_b, verbose=False)

    traj_a = np.load(out_a / "trajectory.npz")
    traj_b = np.load(out_b / "trajectory.npz")

    # The initial state alone should differ (goal assignment is seeded).
    assert not np.array_equal(traj_a["goals"], traj_b["goals"]), (
        "Different seeds produced identical goal assignments — RNG not actually used"
    )
    # And the trajectories of x should differ somewhere.
    assert not np.array_equal(traj_a["x"], traj_b["x"]), (
        "Different seeds produced identical trajectories"
    )
    print("test_different_seed_different_trajectory OK")


def test_trajectory_files_exist_and_load():
    cfg = _short_cfg(seed=99)
    out = run(cfg, verbose=False)

    assert (out / "config.json").exists(), "config.json missing"
    assert (out / "trajectory.npz").exists(), "trajectory.npz missing"
    assert (out / "params_final.pt").exists(), "params_final.pt missing"

    # Config roundtrips.
    cfg_loaded = Config.load(out / "config.json")
    assert cfg_loaded.seed == cfg.seed
    assert cfg_loaded.n_steps == cfg.n_steps

    # Trajectory has expected length (n_steps + 1 because we record t=0).
    traj = np.load(out / "trajectory.npz")
    assert traj["x"].shape[0] == cfg.n_steps + 1
    assert traj["x"].shape[1:] == (cfg.N, cfg.N)
    # Goals are per-step (T, N, N); legacy 2D is also acceptable to loaders.
    assert traj["goals"].ndim in (2, 3)
    if traj["goals"].ndim == 3:
        assert traj["goals"].shape == (cfg.n_steps + 1, cfg.N, cfg.N)
    else:
        assert traj["goals"].shape == (cfg.N, cfg.N)
    print("test_trajectory_files_exist_and_load OK")


def test_initial_snapshot_has_nan_loss():
    """Step 0 has no learning yet, so its loss fields should be NaN."""
    cfg = _short_cfg(seed=7)
    out = run(cfg, verbose=False)
    traj = np.load(out / "trajectory.npz")
    assert np.isnan(traj["loss_reproduce_mean"][0])
    assert np.isnan(traj["loss_eliminate_mean"][0])
    # Step 1 onwards should have real values (assuming any reproducer/eliminator alive).
    # We can't be sure both goals have any alive cell, but at least one should.
    assert not (np.isnan(traj["loss_reproduce_mean"][1])
                and np.isnan(traj["loss_eliminate_mean"][1]))
    print("test_initial_snapshot_has_nan_loss OK")


if __name__ == "__main__":
    test_same_seed_same_trajectory()
    test_different_seed_different_trajectory()
    test_trajectory_files_exist_and_load()
    test_initial_snapshot_has_nan_loss()
    print("\nAll reproducibility tests passed.")
