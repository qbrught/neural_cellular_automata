"""Run a discovery trial and render VLM evidence (summary.png)."""

from __future__ import annotations

from pathlib import Path

from config import Config
from simulate import run as run_simulation
from visualise import render_summary


def run_trial(cfg: Config, trial_dir: Path, *, verbose: bool = False) -> Path:
    """Run sim with learning on, write trajectory + summary.png under trial_dir."""
    trial_dir = Path(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)

    cfg.learn = True
    cfg.save_state_vectors = False
    cfg.__post_init__()

    run_simulation(cfg, output_dir=trial_dir, verbose=verbose)
    render_summary(trial_dir / "trajectory.npz", trial_dir / "summary.png")
    return trial_dir
