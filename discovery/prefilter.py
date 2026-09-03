"""Cheap metric-based reject so we only spend VLM calls on non-obvious runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Tunable thresholds — start conservative (reject only clear trash).
EXTINCTION_FRACTION = 0.25  # if dead by this fraction of run and stays dead
STATIC_TAIL_FRAC = 0.25  # last this fraction of steps
STATIC_ALIVE_RANGE = 2  # max(alive)-min(alive) over tail must exceed this
SATURATION_FRAC = 0.95  # mean alive / N^2 over second half
SATURATION_MIN_STEPS = 50


@dataclass
class PrefilterResult:
    """Cheap reject/pass decision plus a short reason string."""
    passed: bool
    reason: str


def metrics_from_trajectory(traj_path: Path) -> dict[str, np.ndarray]:
    """Load alive / repro / elim / step arrays from a trajectory.npz."""
    traj = np.load(traj_path)
    return {
        "alive": traj["alive_count"].astype(np.int32),
        "repro": traj["reproducer_alive"].astype(np.int32),
        "elim": traj["eliminator_alive"].astype(np.int32),
        "steps": traj["step"].astype(np.int32),
    }


def prefilter(
    metrics: dict[str, np.ndarray],
    *,
    N: int,
) -> PrefilterResult:
    """Return whether a run is worth sending to the VLM."""
    alive = metrics["alive"]
    t = len(alive)
    if t < 2:
        return PrefilterResult(False, "trajectory too short")

    n_cells = N * N

    # Extinction: hits zero early and stays zero.
    ext_idx = max(1, int(t * EXTINCTION_FRACTION))
    if alive[-1] == 0:
        # Find first permanent zero run to the end.
        zeros = np.where(alive == 0)[0]
        if len(zeros) > 0:
            first_zero = int(zeros[0])
            if first_zero <= ext_idx and np.all(alive[first_zero:] == 0):
                return PrefilterResult(
                    False, f"extinction by step {first_zero}"
                )
        if np.all(alive[ext_idx:] == 0):
            return PrefilterResult(False, "extinct for most of run")

    # Static tail: population barely moves late in the run.
    tail_start = max(0, t - max(5, int(t * STATIC_TAIL_FRAC)))
    tail = alive[tail_start:]
    if len(tail) >= 5 and (int(tail.max()) - int(tail.min())) <= STATIC_ALIVE_RANGE:
        # Allow static only if there's some life and some structure — still reject
        # fully frozen counts (including all-dead already handled).
        return PrefilterResult(
            False,
            f"static population tail (alive range {int(tail.max()) - int(tail.min())})",
        )

    # Near-full saturation freeze.
    if t >= SATURATION_MIN_STEPS:
        half = alive[t // 2 :]
        mean_frac = float(half.mean()) / n_cells
        if mean_frac >= SATURATION_FRAC:
            half_range = int(half.max()) - int(half.min())
            if half_range <= STATIC_ALIVE_RANGE:
                return PrefilterResult(
                    False, f"saturated freeze (mean alive frac {mean_frac:.2f})"
                )

    return PrefilterResult(True, "ok")
