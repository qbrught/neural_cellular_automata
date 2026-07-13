"""Visualise an NCSA trajectory.

Reads a trajectory.npz file (and optionally config.json) and produces:
  - summary.svg : final-state grid + per-step time-series
  - animation.mp4 : animated grid (if requested)

Colour encoding for the grid:
  dark grey   : dead
  green       : alive reproducer
  red         : alive eliminator

The goal field is per-cell and immutable across steps, so we can colour
every alive cell by (alive flag x goal type) using a single static palette.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, colors

from state import GOAL_ELIMINATE, GOAL_REPRODUCE

# Cell display state codes:
#   0 = dead
#   1 = alive reproducer
#   2 = alive eliminator
DEAD = 0
ALIVE_REPRO = 1
ALIVE_ELIM = 2

_CMAP = colors.ListedColormap(["#222222", "#3ec96b", "#e0492f"])
_NORM = colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], _CMAP.N)
# --- Poster styling --------------------------------------------------
# A1 = 23.4 x 33.1 in. Each figure is sized at its intended physical
# footprint on the sheet, so a point below is a literal point on paper.
A1_WIDTH_IN = 23.4
A1_HEIGHT_IN = 33.1

POSTER_RC = {
    "font.size": 22,
    "axes.titlesize": 30,
    "axes.labelsize": 26,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 22,
    "lines.linewidth": 3.0,
    "axes.linewidth": 1.5,
    "grid.linewidth": 1.2,
}
def _legend_handles() -> list:
    """Handles for the goal/alive colour legend (shared across renderers)."""
    return [
        plt.Rectangle((0, 0), 1, 1, color="#3ec96b", label="alive reproducer"),
        plt.Rectangle((0, 0), 1, 1, color="#e0492f", label="alive eliminator"),
        plt.Rectangle((0, 0), 1, 1, color="#222222", label="dead"),
    ]
    
def render_alive_count(
    traj_path: Path,
    out_path: Path | None = None,
    figsize: tuple[float, float] = (18.0, 6.0),
    per_goal: bool = True,
) -> Path:
    """Render just the alive-count trajectory as a wide SVG (3:1)."""
    traj_path = Path(traj_path)
    traj = np.load(traj_path)
    if out_path is None:
        out_path = traj_path.with_name("alive_count.svg")

    steps = traj["step"]

    with plt.rc_context(POSTER_RC):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(steps, traj["alive_count"], color="black", label="total")
        if per_goal:
            ax.plot(steps, traj["reproducer_alive"],
                    color="#3ec96b", label="reproducers")
            ax.plot(steps, traj["eliminator_alive"],
                    color="#e0492f", label="eliminators")
        ax.set_xlabel("step")
        ax.set_ylabel("alive count")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(out_path, format="svg")
    plt.close(fig)
    return out_path

def render_final_grid(
    traj_path: Path,
    out_path: Path | None = None,
    size: float = 6.0,
) -> Path:
    """Render just the final grid state as a square SVG (no axes/legend/title).

    Args:
        traj_path: path to trajectory.npz
        out_path: output path (default: final_grid.svg next to traj)
        size: figure side length in inches (square)
    """
    traj_path = Path(traj_path)
    traj = np.load(traj_path)
    if out_path is None:
        out_path = traj_path.with_name("final_grid.svg")

    x = traj["x"]
    goals = traj["goals"]
    final_display = _display_grid(x[-1], goals)

    fig = plt.figure(figsize=(size, size))
    ax = fig.add_axes([0, 0, 1, 1])  # axes fill the entire figure — no margins
    ax.imshow(
        final_display, cmap=_CMAP, norm=_NORM,
        interpolation="nearest", aspect="auto",
    )
    ax.set_axis_off()
    # ax.legend(
    #     handles=_legend_handles(),
    #     loc="upper right",
    #     fontsize=10,
    #     framealpha=0.9,
    #     borderaxespad=1.0,  # inset from the frame edge
    # )
    fig.savefig(out_path, format="svg")  # no bbox_inches="tight" — keep it square
    plt.close(fig)
    return out_path

def _display_grid(x: np.ndarray, goals: np.ndarray) -> np.ndarray:
    """Combine (alive, goal) into a single int grid for display."""
    out = np.zeros_like(x, dtype=np.uint8)
    alive = x.astype(bool)
    out[alive & (goals == GOAL_REPRODUCE)] = ALIVE_REPRO
    out[alive & (goals == GOAL_ELIMINATE)] = ALIVE_ELIM
    return out


def render_summary(traj_path: Path, out_path: Path | None = None) -> Path:
    """Render a 4-panel SVG: final grid, alive count, per-goal count, loss curves."""
    traj_path = Path(traj_path)
    traj = np.load(traj_path)
    if out_path is None:
        out_path = traj_path.with_name("summary.svg")

    x = traj["x"]
    goals = traj["goals"]
    steps = traj["step"]
    n_repro_alive = traj["reproducer_alive"]
    n_elim_alive = traj["eliminator_alive"]
    total_alive = traj["alive_count"]
    loss_r = traj["loss_reproduce_mean"]
    loss_e = traj["loss_eliminate_mean"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # --- Final grid ---
    ax = axes[0, 0]
    final_display = _display_grid(x[-1], goals)
    ax.imshow(final_display, cmap=_CMAP, norm=_NORM, interpolation="nearest")
    ax.set_title(f"Final state (step {steps[-1]})")
    ax.set_xticks([])
    ax.set_yticks([])
    # Manual legend.
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color="#3ec96b", label="alive reproducer"),
        plt.Rectangle((0, 0), 1, 1, color="#e0492f", label="alive eliminator"),
        plt.Rectangle((0, 0), 1, 1, color="#222222", label="dead"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              framealpha=0.9)

    # --- Total alive ---
    ax = axes[0, 1]
    ax.plot(steps, total_alive, color="black", linewidth=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("alive count")
    ax.set_title("Total alive")
    ax.grid(True, alpha=0.3)

    # --- Per-goal alive ---
    ax = axes[1, 0]
    ax.plot(steps, n_repro_alive, color="#3ec96b", label="reproducers", linewidth=1.2)
    ax.plot(steps, n_elim_alive, color="#e0492f", label="eliminators", linewidth=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("alive count")
    ax.set_title("Alive by goal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Losses ---
    ax = axes[1, 1]
    # Skip step 0 which has NaN.
    valid = ~np.isnan(loss_r) | ~np.isnan(loss_e)
    if valid.any():
        ax.plot(steps[~np.isnan(loss_r)], loss_r[~np.isnan(loss_r)],
                color="#3ec96b", label="reproducer mean loss", linewidth=1.2)
        ax.plot(steps[~np.isnan(loss_e)], loss_e[~np.isnan(loss_e)],
                color="#e0492f", label="eliminator mean loss", linewidth=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Mean per-cell loss by goal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)

    fig.suptitle(f"NCSA trajectory: {traj_path.parent.name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return out_path


def render_animation(
    traj_path: Path,
    out_path: Path | None = None,
    fps: int = 15,
    stride: int = 1,
) -> Path:
    """Render an MP4 (or GIF, depending on extension) animation of the grid.

    Note: this stays a video format (mp4/gif) rather than SVG, since SVG
    has no native animation-as-video output comparable to ffmpeg/pillow
    writers. Pass a .gif out_path if you want a looping image instead.

    Args:
        traj_path: path to trajectory.npz
        out_path: output path (default: animation.mp4 next to traj)
        fps: frames per second
        stride: take every Nth step (1 = all frames)
    """
    traj_path = Path(traj_path)
    traj = np.load(traj_path)
    if out_path is None:
        out_path = traj_path.with_name("animation.mp4")

    x = traj["x"][::stride]
    steps = traj["step"][::stride]
    goals = traj["goals"]
    n_repro = traj["reproducer_alive"][::stride]
    n_elim = traj["eliminator_alive"][::stride]

    fig, (ax_grid, ax_curve) = plt.subplots(
        1, 2, figsize=(11, 5),
        gridspec_kw={"width_ratios": [1, 1.4]},
    )

    im = ax_grid.imshow(
        _display_grid(x[0], goals),
        cmap=_CMAP, norm=_NORM, interpolation="nearest",
    )
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    title_grid = ax_grid.set_title(f"step {steps[0]}")

    ax_curve.plot(steps, n_repro, color="#3ec96b",
                  label="reproducers", linewidth=1.2)
    ax_curve.plot(steps, n_elim, color="#e0492f",
                  label="eliminators", linewidth=1.2)
    ax_curve.set_xlabel("step")
    ax_curve.set_ylabel("alive count")
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(loc="upper right")
    cursor = ax_curve.axvline(steps[0], color="black", linewidth=0.8, alpha=0.6)

    def update(frame_idx: int):
        im.set_data(_display_grid(x[frame_idx], goals))
        title_grid.set_text(f"step {steps[frame_idx]}")
        cursor.set_xdata([steps[frame_idx]])
        return im, title_grid, cursor

    anim = animation.FuncAnimation(
        fig, update, frames=len(x), interval=1000 / fps, blit=False,
    )

    if str(out_path).endswith(".gif"):
        anim.save(out_path, writer="pillow", fps=fps)
    else:
        anim.save(out_path, writer="ffmpeg", fps=fps)
    plt.close(fig)
    return out_path