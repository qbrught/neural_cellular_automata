"""Sparse grid frames and off-vs-on montages for the thesis pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualise import (
    _CMAP,
    _NORM,
    _composite_env_rgb,
    _display_grid,
    _legend_handles,
)


def load_frames(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _has_env(frames: dict[str, np.ndarray]) -> bool:
    return "kappa_R" in frames and "occupancy" in frames


def render_frame_rgb(frames: dict[str, np.ndarray], index: int) -> np.ndarray:
    """(N,N,3) float RGB in 0..1, or 2D int display if no env overlay."""
    x = frames["x"][index]
    g = frames["goals"][index]
    display = _display_grid(x, g)
    if not _has_env(frames):
        return display
    eta_r = frames.get("eta_scale_R")
    eta_e = frames.get("eta_scale_E")
    return _composite_env_rgb(
        display,
        frames["occupancy"],
        frames["kappa_R"],
        frames["kappa_E"],
        eta_r,
        eta_e,
    )


def save_frame_png(
    frames: dict[str, np.ndarray],
    index: int,
    out_path: Path,
    *,
    title: str | None = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = render_frame_rgb(frames, index)
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    if img.ndim == 2:
        ax.imshow(img, cmap=_CMAP, norm=_NORM, interpolation="nearest")
    else:
        ax.imshow(img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    step = int(frames["steps"][index])
    ax.set_title(title or f"t={step}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def render_run_frames(run_dir: Path, out_dir: Path | None = None) -> list[Path]:
    """Write t0 / mid / late PNGs next to frames.npz."""
    run_dir = Path(run_dir)
    src = run_dir / "frames.npz"
    if not src.is_file():
        return []
    frames = load_frames(src)
    dest = Path(out_dir) if out_dir is not None else run_dir / "frames"
    dest.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    n = int(frames["x"].shape[0])
    for i in range(n):
        step = int(frames["steps"][i])
        paths.append(save_frame_png(frames, i, dest / f"t{step}.png"))
    return paths


def render_pair_montage(
    off_frames: dict[str, np.ndarray],
    on_frames: dict[str, np.ndarray],
    out_path: Path,
    *,
    off_label: str,
    on_label: str,
    seed: int,
) -> Path:
    """Rows = off/on, columns = snapshot times (aligned by min count)."""
    n = min(int(off_frames["x"].shape[0]), int(on_frames["x"].shape[0]))
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 6.2))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    rows = ((off_frames, off_label), (on_frames, on_label))
    for r, (fr, lab) in enumerate(rows):
        for c in range(n):
            ax = axes[r, c]
            img = render_frame_rgb(fr, c)
            if img.ndim == 2:
                ax.imshow(img, cmap=_CMAP, norm=_NORM, interpolation="nearest")
            else:
                ax.imshow(img, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            step = int(fr["steps"][c])
            if r == 0:
                ax.set_title(f"t={step}", fontsize=10)
            if c == 0:
                ax.set_ylabel(lab, fontsize=10)
    fig.suptitle(f"{off_label} vs {on_label} · seed {seed}", fontsize=11)
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_comparison_frames(
    results: list[dict[str, Any]],
    out_dir: Path,
    *,
    off: str,
    on: str,
    visual_seeds: tuple[int, ...] | list[int],
) -> list[Path]:
    """Montages for each visual seed that has frames on both arms."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by: dict[tuple[str, int], Path] = {}
    for r in results:
        fp = r.get("frames_path")
        if fp and Path(fp).is_file():
            by[(str(r["version_id"]), int(r["seed"]))] = Path(fp)
    paths: list[Path] = []
    for seed in visual_seeds:
        po = by.get((off, int(seed)))
        pn = by.get((on, int(seed)))
        if po is None or pn is None:
            continue
        dest = out_dir / f"seed_{seed}_montage.png"
        paths.append(
            render_pair_montage(
                load_frames(po),
                load_frames(pn),
                dest,
                off_label=off,
                on_label=on,
                seed=int(seed),
            )
        )
    return paths
