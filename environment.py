"""Frozen spatial environment overlay on the regular N×N torus (Experiment G).

Irregularity is a handful of (N, N) tensors generated once from a recipe
(preset + knobs + env_seed, optional env_regions). The lattice, Moore-8
neighbourhood, and gather_neighbours are unchanged.

Maps:
  occupancy     — {0,1} habitable mask
  kappa_R/E     — transfer conductivity, type-conditioned
  eta_scale_R/E — per-cell SGD scale, type-conditioned

Flag-off (`environment_heterogeneous=False`) returns identity via torch.ones
and never constructs a Generator, so the run-seed RNG stream is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from config import Config
from grid import gather_neighbours
from state import GOAL_REPRODUCE, State

PRESETS: tuple[str, ...] = (
    "identity",
    "vertical_band",
    "horizontal_band",
    "torus_wall",
    "center_blob",
    "blobs",
    "blobs_soft",
    "checker",
    "split_types",
    "learning_hotspot",
    "custom",
)

# Which Config knobs each preset reads (others are stored but unused).
PRESET_KNOBS: dict[str, tuple[str, ...]] = {
    "identity": (),
    "vertical_band": (
        "env_dead_frac", "env_kappa_lo", "env_kappa_hi",
        "env_occupancy_blocks", "env_affect_R", "env_affect_E",
    ),
    "horizontal_band": (
        "env_dead_frac", "env_kappa_lo", "env_kappa_hi",
        "env_occupancy_blocks", "env_affect_R", "env_affect_E",
    ),
    "torus_wall": (
        "env_dead_frac", "env_kappa_lo", "env_kappa_hi",
        "env_occupancy_blocks", "env_affect_R", "env_affect_E",
    ),
    "center_blob": (
        "env_blob_radius", "env_kappa_lo", "env_kappa_hi",
        "env_occupancy_blocks", "env_affect_R", "env_affect_E",
    ),
    "blobs": (
        "env_n_blobs", "env_blob_radius", "env_kappa_lo", "env_kappa_hi",
        "env_occupancy_blocks", "env_affect_R", "env_affect_E",
    ),
    "blobs_soft": (
        "env_n_blobs", "env_blob_radius", "env_kappa_lo", "env_kappa_hi",
        "env_occupancy_blocks", "env_affect_R", "env_affect_E",
    ),
    "checker": (
        "env_kappa_lo", "env_kappa_hi",
        "env_occupancy_blocks", "env_affect_R", "env_affect_E",
    ),
    "split_types": (
        "env_kappa_lo", "env_kappa_hi", "env_affect_R", "env_affect_E",
    ),
    "learning_hotspot": (
        "env_blob_radius", "env_eta_lo", "env_eta_hi",
        "env_affect_R", "env_affect_E",
    ),
    "custom": ("env_regions", "env_occupancy_blocks", "env_affect_R", "env_affect_E"),
}

@dataclass
class Environment:
    occupancy: Tensor       # (N, N) float {0,1}, requires_grad=False
    kappa_R: Tensor
    kappa_E: Tensor
    eta_scale_R: Tensor
    eta_scale_E: Tensor
    extras: dict[str, Tensor] = field(default_factory=dict)

    def tensors(self) -> list[Tensor]:
        return [
            self.occupancy, self.kappa_R, self.kappa_E,
            self.eta_scale_R, self.eta_scale_E, *self.extras.values(),
        ]


def _ones(N: int, device: str, dtype: torch.dtype) -> Tensor:
    t = torch.ones(N, N, device=device, dtype=dtype)
    t.requires_grad_(False)
    return t


def _false_mask(N: int, device: str) -> Tensor:
    return torch.zeros(N, N, device=device, dtype=torch.bool)


def identity_environment(
    N: int, device: str = "cpu", dtype: torch.dtype = torch.float32,
) -> Environment:
    """torch.ones only. No Generator, no rand, no manual_seed."""
    ones = _ones(N, device, dtype)
    return Environment(
        occupancy=ones,
        kappa_R=ones.clone(),
        kappa_E=ones.clone(),
        eta_scale_R=ones.clone(),
        eta_scale_E=ones.clone(),
    )


def toroidal_delta(coord: Tensor, center: float, N: int) -> Tensor:
    """Wrapped |coord - center| on the torus: min(d % N, N - d % N)."""
    d = (coord.float() - float(center)) % N
    return torch.minimum(d, N - d)


def toroidal_disk_mask(
    N: int, cy: float, cx: float, radius: float, device: str,
) -> Tensor:
    """Boolean (N,N). Inclusive: dx^2 + dy^2 <= radius^2. Wrap-aware."""
    ii = torch.arange(N, device=device).view(N, 1).expand(N, N)
    jj = torch.arange(N, device=device).view(1, N).expand(N, N)
    dy = toroidal_delta(ii, cy, N)
    dx = toroidal_delta(jj, cx, N)
    return (dy * dy + dx * dx) <= (float(radius) ** 2)


def band_indices(center: int, width: int, N: int) -> list[int]:
    """`width` consecutive cells centered on `center`, toroidal.

    width <= 0 → empty. Example: center=0, width=2, N=16 → [15, 0].
    """
    if width <= 0:
        return []
    start = int(center) - (width // 2)
    return [(start + k) % N for k in range(int(width))]


def inclusive_span(a: int, b: int, N: int) -> list[int]:
    """Inclusive [a, b]. If b >= a: range(a, b+1) then % N.

    If b < a: wrap from a to N-1 then 0 to b.
    """
    if b >= a:
        return [i % N for i in range(int(a), int(b) + 1)]
    return [i % N for i in list(range(int(a), N)) + list(range(0, int(b) + 1))]


def type_select(field_R: Tensor, field_E: Tensor, goals: Tensor) -> Tensor:
    """(N,N) = field_R * 1_{g=R} + field_E * 1_{g=E}. No grad through goals."""
    is_r = (goals == GOAL_REPRODUCE).to(dtype=field_R.dtype)
    return field_R * is_r + field_E * (1.0 - is_r)


def edge_kappa_product(state: State, env: Environment) -> Tensor:
    """(N, N, 8) = κ_cell.unsqueeze(-1) * gather_neighbours(κ_cell).

    κ-only. Commutative. The single helper used by incoming and outgoing.
    """
    kappa_cell = type_select(env.kappa_R, env.kappa_E, state.goals)
    return kappa_cell.unsqueeze(-1) * gather_neighbours(kappa_cell)


def eta_map(state: State, cfg: Config, env: Environment | None) -> Tensor:
    """(N, N) per-cell learning rate. env=None → fill(cfg.eta)."""
    if env is None:
        return torch.full(
            (cfg.N, cfg.N),
            float(cfg.eta),
            device=state.x.device,
            dtype=state.x.dtype,
        )
    scale = type_select(env.eta_scale_R, env.eta_scale_E, state.goals)
    return float(cfg.eta) * scale


def apply_occupancy(x: Tensor, env: Environment | None) -> Tensor:
    """x * occupancy, or x if env is None."""
    if env is None:
        return x
    return x * env.occupancy


def _band_mask(N: int, indices: list[int], axis: str, device: str) -> Tensor:
    mask = _false_mask(N, device)
    if not indices:
        return mask
    idx = torch.tensor(indices, device=device, dtype=torch.long)
    if axis == "v":
        mask[:, idx] = True
    else:
        mask[idx, :] = True
    return mask


def _kappa_from_mask(
    dead_mask: Tensor, kappa_lo: float, kappa_hi: float, dtype: torch.dtype,
) -> Tensor:
    lo = torch.tensor(kappa_lo, device=dead_mask.device, dtype=dtype)
    hi = torch.tensor(kappa_hi, device=dead_mask.device, dtype=dtype)
    return torch.where(dead_mask, lo, hi)


def _blob_centers(cfg: Config, gen: torch.Generator) -> Tensor:
    n = int(cfg.env_n_blobs)
    return torch.randint(0, cfg.N, (n, 2), generator=gen, device="cpu")


def _union_disks(
    N: int, centers: Tensor, radius: float, device: str,
) -> Tensor:
    mask = _false_mask(N, device)
    if centers.numel() == 0:
        return mask
    for k in range(centers.shape[0]):
        cy = float(centers[k, 0].item())
        cx = float(centers[k, 1].item())
        mask = mask | toroidal_disk_mask(N, cy, cx, radius, device)
    return mask


def _soft_blob_kappa(
    N: int,
    centers: Tensor,
    sigma: float,
    kappa_lo: float,
    kappa_hi: float,
    device: str,
    dtype: torch.dtype,
) -> Tensor:
    """κ = κ_hi - (κ_hi-κ_lo) * clip(Σ_k exp(-(dx²+dy²)/(2σ²)), 0, 1)."""
    acc = torch.zeros(N, N, device=device, dtype=dtype)
    if centers.numel() == 0 or sigma <= 0:
        return torch.full((N, N), kappa_hi, device=device, dtype=dtype)
    ii = torch.arange(N, device=device).view(N, 1).expand(N, N)
    jj = torch.arange(N, device=device).view(1, N).expand(N, N)
    two_sigma2 = 2.0 * (sigma ** 2)
    for k in range(centers.shape[0]):
        cy = float(centers[k, 0].item())
        cx = float(centers[k, 1].item())
        dy = toroidal_delta(ii, cy, N)
        dx = toroidal_delta(jj, cx, N)
        acc = acc + torch.exp(-(dx * dx + dy * dy) / two_sigma2)
    mix = acc.clamp(0.0, 1.0)
    return kappa_hi - (kappa_hi - kappa_lo) * mix


def _region_mask(region: dict, N: int, device: str, index: int) -> Tensor:
    shape = region.get("shape")
    if shape == "rect":
        for key in ("r0", "c0", "r1", "c1"):
            if key not in region:
                raise ValueError(f"env_regions[{index}]: rect missing {key!r}")
        rows = inclusive_span(int(region["r0"]), int(region["r1"]), N)
        cols = inclusive_span(int(region["c0"]), int(region["c1"]), N)
        mask = _false_mask(N, device)
        if rows and cols:
            r_idx = torch.tensor(rows, device=device, dtype=torch.long)
            c_idx = torch.tensor(cols, device=device, dtype=torch.long)
            # Cartesian product of inclusive spans.
            mask[r_idx.unsqueeze(1), c_idx.unsqueeze(0)] = True
        return mask
    if shape == "disk":
        for key in ("cy", "cx", "radius"):
            if key not in region:
                raise ValueError(f"env_regions[{index}]: disk missing {key!r}")
        radius = float(region["radius"])
        if radius <= 0:
            raise ValueError(f"env_regions[{index}]: disk radius must be > 0")
        return toroidal_disk_mask(
            N, float(region["cy"]), float(region["cx"]), radius, device,
        )
    if shape == "band":
        for key in ("axis", "center", "width"):
            if key not in region:
                raise ValueError(f"env_regions[{index}]: band missing {key!r}")
        axis = region["axis"]
        if axis not in ("h", "v"):
            raise ValueError(
                f"env_regions[{index}]: band axis must be 'h' or 'v', got {axis!r}"
            )
        width = int(region["width"])
        if width < 0:
            raise ValueError(f"env_regions[{index}]: band width must be >= 0")
        return _band_mask(N, band_indices(int(region["center"]), width, N), axis, device)
    raise ValueError(f"env_regions[{index}]: unknown shape {shape!r}")


def _channel_present(region: dict, key: str) -> bool:
    return key in region and region[key] is not None


def _paint_custom(
    cfg: Config, N: int, device: str, dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, list[tuple[Tensor, float]]]:
    """Returns kappa_R, kappa_E, eta_R, eta_E, dead_mask, occupancy_start, explicit occ writes."""
    kappa_R = _ones(N, device, dtype)
    kappa_E = _ones(N, device, dtype)
    eta_R = _ones(N, device, dtype)
    eta_E = _ones(N, device, dtype)
    dead_mask = _false_mask(N, device)
    occ_writes: list[tuple[Tensor, float]] = []
    regions = cfg.env_regions if cfg.env_regions is not None else []
    if not isinstance(regions, list):
        raise ValueError("env_regions must be a list or None")
    for i, region in enumerate(regions):
        if not isinstance(region, dict):
            raise ValueError(f"env_regions[{i}]: expected dict, got {type(region)!r}")
        mask = _region_mask(region, N, device, i)
        wrote_kappa = False
        if _channel_present(region, "kappa_R"):
            kappa_R = torch.where(
                mask, torch.tensor(float(region["kappa_R"]), device=device, dtype=dtype), kappa_R,
            )
            wrote_kappa = True
        if _channel_present(region, "kappa_E"):
            kappa_E = torch.where(
                mask, torch.tensor(float(region["kappa_E"]), device=device, dtype=dtype), kappa_E,
            )
            wrote_kappa = True
        if _channel_present(region, "eta_R"):
            eta_R = torch.where(
                mask, torch.tensor(float(region["eta_R"]), device=device, dtype=dtype), eta_R,
            )
        if _channel_present(region, "eta_E"):
            eta_E = torch.where(
                mask, torch.tensor(float(region["eta_E"]), device=device, dtype=dtype), eta_E,
            )
        if _channel_present(region, "occupancy"):
            occ_writes.append((mask, float(region["occupancy"])))
        elif wrote_kappa:
            dead_mask = dead_mask | mask
    occupancy = _ones(N, device, dtype)
    return kappa_R, kappa_E, eta_R, eta_E, dead_mask, occupancy, occ_writes


def generate_environment(cfg: Config) -> Environment:
    """If not cfg.environment_heterogeneous: identity, no Generator.

    Else paint (κ, η, dead_mask) from preset; occupancy from dead_mask
    iff env_occupancy_blocks. Never uses cfg.seed.
    """
    N = int(cfg.N)
    device = cfg.device
    dtype = torch.float32
    if not cfg.environment_heterogeneous:
        return identity_environment(N, device, dtype)

    preset = cfg.env_preset
    if preset not in PRESETS:
        raise ValueError(
            f"unknown env_preset {preset!r}. Known: {', '.join(PRESETS)}"
        )

    ones = _ones(N, device, dtype)
    kappa_R = ones.clone()
    kappa_E = ones.clone()
    eta_R = ones.clone()
    eta_E = ones.clone()
    dead_mask = _false_mask(N, device)
    occ_writes: list[tuple[Tensor, float]] = []

    lo = float(cfg.env_kappa_lo)
    hi = float(cfg.env_kappa_hi)
    r = float(cfg.env_blob_radius) * N

    if preset == "identity":
        pass
    elif preset == "vertical_band":
        w = max(0, int(round(float(cfg.env_dead_frac) * N)))
        dead_mask = _band_mask(N, band_indices(N // 2, w, N), "v", device)
        kappa_R = _kappa_from_mask(dead_mask, lo, hi, dtype)
        kappa_E = kappa_R.clone()
    elif preset == "horizontal_band":
        w = max(0, int(round(float(cfg.env_dead_frac) * N)))
        dead_mask = _band_mask(N, band_indices(N // 2, w, N), "h", device)
        kappa_R = _kappa_from_mask(dead_mask, lo, hi, dtype)
        kappa_E = kappa_R.clone()
    elif preset == "torus_wall":
        w = max(0, int(round(0.5 * float(cfg.env_dead_frac) * N)))
        dead_mask = (
            _band_mask(N, band_indices(0, w, N), "v", device)
            | _band_mask(N, band_indices(N // 2, w, N), "v", device)
        )
        kappa_R = _kappa_from_mask(dead_mask, lo, hi, dtype)
        kappa_E = kappa_R.clone()
    elif preset == "center_blob":
        dead_mask = toroidal_disk_mask(N, (N - 1) / 2, (N - 1) / 2, r, device)
        kappa_R = _kappa_from_mask(dead_mask, lo, hi, dtype)
        kappa_E = kappa_R.clone()
    elif preset in ("blobs", "blobs_soft"):
        n_blobs = int(cfg.env_n_blobs)
        if n_blobs > 0:
            # Generator only when this preset actually samples centers.
            gen = torch.Generator().manual_seed(int(cfg.env_seed))
            centers = _blob_centers(cfg, gen)
        else:
            centers = torch.zeros(0, 2, dtype=torch.long)
        dead_mask = _union_disks(N, centers, r, device)
        if preset == "blobs":
            kappa_R = _kappa_from_mask(dead_mask, lo, hi, dtype)
            kappa_E = kappa_R.clone()
        else:
            kappa_R = _soft_blob_kappa(N, centers, r, lo, hi, device, dtype)
            kappa_E = kappa_R.clone()
    elif preset == "checker":
        b = max(1, N // 8)
        ii = torch.arange(N, device=device).view(N, 1).expand(N, N)
        jj = torch.arange(N, device=device).view(1, N).expand(N, N)
        dead_mask = (((ii // b) + (jj // b)) % 2) == 0
        kappa_R = _kappa_from_mask(dead_mask, lo, hi, dtype)
        kappa_E = kappa_R.clone()
    elif preset == "split_types":
        dead_mask = _false_mask(N, device)
        left = torch.arange(N, device=device).view(1, N).expand(N, N) < (N / 2)
        kappa_R = torch.where(
            left,
            torch.tensor(lo, device=device, dtype=dtype),
            torch.tensor(hi, device=device, dtype=dtype),
        )
        kappa_E = torch.where(
            left,
            torch.tensor(hi, device=device, dtype=dtype),
            torch.tensor(lo, device=device, dtype=dtype),
        )
    elif preset == "learning_hotspot":
        dead_mask = _false_mask(N, device)
        hot = toroidal_disk_mask(N, (N - 1) / 2, (N - 1) / 2, r, device)
        eta_lo = float(cfg.env_eta_lo)
        eta_hi = float(cfg.env_eta_hi)
        eta_R = torch.where(
            hot,
            torch.tensor(eta_hi, device=device, dtype=dtype),
            torch.tensor(eta_lo, device=device, dtype=dtype),
        )
        eta_E = eta_R.clone()
    elif preset == "custom":
        kappa_R, kappa_E, eta_R, eta_E, dead_mask, _, occ_writes = _paint_custom(
            cfg, N, device, dtype,
        )

    occupancy = ones.clone()
    if cfg.env_occupancy_blocks:
        occupancy = (~dead_mask).to(dtype=dtype)
    for mask, val in occ_writes:
        occupancy = torch.where(
            mask, torch.tensor(val, device=device, dtype=dtype), occupancy,
        )

    if not cfg.env_affect_R:
        kappa_R = ones.clone()
        eta_R = ones.clone()
    if not cfg.env_affect_E:
        kappa_E = ones.clone()
        eta_E = ones.clone()

    env = Environment(
        occupancy=occupancy.detach(),
        kappa_R=kappa_R.detach(),
        kappa_E=kappa_E.detach(),
        eta_scale_R=eta_R.detach(),
        eta_scale_E=eta_E.detach(),
    )
    for t in env.tensors():
        t.requires_grad_(False)
    return env
