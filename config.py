"""Configuration for a Neural State-Aware Cellular Automaton run.

A Config is a single source of truth: every hyperparameter, the random seed,
and the output location live here. A Config is saved alongside every
trajectory so any run is exactly reproducible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ---- Grid ----
    N: int = 20  # grid side length (N x N cells)

    # ---- State dimensions ----
    d: int = 8  # size of observable state s and memory h
    hidden: int = 16  # MLP hidden width for psi and f

    # ---- Learning ----
    eta: float = 0.01  # SGD learning rate
    n_steps: int | None = 1000  # simulation steps; None means run indefinitely
    learn: bool = True
    # ---- Survival rule weights ----
    # Logistic combination of:
    #   w0: bias (negative -> bias toward death)
    #   w1: total alive neighbours
    #   w2: alive reproducer neighbours
    #   w3: alive eliminator neighbours
    #   w4: weighted vote sum (sum over neighbours of rho_j * v_{j->i})
    #   w5: f-signal = u . s_proposed_i  (Path 1: gives f a gradient channel)
    w0: float = -1.0
    w1: float = 0.5
    w2: float = 0.5
    w3: float = -0.5
    w4: float = 0.5
    w5: float = 0.5

    # ---- Initialisation ----
    seed: int = 0
    init_noise_std: float = 0.01  # std of noise added to identical-init MLP weights
    init_alive_prob: float = 0.5  # probability each cell starts alive
    u_seed: int = 12345  # seed for the fixed projection vector u (separate from run seed)

    # ---- Output ----
    output_dir: str = "runs"
    save_state_vectors: bool = True  # whether snapshots include s and h tensors

    # ---- Device ----
    device: str = "cpu"  # "cpu" or "cuda"

    # ---- Run metadata (filled in at runtime, not user-set) ----
    run_name: str = field(default="")

    def __post_init__(self) -> None:
        if self.N <= 0:
            raise ValueError(f"N must be positive, got {self.N}")
        if self.d <= 0:
            raise ValueError(f"d must be positive, got {self.d}")
        if self.hidden <= 0:
            raise ValueError(f"hidden must be positive, got {self.hidden}")
        if self.eta <= 0:
            raise ValueError(f"eta must be positive, got {self.eta}")
        if self.n_steps is not None and self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive or None, got {self.n_steps}")
        if self.init_noise_std < 0:
            raise ValueError(
                f"init_noise_std must be non-negative, got {self.init_noise_std}"
            )
        if not 0.0 <= self.init_alive_prob <= 1.0:
            raise ValueError(
                f"init_alive_prob must be in [0,1], got {self.init_alive_prob}"
            )
        if self.device not in ("cpu", "cuda"):
            raise ValueError(f"device must be 'cpu' or 'cuda', got {self.device!r}")

    # ---- Serialisation ----

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with Path(path).open("r") as f:
            data = json.load(f)
        return cls(**data)
