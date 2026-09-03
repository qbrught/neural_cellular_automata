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
    """All run hyperparameters, seed, and output location.

    Saved as JSON next to every trajectory so a run is exactly reproducible.
    """
    # ---- Grid ----
    N: int = 20  # grid side length (N x N cells)

    # ---- State dimensions ----
    d: int = 8  # size of observable state s and memory h
    hidden: int = 16  # MLP hidden width for psi and f

    # ---- Learning ----
    eta: float = 0.01  # SGD learning rate
    n_steps: int | None = 4000  # simulation steps; None means run indefinitely
    learn: bool = True
    # If True, gradient flows through aggregated messages M into senders' ψ
    # message heads (via -p_i → s̃_i → M_i). Default False keeps Path-1
    # locality: each cell's loss only touches params at that cell.
    # See dynamics.local_update for the tradeoff.
    learn_messages: bool = False
    require_alive_neighbour: bool = True  # if True, a cell needs A_i > 0 to be alive next step

    # ---- Research version flags (paper ablation path) ----
    # Step A: typed help/harm votes routed by kin vs foe.
    #   False = original single-channel votes (indiscriminate)
    #   True  = dual votes with goal routing (version A)
    typed_votes: bool = True
    # Step B: eliminators only pressure reproducer (prey) neighbours;
    # fellow eliminators are neutral in the loss (not +kill-all).
    predator_prey_loss: bool = False
    # Step C: on birth (dead→alive), adopt majority goal among pre-step
    # alive Moore neighbours (tie → max-rho neighbour). Survivors/deaths keep
    # goals; rho stays fixed. No parameter inheritance.
    goal_inheritance: bool = False
    # Step D: own goal into f input (type-conditioned local update).
    # f always has a goal input slot for param parity; when False the slot
    # is zeros so goal weights stay inert.
    goal_in_f: bool = False
    # Experiment F: soft coexistence pressure — weak barrier on soft living
    # mass of each goal-class so learning is discouraged from extinguishing
    # either type. Added once to the step total loss (not per-cell broadcast).
    # See research/SOFT_COEXISTENCE_BRIEF.md.
    coexistence_pressure: bool = False
    coexistence_lambda: float = 0.01  # λ strength of barrier
    coexistence_delta: float = 1e-4  # δ floor inside log(ρ̃ + δ)

    # ---- Experiment G: heterogeneous environment on the regular torus ----
    # Frozen spatial field (occupancy / conductivity / η scale). Default off
    # is a skip path: identity maps, no env RNG. Unknown presets are rejected
    # in environment.generate_environment (avoids a Config↔environment cycle).
    environment_heterogeneous: bool = False
    env_preset: str = "identity"
    env_seed: int = 0
    env_dead_frac: float = 0.20
    env_n_blobs: int = 3
    env_blob_radius: float = 0.15
    env_kappa_lo: float = 0.0
    env_kappa_hi: float = 1.0
    env_eta_lo: float = 0.25
    env_eta_hi: float = 1.0
    env_affect_R: bool = True
    env_affect_E: bool = True
    env_occupancy_blocks: bool = False
    env_regions: list | None = None

    # ---- Survival rule weights ----
    # Logistic combination of:
    #   w0: bias (negative -> bias toward death)
    #   w1: total alive neighbours
    #   w2: alive reproducer neighbours
    #   w3: alive eliminator neighbours
    #   w4_help: kin-channel (typed) or sole vote channel (original)
    #   w4_harm: foe-channel vote sum (typed mode only; unused if typed_votes=False)
    #   w5: f-signal = u . s_proposed_i  (Path 1: gives f a gradient channel)
    w0: float = -2.15
    w1: float = -2.06
    w2: float = 3.51
    w3: float = -1.87
    w4_help: float = 0.5
    w4_harm: float = 0.5
    w5: float = -1.34

    # ---- Initialisation ----
    seed: int = 68875
    init_noise_std: float = 0.01  # std of noise added to identical-init MLP weights
    init_alive_prob: float = 0.19  # probability each cell starts alive
    u_seed: int = 119725  # seed for the fixed projection vector u (separate from run seed)

    # ---- Output ----
    output_dir: str = "runs"
    save_state_vectors: bool = True  # whether snapshots include s and h tensors

    # ---- Device ----
    device: str = "cpu"  # "cpu" or "cuda"

    # ---- Run metadata (filled in at runtime, not user-set) ----
    run_name: str = field(default="")

    def __post_init__(self) -> None:
        """Validate ranges. Called again after CLI / UI overrides."""
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
        if self.coexistence_lambda < 0:
            raise ValueError(
                f"coexistence_lambda must be non-negative, got {self.coexistence_lambda}"
            )
        if self.coexistence_delta <= 0:
            raise ValueError(
                f"coexistence_delta must be positive, got {self.coexistence_delta}"
            )
        if not 0.0 <= self.env_dead_frac <= 1.0:
            raise ValueError(
                f"env_dead_frac must be in [0,1], got {self.env_dead_frac}"
            )
        if self.env_n_blobs < 0:
            raise ValueError(
                f"env_n_blobs must be non-negative, got {self.env_n_blobs}"
            )
        if self.env_n_blobs > 32:
            raise ValueError(
                f"env_n_blobs must be <= 32, got {self.env_n_blobs}"
            )
        if self.env_blob_radius <= 0:
            raise ValueError(
                f"env_blob_radius must be positive, got {self.env_blob_radius}"
            )
        if self.env_kappa_lo < 0:
            raise ValueError(
                f"env_kappa_lo must be non-negative, got {self.env_kappa_lo}"
            )
        if self.env_kappa_hi < 0:
            raise ValueError(
                f"env_kappa_hi must be non-negative, got {self.env_kappa_hi}"
            )
        if self.env_eta_lo < 0:
            raise ValueError(
                f"env_eta_lo must be non-negative, got {self.env_eta_lo}"
            )
        if self.env_eta_hi < 0:
            raise ValueError(
                f"env_eta_hi must be non-negative, got {self.env_eta_hi}"
            )
        if self.env_regions is not None and not isinstance(self.env_regions, list):
            raise ValueError(
                f"env_regions must be a list or None, got {type(self.env_regions)!r}"
            )
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
        """Plain dict of all fields (JSON-serialisable)."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Write this config as indented JSON, creating parent dirs."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """Load JSON, migrating legacy ``w4`` and dropping unknown keys."""
        with Path(path).open("r") as f:
            data = json.load(f)
        # Migrate pre-step-A configs that used a single w4 vote weight.
        if "w4" in data and "w4_help" not in data:
            w4 = data.pop("w4")
            data["w4_help"] = w4
            data.setdefault("w4_harm", w4)
            # Historical configs with only w4 predate typed votes.
            data.setdefault("typed_votes", False)
        else:
            data.pop("w4", None)
        # Ignore unknown keys so older/newer JSON stays loadable.
        known = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        data = {k: v for k, v in data.items() if k in known}
        return cls(**data)
