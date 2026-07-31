"""Per-cell state tensors.

There are two kinds of per-cell data here:

1. Mutable state, updated every step:
   - x:  alive/dead flag, shape (N, N), float (held as float so it can act as
         a multiplicative mask without casting; values are exactly 0 or 1)
   - s:  observable state vector, shape (N, N, d)
   - h:  memory vector, shape (N, N, d)

2. Per-cell attributes fixed at init (with one optional exception):
   - goals: per-cell goal in {0=reproduce, 1=eliminate}, shape (N, N), int.
     Fixed for the whole run when ``goal_inheritance=False`` (default).
     With Step C (``goal_inheritance=True``), only **birth** cells
     (dead → alive) may adopt a neighbour's goal; survivors and pure
     deaths keep their labels (dead cells keep a latent goal until revival).
   - rho:   per-cell communication rate in (0, 1), shape (N, N), float.
     Always fixed for the whole run.

The Parameters tensors (psi/f MLP weights) are mutable too but live in
parameters.py because they have a separate lifecycle (gradient updates,
save/load independent of state).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

GOAL_REPRODUCE = 0
GOAL_ELIMINATE = 1


@dataclass
class State:
    x: Tensor      # (N, N)         alive flag, float in {0.0, 1.0}
    s: Tensor      # (N, N, d)      observable state
    h: Tensor      # (N, N, d)      memory
    goals: Tensor  # (N, N)         int in {0, 1}; fixed unless goal_inheritance
    rho: Tensor    # (N, N)         float in (0, 1), always FIXED

    @property
    def N(self) -> int:
        return self.x.shape[0]

    @property
    def d(self) -> int:
        return self.s.shape[-1]

    def reproduce_mask(self) -> Tensor:
        """Boolean (N, N) mask: True where goal == reproduce."""
        return self.goals == GOAL_REPRODUCE

    def eliminate_mask(self) -> Tensor:
        """Boolean (N, N) mask: True where goal == eliminate."""
        return self.goals == GOAL_ELIMINATE

    def reproduce_alive(self) -> Tensor:
        """(N, N) float: 1 where alive and goal == reproduce, else 0."""
        return self.x * self.reproduce_mask().float()

    def eliminate_alive(self) -> Tensor:
        """(N, N) float: 1 where alive and goal == eliminate, else 0."""
        return self.x * self.eliminate_mask().float()


def init_state(
    N: int,
    d: int,
    init_alive_prob: float,
    generator: torch.Generator,
    device: str = "cpu",
) -> State:
    """Initialise per-cell state.

    - x: Bernoulli(init_alive_prob) per cell.
    - s, h: zeros (dead cells have no signal to propagate, alive cells start
            blank and accumulate state through dynamics).
    - goals: Uniform({reproduce, eliminate}) per cell. Fixed for the run
             unless ``Config.goal_inheritance`` (Step C) is on, in which
             case birth cells inherit majority alive-neighbour goals.
    - rho:   Uniform(0, 1) per cell, FIXED for the run.
    """
    x = (
        torch.rand((N, N), device=device, generator=generator)
        < init_alive_prob
    ).float()
    s = torch.zeros((N, N, d), device=device)
    h = torch.zeros((N, N, d), device=device)
    goals = (
        torch.rand((N, N), device=device, generator=generator) < 0.5
    ).long()  # 0 or 1
    rho = torch.rand((N, N), device=device, generator=generator)
    return State(x=x, s=s, h=h, goals=goals, rho=rho)
