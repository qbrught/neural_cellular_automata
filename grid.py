"""Grid: container for a single CA instance + neighbourhood operations.

The 8 Moore-neighbourhood offsets, in fixed order. We use this order
everywhere so that "neighbour index k" is unambiguous.

    (di, dj) for k = 0..7:
        0: (-1, -1)  NW
        1: (-1,  0)  N
        2: (-1,  1)  NE
        3: ( 0, -1)  W
        4: ( 0,  1)  E
        5: ( 1, -1)  SW
        6: ( 1,  0)  S
        7: ( 1,  1)  SE

Neighbour k of cell (i, j) is the cell at (i + di, j + dj) with toroidal
wrap. Equivalently, the tensor "values of neighbour k" is obtained by
rolling the input by (-di, -dj) -- because what's now at position (i, j)
in the rolled tensor came from (i + di, j + dj) in the original.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from parameters import Parameters
from state import State

if TYPE_CHECKING:
    from environment import Environment

# (di, dj) offsets in canonical order.
NEIGHBOUR_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
)
NUM_NEIGHBOURS = 8


def gather_neighbours(field: Tensor) -> Tensor:
    """Gather the 8 Moore-neighbourhood values for every cell.

    Args:
        field: tensor of shape (N, N) or (N, N, ...). Spatial dims first.

    Returns:
        Tensor of shape (N, N, 8) or (N, N, 8, ...).

        result[i, j, k, ...] == field[(i + di_k) % N, (j + dj_k) % N, ...]

    Toroidal boundaries. Order of neighbours is NEIGHBOUR_OFFSETS.
    """
    rolled = []
    for di, dj in NEIGHBOUR_OFFSETS:
        # field[i, j] should now appear at (i - di, j - dj) so that
        # at position (i, j) after the roll we see field[i + di, j + dj].
        r = torch.roll(field, shifts=(-di, -dj), dims=(0, 1))
        rolled.append(r)
    # Stack along a new axis at position 2 (just after the two spatial dims).
    return torch.stack(rolled, dim=2)


@dataclass
class Grid:
    """Owns state and parameters for a single CA run.

    This class is intentionally thin: it's a container, not a god object.
    The actual dynamics live in dynamics.py and learning.py as pure
    functions of (state, params, u, config).
    """
    state: State
    params: Parameters
    u: Tensor  # (d,) fixed projection vector for the f-signal channel
    env: Environment | None = None  # Experiment G overlay; no identity factory

    @property
    def N(self) -> int:
        return self.state.N

    @property
    def d(self) -> int:
        return self.state.d
