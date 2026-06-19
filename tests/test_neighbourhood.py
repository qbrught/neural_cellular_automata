"""Test gather_neighbours: correctness against a hand-checked grid + wrap."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from grid import NEIGHBOUR_OFFSETS, gather_neighbours


def test_gather_neighbours_scalar_field():
    # 4x4 grid where each cell's value is its row*10 + col.
    # i.e. field[i, j] = 10*i + j.
    N = 4
    field = torch.tensor(
        [[10 * i + j for j in range(N)] for i in range(N)], dtype=torch.float32
    )
    # Sanity:
    assert field[2, 3].item() == 23

    neigh = gather_neighbours(field)
    assert neigh.shape == (N, N, 8), neigh.shape

    # For cell (2, 2), neighbours per NEIGHBOUR_OFFSETS:
    # NW=(1,1)=11, N=(1,2)=12, NE=(1,3)=13, W=(2,1)=21, E=(2,3)=23,
    # SW=(3,1)=31, S=(3,2)=32, SE=(3,3)=33
    expected_22 = torch.tensor([11, 12, 13, 21, 23, 31, 32, 33], dtype=torch.float32)
    assert torch.equal(neigh[2, 2], expected_22), (
        f"For cell (2,2), got {neigh[2,2].tolist()}, expected {expected_22.tolist()}"
    )
    print("test_gather_neighbours_scalar_field OK")


def test_gather_neighbours_toroidal_wrap():
    # Same field. Cell (0, 0)'s NW neighbour is (-1, -1) which wraps to (N-1, N-1).
    N = 4
    field = torch.tensor(
        [[10 * i + j for j in range(N)] for i in range(N)], dtype=torch.float32
    )
    neigh = gather_neighbours(field)

    # (0, 0): NW=(N-1,N-1)=33, N=(N-1,0)=30, NE=(N-1,1)=31,
    #         W=(0,N-1)=03, E=(0,1)=01,
    #         SW=(1,N-1)=13, S=(1,0)=10, SE=(1,1)=11
    expected_00 = torch.tensor([33, 30, 31, 3, 1, 13, 10, 11], dtype=torch.float32)
    assert torch.equal(neigh[0, 0], expected_00), (
        f"For cell (0,0), got {neigh[0,0].tolist()}, expected {expected_00.tolist()}"
    )

    # (N-1, N-1): SE wraps to (0, 0).
    # NW=(N-2,N-2)=22, N=(N-2,N-1)=23, NE=(N-2,0)=20,
    # W=(N-1,N-2)=32, E=(N-1,0)=30,
    # SW=(0,N-2)=2, S=(0,N-1)=3, SE=(0,0)=0
    expected_NN = torch.tensor([22, 23, 20, 32, 30, 2, 3, 0], dtype=torch.float32)
    assert torch.equal(neigh[N - 1, N - 1], expected_NN), (
        f"For cell (N-1,N-1), got {neigh[N-1,N-1].tolist()}, "
        f"expected {expected_NN.tolist()}"
    )
    print("test_gather_neighbours_toroidal_wrap OK")


def test_gather_neighbours_vector_field():
    # field of shape (N, N, d). Each cell holds vector [i, j, i+j].
    N, d = 3, 3
    field = torch.zeros(N, N, d)
    for i in range(N):
        for j in range(N):
            field[i, j] = torch.tensor([float(i), float(j), float(i + j)])

    neigh = gather_neighbours(field)
    assert neigh.shape == (N, N, 8, d), neigh.shape

    # Spot-check: cell (1,1)'s E neighbour (offset (0,1)) should be field[1,2] = [1,2,3]
    # E is index 4 in NEIGHBOUR_OFFSETS.
    assert NEIGHBOUR_OFFSETS[4] == (0, 1)
    expected = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(neigh[1, 1, 4], expected), (
        f"Got {neigh[1,1,4].tolist()}, expected {expected.tolist()}"
    )
    print("test_gather_neighbours_vector_field OK")


def test_gather_is_consistent_with_explicit_indexing():
    """Strongest test: for every cell and every neighbour offset, the gathered
    value matches an explicit modular index lookup."""
    N = 5
    torch.manual_seed(0)
    field = torch.randn(N, N, 4)
    neigh = gather_neighbours(field)

    for i in range(N):
        for j in range(N):
            for k, (di, dj) in enumerate(NEIGHBOUR_OFFSETS):
                expected = field[(i + di) % N, (j + dj) % N]
                assert torch.equal(neigh[i, j, k], expected), (
                    f"Mismatch at cell ({i},{j}) neighbour k={k} (offset {di},{dj}): "
                    f"got {neigh[i,j,k].tolist()}, expected {expected.tolist()}"
                )
    print("test_gather_is_consistent_with_explicit_indexing OK")


if __name__ == "__main__":
    test_gather_neighbours_scalar_field()
    test_gather_neighbours_toroidal_wrap()
    test_gather_neighbours_vector_field()
    test_gather_is_consistent_with_explicit_indexing()
    print("\nAll neighbourhood tests passed.")
