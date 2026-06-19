"""Test: batched MLP matches per-cell nn.Linear loop to float precision."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from parameters import (
    batched_mlp,
    f_in_dim,
    f_out_dim,
    init_parameters,
    psi_in_dim,
    psi_out_dim,
)


def reference_mlp_per_cell(x_grid, W1, b1, W2, b2):
    """Compute the per-cell MLP using a Python loop over cells.

    x_grid: (N, N, in_dim) or (N, N, K, in_dim)
    W1, b1, W2, b2: per-cell weight tensors.
    """
    N = W1.shape[0]
    out_dim = W2.shape[-1]
    if x_grid.dim() == 3:
        result = torch.zeros(N, N, out_dim, dtype=x_grid.dtype)
        for i in range(N):
            for j in range(N):
                lin1 = nn.Linear(W1.shape[2], W1.shape[3])
                lin1.weight.data = W1[i, j].T.clone()
                lin1.bias.data = b1[i, j].clone()
                lin2 = nn.Linear(W2.shape[2], W2.shape[3])
                lin2.weight.data = W2[i, j].T.clone()
                lin2.bias.data = b2[i, j].clone()
                h = torch.tanh(lin1(x_grid[i, j]))
                y = lin2(h)
                result[i, j] = y
        return result
    elif x_grid.dim() == 4:
        K = x_grid.shape[2]
        result = torch.zeros(N, N, K, out_dim, dtype=x_grid.dtype)
        for i in range(N):
            for j in range(N):
                lin1 = nn.Linear(W1.shape[2], W1.shape[3])
                lin1.weight.data = W1[i, j].T.clone()
                lin1.bias.data = b1[i, j].clone()
                lin2 = nn.Linear(W2.shape[2], W2.shape[3])
                lin2.weight.data = W2[i, j].T.clone()
                lin2.bias.data = b2[i, j].clone()
                for k in range(K):
                    h = torch.tanh(lin1(x_grid[i, j, k]))
                    y = lin2(h)
                    result[i, j, k] = y
        return result
    else:
        raise ValueError(x_grid.dim())


def test_batched_psi_matches_loop():
    N, d, hidden = 5, 4, 8
    gen = torch.Generator().manual_seed(42)
    params = init_parameters(N, d, hidden, init_noise_std=0.1, generator=gen)

    in_psi = psi_in_dim(d)
    x = torch.randn(N, N, in_psi, generator=gen)

    batched = batched_mlp(x, params.psi_W1, params.psi_b1, params.psi_W2, params.psi_b2)
    looped = reference_mlp_per_cell(
        x, params.psi_W1, params.psi_b1, params.psi_W2, params.psi_b2
    )

    assert batched.shape == (N, N, psi_out_dim(d))
    assert torch.allclose(batched, looped, atol=1e-6), (
        f"Max abs diff: {(batched - looped).abs().max().item()}"
    )
    print("test_batched_psi_matches_loop OK")


def test_batched_psi_matches_loop_with_K_neighbours():
    N, d, hidden, K = 4, 3, 6, 8
    gen = torch.Generator().manual_seed(7)
    params = init_parameters(N, d, hidden, init_noise_std=0.1, generator=gen)

    in_psi = psi_in_dim(d)
    x = torch.randn(N, N, K, in_psi, generator=gen)

    batched = batched_mlp(x, params.psi_W1, params.psi_b1, params.psi_W2, params.psi_b2)
    looped = reference_mlp_per_cell(
        x, params.psi_W1, params.psi_b1, params.psi_W2, params.psi_b2
    )

    assert batched.shape == (N, N, K, psi_out_dim(d))
    assert torch.allclose(batched, looped, atol=1e-6), (
        f"Max abs diff: {(batched - looped).abs().max().item()}"
    )
    print("test_batched_psi_matches_loop_with_K_neighbours OK")


def test_batched_f_matches_loop():
    N, d, hidden = 6, 5, 10
    gen = torch.Generator().manual_seed(123)
    params = init_parameters(N, d, hidden, init_noise_std=0.05, generator=gen)

    in_f = f_in_dim(d)
    x = torch.randn(N, N, in_f, generator=gen)

    batched = batched_mlp(x, params.f_W1, params.f_b1, params.f_W2, params.f_b2)
    looped = reference_mlp_per_cell(
        x, params.f_W1, params.f_b1, params.f_W2, params.f_b2
    )

    assert batched.shape == (N, N, f_out_dim(d))
    assert torch.allclose(batched, looped, atol=1e-6), (
        f"Max abs diff: {(batched - looped).abs().max().item()}"
    )
    print("test_batched_f_matches_loop OK")


def test_each_cell_uses_its_own_weights():
    """A targeted test: cells with different weights produce different outputs
    even from identical inputs."""
    N, d, hidden = 3, 2, 4
    gen = torch.Generator().manual_seed(0)
    params = init_parameters(N, d, hidden, init_noise_std=0.5, generator=gen)

    in_psi = psi_in_dim(d)
    # Same input for every cell.
    x_single = torch.randn(in_psi, generator=gen)
    x = x_single.expand(N, N, in_psi).contiguous()

    out = batched_mlp(x, params.psi_W1, params.psi_b1, params.psi_W2, params.psi_b2)

    # Outputs across cells must differ (the weights differ).
    out_flat = out.reshape(N * N, -1)
    diffs = (out_flat[0:1] - out_flat).abs().sum(dim=-1)
    assert (diffs[1:] > 1e-4).all(), (
        "All cells produced identical output despite having different weights"
    )
    print("test_each_cell_uses_its_own_weights OK")


def test_parameters_save_load(tmp_path_factory=None):
    import tempfile
    N, d, hidden = 4, 3, 5
    gen = torch.Generator().manual_seed(99)
    params = init_parameters(N, d, hidden, init_noise_std=0.1, generator=gen)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    params.save(path)

    from parameters import Parameters
    loaded = Parameters.load(path)
    for t_orig, t_loaded in zip(params.tensors(), loaded.tensors()):
        assert torch.equal(t_orig, t_loaded), "Tensor mismatch after save/load"
    print("test_parameters_save_load OK")


if __name__ == "__main__":
    test_batched_psi_matches_loop()
    test_batched_psi_matches_loop_with_K_neighbours()
    test_batched_f_matches_loop()
    test_each_cell_uses_its_own_weights()
    test_parameters_save_load()
    print("\nAll batched MLP tests passed.")
