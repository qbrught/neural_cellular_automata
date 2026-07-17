"""Per-cell MLP parameters and batched forward pass.

Every cell at grid position (i, j) has its own ψ and f MLPs. We store the
weights as (N, N, ...) tensors so the whole grid is one batched matmul.

Architecture (both ψ and f are 1-hidden-layer MLPs with tanh activation):
    h = tanh(x @ W1 + b1)
    y = h @ W2 + b2

Shapes per cell:
    W1: (in_dim, hidden)
    b1: (hidden,)
    W2: (hidden, out_dim)
    b2: (out_dim,)

Stacked across the grid the leading dims become (N, N, ...).

Dimensions:
    ψ (message function): input is (sender_state, sender_memory, sender_alive,
       sender_goal, receiver_state, receiver_memory, receiver_alive,
       receiver_goal). Output is (message_vector_of_dim_d, v_help, v_harm).
       in_dim_psi  = 2*d + 2*d + 4 = 4d + 4
       out_dim_psi = d + 2
       Typed votes (step A): v_help is routed only to same-goal receivers
       (kin channel); v_harm only to opposite-goal receivers (foe channel).

    f (local update): input is (own_state, own_memory, own_alive,
       aggregated_message_vector_of_dim_d). Output is (new_state, new_memory).
       in_dim_f  = d + d + 1 + d = 3d + 1
       out_dim_f = 2d
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def psi_in_dim(d: int) -> int:
    return 4 * d + 4  # +2 for sender goal, receiver goal


def psi_out_dim(d: int) -> int:
    return d + 2  # message vector (d) + help vote (1) + harm vote (1)


def f_in_dim(d: int) -> int:
    return 3 * d + 1


def f_out_dim(d: int) -> int:
    return 2 * d  # new state (d) + new memory (d)


@dataclass
class Parameters:
    """All learnable per-cell parameters, stored as (N, N, ...) tensors."""

    # ψ: message function
    psi_W1: Tensor  # (N, N, in_psi, hidden)
    psi_b1: Tensor  # (N, N, hidden)
    psi_W2: Tensor  # (N, N, hidden, out_psi)
    psi_b2: Tensor  # (N, N, out_psi)

    # f: local update
    f_W1: Tensor  # (N, N, in_f, hidden)
    f_b1: Tensor  # (N, N, hidden)
    f_W2: Tensor  # (N, N, hidden, out_f)
    f_b2: Tensor  # (N, N, out_f)

    def tensors(self) -> list[Tensor]:
        return [
            self.psi_W1, self.psi_b1, self.psi_W2, self.psi_b2,
            self.f_W1, self.f_b1, self.f_W2, self.f_b2,
        ]

    def requires_grad_(self, flag: bool = True) -> "Parameters":
        for t in self.tensors():
            t.requires_grad_(flag)
        return self

    def detach_clone(self) -> "Parameters":
        return Parameters(
            psi_W1=self.psi_W1.detach().clone(),
            psi_b1=self.psi_b1.detach().clone(),
            psi_W2=self.psi_W2.detach().clone(),
            psi_b2=self.psi_b2.detach().clone(),
            f_W1=self.f_W1.detach().clone(),
            f_b1=self.f_b1.detach().clone(),
            f_W2=self.f_W2.detach().clone(),
            f_b2=self.f_b2.detach().clone(),
        )

    def save(self, path) -> None:
        torch.save({
            "psi_W1": self.psi_W1, "psi_b1": self.psi_b1,
            "psi_W2": self.psi_W2, "psi_b2": self.psi_b2,
            "f_W1": self.f_W1, "f_b1": self.f_b1,
            "f_W2": self.f_W2, "f_b2": self.f_b2,
        }, path)

    @classmethod
    def load(cls, path) -> "Parameters":
        d = torch.load(path, weights_only=True)
        return cls(**d)


def init_parameters(
    N: int,
    d: int,
    hidden: int,
    init_noise_std: float,
    generator: torch.Generator,
    device: str = "cpu",
) -> Parameters:
    """Initialise all per-cell MLP weights.

    Every cell starts with the same nominal weights plus a small per-cell
    noise term, so cells are not literally identical at init (which would
    cause them to evolve identically as well).

    The "nominal" weights use Xavier-like scaling for the inputs and zero
    for biases.
    """
    in_psi, out_psi = psi_in_dim(d), psi_out_dim(d)
    in_f, out_f = f_in_dim(d), f_out_dim(d)

    def make(shape: tuple[int, ...], fan_in: int) -> Tensor:
        # Identical-across-grid base weight, drawn once.
        base = torch.empty(shape[2:], device=device)
        std = (2.0 / fan_in) ** 0.5
        base.normal_(mean=0.0, std=std, generator=generator)
        # Tile across the (N, N) leading dims and add small per-cell noise.
        out = base.expand(shape).clone()
        noise = torch.empty(shape, device=device)
        noise.normal_(mean=0.0, std=init_noise_std, generator=generator)
        return out + noise

    def make_bias(shape: tuple[int, ...]) -> Tensor:
        # Biases start at zero plus per-cell noise.
        out = torch.zeros(shape, device=device)
        noise = torch.empty(shape, device=device)
        noise.normal_(mean=0.0, std=init_noise_std, generator=generator)
        return out + noise

    return Parameters(
        psi_W1=make((N, N, in_psi, hidden), fan_in=in_psi),
        psi_b1=make_bias((N, N, hidden)),
        psi_W2=make((N, N, hidden, out_psi), fan_in=hidden),
        psi_b2=make_bias((N, N, out_psi)),
        f_W1=make((N, N, in_f, hidden), fan_in=in_f),
        f_b1=make_bias((N, N, hidden)),
        f_W2=make((N, N, hidden, out_f), fan_in=hidden),
        f_b2=make_bias((N, N, out_f)),
    )


def batched_mlp(
    x: Tensor,
    W1: Tensor,
    b1: Tensor,
    W2: Tensor,
    b2: Tensor,
) -> Tensor:
    """Forward pass for a batched per-cell 1-hidden-layer tanh MLP.

    Args:
        x: input, shape (..., in_dim) where leading dims align with W1's
           leading (N, N) dims via broadcasting. Typical shapes:
              (N, N, in_dim)            -- one input per cell
              (N, N, K, in_dim)         -- K inputs per cell (e.g. K=8 neighbours)
        W1: (N, N, in_dim, hidden)
        b1: (N, N, hidden)
        W2: (N, N, hidden, out_dim)
        b2: (N, N, out_dim)

    Returns:
        Tensor of shape (..., out_dim).

    The MLP is applied independently for each cell using that cell's
    private weights.
    """
    if x.dim() == 3:
        # (N, N, in_dim) -> (N, N, hidden)
        # einsum: nij,nio -> njo... actually we need (N,N,in)@(N,N,in,hid) -> (N,N,hid)
        h = torch.einsum("nmi,nmih->nmh", x, W1) + b1
        h = torch.tanh(h)
        y = torch.einsum("nmh,nmho->nmo", h, W2) + b2
        return y
    elif x.dim() == 4:
        # (N, N, K, in_dim) -> (N, N, K, out_dim)
        # b1, b2 broadcast over K
        h = torch.einsum("nmki,nmih->nmkh", x, W1) + b1.unsqueeze(2)
        h = torch.tanh(h)
        y = torch.einsum("nmkh,nmho->nmko", h, W2) + b2.unsqueeze(2)
        return y
    else:
        raise ValueError(f"batched_mlp expected 3D or 4D input, got {x.dim()}D")
