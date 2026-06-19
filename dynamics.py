"""CA dynamics: message passing, local update, and the fixed survival rule.

This file holds pure functions of (state, params, config). No mutation,
no side-effects — easy to test, easy to differentiate through.

Survival rule (Option B from the design discussion):

    p_i = sigmoid( w0
                 + w1 * A_i          # total alive neighbours
                 + w2 * R_i          # alive reproducer neighbours
                 + w3 * E_i          # alive eliminator neighbours
                 + w4 * V_i )        # weighted vote sum from neighbours

where V_i = sum_{j in N(i)} rho_j * v_{j->i}.

The hard rule used in the simulation: x_{t+1} = 1[p_i > 0.5].
The soft rule used in the loss:        x_{t+1} ~ p_i  (as a differentiable
                                        scalar in [0,1]).
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from config import Config
from grid import gather_neighbours
from parameters import Parameters, batched_mlp, psi_out_dim
from state import State


@dataclass
class SurvivalInputs:
    """Per-cell neighbourhood aggregates fed into the survival rule.

    All tensors are (N, N).
    """
    A: Tensor          # total alive neighbours
    R: Tensor          # alive reproducer neighbours
    E: Tensor          # alive eliminator neighbours
    V: Tensor          # weighted incoming vote sum
    f_signal: Tensor   # u . s_proposed_i  (Path 1: f's gradient channel)


def make_u(d: int, u_seed: int, device: str = "cpu") -> Tensor:
    """Sample the fixed projection vector u once. Same u_seed => same u.

    u is shape (d,), drawn from N(0, 1). Frozen for the run.
    """
    gen = torch.Generator().manual_seed(u_seed)
    return torch.randn(d, generator=gen, device=device)


def compute_survival_inputs(
    state: State,
    incoming_vote_sum: Tensor,
    s_proposed: Tensor,
    u: Tensor,
) -> SurvivalInputs:
    """Compute A, R, E, V, and the f-signal u . s_proposed_i.

    Args:
        state: current State.
        incoming_vote_sum: (N, N) tensor where entry (i, j) is
            sum_{k in N(i,j)} rho_k * v_{k -> (i,j)}.
        s_proposed: (N, N, d) tensor of f's proposed next-state outputs.
        u: (d,) fixed projection vector.

    Returns:
        SurvivalInputs (A, R, E, V, f_signal), each (N, N).
    """
    # Neighbour alive flags: (N, N, 8)
    alive_n = gather_neighbours(state.x)
    A = alive_n.sum(dim=2)

    # Neighbour reproducer-alive and eliminator-alive flags: (N, N, 8)
    repro_alive_n = gather_neighbours(state.reproduce_alive())
    elim_alive_n = gather_neighbours(state.eliminate_alive())
    R = repro_alive_n.sum(dim=2)
    E = elim_alive_n.sum(dim=2)

    # f-signal: project s_proposed onto u, per cell.
    # (N, N, d) . (d,) -> (N, N)
    f_signal = (s_proposed * u).sum(dim=-1)

    return SurvivalInputs(A=A, R=R, E=E, V=incoming_vote_sum, f_signal=f_signal)


def survival_logit(inputs: SurvivalInputs, cfg: Config) -> Tensor:
    """Logit of the survival probability:
       w0 + w1 A + w2 R + w3 E + w4 V + w5 tanh(u . s_proposed).

    The tanh bounds the f-signal contribution to [-w5, +w5], so even a
    well-trained f can't overwhelm the other survival terms. Without this,
    every cell trivially learns to push u . s_proposed very large and the
    population saturates to all-alive.
    """
    return (
        cfg.w0
        + cfg.w1 * inputs.A
        + cfg.w2 * inputs.R
        + cfg.w3 * inputs.E
        + cfg.w4 * inputs.V
        + cfg.w5 * torch.tanh(inputs.f_signal)
    )


def soft_survival(inputs: SurvivalInputs, cfg: Config) -> Tensor:
    """Differentiable survival probability p_i in [0, 1], shape (N, N)."""
    return torch.sigmoid(survival_logit(inputs, cfg))


def hard_survival(inputs: SurvivalInputs, cfg: Config) -> Tensor:
    """Hard alive/dead next-state in {0, 1}, shape (N, N), float.

    Threshold at p > 0.5, which is equivalent to logit > 0. Using the logit
    directly avoids the sigmoid roundtrip.
    """
    return (survival_logit(inputs, cfg) > 0).float()


# Message passing

@dataclass
class MessagePassOutput:
    """Outputs of the message-passing phase.

    All shapes use d = state.d.

    Attributes:
        aggregated_messages: (N, N, d) — sum of message vectors received by
            each cell. Fed into the local update f.
        incoming_vote_sum: (N, N) — sum over neighbours j of rho_j * v_{j->i}.
            Fed into the survival rule.
        outgoing_votes: (N, N, 8) — vote_{i -> k} for each cell i and each
            of its 8 outgoing edges (the k-th edge sends a message to the
            neighbour at offset NEIGHBOUR_OFFSETS[k] from i). Kept for use
            in the locality-preserving loss.
    """
    aggregated_messages: Tensor
    incoming_vote_sum: Tensor
    outgoing_votes: Tensor


def _build_psi_inputs(state: State) -> Tensor:
    """Construct ψ inputs for every directed edge in the grid.

    For each cell i and each of its 8 neighbours j, ψ_{θ_j}(y_j, y_i) is
    called: the message from j to i, using j's parameters.

    To vectorise this, we work from the receiver's perspective: for each
    cell i, we gather the 8 sender-side states (s_j, h_j, x_j) and pair
    them with i's own (s_i, h_i, x_i).

    The ψ MLP for the *sender* is what we need to apply. So later we'll
    also need a per-edge sender-parameter tensor. This function only builds
    the inputs.

    Returns:
        psi_inputs: (N, N, 8, 4d + 2)
            Along the last axis: [s_j, h_j, x_j, s_i, h_i, x_i]
            where j is the neighbour and i is the receiver (cell at (n,m)).
    """
    # Receiver-side scalars/vectors (just the cell's own).
    s_i = state.s                       # (N, N, d)
    h_i = state.h                       # (N, N, d)
    x_i = state.x.unsqueeze(-1)         # (N, N, 1)

    # Sender-side, gathered from each of 8 neighbours.
    s_j = gather_neighbours(state.s)    # (N, N, 8, d)
    h_j = gather_neighbours(state.h)    # (N, N, 8, d)
    x_j = gather_neighbours(state.x).unsqueeze(-1)  # (N, N, 8, 1)

    # Broadcast receiver across the 8 sender slots.
    K = 8
    s_i_exp = s_i.unsqueeze(2).expand(-1, -1, K, -1)
    h_i_exp = h_i.unsqueeze(2).expand(-1, -1, K, -1)
    x_i_exp = x_i.unsqueeze(2).expand(-1, -1, K, -1)

    return torch.cat([s_j, h_j, x_j, s_i_exp, h_i_exp, x_i_exp], dim=-1)


def _gather_sender_params(W: Tensor) -> Tensor:
    """Gather sender-side parameter tensor across the 8 edges.

    For a parameter tensor W of shape (N, N, ...), produce
    (N, N, 8, ...) such that result[i, j, k, ...] holds the W from the
    sender that is neighbour k of (i, j).
    """
    return gather_neighbours(W)


def _psi_forward_per_edge(psi_inputs: Tensor, params: Parameters) -> Tensor:
    """Apply ψ to every directed edge using the *sender's* per-cell weights.

    Args:
        psi_inputs: (N, N, 8, in_psi)
        params: per-cell parameters

    Returns:
        (N, N, 8, out_psi) where out_psi = d + 1 (message vec + vote scalar)
    """
    # Gather sender params for each edge.
    W1 = _gather_sender_params(params.psi_W1)  # (N, N, 8, in_psi, hidden)
    b1 = _gather_sender_params(params.psi_b1)  # (N, N, 8, hidden)
    W2 = _gather_sender_params(params.psi_W2)  # (N, N, 8, hidden, out_psi)
    b2 = _gather_sender_params(params.psi_b2)  # (N, N, 8, out_psi)

    # Manual two-layer MLP on the per-edge inputs with per-edge weights.
    # psi_inputs: (N, N, 8, in)   W1: (N, N, 8, in, hid)  -> (N, N, 8, hid)
    h = torch.einsum("nmki,nmkih->nmkh", psi_inputs, W1) + b1
    h = torch.tanh(h)
    y = torch.einsum("nmkh,nmkho->nmko", h, W2) + b2
    return y


def message_pass(state: State, params: Parameters) -> MessagePassOutput:
    """Compute messages from every neighbour to every cell.

    Steps:
        1. Build ψ inputs per directed edge.
        2. Apply ψ using the *sender's* weights.
        3. Split outputs into message vector (d) and vote scalar (1).
        4. Multiply each message and vote by the sender's communication rate
           ρ_j (the "rate-gated" message rule).
        5. Mask by sender alive flag (dead cells emit nothing).
        6. Aggregate: sum messages and votes over the 8 incoming edges.

    Returns:
        MessagePassOutput with aggregated_messages (N, N, d),
        incoming_vote_sum (N, N), outgoing_votes (N, N, 8).
    """
    d = state.d
    out_psi = psi_out_dim(d)
    assert out_psi == d + 1

    psi_inputs = _build_psi_inputs(state)           # (N, N, 8, 4d+2)
    psi_out = _psi_forward_per_edge(psi_inputs, params)  # (N, N, 8, d+1)

    # Split into messages and votes.
    messages = psi_out[..., :d]                     # (N, N, 8, d)
    votes_received = psi_out[..., d]                # (N, N, 8)  scalar per edge

    # Gate by sender's rho and alive flag.
    # The k-th slot is what arrives from neighbour k of the receiver.
    sender_rho = gather_neighbours(state.rho)       # (N, N, 8)
    sender_alive = gather_neighbours(state.x)       # (N, N, 8)
    gate = (sender_rho * sender_alive)              # (N, N, 8)

    messages = messages * gate.unsqueeze(-1)
    votes_received = votes_received * gate

    # Aggregate over the 8 incoming edges.
    aggregated_messages = messages.sum(dim=2)       # (N, N, d)
    incoming_vote_sum = votes_received.sum(dim=2)   # (N, N)

    outgoing_votes = _compute_outgoing_votes(state, params)
    return MessagePassOutput(
        aggregated_messages=aggregated_messages,
        incoming_vote_sum=incoming_vote_sum,
        outgoing_votes=outgoing_votes,
    )


def _compute_outgoing_votes(state: State, params: Parameters) -> Tensor:
    """Compute v_{i -> k} for each cell i and each of its 8 outgoing edges.

    From cell i's POV, it sends to each of its 8 neighbours. The receiver
    of edge k is the cell at offset NEIGHBOUR_OFFSETS[k] from i. ψ takes
    (y_sender=i, y_receiver=neighbour-k-of-i).

    Returns:
        (N, N, 8) — gated by rho_i and x_i, just like incoming_vote_sum.
    """
    d = state.d

    # Sender-side: own cell, broadcast over 8 outgoing edges.
    s_i = state.s.unsqueeze(2).expand(-1, -1, 8, -1)  # (N, N, 8, d)
    h_i = state.h.unsqueeze(2).expand(-1, -1, 8, -1)  # (N, N, 8, d)
    x_i = state.x.unsqueeze(-1).unsqueeze(2).expand(-1, -1, 8, -1)  # (N, N, 8, 1)

    # Receiver-side: the neighbour at each outgoing edge.
    # gather_neighbours(state.s)[i, j, k] is the s of neighbour-k of (i,j),
    # which is exactly the receiver of i's outgoing edge k.
    s_r = gather_neighbours(state.s)
    h_r = gather_neighbours(state.h)
    x_r = gather_neighbours(state.x).unsqueeze(-1)

    psi_in = torch.cat([s_i, h_i, x_i, s_r, h_r, x_r], dim=-1)  # (N, N, 8, 4d+2)

    # Apply ψ using i's OWN per-cell weights (no gather), broadcasting over edges.
    psi_out = batched_mlp(psi_in, params.psi_W1, params.psi_b1,
                          params.psi_W2, params.psi_b2)  # (N, N, 8, d+1)
    votes = psi_out[..., d]  # (N, N, 8)

    # Gate by sender's own rho and alive.
    gate = (state.rho * state.x).unsqueeze(-1)  # (N, N, 1)
    return votes * gate


# Local update

def local_update(
    state: State,
    aggregated_messages: Tensor,
    params: Parameters,
) -> tuple[Tensor, Tensor]:
    """Run f to produce proposed next (s, h) for every cell.

    f input: (own_s, own_h, own_x, aggregated_messages).  Shape (N,N, 3d+1)
    f output: (new_s, new_h).                              Shape (N,N, 2d)

    Returns (s_proposed, h_proposed), each (N, N, d). These are the
    *proposals* before survival is applied; surviving cells use them,
    dead cells get them zeroed out.

    LOCALITY NOTE (Path 1): we detach `aggregated_messages` before feeding it
    into f. Without this, gradient through `s_proposed[i]` would flow back
    through the messages cell `i` received from its 8 neighbours, into those
    neighbours' psi parameters — a locality violation. With the detach,
    `s_proposed[i]` only carries gradient back to `params.f[i]`, which is
    what we want for the f-signal channel. The message-as-f-input channel
    will be activated in a later version that runs BPTT through messages.
    """
    d = state.d
    own_x = state.x.unsqueeze(-1)  # (N, N, 1)
    f_in = torch.cat(
        [state.s, state.h, own_x, aggregated_messages.detach()],
        dim=-1,
    )
    f_out = batched_mlp(f_in, params.f_W1, params.f_b1, params.f_W2, params.f_b2)
    s_proposed = f_out[..., :d]
    h_proposed = f_out[..., d:]
    return s_proposed, h_proposed


# Full forward step

@dataclass
class StepOutput:
    """Everything produced by one CA step, including intermediates the
    learning code will need."""
    next_state: State

    # Intermediates kept for the learning step:
    survival_inputs: SurvivalInputs    # A, R, E, V at current step
    outgoing_votes: Tensor             # (N, N, 8) — current step's votes
    s_proposed: Tensor                 # (N, N, d)
    h_proposed: Tensor                 # (N, N, d)


def forward_step(state: State, params: Parameters, u: Tensor, cfg: Config) -> StepOutput:
    """One forward CA step. No gradient updates here.

    Order:
        1. Message pass: compute aggregated messages and incoming vote sum.
        2. Local update: propose new s and h.
        3. Survival: compute A, R, E from current state; combine with V and
           the f-signal (u . s_proposed); hard threshold for the new alive flag.
        4. Mask: zero out s and h for cells that died.
    """
    mp = message_pass(state, params)
    s_proposed, h_proposed = local_update(state, mp.aggregated_messages, params)

    surv_in = compute_survival_inputs(state, mp.incoming_vote_sum, s_proposed, u)
    x_next = hard_survival(surv_in, cfg)

    # Apply alive mask.
    mask = x_next.unsqueeze(-1)  # (N, N, 1)
    s_next = s_proposed * mask
    h_next = h_proposed * mask

    next_state = State(
        x=x_next.detach(),
        s=s_next.detach(),
        h=h_next.detach(),
        goals=state.goals,  # immutable
        rho=state.rho,      # immutable
    )
    return StepOutput(
        next_state=next_state,
        survival_inputs=surv_in,
        outgoing_votes=mp.outgoing_votes,
        s_proposed=s_proposed,
        h_proposed=h_proposed,
    )
