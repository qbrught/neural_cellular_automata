"""CA dynamics: message passing, local update, and the fixed survival rule.

This file holds pure functions of (state, params, config). No mutation,
no side-effects — easy to test, easy to differentiate through.

Survival rule (typed votes, step A):

    p_i = sigmoid( w0
                 + w1 * A_i            # total alive neighbours
                 + w2 * R_i            # alive reproducer neighbours
                 + w3 * E_i            # alive eliminator neighbours
                 + w4_help * V_kin_i   # same-goal senders' help votes
                 + w4_harm * V_foe_i   # opposite-goal senders' harm votes
                 + w5 * tanh(u · s̃_i) )

where
    V_kin_i = sum_{j in N(i), g_j=g_i}  rho_j * v_help_{j->i}
    V_foe_i = sum_{j in N(i), g_j≠g_i}  rho_j * v_harm_{j->i}

ψ emits both v_help and v_harm on every edge; routing by goal match decides
which channel reaches the receiver. This makes help/harm physically typed
rather than a single indiscriminate vote scalar.

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
    V_kin: Tensor      # same-goal (kin) help-vote sum
    V_foe: Tensor      # opposite-goal (foe) harm-vote sum
    f_signal: Tensor   # u . s_proposed_i  (Path 1: f's gradient channel)


def make_u(d: int, u_seed: int, device: str = "cpu") -> Tensor:
    """Sample the fixed projection vector u once. Same u_seed => same u.

    u is shape (d,), drawn from N(0, 1). Frozen for the run.
    """
    gen = torch.Generator().manual_seed(u_seed)
    return torch.randn(d, generator=gen, device=device)


def compute_survival_inputs(
    state: State,
    V_kin: Tensor,
    V_foe: Tensor,
    s_proposed: Tensor,
    u: Tensor,
) -> SurvivalInputs:
    """Compute A, R, E, typed vote sums, and the f-signal u . s_proposed_i.

    Args:
        state: current State.
        V_kin: (N, N) sum of rho-gated help votes from same-goal neighbours.
        V_foe: (N, N) sum of rho-gated harm votes from opposite-goal neighbours.
        s_proposed: (N, N, d) tensor of f's proposed next-state outputs.
        u: (d,) fixed projection vector.

    Returns:
        SurvivalInputs, each field (N, N).
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

    return SurvivalInputs(
        A=A, R=R, E=E, V_kin=V_kin, V_foe=V_foe, f_signal=f_signal,
    )


def survival_logit(inputs: SurvivalInputs, cfg: Config) -> Tensor:
    """Logit of the survival probability:
       w0 + w1 A + w2 R + w3 E + w4_help V_kin + w4_harm V_foe
       + w5 tanh(u . s_proposed).

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
        + cfg.w4_help * inputs.V_kin
        + cfg.w4_harm * inputs.V_foe
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
    alive_next = (survival_logit(inputs, cfg) > 0).float()
    if cfg.require_alive_neighbour:
        alive_next = alive_next * (inputs.A > 0).float()
    return alive_next


# Message passing

@dataclass
class MessagePassOutput:
    """Outputs of the message-passing phase.

    All shapes use d = state.d.

    Attributes:
        aggregated_messages: (N, N, d) — sum of message vectors received by
            each cell. Fed into the local update f.
        V_kin: (N, N) — sum of rho_j * v_help from same-goal neighbours.
        V_foe: (N, N) — sum of rho_j * v_harm from opposite-goal neighbours.
        outgoing_votes: (N, N, 8, 2) — [help, harm] from i along each of its
            8 outgoing edges, gated by rho_i and x_i (not yet goal-routed).
            Routing is applied when aggregating into V_kin / V_foe.
    """
    aggregated_messages: Tensor
    V_kin: Tensor
    V_foe: Tensor
    outgoing_votes: Tensor


def _build_psi_inputs(state: State) -> Tensor:
    """Construct ψ inputs for every directed edge in the grid.

    For each cell i and each of its 8 neighbours j, ψ_{θ_j}(y_j, y_i) is
    called: the message from j to i, using j's parameters.

    To vectorise this, we work from the receiver's perspective: for each
    cell i, we gather the 8 sender-side states (s_j, h_j, x_j, g_j) and pair
    them with i's own (s_i, h_i, x_i, g_i).

    The ψ MLP for the *sender* is what we need to apply. So later we'll
    also need a per-edge sender-parameter tensor. This function only builds
    the inputs.

    Returns:
        psi_inputs: (N, N, 8, 4d + 4)
            Along the last axis: [s_j, h_j, x_j, g_j, s_i, h_i, x_i, g_i]
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
    g_j = gather_neighbours(state.goals.float()).unsqueeze(-1)  # (N, N, 8, 1)
    g_i = state.goals.float().unsqueeze(-1)                      # (N, N, 1)

    # Broadcast receiver across the 8 sender slots.
    K = 8
    s_i_exp = s_i.unsqueeze(2).expand(-1, -1, K, -1)
    h_i_exp = h_i.unsqueeze(2).expand(-1, -1, K, -1)
    x_i_exp = x_i.unsqueeze(2).expand(-1, -1, K, -1)
    g_i_exp = g_i.unsqueeze(2).expand(-1, -1, K, -1)

    return torch.cat([s_j, h_j, x_j, g_j, s_i_exp, h_i_exp, x_i_exp, g_i_exp], dim=-1)


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
        (N, N, 8, out_psi) where out_psi = d + 2 (message + help + harm)
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


def _goal_match_mask(state: State) -> tuple[Tensor, Tensor]:
    """Per incoming edge: same-goal and opposite-goal masks.

    Returns:
        same, diff: each (N, N, 8) float in {0, 1}, from the *receiver's*
        perspective (slot k = neighbour k of the receiver).
    """
    sender_goals = gather_neighbours(state.goals.float())  # (N, N, 8)
    receiver_goals = state.goals.float().unsqueeze(-1)     # (N, N, 1)
    same = (sender_goals == receiver_goals).float()
    diff = 1.0 - same
    return same, diff


def _route_votes(
    help_votes: Tensor,
    harm_votes: Tensor,
    gate: Tensor,
    same: Tensor,
    diff: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply rho/alive gate and goal routing; sum over the 8 edges.

    Args:
        help_votes, harm_votes, gate, same, diff: each (N, N, 8)

    Returns:
        V_kin, V_foe: each (N, N)
    """
    V_kin = (help_votes * gate * same).sum(dim=2)
    V_foe = (harm_votes * gate * diff).sum(dim=2)
    return V_kin, V_foe


def message_pass(
    state: State,
    params: Parameters,
    typed_votes: bool = True,
) -> MessagePassOutput:
    """Compute messages from every neighbour to every cell.

    Steps:
        1. Build ψ inputs per directed edge.
        2. Apply ψ using the *sender's* weights.
        3. Split outputs into message vector (d), v_help, v_harm.
        4. Multiply messages by sender rho and alive flag.
        5. Aggregate votes:
             typed_votes=True  (version A): help→kin, harm→foe
             typed_votes=False (original):  help on all edges → V_kin; V_foe=0
        6. Aggregate messages over the 8 incoming edges.

    Returns:
        MessagePassOutput with aggregated_messages, V_kin, V_foe,
        outgoing_votes (N, N, 8, 2).
    """
    d = state.d
    out_psi = psi_out_dim(d)
    assert out_psi == d + 2

    psi_inputs = _build_psi_inputs(state)           # (N, N, 8, 4d+4)
    psi_out = _psi_forward_per_edge(psi_inputs, params)  # (N, N, 8, d+2)

    # Split into messages and dual votes.
    messages = psi_out[..., :d]                     # (N, N, 8, d)
    help_votes = psi_out[..., d]                    # (N, N, 8)
    harm_votes = psi_out[..., d + 1]                # (N, N, 8)

    # Gate by sender's rho and alive flag.
    # The k-th slot is what arrives from neighbour k of the receiver.
    sender_rho = gather_neighbours(state.rho)       # (N, N, 8)
    sender_alive = gather_neighbours(state.x)       # (N, N, 8)
    gate = sender_rho * sender_alive                # (N, N, 8)

    messages = messages * gate.unsqueeze(-1)

    if typed_votes:
        same, diff = _goal_match_mask(state)
        V_kin, V_foe = _route_votes(help_votes, harm_votes, gate, same, diff)
    else:
        # Original: one indiscriminate vote channel (use help head for all).
        V_kin = (help_votes * gate).sum(dim=2)
        V_foe = torch.zeros_like(V_kin)

    aggregated_messages = messages.sum(dim=2)       # (N, N, d)

    outgoing_votes = _compute_outgoing_votes(state, params)
    return MessagePassOutput(
        aggregated_messages=aggregated_messages,
        V_kin=V_kin,
        V_foe=V_foe,
        outgoing_votes=outgoing_votes,
    )


def _compute_outgoing_votes(state: State, params: Parameters) -> Tensor:
    """Compute (v_help, v_harm)_{i -> k} for each cell i and each outgoing edge.

    From cell i's POV, it sends to each of its 8 neighbours. The receiver
    of edge k is the cell at offset NEIGHBOUR_OFFSETS[k] from i. ψ takes
    (y_sender=i, y_receiver=neighbour-k-of-i).

    Returns:
        (N, N, 8, 2) — channel 0 = help, channel 1 = harm; gated by
        rho_i and x_i. Goal routing is *not* applied here (that depends on
        the receiver's goal match and is done at aggregation time).
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
    g_r = gather_neighbours(state.goals.float()).unsqueeze(-1)   # receiver goal
    g_i = state.goals.float().unsqueeze(-1).unsqueeze(2).expand(-1, -1, 8, -1)

    psi_in = torch.cat([s_i, h_i, x_i, g_i, s_r, h_r, x_r, g_r], dim=-1)  # (N,N,8,4d+4)

    # Apply ψ using i's OWN per-cell weights (no gather), broadcasting over edges.
    psi_out = batched_mlp(psi_in, params.psi_W1, params.psi_b1,
                          params.psi_W2, params.psi_b2)  # (N, N, 8, d+2)
    help_votes = psi_out[..., d]       # (N, N, 8)
    harm_votes = psi_out[..., d + 1]   # (N, N, 8)

    # Gate by sender's own rho and alive.
    gate = (state.rho * state.x).unsqueeze(-1)  # (N, N, 1)
    help_votes = help_votes * gate
    harm_votes = harm_votes * gate
    return torch.stack([help_votes, harm_votes], dim=-1)  # (N, N, 8, 2)


# Local update

def local_update(
    state: State,
    aggregated_messages: Tensor,
    params: Parameters,
    learn_messages: bool = False,
) -> tuple[Tensor, Tensor]:
    """Run f to produce proposed next (s, h) for every cell.

    f input: (own_s, own_h, own_x, aggregated_messages).  Shape (N,N, 3d+1)
    f output: (new_s, new_h).                              Shape (N,N, 2d)

    Returns (s_proposed, h_proposed), each (N, N, d). These are the
    *proposals* before survival is applied; surviving cells use them,
    dead cells get them zeroed out.

    Message gradient (`learn_messages`, default False):
      * False (Path 1 locality): detach M before f. Then s_proposed[i] only
        carries gradient to params.f[i]. ψ's message head gets no training
        signal; M still affects the forward pass as a frozen feature.
      * True: keep M live. Gradient of -p_i through u·s_i flows into the
        messages cell i received, and therefore into the *senders'* ψ
        message heads (one-hop leakage). Under losses.sum().backward() each
        cell's message head is trained by its neighbours' self-survival
        terms — which is the natural credit assignment for an influence
        channel. Vote-channel locality (V-detach trick) is unchanged.
    """
    d = state.d
    own_x = state.x.unsqueeze(-1)  # (N, N, 1)
    messages = aggregated_messages if learn_messages else aggregated_messages.detach()
    f_in = torch.cat(
        [state.s, state.h, own_x, messages],
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
    survival_inputs: SurvivalInputs    # A, R, E, V_kin, V_foe, f_signal
    outgoing_votes: Tensor             # (N, N, 8, 2) — help/harm per edge
    s_proposed: Tensor                 # (N, N, d)
    h_proposed: Tensor                 # (N, N, d)


def forward_step(state: State, params: Parameters, u: Tensor, cfg: Config) -> StepOutput:
    """One forward CA step. No gradient updates here.

    Order:
        1. Message pass: messages + typed vote aggregates.
        2. Local update: propose new s and h.
        3. Survival: A, R, E + V_kin, V_foe + f-signal; hard threshold.
        4. Mask: zero out s and h for cells that died.
    """
    mp = message_pass(state, params, typed_votes=cfg.typed_votes)
    s_proposed, h_proposed = local_update(
        state, mp.aggregated_messages, params, learn_messages=cfg.learn_messages
    )

    surv_in = compute_survival_inputs(
        state, mp.V_kin, mp.V_foe, s_proposed, u,
    )
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
