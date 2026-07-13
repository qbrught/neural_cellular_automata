"""Per-cell learning step.

The learning signal: cell i wants to maximise (reproduce) or minimise
(eliminate) the soft survival probabilities of its 8 neighbours at the
NEXT step. Loss is computed using outputs of the CURRENT step's message
pass, before the survival rule is applied (so that the soft probability
is a function of this step's votes).

Locality: cell i's gradient must depend only on params[i]. Other cells'
contributions to the vote sums are detached.

Implementation: we already have `outgoing_votes` of shape (N, N, 8) where
entry [i, j, k_out] = v_{(i,j) -> neighbour-k_out-of-(i,j)}. This tensor's
[i, j, :] slice depends only on params[i, j]. For each cell, we construct
its loss using (V_neighbour.detach() + my_outgoing_to_neighbour - my_outgoing_to_neighbour.detach()),
which equals V_neighbour numerically but, in the backward graph, only the
own-outgoing-vote term carries gradient.

The result: loss.sum().backward() populates params.grad correctly with each
cell's gradient living only at its own (i, j) slot.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from config import Config
from dynamics import (
    StepOutput,
    SurvivalInputs,
    soft_survival,
    compute_survival_inputs,
)
from grid import NEIGHBOUR_OFFSETS, gather_neighbours
from parameters import Parameters
from state import GOAL_REPRODUCE, State


# In the canonical NEIGHBOUR_OFFSETS ordering, neighbour-k of cell i and i itself
# from that neighbour's perspective are at "opposite" slots: k_in = 7 - k_out.
# (This invariant is checked in tests/test_vote_consistency.py.)
OPPOSITE_SLOT = 7  # k_in = OPPOSITE_SLOT - k_out


def compute_local_losses(
    state: State,
    step_out: StepOutput,
    cfg: Config,
) -> Tensor:
    """Compute per-cell local losses with gradient-locality detach tricks.

    Loss structure (Path 1, with self-survival term):

        L_i = -p_i  +  sign(g_i) * sum_{j in N(i)} p_j

    where sign(g_i) = -1 for reproducers, +1 for eliminators. The -p_i term
    is unsigned across goals: both reproducers and eliminators prefer to be
    alive themselves (so they can act next step). They differ only on what
    they want for their neighbours.

    Gradient paths (only ones we want to KEEP):
      * p_i through u . s_proposed_i  ->  this cell's f params      (NEW in Path 1)
      * p_j (j neighbour) through V_j ->  this cell's psi params via outgoing vote
                                          (existing detach trick)

    Gradient paths we must KILL via detach:
      * p_i through V_i (votes IN to i): would leak into 8 neighbours' psi.
      * p_j through u . s_proposed_j: would leak into j's f params.

    Returns:
        Tensor of shape (N, N): per-cell losses. Sum -> .backward() respects
        locality automatically.
    """
    N = state.N
    surv_in = step_out.survival_inputs       # A, R, E, V, f_signal at step t
    outgoing = step_out.outgoing_votes       # (N, N, 8): v_{i -> nbr-k}
    s_proposed = step_out.s_proposed         # (N, N, d): f's proposed outputs

    # We need to compute the projection u . s_proposed but with controlled
    # gradient flow. We don't have u here directly -- it's encoded as the
    # f_signal field of surv_in already (f_signal = (s_proposed * u).sum(-1)).
    # Good: that means f_signal is the per-cell scalar, and we just need to
    # decide whether to detach it.
    f_signal = surv_in.f_signal              # (N, N), differentiable through f

    # Term 1: -p_i (self-survival), with V_i detached.
    # We want p_i with gradient ONLY through f_signal_i, not through V_i.
    V_self_detached = surv_in.V.detach()
    f_signal_self_kept = f_signal             # KEEP gradient through f
    A_d = surv_in.A.detach()
    R_d = surv_in.R.detach()
    E_d = surv_in.E.detach()

    logit_self = (
        cfg.w0
        + cfg.w1 * A_d
        + cfg.w2 * R_d
        + cfg.w3 * E_d
        + cfg.w4 * V_self_detached
        + cfg.w5 * torch.tanh(f_signal_self_kept)
    )
    p_self = torch.sigmoid(logit_self)        # (N, N)
    if cfg.require_alive_neighbour:
        p_self = p_self * (surv_in.A > 0).float()
    # Term 2: signed sum over neighbours of p_j, with the existing V-detach
    # trick plus new f_signal detach.
    V_at_neighbour = gather_neighbours(surv_in.V)       # (N, N, 8)
    A_at_neighbour = gather_neighbours(surv_in.A)       # (N, N, 8)
    R_at_neighbour = gather_neighbours(surv_in.R)       # (N, N, 8)
    E_at_neighbour = gather_neighbours(surv_in.E)       # (N, N, 8)
    f_signal_at_neighbour = gather_neighbours(f_signal) # (N, N, 8)

    # V-detach trick on neighbours' V: keep only my own outgoing vote.
    my_vote = outgoing                                  # (N, N, 8), differentiable
    V_patched = (
        V_at_neighbour.detach()
        - my_vote.detach()
        + my_vote
    )                                                   # (N, N, 8)

    # f_signal at neighbours: depends on the neighbour's f, never mine.
    # Detach entirely.
    f_signal_at_neighbour_d = f_signal_at_neighbour.detach()

    logit_at_neighbour = (
        cfg.w0
        + cfg.w1 * A_at_neighbour.detach()
        + cfg.w2 * R_at_neighbour.detach()
        + cfg.w3 * E_at_neighbour.detach()
        + cfg.w4 * V_patched
        + cfg.w5 * torch.tanh(f_signal_at_neighbour_d)
    )
    p_at_neighbour = torch.sigmoid(logit_at_neighbour)  # (N, N, 8)
    if cfg.require_alive_neighbour:
        p_at_neighbour = p_at_neighbour * (A_at_neighbour > 0).float()

    # =========================================================================
    # Combine: self term (always wants self alive) + signed neighbour term.
    # =========================================================================
    # KIN-SELECTIVE LOSS.
    #   reproducer i: wants REPRODUCER neighbours alive, ELIMINATOR neighbours dead
    #   eliminator i: wants ALL neighbours dead (unchanged)
    # Per-neighbour coefficient c[i, k] multiplies p_at_neighbour[i, k].
    nb_is_repro = gather_neighbours(
        (state.goals == GOAL_REPRODUCE).float()
    )                                                    # (N, N, 8)
    i_is_repro = (state.goals == GOAL_REPRODUCE).float().unsqueeze(-1)  # (N, N, 1)

    # reproducer's coefficients: -1 on reproducer neighbours, +1 on eliminators
    coef_repro = -1.0 * nb_is_repro + 1.0 * (1.0 - nb_is_repro)
    # eliminator's coefficients: +1 on everyone (kill all)
    coef_elim = torch.ones_like(nb_is_repro)

    coef = i_is_repro * coef_repro + (1.0 - i_is_repro) * coef_elim   # (N, N, 8)
    weighted_neighbours = (coef * p_at_neighbour).sum(dim=2)          # (N, N)

    self_term = -p_self                                  # both goals want self alive
    return self_term + weighted_neighbours               # (N, N)


def gradient_step(
    state: State,
    step_out: StepOutput,
    params: Parameters,
    cfg: Config,
) -> dict[str, float]:
    """Compute per-cell gradients and apply masked SGD.

    Only ALIVE cells at the current step update their parameters
    (dead cells do nothing, per spec).

    Returns a dict of summary statistics for logging.
    """
    losses = compute_local_losses(state, step_out, cfg)   # (N, N)

    # Mask: only alive cells contribute. Multiplying loss by x_i means dead
    # cells have zero contribution -> zero gradient at their parameter slot.
    alive_mask = state.x  # (N, N), float 0/1
    masked_total = (losses * alive_mask).sum()

    # Zero any pre-existing grads on params.
    for t in params.tensors():
        if t.grad is not None:
            t.grad.zero_()

    masked_total.backward()

    # SGD: each param slot updates by -eta * grad. The grad is already zero
    # for dead cells (since their loss was masked to 0). We still apply the
    # mask explicitly so the update is correct even if a future change adds
    # gradient paths that bypass the alive multiplication.
    eta = cfg.eta
    with torch.no_grad():
        for t in params.tensors():
            if t.grad is None:
                continue
            # Broadcast (N, N) alive mask over the parameter's trailing dims.
            m = alive_mask
            while m.dim() < t.grad.dim():
                m = m.unsqueeze(-1)
            t.add_(t.grad * m, alpha=-eta)

    # Summary stats for logging.
    repro_mask = state.reproduce_mask() & (state.x > 0)
    elim_mask = state.eliminate_mask() & (state.x > 0)
    repro_loss = (
        losses[repro_mask].mean().item() if repro_mask.any() else float("nan")
    )
    elim_loss = (
        losses[elim_mask].mean().item() if elim_mask.any() else float("nan")
    )
    return {
        "loss_total": float(masked_total.detach().item()),
        "loss_reproduce_mean": repro_loss,
        "loss_eliminate_mean": elim_loss,
        "n_alive": int(state.x.sum().item()),
    }
