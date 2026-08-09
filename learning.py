"""Per-cell learning step.

The learning signal: cell i wants to maximise (reproduce) or minimise
(eliminate) the soft survival probabilities of its 8 neighbours at the
NEXT step. Loss is computed using outputs of the CURRENT step's message
pass, before the survival rule is applied (so that the soft probability
is a function of this step's votes).

Locality: cell i's gradient must depend only on params[i]. Other cells'
contributions to the vote sums are detached.

Typed votes (step A): survival uses two channels —
  V_kin = sum of help votes from same-goal neighbours
  V_foe = sum of harm votes from opposite-goal neighbours

outgoing_votes is (N, N, 8, 2) with [...,0]=help and [...,1]=harm (rho/x
gated, not goal-routed). When patching a neighbour's V, only the channel
that actually reaches that neighbour (kin→help or foe→harm) carries
gradient from this cell.

The result: loss.sum().backward() populates params.grad correctly with each
cell's gradient living only at its own (i, j) slot.
"""

from __future__ import annotations

import torch
from torch import Tensor

from config import Config
from dynamics import StepOutput
from grid import gather_neighbours
from parameters import Parameters
from state import GOAL_REPRODUCE, State

# In the canonical NEIGHBOUR_OFFSETS ordering, neighbour-k of cell i and i itself
# from that neighbour's perspective are at "opposite" slots: k_in = 7 - k_out.
# (This invariant is checked in tests/test_vote_consistency.py.)
OPPOSITE_SLOT = 7  # k_in = OPPOSITE_SLOT - k_out


def _survival_logit_parts(
    A: Tensor,
    R: Tensor,
    E: Tensor,
    V_kin: Tensor,
    V_foe: Tensor,
    f_signal: Tensor,
    cfg: Config,
) -> Tensor:
    """Linear survival logit from components (same formula as dynamics)."""
    return (
        cfg.w0
        + cfg.w1 * A
        + cfg.w2 * R
        + cfg.w3 * E
        + cfg.w4_help * V_kin
        + cfg.w4_harm * V_foe
        + cfg.w5 * torch.tanh(f_signal)
    )


def compute_p_self(
    state: State,
    step_out: StepOutput,
    cfg: Config,
) -> Tensor:
    """Soft self-survival probabilities with Path-1 detach rules.

    Gradient only through f_signal (this cell's f). Neighbour counts and both
    vote channels are detached. Used by the local loss and by the soft
    coexistence barrier (Experiment F).
    """
    surv_in = step_out.survival_inputs
    f_signal = surv_in.f_signal  # (N, N), differentiable through f
    logit_self = _survival_logit_parts(
        surv_in.A.detach(),
        surv_in.R.detach(),
        surv_in.E.detach(),
        surv_in.V_kin.detach(),
        surv_in.V_foe.detach(),
        f_signal,  # KEEP gradient through f
        cfg,
    )
    p_self = torch.sigmoid(logit_self)  # (N, N)
    if cfg.require_alive_neighbour:
        p_self = p_self * (surv_in.A > 0).float()
    return p_self


def coexistence_barrier(
    state: State,
    p_self: Tensor,
    cfg: Config,
) -> Tensor:
    """Soft two-log barrier on global type densities from self soft masses.

        B = λ ( -log(ρ̃^R + δ) - log(ρ̃^E + δ) )
        ρ̃^R = (∑_i p_i^{self} 1_{g_i=R}) / N²
        ρ̃^E = (∑_i p_i^{self} 1_{g_i=E}) / N²

    Gradient only through each cell's own p_i (self f-path). Add B once to
    the step total loss so λ does not scale with N_alive.
    """
    n_cells = float(cfg.N * cfg.N)
    is_r = (state.goals == GOAL_REPRODUCE).to(dtype=p_self.dtype)
    is_e = 1.0 - is_r
    tilde_r = (p_self * is_r).sum()
    tilde_e = (p_self * is_e).sum()
    delta = float(cfg.coexistence_delta)
    rho_r = tilde_r / n_cells
    rho_e = tilde_e / n_cells
    B = float(cfg.coexistence_lambda) * (
        -torch.log(rho_r + delta) - torch.log(rho_e + delta)
    )
    return B


def compute_local_losses(
    state: State,
    step_out: StepOutput,
    cfg: Config,
    *,
    p_self: Tensor | None = None,
) -> Tensor:
    """Compute per-cell local losses with gradient-locality detach tricks.

    Loss structure (Path 1 + typed neighbour term):

        L_i = -p_i + sum_k c_{i,k} * p_{nbr k}

    where c depends on goals. Reproducers always protect kin and pressure
    foes. Eliminators either pressure all neighbours (default) or, with
    cfg.predator_prey_loss (step B), only reproducer (prey) neighbours.
    The -p_i term is unsigned: both types want themselves alive.

    Soft coexistence (Experiment F) is *not* folded into the per-cell tensor;
    see ``coexistence_barrier`` / ``gradient_step`` (added once to total loss).

    Gradient paths (only ones we want to KEEP):
      * p_i through u . s_proposed_i  ->  this cell's f params
      * p_j through V_kin_j / V_foe_j ->  this cell's psi via the routed
                                          outgoing vote channel
      * p_i through M_i (if cfg.learn_messages): senders' ψ message heads
        (one-hop; see dynamics.local_update). Default off.

    Gradient paths we must KILL via detach:
      * p_i through V_kin_i / V_foe_i (votes IN to i)
      * p_j through u . s_proposed_j
      * p_i through M_i when learn_messages=False: message head stays frozen.

    Args:
        p_self: optional precomputed soft self-survival probs (same graph as
            used for the coexistence barrier). If None, computed here.

    Returns:
        Tensor of shape (N, N): per-cell losses. With learn_messages=False,
        losses.sum().backward() respects per-cell param locality. With
        learn_messages=True, self-survival terms also touch neighbours' ψ
        message heads (intentional).
    """
    surv_in = step_out.survival_inputs
    outgoing = step_out.outgoing_votes       # (N, N, 8, 2): help, harm

    # Term 1: -p_i (self-survival), with both vote channels detached.
    if p_self is None:
        p_self = compute_p_self(state, step_out, cfg)

    # Term 2: neighbours' p_j with V-detach trick on both channels.
    V_kin_at_neighbour = gather_neighbours(surv_in.V_kin)   # (N, N, 8)
    V_foe_at_neighbour = gather_neighbours(surv_in.V_foe)   # (N, N, 8)
    A_at_neighbour = gather_neighbours(surv_in.A)           # (N, N, 8)
    R_at_neighbour = gather_neighbours(surv_in.R)           # (N, N, 8)
    E_at_neighbour = gather_neighbours(surv_in.E)           # (N, N, 8)
    f_signal_at_neighbour = gather_neighbours(surv_in.f_signal)  # (N, N, 8)

    my_help = outgoing[..., 0]                              # (N, N, 8)
    my_harm = outgoing[..., 1]                              # (N, N, 8)

    if cfg.typed_votes:
        # Outgoing edge k reaches a neighbour whose goal matches mine?
        my_goals = state.goals.float().unsqueeze(-1)        # (N, N, 1)
        nbr_goals = gather_neighbours(state.goals.float())  # (N, N, 8)
        same_out = (my_goals == nbr_goals).float()          # (N, N, 8)
        diff_out = 1.0 - same_out
        # Only the routed channel reaches that neighbour:
        my_to_V_kin = my_help * same_out
        my_to_V_foe = my_harm * diff_out
    else:
        # Original: single indiscriminate vote (help head) reaches everyone.
        my_to_V_kin = my_help
        my_to_V_foe = torch.zeros_like(my_help)

    V_kin_patched = (
        V_kin_at_neighbour.detach() - my_to_V_kin.detach() + my_to_V_kin
    )
    V_foe_patched = (
        V_foe_at_neighbour.detach() - my_to_V_foe.detach() + my_to_V_foe
    )

    # f_signal at neighbours: depends on the neighbour's f, never mine.
    f_signal_at_neighbour_d = f_signal_at_neighbour.detach()

    logit_at_neighbour = _survival_logit_parts(
        A_at_neighbour.detach(),
        R_at_neighbour.detach(),
        E_at_neighbour.detach(),
        V_kin_patched,
        V_foe_patched,
        f_signal_at_neighbour_d,
        cfg,
    )
    p_at_neighbour = torch.sigmoid(logit_at_neighbour)  # (N, N, 8)
    if cfg.require_alive_neighbour:
        p_at_neighbour = p_at_neighbour * (A_at_neighbour > 0).float()

    # =========================================================================
    # Combine: self term (always wants self alive) + signed neighbour term.
    # =========================================================================
    # Neighbour coefficients c[i,k] multiply p_at_neighbour[i,k].
    # Negative ⇒ wants neighbour alive; positive ⇒ wants neighbour dead.
    #
    # Reproducer (always kin-selective):
    #   -1 on reproducer neighbours (protect kin)
    #   +1 on eliminator neighbours (pressure foes)
    #
    # Eliminator:
    #   default (predator_prey_loss=False): +1 on everyone (indiscriminate kill)
    #   step B   (predator_prey_loss=True):  +1 on reproducer neighbours only
    #                                       (prey); 0 on fellow eliminators
    nb_is_repro = gather_neighbours(
        (state.goals == GOAL_REPRODUCE).float()
    )                                                    # (N, N, 8)
    i_is_repro = (state.goals == GOAL_REPRODUCE).float().unsqueeze(-1)  # (N, N, 1)

    coef_repro = -1.0 * nb_is_repro + 1.0 * (1.0 - nb_is_repro)
    if cfg.predator_prey_loss:
        # Predator–prey: elim only pressures prey (reproducers).
        coef_elim = 1.0 * nb_is_repro
    else:
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

    When ``cfg.coexistence_pressure`` is on, a soft type-mass barrier B is
    added **once** to the step total (not broadcast per cell):

        total = (losses * alive).sum() + B

    so λ is independent of N_alive. B depends on soft self-masses only
    (Path-1 local through each cell's f).

    Returns a dict of summary statistics for logging.
    """
    # Shared p_self graph for local loss and optional coexistence barrier.
    p_self = compute_p_self(state, step_out, cfg)
    losses = compute_local_losses(state, step_out, cfg, p_self=p_self)  # (N, N)

    # Mask: only alive cells contribute. Multiplying loss by x_i means dead
    # cells have zero contribution -> zero gradient at their parameter slot.
    alive_mask = state.x  # (N, N), float 0/1
    masked_total = (losses * alive_mask).sum()

    barrier = None
    if cfg.coexistence_pressure and cfg.coexistence_lambda > 0:
        barrier = coexistence_barrier(state, p_self, cfg)
        masked_total = masked_total + barrier

    # Zero any pre-existing grads on params.
    for t in params.tensors():
        if t.grad is not None:
            t.grad.zero_()

    masked_total.backward()

    # SGD: each param slot updates by -eta * grad. The grad is already zero
    # for dead cells (since their loss was masked to 0). We still apply the
    # mask explicitly so the update is correct even if a future change adds
    # gradient paths that bypass the alive multiplication.
    # Coexistence barrier gradients also only touch each cell's own f (via
    # p_self); we still mask so dead cells never update.
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
    # Soft type masses (detached) for diagnostics.
    with torch.no_grad():
        n_cells = float(cfg.N * cfg.N)
        is_r = (state.goals == GOAL_REPRODUCE).float()
        tilde_r = float((p_self.detach() * is_r).sum().item())
        tilde_e = float((p_self.detach() * (1.0 - is_r)).sum().item())
        soft_rho_r = tilde_r / n_cells
        soft_rho_e = tilde_e / n_cells
    return {
        "loss_total": float(masked_total.detach().item()),
        "loss_reproduce_mean": repro_loss,
        "loss_eliminate_mean": elim_loss,
        "n_alive": int(state.x.sum().item()),
        "coexistence_barrier": (
            float(barrier.detach().item()) if barrier is not None else 0.0
        ),
        "soft_rho_R": soft_rho_r,
        "soft_rho_E": soft_rho_e,
    }
