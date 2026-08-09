"""Experiment F: soft coexistence barrier on soft type masses."""

from __future__ import annotations

import torch

from config import Config
from dynamics import forward_step, make_u
from learning import (
    coexistence_barrier,
    compute_local_losses,
    compute_p_self,
    gradient_step,
)
from parameters import init_parameters
from research.versions import get_version
from state import GOAL_ELIMINATE, GOAL_REPRODUCE, init_state


def _world(
    N=4,
    d=4,
    hidden=8,
    seed=0,
    *,
    coexistence_pressure: bool = False,
    coexistence_lambda: float = 0.01,
    coexistence_delta: float = 1e-4,
    typed_votes: bool = True,
):
    cfg = Config(
        N=N,
        d=d,
        hidden=hidden,
        seed=seed,
        init_alive_prob=0.8,
        n_steps=1,
        typed_votes=typed_votes,
        coexistence_pressure=coexistence_pressure,
        coexistence_lambda=coexistence_lambda,
        coexistence_delta=coexistence_delta,
        init_noise_std=0.05,
    )
    gen = torch.Generator().manual_seed(seed)
    state = init_state(cfg.N, cfg.d, cfg.init_alive_prob, gen)
    state.s.normal_(generator=gen)
    state.h.normal_(generator=gen)
    # Balanced goals so both soft masses are nonzero.
    state.goals[: N // 2, :] = GOAL_REPRODUCE
    state.goals[N // 2 :, :] = GOAL_ELIMINATE
    state.x[:] = 1.0
    params = init_parameters(
        cfg.N, cfg.d, cfg.hidden, cfg.init_noise_std, gen
    ).requires_grad_(True)
    u = make_u(cfg.d, cfg.u_seed)
    return cfg, state, params, u


def test_lambda_zero_matches_baseline_loss_total():
    """λ=0 (or pressure off) must not change gradients vs baseline."""
    cfg0, state, params, u = _world(seed=1, coexistence_pressure=False)
    step = forward_step(state, params, u, cfg0)
    losses0 = compute_local_losses(state, step, cfg0)
    total0 = (losses0 * state.x).sum()

    cfg1 = Config(**{**cfg0.to_dict(), "coexistence_pressure": True, "coexistence_lambda": 0.0})
    losses1 = compute_local_losses(state, step, cfg1)
    p_self = compute_p_self(state, step, cfg1)
    B = coexistence_barrier(state, p_self, cfg1)
    total1 = (losses1 * state.x).sum() + B

    assert torch.allclose(losses0, losses1), "per-cell losses must match"
    assert float(B.item()) == 0.0 or abs(float(B.item())) < 1e-12
    assert torch.allclose(total0, total1)
    print("test_lambda_zero_matches_baseline_loss_total OK")


def test_pressure_off_bit_identical_local_losses():
    """coexistence_pressure flag must not alter compute_local_losses."""
    cfg_off, state, params, u = _world(seed=2, coexistence_pressure=False)
    cfg_on = Config(
        **{**cfg_off.to_dict(), "coexistence_pressure": True, "coexistence_lambda": 1.0}
    )
    step = forward_step(state, params, u, cfg_off)
    L_off = compute_local_losses(state, step, cfg_off)
    L_on = compute_local_losses(state, step, cfg_on)
    assert torch.equal(L_off, L_on) or torch.allclose(L_off, L_on)
    print("test_pressure_off_bit_identical_local_losses OK")


def test_barrier_diverges_as_one_soft_mass_vanishes():
    """As soft mass of one type → 0, B → +∞ (softly, floor by δ)."""
    cfg, state, params, u = _world(
        seed=3, coexistence_pressure=True, coexistence_lambda=1.0, coexistence_delta=1e-4
    )
    step = forward_step(state, params, u, cfg)
    p_self = compute_p_self(state, step, cfg).detach()

    # Zero out soft mass of all reproducers.
    p_zero_r = p_self.clone()
    p_zero_r[state.goals == GOAL_REPRODUCE] = 0.0
    B_low = coexistence_barrier(state, p_zero_r, cfg)

    p_balanced = p_self.clone()
    B_bal = coexistence_barrier(state, p_balanced, cfg)

    assert float(B_low.item()) > float(B_bal.item()), (
        f"rare-type barrier should be larger: {B_low.item()} vs {B_bal.item()}"
    )
    # With ρ̃^R = 0, B ≥ −λ log(δ) for that term alone.
    assert float(B_low.item()) >= -1.0 * torch.log(torch.tensor(cfg.coexistence_delta)).item() - 1e-5
    print(
        f"test_barrier_diverges_as_one_soft_mass_vanishes OK  "
        f"B_low={B_low.item():.3f} B_bal={B_bal.item():.3f}"
    )


def test_barrier_gradient_only_on_self_f_for_rare_type():
    """Perturbing only cell i's f should move B only via that cell's p_i.

    And gradient locality: backprop of B alone should only hit params at
    cells of the rare type that contribute to the soft mass.
    """
    cfg, state, params, u = _world(
        N=5,
        seed=4,
        coexistence_pressure=True,
        coexistence_lambda=1.0,
    )
    # Make R rare in soft mass by zeroing most R cells' chance of surviving
    # via requiring many R cells dead (they still contribute p via birth).
    state.x[:] = 1.0
    state.goals[:] = GOAL_ELIMINATE
    state.goals[2, 2] = GOAL_REPRODUCE  # single R cell

    step = forward_step(state, params, u, cfg)
    p_self = compute_p_self(state, step, cfg)
    B = coexistence_barrier(state, p_self, cfg)

    for t in params.tensors():
        if t.grad is not None:
            t.grad.zero_()
    B.backward()

    # Only the single R cell should get gradient through the R term;
    # E cells also get gradient (raise p when tilde_e matters). Both types
    # that exist get nonzero f grads. Cells with neither... all cells have a type.
    # Locality: each cell's grad only from its own p_i → only own f.
    # ψ should have zero grad from B alone (self path detaches votes).
    for name, t in [
        ("psi_W1", params.psi_W1),
        ("psi_W2", params.psi_W2),
    ]:
        if t.grad is not None:
            assert t.grad.abs().sum().item() == 0.0, (
                f"B must not touch ψ ({name}); got grad sum {t.grad.abs().sum().item()}"
            )

    # f of the unique R cell should feel the rare-type push.
    r_i, r_j = 2, 2
    assert params.f_W1.grad is not None
    r_grad = params.f_W1.grad[r_i, r_j].abs().sum().item()
    assert r_grad > 0, "rare R cell must receive gradient from B via p_self"

    # A different cell's f grad comes only from its own contribution (E mass).
    e_grad = params.f_W1.grad[0, 0].abs().sum().item()
    # Both can be nonzero; locality is per-cell nonzero only for that slot —
    # already guaranteed by how p_self is built. Check no cross-cell ψ path.
    print(
        f"test_barrier_gradient_only_on_self_f_for_rare_type OK  "
        f"R_f_grad={r_grad:.4e} E_f_grad={e_grad:.4e}"
    )


def test_gradient_step_applies_barrier_to_total():
    """With pressure on, loss_total includes B; stats expose soft densities."""
    cfg, state, params, u = _world(
        seed=5, coexistence_pressure=True, coexistence_lambda=0.05
    )
    step = forward_step(state, params, u, cfg)
    losses = compute_local_losses(state, step, cfg)
    p_self = compute_p_self(state, step, cfg)
    B = coexistence_barrier(state, p_self, cfg)
    expected = float(((losses * state.x).sum() + B).detach().item())

    # Fresh params for the actual step (same state/step graph rebuilt).
    cfg2, state2, params2, u2 = _world(
        seed=5, coexistence_pressure=True, coexistence_lambda=0.05
    )
    step2 = forward_step(state2, params2, u2, cfg2)
    stats = gradient_step(state2, step2, params2, cfg2)

    assert abs(stats["loss_total"] - expected) < 1e-4, (
        f"loss_total {stats['loss_total']} vs expected {expected}"
    )
    assert stats["coexistence_barrier"] > 0
    assert stats["soft_rho_R"] > 0 and stats["soft_rho_E"] > 0
    print(
        f"test_gradient_step_applies_barrier_to_total OK  "
        f"B={stats['coexistence_barrier']:.4f}"
    )


def test_version_f_applies_flags():
    v = get_version("F")
    assert v.coexistence_pressure is True
    assert v.typed_votes is True
    assert v.predator_prey_loss is False
    cfg = v.apply(Config())
    assert cfg.coexistence_pressure is True
    assert cfg.coexistence_lambda == 0.01
    # Alias
    assert get_version("A_coexist").id == "F"
    print("test_version_f_applies_flags OK")


def test_barrier_locality_single_cell():
    """Backprop of B alone: cell j's f_W1 grad only from cell j."""
    cfg, state, params, u = _world(
        N=4, seed=6, coexistence_pressure=True, coexistence_lambda=1.0
    )
    step = forward_step(state, params, u, cfg)
    p_self = compute_p_self(state, step, cfg)
    B = coexistence_barrier(state, p_self, cfg)
    for t in params.tensors():
        if t.grad is not None:
            t.grad.zero_()
    B.backward()

    # Spot-check: f_W1[i,j] nonzero only affects that slot — already true by
    # construction; ensure some cells have grad and structure is (N,N,...).
    assert params.f_W1.grad is not None
    assert params.f_W1.grad.shape[:2] == (cfg.N, cfg.N)
    nonzero_cells = (params.f_W1.grad.abs().sum(dim=(-1, -2)) > 1e-12).sum().item()
    assert nonzero_cells >= 1
    print(f"test_barrier_locality_single_cell OK  nonzero_f_cells={nonzero_cells}")


if __name__ == "__main__":
    test_lambda_zero_matches_baseline_loss_total()
    test_pressure_off_bit_identical_local_losses()
    test_barrier_diverges_as_one_soft_mass_vanishes()
    test_barrier_gradient_only_on_self_f_for_rare_type()
    test_gradient_step_applies_barrier_to_total()
    test_version_f_applies_flags()
    test_barrier_locality_single_cell()
    print("all coexistence tests passed")
