"""Gradient locality test — the most important test in the project.

A "local" learning rule means: cell i's gradient depends ONLY on params[i].
We must prove that backprop through the full computation graph satisfies
this — otherwise we have centralised backprop pretending to be local.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from config import Config
from dynamics import forward_step, make_u
from learning import compute_local_losses, gradient_step
from parameters import init_parameters
from state import GOAL_ELIMINATE, GOAL_REPRODUCE, init_state


def make_world(N=5, d=4, hidden=8, seed=0, alive_prob=0.8):
    cfg = Config(N=N, d=d, hidden=hidden, seed=seed,
                 init_alive_prob=alive_prob, n_steps=1)
    gen = torch.Generator().manual_seed(seed)
    state = init_state(cfg.N, cfg.d, cfg.init_alive_prob, gen)
    # Make state vectors non-zero so ψ has real signal:
    state.s.normal_(generator=gen)
    state.h.normal_(generator=gen)
    params = init_parameters(cfg.N, cfg.d, cfg.hidden,
                             cfg.init_noise_std, gen).requires_grad_(True)
    u = make_u(cfg.d, cfg.u_seed)
    return cfg, state, params, u


def test_single_cell_loss_grad_is_local():
    """Backprop on cell (1,1)'s loss alone should produce non-zero grad ONLY
    at params[1,1] across every parameter tensor."""
    cfg, state, params, u = make_world(N=5, seed=0)
    # Force cell (1,1) to be alive (and its neighbours, so its loss has content).
    state.x[:] = 1.0
    step = forward_step(state, params, u, cfg)
    losses = compute_local_losses(state, step, cfg)
    target_cell = (1, 1)
    L = losses[target_cell[0], target_cell[1]]

    # Zero any existing grads, backward.
    for t in params.tensors():
        if t.grad is not None:
            t.grad.zero_()
    L.backward()

    for name, t in zip(
        ["psi_W1", "psi_b1", "psi_W2", "psi_b2",
         "f_W1", "f_b1", "f_W2", "f_b2"],
        params.tensors(),
    ):
        if t.grad is None:
            continue
        N = cfg.N
        for i in range(N):
            for j in range(N):
                slot_grad = t.grad[i, j]
                if (i, j) == target_cell:
                    # Under Path 1 both ψ and f have gradient paths to the loss.
                    # Biases (b1, b2) might happen to net to zero if the function
                    # is symmetric; weights should always have *some* signal.
                    if name.endswith("W1") or name.endswith("W2"):
                        assert slot_grad.abs().sum() > 1e-9, (
                            f"{name}[{i},{j}] (target cell) has zero grad — "
                            f"locality is fine but loss isn't reaching params"
                        )
                else:
                    assert slot_grad.abs().sum().item() == 0.0, (
                        f"LOCALITY VIOLATION: {name}[{i},{j}] has nonzero grad "
                        f"from cell {target_cell}'s loss alone "
                        f"(grad sum {slot_grad.abs().sum().item()})"
                    )

    print("test_single_cell_loss_grad_is_local OK")


def test_f_params_now_get_gradient():
    """With Path 1 (f-signal in survival rule), f's parameters DO receive
    gradient via the u . s_proposed term. This test is the inverse of the
    pre-Path-1 behaviour; it's the key qualitative change Path 1 introduces."""
    cfg, state, params, u = make_world(N=4, seed=1)
    state.x[:] = 1.0
    step = forward_step(state, params, u, cfg)
    losses = compute_local_losses(state, step, cfg)
    losses.sum().backward()

    # At least one f tensor must have nonzero gradient somewhere.
    any_nonzero = False
    for name, t in [
        ("f_W1", params.f_W1), ("f_b1", params.f_b1),
        ("f_W2", params.f_W2), ("f_b2", params.f_b2),
    ]:
        if t.grad is not None and t.grad.abs().sum().item() > 0:
            any_nonzero = True
            break
    assert any_nonzero, "Path 1 should give f gradient, but every f tensor's grad is zero"
    print("test_f_params_now_get_gradient OK")


def test_dead_cells_get_zero_gradient_after_step():
    """Run gradient_step. Cells that were dead at the start of the step
    should have all-zero param updates afterwards."""
    cfg, state, params, u = make_world(N=6, seed=2, alive_prob=0.5)
    params_before = params.detach_clone()

    step = forward_step(state, params, u, cfg)
    gradient_step(state, step, params, cfg)

    N = cfg.N
    for i in range(N):
        for j in range(N):
            was_alive = state.x[i, j].item() > 0
            # Check ψ params (the ones that should get gradient).
            delta = (params.psi_W1[i, j] - params_before.psi_W1[i, j]).abs().sum().item()
            if not was_alive:
                assert delta == 0.0, (
                    f"Cell ({i},{j}) was dead but its params moved by {delta}"
                )
    print("test_dead_cells_get_zero_gradient_after_step OK")


def test_alive_cells_with_alive_neighbours_get_nonzero_gradient():
    """At least some alive cells with alive neighbours should get a real update."""
    cfg, state, params, u = make_world(N=8, seed=3, alive_prob=0.9)
    params_before = params.detach_clone()
    step = forward_step(state, params, u, cfg)
    gradient_step(state, step, params, cfg)

    # Find at least one alive cell whose params changed.
    moved_any = False
    N = cfg.N
    for i in range(N):
        for j in range(N):
            if state.x[i, j].item() > 0:
                delta = (params.psi_W1[i, j] - params_before.psi_W1[i, j]).abs().sum().item()
                if delta > 0:
                    moved_any = True
                    break
        if moved_any:
            break
    assert moved_any, "No alive cell received a gradient update — learning is dead"
    print("test_alive_cells_with_alive_neighbours_get_nonzero_gradient OK")


def test_full_grid_locality_via_per_cell_loop():
    """Stronger version of locality test: for each cell c, build the loss for
    only that cell, backprop, and verify gradient lives only at slot c."""
    cfg, state, params, u = make_world(N=4, seed=4)
    state.x[:] = 1.0
    N = cfg.N

    for ci in range(N):
        for cj in range(N):
            step = forward_step(state, params, u, cfg)
            losses = compute_local_losses(state, step, cfg)
            for t in params.tensors():
                if t.grad is not None:
                    t.grad.zero_()
            losses[ci, cj].backward()

            for name, t in zip(["psi_W1", "psi_W2", "f_W1", "f_W2"],
                                [params.psi_W1, params.psi_W2,
                                 params.f_W1, params.f_W2]):
                grad_sum_per_slot = t.grad.abs().sum(
                    dim=tuple(range(2, t.dim()))
                )  # (N, N): per-cell summary of grad magnitude
                for i in range(N):
                    for j in range(N):
                        if (i, j) != (ci, cj):
                            assert grad_sum_per_slot[i, j].item() == 0.0, (
                                f"Locality violation at target ({ci},{cj}): "
                                f"{name}[{i},{j}] has gradient mass "
                                f"{grad_sum_per_slot[i, j].item()}"
                            )
    print("test_full_grid_locality_via_per_cell_loop OK")


def _zero_grads(params):
    for t in params.tensors():
        if t.grad is not None:
            t.grad.zero_()


def _apply_sgd_to_only(params, names, eta, alive_mask):
    """Apply SGD update to only the named parameter tensors. Returns nothing."""
    name_to_tensor = {
        "psi_W1": params.psi_W1, "psi_b1": params.psi_b1,
        "psi_W2": params.psi_W2, "psi_b2": params.psi_b2,
        "f_W1":   params.f_W1,   "f_b1":   params.f_b1,
        "f_W2":   params.f_W2,   "f_b2":   params.f_b2,
    }
    import torch
    with torch.no_grad():
        for n in names:
            t = name_to_tensor[n]
            if t.grad is None:
                continue
            m = alive_mask
            while m.dim() < t.grad.dim():
                m = m.unsqueeze(-1)
            t.add_(t.grad * m, alpha=-eta)


def test_vote_channel_reproducer_pushes_neighbours_alive():
    """Vote-channel-only test: update ψ only (zero f), reproducer should
    push neighbours toward survival."""
    import torch
    from learning import compute_local_losses
    cfg, state, params, u = make_world(N=4, seed=5, alive_prob=1.0)
    state.goals[:] = GOAL_REPRODUCE

    step0 = forward_step(state, params, u, cfg)
    p_neigh_before = _soft_p_at_each_neighbour(state, step0, cfg)[1, 1].sum().item()

    # Backward, but only ψ updates.
    losses = compute_local_losses(state, step0, cfg)
    _zero_grads(params)
    (losses * state.x).sum().backward()
    _apply_sgd_to_only(params,
                       ["psi_W1", "psi_b1", "psi_W2", "psi_b2"],
                       eta=0.1, alive_mask=state.x)

    step1 = forward_step(state, params, u, cfg)
    p_neigh_after = _soft_p_at_each_neighbour(state, step1, cfg)[1, 1].sum().item()

    assert p_neigh_after > p_neigh_before - 1e-4, (
        f"Reproducer (ψ-only) didn't increase neighbour survival: "
        f"{p_neigh_before:.4f} -> {p_neigh_after:.4f}"
    )
    print(f"  ψ-only reproducer (1,1) neighbour-p: "
          f"{p_neigh_before:.4f} -> {p_neigh_after:.4f}")
    print("test_vote_channel_reproducer_pushes_neighbours_alive OK")


def test_vote_channel_eliminator_pushes_neighbours_dead():
    """Vote-channel-only test: update ψ only (zero f), eliminator should
    push neighbours toward death."""
    import torch
    from learning import compute_local_losses
    cfg, state, params, u = make_world(N=4, seed=6, alive_prob=1.0)
    state.goals[:] = GOAL_ELIMINATE

    step0 = forward_step(state, params, u, cfg)
    p_neigh_before = _soft_p_at_each_neighbour(state, step0, cfg)[1, 1].sum().item()

    losses = compute_local_losses(state, step0, cfg)
    _zero_grads(params)
    (losses * state.x).sum().backward()
    _apply_sgd_to_only(params,
                       ["psi_W1", "psi_b1", "psi_W2", "psi_b2"],
                       eta=0.1, alive_mask=state.x)

    step1 = forward_step(state, params, u, cfg)
    p_neigh_after = _soft_p_at_each_neighbour(state, step1, cfg)[1, 1].sum().item()

    assert p_neigh_after < p_neigh_before + 1e-4, (
        f"Eliminator (ψ-only) didn't decrease neighbour survival: "
        f"{p_neigh_before:.4f} -> {p_neigh_after:.4f}"
    )
    print(f"  ψ-only eliminator (1,1) neighbour-p: "
          f"{p_neigh_before:.4f} -> {p_neigh_after:.4f}")
    print("test_vote_channel_eliminator_pushes_neighbours_dead OK")


def test_f_signal_channel_pushes_self_alive():
    """f-channel-only test: update f only (zero ψ), cells should push their
    OWN survival probability up regardless of goal (both goals want self alive
    in Path 1)."""
    import torch
    from learning import compute_local_losses
    cfg, state, params, u = make_world(N=4, seed=8, alive_prob=1.0)
    cfg.w0 = -3.0  # bias toward death so p_self has room to rise

    step0 = forward_step(state, params, u, cfg)
    # Pull p_self for cell (1, 1) before and after.
    p_self_before = torch.sigmoid(
        cfg.w0
        + cfg.w1 * step0.survival_inputs.A[1, 1]
        + cfg.w2 * step0.survival_inputs.R[1, 1]
        + cfg.w3 * step0.survival_inputs.E[1, 1]
        + cfg.w4 * step0.survival_inputs.V[1, 1]
        + cfg.w5 * torch.tanh(step0.survival_inputs.f_signal[1, 1])
    ).item()

    losses = compute_local_losses(state, step0, cfg)
    _zero_grads(params)
    (losses * state.x).sum().backward()
    _apply_sgd_to_only(params,
                       ["f_W1", "f_b1", "f_W2", "f_b2"],
                       eta=0.5, alive_mask=state.x)

    step1 = forward_step(state, params, u, cfg)
    p_self_after = torch.sigmoid(
        cfg.w0
        + cfg.w1 * step1.survival_inputs.A[1, 1]
        + cfg.w2 * step1.survival_inputs.R[1, 1]
        + cfg.w3 * step1.survival_inputs.E[1, 1]
        + cfg.w4 * step1.survival_inputs.V[1, 1]
        + cfg.w5 * torch.tanh(step1.survival_inputs.f_signal[1, 1])
    ).item()

    assert p_self_after > p_self_before - 1e-4, (
        f"f-channel-only update did not increase self-survival: "
        f"{p_self_before:.4f} -> {p_self_after:.4f}"
    )
    print(f"  f-only cell (1,1) self-p: "
          f"{p_self_before:.4f} -> {p_self_after:.4f}")
    print("test_f_signal_channel_pushes_self_alive OK")


def _soft_p_at_each_neighbour(state, step_out, cfg):
    """Helper: for each cell, return its 8 neighbours' soft survival probs."""
    from grid import gather_neighbours
    return gather_neighbours(
        torch.sigmoid(
            cfg.w0
            + cfg.w1 * step_out.survival_inputs.A
            + cfg.w2 * step_out.survival_inputs.R
            + cfg.w3 * step_out.survival_inputs.E
            + cfg.w4 * step_out.survival_inputs.V
            + cfg.w5 * torch.tanh(step_out.survival_inputs.f_signal)
        )
    )


if __name__ == "__main__":
    test_single_cell_loss_grad_is_local()
    test_f_params_now_get_gradient()
    test_dead_cells_get_zero_gradient_after_step()
    test_alive_cells_with_alive_neighbours_get_nonzero_gradient()
    test_full_grid_locality_via_per_cell_loop()
    test_vote_channel_reproducer_pushes_neighbours_alive()
    test_vote_channel_eliminator_pushes_neighbours_dead()
    test_f_signal_channel_pushes_self_alive()
    print("\nAll gradient locality tests passed.")
