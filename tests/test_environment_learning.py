"""Experiment G per-cell η scale: freeze, type-specific, G+F, locality."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from config import Config
from dynamics import forward_step, make_u
from environment import generate_environment, identity_environment
from learning import compute_local_losses, gradient_step
from parameters import init_parameters
from state import GOAL_ELIMINATE, GOAL_REPRODUCE, init_state


def _world(N=5, seed=0, alive_prob=1.0, **cfg_kw):
    cfg = Config(
        N=N, d=4, hidden=8, seed=seed, init_alive_prob=alive_prob, n_steps=1, **cfg_kw,
    )
    gen = torch.Generator().manual_seed(seed)
    state = init_state(cfg.N, cfg.d, cfg.init_alive_prob, gen)
    state.s.normal_(generator=gen)
    state.h.normal_(generator=gen)
    params = init_parameters(
        cfg.N, cfg.d, cfg.hidden, cfg.init_noise_std, gen,
    ).requires_grad_(True)
    u = make_u(cfg.d, cfg.u_seed)
    return cfg, state, params, u


def _clone_tensors(params):
    return [t.detach().clone() for t in params.tensors()]


def test_type_specific_eta_freezes_reproducers():
    cfg, state, params, u = _world(
        N=6, seed=1, environment_heterogeneous=True, eta=0.05,
    )
    state.x[:] = 1.0
    state.goals[:] = GOAL_ELIMINATE
    state.goals[:, :3] = GOAL_REPRODUCE
    env = identity_environment(6)
    env.eta_scale_R[:] = 0.0
    env.eta_scale_E[:] = 1.0
    before = _clone_tensors(params)
    step = forward_step(state, params, u, cfg, env=env)
    stats = gradient_step(state, step, params, cfg, env=env)
    r = state.goals == GOAL_REPRODUCE
    e = state.goals == GOAL_ELIMINATE
    for t, b in zip(params.tensors(), before):
        assert torch.equal(t[r], b[r]), "reproducer params must be frozen"
        assert (t[e] - b[e]).abs().sum().item() > 0, "eliminator params must move"
    assert abs(stats["eta_mean_alive"] - 0.5 * cfg.eta) < 1e-8
    print("test_type_specific_eta_freezes_reproducers OK")


def test_eta_scale_zero_is_full_freeze():
    cfg, state, params, u = _world(
        N=4, seed=2, environment_heterogeneous=True, eta=0.1,
    )
    state.x[:] = 1.0
    env = identity_environment(4)
    env.eta_scale_R[:] = 0.0
    env.eta_scale_E[:] = 0.0
    before = _clone_tensors(params)
    step = forward_step(state, params, u, cfg, env=env)
    gradient_step(state, step, params, cfg, env=env)
    for t, b in zip(params.tensors(), before):
        assert torch.equal(t, b)
    print("test_eta_scale_zero_is_full_freeze OK")


def test_gf_eta_zero_slot_deaf_to_barrier():
    """T4b: η_scale=0 slot does not move with coexistence pressure on."""
    cfg, state, params, u = _world(
        N=4, seed=3, environment_heterogeneous=True, eta=0.05,
        coexistence_pressure=True, coexistence_lambda=1.0,
    )
    state.x[:] = 1.0
    # Balanced goals so B is live.
    state.goals[:] = GOAL_ELIMINATE
    state.goals[:2, :] = GOAL_REPRODUCE
    env = identity_environment(4)
    env.eta_scale_R[:] = 1.0
    env.eta_scale_E[:] = 1.0
    env.eta_scale_R[1, 1] = 0.0
    env.eta_scale_E[1, 1] = 0.0
    before = _clone_tensors(params)
    step = forward_step(state, params, u, cfg, env=env)
    stats = gradient_step(state, step, params, cfg, env=env)
    assert stats["coexistence_barrier"] > 0.0
    for t, b in zip(params.tensors(), before):
        assert torch.equal(t[1, 1], b[1, 1]), "frozen slot moved under F barrier"
        # Some other alive cell with η=1 must move (proves B/loss is live).
        delta = (t - b).abs()
        delta[1, 1] = 0
        assert delta.sum().item() > 0, "no other cell moved; barrier may be dead"
    print("test_gf_eta_zero_slot_deaf_to_barrier OK")


def test_flag_off_sgd_unchanged():
    cfg, state, params, u = _world(N=4, seed=4, eta=0.02)
    p_skip = params.detach_clone().requires_grad_(True)
    p_omit = params.detach_clone().requires_grad_(True)
    s1 = forward_step(state, p_skip, u, cfg, env=None)
    gradient_step(state, s1, p_skip, cfg, env=None)
    s2 = forward_step(state, p_omit, u, cfg)
    gradient_step(state, s2, p_omit, cfg)
    for a, b in zip(p_skip.tensors(), p_omit.tensors()):
        assert torch.equal(a, b)
    print("test_flag_off_sgd_unchanged OK")


def test_locality_with_blobs_env():
    """T10: Path-1 locality still holds with a blobs environment."""
    cfg, state, params, u = _world(
        N=5, seed=0, alive_prob=1.0,
        environment_heterogeneous=True, env_preset="blobs",
        env_n_blobs=3, env_blob_radius=0.2, env_seed=1,
    )
    env = generate_environment(cfg)
    state.x[:] = 1.0
    step = forward_step(state, params, u, cfg, env=env)
    losses = compute_local_losses(state, step, cfg)
    target = (1, 1)
    L = losses[target]
    for t in params.tensors():
        if t.grad is not None:
            t.grad.zero_()
    L.backward()
    for name, t in zip(
        ["psi_W1", "psi_b1", "psi_W2", "psi_b2", "f_W1", "f_b1", "f_W2", "f_b2"],
        params.tensors(),
    ):
        if t.grad is None:
            continue
        for i in range(cfg.N):
            for j in range(cfg.N):
                slot = t.grad[i, j]
                if (i, j) == target:
                    if name.endswith("W1") or name.endswith("W2"):
                        assert slot.abs().sum() > 1e-9, f"{name} target has zero grad"
                else:
                    assert slot.abs().sum().item() == 0.0, (
                        f"LOCALITY VIOLATION: {name}[{i},{j}] from {target}"
                    )
    print("test_locality_with_blobs_env OK")


if __name__ == "__main__":
    test_type_specific_eta_freezes_reproducers()
    test_eta_scale_zero_is_full_freeze()
    test_gf_eta_zero_slot_deaf_to_barrier()
    test_flag_off_sgd_unchanged()
    test_locality_with_blobs_env()
    print("\nAll environment learning tests passed.")
