"""Experiment G dynamics: transfer barriers, occupancy, skip-path identity."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from config import Config
from dynamics import forward_step, make_u, message_pass
from environment import (
    apply_occupancy,
    edge_kappa_product,
    generate_environment,
    identity_environment,
)
from grid import NEIGHBOUR_OFFSETS
from learning import gradient_step
from parameters import init_parameters, Parameters
from simulate import build_grid, run
from state import GOAL_ELIMINATE, GOAL_REPRODUCE, init_state


def _clone_params(params: Parameters) -> Parameters:
    return params.detach_clone().requires_grad_(True)


def _world(N=5, seed=0, alive_prob=1.0, **cfg_kw):
    cfg = Config(N=N, d=4, hidden=8, seed=seed, init_alive_prob=alive_prob, n_steps=1, **cfg_kw)
    gen = torch.Generator().manual_seed(seed)
    state = init_state(cfg.N, cfg.d, cfg.init_alive_prob, gen)
    state.s.normal_(generator=gen)
    state.h.normal_(generator=gen)
    params = init_parameters(cfg.N, cfg.d, cfg.hidden, cfg.init_noise_std, gen).requires_grad_(True)
    u = make_u(cfg.d, cfg.u_seed)
    return cfg, state, params, u


def test_flag_off_matches_no_env_kwargs():
    cfg, state, params, u = _world(N=5, seed=1)
    assert cfg.environment_heterogeneous is False
    out_none = forward_step(state, params, u, cfg, env=None)
    out_omit = forward_step(state, params, u, cfg)
    assert torch.equal(out_none.next_state.x, out_omit.next_state.x)
    assert torch.equal(out_none.next_state.s, out_omit.next_state.s)
    assert torch.equal(out_none.next_state.h, out_omit.next_state.h)
    assert torch.equal(out_none.survival_inputs.V_kin, out_omit.survival_inputs.V_kin)
    assert torch.equal(out_none.survival_inputs.V_foe, out_omit.survival_inputs.V_foe)
    assert torch.equal(out_none.outgoing_votes, out_omit.outgoing_votes)

    p1 = _clone_params(params)
    p2 = _clone_params(params)
    # Recompute a fresh step on clones so grads are independent.
    s1 = forward_step(state, p1, u, cfg, env=None)
    gradient_step(state, s1, p1, cfg, env=None)
    s2 = forward_step(state, p2, u, cfg)
    gradient_step(state, s2, p2, cfg)
    for a, b in zip(p1.tensors(), p2.tensors()):
        assert torch.equal(a, b)
    print("test_flag_off_matches_no_env_kwargs OK")


def test_flag_on_without_env_raises():
    cfg, state, params, u = _world(N=4, seed=0, environment_heterogeneous=True)
    try:
        forward_step(state, params, u, cfg, env=None)
        raise AssertionError("expected AssertionError")
    except AssertionError as e:
        assert "environment_heterogeneous" in str(e)
    step = forward_step(state, params, u, cfg, env=identity_environment(4))
    try:
        gradient_step(state, step, params, cfg, env=None)
        raise AssertionError("expected AssertionError")
    except AssertionError as e:
        assert "environment_heterogeneous" in str(e)
    print("test_flag_on_without_env_raises OK")


def test_dead_zone_blocks_messages_and_votes():
    cfg, state, params, u = _world(N=5, seed=2)
    state.x[:] = 1.0
    env = identity_environment(5)
    env.kappa_R[2, 2] = 0.0
    env.kappa_E[2, 2] = 0.0
    mp = message_pass(state, params, env=env)
    assert mp.outgoing_votes[2, 2].abs().sum().item() == 0.0
    assert mp.aggregated_messages[2, 2].abs().sum().item() == 0.0
    assert mp.V_kin[2, 2].item() == 0.0
    assert mp.V_foe[2, 2].item() == 0.0
    # Outgoing *into* (2,2) is zero: each neighbour's slot toward (2,2).
    for k_out, (di, dj) in enumerate(NEIGHBOUR_OFFSETS):
        ni, nj = (2 + di) % 5, (2 + dj) % 5
        k_in = 7 - k_out
        assert mp.outgoing_votes[ni, nj, k_in].abs().sum().item() == 0.0
    print("test_dead_zone_blocks_messages_and_votes OK")


def test_simulate_run_threads_env():
    """T2b: simulate.run with G on actually applies κ (would fail if run forgot env=)."""
    cfg = Config(
        N=8, d=4, hidden=8, n_steps=2, seed=5, learn=False,
        save_state_vectors=False, output_dir=tempfile.mkdtemp(),
        environment_heterogeneous=True, env_preset="vertical_band",
        env_dead_frac=0.25, env_kappa_lo=0.0, env_kappa_hi=1.0,
        env_occupancy_blocks=False,
    )
    out = run(cfg, verbose=False)
    traj = np.load(out / "trajectory.npz")
    assert "kappa_R" in traj
    grid = build_grid(cfg)
    prod = edge_kappa_product(grid.state, grid.env)
    # Any edge with an endpoint on a κ=0 cell is 0.
    dead = grid.env.kappa_R == 0
    assert dead.any(), "band should paint at least one column"
    # Crossing edges: slot k at a live cell whose neighbour is dead-κ.
    from grid import gather_neighbours
    nbr_dead = gather_neighbours(dead.float()) > 0
    self_dead = dead.unsqueeze(-1)
    blocked = (self_dead | nbr_dead)
    assert (prod[blocked.expand_as(prod)] == 0).all()
    # Flag-on without threading would still write maps; assert dynamics used them
    # by checking a live step's outgoing votes on the band are zero.
    state, params, u, env = grid.state, grid.params, grid.u, grid.env
    state.x[:] = 1.0
    mp = message_pass(state, params, env=env)
    band = dead
    assert mp.outgoing_votes[band].abs().sum().item() == 0.0
    # If simulate.run forgot env=, this wouldn't crash — the maps still exist —
    # so also assert the production assert: calling with flag on and env=None fails.
    raised = False
    try:
        forward_step(state, params, u, cfg, env=None)
    except AssertionError:
        raised = True
    assert raised, "forward_step must assert when flag is on and env is None"
    print("test_simulate_run_threads_env OK")


def test_type_specific_transfer():
    cfg, state, params, u = _world(N=6, seed=3)
    state.x[:] = 1.0
    state.goals[:] = GOAL_ELIMINATE
    state.goals[:, :3] = GOAL_REPRODUCE
    env = identity_environment(6)
    env.kappa_R[:] = 0.0
    env.kappa_E[:] = 1.0
    mp = message_pass(state, params, env=env)
    r_mask = state.goals == GOAL_REPRODUCE
    e_mask = state.goals == GOAL_ELIMINATE
    assert mp.outgoing_votes[r_mask].abs().sum().item() == 0.0
    # Eliminator-eliminator edges (right half interior) should still send.
    assert mp.outgoing_votes[e_mask].abs().sum().item() > 0.0
    muted = generate_environment(Config(
        N=8, environment_heterogeneous=True, env_preset="blobs",
        env_n_blobs=3, env_blob_radius=0.2, env_kappa_lo=0.0,
        env_affect_R=False, env_affect_E=True, env_seed=1,
    ))
    assert torch.equal(muted.kappa_R, torch.ones(8, 8))
    assert not torch.equal(muted.kappa_E, torch.ones(8, 8))
    print("test_type_specific_transfer OK")


def test_occupancy_after_forward_step():
    cfg, state, params, u = _world(N=5, seed=4, w0=10.0, require_alive_neighbour=False)
    state.x[:] = 1.0
    env = identity_environment(5)
    env.occupancy[2, 2] = 0.0
    cfg = Config(
        N=5, d=4, hidden=8, seed=4, w0=10.0, require_alive_neighbour=False,
        environment_heterogeneous=True, n_steps=1,
    )
    out = forward_step(state, params, u, cfg, env=env)
    assert out.next_state.x[2, 2].item() == 0.0
    assert torch.equal(out.next_state.s[2, 2], torch.zeros(cfg.d))
    assert torch.equal(out.next_state.h[2, 2], torch.zeros(cfg.d))
    print("test_occupancy_after_forward_step OK")


def test_rho_still_multiplies():
    cfg, state, params, u = _world(N=4, seed=6)
    state.x[:] = 1.0
    state.goals[:] = GOAL_REPRODUCE
    # Case A: κ=1, ρ=0 → no send.
    state.rho[:] = 0.0
    env_k1 = identity_environment(4)
    mp0 = message_pass(state, params, env=env_k1)
    assert mp0.outgoing_votes.abs().sum().item() == 0.0
    assert mp0.aggregated_messages.abs().sum().item() == 0.0
    # Case B: ρ=1, κ=0 → no send.
    state.rho[:] = 1.0
    env_k0 = identity_environment(4)
    env_k0.kappa_R[:] = 0.0
    env_k0.kappa_E[:] = 0.0
    mp_k0 = message_pass(state, params, env=env_k0)
    assert mp_k0.outgoing_votes.abs().sum().item() == 0.0
    # Case C: ρ=1, κ=0.5 everywhere → κ_prod = 0.25, matches ρ=0.25 with κ=1.
    state.rho[:] = 1.0
    env_half = identity_environment(4)
    env_half.kappa_R[:] = 0.5
    env_half.kappa_E[:] = 0.5
    mp_half = message_pass(state, params, env=env_half)
    state.rho[:] = 0.25
    mp_ref = message_pass(state, params, env=None)
    assert torch.allclose(mp_half.outgoing_votes, mp_ref.outgoing_votes, atol=1e-5)
    assert torch.allclose(mp_half.V_kin, mp_ref.V_kin, atol=1e-5)
    print("test_rho_still_multiplies OK")


def test_outgoing_uses_kappa_not_incoming_rhox():
    """T11: outgoing is ρ_i x_i * κ_prod, not gather(ρ)*gather(x)*κ_prod."""
    cfg, state, params, u = _world(N=5, seed=8)
    state.x[:] = 1.0
    state.goals[:] = GOAL_REPRODUCE
    state.rho[:] = 1.0
    # Neighbour of (1,1) to the east is (1,2). Give that neighbour ρ ≠ 1.
    state.rho[1, 2] = 0.3
    env = identity_environment(5)
    env.kappa_R[1, 1] = 0.5
    env.kappa_E[1, 1] = 0.5
    env.kappa_R[1, 2] = 0.8
    env.kappa_E[1, 2] = 0.8
    mp = message_pass(state, params, env=env)
    # East is NEIGHBOUR_OFFSETS index 4: (0, 1)
    k_out = 4
    k_in = 7 - k_out
    sent = mp.outgoing_votes[1, 1, k_out, 0]
    # Incoming at (1,2), slot k_in, is from (1,1).
    # Reconstruct incoming help from V isn't needed: compare outgoing to
    # the paired incoming gate * raw vote. Use the vote-consistency pairing.
    from dynamics import _build_psi_inputs, _psi_forward_per_edge
    from grid import gather_neighbours
    d = state.d
    psi_out = _psi_forward_per_edge(_build_psi_inputs(state), params)
    help_raw = psi_out[..., d]
    kappa_prod = edge_kappa_product(state, env)
    incoming_gate = (
        gather_neighbours(state.rho) * gather_neighbours(state.x) * kappa_prod
    )
    got = help_raw[1, 2, k_in] * incoming_gate[1, 2, k_in]
    assert torch.allclose(sent, got, atol=1e-5), (sent.item(), got.item())
    # The wrong helper (incoming ρxκ used as outgoing) would multiply extra ρ_nbr.
    wrong = sent * state.rho[1, 2]  # extra ρ_nbr=0.3
    assert not torch.allclose(sent, wrong, atol=1e-4)
    print("test_outgoing_uses_kappa_not_incoming_rhox OK")


def test_apply_occupancy_helper():
    x = torch.ones(3, 3)
    assert torch.equal(apply_occupancy(x, None), x)
    print("test_apply_occupancy_helper OK")


if __name__ == "__main__":
    test_flag_off_matches_no_env_kwargs()
    test_flag_on_without_env_raises()
    test_dead_zone_blocks_messages_and_votes()
    test_simulate_run_threads_env()
    test_type_specific_transfer()
    test_occupancy_after_forward_step()
    test_rho_still_multiplies()
    test_outgoing_uses_kappa_not_incoming_rhox()
    test_apply_occupancy_helper()
    print("\nAll environment dynamics tests passed.")
