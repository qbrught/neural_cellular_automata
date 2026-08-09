"""Step D: goal_in_f feeds own goal into local update f."""

from __future__ import annotations

import torch

from config import Config
from dynamics import forward_step, local_update, make_u, message_pass
from parameters import f_in_dim, init_parameters
from simulate import build_grid, set_global_seed
from state import GOAL_ELIMINATE, GOAL_REPRODUCE, init_state


def _world(seed: int = 0, *, goal_in_f: bool = False, N: int = 6):
    cfg = Config(
        N=N,
        d=4,
        hidden=8,
        n_steps=5,
        learn=True,
        typed_votes=True,
        predator_prey_loss=True,
        goal_inheritance=False,
        goal_in_f=goal_in_f,
        seed=seed,
        init_noise_std=0.05,
        init_alive_prob=0.4,
    )
    gen = set_global_seed(seed)
    state = init_state(cfg.N, cfg.d, cfg.init_alive_prob, gen, device=cfg.device)
    state.s.normal_(0.0, cfg.init_noise_std, generator=gen)
    state.h.normal_(0.0, cfg.init_noise_std, generator=gen)
    # Force a balanced, non-degenerate goal map for the test.
    state.goals[: N // 2, :] = GOAL_REPRODUCE
    state.goals[N // 2 :, :] = GOAL_ELIMINATE
    params = init_parameters(
        cfg.N, cfg.d, cfg.hidden, cfg.init_noise_std, gen, device=cfg.device
    ).requires_grad_(True)
    u = make_u(cfg.d, cfg.u_seed, device=cfg.device)
    return cfg, state, params, u


def test_f_in_dim_includes_goal_slot():
    d = 8
    assert f_in_dim(d) == 3 * d + 2


def test_goal_in_f_changes_proposal_vs_zero_slot():
    """Same weights/state: goal feature on vs off must change s_proposed."""
    cfg_off, state, params, u = _world(seed=11, goal_in_f=False)
    mp = message_pass(state, params, typed_votes=True)
    s_off, h_off = local_update(
        state, mp.aggregated_messages, params, goal_in_f=False
    )
    s_on, h_on = local_update(
        state, mp.aggregated_messages, params, goal_in_f=True
    )
    # Goal weights are random at init → nonzero contribution when goal≠0.
    delta_s = (s_on - s_off).abs().sum().item()
    delta_h = (h_on - h_off).abs().sum().item()
    assert delta_s + delta_h > 1e-6, (
        "goal_in_f should change f outputs when goals are nonzero"
    )
    print(f"test_goal_in_f_changes_proposal OK  Δs+Δh={delta_s + delta_h:.4f}")


def test_goal_in_f_type_swap_breaks_symmetry_of_f():
    """Swapping goals changes f outputs only when goal_in_f is on."""
    from dataclasses import replace
    from state import State

    cfg, state, params, u = _world(seed=3, goal_in_f=True)
    mp = message_pass(state, params, typed_votes=True)
    s0, _ = local_update(state, mp.aggregated_messages, params, goal_in_f=True)

    state_swapped = State(
        x=state.x,
        s=state.s,
        h=state.h,
        goals=1 - state.goals,
        rho=state.rho,
    )
    s1, _ = local_update(
        state_swapped, mp.aggregated_messages, params, goal_in_f=True
    )
    assert (s0 - s1).abs().sum().item() > 1e-6

    # Off: goal slot zero → swap should not change f input → same outputs.
    s0b, _ = local_update(state, mp.aggregated_messages, params, goal_in_f=False)
    s1b, _ = local_update(
        state_swapped, mp.aggregated_messages, params, goal_in_f=False
    )
    assert (s0b - s1b).abs().max().item() < 1e-6
    print("test_goal_in_f_type_swap_breaks_symmetry_of_f OK")


def test_forward_step_respects_goal_in_f_flag():
    from dataclasses import replace

    cfg_off, state, params, u = _world(seed=5, goal_in_f=False)
    out_off = forward_step(state, params, u, cfg_off)
    cfg_on = replace(cfg_off, goal_in_f=True)
    out_on = forward_step(state, params, u, cfg_on)
    d = (out_off.s_proposed - out_on.s_proposed).abs().sum().item()
    assert d > 1e-6
    print(f"test_forward_step_respects_goal_in_f_flag OK  Δs={d:.4f}")


def test_version_d_runs():
    from research.versions import get_version
    from research.runner import run_experiment

    base = Config(N=8, d=4, hidden=8, n_steps=20, learn=True, seed=0)
    for vid in ("D_fixed", "D"):
        spec = get_version(vid)
        assert spec.implemented and spec.goal_in_f
        r = run_experiment(spec, base, seed=0, n_steps=15)
        assert "summary" in r
        assert r["summary"]["T"] == 15
    print("test_version_d_runs OK")


if __name__ == "__main__":
    test_f_in_dim_includes_goal_slot()
    test_goal_in_f_changes_proposal_vs_zero_slot()
    test_goal_in_f_type_swap_breaks_symmetry_of_f()
    test_forward_step_respects_goal_in_f_flag()
    test_version_d_runs()
    print("all goal_in_f tests OK")
