"""Test the full forward step (no learning yet)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from config import Config
from dynamics import forward_step, make_u
from parameters import init_parameters
from state import init_state


def fresh_grid(N=10, d=4, hidden=8, seed=0, alive_prob=0.5):
    cfg = Config(N=N, d=d, hidden=hidden, seed=seed, init_alive_prob=alive_prob,
                 n_steps=1)
    gen = torch.Generator().manual_seed(seed)
    state = init_state(cfg.N, cfg.d, cfg.init_alive_prob, gen)
    params = init_parameters(cfg.N, cfg.d, cfg.hidden, cfg.init_noise_std, gen)
    u = make_u(cfg.d, cfg.u_seed)
    return cfg, state, params, u


def test_step_output_shapes():
    cfg, state, params, u = fresh_grid(N=8, d=4)
    out = forward_step(state, params, u, cfg)
    N, d = cfg.N, cfg.d
    assert out.next_state.x.shape == (N, N)
    assert out.next_state.s.shape == (N, N, d)
    assert out.next_state.h.shape == (N, N, d)
    assert out.outgoing_votes.shape == (N, N, 8, 2)
    assert out.s_proposed.shape == (N, N, d)
    assert out.h_proposed.shape == (N, N, d)
    for t in [out.survival_inputs.A, out.survival_inputs.R,
              out.survival_inputs.E, out.survival_inputs.V_kin,
              out.survival_inputs.V_foe]:
        assert t.shape == (N, N)
    print("test_step_output_shapes OK")


def test_step_no_nans():
    cfg, state, params, u = fresh_grid(N=10, d=8, hidden=16)
    out = forward_step(state, params, u, cfg)
    assert not torch.isnan(out.next_state.x).any()
    assert not torch.isnan(out.next_state.s).any()
    assert not torch.isnan(out.next_state.h).any()
    assert not torch.isnan(out.outgoing_votes).any()
    print("test_step_no_nans OK")


def test_dead_cells_have_zero_state():
    """After a step, every cell with x_next == 0 must have s_next == 0 and h_next == 0."""
    cfg, state, params, u = fresh_grid(N=10, d=4)
    out = forward_step(state, params, u, cfg)
    dead = (out.next_state.x == 0.0)
    assert (out.next_state.s[dead] == 0.0).all()
    assert (out.next_state.h[dead] == 0.0).all()
    print("test_dead_cells_have_zero_state OK")


def test_goals_and_rho_are_immutable():
    cfg, state, params, u = fresh_grid()
    g0 = state.goals.clone()
    r0 = state.rho.clone()
    out = forward_step(state, params, u, cfg)
    assert torch.equal(out.next_state.goals, g0)
    assert torch.equal(out.next_state.rho, r0)
    print("test_goals_and_rho_are_immutable OK")


def test_alive_x_values_in_zero_one():
    cfg, state, params, u = fresh_grid()
    out = forward_step(state, params, u, cfg)
    vals = out.next_state.x.unique()
    assert set(vals.tolist()).issubset({0.0, 1.0})
    print("test_alive_x_values_in_zero_one OK")


def test_multistep_run_remains_bounded():
    """Run 200 steps; alive count should stay in a sensible range, no NaNs anywhere."""
    cfg, state, params, u = fresh_grid(N=20, d=8, hidden=16, seed=42)
    alive_history = []
    N = cfg.N

    for t in range(200):
        out = forward_step(state, params, u, cfg)
        state = out.next_state
        alive = int(state.x.sum().item())
        alive_history.append(alive)
        assert not torch.isnan(state.s).any(), f"NaN in s at step {t}"
        assert not torch.isnan(state.h).any(), f"NaN in h at step {t}"
        # State vectors should remain bounded (tanh output, gated, summed over 8 neighbours)
        assert state.s.abs().max() < 1e3, f"s exploded at step {t}: max {state.s.abs().max().item()}"
        assert state.h.abs().max() < 1e3, f"h exploded at step {t}: max {state.h.abs().max().item()}"

    print(f"  Alive trajectory (every 20 steps): "
          f"{[alive_history[i] for i in range(0, 200, 20)]}")
    print(f"  Final alive: {alive_history[-1]}, total cells: {N*N}")
    print("test_multistep_run_remains_bounded OK")


def test_zero_alive_grid_stays_dead():
    """An all-dead grid has A=R=E=V=0, so logit=w0=-1<0 -> all stay dead."""
    cfg, state, params, u = fresh_grid(N=8, d=4, alive_prob=0.0)
    assert state.x.sum().item() == 0
    out = forward_step(state, params, u, cfg)
    assert out.next_state.x.sum().item() == 0
    print("test_zero_alive_grid_stays_dead OK")


def test_step_does_not_modify_input_state():
    """forward_step is pure: input state tensors should be unchanged afterward."""
    cfg, state, params, u = fresh_grid()
    x0, s0, h0 = state.x.clone(), state.s.clone(), state.h.clone()
    _ = forward_step(state, params, u, cfg)
    assert torch.equal(state.x, x0)
    assert torch.equal(state.s, s0)
    assert torch.equal(state.h, h0)
    print("test_step_does_not_modify_input_state OK")


if __name__ == "__main__":
    test_step_output_shapes()
    test_step_no_nans()
    test_dead_cells_have_zero_state()
    test_goals_and_rho_are_immutable()
    test_alive_x_values_in_zero_one()
    test_zero_alive_grid_stays_dead()
    test_step_does_not_modify_input_state()
    test_multistep_run_remains_bounded()
    print("\nAll simulation step tests passed.")
