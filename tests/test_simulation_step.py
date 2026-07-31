"""Test the full forward step (no learning yet)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from config import Config
from dynamics import forward_step, inherit_goals, make_u
from parameters import init_parameters
from state import GOAL_ELIMINATE, GOAL_REPRODUCE, State, init_state


def fresh_grid(N=10, d=4, hidden=8, seed=0, alive_prob=0.5, **cfg_kw):
    cfg = Config(
        N=N, d=d, hidden=hidden, seed=seed, init_alive_prob=alive_prob,
        n_steps=1, **cfg_kw,
    )
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


def test_goals_immutable_when_inheritance_off():
    """goal_inheritance=False → goals unchanged after a step (pre-C behavior)."""
    cfg, state, params, u = fresh_grid(goal_inheritance=False)
    g0 = state.goals.clone()
    out = forward_step(state, params, u, cfg)
    assert torch.equal(out.next_state.goals, g0)
    print("test_goals_immutable_when_inheritance_off OK")


def test_rho_always_immutable():
    """rho is fixed for the whole run regardless of goal_inheritance."""
    for flag in (False, True):
        cfg, state, params, u = fresh_grid(goal_inheritance=flag)
        r0 = state.rho.clone()
        out = forward_step(state, params, u, cfg)
        assert torch.equal(out.next_state.rho, r0), (
            f"rho mutated with goal_inheritance={flag}"
        )
    print("test_rho_always_immutable OK")


def test_inherit_goals_majority_synthetic():
    """Unit test of inherit_goals without a full sim: majority elim wins."""
    N = 5
    cfg = Config(N=N, d=4, hidden=8, goal_inheritance=True, n_steps=1)
    x = torch.zeros(N, N)
    # Surround center (2,2) with 5 alive eliminators and 1 alive reproducer.
    # Moore nbrs of (2,2): all cells in [1:4, 1:4] except center.
    goals = torch.zeros(N, N, dtype=torch.long)  # default REPRODUCE
    # Five elim alive on NW,N,NE,W,E of center; one repro on S.
    for i, j in [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3)]:
        x[i, j] = 1.0
        goals[i, j] = GOAL_ELIMINATE
    x[3, 2] = 1.0
    goals[3, 2] = GOAL_REPRODUCE
    # Center dead with latent REPRODUCE goal.
    x[2, 2] = 0.0
    goals[2, 2] = GOAL_REPRODUCE
    rho = torch.full((N, N), 0.5)
    state = State(
        x=x,
        s=torch.zeros(N, N, cfg.d),
        h=torch.zeros(N, N, cfg.d),
        goals=goals,
        rho=rho,
    )
    x_next = x.clone()
    x_next[2, 2] = 1.0  # force birth at center

    g_next = inherit_goals(state, x_next, cfg)
    assert int(g_next[2, 2].item()) == GOAL_ELIMINATE, (
        f"center should inherit ELIMINATE (majority), got {g_next[2, 2].item()}"
    )
    # Non-birth cells keep goals.
    mask = torch.ones(N, N, dtype=torch.bool)
    mask[2, 2] = False
    assert torch.equal(g_next[mask], goals[mask])
    print("test_inherit_goals_majority_synthetic OK")


def test_birth_inherits_majority_goal():
    """Full forward_step: dead center with majority elim neighbours → ELIMINATE."""
    N = 5
    # Force survival whenever A>0 so center with alive nbrs will birth.
    cfg = Config(
        N=N, d=4, hidden=8, seed=0, n_steps=1,
        goal_inheritance=True,
        require_alive_neighbour=True,
        w0=10.0, w1=0.0, w2=0.0, w3=0.0,
        w4_help=0.0, w4_harm=0.0, w5=0.0,
        typed_votes=True,
    )
    gen = torch.Generator().manual_seed(0)
    state = init_state(N, cfg.d, 0.0, gen)  # all dead initially
    # Build ring of elim majority around center.
    state.x[:] = 0.0
    state.goals[:] = GOAL_REPRODUCE
    for i, j in [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3)]:
        state.x[i, j] = 1.0
        state.goals[i, j] = GOAL_ELIMINATE
    state.x[3, 2] = 1.0
    state.goals[3, 2] = GOAL_REPRODUCE
    state.x[2, 2] = 0.0
    state.goals[2, 2] = GOAL_REPRODUCE
    state.s.normal_(std=0.01, generator=gen)
    state.h.normal_(std=0.01, generator=gen)
    params = init_parameters(N, cfg.d, cfg.hidden, cfg.init_noise_std, gen)
    u = make_u(cfg.d, cfg.u_seed)

    out = forward_step(state, params, u, cfg)
    assert out.next_state.x[2, 2].item() == 1.0, "center should birth"
    assert int(out.next_state.goals[2, 2].item()) == GOAL_ELIMINATE
    print("test_birth_inherits_majority_goal OK")


def test_surviving_cell_keeps_goal():
    """Alive→alive cell does not flip goal even if surrounded by opposite type."""
    N = 5
    cfg = Config(
        N=N, d=4, hidden=8, seed=1, n_steps=1,
        goal_inheritance=True,
        require_alive_neighbour=True,
        w0=10.0, w1=0.0, w2=0.0, w3=0.0,
        w4_help=0.0, w4_harm=0.0, w5=0.0,
    )
    gen = torch.Generator().manual_seed(1)
    state = init_state(N, cfg.d, 0.0, gen)
    state.x[:] = 0.0
    state.goals[:] = GOAL_ELIMINATE
    # Center alive REPRODUCE, all 8 neighbours alive ELIMINATE.
    state.x[2, 2] = 1.0
    state.goals[2, 2] = GOAL_REPRODUCE
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            state.x[2 + di, 2 + dj] = 1.0
            state.goals[2 + di, 2 + dj] = GOAL_ELIMINATE
    state.s.normal_(std=0.01, generator=gen)
    state.h.normal_(std=0.01, generator=gen)
    params = init_parameters(N, cfg.d, cfg.hidden, cfg.init_noise_std, gen)
    u = make_u(cfg.d, cfg.u_seed)

    out = forward_step(state, params, u, cfg)
    assert out.next_state.x[2, 2].item() == 1.0
    assert int(out.next_state.goals[2, 2].item()) == GOAL_REPRODUCE
    print("test_surviving_cell_keeps_goal OK")


def test_death_keeps_latent_goal():
    """Alive→dead keeps previous goal (latent while dead)."""
    N = 5
    # Force death: large negative bias, no neighbour requirement needed.
    cfg = Config(
        N=N, d=4, hidden=8, seed=2, n_steps=1,
        goal_inheritance=True,
        require_alive_neighbour=False,
        w0=-10.0, w1=0.0, w2=0.0, w3=0.0,
        w4_help=0.0, w4_harm=0.0, w5=0.0,
    )
    gen = torch.Generator().manual_seed(2)
    state = init_state(N, cfg.d, 1.0, gen)  # all alive
    state.goals[2, 2] = GOAL_ELIMINATE
    g0 = state.goals[2, 2].item()
    params = init_parameters(N, cfg.d, cfg.hidden, cfg.init_noise_std, gen)
    u = make_u(cfg.d, cfg.u_seed)

    out = forward_step(state, params, u, cfg)
    assert out.next_state.x[2, 2].item() == 0.0
    assert int(out.next_state.goals[2, 2].item()) == g0
    print("test_death_keeps_latent_goal OK")


def test_inherit_goals_tie_breaks_by_max_rho():
    """Equal n_repro / n_elim → parent is max-rho alive neighbour."""
    N = 3
    cfg = Config(N=N, d=4, hidden=8, goal_inheritance=True, n_steps=1)
    x = torch.zeros(N, N)
    goals = torch.zeros(N, N, dtype=torch.long)
    rho = torch.full((N, N), 0.1)
    # Center dead. Two alive nbrs: E=elim rho=0.9, W=repro rho=0.2.
    # With toroidal Moore, also other nbrs dead.
    x[1, 0] = 1.0
    goals[1, 0] = GOAL_REPRODUCE
    rho[1, 0] = 0.2
    x[1, 2] = 1.0
    goals[1, 2] = GOAL_ELIMINATE
    rho[1, 2] = 0.9
    x[1, 1] = 0.0
    goals[1, 1] = GOAL_REPRODUCE  # latent
    state = State(
        x=x,
        s=torch.zeros(N, N, cfg.d),
        h=torch.zeros(N, N, cfg.d),
        goals=goals,
        rho=rho,
    )
    x_next = x.clone()
    x_next[1, 1] = 1.0
    g_next = inherit_goals(state, x_next, cfg)
    assert int(g_next[1, 1].item()) == GOAL_ELIMINATE
    print("test_inherit_goals_tie_breaks_by_max_rho OK")


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
    x0, s0, h0, g0 = (
        state.x.clone(), state.s.clone(), state.h.clone(), state.goals.clone(),
    )
    _ = forward_step(state, params, u, cfg)
    assert torch.equal(state.x, x0)
    assert torch.equal(state.s, s0)
    assert torch.equal(state.h, h0)
    assert torch.equal(state.goals, g0)
    print("test_step_does_not_modify_input_state OK")


if __name__ == "__main__":
    test_step_output_shapes()
    test_step_no_nans()
    test_dead_cells_have_zero_state()
    test_goals_immutable_when_inheritance_off()
    test_rho_always_immutable()
    test_inherit_goals_majority_synthetic()
    test_birth_inherits_majority_goal()
    test_surviving_cell_keeps_goal()
    test_death_keeps_latent_goal()
    test_inherit_goals_tie_breaks_by_max_rho()
    test_alive_x_values_in_zero_one()
    test_zero_alive_grid_stays_dead()
    test_step_does_not_modify_input_state()
    test_multistep_run_remains_bounded()
    print("\nAll simulation step tests passed.")
