"""Tests for cell-level functional class divergence (probe bank + Δ)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from parameters import init_parameters, psi_out_dim
from research.functional import (
    FAMILY_COMMON,
    FAMILY_REALIZED,
    FAMILY_WEIGHTS_ONLY,
    VOTE_PROBES,
    build_common_probe_inputs,
    build_vote_probe_inputs,
    class_distance_stats,
    common_kin_mask,
    common_probe_names,
    evaluate_snapshot,
    pairwise_euclidean,
    psi_on_probes,
    realized_votes,
    zscore,
)
from state import GOAL_ELIMINATE, GOAL_REPRODUCE, State


def _world(N=4, d=4, hidden=8, seed=0, noise=0.0):
    gen = torch.Generator().manual_seed(seed)
    goals = torch.zeros(N, N, dtype=torch.long)
    goals[:, N // 2 :] = GOAL_ELIMINATE
    x = torch.ones(N, N)
    s = torch.zeros(N, N, d)
    h = torch.zeros(N, N, d)
    rho = torch.ones(N, N) * 0.5
    state = State(x=x, s=s, h=h, goals=goals, rho=rho)
    params = init_parameters(N, d, hidden, noise, gen)
    u = torch.randn(d, generator=gen)
    return state, params, u


def _clone_params_from_origin(params):
    for t in params.tensors():
        base = t[0, 0].detach().clone()
        t.copy_(base.view(1, 1, *base.shape).expand_as(t).contiguous())
    return params


def test_vote_probe_input_layout():
    N, d = 3, 4
    goals = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    psi_in = build_vote_probe_inputs(goals, d)
    assert psi_in.shape == (N, N, len(VOTE_PROBES), 4 * d + 4)
    # Sender x=1, s=h=0 for every probe.
    assert torch.allclose(psi_in[..., :d], torch.zeros(N, N, len(VOTE_PROBES), d))
    x_s = psi_in[..., 2 * d]
    assert torch.allclose(x_s, torch.ones_like(x_s))
    g_s = psi_in[..., 2 * d + 1]
    assert torch.allclose(g_s, goals.float().unsqueeze(-1).expand_as(g_s))
    # kin probe 0: receiver goal matches sender.
    g_r0 = psi_in[:, :, 0, -1]
    assert torch.allclose(g_r0, goals.float())
    # foe probe 1: receiver goal flipped.
    g_r1 = psi_in[:, :, 1, -1]
    assert torch.allclose(g_r1, 1.0 - goals.float())


def test_identical_weights_zero_weights_only_delta():
    state, params, u = _world(noise=0.01)
    _clone_params_from_origin(params)
    snap = evaluate_snapshot(
        params, state, u, typed_votes=True, goal_in_f=False, embed=False
    )
    w = snap.families[FAMILY_WEIGHTS_ONLY]
    # Goal bits are zeroed, weights identical → every cell the same vector.
    assert np.allclose(w.X, w.X[0:1], atol=1e-5)
    assert abs(w.stats_all["delta"]) < 1e-6 or not np.isfinite(w.stats_all["delta"])
    # Realized votes still see own goal, so R vs E inputs differ.
    r = snap.families[FAMILY_REALIZED]
    assert r.X.shape[1] == len(VOTE_PROBES)
    # Shared-question bank: identical weights → identical vectors, Δ ~ 0.
    c = snap.families[FAMILY_COMMON]
    assert c.X.shape[1] == 2 * len(VOTE_PROBES)
    assert np.allclose(c.X, c.X[0:1], atol=1e-5)
    assert abs(c.stats_all["delta"]) < 1e-6 or not np.isfinite(c.stats_all["delta"])
    # Own-goal realized saturates: two point masses, ARI = 1.
    assert r.stats_all["delta"] > 4.0
    assert r.ari_all > 0.9


def test_common_probe_inputs_ignore_cell_goals():
    N, d = 3, 4
    goals = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    psi_in = build_common_probe_inputs(N, d, "cpu")
    assert psi_in.shape == (N, N, 2 * len(VOTE_PROBES), 4 * d + 4)
    names = common_probe_names()
    assert names[0] == "R:kin_alive_blank"
    assert names[len(VOTE_PROBES)] == "E:kin_alive_blank"
    # First six: g_s = 0 for every cell; next six: g_s = 1.
    g_s_r = psi_in[:, :, :6, 2 * d + 1]
    g_s_e = psi_in[:, :, 6:, 2 * d + 1]
    assert torch.allclose(g_s_r, torch.zeros_like(g_s_r))
    assert torch.allclose(g_s_e, torch.ones_like(g_s_e))
    # Independent of the actual goal map.
    g_s_own = build_vote_probe_inputs(goals, d)[..., 2 * d + 1]
    assert not torch.allclose(psi_in[:, :, 0, 2 * d + 1], g_s_own[:, :, 0])
    assert common_kin_mask().sum() == 6


def test_goal_column_split_visible_to_common_not_weights():
    state, params, u = _world(N=6, d=4, noise=0.0)
    _clone_params_from_origin(params)
    d = state.d
    gs = 2 * d + 1
    is_r = state.goals == GOAL_REPRODUCE
    sign = torch.where(is_r, 3.0, -3.0)
    params.psi_W1[:, :, gs, :] = params.psi_W1[:, :, gs, :] + sign.unsqueeze(-1)
    snap = evaluate_snapshot(
        params, state, u, typed_votes=True, goal_in_f=False, embed=False
    )
    # Goal-input columns never fire when bits are zeroed.
    w = snap.families[FAMILY_WEIGHTS_ONLY]
    assert np.allclose(w.X, w.X[0:1], atol=1e-5)
    # Shared questions still include goal bits, so the maps split.
    c = snap.families[FAMILY_COMMON]
    assert c.stats_all["delta"] > 0.5
    assert c.ari_all > 0.9


def test_realized_original_ignores_harm_head():
    state, params, u = _world(noise=0.0)
    _clone_params_from_origin(params)
    d = state.d
    # Huge harm bias: if routing used harm, realized would jump.
    params.psi_b2[..., d + 1] = 50.0
    psi_in = build_vote_probe_inputs(state.goals, d)
    out = psi_on_probes(params, psi_in)
    orig = realized_votes(out, typed_votes=False)
    typed = realized_votes(out, typed_votes=True)
    help_v = out[..., d]
    assert torch.allclose(orig, help_v)
    kin = torch.tensor([p["relation"] == "kin" for p in VOTE_PROBES])
    # Foe slots under typed routing should be the huge harm values.
    assert typed[:, :, ~kin].abs().mean() > 10.0


def test_class_biased_heads_recover_goals():
    state, params, u = _world(N=6, d=4, noise=0.0)
    _clone_params_from_origin(params)
    d = state.d
    is_r = (state.goals == GOAL_REPRODUCE)
    # R cells: large help. E cells: large harm. Typed realized then splits.
    params.psi_b2[..., d] = torch.where(is_r, 8.0, -8.0)
    params.psi_b2[..., d + 1] = torch.where(is_r, -8.0, 8.0)
    snap = evaluate_snapshot(
        params, state, u, typed_votes=True, goal_in_f=False, embed=False
    )
    fam = snap.families[FAMILY_REALIZED]
    assert fam.stats_all["delta"] > 0.5
    assert fam.ari_all > 0.9
    w = snap.families[FAMILY_WEIGHTS_ONLY]
    # Goal-ablated still sees the weight split.
    assert w.ari_all > 0.9


def test_pairwise_and_zscore():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    D = pairwise_euclidean(zscore(X))
    assert D.shape == (3, 3)
    assert np.allclose(np.diag(D), 0.0, atol=1e-6)
    goals = np.array([0, 0, 1])
    st = class_distance_stats(D, goals)
    assert st["n_r"] == 2
    assert st["n_e"] == 1
    assert np.isfinite(st["within_r"])
    assert not np.isfinite(st["within_e"])  # one eliminator: no pair
    assert not np.isfinite(st["delta"])  # energy contrast needs both types ≥2


def test_equal_pool_is_energy_half():
    # 3 R, 2 E. within_r=1, within_e=5, between=3.
    D = np.zeros((5, 5))
    for a, b in ((0, 1), (0, 2), (1, 2)):
        D[a, b] = D[b, a] = 1.0
    D[3, 4] = D[4, 3] = 5.0
    for a in (0, 1, 2):
        for b in (3, 4):
            D[a, b] = D[b, a] = 3.0
    st = class_distance_stats(D, np.array([0, 0, 0, 1, 1]))
    assert st["within_r"] == 1.0
    assert st["within_e"] == 5.0
    assert st["within"] == 3.0  # 0.5 * (1+5), not pair-count 2.0
    assert st["between"] == 3.0
    assert st["delta"] == 0.0  # E/2


def test_alive_only_rezscores():
    state, params, u = _world(N=8, d=4, noise=0.0, seed=8)
    _clone_params_from_origin(params)
    state.x[:4, :] = 0
    params.psi_b2[:4, :, state.d] = 50.0
    snap = evaluate_snapshot(
        params, state, u, typed_votes=True, goal_in_f=False, embed=False
    )
    fam = snap.families[FAMILY_REALIZED]
    Xa = fam.X[snap.alive]
    st_re = class_distance_stats(pairwise_euclidean(zscore(Xa)), snap.goals[snap.alive])
    assert np.isfinite(fam.stats_alive["delta"])
    assert abs(fam.stats_alive["delta"] - st_re["delta"]) < 1e-8


def test_psi_out_dim_matches_split():
    state, params, u = _world()
    d = state.d
    assert psi_out_dim(d) == d + 2
    psi_in = build_vote_probe_inputs(state.goals, d)
    out = psi_on_probes(params, psi_in)
    assert out.shape[-1] == d + 2


if __name__ == "__main__":
    test_vote_probe_input_layout()
    print("test_vote_probe_input_layout OK")
    test_identical_weights_zero_weights_only_delta()
    print("test_identical_weights_zero_weights_only_delta OK")
    test_common_probe_inputs_ignore_cell_goals()
    print("test_common_probe_inputs_ignore_cell_goals OK")
    test_goal_column_split_visible_to_common_not_weights()
    print("test_goal_column_split_visible_to_common_not_weights OK")
    test_realized_original_ignores_harm_head()
    print("test_realized_original_ignores_harm_head OK")
    test_class_biased_heads_recover_goals()
    print("test_class_biased_heads_recover_goals OK")
    test_pairwise_and_zscore()
    print("test_pairwise_and_zscore OK")
    test_equal_pool_is_energy_half()
    print("test_equal_pool_is_energy_half OK")
    test_alive_only_rezscores()
    print("test_alive_only_rezscores OK")
    test_psi_out_dim_matches_split()
    print("test_psi_out_dim_matches_split OK")
    print("all functional divergence tests passed")
