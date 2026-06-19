"""Test the survival rule: hand-checked scenarios."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from config import Config
from dynamics import SurvivalInputs, hard_survival, soft_survival, survival_logit


def make_cfg(w0=-1.0, w1=0.5, w2=0.5, w3=-0.5, w4=0.5, w5=0.0):
    """Test config. Defaults w5=0 so legacy scenarios that don't supply
    f_signal aren't affected by it."""
    return Config(w0=w0, w1=w1, w2=w2, w3=w3, w4=w4, w5=w5, N=1, d=1, hidden=1, n_steps=1)


def make_inputs(A, R, E, V, f_signal=0.0):
    """Single-cell SurvivalInputs (each (1,1))."""
    t = lambda v: torch.tensor([[float(v)]])
    return SurvivalInputs(A=t(A), R=t(R), E=t(E), V=t(V), f_signal=t(f_signal))


def test_isolated_cell_dies():
    """Zero neighbours alive, no votes -> logit = w0 = -1 < 0 -> dead."""
    cfg = make_cfg()
    inp = make_inputs(A=0, R=0, E=0, V=0)
    assert hard_survival(inp, cfg).item() == 0.0
    assert soft_survival(inp, cfg).item() < 0.5
    print("test_isolated_cell_dies OK")


def test_surrounded_by_reproducers_lives():
    """8 reproducer neighbours alive: logit = -1 + 0.5*8 + 0.5*8 + 0 + 0 = 7 > 0 -> alive."""
    cfg = make_cfg()
    inp = make_inputs(A=8, R=8, E=0, V=0)
    assert hard_survival(inp, cfg).item() == 1.0
    assert soft_survival(inp, cfg).item() > 0.5
    print("test_surrounded_by_reproducers_lives OK")


def test_surrounded_by_eliminators_dies():
    """8 eliminator neighbours alive: logit = -1 + 0.5*8 + 0 - 0.5*8 + 0 = -1 < 0 -> dead.

    This is the 'eliminators are killers' behaviour: w1 boost is exactly cancelled
    by w3 penalty, leaving only the bias.
    """
    cfg = make_cfg()
    inp = make_inputs(A=8, R=0, E=8, V=0)
    assert hard_survival(inp, cfg).item() == 0.0
    assert soft_survival(inp, cfg).item() < 0.5
    print("test_surrounded_by_eliminators_dies OK")


def test_mixed_neighbourhood_threshold():
    """3 reproducers, 1 eliminator: logit = -1 + 0.5*4 + 0.5*3 - 0.5*1 = 2.0 > 0 -> alive."""
    cfg = make_cfg()
    inp = make_inputs(A=4, R=3, E=1, V=0)
    assert hard_survival(inp, cfg).item() == 1.0
    print("test_mixed_neighbourhood_threshold OK")


def test_positive_votes_save_a_cell():
    """Without votes, 1 reproducer alone is logit = -1 + 0.5 + 0.5 = 0.0 -> not >0 -> dead.
    With a strong positive vote, the cell flips alive."""
    cfg = make_cfg()
    no_vote = make_inputs(A=1, R=1, E=0, V=0)
    with_vote = make_inputs(A=1, R=1, E=0, V=10.0)
    assert hard_survival(no_vote, cfg).item() == 0.0
    assert hard_survival(with_vote, cfg).item() == 1.0
    print("test_positive_votes_save_a_cell OK")


def test_negative_votes_kill_a_cell():
    """3 reproducers should normally keep alive (logit = -1 + 0.5*3 + 0.5*3 = 2.0).
    A large negative vote sum overwhelms it."""
    cfg = make_cfg()
    no_vote = make_inputs(A=3, R=3, E=0, V=0)
    big_neg = make_inputs(A=3, R=3, E=0, V=-10.0)
    assert hard_survival(no_vote, cfg).item() == 1.0
    assert hard_survival(big_neg, cfg).item() == 0.0
    print("test_negative_votes_kill_a_cell OK")


def test_soft_and_hard_agree_at_extremes():
    """When the logit is very positive, both p≈1 and hard=1; when very negative, both ≈0."""
    cfg = make_cfg()
    very_pos = make_inputs(A=0, R=0, E=0, V=100.0)
    very_neg = make_inputs(A=0, R=0, E=0, V=-100.0)
    assert hard_survival(very_pos, cfg).item() == 1.0
    assert soft_survival(very_pos, cfg).item() > 0.999
    assert hard_survival(very_neg, cfg).item() == 0.0
    assert soft_survival(very_neg, cfg).item() < 0.001
    print("test_soft_and_hard_agree_at_extremes OK")


def test_soft_is_differentiable_in_V():
    """The soft survival should produce a non-zero gradient w.r.t. the vote sum."""
    cfg = make_cfg()
    V = torch.tensor([[1.5]], requires_grad=True)
    inp = SurvivalInputs(
        A=torch.tensor([[2.0]]),
        R=torch.tensor([[1.0]]),
        E=torch.tensor([[1.0]]),
        V=V,
        f_signal=torch.tensor([[0.0]]),
    )
    p = soft_survival(inp, cfg)
    p.sum().backward()
    assert V.grad is not None
    assert V.grad.abs().item() > 0
    print("test_soft_is_differentiable_in_V OK")


def test_logit_formula_is_exact():
    """Direct numeric check that survival_logit computes the expected linear combination."""
    cfg = make_cfg(w0=-0.7, w1=0.3, w2=0.4, w3=-0.6, w4=0.2)
    inp = make_inputs(A=5, R=3, E=2, V=1.5)
    # -0.7 + 0.3*5 + 0.4*3 + (-0.6)*2 + 0.2*1.5
    # = -0.7 + 1.5 + 1.2 - 1.2 + 0.3 = 1.1
    out = survival_logit(inp, cfg).item()
    assert abs(out - 1.1) < 1e-6, f"Expected 1.1, got {out}"
    print("test_logit_formula_is_exact OK")


def test_full_grid_shapes():
    """Survival functions work on (N,N) grids, not just single cells."""
    cfg = Config(N=5, d=4, hidden=8, n_steps=1)
    N = cfg.N
    inputs = SurvivalInputs(
        A=torch.randint(0, 9, (N, N)).float(),
        R=torch.randint(0, 9, (N, N)).float(),
        E=torch.randint(0, 9, (N, N)).float(),
        V=torch.randn(N, N),
        f_signal=torch.randn(N, N),
    )
    p = soft_survival(inputs, cfg)
    x = hard_survival(inputs, cfg)
    assert p.shape == (N, N)
    assert x.shape == (N, N)
    assert ((p >= 0) & (p <= 1)).all()
    assert ((x == 0) | (x == 1)).all()
    print("test_full_grid_shapes OK")


if __name__ == "__main__":
    test_isolated_cell_dies()
    test_surrounded_by_reproducers_lives()
    test_surrounded_by_eliminators_dies()
    test_mixed_neighbourhood_threshold()
    test_positive_votes_save_a_cell()
    test_negative_votes_kill_a_cell()
    test_soft_and_hard_agree_at_extremes()
    test_soft_is_differentiable_in_V()
    test_logit_formula_is_exact()
    test_full_grid_shapes()
    print("\nAll survival rule tests passed.")
