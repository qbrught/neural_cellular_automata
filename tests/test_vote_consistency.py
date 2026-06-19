"""Consistency check: outgoing_votes[i, k] should equal the value that neighbour k of i
sees as the vote arriving from i.

This is a critical invariant. If it doesn't hold, the locality-preserving loss in
learning.py will train on votes that don't match the votes used by the simulation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from config import Config
from grid import NEIGHBOUR_OFFSETS
from parameters import init_parameters
from state import init_state
from dynamics import message_pass


def test_outgoing_matches_incoming():
    cfg = Config(N=6, d=4, hidden=8, n_steps=1, init_alive_prob=0.7)
    gen = torch.Generator().manual_seed(0)
    state = init_state(cfg.N, cfg.d, cfg.init_alive_prob, gen)
    # Give cells nonzero state vectors so ψ has signal:
    state.s.normal_(generator=gen)
    state.h.normal_(generator=gen)
    params = init_parameters(cfg.N, cfg.d, cfg.hidden, cfg.init_noise_std, gen)

    mp = message_pass(state, params)
    N = cfg.N

    # outgoing_votes[i, j, k]: vote from cell (i,j) to neighbour-k-of-(i,j).
    # incoming, when reconstructed: votes_received_individual[receiver, k_at_receiver]
    # is the vote into `receiver` from its neighbour-k.
    #
    # For any cell (i, j) and outgoing edge k_out (sending to receiver at offset
    # NEIGHBOUR_OFFSETS[k_out]), the receiver R = ((i+di)%N, (j+dj)%N) has (i, j) as
    # one of *its* 8 neighbours -- at the opposite offset (-di, -dj), which is
    # NEIGHBOUR_OFFSETS[7 - k_out] in our canonical ordering. Verify this opposite-index
    # property first.
    for k_out, (di, dj) in enumerate(NEIGHBOUR_OFFSETS):
        k_in = 7 - k_out
        di_back, dj_back = NEIGHBOUR_OFFSETS[k_in]
        assert (di_back, dj_back) == (-di, -dj), (
            f"k_out={k_out} ({di},{dj}) does not pair with k_in={k_in} ({di_back},{dj_back})"
        )

    # We don't have the per-edge incoming votes broken out individually from
    # message_pass (it returns the sum). Re-compute them here using the same logic
    # but inline, to compare against outgoing_votes.
    from grid import gather_neighbours
    from dynamics import _build_psi_inputs, _psi_forward_per_edge

    d = state.d
    psi_in = _build_psi_inputs(state)
    psi_out = _psi_forward_per_edge(psi_in, params)
    votes_received_raw = psi_out[..., d]  # (N, N, 8) before gating

    sender_rho = gather_neighbours(state.rho)
    sender_alive = gather_neighbours(state.x)
    votes_received = votes_received_raw * sender_rho * sender_alive  # (N, N, 8)

    # Now compare: for each cell (i, j) and outgoing direction k_out,
    # the vote it sends should appear at its receiver's slot k_in = 7 - k_out.
    outgoing = mp.outgoing_votes  # (N, N, 8)
    max_diff = 0.0
    for i in range(N):
        for j in range(N):
            for k_out, (di, dj) in enumerate(NEIGHBOUR_OFFSETS):
                k_in = 7 - k_out
                ri, rj = (i + di) % N, (j + dj) % N
                sent_by_ij = outgoing[i, j, k_out].item()
                received_at_r = votes_received[ri, rj, k_in].item()
                diff = abs(sent_by_ij - received_at_r)
                if diff > max_diff:
                    max_diff = diff

    print(f"Max diff between outgoing and incoming (paired): {max_diff:.3e}")
    assert max_diff < 1e-5, f"Outgoing/incoming vote pairing inconsistent! Max diff {max_diff}"
    print("test_outgoing_matches_incoming OK")


def test_incoming_sum_matches_sum_of_per_edge():
    """A sanity check: the per-edge votes_received summed over k must equal the
    incoming_vote_sum returned by message_pass."""
    cfg = Config(N=5, d=3, hidden=6, n_steps=1)
    gen = torch.Generator().manual_seed(1)
    state = init_state(cfg.N, cfg.d, cfg.init_alive_prob, gen)
    state.s.normal_(generator=gen)
    state.h.normal_(generator=gen)
    params = init_parameters(cfg.N, cfg.d, cfg.hidden, cfg.init_noise_std, gen)

    mp = message_pass(state, params)

    from grid import gather_neighbours
    from dynamics import _build_psi_inputs, _psi_forward_per_edge
    d = state.d
    psi_in = _build_psi_inputs(state)
    psi_out = _psi_forward_per_edge(psi_in, params)
    votes_received_raw = psi_out[..., d]
    sender_rho = gather_neighbours(state.rho)
    sender_alive = gather_neighbours(state.x)
    votes_received = votes_received_raw * sender_rho * sender_alive

    assert torch.allclose(votes_received.sum(dim=2), mp.incoming_vote_sum, atol=1e-6)
    print("test_incoming_sum_matches_sum_of_per_edge OK")


if __name__ == "__main__":
    test_outgoing_matches_incoming()
    test_incoming_sum_matches_sum_of_per_edge()
    print("\nAll vote-consistency tests passed.")
