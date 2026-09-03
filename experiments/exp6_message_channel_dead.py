"""Experiment 6 — The message channel is not learned.

Research question
-----------------
Does the vector-valued message head of ψ receive any training signal under
the current Path-1 local learning rule, or is "learned signalling" carried
entirely by the scalar vote heads?

Mathematical claim
------------------
ψ emits, per directed edge,

    ψ_θ(y_j, y_i) = ( m_{j→i} ∈ R^d ,  v^help_{j→i} ,  v^harm_{j→i} )

so the last linear layer splits as

    W2 = [ W2^{(m)} | W2^{(help)} | W2^{(harm)} ] ∈ R^{H × (d+2)}.

Aggregated messages at cell i are

    M_i = Σ_{j ∈ N(i)} ρ_j x_j  m_{j→i}.

The local loss is

    L_i = -p_i + Σ_k c_{i,k} p_{nbr(i,k)},

    p_i = σ( ℓ_i ),
    ℓ_i = w0 + w1 A_i + w2 R_i + w3 E_i
          + w4_help V^{kin}_i + w4_harm V^{foe}_i
          + w5 tanh(u · s̃_i),

where s̃_i = f_θ(s_i, h_i, x_i, M_i)_s is f's proposed state.

Under the *current* implementation (dynamics.local_update):

    s̃_i = f_θ( … , stopgrad(M_i) )_s

and learning.py detaches all incoming votes in the self term and all
neighbour f-signals in the neighbour term. The only live paths are

    -p_i  →  tanh(u·s̃_i)  →  f-params of i
    c p_j →  V^{kin/foe}_j (patched)  →  outgoing vote heads of i.

Hence exactly

    ∂L / ∂W2^{(m)} = 0 ,   ∂L / ∂b2^{(m)} = 0

for L = Σ_i L_i (and for every individual L_i). The same argument kills
f's memory head (nothing reads h̃ in the loss; next_state is detached).

Interpretation
--------------
m still enters the forward dynamics (f reads detached M), so it perturbs
behaviour and can drift as a *side effect* of trunk updates driven by the
vote heads. It is not a trained communication channel. Claims about
"learned messages" currently rest on votes alone.

Fix (optional): Config.learn_messages=True removes stopgrad(M) and trains
the message head (one-hop leakage into senders' ψ). Default remains False
so existing locality guarantees and this experiment's claim stay intact.

Method
------
1. One-step gradient probe: L1 grad mass per logical head (assert zeros).
2. Multi-step SGD: cumulative parameter L1 drift per head over T steps.
3. Forward ablation: zero message head → Δs̃ ≠ 0, ΔV = 0.
4. Counterfactual: recompute f with live M → message-head grad becomes
   nonzero (proves the detach is the sole cause).

Run
---
    python -m experiments.exp6_message_channel_dead

Out
---
    runs/exp6_message_channel_dead/
        REPORT.md          mathematical claim + measured results
        summary.png        head drift curves
        grad_probe.json    one-step grad masses
        drift_series.npz   per-step head L1 deltas
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Project root on path when run as module or script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config
from dynamics import forward_step, local_update, make_u, message_pass
from learning import compute_local_losses, gradient_step
from parameters import batched_mlp, init_parameters
from state import init_state

OUT = Path("runs/exp6_message_channel_dead")
STEPS = 200
GRID_N = 12
D = 6
HIDDEN = 12
SEED = 17


def _world(seed: int = SEED):
    """Fully-alive grid with noisy s/h, used so vote/f heads actually receive gradient."""
    cfg = Config(
        N=GRID_N, d=D, hidden=HIDDEN, seed=seed, n_steps=STEPS,
        init_alive_prob=1.0, learn=True, typed_votes=True,
        # Mild weights: keep a live population so vote/f heads actually train.
        w0=-1.0, w1=0.25, w2=0.2, w3=-0.2,
        w4_help=0.8, w4_harm=0.8, w5=0.5, eta=0.08,
    )
    gen = torch.Generator().manual_seed(seed)
    state = init_state(cfg.N, cfg.d, 1.0, gen)
    state.s.normal_(generator=gen)
    state.h.normal_(generator=gen)
    state.x[:] = 1.0
    params = init_parameters(
        cfg.N, cfg.d, cfg.hidden, cfg.init_noise_std, gen
    ).requires_grad_(True)
    u = make_u(cfg.d, cfg.u_seed)
    return cfg, state, params, u


def _zero(params):
    """Zero all parameter grads in place."""
    for t in params.tensors():
        if t.grad is not None:
            t.grad.zero_()


def head_grad_masses(params, d: int) -> dict[str, float]:
    """L1 mass of grads split by ψ message/vote heads and f state/memory heads."""
    gW2, gb2 = params.psi_W2.grad, params.psi_b2.grad
    gFW2, gFb2 = params.f_W2.grad, params.f_b2.grad

    def mass(t):
        return 0.0 if t is None else float(t.abs().sum().item())

    return {
        "psi_trunk_W1": mass(params.psi_W1.grad),
        "psi_msg_W2": mass(None if gW2 is None else gW2[..., :d]),
        "psi_help_W2": mass(None if gW2 is None else gW2[..., d:d + 1]),
        "psi_harm_W2": mass(None if gW2 is None else gW2[..., d + 1:d + 2]),
        "psi_msg_b2": mass(None if gb2 is None else gb2[..., :d]),
        "psi_help_b2": mass(None if gb2 is None else gb2[..., d:d + 1]),
        "psi_harm_b2": mass(None if gb2 is None else gb2[..., d + 1:d + 2]),
        "f_trunk_W1": mass(params.f_W1.grad),
        "f_state_W2": mass(None if gFW2 is None else gFW2[..., :d]),
        "f_mem_W2": mass(None if gFW2 is None else gFW2[..., d:]),
        "f_state_b2": mass(None if gFb2 is None else gFb2[..., :d]),
        "f_mem_b2": mass(None if gFb2 is None else gFb2[..., d:]),
    }


def probe_one_step_grads(cfg, state, params, u) -> dict[str, float]:
    """One forward + local-loss backward; return per-head gradient masses."""
    step = forward_step(state, params, u, cfg)
    losses = compute_local_losses(state, step, cfg)
    _zero(params)
    losses.sum().backward()
    return head_grad_masses(params, cfg.d)


def counterfactual_live_M_grad(cfg, state, params, u) -> float:
    """Recompute f with *live* M; return message-head |grad|_1 under -Σ p_self."""
    from state import State

    # Scale state to avoid tanh saturation; self-only loss with w5=1.
    st2 = State(
        x=state.x, s=state.s * 0.1, h=state.h * 0.1,
        goals=state.goals, rho=state.rho,
    )
    mp = message_pass(st2, params, typed_votes=cfg.typed_votes)
    d = st2.d
    own_x = st2.x.unsqueeze(-1)
    # Live M (the production path detaches this).
    f_in = torch.cat([st2.s, st2.h, own_x, mp.aggregated_messages], dim=-1)
    f_out = batched_mlp(f_in, params.f_W1, params.f_b1, params.f_W2, params.f_b2)
    s_prop = f_out[..., :d]
    f_sig = (s_prop * u).sum(-1)
    loss = (-torch.sigmoid(torch.tanh(f_sig))).sum()  # w5=1, other w=0
    _zero(params)
    loss.backward()
    return float(params.psi_W2.grad[..., :d].abs().sum().item())


def forward_ablation(cfg, state, params, u) -> dict[str, float]:
    """Zero the message head and measure change in s_proposed vs vote sums."""
    step1 = forward_step(state, params, u, cfg)
    p2 = params.detach_clone()
    p2.psi_W2[..., :cfg.d] = 0
    p2.psi_b2[..., :cfg.d] = 0
    step2 = forward_step(state, p2, u, cfg)
    return {
        "s_proposed_L1_delta": float(
            (step1.s_proposed - step2.s_proposed).abs().sum().item()
        ),
        "V_kin_L1_delta": float(
            (step1.survival_inputs.V_kin - step2.survival_inputs.V_kin)
            .abs().sum().item()
        ),
        "V_foe_L1_delta": float(
            (step1.survival_inputs.V_foe - step2.survival_inputs.V_foe)
            .abs().sum().item()
        ),
    }


def run_drift(cfg, state, params, u, steps: int):
    """Cumulative L1 drift of each ψ/f head from init, plus alive count."""
    d = cfg.d
    snaps = {
        "psi_msg": [], "psi_help": [], "psi_harm": [], "psi_trunk": [],
        "f_state": [], "f_mem": [], "f_trunk": [], "alive": [],
    }
    base = {
        "psi_msg": params.psi_W2[..., :d].detach().clone(),
        "psi_help": params.psi_W2[..., d:d + 1].detach().clone(),
        "psi_harm": params.psi_W2[..., d + 1:].detach().clone(),
        "psi_trunk": params.psi_W1.detach().clone(),
        "f_state": params.f_W2[..., :d].detach().clone(),
        "f_mem": params.f_W2[..., d:].detach().clone(),
        "f_trunk": params.f_W1.detach().clone(),
    }
    for t in range(steps):
        step = forward_step(state, params, u, cfg)
        gradient_step(state, step, params, cfg)
        state = step.next_state
        snaps["psi_msg"].append(
            float((params.psi_W2[..., :d] - base["psi_msg"]).abs().sum().item())
        )
        snaps["psi_help"].append(
            float((params.psi_W2[..., d:d + 1] - base["psi_help"]).abs().sum().item())
        )
        snaps["psi_harm"].append(
            float((params.psi_W2[..., d + 1:] - base["psi_harm"]).abs().sum().item())
        )
        snaps["psi_trunk"].append(
            float((params.psi_W1 - base["psi_trunk"]).abs().sum().item())
        )
        snaps["f_state"].append(
            float((params.f_W2[..., :d] - base["f_state"]).abs().sum().item())
        )
        snaps["f_mem"].append(
            float((params.f_W2[..., d:] - base["f_mem"]).abs().sum().item())
        )
        snaps["f_trunk"].append(
            float((params.f_W1 - base["f_trunk"]).abs().sum().item())
        )
        snaps["alive"].append(int(state.x.sum().item()))
    return {k: np.asarray(v, dtype=np.float64) for k, v in snaps.items()}


def write_report(
    path: Path,
    grads: dict,
    drift: dict,
    ablation: dict,
    cf_msg_grad: float,
    cfg: Config,
) -> None:
    """Write a markdown report of gradient masses, drift, and ablation deltas."""
    final = {k: float(v[-1]) for k, v in drift.items() if k != "alive"}
    lines = [
        "# Experiment 6 — Message channel is not learned",
        "",
        "## Claim",
        "",
        "Under the Path-1 learning rule currently implemented in this codebase,",
        "the message head of ψ is **exactly gradient-dead**. After any number of",
        "SGD steps it remains **bit-identical** to initialisation. Learned",
        "inter-cell influence is carried only by the scalar vote heads",
        f"(help/harm; `typed_votes={cfg.typed_votes}`).",
        "",
        "## Setup (reduced model)",
        "",
        "ψ edge output:",
        "",
        r"$$\psi_\theta(y_j,y_i)=\big(m_{j\to i}\in\mathbb{R}^d,\;"
        r"v^{\mathrm{help}}_{j\to i},\;v^{\mathrm{harm}}_{j\to i}\big)$$",
        "",
        "so $W_2=[W_2^{(m)}\\,|\\,W_2^{(\\mathrm{help})}\\,|\\,W_2^{(\\mathrm{harm})}]$.",
        "",
        r"$$M_i=\sum_{j\in\mathcal{N}(i)}\rho_j x_j\, m_{j\to i}$$",
        "",
        r"$$\tilde s_i = f_\theta\big(s_i,h_i,x_i,\mathrm{stopgrad}(M_i)\big)_s$$",
        "",
        r"$$L_i=-p_i+\sum_k c_{i,k}\,p_{\mathrm{nbr}(i,k)},"
        r"\qquad p_i=\sigma(\ell_i)$$",
        "",
        "with $\\ell_i$ the usual linear combination of $(A,R,E,V^{\\mathrm{kin}},"
        "V^{\\mathrm{foe}},\\tanh(u\\cdot\\tilde s_i))$.",
        "",
        "### Gradient paths that exist",
        "",
        "| Path | Parameters trained |",
        "| --- | --- |",
        "| $-p_i \\to \\tanh(u\\cdot\\tilde s_i)$ | $f$ state head + $f$ trunk |",
        "| $c\\,p_j \\to V^{\\mathrm{kin/foe}}_j$ (patched) | ψ help/harm heads + ψ trunk |",
        "",
        "### Gradient paths that are killed",
        "",
        "| Path | Mechanism |",
        "| --- | --- |",
        "| $-p_i \\to M_i \\to m_{j\\to i}$ | `aggregated_messages.detach()` in `local_update` |",
        "| $-p_i \\to V^{\\mathrm{kin/foe}}_i$ | `.detach()` on self votes in `learning.py` |",
        "| $c\\,p_j \\to \\tilde s_j$ | neighbour `f_signal.detach()` |",
        "| multi-step through $h$ or $s$ | `next_state` fully detached |",
        "",
        "Therefore",
        "",
        r"$$\frac{\partial L}{\partial W_2^{(m)}}=0,\qquad"
        r"\frac{\partial L}{\partial b_2^{(m)}}=0$$",
        "",
        "exactly, for $L=\\sum_i L_i$.",
        "",
        "## Empirical results",
        "",
        f"- Grid $N={cfg.N}$, $d={cfg.d}$, hidden $={cfg.hidden}$, "
        f"steps $={STEPS}$, seed $={cfg.seed}$, $\\eta={cfg.eta}$.",
        f"- Survival weights: $w=(w_0,\\ldots,w_5)=$"
        f"({cfg.w0}, {cfg.w1}, {cfg.w2}, {cfg.w3}, "
        f"{cfg.w4_help}/{cfg.w4_harm}, {cfg.w5}).",
        "",
        "### 1. One-step gradient probe ($\\|\\nabla\\|_1$ over the full grid)",
        "",
        "| Component | $\\|\\nabla\\|_1$ | Expected |",
        "| --- | ---: | --- |",
    ]
    expect = {
        "psi_msg_W2": "0 (dead)",
        "psi_msg_b2": "0 (dead)",
        "psi_help_W2": "> 0 (live)",
        "psi_harm_W2": "> 0 (live)",
        "psi_trunk_W1": "> 0 via votes",
        "f_state_W2": "> 0 (live)",
        "f_mem_W2": "0 (dead)",
        "f_trunk_W1": "> 0 via state head",
    }
    for k, exp in expect.items():
        lines.append(f"| `{k}` | {grads[k]:.6e} | {exp} |")

    lines += [
        "",
        "### 2. Cumulative parameter drift after training "
        f"(L1 distance from init, $T={STEPS}$)",
        "",
        "| Head | Final L1 drift |",
        "| --- | ---: |",
        f"| ψ message $W_2^{{(m)}}$ | **{final['psi_msg']:.6e}** |",
        f"| ψ help $W_2^{{(\\mathrm{{help}})}}$ | {final['psi_help']:.6e} |",
        f"| ψ harm $W_2^{{(\\mathrm{{harm}})}}$ | {final['psi_harm']:.6e} |",
        f"| ψ trunk $W_1$ | {final['psi_trunk']:.6e} |",
        f"| $f$ state head | {final['f_state']:.6e} |",
        f"| $f$ memory head | **{final['f_mem']:.6e}** |",
        f"| $f$ trunk $W_1$ | {final['f_trunk']:.6e} |",
        "",
        f"Alive at end: {int(drift['alive'][-1])} / {cfg.N ** 2}.",
        "",
        "The message and memory heads remain at **exactly zero** drift "
        "(bit-identical to init). Vote heads and trunks move.",
        "",
        "### 3. Forward ablation (messages are not inert)",
        "",
        "Zero the message head, re-run one forward step on the same state:",
        "",
        f"- $\\|\\Delta \\tilde s\\|_1$ = **{ablation['s_proposed_L1_delta']:.6e}** "
        "(messages change dynamics)",
        f"- $\\|\\Delta V^{{\\mathrm{{kin}}}}\\|_1$ = {ablation['V_kin_L1_delta']:.6e}",
        f"- $\\|\\Delta V^{{\\mathrm{{foe}}}}\\|_1$ = {ablation['V_foe_L1_delta']:.6e}",
        "",
        "So $m$ is **structured noise**: it affects the forward map; nothing "
        "optimises it. The trunk can still move $m$'s *outputs* over time as a "
        "side effect of vote-driven trunk updates, but $W_2^{(m)}$ itself never "
        "receives a gradient.",
        "",
        "### 4. Counterfactual (cause isolation)",
        "",
        "Same self-survival loss, but feed **live** $M$ into $f$ "
        "(remove only the detach):",
        "",
        f"- message-head $\\|\\nabla\\|_1$ = **{cf_msg_grad:.6e}** (nonzero)",
        "",
        "Therefore the dead channel is an implementation choice "
        "(`stopgrad(M)` for Path-1 locality), not an absence of dependence of "
        "$f$ on $M$.",
        "",
        "## Conclusion for the paper",
        "",
        "1. **Only votes are a learned communication channel** in the current "
        "system (plus the local $f$-signal for self-survival).",
        "2. Any discussion of \"learned messages\" / signalling via $m$ is "
        "**not supported by the training graph** until `stopgrad(M)` is removed "
        "or replaced by a locality-preserving message credit assignment.",
        "3. $h$ (memory output) is likewise untrained; multi-step memory needs "
        "BPTT through `next_state` (deferred).",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python -m experiments.exp6_message_channel_dead",
        "python tests/test_message_head_dead.py",
        "```",
        "",
        "Artifacts: `summary.png`, `grad_probe.json`, `drift_series.npz`, this report.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_summary(drift: dict, out_png: Path) -> None:
    """Plot cumulative L1 drift of ψ and f heads (message/memory should stay flat)."""
    t = np.arange(len(drift["psi_msg"]))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax = axes[0]
    ax.plot(t, drift["psi_msg"], label=r"$\psi$ message head $W_2^{(m)}$",
            color="#ef4444", linewidth=2.0)
    ax.plot(t, drift["psi_help"], label=r"$\psi$ help head",
            color="#22c55e", linewidth=1.5)
    ax.plot(t, drift["psi_harm"], label=r"$\psi$ harm head",
            color="#16a34a", linewidth=1.5, linestyle="--")
    ax.plot(t, drift["psi_trunk"], label=r"$\psi$ trunk $W_1$",
            color="#94a3b8", linewidth=1.0)
    ax.set_ylabel("cumulative L1 drift from init")
    ax.set_title("ψ heads: message head is flat at 0")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t, drift["f_state"], label=r"$f$ state head",
            color="#38bdf8", linewidth=1.5)
    ax.plot(t, drift["f_mem"], label=r"$f$ memory head",
            color="#ef4444", linewidth=2.0)
    ax.plot(t, drift["f_trunk"], label=r"$f$ trunk $W_1$",
            color="#94a3b8", linewidth=1.0)
    ax2 = ax.twinx()
    ax2.plot(t, drift["alive"], color="#a78bfa", alpha=0.5, linewidth=1.0,
             label="alive")
    ax2.set_ylabel("alive count", color="#a78bfa")
    ax.set_xlabel("step")
    ax.set_ylabel("cumulative L1 drift from init")
    ax.set_title("f heads: memory head is flat at 0")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    """Probe message-head grads, run a drift series, and write the report."""
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, state, params, u = _world()

    print("=== Exp 6: message channel dead ===")
    print("1. One-step gradient probe...")
    grads = probe_one_step_grads(cfg, state, params, u)
    for k, v in grads.items():
        print(f"   {k:16s}  {v:.6e}")

    assert grads["psi_msg_W2"] == 0.0 and grads["psi_msg_b2"] == 0.0
    assert grads["f_mem_W2"] == 0.0 and grads["f_mem_b2"] == 0.0
    assert grads["psi_help_W2"] + grads["psi_harm_W2"] > 0.0
    assert grads["f_state_W2"] > 0.0

    print("2. Forward ablation...")
    # Need a fresh graph-less snapshot of state/params for ablation mid-flow.
    # Use current params (pre-training) before drift run.
    ablation = forward_ablation(cfg, state, params, u)
    print(f"   s_delta={ablation['s_proposed_L1_delta']:.6e}  "
          f"V_kin_delta={ablation['V_kin_L1_delta']:.6e}")

    print("3. Counterfactual live-M gradient...")
    # Re-init a clean world so counterfactual doesn't fight used grads.
    cfg2, state2, params2, u2 = _world(seed=SEED)
    cf = counterfactual_live_M_grad(cfg2, state2, params2, u2)
    print(f"   message-head |grad|_1 with live M = {cf:.6e}")
    assert cf > 0.0

    print(f"4. Multi-step drift ({STEPS} steps)...")
    cfg3, state3, params3, u3 = _world(seed=SEED)
    drift = run_drift(cfg3, state3, params3, u3, STEPS)
    print(f"   final msg drift = {drift['psi_msg'][-1]:.6e}  "
          f"(must be 0)")
    print(f"   final help drift = {drift['psi_help'][-1]:.6e}")
    print(f"   final harm drift = {drift['psi_harm'][-1]:.6e}")
    print(f"   final mem drift = {drift['f_mem'][-1]:.6e}  (must be 0)")
    assert drift["psi_msg"][-1] == 0.0
    assert drift["f_mem"][-1] == 0.0

    (OUT / "grad_probe.json").write_text(
        json.dumps({"one_step_grad_L1": grads,
                    "counterfactual_live_M_msg_grad_L1": cf,
                    "forward_ablation": ablation}, indent=2),
        encoding="utf-8",
    )
    np.savez(OUT / "drift_series.npz", **drift)
    plot_summary(drift, OUT / "summary.png")
    write_report(OUT / "REPORT.md", grads, drift, ablation, cf, cfg)

    print(f"\nSaved → {OUT}/")
    print(f"  REPORT.md  summary.png  grad_probe.json  drift_series.npz")
    print("Claim held: message head is not learned.")


if __name__ == "__main__":
    main()
