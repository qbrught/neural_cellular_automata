"""Thesis comparison registry.

Each Comparison is one results chapter: a named set of arms on one base
config, an optional off/on isolation pair, and the prose + primary metrics
the report should lead with.

Add a letter: append to COMPARISONS (and a VersionSpec in versions.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from research.protocol import F_LAMBDAS
from research.versions import VersionSpec, get_version


@dataclass(frozen=True)
class Arm:
    """One simulated condition."""

    id: str
    version: str
    spec_overrides: dict[str, Any] = field(default_factory=dict)

    def spec(self) -> VersionSpec:
        v = get_version(self.version)
        if self.spec_overrides:
            return replace(v, **self.spec_overrides)
        return v


@dataclass(frozen=True)
class Comparison:
    """One thesis chapter / isolation / sweep / ladder."""

    id: str
    title: str
    arms: tuple[Arm, ...]
    # Isolation pair (arm ids). None for ladder / multi-arm sweeps.
    off: str | None
    on: str | None
    config_id: str = "sym"
    # Extra summary keys to headline besides Φ / Φ_late / min-type / alive.
    primary_metrics: tuple[str, ...] = ()
    motivation: str = ""
    visual_prompt: str = ""
    hypothesis: str = ""
    kind: str = "isolation"  # isolation | ladder | sweep | identity
    default: bool = True


def _arm(vid: str) -> Arm:
    return Arm(id=vid, version=vid)


def _iso(
    cid: str,
    title: str,
    off: str,
    on: str,
    **kwargs: Any,
) -> Comparison:
    return Comparison(
        id=cid,
        title=title,
        arms=(_arm(off), _arm(on)),
        off=off,
        on=on,
        **kwargs,
    )


COMPARISONS: tuple[Comparison, ...] = (
    _iso(
        "A",
        "A — typed help/harm votes",
        "original",
        "A",
        primary_metrics=(
            "corr_ra_density",
            "late_R_vote_disc",
            "late_E_vote_disc",
            "late_death_rate_cross_minus_same",
        ),
        motivation=(
            "Original physics dumps one vote scalar on every neighbour. "
            "A routes ψ's help head only to same-goal neighbours and the harm "
            "head only to opposite-goal neighbours, so influence is typed."
        ),
        visual_prompt=(
            "Same seed, toggle typed_votes. Do clusters of green/red separate? "
            "Do fronts look kin-protective rather than a single mixed soup?"
        ),
        hypothesis=(
            "Alive-count coupling drops; vote specialization (help→kin, harm→foe) "
            "emerges; Φ_late rises vs original without requiring w2≠w3."
        ),
    ),
    _iso(
        "B",
        "B — predator–prey loss",
        "A",
        "B",
        primary_metrics=(
            "late_death_E_cross_minus_same",
            "late_E_vote_disc",
            "corr_ra_ea",
        ),
        motivation=(
            "Under A, eliminators are rewarded for killing everyone, including "
            "fellow E. B zeros the loss on E–E neighbours so eliminators pressure "
            "only reproducer prey. Survival and vote routing stay as in A."
        ),
        visual_prompt=(
            "Compare A vs B on the same seed. Do red cells still collapse into "
            "self-destructive patches, or do they persist as predators on green?"
        ),
        hypothesis=(
            "E-on-E death falls; eliminator harm specializes on prey; less "
            "self-destructive elim pressure than A."
        ),
    ),
    _iso(
        "C_only",
        "C_only — goal inheritance (isolated)",
        "original",
        "C_only",
        primary_metrics=(
            "goal_frac_repro_drift",
            "mean_abs_goal_frac_delta",
            "corr_ra_density",
        ),
        motivation=(
            "Goals are otherwise frozen at init. C_only lets a birth (dead→alive) "
            "adopt the majority pre-step alive-neighbour goal, on original "
            "(indiscriminate) votes and loss — colonization without A/B."
        ),
        visual_prompt=(
            "Watch births: do colour regions expand as invasion fronts, or do "
            "revivals keep the latent init map?"
        ),
        hypothesis=(
            "Type fractions become dynamical; |g_frac drift| ≫ 0 vs original."
        ),
    ),
    _iso(
        "C",
        "C — goal inheritance on A+B",
        "B",
        "C",
        primary_metrics=(
            "goal_frac_repro_drift",
            "mean_abs_goal_frac_delta",
            "corr_ra_density",
        ),
        motivation=(
            "Same birth-inheritance rule as C_only, stacked on typed votes and "
            "predator–prey loss. Survivors keep goals; ρ and MLP weights are "
            "not copied."
        ),
        visual_prompt=(
            "Invasion fronts on A+B physics: does one colour colonize, and does "
            "that match g_frac drift rather than a density-tracking rewrite?"
        ),
        hypothesis=(
            "Φ vs initial f0 rises because living type fractions leave f0; "
            "check that corr(ra, dens) does not sneak back to 1 (rewrite null)."
        ),
    ),
    _iso(
        "D_fixed",
        "D_fixed — goal-conditioned f (goals fixed)",
        "B",
        "D_fixed",
        primary_metrics=(
            "late_f_signal_type_gap",
            "late_s_norm_type_gap",
        ),
        motivation=(
            "Local update f always has a goal input slot. Off, the slot is zeros. "
            "On, f sees g_i so proposed (s,h) and the f-survival channel can "
            "become type-specific. Goals stay fixed (no C)."
        ),
        visual_prompt=(
            "Count-level grids may look like B. Look at whether the two colours "
            "hold different textures / persistence, not just different counts."
        ),
        hypothesis=(
            "Type-conditioned f raises f-signal / state specialization even if "
            "count-level Φ stays near B."
        ),
    ),
    _iso(
        "D",
        "D — goal-conditioned f on A+B+C",
        "C",
        "D",
        primary_metrics=(
            "late_f_signal_type_gap",
            "goal_frac_repro_drift",
            "late_s_norm_type_gap",
        ),
        motivation=(
            "Same goal-in-f slot as D_fixed, now with dynamical goals (C). "
            "Colonization plus type-conditioned local updates."
        ),
        visual_prompt=(
            "Relative to C: do invasion fronts look sharper, or is D visually "
            "the same colonization with a different internal policy?"
        ),
        hypothesis=(
            "Type-specific internal dynamics plus dynamical goals: Φ at least "
            "as large as C, with a larger f-signal gap."
        ),
    ),
    _iso(
        "E",
        "E — symmetric w2=w3 on asymmetric physics",
        "A",
        "E",
        config_id="asym",
        primary_metrics=("corr_ra_density",),
        motivation=(
            "Benchmark defaults give reproducers and eliminators unequal "
            "neighbour weights (w2≠w3). E sets w2=w3=mean(w2,w3) with no new "
            "code path, so count-level Φ cannot be blamed on that asymmetry."
        ),
        visual_prompt=(
            "On the asymmetric base, does symmetrizing w2/w3 visibly flatten "
            "the green/red imbalance that A showed?"
        ),
        hypothesis=(
            "If Φ under E stays comparable to A, typed votes (not w2≠w3) drive "
            "divergence; if Φ collapses, the old signal was partly physics."
        ),
    ),
    _iso(
        "F",
        "F — soft coexistence pressure",
        "A",
        "F",
        primary_metrics=(
            "late_min_type_frac",
            "late_min_soft_rho",
            "mean_coexistence_barrier",
        ),
        motivation=(
            "Local losses do not care whether both types remain globally. F adds "
            "B = λ(−log ρ̃^R − log ρ̃^E) once per step on soft self-masses, a "
            "viability regularizer rather than a 50/50 quota."
        ),
        visual_prompt=(
            "On seeds where A goes to one colour, does F keep both colours "
            "alive, or is the grid indistinguishable from A?"
        ),
        hypothesis=(
            "F raises the two-process rate (large Φ_late AND both types present) "
            "vs A; specialization should survive if λ is small."
        ),
    ),
    _iso(
        "G",
        "G — frozen transfer blobs",
        "A",
        "G",
        primary_metrics=(
            "late_frac_alive_low_kappa",
            "late_kappa_edge_mean",
        ),
        motivation=(
            "G leaves vote routing and loss as in A. A frozen overlay multiplies "
            "messages and votes by type-selected κ on every edge. Suite G is "
            "three hard transfer-dead disks (κ=0); occupancy is off, so cells "
            "may still live inside a blob."
        ),
        visual_prompt=(
            "Blue low-κ disks should fragment information. Do patterns stall at "
            "blob edges? Are interiors occupied but quiet?"
        ),
        hypothesis=(
            "Spatial transfer barriers change Φ vs A by fragmenting information "
            "flow; alive fraction in low-κ regions is a G-specific signature."
        ),
    ),
    _iso(
        "G_learn",
        "G_learn — learning-rate hotspot",
        "A",
        "G_learn",
        primary_metrics=(
            "late_eta_mean_alive",
            "late_f_signal_type_gap",
        ),
        motivation=(
            "Same overlay machinery as G, but κ stays 1 and η_scale is 1 in a "
            "center disk and 0.25 outside. Adaptation concentrates in the hotspot."
        ),
        visual_prompt=(
            "Amber low-η exterior vs full-rate center: does structure persist "
            "in the hotspot while the exterior looks closer to frozen A?"
        ),
        hypothesis=(
            "Spatially varying SGD changes Φ via slower exterior updates, not "
            "blocked messages."
        ),
    ),
    Comparison(
        id="ladder",
        title="Full version ladder (appendix)",
        arms=tuple(
            _arm(v)
            for v in (
                "original",
                "A",
                "E",
                "F",
                "G",
                "G_learn",
                "B",
                "C_only",
                "C",
                "D_fixed",
                "D",
            )
        ),
        off=None,
        on=None,
        primary_metrics=(),
        motivation=(
            "Stacked view of every implemented version on symmetric physics. "
            "Not an isolation: use the pairwise chapters for claims, this table "
            "for the appendix overview."
        ),
        visual_prompt="Scan late-grid montages across the ladder on one seed.",
        hypothesis="Typed votes and inheritance move Φ; D_fixed and F may not.",
        kind="ladder",
        default=False,
    ),
    Comparison(
        id="F_lambda",
        title="F — λ sweep",
        arms=(_arm("A"),)
        + tuple(
            Arm(
                id=f"F_lam{lam:g}",
                version="F",
                spec_overrides={"coexistence_lambda": lam},
            )
            for lam in F_LAMBDAS
        ),
        off="A",
        on=None,
        primary_metrics=(
            "late_min_type_frac",
            "mean_coexistence_barrier",
            "phi_class_late",
        ),
        motivation=(
            "Same F barrier, λ ∈ {0.01, 0.1, 1}. Tests whether a larger penalty "
            "converts attrition-Φ into two-process coexistence, or does nothing "
            "because the gradient only rides the weak f-survival channel."
        ),
        visual_prompt="High-λ vs A on an attrition seed and on a full-lattice seed.",
        hypothesis="If dynamics are flat in λ, the channel is too weak, not λ too small.",
        kind="sweep",
        default=False,
    ),
    Comparison(
        id="E_identity",
        title="E identity check on already-symmetric weights",
        arms=(_arm("A"), _arm("E")),
        off="A",
        on="E",
        config_id="sym",
        primary_metrics=(),
        motivation=(
            "On benchmark_sym_w.json, A and E are the same mechanism (E only "
            "re-averages already-equal w2,w3). ΔΦ should be ~0."
        ),
        visual_prompt="Grids should match aside from tiny float noise in apply().",
        hypothesis="Mean ΔΦ (E−A) ≈ 0; any gap is apply() rounding.",
        kind="identity",
        default=False,
    ),
)


def get_comparison(cid: str) -> Comparison:
    for c in COMPARISONS:
        if c.id == cid:
            return c
    known = ", ".join(c.id for c in COMPARISONS)
    raise KeyError(f"Unknown comparison {cid!r}. Known: {known}")


def parse_letter_list(spec: str | None, *, include_optional: bool = False) -> list[Comparison]:
    """Parse 'A,B,C' or 'all' or None (defaults)."""
    if spec is None or spec.strip() == "":
        return [c for c in COMPARISONS if c.default or include_optional]
    raw = spec.strip()
    if raw.lower() == "all":
        return list(COMPARISONS)
    if raw.lower() == "default":
        return [c for c in COMPARISONS if c.default]
    out: list[Comparison] = []
    seen: set[str] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part or part in seen:
            continue
        seen.add(part)
        out.append(get_comparison(part))
    return out
