"""Paper version registry.

Each version is a named ablation of the system defined by Config flags.
The suite applies these flags on top of a shared base config + seed so
comparisons isolate the mechanism change, not hyperparameters.

Roadmap (matches the planned A→D path, plus environment ablations):
  original  — single indiscriminate vote channel
  A         — typed help/harm votes routed by kin/foe   [implemented]
  B         — predator–prey losses                       [implemented]
  C_only    — goal inheritance alone (no A/B)            [implemented]
  C         — goal inheritance on top of A+B             [implemented]
  D_fixed   — goal-conditioned f on A+B (fixed goals)    [implemented]
  D         — goal-conditioned f on A+B+C                [implemented]
  E         — A + symmetric survival weights w2=w3       [implemented]
  F         — A + soft coexistence pressure              [implemented]
  G         — A + frozen transfer blobs                  [implemented]
  G_learn   — A + learning hotspot                       [implemented]
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from config import Config


@dataclass(frozen=True)
class VersionSpec:
    """One paper version / ablation."""

    id: str
    title: str
    description: str
    # Flags applied to a base Config.
    typed_votes: bool = True
    predator_prey_loss: bool = False
    goal_inheritance: bool = False
    goal_in_f: bool = False
    # Experiment F: soft coexistence barrier on soft type masses.
    coexistence_pressure: bool = False
    coexistence_lambda: float = 0.01
    coexistence_delta: float = 1e-4
    # Experiment G: frozen spatial environment overlay.
    environment_heterogeneous: bool = False
    env_preset: str = "identity"
    env_n_blobs: int = 3
    env_blob_radius: float = 0.15
    env_kappa_lo: float = 0.0
    env_kappa_hi: float = 1.0
    env_eta_lo: float = 1.0   # identity; A–G must not inject a hotspot
    env_eta_hi: float = 1.0
    env_occupancy_blocks: bool = False
    env_affect_R: bool = True
    env_affect_E: bool = True
    # Environment ablation (not a dynamics-code path): force w2 = w3.
    # When True, both are set to the mean of the base config's w2 and w3
    # so average type-neighbour pressure is preserved.
    symmetrize_RE_weights: bool = False
    # Whether this version is fully implemented (suite can run it).
    implemented: bool = True
    # Free-text experimental hypothesis for the report template.
    hypothesis: str = ""

    def apply(self, cfg: Config) -> Config:
        """Return a copy of cfg with this version's flags set."""
        if not self.implemented:
            raise RuntimeError(
                f"Version {self.id!r} is not implemented yet "
                f"({self.title}). Implement the mechanism before running."
            )
        out = replace(
            cfg,
            typed_votes=self.typed_votes,
            predator_prey_loss=self.predator_prey_loss,
            goal_inheritance=self.goal_inheritance,
            goal_in_f=self.goal_in_f,
            coexistence_pressure=self.coexistence_pressure,
            coexistence_lambda=self.coexistence_lambda,
            coexistence_delta=self.coexistence_delta,
            environment_heterogeneous=self.environment_heterogeneous,
            env_preset=self.env_preset,
            env_n_blobs=self.env_n_blobs,
            env_blob_radius=self.env_blob_radius,
            env_kappa_lo=self.env_kappa_lo,
            env_kappa_hi=self.env_kappa_hi,
            env_eta_lo=self.env_eta_lo,
            env_eta_hi=self.env_eta_hi,
            env_occupancy_blocks=self.env_occupancy_blocks,
            env_affect_R=self.env_affect_R,
            env_affect_E=self.env_affect_E,
        )
        if self.symmetrize_RE_weights:
            w = 0.5 * (float(out.w2) + float(out.w3))
            out = replace(out, w2=w, w3=w)
        # Hand-authored terrain: keep custom env_regions from the base config
        # when this version is an environment ablation (G / G_learn). A–F still
        # force env_preset=identity so the regions are inert under flag-off.
        if (
            self.environment_heterogeneous
            and cfg.env_preset == "custom"
            and isinstance(cfg.env_regions, list)
            and len(cfg.env_regions) > 0
        ):
            out = replace(out, env_preset="custom", env_regions=list(cfg.env_regions))
        return out

    def flag_summary(self) -> str:
        """Short human-readable flag dump for VERSION.txt / notes."""
        return (
            f"typed_votes={self.typed_votes}\n"
            f"predator_prey_loss={self.predator_prey_loss}\n"
            f"goal_inheritance={self.goal_inheritance}\n"
            f"goal_in_f={self.goal_in_f}\n"
            f"coexistence_pressure={self.coexistence_pressure}\n"
            f"coexistence_lambda={self.coexistence_lambda}\n"
            f"coexistence_delta={self.coexistence_delta}\n"
            f"environment_heterogeneous={self.environment_heterogeneous}\n"
            f"env_preset={self.env_preset}\n"
            f"env_n_blobs={self.env_n_blobs}\n"
            f"env_blob_radius={self.env_blob_radius}\n"
            f"env_kappa_lo={self.env_kappa_lo}\n"
            f"env_kappa_hi={self.env_kappa_hi}\n"
            f"env_eta_lo={self.env_eta_lo}\n"
            f"env_eta_hi={self.env_eta_hi}\n"
            f"env_occupancy_blocks={self.env_occupancy_blocks}\n"
            f"env_affect_R={self.env_affect_R}\n"
            f"env_affect_E={self.env_affect_E}\n"
            f"symmetrize_RE_weights={self.symmetrize_RE_weights}\n"
        )


VERSIONS: dict[str, VersionSpec] = {
    "original": VersionSpec(
        id="original",
        title="Original (indiscriminate votes)",
        description=(
            "Single vote channel: ψ help-head is applied to all receivers; "
            "V_foe is zero. Kin-selective loss may still be active. "
            "This is the pre-step-A baseline for measuring typed-vote impact."
        ),
        typed_votes=False,
        implemented=True,
        hypothesis=(
            "Reproducer and eliminator alive counts track total density "
            "(corr≈1); votes do not specialize by receiver type."
        ),
    ),
    "A": VersionSpec(
        id="A",
        title="Step A — typed help/harm votes",
        description=(
            "ψ emits (v_help, v_harm). Survival uses "
            "V_kin = same-goal help votes and V_foe = opposite-goal harm votes "
            "with separate weights w4_help / w4_harm."
        ),
        typed_votes=True,
        implemented=True,
        hypothesis=(
            "Alive-count coupling drops; vote specialization "
            "(help→kin, harm→foe) emerges; type ratio can drift over time."
        ),
    ),
    "B": VersionSpec(
        id="B",
        title="Step B — predator–prey loss",
        description=(
            "Builds on A (typed votes). Loss is asymmetric: eliminators "
            "pressure only reproducer (prey) neighbours and are neutral on "
            "fellow eliminators; reproducers still protect kin and pressure foes."
        ),
        typed_votes=True,
        predator_prey_loss=True,
        implemented=True,
        hypothesis=(
            "Eliminator harm specializes on prey; death gap and vote disc "
            "for E improve vs A; less self-destructive elim pressure."
        ),
    ),
    "C_only": VersionSpec(
        id="C_only",
        title="Step C only — goal inheritance (isolated)",
        description=(
            "Goal inheritance alone: on birth, cells adopt majority pre-step "
            "alive-neighbour goals. Votes stay original (indiscriminate); "
            "loss stays pre-B (eliminators pressure all neighbours). "
            "Use this to measure colonization without A/B confounders."
        ),
        typed_votes=False,
        predator_prey_loss=False,
        goal_inheritance=True,
        implemented=True,
        hypothesis=(
            "Type fractions become dynamical via colonization on original "
            "physics; compare to original (no inherit) and to full C (A+B+inherit)."
        ),
    ),
    "C": VersionSpec(
        id="C",
        title="Step C — goal inheritance on A+B",
        description=(
            "On birth (dead→alive), a cell adopts the majority goal among "
            "pre-step alive Moore neighbours (tie → max-rho neighbour). "
            "Survivors and pure deaths keep goals; rho stays fixed. "
            "Full stack: typed votes (A) + predator–prey loss (B) + inheritance."
        ),
        typed_votes=True,
        predator_prey_loss=True,
        goal_inheritance=True,
        implemented=True,
        hypothesis=(
            "Goal-frac drifts (|final−init| ≫ 0); corr(ra, dens_using_init_frac) "
            "drops; type ratios and segregation become more dynamical (invasion)."
        ),
    ),
    "D_fixed": VersionSpec(
        id="D_fixed",
        title="Step D fixed — goal-conditioned f (no inheritance)",
        description=(
            "Local update f receives own goal as an input feature so "
            "proposed (s,h) and the f-survival channel can become "
            "type-specific. Goals stay fixed (no C). Stack: A+B+goal_in_f."
        ),
        typed_votes=True,
        predator_prey_loss=True,
        goal_inheritance=False,
        goal_in_f=True,
        implemented=True,
        hypothesis=(
            "With fixed goals, type-conditioned f raises policy/state "
            "specialization and may lift count-level class divergence Φ "
            "versus B alone."
        ),
    ),
    "D": VersionSpec(
        id="D",
        title="Step D — goal-conditioned f on A+B+C",
        description=(
            "Local update f receives own goal so memory/state strategies "
            "can become type-specific. Full stack: typed votes (A), "
            "predator–prey loss (B), goal inheritance (C), goal_in_f (D)."
        ),
        typed_votes=True,
        predator_prey_loss=True,
        goal_inheritance=True,
        goal_in_f=True,
        implemented=True,
        hypothesis=(
            "Type-specific internal dynamics plus dynamical goals: "
            "colonization with type-conditioned local updates."
        ),
    ),
    "E": VersionSpec(
        id="E",
        title="Step E — typed votes + symmetric R/E survival weights",
        description=(
            "Same as A (typed help/harm votes) but environment ablation: "
            "w2 = w3 = mean(base.w2, base.w3). Reproducer and eliminator "
            "neighbour counts enter the survival logit with equal weight, so "
            "count-level class divergence cannot be blamed on asymmetric "
            "fixed physics (benchmark has w2 ≫ w3). No new code path — only "
            "config weights."
        ),
        typed_votes=True,
        predator_prey_loss=False,
        goal_inheritance=False,
        goal_in_f=False,
        symmetrize_RE_weights=True,
        implemented=True,
        hypothesis=(
            "If Φ under E remains comparable to A at long horizon, typed "
            "votes alone drive fixed-goal class divergence; if Φ collapses "
            "toward original, benchmark divergence was partly w2≠w3."
        ),
    ),
    "F": VersionSpec(
        id="F",
        title="Step F — typed votes + soft coexistence pressure",
        description=(
            "Builds on A (typed votes). Adds a weak global soft barrier on "
            "soft living mass of each goal-class: "
            "B = λ(−log ρ̃^R − log ρ̃^E), added once to the step total loss. "
            "ρ̃ uses self soft survival probs only (Path-1 local through f). "
            "Viability regularizer so both classes stay observable under "
            "long-horizon learning; not forced 50/50. Default λ=0.01."
        ),
        typed_votes=True,
        predator_prey_loss=False,
        goal_inheritance=False,
        goal_in_f=False,
        coexistence_pressure=True,
        coexistence_lambda=0.01,
        coexistence_delta=1e-4,
        implemented=True,
        hypothesis=(
            "Under w2=w3, A+F raises seed success rate for large Φ_late "
            "vs A alone by reducing type extinction / full-grid nulls, "
            "without collapsing specialization if λ is small."
        ),
    ),
    "G": VersionSpec(
        id="G",
        title="Step G — typed votes + transfer blobs",
        description=(
            "Builds on A (typed votes). Frozen spatial environment: "
            "three hard transfer-dead disks (κ_lo=0) on the torus. "
            "η_scale stays 1. Occupancy off (transfer-only). "
            "env_seed is not pinned so suite --seeds share terrain."
        ),
        typed_votes=True,
        environment_heterogeneous=True,
        env_preset="blobs",
        env_n_blobs=3,
        env_blob_radius=0.15,
        env_kappa_lo=0.0,
        env_kappa_hi=1.0,
        env_eta_lo=1.0,
        env_eta_hi=1.0,
        env_occupancy_blocks=False,
        env_affect_R=True,
        env_affect_E=True,
        implemented=True,
        hypothesis=(
            "Spatial transfer barriers change Φ vs A by fragmenting "
            "information flow without changing vote routing or loss shape."
        ),
    ),
    "G_learn": VersionSpec(
        id="G_learn",
        title="Step G_learn — typed votes + learning hotspot",
        description=(
            "Builds on A. Center disk learns at η_hi=1; exterior at η_lo=0.25. "
            "κ stays 1 (no transfer barrier). Occupancy off."
        ),
        typed_votes=True,
        environment_heterogeneous=True,
        env_preset="learning_hotspot",
        env_eta_lo=0.25,
        env_eta_hi=1.0,
        implemented=True,
        hypothesis=(
            "Spatially varying learning rate concentrates adaptation in a "
            "hotspot; Φ vs A may change via slower exterior SGD rather than "
            "blocked messages."
        ),
    ),
}

# Display / ablation order for charts and reports (not alphabetical).
# E sits next to A so the w2=w3 identity check is obvious on a symmetric base.
LADDER_ORDER: tuple[str, ...] = (
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


def ladder_sort_key(version_id: str) -> tuple[int, str]:
    """Sort key: canonical ladder, then alphabetical fallback."""
    try:
        return (LADDER_ORDER.index(version_id), version_id)
    except ValueError:
        return (len(LADDER_ORDER), version_id)


# Aliases for CLI convenience.
_ALIASES: dict[str, str] = {
    "c_only": "C_only",
    "C-only": "C_only",
    "inheritance": "C_only",
    "inherit_only": "C_only",
    "d_fixed": "D_fixed",
    "D-fixed": "D_fixed",
    "Df": "D_fixed",
    "A_sym": "E",
    "a_sym": "E",
    "A_coexist": "F",
    "a_coexist": "F",
    "A+F": "F",
    "A_env": "G",
    "a_env": "G",
    "A_hetero": "G",
    "A+G": "G",
    "A_env_learn": "G_learn",
    "a_env_learn": "G_learn",
}


def get_version(version_id: str) -> VersionSpec:
    key = version_id.strip()
    key = _ALIASES.get(key, key)
    if key not in VERSIONS:
        known = ", ".join(VERSIONS)
        raise KeyError(f"Unknown version {version_id!r}. Known: {known}")
    return VERSIONS[key]


def parse_version_list(spec: str) -> list[VersionSpec]:
    """Parse 'original,A' or 'all' into implemented VersionSpecs."""
    spec = spec.strip()
    if spec.lower() == "all":
        ids = [v.id for v in VERSIONS.values() if v.implemented]
    else:
        ids = [p.strip() for p in spec.split(",") if p.strip()]
    out = [get_version(i) for i in ids]
    for v in out:
        if not v.implemented:
            raise RuntimeError(
                f"Version {v.id} is not implemented yet. "
                f"Remove it from --versions or implement the feature first."
            )
    return out
