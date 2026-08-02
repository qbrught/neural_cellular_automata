"""Paper version registry.

Each version is a named ablation of the system defined by Config flags.
The suite applies these flags on top of a shared base config + seed so
comparisons isolate the mechanism change, not hyperparameters.

Roadmap (matches the planned A→D path):
  original  — single indiscriminate vote channel
  A         — typed help/harm votes routed by kin/foe   [implemented]
  B         — predator–prey losses                       [implemented]
  C_only    — goal inheritance alone (no A/B)            [implemented]
  C         — goal inheritance on top of A+B             [implemented]
  D         — own goal into f                            [flag reserved]
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
        return replace(
            cfg,
            typed_votes=self.typed_votes,
            predator_prey_loss=self.predator_prey_loss,
            goal_inheritance=self.goal_inheritance,
            goal_in_f=self.goal_in_f,
        )

    def flag_summary(self) -> str:
        """Short human-readable flag dump for VERSION.txt / notes."""
        return (
            f"typed_votes={self.typed_votes}\n"
            f"predator_prey_loss={self.predator_prey_loss}\n"
            f"goal_inheritance={self.goal_inheritance}\n"
            f"goal_in_f={self.goal_in_f}\n"
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
    "D": VersionSpec(
        id="D",
        title="Step D — goal-conditioned f",
        description=(
            "Local update f receives own goal so memory/state strategies "
            "can become type-specific."
        ),
        typed_votes=True,
        predator_prey_loss=True,
        goal_inheritance=True,
        goal_in_f=True,
        implemented=False,
        hypothesis=(
            "Type-specific internal dynamics (clustering vs hunting) "
            "appear in s/h and local environment stats."
        ),
    ),
}

# Aliases for CLI convenience.
_ALIASES: dict[str, str] = {
    "c_only": "C_only",
    "C-only": "C_only",
    "inheritance": "C_only",
    "inherit_only": "C_only",
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
