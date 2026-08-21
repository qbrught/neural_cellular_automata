"""Gemini Flash VLM judge + guided next-config proposals."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from discovery.sample import FIELD_RANGES, GUIDABLE_FIELDS


SYSTEM_BRIEF = """You evaluate Neural State-Aware Cellular Automaton (NCSA) runs and
steer the search toward interesting, non-crashing dynamics.

Each cell is a tiny MLP; learning is ON. Survival is a fixed logistic rule with
weights w0..w5:
  w0 bias (more negative → more death)
  w1 total alive neighbours
  w2 alive reproducer neighbours (often helps population persist)
  w3 alive eliminator neighbours (often negative = penalty)
  w4_help kin help votes; w4_harm foe harm votes (typed channels)
  w5 f-signal from local state update (learnable self-survival channel)
Also: init_alive_prob, eta (learning rate), init_noise_std.

The image is a summary panel:
- Top-left: final grid — green = alive reproducer, red = alive eliminator, dark = dead
- Top-right: total alive vs step
- Bottom-left: alive by goal vs step
- Bottom-right: mean losses vs step

You both (1) judge if the run is worth saving and (2) propose the NEXT config
to try, based on this run and recent history. Prefer small directed steps when
recovering from extinction/static, larger jumps only if stuck."""

# Extra brief when discovering under Step E (typed votes + symmetric w2=w3).
SYSTEM_BRIEF_E = """
=== STEP E / SYMMETRY NULL (this catalog) ===
typed_votes ON. Survival weights are type-symmetric: w2 = w3. Reproducer and
eliminator neighbour counts enter the survival logit with equal weight, so
class divergence cannot be blamed on fixed physics favoring one type.

What to SEEK and SAVE under E:
- Two types with distinct temporal curves (anti-correlated, delayed, or
  role-swapping) despite equal neighbour weights
- Spatial domains, fronts, or clustered niches that persist
- Learning-driven phase change that is not just one type dying
- Residual structure vs the density-tracking null (repro ≠ scaled total)

What to REJECT under E (the scientific null):
- Density-tracking: green and red alive curves are scaled copies of total
- Saturated mixed slab with no spatial or temporal structure
- Unstructured flicker
- Immediate total extinction with no mixed transient

Learning-driven competitive exclusion AFTER a mixed/saturated transient
(one type declines, the other holds) is a hit, not a reject.
"""

# Extra brief when only the IC seed changes (frozen physics).
SYSTEM_BRIEF_SEED_ONLY = """
=== SEED SEARCH — SEED BANK (frozen physics) ===
Survival weights, eta, init density, and u_seed are FIXED. Each trial is a
new random initial condition (seed) on the same rule.

Do NOT propose weight changes. Search continues by drawing a new seed.
"""

# Original + w2=w3: fair IC bank for later flag ablations. Do NOT filter for
# class divergence — that would re-select the A/E attrition motif.
SYSTEM_BRIEF_ORIGINAL_SEEDS = """
=== ORIGINAL BASELINE SEED BANK (indiscriminate votes, w2=w3) ===
typed_votes OFF. Survival weights are type-symmetric (w2=w3). This is the
density-tracking null: no typed routing and no R/E neighbour-weight bias.

The catalog is a FAIR set of initial conditions for later A/B/C/D/E/F
comparisons. Do NOT prefer seeds where one type wipes the other — that
biases the ablation toward A's attrition attractor.

SAVE (interesting=true, novel=true, worth_saving=true) when the run is viable:
- Both types still present late (mixed slab, slow co-decay, mild imbalance)
- Spatial texture or delayed change even if green/red track total density
- Oscillations / waves / persistent mixed life
Mixed density-tracking ICs ARE wanted. Mention late mix vs wipe in the one_liner.

REJECT:
- Total extinction (whole grid dead early and stays dead)
- Immediate empty freeze
Do NOT reject a seed because green and red track density. That is the baseline.
Do NOT reject mixed full-alive slabs unless they are totally static AND you
already have several of that exact story — still prefer them over exclusion.

Prefer diversity: mix of (persistent mix, slow decay, mild spatial structure).
Avoid filling the catalog with 20 copies of "reproducers exclude eliminators".
"""

# Extra brief when only the IC seed changes under E (typed votes + w2=w3).
SYSTEM_BRIEF_E_SEEDS = """
=== E SEED BANK (typed votes, w2=w3) ===
Save independent ICs that beat the density-tracking null under typed votes.
Same motif on a new seed is still worth saving. Reject extinction / mixed slabs.
"""

# Extra brief when discovering under Step C (goal inheritance / colonization).
SYSTEM_BRIEF_C = """
=== STEP C REGIME (this catalog) ===
goal_inheritance is ON: when a dead cell becomes alive (birth), it adopts the
majority goal of its pre-step alive Moore neighbours (colonization). Goals are
NOT fixed labels — type fractions can expand or collapse.

Also on: typed help/harm votes (A) and predator–prey loss (B).

What to SEEK and SAVE under C:
- Moving invasion fronts / domain walls between green and red that persist
- Coexistence with slow colonization (both colors remain for long times)
- Oscillating or cyclic type-fraction wars (not a one-shot wipe)
- Spatial niches: one type holds cores, the other borders, with births flipping
- Delayed takeover or phase change in type composition (not just density)

What to REJECT under C (common failure mode):
- Quick monoculture: one color colonizes the whole grid, then saturates all-alive
- Static full-green or full-red slab after ~mid run
- Extinction of the whole population
- Unstructured flicker with no type structure

When recovering from monoculture wipeouts: weaken the dominant type's advantage
(e.g. lower w2 if green always wins, make w3 more negative / less friendly to
elim if red dominates, increase death bias w0 more negative, lower init density,
tune w4_help/w4_harm so fronts stay contested). Prefer configs that KEEP both
colors visible late in the run.
"""

JUDGE_INSTRUCTIONS = """Tasks:
A) Judge whether THIS run is worth saving in a curated discovery set.
B) Propose the next config to try (guided search).

=== Judgment ===
interesting: non-trivial structure or temporal behavior (waves, coexistence,
expanding/contracting fronts, persistent niches, phase changes, patterned domains).
Reject pure extinction, pure full-alive slabs, pure unstructured noise, or dead freeze.
{version_criteria}

novel: not the same story as any listed prior discovery.

worth_saving: true only if interesting AND novel.

one_liner: ≤~20 words catalog caption if worth_saving or interesting; else short note.
  Do NOT put a version tag in the one_liner (catalog adds that).

boring_reason: if not interesting, short reason; else null.
similarity_note: if not novel, which prior it resembles; else null.

analysis: 1-3 sentences. What happened dynamically, and how the knobs likely caused it.

=== Guided next config ===
strategy: one of
  recover_extinction | break_static | refine_interesting | diversify | explore_jump
  | break_monoculture

rationale: short explanation of the proposed move (e.g. "raise w2 after collapse").

next_config: object with ABSOLUTE values for knobs to set on the NEXT run.
  Include only fields you want to change (others keep current values).
  Allowed keys: {keys}
  Valid ranges (must respect):
{ranges}

Guidance rules:
- If extinct / near-dead: move toward survival with SMALL steps (e.g. raise w2,
  make w0 less negative, slightly higher init_alive_prob). Do NOT randomize everything.
- If static/saturated freeze: adjust w4_help/w4_harm/w5/eta or death bias modestly to restore change.
- If one type wiped the other then filled the grid: break_monoculture — reduce the
  winner's survival edge and rebalance w2/w3/w0/votes so both colors can persist.
- If interesting but not novel: keep the basin, small refine OR diversify one channel.
- If interesting and saved: refine locally (small jitter in weights).
- If stuck repeating failures: explore_jump with a bolder but still in-range change.
- Prefer changing 2-5 knobs, not all nine, unless explore_jump.
- Always return a next_config (never null) so search can continue.

Return ONLY JSON with keys:
interesting, novel, worth_saving, one_liner, boring_reason, similarity_note,
analysis, strategy, rationale, next_config
"""


@dataclass
class JudgeResult:
    interesting: bool
    novel: bool
    worth_saving: bool
    one_liner: str
    boring_reason: str | None
    similarity_note: str | None
    analysis: str = ""
    strategy: str = ""
    rationale: str = ""
    next_config: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def _ranges_block() -> str:
    lines = []
    for k in GUIDABLE_FIELDS:
        lo, hi = FIELD_RANGES[k]
        lines.append(f"  {k}: [{lo}, {hi}]")
    return "\n".join(lines)


def _parse_result(data: dict[str, Any]) -> JudgeResult:
    interesting = bool(data.get("interesting"))
    novel = bool(data.get("novel"))
    worth = bool(data.get("worth_saving")) and interesting and novel
    one_liner = (data.get("one_liner") or "").strip()
    if worth and not one_liner:
        one_liner = "Interesting dynamics (no caption returned)."

    next_cfg = data.get("next_config") or {}
    if not isinstance(next_cfg, dict):
        next_cfg = {}

    return JudgeResult(
        interesting=interesting,
        novel=novel,
        worth_saving=worth,
        one_liner=one_liner,
        boring_reason=data.get("boring_reason"),
        similarity_note=data.get("similarity_note"),
        analysis=(data.get("analysis") or "").strip(),
        strategy=(data.get("strategy") or "").strip(),
        rationale=(data.get("rationale") or "").strip(),
        next_config=next_cfg,
        raw=data,
        error=None,
    )


def _is_inheritance_version(version: str | None) -> bool:
    if not version:
        return False
    v = version.strip().lower().replace("-", "_")
    return v in ("c", "c_only", "inheritance", "inherit_only")


def _is_symmetry_null_version(version: str | None) -> bool:
    if not version:
        return False
    v = version.strip().lower().replace("-", "_")
    return v in ("e", "a_sym")


def _is_original_version(version: str | None) -> bool:
    if not version:
        return False
    return version.strip().lower() == "original"


def _version_criteria(version: str | None, seed_only: bool = False) -> str:
    extra = ""
    if _is_inheritance_version(version):
        extra = (
            "Under goal inheritance: interesting means contested colonization "
            "(both types late, moving fronts, type-fraction dynamics). "
            "Reject fast monoculture wipeouts and all-alive single-color slabs."
        )
        if version and version.strip().lower().replace("-", "_") in (
            "c_only", "inheritance", "inherit_only",
        ):
            extra += (
                " This is C_only: inheritance ON but typed votes OFF and "
                "predator–prey loss OFF (original vote/loss physics + colonization)."
            )
    elif _is_symmetry_null_version(version):
        extra = (
            "Under symmetric w2=w3: interesting means type dynamics that beat "
            "the density-tracking null (distinct curves, spatial domains). "
            "Reject scaled-copy populations, one-type wipeouts, and mixed slabs."
        )
    if seed_only and _is_original_version(version):
        extra += (
            " Original+w2=w3 seed bank: save viable mixed ICs (both types late). "
            "Density-tracking mixed life is a SAVE. Competitive exclusion is "
            "allowed but do not fill the catalog with only that motif. "
            "Reject total extinction."
        )
    elif seed_only:
        extra += (
            " Seed-bank mode: save every non-null IC even if the motif matches "
            "a prior catalog entry. Set novel=true unless the run is a saturated "
            "slab, density-tracking null, or extinction. One-liners should note "
            "who leads and roughly when divergence starts."
        )
    return extra


def _system_brief(version: str | None, seed_only: bool = False) -> str:
    brief = SYSTEM_BRIEF
    if _is_inheritance_version(version):
        brief = brief + "\n" + SYSTEM_BRIEF_C
        if version and version.strip().lower().replace("-", "_") in (
            "c_only", "inheritance", "inherit_only",
        ):
            brief += (
                "\n=== C_only (isolated inheritance) ===\n"
                "typed_votes=False, predator_prey_loss=False. Votes are "
                "indiscriminate; eliminators still learn to pressure all neighbours. "
                "Only colonization (birth goal adoption) is new. Seek patterns "
                "driven by inheritance alone, not typed warfare.\n"
            )
    if _is_symmetry_null_version(version):
        brief = brief + "\n" + SYSTEM_BRIEF_E
    if seed_only:
        brief = brief + "\n" + SYSTEM_BRIEF_SEED_ONLY
        if _is_original_version(version):
            brief = brief + "\n" + SYSTEM_BRIEF_ORIGINAL_SEEDS
        elif _is_symmetry_null_version(version):
            brief = brief + "\n" + SYSTEM_BRIEF_E_SEEDS
    return brief


def _build_user_text(
    *,
    catalog_one_liners: list[str],
    config_knobs: dict[str, Any],
    metrics_summary: str,
    prefilter_reason: str | None,
    history_lines: list[str],
    version: str | None = None,
    seed_only: bool = False,
) -> str:
    keys = ", ".join(GUIDABLE_FIELDS)
    instructions = JUDGE_INSTRUCTIONS.format(
        keys=keys,
        ranges=_ranges_block(),
        version_criteria=_version_criteria(version, seed_only=seed_only),
    )
    if seed_only:
        if _is_original_version(version):
            instructions += (
                "\n\nORIGINAL BASELINE SEED-BANK: next_config ignored. "
                "worth_saving=true for viable mixed ICs (both types persist). "
                "Density-tracking is a SAVE, not a reject. Prefer diversity "
                "over 20 copies of competitive exclusion.\n"
            )
        else:
            instructions += (
                "\n\nSEED-BANK MODE: next_config will be ignored. Return an empty "
                "next_config. worth_saving=true for every non-null seed, including "
                "repeats of a known motif on a new seed. novel=false only for the "
                "density-tracking / saturated-slab / extinction null.\n"
            )

    if catalog_one_liners:
        prior = "\n".join(f"- {line}" for line in catalog_one_liners)
    else:
        prior = "- (none yet)"

    if history_lines:
        hist = "\n".join(history_lines)
    else:
        hist = "- (none — this is the first trial)"

    pf = prefilter_reason or "passed (or not applied)"
    ver = f"Paper version under search: {version}\n\n" if version else ""
    if seed_only:
        ver += "Mode: seed-only (weights frozen; only seed changes).\n\n"

    return (
        f"{instructions}\n\n"
        f"{ver}"
        f"Already saved discoveries (id: one-liner):\n{prior}\n\n"
        f"Recent trial history (oldest → newest):\n{hist}\n\n"
        f"CURRENT trial config knobs:\n{json.dumps(config_knobs, indent=2)}\n\n"
        f"CURRENT metrics:\n{metrics_summary}\n\n"
        f"Prefilter: {pf}\n"
    )


def judge_trial(
    summary_png: Path,
    catalog_one_liners: list[str],
    *,
    config_knobs: dict[str, Any],
    metrics_summary: str,
    prefilter_reason: str | None = None,
    history_lines: list[str] | None = None,
    model: str = "gemini-3.5-flash",
    max_retries: int = 2,
    temperature: float = 0.35,
    version: str | None = None,
    seed_only: bool = False,
) -> JudgeResult:
    """Call Gemini on summary.png + config/history; return judgment + next_config."""
    summary_png = Path(summary_png)
    empty = JudgeResult(
        interesting=False,
        novel=False,
        worth_saving=False,
        one_liner="",
        boring_reason=None,
        similarity_note=None,
    )
    if not summary_png.exists():
        empty.boring_reason = "summary.png missing"
        empty.error = "summary.png missing"
        return empty

    key = _api_key()
    if not key:
        empty.error = "Set GEMINI_API_KEY or GOOGLE_API_KEY"
        return empty

    image_bytes = summary_png.read_bytes()
    user_text = _build_user_text(
        catalog_one_liners=catalog_one_liners,
        config_knobs=config_knobs,
        metrics_summary=metrics_summary,
        prefilter_reason=prefilter_reason,
        history_lines=history_lines or [],
        version=version,
        seed_only=seed_only,
    )
    last_err: str | None = None
    system = _system_brief(version, seed_only=seed_only)

    for attempt in range(max_retries + 1):
        try:
            data = _call_gemini(
                model=model,
                api_key=key,
                image_bytes=image_bytes,
                user_text=user_text,
                temperature=temperature,
                system_instruction=system,
            )
            return _parse_result(data)
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))

    empty.error = last_err
    return empty


def _call_gemini(
    *,
    model: str,
    api_key: str,
    image_bytes: bytes,
    user_text: str,
    temperature: float,
    system_instruction: str = SYSTEM_BRIEF,
) -> dict[str, Any]:
    try:
        return _call_google_genai(
            model=model,
            api_key=api_key,
            image_bytes=image_bytes,
            user_text=user_text,
            temperature=temperature,
            system_instruction=system_instruction,
        )
    except ImportError:
        return _call_generativeai(
            model=model,
            api_key=api_key,
            image_bytes=image_bytes,
            user_text=user_text,
            temperature=temperature,
            system_instruction=system_instruction,
        )


def _call_google_genai(
    *,
    model: str,
    api_key: str,
    image_bytes: bytes,
    user_text: str,
    temperature: float,
    system_instruction: str = SYSTEM_BRIEF,
) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                types.Part.from_text(text=user_text),
            ],
        )
    ]
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        response_mime_type="application/json",
    )
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    text = getattr(response, "text", None) or ""
    if not text and getattr(response, "candidates", None):
        parts = response.candidates[0].content.parts
        text = "".join(getattr(p, "text", "") or "" for p in parts)
    return _extract_json(text)


def _call_generativeai(
    *,
    model: str,
    api_key: str,
    image_bytes: bytes,
    user_text: str,
    temperature: float,
    system_instruction: str = SYSTEM_BRIEF,
) -> dict[str, Any]:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_instruction,
        generation_config={
            "temperature": temperature,
            "response_mime_type": "application/json",
        },
    )
    response = m.generate_content(
        [
            {"mime_type": "image/png", "data": image_bytes},
            user_text,
        ]
    )
    return _extract_json(response.text or "")
