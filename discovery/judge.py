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

Each cell is a tiny MLP with a fixed goal (reproduce or eliminate); learning is ON.
Survival is a fixed logistic rule with weights w0..w5:
  w0 bias (more negative → more death)
  w1 total alive neighbours
  w2 alive reproducer neighbours (often helps population persist)
  w3 alive eliminator neighbours (often negative = penalty)
  w4 weighted vote sum (learnable communication channel)
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


JUDGE_INSTRUCTIONS = """Tasks:
A) Judge whether THIS run is worth saving in a curated discovery set.
B) Propose the next config to try (guided search).

=== Judgment ===
interesting: non-trivial structure or temporal behavior (waves, coexistence,
expanding/contracting fronts, persistent niches, phase changes, patterned domains).
Reject pure extinction, pure full-alive slabs, pure unstructured noise, or dead freeze.

novel: not the same story as any listed prior discovery.

worth_saving: true only if interesting AND novel.

one_liner: ≤~20 words catalog caption if worth_saving or interesting; else short note.

boring_reason: if not interesting, short reason; else null.
similarity_note: if not novel, which prior it resembles; else null.

analysis: 1-3 sentences. What happened dynamically, and how the knobs likely caused it.

=== Guided next config ===
strategy: one of
  recover_extinction | break_static | refine_interesting | diversify | explore_jump

rationale: short explanation of the proposed move (e.g. "raise w2 after collapse").

next_config: object with ABSOLUTE values for knobs to set on the NEXT run.
  Include only fields you want to change (others keep current values).
  Allowed keys: {keys}
  Valid ranges (must respect):
{ranges}

Guidance rules:
- If extinct / near-dead: move toward survival with SMALL steps (e.g. raise w2,
  make w0 less negative, slightly higher init_alive_prob). Do NOT randomize everything.
- If static/saturated freeze: adjust w4/w5/eta or death bias modestly to restore change.
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


def _build_user_text(
    *,
    catalog_one_liners: list[str],
    config_knobs: dict[str, Any],
    metrics_summary: str,
    prefilter_reason: str | None,
    history_lines: list[str],
) -> str:
    keys = ", ".join(GUIDABLE_FIELDS)
    instructions = JUDGE_INSTRUCTIONS.format(keys=keys, ranges=_ranges_block())

    if catalog_one_liners:
        prior = "\n".join(f"- {line}" for line in catalog_one_liners)
    else:
        prior = "- (none yet)"

    if history_lines:
        hist = "\n".join(history_lines)
    else:
        hist = "- (none — this is the first trial)"

    pf = prefilter_reason or "passed (or not applied)"

    return (
        f"{instructions}\n\n"
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
    )
    last_err: str | None = None

    for attempt in range(max_retries + 1):
        try:
            data = _call_gemini(
                model=model,
                api_key=key,
                image_bytes=image_bytes,
                user_text=user_text,
                temperature=temperature,
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
) -> dict[str, Any]:
    try:
        return _call_google_genai(
            model=model,
            api_key=api_key,
            image_bytes=image_bytes,
            user_text=user_text,
            temperature=temperature,
        )
    except ImportError:
        return _call_generativeai(
            model=model,
            api_key=api_key,
            image_bytes=image_bytes,
            user_text=user_text,
            temperature=temperature,
        )


def _call_google_genai(
    *,
    model: str,
    api_key: str,
    image_bytes: bytes,
    user_text: str,
    temperature: float,
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
        system_instruction=SYSTEM_BRIEF,
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
) -> dict[str, Any]:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(
        model_name=model,
        system_instruction=SYSTEM_BRIEF,
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
