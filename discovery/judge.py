"""Gemini Flash VLM judge: interesting + novel + one-liner."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SYSTEM_BRIEF = """You evaluate Neural State-Aware Cellular Automaton (NCSA) runs.
Each cell is a tiny MLP with a fixed goal (reproduce or eliminate) and learning is ON.
Survival follows a fixed global rule; cells learn to exploit it.

The image is a summary panel:
- Top-left: final grid — green = alive reproducer, red = alive eliminator, dark = dead
- Top-right: total alive vs step
- Bottom-left: alive by goal vs step
- Bottom-right: mean losses vs step

Judge whether the dynamics are worth keeping in a curated discovery set."""


JUDGE_INSTRUCTIONS = """Decide if this run shows dynamics worth saving.

interesting: non-trivial structure or temporal behavior (waves, coexistence,
expanding/contracting fronts, persistent niches, clear phase changes, patterned
domains). Reject pure extinction, pure full-alive slabs, pure unstructured noise,
or immediate freeze with nothing going on.

novel: not the same story as any listed prior discovery. Superficial seed changes
that yield the same qualitative dynamics are NOT novel.

worth_saving: true only if interesting AND novel.

one_liner: at most ~20 words, concrete, no fluff — suitable as a catalog caption.

boring_reason: short string if not interesting, else null.
similarity_note: if not novel, say which prior discovery it resembles; else null.

Return ONLY a JSON object with keys:
interesting, novel, worth_saving, one_liner, boring_reason, similarity_note
(booleans for the first three; strings or null for the rest)."""


@dataclass
class JudgeResult:
    interesting: bool
    novel: bool
    worth_saving: bool
    one_liner: str
    boring_reason: str | None
    similarity_note: str | None
    raw: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    # Strip markdown fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def _parse_result(data: dict[str, Any]) -> JudgeResult:
    interesting = bool(data.get("interesting"))
    novel = bool(data.get("novel"))
    # Trust model but enforce logical AND if they drift.
    worth = bool(data.get("worth_saving")) and interesting and novel
    one_liner = (data.get("one_liner") or "").strip()
    if worth and not one_liner:
        one_liner = "Interesting dynamics (no caption returned)."
    return JudgeResult(
        interesting=interesting,
        novel=novel,
        worth_saving=worth,
        one_liner=one_liner,
        boring_reason=data.get("boring_reason"),
        similarity_note=data.get("similarity_note"),
        raw=data,
        error=None,
    )


def _build_user_text(catalog_one_liners: list[str]) -> str:
    if catalog_one_liners:
        prior = "\n".join(f"- {line}" for line in catalog_one_liners)
    else:
        prior = "- (none yet)"
    return (
        f"{JUDGE_INSTRUCTIONS}\n\n"
        f"Already saved discoveries (id: one-liner):\n{prior}\n"
    )


def judge_trial(
    summary_png: Path,
    catalog_one_liners: list[str],
    *,
    model: str = "gemini-3.5-flash",
    max_retries: int = 2,
    temperature: float = 0.3,
) -> JudgeResult:
    """Call Gemini Flash on summary.png; return structured judgment."""
    summary_png = Path(summary_png)
    if not summary_png.exists():
        return JudgeResult(
            interesting=False,
            novel=False,
            worth_saving=False,
            one_liner="",
            boring_reason="summary.png missing",
            similarity_note=None,
            error="summary.png missing",
        )

    key = _api_key()
    if not key:
        return JudgeResult(
            interesting=False,
            novel=False,
            worth_saving=False,
            one_liner="",
            boring_reason=None,
            similarity_note=None,
            error="Set GEMINI_API_KEY or GOOGLE_API_KEY",
        )

    image_bytes = summary_png.read_bytes()
    user_text = _build_user_text(catalog_one_liners)
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
        except Exception as e:  # noqa: BLE001 — surface any API/parse failure
            last_err = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))

    return JudgeResult(
        interesting=False,
        novel=False,
        worth_saving=False,
        one_liner="",
        boring_reason=None,
        similarity_note=None,
        error=last_err,
    )


def _call_gemini(
    *,
    model: str,
    api_key: str,
    image_bytes: bytes,
    user_text: str,
    temperature: float,
) -> dict[str, Any]:
    """Call Google GenAI; prefer google.genai, fall back to google.generativeai."""
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
        # Fallback scrape
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
