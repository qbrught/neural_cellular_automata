"""Cross-version comparison tables and aggregations."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from research.metrics import TABLE_COLUMNS


def aggregate_by_version(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean ± over seeds for each scalar summary key, per version."""
    by_v: dict[str, list[dict]] = {}
    for r in results:
        by_v.setdefault(r["version_id"], []).append(r["summary"])

    out: dict[str, dict[str, float]] = {}
    for vid, rows in by_v.items():
        keys = rows[0].keys()
        agg: dict[str, float] = {}
        for k in keys:
            vals = np.array([row[k] for row in rows], dtype=float)
            # extinct_step may be None
            if k == "extinct_step":
                nums = [row[k] for row in rows if row[k] is not None]
                agg["extinct_rate"] = float(len(nums) / len(rows))
                agg["mean_extinct_step"] = (
                    float(np.mean(nums)) if nums else float("nan")
                )
                continue
            agg[f"{k}_mean"] = float(np.nanmean(vals))
            agg[f"{k}_std"] = float(np.nanstd(vals)) if len(vals) > 1 else 0.0
        out[vid] = agg
    return out


def write_summary_csv(results: list[dict[str, Any]], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        path.write_text("")
        return path
    has_config = any(r.get("config_id") for r in results)
    base = (["config_id", "config_title"] if has_config else []) + ["version", "seed"]
    fieldnames = base + list(results[0]["summary"].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {
                "version": r["version_id"],
                "seed": r["seed"],
                **r["summary"],
            }
            if has_config:
                row["config_id"] = r.get("config_id", "")
                row["config_title"] = r.get("config_title", "")
            w.writerow(row)
    return path


def markdown_comparison_table(
    results: list[dict[str, Any]],
    *,
    include_config: bool = False,
) -> str:
    """One row per version (mean over seeds) of key paper metrics.

    If include_config and results carry config_id, rows are (config, version).
    """
    # Group key: (config_id or "", version_id)
    by_key: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        cid = str(r.get("config_id") or "") if include_config else ""
        by_key.setdefault((cid, r["version_id"]), []).append(r["summary"])

    headers = (
        (["config", "version", "n_seeds"] if include_config else ["version", "n_seeds"])
        + [h for _, h, _ in TABLE_COLUMNS]
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for (cid, vid) in sorted(by_key):
        rows = by_key[(cid, vid)]
        cells = ([cid, vid, str(len(rows))] if include_config else [vid, str(len(rows))])
        for key, _h, fmt in TABLE_COLUMNS:
            if fmt == "s":
                # extinct step: show rate
                n_ext = sum(1 for row in rows if row.get("extinct_step") is not None)
                cells.append(f"{n_ext}/{len(rows)}")
                continue
            vals = []
            for row in rows:
                v = row.get(key)
                if v is None:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(fv):
                    vals.append(fv)
            if vals:
                mean = float(np.mean(vals))
                cells.append(format(mean, fmt))
            else:
                cells.append("nan")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def delta_table(results: list[dict[str, Any]], baseline: str = "original") -> str:
    """Markdown table of (version − baseline) for key metrics."""
    by_v: dict[str, list[dict]] = {}
    for r in results:
        by_v.setdefault(r["version_id"], []).append(r["summary"])
    if baseline not in by_v:
        return f"_Baseline {baseline!r} not in results; skip delta table._"

    def mean_key(vid: str, key: str) -> float:
        vals = []
        for row in by_v[vid]:
            v = row.get(key)
            if v is None:
                vals.append(np.nan)
            else:
                vals.append(float(v))
        return float(np.nanmean(vals))

    headers = ["version"] + [f"Δ {h}" for _, h, _ in TABLE_COLUMNS if _ != "s"]
    # skip extinct for deltas
    keys = [(k, h, f) for k, h, f in TABLE_COLUMNS if f != "s"]
    headers = ["version"] + [f"Δ {h}" for _, h, _ in keys]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    base = {k: mean_key(baseline, k) for k, _, _ in keys}
    for vid in sorted(by_v):
        if vid == baseline:
            continue
        cells = [f"{vid} − {baseline}"]
        for k, _h, fmt in keys:
            d = mean_key(vid, k) - base[k]
            cells.append(format(d, f"{'+' if d >= 0 else ''}{fmt}") if np.isfinite(d) else "nan")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
