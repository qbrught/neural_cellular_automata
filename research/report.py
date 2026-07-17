"""Write human-readable research reports for a suite run."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from research.compare import (
    aggregate_by_version,
    delta_table,
    markdown_comparison_table,
    write_summary_csv,
)
from research.versions import VERSIONS, VersionSpec


def write_report(
    results: list[dict[str, Any]],
    versions: list[VersionSpec],
    out_dir: Path,
    *,
    chart_paths: dict[str, Path] | None = None,
    base_config_note: str = "",
    n_steps: int,
    seeds: list[int],
) -> Path:
    """Write REPORT.md + NOTES.md + summary.csv under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_summary_csv(results, out_dir / "summary.csv")

    lines: list[str] = []
    lines.append("# Research suite report")
    lines.append("")
    lines.append(f"- Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- Steps per run: **{n_steps}**")
    lines.append(f"- Seeds: `{seeds}`")
    lines.append(
        f"- Versions: {', '.join(f'`{v.id}`' for v in versions)}"
    )
    if base_config_note:
        lines.append(f"- Base config: {base_config_note}")
    lines.append("")
    lines.append("## How to use this folder")
    lines.append("")
    lines.append("| Path | Contents |")
    lines.append("| --- | --- |")
    lines.append("| `REPORT.md` | This file — quantitative comparison |")
    lines.append("| `NOTES.md` | Blank template for manual observations |")
    lines.append("| `summary.csv` | Per (version, seed) scalar metrics |")
    lines.append("| `comparison/` | Overlay charts across versions |")
    lines.append("| `versions/<id>/seed_<n>/` | Per-run series, charts, config |")
    lines.append("")

    lines.append("## Versions under test")
    lines.append("")
    for v in versions:
        lines.append(f"### `{v.id}` — {v.title}")
        lines.append("")
        lines.append(v.description)
        lines.append("")
        lines.append(f"**Hypothesis:** {v.hypothesis}")
        lines.append("")
        lines.append(
            f"Flags: `typed_votes={v.typed_votes}`, "
            f"`predator_prey_loss={v.predator_prey_loss}`, "
            f"`goal_inheritance={v.goal_inheritance}`, "
            f"`goal_in_f={v.goal_in_f}`"
        )
        lines.append("")

    lines.append("## Summary table (mean over seeds)")
    lines.append("")
    lines.append(markdown_comparison_table(results))
    lines.append("")
    lines.append("### Deltas vs `original`")
    lines.append("")
    lines.append(delta_table(results, baseline="original"))
    lines.append("")

    lines.append("## Metric definitions")
    lines.append("")
    lines.append(
        "| Metric | Meaning |\n"
        "| --- | --- |\n"
        "| corr(ra,ea) | Correlation of reproducer vs eliminator alive counts. "
        "~1 means both tracks are the same density curve. |\n"
        "| corr(ra, dens) | Correlation of ra with (goal_frac × total alive). "
        "High ⇒ type counts are pure density samples. |\n"
        "| \\|ra residual\\| | Mean \\|ra − frac·total\\|. Higher ⇒ type-specific dynamics. |\n"
        "| ra/ea early/late | Population ratio; drift shows role divergence over time. |\n"
        "| corr(Lr,Le) | Loss coupling. ~−1 under mirror losses; higher if roles differ. |\n"
        "| R/E vote disc late | Vote specialization (help_kin−harm_foe for R; "
        "harm_foe−help_kin for E) in the last 10% of steps. |\n"
        "| extinct@ | Fraction of seeds that hit total extinction. |\n"
    )
    lines.append("")

    if chart_paths:
        lines.append("## Comparison charts")
        lines.append("")
        out_res = out_dir.resolve()
        for key, p in chart_paths.items():
            try:
                rel = Path(p).resolve().relative_to(out_res)
            except ValueError:
                rel = Path(p)
            lines.append(f"### {key}")
            lines.append("")
            lines.append(f"![{key}]({rel.as_posix()})")
            lines.append("")

    lines.append("## Per-version mean scalars")
    lines.append("")
    agg = aggregate_by_version(results)
    for vid in sorted(agg):
        lines.append(f"### `{vid}`")
        lines.append("")
        lines.append("```")
        for k, v in sorted(agg[vid].items()):
            if k.endswith("_std"):
                continue
            std_k = k.replace("_mean", "_std") if k.endswith("_mean") else None
            if k.endswith("_mean") and std_k in agg[vid]:
                lines.append(f"  {k[:-5]:28s} {v: .4f}  ± {agg[vid][std_k]:.4f}")
            else:
                lines.append(f"  {k:28s} {v}")
        lines.append("```")
        lines.append("")

    lines.append("## Paper writing prompts")
    lines.append("")
    lines.append(
        "1. Does step A reduce `corr(ra,ea)` relative to original?\n"
        "2. Does late vote discrimination rise under A (especially reproducers)?\n"
        "3. Does the ra/ea ratio drift more under A than original?\n"
        "4. Any extinction / viability cost of the change?\n"
        "5. Manual UI observation (see NOTES.md): do clusters / fronts look different?\n"
    )

    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n")

    notes = [
        "# Manual observation notes",
        "",
        f"Suite run directory: `{out_dir.resolve()}`",
        "",
        "Use the interactive UI (`python server.py`) with the version flags:",
        "",
        "- **original**: `typed_votes=0`",
        "- **A**: `typed_votes=1`",
        "",
        "Match seeds from this suite for fair visual comparison.",
        "",
        "## Checklist",
        "",
        "- [ ] Original: describe visual dynamics (clusters, fronts, flicker)",
        "- [ ] A: same seed — what changed?",
        "- [ ] Does one type dominate spatially?",
        "- [ ] Do votes *look* more targeted in the live counters / behaviour?",
        "- [ ] Anything the metrics miss?",
        "",
        "## Free notes",
        "",
        "_Write here._",
        "",
    ]
    (out_dir / "NOTES.md").write_text("\n".join(notes))

    # Index of all versions for the paper
    index_lines = [
        "# Research results index",
        "",
        f"Latest report: [{report_path.name}]({report_path.name})",
        "",
        "## Registered versions",
        "",
    ]
    for vid, v in VERSIONS.items():
        status = "implemented" if v.implemented else "planned"
        index_lines.append(f"- `{vid}` ({status}): {v.title}")
    (out_dir / "INDEX.md").write_text("\n".join(index_lines) + "\n")

    return report_path
