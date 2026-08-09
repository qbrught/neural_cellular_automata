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
    config_id: str | None = None,
    config_title: str | None = None,
    config_description: str | None = None,
    config_path: str | None = None,
    suite_name: str | None = None,
) -> Path:
    """Write REPORT.md + NOTES.md + summary.csv under out_dir.

    When config_id is set, the report title names the base config so multi-config
    suites stay readable.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_summary_csv(results, out_dir / "summary.csv")

    version_ids = ", ".join(v.id for v in versions)
    if config_id:
        heading = f"Research suite report: {version_ids} on `{config_id}`"
    elif suite_name:
        heading = f"Research suite report: {suite_name}"
    else:
        heading = "Research suite report"

    lines: list[str] = []
    lines.append(f"# {heading}")
    lines.append("")
    lines.append(f"- Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
    if suite_name:
        lines.append(f"- Suite run: **{suite_name}**")
    if config_id:
        lines.append(f"- **Base config id:** `{config_id}`")
    if config_title:
        lines.append(f"- **Config title:** {config_title}")
    if config_description and config_description != config_title:
        lines.append(f"- **Config description:** {config_description}")
    if config_path:
        lines.append(f"- **Config path:** `{config_path}`")
    elif base_config_note:
        lines.append(f"- Base config: {base_config_note}")
    lines.append(f"- Steps per run: **{n_steps}**")
    lines.append(f"- Seeds: `{seeds}`")
    lines.append(
        f"- Versions: {', '.join(f'`{v.id}`' for v in versions)}"
    )
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

    if config_id:
        lines.append("## Base config under test")
        lines.append("")
        lines.append(f"| Field | Value |")
        lines.append(f"| --- | --- |")
        lines.append(f"| id | `{config_id}` |")
        if config_title:
            lines.append(f"| title | {config_title} |")
        if config_path:
            lines.append(f"| path | `{config_path}` |")
        if config_description:
            lines.append(f"| description | {config_description} |")
        lines.append("")
        lines.append(
            "Version flags are applied **on top of** this base config; "
            "survival weights, η, init density, etc. stay fixed for fair comparison."
        )
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
            f"`goal_in_f={v.goal_in_f}`, "
            f"`coexistence_pressure={getattr(v, 'coexistence_pressure', False)}`, "
            f"`coexistence_lambda={getattr(v, 'coexistence_lambda', 0.01)}`, "
            f"`symmetrize_RE_weights={getattr(v, 'symmetrize_RE_weights', False)}`"
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
    # Also show deltas vs first version if original missing (e.g. A,B,C only)
    if not any(r["version_id"] == "original" for r in results) and versions:
        base_v = versions[0].id
        if any(r["version_id"] == base_v for r in results) and len(versions) > 1:
            lines.append(f"### Deltas vs `{base_v}` (suite baseline)")
            lines.append("")
            lines.append(delta_table(results, baseline=base_v))
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
        "| g_frac drift | Final − initial goal=REPRO fraction (all cells); ~0 without C. |\n"
        "| mean\\|Δg_alive\\| | Mean step-to-step \\|Δ\\| of alive type fraction. |\n"
        "| corr(Lr,Le) | Loss coupling. ~−1 under mirror losses; higher if roles differ. |\n"
        "| R/E vote disc late | Vote specialization (help_kin−harm_foe for R; "
        "harm_foe−help_kin for E) in the last 10% of steps. |\n"
        "| death same/cross late | Sender death rate on same-type vs cross-type "
        "directed Moore edges (both endpoints alive at t). |\n"
        "| death gap late | cross − same death rate; >0 means cross-type contact "
        "is more lethal. |\n"
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
    if config_id:
        lines.append(
            f"Config under discussion: **`{config_id}`**"
            + (f" — {config_description}" if config_description else "")
            + "\n"
        )
    lines.append(
        "1. Does step A reduce `corr(ra,ea)` relative to original / B?\n"
        "2. Does late vote discrimination rise under A/B (especially reproducers)?\n"
        "3. Does the ra/ea ratio drift more under later steps?\n"
        "4. Under C: is `|goal_frac_final − goal_frac_init|` large? Does one type colonize?\n"
        "5. Any extinction / viability cost of the change on *this* config?\n"
        "6. Manual UI observation (see NOTES.md): do clusters / fronts look different?\n"
    )

    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n")

    notes = [
        "# Manual observation notes",
        "",
        f"Suite run directory: `{out_dir.resolve()}`",
    ]
    if suite_name:
        notes.append(f"Suite name: `{suite_name}`")
    if config_id:
        notes.append(f"Base config: `{config_id}`")
        if config_description:
            notes.append(f"Description: {config_description}")
        if config_path:
            notes.append(f"Path: `{config_path}`")
    notes += [
        "",
        "Use the interactive UI (`python server.py`) with the version flags:",
        "",
        "- **original**: `typed_votes=0`",
        "- **A**: `typed_votes=1`",
        "- **B**: `typed_votes=1`, `predator_prey_loss=1`",
        "- **C**: A+B + `goal_inheritance=1`",
        "",
        "Match seeds from this suite for fair visual comparison.",
        "Load the same base weights as this config when comparing in the UI.",
        "",
        "## Checklist",
        "",
        f"- [ ] Config `{config_id or 'base'}`: describe visual dynamics",
        "- [ ] A vs B: what changed?",
        "- [ ] C: does one type colonize? invasion fronts?",
        "- [ ] Does one type dominate spatially?",
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
        "# Config result index",
        "",
        f"Report: [{report_path.name}]({report_path.name})",
        "",
    ]
    if config_id:
        index_lines.append(f"**Config:** `{config_id}`")
        if config_description:
            index_lines.append(f"**Description:** {config_description}")
        index_lines.append("")
    index_lines.append("## Registered versions")
    index_lines.append("")
    for vid, v in VERSIONS.items():
        status = "implemented" if v.implemented else "planned"
        index_lines.append(f"- `{vid}` ({status}): {v.title}")
    (out_dir / "INDEX.md").write_text("\n".join(index_lines) + "\n")

    return report_path


def write_multi_config_index(
    out_root: Path,
    *,
    suite_name: str,
    versions: list[VersionSpec],
    seeds: list[int],
    n_steps: int,
    config_summaries: list[dict[str, Any]],
    all_results: list[dict[str, Any]],
) -> Path:
    """Top-level INDEX.md + combined summary for multi-config suite runs."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    write_summary_csv(all_results, out_root / "summary_all.csv")

    version_ids = ", ".join(f"`{v.id}`" for v in versions)
    lines: list[str] = [
        f"# Multi-config research suite: `{suite_name}`",
        "",
        f"- Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Versions: {version_ids}",
        f"- Seeds: `{seeds}`",
        f"- Steps per run: **{n_steps}**",
        f"- Configs: **{len(config_summaries)}**",
        "",
        "Each config has its own subfolder with a full `REPORT.md`, charts, and per-seed artifacts.",
        "",
        "## Configs in this suite",
        "",
        "| Config | Title / description | Report |",
        "| --- | --- | --- |",
    ]
    for cs in config_summaries:
        cid = cs["config_id"]
        title = cs.get("config_description") or cs.get("config_title") or cid
        # Truncate long one-liners for the table
        if len(title) > 90:
            title = title[:87] + "..."
        rel = cs.get("rel_report", f"configs/{cid}/REPORT.md")
        lines.append(f"| `{cid}` | {title} | [{rel}]({rel}) |")

    lines.append("")
    lines.append("## Cross-config summary table")
    lines.append("")
    lines.append(
        "One row per (config, version), mean over seeds. "
        "Use this to see whether C’s goal drift / colonization is config-dependent."
    )
    lines.append("")
    lines.append(markdown_comparison_table(all_results, include_config=True))
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("| Path | Contents |")
    lines.append("| --- | --- |")
    lines.append("| `summary_all.csv` | All (config, version, seed) rows |")
    lines.append("| `manifest.json` | Machine-readable suite manifest |")
    lines.append("| `configs/<id>/` | Per-config REPORT, charts, versions |")
    lines.append("")
    lines.append("## How to re-run")
    lines.append("")
    lines.append("```bash")
    ids = ",".join(cs["config_id"] for cs in config_summaries)
    vstr = ",".join(v.id for v in versions)
    sstr = ",".join(str(s) for s in seeds)
    if all(c["config_id"].startswith("disc_") for c in config_summaries):
        lines.append(
            f"python -m research.suite run --versions {vstr} "
            f"--discoveries {ids} --n-steps {n_steps} --seeds {sstr} "
            f"--name {suite_name}"
        )
    else:
        paths = ",".join(cs.get("config_path", cs["config_id"]) for cs in config_summaries)
        lines.append(
            f"python -m research.suite run --versions {vstr} "
            f"--configs {paths} --n-steps {n_steps} --seeds {sstr} "
            f"--name {suite_name}"
        )
    lines.append("```")
    lines.append("")

    index_path = out_root / "INDEX.md"
    index_path.write_text("\n".join(lines) + "\n")

    # Also write a top-level REPORT.md pointing at the index (convenience)
    top_report = [
        f"# {suite_name}",
        "",
        f"Multi-config suite over **{len(config_summaries)}** base configs, "
        f"versions {version_ids}.",
        "",
        f"Start here: **[INDEX.md](INDEX.md)** (per-config links + cross-config table).",
        "",
        "Combined CSV: [`summary_all.csv`](summary_all.csv)",
        "",
        "## Per-config reports",
        "",
    ]
    for cs in config_summaries:
        cid = cs["config_id"]
        desc = cs.get("config_description") or ""
        rel = cs.get("rel_report", f"configs/{cid}/REPORT.md")
        top_report.append(f"- [`{cid}`]({rel})" + (f" — {desc}" if desc else ""))
    top_report.append("")
    (out_root / "REPORT.md").write_text("\n".join(top_report) + "\n")

    return index_path
