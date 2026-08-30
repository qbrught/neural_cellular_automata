"""Phi_late vs min-type map for the experiment summary figure."""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

VERSIONS = ["original", "A", "B", "C_only", "C", "D_fixed", "D", "F"]
COLORS = {
    "original": "#4d4d4d",
    "A": "#d62728",
    "B": "#2ca02c",
    "C_only": "#9467bd",
    "C": "#8c564b",
    "D_fixed": "#17becf",
    "D": "#1f77b4",
    "F": "#ff7f0e",
}
LABELS = {
    "original": "original",
    "A": "typed votes",
    "B": "predator–prey",
    "C_only": "inheritance only",
    "C": "inheritance",
    "D_fixed": "goal in $f$ (fixed)",
    "D": "goal in $f$",
    "F": "viability barrier",
}


def load(path: Path, bank: str) -> list[dict]:
    seen: set[tuple] = set()
    rows = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row["config_id"] != "sym":
                continue
            v = row["version"]
            if v not in VERSIONS:
                continue
            key = (v, row["seed"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "bank": bank,
                    "version": v,
                    "phi_late": float(row["phi_class_late"]),
                    "min_type": float(row["late_min_type_frac"]),
                }
            )
    return rows


def main() -> None:
    trio = load(ROOT / "research_results/thesis_hist3_ladder/summary_all.csv", "trio")
    catalog = load(
        ROOT / "research_results/thesis_orig_sym_20/summary_all.csv", "catalog"
    )
    rows = trio + catalog

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 7.5,
            "axes.grid": True,
            "grid.alpha": 0.28,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    for ax, bank, title in zip(
        axes,
        ("trio", "catalog"),
        ("Unfiltered trio", "Searched catalog"),
    ):
        subset = [r for r in rows if r["bank"] == bank]
        for v in VERSIONS:
            pts = [r for r in subset if r["version"] == v]
            if not pts:
                continue
            ax.scatter(
                [r["min_type"] for r in pts],
                [r["phi_late"] for r in pts],
                c=COLORS[v],
                s=36 if bank == "trio" else 22,
                alpha=0.85 if bank == "trio" else 0.7,
                edgecolors="k",
                linewidths=0.35,
                zorder=3,
                label=LABELS[v],
            )
        ax.axhline(0.2, color="0.45", ls="--", lw=0.8)
        ax.axvline(0.10, color="0.45", ls="--", lw=0.8)
        ax.set_xlabel("late min-type")
        ax.set_title(title)
        ax.set_xlim(-0.03, 0.55)
        ax.set_ylim(-0.03, 0.62)
        ax.text(0.012, 0.57, "attrition", color="#d62728", fontsize=7.5)
        ax.text(0.22, 0.57, "mix moved, both present", color="0.25", fontsize=7.5)
        ax.text(0.28, 0.04, "density-null", color="0.35", fontsize=7.5)
    axes[0].set_ylabel(r"$\Phi_{\mathrm{late}}$")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS[v],
            markeredgecolor="k",
            markersize=6,
            label=LABELS[v],
        )
        for v in VERSIONS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"experiment_phi_map.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'experiment_phi_map.pdf'}")


if __name__ == "__main__":
    main()
