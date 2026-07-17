"""Research experiment suite for paper-ready version comparisons.

Compare system versions (original, A, B, C, D) under fixed seeds with
shared metrics, charts, and markdown reports.

Quick start:
    python -m research.suite run --versions original,A --n-steps 400
    python -m research.suite list
    open research_results/<run_id>/REPORT.md
"""

from research.versions import VERSIONS, VersionSpec, get_version

__all__ = ["VERSIONS", "VersionSpec", "get_version"]
