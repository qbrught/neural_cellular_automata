"""Research comparisons for paper versions (original, A–G).

Thesis entry: python -m research.pipeline
Ad-hoc version lists: python -m research.suite
"""

from research.versions import VERSIONS, VersionSpec, get_version

__all__ = ["VERSIONS", "VersionSpec", "get_version"]
