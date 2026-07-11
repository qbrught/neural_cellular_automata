"""Append-only discovery catalog (jsonl + human-readable markdown)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class CatalogEntry:
    id: str
    one_liner: str
    config_path: str
    summary_path: str
    cycle: int
    created_at: str
    judge: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "one_liner": self.one_liner,
            "config_path": self.config_path,
            "summary_path": self.summary_path,
            "cycle": self.cycle,
            "created_at": self.created_at,
            "judge": self.judge,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CatalogEntry":
        return cls(
            id=data["id"],
            one_liner=data["one_liner"],
            config_path=data["config_path"],
            summary_path=data["summary_path"],
            cycle=int(data["cycle"]),
            created_at=data["created_at"],
            judge=data.get("judge") or {},
        )


_DISC_ID_RE = re.compile(r"^disc_(\d+)$")


class Catalog:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.root / "catalog.jsonl"
        self.md_path = self.root / "catalog.md"
        self.entries: list[CatalogEntry] = []
        self._load()

    def _load(self) -> None:
        self.entries = []
        if not self.jsonl_path.exists():
            return
        with self.jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.entries.append(CatalogEntry.from_dict(json.loads(line)))
        self._reconcile_orphans()

    def _disc_number(self, disc_id: str) -> int | None:
        m = _DISC_ID_RE.match(disc_id)
        return int(m.group(1)) if m else None

    def _max_disc_number(self) -> int:
        max_n = 0
        for e in self.entries:
            n = self._disc_number(e.id)
            if n is not None:
                max_n = max(max_n, n)
        for path in self.root.iterdir():
            if not path.is_dir():
                continue
            n = self._disc_number(path.name)
            if n is not None:
                max_n = max(max_n, n)
        return max_n

    def _reconcile_orphans(self) -> None:
        """Register disc_* folders that exist on disk but are missing from jsonl."""
        known = {e.id for e in self.entries}
        for path in sorted(self.root.iterdir()):
            if not path.is_dir():
                continue
            disc_id = path.name
            n = self._disc_number(disc_id)
            if n is None or disc_id in known:
                continue
            if not (path / "config.json").exists():
                continue

            one_liner = disc_id
            cycle = 0
            judge: dict[str, Any] = {}
            meta_path = path / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                one_liner = meta.get("one_liner") or one_liner
                cycle = int(meta.get("cycle", 0))
                judge = meta.get("judge") or {}

            note_path = path / "note.txt"
            if one_liner == disc_id and note_path.exists():
                one_liner = note_path.read_text(encoding="utf-8").splitlines()[0].strip()

            self._persist_entry(
                CatalogEntry(
                    id=disc_id,
                    one_liner=one_liner,
                    config_path=f"{disc_id}/config.json",
                    summary_path=f"{disc_id}/summary.png",
                    cycle=cycle,
                    created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    judge=judge,
                )
            )

    @property
    def count(self) -> int:
        return len(self.entries)

    def one_liners(self) -> list[str]:
        return [f"{e.id}: {e.one_liner}" for e in self.entries]

    def next_id(self) -> str:
        return f"disc_{self._max_disc_number() + 1:04d}"

    def get_entry(self, disc_id: str) -> CatalogEntry | None:
        for e in self.entries:
            if e.id == disc_id:
                return e
        return None

    def random_base_config_path(self, rng) -> Path | None:
        if not self.entries:
            return None
        e = rng.choice(self.entries)
        path = self.root / e.id / "config.json"
        if path.exists():
            return path
        # Fall back to stored relative path
        alt = Path(e.config_path)
        return alt if alt.exists() else None

    def _persist_entry(self, entry: CatalogEntry) -> CatalogEntry:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.entries.append(entry)
        self.write_md()
        return entry

    def append(
        self,
        disc_id: str,
        one_liner: str,
        *,
        cycle: int,
        judge: dict[str, Any],
    ) -> CatalogEntry:
        entry = CatalogEntry(
            id=disc_id,
            one_liner=one_liner,
            config_path=f"{disc_id}/config.json",
            summary_path=f"{disc_id}/summary.png",
            cycle=cycle,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            judge=judge,
        )
        return self._persist_entry(entry)

    def write_md(self) -> None:
        lines = [
            "# NCSA Discovery Catalog",
            "",
            f"Total discoveries: **{self.count}**",
            "",
            "| ID | One-liner | Config |",
            "|----|-----------|--------|",
        ]
        for e in self.entries:
            # Escape pipes in one-liners for markdown tables.
            ol = e.one_liner.replace("|", "\\|")
            lines.append(
                f"| {e.id} | {ol} | [{e.config_path}]({e.config_path}) |"
            )
        lines.append("")
        tmp = self.md_path.with_suffix(".md.tmp")
        tmp.write_text("\n".join(lines), encoding="utf-8")
        tmp.replace(self.md_path)
