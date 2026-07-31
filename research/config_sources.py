"""Resolve base configs for multi-config research suite runs.

Supports:
  - explicit Config JSON paths
  - discovery directories / ids under ``discoveries/``
  - catalog one-liners for human-readable titles
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from config import Config

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DISCOVERIES_ROOT = _ROOT / "discoveries"
DEFAULT_CATALOG = DEFAULT_DISCOVERIES_ROOT / "catalog.jsonl"
DEFAULT_BENCHMARK = Path(__file__).parent / "configs" / "benchmark.json"


@dataclass(frozen=True)
class ConfigSource:
    """One base config arm of a suite run (hyperparams fixed; versions vary)."""

    id: str
    path: Path
    title: str
    description: str
    cfg: Config

    @property
    def short_label(self) -> str:
        """Compact label for chart titles / logs."""
        return self.id

    @property
    def full_title(self) -> str:
        if self.description:
            return f"{self.id} — {self.description}"
        return self.title or self.id


def load_discovery_catalog(
    catalog_path: Path | None = None,
) -> dict[str, dict]:
    """Load discoveries/catalog.jsonl keyed by discovery id."""
    path = Path(catalog_path) if catalog_path else DEFAULT_CATALOG
    if not path.is_file():
        return {}
    out: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec.get("id")
            if rid:
                out[str(rid)] = rec
    return out


def _slug_from_path(path: Path) -> str:
    path = path.resolve()
    # discoveries/disc_0001/config.json → disc_0001
    if path.name == "config.json" and path.parent.name.startswith("disc_"):
        return path.parent.name
    if path.is_dir() and path.name.startswith("disc_"):
        return path.name
    # research/configs/benchmark.json → benchmark
    return path.stem if path.suffix == ".json" else path.name


def _resolve_config_json(raw: Path) -> Path:
    """Accept a JSON file or a directory containing config.json."""
    p = Path(raw)
    if not p.is_absolute():
        # Try cwd-relative first, then project root.
        candidates = [p, _ROOT / p]
    else:
        candidates = [p]
    for c in candidates:
        if c.is_file() and c.suffix == ".json":
            return c.resolve()
        if c.is_dir():
            cfg = c / "config.json"
            if cfg.is_file():
                return cfg.resolve()
    raise FileNotFoundError(
        f"Could not resolve config JSON from {raw!r} "
        f"(expected .json file or directory with config.json)"
    )


def config_source_from_path(
    path: Path | str,
    *,
    catalog: dict[str, dict] | None = None,
    id_override: str | None = None,
) -> ConfigSource:
    """Build a ConfigSource from a config.json path or discovery directory."""
    json_path = _resolve_config_json(Path(path))
    cid = id_override or _slug_from_path(json_path)
    cfg = Config.load(json_path)
    cat = catalog if catalog is not None else load_discovery_catalog()
    rec = cat.get(cid, {})
    one_liner = (
        rec.get("one_liner")
        or (rec.get("judge") or {}).get("one_liner")
        or ""
    )
    title = one_liner or cid
    return ConfigSource(
        id=cid,
        path=json_path,
        title=title,
        description=one_liner,
        cfg=cfg,
    )


def resolve_discovery_ids(
    ids: str,
    *,
    discoveries_root: Path | None = None,
    catalog: dict[str, dict] | None = None,
) -> list[ConfigSource]:
    """Resolve ``disc_0001,disc_0003`` or ``all`` under discoveries/.

    Also accepts integer-ish ids like ``1,3,5`` → disc_0001, disc_0003, disc_0005.
    """
    root = Path(discoveries_root) if discoveries_root else DEFAULT_DISCOVERIES_ROOT
    cat = catalog if catalog is not None else load_discovery_catalog()
    raw = ids.strip()
    if raw.lower() == "all":
        # Prefer catalog order; fall back to disc_* dirs.
        if cat:
            id_list = list(cat.keys())
        else:
            id_list = sorted(
                p.name for p in root.iterdir()
                if p.is_dir() and p.name.startswith("disc_")
            )
    else:
        id_list = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            # "1" / "0001" / "disc_0001" all accepted
            if part.isdigit():
                part = f"disc_{int(part):04d}"
            elif part.startswith("disc_"):
                pass
            else:
                # bare slug: leave as-is (e.g. custom folder names)
                pass
            id_list.append(part)

    sources: list[ConfigSource] = []
    for did in id_list:
        dpath = root / did
        if not (dpath / "config.json").is_file() and not dpath.is_file():
            # also allow discoveries/disc_0001/config.json already in id
            alt = root / did / "config.json"
            if not alt.is_file():
                raise FileNotFoundError(
                    f"Discovery {did!r} not found under {root} "
                    f"(expected {dpath / 'config.json'})"
                )
        sources.append(
            config_source_from_path(dpath if dpath.is_dir() else root / did, catalog=cat)
        )
    if not sources:
        raise ValueError(f"No discovery configs resolved from {ids!r}")
    return sources


def resolve_config_list(
    specs: str,
    *,
    catalog: dict[str, dict] | None = None,
) -> list[ConfigSource]:
    """Resolve comma-separated paths (files or dirs) into ConfigSources."""
    cat = catalog if catalog is not None else load_discovery_catalog()
    parts = [p.strip() for p in specs.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty --configs list")
    return [config_source_from_path(p, catalog=cat) for p in parts]


def resolve_suite_configs(
    *,
    config: Path | None = None,
    configs: str | None = None,
    discoveries: str | None = None,
    discoveries_root: Path | None = None,
) -> list[ConfigSource]:
    """Pick configs from CLI flags. Exactly one source style should win.

    Priority if multiple set: discoveries > configs > config > default benchmark.
    """
    cat = load_discovery_catalog()
    if discoveries:
        return resolve_discovery_ids(
            discoveries,
            discoveries_root=discoveries_root,
            catalog=cat,
        )
    if configs:
        return resolve_config_list(configs, catalog=cat)
    if config is not None:
        return [config_source_from_path(config, catalog=cat)]
    return [config_source_from_path(DEFAULT_BENCHMARK, catalog=cat, id_override="benchmark")]
