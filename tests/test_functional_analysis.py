"""Discovery + map-split helpers for functional analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.comparisons import get_comparison
from research.functional_analysis import map_split, mix_two_process, rows_for_comparison
from research.response_analysis import discover_run_dirs


def test_discover_pipeline_cache(tmp_path: Path | None = None):
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as td:
        root = Path(td)
        seed = root / "cache" / "sym" / "A" / "seed_42"
        seed.mkdir(parents=True)
        (seed / "params_final.pt").write_bytes(b"x")
        (seed / "state_final.pt").write_bytes(b"x")
        # versions tree also
        vseed = root / "versions" / "original" / "seed_7"
        vseed.mkdir(parents=True)
        (vseed / "params_final.pt").write_bytes(b"x")
        (vseed / "state_final.pt").write_bytes(b"x")
        found = discover_run_dirs(root)
        names = sorted(p.name for p in found)
        assert names == ["seed_42", "seed_7"]
    print("test_discover_pipeline_cache OK")


def test_map_split_and_mix_two_process():
    rec = {
        "learned": {
            "delta_common_all_late": 0.9,
            "ari_common_all_late": 0.4,
        },
        "scalars": {
            "phi_class_late": 0.3,
            "late_min_type_frac": 0.2,
        },
    }
    assert map_split(rec)
    assert mix_two_process(rec)
    rec["scalars"]["late_min_type_frac"] = 0.01
    assert not map_split(rec)
    rec["scalars"]["late_min_type_frac"] = 0.2
    rec["learned"]["ari_common_all_late"] = 0.05
    assert not map_split(rec)
    print("test_map_split_and_mix_two_process OK")


def test_rows_for_comparison_scopes_config():
    records = [
        {"version_id": "A", "config_id": "sym", "seed": 1},
        {"version_id": "A", "config_id": "asym", "seed": 1},
        {"version_id": "E", "config_id": "asym", "seed": 1},
        {"version_id": "original", "config_id": "sym", "seed": 1},
    ]
    e_rows = rows_for_comparison(get_comparison("E"), records)
    assert {(r["version_id"], r["config_id"]) for r in e_rows} == {
        ("A", "asym"),
        ("E", "asym"),
    }
    a_rows = rows_for_comparison(get_comparison("A"), records)
    assert {(r["version_id"], r["config_id"]) for r in a_rows} == {
        ("original", "sym"),
        ("A", "sym"),
    }
    print("test_rows_for_comparison_scopes_config OK")


if __name__ == "__main__":
    test_discover_pipeline_cache()
    test_map_split_and_mix_two_process()
    test_rows_for_comparison_scopes_config()
    print("all functional_analysis tests passed")
