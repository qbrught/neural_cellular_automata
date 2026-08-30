"""Discovery + map-split helpers for functional analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.functional_analysis import map_split, mix_two_process
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


if __name__ == "__main__":
    test_discover_pipeline_cache()
    test_map_split_and_mix_two_process()
    print("all functional_analysis tests passed")
