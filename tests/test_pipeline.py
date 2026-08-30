"""Thesis pipeline: registry, stats, diagnostics, smoke run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from research.comparisons import COMPARISONS, get_comparison, parse_letter_list
from research.metrics import summarize_run
from research.pipeline import unique_jobs
from research.protocol import (
    FRAME_FRACS,
    N_STEPS,
    SEEDS_ORIGINAL_SYM,
    SEEDS_THESIS,
    frame_steps,
)
from research.stats import paired_delta_test
from research.versions import VERSIONS


def test_frame_steps_includes_start_and_end():
    steps = frame_steps(4000, FRAME_FRACS)
    assert steps[0] == 0
    assert steps[-1] == 4000
    assert 2000 in steps
    print("test_frame_steps_includes_start_and_end OK", steps)


def test_default_comparisons_resolve():
    defaults = parse_letter_list(None)
    ids = [c.id for c in defaults]
    assert ids == [c.id for c in COMPARISONS if c.default]
    for c in defaults:
        assert c.config_id in ("sym", "asym")
        for arm in c.arms:
            spec = arm.spec()
            assert spec.id in VERSIONS or arm.version in VERSIONS
            spec.apply  # callable
        if c.off:
            assert any(a.id == c.off for a in c.arms)
        if c.on:
            assert any(a.id == c.on for a in c.arms)
    print("test_default_comparisons_resolve OK", ids)


def test_unique_jobs_dedupes_shared_arms():
    comps = [get_comparison("A"), get_comparison("B"), get_comparison("F")]
    jobs = unique_jobs(comps, seeds=[1, 2])
    keys = [(j.config_id, j.arm.id, j.seed) for j in jobs]
    assert len(keys) == len(set(keys))
    arms = {(j.arm.id, j.seed) for j in jobs}
    # original, A, B, F × 2 seeds on sym
    assert ("original", 1) in arms
    assert ("A", 1) in arms
    assert ("B", 1) in arms
    assert ("F", 1) in arms
    assert len(jobs) == 8
    print("test_unique_jobs_dedupes_shared_arms OK", len(jobs))


def test_f_lambda_arms_override_lambda():
    c = get_comparison("F_lambda")
    lams = [a.spec().coexistence_lambda for a in c.arms if a.version == "F"]
    assert lams == [0.01, 0.1, 1.0]
    print("test_f_lambda_arms_override_lambda OK", lams)


def test_paired_delta_test_sign():
    off = np.array([0.1, 0.2, 0.15, 0.12, 0.18, 0.11])
    on = off + 0.2
    st = paired_delta_test(off, on, n_iter=200, seed=0)
    assert st["n"] == 6
    assert abs(st["mean_delta"] - 0.2) < 1e-9
    assert st["n_pos"] == 6
    assert st["ci95_lo"] > 0
    print("test_paired_delta_test_sign OK", st["mean_delta"], st["p_value"])


def test_summarize_new_keys():
    T = 20
    ra = np.linspace(10, 20, T)
    ea = np.linspace(10, 5, T)
    alive = ra + ea
    series = {
        "reproducer_alive": ra,
        "eliminator_alive": ea,
        "alive": alive,
        "loss_r": np.zeros(T),
        "loss_e": np.zeros(T),
        "f_signal_type_gap": np.linspace(0, 1, T),
        "frac_alive_low_kappa": np.full(T, 0.25),
        "death_rate_E_cross_minus_same": np.full(T, 0.1),
    }
    s = summarize_run(series, goal_frac_repro=0.5)
    assert "late_f_signal_type_gap" in s
    assert abs(s["late_frac_alive_low_kappa"] - 0.25) < 1e-9
    assert "late_death_rate_E_cross_minus_same" in s
    print("test_summarize_new_keys OK")


def test_pipeline_smoke(tmp_path: Path | None = None):
    from research.pipeline import main as pipeline_main

    out = Path("research_results") / "thesis_test_smoke"
    if tmp_path is not None:
        out_parent = tmp_path
        name = "thesis_test_smoke"
        pipeline_main(
            [
                "run",
                "--letters",
                "A",
                "--n-steps",
                "4",
                "--seeds",
                "7",
                "--name",
                name,
                "--output-dir",
                str(out_parent),
            ]
        )
        root = out_parent / name
    else:
        pipeline_main(
            [
                "run",
                "--letters",
                "A",
                "--n-steps",
                "4",
                "--seeds",
                "7",
                "--name",
                "thesis_test_smoke",
            ]
        )
        root = out
    assert (root / "INDEX.md").is_file()
    assert (root / "protocol.json").is_file()
    cache_a = root / "cache" / "sym" / "A" / "seed_7"
    assert (cache_a / "series.npz").is_file()
    assert (cache_a / "frames.npz").is_file()
    report = (root / "comparisons" / "A" / "REPORT.md").read_text()
    assert "## 1. Motivation" in report
    assert "## 2. Visual" in report
    assert "## 3. Objective" in report
    assert "## 4. Conclusions" in report
    proto = json.loads((root / "protocol.json").read_text())
    assert proto["n_steps"] == 4
    series = np.load(cache_a / "series.npz")
    assert "f_signal_type_gap" in series.files
    assert "frac_alive_low_kappa" in series.files
    print("test_pipeline_smoke OK", root)


def test_protocol_seeds_original_sym():
    assert SEEDS_THESIS == SEEDS_ORIGINAL_SYM
    assert len(SEEDS_THESIS) == 20
    assert SEEDS_THESIS[0] == 459903122
    assert N_STEPS == 4000
    print("test_protocol_seeds_original_sym OK", len(SEEDS_THESIS))


if __name__ == "__main__":
    test_frame_steps_includes_start_and_end()
    test_default_comparisons_resolve()
    test_unique_jobs_dedupes_shared_arms()
    test_f_lambda_arms_override_lambda()
    test_paired_delta_test_sign()
    test_summarize_new_keys()
    test_protocol_seeds_original_sym()
    test_pipeline_smoke()
    print("all pipeline tests OK")
