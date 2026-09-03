"""Experiment G UI field mapping — imports ui_config, not server."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

from config import Config
from environment import PRESETS
from ui_config import (
    UI_CHOICES,
    UI_FIELD_GROUP,
    UI_FIELDS,
    cfg_to_payload,
    payload_to_cfg,
)


G_BOOLS = (
    "environment_heterogeneous",
    "env_affect_R",
    "env_affect_E",
    "env_occupancy_blocks",
)
G_NUMERICS = (
    "env_seed",
    "env_dead_frac",
    "env_n_blobs",
    "env_blob_radius",
    "env_kappa_lo",
    "env_kappa_hi",
    "env_eta_lo",
    "env_eta_hi",
)


def test_g_fields_in_ui_fields():
    """G is implemented but hidden from the interactive UI (not in the report)."""
    names = {n for (n, *_rest) in UI_FIELDS}
    for name in G_BOOLS + G_NUMERICS + ("env_preset", "env_regions"):
        assert name not in names, f"{name} should be hidden from UI_FIELDS"
    for name in G_BOOLS:
        assert name not in UI_FIELD_GROUP
    print("test_g_fields_in_ui_fields OK")


def test_cfg_to_payload_includes_choices():
    payload = cfg_to_payload(Config())
    assert payload["choices"]["env_preset"] == list(PRESETS)
    assert UI_CHOICES["env_preset"] == list(PRESETS)
    names = [f["name"] for f in payload["fields"]]
    assert "env_preset" not in names
    print("test_cfg_to_payload_includes_choices OK")


def test_payload_to_cfg_bools_and_choice():
    """G keys in a payload are ignored while those fields are hidden from the UI."""
    cfg = payload_to_cfg({
        "environment_heterogeneous": "1",
        "env_preset": "blobs",
        "env_n_blobs": "3",
        "env_affect_R": "0",
        "env_occupancy_blocks": "1",
        "N": "12",
    })
    defaults = Config()
    assert cfg.environment_heterogeneous is defaults.environment_heterogeneous
    assert cfg.env_preset == defaults.env_preset
    assert cfg.env_n_blobs == defaults.env_n_blobs
    assert cfg.env_affect_R is defaults.env_affect_R
    assert cfg.env_occupancy_blocks is defaults.env_occupancy_blocks
    assert cfg.N == 12
    assert cfg.env_regions is None
    print("test_payload_to_cfg_bools_and_choice OK")


def test_payload_to_cfg_native_and_empty_bools():
    """JSON true/false from the UI, and empty string, must not raise."""
    defaults = Config()
    cfg = payload_to_cfg({
        "learn": True,
        "learn_messages": False,
        "typed_votes": True,
        "environment_heterogeneous": False,
        "env_affect_R": True,
        "env_affect_E": False,
    })
    assert cfg.learn is True
    assert cfg.learn_messages is False
    assert cfg.typed_votes is True
    # G fields are not in UI_FIELDS, so payload values do not override defaults.
    assert cfg.environment_heterogeneous is defaults.environment_heterogeneous
    assert cfg.env_affect_R is defaults.env_affect_R
    assert cfg.env_affect_E is defaults.env_affect_E

    empty = payload_to_cfg({
        "learn": "",
        "typed_votes": "",
        "environment_heterogeneous": "",
        "env_occupancy_blocks": None,
    })
    assert empty.learn is defaults.learn
    assert empty.typed_votes is defaults.typed_votes
    assert empty.environment_heterogeneous is defaults.environment_heterogeneous
    assert empty.env_occupancy_blocks is defaults.env_occupancy_blocks

    payload = cfg_to_payload(Config())
    kinds = {f["name"]: f["kind"] for f in payload["fields"]}
    assert kinds["learn"] == "bool"
    assert payload["values"]["learn"] is True
    assert payload["values"]["learn_messages"] is False
    print("test_payload_to_cfg_native_and_empty_bools OK")


def test_payload_to_cfg_env_regions_json():
    """env_regions is ignored while G is hidden from UI_FIELDS (no parse error)."""
    regions = [
        {"shape": "disk", "cy": 5, "cx": 7, "radius": 3, "kappa_R": 0, "kappa_E": 0},
    ]
    cfg = payload_to_cfg({
        "environment_heterogeneous": "1",
        "env_preset": "custom",
        "env_regions": json.dumps(regions),
    })
    defaults = Config()
    assert cfg.env_preset == defaults.env_preset
    assert cfg.env_regions is None
    cfg2 = payload_to_cfg({
        "environment_heterogeneous": 1,
        "env_preset": "custom",
        "env_regions": regions,
    })
    assert cfg2.env_regions is None
    cfg3 = payload_to_cfg({"env_regions": ""})
    assert cfg3.env_regions is None
    payload_to_cfg({"env_regions": "{not json"})
    payload_to_cfg({"env_regions": '{"shape": "disk"}'})
    print("test_payload_to_cfg_env_regions_json OK")


def test_payload_missing_preset_falls_back_to_identity():
    cfg = payload_to_cfg({"environment_heterogeneous": "1"})
    assert cfg.env_preset == "identity"
    print("test_payload_missing_preset_falls_back_to_identity OK")


def test_snapshot_sends_env_maps_on_reset_only():
    from simulation_engine import SimulationEngine
    eng = SimulationEngine(Config(
        N=6, d=4, hidden=8, n_steps=2, learn=False,
        environment_heterogeneous=True, env_preset="vertical_band",
        env_dead_frac=0.25,
    ))
    reset_frame = eng._snapshot_locked(None, include_env=True)
    assert reset_frame["env_active"] is True
    assert "env_kappa_b64" in reset_frame
    assert "env_occ_b64" in reset_frame
    assert "env_eta_hi" in reset_frame
    tick_frame = eng._snapshot_locked(None, include_env=False)
    assert tick_frame["env_active"] is True
    assert "env_kappa_b64" not in tick_frame
    off = SimulationEngine(Config(N=4, d=4, hidden=8, n_steps=1, learn=False))
    off_frame = off._snapshot_locked(None, include_env=True)
    assert off_frame["env_active"] is False
    assert "env_kappa_b64" not in off_frame
    eng.shutdown()
    off.shutdown()
    print("test_snapshot_sends_env_maps_on_reset_only OK")


if __name__ == "__main__":
    test_g_fields_in_ui_fields()
    test_cfg_to_payload_includes_choices()
    test_payload_to_cfg_bools_and_choice()
    test_payload_to_cfg_native_and_empty_bools()
    test_payload_to_cfg_env_regions_json()
    test_payload_missing_preset_falls_back_to_identity()
    test_snapshot_sends_env_maps_on_reset_only()
    print("\nAll environment UI tests passed.")
