"""Experiment G UI field mapping — imports ui_config, not server."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    names = {n for (n, *_rest) in UI_FIELDS}
    for name in G_BOOLS + G_NUMERICS + ("env_preset",):
        assert name in names, f"{name} missing from UI_FIELDS"
    kinds = {n: k for (n, _l, k, *_r) in UI_FIELDS}
    assert kinds["env_preset"] == "choice"
    for name in G_BOOLS:
        assert name in UI_FIELD_GROUP
    print("test_g_fields_in_ui_fields OK")


def test_cfg_to_payload_includes_choices():
    payload = cfg_to_payload(Config())
    assert payload["choices"]["env_preset"] == list(PRESETS)
    assert UI_CHOICES["env_preset"] == list(PRESETS)
    names = [f["name"] for f in payload["fields"]]
    assert "env_preset" in names
    preset_field = next(f for f in payload["fields"] if f["name"] == "env_preset")
    assert preset_field["kind"] == "choice"
    assert preset_field["group"] == "Experiment G"
    print("test_cfg_to_payload_includes_choices OK")


def test_payload_to_cfg_bools_and_choice():
    cfg = payload_to_cfg({
        "environment_heterogeneous": "1",
        "env_preset": "blobs",
        "env_n_blobs": "3",
        "env_affect_R": "0",
        "env_occupancy_blocks": "1",
        "N": "12",
    })
    assert cfg.environment_heterogeneous is True
    assert cfg.env_preset == "blobs"
    assert cfg.env_n_blobs == 3
    assert cfg.env_affect_R is False
    assert cfg.env_occupancy_blocks is True
    assert cfg.N == 12
    # env_regions is not in UI_FIELDS → stays default None
    assert cfg.env_regions is None
    print("test_payload_to_cfg_bools_and_choice OK")


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
    test_payload_missing_preset_falls_back_to_identity()
    test_snapshot_sends_env_maps_on_reset_only()
    print("\nAll environment UI tests passed.")
