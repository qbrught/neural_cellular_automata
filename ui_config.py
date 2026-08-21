"""UI field metadata and Config payload conversion.

Extracted from server.py so tests can import this without FastAPI /
SimulationEngine. The interactive UI still auto-renders from UI_FIELDS.
"""

from __future__ import annotations

from dataclasses import asdict, fields

from config import Config
from environment import PRESET_KNOBS, PRESETS

# (field_name, label, kind, min, max, step)
# kind is "int" | "float" | "intornone" | "choice"
UI_FIELDS = [
    ("N", "Grid size N", "int", 5, 200, 1),
    ("d", "State dim d", "int", 1, 64, 1),
    ("hidden", "MLP hidden", "int", 4, 128, 1),
    ("eta", "Learning rate η", "float", 0.0, 1.0, 0.001),
    ("learn", "Learn (0/1)", "int", 0, 1, 1),
    ("learn_messages", "Learn messages (0/1)", "int", 0, 1, 1),
    ("require_alive_neighbour", "Require alive nbr (0/1)", "int", 0, 1, 1),
    ("typed_votes", "Typed votes A (0/1)", "int", 0, 1, 1),
    ("predator_prey_loss", "Pred-prey loss B (0/1)", "int", 0, 1, 1),
    ("goal_inheritance", "Goal inherit C (0/1)", "int", 0, 1, 1),
    ("goal_in_f", "Goal in f D (0/1)", "int", 0, 1, 1),
    ("coexistence_pressure", "Coexist pressure F (0/1)", "int", 0, 1, 1),
    ("coexistence_lambda", "Coexist λ", "float", 0.0, 2.0, 0.001),
    ("coexistence_delta", "Coexist δ", "float", 1e-8, 0.1, 1e-5),
    ("environment_heterogeneous", "Heterogeneous env G (0/1)", "int", 0, 1, 1),
    ("env_preset", "Env preset G", "choice", 0, 0, 1),
    ("env_seed", "Env seed G", "int", 0, 1_000_000, 1),
    ("env_dead_frac", "Env dead frac G", "float", 0.0, 1.0, 0.01),
    ("env_n_blobs", "Env n blobs G", "int", 0, 32, 1),
    ("env_blob_radius", "Env blob radius G", "float", 0.01, 1.0, 0.01),
    ("env_kappa_lo", "Env κ_lo G", "float", 0.0, 2.0, 0.01),
    ("env_kappa_hi", "Env κ_hi G", "float", 0.0, 2.0, 0.01),
    ("env_eta_lo", "Env η_lo G", "float", 0.0, 4.0, 0.01),
    ("env_eta_hi", "Env η_hi G", "float", 0.0, 4.0, 0.01),
    ("env_affect_R", "Env affect R (0/1)", "int", 0, 1, 1),
    ("env_affect_E", "Env affect E (0/1)", "int", 0, 1, 1),
    ("env_occupancy_blocks", "Env occupancy blocks (0/1)", "int", 0, 1, 1),
    ("w0", "w₀ bias", "float", -5.0, 5.0, 0.01),
    ("w1", "w₁ alive", "float", -5.0, 5.0, 0.01),
    ("w2", "w₂ reproducer", "float", -5.0, 5.0, 0.01),
    ("w3", "w₃ eliminator", "float", -5.0, 5.0, 0.01),
    ("w4_help", "w₄ help (kin)", "float", -5.0, 5.0, 0.01),
    ("w4_harm", "w₄ harm (foe)", "float", -5.0, 5.0, 0.01),
    ("w5", "w₅ f-signal", "float", -5.0, 5.0, 0.01),
    ("seed", "Seed", "int", 0, 1_000_000, 1),
    ("init_alive_prob", "Init alive prob", "float", 0.0, 1.0, 0.01),
    ("init_noise_std", "Init noise std", "float", 0.0, 1.0, 0.001),
    ("u_seed", "u seed", "int", 0, 1_000_000, 1),
    ("n_steps", "n_steps (blank=∞)", "intornone", 0, 10_000_000, 1),
]

UI_CHOICES = {"env_preset": list(PRESETS)}

UI_FIELD_GROUP = {
    "environment_heterogeneous": "Experiment G",
    "env_preset": "Experiment G",
    "env_seed": "Experiment G",
    "env_dead_frac": "Experiment G",
    "env_n_blobs": "Experiment G",
    "env_blob_radius": "Experiment G",
    "env_kappa_lo": "Experiment G",
    "env_kappa_hi": "Experiment G",
    "env_eta_lo": "Experiment G",
    "env_eta_hi": "Experiment G",
    "env_affect_R": "Experiment G",
    "env_affect_E": "Experiment G",
    "env_occupancy_blocks": "Experiment G",
}

_BOOL_FIELDS = {
    "learn",
    "learn_messages",
    "require_alive_neighbour",
    "typed_votes",
    "predator_prey_loss",
    "goal_inheritance",
    "goal_in_f",
    "coexistence_pressure",
    "environment_heterogeneous",
    "env_affect_R",
    "env_affect_E",
    "env_occupancy_blocks",
}


def cfg_to_payload(cfg: Config) -> dict:
    return {
        "fields": [
            {
                "name": n,
                "label": lbl,
                "kind": kind,
                "min": mn,
                "max": mx,
                "step": st,
                "group": UI_FIELD_GROUP.get(n),
            }
            for (n, lbl, kind, mn, mx, st) in UI_FIELDS
        ],
        "values": asdict(cfg),
        "choices": UI_CHOICES,
        "preset_knobs": PRESET_KNOBS,
    }


def payload_to_cfg(values: dict) -> Config:
    """Build a Config from a dict of field-name -> value.

    Tolerates string values from form posts. Empty string for n_steps -> None.
    Sidebar apply does not include env_regions (resets to None like output_dir).
    """
    out: dict = {}
    field_kinds = {n: kind for (n, _l, kind, _mn, _mx, _s) in UI_FIELDS}
    for k, v in values.items():
        if k not in field_kinds:
            continue
        kind = field_kinds[k]
        if k in _BOOL_FIELDS:
            out[k] = bool(int(v))
        elif kind == "intornone":
            if v is None or v == "" or v == "null":
                out[k] = None
            else:
                out[k] = int(v)
        elif kind == "choice":
            out[k] = str(v)
        elif kind == "int":
            out[k] = int(v)
        elif kind == "float":
            out[k] = float(v)
    defaults = Config()
    full = {f.name: getattr(defaults, f.name) for f in fields(Config)}
    full.update(out)
    return Config(**full)
