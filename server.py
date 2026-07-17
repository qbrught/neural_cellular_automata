"""Interactive web UI backend for NCSA.

Architecture:
  - One SimulationEngine instance owns the simulation state and runs in a
    background thread.
  - WebSocket /ws streams per-step frames to any connected client.
  - HTTP endpoints control the engine (start, pause, reset, config).

Refactor isolation:
  This file should not import tensors directly. It uses only the public
  functions from simulate / dynamics / learning. If the simulation's
  internal shapes change, only `_snapshot()` needs updating.

Run:
  python -m uvicorn server:app --host 127.0.0.1 --port 8765
  (or use the convenience `python server.py`)
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import threading
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from config import Config
from dynamics import forward_step
from learning import gradient_step
from simulate import build_grid
from state import GOAL_REPRODUCE, GOAL_ELIMINATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ncsa.server")

from simulation_engine import SimulationEngine


# Config helpers — what fields the UI exposes, with metadata for slider ranges.

# (field_name, label, kind, min, max, step)  — kind is "int" | "float" | "intornone"
UI_FIELDS = [
    ("N", "Grid size N", "int", 5, 200, 1),
    ("d", "State dim d", "int", 1, 64, 1),
    ("hidden", "MLP hidden", "int", 4, 128, 1),
    ("eta", "Learning rate η", "float", 0.0, 1.0, 0.001),
    ("learn", "Learn (0/1)", "int", 0, 1, 1),
    ("require_alive_neighbour", "Require alive nbr (0/1)", "int", 0, 1, 1),
    ("typed_votes", "Typed votes A (0/1)", "int", 0, 1, 1),
    ("predator_prey_loss", "Pred-prey loss B (0/1)", "int", 0, 1, 1),
    ("goal_inheritance", "Goal inherit C (0/1)", "int", 0, 1, 1),
    ("goal_in_f", "Goal in f D (0/1)", "int", 0, 1, 1),
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


def cfg_to_payload(cfg: Config) -> dict:
    return {
        "fields": [
            {"name": n, "label": lbl, "kind": kind, "min": mn, "max": mx, "step": st}
            for (n, lbl, kind, mn, mx, st) in UI_FIELDS
        ],
        "values": asdict(cfg),
    }


def payload_to_cfg(values: dict) -> Config:
    """Build a Config from a dict of field-name -> value.

    Tolerates string values from form posts. Empty string for n_steps -> None.
    """
    converters = {
        "int": int,
        "float": float,
    }
    out: dict = {}
    field_kinds = {n: kind for (n, _l, kind, _mn, _mx, _s) in UI_FIELDS}
    for k, v in values.items():
        if k not in field_kinds:
            # Unknown field; ignore (lets us add/remove without breaking the UI).
            continue
        kind = field_kinds[k]
        if k in (
            "learn",
            "require_alive_neighbour",
            "typed_votes",
            "predator_prey_loss",
            "goal_inheritance",
            "goal_in_f",
        ):
            out[k] = bool(int(v))
        elif kind == "intornone":
            if v is None or v == "" or v == "null":
                out[k] = None
            else:
                out[k] = int(v)
        elif kind == "int":
            out[k] = int(v)
        elif kind == "float":
            out[k] = float(v)
    # Fall back to defaults for anything not specified.
    defaults = Config()
    full = {f.name: getattr(defaults, f.name) for f in fields(Config)}
    full.update(out)
    return Config(**full)


# FastAPI app

app = FastAPI(title="NCSA Interactive")
engine = SimulationEngine(Config())

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def _on_startup() -> None:
    engine.attach_loop(asyncio.get_running_loop())


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    engine.shutdown()


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/config")
async def get_config():
    return cfg_to_payload(engine.get_config())


@app.post("/api/config")
async def post_config(payload: dict):
    """Update the config. Body: {"values": {...}, "reset": bool}.

    If "reset" is true (default), rebuilds the grid with the new config.
    Otherwise just updates fields that don't require rebuild (currently
    only n_steps and eta -- the rest require a rebuild to take effect).
    """
    values = payload.get("values", {})
    do_reset = payload.get("reset", True)
    try:
        new_cfg = payload_to_cfg(values)
    except (ValueError, TypeError) as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    engine.set_config(new_cfg, do_reset=do_reset)
    return {"ok": True}


@app.post("/api/start")
async def post_start():
    engine.start()
    return {"ok": True, "running": True}


@app.post("/api/pause")
async def post_pause():
    engine.pause()
    return {"ok": True, "running": False}


@app.post("/api/reset")
async def post_reset(payload: dict | None = None):
    """Optionally update config first, then reset."""
    if payload and "values" in payload:
        try:
            new_cfg = payload_to_cfg(payload["values"])
        except (ValueError, TypeError) as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
        engine.set_config(new_cfg, do_reset=True)
    else:
        engine.reset()
    return {"ok": True}


@app.post("/api/speed")
async def post_speed(payload: dict):
    sps = float(payload.get("steps_per_second", 30.0))
    engine.set_speed(sps)
    return {"ok": True, "steps_per_second": sps}


@app.get("/api/config/download")
async def download_config():
    cfg = engine.get_config()
    body = json.dumps(asdict(cfg), indent=2).encode("utf-8")
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="ncsa_config.json"'},
    )


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    q = engine.subscribe()
    # Send current snapshot immediately.
    engine._broadcast_snapshot()
    try:
        while True:
            frame = await q.get()
            await ws.send_text(json.dumps(frame))
    except WebSocketDisconnect:
        pass
    finally:
        engine.unsubscribe(q)


# CLI entry: `python server.py`

def main() -> None:
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":
    main()
