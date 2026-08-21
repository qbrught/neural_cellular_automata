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
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from config import Config
from ui_config import cfg_to_payload, payload_to_cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ncsa.server")

from simulation_engine import SimulationEngine

# UI_FIELDS / cfg_to_payload / payload_to_cfg live in ui_config.py so tests
# can import them without constructing FastAPI or SimulationEngine.

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
