
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

from config import Config
from dynamics import forward_step
from learning import gradient_step
from simulate import build_grid
from state import GOAL_REPRODUCE, GOAL_ELIMINATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ncsa.server")





# Simulation engine — owns state, runs in background thread, broadcasts frames.

class SimulationEngine:
    """Holds the simulation state and runs it on a background thread.

    Thread safety:
      - `_lock` guards all mutations of state/params/u/cfg/running flags.
      - Tick loop holds the lock briefly to step, releases for sleep.
      - HTTP handlers acquire the lock for any control change.
    """

    def __init__(self, initial_cfg: Config):
        self._lock = threading.RLock()
        self._cfg = initial_cfg
        self._state = None
        self._params = None
        self._u = None
        self._env = None
        self._step_idx = 0
        self._running = False
        self._steps_per_second = 30.0  # UI-controlled tick rate
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Async event loop for broadcasting (set when first ws client connects).
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: set[asyncio.Queue] = set()
        # Initialise simulation now (so a snapshot is available before start).
        self._reset_locked()

    # ----- state management (locked) -----

    def _reset_locked(self) -> None:
        log.info("Reset with cfg: %s", self._cfg)
        grid = build_grid(self._cfg)
        self._state = grid.state
        self._params = grid.params
        self._u = grid.u
        self._env = grid.env
        self._step_idx = 0

    def reset(self) -> None:
        with self._lock:
            was_running = self._running
            self._running = False  # stop ticking during reset
            self._reset_locked()
        if was_running:
            self.start()
        self._broadcast_snapshot()

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._thread = threading.Thread(target=self._tick_loop, daemon=True)
                self._thread.start()

    def pause(self) -> None:
        with self._lock:
            self._running = False

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def set_config(self, new_cfg: Config, do_reset: bool = True) -> None:
        """Replace config. If do_reset, rebuild the grid (most fields require this)."""
        with self._lock:
            was_running = self._running
            self._running = False
            self._cfg = new_cfg
            if do_reset:
                self._reset_locked()
        if was_running:
            self.start()
        self._broadcast_snapshot()

    def set_speed(self, steps_per_second: float) -> None:
        with self._lock:
            self._steps_per_second = max(0.1, min(500.0, float(steps_per_second)))

    def get_config(self) -> Config:
        with self._lock:
            return self._cfg

    # ----- subscription -----

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._main_loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    # ----- the actual tick -----

    def _tick_loop(self) -> None:
        log.info("Tick loop started")
        last_emit = 0.0
        while not self._stop.is_set():
            with self._lock:
                running = self._running
                sps = self._steps_per_second
                n_steps = self._cfg.n_steps
                if running and n_steps is not None and self._step_idx >= n_steps:
                    log.info("Reached n_steps=%d; pausing.", n_steps)
                    self._running = False
                    running = False

            if not running:
                # Idle: poll for resume.
                time.sleep(0.05)
                continue

            # One step.
            with self._lock:
                try:
                    step_out = forward_step(
                        self._state, self._params, self._u, self._cfg, env=self._env,
                    )
                    if self._cfg.learn:
                        stats = gradient_step(
                            self._state, step_out, self._params, self._cfg, env=self._env,
                        )
                    else:
                        stats = {
                            "loss_total": 0.0,
                            "loss_reproduce_mean": float("nan"),
                            "loss_eliminate_mean": float("nan"),
                            "n_alive": int(self._state.x.sum().item()),
                        }
                    self._state = step_out.next_state
                    self._step_idx += 1
                    frame = self._snapshot_locked(stats)
                except Exception:
                    log.exception("Tick failed; pausing")
                    self._running = False
                    continue

            # Throttle emission: at most ~30 fps regardless of step rate.
            now = time.monotonic()
            if now - last_emit >= 1.0 / 30.0:
                self._broadcast_frame(frame)
                last_emit = now

            # Sleep between steps to honour sps.
            time.sleep(max(0.0, 1.0 / sps))
        log.info("Tick loop stopped")

    # ----- snapshot serialisation -----

    def _snapshot_locked(self, stats: dict | None) -> dict:
        """Build a JSON-able dict describing the current frame.

        Grid is encoded as a base64 string of (N*N) bytes:
          0 = dead, 1 = alive reproducer, 2 = alive eliminator.
        That's ~13KB for a 100x100 grid -- well under any WS limit.
        """
        s = self._state
        x = s.x.detach().cpu().numpy().astype(np.uint8)
        g = s.goals.detach().cpu().numpy().astype(np.uint8)
        # 0 dead, 1 repro alive, 2 elim alive
        display = np.zeros_like(x, dtype=np.uint8)
        alive = x.astype(bool)
        display[alive & (g == GOAL_REPRODUCE)] = 1
        display[alive & (g == GOAL_ELIMINATE)] = 2
        grid_b64 = base64.b64encode(display.tobytes()).decode("ascii")

        alive_total = int(x.sum())
        repro_alive = int(((g == GOAL_REPRODUCE) & alive).sum())
        elim_alive = int(((g == GOAL_ELIMINATE) & alive).sum())

        return {
            "type": "frame",
            "step": self._step_idx,
            "n_steps": self._cfg.n_steps,
            "running": self._running,
            "N": self._cfg.N,
            "grid_b64": grid_b64,
            "alive": alive_total,
            "dead": self._cfg.N * self._cfg.N - alive_total,
            "reproducer_alive": repro_alive,
            "eliminator_alive": elim_alive,
            "loss_reproduce_mean": (
                None if stats is None or stats["loss_reproduce_mean"] != stats["loss_reproduce_mean"]
                else stats["loss_reproduce_mean"]
            ),
            "loss_eliminate_mean": (
                None if stats is None or stats["loss_eliminate_mean"] != stats["loss_eliminate_mean"]
                else stats["loss_eliminate_mean"]
            ),
        }

    def _broadcast_frame(self, frame: dict) -> None:
        loop = self._main_loop
        if loop is None:
            return
        # Drop frames if any subscriber is slow (maxsize=2).
        for q in list(self._subscribers):
            try:
                loop.call_soon_threadsafe(q.put_nowait, frame)
            except asyncio.QueueFull:
                pass  # client slow; drop
            except RuntimeError:
                pass  # loop closing

    def _broadcast_snapshot(self) -> None:
        """Send a current-state frame to all subscribers (e.g. after reset)."""
        with self._lock:
            frame = self._snapshot_locked(None)
        self._broadcast_frame(frame)
