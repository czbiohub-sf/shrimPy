"""DynaTrack worker process for offloading preprocessing and PCC.

Runs the heavy computation (deskew, phase reconstruction, VS, PCC) in a
separate process with its own GIL and GPU context, so it doesn't interfere
with the acquisition thread.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time as _time

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from shrimpy.mantis.dynatrack import DynaTrackConfig, PositionCoordinates

logger = logging.getLogger(__name__)


class DynaTrackWorker:
    """Manages a subprocess that runs DynaTrack computation.

    The worker process initializes its own preprocessor (TF, VS model) and
    maintains its own reference stacks. The main process sends frame data
    via a queue and receives updated coordinates back.
    """

    def __init__(
        self,
        config: DynaTrackConfig,
        zyx_shape: tuple[int, int, int],
        debug_zarr_path: Path | None = None,
        debug_position_names: dict[int, str] | None = None,
    ) -> None:
        self._config = config
        self._zyx_shape = zyx_shape
        self._debug_zarr_path = debug_zarr_path
        self._debug_position_names = debug_position_names or {}
        self._process: mp.Process | None = None
        self._input_queue: mp.Queue | None = None
        self._output_queue: mp.Queue | None = None

    def start(self) -> None:
        """Spawn the worker process and wait for it to be ready."""
        ctx = mp.get_context("spawn")
        self._input_queue = ctx.Queue()
        self._output_queue = ctx.Queue()

        self._process = ctx.Process(
            target=_worker_loop,
            args=(
                self._config,
                self._zyx_shape,
                self._input_queue,
                self._output_queue,
                self._debug_zarr_path,
                self._debug_position_names,
            ),
            daemon=True,
        )
        self._process.start()

        # Wait for the worker to signal it's ready
        msg = self._output_queue.get(timeout=120)
        if msg["type"] != "ready":
            raise RuntimeError(f"DynaTrack worker failed to start: {msg}")
        logger.info("DynaTrack worker process started (pid=%d)", self._process.pid)

    def submit(
        self,
        timepoint_index: int,
        position_index: int,
        position: PositionCoordinates,
        data: list[np.ndarray],
    ) -> None:
        """Send a job to the worker process (non-blocking)."""
        self._input_queue.put(
            {
                "type": "update",
                "timepoint_index": timepoint_index,
                "position_index": position_index,
                "position": (position.x, position.y, position.z),
                "data": data,
            }
        )

    def get_result(self, timeout: float = 120) -> dict | None:
        """Get the next result from the worker (blocking).

        Returns
        -------
        dict with keys: position_index, x, y, z, elapsed
        or None if no result is available within timeout.
        """
        try:
            msg = self._output_queue.get(timeout=timeout)
            if msg["type"] == "result":
                return msg
            elif msg["type"] == "error":
                logger.error("DynaTrack worker error: %s", msg["error"])
                return None
        except Exception:
            return None

    def try_get_result(self) -> dict | None:
        """Non-blocking check for a result."""
        try:
            if not self._output_queue.empty():
                return self.get_result(timeout=0.1)
        except Exception:
            pass
        return None

    def shutdown(self, timeout: float = 120) -> None:
        """Signal the worker to stop and wait for it to finish."""
        if self._input_queue is not None:
            self._input_queue.put({"type": "shutdown"})
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                logger.warning("DynaTrack worker did not exit, terminating")
                self._process.terminate()
        self._process = None


def _worker_loop(
    config: DynaTrackConfig,
    zyx_shape: tuple[int, int, int],
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    debug_zarr_path: Path | None = None,
    debug_position_names: dict[int, str] | None = None,
) -> None:
    """Main loop for the DynaTrack worker process."""
    # Configure logging in the subprocess
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    log = logging.getLogger("shrimpy.mantis.dynatrack_worker")

    try:
        from shrimpy.mantis.dynatrack import DynaTrackUpdater
        from shrimpy.mantis.dynatrack_preprocessing import build_preprocessor

        log.info("DynaTrack worker: initializing preprocessor for shape %s...", zyx_shape)

        preprocessor = None
        if config.preprocessing:
            preprocessor = build_preprocessor(config, zyx_shape)

        updater = DynaTrackUpdater(config=config, preprocessor=preprocessor)
        if debug_zarr_path:
            updater._debug_zarr_path = debug_zarr_path
            updater._debug_position_names = debug_position_names or {}

        output_queue.put({"type": "ready"})
        log.info("DynaTrack worker: ready")

    except Exception as e:
        output_queue.put({"type": "error", "error": str(e)})
        return

    while True:
        try:
            msg = input_queue.get()
        except Exception:
            break

        if msg["type"] == "shutdown":
            log.info("DynaTrack worker: shutting down")
            break

        if msg["type"] == "update":
            t0 = _time.monotonic()
            t_idx = msg["timepoint_index"]
            p_idx = msg["position_index"]
            px, py, pz = msg["position"]

            from shrimpy.mantis.position_update import PositionCoordinates

            position = PositionCoordinates(x=px, y=py, z=pz)
            data = msg["data"]

            n_frames = len(data) if data else 0
            log.info(
                "DynaTrack worker: processing p=%d t=%d (%d frames)", p_idx, t_idx, n_frames
            )

            try:
                updated = updater.update(t_idx, p_idx, position, data)
                elapsed = _time.monotonic() - t0
                output_queue.put(
                    {
                        "type": "result",
                        "position_index": p_idx,
                        "timepoint_index": t_idx,
                        "x": updated.x,
                        "y": updated.y,
                        "z": updated.z,
                        "elapsed": elapsed,
                    }
                )
                log.info(
                    "DynaTrack worker: p=%d t=%d -> (%.2f, %.2f, %s) in %.1fs",
                    p_idx,
                    t_idx,
                    updated.x,
                    updated.y,
                    updated.z,
                    elapsed,
                )
            except Exception as e:
                log.exception("DynaTrack worker: update failed for p=%d t=%d", p_idx, t_idx)
                output_queue.put(
                    {
                        "type": "error",
                        "error": str(e),
                        "position_index": p_idx,
                        "timepoint_index": t_idx,
                    }
                )
