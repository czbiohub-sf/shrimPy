"""Feeds live acquisition frames to an out-of-process napari viewer.

:class:`ViewerFeeder` connects to a :class:`~pymmcore_plus.CMMCorePlus` MDA event stream
and, for every frame, copies the pixels into a bounded shared-memory ring and pushes a
tiny coordinate message onto a queue. A child process (see :mod:`._napari_process`) reads
those and renders them.

Design contract: **nothing here may ever block or crash the acquisition.** The
``frameReady`` callback runs on the acquisition thread, so every handler is wrapped in a
blanket ``try/except``, queue writes are non-blocking (frames are dropped if the viewer
falls behind), and the viewer lives in a separate process so even a hard crash (segfault,
GUI hang) cannot touch the running acquisition.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue as _queue

from typing import TYPE_CHECKING, Any

import numpy as np

from shrimpy.viewer._napari_process import run_viewer
from shrimpy.viewer.ring_buffer import RingBuffer

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from useq import MDAEvent, MDASequence

logger = logging.getLogger(__name__)

# Max coordinate messages buffered for the viewer before we start dropping frames.
# Small on purpose: a slow/stalled viewer must never accumulate unbounded backlog.
_QUEUE_MAXSIZE = 256


class ViewerFeeder:
    """Bridges a running MDA to a separate-process napari viewer.

    Parameters
    ----------
    core : CMMCorePlus
        The core whose ``mda`` events drive the acquisition.
    cache_mb : float
        Approximate RAM budget for the shared-memory ring, in megabytes. The number of
        cached frames is ``cache_mb`` / frame-size, capped at the dataset's frame count.
    """

    def __init__(self, core: CMMCorePlus, *, cache_mb: float = 2048.0) -> None:
        self._core = core
        self._cache_mb = cache_mb
        self._queue: mp.Queue = mp.Queue(maxsize=_QUEUE_MAXSIZE)
        self._proc: mp.Process | None = None
        self._ring: RingBuffer | None = None
        self._frame_counter = 0
        self._sizes: dict[str, int] = {}
        self._channels: list[str] = []
        # Grid (`g`) FOVs per stage position; folded into the position axis.
        self._n_grid = 1
        # True total frames in the dataset (all axes), used to cap the ring size.
        self._total_frames = 1

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        """Launch the viewer process and subscribe to MDA events."""
        self._proc = mp.Process(
            target=run_viewer, args=(self._queue,), name="shrimpy-napari-viewer", daemon=True
        )
        self._proc.start()
        events = self._core.mda.events
        events.sequenceStarted.connect(self._on_sequence_started)
        events.frameReady.connect(self._on_frame_ready)
        events.sequenceFinished.connect(self._on_sequence_finished)
        logger.info("napari viewer process started (pid=%s)", self._proc.pid)

    def join(self) -> None:
        """Block until the user closes the viewer window (if it is still alive)."""
        if self._proc is not None and self._proc.is_alive():
            self._proc.join()

    def cleanup(self) -> None:
        """Disconnect events, release shared memory, and tear down the process."""
        events = self._core.mda.events
        for sig, slot in (
            (events.sequenceStarted, self._on_sequence_started),
            (events.frameReady, self._on_frame_ready),
            (events.sequenceFinished, self._on_sequence_finished),
        ):
            try:
                sig.disconnect(slot)
            except Exception:  # noqa: BLE001 - disconnect is best-effort
                pass
        if self._ring is not None:
            try:
                self._ring.close()
            except Exception:  # noqa: BLE001
                logger.debug("Failed to close ring buffer", exc_info=True)
            self._ring = None
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()

    # -- event handlers (run on the acquisition thread) ------------------------

    def _on_sequence_started(self, sequence: MDASequence, meta: object = None) -> None:
        """Capture dataset dimensions and channel names for the viewer."""
        try:
            sizes = sequence.sizes
            # Fold the grid axis into the position axis: a "position" is a stage
            # position x grid FOV, matching the well/FOV layout written to disk.
            self._n_grid = int(sizes.get("g") or 1)
            n_position = int(sizes.get("p") or 1) * self._n_grid
            self._sizes = {
                "position": n_position,
                "t": int(sizes.get("t") or 1),
                "z": int(sizes.get("z") or 1),
            }
            self._channels = [c.config for c in sequence.channels] or ["default"]
            # True frame count across every axis (incl. c and g) -- caps the ring size.
            self._total_frames = max(1, int(np.prod([max(1, int(v)) for v in sizes.values()])))
        except Exception:  # noqa: BLE001 - never propagate into the runner
            logger.debug("Failed to read sequence metadata for viewer", exc_info=True)

    def _on_frame_ready(
        self, image: np.ndarray, event: MDAEvent, metadata: dict | None = None
    ) -> None:
        """Copy the frame into the ring and notify the viewer. Never raises."""
        try:
            if self._ring is None:
                self._init_ring(image)
            assert self._ring is not None
            slot = self._frame_counter % self._ring.n_slots
            self._frame_counter += 1
            self._ring.write(slot, image)
            idx = event.index
            msg = {
                "kind": "frame",
                "slot": slot,
                # Combine stage position (p) and grid FOV (g) into one position index.
                "position": int(idx.get("p", 0)) * self._n_grid + int(idx.get("g", 0)),
                "t": int(idx.get("t", 0)),
                "z": int(idx.get("z", 0)),
                "c": int(idx.get("c", 0)),
            }
            self._put(msg)
        except Exception:  # noqa: BLE001 - viewer must never break acquisition
            logger.debug("Viewer frame handler error (ignored)", exc_info=True)

    def _on_sequence_finished(self, sequence: MDASequence) -> None:
        try:
            self._put({"kind": "finish"})
        except Exception:  # noqa: BLE001
            logger.debug("Failed to send finish to viewer", exc_info=True)

    # -- helpers ---------------------------------------------------------------

    def _init_ring(self, image: np.ndarray) -> None:
        """Allocate the ring on the first frame, then send the viewer a 'start' message."""
        frame_shape = tuple(image.shape)
        dtype = np.dtype(image.dtype)
        frame_bytes = int(np.prod(frame_shape) * dtype.itemsize)
        budget_frames = max(8, int(self._cache_mb * 1e6 // max(1, frame_bytes)))
        n_slots = min(budget_frames, self._total_frames)
        self._ring = RingBuffer.create(n_slots, frame_shape, dtype)
        logger.info(
            "napari ring buffer: %d slots x %s %s (~%.0f MB)",
            n_slots,
            frame_shape,
            dtype,
            n_slots * frame_bytes / 1e6,
        )
        self._put(
            {
                "kind": "start",
                "shm_name": self._ring.name,
                "n_slots": n_slots,
                "frame_shape": frame_shape,
                "dtype": dtype.str,
                "sizes": dict(self._sizes),
                "channels": list(self._channels),
            }
        )

    def _put(self, msg: dict[str, Any]) -> None:
        """Non-blocking queue put; silently drop if the viewer is behind."""
        try:
            self._queue.put_nowait(msg)
        except _queue.Full:
            pass
