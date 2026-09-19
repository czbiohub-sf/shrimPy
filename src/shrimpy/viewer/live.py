"""Launch the napari viewer alongside a running acquisition.

:class:`LiveViewer` starts :func:`shrimpy.viewer._napari_process.run_viewer` in a child
process and tells it which OME-Zarr store the acquisition is writing. That is the whole
contract: the viewer reads the store from disk itself, so **no image data crosses the
process boundary** and nothing the viewer does -- a slow repaint, a GUI hang, a segfault
-- can touch the acquisition. The acquisition thread is never called back into either;
the only wiring is two lifecycle signals.

The child is started *before* the acquisition so napari's slow import overlaps with the
first frames, and it is told the store path once
:meth:`~shrimpy.engines.base_engine.BaseEngine.acquire` has named it. Data acquired
during that window is not lost, unlike a stream of frames pushed at a viewer that is not
listening yet: it is on disk, and the viewer picks it up when it opens the store.
"""

from __future__ import annotations

import logging
import multiprocessing as mp

from typing import TYPE_CHECKING, Any

from shrimpy.viewer._napari_process import DEFAULT_REFRESH_MS, run_viewer
from shrimpy.viewer.store import DEFAULT_CACHE_MB

if TYPE_CHECKING:
    from useq import MDASequence

    from shrimpy.engines.base_engine import BaseEngine

logger = logging.getLogger(__name__)


class LiveViewer:
    """Shows an engine's acquisition in an out-of-process napari window.

    Parameters
    ----------
    engine : BaseEngine
        The engine that will run the acquisition; its ``data_path`` names the store to
        view and its core's MDA events drive the viewer's lifecycle.
    deskew : bool
        Whether this microscope uses oblique-plane deskew. When True (e.g. mantis) the
        viewer shows the Deskew widget with deskew on by default (toggleable), provided
        the store records a scan step and holds a real z-stack. Other microscopes
        (e.g. iSIM) pass False.
    cache_mb : float
        RAM budget for the viewer's cache of decompressed volumes.
    refresh_ms : int
        How often the viewer re-reads the store for newly acquired data.
    """

    def __init__(
        self,
        engine: BaseEngine,
        *,
        deskew: bool = False,
        cache_mb: float = DEFAULT_CACHE_MB,
        refresh_ms: int = DEFAULT_REFRESH_MS,
    ) -> None:
        self._engine = engine
        self._deskew = deskew
        self._cache_mb = cache_mb
        self._refresh_ms = refresh_ms
        self._control: mp.Queue = mp.Queue()
        self._process: mp.Process | None = None

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        """Launch the viewer process and subscribe to the acquisition's lifecycle."""
        self._process = mp.Process(
            target=run_viewer,
            args=(None, self._control),
            kwargs={
                "deskew": self._deskew,
                "cache_mb": self._cache_mb,
                "refresh_ms": self._refresh_ms,
            },
            name="shrimpy-napari-viewer",
            daemon=True,
        )
        self._process.start()
        events = self._engine.mmcore.mda.events
        events.sequenceStarted.connect(self._on_sequence_started)
        events.sequenceFinished.connect(self._on_sequence_finished)
        logger.info("napari viewer process started (pid=%s)", self._process.pid)

    def join(self) -> None:
        """Block until the user closes the viewer window (if it is still alive)."""
        if self._process is not None and self._process.is_alive():
            self._process.join()

    def cleanup(self) -> None:
        """Disconnect the lifecycle signals and tear the process down."""
        events = self._engine.mmcore.mda.events
        for signal, slot in (
            (events.sequenceStarted, self._on_sequence_started),
            (events.sequenceFinished, self._on_sequence_finished),
        ):
            try:
                signal.disconnect(slot)
            except Exception:  # noqa: BLE001 - disconnect is best-effort
                pass
        if self._process is not None and self._process.is_alive():
            self._process.terminate()

    # -- event handlers (run on the acquisition thread) ------------------------

    def _on_sequence_started(self, sequence: MDASequence, meta: object = None) -> None:
        """Tell the viewer which store to open. Never raises."""
        path = self._engine.data_path
        if path is None:
            logger.warning("Acquisition has no output path; viewer has nothing to show.")
            return
        self._send({"kind": "open", "path": str(path)})

    def _on_sequence_finished(self, sequence: MDASequence) -> None:
        self._send({"kind": "finish"})

    def _send(self, message: dict[str, Any]) -> None:
        """Queue a lifecycle message; a dead or wedged viewer must not break the run."""
        try:
            self._control.put_nowait(message)
        except Exception:  # noqa: BLE001 - viewer must never break acquisition
            logger.debug("Could not notify viewer (ignored)", exc_info=True)
