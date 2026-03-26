"""Replay camera that serves frames from a pre-acquired OME-Zarr dataset.

Patches a CMMCorePlus instance at the Python level so that ``snapImage`` /
``getImage`` (and the sequenced-acquisition equivalents) return data read from
an existing OME-Zarr plate rather than from a live camera.  This enables
offline testing of DynaTrack and other analysis pipelines without hardware.

The patching approach follows the ``Sample.patch()`` pattern from
``pymmcore_plus.experimental.simulate``.

Usage
-----
::

    from shrimpy.mantis.replay_camera import ReplayCamera

    camera = ReplayCamera("/path/to/dataset.ome.zarr")
    core = CMMCorePlus()
    core.loadSystemConfiguration()  # demo config for stages

    with camera.patch(core):
        engine = MantisEngine(core)
        core.mda.run(sequence)  # frames come from the dataset
"""

from __future__ import annotations

import logging

from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from iohub.ngff import open_ome_zarr
from pymmcore_plus.core._sequencing import SequencedEvent
from pymmcore_plus.experimental.simulate._sample import patch_with_object

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any

    from pymmcore_plus import CMMCorePlus
    from useq import MDAEvent

logger = logging.getLogger(__name__)


class ReplayCamera:
    """Patches CMMCorePlus to serve frames from an OME-Zarr dataset.

    Parameters
    ----------
    data_path : str | Path
        Path to an HCS-layout OME-Zarr plate store.  Positions are
        enumerated in the order returned by ``plate.positions()`` and
        mapped 1-to-1 to MDA position indices.
    """

    def __init__(self, data_path: str | Path) -> None:
        self._data_path = Path(data_path)
        self._plate = open_ome_zarr(self._data_path, layout="hcs", mode="r")

        # Build ordered list of (name, position_node) tuples
        self._positions: list[tuple[str, Any]] = list(self._plate.positions())
        if not self._positions:
            raise ValueError(f"No positions found in {self._data_path}")

        # Cache shape from the first position (all assumed identical)
        _first_pos = self._positions[0][1]
        self._shape = _first_pos["0"].shape  # (T, C, Z, Y, X)
        self._dtype = _first_pos["0"].dtype

        logger.info(
            "ReplayCamera opened %s with %d positions, shape=%s, dtype=%s",
            self._data_path,
            len(self._positions),
            self._shape,
            self._dtype,
        )

    # ------ Public properties ------

    @property
    def num_positions(self) -> int:
        return len(self._positions)

    @property
    def shape(self) -> tuple[int, ...]:
        """Per-position data shape: ``(T, C, Z, Y, X)``."""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    # ------ Data access ------

    def get_frame(self, p: int, t: int, c: int, z: int) -> np.ndarray:
        """Read a single 2-D frame from the dataset.

        Parameters
        ----------
        p : int
            Position index.
        t, c, z : int
            Timepoint, channel, and z-slice indices.

        Returns
        -------
        np.ndarray
            2-D array of shape ``(Y, X)``.
        """
        if p >= len(self._positions):
            raise IndexError(
                f"Position index {p} out of range (dataset has {len(self._positions)})"
            )
        _, pos_node = self._positions[p]
        # Clamp indices to dataset bounds to allow cycling / overflow gracefully
        nt, nc, nz, _ny, _nx = pos_node["0"].shape
        t = t % nt
        c = c % nc
        z = z % nz
        return np.asarray(pos_node["0"][t, c, z])

    # ------ Patching ------

    @contextmanager
    def patch(self, core: CMMCorePlus) -> Generator[ReplayCamera, None, None]:
        """Context manager that patches *core* to serve replay frames.

        While active, ``snapImage`` / ``getImage`` and the sequenced-acquisition
        methods on *core* are replaced with implementations that read from the
        OME-Zarr dataset instead of talking to hardware.

        Yields
        ------
        ReplayCamera
            This instance.
        """
        patcher = _ReplayCameraPatcher(core, self)

        # Connect to MDA event signals so we track the current event
        core.mda.events.eventStarted.connect(patcher._on_event_started)
        try:
            with patch_with_object(core, patcher):
                yield self
        finally:
            core.mda.events.eventStarted.disconnect(patcher._on_event_started)


class _ReplayCameraPatcher:
    """Houses the replacement methods injected into CMMCorePlus."""

    def __init__(self, core: CMMCorePlus, camera: ReplayCamera) -> None:
        self._core = core
        self._camera = camera

        # Current MDA event indices (updated by signal)
        self._t: int = 0
        self._p: int = 0
        self._c: int = 0
        self._z: int = 0

        # Original methods we delegate to in some cases
        self._original_snapImage = core.snapImage

        # Sequence acquisition state
        self._seq_queue: deque[np.ndarray] = deque()
        self._seq_running: bool = False

    # ------ MDA event tracking ------

    def _on_event_started(self, event: MDAEvent) -> None:
        """Stash the current TCPZ indices from the running MDA event."""
        idx = event.index
        self._t = idx.get("t", 0)
        self._p = idx.get("p", 0)
        self._c = idx.get("c", 0)
        self._z = idx.get("z", 0)

    # ------ Snap / getImage replacements ------

    def snapImage(self) -> None:  # noqa: N802
        """No-op snap — the frame is already available in the dataset."""

    def getImage(self, *_: Any, **__: Any) -> np.ndarray:  # noqa: N802
        """Return the frame for the current MDA event indices."""
        return self._camera.get_frame(self._p, self._t, self._c, self._z)

    def getLastImage(self, *_: Any, **__: Any) -> np.ndarray:  # noqa: N802
        """Alias for getImage (used in live-mode paths)."""
        return self.getImage()

    def getImageWidth(self) -> int:  # noqa: N802
        return self._camera.shape[-1]  # X

    def getImageHeight(self) -> int:  # noqa: N802
        return self._camera.shape[-2]  # Y

    def getImageBitDepth(self) -> int:  # noqa: N802
        return int(np.dtype(self._camera.dtype).itemsize * 8)

    # ------ Sequenced acquisition replacements ------

    def startSequenceAcquisition(  # noqa: N802
        self,
        numImages: int,  # noqa: N803
        intervalMs: float = 0.0,  # noqa: N803
        stopOnOverflow: bool = True,  # noqa: N803
    ) -> None:
        """Populate the frame queue for a hardware-sequenced burst.

        The engine will call this for ``SequencedEvent`` groups.  We pre-load
        all the frames that the sequence would produce.
        """
        self._seq_queue.clear()
        # We don't know the exact sub-events here; frames will be pushed
        # via _prepare_sequence_frames if available, or we fall back to
        # repeating the current frame.
        for _ in range(numImages):
            self._seq_queue.append(self._camera.get_frame(self._p, self._t, self._c, self._z))
        self._seq_running = True

    def getRemainingImageCount(self) -> int:  # noqa: N802
        return len(self._seq_queue)

    def isSequenceRunning(self, *_: Any) -> bool:  # noqa: N802
        return self._seq_running and len(self._seq_queue) > 0

    def popNextImageAndMD(self) -> tuple[np.ndarray, dict]:  # noqa: N802
        """Pop the next frame and empty metadata from the sequence queue."""
        if not self._seq_queue:
            raise RuntimeError("No images in sequence queue")
        frame = self._seq_queue.popleft()
        if not self._seq_queue:
            self._seq_running = False
        return frame, {}

    def stopSequenceAcquisition(self) -> None:  # noqa: N802
        self._seq_queue.clear()
        self._seq_running = False

    # ------ Helpers to pre-populate sequence frames from events ------

    def prepare_sequence_frames(self, sequenced_event: SequencedEvent) -> None:
        """Pre-load frames for a SequencedEvent so they are served in order.

        Call this *before* the engine calls ``startSequenceAcquisition`` if
        you have access to the ``SequencedEvent`` and want accurate per-sub-event
        frames.
        """
        self._seq_queue.clear()
        for sub in sequenced_event.events:
            idx = sub.index
            p = idx.get("p", 0)
            t = idx.get("t", 0)
            c = idx.get("c", 0)
            z = idx.get("z", 0)
            self._seq_queue.append(self._camera.get_frame(p, t, c, z))
        self._seq_running = True
