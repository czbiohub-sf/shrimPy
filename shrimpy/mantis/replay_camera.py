"""Replay camera that serves frames from a pre-acquired OME-Zarr dataset.

Implements a UniMMCore ``SimpleCameraDevice`` that reads frames from an
existing OME-Zarr HCS-layout plate store.  This enables offline testing of
DynaTrack and other analysis pipelines without hardware.

Usage
-----
::

    from pymmcore_plus.experimental.unicore.core._unicore import UniMMCore
    from shrimpy.mantis.replay_camera import ReplayCamera

    core = UniMMCore()
    camera = ReplayCamera("/path/to/dataset.ome.zarr")
    core.loadPyDevice("Camera", camera)
    core.initializeAllDevices()
    core.setCameraDevice("Camera")

    # Connect to MDA events so the camera tracks current TCZP indices
    camera.connect_to_mda(core)

    engine = MantisEngine(core)
    core.mda.run(sequence)

    camera.disconnect_from_mda(core)
"""

from __future__ import annotations

import logging

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from iohub.ngff import open_ome_zarr
from pymmcore_plus.experimental.unicore import SimpleCameraDevice

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import DTypeLike
    from pymmcore_plus import CMMCorePlus
    from useq import MDAEvent

logger = logging.getLogger(__name__)


class ReplayCamera(SimpleCameraDevice):
    """UniMMCore camera device that replays frames from an OME-Zarr dataset.

    Parameters
    ----------
    data_path : str | Path
        Path to an HCS-layout OME-Zarr plate store.  Positions are
        enumerated in the order returned by ``plate.positions()`` and
        mapped 1-to-1 to MDA position indices.
    """

    def __init__(self, data_path: str | Path) -> None:
        super().__init__()
        self._data_path = Path(data_path)
        self._plate = open_ome_zarr(self._data_path, layout="hcs", mode="r")

        # Build ordered list of (name, position_node) tuples
        self._positions: list[tuple[str, object]] = list(self._plate.positions())
        if not self._positions:
            raise ValueError(f"No positions found in {self._data_path}")

        # Cache shape from the first position (all assumed identical)
        _first_pos = self._positions[0][1]
        self._tcz_shape = _first_pos["0"].shape  # (T, C, Z, Y, X)
        self._ny: int = self._tcz_shape[-2]
        self._nx: int = self._tcz_shape[-1]
        self._dtype_val: np.dtype = _first_pos["0"].dtype
        self._exposure: float = 10.0

        # Current MDA event indices (updated by signal)
        self._t: int = 0
        self._p: int = 0
        self._c: int = 0
        self._z: int = 0

        logger.info(
            "ReplayCamera opened %s with %d positions, shape=%s, dtype=%s",
            self._data_path,
            len(self._positions),
            self._tcz_shape,
            self._dtype_val,
        )

    # ------ Public properties ------

    @property
    def num_positions(self) -> int:
        return len(self._positions)

    @property
    def tcz_shape(self) -> tuple[int, ...]:
        """Per-position data shape: ``(T, C, Z, Y, X)``."""
        return self._tcz_shape

    # ------ SimpleCameraDevice interface ------

    def sensor_shape(self) -> tuple[int, int]:
        return (self._ny, self._nx)

    def dtype(self) -> DTypeLike:
        return self._dtype_val

    def get_exposure(self) -> float:
        return self._exposure

    def set_exposure(self, exposure: float) -> None:
        self._exposure = exposure

    def snap(self, buffer: np.ndarray) -> Mapping:
        """Read the frame for the current TCZP indices into *buffer*."""
        frame = self.get_frame(self._p, self._t, self._c, self._z)
        buffer[:] = frame
        return {}

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

    # ------ MDA event tracking ------

    def connect_to_mda(self, core: CMMCorePlus) -> None:
        """Connect to MDA event signals to track current TCZP indices."""
        core.mda.events.eventStarted.connect(self._on_event_started)

    def disconnect_from_mda(self, core: CMMCorePlus) -> None:
        """Disconnect from MDA event signals."""
        core.mda.events.eventStarted.disconnect(self._on_event_started)

    def _on_event_started(self, event: MDAEvent) -> None:
        """Stash the current TCPZ indices from the running MDA event."""
        idx = event.index
        self._t = idx.get("t", 0)
        self._p = idx.get("p", 0)
        self._c = idx.get("c", 0)
        self._z = idx.get("z", 0)
