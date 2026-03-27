"""Replay camera that serves frames from a pre-acquired OME-Zarr dataset.

Implements a UniMMCore ``SimpleCameraDevice`` that reads frames from an
existing OME-Zarr FOV (single position, 5D TCZYX). This enables offline
testing of DynaTrack and other analysis pipelines without hardware.

Features
--------
- **Channel switching**: Set the ``Channel`` property to match dataset channel
  names. Unknown channels return zeros with a warning.
- **Timepoint auto-increment**: Each snap returns the next timepoint, looping
  over the total number of timepoints.
- **Z-stage tracking**: By default returns the middle z-slice. When connected
  to a Z stage (via ``connect_z_stage``), the z-index shifts with stage position.
- **MDA integration**: ``connect_to_mda`` overrides timepoint/z from MDA events.

Usage (config file)
-------------------
::

    # py pyDevice,Camera,shrimpy.mantis.replay_camera,ReplayCamera
    # py Property,Camera,DataPath,/path/to/dataset.zarr/0/2/003
    Property, Core, Initialize, 1
    Property, Core, Camera, Camera

Usage (programmatic)
--------------------
::

    from pymmcore_plus.experimental.unicore.core._unicore import UniMMCore
    from shrimpy.mantis.replay_camera import ReplayCamera

    core = UniMMCore()
    camera = ReplayCamera()
    camera._data_path = "/path/to/dataset.zarr/0/2/003"
    core.loadPyDevice("Camera", camera)
    core.initializeDevice("Camera")
    core.setCameraDevice("Camera")

    # For Z-stage tracking in GUI mode
    camera.connect_z_stage(core, "Z")

    # For MDA mode
    camera.connect_to_mda(core)
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
    """UniMMCore camera that replays frames from an OME-Zarr FOV dataset.

    The dataset must be a single FOV opened with ``open_ome_zarr(path,
    layout='fov')``, containing 5D data in TCZYX order.

    Set the ``DataPath`` property (pre-init) before calling ``initialize()``.
    """

    def __init__(self) -> None:
        super().__init__()

        # Pre-init state
        self._data_path: str = ""

        # Dataset state (populated in initialize)
        self._dataset = None
        self._data_array = None
        self._nt: int = 0
        self._nc: int = 0
        self._nz: int = 0
        self._ny: int = 512
        self._nx: int = 512
        self._dtype_val: np.dtype = np.dtype(np.uint16)
        self._channel_names: list[str] = []
        self._z_scale: float = 1.0  # um per z-step

        # Current acquisition state
        self._channel_name: str = ""
        self._channel_index: int = 0
        self._t_index: int = 0
        self._z_center: int = 0  # center z-index (nz // 2)
        self._z_position: float = 0.0  # current z-stage position in um
        self._z_origin: float = 0.0  # z-stage position at center of stack
        self._exposure: float = 10.0
        self._mda_connected: bool = False

        # Signal disconnectors
        self._z_disconnect = None
        self._mda_disconnect = None

        # Register pre-init property for the dataset path
        self.register_property(
            name="DataPath",
            property_type=str,
            default_value="",
            getter=lambda d: d._data_path,
            setter=lambda d, v: setattr(d, "_data_path", v),
            is_pre_init=True,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Open the dataset and register channel/position properties."""
        if not self._data_path:
            raise RuntimeError(
                "ReplayCamera: DataPath property must be set before initialization"
            )

        path = Path(self._data_path)
        if not path.exists():
            raise FileNotFoundError(f"ReplayCamera: dataset not found at {path}")

        self._dataset = open_ome_zarr(str(path), layout="fov", mode="r")
        self._data_array = self._dataset["0"]

        shape = self._data_array.shape
        if len(shape) != 5:
            raise ValueError(f"ReplayCamera: expected 5D TCZYX data, got shape {shape}")
        self._nt, self._nc, self._nz, self._ny, self._nx = shape
        self._dtype_val = self._data_array.dtype
        self._channel_names = list(self._dataset.channel_names)
        self._z_center = self._nz // 2

        # Read z-scale from OME-NGFF metadata
        multiscales = self._dataset.zattrs.get("multiscales", [{}])
        datasets = multiscales[0].get("datasets", [{}]) if multiscales else [{}]
        transforms = datasets[0].get("coordinateTransformations", [])
        for t in transforms:
            if t.get("type") == "scale":
                # Scale order matches axes: T, C, Z, Y, X
                self._z_scale = t["scale"][2]
                break

        # Set default channel
        if self._channel_names:
            self._channel_name = self._channel_names[0]
            self._channel_index = 0

        # Register the Channel property (any string accepted)
        self.register_property(
            name="Channel",
            property_type=str,
            default_value=self._channel_name,
            getter=lambda d: d._channel_name,
            setter=lambda d, v: d._set_channel(v),
        )

        logger.info(
            "ReplayCamera initialized: %s | shape=%s (TCZYX) | channels=%s | z_scale=%.4f um",
            self._data_path,
            shape,
            self._channel_names,
            self._z_scale,
        )

    def shutdown(self) -> None:
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None

    # ------------------------------------------------------------------
    # SimpleCameraDevice interface
    # ------------------------------------------------------------------

    def sensor_shape(self) -> tuple[int, int]:
        return (self._ny, self._nx)

    def dtype(self) -> DTypeLike:
        return self._dtype_val

    def get_exposure(self) -> float:
        return self._exposure

    def set_exposure(self, exposure: float) -> None:
        self._exposure = exposure

    def snap(self, buffer: np.ndarray) -> Mapping:
        """Return the frame for the current channel, timepoint, and z-index.

        After each snap, the timepoint counter auto-increments (wrapping at
        the end of the dataset). In MDA mode, the timepoint is overridden
        by ``eventStarted`` events.
        """
        t = self._t_index % self._nt if self._nt > 0 else 0
        z = self._get_z_index()

        if self._channel_index < 0 or self._channel_index >= self._nc:
            # Channel not in dataset — return zeros
            buffer[:] = 0
        else:
            buffer[:] = np.asarray(self._data_array[t, self._channel_index, z])

        # Auto-increment timepoint (MDA mode overrides via event tracking)
        if not self._mda_connected:
            self._t_index += 1

        return {"TimeIndex": str(t), "ZIndex": str(z), "Channel": self._channel_name}

    # ------------------------------------------------------------------
    # Channel management
    # ------------------------------------------------------------------

    def _set_channel(self, name: str) -> None:
        """Set the active channel by name."""
        self._channel_name = name
        if name in self._channel_names:
            self._channel_index = self._channel_names.index(name)
        else:
            logger.warning(
                "ReplayCamera: channel '%s' not in dataset (available: %s). Returning zeros.",
                name,
                self._channel_names,
            )
            self._channel_index = -1

    # ------------------------------------------------------------------
    # Z-position tracking
    # ------------------------------------------------------------------

    def _get_z_index(self) -> int:
        """Map the current z-stage position to a dataset z-index."""
        if self._z_scale == 0:
            return self._z_center
        offset = round((self._z_position - self._z_origin) / self._z_scale)
        z = self._z_center + offset
        return max(0, min(self._nz - 1, z))

    def connect_z_stage(self, core: CMMCorePlus, device_name: str | None = None) -> None:
        """Connect z-index tracking to a stage device via core events.

        When the named stage device moves, the camera's z-index updates
        correspondingly. If *device_name* is ``None``, the core's current
        focus device is used.

        Parameters
        ----------
        core : CMMCorePlus
            The core instance (not the device proxy — use the actual core).
        device_name : str | None
            Stage device label to track. Defaults to ``core.getFocusDevice()``.
        """
        if device_name is None:
            device_name = core.getFocusDevice()

        # Store the current stage position as origin (maps to center z)
        try:
            self._z_origin = core.getPosition(device_name)
            self._z_position = self._z_origin
        except Exception:
            self._z_origin = 0.0
            self._z_position = 0.0

        def _on_stage_changed(dev: str, pos: float) -> None:
            if dev == device_name:
                self._z_position = pos

        core.events.stagePositionChanged.connect(_on_stage_changed)
        self._z_disconnect = lambda: core.events.stagePositionChanged.disconnect(
            _on_stage_changed
        )
        logger.info(
            "ReplayCamera: tracking Z stage '%s' (origin=%.2f, scale=%.4f um/step)",
            device_name,
            self._z_origin,
            self._z_scale,
        )

    def disconnect_z_stage(self) -> None:
        """Disconnect from Z stage tracking."""
        if self._z_disconnect is not None:
            self._z_disconnect()
            self._z_disconnect = None

    # ------------------------------------------------------------------
    # MDA event tracking
    # ------------------------------------------------------------------

    def connect_to_mda(self, core: CMMCorePlus) -> None:
        """Connect to MDA event signals to track TCZP indices.

        In MDA mode, timepoint auto-increment is disabled; the timepoint
        and z-position are set by each ``eventStarted`` signal.
        """
        core.mda.events.eventStarted.connect(self._on_event_started)
        self._mda_connected = True
        self._mda_disconnect = lambda: (
            core.mda.events.eventStarted.disconnect(self._on_event_started),
            setattr(self, "_mda_connected", False),
        )

    def disconnect_from_mda(self) -> None:
        """Disconnect from MDA event signals."""
        if self._mda_disconnect is not None:
            self._mda_disconnect()
            self._mda_disconnect = None
            self._mda_connected = False

    def _on_event_started(self, event: MDAEvent) -> None:
        """Update state from the running MDA event."""
        idx = event.index
        self._t_index = idx.get("t", 0)

        # Channel from event
        if event.channel and event.channel.config:
            self._set_channel(event.channel.config)
        elif "c" in idx:
            c = idx["c"]
            if 0 <= c < len(self._channel_names):
                self._set_channel(self._channel_names[c])

        # Z position from event
        if event.z_pos is not None:
            self._z_position = event.z_pos

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def num_positions(self) -> int:
        return 1  # single FOV

    @property
    def data_shape(self) -> tuple[int, ...]:
        """Per-position data shape: ``(T, C, Z, Y, X)``."""
        return (self._nt, self._nc, self._nz, self._ny, self._nx)

    @property
    def channel_names(self) -> list[str]:
        return list(self._channel_names)

    @property
    def z_scale(self) -> float:
        return self._z_scale

    def get_frame(self, t: int, c: int, z: int) -> np.ndarray:
        """Read a single 2-D frame directly from the dataset."""
        if self._data_array is None:
            raise RuntimeError("ReplayCamera not initialized")
        t = t % self._nt
        c = c % self._nc
        z = z % self._nz
        return np.asarray(self._data_array[t, c, z])
