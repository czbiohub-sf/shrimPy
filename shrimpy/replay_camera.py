"""Replay camera that serves frames from a pre-acquired OME-Zarr dataset.

Implements a UniMMCore ``SimpleCameraDevice`` that reads frames from an
existing OME-Zarr dataset (5D TCZYX). Both single-FOV datasets and
multi-position HCS plates are supported. This enables offline testing of
DynaTrack and other analysis pipelines without hardware.

Features
--------
- **Multi-position (HCS)**: Point ``DataPath`` at a plate root to expose every
  position under its HCS key (e.g. ``"0/2/000"``). The camera defaults to the
  first position. In MDA mode, the position is matched to each event's
  ``pos_name`` (falling back to the position index).
- **Channel switching**: Set the ``Channel`` property to match dataset channel
  names. Unknown channels return zeros with a warning.
- **Timepoint auto-increment**: Each snap returns the next timepoint, looping
  over the total number of timepoints.
- **Z-stage tracking**: By default returns the middle z-slice. When connected
  to a Z stage (via ``connect_z_stage``), the z-index shifts with stage position.
- **MDA integration**: ``connect_to_mda`` overrides timepoint/z/position from
  MDA events.

Usage (config file)
-------------------
::

    # Single FOV — point at a position within the store
    # py pyDevice,Camera,shrimpy.replay_camera,ReplayCamera
    # py Property,Camera,DataPath,/path/to/dataset.zarr/0/2/003

    # HCS plate — point at the plate root (defaults to the first position)
    # py Property,Camera,DataPath,/path/to/plate.zarr
    Property, Core, Initialize, 1
    Property, Core, Camera, Camera

Usage (programmatic)
--------------------
::

    from pymmcore_plus.experimental.unicore.core._unicore import UniMMCore
    from shrimpy.replay_camera import ReplayCamera

    core = UniMMCore()
    camera = ReplayCamera()
    camera._data_path = "/path/to/plate.zarr"  # or a single-FOV path
    core.loadPyDevice("Camera", camera)
    core.initializeDevice("Camera")
    core.setCameraDevice("Camera")

    # Switch position manually (GUI mode)
    camera.set_position("0/2/000")

    # For Z-stage tracking in GUI mode
    camera.connect_z_stage(core, "Z")

    # For MDA mode (position tracked from event pos_name)
    camera.connect_to_mda(core)
"""

from __future__ import annotations

import logging

from collections import deque
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

# Key used for the single position of a non-HCS (FOV) dataset.
DEFAULT_POSITION_KEY = "0"


class ReplayCamera(SimpleCameraDevice):
    """UniMMCore camera that replays frames from an OME-Zarr dataset.

    The dataset may be a single FOV (``layout='fov'``) or an HCS plate
    (``layout='hcs'``), containing 5D data in TCZYX order.

    Set the ``DataPath`` property (pre-init) before calling ``initialize()``.

    Since data is typically chunked in ZYX, reading one Z slice loads the whole
    volume off disk; the most recently read ZYX volume is cached so that
    sibling Z slices (same position/timepoint/channel) are served from memory.
    """

    def __init__(self) -> None:
        super().__init__()

        # Pre-init state
        self._data_path: str = ""

        # Cache of the most recently decoded ZYX volume and its (position, t, c)
        self._cached_volume: np.ndarray | None = None
        self._cached_key: tuple[str, int, int] | None = None

        # Dataset state (populated in initialize)
        self._dataset = None
        # Position key (e.g. "0/2/000") -> lazy dask array for that FOV. A
        # single-FOV dataset yields one entry keyed by DEFAULT_POSITION_KEY.
        self._data_arrays: dict[str, object] = {}
        self._positions: dict = {}  # position key -> iohub Position node
        self._position_keys: list[str] = []
        self._current_position_key: str = ""
        self._data_array = None  # dask array for the active position
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

        # Queue of z-indices for sequenced (hardware-triggered) acquisitions
        self._z_queue: deque[int] = deque()

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
        """Open the dataset and register channel/position properties.

        Supports both single-FOV datasets (``layout='fov'``) and HCS plates
        (``layout='hcs'``) with multiple positions. For a plate, every
        position is exposed under its HCS key (e.g. ``"0/2/000"``) and the
        camera defaults to the first position.
        """
        if not self._data_path:
            raise RuntimeError(
                "ReplayCamera: DataPath property must be set before initialization"
            )

        path = Path(self._data_path)
        if not path.exists():
            raise FileNotFoundError(f"ReplayCamera: dataset not found at {path}")

        logger.info("ReplayCamera: opening dataset (lazy, backed by dask)...")
        self._dataset = open_ome_zarr(str(path), layout="auto", mode="r")

        # Collect all positions and their lazy dask arrays. Frames are read
        # from disk on demand rather than holding data in memory.
        self._positions = self._collect_positions(self._dataset)
        if not self._positions:
            raise ValueError(f"ReplayCamera: no positions found in dataset at {path}")

        for key, position in self._positions.items():
            data_array = position.data.dask_array()
            shape = data_array.shape
            if len(shape) != 5:
                raise ValueError(
                    f"ReplayCamera: expected 5D TCZYX data at position '{key}', "
                    f"got shape {shape}"
                )
            self._data_arrays[key] = data_array
        self._position_keys = list(self._data_arrays)

        # All positions share the same TCZYX dimensions, dtype, channels, and
        # z-scale; derive them once from the first position.
        first_key = self._position_keys[0]
        first_position = self._positions[first_key]
        first_array = self._data_arrays[first_key]
        self._nt, self._nc, self._nz, self._ny, self._nx = first_array.shape
        self._dtype_val = first_array.dtype
        self._z_center = self._nz // 2
        self._channel_names = list(first_position.channel_names)
        self._z_scale = self._read_z_scale(first_position)

        # Default to the first position (swaps in its dask array)
        self._set_position(first_key)

        # Set default channel from the active position
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

        # Register the Position property (HCS key), restricted to known
        # positions. Lets GUI users switch FOVs; MDA events override it.
        self.register_property(
            name="Position",
            property_type=str,
            default_value=self._current_position_key,
            allowed_values=self._position_keys,
            getter=lambda d: d._current_position_key,
            setter=lambda d, v: d._set_position(v),
        )

        logger.info(
            "ReplayCamera initialized: %s | positions=%s | shape=%s (TCZYX) | "
            "channels=%s | z_scale=%.4f um",
            self._data_path,
            self._position_keys,
            self.data_shape,
            self._channel_names,
            self._z_scale,
        )

    @staticmethod
    def _collect_positions(dataset) -> dict:
        """Map position key -> Position node for FOV or HCS-plate datasets.

        For a single FOV the returned mapping has one entry keyed by
        ``DEFAULT_POSITION_KEY``. For an HCS plate the keys are the position
        paths within the plate (e.g. ``"0/2/000"``).
        """
        # Plate exposes a positions() generator yielding (path, Position)
        if hasattr(dataset, "positions"):
            return {key: position for key, position in dataset.positions()}
        # Single FOV (Position node) exposes `.data` directly
        return {DEFAULT_POSITION_KEY: dataset}

    @staticmethod
    def _read_z_scale(position) -> float:
        """Read the z-step size (um) from a position's OME-NGFF metadata."""
        multiscales = position.zattrs.get("multiscales", [{}])
        datasets = multiscales[0].get("datasets", [{}]) if multiscales else [{}]
        transforms = datasets[0].get("coordinateTransformations", [])
        for t in transforms:
            if t.get("type") == "scale":
                # Scale order matches axes: T, C, Z, Y, X
                return t["scale"][2]
        return 1.0

    def shutdown(self) -> None:
        self._cached_volume = None
        self._cached_key = None
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

    def _get_volume(self, position_key: str, t: int, c: int) -> np.ndarray:
        """Return the ZYX volume for ``(position_key, t, c)``, caching it.

        Because the data is chunked in ZYX, computing any single slice reads
        and decompresses the whole volume anyway. We compute the full ZYX
        volume once and keep only the most recent one in memory so subsequent
        z-slices from the same position/timepoint/channel are served instantly.
        """
        cache_key = (position_key, t, c)
        if self._cached_key == cache_key and self._cached_volume is not None:
            return self._cached_volume

        volume = np.asarray(self._data_arrays[position_key][t, c].compute())
        self._cached_volume = volume
        self._cached_key = cache_key
        return volume

    def snap(self, buffer: np.ndarray) -> Mapping:
        """Return the frame for the current channel, timepoint, and z-index.

        After each snap, the timepoint counter auto-increments (wrapping at
        the end of the dataset). In MDA mode, the timepoint is overridden
        by ``eventStarted`` events.

        For sequenced (hardware-triggered) acquisitions, z-indices are
        pre-queued by ``_on_event_started`` and popped on each snap.
        """
        t = self._t_index % self._nt if self._nt > 0 else 0

        # Use queued z-index for sequenced acquisitions, else compute from position
        if self._z_queue:
            z = self._z_queue.popleft()
        else:
            z = self._get_z_index()

        if self._channel_index < 0 or self._channel_index >= self._nc:
            # Channel not in dataset — return zeros
            buffer[:] = 0
        else:
            # Pull the (cached) ZYX volume and copy out the requested z-slice
            volume = self._get_volume(self._current_position_key, t, self._channel_index)
            buffer[:] = volume[z]

        # Auto-increment timepoint (MDA mode overrides via event tracking)
        if not self._mda_connected:
            self._t_index += 1

        return {
            "TimeIndex": str(t),
            "ZIndex": str(z),
            "Channel": self._channel_name,
            "Position": self._current_position_key,
        }

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _set_position(self, key: str) -> None:
        """Switch the active position (FOV) to the given HCS key.

        Only the active dask array is swapped; all positions are assumed to
        share the same TCZYX shape, dtype, channels, and z-scale (derived
        once at initialization).
        """
        if key not in self._data_arrays:
            logger.warning(
                "ReplayCamera: position '%s' not in dataset (available: %s). "
                "Keeping current position '%s'.",
                key,
                self._position_keys,
                self._current_position_key,
            )
            return

        self._current_position_key = key
        self._data_array = self._data_arrays[key]

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
        """Update state from the running MDA event.

        For ``SequencedEvent`` (hardware-triggered bursts), the z-indices
        of all sub-events are queued so that each ``snap()`` returns the
        correct z-slice.

        The position is matched by ``pos_name`` against the HCS keys (e.g.
        ``"0/2/000"``), falling back to the position index ``p``.
        """
        from pymmcore_plus.core._sequencing import SequencedEvent

        if isinstance(event, SequencedEvent):
            sub_events = event.events
            first = sub_events[0]
            idx = first.index
            self._t_index = idx.get("t", 0)

            # Position from first sub-event (matched by name, then index)
            self._set_position_from_event(first)

            # Channel from first sub-event
            if first.channel and first.channel.config:
                self._set_channel(first.channel.config)
            elif "c" in idx:
                c = idx["c"]
                if 0 <= c < len(self._channel_names):
                    self._set_channel(self._channel_names[c])

            # Queue z-indices for all sub-events
            self._z_queue.clear()
            for sub in sub_events:
                self._z_queue.append(sub.index.get("z", self._z_center))
        else:
            idx = event.index
            self._t_index = idx.get("t", 0)
            self._z_queue.clear()

            # Position from event (matched by name, then index)
            self._set_position_from_event(event)

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

    def _set_position_from_event(self, event: MDAEvent) -> None:
        """Switch position from an MDA (sub-)event.

        Prefers matching ``event.pos_name`` against the HCS position keys,
        then falls back to the position index ``p`` in the event index.
        Single-position datasets are left untouched.
        """
        if len(self._position_keys) <= 1:
            return

        pos_name = getattr(event, "pos_name", None)
        if pos_name and pos_name in self._data_arrays:
            self._set_position(pos_name)
            return

        p = event.index.get("p")
        if p is not None and 0 <= p < len(self._position_keys):
            self._set_position(self._position_keys[p])

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def num_positions(self) -> int:
        return len(self._position_keys)

    @property
    def position_keys(self) -> list[str]:
        """HCS keys of all available positions (e.g. ``["0/2/000", ...]``)."""
        return list(self._position_keys)

    @property
    def current_position(self) -> str:
        """HCS key of the active position."""
        return self._current_position_key

    @property
    def data_shape(self) -> tuple[int, ...]:
        """Active-position data shape: ``(T, C, Z, Y, X)``."""
        return (self._nt, self._nc, self._nz, self._ny, self._nx)

    @property
    def channel_names(self) -> list[str]:
        return list(self._channel_names)

    @property
    def z_scale(self) -> float:
        return self._z_scale

    def set_position(self, key: str) -> None:
        """Switch the active position by HCS key (e.g. ``"0/2/000"``)."""
        self._set_position(key)

    def get_frame(self, t: int, c: int, z: int, position: str | None = None) -> np.ndarray:
        """Read a single 2-D frame on demand from the lazy dataset.

        If *position* is given, reads from that position instead of the
        active one, without changing the active position.
        """
        key = self._current_position_key if position is None else position
        if key not in self._data_arrays:
            raise RuntimeError(
                f"ReplayCamera not initialized or unknown position '{position}'"
            )
        volume = self._get_volume(key, t % self._nt, c % self._nc)
        # Copy so callers can't mutate the cached volume through the view
        return volume[z % self._nz].copy()
