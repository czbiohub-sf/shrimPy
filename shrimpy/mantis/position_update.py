from __future__ import annotations

import logging
import threading

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from useq import MDAEvent, MDASequence

logger = logging.getLogger(__name__)


@dataclass
class PositionCoordinates:
    """Coordinates for a single stage position."""

    x: float
    y: float
    z: float | None = None


class PositionStore:
    """Thread-safe store for dynamically updated stage positions.

    Initialized from the MDASequence's stage_positions and supports
    concurrent reads (from the event loop) and writes (from the
    updater thread).
    """

    def __init__(self) -> None:
        self._positions: dict[int, PositionCoordinates] = {}
        self._lock = threading.Lock()

    def initialize_from_sequence(
        self, sequence: MDASequence, z_device: str | None = None
    ) -> None:
        """Populate the store from the sequence's stage_positions.

        Parameters
        ----------
        sequence : MDASequence
            The acquisition sequence.
        z_device : str | None
            If set, read the initial Z value from the position's device
            properties (e.g. ``ObjectiveZ``) instead of ``pos.z``.
        """
        with self._lock:
            self._positions.clear()
            for idx, pos in enumerate(sequence.stage_positions):
                z = pos.z
                if z_device and pos.properties:
                    for dev, prop, value in pos.properties:
                        if dev == z_device and prop == "Position":
                            z = float(value)
                            break
                self._positions[idx] = PositionCoordinates(
                    x=pos.x if pos.x is not None else 0.0,
                    y=pos.y if pos.y is not None else 0.0,
                    z=z,
                )

    def get_position(self, position_index: int) -> PositionCoordinates | None:
        """Return the current coordinates for position_index, or None."""
        with self._lock:
            pos = self._positions.get(position_index)
            if pos is None:
                return None
            return PositionCoordinates(x=pos.x, y=pos.y, z=pos.z)

    def update_position(
        self, position_index: int, x: float, y: float, z: float | None = None
    ) -> None:
        """Write updated coordinates for position_index."""
        with self._lock:
            self._positions[position_index] = PositionCoordinates(x=x, y=y, z=z)

    def get_all_positions(self) -> dict[int, PositionCoordinates]:
        """Return a snapshot of all positions (copy)."""
        with self._lock:
            return {
                k: PositionCoordinates(x=v.x, y=v.y, z=v.z) for k, v in self._positions.items()
            }

    @property
    def num_positions(self) -> int:
        with self._lock:
            return len(self._positions)


@dataclass
class PositionUpdateConfig:
    """Configuration for position updating, read from sequence metadata."""

    enabled: bool = False
    update_channel: int | None = 0  # channel index to cache; None = all channels
    z_device: str | None = None  # device name for Z updates (e.g. "ObjectiveZ")


class PositionUpdater:
    """Base class for position updaters.

    Subclass to add stateful tracking (e.g. reference images, models).
    The ``update`` method is called in a background thread after each
    position's z-stack is acquired.
    """

    def update(
        self,
        timepoint_index: int,
        position_index: int,
        position: PositionCoordinates,
        data: list[np.ndarray] | None = None,
    ) -> PositionCoordinates:
        """Return updated coordinates for the given position.

        Parameters
        ----------
        timepoint_index : int
            The current timepoint index.
        position_index : int
            The position that was just acquired.
        position : PositionCoordinates
            Current coordinates for this position.
        data : list[np.ndarray] | None
            Frames acquired for this position (one 2D array per z-slice),
            or None if frame collection is not available.

        Returns
        -------
        PositionCoordinates
            Updated coordinates for this position.
        """
        return position


class PositionUpdateManager:
    """Manages async position updates after each position is acquired.

    Holds the PositionStore and a single-worker ThreadPoolExecutor (or a
    separate worker process for DynaTrack with preprocessing).
    After each position's z-stack completes, the updater is called with
    all current positions, the index of the completed position, and the
    acquired frame data. It returns updated coordinates for that position.
    """

    def __init__(
        self,
        config: PositionUpdateConfig,
        position_store: PositionStore,
        updater: PositionUpdater | None = None,
    ) -> None:
        self.config = config
        self.position_store = position_store
        self._updater = updater or PositionUpdater()
        self._executor: ThreadPoolExecutor | None = None
        self._pending_future: Future | None = None
        self._worker = None  # DynaTrackWorker for subprocess mode

    def apply_position_update(self, event: MDAEvent) -> MDAEvent:
        """Replace event's x/y/z with current values from the position store.

        When ``z_device`` is configured, the Z coordinate is written to the
        event's device properties (e.g. ``ObjectiveZ.Position``) rather than
        ``z_pos``, which typically controls the fast scanning stage (PiezoZ).
        """
        from pymmcore_plus.core._sequencing import SequencedEvent

        if isinstance(event, SequencedEvent):
            p_idx = event.events[0].index.get("p")
        else:
            p_idx = event.index.get("p")

        if p_idx is None:
            return event

        coords = self.position_store.get_position(p_idx)
        if coords is None:
            return event

        update: dict = {}
        if coords.x is not None:
            update["x_pos"] = coords.x
        if coords.y is not None:
            update["y_pos"] = coords.y

        z_device = self.config.z_device
        if coords.z is not None:
            if z_device:
                # Write Z to device property instead of z_pos
                props = list(event.properties) if event.properties else []
                # Replace existing property or append
                replaced = False
                for i, (dev, prop, _val) in enumerate(props):
                    if dev == z_device and prop == "Position":
                        props[i] = (dev, prop, coords.z)
                        replaced = True
                        break
                if not replaced:
                    props.append((z_device, "Position", coords.z))
                update["properties"] = props
            else:
                update["z_pos"] = coords.z

        if not update:
            return event

        logger.debug(
            f"Position update: overriding p={p_idx} to "
            f"x={update.get('x_pos')}, y={update.get('y_pos')}, "
            f"z={coords.z} (device={z_device or 'z_pos'})"
        )
        return event.model_copy(update=update)

    def start(self) -> None:
        """Initialize the executor. Called during setup_sequence."""
        if self.config.enabled:
            if self._worker is not None:
                # Worker process mode — start the subprocess
                self._worker.start()
                # Drain any pending results in a background thread
                self._executor = ThreadPoolExecutor(max_workers=1)
            else:
                self._executor = ThreadPoolExecutor(max_workers=1)
            self._pending_future = None

    def shutdown(self) -> None:
        """Shutdown the executor and worker process."""
        # Drain final result from worker process
        if self._worker is not None:
            if self._pending_future is not None and not self._pending_future.done():
                logger.info("Waiting for position updater to complete final update...")
                self._pending_future.result(timeout=120)
            self._worker.shutdown()
            self._worker = None
        elif self._pending_future is not None and not self._pending_future.done():
            logger.info("Waiting for position updater to complete final update...")
            self._pending_future.result(timeout=60)

        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def on_position_complete(
        self,
        timepoint_index: int,
        position_index: int,
        data: list[np.ndarray] | None = None,
    ) -> None:
        """Called when a position's z-stack has been fully acquired.

        Submits the update computation to the thread/process pool.
        """
        if not self.config.enabled or self._executor is None:
            return

        position = self.position_store.get_position(position_index)
        if position is None:
            return

        if self._worker is not None:
            # Submit and wait in background thread (serialized by single-worker executor)
            # so only one position's data is in the mp.Queue at a time
            self._pending_future = self._executor.submit(
                self._submit_and_wait_worker,
                timepoint_index,
                position_index,
                position,
                data,
            )
        else:
            self._pending_future = self._executor.submit(
                self._run_updater, timepoint_index, position_index, position, data
            )

    def _run_updater(
        self,
        timepoint_index: int,
        position_index: int,
        position: PositionCoordinates,
        data: list[np.ndarray] | None,
    ) -> None:
        """Execute the updater and write the result to the position store."""
        import time as _time

        t0 = _time.monotonic()
        n_frames = len(data) if data else 0
        logger.info(
            f"Position updater: starting p={position_index} t={timepoint_index} "
            f"({n_frames} frames)"
        )
        try:
            updated = self._updater.update(timepoint_index, position_index, position, data)
            elapsed = _time.monotonic() - t0
            self.position_store.update_position(
                position_index, updated.x, updated.y, updated.z
            )
            logger.info(
                f"Position update: p={position_index} t={timepoint_index} "
                f"-> x={updated.x:.2f}, y={updated.y:.2f}, z={updated.z} "
                f"({elapsed:.1f}s)"
            )
        except Exception:
            logger.exception(
                f"Position update failed for p={position_index} at "
                f"t={timepoint_index}, keeping previous position"
            )

    def _submit_and_wait_worker(
        self,
        timepoint_index: int,
        position_index: int,
        position: PositionCoordinates,
        data: list[np.ndarray] | None,
    ) -> None:
        """Submit data to the worker and wait for the result.

        Runs in a background thread. By serializing submit + wait, only one
        position's frame data is in the mp.Queue at a time, reducing memory.
        """
        self._worker.submit(timepoint_index, position_index, position, data)
        del data  # free main-process copy after it's been pickled to the queue
        result = self._worker.get_result(timeout=120)
        if result is None:
            logger.warning(
                f"Position update: no result from worker for p={position_index} "
                f"t={timepoint_index}"
            )
            return
        self.position_store.update_position(
            result["position_index"], result["x"], result["y"], result["z"]
        )
        logger.info(
            f"Position update (worker): p={result['position_index']} "
            f"t={result['timepoint_index']} -> x={result['x']:.2f}, "
            f"y={result['y']:.2f}, z={result['z']} ({result['elapsed']:.1f}s)"
        )
