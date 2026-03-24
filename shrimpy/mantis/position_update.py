from __future__ import annotations

import logging
import threading

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from useq import MDASequence

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

    def initialize_from_sequence(self, sequence: MDASequence) -> None:
        """Populate the store from the sequence's stage_positions."""
        with self._lock:
            self._positions.clear()
            for idx, pos in enumerate(sequence.stage_positions):
                self._positions[idx] = PositionCoordinates(
                    x=pos.x if pos.x is not None else 0.0,
                    y=pos.y if pos.y is not None else 0.0,
                    z=pos.z,
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

    Holds the PositionStore and a single-worker ThreadPoolExecutor.
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

    def start(self) -> None:
        """Initialize the executor. Called during setup_sequence."""
        if self.config.enabled:
            self._executor = ThreadPoolExecutor(max_workers=1)
            self._pending_future = None

    def shutdown(self) -> None:
        """Shutdown the executor. Called during teardown_sequence."""
        if self._executor is not None:
            if self._pending_future is not None and not self._pending_future.done():
                logger.info("Waiting for position updater to complete final update...")
                self._pending_future.result(timeout=60)
            self._executor.shutdown(wait=True)
            self._executor = None

    def on_position_complete(
        self,
        timepoint_index: int,
        position_index: int,
        data: list[np.ndarray] | None = None,
    ) -> None:
        """Called when a position's z-stack has been fully acquired.

        Submits the update computation to the thread pool.
        """
        if not self.config.enabled or self._executor is None:
            return

        position = self.position_store.get_position(position_index)
        if position is None:
            return

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
        try:
            updated = self._updater.update(timepoint_index, position_index, position, data)
            self.position_store.update_position(
                position_index, updated.x, updated.y, updated.z
            )
            logger.info(
                f"Position update: p={position_index} at t={timepoint_index} "
                f"-> x={updated.x:.2f}, y={updated.y:.2f}, z={updated.z}"
            )
        except Exception:
            logger.exception(
                f"Position update failed for p={position_index} at "
                f"t={timepoint_index}, keeping previous position"
            )
