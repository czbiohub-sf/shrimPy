from __future__ import annotations

import logging
import threading

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

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


class PositionUpdateManager:
    """Manages async position updates after each position is acquired.

    Holds the PositionStore and a single-worker ThreadPoolExecutor.
    After each position's z-stack completes, the updater function is
    called with all current positions and the index of the position
    that just completed. It returns updated coordinates for that
    position only.
    """

    def __init__(
        self,
        config: PositionUpdateConfig,
        position_store: PositionStore,
        updater_fn: (
            Callable[[int, int, dict[int, PositionCoordinates]], PositionCoordinates] | None
        ) = None,
    ) -> None:
        self.config = config
        self.position_store = position_store
        self._updater_fn = updater_fn or self._default_updater
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

    def on_position_complete(self, timepoint_index: int, position_index: int) -> None:
        """Called when a position's z-stack has been fully acquired.

        Submits the update computation to the thread pool.
        """
        if not self.config.enabled or self._executor is None:
            return

        positions_snapshot = self.position_store.get_all_positions()
        self._pending_future = self._executor.submit(
            self._run_updater, timepoint_index, position_index, positions_snapshot
        )

    def _run_updater(
        self,
        timepoint_index: int,
        position_index: int,
        positions: dict[int, PositionCoordinates],
    ) -> None:
        """Execute the updater and write the result to the position store."""
        try:
            updated = self._updater_fn(timepoint_index, position_index, positions)
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

    @staticmethod
    def _default_updater(
        timepoint_index: int,
        position_index: int,
        positions: dict[int, PositionCoordinates],
    ) -> PositionCoordinates:
        """Stub updater — returns the position unchanged."""
        return positions[position_index]
