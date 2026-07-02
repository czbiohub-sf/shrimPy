"""Engine-facing coordinator for DynaTrack position tracking.

Acquisition engines interact with DynaTrack exclusively through the
:class:`DynaTrack` class; the underlying position-update infrastructure
(PositionStore, PositionUpdater, PositionUpdateManager) is an implementation
detail of this package.
"""

from __future__ import annotations

import logging
import os

from pathlib import Path
from typing import TYPE_CHECKING

import psutil

from shrimpy.dynatrack.position_update import PositionStore, PositionUpdateManager
from shrimpy.dynatrack.tracking import DynaTrackConfig, DynaTrackUpdater

if TYPE_CHECKING:
    import numpy as np

    from useq import MDAEvent, MDASequence

    from shrimpy.dynatrack.position_update import PositionUpdater

_PROC = psutil.Process(os.getpid())


def _rss_gb() -> float:
    return _PROC.memory_info().rss / (1024**3)


logger = logging.getLogger(__name__)


class DynaTrack:
    """Coordinates DynaTrack position tracking for one acquisition.

    Owns the position store, the async update manager, per-position frame
    buffering, and (by default) the worker subprocess that runs the heavy
    tracking computation. An engine drives it through five calls:

    1. :meth:`from_metadata` in ``setup_sequence`` (returns ``None`` when
       tracking is disabled)
    2. connect :meth:`on_frame_ready` to the core's ``frameReady`` signal
    3. :meth:`start` once hardware setup has applied the ROI
    4. :meth:`apply_position_update` and :meth:`drain_pending` from the
       event iterator
    5. :meth:`shutdown` in ``teardown_sequence``

    Parameters
    ----------
    config : DynaTrackConfig
        Tracking configuration (typically parsed from acquisition metadata).
    sequence : MDASequence
        The acquisition sequence; provides the initial stage positions and
        the number of z-slices per stack.
    data_path : Path | None
        Acquisition output directory; when set (and ``config.save_debug``),
        debug output is written alongside the data.
    updater : PositionUpdater | None
        Override the tracking updater. When provided, updates run in-process
        on a background thread instead of the default DynaTrack worker
        subprocess (useful for tests and custom trackers).
    """

    def __init__(
        self,
        config: DynaTrackConfig,
        sequence: MDASequence,
        data_path: Path | None = None,
        updater: PositionUpdater | None = None,
    ) -> None:
        self.config = config
        self._store = PositionStore()
        self._store.initialize_from_sequence(sequence, z_device=config.z_device)
        self._expected_slices = max(sequence.sizes.get("z", 1), 1)
        self._frames: dict[tuple[int, int], list[np.ndarray]] = {}
        # Resolve the input channel name to its index in the sequence (used to
        # filter frames in on_frame_ready). None = buffer all channels.
        self._input_channel_index = self._resolve_input_channel(config.input_channel, sequence)

        self._debug_zarr_path: Path | None = None
        self._debug_position_names: dict[int, str] = {}
        if config.save_debug and data_path:
            self._debug_zarr_path = Path(data_path) / "dynatrack_debug.zarr"
            self._debug_position_names = {
                idx: pos.name or f"p{idx}" for idx, pos in enumerate(sequence.stage_positions)
            }

        self._use_worker = updater is None
        if updater is None:
            # Also used in worker mode: the manager consults the updater's
            # wants_reference_refresh (pure config logic) even though the
            # actual tracking runs in the worker subprocess.
            updater = DynaTrackUpdater(config=config)
        if isinstance(updater, DynaTrackUpdater):
            updater._debug_zarr_path = self._debug_zarr_path
            updater._debug_position_names = self._debug_position_names
        self._manager = PositionUpdateManager(
            self._store, updater=updater, z_device=config.z_device
        )

    @staticmethod
    def _resolve_input_channel(name: str | None, sequence: MDASequence) -> int | None:
        """Resolve an input channel name to its index in the sequence.

        Returns ``None`` when ``name`` is ``None`` (buffer all channels).
        Raises ``ValueError`` if the name is not one of the sequence's channels.
        """
        if name is None:
            return None
        channel_names = [ch.config for ch in sequence.channels]
        if name not in channel_names:
            raise ValueError(
                f"DynaTrack input_channel {name!r} is not one of the acquisition "
                f"channels {channel_names}."
            )
        return channel_names.index(name)

    @classmethod
    def from_metadata(
        cls,
        meta: dict | None,
        sequence: MDASequence,
        data_path: Path | None = None,
    ) -> DynaTrack | None:
        """Build a DynaTrack coordinator from acquisition metadata.

        Parameters
        ----------
        meta : dict | None
            The ``dynatrack`` section of the microscope metadata (e.g.
            ``sequence.metadata['mantis']['dynatrack']``), mapping directly
            onto :class:`DynaTrackConfig` fields.
        sequence : MDASequence
            The acquisition sequence.
        data_path : Path | None
            Acquisition output directory; when set, the shift log is written
            to ``<data_path>/dynatrack_log.csv`` (unless ``shift_log_path``
            is configured explicitly).

        Returns
        -------
        DynaTrack | None
            ``None`` when tracking is disabled or the sequence has no stage
            positions.
        """
        if not meta or not meta.get("enabled", False):
            return None
        if not sequence.stage_positions:
            return None
        meta = dict(meta)
        if data_path is not None:
            meta.setdefault("shift_log_path", str(Path(data_path) / "dynatrack_log.csv"))
        config = DynaTrackConfig(**meta)
        return cls(config=config, sequence=sequence, data_path=data_path)

    @property
    def position_store(self) -> PositionStore:
        return self._store

    @property
    def num_positions(self) -> int:
        return self._store.num_positions

    def start(
        self,
        zyx_shape: tuple[int, int, int] | None = None,
        log_file_path: Path | None = None,
    ) -> None:
        """Start background update processing.

        By default this spawns the DynaTrack worker subprocess, which keeps
        torch/GPU work isolated from the acquisition process. The worker
        needs the acquired frame shape, so call this after hardware setup has
        applied the ROI. When a custom ``updater`` was passed to the
        constructor, updates run in-process and both arguments are ignored.

        Parameters
        ----------
        zyx_shape : tuple[int, int, int] | None
            (Z, Y, X) shape of the acquired stacks; required in worker mode.
        log_file_path : Path | None
            Log file the worker subprocess should append to.
        """
        worker = None
        if self._use_worker:
            if zyx_shape is None:
                raise ValueError("zyx_shape is required to start the DynaTrack worker")

            from shrimpy.dynatrack.worker import DynaTrackWorker

            logger.info(f"DynaTrack: starting worker process for shape {zyx_shape}")
            if log_file_path is None:
                logger.warning(
                    "DynaTrack: no log file path provided; "
                    "worker subprocess logs will go to stderr only"
                )
            worker = DynaTrackWorker(
                config=self.config,
                zyx_shape=zyx_shape,
                debug_zarr_path=self._debug_zarr_path,
                debug_position_names=self._debug_position_names,
                log_file_path=log_file_path,
            )
        self._manager.start(worker=worker)

    def on_frame_ready(self, img: np.ndarray, event: MDAEvent) -> None:
        """Buffer frames per (timepoint, position) and flush completed stacks.

        Connect to the core's ``frameReady`` signal. Counts z-slices per
        (timepoint, position) and submits the stack for tracking as soon as
        all expected slices have been collected. Only frames from
        ``config.input_channel`` are buffered (``None`` = all channels).
        """
        channel_index = self._input_channel_index
        if channel_index is not None and event.index.get("c") != channel_index:
            return

        t_idx = event.index.get("t", 0)
        p_idx = event.index.get("p", 0)
        tp = (t_idx, p_idx)
        self._frames.setdefault(tp, []).append(img.copy())

        # Flush when all z-slices for this position have arrived
        if len(self._frames[tp]) >= self._expected_slices:
            frames = self._frames.pop(tp)
            pending_bytes = sum(
                sum(a.nbytes for a in frames_list) for frames_list in self._frames.values()
            )
            logger.debug(
                f"DynaTrack[mem]: stack complete p={p_idx} t={t_idx} "
                f"rss={_rss_gb():.2f} GB frames_buf_pending={len(self._frames)} "
                f"({pending_bytes / 1024**3:.2f} GB)"
            )
            self._manager.on_position_complete(t_idx, p_idx, frames)

    def apply_position_update(self, event: MDAEvent) -> MDAEvent:
        """Replace the event's x/y/z with current values from the position store."""
        return self._manager.apply_position_update(event)

    def drain_pending(self, timeout: float = 120) -> None:
        """Block until the in-flight position update completes."""
        self._manager.drain_pending(timeout=timeout)

    def shutdown(self) -> None:
        """Discard buffered frames and shut down the manager (and worker)."""
        self._frames = {}
        self._manager.shutdown()
