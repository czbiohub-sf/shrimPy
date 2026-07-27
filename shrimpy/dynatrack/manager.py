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
        Acquisition output directory; when set (and ``config.debug``),
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
        pixel_size_um: float | None = None,
        z_step_um: float | None = None,
    ) -> None:
        self.config = config
        self._store = PositionStore()
        self._store.initialize_from_sequence(sequence, z_device=config.z_device)
        self._expected_slices = max(sequence.sizes.get("z", 1), 1)
        self._frames: dict[tuple[int, int], list[np.ndarray]] = {}
        # XY pixel size and Z step (microns) used to convert pixel shifts to
        # microns; derived from the core / z_plan by from_metadata.
        self._pixel_size_um = pixel_size_um
        self._z_step_um = z_step_um
        # Resolve the input channel name to its index in the sequence (used to
        # filter frames in on_frame_ready).
        self._input_channel_index = self._resolve_input_channel(config.input_channel, sequence)
        self._validate_tracking_channel(config, sequence)

        self._debug_zarr_path: Path | None = None
        self._debug_position_names: dict[int, str] = {}
        if config.debug and data_path:
            self._debug_zarr_path = Path(data_path) / "dynatrack_debug.zarr"
            self._debug_position_names = {
                idx: pos.name or f"p{idx}" for idx, pos in enumerate(sequence.stage_positions)
            }

        self._use_worker = updater is None
        if updater is None:
            # Consultation-only in worker mode: the manager reads this updater's
            # wants_reference_refresh (pure config logic); the actual tracking
            # (with the real scales) runs in the worker subprocess, so scale is
            # not needed here.
            updater = DynaTrackUpdater(config=config)
        if isinstance(updater, DynaTrackUpdater):
            updater._debug_zarr_path = self._debug_zarr_path
            updater._debug_position_names = self._debug_position_names
        self._manager = PositionUpdateManager(
            self._store, updater=updater, z_device=config.z_device
        )

    @staticmethod
    def _validate_tracking_channel(config: DynaTrackConfig, sequence: MDASequence) -> None:
        """Validate ``tracking_channel`` against the preprocessing pipeline.

        Without VS, it must name one of the acquisition input channels (tracking
        runs on that channel's raw/deskewed/phase volume). With VS, it must be
        one of ``virtual_staining.target_channels``. The reserved names
        ``"phase"``, ``"deskewed"``, and any ``"vs_*"`` name are rejected as
        ambiguous / bug-prone.
        """
        tc = config.tracking_channel
        preprocessing = config.preprocessing or []
        uses_vs = "vs" in preprocessing

        if tc == "phase" or tc == "deskewed" or tc.startswith("vs_"):
            raise ValueError(
                f"tracking_channel={tc!r} is not allowed. Use an input channel "
                "name (raw/deskew/phase pipelines) or a "
                "virtual_staining.target_channels name (VS pipeline); 'phase', "
                "'deskewed', and 'vs_*' are reserved."
            )

        if uses_vs:
            targets = (config.virtual_staining or {}).get("target_channels") or [
                "nuclei",
                "membrane",
            ]
            if tc not in targets:
                raise ValueError(
                    f"tracking_channel={tc!r} must be one of "
                    f"virtual_staining.target_channels {targets} when preprocessing "
                    "includes 'vs'."
                )
        else:
            input_channels = [ch.config for ch in sequence.channels]
            if tc not in input_channels:
                raise ValueError(
                    f"tracking_channel={tc!r} must be one of the acquisition channels "
                    f"{input_channels} when not using VS preprocessing."
                )

    @staticmethod
    def _resolve_input_channel(name: str, sequence: MDASequence) -> int:
        """Resolve the input channel name to its index in the sequence.

        Raises ``ValueError`` if the name is not one of the sequence's channels.
        """
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
        pixel_size_um: float | None = None,
    ) -> DynaTrack | None:
        """Build a DynaTrack coordinator from acquisition metadata.

        The XY pixel size and Z step are the single source of truth for all
        scale parameters: ``pixel_size_um`` (from ``core.getPixelSizeUm()``)
        and the sequence's ``z_plan.step``. They are injected into the
        ``deskew`` (``pixel_size_um`` / ``scan_step_um``) and ``phase``
        (``transfer_function.yx_pixel_size`` / ``z_pixel_size``) sub-configs
        and used to convert pixel shifts to microns, so those values are never
        specified in the config (avoiding drift).

        Parameters
        ----------
        meta : dict | None
            The ``dynatrack`` section of the microscope metadata (e.g.
            ``sequence.metadata['mantis']['dynatrack']``), mapping directly
            onto :class:`DynaTrackConfig` fields.
        sequence : MDASequence
            The acquisition sequence; ``z_plan.step`` provides the Z step.
        data_path : Path | None
            Acquisition output directory; when set, the shift log is written
            to ``<data_path>/dynatrack_log.csv`` (unless ``shift_log_path``
            is configured explicitly).
        pixel_size_um : float | None
            XY pixel size in microns, from ``core.getPixelSizeUm()``.

        Returns
        -------
        DynaTrack | None
            ``None`` when tracking is disabled or the sequence has no stage
            positions.

        Raises
        ------
        ValueError
            If ``pixel_size_um`` is unset/zero (pixel size not calibrated) or
            the sequence's z_plan has no step.
        """
        if not meta or not meta.get("enabled", False):
            return None
        if not sequence.stage_positions:
            return None
        if not pixel_size_um:
            raise ValueError(
                "DynaTrack: pixel size is not set (core.getPixelSizeUm() returned "
                "0 or None); calibrate the pixel size in Micro-Manager."
            )
        z_step_um = getattr(sequence.z_plan, "step", None) if sequence.z_plan else None
        if not z_step_um:
            raise ValueError(
                "DynaTrack: the sequence z_plan has no step; a stepped z_plan is "
                "required to derive the Z scale."
            )
        meta = cls._inject_scales(meta, pixel_size_um, z_step_um)
        if data_path is not None:
            meta.setdefault("shift_log_path", str(Path(data_path) / "dynatrack_log.csv"))
        config = DynaTrackConfig(**meta)
        return cls(
            config=config,
            sequence=sequence,
            data_path=data_path,
            pixel_size_um=pixel_size_um,
            z_step_um=z_step_um,
        )

    @staticmethod
    def _inject_scales(meta: dict, pixel_size_um: float, z_step_um: float) -> dict:
        """Return a copy of ``meta`` with the pixel size / z step injected.

        Feeds the ``deskew`` and ``phase`` sub-configs their pixel/step
        parameters from the single source of truth, so they are not specified
        (and cannot drift) in the config.
        """
        import copy

        meta = copy.deepcopy(meta)
        deskew = meta.get("deskew")
        if deskew is not None:
            deskew["pixel_size_um"] = pixel_size_um
            deskew["scan_step_um"] = z_step_um
        phase = meta.get("phase")
        if phase is not None:
            tf = phase.setdefault("transfer_function", {})
            tf["yx_pixel_size"] = pixel_size_um
            tf["z_pixel_size"] = z_step_um
        return meta

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
                scale_yx=self._pixel_size_um,
                scale_z=self._z_step_um,
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
        ``config.input_channel`` are buffered.
        """
        if event.index.get("c") != self._input_channel_index:
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
