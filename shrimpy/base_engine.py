"""Shared MDA engine for shrimPy microscopes.

:class:`BaseEngine` collects the acquisition behavior that is common to every
microscope shrimPy drives (mantis, iSIM, Dragonfly, ...):

- hardware-sequenced acquisition defaults (``use_hardware_sequencing``,
  ``force_set_xy_position``),
- verbose hardware logging (property changes, ROI changes, XY stage moves),
- continuous-autofocus handling, including the simulated ``demo-PFS`` method
  used with the Micro-Manager demo config, and skipping events whose autofocus
  did not engage,
- resetting hardware properties in ``teardown_sequence``,
- the :meth:`BaseEngine.acquire` entry point that runs an ``MDASequence`` and
  writes OME-Zarr.

Microscope-specific engines subclass it and override the pieces that differ:

- :meth:`BaseEngine.engage_autofocus` — the hardware autofocus routine. The
  base implementation raises ``NotImplementedError``; it is only reached when
  autofocus is enabled with a method other than ``demo-PFS``.
- ``setup_sequence`` / ``setup_event`` / ``teardown_sequence`` — call
  ``super()`` and add microscope-specific hardware setup around it.

See :mod:`shrimpy.mantis.mantis_engine` for the reference implementation.
"""

from __future__ import annotations

import json
import logging
import os

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import psutil

from ome_writers import AcquisitionSettings
from pymmcore_plus.core import CMMCorePlus
from pymmcore_plus.core._constants import Keyword
from pymmcore_plus.core._sequencing import SequencedEvent
from pymmcore_plus.mda import MDAEngine, SkipEvent
from pymmcore_plus.metadata import SummaryMetaV1
from pymmcore_plus.metadata.serialize import to_builtins
from useq import MDAEvent, MDASequence

from shrimpy.config import ShrimpyMetadata, load_config

logger = logging.getLogger(__name__)

DEMO_PFS_METHOD = "demo-PFS"
DEMO_PFS_SUCCESS_RATE = 0.5  # probability that a demo-PFS call succeeds

_PROC = psutil.Process(os.getpid())


def _rss_gb() -> float:
    return _PROC.memory_info().rss / (1024**3)


class BaseEngine(MDAEngine):
    """Base MDA engine shared by all shrimPy microscopes.

    Parameters
    ----------
    mmc : CMMCorePlus
        The Micro-Manager core instance. The engine registers itself with
        ``mmc.mda`` and connects to the core's property / ROI / stage signals
        for logging.
    *args, **kwargs
        Forwarded to :class:`~pymmcore_plus.mda.MDAEngine`. shrimPy defaults
        ``use_hardware_sequencing`` to True and ``force_set_xy_position`` to
        False; subclasses may set microscope-specific defaults (e.g. acquisition
        timeouts) before calling ``super().__init__()``.
    """

    def __init__(self, mmc: CMMCorePlus, *args, **kwargs):
        kwargs.setdefault("use_hardware_sequencing", True)
        kwargs.setdefault("force_set_xy_position", False)
        super().__init__(mmc, *args, **kwargs)
        self._use_autofocus = False
        self._autofocus_success = False
        self._autofocus_stage = None
        self._autofocus_method = None
        self._autofocus_fail_at_index = None
        self._xy_stage_device = None
        self._data_path: Path | None = None

        # Register event callbacks for logging
        mmc.mda.set_engine(self)
        mmc.events.propertyChanged.connect(self._on_property_changed)
        mmc.events.roiSet.connect(self._on_roi_set)
        mmc.events.XYStagePositionChanged.connect(self._on_xy_stage_position_changed)

    # ------------------------------------------------------------------
    # Logging callbacks
    # ------------------------------------------------------------------

    def _on_property_changed(self, device: str, property_name: str, value: str) -> None:
        """Log property changes at debug level."""
        # Ignore select property changes
        if property_name in ("PFS Status", "PFS in Range", "FocusMaintenance"):
            return
        logger.debug(f"Property changed: {device}.{property_name} = {value}")

    def _on_roi_set(self, camera: str, x: int, y: int, width: int, height: int) -> None:
        """Log ROI changes at debug level."""
        logger.debug(
            f"Setting ROI on {camera} to x={x}, y={y}, width={width}, height={height}"
        )

    def _on_xy_stage_position_changed(self, device: str, x: float, y: float) -> None:
        """Log stage position changes at debug level."""
        logger.debug(f"XY stage position changed: device={device}, x={x:.2f}, y={y:.2f}")

    # ------------------------------------------------------------------
    # MDAEngine protocol
    # ------------------------------------------------------------------

    def setup_sequence(self, sequence: MDASequence) -> SummaryMetaV1 | None:
        """Configure shared hardware settings before the sequence starts.

        The microscope settings are read from ``sequence.metadata`` and
        validated by :class:`~shrimpy.config.ShrimpyMetadata`; missing sections
        fall back to their defaults (autofocus disabled).
        """
        logger.info("Setting up hardware for acquisition sequence")

        core = self.mmcore
        meta = ShrimpyMetadata.from_sequence(sequence)

        # Set autofocus settings
        autofocus = meta.autofocus
        if autofocus.enabled:
            self._use_autofocus = True
            self._autofocus_stage = autofocus.stage
            self._autofocus_method = autofocus.method
            logger.info(f"Enabling autofocus with method: {self._autofocus_method}")
            if not self._autofocus_method == DEMO_PFS_METHOD:
                core.setAutoFocusDevice(self._autofocus_method)
        else:
            logger.info("Autofocus is disabled for this acquisition")

        # Store XY stage device name
        self._xy_stage_device = core.getXYStageDevice()
        logger.debug(f"XY stage device: {self._xy_stage_device}")

        # Call parent setup so SummaryMetaV1 captures the fully configured
        # hardware state and the setup event applies the ROI.
        return super().setup_sequence(sequence)

    def setup_event(self, event: MDAEvent) -> None:
        """Move to the event position, engage autofocus, and prepare hardware."""
        # Set XY stage position and engage autofocus
        # Note: this command will not move the stage if the target position is the same
        # as the last commanded position and force_set_xy_position is False.
        self._set_event_xy_position(event)
        # _set_event_xy_position does not wait for the stage to reach the target position
        if self._xy_stage_device:
            self.mmcore.waitForDevice(self._xy_stage_device)

        # Engage autofocus
        self._engage_autofocus(event)

        # Skip acquisition if autofocus failed
        if self._use_autofocus and not self._autofocus_success:
            num_frames = len(event.events) if isinstance(event, SequencedEvent) else 1
            raise SkipEvent(num_frames=num_frames, reason="autofocus failed")

        self._log_memory_usage()

        # Call parent setup_event
        super().setup_event(event)

    def teardown_sequence(self, sequence: MDASequence) -> None:
        """Return the hardware to a safe idle state after the sequence."""
        super().teardown_sequence(sequence)

        core = self.mmcore
        meta = ShrimpyMetadata.from_sequence(sequence)

        if reset_hardware_sequencing_settings := meta.reset_hardware_sequencing_settings:
            logger.info(
                f"Resetting {len(reset_hardware_sequencing_settings)} hardware sequencing settings"
            )
            for setting in reset_hardware_sequencing_settings:
                logger.debug(f"  Setting {setting[0]}.{setting[1]} = {setting[2]}")
                core.setProperty(setting[0], setting[1], setting[2])
        else:
            logger.debug("No reset hardware sequencing settings specified")

    def _set_event_properties(self, properties: Iterable[tuple]) -> None:
        """Set properties for the current event."""
        for device, prop, value in properties:
            if (
                prop == Keyword.Position
                and device == self._autofocus_stage
                and self._use_autofocus
            ):
                # Skip setting Z position if autofocus is enabled to avoid
                # disengaging autofocus lock; autofocus algorithm will set Z
                # position independently
                logger.debug(
                    "Skipping Z set on autofocus stage: %s.%s = %s", device, prop, value
                )
                continue
            super()._set_event_properties([(device, prop, value)])

    def _log_memory_usage(self) -> None:
        """Log process memory and circular buffer occupancy at debug level."""
        free_capacity = self.mmcore.getBufferFreeCapacity()
        total_capacity = self.mmcore.getBufferTotalCapacity()
        logger.debug(f"Circular buffer capacity: {free_capacity} / {total_capacity} frames")
        logger.debug(
            f"{type(self).__name__}[mem]: setup_event rss={_rss_gb():.2f} GB "
            f"mm_buf_used={total_capacity - free_capacity}/{total_capacity}"
        )

    # ------------------------------------------------------------------
    # Autofocus
    # ------------------------------------------------------------------

    def _engage_autofocus(self, event: MDAEvent) -> None:
        """Engage autofocus for ``event``, recording the outcome.

        Dispatches to the simulated :meth:`_engage_demo_pfs` when the
        configured method is ``demo-PFS``, and to the microscope-specific
        :meth:`engage_autofocus` otherwise. The outcome is stored in
        ``self._autofocus_success``; :meth:`setup_event` skips the event when
        autofocus is enabled but did not engage.
        """
        if not self._use_autofocus:
            logger.debug("Autofocus is disabled.")
            return

        if self._autofocus_method == DEMO_PFS_METHOD:
            self._autofocus_success = self._engage_demo_pfs(
                event=event,
                fail_at_index=self._autofocus_fail_at_index,
            )
            return

        self._autofocus_success = bool(self.engage_autofocus(event))

    def engage_autofocus(self, event: MDAEvent) -> bool:
        """Engage the microscope's hardware autofocus for ``event``.

        Subclasses must implement this method; the acquisition of any event for
        which it returns False is skipped (see :meth:`setup_event`). It is only
        called when autofocus is enabled with a method other than ``demo-PFS``.

        Parameters
        ----------
        event : MDAEvent
            The event that is about to be acquired.

        Returns
        -------
        bool
            True if autofocus engaged successfully.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement engage_autofocus(); "
            f"autofocus method {self._autofocus_method!r} is not supported. "
            "Override engage_autofocus() in the microscope engine, disable "
            f"autofocus, or use the {DEMO_PFS_METHOD!r} method."
        )

    def _engage_demo_pfs(
        self,
        event: MDAEvent | None = None,
        success_rate: float = DEMO_PFS_SUCCESS_RATE,
        fail_at_index: list[dict] | None = None,
    ) -> bool:
        """Engage demo PFS continuous autofocus.

        If ``fail_at_index`` is provided, autofocus deterministically fails
        when the event index matches any entry in the list. Otherwise, success
        is random based on ``success_rate``.

        Parameters
        ----------
        event : MDAEvent | None
            The current MDA event (used for deterministic failure matching).
        success_rate : float
            The probability of success for the demo PFS call. Only used when
            ``fail_at_index`` is not provided.
        fail_at_index : list[dict] | None
            List of index dicts to fail at, e.g. ``[{"p": 0}, {"t": 1, "p": 2}]``.
            Each dict is matched against the event index — if all keys in the
            dict match the event index, autofocus fails at that event.

        Returns
        -------
        bool
            True if the simulated autofocus call succeeded.
        """
        if fail_at_index is not None and event is not None:
            # For SequencedEvents, use the first sub-event's index
            event_index = (
                event.events[0].index if isinstance(event, SequencedEvent) else event.index
            )
            success = not any(
                all(event_index.get(k) == v for k, v in idx.items()) for idx in fail_at_index
            )
        else:
            success = np.random.random() < success_rate

        if success:
            logger.debug(f"{DEMO_PFS_METHOD} call succeeded")
        else:
            logger.debug(f"{DEMO_PFS_METHOD} call failed")

        return success

    def _get_autofocus_z_position(self, event: MDAEvent) -> float:
        """Return the target Z position of the autofocus stage for ``event``.

        Z positions are not written to the autofocus stage while autofocus is
        enabled (see :meth:`_set_event_properties`), so the target position is
        read from the event's properties when present, and from the stage's
        current position otherwise.
        """
        if event.properties:
            for dev, prop, value in event.properties:
                if dev == self._autofocus_stage and prop == "Position":
                    return value
        return self.mmcore.getPosition(self._autofocus_stage)

    # ------------------------------------------------------------------
    # Acquisition entry point
    # ------------------------------------------------------------------

    def acquire(
        self,
        output_dir: str | Path,
        name: str,
        mda_config: MDASequence | str | Path,
    ) -> None:
        """Run an acquisition and write the data as OME-Zarr.

        Parameters
        ----------
        output_dir : str | Path
            Directory where acquisition data will be saved.
        name : str
            Base acquisition name; an index suffix will be appended automatically.
        mda_config : MDASequence | str | Path
            An MDASequence object or path to an acquisition configuration YAML
            file (an MDASequence with the microscope settings under
            ``metadata``; see :mod:`shrimpy.config`).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        name = _get_next_acquisition_name(output_dir, name)

        if isinstance(mda_config, MDASequence):
            sequence = mda_config
        else:
            logger.info(f"Loading acquisition config from {mda_config}")
            # Validates the shrimPy metadata sections before any hardware setup
            sequence = load_config(mda_config)

        data_path = output_dir / f"{name}.ome.zarr"
        self._data_path = data_path

        # Write summary metadata after the zarr store is created
        # TODO: remove once ome-writers supports root-level metadata natively
        def _write_summary_metadata(_seq: MDASequence, meta: object) -> None:
            self.mmcore.mda.events.sequenceStarted.disconnect(_write_summary_metadata)
            if meta and isinstance(meta, dict):
                meta_path = data_path / "summary_metadata.json"
                meta_path.write_text(json.dumps(to_builtins(meta)))

        self.mmcore.mda.events.sequenceStarted.connect(_write_summary_metadata)

        logger.info(f"Starting acquisition: {name}")
        self.mmcore.mda.run(
            sequence,
            output=AcquisitionSettings(
                root_path=data_path, compression="blosc-zstd", format="acquire-zarr"
            ),
            dimension_overrides={"z": {"chunk_size": min(512, sequence.sizes["z"])}},
            overwrite=False,
        )
        logger.info("Acquisition completed successfully")


def _get_next_acquisition_name(output_dir: Path, name: str) -> str:
    """Get next available acquisition name with incremented index.

    Parameters
    ----------
    output_dir : Path
        Output directory where acquisitions are saved.
    name : str
        Base acquisition name.

    Returns
    -------
    str
        Acquisition name with index (e.g., "acq_1", "acq_2", etc.).
    """
    idx = 1
    while True:
        indexed_name = f"{name}_{idx}"
        data_path = output_dir / f"{indexed_name}.ome.zarr"
        if not data_path.exists():
            return indexed_name
        idx += 1
