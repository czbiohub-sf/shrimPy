from __future__ import annotations

import json
import logging
import time

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from ome_writers import (
    AcquisitionSettings,
)
from pymmcore_plus.core import CMMCorePlus
from pymmcore_plus.core._constants import Keyword
from pymmcore_plus.core._sequencing import SequencedEvent
from pymmcore_plus.mda import MDAEngine, SkipEvent
from pymmcore_plus.metadata import SummaryMetaV1
from pymmcore_plus.metadata.serialize import to_builtins
from useq import MDAEvent, MDASequence

from shrimpy.mantis.autoexposure import load_manual_illumination_settings
from shrimpy.mantis.lasers.vortran import VortranLaser, setup_vortran_laser

# Get the logger instance (will be configured by the CLI entry point)
logger = logging.getLogger(__name__)

MANTIS_XY_STAGE_NAME = "XYStage:XY:31"
DEMO_PFS_METHOD = "demo-PFS"
SLOW_XY_STAGE_SPEED = 2.0  # in mm/s, used for short moves to maintain autofocus lock
FAST_XY_STAGE_SPEED = 5.75  # in mm/s, used for long moves
NEGLIGIBLE_XY_DISTANCE = 1  # in um, moves below this are ignored
SHORT_XY_DISTANCE = 2000  # in um, threshold between slow and fast speed


class MantisEngine(MDAEngine):
    """Custom MDA engine for the Mantis microscope.

    This engine extends the default MDAEngine to handle mantis-specific
    hardware setup and configuration, including:
    - TriggerScope sequencing configuration
    - ROI setup
    - Axial Piezo (AP Galvo) focus control
    - TTL blanking
    - Autofocus after XY stage movements
    """

    def __init__(self, mmc: CMMCorePlus, *args, **kwargs):
        """Initialize and register the MantisEngine with the core.

        Parameters
        ----------
        mmc : CMMCorePlus
            The Micro-Manager core instance
        """
        kwargs.setdefault("use_hardware_sequencing", True)
        kwargs.setdefault("force_set_xy_position", False)
        # Set acquisition timeout to guard against stalling due to dropped frames
        # or missed trigger pulses
        kwargs.setdefault("timeout_base", 2.0)
        kwargs.setdefault("timeout_multiplier", 1.0)
        kwargs.setdefault("timeout_first_frame", None)
        kwargs.setdefault("timeout_action", "warn")
        super().__init__(mmc, *args, **kwargs)
        self._use_autofocus = False
        self._autofocus_success = False
        self._autofocus_stage = None
        self._autofocus_method = None
        self._autofocus_fail_at_index = None
        self._xy_stage_device = None
        self._xy_stage_speed = None
        self._autoexposure_method: str | None = None
        self._illumination_settings = None
        self._lasers: dict[str, VortranLaser] = {}
        self._current_well_id: str | None = None
        self._last_applied_key: tuple[str, str] | None = None

        # Register event callbacks for logging
        mmc.mda.set_engine(self)
        mmc.events.propertyChanged.connect(self._on_property_changed)
        mmc.events.roiSet.connect(self._on_roi_set)
        mmc.events.XYStagePositionChanged.connect(self._on_xy_stage_position_changed)

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

    def _disconnect_lasers(self) -> None:
        """Close serial ports for any lasers opened during ``_setup_manual_exposure``."""
        for name, laser in self._lasers.items():
            try:
                laser.disconnect()
            except Exception as e:
                logger.warning(f"Failed to disconnect laser '{name}': {e}")
        self._lasers = {}

    def _setup_autoexposure(self, sequence: MDASequence, microscope_meta: dict) -> None:
        """Dispatch autoexposure setup based on ``autoexposure.method``.

        Reads ``microscope_meta["autoexposure"]``. The block is opt-in via an
        ``enabled`` flag (mirroring the ``autofocus`` config). When enabled,
        ``method`` selects the strategy:

        - ``"manual"`` → :meth:`_setup_manual_exposure` (per-well CSV)
        - anything else → ``NotImplementedError``
        """
        # Clear any state left over from a prior sequence
        self._autoexposure_method = None
        self._illumination_settings = None
        self._current_well_id = None
        self._last_applied_key = None
        self._disconnect_lasers()

        cfg = microscope_meta.get("autoexposure")
        if not cfg or not cfg.get("enabled"):
            logger.info("Autoexposure is disabled for this acquisition")
            return

        method = cfg.get("method")
        logger.info(f"Enabling autoexposure with method: {method}")
        if method == "manual":
            self._setup_manual_exposure(sequence, cfg)
        else:
            raise NotImplementedError(f"Autoexposure method '{method}' is not implemented.")
        self._autoexposure_method = method

    def _setup_manual_exposure(self, sequence: MDASequence, cfg: dict) -> None:
        """Load the illumination CSV and validate it against the sequence.

        Reads from the ``autoexposure`` config block:

        - ``csv_path`` (str): path to ``illumination.csv``. The CSV must have
          columns ``well_id``, ``exposure_time_ms``, ``laser_power_mW``.
        - ``lasers`` (dict, optional): mapping of laser name → COM port
          (e.g. ``{"488": "COM6", "561": "COM13"}``). Each laser is opened
          via :func:`setup_vortran_laser` and the per-well ``laser_power_mW``
          is written to every configured laser on well transitions.
        """
        csv_path = cfg.get("csv_path")
        if not csv_path:
            raise ValueError("autoexposure.method='manual' requires 'csv_path' to be set.")
        csv_path = Path(csv_path)
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Manual autoexposure illumination CSV not found: {csv_path}"
            )

        illumination = load_manual_illumination_settings(csv_path)

        # Validate that every well referenced by the sequence appears in the CSV.
        sequence_wells = {parse_well_id(p.name) for p in (sequence.stage_positions or [])}
        csv_wells = set(illumination.index.get_level_values("well_id"))
        missing = sequence_wells - csv_wells
        if missing:
            raise ValueError(
                f"illumination.csv is missing well_ids present in the sequence: "
                f"{sorted(missing)}"
            )

        # Validate that every channel referenced by the CSV is in the sequence.
        sequence_channels = {c.config for c in (sequence.channels or [])}
        csv_channels = set(illumination.index.get_level_values("channel_name"))
        unknown = csv_channels - sequence_channels
        if unknown:
            raise ValueError(
                f"illumination.csv references channels not present in the "
                f"MDASequence: {sorted(unknown)} (sequence channels: "
                f"{sorted(sequence_channels)})"
            )

        lasers_cfg = cfg.get("lasers") or {}
        if not isinstance(lasers_cfg, dict):
            raise ValueError("autoexposure.lasers must be a mapping of laser name → COM port.")
        for name, com_port in lasers_cfg.items():
            logger.info(f"Connecting to Vortran laser '{name}' on {com_port}")
            self._lasers[str(name)] = setup_vortran_laser(str(com_port))

        self._illumination_settings = illumination
        logger.info(
            f"Manual autoexposure enabled with {len(illumination)} wells from {csv_path}"
        )

    def _lookup_illumination_row(self, event: MDAEvent):
        """Return ``(well_id, channel, row)`` for ``event``, or ``None``.

        Used by both ``event_iterator`` (to rewrite ``exposure``) and
        ``setup_event`` (to set laser ``pulse_power``). Returns ``None`` when
        autoexposure is disabled, the event has no channel, or no row matches
        the event's ``(well_id, channel)`` pair.
        """
        if self._autoexposure_method != "manual" or self._illumination_settings is None:
            return None

        # SequencedEvents carry pos_name/channel on the first sub-event
        pos_name = event.pos_name
        channel_obj = event.channel
        if isinstance(event, SequencedEvent) and event.events:
            if pos_name is None:
                pos_name = event.events[0].pos_name
            if channel_obj is None:
                channel_obj = event.events[0].channel
        well_id = parse_well_id(pos_name)
        channel = channel_obj.config if channel_obj else None
        if channel is None:
            return None

        try:
            row = self._illumination_settings.loc[(well_id, channel)]
        except KeyError:
            return None
        return well_id, channel, row

    def event_iterator(self, events: Iterable[MDAEvent]) -> Iterator[MDAEvent]:
        """Apply per-(well, channel) exposure overrides, then hand off to the
        parent's hardware-sequencing pass.

        Doing the override here (instead of in ``setup_event``) means the
        rewritten exposure is what the ``MDARunner`` logs, and the
        sequencer downstream sees authoritative exposures when deciding
        which events can be combined into a :class:`SequencedEvent`.
        """
        rewritten = self._override_exposures(events)
        yield from super().event_iterator(rewritten)

    def _override_exposures(self, events: Iterable[MDAEvent]) -> Iterator[MDAEvent]:
        """Yield each event with its exposure rewritten from the CSV.

        Events with no matching ``(well_id, channel)`` row pass through
        unchanged.
        """
        for event in events:
            match = self._lookup_illumination_row(event)
            if match is None:
                yield event
                continue
            exposure_ms = float(match[2]["exposure_time_ms"])
            if event.exposure != exposure_ms:
                yield event.model_copy(update={"exposure": exposure_ms})
            else:
                yield event

    def _apply_autoexposure(self, event: MDAEvent) -> None:
        """Dispatch per-event autoexposure based on the configured method.

        Exposure overrides are handled in :meth:`event_iterator`; this hook
        is for hardware that needs to be commanded at acquisition time, such
        as laser power on a COM-port-controlled laser.
        """
        if self._autoexposure_method is None:
            return
        if self._autoexposure_method == "manual":
            self._apply_manual_autoexposure(event)
            return
        raise NotImplementedError(
            f"Autoexposure method '{self._autoexposure_method}' is not implemented."
        )

    def _apply_manual_autoexposure(self, event: MDAEvent) -> None:
        """Apply per-(well, channel) laser power on transitions.

        Exposure-time override is handled upstream in
        :meth:`_override_exposures`; this method only writes
        ``pulse_power`` to the laser named by the matching CSV row.
        Throttled to actual ``(well, channel)`` transitions to avoid
        redundant serial traffic.
        """
        match = self._lookup_illumination_row(event)
        if match is None:
            # Track well transitions for the "no settings for well" warning
            pos_name = event.pos_name
            if isinstance(event, SequencedEvent) and event.events and pos_name is None:
                pos_name = event.events[0].pos_name
            well_id = parse_well_id(pos_name)
            if self._illumination_settings is not None and well_id not in (
                self._illumination_settings.index.get_level_values("well_id")
            ):
                if well_id != self._current_well_id:
                    logger.warning(
                        f"No illumination settings for well '{well_id}'; "
                        "leaving laser power unchanged."
                    )
                    self._current_well_id = well_id
            return

        well_id, channel, row = match
        self._current_well_id = well_id
        laser_name = str(row["laser_name"]) if row["laser_name"] else ""
        laser_power = float(row["laser_power_mW"])

        key = (well_id, channel)
        if key == self._last_applied_key:
            return
        logger.info(
            f"Applying manual autoexposure for well '{well_id}', channel "
            f"'{channel}': laser '{laser_name}' power={laser_power} mW"
        )
        if laser_name:
            laser = self._lasers.get(laser_name)
            if laser is not None:
                laser.pulse_power = laser_power
            else:
                logger.warning(
                    f"Laser '{laser_name}' referenced by illumination CSV is "
                    "not configured under autoexposure.lasers; skipping power set."
                )
        self._last_applied_key = key

    def setup_sequence(self, sequence: MDASequence) -> SummaryMetaV1 | None:
        """Setup mantis-specific hardware before the sequence starts.

        Reads mantis-specific settings from sequence.metadata['mantis'] if present,
        otherwise uses default values.
        """
        logger.info("Setting up Mantis-specific hardware for acquisition sequence")

        core = self.mmcore

        # Extract mantis settings from metadata
        microscope_meta = sequence.metadata.get("mantis", {}) if sequence.metadata else {}

        # Set autofocus settings
        if autofocus := microscope_meta.get("autofocus"):
            if autofocus.get("enabled"):
                self._use_autofocus = True
                self._autofocus_stage = autofocus.get("stage")
                self._autofocus_method = autofocus.get("method")
                logger.info(f"Enabling autofocus with method: {self._autofocus_method}")
                if not self._autofocus_method == DEMO_PFS_METHOD:
                    core.setAutoFocusDevice(self._autofocus_method)
            else:
                logger.info("Autofocus is disabled for this acquisition")

        # Store XY stage device name
        self._xy_stage_device = core.getXYStageDevice()
        logger.debug(f"XY stage device: {self._xy_stage_device}")

        # Set up autoexposure (per-well exposure time and laser power)
        self._setup_autoexposure(sequence, microscope_meta)

        logger.info("Mantis hardware setup completed successfully")

        # Call parent setup last so SummaryMetaV1 captures the fully
        # configured hardware state (ROI, focus device, etc.).
        return super().setup_sequence(sequence)

    def setup_event(self, event: MDAEvent) -> None:
        """Prepare mantis hardware for each event."""
        # Per-event laser power (per (well, channel)). Exposure-time
        # overrides are applied upstream in event_iterator so they show
        # up in the MDARunner logs and inform hardware-sequencing.
        self._apply_autoexposure(event)

        # Set XY stage position and engage autofocus
        # Note: this command will not move the stage if the target position is the same
        # as the last commanded position and force_set_xy_position is False.
        # TODO: debug resetting xy stage speed
        # self._adjust_xy_stage_speed(event)
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

        # Call parent setup_event
        super().setup_event(event)

    def teardown_sequence(self, sequence):
        super().teardown_sequence(sequence)

        # Close any Vortran laser COM ports opened for autoexposure
        self._disconnect_lasers()

        core = self.mmcore
        microscope_meta = sequence.metadata.get("mantis", {}) if sequence.metadata else {}

        if reset_hardware_sequencing_settings := microscope_meta.get(
            "reset_hardware_sequencing_settings"
        ):
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
                continue
            super()._set_event_properties([(device, prop, value)])

    def _adjust_xy_stage_speed(self, event: MDAEvent) -> None:
        """Modulate XY stage speed based on distance to target position.

        This method adjusts the XY stage speed before moving to a new position,
        using a slower speed for short moves to help maintain autofocus lock.

        Parameters
        ----------
        event : MDAEvent
            The MDA event containing the target XY position.
        """
        if not self._use_autofocus or not self._xy_stage_device:
            return

        # Only adjust speed for Mantis XY stage, not demo XY stage
        if self._xy_stage_device != MANTIS_XY_STAGE_NAME:
            return

        last_x, last_y = self.mmcore._last_xy_position.get(None) or (None, None)
        target_x, target_y = event.x_pos, event.y_pos

        if not all(v is not None for v in [last_x, last_y, target_x, target_y]):
            return

        distance = np.linalg.norm([target_x - last_x, target_y - last_y])
        # If the move is negligible, skip speed adjustment
        if distance < NEGLIGIBLE_XY_DISTANCE:
            return

        speed = SLOW_XY_STAGE_SPEED if distance < SHORT_XY_DISTANCE else FAST_XY_STAGE_SPEED

        # If the speed is already set appropriately, no need to update
        if self._xy_stage_speed == speed:
            return

        self.mmcore.setProperty(self._xy_stage_device, "MotorSpeedX-S(mm/s)", speed)
        self.mmcore.setProperty(self._xy_stage_device, "MotorSpeedY-S(mm/s)", speed)

        self._xy_stage_speed = speed
        logger.debug(f"Set stage speed to {speed} mm/s")

    def _engage_autofocus(self, event: MDAEvent) -> None:
        if not self._use_autofocus:
            logger.debug("Autofocus is disabled.")
            return

        if self._autofocus_method == DEMO_PFS_METHOD:
            self._engage_demo_pfs(
                event=event,
                success_rate=0.5,
                fail_at_index=self._autofocus_fail_at_index,
            )
        else:
            z_position = None
            if event.properties:
                for dev, prop, value in event.properties:
                    if dev == self._autofocus_stage and prop == "Position":
                        z_position = value
                        break
            if z_position is None:
                z_position = self.mmcore.getPosition(self._autofocus_stage)
            self._engage_nikon_pfs(self._autofocus_stage, z_position)

    def _engage_demo_pfs(
        self,
        event: MDAEvent | None = None,
        success_rate: float = 0.9,
        fail_at_index: list[dict] | None = None,
    ):
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
        """
        if fail_at_index is not None and event is not None:
            # For SequencedEvents, use the first sub-event's index
            event_index = (
                event.events[0].index if isinstance(event, SequencedEvent) else event.index
            )
            self._autofocus_success = not any(
                all(event_index.get(k) == v for k, v in idx.items()) for idx in fail_at_index
            )
        else:
            self._autofocus_success = np.random.random() < success_rate

        if self._autofocus_success:
            logger.debug(f"{DEMO_PFS_METHOD} call succeeded")
        else:
            logger.debug(f"{DEMO_PFS_METHOD} call failed")

    def _engage_nikon_pfs(self, z_stage_name: str, z_position: float):
        """
                Attempt to engage Nikon PFS continuous autofocus. This function will log a
                message and continue if continuous autofocus is already engaged. Otherwise,
                it will attempt to engage autofocus, moving the z stage by amounts given in
                `z_offsets`, if necessary.
        `
                Parameters`
                ----------
                z_stage_name : str
                    The name of the z stage device which will be moved to help engage autofocus.
                z_position : float
                    The target position at which autofocus will be engaged.
        """
        core = self.mmcore
        self._autofocus_success = False
        z_offsets = [0, -10, 10, -20, 20, -30, 30]  # in um

        # Turn on autofocus if it has been turned off. This call has no effect is
        # continuous autofocus is already engaged
        try:
            core.fullFocus()
            time.sleep(0.2)  # needed before we can call isContinuousFocusLocked()
            logger.debug("Call to fullFocus() succeeded")
        except Exception:
            # TODO: Test is this make a difference
            # We should see fewer instances of autofocus failing on the first channel but engaging on the next channel
            # If so, engage autofocus only on the first channel
            time.sleep(5)  # wait for oil to catch up, usually needed when switching wells
            logger.debug("Call to fullFocus() failed")

        # Check if autofocus is already engaged
        if core.isContinuousFocusLocked():
            self._autofocus_success = True
            logger.debug("Continuous autofocus is already engaged")
            return

        for z_offset in z_offsets:
            core.setPosition(z_stage_name, z_position + z_offset)
            core.waitForDevice(z_stage_name)

            # This call engages autofocus
            core.enableContinuousFocus(True)
            time.sleep(1)  # Wait for autofocus to engage

            if core.isContinuousFocusLocked():
                self._autofocus_success = True
                logger.debug(f"Continuous autofocus engaged with Z offset of {z_offset} um")
                break
            else:
                logger.debug(f"Autofocus call failed with Z offset of {z_offset} um")

        if not self._autofocus_success:
            # return z stage to original position if autofocus attempts failed
            core.setPosition(z_stage_name, z_position)
            core.waitForDevice(z_stage_name)

            logger.error(f"Autofocus call failed after {len(z_offsets)} attempts")

    def acquire(
        self,
        output_dir: str | Path,
        name: str,
        mda_config: MDASequence | str | Path,
    ) -> None:
        """Run a Mantis microscope acquisition.

        Parameters
        ----------
        output_dir : str | Path
            Directory where acquisition data will be saved.
        name : str
            Base acquisition name; an index suffix will be appended automatically.
        mda_config : MDASequence | str | Path
            An MDASequence object or path to an MDA sequence configuration YAML file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        name = _get_next_acquisition_name(output_dir, name)

        if isinstance(mda_config, MDASequence):
            sequence = mda_config
        else:
            logger.info(f"Loading MDA sequence from {mda_config}")
            sequence = MDASequence.from_file(mda_config)

        data_path = output_dir / f"{name}.ome.zarr"

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


def parse_well_id(position_name: str | None) -> str:
    """Extract the well_id portion from a useq position name.

    Supports the historical mantis position-naming conventions:

    - ``"A1-Site_0"`` → ``"A1"`` (HCS plate format)
    - ``"1-Pos000_000"`` → ``"1"`` (Micro-Manager position list format)
    - ``"A1_0000"`` → ``"A1"`` (pymmcore-plus default format)
    - anything else → the full name (or ``"0"`` if empty)
    """
    if not position_name:
        return "0"
    # Order matters: try the more specific HCS/MM separators before the
    # generic underscore fallback, otherwise "A1-Site_0" would be split on
    # "_" first and yield "A1-Site" instead of "A1".
    for sep in ("-Site_", "-Pos", "_"):
        if sep in position_name:
            return position_name.split(sep, 1)[0]
    return position_name
