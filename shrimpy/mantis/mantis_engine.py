from __future__ import annotations

import json
import logging
import time

from collections.abc import Iterable
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
        self._laser_power_property: tuple[str, str] | None = None
        self._current_well_id: str | None = None

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
        self._laser_power_property = None
        self._current_well_id = None

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
        - ``laser_power_property`` ([device, property], optional): which core
          property to write ``laser_power_mW`` to. If omitted, only the
          camera exposure time is applied.
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

        # Validate that every well referenced by the sequence appears in the CSV
        sequence_wells = {parse_well_id(p.name) for p in (sequence.stage_positions or [])}
        csv_wells = set(illumination.index.values)
        missing = sequence_wells - csv_wells
        if missing:
            raise ValueError(
                f"illumination.csv is missing well_ids present in the sequence: "
                f"{sorted(missing)}"
            )

        laser_prop = cfg.get("laser_power_property")
        if laser_prop is not None:
            if not (isinstance(laser_prop, (list, tuple)) and len(laser_prop) == 2):
                raise ValueError(
                    "autoexposure.laser_power_property must be [device, property]."
                )
            self._laser_power_property = (str(laser_prop[0]), str(laser_prop[1]))

        self._illumination_settings = illumination
        logger.info(
            f"Manual autoexposure enabled with {len(illumination)} wells from {csv_path}"
        )

    def _apply_autoexposure(self, event: MDAEvent) -> MDAEvent:
        """Dispatch per-event autoexposure based on the configured method.

        Returns a possibly-modified event. The caller must pass the returned
        event to ``super().setup_event`` so the engine's own ``setExposure``
        call uses the override instead of the value baked into the original
        event.
        """
        if self._autoexposure_method is None:
            return event
        if self._autoexposure_method == "manual":
            return self._apply_manual_autoexposure(event)
        raise NotImplementedError(
            f"Autoexposure method '{self._autoexposure_method}' is not implemented."
        )

    def _apply_manual_autoexposure(self, event: MDAEvent) -> MDAEvent:
        """Override event exposure (and apply laser power) for the event's well.

        Uses the per-well values loaded from the illumination CSV during
        ``_setup_manual_exposure``.
        """
        if self._illumination_settings is None:
            return event

        # SequencedEvents carry their pos_name on the first sub-event
        pos_name = event.pos_name
        if pos_name is None and isinstance(event, SequencedEvent) and event.events:
            pos_name = event.events[0].pos_name
        well_id = parse_well_id(pos_name)

        if well_id not in self._illumination_settings.index:
            if well_id != self._current_well_id:
                logger.warning(
                    f"No illumination settings for well '{well_id}'; "
                    "leaving event exposure unchanged."
                )
                self._current_well_id = well_id
            return event

        row = self._illumination_settings.loc[well_id]
        exposure_ms = float(row["exposure_time_ms"])
        laser_power = float(row["laser_power_mW"])

        # Laser power only needs to change on well transitions; exposure must
        # be re-applied to every event since super().setup_event(event) will
        # call setExposure(event.exposure) using the override below.
        if well_id != self._current_well_id:
            logger.info(
                f"Applying manual autoexposure for well '{well_id}': "
                f"exposure={exposure_ms} ms, laser_power={laser_power} mW"
            )
            if self._laser_power_property is not None:
                device, prop = self._laser_power_property
                self.mmcore.setProperty(device, prop, laser_power)
            self._current_well_id = well_id

        # Rewrite the event so the parent engine applies our exposure
        return event.model_copy(update={"exposure": exposure_ms})

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
        # Override the event's exposure (and apply laser power) per-well.
        # The returned event must be used for super().setup_event below,
        # otherwise the parent engine's setExposure(event.exposure) call
        # would clobber any per-well exposure set here.
        event = self._apply_autoexposure(event)

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
    for sep in ("_", "-Site_", "-Pos"):
        if sep in position_name:
            return position_name.split(sep, 1)[0]
    return position_name
