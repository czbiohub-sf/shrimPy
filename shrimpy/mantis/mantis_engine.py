from __future__ import annotations

import json
import logging
import time

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ome_writers import (
    AcquisitionSettings,
    useq_to_acquisition_settings,
)
from pymmcore_plus.core import CMMCorePlus
from pymmcore_plus.core._constants import Keyword
from pymmcore_plus.core._sequencing import SequencedEvent
from pymmcore_plus.mda import MDAEngine, SkipEvent
from pymmcore_plus.metadata import SummaryMetaV1
from pymmcore_plus.metadata.serialize import to_builtins
from useq import MDAEvent, MDASequence

from shrimpy.mantis.dynatrack import DynaTrackConfig, DynaTrackUpdater
from shrimpy.mantis.position_update import (
    PositionStore,
    PositionUpdateConfig,
    PositionUpdateManager,
)

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
        self._position_update_manager: PositionUpdateManager | None = None
        self._position_update_frames: dict[tuple[int, int], list[np.ndarray]] = {}
        self._position_update_expected_slices: int = 1

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

        # Setup position updating
        position_update_meta = microscope_meta.get("position_update", {})
        position_update_config = PositionUpdateConfig(
            enabled=position_update_meta.get("enabled", False),
            update_channel=position_update_meta.get("update_channel"),
            z_device=position_update_meta.get("z_device"),
        )
        if position_update_config.enabled and sequence.stage_positions:
            position_store = PositionStore()
            position_store.initialize_from_sequence(
                sequence, z_device=position_update_config.z_device
            )

            # Create DynaTrack updater if config is provided
            dynatrack_meta = position_update_meta.get("dynatrack")
            if dynatrack_meta:
                dynatrack_config = DynaTrackConfig(**dynatrack_meta)
                updater = DynaTrackUpdater(config=dynatrack_config)
                logger.info(
                    f"DynaTrack enabled: scale_yx={dynatrack_config.scale_yx}, "
                    f"scale_z={dynatrack_config.scale_z}, "
                    f"interval={dynatrack_config.tracking_interval}"
                )
            else:
                updater = None

            self._position_update_manager = PositionUpdateManager(
                config=position_update_config,
                position_store=position_store,
                updater=updater,
            )
            self._position_update_manager.start()
            self._position_update_frames = {}
            self._position_update_expected_slices = max(sequence.sizes.get("z", 1), 1)
            self.mmcore.mda.events.frameReady.connect(self._on_frame_ready)
            logger.info(
                f"Position updating enabled with {position_store.num_positions} positions"
            )
        else:
            self._position_update_manager = None

        logger.info("Mantis hardware setup completed successfully")

        # Call parent setup last so SummaryMetaV1 captures the fully
        # configured hardware state (ROI, focus device, etc.).
        return super().setup_sequence(sequence)

    def event_iterator(self, events: Iterable[MDAEvent]):
        """Wrap event iteration to apply position updates before logging.

        By applying position updates here (before the MDA runner emits
        ``eventStarted``), the logged event reflects the corrected
        coordinates rather than the original sequence values.
        """
        for event in super().event_iterator(events):
            if self._position_update_manager is not None:
                event = self._position_update_manager.apply_position_update(event)
            yield event

    def setup_event(self, event: MDAEvent) -> None:
        """Prepare mantis hardware for each event."""
        # Set XY stage position and engage autofocus
        # Note: this command will not move the stage if the target position is the same
        # as the last commanded position and force_set_xy_position is False.
        # TODO: debug resetting xy stage speed
        # self._adjust_xy_stage_speed(event)
        self._set_event_xy_position(event)
        # _set_event_xy_position does not wait for the stage to reach the target position
        if self._xy_stage_device:
            self.mmcore.waitForDevice(self._xy_stage_device)

        # Set Z position for the event only if not using autofocus; calling
        # setPosition will disengage continuous autofocus. The autofocus algorithm
        # sets the z position independently.
        # TODO: probably not needed, Z position is passed thru device properties in most cases
        if not self._use_autofocus and self._autofocus_stage and event.z_pos is not None:
            self.mmcore.setPosition(self._autofocus_stage, event.z_pos)
            self.mmcore.waitForDevice(self._autofocus_stage)

        # Engage autofocus
        self._engage_autofocus(event)

        # Skip acquisition if autofocus failed
        if self._use_autofocus and not self._autofocus_success:
            num_frames = len(event.events) if isinstance(event, SequencedEvent) else 1
            raise SkipEvent(num_frames=num_frames, reason="autofocus failed")

        # Call parent setup_event
        super().setup_event(event)

    def _on_frame_ready(self, img: np.ndarray, event: MDAEvent) -> None:
        """Buffer frames for position updating via frameReady callback.

        Counts z-slices per (timepoint, position) and triggers the updater
        as soon as all expected slices have been collected.
        """
        if self._position_update_manager is None:
            return

        # Only cache frames from the configured channel (None = all channels)
        update_channel = self._position_update_manager.config.update_channel
        if update_channel is not None and event.index.get("c") != update_channel:
            return

        t_idx = event.index.get("t", 0)
        p_idx = event.index.get("p", 0)
        tp = (t_idx, p_idx)

        if tp not in self._position_update_frames:
            self._position_update_frames[tp] = []
        self._position_update_frames[tp].append(img.copy())

        # Flush when all z-slices for this position have arrived
        if len(self._position_update_frames[tp]) >= self._position_update_expected_slices:
            frames = self._position_update_frames.pop(tp)
            self._position_update_manager.on_position_complete(t_idx, p_idx, frames)

    def teardown_sequence(self, sequence):
        # Position update: disconnect callback and shutdown
        if self._position_update_manager is not None:
            self.mmcore.mda.events.frameReady.disconnect(self._on_frame_ready)
            self._position_update_frames = {}
            self._position_update_manager.shutdown()
            self._position_update_manager = None

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

    def _create_stream_settings(
        self, sequence: MDASequence, data_path: str | Path
    ) -> AcquisitionSettings:
        """Create acquisition settings for an OME-ZARR stream.

        Parameters
        ----------
        sequence : MDASequence
            The acquisition sequence (used to determine dimensions and chunk shapes).
        data_path : str | Path
            Path where the .ome.zarr store will be created.

        Returns
        -------
        AcquisitionSettings
            Settings to pass to ``create_stream()``.
        """
        core = self.mmcore
        # ROI is read from metadata because it may not have been applied yet
        roi = sequence.metadata.get("mantis", {}).get("roi")
        if roi:
            image_width, image_height = roi[-2], roi[-1]
        else:
            image_width = core.getImageWidth()
            image_height = core.getImageHeight()
        pixel_size_um = core.getPixelSizeUm()

        # TODO: current implementation of ome-writers handlers overwrites provided chunking
        chunk_shapes = {
            "t": 1,
            "c": 1,
            "z": min(512, sequence.sizes["z"]),
            "y": image_height,
            "x": image_width,
        }

        acq_settings = useq_to_acquisition_settings(
            sequence,
            image_width=image_width,
            image_height=image_height,
            pixel_size_um=pixel_size_um,
            chunk_shapes=chunk_shapes,
        )

        return AcquisitionSettings(
            root_path=data_path,
            dtype="uint16",
            compression="blosc-zstd",
            format="acquire-zarr",
            overwrite=False,
            **acq_settings,
        )

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
        settings = self._create_stream_settings(sequence, data_path)

        # Write summary metadata after the zarr store is created
        # TODO: remove once ome-writers supports root-level metadata natively
        def _write_summary_metadata(_seq: MDASequence, meta: object) -> None:
            self.mmcore.mda.events.sequenceStarted.disconnect(_write_summary_metadata)
            if meta and isinstance(meta, dict):
                meta_path = data_path / "summary_metadata.json"
                meta_path.write_text(json.dumps(to_builtins(meta)))

        self.mmcore.mda.events.sequenceStarted.connect(_write_summary_metadata)

        logger.info(f"Starting acquisition: {name}")
        self.mmcore.mda.run(sequence, output=settings)
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
