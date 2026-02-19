from __future__ import annotations

import logging
import time

from pathlib import Path

import numpy as np
import useq

from ome_writers import (
    AcquisitionSettings,
    create_stream,
    useq_to_acquisition_settings,
)
from pymmcore_plus.core import CMMCorePlus
from pymmcore_plus.core._sequencing import SequencedEvent
from pymmcore_plus.mda import MDAEngine
from pymmcore_plus.metadata import SummaryMetaV1
from useq import MDAEvent, MDASequence

# Get the logger instance (will be configured by the CLI entry point)
logger = logging.getLogger(__name__)

DEMO_PFS_METHOD = "demo-PFS"


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
        super().__init__(mmc, *args, **kwargs)
        self._use_autofocus = False
        self._autofocus_success = False
        self._autofocus_stage = None
        self._autofocus_method = None
        self._xy_stage_device = None
        self._slow_speed = 2.0  # mm/s for short distances
        self._fast_speed = 5.75  # mm/s for long distances
        self._short_distance_threshold = 2000  # um

        # Register event callbacks for logging
        mmc.mda.set_engine(self)
        mmc.events.propertyChanged.connect(self._on_property_changed)
        mmc.events.roiSet.connect(self._on_roi_set)
        mmc.events.XYStagePositionChanged.connect(self._on_xy_stage_position_changed)

    def _on_property_changed(self, device: str, property_name: str, value: str) -> None:
        """Log property changes at debug level."""
        logger.debug(f"Property changed: {device}.{property_name} = {value}")

    def _on_roi_set(self, camera: str, x: int, y: int, width: int, height: int) -> None:
        """Log ROI changes at debug level."""
        logger.debug(
            f"Setting ROI on {camera} to x={x}, y={y}, width={width}, height={height}"
        )

    def _on_xy_stage_position_changed(self, device: str, x: float, y: float) -> None:
        """Log stage position changes at debug level.

        None: The DXYStage device adapter fires this callback twice per move.
        """
        logger.debug(f"XY stage position changed: device={device}, x={x:.2f}, y={y:.2f}")

    def setup_sequence(self, sequence: MDASequence) -> SummaryMetaV1 | None:
        """Setup mantis-specific hardware before the sequence starts.

        Reads mantis-specific settings from sequence.metadata['mantis'] if present,
        otherwise uses default values.
        """
        logger.info("Setting up Mantis-specific hardware for acquisition sequence")

        # Call parent setup first
        summary = super().setup_sequence(sequence)
        core = self.mmcore

        # Extract mantis settings from metadata
        microscope_meta = sequence.metadata.get("mantis", {}) if sequence.metadata else {}

        # Apply initialization settings
        # TODO: move to proper place
        if initialization_settings := microscope_meta.get("initialization_settings"):
            logger.info(f"Applying {len(initialization_settings)} initialization settings")
            for setting in initialization_settings:
                logger.debug(f"  Setting {setting[0]}.{setting[1]} = {setting[2]}")
                core.setProperty(setting[0], setting[1], setting[2])
        else:
            logger.debug("No initialization settings specified")

        # Apply ROI settings
        if roi := microscope_meta.get("roi"):
            core.clearROI()
            core.setROI(*roi)

        # Apply setup hardware sequencing settings
        # TODO: reset hardware sequencing settings after acquisition
        if setup_hardware_sequencing_settings := microscope_meta.get(
            "setup_hardware_sequencing_settings"
        ):
            logger.info(
                f"Applying {len(setup_hardware_sequencing_settings)} hardware sequencing settings"
            )
            for setting in setup_hardware_sequencing_settings:
                logger.debug(f"  Setting {setting[0]}.{setting[1]} = {setting[2]}")
                core.setProperty(setting[0], setting[1], setting[2])
        else:
            logger.debug("No hardware sequencing settings specified")

        # Set focus device
        if z_stage := microscope_meta.get("z_stage"):
            logger.info(f"Setting focus device to: {z_stage}")
            core.setProperty("Core", "Focus", z_stage)
        else:
            logger.debug(f"Using default focus device: {core.getFocusDevice()}")

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
        # TODO: confirm this is required
        self._xy_stage_device = core.getXYStageDevice()
        logger.debug(f"XY stage device: {self._xy_stage_device}")

        logger.info("Mantis hardware setup completed successfully")

        return summary

    def setup_event(self, event: useq.MDAEvent) -> None:
        """Prepare mantis hardware for each event."""
        # Set XY stage position and engage autofocus
        # Note: this command will not move the stage if the target position is the same
        # as the last commanded position and force_set_xy_position is False.
        # TODO: modulate stage speed
        self._set_event_xy_position(event)

        # Set Z position for the event only if not using autofocus; calling
        # setPosition will disengage continuous autofocus. The autofocus algorithm
        # sets the z position independently.
        if not self._use_autofocus and self._autofocus_stage and event.z_pos is not None:
            self.mmcore.setPosition(self._autofocus_stage, event.z_pos)
            self.mmcore.waitForDevice(self._autofocus_stage)

        # Engage autofocus
        self._engage_autofocus(event)

        # Call parent setup_event
        super().setup_event(event)

    def exec_event(self, event: MDAEvent):
        if self._use_autofocus and not self._autofocus_success:
            # Pad zarr dataset with empty images if autofocus failed at this position
            logger.debug("Autofocus failed, padding zarr dataset with zeros")
            image_height = self.mmcore.getImageHeight()
            image_width = self.mmcore.getImageWidth()
            dtype = f"uint{self.mmcore.getImageBitDepth()}"
            _img = np.zeros((image_height, image_width), dtype=dtype)
            if isinstance(event, SequencedEvent):
                for _event in event.events:
                    yield (_img, _event, self.get_frame_metadata(_event))
            else:
                yield (_img, event, self.get_frame_metadata(event))
        else:
            yield from super().exec_event(event)

    # def _set_event_xy_position(self, event: MDAEvent) -> None:
    #     """Override XY position setting to handle autofocus after stage movement.

    #     Changes from default MDAEngine behavior:
    #     1. SPEED ADJUSTMENT: Sets stage speed based on distance before movement
    #        - Default: Uses pre-configured stage speed
    #        - Mantis: 2.0 mm/s for <2000 um, 5.75 mm/s for longer moves
    #        - Reason: Slow speed keeps continuous autofocus engaged during short moves

    #     2. EXPLICIT WAIT: Calls core.waitForDevice() after setXYPosition()
    #        - Default: Does NOT wait for stage to finish moving
    #        - Mantis: Waits for movement completion before proceeding
    #        - Reason: Ensures stage is settled before engaging autofocus

    #     3. AUTOFOCUS: Engages continuous autofocus after each position change
    #        - Default: No automatic autofocus behavior
    #        - Mantis: Calls _engage_autofocus() with multiple Z offset attempts
    #        - Reason: Re-establish Nikon PFS lock after stage movement

    #     4. POSITION TRACKING: Uses local _last_xy_position instead of core's cache
    #        - Note: This differs from default's use of core._last_xy_position
    #        - TODO: Consider using core._last_xy_position for consistency
    #     """
    #     event_x, event_y = event.x_pos, event.y_pos

    #     # If neither coordinate is provided, do nothing
    #     if event_x is None and event_y is None:
    #         return

    #     core = self.mmcore

    #     # Skip if no XY stage device is found
    #     if not self._xy_stage_device:
    #         logger.warning("No XY stage device found. Cannot set XY position.")
    #         return

    #     # Get current position
    #     try:
    #         current_x, current_y = core.getXYPosition()
    #     except Exception as e:
    #         logger.warning(f"Failed to get current XY position: {e}")
    #         current_x, current_y = 0, 0

    #     # Use current position if event position is None
    #     target_x = current_x if event_x is None else event_x
    #     target_y = current_y if event_y is None else event_y

    #     # Check if position actually changed
    #     if self._last_xy_position is not None and np.allclose(
    #         [target_x, target_y], self._last_xy_position, rtol=0, atol=0.01
    #     ):
    #         logger.debug(
    #             f"Stage position unchanged ({target_x:.2f}, {target_y:.2f}), skipping move"
    #         )
    #         return

    #     # Adjust stage speed based on distance (only if using autofocus)
    #     if self._use_autofocus and self._xy_stage_device:
    #         distance = np.linalg.norm([target_x - current_x, target_y - current_y])
    #         speed = (
    #             self._slow_speed
    #             if distance < self._short_distance_threshold
    #             else self._fast_speed
    #         )

    #         try:
    #             # Set XY stage speed (mantis-specific property names)
    #             core.setProperty(self._xy_stage_device, "MotorSpeedX-S(mm/s)", speed)
    #             core.setProperty(self._xy_stage_device, "MotorSpeedY-S(mm/s)", speed)
    #             logger.debug(f"Set stage speed to {speed} mm/s for distance {distance:.1f} um")
    #         except Exception as e:
    #             logger.debug(f"Could not set stage speed: {e}")

    #     # Move to the target position
    #     try:
    #         logger.debug(f"Moving stage to ({target_x:.2f}, {target_y:.2f})")
    #         core.setXYPosition(target_x, target_y)
    #         core.waitForDevice(self._xy_stage_device)
    #         self._last_xy_position = [target_x, target_y]
    #     except Exception as e:
    #         logger.warning(f"Failed to set XY position: {e}")
    #         return

    #     # Attempt autofocus after stage movement
    #     if self._use_autofocus:
    #         self._engage_autofocus(event)

    def _engage_autofocus(self, event: MDAEvent) -> None:
        if not self._use_autofocus:
            logger.debug("Autofocus is disabled.")
            return

        if self._autofocus_method == DEMO_PFS_METHOD:
            self._engage_demo_pfs(success_rate=0.5)
        else:
            # TODO: fix after resolving https://github.com/czbiohub-sf/shrimPy/issues/242
            z_position = self.mmcore.getPosition(self._autofocus_stage)
            # MDA events have z_pos of the scan stage
            # z_position = (
            #     event.z_pos
            #     if event.z_pos is not None
            #     else self.mmcore.getPosition(self._autofocus_stage)
            # )
            self._engage_nikon_pfs(self._autofocus_stage, z_position)

    def _engage_demo_pfs(self, success_rate: float = 0.9):
        """
        Engage demo PFS continuous autofocus.

        Parameters
        ----------
        success_rate : float
            The probability of success for the demo PFS call.
        """
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

        # Try to engage autofocus with fullFocus
        try:
            core.fullFocus()
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
        mda_config: str | Path,
    ) -> None:
        """Run a Mantis microscope acquisition.

        Parameters
        ----------
        output_dir : str | Path
            Directory where acquisition data will be saved.
        name : str
            Base acquisition name; an index suffix will be appended automatically.
        mda_config : str | Path
            Path to the MDA sequence configuration YAML file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        name = _get_next_acquisition_name(output_dir, name)

        logger.info(f"Loading MDA sequence from {mda_config}")
        sequence = MDASequence.from_file(mda_config)

        data_path = output_dir / f"{name}.ome.zarr"
        logger.info(f"Initializing OME-ZARR writer at {data_path}")

        core = self.mmcore
        # TODO: ROI is read from metadata because it has not been applied yet
        roi = sequence.metadata.get("mantis", {}).get("roi")
        if roi:
            image_width, image_height = roi[-2], roi[-1]
        else:
            image_width = core.getImageWidth()
            image_height = core.getImageHeight()
        pixel_size_um = core.getPixelSizeUm()

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

        settings = AcquisitionSettings(
            root_path=data_path,
            dtype="uint16",
            compression="blosc-zstd",
            format="acquire-zarr",
            overwrite=False,
            **acq_settings,
        )

        logger.info(f"Starting acquisition: {name}")
        with create_stream(settings) as stream:

            @self.mmcore.mda.events.frameReady.connect
            def _on_frame_ready(frame: np.ndarray, _event: useq.MDAEvent, _meta: dict) -> None:
                stream.append(frame)

            logger.info("Starting MDA acquisition sequence")
            self.mmcore.mda.run(sequence)

        logger.info("Acquisition completed successfully")

    @staticmethod
    def initialize_core(mm_config: str | Path | None = None) -> CMMCorePlus:
        """Initialize and configure the Core instance for Mantis.

        Parameters
        ----------
        mm_config : str | Path | None
            Path to the Micro-Manager configuration file. If None, uses default demo config.

        Returns
        -------
        CMMCorePlus (or UniMMCore)
            Configured core instance ready for use.
        """
        logger.info("Initializing Micro-Manager core")
        core = CMMCorePlus().instance()

        if mm_config is None:
            logger.info("No configuration file provided. Using MMConfig_demo.cfg.")
            _config = None
        else:
            logger.info(f"Loading Micro-Manager configuration from: {mm_config}")
            _config = mm_config

        core.loadSystemConfiguration(_config)
        logger.info("Micro-Manager core initialized successfully")

        return core


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
