from __future__ import annotations

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

from shrimpy.mantis.mantis_logger import configure_mantis_logger, get_mantis_logger

# Get the logger instance
logger = get_mantis_logger()

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
        """Initialize the MantisEngine.

        Parameters
        ----------
        mmc : CMMCorePlus
            The Micro-Manager core instance
        """
        super().__init__(mmc, *args, **kwargs)
        self._use_autofocus = False
        self._autofocus_success = False
        self._autofocus_stage = None
        self._autofocus_method = None
        self._xy_stage_device = None
        self._last_xy_position = None
        self._slow_speed = 2.0  # mm/s for short distances
        self._fast_speed = 5.75  # mm/s for long distances
        self._short_distance_threshold = 2000  # um

        # Register event callbacks for logging
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
        """Log stage position changes at debug level."""
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
        core.events.XYStagePositionChanged.connect(self.on_xy_stage_moved)
        self._xy_stage_device = core.getXYStageDevice()
        logger.debug(f"XY stage device: {self._xy_stage_device}")

        logger.info("Mantis hardware setup completed successfully")

        return summary

    def setup_event(self, event: useq.MDAEvent) -> None:
        """Prepare mantis hardware for each event."""
        # Log sequenced event details to show all channels being acquired
        # from pymmcore_plus.core._sequencing import SequencedEvent

        # TODO: consider removing this, pymmcore_plus already logs this?
        # if isinstance(event, SequencedEvent):
        #     channels = [e.channel.config if e.channel else 'None' for e in event.events]
        #     unique_channels = list(dict.fromkeys(channels))  # preserve order, remove dupes
        #     logger.info(
        #         f"Sequenced event will acquire {len(event.events)} images: "
        #         f"{len(unique_channels)} channels {unique_channels} × "
        #         f"{len(event.events)//len(unique_channels)} z-slices"
        #     )

        # Call parent setup
        super().setup_event(event)

    def exec_event(self, event: MDAEvent):
        if self._use_autofocus and not self._autofocus_success:
            # Pad zarr dataset with empty images if autofocus failed at this position
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

    def _set_event_xy_position(self, event: MDAEvent) -> None:
        """Override XY position setting to handle autofocus after stage movement.

        Changes from default MDAEngine behavior:
        1. SPEED ADJUSTMENT: Sets stage speed based on distance before movement
           - Default: Uses pre-configured stage speed
           - Mantis: 2.0 mm/s for <2000 um, 5.75 mm/s for longer moves
           - Reason: Slow speed keeps continuous autofocus engaged during short moves

        2. EXPLICIT WAIT: Calls core.waitForDevice() after setXYPosition()
           - Default: Does NOT wait for stage to finish moving
           - Mantis: Waits for movement completion before proceeding
           - Reason: Ensures stage is settled before engaging autofocus

        3. AUTOFOCUS: Engages continuous autofocus after each position change
           - Default: No automatic autofocus behavior
           - Mantis: Calls _engage_autofocus() with multiple Z offset attempts
           - Reason: Re-establish Nikon PFS lock after stage movement

        4. POSITION TRACKING: Uses local _last_xy_position instead of core's cache
           - Note: This differs from default's use of core._last_xy_position
           - TODO: Consider using core._last_xy_position for consistency
        """
        event_x, event_y = event.x_pos, event.y_pos

        # If neither coordinate is provided, do nothing
        if event_x is None and event_y is None:
            return

        core = self.mmcore

        # Skip if no XY stage device is found
        if not self._xy_stage_device:
            logger.warning("No XY stage device found. Cannot set XY position.")
            return

        # Get current position
        try:
            current_x, current_y = core.getXYPosition()
        except Exception as e:
            logger.warning(f"Failed to get current XY position: {e}")
            current_x, current_y = 0, 0

        # Use current position if event position is None
        target_x = current_x if event_x is None else event_x
        target_y = current_y if event_y is None else event_y

        # Check if position actually changed
        if self._last_xy_position is not None and np.allclose(
            [target_x, target_y], self._last_xy_position, rtol=0, atol=0.01
        ):
            logger.debug(
                f"Stage position unchanged ({target_x:.2f}, {target_y:.2f}), skipping move"
            )
            return

        # Adjust stage speed based on distance (only if using autofocus)
        if self._use_autofocus and self._xy_stage_device:
            distance = np.linalg.norm([target_x - current_x, target_y - current_y])
            speed = (
                self._slow_speed
                if distance < self._short_distance_threshold
                else self._fast_speed
            )

            try:
                # Set XY stage speed (mantis-specific property names)
                core.setProperty(self._xy_stage_device, "MotorSpeedX-S(mm/s)", speed)
                core.setProperty(self._xy_stage_device, "MotorSpeedY-S(mm/s)", speed)
                logger.debug(f"Set stage speed to {speed} mm/s for distance {distance:.1f} um")
            except Exception as e:
                logger.debug(f"Could not set stage speed: {e}")

        # Move to the target position
        try:
            logger.debug(f"Moving stage to ({target_x:.2f}, {target_y:.2f})")
            core.setXYPosition(target_x, target_y)
            core.waitForDevice(self._xy_stage_device)
            self._last_xy_position = [target_x, target_y]
        except Exception as e:
            logger.warning(f"Failed to set XY position: {e}")
            return

        # Attempt autofocus after stage movement
        if self._use_autofocus:
            self._engage_autofocus(event)

    def _engage_autofocus(self, event: MDAEvent):
        """Attempt to engage continuous autofocus after stage movement.

        This method tries to engage Nikon PFS continuous autofocus, attempting
        different Z offsets if the initial attempt fails.

        Parameters
        ----------
        event : MDAEvent
            The current event being executed
        """
        core = self.mmcore

        # Get the starting Z position
        try:
            z_position = (
                event.z_pos
                if event.z_pos is not None
                else core.getPosition(self._autofocus_stage)
            )
        except Exception:
            z_position = core.getPosition(self._autofocus_stage)

        logger.debug(f"Engaging autofocus at Z position {z_position:.2f} um")

        self._autofocus_success = False
        error_occurred = False
        z_offsets = [0, -10, 10, -20, 20, -30, 30]  # in um

        # TODO: refactor
        if self._autofocus_method == DEMO_PFS_METHOD:
            self._autofocus_success = np.random.random() > 0.5  # 50% chance of failure
            if self._autofocus_success:
                logger.debug(f"{DEMO_PFS_METHOD} call succeeded")
            else:
                logger.debug(f"{DEMO_PFS_METHOD} call failed")
            return

        # Try to engage autofocus with fullFocus
        try:
            core.fullFocus()
            logger.debug("Call to fullFocus() succeeded")
        except Exception:
            logger.error("Call to fullFocus() failed")

        # Check if autofocus is already engaged
        if core.isContinuousFocusLocked():
            self._autofocus_success = True
            logger.debug("Continuous autofocus is already engaged")
        else:
            # Try different Z offsets
            for z_offset in z_offsets:
                try:
                    core.setPosition(self._autofocus_stage, z_position + z_offset)
                    core.waitForDevice(self._autofocus_stage)

                    core.enableContinuousFocus(True)
                    time.sleep(1)  # Wait for autofocus to engage

                    if core.isContinuousFocusLocked():
                        self._autofocus_success = True
                        if error_occurred:
                            logger.debug(
                                f"Continuous autofocus engaged with Z offset of {z_offset} um"
                            )
                        break
                    else:
                        error_occurred = True
                        logger.debug(f"Autofocus call failed with Z offset of {z_offset} um")
                except Exception as e:
                    logger.debug(f"Error during autofocus attempt at offset {z_offset}: {e}")
                    error_occurred = True

        if not self._autofocus_success:
            logger.error(f"Autofocus call failed after {len(z_offsets)} attempts")

    def on_xy_stage_moved(self, x: float, y: float) -> None:
        """Handle XY stage movement events."""
        # TODO: throws error, x = 'XY' rather than float
        pass
        # logger.debug(f"XY stage position changed: ({x:.2f}, {y:.2f})")


def initialize_mantis_core(mm_config: str | Path | None = None) -> CMMCorePlus:
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


def create_mantis_engine(
    core: CMMCorePlus, use_hardware_sequencing: bool = True
) -> MantisEngine:
    """Create and register a MantisEngine with the core.

    Parameters
    ----------
    core : UniMMCore
        The core instance to attach the engine to.
    use_hardware_sequencing : bool
        Whether to enable hardware sequencing (default: True).

    Returns
    -------
    MantisEngine
        The created and registered engine instance.
    """
    logger.info(f"Creating MantisEngine (hardware_sequencing={use_hardware_sequencing})")
    engine = MantisEngine(core, use_hardware_sequencing=use_hardware_sequencing)
    core.mda.set_engine(engine)
    return engine


def acquire(
    mm_config: str | Path,
    mda_config: str | Path,
    output_dir: str | Path,
    name: str = "mantis_acquisition",
) -> None:
    """Run a Mantis microscope acquisition.

    Parameters
    ----------
    mm_config : str | Path
        Path to Micro-Manager configuration file.
    mda_config : str | Path
        Path to MDA sequence configuration YAML file.
    output_dir : str | Path
        Output directory where acquisition data and logs will be saved.
    name : str
        Name of the acquisition (used for log files and output).
    """
    from shrimpy.mantis.mantis_logger import log_conda_environment

    # Convert to Path objects
    mm_config = Path(mm_config)
    mda_config = Path(mda_config)
    output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure mantis logger
    logger = configure_mantis_logger(output_dir, name)

    # Log conda environment
    log_dir = output_dir / "logs"
    output_log, errors = log_conda_environment(log_dir)
    if output_log:
        logger.debug(output_log.decode("ascii").strip())
    if errors:
        logger.error(errors.decode("ascii"))

    # Load the sequence
    logger.info(f"Loading MDA sequence from {mda_config}")
    sequence = MDASequence.from_file(mda_config)

    # Initialize core and engine using common functions
    core = initialize_mantis_core(mm_config)
    use_hardware_sequencing = sequence.metadata.get("mantis").get("use_hardware_sequencing")
    create_mantis_engine(core, use_hardware_sequencing=use_hardware_sequencing)

    # Setup data writer
    data_path = output_dir / f"{name}.ome.zarr"
    logger.info(f"Initializing OME-ZARR writer at {data_path}")

    # Get image dimensions from core
    # TODO: we are getting the ROI from the mda config because it has not been applied yet
    roi = sequence.metadata.get("mantis").get("roi")
    if roi:
        image_width = roi[-2]
        image_height = roi[-1]
    else:
        image_width = core.getImageWidth()
        image_height = core.getImageHeight()
    pixel_size_um = core.getPixelSizeUm()
    dtype = "uint16"

    # Define chunk shapes for optimal performance
    chunk_shapes = {
        "t": 1,
        "c": 1,
        "z": min(512, sequence.sizes["z"]),
        "y": image_height,
        "x": image_width,
    }

    # Convert MDASequence to acquisition settings
    acq_settings = useq_to_acquisition_settings(
        sequence,
        image_width=image_width,
        image_height=image_height,
        pixel_size_um=pixel_size_um,
        chunk_shapes=chunk_shapes,
    )

    # Create acquisition settings with compression
    settings = AcquisitionSettings(
        root_path=data_path,
        dtype=dtype,
        compression="blosc-zstd",
        format="acquire-zarr",
        overwrite=False,
        **acq_settings,
    )

    with create_stream(settings) as stream:
        # Append frames to the stream on frameReady event
        @core.mda.events.frameReady.connect
        def _on_frame_ready(frame: np.ndarray, event: useq.MDAEvent, metadata: dict) -> None:
            stream.append(frame)

        # Run the acquisition
        logger.info("Starting MDA acquisition sequence")
        core.mda.run(sequence)

        # Cleanup
        logger.info("Acquisition completed successfully")
