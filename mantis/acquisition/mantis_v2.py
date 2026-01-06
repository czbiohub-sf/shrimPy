from __future__ import annotations

import time

import numpy as np
import useq
from useq import MDASequence, MDAEvent

from pymmcore_plus._logger import logger
from pymmcore_plus.experimental.unicore import UniMMCore
from pymmcore_plus.core import CMMCorePlus
from pymmcore_plus.mda import MDAEngine, mda_listeners_connected
from pymmcore_plus.mda.handlers import OMETiffWriter, OMEZarrWriter
from pymmcore_plus.metadata import SummaryMetaV1


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
        self._autofocus_stage = None
        self._xy_stage_device = None
        self._last_xy_position = None
        self._slow_speed = 2.0  # mm/s for short distances
        self._fast_speed = 5.75  # mm/s for long distances
        self._short_distance_threshold = 2000  # um

    
    def setup_sequence(self, sequence: MDASequence) -> SummaryMetaV1 | None:
        """Setup mantis-specific hardware before the sequence starts.
        
        Reads mantis-specific settings from sequence.metadata['mantis'] if present,
        otherwise uses default values.
        """
        # Call parent setup first
        summary = super().setup_sequence(sequence)
        
        # Extract mantis settings from metadata
        mantis_meta = sequence.metadata.get('mantis', {}) if sequence.metadata else {}
        
        core = self.mmcore
        
        # Apply ROI settings
        if roi := mantis_meta.get('roi'):
            core.setROI(*roi)
        else:
            # Default ROI for label-free acquisition
            core.setROI(0, 512, 2048, 256)
        
        # Apply TriggerScope settings
        if ts := mantis_meta.get('trigger_scope'):
            if dac := ts.get('dac_sequencing'):
                core.setProperty(dac, "Sequence", "On")
            if ttl := ts.get('ttl_blanking'):
                core.setProperty(ttl, "Blanking", "On")
        else:
            # Default TriggerScope settings
            core.setProperty("TS_DAC01", "Sequence", "On")
            core.setProperty("TS_TTL1-8", "Blanking", "On")
        
        # Set focus device
        focus_device = mantis_meta.get('focus_device', 'AP Galvo')
        core.setProperty("Core", "Focus", focus_device)
        
        core.events.XYStagePositionChanged.connect(self.on_xy_stage_moved)
        
        # TODO: These hardcoded defaults will be replaced with proper configuration
        # reading from sequence.metadata['mantis'] once the metadata structure is implemented
        self._use_autofocus = True  # Enable autofocus by default
        self._autofocus_stage = core.getFocusDevice()  # Use the default focus device
        
        # Setup autofocus device if enabled
        if self._use_autofocus:
            autofocus_method = 'PFS'  # Hardcoded to Nikon PFS for now
            logger.debug(f'Setting autofocus method as {autofocus_method}')
            try:
                core.setAutoFocusDevice(autofocus_method)
            except Exception as e:
                logger.warning(f'Could not set autofocus device: {e}')
                self._use_autofocus = False
        else:
            logger.debug('Autofocus is not enabled')
        
        # Store XY stage device name
        self._xy_stage_device = core.getXYStageDevice()
        
        return summary
    
    def setup_event(self, event: useq.MDAEvent) -> None:
        """Prepare mantis hardware for each event."""
        # Custom pre-setup logic could go here
        
        # Call parent setup
        super().setup_event(event)
        
        # Custom post-setup logic could go here
    
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
        if (self._last_xy_position is not None and 
            np.allclose([target_x, target_y], self._last_xy_position, rtol=0, atol=0.01)):
            return
        
        # Adjust stage speed based on distance (only if using autofocus)
        if self._use_autofocus and self._xy_stage_device:
            distance = np.linalg.norm([target_x - current_x, target_y - current_y])
            speed = self._slow_speed if distance < self._short_distance_threshold else self._fast_speed
            
            try:
                # Set XY stage speed (mantis-specific property names)
                core.setProperty(self._xy_stage_device, 'MotorSpeedX-S(mm/s)', speed)
                core.setProperty(self._xy_stage_device, 'MotorSpeedY-S(mm/s)', speed)
                logger.debug(f'Set stage speed to {speed} mm/s for distance {distance:.1f} um')
            except Exception as e:
                logger.debug(f'Could not set stage speed: {e}')
        
        # Move to the target position
        try:
            logger.debug(f'Moving stage to ({target_x:.2f}, {target_y:.2f})')
            core.setXYPosition(target_x, target_y)
            core.waitForDevice(self._xy_stage_device)
            self._last_xy_position = [target_x, target_y]
        except Exception as e:
            logger.warning(f"Failed to set XY position: {e}")
            return
        
        # Attempt autofocus after stage movement
        if self._use_autofocus and self._autofocus_stage:
            self._engage_autofocus(event)
    
    def _engage_autofocus(self, event: MDAEvent) -> bool:
        """Attempt to engage continuous autofocus after stage movement.
        
        This method tries to engage Nikon PFS continuous autofocus, attempting
        different Z offsets if the initial attempt fails.
        
        Parameters
        ----------
        event : MDAEvent
            The current event being executed
            
        Returns
        -------
        bool
            True if autofocus successfully engaged, False otherwise
        """
        core = self.mmcore
        
        # Get the starting Z position
        try:
            z_position = event.z_pos if event.z_pos is not None else core.getPosition(self._autofocus_stage)
        except Exception:
            z_position = core.getPosition(self._autofocus_stage)
        
        logger.debug(f'Engaging autofocus at Z position {z_position:.2f} um')
        
        autofocus_success = False
        error_occurred = False
        z_offsets = [0, -10, 10, -20, 20, -30, 30]  # in um
        
        # Try to engage autofocus with fullFocus
        try:
            core.fullFocus()
            logger.debug('Call to fullFocus() succeeded')
        except Exception:
            logger.debug('Call to fullFocus() failed')
        
        # Check if autofocus is already engaged
        if core.isContinuousFocusLocked():
            autofocus_success = True
            logger.debug('Continuous autofocus is already engaged')
        else:
            # Try different Z offsets
            for z_offset in z_offsets:
                try:
                    core.setPosition(self._autofocus_stage, z_position + z_offset)
                    core.waitForDevice(self._autofocus_stage)
                    
                    core.enableContinuousFocus(True)
                    time.sleep(1)  # Wait for autofocus to engage
                    
                    if core.isContinuousFocusLocked():
                        autofocus_success = True
                        if error_occurred:
                            logger.debug(f'Continuous autofocus engaged with Z offset of {z_offset} um')
                        break
                    else:
                        error_occurred = True
                        logger.debug(f'Autofocus call failed with Z offset of {z_offset} um')
                except Exception as e:
                    logger.debug(f'Error during autofocus attempt at offset {z_offset}: {e}')
                    error_occurred = True
        
        if not autofocus_success:
            logger.error(f'Autofocus call failed after {len(z_offsets)} attempts')
        
        return autofocus_success


    def on_xy_stage_moved(self, x: float, y: float) -> None:
        """Handle XY stage movement events."""
        print(f"XY Stage moved to X: {x}, Y: {y}")


def initialize_mantis_core(config_path: str | None = None) -> CMMCorePlus:
    """Initialize and configure the Core instance for Mantis.
    
    Parameters
    ----------
    config_path : str | None
        Path to the Micro-Manager configuration file. If None, uses default demo config.
    
    Returns
    -------
    UniMMCore
        Configured core instance ready for use.
    """
    core = CMMCorePlus().instance()
    
    if config_path is None:
        config_path = "C:\\Users\\Cameron\\justin\\shrimPy\\CompMicro_MMConfigs\\Dev_Computer\\mantis2-demo.cfg"
    
    core.loadSystemConfiguration(config_path)
    # core.setPixelSizeConfig("Res40x")  # Uncomment if needed
    
    return core


def create_mantis_engine(core: UniMMCore, use_hardware_sequencing: bool = True) -> MantisEngine:
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
    engine = MantisEngine(core, use_hardware_sequencing=use_hardware_sequencing)
    core.mda.set_engine(engine)
    return engine


# Main execution code
if __name__ == "__main__":
    mmconfig_file = "C:\\Users\\Cameron\\justin\\shrimPy\\CompMicro_MMConfigs\\Dev_Computer\\mantis2-demo.cfg"
    mda_sequence_file = "C:\\Users\\Cameron\\justin\\shrimPy\\examples\\acquisition_settings\\example_mda_sequence.yaml"

    # Initialize core and engine using common functions
    core = initialize_mantis_core(mmconfig_file)
    mantis_engine = create_mantis_engine(core)

    # Load the sequence
    sequence = MDASequence.from_file(mda_sequence_file)

    # Run the acquisition
    core.mda.run(sequence)
