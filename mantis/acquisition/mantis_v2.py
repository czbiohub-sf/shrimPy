from __future__ import annotations

import useq
from useq import MDASequence

from pymmcore_plus.experimental.unicore import UniMMCore
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
    """
    
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
        
        return summary
    
    def setup_event(self, event: useq.MDAEvent) -> None:
        """Prepare mantis hardware for each event."""
        # Custom pre-setup logic could go here
        
        # Call parent setup
        super().setup_event(event)
        
        # Custom post-setup logic could go here

    def on_xy_stage_moved(self, x: float, y: float) -> None:
        """Handle XY stage movement events."""
        print(f"XY Stage moved to X: {x}, Y: {y}")


def initialize_mantis_core(config_path: str | None = None) -> UniMMCore:
    """Initialize and configure the UniMMCore instance for Mantis.
    
    Parameters
    ----------
    config_path : str | None
        Path to the Micro-Manager configuration file. If None, uses default demo config.
    
    Returns
    -------
    UniMMCore
        Configured core instance ready for use.
    """
    core = UniMMCore()
    
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
