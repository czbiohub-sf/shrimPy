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
        """Setup mantis-specific hardware before the sequence starts."""
        # Call parent setup first
        summary = super().setup_sequence(sequence)
        
        # Mantis-specific hardware setup
        core = self.mmcore
        
        # Set the ROI for label-free acquisition
        core.setROI(0, 512, 2048, 256)
        
        # Enable TriggerScope sequencing
        core.setProperty("TS_DAC01", "Sequence", "On")
        
        # Set the focus device to the Axial Piezo (AP Galvo)
        core.setProperty("Core", "Focus", "AP Galvo")
        
        # Enable TTL blanking
        core.setProperty("TS_TTL1-8", "Blanking", "On")
        
        return summary
    
    def setup_event(self, event: useq.MDAEvent) -> None:
        """Prepare mantis hardware for each event."""
        # Custom pre-setup logic could go here
        
        # Call parent setup
        super().setup_event(event)
        
        # Custom post-setup logic could go here


# Main execution code
if __name__ == "__main__":
    mmconfig_file = "C:\\Users\\Cameron\\justin\\shrimPy\\CompMicro_MMConfigs\\Dev_Computer\\mantis2-demo.cfg"
    mda_sequence_file = "C:\\Users\\Cameron\\justin\\shrimPy\\examples\\acquisition_settings\\example_mda_sequence.yaml"

    # Create and configure core
    core = UniMMCore()
    core.loadSystemConfiguration(mmconfig_file)
    # core.setPixelSizeConfig("Res40x")

    # Create and register the custom Mantis engine
    mantis_engine = MantisEngine(core)
    core.mda.set_engine(mantis_engine)

    # Load the sequence
    sequence = MDASequence.from_file(mda_sequence_file)

    # Run the acquisition
    core.mda.run(sequence)
