from __future__ import annotations

from useq import MDASequence

from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda import mda_listeners_connected
from pymmcore_plus.mda.handlers import OMETiffWriter, OMEZarrWriter
import time

mmconfig_file = "C:\\Users\\Cameron\\justin\\shrimPy\\CompMicro_MMConfigs\\Dev_Computer\\mantis2-demo.cfg"
mda_sequence_file = "C:\\Users\\Cameron\\justin\\shrimPy\\examples\\acquisition_settings\\example_mda_sequence.yaml"

core = CMMCorePlus.instance()
core.loadSystemConfiguration(mmconfig_file)
#core.setPixelSizeConfig("Res40x")

# sequence = MDASequence(
#     channels=["BF", "GFP"],
#     stage_positions=[{"x": 1, "y": 1, "name": "some position"}, {"x": 0, "y": 0}],
#     time_plan={"interval": 1.0, "loops": 3},
#     z_plan={"range": 200, "step": 0.33333333},
#     axis_order="tpcz",
# )

sequence = MDASequence.from_file(mda_sequence_file)

# Set the ROI
core.setROI(0,512,2048,256)

# The following properties are necesary for proper sequencing
core.setProperty("TS_DAC01", "Sequence", "On")
core.setProperty("Core", "Focus", "AP Galvo")
core.setProperty("TS_TTL1-8", "Blanking", "On")

#with mda_listeners_connected(OMETiffWriter("example.ome.tiff")):
#    core.mda.run(sequence)
core.mda.run(sequence)
