# /// requires
# ome-writers[all]==v0.1.0rc1
# pymmcore-plus>=0.16.0
# ///

from pathlib import Path

import numpy as np
import shutil
import time

from ome_writers import (
    AcquisitionSettings,
    create_stream,
    dims_from_useq,
    useq_to_acquisition_settings,
)

from useq import MDASequence, MDAEvent, WellPlatePlan, GridRowsColumns
from useq._position import Position
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda import MDAEngine
from pymmcore_plus.core._sequencing import SequencedEvent

core = CMMCorePlus.instance()
core.loadSystemConfiguration()
# Set realistic ROI for mantis microscope
core.setProperty("Camera", "OnCameraCCDXSize", "2048")
core.setProperty("Camera", "OnCameraCCDYSize", "256")
core.setProperty("XY", "Velocity", "10000")
# Enable Z sequencing
core.setProperty("Z", "UseSequences", "Yes")

data_dir = Path("~/Documents/test/test_14_acq_zarr").expanduser()
# stage_positions = tuple(
#     Position(
#         x=row, 
#         y=col, 
#         name=f"Pos{row * 4 + col}", 
#         row=str(row), 
#         col=str(col + 1)
#     )
#     for row in range(2)   # 2 rows
#     for col in range(4)   # 4 columns
# )

stage_positions = WellPlatePlan(
    plate="12-well",
    a1_center_xy=(0, 0),
    selected_wells=((0, 1), (0)),
    well_points_plan=GridRowsColumns(rows=2, columns=2),
)

seq = MDASequence(
    stage_positions=stage_positions,
    time_plan={"interval": 1, "loops": 5},
    channels=(
        {"config": "DAPI", "exposure": 2},
        {"config": "FITC", "exposure": 10},
    ),
    z_plan={"range": 10, "step": 0.2},
    axis_order="tpcz",
)

# # Convert the MDASequence to ome_writers dimensions
image_width = core.getImageWidth()
image_height = core.getImageHeight()
pixel_size_um = core.getPixelSizeUm()
dtype = f"uint{core.getImageBitDepth()}"
chunk_shapes = {"t": 1, "c": 1, "z": 32, "y": image_height, "x": image_width}
BACKEND = "acquire-zarr"
# BACKEND = "tifffile"
suffix = ".ome.tiff" if BACKEND == "tifffile" else ".ome.zarr"

acq_settings = useq_to_acquisition_settings(
    seq, image_width=image_width, image_height=image_height, pixel_size_um=pixel_size_um, chunk_shapes=chunk_shapes
)

skip_acquisition_indices = [
    {'t': 1, 'p': 0},
    {'t': 1, 'p': 1},
    {'t': 2, 'p': 0},
    {'t': 5, 'p': 0},
    {'t': 5, 'p': 1},
]
# Subclass acquisition engine to skip acquisition for certain timepoints and positions,
# simulating O1 autofocus failure.
class MyEngine(MDAEngine):
    def exec_event(self, event: MDAEvent):
        if isinstance(event, SequencedEvent):
            # Similated autofocus failure only implemented for sequenced events
            first_event = event.events[0]
            event_pt_idx = {'t': first_event.index['t'], 'p': first_event.index['p']}
            if event_pt_idx in skip_acquisition_indices:
                # For preset (P,T) indices, pad with zeros instead of acquiring data
                print(f"Skipping acquisition for event index {first_event.index}")
                _img = np.zeros((image_height, image_width), dtype=dtype)
                for _event in event.events:
                    yield (_img, _event, self.get_frame_metadata(_event))
            else:
                yield from super().exec_event(event)
        else:
            yield from super().exec_event(event)

if not data_dir.exists():
    data_dir.mkdir(parents=True)
else:
    shutil.rmtree(data_dir)

settings = AcquisitionSettings(
    root_path=data_dir / f"example_acq{suffix}",
    dtype=dtype,
    # Specify compression
    compression="blosc-zstd" if BACKEND == "acquire-zarr" else None,
    format=BACKEND,
    overwrite=False,
    **acq_settings,
)
stream = create_stream(settings)

@core.mda.events.frameReady.connect
def _on_frame(frame: np.ndarray, event: MDAEvent, metadata: dict) -> None:
    stream.append(frame)


start_time = time.time()
core.mda.set_engine(MyEngine(core))
core.mda.run(seq)
end_time = time.time()

stream.close()
print(f"Time taken: {end_time - start_time} seconds")
