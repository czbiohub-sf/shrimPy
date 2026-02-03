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
)

from useq import MDASequence, MDAEvent
from useq._position import Position
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda import MDAEngine
from pymmcore_plus.core._sequencing import SequencedEvent

core = CMMCorePlus.instance()
core.loadSystemConfiguration()
# Set realistic ROI for mantis microscope
core.setProperty("Camera", "OnCameraCCDXSize", "2048")
core.setProperty("Camera", "OnCameraCCDYSize", "256")
# Enable Z sequencing
core.setProperty("Z", "UseSequences", "Yes")

data_dir = Path("~/Documents/test/test_14_acq_zarr").expanduser()
if not data_dir.exists():
    data_dir.mkdir(parents=True)
else:
    shutil.rmtree(data_dir)

seq = MDASequence(
    channels=(
        {"config": "DAPI", "exposure": 2},
        {"config": "FITC", "exposure": 10},
    ),
    stage_positions=(
        Position(x=0, y=0, name="Pos0", row="0", col="1"),
        Position(x=1, y=1, name="Pos1", row="0", col="1"),
        Position(x=0, y=1, name="Pos2", row="0", col="2"),
        Position(x=1, y=0, name="Pos3", row="0", col="2"),
    ),
    # stage_positions=(
    #     {"x": 0, "y": 0, "name": "Pos0"},
    #     {"x": 1, "y": 1, "name": "Pos1"},
    #     {"x": 0, "y": 1, "name": "Pos2"},
    #     {"x": 1, "y": 0, "name": "Pos3"},
    # ),
    time_plan={"interval": 1, "loops": 5},
    z_plan={"range": 10, "step": 0.2},
    axis_order="tpcz",
)

# # Convert the MDASequence to ome_writers dimensions
image_width = core.getImageWidth()
image_height = core.getImageHeight()
pixel_size_um = core.getPixelSizeUm()
dtype = f"uint{core.getImageBitDepth()}"
BACKEND = "acquire-zarr"
# BACKEND = "tifffile"
suffix = ".ome.tiff" if BACKEND == "tifffile" else ".ome.zarr"

# TODO: current it's not possible to convert HCS plate format from MDASequence to ome_writers dimensions
# need to update dims_from_useq function
dims = dims_from_useq(
    seq, image_width=image_width, image_height=image_height, pixel_size_um=pixel_size_um
)
# Specify chunk shapes for each dimension, could be part of dims_from_useq
chunk_shapes = {"t": 1, "c": 1, "z": 32, "y": image_height, "x": image_width}
for dim in dims:
    if dim.name in chunk_shapes:
        dim.chunk_size = chunk_shapes[dim.name]

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


settings = AcquisitionSettings(
    root_path=data_dir / f"example_acq{suffix}",
    dimensions=dims,
    dtype=dtype,
    # Specify compression
    compression="blosc-zstd" if BACKEND == "acquire-zarr" else None,
    backend=BACKEND,
    overwrite=False,
)
stream = create_stream(settings)

@core.mda.events.frameReady.connect
def _on_frame(frame: np.ndarray, event: MDAEvent, metadata: dict) -> None:
    stream.append(frame)


start_time = time.time()
core.mda.set_engine(MyEngine(core))
core.mda.run(seq)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
