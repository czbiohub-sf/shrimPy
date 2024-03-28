# %%
from pathlib import Path
import numpy as np
from iohub import open_ome_zarr

input_data_path = "/hpc/projects/intracellular_dashboard/ops/2024_03_05_registration_test/fixed/test-register/small_stitch_test.zarr"
output_data_path = "/hpc/projects/intracellular_dashboard/ops/2024_03_05_registration_test/fixed/test-register/small_stitch_test_stitched.zarr"

# %%
with open_ome_zarr(input_data_path, mode="r") as input_dataset:
    well_name, _ = next(input_dataset.wells())
    _, sample_position = next(input_dataset.positions())
    array_shape = sample_position.data.shape
    channels = input_dataset.channel_names

    stitched_array = np.zeros(array_shape, dtype=np.float32)
    denominator = np.zeros(array_shape, dtype=np.uint8)

    for position_name, position in input_dataset.positions():
        stitched_array += position.data
        denominator += np.bool_(position.data)

denominator[denominator == 0] = 1
stitched_array /= denominator

with open_ome_zarr(
    output_data_path,
    layout='hcs',
    channel_names=channels,
    mode="w-"
) as output_dataset:
    position = output_dataset.create_position(*Path(well_name, '0').parts)
    position.create_image('0', stitched_array, chunks=(1, 1, 1, 4096, 4096))

# %%
