import datetime
import glob

from pathlib import Path

import numpy as np

from iohub import open_ome_zarr
from natsort import natsorted
from slurmkit import SlurmParams, slurm_function, submit_function

from mantis.analysis.stitch import get_stitch_output_shape, calculate_shift, shift_image, get_grid_rows_cols

from mantis.cli.utils import (
    create_empty_hcs_zarr,
    process_single_position_v2,
)

col_translation = (967.9, -7.45)
row_translation = (7.78, 969)

# io parameters
# input_paths = "/hpc/projects/intracellular_dashboard/ops/2024_03_05_registration_test/fixed/2-virtual-stain/fcmae-2d/mean_projection/A3_20x_38x38_1.zarr/*/*/*"
# output_data_path = "/hpc/projects/intracellular_dashboard/ops/2024_03_05_registration_test/fixed/test-register/large_stitch_test.zarr"
# channels = ['Nucleus_prediction']

input_paths = "/hpc/projects/intracellular_dashboard/ops/2024_03_05_registration_test/fixed/0-convert/grid_test_3.zarr/*/*/*"
output_data_path = "/hpc/projects/intracellular_dashboard/ops/2024_03_05_registration_test/fixed/test-register/small_stitch_test.zarr"
channels = ['Default']

# sbatch and resource parameters
cpus_per_task = 1
mem_per_cpu = "16G"
time = 10  # minutes
partition = 'cpu'

# NOTE: parameters from here and below should not have to be changed
input_paths = [Path(path) for path in natsorted(glob.glob(input_paths))]
output_path = Path(output_data_path)
slurm_out_path = Path(output_data_path).parent / "slurm_output" / "stitch-%j.out"

with open_ome_zarr(str(input_paths[0]), mode="r") as input_dataset:
    dataset_channel_names = input_dataset.channel_names
    T, C, Z, Y, X = input_dataset.data.shape
    # scale = tuple(input_dataset.scale)

grid_rows, grid_cols = get_grid_rows_cols(Path(*input_paths[0].parts[:-3]))
n_rows = len(grid_rows)
n_cols = len(grid_cols)

output_shape, global_translation = get_stitch_output_shape(
    n_rows, n_cols, Y, X, col_translation, row_translation
)

# Create the output zarr mirroring input positions
# Takes a while
create_empty_hcs_zarr(
    store_path=output_data_path,
    position_keys=[p.parts[-3:] for p in input_paths],
    shape=(T, len(channels), Z) + output_shape,
    chunks=(1, 1, 1, 4096, 4096),
    # scale=scale,
    channel_names=channels,
    dtype=np.float32,
)

# debug
# process_single_position_v2(
#     shift_image,
#     input_data_path=input_paths[0],
#     output_path=output_path,
#     input_channel_idx=[dataset_channel_names.index(ch) for ch in channels],
#     output_channel_idx=list(range(len(channels))),
#     num_processes=cpus_per_task,
#     output_shape=output_shape,
#     shift=(0, 0),
#     verbose=True,
# )

# prepare slurm parameters
params = SlurmParams(
    partition=partition,
    cpus_per_task=cpus_per_task,
    mem_per_cpu=mem_per_cpu,
    time=datetime.timedelta(minutes=time),
    output=slurm_out_path,
)

# wrap our deskew_single_position() function with slurmkit
slurm_func = slurm_function(process_single_position_v2)

# generate an array of jobs by passing the in_path and out_path to slurm wrapped function
for in_path in input_paths:
    col_idx, row_idx = (int(in_path.name[:3]), int(in_path.name[3:]))
    shift = calculate_shift(
        col_idx, row_idx, col_translation, row_translation, global_translation
    )

    func = slurm_func(
        shift_image,
        input_channel_idx=[dataset_channel_names.index(ch) for ch in channels],
        output_channel_idx=list(range(len(channels))),
        num_processes=cpus_per_task,
        output_shape=output_shape,
        shift=shift,
        verbose=True,
    )

    submit_function(
        func,
        slurm_params=params,
        input_data_path=in_path,
        output_path=output_path,
    )
