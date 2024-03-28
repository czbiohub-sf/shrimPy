import datetime
import glob

from pathlib import Path
import click

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
verbose = True

# io parameters
# dataset = 'B3_600k_20x20_timelapse_1'
# channels = ['Nucleus_prediction']
# input_paths = f"/hpc/projects/intracellular_dashboard/ops/2024_03_05_registration_test/live/2-virtual-stain/fcmae-2d/mean_projection/{dataset}.zarr/*/*/*"
# temp_path = Path(f"/hpc/projects/intracellular_dashboard/ops/2024_03_05_registration_test/live/3-stitch/TEMP_{dataset}.zarr")
# output_path = Path(f"/hpc/projects/intracellular_dashboard/ops/2024_03_05_registration_test/live/3-stitch/{dataset}.zarr")

dataset = 'grid_test_3'
channels = ['Default']
input_paths = f"/hpc/projects/intracellular_dashboard/ops/2024_03_05_registration_test/fixed/0-convert/{dataset}.zarr/*/*/*"
temp_path = Path(f"/hpc/projects/intracellular_dashboard/ops/2024_03_05_registration_test/fixed/test-register/TEMP_{dataset}.zarr")
output_path = Path(f"/hpc/projects/intracellular_dashboard/ops/2024_03_05_registration_test/fixed/test-register/{dataset}.zarr")

# sbatch and resource parameters
cpus_per_task = 1
mem_per_cpu = "16G"
time = 10  # minutes
partition = 'cpu'


def stitch_shifted_store(input_data_path, output_data_path):
    click.echo(f'Stitching zarr store: {input_data_path}')
    with open_ome_zarr(input_data_path, mode="r") as input_dataset:
        well_name, _ = next(input_dataset.wells())
        _, sample_position = next(input_dataset.positions())
        array_shape = sample_position.data.shape
        channels = input_dataset.channel_names

        stitched_array = np.zeros(array_shape, dtype=np.float32)
        denominator = np.zeros(array_shape, dtype=np.uint8)

        j = 0
        for _, position in input_dataset.positions():
            if verbose:
                click.echo(f'Processing position {j}')
            stitched_array += position.data
            denominator += np.bool_(position.data)
            j += 1

    denominator[denominator == 0] = 1
    stitched_array /= denominator

    click.echo(f'Saving stitched array in :{output_data_path}')
    with open_ome_zarr(
        output_data_path,
        layout='hcs',
        channel_names=channels,
        mode="w-"
    ) as output_dataset:
        position = output_dataset.create_position(*Path(well_name, '0').parts)
        position.create_image('0', stitched_array, chunks=(1, 1, 1, 4096, 4096))


# NOTE: parameters from here and below should not have to be changed
input_paths = [Path(path) for path in natsorted(glob.glob(input_paths))]
slurm_out_path = temp_path.parent / "slurm_output" / "stitch-%j.out"

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
click.echo('Creating output zarr store')
create_empty_hcs_zarr(
    store_path=temp_path,
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
#     time_indices='all',
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
click.echo('Submitting SLURM jobs')
jobs = []
for in_path in input_paths:
    col_idx, row_idx = (int(in_path.name[:3]), int(in_path.name[3:]))
    shift = calculate_shift(
        col_idx, row_idx, col_translation, row_translation, global_translation
    )

    func = slurm_func(
        shift_image,
        time_indices='all',
        input_channel_idx=[dataset_channel_names.index(ch) for ch in channels],
        output_channel_idx=list(range(len(channels))),
        num_processes=cpus_per_task,
        output_shape=output_shape,
        shift=shift,
        verbose=True,
    )

    jobs.append(
        submit_function(
            func,
            slurm_params=params,
            input_data_path=in_path,
            output_path=temp_path,
        )
    )

submit_function(
    slurm_function(stitch_shifted_store)(temp_path, output_path),
    slurm_params=SlurmParams(
        partition=partition,
        cpus_per_task=8,
        mem_per_cpu='16G',
        time=datetime.timedelta(hours=12),
        output=slurm_out_path,
    ),
    dependencies=jobs,
)
