import datetime
import glob


from pathlib import Path

import click
import numpy as np

from iohub import open_ome_zarr
from natsort import natsorted
from slurmkit import SlurmParams, slurm_function, submit_function

from mantis.cli.utils import (
    copy_n_paste_czyx,
    create_empty_hcs_zarr,
    process_single_position_v2,
)

# io parameters
input_position_dirpath = '/input_source.zarr/*/*/*'
output_dirpath = './test_output.zarr'

# sbatch and resource parameters
cpus_per_task = 4
mem_per_cpu = "16G"
time = 60  # minutes
partition = 'cpu'
simultaneous_processes_per_node = (
    8  # number of processes that are run in parallel on a single node
)

# NOTE: parameters from here and below should not have to be changed
input_position_dirpath = [Path(path) for path in natsorted(glob.glob(input_position_dirpath))]
output_dirpath = Path(output_dirpath)

click.echo(f"in_path: {input_position_dirpath[0]}, out_path: {output_dirpath}")
slurm_out_path = output_dirpath.parent / "slurm_output" / "register-%j.out"


# Calculate the output voxel size from the input scale and affine transform
with open_ome_zarr(input_position_dirpath[0]) as input_dataset:
    T, C, Z, Y, X = input_dataset.data.shape
    channel_names = input_dataset.channel_names
    output_voxel_size = input_dataset.data.shape[-3:]
    input_voxel_size = input_dataset.scale[-3:]

# Channels to process
channels_to_process = ['Channel_1', 'Channel_2']

# Slicing parameters used for cropping/copy_n_paste
T_slice = slice(0, T)
Z_slice = slice(0, Z)
Y_slice = slice(0, Y)
X_slice = slice(0, X)

# TODO: start or stop may be None
cropped_shape_zyx = (
    Z_slice.stop - Z_slice.start,
    Y_slice.stop - Y_slice.start,
    X_slice.stop - X_slice.start,
)

# Overwrite the previous target shape
Z_target, Y_target, X_target = cropped_shape_zyx[-3:]
time_indices = list(T_slice)

# Logic to know what channels to process
input_channel_idx = []
output_channel_idx = []
i = 0
for chan in channels_to_process:
    if chan in channel_names:
        input_channel_idx.append(channel_names.index(chan))
        output_channel_idx.append(i)
        i += 1
    else:
        raise ValueError(f"Channel {chan} not found in input dataset")

click.echo(f'Shape of cropped output dataset (z,y,x): {cropped_shape_zyx}\n')

output_metadata = {
    "shape": (len(time_indices), len(output_channel_idx), Z_target, Y_target, X_target),
    "chunks": None,
    "scale": (1,) * 2 + tuple(output_voxel_size),
    "channel_names": channels_to_process,
    "dtype": np.float32,
}

# Create the output zarr mirroring input_position_dirpath
create_empty_hcs_zarr(
    store_path=output_dirpath,
    position_keys=[p.parts[-3:] for p in input_position_dirpath],
    **output_metadata,
)

# Copy_n_paste arguments
copy_n_paste_kwargs = {"czyx_slicing_params": ([Z_slice, Y_slice, X_slice])}

# prepare slurm parameters
params = SlurmParams(
    partition=partition,
    cpus_per_task=cpus_per_task,
    mem_per_cpu=mem_per_cpu,
    time=datetime.timedelta(minutes=time),
    output=slurm_out_path,
)

# wrap our utils.process_single_position() function with slurmkit
slurm_process_single_position = slurm_function(process_single_position_v2)

copy_n_paste_func = slurm_process_single_position(
    func=copy_n_paste_czyx,
    output_path=output_dirpath,
    time_indices=time_indices,
    num_processes=simultaneous_processes_per_node,
    **copy_n_paste_kwargs,
)

# Copy over the channels that were not processed
for input_position_path in input_position_dirpath:
    submit_function(
        copy_n_paste_func,
        slurm_params=params,
        input_data_path=input_position_path,
    )
