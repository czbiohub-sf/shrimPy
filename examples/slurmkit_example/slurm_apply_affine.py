import datetime
import glob
import os

from pathlib import Path

import click
import numpy as np

from iohub import open_ome_zarr
from natsort import natsorted
from slurmkit import SlurmParams, slurm_function, submit_function

from mantis.analysis.AnalysisSettings import RegistrationSettings
from mantis.analysis.register import apply_affine_transform, find_overlapping_volume
from mantis.cli.apply_affine import rescale_voxel_size
from mantis.cli.utils import (
    copy_n_paste_czyx,
    create_empty_hcs_zarr,
    process_single_position_v2,
    yaml_to_model,
)

# io parameters
source_position_dirpaths = '/input_source.zarr/*/*/*'
target_position_dirpaths = '/input_target.zarr/*/*/*'
config_filepath = (
    '../mantis/analysis/settings/example_apply_affine_settings.yml'
)
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
source_position_dirpaths = [
    Path(path) for path in natsorted(glob.glob(source_position_dirpaths))
]
target_position_dirpaths = [
    Path(path) for path in natsorted(glob.glob(target_position_dirpaths))
]
output_dirpath = Path(output_dirpath)
config_filepath = Path(config_filepath)

click.echo(f"in_path: {source_position_dirpaths[0]}, out_path: {output_dirpath}")
slurm_out_path = output_dirpath.parent / "slurm_output" / "register-%j.out"

# Parse from the yaml file
settings = yaml_to_model(config_filepath, RegistrationSettings)
matrix = np.array(settings.affine_transform_zyx)
keep_overhang = settings.keep_overhang

# Calculate the output voxel size from the input scale and affine transform
with open_ome_zarr(source_position_dirpaths[0]) as source_dataset:
    T, C, Z, Y, X = source_dataset.data.shape
    source_channel_names = source_dataset.channel_names
    source_shape_zyx = source_dataset.data.shape[-3:]
    source_voxel_size = source_dataset.scale[-3:]
    output_voxel_size = rescale_voxel_size(matrix[:3, :3], source_voxel_size)

with open_ome_zarr(target_position_dirpaths[0]) as target_dataset:
    target_channel_names = target_dataset.channel_names
    Z_target, Y_target, X_target = target_dataset.data.shape[-3:]
    target_shape_zyx = target_dataset.data.shape[-3:]

click.echo('\nREGISTRATION PARAMETERS:')
click.echo(f'Transformation matrix:\n{matrix}')
click.echo(f'Voxel size: {output_voxel_size}')

# Logic to parse time indices
if settings.time_indices == "all":
    time_indices = list(range(T))
elif isinstance(settings.time_indices, list):
    time_indices = settings.time_indices
elif isinstance(settings.time_indices, int):
    time_indices = [settings.time_indices]

output_channel_names = target_channel_names
if target_position_dirpaths != source_position_dirpaths:
    output_channel_names += source_channel_names

if not keep_overhang:
    # Find the largest interior rectangle
    click.echo('\nFinding largest overlapping volume between source and target datasets')
    Z_slice, Y_slice, X_slice = find_overlapping_volume(
        source_shape_zyx, target_shape_zyx, matrix
    )
    # TODO: start or stop may be None
    cropped_target_shape_zyx = (
        Z_slice.stop - Z_slice.start,
        Y_slice.stop - Y_slice.start,
        X_slice.stop - X_slice.start,
    )
    # Overwrite the previous target shape
    Z_target, Y_target, X_target = cropped_target_shape_zyx[-3:]
    click.echo(f'Shape of cropped output dataset: {target_shape_zyx}\n')
else:
    Z_slice, Y_slice, X_slice = (
        slice(0, Z_target),
        slice(0, Y_target),
        slice(0, X_target),
    )

output_metadata = {
    "shape": (len(time_indices), len(output_channel_names), Z_target, Y_target, X_target),
    "chunks": None,
    "scale": (1,) * 2 + tuple(output_voxel_size),
    "channel_names": output_channel_names,
    "dtype": np.float32,
}

# Create the output zarr mirroring source_position_dirpaths
create_empty_hcs_zarr(
    store_path=output_dirpath,
    position_keys=[p.parts[-3:] for p in source_position_dirpaths],
    **output_metadata,
)

# Get the affine transformation matrix
# NOTE: add any extra metadata if needed:
extra_metadata = {
    'affine_transformation': {
        'transform_matrix': matrix.tolist(),
    }
}

affine_transform_args = {
    'matrix': matrix,
    'output_shape_zyx': target_shape_zyx,  # NOTE: this is the shape of the original target dataset
    'crop_output_slicing': ([Z_slice, Y_slice, X_slice] if not keep_overhang else None),
    'extra_metadata': extra_metadata,
}

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
register_func = slurm_process_single_position(
    func=apply_affine_transform,
    output_path=output_dirpath,
    time_indices=time_indices,
    num_processes=simultaneous_processes_per_node,
    **affine_transform_args,
)

copy_n_paste_func = slurm_process_single_position(
    func=copy_n_paste_czyx,
    output_path=output_dirpath,
    time_indices=time_indices,
    num_processes=simultaneous_processes_per_node,
    **copy_n_paste_kwargs,
)

# NOTE: channels will not be processed in parallel
# NOTE: the the source and target datastores may be the same (e.g. Hummingbird datasets)
# apply affine transform to channels in the source datastore that should be registered
# as given in the config file (i.e. settings.source_channel_names)
for input_position_path in source_position_dirpaths:
    for channel_name in source_channel_names:
        if channel_name in settings.source_channel_names:
            submit_function(
                register_func,
                slurm_params=params,
                input_data_path=input_position_path,
                input_channel_idx=[source_channel_names.index(channel_name)],
                output_channel_idx=[output_channel_names.index(channel_name)],
            )

# Copy over the channels that were not processed
for input_position_path in target_position_dirpaths:
    for channel_name in target_channel_names:
        if channel_name not in settings.source_channel_names:
            submit_function(
                copy_n_paste_func,
                slurm_params=params,
                input_data_path=input_position_path,
                input_channel_idx=[target_channel_names.index(channel_name)],
                output_channel_idx=[output_channel_names.index(channel_name)],
            )
