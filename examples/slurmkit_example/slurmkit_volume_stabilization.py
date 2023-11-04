import datetime
import os
import glob
from mantis.cli import utils
from slurmkit import SlurmParams, slurm_function, submit_function
from natsort import natsorted
import click
from iohub import open_ome_zarr
from pathlib import Path
from mantis.cli.stabilization import estimate_YX_stabilization

# NOTE: this pipeline uses the focus found on well one for all. Perhaps this should be done per FOV(?)

# io parameters
input_position_dirpaths = "./input.zarr/0/0/0"
output_data_path = "./stabilized.zarr"

# convert to Path
output_data_path = Path(output_data_path)

# batch and resource parameters
cpus_per_task = 18
mem_per_cpu = "5G"
time = 60  # minutes
simultaneous_processes_per_node = 16
Z_CHUNK = 5
channel_for_stabilization = 0
CROP_XY = 300

# Path handling
input_position_dirpaths = natsorted(glob.glob(input_position_dirpaths))
output_dir = output_data_path.parent

output_paths = utils.get_output_paths(input_position_dirpaths, output_data_path)
click.echo(f"in: {input_position_dirpaths}, out: {output_paths}")
slurm_out_path = str(os.path.join(output_dir, "slurm_output/register-%j.out"))

# Additional registraion arguments
input_position = open_ome_zarr(input_position_dirpaths[0])  # take position 0 to get metadata
output_shape_zyx = input_position[0].shape[-3:]
output_voxel_size = input_position.scale[-3:]
chunk_zyx_shape = (Z_CHUNK, output_shape_zyx[-2], output_shape_zyx[-1])

# Estimate the stabilization transformations in XY
YX_shift_matrices = estimate_YX_stabilization(
    input_position=input_position,
    output_dir_paths=output_dir,
    c_idx=channel_for_stabilization,
    crop_xy=CROP_XY,
)

# Create the empty store
utils.create_empty_zarr(
    position_paths=input_position_dirpaths,
    output_path=output_data_path,
    output_zyx_shape=output_shape_zyx,
    chunk_zyx_shape=chunk_zyx_shape,
    voxel_size=tuple(output_voxel_size),
)

extra_metadata = {}
extra_arguments = {
    "extra_metadata": extra_metadata,
}

# prepare slurm parameters
params = SlurmParams(
    partition="preempted",
    cpus_per_task=cpus_per_task,
    mem_per_cpu=mem_per_cpu,
    time=datetime.timedelta(minutes=time),
    output=slurm_out_path,
)

# wrap our utils.process_single_position() function with slurmkit
slurm_process_single_position = slurm_function(utils.apply_stabilization_over_time_ants)
register_func = slurm_process_single_position(
    list_of_shifts_ants_style=YX_shift_matrices,
    num_processes=simultaneous_processes_per_node,
    **extra_arguments,
)

# generate an array of jobs by passing the in_path and out_path to slurm wrapped function
register_jobs = [
    submit_function(
        register_func,
        slurm_params=params,
        input_data_path=in_path,
        output_path=out_path,
    )
    for in_path, out_path in zip(input_position_dirpaths, output_paths)
]
