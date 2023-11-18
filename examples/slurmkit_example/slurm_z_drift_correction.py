import datetime
import os
import glob
from mantis.cli import utils
from slurmkit import SlurmParams, slurm_function, submit_function
from natsort import natsorted
import click
from mantis.cli.apply_affine import (
    registration_params_from_file,
    rotate_n_affine_transform,
)
import numpy as np
from iohub import open_ome_zarr
from pathlib import Path
import ants
from mantis.cli.stabilization import calculate_z_drift

# NOTE: this pipeline uses the focus found on well one for all. Perhaps this should be done per FOV(?)

# io parameters
input_position_dirpaths = "./timelapse_1/registered_output.zarr/*/*/*"
z_shifts_matrices_filepath = Path("./z_shifts.npy")
output_data_path = "./registered_phase_z_stabilized_all_positions.zarr"

# sbatch and resource parameters
cpus_per_task = 32
mem_per_cpu = "4G"
time = 80  # minutes
simultaneous_processes_per_node = 30
Z_CHUNK = 5


# Path handling
input_position_dirpaths = natsorted(glob.glob(input_position_dirpaths))
output_dir = os.path.dirname(output_data_path)

output_paths = utils.get_output_paths(input_position_dirpaths, output_data_path)
click.echo(f"in: {input_position_dirpaths}, out: {output_paths}")
slurm_out_path = str(os.path.join(output_dir, "slurm_output/register-%j.out"))

# Additional registraion arguments
input_position = open_ome_zarr(input_position_dirpaths[0])  # take position 0 to get metadata
output_shape_zyx = input_position[0].shape[-3:]
output_voxel_size = input_position.scale[-3:]
chunk_zyx_shape = (Z_CHUNK, output_shape_zyx[-2], output_shape_zyx[-1])

# Convert string paths to Path objects
output_dirpath = Path(output_data_path)

# Create the empty store
utils.create_empty_zarr(
    position_paths=input_position_dirpaths,
    output_path=output_dirpath,
    output_zyx_shape=output_shape_zyx,
    chunk_zyx_shape=chunk_zyx_shape,
    voxel_size=tuple(output_voxel_size),
)

assert z_shifts_matrices_filepath.exists()
z_shifts_matrices = np.load(z_shifts_matrices_filepath)

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
    list_of_shifts=z_shifts_matrices,
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
