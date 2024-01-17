import datetime
import os
import glob
from mantis.analysis.register import affine_transform
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


# io parameters
input_position_dirpaths = "/hpc/projects/comp.micro/mantis/2023_08_18_A549_DRAQ5/1-phase/phase_filtered_rechunked.zarr/*/*/*"
target_position_dirpaths = "/hpc/projects/comp.micro/mantis/2023_08_18_A549_DRAQ5/2-deskew/a549_draq5_deskewed.zarr/0/10/0"
registration_mat_manual = "/hpc/projects/comp.micro/mantis/2023_08_18_A549_DRAQ5/3-registration/manual_reg/tx_manual.mat"
registration_mat_optimized = "/hpc/projects/comp.micro/mantis/2023_08_18_A549_DRAQ5/3-registration/manual_reg/tx_opt.mat"
output_data_path = "/hpc/projects/comp.micro/mantis/2023_08_18_A549_DRAQ5/3-registration/manual_reg/draq5_registered_phase.zarr"


# sbatch and resource parameters
cpus_per_task = 6
mem_per_cpu = "8G"
time = 40  # minutes
simultaneous_processes_per_node = 5

# path handling
input_position_dirpaths = natsorted(glob.glob(input_position_dirpaths))
target_position_dirpaths = natsorted(glob.glob(target_position_dirpaths))
target_position = open_ome_zarr(target_position_dirpaths[0])

output_dir = os.path.dirname(output_data_path)

output_paths = utils.get_output_paths(input_position_dirpaths, output_data_path)
click.echo(f"in: {input_position_dirpaths}, out: {output_paths}")
slurm_out_path = str(os.path.join(output_dir, "slurm_output/register-%j.out"))

# Additional registraion arguments
output_shape_zyx = target_position[0].shape[-3:]
output_voxel_size = target_position.scale[-3:]

# Convert string paths to Path objects
output_dirpath = Path(output_data_path)

# Create the empty store
utils.create_empty_zarr(
    position_paths=input_position_dirpaths,
    output_path=output_dirpath,
    output_zyx_shape=output_shape_zyx,
    chunk_zyx_shape=None,
    voxel_size=tuple(output_voxel_size),
)

ants_composed_matrix_list = [registration_mat_optimized, registration_mat_manual]
extra_metadata = {}
affine_transform_args = {
    "ants_transform_file_list": ants_composed_matrix_list,
    "output_shape_zyx": output_shape_zyx,
    "extra_metadata": extra_metadata,
}

# prepare slurm parameters
params = SlurmParams(
    partition="cpu",
    cpus_per_task=cpus_per_task,
    mem_per_cpu=mem_per_cpu,
    time=datetime.timedelta(minutes=time),
    output=slurm_out_path,
)

# wrap our utils.process_single_position() function with slurmkit
slurm_process_single_position = slurm_function(utils.process_single_position)
register_func = slurm_process_single_position(
    func=affine_transform,
    num_processes=simultaneous_processes_per_node,
    **affine_transform_args,
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
