import datetime
import os
import glob
from mantis.cli import utils
from slurmkit import SlurmParams, slurm_function, submit_function
from natsort import natsorted
import click
from iohub import open_ome_zarr
from pathlib import Path
from mantis.cli.stabilization import calculate_z_drift, calculate_yx_stabilization
import numpy as np

# NOTE: this pipeline uses the focus found on well one for all. Perhaps this should be done per FOV(?)

# io parameters
input_position_dirpaths = "/hpc/projects/comp.micro/mantis/2023_11_08_Opencell_infection/4-registration/OC43_infection_timelapse_3_registered.zarr/*/*/*"
output_data_path = "./OC43_infection_timelapse_3_registered_stabilized_2.zarr"

# convert to Path
output_data_path = Path(output_data_path)

# batch and resource parameters
cpus_per_task = 20
mem_per_cpu = "8G"
time = 300  # minutes
simultaneous_processes_per_node = 12
Z_CHUNK = 5
channel_for_stabilization = 0
crop_size_xy = [300, 300]
stabilization_verbose = True
estimate_z_drift = True
estimate_yx_drift = False
stabilzation_channel_index = 0

# Path handling
input_position_dirpaths = [
    Path(path) for path in natsorted(glob.glob(input_position_dirpaths))
]
output_dir = output_data_path.parent

output_paths = utils.get_output_paths(input_position_dirpaths, output_data_path)
click.echo(f"in: {input_position_dirpaths}, out: {output_paths}")
slurm_out_path = str(os.path.join(output_dir, "slurm_output/register-%j.out"))

# Additional registraion arguments
input_position = open_ome_zarr(input_position_dirpaths[0])  # take position 0 to get metadata
output_shape_zyx = input_position[0].shape[-3:]
output_voxel_size = input_position.scale[-3:]
chunk_zyx_shape = (Z_CHUNK, output_shape_zyx[-2], output_shape_zyx[-1])

# Estimate z drift
if estimate_z_drift:
    T_z_drift_mats = calculate_z_drift(
        input_data_paths=input_position_dirpaths,
        output_folder_path=output_dir,
        num_processes=5,
        crop_size_xy=crop_size_xy,
        verbose=stabilization_verbose,
    )
    if not estimate_yx_drift:
        combined_mats = T_z_drift_mats

# Estimate yx drift
if estimate_yx_drift:
    T_translation_mats = calculate_yx_stabilization(
        input_data_path=input_position_dirpaths,
        output_folder_path=output_dir,
        c_idx=stabilzation_channel_index,
        crop_size_xy=crop_size_xy,
        verbose=stabilization_verbose,
    )
    if estimate_z_drift:
        if T_translation_mats.shape[0] != T_z_drift_mats.shape[0]:
            raise ValueError(
                "The number of translation matrices and z drift matrices must be the same"
            )
        else:
            combined_mats = [
                np.dot(T_translation_mat, T_z_drift_mat)
                for T_translation_mat, T_z_drift_mat in zip(T_translation_mats, T_z_drift_mats)
            ]

    else:
        combined_mats = T_translation_mats

# Save the combined matrices
print(f'Drift correction matrices: {combined_mats}')
combined_mats_filepath = output_dir / "combined_mats.npy"
np.save(combined_mats_filepath, combined_mats)

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
    list_of_shifts=combined_mats,
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
