import largestinteriorrectangle as lir
import datetime
import os
import glob
from mantis.cli import utils
from slurmkit import SlurmParams, slurm_function, submit_function
from natsort import natsorted
import click
import numpy as np
from iohub import open_ome_zarr
from pathlib import Path
import ants
import datetime
import os
import glob
from mantis.cli import utils
from slurmkit import SlurmParams, slurm_function, submit_function
from natsort import natsorted
import click
from iohub import open_ome_zarr
from dataclasses import asdict

cropping_methods = {
    "all": 0,
    "custom_roi": 1,
    "custom_slices": 2,
}

# Get two sample FOVs from the target and source respectively
input_position_dirpaths = "/hpc/projects/comp.micro/mantis/2023_08_18_A549_DRAQ5/3-registration/manual_reg/draq5_registered_phase.zarr/0/0/0"
output_data_path = "./cropped.zarr"

# Used for finding the largestinteriorrectangle
raw_input_position_dirpaths = "/hpc/projects/comp.micro/mantis/2023_08_18_A549_DRAQ5/1-phase/phase_filtered_rechunked.zarr/0/0/0"
target_position_dirpaths = "/hpc/projects/comp.micro/mantis/2023_08_18_A549_DRAQ5/2-deskew/a549_draq5_deskewed.zarr/0/0/0"

# Load the transforms
registration_mat_manual = "/hpc/projects/comp.micro/mantis/2023_08_18_A549_DRAQ5/3-registration/manual_reg/tx_manual.mat"
registration_mat_optimized = "/hpc/projects/comp.micro/mantis/2023_08_18_A549_DRAQ5/3-registration/manual_reg/tx_opt.mat"

# Z-chunking factor. Big datasets require z-chunking to avoid blocs issues
Z_CHUNK = 5  # CHUNK XY ZYX or non will be 500MB
cropping_mode = "custom_slices"
indices_to_remove = []

# sbatch and resource parameters
cpus_per_task = 3
mem_per_cpu = "8G"
time = 40  # minutes
simultaneous_processes_per_node = 3

# path handling
input_paths = [Path(path) for path in natsorted(glob.glob(input_position_dirpaths))]
target_paths = [Path(path) for path in natsorted(glob.glob(target_position_dirpaths))]
raw_input_paths = [
    Path(path) for path in natsorted(glob.glob(raw_input_position_dirpaths))
]

output_data_path = Path(output_data_path)
output_dir = output_data_path.parent
output_paths = utils.get_output_paths(input_paths, output_data_path)
slurm_out_path = str(os.path.join(output_dir, "slurm_output/deskew-%j.out"))

# Remove desired rois
input_paths = [
    element
    for index, element in enumerate(input_paths)
    if index not in indices_to_remove
]
output_paths = [
    element
    for index, element in enumerate(output_paths)
    if index not in indices_to_remove
]
print(f"Removing positions idx: {indices_to_remove}")
click.echo(f"in: {input_paths}, out: {output_paths}")

with open_ome_zarr(input_paths[0], mode="r") as input_dataset:
    Z_input, Y_input, X_input = input_dataset.data.shape[-3:]
    voxel_size = tuple(input_dataset.scale[-3:])

with open_ome_zarr(target_paths[0], mode="r") as target_dataset:
    Z_target, Y_target, X_target = target_dataset.data.shape[-3:]

with open_ome_zarr(raw_input_paths[0], mode="r") as raw_phase_dataset:
    Z_raw, Y_raw, X_raw = raw_phase_dataset.data.shape[-3:]


# Additional deskew arguments
# Get the deskewing parameters
# Load the first position to infer dataset information
if cropping_mode in cropping_methods:
    method = cropping_methods[cropping_mode]
    if method == 0:
        # # NOTE :crop here and chunksize
        # # Slicing Parameters
        Z_slice = slice(None)
        Y_slice = slice(None)
        X_slice = slice(None)
        chunk_zyx_shape = None
        output_shape_zyx = (Z_input, Y_input, X_input)

    elif method == 1:
        assert HALF_CROP_LENGTH is not None
        Z_slice = slice(None)
        Y_slice = slice(
            Y_input // 2 - HALF_CROP_LENGTH, Y_input // 2 + HALF_CROP_LENGTH
        )
        X_slice = slice(
            X_input // 2 - HALF_CROP_LENGTH, X_input // 2 + HALF_CROP_LENGTH
        )
        chunk_zyx_shape = (Z_CHUNK, output_shape_zyx[-2], output_shape_zyx[-1])
        output_shape_zyx = (
            Z_input,
            int(Y_slice.stop - Y_slice.start),
            int(X_slice.stop - X_slice.start),
        )
    elif method == 2:
        # # NOTE :crop here and chunksize
        # # Slicing Parameters
        Z_slice, Y_slice, X_slice = utils.find_lir_slicing_params(
            (Z_raw, Y_raw, X_raw),
            (Z_target, Y_target, X_target),
            registration_mat_optimized,
            registration_mat_manual,
        )
        output_shape_zyx = (
            int(Z_slice.stop - Z_slice.start),
            int(Y_slice.stop - Y_slice.start),
            int(X_slice.stop - X_slice.start),
        )
        chunk_zyx_shape = (Z_CHUNK, output_shape_zyx[-2], output_shape_zyx[-1])

    zyx_slicing_params = [Z_slice, Y_slice, X_slice]

else:
    raise ValueError("Cropping mode not in cropping_methods")


# Create a zarr store output to mirror the input
utils.create_empty_zarr(
    position_paths=input_paths,
    output_path=output_data_path,
    output_zyx_shape=output_shape_zyx,
    chunk_zyx_shape=chunk_zyx_shape,
    voxel_size=voxel_size,
)

extra_metadata = {}
copy_n_paste_args = {"zyx_slicing_params": zyx_slicing_params}

# prepare slurm parameters
params = SlurmParams(
    partition="cpu",
    cpus_per_task=cpus_per_task,
    mem_per_cpu=mem_per_cpu,
    time=datetime.timedelta(minutes=time),
    output=slurm_out_path,
)

# wrap our deskew_single_position() function with slurmkit
slurm_crop_single_position = slurm_function(utils.process_single_position)
crop_func = slurm_crop_single_position(
    func=utils.copy_n_paste,
    num_processes=simultaneous_processes_per_node,
    **copy_n_paste_args,
)

# generate an array of jobs by passing the in_path and out_path to slurm wrapped function
crop_jobs = [
    submit_function(
        crop_func,
        slurm_params=params,
        input_data_path=in_path,
        output_path=out_path,
    )
    for in_path, out_path in zip(input_paths, output_paths)
]
