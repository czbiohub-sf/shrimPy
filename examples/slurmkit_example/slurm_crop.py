import datetime
import os
import glob
from mantis.cli import utils
from slurmkit import SlurmParams, slurm_function, submit_function
from natsort import natsorted
import click
from iohub import open_ome_zarr
from dataclasses import asdict


# io parameters
input_paths = "/hpc/projects/comp.micro/mantis/2023_08_09_HEK_PCNA_H2B/1-deskew/pcna_rac1_virtual_staining_1/deskewed.zarr/*/*/*"
output_data_path = "./deskewed_pcna_partition.zarr"

# Z-chunking factor. Big datasets require z-chunking to avoid blocs issues
Z_CHUNK = None  # CHUNK XY ZYX or non will be 500MB
no_roi_cropping = True
indices_to_remove = list(range(144, 216))

# Cropping the FOV parameters
if not no_roi_cropping:
    HALF_CROP_LENGTH = 400


# sbatch and resource parameters
cpus_per_task = 3
mem_per_cpu = "8G"
time = 40  # minutes
simultaneous_processes_per_node = 3


# path handling
input_paths = natsorted(glob.glob(input_paths))
output_dir = os.path.dirname(output_data_path)
output_paths = utils.get_output_paths(input_paths, output_data_path)
slurm_out_path = str(os.path.join(output_dir, "slurm_output/deskew-%j.out"))

# Remove desired rois
input_paths = [
    element for index, element in enumerate(input_paths) if index not in indices_to_remove
]
output_paths = [
    element for index, element in enumerate(output_paths) if index not in indices_to_remove
]
print(f"Removing positions idx: {indices_to_remove}")
click.echo(f"in: {input_paths}, out: {output_paths}")

# Additional deskew arguments
# Get the deskewing parameters
# Load the first position to infer dataset information
with open_ome_zarr(input_paths[0], mode="r") as input_dataset:
    T, C, Z, Y, X = input_dataset.data.shape
    voxel_size = tuple(input_dataset.scale[-3:])

    if no_roi_cropping:
        # # NOTE :crop here and chunksize
        # # Slicing Parameters
        Z_slice = slice(None)
        Y_slice = slice(None)
        X_slice = slice(None)
        chunk_zyx_shape = None
        output_shape_zyx = (Z, Y, X)
    else:
        Z_slice = slice(None)
        Y_slice = slice(Y // 2 - HALF_CROP_LENGTH, Y // 2 + HALF_CROP_LENGTH)
        X_slice = slice(X // 2 - HALF_CROP_LENGTH, X // 2 + HALF_CROP_LENGTH)
        chunk_zyx_shape = (Z_CHUNK, output_shape_zyx[-2], output_shape_zyx[-1])
        output_shape_zyx = (
            Z,
            int(Y_slice.stop - Y_slice.start),
            int(X_slice.stop - X_slice.start),
        )
    zyx_slicing_params = [Z_slice, Y_slice, X_slice]


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
