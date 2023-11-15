import datetime
import os
import glob
from mantis.cli import utils
from slurmkit import SlurmParams, slurm_function, submit_function
from natsort import natsorted
import numpy as np
from iohub import open_ome_zarr
from pathlib import Path
from mantis.analysis.AnalysisSettings import RegistrationSettings
from mantis.cli.utils import yaml_to_model

cropping_methods = {
    "all": 0,
    "custom_roi": 1,
    "custom_slices": 2,
}

# Get two sample FOVs from the target and source respectively
input_position_dirpaths = "/hpc/projects/comp.micro/mantis/2023_11_08_Opencell_infection/4-registration/HSP90AB1_registration_v2_1_registered.zarr/0/9/000000"
raw_input_input_dirpaths = "/hpc/projects/comp.micro/mantis/2023_11_08_Opencell_infection/1-recon/HSP90AB1_registration_v2_1_phase.zarr/0/9/000000"
output_data_path = "./cropped_v2.zarr"
config_filepath = "./optimized_registration_nuc_3.yml"

# Parse from the yaml file
settings = yaml_to_model(config_filepath, RegistrationSettings)
matrix = np.array(settings.affine_transform_zyx)
target_shape_zyx = tuple(settings.output_shape_zyx)

# Z-chunking factor. Big datasets require z-chunking to avoid blocs issues
Z_CHUNK = 5  # CHUNK XY ZYX or non will be 500MB

# sbatch and resource parameters
cpus_per_task = 3
mem_per_cpu = "8G"
time = 40  # minutes
simultaneous_processes_per_node = 3

# path handling
input_paths = [Path(path) for path in natsorted(glob.glob(input_position_dirpaths))]

output_data_path = Path(output_data_path)
output_dir = output_data_path.parent
output_paths = utils.get_output_paths(input_paths, output_data_path)
slurm_out_path = str(os.path.join(output_dir, "slurm_output/deskew-%j.out"))

with open_ome_zarr(input_paths[0], mode="r") as input_dataset:
    voxel_size = tuple(input_dataset.scale[-3:])

with open_ome_zarr(raw_input_input_dirpaths, mode="r") as raw_input_dataset:
    raw_input_shape_zyx = raw_input_dataset.data.shape[-3:]


print(f'matrix: {matrix}')
# Find the largest interior rectangle
Z_slice, Y_slice, X_slice = utils.find_lir_slicing_params(
    input_zyx_shape=raw_input_shape_zyx,
    target_zyx_shape=target_shape_zyx,
    transformation_matrix=matrix,
)
print(f"Z_slice: {Z_slice}")
print(f"Y_slice: {Y_slice}")
print(f"X_slice: {X_slice}")

output_shape_zyx = (
    int(Z_slice.stop - Z_slice.start),
    int(Y_slice.stop - Y_slice.start),
    int(X_slice.stop - X_slice.start),
)
chunk_zyx_shape = (Z_CHUNK, output_shape_zyx[-2], output_shape_zyx[-1])

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
