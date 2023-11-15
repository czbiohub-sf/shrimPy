import datetime
import os
import glob
from mantis.cli import utils
from slurmkit import SlurmParams, slurm_function, submit_function
from natsort import natsorted
import click
import numpy as np
from iohub import open_ome_zarr
from mantis.analysis.AnalysisSettings import RegistrationSettings
from pathlib import Path
from typing import List
from mantis.cli.parsing import config_filepath, input_position_dirpaths, output_dirpath
from mantis.cli.utils import yaml_to_model
from mantis.cli.apply_affine import apply_affine_to_scale

# io parameters
source_data_paths = '/hpc/projects/comp.micro/mantis/2023_08_09_HEK_PCNA_H2B/2-phase3D/pcna_rac1_virtual_staining_b1_redo_1/phase3D.zarr/0/0/0'
target_data_paths = '/hpc/projects/comp.micro/mantis/2023_08_09_HEK_PCNA_H2B/1-deskew/pcna_rac1_virtual_staining_b1_redo_1/deskewed.zarr/0/0/0'
output_data_path = './registered_output.zarr'
registration_param_path = './register.yml'

# sbatch and resource parameters
cpus_per_task = 16
mem_per_cpu = "16G"
time = 40  # minutes
simultaneous_processes_per_node = 5

# path handling
source_data_paths = natsorted(glob.glob(source_data_paths))
target_data_paths = natsorted(glob.glob(target_data_paths))
output_dir = os.path.dirname(output_data_path)
output_paths = utils.get_output_paths(source_data_paths, output_data_path)
click.echo(f"in: {source_data_paths}, out: {output_paths}")
slurm_out_path = str(os.path.join(output_dir, "slurm_output/register-%j.out"))

# Additional registraion arguments
# Parse from the yaml file
settings = yaml_to_model(config_filepath, RegistrationSettings)
matrix = np.array(settings.affine_transform_zyx)
output_shape_zyx = tuple(settings.output_shape_zyx)

# Calculate the output voxel size from the input scale and affine transform
with open_ome_zarr(input_position_dirpaths[0]) as input_dataset:
    output_voxel_size = apply_affine_to_scale(matrix[:3, :3], input_dataset.scale[-3:])

# Get the affine transformation matrix
extra_metadata = {
    'affine_transformation': {
        'transform_matrix': matrix.tolist(),
    }
}
affine_transform_args = {
    'matrix': matrix,
    'output_shape_zyx': settings.output_shape_zyx,
    'extra_metadata': extra_metadata,
}
utils.create_empty_zarr(
    position_paths=source_data_paths,
    output_path=output_data_path,
    output_zyx_shape=output_shape_zyx,
    chunk_zyx_shape=None,
    voxel_size=voxel_size,
)

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
    func=utils.affine_transform,
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
    for in_path, out_path in zip(source_data_paths, output_paths)
]
