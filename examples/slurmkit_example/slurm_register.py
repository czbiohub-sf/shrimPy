import datetime
import os
import glob
from mantis.cli import utils
from slurmkit import SlurmParams, slurm_function, submit_function
from natsort import natsorted
import click
from mantis.cli.register import registration_params_from_file
import numpy as np
from scipy.ndimage import affine_transform

# io parameters
input_paths = '/hpc/projects/comp.micro/mantis/2023_05_10_PCNA_RAC1/timelapse_1/3-reconstruct-all-slurmkit/phase_3D.zarr/0/0/0'
output_data_path = './registered_output.zarr'
registration_param_path = './registration_parameters.yml'

# sbatch and resource parameters
cpus_per_task = 16
mem_per_cpu = "16G"
time = 40  # minutes

# path handling
input_paths = natsorted(glob.glob(input_paths))
output_dir = os.path.dirname(output_data_path)
output_paths = utils.get_output_paths(input_paths, output_data_path)
click.echo(f"in: {input_paths}, out: {output_paths}")
slurm_out_path = str(os.path.join(output_dir, "slurm_output/register-%j.out"))

# Additional registraion arguments
# Parse from the yaml file
settings = registration_params_from_file(registration_param_path)
matrix = np.linalg.inv(np.array(settings.affine_transform_zyx))
output_shape = tuple(settings.output_shape)
voxel_size = tuple(settings.voxel_size)  # initialize zarr
# TODO: make this variable?
chunk_zyx_shape = (output_shape[0] // 10,) + output_shape[1:]
extra_metadata = {
    'registration': {
        'affine_matrix': matrix.tolist(),
        'fluor_channel_90deg_CCW_rot': settings.fluor_channel_90deg_CCW_rotation,
    }
}
affine_transform_args = {
    'matrix': matrix,
    'output_shape': settings.output_shape,
    'extra_metadata': extra_metadata,
}
utils.create_empty_zarr(
    position_paths=input_paths,
    output_path=output_data_path,
    output_zyx_shape=output_shape,
    chunk_zyx_shape=chunk_zyx_shape,
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
    func=affine_transform, num_processes=cpus_per_task, **affine_transform_args
)

# generate an array of jobs by passing the in_path and out_path to slurm wrapped function
register_jobs = [
    submit_function(
        register_func,
        slurm_params=params,
        input_data_path=in_path,
        output_path=out_path,
    )
    for in_path, out_path in zip(input_paths, output_paths)
]
