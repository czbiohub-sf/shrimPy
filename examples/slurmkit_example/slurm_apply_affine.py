import datetime
import os
import glob
from mantis.cli import utils
from slurmkit import SlurmParams, slurm_function, submit_function
from natsort import natsorted
import click
from mantis.cli.apply_affine import registration_params_from_file, rotate_n_affine_transform
import numpy as np
from iohub import open_ome_zarr


# io parameters
labelfree_data_paths = '/hpc/projects/comp.micro/mantis/2023_08_09_HEK_PCNA_H2B/2-phase3D/pcna_rac1_virtual_staining_b1_redo_1/phase3D.zarr/0/0/0'
lightsheet_data_paths = '/hpc/projects/comp.micro/mantis/2023_08_09_HEK_PCNA_H2B/1-deskew/pcna_rac1_virtual_staining_b1_redo_1/deskewed.zarr/0/0/0'
output_data_path = './registered_output.zarr'
registration_param_path = './register.yml'

# sbatch and resource parameters
cpus_per_task = 16
mem_per_cpu = "16G"
time = 40  # minutes
simultaneous_processes_per_node = 5

# path handling
labelfree_data_paths = natsorted(glob.glob(labelfree_data_paths))
lightsheet_data_paths = natsorted(glob.glob(lightsheet_data_paths))
output_dir = os.path.dirname(output_data_path)
output_paths = utils.get_output_paths(labelfree_data_paths, output_data_path)
click.echo(f"in: {labelfree_data_paths}, out: {output_paths}")
slurm_out_path = str(os.path.join(output_dir, "slurm_output/register-%j.out"))

# Additional registraion arguments
# Parse from the yaml file
settings = registration_params_from_file(registration_param_path)
matrix = np.array(settings.affine_transform_zyx)
output_shape_zyx = tuple(settings.output_shape_zyx)

# Get the output voxel_size
with open_ome_zarr(lightsheet_data_paths[0]) as light_sheet_position:
    voxel_size = tuple(light_sheet_position.scale[-3:])

extra_metadata = {
    'registration': {
        'affine_matrix': matrix.tolist(),
    }
}
affine_transform_args = {
    'matrix': matrix,
    'output_shape_zyx': settings.output_shape_zyx,
    'extra_metadata': extra_metadata,
}
utils.create_empty_zarr(
    position_paths=labelfree_data_paths,
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
    func=rotate_n_affine_transform,
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
    for in_path, out_path in zip(labelfree_data_paths, output_paths)
]
