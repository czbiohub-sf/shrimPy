import datetime
import os
import glob
from mantis.cli.deskew import deskew_single_position

from mantis.cli.utils import (
    create_empty_zarr,
    get_output_paths,
)
from slurmkit import SlurmParams, slurm_function, submit_function
from natsort import natsorted
import click

# deskew parameters
# TODO: make sure that this settings file sets `keep_overhang` to false
deskew_param_path = './deskew_settings.yml'

# io parameters
input_paths = '/hpc/projects/comp.micro/mantis/2023_05_10_PCNA_RAC1/timelapse_2_3/0-crop-convert-zarr/sample_short.zarr/*/*/*'
output_data_path = './deskewed.zarr'

# sbatch and resource parameters
cpus_per_task = 16
mem_per_cpu = "16G"
time = 40  # minutes

# path handling
input_paths = natsorted(glob.glob(input_paths))
output_dir = os.path.dirname(output_data_path)
output_paths = get_output_paths(input_paths, output_data_path)
click.echo(f"in: {input_paths}, out: {output_paths}")
slurm_out_path = str(os.path.join(output_dir, "slurm_output/deskew-%j.out"))

# initialize zarr
create_empty_zarr(input_paths, deskew_param_path, output_data_path)

# prepare slurm parameters
params = SlurmParams(
    partition="cpu",
    cpus_per_task=cpus_per_task,
    mem_per_cpu=mem_per_cpu,
    time=datetime.timedelta(minutes=time),
    output=slurm_out_path,
)

# wrap our deskew_single_position() function with slurmkit
slurm_deskew_single_position = slurm_function(deskew_single_position)
deskew_func = slurm_deskew_single_position(
    deskew_param_path=deskew_param_path,
    num_processes=cpus_per_task,
)

# generate an array of jobs by passing the in_path and out_path to slurm wrapped function
deskew_jobs = [
    submit_function(
        deskew_func,
        slurm_params=params,
        input_data_path=in_path,
        output_path=out_path,
    )
    for in_path, out_path in zip(input_paths, output_paths)
]
