import datetime
import os
import glob
from mantis.cli.deskew import (
    deskew_cli,
    get_deskew_params,
    create_empty_zarr,
    get_output_paths,
)
from slurmkit import SlurmParams, slurm_function, submit_function
from natsort import natsorted
import click

# TODO: decide what logging we use (i.e click.echo, print, logging)

## PYTHON SCRIPT GLOBAL PARAMETERS
input_data_path = '/hpc/projects/comp.micro/mantis/2023_05_10_PCNA_RAC1/0-crop-convert-zarr/sample_short.zarr/*/*/*'
output_data_path = './deskewed.zarr'
deskew_param_path = './deskew_settings.yml'
keep_overhang = False

## SLURM SBATCH GLOBAL PARAMETERS
cpu_per_task = 16
mem = "16G"
time = 40

# ---------------INITIALIZATION AND DATA PARSING ----------------------
# Deskewing I/O
input_data_path = glob.glob(input_data_path)
input_data_path = natsorted(input_data_path)
output_dir = os.path.dirname(os.path.join(os.getcwd(), output_data_path))
list_output_pos = get_output_paths(input_data_path, output_data_path)
click.echo(f"in: {input_data_path}, out: {list_output_pos}")

# Slurm I/O
slurm_out_path = str(os.path.join(output_dir, "slurm_output/deskew-%j.out"))

# Initialize the array
create_empty_zarr(input_data_path, deskew_param_path, output_data_path, keep_overhang)

# -------------------- SLURM PARAMETERS ----------------------------------

# Slurm Parameters for each node
# NOTE: These parameters have to be tunned depending on the size of the job
# These parameters could be reused by multiple jobs. The job-name by default takes the slurm_function's name
params = SlurmParams(
    partition="cpu",
    cpus_per_task=cpu_per_task,
    mem=mem,
    time=datetime.timedelta(minutes=time),
    output=slurm_out_path,
)

# Wrap our deskew_cli() function with slurmkit
slurm_deskew_cli = slurm_function(deskew_cli)
deskew_func = slurm_deskew_cli(
    deskew_param_path=deskew_param_path, keep_overhang=keep_overhang
)

# Generates the array of jobs by passing the in_path and out_path to slurm wrapped function
deskew_jobs = [
    submit_function(
        deskew_func,
        slurm_params=params,
        input_data_path=in_path,
        output_path=out_path,
    )
    for in_path, out_path in zip(input_data_path, list_output_pos)
]
