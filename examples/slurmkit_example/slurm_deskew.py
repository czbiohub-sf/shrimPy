import datetime
import os
import glob
from mantis.cli import utils
from slurmkit import SlurmParams, slurm_function, submit_function
from natsort import natsorted
import click
from mantis.analysis.deskew import deskew_data, get_deskewed_data_shape
from mantis.cli.deskew import deskew_params_from_file
from iohub import open_ome_zarr
from dataclasses import asdict

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
output_paths = utils.get_output_paths(input_paths, output_data_path)
click.echo(f"in: {input_paths}, out: {output_paths}")
slurm_out_path = str(os.path.join(output_dir, "slurm_output/deskew-%j.out"))

# Additional deskew arguments
# Get the deskewing parameters
# Load the first position to infer dataset information
with open_ome_zarr(str(input_paths[0]), mode="r") as input_dataset:
    T, C, Z, Y, X = input_dataset.data.shape
    settings = deskew_params_from_file(deskew_param_path)
    deskewed_shape, voxel_size = get_deskewed_data_shape(
        (Z, Y, X),
        settings.ls_angle_deg,
        settings.px_to_scan_ratio,
        settings.keep_overhang,
        settings.pixel_size_um,
    )

    # Create a zarr store output to mirror the input
    utils.create_empty_zarr(
        input_paths,
        output_data_path,
        output_zyx_shape=deskewed_shape,
        chunk_zyx_shape=deskewed_shape,
        voxel_size=voxel_size,
    )

deskew_args = {
    'ls_angle_deg': settings.ls_angle_deg,
    'px_to_scan_ratio': settings.px_to_scan_ratio,
    'keep_overhang': settings.keep_overhang,
    'extra_metadata': {'deskew': asdict(settings)},
}

# prepare slurm parameters
params = SlurmParams(
    partition="cpu",
    cpus_per_task=cpus_per_task,
    mem_per_cpu=mem_per_cpu,
    time=datetime.timedelta(minutes=time),
    output=slurm_out_path,
)

# wrap our deskew_single_position() function with slurmkit
slurm_deskew_single_position = slurm_function(utils.process_single_position)
deskew_func = slurm_deskew_single_position(
    func=deskew_data, num_processes=cpus_per_task, **deskew_args
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
