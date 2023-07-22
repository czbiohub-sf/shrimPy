import multiprocessing as mp

from pathlib import Path

import click

from iohub.ngff import Plate, open_ome_zarr
from natsort import natsorted

from mantis.cli import utils
from mantis.cli.parsing import input_data_paths_argument, output_dataset_options
from scipy.ndimage import affine_transform


@click.command()
@input_data_paths_argument()
@click.argument("registration_param_path", type=click.Path(exists=True))
@output_dataset_options(default="./registered.zarr")
# @click.option("--inverse", "-i", default=False, help="Apply the inverse transform")
@click.option(
    "--num-processes",
    "-j",
    default=mp.cpu_count(),
    help="Number of cores",
    required=False,
    type=int,
)
def register(
    input_paths: list[str],
    registration_param_path: str,
    output_path: str,
    num_processes: int,
):
    "Registers a single position across T and C axes using a parameter file generated by estimate_deskew.py"
    if isinstance(open_ome_zarr(input_paths[0]), Plate):
        raise ValueError(
            "Please supply a single position instead of an HCS plate. Likely fix: replace input.zarr with 'input.zarr/0/0/0'"
        )

    # Sort the input as nargs=-1 will not be natsorted
    input_paths = [Path(path) for path in natsorted(input_paths)]

    # Convert string paths to Path objects
    output_path = Path(output_path)
    registration_param_path = Path(registration_param_path)

    # Handle single position or wildcard filepath
    output_paths = utils.get_output_paths(input_paths, output_path)
    click.echo(f"List of input_pos:{input_paths} output_pos:{output_paths}")

    # Create a zarr store output to mirror the input
    utils.create_empty_zarr(input_paths, registration_param_path, output_path)

    # Get the affine transformation matrix

    with open_ome_zarr(registration_param_path, mode="r") as registration_parameters:
        matrix = registration_parameters["affine_transform_zyx"][0, 0, 0]
        output_shape = tuple(registration_parameters.zattrs["registration"]["channel_2_shape"])

    affine_transform_args = {'matrix': matrix, 'output_shape': output_shape}

    # Loop over positions
    for input_position_path, output_position_path in zip(input_paths, output_paths):
        utils.process_single_position(
            affine_transform,
            input_data_path=input_position_path,
            output_path=output_position_path,
            num_processes=num_processes,
            **affine_transform_args,
        )
