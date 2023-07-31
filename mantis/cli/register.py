import multiprocessing as mp


import click

from iohub.ngff import open_ome_zarr

from mantis.cli import utils
from mantis.cli.parsing import (
    input_data_paths_argument,
    output_dataset_options,
    registration_param_argument,
)
from scipy.ndimage import affine_transform
import numpy as np
from pathlib import Path


@click.command()
@input_data_paths_argument()
@registration_param_argument()
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
    input_paths: list[Path],
    registration_param_path: Path,
    output_path: Path,
    num_processes: int,
):
    "Registers a single position across T and C axes using the pathfile for affine transform"

    # Handle single position or wildcard filepath
    output_paths = utils.get_output_paths(input_paths, output_path)
    click.echo(f"List of input_pos:{input_paths} output_pos:{output_paths}")

    # Create a zarr store output to mirror the input
    with open_ome_zarr(registration_param_path, mode="r") as registration_parameters:
        matrix = np.linalg.inv(registration_parameters["affine_transform_zyx"][0, 0, 0])
        output_shape = tuple(registration_parameters.zattrs["registration"]["channel_2_shape"])
        voxel_size = tuple(registration_parameters.zattrs["registration"]["voxel_size"])
        click.echo('\nREGISTRATION PARAMETERS:')
        click.echo(f'Affine transform: {matrix}')
        click.echo(f'Position zyx shape: {output_shape}')
        click.echo(f'Voxel size: {voxel_size}')
        # TODO: dont know what would be the best chunking size. we have a limit with blosc
        chunk_zyx_shape = (output_shape[0] // 10,) + output_shape[1:]
        click.echo(f'Chunk zyx size: {chunk_zyx_shape}')

    utils.create_empty_zarr(
        position_paths=input_paths,
        output_path=output_path,
        output_zyx_shape=output_shape,
        chunk_zyx_shape=chunk_zyx_shape,
        voxel_size=voxel_size,
    )

    # Get the affine transformation matrix

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
