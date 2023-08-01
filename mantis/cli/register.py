import multiprocessing as mp


import click


from mantis.cli import utils
from mantis.cli.parsing import (
    input_data_paths_argument,
    output_dataset_options,
    registration_param_argument,
)
from mantis.analysis.AnalysisSettings import RegistrationSettings

from scipy.ndimage import affine_transform
import numpy as np
from pathlib import Path
import yaml
from dataclasses import asdict


def registration_params_from_file(registration_param_path: Path) -> RegistrationSettings:
    """Parse the deskewing parameters from the yaml file"""
    # Load params
    with open(registration_param_path) as file:
        raw_settings = yaml.safe_load(file)
    settings = RegistrationSettings(**raw_settings)
    click.echo(f"Registration parameters: {asdict(settings)}")
    return settings


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

    # Parse from the yaml file
    settings = registration_params_from_file(registration_param_path)
    matrix = np.linalg.inv(np.array(settings.affine_transform_zyx))
    output_shape = tuple(settings.output_shape)
    voxel_size = tuple(settings.voxel_size)

    click.echo('\nREGISTRATION PARAMETERS:')
    click.echo(f'Affine transform: {matrix}')
    click.echo(f'Output shape: {output_shape}')
    click.echo(f'Voxel size: {voxel_size}')
    chunk_zyx_shape = (output_shape[0] // 10,) + output_shape[1:]
    click.echo(f'Chunk size output {chunk_zyx_shape}')

    utils.create_empty_zarr(
        position_paths=input_paths,
        output_path=output_path,
        output_zyx_shape=output_shape,
        chunk_zyx_shape=chunk_zyx_shape,
        voxel_size=voxel_size,
    )

    # Get the affine transformation matrix
    # TODO: add the metadta from yaml
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

    # Loop over positions
    for input_position_path, output_position_path in zip(input_paths, output_paths):
        utils.process_single_position(
            affine_transform,
            input_data_path=input_position_path,
            output_path=output_position_path,
            num_processes=num_processes,
            **affine_transform_args,
        )
