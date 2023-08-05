import multiprocessing as mp

from dataclasses import asdict
from pathlib import Path
from typing import List

import click
import numpy as np
import yaml

from scipy.ndimage import affine_transform

from mantis.analysis.AnalysisSettings import RegistrationSettings
from mantis.cli import utils
from mantis.cli.parsing import config_filepath, input_position_dirpaths, output_dirpath


def registration_params_from_file(registration_param_path: Path) -> RegistrationSettings:
    """Parse the deskewing parameters from the yaml file"""
    # Load params
    with open(registration_param_path) as file:
        raw_settings = yaml.safe_load(file)
    settings = RegistrationSettings(**raw_settings)
    click.echo(f"Registration parameters: {asdict(settings)}")
    return settings


@click.command()
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
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
    input_position_dirpaths: List[str],
    config_filepath: str,
    output_dirpath: str,
    num_processes: int,
):
    """
    Register a single position across T and C axes using the pathfile for affine transform

    >> mantis register -i ./input.zarr/*/*/* -c ./deskew_params.yml -o ./output.zarr
    """

    # Handle single position or wildcard filepath
    output_paths = utils.get_output_paths(input_position_dirpaths, output_dirpath)
    click.echo(f"List of input_pos:{input_position_dirpaths} output_pos:{output_paths}")

    # Parse from the yaml file
    settings = registration_params_from_file(config_filepath)
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
        position_paths=input_position_dirpaths,
        output_path=output_dirpath,
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
    for input_position_path, output_position_path in zip(
        input_position_dirpaths, output_paths
    ):
        utils.process_single_position(
            affine_transform,
            input_data_path=input_position_path,
            output_path=output_position_path,
            num_processes=num_processes,
            **affine_transform_args,
        )
