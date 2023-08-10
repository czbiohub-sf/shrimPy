import multiprocessing as mp

from dataclasses import asdict
from pathlib import Path
from typing import List

import click
import numpy as np
import yaml

from iohub import open_ome_zarr
from scipy.ndimage import affine_transform

from mantis.analysis.AnalysisSettings import RegistrationSettings
from mantis.cli import utils
from mantis.cli.parsing import (
    config_filepath,
    labelfree_position_dirpaths,
    lightsheet_position_dirpaths,
    output_dirpath,
)


def registration_params_from_file(registration_param_path: Path) -> RegistrationSettings:
    """Parse the deskewing parameters from the yaml file"""
    # Load params
    with open(registration_param_path) as file:
        raw_settings = yaml.safe_load(file)
    settings = RegistrationSettings(**raw_settings)
    click.echo(f"Registration parameters: {asdict(settings)}")
    return settings


def rotate_n_affine_transform(zyx_data, matrix, output_shape_zyx, k_90deg_rot: int = 0):
    rotate_volume = np.rot90(zyx_data, k=k_90deg_rot, axes=(1, 2))
    affine_volume = affine_transform(
        rotate_volume, matrix=matrix, output_shape=output_shape_zyx
    )
    return affine_volume


@click.command()
@labelfree_position_dirpaths()
@lightsheet_position_dirpaths()
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
def apply_affine(
    labelfree_position_dirpaths: List[str],  # TODO copy from here to output?
    lightsheet_position_dirpaths: List[str],
    config_filepath: str,
    output_dirpath: str,
    num_processes: int,
):
    """
    Apply an affine transformation a single position across T and C axes using the pathfile for affine transform

    >> mantis apply_affine -lf ./acq_name_lightsheet_deskewed.zarr/*/*/* -ls ./acq_name_lightsheet_deskewed.zarr/*/*/* -c ./register.yml -o ./acq_name_registerred.zarr
    """

    # Handle single position or wildcard filepath
    output_paths = utils.get_output_paths(lightsheet_position_dirpaths, output_dirpath)
    click.echo(f"List of input_pos:{lightsheet_position_dirpaths} output_pos:{output_paths}")

    # Parse from the yaml file
    settings = registration_params_from_file(config_filepath)
    matrix = np.linalg.inv(np.array(settings.affine_transform_zyx))
    output_shape_zyx = tuple(settings.output_shape_zyx)
    k_90deg_rot = settings.k_90deg_rot

    # Get the voxel size from the lightsheet data
    with open_ome_zarr(lightsheet_position_dirpaths[0]) as ls_position:
        voxel_size = ls_position.scale

    click.echo('\nREGISTRATION PARAMETERS:')
    click.echo(f'Affine transform: {matrix}')
    click.echo(f'Output shape: {output_shape_zyx}')
    click.echo(f'Voxel size: {voxel_size}')
    chunk_zyx_shape = (output_shape_zyx[0] // 10,) + output_shape_zyx[1:]
    click.echo(f'Chunk size output {chunk_zyx_shape}')

    utils.create_empty_zarr(
        position_paths=lightsheet_position_dirpaths,
        output_path=output_dirpath,
        output_zyx_shape=output_shape_zyx,
        chunk_zyx_shape=chunk_zyx_shape,
        voxel_size=voxel_size,
    )

    # Get the affine transformation matrix
    # TODO: add the metadta from yaml
    extra_metadata = {
        'affine_transformation': {
            'affine_matrix': matrix.tolist(),
            'k_90deg_rot': k_90deg_rot,
        }
    }
    affine_transform_args = {
        'matrix': matrix,
        'output_shape': settings.output_shape_zyx,
        'k_90deg_rot': k_90deg_rot,
        'extra_metadata': extra_metadata,
    }

    # Loop over positions
    for input_position_path, output_position_path in zip(
        lightsheet_position_dirpaths, output_paths
    ):
        utils.process_single_position(
            rotate_n_affine_transform,
            input_data_path=input_position_path,
            output_path=output_position_path,
            num_processes=num_processes,
            **affine_transform_args,
        )
