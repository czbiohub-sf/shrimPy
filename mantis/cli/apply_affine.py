import multiprocessing as mp

from pathlib import Path
from typing import List

import click
import numpy as np
import yaml

from iohub import open_ome_zarr
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
    return settings


def rotate_n_affine_transform(
    zyx_data, matrix, output_shape_zyx, pre_affine_90degree_rotations_about_z: int = 0
):
    if pre_affine_90degree_rotations_about_z != 0:
        rotate_volume = np.rot90(
            zyx_data, k=pre_affine_90degree_rotations_about_z, axes=(1, 2)
        )
    affine_volume = affine_transform(
        rotate_volume, matrix=matrix, output_shape=output_shape_zyx
    )
    return affine_volume


def apply_affine_to_scale(affine_matrix, input_scale):
    return np.linalg.norm(affine_matrix, axis=1) * input_scale


@click.command()
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@click.option(
    "--num-processes",
    "-j",
    default=mp.cpu_count(),
    help="Number of cores",
    required=False,
    type=int,
)
def apply_affine(
    input_position_dirpaths: List[str],
    config_filepath: str,
    output_dirpath: str,
    num_processes: int,
):
    """
    Apply an affine transformation to a single position across T and C axes using the pathfile for affine transform to the phase channel

    >> mantis apply_affine -i ./acq_name_lightsheet_deskewed.zarr/*/*/* -c ./register.yml -o ./acq_name_registerred.zarr
    """
    # Convert string paths to Path objects
    output_dirpath = Path(output_dirpath)
    config_filepath = Path(config_filepath)

    # Handle single position or wildcard filepath
    output_paths = utils.get_output_paths(input_position_dirpaths, output_dirpath)
    click.echo(f"List of input_pos:{input_position_dirpaths} output_pos:{output_paths}")

    # Parse from the yaml file
    settings = registration_params_from_file(config_filepath)
    matrix = np.array(settings.affine_transform_zyx)
    output_shape_zyx = tuple(settings.output_shape_zyx)
    pre_affine_90degree_rotations_about_z = settings.pre_affine_90degree_rotations_about_z

    # Calculate the output voxel size from the input scale and affine transform
    with open_ome_zarr(input_position_dirpaths[0]) as input_dataset:
        output_voxel_size = apply_affine_to_scale(matrix[:3, :3], input_dataset.scale[-3:])

    click.echo('\nREGISTRATION PARAMETERS:')
    click.echo(f'Affine transform: {matrix}')
    click.echo(f'Voxel size: {output_voxel_size}')

    z_chunk_factor = 10
    chunk_zyx_shape = (
        output_shape_zyx[0] // z_chunk_factor
        if output_shape_zyx[0] > z_chunk_factor
        else output_shape_zyx[0],
        output_shape_zyx[1],
        output_shape_zyx[2],
    )

    utils.create_empty_zarr(
        position_paths=input_position_dirpaths,
        output_path=output_dirpath,
        output_zyx_shape=output_shape_zyx,
        chunk_zyx_shape=chunk_zyx_shape,
        voxel_size=tuple(output_voxel_size),
    )

    # Get the affine transformation matrix
    extra_metadata = {
        'affine_transformation': {
            'affine_matrix': matrix.tolist(),
            'pre_affine_90degree_rotations_about_z': pre_affine_90degree_rotations_about_z,
        }
    }
    affine_transform_args = {
        'matrix': matrix,
        'output_shape_zyx': settings.output_shape_zyx,
        'pre_affine_90degree_rotations_about_z': pre_affine_90degree_rotations_about_z,
        'extra_metadata': extra_metadata,
    }

    # Loop over positions
    for input_position_path, output_position_path in zip(
        input_position_dirpaths, output_paths
    ):
        utils.process_single_position(
            rotate_n_affine_transform,
            input_data_path=input_position_path,
            output_path=output_position_path,
            num_processes=num_processes,
            **affine_transform_args,
        )
