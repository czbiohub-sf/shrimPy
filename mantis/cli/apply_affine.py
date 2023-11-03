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
from mantis.cli.utils import yaml_to_model


def apply_affine_to_scale(affine_matrix, input_scale):
    return np.linalg.norm(affine_matrix, axis=1) * input_scale


@click.command()
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@click.option(
    "--num-processes",
    "-j",
    default=1,
    help="Number of parallel processes",
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
    settings = yaml_to_model(config_filepath, RegistrationSettings)
    matrix = np.array(settings.affine_transform_zyx)
    output_shape_zyx = tuple(settings.output_shape_zyx)

    # Calculate the output voxel size from the input scale and affine transform
    with open_ome_zarr(input_position_dirpaths[0]) as input_dataset:
        output_voxel_size = apply_affine_to_scale(matrix[:3, :3], input_dataset.scale[-3:])

    click.echo('\nREGISTRATION PARAMETERS:')
    click.echo(f'Affine transform: {matrix}')
    click.echo(f'Voxel size: {output_voxel_size}')

    utils.create_empty_zarr(
        position_paths=input_position_dirpaths,
        output_path=output_dirpath,
        output_zyx_shape=output_shape_zyx,
        chunk_zyx_shape=None,
        voxel_size=tuple(output_voxel_size),
    )

    # Get the affine transformation matrix
    extra_metadata = {
        'affine_transformation': {
            'transform_matrix': matrix.tolist(),
        }
    }
    affine_transform_args = {
        'matrix': matrix,
        'output_shape_zyx': settings.output_shape_zyx,
        'extra_metadata': extra_metadata,
    }

    # Loop over positions
    for input_position_path, output_position_path in zip(
        input_position_dirpaths, output_paths
    ):
        utils.process_single_position(
            utils.ants_affine_transform,
            input_data_path=input_position_path,
            output_path=output_position_path,
            num_processes=num_processes,
            **affine_transform_args,
        )
