# %%%
from iohub import open_ome_zarr

from mantis.cli.parsing import (
    input_data_path_argument,
    deskew_param_argument,
    output_dataset_options,
)
from mantis.analysis.deskew import get_deskewed_data_shape
import yaml
from mantis.analysis.AnalysisSettings import DeskewSettings
from dataclasses import asdict

# debugging
from tqdm import tqdm
import click
import numpy as np
from datetime import datetime
import os
from iohub.ngff_meta import TransformationMeta


def _get_deskew_params(deskew_params_path):
    # Load params
    with open(deskew_params_path) as file:
        raw_settings = yaml.safe_load(file)
    settings = DeskewSettings(**raw_settings)
    print(f"Deskewing parameters: {asdict(settings)}")
    return settings


@click.group()
def cli():
    pass


@cli.command()
@input_data_path_argument()
@deskew_param_argument()
@output_dataset_options(default="./deskewed.zarr")
@click.option(
    "--keep-overhang",
    "-ko",
    default=False,
    is_flag=True,
    help="Keep the overhanging region.",
)
@click.help_option("-h", "--help")
def create_empty_zarr(input_data_path, deskew_param_path, output_path, keep_overhang):
    # Load the "0" position to infer dataset information
    input_dataset = open_ome_zarr(str(input_data_path[0]), mode="r")
    T, C, Z, Y, X = input_dataset.data.shape

    # Get the deskewing parameters
    settings = _get_deskew_params(deskew_param_path)
    deskewed_shape, voxel_size = get_deskewed_data_shape(
        (Z, Y, X),
        settings.pixel_size_um,
        settings.ls_angle_deg,
        settings.px_to_scan_ratio,
        keep_overhang,
    )

    click.echo("Creating empty array...")

    # Handle transforms and metadata
    transform = TransformationMeta(
        type="scale",
        scale=2 * (1,) + voxel_size,
    )

    # Prepare output dataset
    channel_names = input_dataset.channel_names

    # Output shape based on the type of reconstruction
    output_shape = (T, len(channel_names)) + deskewed_shape
    click.echo(f"Number of positions: {len(input_data_path)}")
    click.echo(f"Output shape: {output_shape}")

    # Create output dataset
    output_dataset = open_ome_zarr(
        output_path, layout="hcs", mode="w", channel_names=channel_names
    )
    # This takes care of the logic for single position or multiple position by wildcards
    for filepath in input_data_path:
        path_strings = filepath.split(os.path.sep)[-3:]
        pos = output_dataset.create_position(
            str(path_strings[0]), str(path_strings[1]), str(path_strings[2])
        )

        _ = pos.create_zeros(
            name="0",
            shape=(
                T,
                C,
            )
            + deskewed_shape,
            chunks=(1, 1, 64) + deskewed_shape[-2:],
            dtype=np.uint16,
            transform=[transform],
        )


# def create_empty_zarr(input_data_path, output_data_path):
#     print(f'input:{input_data_path}')
#     print(f'output:{output_data_path}')

#     # Load datasets
#     # Take position 0 as sample

#     #Wildcard pattern
#     wildcard_pattern='*/*/*'

#     # Helper function to do the job o
#     position_paths = get_files_with_pattern(input_data_path, wildcard_pattern)

#     input_dataset = open_ome_zarr(str(position_paths[0]), mode="r")
#     channel_names = input_dataset.channel_names

#     # Create output dataset
#     output_dataset = open_ome_zarr(
#         output_data_path, layout="hcs", mode="a", channel_names=channel_names
#     )
#     for filepath in tqdm(position_paths):
#         path_strings = filepath.split(os.path.sep)[-3:]
#         pos = output_dataset.create_position(
#             str(path_strings[0]), str(path_strings[1]), str(path_strings[2])
#         )
#         _ = pos.create_zeros(
#             name="0",
#             shape=input_dataset.data.shape,
#             dtype=np.uint16,
#             chunks=(
#                 1,
#                 1,
#                 1,
#             )
#             + input_dataset.data.shape[3:],  # chunk by YX
#         )


if __name__ == "__main__":
    create_empty_zarr()
