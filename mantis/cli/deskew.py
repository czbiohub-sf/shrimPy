from pathlib import Path
from typing import List

import click
import torch

from iohub.ngff import open_ome_zarr

from mantis.analysis.AnalysisSettings import DeskewSettings
from mantis.analysis.deskew import deskew_data, get_deskewed_data_shape
from mantis.cli import utils
from mantis.cli.parsing import config_filepath, input_position_dirpaths, output_dirpath
from mantis.cli.utils import yaml_to_model

# Needed for multiprocessing with GPUs
# https://github.com/pytorch/pytorch/issues/40403#issuecomment-1422625325
torch.multiprocessing.set_start_method('spawn', force=True)


@click.command(deprecated=True)
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@click.option(
    "--num-processes",
    "-j",
    default=1,
    help="Number of cores",
    required=False,
    type=int,
)
def deskew(
    input_position_dirpaths: List[str],
    config_filepath: str,
    output_dirpath: str,
    num_processes: int,
):
    """
    Deskew a single position across T and C axes using a configuration file generated by estimate_deskew.py

    >> mantis deskew -i ./input.zarr/*/*/* -c ./deskew_params.yml -o ./output.zarr
    """
    click.echo(
        '"This function is being moved to the biahub library, available at https://github.com/czbiohub-sf/biahub"'
    )

    # Convert string paths to Path objects
    output_dirpath = Path(output_dirpath)
    config_filepath = Path(config_filepath)

    # Handle single position or wildcard filepath
    output_paths = utils.get_output_paths(input_position_dirpaths, output_dirpath)
    click.echo(f'List of input_pos:{input_position_dirpaths} output_pos:{output_paths}')

    # Get the deskewing parameters
    # Load the first position to infer dataset information
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        T, C, Z, Y, X = input_dataset.data.shape
        settings = yaml_to_model(config_filepath, DeskewSettings)
        deskewed_shape, voxel_size = get_deskewed_data_shape(
            (Z, Y, X),
            settings.ls_angle_deg,
            settings.px_to_scan_ratio,
            settings.keep_overhang,
            settings.average_n_slices,
            settings.pixel_size_um,
        )

        # Create a zarr store output to mirror the input
        utils.create_empty_zarr(
            input_position_dirpaths,
            output_dirpath,
            output_zyx_shape=deskewed_shape,
            chunk_zyx_shape=None,
            voxel_size=voxel_size,
        )

    deskew_args = {
        'ls_angle_deg': settings.ls_angle_deg,
        'px_to_scan_ratio': settings.px_to_scan_ratio,
        'keep_overhang': settings.keep_overhang,
        'average_n_slices': settings.average_n_slices,
        'extra_metadata': {'deskew': settings.dict()},
    }

    # Loop over positions
    for input_position_path, output_position_path in zip(
        input_position_dirpaths, output_paths
    ):
        utils.process_single_position(
            deskew_data,
            input_data_path=input_position_path,
            output_path=output_position_path,
            num_processes=num_processes,
            **deskew_args,
        )


if __name__ == "__main__":
    deskew()
