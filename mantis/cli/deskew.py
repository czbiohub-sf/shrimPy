import itertools
import multiprocessing as mp

from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import List

import click
import numpy as np
import yaml

from iohub.ngff import Plate, Position, open_ome_zarr
from iohub.ngff_meta import TransformationMeta
from natsort import natsorted

from mantis.analysis.AnalysisSettings import DeskewSettings
from mantis.analysis.deskew import deskew_data, get_deskewed_data_shape
from mantis.cli.parsing import (
    deskew_param_argument,
    input_data_paths_argument,
    output_dataset_options,
)


# TODO: consider refactoring to utils
def deskew_params_from_file(deskew_param_path: Path) -> DeskewSettings:
    """Parse the deskewing parameters from the yaml file"""
    # Load params
    with open(deskew_param_path) as file:
        raw_settings = yaml.safe_load(file)
    settings = DeskewSettings(**raw_settings)
    click.echo(f"Deskewing parameters: {asdict(settings)}")
    return settings


def create_empty_zarr(
    position_paths: List[Path], deskew_param_path: Path, output_path: Path, keep_overhang: bool
) -> None:
    """Create an empty zarr array for the deskewing"""
    # Load the first position to infer dataset information
    input_dataset = open_ome_zarr(str(position_paths[0]), mode="r")
    T, C, Z, Y, X = input_dataset.data.shape

    # Get the deskewing parameters
    settings = deskew_params_from_file(deskew_param_path)
    deskewed_shape, voxel_size = get_deskewed_data_shape(
        (Z, Y, X),
        settings.ls_angle_deg,
        settings.px_to_scan_ratio,
        keep_overhang,
        settings.pixel_size_um,
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
    click.echo(f"Number of positions: {len(position_paths)}")
    click.echo(f"Output shape: {output_shape}")
    # Create output dataset
    output_dataset = open_ome_zarr(
        output_path, layout="hcs", mode="w", channel_names=channel_names
    )
    chunk_size = (1, 1, 64) + deskewed_shape[-2:]
    click.echo(f"Chunk size {chunk_size}")

    # This takes care of the logic for single position or multiple position by wildcards
    for path in position_paths:
        path_strings = Path(path).parts[-3:]
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
            chunks=chunk_size,
            dtype=np.uint16,
            transform=[transform],
        )

    input_dataset.close()


def get_output_paths(input_paths: List[Path], output_zarr_path: Path) -> List[Path]:
    """Generates a mirrored output path list given an input list of positions"""
    list_output_path = []
    for path in input_paths:
        # Select the Row/Column/FOV parts of input path
        path_strings = Path(path).parts[-3:]
        # Append the same Row/Column/FOV to the output zarr path
        list_output_path.append(Path(output_zarr_path, *path_strings))
    return list_output_path


def deskew_zyx_and_save(
    position: Position, output_path: Path, settings, keep_overhang: bool, t: int, c: int
) -> None:
    """Load a zyx array from a Position object, deskew it, and save the result to file"""
    click.echo(f"Deskewing c={c}, t={t}")
    zyx_data = position[0][t, c]

    # Deskew
    deskewed = deskew_data(
        zyx_data, settings.ls_angle_deg, settings.px_to_scan_ratio, keep_overhang
    )
    # Write to file
    with open_ome_zarr(output_path, mode="r+") as output_dataset:
        output_dataset[0][t, c] = deskewed
        output_dataset.zattrs["deskewing"] = asdict(settings)

    click.echo(f"Finished Writing.. c={c}, t={t}")


def deskew_single_position(
    input_data_path: Path,
    output_path: Path = './deskewed.zarr',
    deskew_param_path: Path = './deskew_setting.yml',
    keep_overhang: bool = False,
    num_processes: int = mp.cpu_count(),
) -> None:
    """Deskew a single position with multiprocessing parallelization over T and C"""

    # Get the reader and writer
    click.echo(f'Input data path:\t{input_data_path}')
    click.echo(f'Output data path:\t{str(output_path)}')
    input_dataset = open_ome_zarr(str(input_data_path))
    click.echo(input_dataset.print_tree())

    settings = deskew_params_from_file(deskew_param_path)
    T, C, Z, Y, X = input_dataset.data.shape
    click.echo(f'Dataset shape:\t{input_dataset.data.shape}')

    # Loop through (T, C), deskewing and writing as we go
    click.echo(f"Starting multiprocess pool with {num_processes} processes")
    with mp.Pool(num_processes) as p:
        p.starmap(
            partial(
                deskew_zyx_and_save, input_dataset, str(output_path), settings, keep_overhang
            ),
            itertools.product(range(T), range(C)),
        )


@click.command()
@input_data_paths_argument()
@deskew_param_argument()
@output_dataset_options(default="./deskewed.zarr")
@click.option(
    "--keep-overhang",
    "-ko",
    default=False,
    is_flag=True,
    help="Keep the overhanging region.",
)
@click.option(
    "--num-processes",
    "-j",
    default=mp.cpu_count(),
    help="Number of cores",
    required=False,
    type=int,
)
def deskew(
    input_paths: List[str],
    deskew_param_path: str,
    output_path: str,
    keep_overhang: bool,
    num_processes: int,
):
    "Deskews a single position across T and C axes using a parameter file generated by estimate_deskew.py"
    if isinstance(open_ome_zarr(input_paths[0]), Plate):
        raise ValueError(
            "Please supply a single position instead of an HCS plate. Likely fix: replace input.zarr with 'input.zarr/0/0/0'"
        )

    # Sort the input as nargs=-1 will not be natsorted
    input_paths = [Path(path) for path in natsorted(input_paths)]

    # Convert string paths to Path objects
    output_path = Path(output_path)
    deskew_param_path = Path(deskew_param_path)

    # Handle single position or wildcard filepath
    output_paths = get_output_paths(input_paths, output_path)
    click.echo(f'List of input_pos:{input_paths} output_pos:{output_paths}')

    # Create a zarr store output to mirror the input
    create_empty_zarr(input_paths, deskew_param_path, output_path, keep_overhang)

    # Loop over positions
    for input_position_path, output_position_path in zip(input_paths, output_paths):
        deskew_single_position(
            input_data_path=input_position_path,
            output_path=output_position_path,
            deskew_param_path=deskew_param_path,
            keep_overhang=keep_overhang,
            num_processes=num_processes,
        )
