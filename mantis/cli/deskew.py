import multiprocessing as mp
import itertools
import os
import click
import numpy as np
import yaml
from pathlib import Path
from typing import List
from numpy.typing import ArrayLike

from iohub import open_ome_zarr
from iohub.ngff_meta import TransformationMeta
from mantis.analysis.AnalysisSettings import DeskewSettings
from mantis.analysis.deskew import deskew_data, get_deskewed_data_shape

from dataclasses import asdict
from functools import partial
from mantis.cli.parsing import (
    input_data_paths_argument,
    deskew_param_argument,
    output_dataset_options,
)
from natsort import natsorted


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
    position_paths: List[str], deskew_param_path: Path, output_path: Path, keep_overhang: bool
) -> None:
    """Create an empty zarr array for the deskewing"""
    # Load the "0" position to infer dataset information
    input_dataset = open_ome_zarr(str(position_paths[0]), mode="r")
    T, C, Z, Y, X = input_dataset.data.shape

    # Get the deskewing parameters
    settings = deskew_params_from_file(deskew_param_path)
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
    click.echo(f"Number of positions: {len(position_paths)}")
    click.echo(f"Output shape: {output_shape}")
    # Create output dataset
    output_dataset = open_ome_zarr(
        output_path, layout="hcs", mode="w", channel_names=channel_names
    )
    chunk_size = (1, 1, 64) + deskewed_shape[-2:]
    click.echo(f"Chunk size {chunk_size}")

    # This takes care of the logic for single position or multiple position by wildcards
    for filepath in position_paths:
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
            chunks=chunk_size,
            dtype=np.uint16,
            transform=[transform],
        )

    input_dataset.close()


def get_output_paths(list_pos: List[str], output_path: Path) -> List[str]:
    """Generates a mirrored output path list given an the input list of positions"""
    list_output_path = []
    for filepath in list_pos:
        path_strings = filepath.split(os.path.sep)[-3:]
        list_output_path.append(os.path.join(output_path, *path_strings))
    return list_output_path


def single_process(
    data_array: ArrayLike, output_path: Path, settings, keep_overhang: bool, t: int, c: int
) -> None:
    """Process a single position"""
    click.echo(f"Deskewing c={c}, t={t}")
    data = data_array[0][t, c]

    # Deskew
    deskewed = deskew_data(
        data, settings.px_to_scan_ratio, settings.ls_angle_deg, keep_overhang
    )
    # Write to file
    with open_ome_zarr(output_path, mode="r+") as output_dataset:
        output_dataset[0][t, c] = deskewed

    click.echo(f"Finished Writing.. c={c}, t={t}")


def deskew_cli(
    input_data_path: Path,
    output_path: Path = './deskewed.zarr',
    deskew_param_path: Path = './deskew.zarr',
    keep_overhang: bool = False,
    num_cores: int = mp.cpu_count(),
) -> None:
    """Deskew a single position and parallelized over T and C"""

    # Get the reader and writer
    click.echo(f'Input data path:\t{input_data_path}')
    click.echo(f'Output data path:\t{str(output_path)}')
    input_dataset = open_ome_zarr(str(input_data_path))
    click.echo(input_dataset.print_tree())

    settings = deskew_params_from_file(deskew_param_path)
    T, C, Z, Y, X = input_dataset.data.shape
    click.echo(f'Dataset shape:\t{input_dataset.data.shape}')

    deskewed_shape, voxel_size = get_deskewed_data_shape(
        (Z, Y, X),
        settings.pixel_size_um,
        settings.ls_angle_deg,
        settings.px_to_scan_ratio,
        keep_overhang,
    )

    # Loop through (T, C), deskewing and writing as we go
    click.echo(f"Starting multiprocess pool with cores {num_cores}")
    with mp.Pool(num_cores) as p:
        p.starmap(
            partial(single_process, input_dataset, str(output_path), settings, keep_overhang),
            itertools.product(range(T), range(C)),
        )
    # Write metadata
    output_zarr_root = str(output_path).split(os.path.sep)[:-3]
    output_zarr_root = os.path.join(*output_zarr_root)
    click.echo(f'output_zarr_root \t{output_zarr_root}')
    with open_ome_zarr(output_zarr_root, mode='r+') as dataset:
        dataset.zattrs["deskewing"] = asdict(settings)
        # TODO: not sure what this metadata was for
        # dataset.zattrs["mm-meta"] = input_dataset.mm_meta["Summary"]


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
    "--num-cores",
    "-j",
    default=mp.cpu_count(),
    help="Number of cores",
    required=False,
    type=int,
)
def deskew(input_paths, deskew_param_path, output_path, keep_overhang, num_cores):
    "Deskews a single position across T and C axes using a parameter file generated by estimate_deskew.py"

    # Sort the input as nargs=-1 will not be natsorted
    input_paths = natsorted(input_paths)

    # Handle single position or wildcard filepath
    output_paths = get_output_paths(input_paths, output_path)
    click.echo(f'List of input pos:{input_paths} output_pos:{output_paths}')

    # Create a zarr store output to mirror the input
    create_empty_zarr(input_paths, deskew_param_path, output_path, keep_overhang)

    # Loop over positions
    for input_path, output_path in zip(input_paths, output_paths):
        deskew_cli(
            input_data_path=input_path,
            output_path=output_path,
            deskew_param_path=deskew_param_path,
            keep_overhang=keep_overhang,
            num_cores=num_cores,
        )
