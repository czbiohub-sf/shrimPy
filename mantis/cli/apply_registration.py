import contextlib
import io
import itertools
import multiprocessing as mp

from functools import partial
from pathlib import Path
from typing import List

import click

from iohub.ngff import Plate, Position, open_ome_zarr
from iohub.ngff_meta import TransformationMeta
from natsort import natsorted

from mantis.analysis import registration
from mantis.cli.deskew import get_output_paths
from mantis.cli.parsing import input_data_paths_argument, output_dataset_options


def create_empty_zarr(
    position_paths: List[Path], registration_param_path: Path, output_path: Path
) -> None:
    """Create an empty zarr array for the registering channel"""
    # Load the first position to infer dataset information
    input_dataset = open_ome_zarr(str(position_paths[0]), mode="r")
    T, C, Z, Y, X = input_dataset.data.shape

    # Get the deskewing parameters
    # TODO: where should we get the voxel size from?
    with open_ome_zarr(registration_param_path, mode="r") as parameters:
        zyx_output_shape = tuple(parameters.zattrs["registration"]["channel_2_shape"])
        voxel_size = tuple(parameters.zattrs["registration"]["voxel_size"])
    click.echo("Creating empty array...")

    # Handle transforms and metadata
    transform = TransformationMeta(
        type="scale",
        scale=2 * (1,) + voxel_size,
    )

    # Prepare output dataset
    channel_names = input_dataset.channel_names

    # Output shape based on the type of reconstruction
    output_shape = (T, len(channel_names)) + zyx_output_shape
    click.echo(f"Number of positions: {len(position_paths)}")
    click.echo(f"Output shape: {output_shape}")
    # Create output dataset
    output_dataset = open_ome_zarr(
        output_path, layout="hcs", mode="w", channel_names=channel_names
    )
    chunk_size = (1, 1, 10) + zyx_output_shape[1:]
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
            + zyx_output_shape,
            chunks=chunk_size,
            dtype=input_dataset[0].dtype,
            transform=[transform],
        )

    input_dataset.close()


def register_zyx_and_save(
    position: Position,
    output_path: Path,
    registration_param_path: Path,
    t_idx: int,
    c_idx: int,
) -> None:
    """Load a zyx array from a Position object, deskew it, and save the result to file"""
    click.echo(f"Registering c={c_idx}, t={t_idx}")
    zyx_data = position[0][t_idx, c_idx]

    with open_ome_zarr(registration_param_path, mode="r") as registration_parameters:
        affine_transform_zyx = registration_parameters["affine_transform_zyx"][0, 0, 0]
        output_shape = tuple(registration_parameters.zattrs["registration"]["channel_2_shape"])

    # Deskew
    registered = registration.register_data(
        zyx_data, affine_transform_zyx, output_shape, inverse=True
    )
    # Write to file
    with open_ome_zarr(output_path, mode="r+") as output_dataset:
        output_dataset[0][t_idx, c_idx] = registered
        output_dataset.zattrs["registration_parameters"] = str(registration_param_path)

    click.echo(f"Finished Writing.. c={c_idx}, t={t_idx}")


def deskew_single_position(
    input_data_path: Path,
    output_path: Path = "./registered.zarr",
    registration_param_path: Path = "./registration_parameters.zarr",
    num_processes: int = mp.cpu_count(),
) -> None:
    """Register a single position with multiprocessing parallelization over T and C"""

    # Get the reader and writer
    click.echo(f"Input data path:\t{input_data_path}")
    click.echo(f"Output data path:\t{str(output_path)}")
    input_dataset = open_ome_zarr(str(input_data_path))
    stdout_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer):
        input_dataset.print_tree()
    click.echo(f" Zarr Store info: {stdout_buffer.getvalue()}")

    T, C, Z, Y, X = input_dataset.data.shape

    click.echo(f"Dataset shape:\t{input_dataset.data.shape}")

    # Loop through (T, C), deskewing and writing as we go
    click.echo(f"Starting multiprocess pool with {num_processes} processes")
    with mp.Pool(num_processes) as p:
        p.starmap(
            partial(
                register_zyx_and_save,
                input_dataset,
                str(output_path),
                registration_param_path,
            ),
            itertools.product(range(T), range(C)),
        )


@click.command()
@input_data_paths_argument()
@click.argument("registration_param_path", type=click.Path(exists=True))
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
    input_paths: List[str],
    registration_param_path: str,
    output_path: str,
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
    registration_param_path = Path(registration_param_path)

    # Handle single position or wildcard filepath
    output_paths = get_output_paths(input_paths, output_path)
    click.echo(f"List of input_pos:{input_paths} output_pos:{output_paths}")

    # Create a zarr store output to mirror the input
    create_empty_zarr(input_paths, registration_param_path, output_path)

    # Loop over positions
    for input_position_path, output_position_path in zip(input_paths, output_paths):
        deskew_single_position(
            input_data_path=input_position_path,
            output_path=output_position_path,
            registration_param_path=registration_param_path,
            num_processes=num_processes,
        )
