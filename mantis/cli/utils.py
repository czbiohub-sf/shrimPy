import contextlib
import inspect
import io
import itertools
import multiprocessing as mp

from functools import partial
from pathlib import Path
from typing import Tuple

import click

from iohub.ngff import Position, open_ome_zarr
from iohub.ngff_meta import TransformationMeta


def create_empty_zarr(
    position_paths: list[Path],
    output_path: Path,
    output_zyx_shape: Tuple[int],
    chunk_zyx_shape: Tuple[int] = None,
    voxel_size: Tuple[int, float] = (1, 1, 1),
) -> None:
    """Create an empty zarr array for the deskewing"""
    # Load the first position to infer dataset information
    input_dataset = open_ome_zarr(str(position_paths[0]), mode="r")
    T, C, Z, Y, X = input_dataset.data.shape

    click.echo("Creating empty array...")

    # Handle transforms and metadata
    transform = TransformationMeta(
        type="scale",
        scale=2 * (1,) + voxel_size,
    )

    # Prepare output dataset
    channel_names = input_dataset.channel_names

    # Output shape based on the type of reconstruction
    output_shape = (T, len(channel_names)) + output_zyx_shape
    click.echo(f"Number of positions: {len(position_paths)}")
    click.echo(f"Output shape: {output_shape}")
    # Create output dataset
    output_dataset = open_ome_zarr(
        output_path, layout="hcs", mode="w", channel_names=channel_names
    )
    if chunk_zyx_shape is None:
        chunk_zyx_shape = output_zyx_shape
    chunk_size = 2 * (1,) + chunk_zyx_shape
    click.echo(f"Chunk size {chunk_size}")

    # This takes care of the logic for single position or multiple position by wildcards
    for path in position_paths:
        path_strings = Path(path).parts[-3:]
        pos = output_dataset.create_position(
            str(path_strings[0]), str(path_strings[1]), str(path_strings[2])
        )

        _ = pos.create_zeros(
            name="0",
            shape=output_shape,
            chunks=chunk_size,
            dtype=input_dataset[0].dtype,
            transform=[transform],
        )

    input_dataset.close()


def get_output_paths(input_paths: list[Path], output_zarr_path: Path) -> list[Path]:
    """Generates a mirrored output path list given an input list of positions"""
    list_output_path = []
    for path in input_paths:
        # Select the Row/Column/FOV parts of input path
        path_strings = Path(path).parts[-3:]
        # Append the same Row/Column/FOV to the output zarr path
        list_output_path.append(Path(output_zarr_path, *path_strings))
    return list_output_path


def apply_transform_to_zyx_and_save(
    func, position: Position, output_path: Path, t_idx: int, c_idx: int, **kwargs
) -> None:
    """Load a zyx array from a Position object, apply a transformation and save the result to file"""
    click.echo(f"Processing c={c_idx}, t={t_idx}")
    zyx_data = position[0][t_idx, c_idx]

    # Apply transformation
    registered_zyx = func(zyx_data, **kwargs)

    # Write to file
    with open_ome_zarr(output_path, mode="r+") as output_dataset:
        output_dataset[0][t_idx, c_idx] = registered_zyx

    click.echo(f"Finished Writing.. c={c_idx}, t={t_idx}")


def process_single_position(
    func,
    input_data_path: Path,
    output_path: Path,
    num_processes: int = mp.cpu_count(),
    **kwargs,
) -> None:
    """Register a single position with multiprocessing parallelization over T and C"""
    # Function to be applied
    click.echo(f"Function to be applied: \t{func}")

    # Get the reader and writer
    click.echo(f"Input data path:\t{input_data_path}")
    click.echo(f"Output data path:\t{str(output_path)}")
    input_dataset = open_ome_zarr(str(input_data_path))
    stdout_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer):
        input_dataset.print_tree()
    click.echo(f" Zarr Store info: {stdout_buffer.getvalue()}")

    T, C, _, _, _ = input_dataset.data.shape
    click.echo(f"Input dataset shape:\t{input_dataset.data.shape}")

    # Check the arguments for the function
    all_func_params = inspect.signature(func).parameters.keys()
    # Extract the relevant kwargs for the function 'func'
    func_args = {}
    non_func_args = {}

    for k, v in kwargs.items():
        if k in all_func_params:
            func_args[k] = v
        else:
            non_func_args[k] = v

    # Write the settings into the metadata if existing
    # TODO: alternatively we can throw all extra arguments as metadata.
    if 'extra_metadata' in non_func_args:
        # For each dictionary in the nest
        with open_ome_zarr(output_path, mode='r+') as output_dataset:
            for params_metadata_keys in kwargs['extra_metadata'].keys():
                output_dataset.zattrs['extra_metadata'] = non_func_args['extra_metadata']

    # Loop through (T, C), deskewing and writing as we go
    click.echo(f"\nStarting multiprocess pool with {num_processes} processes")
    with mp.Pool(num_processes) as p:
        p.starmap(
            partial(
                apply_transform_to_zyx_and_save,
                func,
                input_dataset,
                str(output_path),
                **func_args,
            ),
            itertools.product(range(T), range(C)),
        )
