import contextlib
import io
import itertools
import multiprocessing as mp

from functools import partial
from pathlib import Path

import ants
import click
import numpy as np

from iohub.ngff import Position, open_ome_zarr

from mantis.analysis.register import numpy_to_ants_transform_zyx
from mantis.cli.utils import _check_nan_n_zeros


def stabilization_over_time_ants(
    position: Position,
    output_path: Path,
    list_of_shifts: np.ndarray,
    input_channel_idx: list,
    output_channel_idx: list,
    t_idx: int,
    c_idx: int,
    **kwargs,
) -> None:
    """Load a zyx array from a Position object, apply a transformation and save the result to file"""

    click.echo(f"Processing c={c_idx}, t={t_idx}")
    tx_shifts = numpy_to_ants_transform_zyx(list_of_shifts[t_idx])

    # Process CZYX vs ZYX
    if input_channel_idx is not None:
        czyx_data = position.data.oindex[t_idx, input_channel_idx]
        if not _check_nan_n_zeros(czyx_data):
            for c in input_channel_idx:
                print(f'czyx_data.shape {czyx_data.shape}')
                zyx_data_ants = ants.from_numpy(czyx_data[0])
                registered_zyx = tx_shifts.apply_to_image(
                    zyx_data_ants, reference=zyx_data_ants
                )
                # Write to file
                with open_ome_zarr(output_path, mode="r+") as output_dataset:
                    output_dataset[0].oindex[
                        t_idx, output_channel_idx
                    ] = registered_zyx.numpy()
            click.echo(f"Finished Writing.. t={t_idx}")
        else:
            click.echo(f"Skipping t={t_idx} due to all zeros or nans")
    else:
        zyx_data = position.data.oindex[t_idx, c_idx]
        # Checking if nans or zeros and skip processing
        if not _check_nan_n_zeros(zyx_data):
            zyx_data_ants = ants.from_numpy(zyx_data)
            # Apply transformation
            registered_zyx = tx_shifts.apply_to_image(zyx_data_ants, reference=zyx_data_ants)

            # Write to file
            with open_ome_zarr(output_path, mode="r+") as output_dataset:
                output_dataset[0][t_idx, c_idx] = registered_zyx.numpy()

            click.echo(f"Finished Writing.. c={c_idx}, t={t_idx}")
        else:
            click.echo(f"Skipping c={c_idx}, t={t_idx} due to all zeros or nans")


def apply_stabilization_over_time_ants(
    list_of_shifts: list,
    input_data_path: Path,
    output_path: Path,
    time_indices: list = [0],
    input_channel_idx: list = [],
    output_channel_idx: list = [],
    num_processes: int = 1,
    **kwargs,
) -> None:
    """Apply stabilization over time"""
    # Function to be applied
    # Get the reader and writer
    click.echo(f"Input data path:\t{input_data_path}")
    click.echo(f"Output data path:\t{str(output_path)}")
    input_dataset = open_ome_zarr(str(input_data_path))
    stdout_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer):
        input_dataset.print_tree()
    click.echo(f" Input data tree: {stdout_buffer.getvalue()}")

    T, C, _, _, _ = input_dataset.data.shape

    # Write the settings into the metadata if existing
    # TODO: alternatively we can throw all extra arguments as metadata.
    if 'extra_metadata' in kwargs:
        # For each dictionary in the nest
        with open_ome_zarr(output_path, mode='r+') as output_dataset:
            for params_metadata_keys in kwargs['extra_metadata'].keys():
                output_dataset.zattrs['extra_metadata'] = kwargs['extra_metadata']

    # Loop through (T, C), deskewing and writing as we go
    click.echo(f"\nStarting multiprocess pool with {num_processes} processes")

    if input_channel_idx is None or len(input_channel_idx) == 0:
        # If C is not empty, use itertools.product with both ranges
        _, C, _, _, _ = input_dataset.data.shape
        iterable = itertools.product(time_indices, range(C))
        partial_stabilization_over_time_ants = partial(
            stabilization_over_time_ants,
            input_dataset,
            output_path / Path(*input_data_path.parts[-3:]),
            list_of_shifts,
            None,
            None,
        )
    else:
        # If C is empty, use only the range for time_indices
        iterable = itertools.product(time_indices)
        partial_stabilization_over_time_ants = partial(
            stabilization_over_time_ants,
            input_dataset,
            output_path / Path(*input_data_path.parts[-3:]),
            list_of_shifts,
            input_channel_idx,
            output_channel_idx,
            c_idx=0,
        )

    with mp.Pool(num_processes) as p:
        p.starmap(
            partial_stabilization_over_time_ants,
            iterable,
        )
    input_dataset.close()
