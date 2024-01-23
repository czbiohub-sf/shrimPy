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

from mantis.analysis.AnalysisSettings import StabilizationSettings
from mantis.analysis.register import convert_transform_to_ants
from mantis.cli.parsing import config_filepath, input_position_dirpaths, output_dirpath
from mantis.cli.utils import _check_nan_n_zeros, create_empty_hcs_zarr, yaml_to_model

# TODO: this variable should probably be exposed?
Z_CHUNK = 5


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
    tx_shifts = convert_transform_to_ants(list_of_shifts[t_idx])

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


@click.command()
@input_position_dirpaths()
@output_dirpath()
@config_filepath()
@click.option(
    "--num-processes",
    "-j",
    default=1,
    help="Number of parallel processes. Default is 1.",
    required=False,
    type=int,
)
def apply_stabilization(
    input_position_dirpaths, output_dirpath, config_filepath, num_processes
):
    """
    Stabilize the timelapse input based on single position and channel.

    This function applies stabilization to the input data. It can estimate both yx and z drifts.
    The level of verbosity can be controlled with the stabilization_verbose flag.
    The size of the crop in xy can be specified with the crop-size-xy option.

    Example usage:
    mantis stabilize-timelapse -i ./timelapse.zarr/0/0/0 -o ./stabilized_timelapse.zarr -c ./file_w_matrices.yml -v

    """
    assert config_filepath.suffix == ".yml", "Config file must be a yaml file"

    # Load the config file
    settings = yaml_to_model(config_filepath, StabilizationSettings)

    combined_mats = settings.affine_transform_zyx_list
    combined_mats = np.array(combined_mats)

    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        T, C, Z, Y, X = dataset.data.shape
        channel_names = dataset.channel_names

    chunk_zyx_shape = (Z_CHUNK, Y, X)
    output_metadata = {
        "shape": (T, C, Z, Y, X),
        "chunks": (1,) * 2 + chunk_zyx_shape,
        "scale": dataset.scale,
        "channel_names": channel_names,
        "dtype": np.float32,
    }

    # Create the output zarr mirroring input_position_dirpaths
    create_empty_hcs_zarr(
        store_path=output_dirpath,
        position_keys=[p.parts[-3:] for p in input_position_dirpaths],
        **output_metadata,
    )

    # Apply the affine transformation to the input data
    for input_path in input_position_dirpaths:
        apply_stabilization_over_time_ants(
            list_of_shifts=combined_mats,
            input_data_path=input_path,
            output_path=output_dirpath,
            time_indices=list(range(T)),
            input_channel_idx=None,
            output_channel_idx=None,
            num_processes=num_processes,
        )


if __name__ == "__main__":
    apply_stabilization()
