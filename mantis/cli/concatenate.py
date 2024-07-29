import glob

from pathlib import Path

import click
import numpy as np

from iohub import open_ome_zarr
from natsort import natsorted

from mantis.analysis.AnalysisSettings import ConcatenateSettings
from mantis.cli.parsing import config_filepath, output_dirpath
from mantis.cli.utils import (
    copy_n_paste_czyx,
    create_empty_hcs_zarr,
    process_single_position_v2,
    yaml_to_model,
)


def get_channel_combiner_metadata(
    data_paths_list: list[str], processing_channel_names: list[str]
):
    all_data_paths = []
    all_channel_names = []
    input_channel_idx = []
    output_channel_idx = []
    out_chan_idx_counter = 0
    for paths, per_datapath_channels in zip(data_paths_list, processing_channel_names):
        # Parse the data paths
        parsed_paths = [Path(path) for path in natsorted(glob.glob(paths))]
        all_data_paths.extend(parsed_paths)

        output_channel_indices = []
        input_channel_indices = []

        # NOTE: taking first file as sample to get the channel names
        dataset = open_ome_zarr(parsed_paths[0])
        channel_names = dataset.channel_names

        # Parse channels
        if per_datapath_channels == 'all':
            for i in range(len(channel_names)):
                input_channel_indices.append(i)
            for i in range(out_chan_idx_counter, out_chan_idx_counter + len(channel_names)):
                output_channel_indices.append(i)
            out_chan_idx_counter += len(channel_names)
            all_channel_names.extend(channel_names)

        elif isinstance(per_datapath_channels, list):
            for channel in per_datapath_channels:
                if channel in channel_names:
                    input_channel_indices.append(channel_names.index(channel))
                    output_channel_indices.append(out_chan_idx_counter)
                    out_chan_idx_counter += 1
                    all_channel_names.append(channel)
        dataset.close()

        # Create a list of len paths
        input_channel_idx.extend([input_channel_indices for _ in parsed_paths])
        output_channel_idx.extend([output_channel_indices for _ in parsed_paths])

    return all_data_paths, all_channel_names, input_channel_idx, output_channel_idx


def get_slice(slice_param, max_value):
    return slice(0, max_value) if slice_param == 'all' else slice(*slice_param)


@click.command()
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
def concatenate(config_filepath: str, output_dirpath: str, num_processes: int):
    """
    Concatenate datasets (with optional cropping)

    >> mantis concatenate -c ./concat.yml -o ./output_concat.zarr -j 8
    """
    # Convert to Path objects
    config_filepath = Path(config_filepath)
    output_dirpath = Path(output_dirpath)

    settings = yaml_to_model(config_filepath, ConcatenateSettings)

    (
        all_data_paths,
        all_channel_names,
        input_channel_idx_list,
        output_channel_idx_list,
    ) = get_channel_combiner_metadata(settings.concat_data_paths, settings.channel_names)

    # Open dummy FOV to get overall shape
    # NOTE: assumes all the zarrs will have the same shape
    with open_ome_zarr(all_data_paths[0]) as dataset:
        T, C, Z, Y, X = dataset.data.shape
        output_voxel_size = dataset.scale[-3:]

    # Logic to parse time indices
    if settings.time_indices == "all":
        time_indices = list(range(T))
    elif isinstance(settings.time_indices, list):
        time_indices = settings.time_indices
    elif isinstance(settings.time_indices, int):
        time_indices = [settings.time_indices]

    # Crop the data
    Z_slice = get_slice(settings.Z_slice, Z)
    Y_slice = get_slice(settings.Y_slice, Y)
    X_slice = get_slice(settings.X_slice, X)

    cropped_shape_zyx = (
        abs(Z_slice.stop - Z_slice.start),
        abs(Y_slice.stop - Y_slice.start),
        abs(X_slice.stop - X_slice.start),
    )

    # Ensure that the cropped shape is within the bounds of the original shape
    if cropped_shape_zyx[0] > Z or cropped_shape_zyx[1] > Y or cropped_shape_zyx[2] > X:
        raise ValueError("The cropped shape is larger than the original shape.")

    # TODO: make assertion for chunk size?
    if settings.chunks_czyx is not None:
        chunk_size = [1] + list(settings.chunks_czyx)
    else:
        chunk_size = settings.chunks_czyx

    # Logic for creation of zarr and metadata
    output_metadata = {
        "shape": (len(time_indices), len(all_channel_names)) + tuple(cropped_shape_zyx),
        "chunks": chunk_size,
        "scale": (1,) * 2 + tuple(output_voxel_size),
        "channel_names": all_channel_names,
        "dtype": np.float32,
    }

    # Create the output zarr mirroring source_position_dirpaths
    create_empty_hcs_zarr(
        store_path=output_dirpath,
        position_keys=[p.parts[-3:] for p in all_data_paths],
        **output_metadata,
    )

    copy_n_paste_kwargs = {"czyx_slicing_params": ([Z_slice, Y_slice, X_slice])}

    for input_position_path, input_channel_idx, output_channel_idx in zip(
        all_data_paths, input_channel_idx_list, output_channel_idx_list
    ):
        click.echo(f'input_channel_idx: {input_channel_idx}')

        process_single_position_v2(
            copy_n_paste_czyx,
            input_data_path=input_position_path,
            output_path=output_dirpath,
            time_indices=time_indices,
            input_channel_idx=input_channel_idx,
            output_channel_idx=output_channel_idx,
            num_processes=num_processes,
            **copy_n_paste_kwargs,
        )


if __name__ == "__main__":
    concatenate()
