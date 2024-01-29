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
from mantis.cli.utils import (
    create_empty_hcs_zarr,
    yaml_to_model,
    process_single_position_v2,
)

# TODO: this variable should probably be exposed?
Z_CHUNK = 5


def stabilize_zyx(zyx_data : np.ndarray, list_of_shifts : list[np.ndarray], t_idx : int, **kwargs):
    """Apply stabilization to a single zyx array"""
    click.echo(f'shifting matrix with t_idx:{t_idx}-- {list_of_shifts[t_idx]}')
    tx_shifts = convert_transform_to_ants(list_of_shifts[t_idx])
    zyx_data_ants = ants.from_numpy(zyx_data)
    registered_zyx = tx_shifts.apply_to_image(zyx_data_ants, reference=zyx_data_ants)
    return registered_zyx.numpy()


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

    stabilize_z_args = {"list_of_shifts": combined_mats}

    # Apply the affine transformation to the input data
    for input_path in input_position_dirpaths:
        process_single_position_v2(
            stabilize_zyx,
            input_data_path=input_path,
            output_path=output_dirpath,
            time_indices=list(range(T)),
            input_channel_idx=None,
            output_channel_idx=None,
            num_processes=num_processes,
            **stabilize_z_args,
        )

if __name__ == "__main__":
    apply_stabilization()
