import ants
import click
import numpy as np

from iohub.ngff import open_ome_zarr

from mantis.analysis.AnalysisSettings import StabilizationSettings
from mantis.analysis.register import convert_transform_to_ants
from mantis.cli.parsing import config_filepath, input_position_dirpaths, output_dirpath
from mantis.cli.utils import (
    copy_n_paste_czyx,
    create_empty_hcs_zarr,
    process_single_position_v2,
    yaml_to_model,
)


def apply_stabilization_transform(
    zyx_data: np.ndarray, list_of_shifts: list[np.ndarray], t_idx: int, **kwargs
):
    """Apply stabilization to a single zyx"""
    click.echo(f'shifting matrix with t_idx:{t_idx} \n{list_of_shifts[t_idx]}')
    Z, Y, X = zyx_data.shape[-3:]

    # Get the transformation matrix for the current time index
    tx_shifts = convert_transform_to_ants(list_of_shifts[t_idx])

    if zyx_data.ndim == 4:
        stabilized_czyx = np.zeros((zyx_data.shape[0], Z, Y, X), dtype=np.float32)
        for c in range(zyx_data.shape[0]):
            stabilized_czyx[c] = apply_stabilization_transform(
                zyx_data[c], list_of_shifts, t_idx
            )
        return stabilized_czyx
    else:
        zyx_data = np.nan_to_num(zyx_data, nan=0)
        zyx_data_ants = ants.from_numpy(zyx_data.astype(np.float32))
        stabilized_zyx = tx_shifts.apply_to_image(
            zyx_data_ants, reference=zyx_data_ants
        ).numpy()

    return stabilized_zyx


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
def stabilize(input_position_dirpaths, output_dirpath, config_filepath, num_processes):
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
    stabilization_channels = settings.stabilization_channels

    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        T, C, Z, Y, X = dataset.data.shape
        channel_names = dataset.channel_names
        for channel in stabilization_channels:
            if channel not in channel_names:
                raise ValueError(f"Channel <{channel}> not found in the input data")

        # NOTE: these can be modified to crop the output
        Z_slice, Y_slice, X_slice = (
            slice(0, Z),
            slice(0, Y),
            slice(0, X),
        )
        Z = Z_slice.stop - Z_slice.start
        Y = Y_slice.stop - Y_slice.start
        X = X_slice.stop - X_slice.start

    # Logic to parse time indices
    if settings.time_indices == "all":
        time_indices = list(range(T))
    elif isinstance(settings.time_indices, list):
        time_indices = settings.time_indices
    elif isinstance(settings.time_indices, int):
        time_indices = [settings.time_indices]

    output_metadata = {
        "shape": (len(time_indices), len(channel_names), Z, Y, X),
        "chunks": None,
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

    stabilize_zyx_args = {"list_of_shifts": combined_mats}
    copy_n_paste_kwargs = {"czyx_slicing_params": ([Z_slice, Y_slice, X_slice])}

    # apply stabilization to channels in the chosen channels and else copy the rest
    for input_position_path in input_position_dirpaths:
        for channel_name in channel_names:
            if channel_name in stabilization_channels:
                process_single_position_v2(
                    apply_stabilization_transform,
                    input_data_path=input_position_path,  # source store
                    output_path=output_dirpath,
                    time_indices=time_indices,
                    input_channel_idx=[channel_names.index(channel_name)],
                    output_channel_idx=[channel_names.index(channel_name)],
                    num_processes=num_processes,  # parallel processing over time
                    **stabilize_zyx_args,
                )
            else:
                process_single_position_v2(
                    copy_n_paste_czyx,
                    input_data_path=input_position_path,  # target store
                    output_path=output_dirpath,
                    time_indices=time_indices,
                    input_channel_idx=[channel_names.index(channel_name)],
                    output_channel_idx=[channel_names.index(channel_name)],
                    num_processes=num_processes,
                    **copy_n_paste_kwargs,
                )


if __name__ == "__main__":
    stabilize()
