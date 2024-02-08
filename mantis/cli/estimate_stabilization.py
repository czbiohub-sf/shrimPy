import itertools
import multiprocessing as mp

from functools import partial
from pathlib import Path
from typing import Tuple

import click
import numpy as np
import pandas as pd

from iohub.ngff import open_ome_zarr
from pandas import DataFrame
from pystackreg import StackReg
from waveorder.focus import focus_from_transverse_band

from mantis.analysis.AnalysisSettings import StabilizationSettings
from mantis.cli.parsing import input_position_dirpaths, output_filepath
from mantis.cli.utils import model_to_yaml

NA_DET = 1.35
LAMBDA_ILL = 0.500


# TODO: Do we need to compute focus fiding on n_number of channels?
def estimate_position_focus(
    input_data_path: Path,
    input_channel_indices: Tuple[int, ...],
    crop_size_xy: list[int, int],
):
    position_idx, time_min, channel, channel_idx, focal_idx = [], [], [], [], []

    with open_ome_zarr(input_data_path) as dataset:
        channel_names = dataset.channel_names
        T, C, Z, Y, X = dataset[0].shape
        T_scale, _, _, _, X_scale = dataset.scale

        for tc_idx in itertools.product(range(T), input_channel_indices):
            data_zyx = dataset.data[tc_idx][
                :,
                Y // 2 - crop_size_xy[1] // 2 : Y // 2 + crop_size_xy[1] // 2,
                X // 2 - crop_size_xy[0] // 2 : X // 2 + crop_size_xy[0] // 2,
            ]

            # if the FOV is empty, set the focal plane to 0
            if np.sum(data_zyx) == 0:
                focal_plane = 0
            else:
                focal_plane = focus_from_transverse_band(
                    data_zyx,
                    NA_det=NA_DET,
                    lambda_ill=LAMBDA_ILL,
                    pixel_size=X_scale,
                )

            pos_idx = '/'.join(input_data_path.parts[-3:]).replace('/', '_')
            position_idx.append(pos_idx)
            time_min.append(tc_idx[0] * T_scale)
            channel.append(channel_names[tc_idx[1]])
            channel_idx.append(tc_idx[1])
            focal_idx.append(focal_plane)

    position_stats_stabilized = {
        "position_idx": position_idx,
        "time_min": time_min,
        "channel": channel,
        "channel_idx": channel_idx,
        "focal_idx": focal_idx,
    }
    return position_stats_stabilized


def get_mean_z_positions(
    input_dataframe: DataFrame, z_drift_channel_idx: int = 0, verbose: bool = False
) -> None:
    import matplotlib.pyplot as plt

    z_drift_df = pd.read_csv(input_dataframe)

    # Filter the DataFrame for 'channel A'
    phase_3D_df = z_drift_df[z_drift_df["channel_idx"] == z_drift_channel_idx]
    # Sort the DataFrame based on 'time_min'
    phase_3D_df = phase_3D_df.sort_values("time_min")

    # TODO: this is a hack to deal with the fact that the focus finding function returns 0 if it fails
    phase_3D_df["focal_idx"] = phase_3D_df["focal_idx"].replace(0, method="ffill")
    # phase_3D_df["focal_idx"] = phase_3D_df["focal_idx"].replace(0,np.nan)

    # Get the mean of positions for each time point
    average_focal_idx = phase_3D_df.groupby("time_min")["focal_idx"].mean().reset_index()
    if verbose:
        # Get the moving average of the focal_idx
        plt.plot(average_focal_idx["focal_idx"], linestyle="--", label="mean of all positions")
        plt.legend()
        plt.savefig("./z_drift.png")
    return average_focal_idx["focal_idx"].values


def estimate_z_stabilization(
    input_data_paths: Path,
    output_folder_path: Path,
    z_drift_channel_idx: int = 0,
    num_processes: int = 1,
    crop_size_xy: list[int, int] = [600, 600],
    verbose: bool = False,
) -> np.ndarray:
    output_folder_path.mkdir(parents=True, exist_ok=True)

    fun = partial(
        estimate_position_focus,
        input_channel_indices=(z_drift_channel_idx,),
        crop_size_xy=crop_size_xy,
    )
    # TODO: do we need to natsort the input_data_paths?
    with mp.Pool(processes=num_processes) as pool:
        position_stats_stabilized = pool.map(fun, input_data_paths)

    df = pd.concat([pd.DataFrame.from_dict(stats) for stats in position_stats_stabilized])
    df.to_csv(output_folder_path / 'positions_focus.csv', index=False)

    # Calculate and save the output file
    z_drift_offsets = get_mean_z_positions(
        output_folder_path / 'positions_focus.csv',
        z_drift_channel_idx=z_drift_channel_idx,
        verbose=verbose,
    )

    # Calculate the z focus shift matrices
    z_focus_shift = [np.eye(4)]
    # Find the z focus shift matrices for each time point based on the z_drift_offsets relative to the first timepoint.
    z_val = z_drift_offsets[0]
    for z_val_next in z_drift_offsets[1:]:
        shift = np.eye(4)
        shift[0, 3] = z_val_next - z_val
        z_focus_shift.append(shift)
    z_focus_shift = np.array(z_focus_shift)

    if verbose:
        click.echo(f"Saving z focus shift matrices to {output_folder_path}")
        z_focus_shift_filepath = output_folder_path / "z_focus_shift.npy"
        np.save(z_focus_shift_filepath, z_focus_shift)

    return z_focus_shift


def estimate_xy_stabilization(
    input_data_paths: Path,
    output_folder_path: Path,
    c_idx: int = 0,
    crop_size_xy: list[int, int] = (400, 400),
    verbose: bool = False,
) -> np.ndarray:
    input_position = open_ome_zarr(input_data_paths[0])
    output_folder_path = Path(output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    focus_params = {
        "NA_det": NA_DET,
        "lambda_ill": LAMBDA_ILL,
        "pixel_size": input_position.scale[-1],
    }

    # Get metadata
    T, C, Z, Y, X = input_position.data.shape
    X_slice = slice(X // 2 - crop_size_xy[0] // 2, X // 2 + crop_size_xy[0] // 2)
    Y_slice = slice(Y // 2 - crop_size_xy[1] // 2, Y // 2 + crop_size_xy[1] // 2)

    z_idx = focus_from_transverse_band(
        input_position[0][
            0,
            c_idx,
            :,
            Y_slice,
            X_slice,
        ],
        **focus_params,
    )
    if verbose:
        click.echo(f"Estimated in-focus slice: {z_idx}")

    # Load timelapse
    xy_timelapse = input_position[0][:T, c_idx, z_idx, Y_slice, X_slice]
    minimum = xy_timelapse.min()

    xy_timelapse = xy_timelapse + minimum  # Ensure negative values are not present

    # register each frame to the previous (already registered) one
    # this is what the original StackReg ImageJ plugin uses
    sr = StackReg(StackReg.TRANSLATION)

    T_stackreg = sr.register_stack(xy_timelapse, reference="previous", axis=0)

    # Swap values in the array since stackreg is xy and we need yx
    for subarray in T_stackreg:
        subarray[0, 2], subarray[1, 2] = subarray[1, 2], subarray[0, 2]

    T_zyx_shift = np.zeros((T_stackreg.shape[0], 4, 4))
    T_zyx_shift[:, 1:4, 1:4] = T_stackreg
    T_zyx_shift[:, 0, 0] = 1

    # Save the translation matrices
    if verbose:
        click.echo(f"Saving translation matrices to {output_folder_path}")
        yx_shake_translation_tx_filepath = (
            output_folder_path / "yx_shake_translation_tx_ants.npy"
        )
        np.save(yx_shake_translation_tx_filepath, T_zyx_shift)

    input_position.close()

    return T_zyx_shift


@click.command()
@input_position_dirpaths()
@output_filepath()
@click.option(
    "--num-processes",
    "-j",
    default=1,
    help="Number of parallel processes. Default is 1.",
    required=False,
    type=int,
)
@click.option(
    "--channel-index",
    "-c",
    default=0,
    help="Channel index used for estimating stabilization parameters. Default is 0.",
    required=False,
    type=int,
)
@click.option(
    "--stabilize_xy",
    "-y",
    is_flag=True,
    help="Estimate yx drift and apply to the input data. Default is False.",
)
@click.option(
    "--stabilize_z",
    "-z",
    is_flag=True,
    help="Estimate z drift and apply to the input data. Default is False.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    type=bool,
    help="Stabilization verbose. Default is False.",
)
@click.option(
    "--crop-size-xy",
    nargs=2,
    type=int,
    default=[300, 300],
    help="Crop size in xy. Enter two integers. Default is 300 300.",
)
@click.option(
    "--stabilization-channel-indices",
    help="Indices of channels which will be stabilized. Default is all channels.",
    multiple=True,
    type=int,
    default=[],
)
def estimate_stabilization(
    input_position_dirpaths,
    output_filepath,
    num_processes,
    channel_index,
    stabilize_xy,
    stabilize_z,
    verbose,
    crop_size_xy,
    stabilization_channel_indices,
):
    """
    Estimate the Z and/or XY timelapse stabilization matrices.

    This function estimates xy and z drifts and returns the affine matrices per timepoint taking t=0 as reference saved as a yaml file.
    The level of verbosity can be controlled with the verbose flag.
    The size of the crop in xy can be specified with the crop-size-xy option.

    Example usage:
    mantis stabilization -i ./timelapse.zarr/0/0/0 -o ./stabilization.yml -y -z -v --crop-size-xy 300 300

    Note: the verbose output will be saved at the same level as the output zarr.
    """
    assert (
        stabilize_xy or stabilize_z
    ), "At least one of 'stabilize_xy' or 'stabilize_z' must be selected"

    assert output_filepath.suffix == ".yml", "Output file must be a yaml file"

    output_dirpath = output_filepath.parent
    output_dirpath.mkdir(parents=True, exist_ok=True)

    # Channel names to process
    stabilization_channel_names = []
    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        channel_names = dataset.channel_names
    if len(stabilization_channel_indices) < 1:
        stabilization_channel_indices = range(len(channel_names))
        stabilization_channel_names = channel_names
    else:
        # Make the input a list
        stabilization_channel_indices = list(stabilization_channel_indices)
        stabilization_channel_names = []
        # Check the channel indeces are valid
        for c_idx in stabilization_channel_indices:
            if c_idx not in range(len(channel_names)):
                raise ValueError(
                    f"Channel index {c_idx} is not valid. Please provide channel indeces from 0 to {len(channel_names)-1}"
                )
            else:
                stabilization_channel_names.append(channel_names[c_idx])

    # Estimate z drift
    if stabilize_z:
        click.echo("Estimating z stabilization parameters")
        T_z_drift_mats = estimate_z_stabilization(
            input_data_paths=input_position_dirpaths,
            output_folder_path=output_dirpath,
            z_drift_channel_idx=channel_index,
            num_processes=num_processes,
            crop_size_xy=crop_size_xy,
            verbose=verbose,
        )
        stabilization_type = "z"

    # Estimate yx drift
    if stabilize_xy:
        click.echo("Estimating xy stabilization parameters")
        T_translation_mats = estimate_xy_stabilization(
            input_data_paths=input_position_dirpaths,
            output_folder_path=output_dirpath,
            c_idx=channel_index,
            crop_size_xy=crop_size_xy,
            verbose=verbose,
        )
        stabilization_type = "xy"

    if stabilize_z and stabilize_xy:
        if T_translation_mats.shape[0] != T_z_drift_mats.shape[0]:
            raise ValueError(
                "The number of translation matrices and z drift matrices must be the same"
            )
        combined_mats = np.array([a @ b for a, b in zip(T_translation_mats, T_z_drift_mats)])
        stabilization_type = "xyz"

    # NOTE: we've checked that one of the two conditions below is true
    elif stabilize_z:
        combined_mats = T_z_drift_mats
    elif stabilize_xy:
        combined_mats = T_translation_mats

    # Save the combined matrices
    model = StabilizationSettings(
        stabilization_type=stabilization_type,
        stabilization_estimation_channel=channel_names[channel_index],
        stabilization_channels=stabilization_channel_names,
        affine_transform_zyx_list=combined_mats.tolist(),
        time_indices="all",
    )
    model_to_yaml(model, output_filepath)


if __name__ == "__main__":
    estimate_stabilization()
