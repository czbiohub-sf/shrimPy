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


def calculate_z_drift(
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
    for i in range(len(z_drift_offsets) - 1):
        z_val = z_drift_offsets[0]
        z_val_next = z_drift_offsets[i + 1]
        z_focus_shift.append(
            np.array(
                [
                    [1, 0, 0, (z_val_next - z_val)],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
        )
    z_focus_shift = np.array(z_focus_shift)
    if verbose:
        print(f"Saving z focus shift matrices to {output_folder_path}")
        z_focus_shift_filepath = output_folder_path / "z_focus_shift.npy"
        np.save(z_focus_shift_filepath, z_focus_shift)

    return z_focus_shift


def calculate_yx_stabilization(
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

    print('Estimating in-focus slice')
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
    print(f"Estimated in-focus slice: {z_idx}")
    # Load timelapse
    xy_timelapse = input_position[0][:T, c_idx, z_idx, Y_slice, X_slice]
    minimum = xy_timelapse.min()

    xy_timelapse = xy_timelapse + minimum  # Ensure negative values are not present

    # register each frame to the previous (already registered) one
    # this is what the original StackReg ImageJ plugin uses
    sr = StackReg(StackReg.TRANSLATION)

    print("Finding XY translation matrices")
    T_stackreg = sr.register_stack(xy_timelapse, reference="previous", axis=0)

    # Swap values in the array since stackreg is xy and we need yx
    for subarray in T_stackreg:
        subarray[0, 2], subarray[1, 2] = subarray[1, 2], subarray[0, 2]

    T_zyx_shift = np.zeros((T_stackreg.shape[0], 4, 4))
    T_zyx_shift[:, 1:4, 1:4] = T_stackreg
    T_zyx_shift[:, 0, 0] = 1

    # Save the translation matrices
    if verbose:
        print(f"Saving translation matrices to {output_folder_path}")
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
    "--estimate-yx-drift",
    "-y",
    is_flag=True,
    help="Estimate yx drift and apply to the input data. Default is False.",
)
@click.option(
    "--estimate-z-drift",
    "-z",
    is_flag=True,
    help="Estimate z drift and apply to the input data. Default is False.",
)
@click.option(
    "--stabilization-verbose",
    "-v",
    is_flag=True,
    type=bool,
    help="Stabilization verbose. Default is False.",
)
@click.option(
    "--crop-size-xy",
    "-s",
    nargs=2,
    type=int,
    default=[300, 300],
    help="Crop size in xy. Enter two integers. Default is 300 300.",
)
def estimate_stabilization(
    input_position_dirpaths,
    output_filepath,
    num_processes,
    channel_index,
    estimate_yx_drift,
    estimate_z_drift,
    stabilization_verbose,
    crop_size_xy,
):
    """
    Estimate the Z and/or XY timelapse stabilization matrices.

    This function estimates yx and z drifts and returns the affine matrices per timepoint taking t=0 as reference saved as a yaml file.
    The level of verbosity can be controlled with the stabilization_verbose flag.
    The size of the crop in xy can be specified with the crop-size-xy option.

    Example usage:
    mantis stabilization -i ./timelapse.zarr/0/0/0 -o ./stabilization.yml -y -z -v -s 300 300

    Note: the verbose output will be saved at the same level as the output zarr.
    """
    assert (
        estimate_yx_drift or estimate_z_drift
    ), "At least one of estimate_yx_drift or estimate_z_drift must be selected"

    assert output_filepath.suffix == ".yml", "Output file must be a yaml file"

    output_dirpath = output_filepath.parent
    output_dirpath.mkdir(parents=True, exist_ok=True)

    # Estimate z drift
    if estimate_z_drift:
        T_z_drift_mats = calculate_z_drift(
            input_data_paths=input_position_dirpaths,
            output_folder_path=output_dirpath,
            z_drift_channel_idx=channel_index,
            num_processes=num_processes,
            crop_size_xy=crop_size_xy,
            verbose=stabilization_verbose,
        )

    # Estimate yx drift
    if estimate_yx_drift:
        T_translation_mats = calculate_yx_stabilization(
            input_data_paths=input_position_dirpaths,
            output_folder_path=output_dirpath,
            c_idx=channel_index,
            crop_size_xy=crop_size_xy,
            verbose=stabilization_verbose,
        )

    if estimate_z_drift and estimate_yx_drift:
        if T_translation_mats.shape[0] != T_z_drift_mats.shape[0]:
            raise ValueError(
                "The number of translation matrices and z drift matrices must be the same"
            )
        combined_mats = np.array([a @ b for a, b in zip(T_translation_mats, T_z_drift_mats)])
    # note: we've checked that one of the two conditions below is true
    elif estimate_z_drift:
        combined_mats = T_z_drift_mats
    elif estimate_yx_drift:
        combined_mats = T_translation_mats

    # Save the combined matrices
    model = StabilizationSettings(
        focus_finding_channel_index=channel_index,
        affine_transform_zyx_list=combined_mats.tolist(),
    )
    model_to_yaml(model, output_filepath)


if __name__ == "__main__":
    estimate_stabilization()
