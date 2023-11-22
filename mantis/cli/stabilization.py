import pandas as pd
from pathlib import Path
from natsort import natsorted
import multiprocessing as mp
from tqdm import tqdm
from iohub.ngff import open_ome_zarr
from waveorder.focus import focus_from_transverse_band
import glob
import os
import numpy as np
from pystackreg import StackReg
from mantis.cli import utils
from mantis.cli.parsing import (
    config_filepath,
    output_filepath,
    output_dirpath,
    input_position_dirpaths,
)
import click
from typing import Tuple

NA_DET = 1.35
LAMBDA_ILL = 0.500
Z_CHUNK = 5


def estimate_position_focus(
    input_data_path: Path,
    crop_size_xy: list[int, int],
    output_dir: Path,
):
    with open_ome_zarr(input_data_path) as dataset:
        channel_names = dataset.channel_names
        T, C, Z, Y, X = dataset[0].shape
        T_scale, _, _, _, X_scale = dataset.scale

        focus_params = {
            "NA_det": NA_DET,
            "lambda_ill": LAMBDA_ILL,
            "pixel_size": X_scale,
        }

        position_stats_stablized = {
            "position_idx": [],
            "time_min": [],
            "channel": [],
            "focal_idx": [],
        }
        print(f"Processing {input_data_path}")
        for t_idx in tqdm(range(T)):
            for c_idx in range(C):
                focal_plane = focus_from_transverse_band(
                    dataset[0][
                        t_idx,
                        c_idx,
                        :,
                        Y // 2 - crop_size_xy[1] // 2 : Y // 2 + crop_size_xy[1] // 2,
                        X // 2 - crop_size_xy[0] // 2 : X // 2 + crop_size_xy[0] // 2,
                    ],
                    **focus_params,
                )
                pos_idx = '/'.join(input_data_path.parts[-3:]).replace('/', '_')
                position_stats_stablized["position_idx"].append(pos_idx)
                position_stats_stablized["time_min"].append(t_idx * T_scale)
                position_stats_stablized["channel"].append(channel_names[c_idx])
                position_stats_stablized["focal_idx"].append(focal_plane)

        position_focus_stats_df = pd.DataFrame(position_stats_stablized)
        filename = output_dir / f"position_{pos_idx}_stats.csv"
        position_focus_stats_df.to_csv(filename)


def combine_dataframes(
    input_dir: Path, output_csv_file_path: Path, remove_intermediate_csv: bool = False
):
    """
    Combine all csv files in input_dir into a single csv file at output_csv_path

    Parameters
    ----------
    input_dir : Path
        Folder containing CSV
    output_csv_path : Path
        Path to output CSV file
    remove_intermediate_csv : bool, optional
        Remove the intermediate csv after merging, by default False
    """
    input_dir = Path(input_dir)
    dataframes = []

    for csv_file in natsorted(input_dir.glob("*.csv")):
        dataframes.append(pd.read_csv(csv_file))
        if remove_intermediate_csv:
            os.remove(csv_file)
    if remove_intermediate_csv:
        os.rmdir(input_dir)
    pd.concat(dataframes, ignore_index=True).to_csv(output_csv_file_path, index=False)


def get_mean_z_positions(input_dataframe, verbose=False) -> None:
    import matplotlib.pyplot as plt

    z_drift_df = pd.read_csv(input_dataframe)
    # Filter the 0 in focal_idx
    z_drift_df = z_drift_df[z_drift_df["focal_idx"] != 0]
    # Filter the DataFrame for 'channel A'
    phase_3D_df = z_drift_df[z_drift_df["channel"] == "Phase3D"]
    # Sort the DataFrame based on 'time_min'
    phase_3D_df = phase_3D_df.sort_values("time_min")
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
    num_processes: int = 1,
    crop_size_xy: list[int, int] = 300,
    verbose: bool = False,
) -> None:
    output_folder_path.mkdir(parents=True, exist_ok=True)

    position_focus_folder = output_folder_path / 'positions_focus'
    position_focus_folder.mkdir(parents=True, exist_ok=True)

    with mp.Pool(processes=num_processes) as pool:
        list(
            tqdm(
                pool.starmap(
                    estimate_position_focus,
                    [
                        (input_data_path, crop_size_xy, position_focus_folder)
                        for input_data_path in input_data_paths
                    ],
                ),
                total=len(input_data_paths),
            )
        )
        pool.close()
        pool.join()

    combine_dataframes(
        position_focus_folder,
        output_folder_path / 'positions_focus.csv',
        remove_intermediate_csv=True,
    )

    # Calculate and save the output file
    output_file = output_folder_path / "z_shifts_matrices.npy"
    z_drift_offsets = get_mean_z_positions(
        output_folder_path / 'positions_focus.csv', verbose=verbose
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

    print(f'Saving z_focus_shifts to {output_file}')
    np.save(output_file, z_focus_shift)
    return z_focus_shift


def calculate_yx_stabilization(
    input_data_path: Path,
    output_folder_path: Path,
    c_idx: int = 0,
    crop_size_xy: list[int, int] = (200, 200),
    verbose: bool = False,
):
    input_position = open_ome_zarr(input_data_path[0])
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
    # Load timelapse
    xy_timelapse = input_position[0][:T, c_idx, z_idx, Y_slice, X_slice]
    minimum = xy_timelapse.min()

    xy_timelapse = xy_timelapse + minimum  # Ensure negative values are not present

    # register each frame to the previous (already registered) one
    # this is what the original StackReg ImageJ plugin uses
    sr = StackReg(StackReg.TRANSLATION)

    print("Finding XY translation matrices")
    T_stackreg = sr.register_stack(xy_timelapse, reference="previous", axis=0)
    T_zyx_shift = np.zeros((T_stackreg.shape[0], 4, 4))
    T_zyx_shift[:, 1:4, 1:4] = T_stackreg
    T_zyx_shift[:, 0, 0] = 1

    # Save the translation matrices
    yx_shake_translation_tx_filepath = output_folder_path / "yx_shake_translation_tx_ants.npy"
    np.save(yx_shake_translation_tx_filepath, T_zyx_shift)

    input_position.close()

    return T_zyx_shift


@click.command()
@input_position_dirpaths()
@output_dirpath()
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
def stabilize_timelapse(
    input_position_dirpaths,
    output_dirpath,
    num_processes,
    channel_index,
    estimate_yx_drift,
    estimate_z_drift,
    stabilization_verbose,
    crop_size_xy,
):
    """
    Stabilize the timelapse input based on single position and channel.

    This function applies stabilization to the input data. It can estimate both yx and z drifts.
    The level of verbosity can be controlled with the stabilization_verbose flag.
    The size of the crop in xy can be specified with the crop-size-xy option.

    Example usage:
    mantis stabilization -i ./timelapse.zarr/0/0/0 -o ./stabilization.zarr -d -v -z -s 300 300

    Note: the verbose output will be saved at the same level as the output zarr.
    """
    assert (
        estimate_yx_drift or estimate_z_drift
    ), "At least one of estimate_yx_drift or estimate_z_drift must be selected"

    output_dirpath_parent = output_dirpath.parent
    output_dirpath_parent.mkdir(parents=True, exist_ok=True)

    # Estimate z drift
    if estimate_z_drift:
        T_z_drift_mats = calculate_z_drift(
            input_data_paths=input_position_dirpaths,
            output_folder_path=output_dirpath_parent,
            num_processes=num_processes,
            crop_size_xy=crop_size_xy,
            verbose=stabilization_verbose,
        )
        if not estimate_yx_drift:
            combined_mats = T_z_drift_mats

    # Estimate yx drift
    if estimate_yx_drift:
        T_translation_mats = calculate_yx_stabilization(
            input_data_path=input_position_dirpaths,
            output_folder_path=output_dirpath_parent,
            c_idx=channel_index,
            crop_size_xy=crop_size_xy,
            verbose=stabilization_verbose,
        )
        if estimate_z_drift:
            if T_translation_mats.shape[0] != T_z_drift_mats.shape[0]:
                raise ValueError(
                    "The number of translation matrices and z drift matrices must be the same"
                )
            else:
                combined_mats = [
                    np.dot(T_translation_mat, T_z_drift_mat)
                    for T_translation_mat, T_z_drift_mat in zip(
                        T_translation_mats, T_z_drift_mats
                    )
                ]

        else:
            combined_mats = T_translation_mats

    # Save the combined matrices
    combined_mats_filepath = output_dirpath_parent / "combined_mats.npy"
    np.save(combined_mats_filepath, combined_mats)

    # Open test dataset to get the output shape and voxel size
    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        output_zyx_shape = dataset.data.shape[-3:]
        voxel_size = dataset.scale[-3:]

    # Create empty zarr
    utils.create_empty_zarr(
        input_position_dirpaths,
        output_dirpath,
        output_zyx_shape=output_zyx_shape,
        chunk_zyx_shape=(Z_CHUNK, output_zyx_shape[-2], output_zyx_shape[-1]),
        voxel_size=tuple(voxel_size),
    )
    output_zarr_path = utils.get_output_paths(input_position_dirpaths, output_dirpath)

    # Apply the affine transformation to the input data
    for input_path, output_path in zip(input_position_dirpaths, output_zarr_path):
        utils.apply_stabilization_over_time_ants(
            list_of_shifts=combined_mats,
            input_data_path=input_path,
            output_path=output_path,
            num_processes=num_processes,
        )
