import pandas as pd
from pathlib import Path
from natsort import natsorted
import multiprocessing as mp
from tqdm import tqdm
from iohub import open_ome_zarr
from waveorder.focus import focus_from_transverse_band
import glob
import os

NA_DET = 1.35
LAMBDA_ILL = 0.500


def estimate_position_focus(
    input_data_path,
    crop_size_x,
    crop_size_y,
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

        for t_idx in tqdm(range(T)):
            for c_idx in range(C):
                focal_plane = focus_from_transverse_band(
                    dataset[0][
                        t_idx,
                        c_idx,
                        :,
                        Y // 2 - crop_size_y // 2 : Y // 2 + crop_size_y // 2,
                        X // 2 - crop_size_x // 2 : X // 2 + crop_size_x // 2,
                    ],
                    **focus_params,
                )
                pos_idx = '/'.join(input_data_path.parts[-3:])
                position_stats_stablized["position_idx"].append(pos_idx)
                position_stats_stablized["time_min"].append(t_idx * T_scale)
                position_stats_stablized["channel"].append(channel_names[c_idx])
                position_stats_stablized["focal_idx"].append(focal_plane)

        position_focus_stats_df = pd.DataFrame(position_stats_stablized)
        filename = output_dir / f"position_" + pos_idx + "_stats.csv"
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
        dataframes.append(
            pd.read_csv(csv_file).assign(position=int(csv_file.stem.split("_")[1]))
        )
        if remove_intermediate_csv:
            os.remove(csv_file)

    pd.concat(dataframes, ignore_index=True).to_csv(output_csv_file_path, index=False)


def calculate_z_drift(
    input_data_paths: Path,
    output_folder_path: Path,
    num_processes: int = 1,
    crop_size_x: int = 300,
    crop_size_y: int = 300,
) -> None:
    input_data_paths = [Path(path) for path in natsorted(glob.glob(input_data_paths))]

    output_folder_path = Path(output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    position_focus_folder = output_folder_path / 'positions_focus'
    with mp.Pool(processes=num_processes) as pool:
        list(
            tqdm(
                pool.imap(
                    lambda input_data_path: estimate_position_focus(
                        input_data_path, crop_size_x, crop_size_y, position_focus_folder
                    ),
                    input_data_paths,
                ),
                total=len(input_data_paths),
            )
        )
        pool.close()
        pool.join()

    combine_dataframes(
        position_focus_folder,
        output_folder_path + 'positions_focus.csv',
        remove_intermediate_csv=True,
    )


if __name__ == '__main__':
    input_data_path = Path(
        "/hpc/projects/comp.micro/mantis/2023_09_21_OpenCell_targets/1-recon/opencell_timelapse_2_phase.zarr"
    )
    output_folder_path = "./test_position_stats"
    num_processes = 16
    crop_size_x = 1200  # Size of center crop from FOV
    crop_size_y = 1200  # Size of center crop from FOV
