from mantis.analysis import visualization as vz
from pathlib import Path
from iohub import open_ome_zarr
import numpy as np
import glob
from natsort import natsorted
from mantis.cli import utils

if __name__ == "__main__":
    HSV_method = "JCh"  # "RO" or "PRO" or "JCh"
    hsv_channels = ["Retardance - recon", "Orientation - recon"]
    num_processes = 10
    input_data_paths = "/hpc/projects/comp.micro/zebrafish/2023_02_02_zebrafish_casper/2-prototype-reconstruction/fov6-reconstruction.zarr/*/*/*"
    output_data_path = f"./test_{HSV_method}_new_method2.zarr"

    input_data_paths = [Path(path) for path in natsorted(glob.glob(input_data_paths))]
    output_data_path = Path(output_data_path)

    # Taking the input sample
    with open_ome_zarr(input_data_paths[0], mode="r") as dataset:
        T, C, Z, Y, X = dataset.data.shape
        dataset_scale = dataset.scale
        channel_names = dataset.channel_names

        input_channel_idx = []
        # FIXME: these should be part of a config
        # FIXME:this is hardcoded to spit out 3 chans for RGB
        output_channel_idx = [0, 1, 2]
        time_indices = list(range(T))

        if HSV_method == "PRO":
            # hsv_channels = ["Orientation", "Retardance", "Phase"]
            HSV_func = vz.HSV_PRO
            hsv_func_kwargs = dict(
                channel_order=output_channel_idx, max_val_V=0.5, max_val_S=1.0
            )

        elif HSV_method == "RO":
            # hsv_channels = [
            #     "Orientation",
            #     "Retardance",
            # ]
            HSV_func = vz.HSV_RO
            hsv_func_kwargs = dict(channel_order=output_channel_idx, max_val_V=0.5)

        elif HSV_method == "JCh":
            # hsv_channels = ["Orientation", "Retardance"]
            HSV_func = vz.JCh_mapping
            hsv_func_kwargs = dict(
                channel_order=output_channel_idx, max_val_ret=150, noise_level=1
            )
        for channel in hsv_channels:
            if channel in channel_names:
                input_channel_idx.append(channel_names.index(channel))
        rgb_channel_names = ["Red", "Green", "Blue"]

    # Here the functions will output an RGB image
    output_metadata = {
        "shape": (len(time_indices), len(rgb_channel_names), Z, Y, X),
        "chunks": None,
        "scale": dataset_scale,
        "channel_names": rgb_channel_names,
        "dtype": np.float32,
    }

    utils.create_empty_hcs_zarr(
        store_path=output_data_path,
        position_keys=[p.parts[-3:] for p in input_data_paths],
        **output_metadata,
    )

    for input_position_path in input_data_paths:
        utils.process_single_position_v2(
            HSV_func,
            input_data_path=input_position_path,  # source store
            output_path=output_data_path,  # target store
            time_indices=time_indices,
            input_channel_idx=input_channel_idx,
            output_channel_idx=output_channel_idx,
            num_processes=num_processes,  # parallel processing over time
            **hsv_func_kwargs,
        )

# %%
