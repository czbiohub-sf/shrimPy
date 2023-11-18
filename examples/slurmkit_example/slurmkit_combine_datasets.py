# %%

from iohub import open_ome_zarr
import numpy as np
from pathlib import Path

from mantis.cli import utils
from natsort import natsorted
import glob
import multiprocessing as mp
import itertools
from functools import partial

# %%
phase_data_paths = '/hpc/projects/comp.micro/mantis/2023_11_08_Opencell_infection/6-phase_stabilized_crop/OC43_infection_timelapse_3_registered_stabilized_cropped.zarr/0/2/000000'
fluor_data_paths = '/hpc/projects/comp.micro/mantis/2023_11_08_Opencell_infection/6-fluor-cropped/OC43_infection_timelapse_3_cropped.zarr/0/2/000000'
output_data_path = './test_output.zarr'

# Add the paths to lists to be processed
data_paths = [phase_data_paths, fluor_data_paths]

Z_CHUNK = 5
num_processes = 3

all_channel_names = []
all_data_paths = []
input_channel_indeces = []
output_channel_indeces = []
for idx, paths in enumerate(data_paths):
    # Parse the data paths
    parsed_paths = [Path(path) for path in natsorted(glob.glob(paths))]
    all_data_paths.extend(parsed_paths)

    # Open the dataset and get the channel names
    with open_ome_zarr(parsed_paths[0]) as dataset:
        T, C, Z, Y, X = dataset.data.shape
        channel_names = dataset.channel_names

        # Add the current length of all_channel_names to each channel index to ensure uniqueness
        output_indices = [len(all_channel_names) + i for i in range(len(channel_names))]
        # Add the channel names to the list
        all_channel_names.extend(channel_names)
        # Generate a matching list of input_channel_indices for each file in parsed_paths
        input_channel_indices = [
            [index for index, name in enumerate(channel_names)] for _ in parsed_paths
        ]
        input_channel_indeces.extend(input_channel_indices)

        # Add the output indices for each file in parsed_paths
        output_channel_indeces.extend([output_indices for _ in parsed_paths])

print(f'channel_names: {all_channel_names}')

chunk_zyx_shape = (Z_CHUNK, Y, X)
output_metadata = {
    "shape": (T, len(all_channel_names), Z, Y, X),
    "chunks": (1,) * 2 + chunk_zyx_shape,
    "scale": dataset.scale,
    "channel_names": all_channel_names,
    "dtype": np.float32,
}

# Using the first path as a template for the input data path
utils.create_empty_hcs_zarr(
    store_path=output_data_path,
    position_keys=[p.parts[-3:] for p in all_data_paths[0:1]],
    **output_metadata,
)

Z_slice = slice(0, Z)
Y_slice = slice(0, Y)
X_slice = slice(0, X)

copy_n_paste_kwargs = {
    "czyx_slicing_params": [Z_slice, Y_slice, X_slice],
}

# %%
for in_path, input_c_idx, output_c_idx in zip(
    all_data_paths, input_channel_indeces, output_channel_indeces
):
    utils.merge_datasets(
        input_data_path=in_path,
        output_path=output_data_path,
        time_indices='all',
        input_channel_idx=input_c_idx,
        output_channel_idx=output_c_idx,
        num_processes=num_processes,
        **copy_n_paste_kwargs,
    )
