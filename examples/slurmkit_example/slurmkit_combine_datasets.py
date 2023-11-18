# %%
import datetime
import os
import glob
from mantis.cli import utils
from slurmkit import SlurmParams, slurm_function, submit_function
from natsort import natsorted
import click
import numpy as np
from iohub import open_ome_zarr
from mantis.analysis.AnalysisSettings import RegistrationSettings
from pathlib import Path
from typing import List
from mantis.cli.utils import yaml_to_model
from mantis.cli.apply_affine import apply_affine_to_scale

# %%
# Input data paths
phase_data_paths = '/hpc/projects/comp.micro/mantis/2023_11_08_Opencell_infection/6-phase_stabilized_crop/OC43_infection_timelapse_3_registered_stabilized_cropped.zarr/0/2/000000'
fluor_data_paths = '/hpc/projects/comp.micro/mantis/2023_11_08_Opencell_infection/6-fluor-cropped/OC43_infection_timelapse_3_cropped.zarr/0/2/000000'
virtual_path = '/hpc/projects/comp.micro/mantis/2023_11_08_Opencell_infection/6-VS_stabilized_crop/OC43_infection_timelapse_3_registered_stabilized_VS_cropped.zarr/0/2/000000'
output_data_path = './test_output_2.zarr'

# Add the paths to lists to be processed
data_paths = [phase_data_paths, fluor_data_paths, virtual_path]

# sbatch and resource parameters
cpus_per_task = 12
mem_per_cpu = "8G"
time = 60  # minutes
simultaneous_processes_per_node = 10
Z_CHUNK = 5

#####
output_data_path = Path(output_data_path)
output_dir = output_data_path.parent
slurm_out_path = str(os.path.join(output_dir, "slurm_output/data_combiner-%j.out"))

# Logic to handle the datapaths and the
(
    all_data_paths,
    all_channel_names,
    input_channel_indeces,
    output_channel_indeces,
) = utils.get_channel_combiner_metadata(data_paths)

with open_ome_zarr(all_data_paths[0]) as dataset:
    T, C, Z, Y, X = dataset.data.shape
    chunk_zyx_shape = (Z_CHUNK, Y, X)
    output_metadata = {
        "shape": (T, len(all_channel_names), Z, Y, X),
        "chunks": (1,) * 2 + chunk_zyx_shape,
        "scale": dataset.scale,
        "channel_names": all_channel_names,
        "dtype": np.float32,
    }
    # CROPPING PARAMETERS
    Z_slice = slice(0, Z)
    Y_slice = slice(0, Y)
    X_slice = slice(0, X)


# Using the first path as a template for the input data path
utils.create_empty_hcs_zarr(
    store_path=output_data_path,
    position_keys=[p.parts[-3:] for p in all_data_paths[0:1]],
    **output_metadata,
)

copy_n_paste_kwargs = {
    "czyx_slicing_params": [Z_slice, Y_slice, X_slice],
}

# prepare slurm parameters
params = SlurmParams(
    partition="preempted",
    cpus_per_task=cpus_per_task,
    mem_per_cpu=mem_per_cpu,
    time=datetime.timedelta(minutes=time),
    output=slurm_out_path,
)

# wrap our utils.process_single_position() function with slurmkit
slurm_process_single_position = slurm_function(utils.merge_datasets)
combine_func = slurm_process_single_position(
    time_indices='all',
    num_processes=simultaneous_processes_per_node,
    **copy_n_paste_kwargs,
)

# generate an array of jobs by passing the in_path and out_path to slurm wrapped function
register_jobs = [
    submit_function(
        combine_func,
        slurm_params=params,
        input_data_path=in_path,
        output_path=output_data_path,
        input_channel_idx=input_c_idx,
        output_channel_idx=output_c_idx,
    )
    for in_path, input_c_idx, output_c_idx in zip(
        all_data_paths, input_channel_indeces, output_channel_indeces
    )
]
