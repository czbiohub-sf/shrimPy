import datetime
import os
import glob
from mantis.cli import utils
from slurmkit import SlurmParams, slurm_function, submit_function
from natsort import natsorted
import click
from iohub import open_ome_zarr
from pathlib import Path
from mantis.cli.stabilization import calculate_z_drift, calculate_yx_stabilization
import numpy as np
from mantis.analysis.AnalysisSettings import StabilizationSettings

# NOTE: this pipeline uses the focus found on well one for all. Perhaps this should be done per FOV(?)

# io parameters
input_position_dirpaths = "/hpc/projects/comp.micro/mantis/2023_09_22_A549_0.52NA_illum/4.1-virutal-staining-v2/A549_MitoViewGreen_LysoTracker_W3_FOV5_1_phase_VS.zarr/0/FOV0/0"
output_dirpath = "./test_stabilized.zarr"
config_filepath = "/hpc/projects/comp.micro/mantis/2023_09_22_A549_0.52NA_illum/4.1-virutal-staining-v2/test_stabilization.yml"

# batch and resource parameters
partition = 'preempted'
cpus_per_task = 8
mem_per_cpu = "4G"
time = 300  # minutes
simultaneous_processes_per_node = 4
slurmkit_array_chunk = 20
Z_CHUNK=20


# convert to Path
input_position_dirpaths = [Path(p) for p in natsorted(glob.glob(input_position_dirpaths))]
output_dirpath = Path(output_dirpath)
config_filepath = Path(config_filepath)

settings = utils.yaml_to_model(config_filepath, StabilizationSettings)
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
utils.create_empty_hcs_zarr(
    store_path=output_dirpath,
    position_keys=[p.parts[-3:] for p in input_position_dirpaths],
    **output_metadata,
)

slurm_out_path = str(os.path.join(output_dirpath.parent, "slurm_output/stabilization-%j.out"))

# prepare slurm parameters
params = SlurmParams(
    partition=partition,
    cpus_per_task=cpus_per_task,
    mem_per_cpu=mem_per_cpu,
    time=datetime.timedelta(minutes=time),
    output=slurm_out_path,
)

# wrap our utils.process_single_position() function with slurmkit
slurm_process_single_position = slurm_function(utils.apply_stabilization_over_time_ants)
stabilization_function = slurm_process_single_position(
    list_of_shifts=combined_mats,
    num_processes=simultaneous_processes_per_node,
    time_indices = list(range(T)),
    input_channel_idx = None,
    output_channel_idx = None
)

# Making batches of jobs to avoid IO overload

stabilization_jobs = []
for i in range(0, len(input_position_dirpaths), slurmkit_array_chunk):
    chunk_input_paths = input_position_dirpaths[i : i + slurmkit_array_chunk]

    if i == 0:
        stabilization_jobs = [
            submit_function(
                stabilization_function,
                slurm_params=params,
                input_data_path=in_path,
                output_path=output_dirpath,
            )
            for in_path in chunk_input_paths
        ]

    else:
        stabilization_jobs = [
            submit_function(
                stabilization_function,
                slurm_params=params,
                input_data_path=in_path,
                output_path=output_dirpath,
                dependencies=stabilization_jobs,
            )
            for in_path in chunk_input_paths
        ]