import datetime
import os
import glob
from slurmkit import SlurmParams, slurm_function, submit_function
from natsort import natsorted
from iohub import open_ome_zarr
from pathlib import Path
from mantis.cli.utils import (
    yaml_to_model,
    create_empty_hcs_zarr,
    process_single_position_v2,
)
from mantis.cli.stabilize import apply_stabilization_transform
import numpy as np
from mantis.analysis.AnalysisSettings import StabilizationSettings

# NOTE: this pipeline uses the focus found on well one for all. Perhaps this should be done per FOV(?)

# io parameters
input_position_dirpaths = "/hpc/projects/comp.micro/mantis/2023_11_08_Opencell_infection_redo/1-recon/OC43_infection_timelapse_3.zarr/*/*/*"
output_dirpath = "/hpc/projects/comp.micro/mantis/2023_11_08_Opencell_infection_redo/3-stabilization/OC43_infection_timelapse_3.zarr"
config_filepath = "/hpc/projects/comp.micro/mantis/2023_11_08_Opencell_infection_redo/3-stabilization/stabilization.yml"

# batch and resource parameters
partition = 'preempted'
cpus_per_task = 5
mem_per_cpu = "4G"
time = 60  # minutes
simultaneous_processes_per_node = 5

# convert to Path
input_position_dirpaths = [Path(p) for p in natsorted(glob.glob(input_position_dirpaths))]
output_dirpath = Path(output_dirpath)
config_filepath = Path(config_filepath)

settings = yaml_to_model(config_filepath, StabilizationSettings)
combined_mats = settings.affine_transform_zyx_list
combined_mats = np.array(combined_mats)

with open_ome_zarr(input_position_dirpaths[0]) as dataset:
    T, C, Z, Y, X = dataset.data.shape
    channel_names = dataset.channel_names

output_metadata = {
    "shape": (T, C, Z, Y, X),
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
slurm_process_single_position = slurm_function(process_single_position_v2)
stabilization_function = slurm_process_single_position(
    func=apply_stabilization_transform,
    list_of_shifts=combined_mats,
    time_indices=list(range(T)),
    input_channel_idx=None,
    output_channel_idx=None,
    num_processes=simultaneous_processes_per_node,
)

stabilization_jobs = [
    submit_function(
        stabilization_function,
        slurm_params=params,
        input_data_path=in_path,
        output_path=output_dirpath,
    )
    for in_path in input_position_dirpaths
]
