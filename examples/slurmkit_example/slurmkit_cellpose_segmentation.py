# %%
from iohub import open_ome_zarr
import numpy as np
from mantis.cli import utils
import glob
from natsort import natsorted
from pathlib import Path
from slurmkit import SlurmParams, slurm_function, submit_function
from mantis.analysis.AnalysisSettings import CellposeSegmentationSettings
import click
import datetime


# %%
input_dataset_paths = "/hpc/projects/comp.micro/mantis/2023_09_22_A549_0.52NA_illum/4.1-virutal-staining-v2/A549_MitoViewGreen_LysoTracker_W3_FOV5_1_phase_VS.zarr/0/FOV0/0"
config_file = "/hpc/projects/comp.micro/mantis/2023_09_22_A549_0.52NA_illum/4.2-segmentation/segmentation_config.yml"
output_data_path = (
    "./A549_MitoViewGreen_LysoTracker_W3_FOV5_1_phase_VS_segmentation_2.zarr"
)

# sbatch and resource parameters
partition = "gpu"
cpus_per_task = 16
mem_per_cpu = "18G"
time = 300  # minutes
simultaneous_processes_per_node = 12

input_paths = [Path(path) for path in natsorted(glob.glob(input_dataset_paths))]
output_data_path = Path(output_data_path)
click.echo(f"in: {input_paths}, out: {output_data_path}")
slurm_out_path = str(output_data_path.parent / f"slurm_output/segment2-%j.out")

settings = utils.yaml_to_model(config_file, CellposeSegmentationSettings)
kwargs = {"cellpose_kwargs": settings.dict()}
print(f"Using settings: {kwargs}")
# %%
with open_ome_zarr(input_paths[0]) as dataset:
    T, C, Z, Y, X = dataset.data.shape
    channel_names = dataset.channel_names
chunk_zyx_shape = None
channel_names = ["label_nuc", "label_mem"]

output_metadata = {
    "shape": (T, len(channel_names), Z, Y, X),
    "chunks": None,
    "scale": dataset.scale,
    "channel_names": channel_names,
    "dtype": np.float32,
}

utils.create_empty_hcs_zarr(
    store_path=output_data_path,
    position_keys=[p.parts[-3:] for p in input_paths],
    **output_metadata,
)

# prepare slurm parameters
params = SlurmParams(
    partition=partition,
    gpus=1,
    cpus_per_task=cpus_per_task,
    mem_per_cpu=mem_per_cpu,
    time=datetime.timedelta(minutes=time),
    output=slurm_out_path,
)

# wrap our utils.process_single_position() function with slurmkit
slurm_process_single_position = slurm_function(utils.process_single_position_v2)
segmentation_func = slurm_process_single_position(
    func=utils.nuc_mem_segmentation,
    time_indices=list(range(T)),
    input_channel_idx=[0, 1],  # chanesl in the input dataset
    output_channel_idx=[0, 1],  # channels in the output dataset
    num_processes=simultaneous_processes_per_node,
    **kwargs,
)

# Making batches of jobs to avoid IO overload
slurmkit_array_chunk = 20
segment_jobs = []
for i in range(0, len(input_paths), slurmkit_array_chunk):
    chunk_input_paths = input_paths[i : i + slurmkit_array_chunk]

    if i == 0:
        segment_jobs = [
            submit_function(
                segmentation_func,
                slurm_params=params,
                input_data_path=in_path,
                output_path=output_data_path,
            )
            for in_path in chunk_input_paths
        ]

    else:
        segment_jobs = [
            submit_function(
                segmentation_func,
                slurm_params=params,
                input_data_path=in_path,
                output_path=output_data_path,
                dependencies=segment_jobs,
            )
            for in_path in chunk_input_paths
        ]
