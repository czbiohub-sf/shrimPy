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
input_dataset_paths = "/hpc/projects/comp.micro/mantis/2023_09_21_OpenCell_targets/4-virtual_staining/H2B_CAAX_timelapse_2/prediction.zarr/0/23/000001"
config_file = "/home/eduardo.hirata/repos/mantis/mantis/cli/test_edhirata/example_cellpose_segmentation_settings.yml"
output_data_path = "./mask_nuc_mem_all_pos_v2.zarr"

# sbatch and resource parameters
partition = 'gpu'
cpus_per_task = 8
mem_per_cpu = "16G"
time = 60  # minutes
simultaneous_processes_per_node = 5

input_paths = [Path(path) for path in natsorted(glob.glob(input_dataset_paths))]
output_data_path = Path(output_data_path)
click.echo(f"in: {input_paths}, out: {output_data_path}")
slurm_out_path = str(output_data_path.parent / f"slurm_output/register-%j.out")

settings = utils.yaml_to_model(config_file, CellposeSegmentationSettings)
kwargs = {"cellpose_kwargs": settings.dict()}
print(f'Using settings: {kwargs}')
# %%
with open_ome_zarr(input_paths[0]) as dataset:
    T, C, Z, Y, X = dataset.data.shape
    channel_names = dataset.channel_names
chunk_zyx_shape = (1, Y, X)
channel_names = ["label_nuc", "label_mem"]

output_metadata = {
    "shape": (T, len(channel_names), 1, Y, X),
    "chunks": (1,) * 2 + chunk_zyx_shape,
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
    channel_indices=[0, 1],
    num_processes=simultaneous_processes_per_node,
    **kwargs,
)

# generate an array of jobs by passing the in_path and out_path to slurm wrapped function
register_jobs = [
    submit_function(
        segmentation_func,
        slurm_params=params,
        input_data_path=in_path,
        output_path=output_data_path / Path(*in_path.parts[-3:]),
    )
    for in_path in input_paths
]
