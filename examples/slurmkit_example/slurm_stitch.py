import datetime
import glob

from pathlib import Path
import click

import numpy as np

from iohub import open_ome_zarr
from natsort import natsorted
from slurmkit import SlurmParams, slurm_function, submit_function

from mantis.analysis.stitch import (
    get_stitch_output_shape, get_image_shift, get_grid_rows_cols
)

from mantis.cli.stitch import (
    _preprocess_and_shift, _stitch_shifted_store
)

from mantis.cli.utils import (
    create_empty_hcs_zarr,
    process_single_position_v2,
)

from mantis.analysis.AnalysisSettings import StitchSettings
from mantis.cli.utils import yaml_to_model


verbose = True

# io parameters
# dataset = 'A3_600k_38x38_1'

# input_paths = f"/hpc/projects/intracellular_dashboard/ops/2024_03_05_registration_test/live/2-virtual-stain/fcmae-2d/{dataset}.zarr/*/*/*"
# temp_path = Path(f"/hpc/scratch/group.comp.micro/TEMP_{dataset}.zarr")
# output_path = Path(f"/hpc/projects/intracellular_dashboard/ops/2024_03_05_registration_test/live/3-stitch/{dataset}.zarr")
# config_filepath = Path("/hpc/projects/intracellular_dashboard/ops/2024_03_05_registration_test/live/3-stitch/stitch_settings.yml")

dataset = 'kidney_grid_test_1'

input_paths = f"/hpc/projects/intracellular_dashboard/ops/2024_03_05_registration_test/kidney_tissue/0-convert/dragonfly/{dataset}.zarr/*/*/*"
temp_path = Path(f"/hpc/scratch/group.comp.micro/TEMP_{dataset}.zarr")
output_path = Path(f"/hpc/projects/intracellular_dashboard/ops/2024_03_05_registration_test/kidney_tissue/1-stitch/dragonfly/{dataset}_2.zarr")
config_filepath = Path("/hpc/projects/intracellular_dashboard/ops/2024_03_05_registration_test/kidney_tissue/1-stitch/dragonfly/stitch_settings.yml")

# sbatch and resource parameters
cpus_per_task = 1
mem_per_cpu = "16G"
time = 10  # minutes
partition = 'cpu'

# NOTE: parameters from here and below should not have to be changed
input_paths = [Path(path) for path in natsorted(glob.glob(input_paths))]
slurm_out_path = output_path.parent / "slurm_output" / "stitch-%j.out"

settings = yaml_to_model(config_filepath, StitchSettings)

with open_ome_zarr(str(input_paths[0]), mode="r") as input_dataset:
    input_dataset_channels = input_dataset.channel_names
    T, C, Z, Y, X = input_dataset.data.shape
    # scale = tuple(input_dataset.scale)

if settings.channels is None:
    settings.channels = input_dataset_channels

grid_rows, grid_cols = get_grid_rows_cols(Path(*input_paths[0].parts[:-3]))
n_rows = len(grid_rows)
n_cols = len(grid_cols)

output_shape, global_translation = get_stitch_output_shape(
    n_rows, n_cols, Y, X, settings.column_translation, settings.row_translation
)

# Create the output zarr mirroring input positions
# Takes a while, 10 minutes ?!
click.echo('Creating output zarr store')
create_empty_hcs_zarr(
    store_path=temp_path,
    position_keys=[p.parts[-3:] for p in input_paths],
    shape=(T, len(settings.channels), Z) + output_shape,
    chunks=(1, 1, 1, 4096, 4096),
    # scale=scale,
    channel_names=settings.channels,
    dtype=np.float32,
)

# debug
# process_single_position_v2(
#     _preprocess_and_shift,
#     time_indices='all',
#     input_data_path=input_paths[0],
#     output_path=temp_path,
#     input_channel_idx=[input_dataset_channels.index(ch) for ch in settings.channels],
#     output_channel_idx=list(range(len(settings.channels))),
#     num_processes=cpus_per_task,
#     settings=settings,
#     output_shape=output_shape,
#     shift=(0, 0),
#     verbose=True,
# )

# prepare slurm parameters
params = SlurmParams(
    partition=partition,
    cpus_per_task=cpus_per_task,
    mem_per_cpu=mem_per_cpu,
    time=datetime.timedelta(minutes=time),
    output=slurm_out_path,
)

# wrap our deskew_single_position() function with slurmkit
slurm_func = slurm_function(process_single_position_v2)

# generate an array of jobs by passing the in_path and out_path to slurm wrapped function
click.echo('Submitting SLURM jobs')
jobs = []
for in_path in input_paths:
    col_idx, row_idx = (int(in_path.name[:3]), int(in_path.name[3:]))
    shift = get_image_shift(
        col_idx, row_idx, settings.column_translation, settings.row_translation, global_translation
    )

    func = slurm_func(
        _preprocess_and_shift,
        time_indices='all',
        input_channel_idx=[input_dataset_channels.index(ch) for ch in settings.channels],
        output_channel_idx=list(range(len(settings.channels))),
        num_processes=cpus_per_task,
        settings=settings.preprocessing,
        output_shape=output_shape,
        shift=shift,
        verbose=True,
    )

    jobs.append(
        submit_function(
            func,
            slurm_params=params,
            input_data_path=in_path,
            output_path=temp_path,
        )
    )

submit_function(
    slurm_function(_stitch_shifted_store)(temp_path, output_path, settings.postprocessing, verbose),
    slurm_params=SlurmParams(
        partition=partition,
        cpus_per_task=8,
        mem_per_cpu='16G',
        time=datetime.timedelta(hours=12),
        output=slurm_out_path,
    ),
    dependencies=jobs,
)
