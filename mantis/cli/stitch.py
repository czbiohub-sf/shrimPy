import datetime
import shutil

from pathlib import Path

import click
import numpy as np
import pandas as pd

from iohub import open_ome_zarr
from iohub.ngff_meta import TransformationMeta
from slurmkit import SlurmParams, slurm_function, submit_function

from mantis.analysis.AnalysisSettings import StitchSettings
from mantis.analysis.stitch import (
    _preprocess_and_shift,
    _stitch_shifted_store,
    get_grid_rows_cols,
    get_image_shift,
    get_stitch_output_shape,
)
from mantis.cli.parsing import config_filepath, input_position_dirpaths, output_dirpath
from mantis.cli.utils import create_empty_hcs_zarr, process_single_position_v2, yaml_to_model


@click.command()
@input_position_dirpaths()
@output_dirpath()
@config_filepath()
@click.option(
    "--temp-path", default='/hpc/scratch/group.comp.micro/', help="Path to temporary directory"
)
@click.option("--slurm", "-s", is_flag=True, help="Run stitching on SLURM")
def stitch(
    input_position_dirpaths: list[str],
    output_dirpath: str,
    config_filepath: str,
    temp_path: str,
    slurm: bool,
) -> None:
    """
    Stitch a Zarr store of multi-position data.

    Args:
        input_position_dirpaths (str):
            The path to the input Zarr store.
        output_dirpath (str):
            The path to the output Zarr store. Channels will be appended is the store already exists.
        config_filepath (str):
            The path to the YAML file containing the stitching settings.
        temp_path (str):
            Path to temporary directory, ideally with fast read/write speeds.
        slurm (bool):
            Run stitching on SLURM.
    """
    if not slurm:
        raise NotImplementedError("Only SLURM mode is supported.")

    slurm_out_path = Path(output_dirpath).parent / "slurm_output" / "stitch-%j.out"
    shifted_store_path = Path(temp_path, f"TEMP_{input_position_dirpaths[0].parts[-4]}")
    settings = yaml_to_model(config_filepath, StitchSettings)

    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        input_dataset_channels = input_dataset.channel_names
        T, C, Z, Y, X = input_dataset.data.shape
        scale = tuple(input_dataset.scale)

    if settings.channels is None:
        settings.channels = input_dataset_channels

    assert all(
        channel in input_dataset_channels for channel in settings.channels
    ), "Invalid channel(s) provided."

    wells = list(set([Path(*p.parts[-3:-1]) for p in input_position_dirpaths]))
    grid_rows, grid_cols = get_grid_rows_cols(Path(*input_position_dirpaths[0].parts[:-3]))
    n_rows = len(grid_rows)
    n_cols = len(grid_cols)

    if settings.total_translation is None:
        output_shape, global_translation = get_stitch_output_shape(
            n_rows, n_cols, Y, X, settings.column_translation, settings.row_translation
        )
    else:
        df = pd.DataFrame.from_dict(
            settings.total_translation, orient="index", columns=["shift-y", "shift-x"]
        )
        output_shape = (
            np.ceil(df["shift-y"].max() + Y).astype(int),
            np.ceil(df["shift-x"].max() + X).astype(int),
        )

    # create temp zarr store
    click.echo(f'Creating temporary zarr store at {shifted_store_path}')
    create_empty_hcs_zarr(
        store_path=shifted_store_path,
        position_keys=[p.parts[-3:] for p in input_position_dirpaths],
        shape=(T, len(settings.channels), Z) + output_shape,
        chunks=(1, 1, 1, 4096, 4096),
        channel_names=settings.channels,
        dtype=np.float32,
    )

    # prepare slurm parameters
    params = SlurmParams(
        partition='preempted',
        cpus_per_task=6,
        mem_per_cpu='24G',
        time=datetime.timedelta(minutes=10),
        output=slurm_out_path,
    )

    # wrap our deskew_single_position() function with slurmkit
    slurm_func = slurm_function(process_single_position_v2)(
        _preprocess_and_shift,
        input_channel_idx=[input_dataset_channels.index(ch) for ch in settings.channels],
        output_channel_idx=list(range(len(settings.channels))),
        num_processes=6,
        settings=settings.preprocessing,
        output_shape=output_shape,
        verbose=True,
    )

    click.echo('Submitting SLURM jobs')
    shift_jobs = []
    for in_path in input_position_dirpaths:
        well = Path(*in_path.parts[-3:-1])
        col, row = (in_path.name[:3], in_path.name[3:])

        if settings.total_translation is None:
            shift = get_image_shift(
                int(col),
                int(row),
                settings.column_translation,
                settings.row_translation,
                global_translation,
            )
        else:
            # COL+ROW order here is important
            shift = settings.total_translation[str(well / (col + row))]

        shift_jobs.append(
            submit_function(
                slurm_func,
                slurm_params=params,
                shift_x=shift[-1],
                shift_y=shift[-2],
                input_data_path=in_path,
                output_path=shifted_store_path,
            )
        )

    # create output zarr store
    with open_ome_zarr(
        output_dirpath, layout='hcs', mode="w-", channel_names=settings.channels
    ) as output_dataset:
        for well in wells:
            pos = output_dataset.create_position(*Path(well, '0').parts)
            pos.create_zeros(
                name='0',
                shape=(T, len(settings.channels), Z) + output_shape,
                dtype=np.float32,
                chunks=(1, 1, 1, 4096, 4096),
                transform=[TransformationMeta(type="scale", scale=scale)],
            )

    stitch_job = submit_function(
        slurm_function(_stitch_shifted_store)(
            shifted_store_path, output_dirpath, settings.postprocessing, verbose=True
        ),
        slurm_params=SlurmParams(
            partition='cpu',
            cpus_per_task=32,
            mem_per_cpu='8G',
            time=datetime.timedelta(hours=12),
            output=slurm_out_path,
        ),
        dependencies=shift_jobs,
    )

    click.echo(f'Removing temporary zarr store at {shifted_store_path}')
    submit_function(
        slurm_function(shutil.rmtree)(shifted_store_path),
        slurm_params=SlurmParams(
            partition='cpu',
            cpus_per_task=1,
            mem_per_cpu='12G',
            time=datetime.timedelta(hours=1),
            output=slurm_out_path,
        ),
        dependencies=stitch_job,
    )


if __name__ == '__main__':
    stitch()
