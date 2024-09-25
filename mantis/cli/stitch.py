import datetime
import shutil
import warnings

from pathlib import Path

import click
import numpy as np
import pandas as pd

from iohub import open_ome_zarr
from iohub.ngff import TransformationMeta
from slurmkit import HAS_SLURM, SlurmParams, slurm_function, submit_function

from mantis.analysis.AnalysisSettings import StitchSettings
from mantis.analysis.stitch import (
    get_grid_rows_cols,
    get_image_shift,
    get_stitch_output_shape,
    preprocess_and_shift,
    stitch_shifted_store,
)
from mantis.cli.parsing import config_filepath, input_position_dirpaths, output_dirpath
from mantis.cli.utils import create_empty_hcs_zarr, process_single_position_v2, yaml_to_model


@click.command()
@input_position_dirpaths()
@output_dirpath()
@config_filepath()
@click.option(
    "--temp-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default='./',
    help="Path to temporary directory, ideally with fast read/write speeds, e.g. /hpc/scratch/group.comp.micro/",
)
def stitch(
    input_position_dirpaths: list[Path],
    output_dirpath: str,
    config_filepath: str,
    temp_path: str,
) -> None:
    """
    Stitch positions in wells of a zarr store using a configuration file generated by estimate-stitch.

    >>> mantis stitch -i ./input.zarr/*/*/* -c ./stitch_params.yml -o ./output.zarr --temp-path /hpc/scratch/group.comp.micro/
    """
    # Emitting a deprecation warning
    warnings.warn(
        "This function is being moved to the biahub library, available at https://github.com/czbiohub-sf/biahub",
        DeprecationWarning,
        stacklevel=2,
    )

    if not HAS_SLURM:
        warnings.warn(
            "This function is intended to be used with SLURM. "
            "Running on local machine instead."
        )

    slurm_out_path = Path(output_dirpath).parent / "slurm_output" / "stitch-%j.out"
    shifted_store_path = Path(temp_path, f"TEMP_{input_position_dirpaths[0].parts[-4]}")
    settings = yaml_to_model(config_filepath, StitchSettings)

    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        input_dataset_channels = input_dataset.channel_names
        T, C, Z, Y, X = input_dataset.data.shape
        scale = tuple(input_dataset.scale)
        chunks = input_dataset.data.chunks

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
    stitched_shape = (T, len(settings.channels), Z) + output_shape
    stitched_chunks = chunks[:3] + (4096, 4096)
    create_empty_hcs_zarr(
        store_path=shifted_store_path,
        position_keys=[p.parts[-3:] for p in input_position_dirpaths],
        shape=stitched_shape,
        chunks=stitched_chunks,
        channel_names=settings.channels,
        dtype=np.float32,
    )

    # prepare slurm parameters
    params = SlurmParams(
        partition='preempted',
        cpus_per_task=6,
        mem_per_cpu='24G',
        time=datetime.timedelta(minutes=30),
        output=slurm_out_path,
    )

    # Shift each FOV to its final position in the stitched image
    slurm_func = slurm_function(process_single_position_v2)(
        preprocess_and_shift,
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
                shape=stitched_shape,
                dtype=np.float32,
                chunks=stitched_chunks,
                transform=[TransformationMeta(type="scale", scale=scale)],
            )

    # Stitch pre-shifted images
    stitch_job = submit_function(
        slurm_function(stitch_shifted_store)(
            shifted_store_path,
            output_dirpath,
            settings.postprocessing,
            blending='average',
            verbose=True,
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

    # Delete temporary store
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
