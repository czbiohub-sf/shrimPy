import datetime
import time

from pathlib import Path

import click
import pandas as pd

from iohub import open_ome_zarr
from slurmkit import SlurmParams, slurm_function, submit_function

from mantis.analysis.AnalysisSettings import ProcessingSettings, StitchSettings
from mantis.analysis.stitch import (
    cleanup_shifts,
    compute_total_translation,
    consolidate_zarr_fov_shifts,
    estimate_zarr_fov_shifts,
    get_grid_rows_cols,
)
from mantis.cli.parsing import input_position_dirpaths, output_filepath
from mantis.cli.utils import model_to_yaml


def write_config_file(
    shifts: pd.DataFrame, output_filepath: str, channel: str, fliplr: bool, flipud: bool
):
    total_translation_dict = shifts.apply(
        lambda row: [float(row['shift-y'].round(2)), float(row['shift-x'].round(2))], axis=1
    ).to_dict()

    settings = StitchSettings(
        channels=[channel],
        preprocessing=ProcessingSettings(fliplr=fliplr, flipud=flipud),
        postprocessing=ProcessingSettings(),
        total_translation=total_translation_dict,
    )
    model_to_yaml(settings, output_filepath)


@click.command()
@input_position_dirpaths()
@output_filepath()
@click.option(
    "--channel",
    required=True,
    type=str,
    help="Channel to use for estimating stitch parameters",
)
@click.option(
    "--percent-overlap", "-p", required=True, type=float, help="Percent overlap between images"
)
@click.option("--fliplr", is_flag=True, help="Flip images left-right before stitching")
@click.option("--flipud", is_flag=True, help="Flip images up-down before stitching")
@click.option("--slurm", "-s", is_flag=True, help="Run stitching on SLURM")
def estimate_stitch(
    input_position_dirpaths: list[Path],
    output_filepath: str,
    channel: str,
    percent_overlap: float,
    fliplr: bool,
    flipud: bool,
    slurm: bool,
):
    """
    Estimate stitching parameters for positions in wells of a zarr store.
    Position names must follow the naming format XXXYYY, e.g. 000000, 000001, 001000, etc.
    as created by the Micro-manager Tile Creator: https://micro-manager.org/Micro-Manager_User's_Guide#positioning
    Assumes all wells have the save FOV grid layout.

    >>> mantis estimate-stitch -i ./input.zarr/*/*/* -o ./stitch_params.yml --channel DAPI --percent-overlap 0.05 --slurm
    """
    assert 0 <= percent_overlap <= 1, "Percent overlap must be between 0 and 1"

    input_zarr_path = Path(*input_position_dirpaths[0].parts[:-3])
    output_filepath = Path(output_filepath)
    csv_filepath = (
        output_filepath.parent
        / f"stitch_shifts_{input_zarr_path.name.replace('.zarr', '.csv')}"
    )

    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        assert (
            channel in dataset.channel_names
        ), f"Channel {channel} not found in input zarr store"
        tcz_idx = (0, dataset.channel_names.index(channel), dataset.data.shape[-3] // 2)
        pixel_size_um = dataset.scale[-1]
    if pixel_size_um == 1.0:
        response = input(
            'The pixel size is equal to the default value of 1.0 um. ',
            'Inaccurate pixel size will affect stitching outlier removal. ',
            'Continue? [y/N]: ',
        )
        if response.lower() != 'y':
            return

    # here we assume that all wells have the same fov grid
    click.echo('Indexing input zarr store')
    wells = list(set([Path(*p.parts[-3:-1]) for p in input_position_dirpaths]))
    grid_rows, grid_cols = get_grid_rows_cols(input_zarr_path)
    row_fov0 = [col + row for row in grid_rows[:-1] for col in grid_cols]
    row_fov1 = [col + row for row in grid_rows[1:] for col in grid_cols]
    col_fov0 = [col + row for col in grid_cols[:-1] for row in grid_rows]
    col_fov1 = [col + row for col in grid_cols[1:] for row in grid_rows]
    estimate_shift_params = {
        "tcz_index": tcz_idx,
        "percent_overlap": percent_overlap,
        "fliplr": fliplr,
        "flipud": flipud,
    }

    # define slurm parameters
    if slurm:
        slurm_out_path = output_filepath.parent / "slurm_output" / "shift-%j.out"
        csv_dirpath = (
            output_filepath.parent / 'raw_shifts' / input_zarr_path.name.replace('.zarr', '')
        )
        csv_dirpath.mkdir(parents=True, exist_ok=False)
        params = SlurmParams(
            partition="preempted",
            cpus_per_task=1,
            mem_per_cpu='8G',
            time=datetime.timedelta(minutes=10),
            output=slurm_out_path,
        )
        slurm_func = {
            direction: slurm_function(estimate_zarr_fov_shifts)(
                direction=direction,
                output_dirname=csv_dirpath,
                **estimate_shift_params,
            )
            for direction in ("row", "col")
        }

    click.echo('Estimating FOV shifts...')
    shifts, jobs = [], []
    for well_name in wells:
        for direction, fovs in zip(
            ("row", "col"), (zip(row_fov0, row_fov1), zip(col_fov0, col_fov1))
        ):
            for fov0, fov1 in fovs:
                fov0_zarr_path = Path(input_zarr_path, well_name, fov0)
                fov1_zarr_path = Path(input_zarr_path, well_name, fov1)
                if slurm:
                    job_id = submit_function(
                        slurm_func[direction],
                        slurm_params=params,
                        fov0_zarr_path=fov0_zarr_path,
                        fov1_zarr_path=fov1_zarr_path,
                    )
                    jobs.append(job_id)
                else:
                    shift_params = estimate_zarr_fov_shifts(
                        fov0_zarr_path=fov0_zarr_path,
                        fov1_zarr_path=fov1_zarr_path,
                        direction=direction,
                        **estimate_shift_params,
                    )
                    shifts.append(shift_params)

    click.echo('Consolidating FOV shifts...')
    if slurm:
        submit_function(
            slurm_function(consolidate_zarr_fov_shifts)(
                input_dirname=csv_dirpath,
                output_filepath=csv_filepath,
            ),
            slurm_params=params,
            dependencies=jobs,
        )

        # wait for csv_filepath to be created, capped at 5 min
        t_start = time.time()
        while not csv_filepath.exists() and time.time() - t_start < 300:
            time.sleep(1)
    else:
        df = pd.concat(shifts, ignore_index=True)
        df.to_csv(csv_filepath, index=False)

    cleanup_shifts(csv_filepath, pixel_size_um)
    shifts = compute_total_translation(csv_filepath)
    write_config_file(shifts, output_filepath, channel, fliplr, flipud)


if __name__ == "__main__":
    estimate_stitch()
