import datetime
import time

from pathlib import Path
from typing import Literal

import click
import numpy as np
import pandas as pd

from iohub import open_ome_zarr
from slurmkit import SlurmParams, slurm_function, submit_function

from mantis.analysis.AnalysisSettings import ProcessingSettings, StitchSettings
from mantis.analysis.stitch import estimate_shift, get_grid_rows_cols
from mantis.cli.parsing import input_zarr_path, output_filepath
from mantis.cli.utils import model_to_yaml


def estimate_zarr_fov_shifts(
    fov0_zarr_path: str,
    fov1_zarr_path: str,
    percent_overlap: float,
    fliplr: bool,
    flipud: bool,
    direction: Literal["row", "col"],
    output_dirname: str = None,
):
    fov0_zarr_path = Path(fov0_zarr_path)
    fov1_zarr_path = Path(fov1_zarr_path)
    well_name = Path(*fov0_zarr_path.parts[-3:-1])
    fov0 = fov0_zarr_path.name
    fov1 = fov1_zarr_path.name

    # TODO: hardcoded image coords
    im0 = open_ome_zarr(fov0_zarr_path).data[0, 0, 0]
    im1 = open_ome_zarr(fov1_zarr_path).data[0, 0, 0]

    if fliplr:
        im0 = np.fliplr(im0)
        im1 = np.fliplr(im1)
    if flipud:
        im0 = np.flipud(im0)
        im1 = np.flipud(im1)

    shift = estimate_shift(im0, im1, percent_overlap, direction)

    df = pd.DataFrame(
        {
            "well": str(well_name),
            "fov0": fov0,
            "fov1": fov1,
            "shift-x": shift[0],
            "shift-y": shift[1],
            "direction": direction,
        },
        index=[0],
    )
    if output_dirname:
        df.to_csv(
            Path(output_dirname, f"{'_'.join(well_name.parts + (fov0, fov1))}_shift.csv"),
            index=False,
        )
    else:
        return df


def consolidate_zarr_fov_shifts(
    input_dirname: str,
    output_filepath: str,
):
    # read all csv files in input_dirname and combine into a single dataframe
    csv_files = Path(input_dirname).rglob("*_shift.csv")
    df = pd.concat(
        [pd.read_csv(csv_file, dtype={'fov0': str, 'fov1': str}) for csv_file in csv_files],
        ignore_index=True,
    )
    df.to_csv(output_filepath, index=False)


def cleanup_shifts(csv_filepath: str):
    df = pd.read_csv(csv_filepath, dtype={'fov0': str, 'fov1': str})
    df['shift-x-raw'] = df['shift-x']
    df['shift-y-raw'] = df['shift-y']

    # replace row shifts with median value calculated across all columns
    _df = df[df['direction'] == 'row']
    # group by well and last three characters of fov0
    groupby = _df.groupby(['well', _df['fov0'].str[-3:]])
    df.loc[df['direction'] == 'row', 'shift-x'] = groupby['shift-x-raw'].transform('median')
    df.loc[df['direction'] == 'row', 'shift-y'] = groupby['shift-y-raw'].transform('median')

    df.to_csv(csv_filepath, index=False)


def compute_total_translation(csv_filepath: str) -> pd.DataFrame:
    df = pd.read_csv(csv_filepath, dtype={'fov0': str, 'fov1': str})

    df['row'] = df['fov1'].str[-3:].astype(int)
    df['col'] = df['fov1'].str[:3].astype(int)
    df.set_index('fov1', inplace=True)
    df.sort_index(inplace=True)

    total_shift = []
    for well in df['well'].unique():
        _df = df[(df['direction'] == 'col') & (df['well'] == well)]
        col_shifts = _df.groupby('row')[['shift-x', 'shift-y']].cumsum()
        _df = df[(df['direction'] == 'row') & (df['well'] == well)]
        row_shifts = _df.groupby('col')[['shift-x', 'shift-y']].cumsum()
        _total_shift = col_shifts.add(row_shifts, fill_value=0)

        # add row 000000
        _total_shift = pd.concat(
            [pd.DataFrame({'shift-x': 0, 'shift-y': 0}, index=['000000']), _total_shift]
        )

        # add global offset to remove negative values
        _total_shift['shift-x'] += -np.minimum(_total_shift['shift-x'].min(), 0)
        _total_shift['shift-y'] += -np.minimum(_total_shift['shift-y'].min(), 0)
        _total_shift.set_index(well + '/' + _total_shift.index, inplace=True)
        total_shift.append(_total_shift)

    return pd.concat(total_shift)


def write_config_file(shifts: pd.DataFrame, output_filepath: str, fliplr: bool, flipud: bool):
    total_translation_dict = shifts.apply(
        lambda row: [float(row['shift-y'].round(2)), float(row['shift-x'].round(2))], axis=1
    ).to_dict()

    settings = StitchSettings(
        total_translation=total_translation_dict,
        preprocessing=ProcessingSettings(fliplr=fliplr, flipud=flipud),
    )
    model_to_yaml(settings, output_filepath)


@click.command()
@input_zarr_path()
@output_filepath()
@click.option(
    "--percent-overlap", "-p", required=True, type=float, help="Percent overlap between images"
)
@click.option("--fliplr", is_flag=True, help="Flip images left-right before stitching")
@click.option("--flipud", is_flag=True, help="Flip images up-down before stitching")
@click.option("--slurm", "-s", is_flag=True, help="Run stitching on SLURM")
def estimate_stitch(
    input_zarr_path: str,
    output_filepath: str,
    percent_overlap: float,
    fliplr: bool,
    flipud: bool,
    slurm: bool,
):
    assert 0 <= percent_overlap <= 1, "Percent overlap must be between 0 and 1"

    input_zarr_path = Path(input_zarr_path)
    output_filepath = Path(output_filepath)
    csv_filepath = (
        output_filepath.parent
        / f"stitch_shifts_{input_zarr_path.name.replace('.zarr', '.csv')}"
    )

    dataset = open_ome_zarr(input_zarr_path)

    # here we assume that all wells have the same fov grid
    grid_rows, grid_cols = get_grid_rows_cols(input_zarr_path)
    row_fov0 = [col + row for row in grid_rows[:-1] for col in grid_cols]
    row_fov1 = [col + row for row in grid_rows[1:] for col in grid_cols]
    col_fov0 = [col + row for col in grid_cols[:-1] for row in grid_rows]
    col_fov1 = [col + row for col in grid_cols[1:] for row in grid_rows]
    estimate_shift_params = {
        "percent_overlap": percent_overlap,
        "fliplr": fliplr,
        "flipud": flipud,
    }

    # define slurm paramters
    if slurm:
        slurm_out_path = output_filepath.parent / "slurm_output" / "shift-%j.out"
        csv_dirpath = (
            output_filepath.parent / 'raw_shifts' / input_zarr_path.name.replace('.zarr', '')
        )
        csv_dirpath.mkdir(parents=True, exist_ok=False)
        params = SlurmParams(
            partition="cpu",
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

    shifts, jobs = [], []
    for well_name, _ in dataset.wells():
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

    cleanup_shifts(csv_filepath)
    shifts = compute_total_translation(csv_filepath)
    write_config_file(shifts, output_filepath, fliplr, flipud)


if __name__ == "__main__":
    estimate_stitch()
