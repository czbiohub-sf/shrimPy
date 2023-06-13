import click
from iohub import open_ome_zarr
from pathlib import Path
import datetime
import numpy as np
import os
import multiprocessing as mp


from mantis.cli.parsing import (
    input_data_path_argument,
    deskew_param_argument,
    output_dataset_options,
)
from mantis.cli.deskew import deskew_cli

from mantis.analysis.deskew import get_deskewed_data_shape
import yaml
from mantis.analysis.AnalysisSettings import DeskewSettings
from dataclasses import asdict

from slurmkit import SlurmParams, slurm_function, submit_function
from iohub.ngff_meta import TransformationMeta
from natsort import natsorted


def _get_deskew_params(deskew_params_path: Path)-> None:
    # Load params
    with open(deskew_params_path) as file:
        raw_settings = yaml.safe_load(file)
    settings = DeskewSettings(**raw_settings)
    click.echo(f"Deskewing parameters: {asdict(settings)}")
    return settings


def create_empty_zarr(
    input_data_path: Path,
    deskew_param_path: Path,
    output_path: Path,
    keep_overhang: bool,
) -> None:
    # Load the "0" position to infer dataset information
    input_data_path = natsorted(input_data_path)
    click.echo(f"Input data folders {input_data_path})")

    input_dataset = open_ome_zarr(str(input_data_path[0]), mode="r")
    T, C, Z, Y, X = input_dataset.data.shape

    # Get the deskewing parameters
    settings = _get_deskew_params(deskew_param_path)
    deskewed_shape, voxel_size = get_deskewed_data_shape(
        (Z, Y, X),
        settings.pixel_size_um,
        settings.ls_angle_deg,
        settings.px_to_scan_ratio,
        keep_overhang,
    )

    click.echo("Creating empty array...")

    # Handle transforms and metadata
    transform = TransformationMeta(
        type="scale",
        scale=2 * (1,) + voxel_size,
    )

    # Prepare output dataset
    channel_names = input_dataset.channel_names

    # Output shape based on the type of reconstruction
    output_shape = (T, len(channel_names)) + deskewed_shape
    click.echo(f"Number of positions: {len(input_data_path)}")
    click.echo(f"Output shape: {output_shape}")
    chunk_size = (1, 1, 64) + deskewed_shape[-2:]
    click.echo(f"Chunk size {chunk_size}")

    # Create output dataset
    output_dataset = open_ome_zarr(
        output_path, layout="hcs", mode="w", channel_names=channel_names
    )
    # This takes care of the logic for single position or multiple position by wildcards
    for filepath in input_data_path:
        path_strings = filepath.split(os.path.sep)[-3:]
        pos = output_dataset.create_position(
            str(path_strings[0]), str(path_strings[1]), str(path_strings[2])
        )
        _ = pos.create_zeros(
            name="0",
            shape=(
                T,
                C,
            )
            + deskewed_shape,
            chunks=chunk_size,
            dtype=np.uint16,
            transform=[transform],
        )


def _get_output_paths(list_pos, output_path):
    # From the position filepath generate the output filepath
    list_output_path = []
    for filepath in list_pos:
        path_strings = filepath.split(os.path.sep)[-3:]
        list_output_path.append(os.path.join(output_path, *path_strings))
    return list_output_path


@slurm_function
def slurm_deskew(
    input_data_path: Path,
    deskew_params_path: Path,
    output_path: Path,
    view: bool,
    keep_overhang: bool,
    num_cores: int,
    slurm: bool,
) -> None:
    click.echo(f"slurm_deskew {input_data_path}, {output_path}")

    # Sort the input as nargs=-1 will not be natsorted
    # input_data_path = natsorted(input_data_path)
    # Handle single position or wildcard filepath
    # click.echo(f"List of input pos:{input_data_path} output_pos:{list_output_pos}")

    if not slurm:
        # Create a zarr store output to mirror the input
        create_empty_zarr(input_data_path, deskew_params_path, output_path, keep_overhang)

    deskew_cli(
        input_data_path=input_data_path,
        deskew_params_path=deskew_params_path,
        output_path=output_path,
        view=view,
        keep_overhang=keep_overhang,
        num_cores=num_cores,
    )


@click.command()
@input_data_path_argument()
@deskew_param_argument()
@output_dataset_options(default="./deskewed.zarr")
@click.option(
    "--view",
    "-v",
    default=False,
    required=False,
    is_flag=True,
    help="View the correctly scaled result in napari",
)
@click.option(
    "--keep-overhang",
    "-ko",
    default=False,
    is_flag=True,
    help="Keep the overhanging region.",
)
@click.option(
    "--num-cores",
    "-j",
    default=mp.cpu_count(),
    help="Number of cores",
    required=False,
    type=int,
)
@click.option(
    "--slurm",
    "-s",
    default=False,
    is_flag=True,
    help="Using slurm",
)
def main(
    input_data_path,
    deskew_param_path,
    output_path,
    view,
    keep_overhang,
    num_cores,
    slurm,
):
    input_data_path = natsorted(input_data_path)
    list_output_pos = _get_output_paths(input_data_path, output_path)
    click.echo(f"in: {input_data_path}, out: {list_output_pos}")

    # Slurm output path
    #TODO: we will probably end up using the logger instead of printf and click.echo.
    output_dir = os.path.dirname(os.path.join(os.getcwd(), output_path))
    slurm_out_path = str(os.path.join(output_dir, "slurm_output/deskew-%j.out"))

    # Initialize the array
    if not slurm:
        create_empty_zarr(input_data_path, deskew_param_path, output_path, keep_overhang)

    # Slurm Parameters for the deskew
    params = SlurmParams(
        partition="cpu",
        cpus_per_task=16,
        mem="64G",
        time=datetime.timedelta(minutes=40),
        output=slurm_out_path,
    )

    deskew_func = slurm_deskew(
        deskew_params_path=deskew_param_path,
        view=view,
        keep_overhang=keep_overhang,
        num_cores=num_cores,
        slurm=slurm,
    )

    # Submits the job with the input_data_path and output_path variables
    # Adding the no_sbatch flag to match our slurm flag
    deskew_jobs = [
        submit_function(
            deskew_func,
            slurm_params=params,
            input_data_path=in_path,
            output_path=out_path,
            no_sbatch=not slurm,
        )
        for in_path, out_path in zip(input_data_path, list_output_pos)
    ]


if __name__ == "__main__":
    main()
