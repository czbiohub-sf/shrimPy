from pathlib import Path
from typing import List

import click
import numpy as np
import torch

from iohub import open_ome_zarr
from waveorder.models.isotropic_fluorescent_thick_3d import apply_inverse_transfer_function

from mantis.analysis.AnalysisSettings import DeconvolveSettings
from mantis.cli.parsing import (
    _str_to_path,
    config_filepath,
    input_position_dirpaths,
    output_dirpath,
)
from mantis.cli.utils import create_empty_hcs_zarr, yaml_to_model


def apply_deconvolve_single_position(
    input_position_dirpath: str, psf_dirpath: str, config_filepath: str, output_dirpath: Path
):
    """
    Apply deconvolution to a single position
    """
    # Read settings
    settings = yaml_to_model(Path(config_filepath), DeconvolveSettings)

    # Load the data
    input_dataset = open_ome_zarr(input_position_dirpath, mode="r")
    output_dataset = open_ome_zarr(output_dirpath, mode="a")
    T, C, Z, Y, X = input_dataset.data.shape

    # Load the PSF
    with open_ome_zarr(psf_dirpath, mode="r") as psf_dataset:
        position = psf_dataset["0/0/0"]
        psf_data = position["0"][0, 0]
        psf_scale = position.scale[-3:]

    click.echo("Padding PSF...")
    zyx_padding = np.array((Z, Y, X)) - np.array(psf_data.shape)
    pad_width = [(x // 2, x // 2) if x % 2 == 0 else (x // 2, x // 2 + 1) for x in zyx_padding]
    padded_average_psf = np.pad(
        psf_data, pad_width=pad_width, mode="constant", constant_values=0
    )

    click.echo("Calculating transfer function...")
    transfer_function = torch.abs(torch.fft.fftn(torch.tensor(padded_average_psf)))
    transfer_function /= torch.max(transfer_function)

    zyx_scale = input_dataset.scale[-3:]

    # Check if scales match
    if psf_scale != zyx_scale:
        click.echo(
            f"Warning: PSF scale {psf_scale} does not match data scale {zyx_scale}. "
            "Consider resampling the PSF."
        )

    for t in range(1):  # T):
        for c in range(C):
            zyx_data = input_dataset["0"][t, c]

            # Apply deconvolution
            click.echo(f"Deconvolving channel {c}/{C-1}, time {t}/{T-1}")
            zyx_data_deconvolved = apply_inverse_transfer_function(
                torch.tensor(zyx_data),
                torch.tensor(transfer_function),
                0,
                regularization_strength=settings.regularization_strength,
            )
            click.echo("Saving to output...")
            output_dataset["0"][t, c] = zyx_data_deconvolved.numpy()

    input_dataset.close()
    output_dataset.close()


@click.command()
@input_position_dirpaths()
@click.option(
    "--psf-dirpath",
    "-p",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    callback=_str_to_path,
    help="Path to psf.zarr",
)
@config_filepath()
@output_dirpath()
def deconvolve(
    input_position_dirpaths: List[str],
    psf_dirpath: str,
    config_filepath: str,
    output_dirpath: str,
):
    """
    Deconvolve across T and C axes using a PSF and a configuration file

    >> mantis deconvolve -i ./input.zarr/*/*/* -p ./psf.zarr -c ./deconvolve_params.yml -o ./output.zarr
    """
    # Convert string paths to Path objects
    output_dirpath = Path(output_dirpath)
    config_filepath = Path(config_filepath)

    # Create output zarr store
    click.echo("Creating empty output zarr...")
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        create_empty_hcs_zarr(
            store_path=output_dirpath,
            position_keys=[p.parts[-3:] for p in input_position_dirpaths],
            channel_names=input_dataset.channel_names,
            shape=input_dataset.data.shape,
            scale=input_dataset.scale,
        )

    # Loop through positions
    for input_position_dirpath in input_position_dirpaths:
        apply_deconvolve_single_position(
            input_position_dirpath,
            psf_dirpath,
            config_filepath,
            output_dirpath / Path(*input_position_dirpath.parts[-3:]),
        )
