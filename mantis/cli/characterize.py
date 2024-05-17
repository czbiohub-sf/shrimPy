import click
import time
import gc
import torch
import warnings

from iohub.ngff import open_ome_zarr
from typing import List
from mantis.analysis.AnalysisSettings import CharacterizeSettings
from mantis.cli.parsing import input_position_dirpaths, output_dirpath, config_filepath
from mantis.cli.utils import yaml_to_model
from mantis.analysis.analyze_psf import detect_peaks, extract_beads, analyze_psf, generate_report


@click.command()
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
def characterize(
    input_position_dirpaths: List[str],
    config_filepath: str,
    output_dirpath: str,
):
    """
    Characterize the point spread function (PSF) from bead images in an html report

    >> mantis characterize -i ./beads.zarr/*/*/* -c ./characterize_params.yml -o ./
    """
    click.echo(f"Loading data...")
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        T, C, Z, Y, X = input_dataset.data.shape
        zyx_data = input_dataset["0"][0, 0]
        zyx_scale = input_dataset.scale[-3:]

    # Read settings
    settings = yaml_to_model(config_filepath, CharacterizeSettings)
    
    click.echo(f"Detecting peaks...")
    t1 = time.time()
    peaks = detect_peaks(
        zyx_data,
        **settings.dict(),
        verbose=True,
    )
    gc.collect()
    torch.cuda.empty_cache()
    t2 = time.time()
    click.echo(f'Time to detect peaks: {t2-t1}')

    t1 = time.time()
    beads, offsets = extract_beads(
        zyx_data=zyx_data,
        points=peaks,
        scale=zyx_scale,
    )

    click.echo(f"Analyzing PSFs...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_gaussian_fit, df_1d_peak_width = analyze_psf(
            zyx_patches=beads,
            bead_offsets=offsets,
            scale=zyx_scale,
        )
    t2 = time.time()
    click.echo(f'Time to analyze PSFs: {t2-t1}')

    # Generate HTML report
    generate_report(
        output_dirpath,
        input_position_dirpaths[0],
        "",
        beads,
        peaks,
        df_gaussian_fit,
        df_1d_peak_width,
        zyx_scale,
        ["AXIS 0", "AXIS 1", "AXIS 2"],
    )