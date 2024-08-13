import gc
import time

from pathlib import Path
from typing import List

import click
import numpy as np
import torch

from iohub.ngff import open_ome_zarr
from iohub.ngff_meta import TransformationMeta

from mantis.analysis.AnalysisSettings import PsfFromBeadsSettings
from mantis.analysis.analyze_psf import detect_peaks, extract_beads
from mantis.cli.parsing import config_filepath, input_position_dirpaths, output_dirpath
from mantis.cli.utils import yaml_to_model


@click.command()
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
def estimate_psf(
    input_position_dirpaths: List[str],
    config_filepath: str,
    output_dirpath: str,
):
    """
    Estimate the point spread function (PSF) from bead images

    >> mantis estimate-psf -i ./beads.zarr/*/*/* -c ./psf_params.yml -o ./psf.zarr
    """
    # Convert string paths to Path objects
    output_dirpath = Path(output_dirpath)
    config_filepath = Path(config_filepath)

    # Load the first position
    click.echo("Loading data...")
    pzyx_data = []
    for input_position_dirpath in input_position_dirpaths:
        with open_ome_zarr(str(input_position_dirpath), mode="r") as input_dataset:
            T, C, Z, Y, X = input_dataset.data.shape
            pzyx_data.append(input_dataset["0"][0, 0])
            zyx_scale = input_dataset.scale[-3:]

    try:
        pzyx_data = np.array(pzyx_data)
    except Exception:
        raise "Concatenating position arrays failed."

    # Read settings
    settings = yaml_to_model(config_filepath, PsfFromBeadsSettings)
    patch_size = (
        settings.axis0_patch_size,
        settings.axis1_patch_size,
        settings.axis2_patch_size,
    )

    # Some of these settings can be moved to PsfFromBeadsSettings as needed
    bead_detection_settings = {
        "block_size": (64, 64, 32),
        "blur_kernel_size": 3,
        "nms_distance": 32,
        "min_distance": 50,
        "threshold_abs": 200.0,
        "max_num_peaks": 2000,
        "exclude_border": (5, 10, 5),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    pbzyx_data = []
    for zyx_data in pzyx_data:
        # Detect and extract bead patches
        click.echo("Detecting beads...")
        t1 = time.time()
        peaks = detect_peaks(
            zyx_data,
            **bead_detection_settings,
            verbose=True,
        )
        gc.collect()

        torch.cuda.empty_cache()
        t2 = time.time()
        click.echo(f'Time to detect peaks: {t2-t1}')

        beads, _ = extract_beads(
            zyx_data=zyx_data,
            points=peaks,
            scale=zyx_scale,
            patch_size=tuple([a * b for a, b in zip(patch_size, zyx_scale)]),
        )

        # Filter PSFs with non-standard shapes
        filtered_beads = [x for x in beads if x.shape == beads[0].shape]
        bzyx_data = np.stack(filtered_beads)
        pbzyx_data.append(bzyx_data)

    bzyx_data = np.concatenate(pbzyx_data)
    click.echo(f"Total beads: {bzyx_data.shape[0]}")

    normalized_bzyx_data = (
        bzyx_data / np.max(bzyx_data, axis=(-3, -2, -1))[:, None, None, None]
    )
    average_psf = np.mean(normalized_bzyx_data, axis=0)

    # Simple background subtraction and normalization
    average_psf -= np.min(average_psf)
    average_psf /= np.max(average_psf)

    # Save
    with open_ome_zarr(
        output_dirpath, layout="hcs", mode="w", channel_names=["PSF"]
    ) as output_dataset:
        pos = output_dataset.create_position("0", "0", "0")
        array = pos.create_zeros(
            name="0",
            shape=2 * (1,) + average_psf.shape,
            chunks=2 * (1,) + average_psf.shape,
            dtype=np.float32,
            transform=[TransformationMeta(type="scale", scale=2 * (1,) + tuple(zyx_scale))],
        )
        array[0, 0] = average_psf
