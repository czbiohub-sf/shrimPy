import gc
import click
import numpy as np
import torch
import time

from iohub.ngff import open_ome_zarr
from iohub.ngff_meta import TransformationMeta
from pathlib import Path
from typing import List
from mantis.analysis.AnalysisSettings import PsfFromBeadsSettings
from mantis.analysis.analyze_psf import detect_peaks, extract_beads
from mantis.cli.parsing import input_position_dirpaths, output_dirpath, config_filepath
from mantis.cli.utils import yaml_to_model


@click.command()
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
def psf_from_beads(
    input_position_dirpaths: List[str],
    config_filepath: str,
    output_dirpath: str,
):
    """
    Estimate the point spread function (PSF) from bead images

    >> mantis psf_from_beads -i ./beads.zarr/*/*/* -c ./psf_params.yml -o ./psf.zarr
    """
    # Convert string paths to Path objects
    output_dirpath = Path(output_dirpath)
    config_filepath = Path(config_filepath)

    # Load the first position (TODO: consider averaging over positions)
    click.echo(f"Loading data...")
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        T, C, Z, Y, X = input_dataset.data.shape
        zyx_data = input_dataset["0"][0, 0]
        zyx_scale = input_dataset.scale[-3:]

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

    # Detect and extract bead patches
    click.echo(f"Detecting beads...")
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
        patch_size_voxels=patch_size,
    )

    # Filter PSFs with non-standard shapes
    filtered_beads = [x for x in beads if x.shape == beads[0].shape]
    bzyx_data = np.stack(filtered_beads)
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
