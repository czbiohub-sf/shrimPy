import csv

from dataclasses import asdict

import click
import napari
import numpy as np
import yaml
import os
from multiprocessing import Pool
import itertools
from functools import partial
import dask.array as da

from iohub import open_ome_zarr
from iohub.ngff_meta import TransformationMeta
from ndtiff import Dataset

from mantis.analysis.AnalysisSettings import DeskewSettings
from mantis.analysis.deskew import deskew_data, get_deskewed_data_shape

N_processes = 6


@click.command()
@click.argument(
    "data_path",
    type=click.Path(exists=True, file_okay=False),
)
@click.argument(
    "output_path",
    type=click.Path(exists=False, file_okay=False),
)
@click.argument(
    "deskew_params_path",
    type=click.Path(exists=True, file_okay=True),
)
@click.option(
    "--positions",
    type=click.Path(exists=True, file_okay=True),
    required=False,
    help="Path to positions CSV log file",
)
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
def deskew(data_path, output_path, deskew_params_path, positions, view, keep_overhang):
    """
    Deskews across P, T, C axes using a parameter file generated with estimate_deskew.py
    """

    assert str(output_path).endswith('.zarr'), "Output path must be a zarr store"

    # Load params
    with open(deskew_params_path) as file:
        raw_settings = yaml.safe_load(file)
    settings = DeskewSettings(**raw_settings)
    print(f"Deskewing parameters: {asdict(settings)}")

    # Load positions log and generate pos_hcs_idx
    if positions is not None:
        with open(positions, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            pos_log = [row for row in reader]

    dataset = Dataset(data_path)
    dask_array = dataset.as_array()
    # reader = read_micromanager(data_path)
    writer = open_ome_zarr(
        output_path, mode="w", layout="hcs", channel_names=dataset.get_channel_names()
    )

    P, T, C, Z, Y, X = *dask_array.shape,
    deskewed_shape, voxel_size = get_deskewed_data_shape(
        (Z, Y, X), settings.pixel_size_um, settings.ls_angle_deg, settings.px_to_scan_ratio
    )

    if positions is not None:
        pos_hcs_idx = [
            (row['well_id'][0], row['well_id'][1:], row['site_num']) for row in pos_log
        ]
    else:
        pos_hcs_idx = [(0, p, 0) for p in range(P)]

    for p in range(P):
        position = writer.create_position(*pos_hcs_idx[p])
        # Handle transforms and metadata
        transform = TransformationMeta(
            type="scale",
            scale=2 * (1,) + voxel_size,
        )
        position.create_zeros(
            name="0",
            shape=(
                T,
                C,
            )
            + deskewed_shape,
            dtype=np.int16,
            transform=[transform],
        )

    # Loop through (P, T, C), deskewing and writing as we go
    with Pool(N_processes) as p:
        p.starmap(
            partial(fun, dask_array, writer, settings, keep_overhang, pos_hcs_idx),
            itertools.product(range(P), range(T), range(C))
        )

    # Write metadata
    writer.zattrs["deskewing"] = asdict(settings)
    writer.zattrs["mm-meta"] = reader.mm_meta["Summary"]

    # Optional view
    # if view:
    #     v = napari.Viewer()
    #     v.add_image(deskewed)
    #     v.layers[-1].scale = voxel_size
    #     napari.run()


def fun(data_array, writer, settings, keep_overhang, pos_hcs_idx, p, t, c):
    # print(f"Deskewing c={c}/{C-1}, t={t}/{T-1}, p={p}/{P-1}")
    data = np.asarray(data_array[p, t, c])  # zyx

    # Deskew
    deskewed = deskew_data(
        data, settings.px_to_scan_ratio, settings.ls_angle_deg, keep_overhang
    )

    writer[os.path.join(*pos_hcs_idx[p])]["0"][t, c] = deskewed # write to zarr


if __name__ == "__main__":
    deskew()
