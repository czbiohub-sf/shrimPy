from mantis.cli.parsing import (
    input_data_path_argument,
    deskew_param_argument,
    output_dataset_options,
)

import multiprocessing as mp
from mantis.cli.printing import echo_headline, echo_settings

import csv
import itertools
import os

from dataclasses import asdict
from functools import partial
import click
import dask.array as da

import napari
import numpy as np
import yaml

from iohub import open_ome_zarr
from iohub.ngff_meta import TransformationMeta

from mantis.analysis.AnalysisSettings import DeskewSettings
from mantis.analysis.deskew import deskew_data, get_deskewed_data_shape
import pdb
from tqdm import tqdm

N_processes = 6
# set environment variable
os.environ['DISPLAY'] = ':1'


# def position_CSV(positions):
#     # Load positions log and generate pos_hcs_idx
#     if positions is not None:
#         with open(positions, newline='') as csvfile:
#             reader = csv.DictReader(csvfile)
#             pos_log = [row for row in reader]

#         pos_hcs_idx = [
#             (row['well_id'][0], row['well_id'][1:], row['site_num']) for row in pos_log
#             ]
#     else:
#         pos_hcs_idx = [(0, p, 0) for p in range(P)]

# def _create_empty_zarr():
#     for p in range(P):
#         position = writer.create_position(*pos_hcs_idx[p])

#         position.create_zeros(
#             name="0",
#             shape=(
#                 T,
#                 C,
#             )
#             + deskewed_shape,
#             chunks=(1, 1, 64) + deskewed_shape[-2:],
#             dtype=np.uint16,
#             transform=[transform],
# )


def _get_deskew_params(deskew_params_path):
    # Load params
    with open(deskew_params_path) as file:
        raw_settings = yaml.safe_load(file)
    settings = DeskewSettings(**raw_settings)
    print(f"Deskewing parameters: {asdict(settings)}")
    return settings


def _create_empty_zarr(position_paths, deskew_params_path, output_path, keep_overhang):
    # Load the "0" position to infer dataset information
    input_dataset = open_ome_zarr(str(position_paths[0]), mode="r")
    T, C, Z, Y, X = input_dataset.data.shape

    # Get the deskewing parameters
    settings = _get_deskew_params(deskew_params_path)
    deskewed_shape, voxel_size = get_deskewed_data_shape(
        (Z, Y, X),
        settings.pixel_size_um,
        settings.ls_angle_deg,
        settings.px_to_scan_ratio,
        keep_overhang,
    )

    #TODO: remove this.hardcoding due to mismatch in shape
    # deskewed_shape = (300, 297, 567)

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
    print(
        f'createzero shape {output_shape}, deskdew_shape {deskewed_shape}, {deskewed_shape[-2:]}'
    )
    click.echo(f"Number of positions: {len(position_paths)}")
    click.echo(f"Output shape: {output_shape}")
    # Create output dataset
    output_dataset = open_ome_zarr(
        output_path, layout="hcs", mode="w", channel_names=channel_names
    )
    # This takes care of the logic for single position or multiple position by wildcards
    for filepath in position_paths:
        path_strings = filepath.split(os.path.sep)[-3:]
        pos = output_dataset.create_position(
            str(path_strings[0]), str(path_strings[1]), str(path_strings[2])
        )

        output_array = pos.create_zeros(
            name="0",
            shape=(
                T,
                C,
            )
            + deskewed_shape,
            chunks=(1, 1, 64) + deskewed_shape[-2:],
            dtype=np.uint16,
            transform=[transform],
        )

    # pdb.set_trace()


def _get_output_paths(list_pos, output_path):
    # From the position filepath generate the output filepath
    list_output_path = []
    for filepath in list_pos:
        path_strings = filepath.split(os.path.sep)[-3:]
        list_output_path.append(os.path.join(output_path, *path_strings))
    return list_output_path


def deskew_cli(input_data_path, output_path, deskew_params_path, view, keep_overhang):
    """ "
    Deskew over a single position and parallelized over T and C
    """

    # Get the reader and writer
    print(f'INPUT DATA:{input_data_path[0]}')
    print(f'OUTPUT_DATA:{str(output_path[0])}')
    input_dataset = open_ome_zarr(str(input_data_path[0]))
    print(input_dataset.print_tree())

    settings = _get_deskew_params(deskew_params_path)
    T, C, Z, Y, X = input_dataset.data.shape
    print(f'dataset shape {input_dataset.data.shape}')

    deskewed_shape, voxel_size = get_deskewed_data_shape(
        (Z, Y, X),
        settings.pixel_size_um,
        settings.ls_angle_deg,
        settings.px_to_scan_ratio,
        keep_overhang,
    )

    # Optional view
    # TODO: implement the viewer capability
    if view:
        viewer = napari.Viewer()
        napari.run()
    else:
        viewer = None

    # Loop through (T, C), deskewing and writing as we go
    with mp.Pool(N_processes) as p:
        p.starmap(
            partial(
                single_process,
                input_dataset,
                str(output_path[0]),
                settings,
                viewer,
                voxel_size,
                keep_overhang,
            ),
            itertools.product(range(T), range(C)),
        )
    # TODO: check if output_path[0] will work when we parallelize

    # Write metadata
    output_zarr_root = str(output_path[0]).split(os.path.sep)[:-3]
    output_zarr_root = os.path.join(*output_zarr_root)
    with open_ome_zarr(output_zarr_root, mode='r+') as dataset:
        dataset.zattrs["deskewing"] = asdict(settings)
        # TODO: not sure what this metadata was for
        # dataset.zattrs["mm-meta"] = input_dataset.mm_meta["Summary"]


def single_process(data_array, output_path, settings, viewer, voxel_size, keep_overhang, t, c):
    print(f"Deskewing c={c}, t={t}")
    with open_ome_zarr(output_path, mode="r+") as output_dataset:
        data = output_dataset[0][t, c]
        print(f'data shape: {data.shape}')

    # Deskew
    deskewed = deskew_data(
        data_array[0][t, c], settings.px_to_scan_ratio, settings.ls_angle_deg, keep_overhang
    )

    # Write to file
    with open_ome_zarr(output_path, mode="r+") as output_dataset:
        output_dataset[0][t, c] = deskewed

    # if viewer is not None:
    #     curr_layer = 'deskew_' + str(t) + 'T_' + str(c) + 'C'
    #     viewer.add_image(deskewed, name=curr_layer)
    #     viewer.layers[curr_layer].scale = voxel_size


def parallel(func=None, **options):
    """
    Decorator for multiprocessing Pool
    """
    if func is None:
        return functools.partial(parallel, **options)

    def wrapper(*arguments):
        # Split the arguments
        position_paths = arguments[0]
        output_paths = arguments[1]
        recon_params = arguments[2:-1]
        # Check number of processes and available cores
        processes = mp.cpu_count() if arguments[-1] > mp.cpu_count() else arguments[-1]

        # Multiprocess per position
        result = []
        with mp.Pool(processes) as pool:
            for pos_in, pos_out in zip(position_paths, output_paths):
                args = (pos_in,) + recon_params + (pos_out,)
                for results in tqdm(pool.starmap(func, [args])):
                    result.append(results)
        return result

    return wrapper


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
def deskew(input_data_path, deskew_param_path, output_path, view, keep_overhang):
    "Deskews single position across T and C axes using a parameter file generated with estimate_deskew.py"

    # Handle single position or wildcard filepath
    list_output_pos = _get_output_paths(input_data_path, output_path)
    print(f'List of input pos:{input_data_path} output_pos:{list_output_pos}')
    # Create a zarr store output to mirror the input
    _create_empty_zarr(input_data_path, deskew_param_path, output_path, keep_overhang)
    # pdb.set_trace()
    deskew_cli(input_data_path, list_output_pos, deskew_param_path, view, keep_overhang)


# mantis deskew /home/eduardo.hirata/CompMicro/projects/mantis/2023_05_10_PCNA_RAC1/0-crop_conver_zarr/sample.zarr/0/0/0 ./deskew_settings.yml
# mantis deskew /home/eduardo.hirata/CompMicro/projects/mantis/2023_05_10_PCNA_RAC1/0-crop_conver_zarr/sample_short.zarr/0/0/0 ./deskew_settings.yml
