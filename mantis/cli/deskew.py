import multiprocessing as mp
import itertools
import os
import click
import napari
import numpy as np
import yaml

from iohub import open_ome_zarr
from iohub.ngff_meta import TransformationMeta
from mantis.analysis.AnalysisSettings import DeskewSettings
from mantis.analysis.deskew import deskew_data, get_deskewed_data_shape
from dataclasses import asdict
from functools import partial
from tqdm import tqdm
from mantis.cli.parsing import (
    input_data_path_argument,
    deskew_param_argument,
    output_dataset_options,
)
from time import time
from natsort import natsorted


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
    click.echo(f"Number of positions: {len(position_paths)}")
    click.echo(f"Output shape: {output_shape}")
    # Create output dataset
    output_dataset = open_ome_zarr(
        output_path, layout="hcs", mode="a", channel_names=channel_names
    )
    # This takes care of the logic for single position or multiple position by wildcards
    for filepath in position_paths:
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
            chunks=(1, 1, 64) + deskewed_shape[-2:],
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


def single_process(data_array, output_path, settings, keep_overhang, t, c):
    click.echo(f"Deskewing c={c}, t={t}")
    click.echo(f'data_array.shape {data_array[0][t, c].shape}')
    start_time = time()
    data = data_array[0][t, c]

    click.echo(f'total time {time()-start_time}')

    # Deskew
    deskewed = deskew_data(
        data, settings.px_to_scan_ratio, settings.ls_angle_deg, keep_overhang, verbose=True
    )
    click.echo(f"Writing.. c={c}, t={t}")
    # Write to file
    with open_ome_zarr(output_path, mode="r+") as output_dataset:
        output_dataset[0][t, c] = deskewed

    click.echo(f"Finished Writing.. c={c}, t={t}")


def deskew_cli(
    input_data_path, output_path, deskew_params_path, view, keep_overhang, num_cores
):
    """
    Deskew a single position and parallelized over T and C
    """

    # Get the reader and writer
    click.echo(f'Input data path:\t{input_data_path}')
    click.echo(f'Output data path:\t{str(output_path)}')
    input_dataset = open_ome_zarr(str(input_data_path))
    click.echo(input_dataset.print_tree())

    settings = _get_deskew_params(deskew_params_path)
    T, C, Z, Y, X = input_dataset.data.shape
    click.echo(f'Dataset shape:\t{input_dataset.data.shape}')

    deskewed_shape, voxel_size = get_deskewed_data_shape(
        (Z, Y, X),
        settings.pixel_size_um,
        settings.ls_angle_deg,
        settings.px_to_scan_ratio,
        keep_overhang,
    )

    # Loop through (T, C), deskewing and writing as we go
    click.echo(f"Starting multiprocess pool with cores {num_cores}")
    with mp.Pool(num_cores) as p:
        p.starmap(
            partial(single_process, input_dataset, str(output_path), settings, keep_overhang),
            itertools.product(range(T), range(C)),
        )
    # TODO: check if output_path[0] will work when we parallelize
    # Write metadata
    output_zarr_root = str(output_path).split(os.path.sep)[:-3]
    output_zarr_root = os.path.join(*output_zarr_root)
    click.echo(f'output_zarr_root \t{output_zarr_root}')
    with open_ome_zarr(output_zarr_root, mode='r+') as dataset:
        dataset.zattrs["deskewing"] = asdict(settings)
        # TODO: not sure what this metadata was for
        # dataset.zattrs["mm-meta"] = input_dataset.mm_meta["Summary"]

    # Optional view
    # TODO: implement the viewer capability
    if view:
        click.echo(f"View mode activated")
        viewer = napari.Viewer()
        napari.run()
    else:
        viewer = None

    # if viewer is not None:
    #     curr_layer = 'deskew_' + str(t) + 'T_' + str(c) + 'C'
    #     viewer.add_image(deskewed, name=curr_layer)
    #     viewer.layers[curr_layer].scale = voxel_size


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
def deskew(
    input_data_path, deskew_param_path, output_path, view, keep_overhang, num_cores, slurm
):
    "Deskews a single position across T and C axes using a parameter file generated by estimate_deskew.py"
    input_data_path = natsorted(input_data_path)
    # Handle single position or wildcard filepath
    list_output_pos = _get_output_paths(input_data_path, output_path)
    print(f'List of input pos:{input_data_path} output_pos:{list_output_pos}')

    if not slurm:
        # Create a zarr store output to mirror the input
        _create_empty_zarr(input_data_path, deskew_param_path, output_path, keep_overhang)

    # Parallel Position
    # mp_position_deskew = parallel(deskew_cli)
    deskew_parameters = (
        input_data_path,
        list_output_pos,
        deskew_param_path,
        view,
        keep_overhang,
        num_cores,
    )
    deskew_params = (deskew_param_path, view, keep_overhang, num_cores)

    # Check number of processes and available cores
    # Multiprocess per position
    for pos_in, pos_out in zip(input_data_path, list_output_pos):
        args = (pos_in,) + (pos_out,) + deskew_params
        deskew_cli(*args)
