import glob
import itertools
import os

from functools import partial
from pathlib import Path
from typing import List

import click
import numpy as np
import torch
import torch.multiprocessing as mp

from iohub import open_ome_zarr
from natsort import natsorted
from waveorder.models import phase_thick_3d


def get_output_paths(list_pos: List[str], output_data_path: Path) -> List[str]:
    """Generates a mirrored output path list given an input list of positions"""
    list_output_path = []
    for filepath in list_pos:
        path_strings = filepath.split(os.path.sep)[-3:]
        list_output_path.append(os.path.join(output_data_path, *path_strings))
    return list_output_path

def create_empty_zarr(position_paths, output_path, recon_params):
    # Load datasets
    # Take position 0 as sample
    input_dataset = open_ome_zarr(str(position_paths[0]), mode="r")
    T, C, Z, Y, X = input_dataset.data.shape

    # -------------------------------
    # TODO: this logic should be replaced by the Transferfunction Settings recOrder
    recon_type = recon_params[0]
    recon_dim = recon_params[1]
    recon_biref = False
    recon_phase = False
    if recon_type == "bire":
        recon_biref = True
    elif recon_type == "phase":
        recon_phase = True
    elif recon_type == "bire_phase":
        recon_phase = True
        recon_biref = True
    # -------------------------------
    # Prepare output dataset
    channel_names = []
    if recon_biref:
        channel_names.append("Retardance")
        channel_names.append("Orientation")
        channel_names.append("BF")
        channel_names.append("Pol")
        output_z_shape = input_dataset.data.shape[2]
    if recon_phase:
        if recon_dim == 2:
            channel_names.append("Phase2D")
            output_z_shape = 1
        elif recon_dim == 3:
            channel_names.append("Phase3D")
            output_z_shape = input_dataset.data.shape[2]
    click.echo(f"channel names {channel_names}")

    # Output shape based on the type of reconstruction
    output_shape = (
        T,
        len(channel_names),
        output_z_shape,
    ) + input_dataset.data.shape[3:]

    # Create output dataset
    output_dataset = open_ome_zarr(
        output_path, layout="hcs", mode="w", channel_names=channel_names
    )
    for filepath in position_paths:
        path_strings = filepath.split(os.path.sep)[-3:]
        pos = output_dataset.create_position(
            str(path_strings[0]), str(path_strings[1]), str(path_strings[2])
        )
        _ = pos.create_zeros(
            name="0",
            shape=output_shape,
            dtype=np.float32,
            chunks=(
                1,
                1,
                1,
            )
            + input_dataset.data.shape[3:],  # chunk by YX
        )
    click.echo(f"output_shape {output_shape}")
    input_dataset.close()

def reconstruct_phase3D_n_save(position, output_path, recon_params, t):
    # Initialize torch module in each worker process
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)

    # Get reconstruction paremeters
    # TODO: these should be obtained from metadata or yaml?
    (
        real_potential_transfer_function,
        imaginary_potential_transfer_function,
        z_padding,
        z_pixel_size,
        wavelength_illumination,
        regularization_strength,
    ) = recon_params

    c = 0
    click.echo(f"Processing c={c}, t={t}")
    zyx_data = position[0][t, c].astype(np.float32)
    click.echo(f"Loaded zyx data c={c}, t={t}")

    # Process volume
    zyx_phase = phase_thick_3d.apply_inverse_transfer_function(
        zyx_data=torch.tensor(zyx_data),
        real_potential_transfer_function=real_potential_transfer_function,
        imaginary_potential_transfer_function=imaginary_potential_transfer_function,
        z_padding=z_padding,
        z_pixel_size=z_pixel_size,
        wavelength_illumination=wavelength_illumination,
        absorption_ratio=0.0,
        method="Tikhonov",
        reg_re=regularization_strength,
    )
    click.echo(f"Finished c={c}, t={t}")

    with open_ome_zarr(output_path, mode="r+") as output_dataset:
        output_dataset[0][t, c] = np.array(zyx_data)

def reconstruct_single_position(
        input_data_path: Path,
        output_path: Path = './deskewed.zarr',
        reconstruction_function = None,
        reconstruction_parameters = None,
        num_processes: int = mp.cpu_count(),
    )->None:

    # Get the reader and writer
    click.echo(f'Input data path:\t{input_data_path}')
    click.echo(f'Output data path:\t{str(output_path)}')
    input_dataset = open_ome_zarr(str(input_data_path))
    click.echo(input_dataset.print_tree())
    T, C, Z, Y, X = input_dataset.data.shape
    click.echo(f'Dataset shape:\t{input_dataset.data.shape}')

    # Loop through (T, C), deskewing and writing as we go
    click.echo(f"Starting multiprocess pool with {num_processes} processes")
    with mp.Pool(num_processes) as p:
        p.starmap(
            partial(
                reconstruction_function, input_dataset, str(output_path), reconstruction_parameters
            ),
            itertools.product(range(T)),
        )
