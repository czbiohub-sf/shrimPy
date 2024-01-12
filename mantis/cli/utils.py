import contextlib
import inspect
import io
import itertools
import multiprocessing as mp

from functools import partial
from pathlib import Path
from typing import Tuple

import ants
import click
import largestinteriorrectangle as lir
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import yaml

from iohub.ngff import Position, open_ome_zarr
from iohub.ngff_meta import TransformationMeta
from numpy.typing import DTypeLike
from tqdm import tqdm


# TODO: replace this with recOrder recOrder.cli.utils.create_empty_hcs()
def create_empty_zarr(
    position_paths: list[Path],
    output_path: Path,
    output_zyx_shape: Tuple[int],
    chunk_zyx_shape: Tuple[int] = None,
    voxel_size: Tuple[int, float] = (1, 1, 1),
) -> None:
    """Create an empty zarr store mirroring another store"""
    DTYPE = np.float32
    MAX_CHUNK_SIZE = 500e6  # in bytes
    bytes_per_pixel = np.dtype(DTYPE).itemsize

    # Load the first position to infer dataset information
    input_dataset = open_ome_zarr(str(position_paths[0]), mode="r")
    T, C, Z, Y, X = input_dataset.data.shape

    click.echo("Creating empty array...")

    # Handle transforms and metadata
    transform = TransformationMeta(
        type="scale",
        scale=2 * (1,) + voxel_size,
    )

    # Prepare output dataset
    channel_names = input_dataset.channel_names

    # Output shape based on the type of reconstruction
    output_shape = (T, len(channel_names)) + output_zyx_shape
    click.echo(f"Number of positions: {len(position_paths)}")
    click.echo(f"Output shape: {output_shape}")

    # Create output dataset
    output_dataset = open_ome_zarr(
        output_path, layout="hcs", mode="w", channel_names=channel_names
    )
    if chunk_zyx_shape is None:
        chunk_zyx_shape = list(output_zyx_shape)
        # chunk_zyx_shape[-3] > 1 ensures while loop will not stall if single
        # XY image is larger than MAX_CHUNK_SIZE
        while (
            chunk_zyx_shape[-3] > 1
            and np.prod(chunk_zyx_shape) * bytes_per_pixel > MAX_CHUNK_SIZE
        ):
            chunk_zyx_shape[-3] = np.ceil(chunk_zyx_shape[-3] / 2).astype(int)
        chunk_zyx_shape = tuple(chunk_zyx_shape)

    chunk_size = 2 * (1,) + chunk_zyx_shape
    click.echo(f"Chunk size: {chunk_size}")

    # This takes care of the logic for single position or multiple position by wildcards
    for path in position_paths:
        path_strings = Path(path).parts[-3:]
        pos = output_dataset.create_position(
            str(path_strings[0]), str(path_strings[1]), str(path_strings[2])
        )

        _ = pos.create_zeros(
            name="0",
            shape=output_shape,
            chunks=chunk_size,
            dtype=DTYPE,
            transform=[transform],
        )

    input_dataset.close()


def get_output_paths(input_paths: list[Path], output_zarr_path: Path) -> list[Path]:
    """Generates a mirrored output path list given an input list of positions"""
    list_output_path = []
    for path in input_paths:
        # Select the Row/Column/FOV parts of input path
        path_strings = Path(path).parts[-3:]
        # Append the same Row/Column/FOV to the output zarr path
        list_output_path.append(Path(output_zarr_path, *path_strings))
    return list_output_path


def apply_function_to_zyx_and_save(
    func, position: Position, output_path: Path, t_idx: int, c_idx: int, **kwargs
) -> None:
    """Load a zyx array from a Position object, apply a transformation and save the result to file"""
    click.echo(f"Processing c={c_idx}, t={t_idx}")

    zyx_data = position[0][t_idx, c_idx]
    if _check_nan_n_zeros(zyx_data):
        click.echo(f"Skipping c={c_idx}, t={t_idx} due to all zeros or nans")
    else:
        # Apply function
        processed_zyx = func(zyx_data, **kwargs)

        # Write to file
        with open_ome_zarr(output_path, mode="r+") as output_dataset:
            output_dataset[0][t_idx, c_idx] = processed_zyx

        click.echo(f"Finished Writing.. c={c_idx}, t={t_idx}")


def process_single_position(
    func,
    input_data_path: Path,
    output_path: Path,
    num_processes: int = mp.cpu_count(),
    **kwargs,
) -> None:
    """Register a single position with multiprocessing parallelization over T and C"""
    # Function to be applied
    click.echo(f"Function to be applied: \t{func}")

    # Get the reader and writer
    click.echo(f"Input data path:\t{input_data_path}")
    click.echo(f"Output data path:\t{str(output_path)}")
    input_dataset = open_ome_zarr(str(input_data_path))
    stdout_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer):
        input_dataset.print_tree()
    click.echo(f" Input data tree: {stdout_buffer.getvalue()}")

    T, C, _, _, _ = input_dataset.data.shape

    # Check the arguments for the function
    all_func_params = inspect.signature(func).parameters.keys()
    # Extract the relevant kwargs for the function 'func'
    func_args = {}
    non_func_args = {}

    for k, v in kwargs.items():
        if k in all_func_params:
            func_args[k] = v
        else:
            non_func_args[k] = v

    # Write the settings into the metadata if existing
    # TODO: alternatively we can throw all extra arguments as metadata.
    if 'extra_metadata' in non_func_args:
        # For each dictionary in the nest
        with open_ome_zarr(output_path, mode='r+') as output_dataset:
            for params_metadata_keys in kwargs['extra_metadata'].keys():
                output_dataset.zattrs['extra_metadata'] = non_func_args['extra_metadata']

    # Loop through (T, C), deskewing and writing as we go
    click.echo(f"\nStarting multiprocess pool with {num_processes} processes")
    with mp.Pool(num_processes) as p:
        p.starmap(
            partial(
                apply_function_to_zyx_and_save,
                func,
                input_dataset,
                str(output_path),
                **func_args,
            ),
            itertools.product(range(T), range(C)),
        )


def scale_affine(start_shape_zyx, scaling_factor_zyx=(1, 1, 1), end_shape_zyx=None):
    center_Y_start, center_X_start = np.array(start_shape_zyx)[-2:] / 2
    if end_shape_zyx is None:
        center_Y_end, center_X_end = (center_Y_start, center_X_start)
    else:
        center_Y_end, center_X_end = np.array(end_shape_zyx)[-2:] / 2

    scaling_matrix = np.array(
        [
            [scaling_factor_zyx[-3], 0, 0, 0],
            [
                0,
                scaling_factor_zyx[-2],
                0,
                -center_Y_start * scaling_factor_zyx[-2] + center_Y_end,
            ],
            [
                0,
                0,
                scaling_factor_zyx[-1],
                -center_X_start * scaling_factor_zyx[-1] + center_X_end,
            ],
            [0, 0, 0, 1],
        ]
    )
    return scaling_matrix


def rotate_affine(
    start_shape_zyx: Tuple, angle: float = 0.0, end_shape_zyx: Tuple = None
) -> np.ndarray:
    """
    Rotate Transformation Matrix

    Parameters
    ----------
    start_shape_zyx : Tuple
        Shape of the input
    angle : float, optional
        Angles of rotation in degrees
    end_shape_zyx : Tuple, optional
       Shape of output space

    Returns
    -------
    np.ndarray
        Rotation matrix
    """
    # TODO: make this 3D?
    center_Y_start, center_X_start = np.array(start_shape_zyx)[-2:] / 2
    if end_shape_zyx is None:
        center_Y_end, center_X_end = (center_Y_start, center_X_start)
    else:
        center_Y_end, center_X_end = np.array(end_shape_zyx)[-2:] / 2

    theta = np.radians(angle)

    rotation_matrix = np.array(
        [
            [1, 0, 0, 0],
            [
                0,
                np.cos(theta),
                -np.sin(theta),
                -center_Y_start * np.cos(theta)
                + np.sin(theta) * center_X_start
                + center_Y_end,
            ],
            [
                0,
                np.sin(theta),
                np.cos(theta),
                -center_Y_start * np.sin(theta)
                - center_X_start * np.cos(theta)
                + center_X_end,
            ],
            [0, 0, 0, 1],
        ]
    )

    affine_rot_n_scale_matrix_zyx = rotation_matrix

    return affine_rot_n_scale_matrix_zyx


def affine_transform(
    zyx_data: np.ndarray,
    matrix: np.ndarray,
    output_shape_zyx: Tuple,
    method='ants',
    crop_output_slicing: bool = None,
) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    zyx_data : np.ndarray
        3D input array to be transformed
    matrix : np.ndarray
        3D Homogenous transformation matrix
    output_shape_zyx : Tuple
        output target zyx shape
    method : str, optional
        method to use for transformation, by default 'ants'
    crop_output : bool, optional
        crop the output to the largest interior rectangle, by default False

    Returns
    -------
    np.ndarray
        registered zyx data
    """

    Z, Y, X = zyx_data.shape[-3:]
    if crop_output_slicing is not None:
        Z_slice, Y_slice, X_slice = crop_output_slicing
        Z = Z_slice.stop - Z_slice.start
        Y = Y_slice.stop - Y_slice.start
        X = X_slice.stop - X_slice.start

    # TODO: based on the signature of this function, it should not be called on 4D array
    if zyx_data.ndim == 4:
        registered_czyx = np.zeros((zyx_data.shape[0], Z, Y, X), dtype=np.float32)
        for c in range(zyx_data.shape[0]):
            registered_czyx[c] = affine_transform(
                zyx_data[c],
                matrix,
                output_shape_zyx,
                method,
                crop_output_slicing,
            )
        return registered_czyx
    else:
        # Convert nans to 0
        zyx_data = np.nan_to_num(zyx_data, nan=0)

        # NOTE: default set to ANTS apply_affine method until we decide we get a benefit from using cupy
        # The ants method on CPU is 10x faster than scipy on CPU. Cupy method has not been bencharked vs ANTs

        if method == 'ants':
            # The output has to be a ANTImage Object
            empty_target_array = np.zeros((output_shape_zyx), dtype=np.float32)
            target_zyx_ants = ants.from_numpy(empty_target_array)

            T_ants = numpy_to_ants_transform_zyx(matrix)

            zyx_data_ants = ants.from_numpy(zyx_data.astype(np.float32))
            registered_zyx = T_ants.apply_to_image(
                zyx_data_ants, reference=target_zyx_ants
            ).numpy()

        elif method == 'scipy':
            registered_zyx = ndi.affine_transform(zyx_data, matrix, output_shape_zyx)

        else:
            raise ValueError(f'Unknown method {method}')

        # Crop the output to the largest interior rectangle
        if crop_output_slicing is not None:
            registered_zyx = registered_zyx[Z_slice, Y_slice, X_slice]

    return registered_zyx


def find_lir(registered_zyx: np.ndarray, plot: bool = False) -> Tuple:
    # Find the lir YX
    registered_yx_bool = registered_zyx[registered_zyx.shape[0] // 2].copy()
    registered_yx_bool = registered_yx_bool > 0 * 1.0
    rectangle_coords_yx = lir.lir(registered_yx_bool)

    x = rectangle_coords_yx[0]
    y = rectangle_coords_yx[1]
    width = rectangle_coords_yx[2]
    height = rectangle_coords_yx[3]
    corner1_xy = (x, y)  # Bottom-left corner
    corner2_xy = (x + width, y)  # Bottom-right corner
    corner3_xy = (x + width, y + height)  # Top-right corner
    corner4_xy = (x, y + height)  # Top-left corner
    rectangle_xy = np.array((corner1_xy, corner2_xy, corner3_xy, corner4_xy))
    X_slice = slice(rectangle_xy.min(axis=0)[0], rectangle_xy.max(axis=0)[0])
    Y_slice = slice(rectangle_xy.min(axis=0)[1], rectangle_xy.max(axis=0)[1])

    # Find the lir Z
    zyx_shape = registered_zyx.shape
    registered_zx_bool = registered_zyx.transpose((2, 0, 1)) > 0
    registered_zx_bool = registered_zx_bool[zyx_shape[0] // 2].copy()
    rectangle_coords_zx = lir.lir(registered_zx_bool)
    x = rectangle_coords_zx[0]
    z = rectangle_coords_zx[1]
    width = rectangle_coords_zx[2]
    height = rectangle_coords_zx[3]
    corner1_zx = (x, z)  # Bottom-left corner
    corner2_zx = (x + width, z)  # Bottom-right corner
    corner3_zx = (x + width, z + height)  # Top-right corner
    corner4_zx = (x, z + height)  # Top-left corner
    rectangle_zx = np.array((corner1_zx, corner2_zx, corner3_zx, corner4_zx))
    Z_slice = slice(rectangle_zx.min(axis=0)[1], rectangle_zx.max(axis=0)[1])

    if plot:
        rectangle_yx = plt.Polygon(
            (corner1_xy, corner2_xy, corner3_xy, corner4_xy),
            closed=True,
            fill=None,
            edgecolor="r",
        )
        # Add the rectangle to the plot
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(registered_yx_bool)
        ax[0].add_patch(rectangle_yx)

        rectangle_zx = plt.Polygon(
            (corner1_zx, corner2_zx, corner3_zx, corner4_zx),
            closed=True,
            fill=None,
            edgecolor="r",
        )
        ax[1].imshow(registered_zx_bool)
        ax[1].add_patch(rectangle_zx)
        plt.savefig("./lir.png")

    return (Z_slice, Y_slice, X_slice)


def copy_n_paste(zyx_data: np.ndarray, zyx_slicing_params: list) -> np.ndarray:
    """
    Load a zyx array and crop given a list of ZYX slices()

    Parameters
    ----------
    zyx_data : np.ndarray
        data to copy
    zyx_slicing_params : list
        list of slicing parameters for z,y,x

    Returns
    -------
    np.ndarray
        crop of the input zyx_data given the slicing parameters
    """
    zyx_data = np.nan_to_num(zyx_data, nan=0)
    zyx_data_sliced = zyx_data[
        zyx_slicing_params[0],
        zyx_slicing_params[1],
        zyx_slicing_params[2],
    ]
    return zyx_data_sliced


def copy_n_paste_czyx(czyx_data: np.ndarray, czyx_slicing_params: list) -> np.ndarray:
    """
    Load a zyx array and crop given a list of ZYX slices()

    Parameters
    ----------
    czyx_data : np.ndarray
        data to copy
    czyx_slicing_params : list
        list of slicing parameters for z,y,x

    Returns
    -------
    np.ndarray
        crop of the input czyx_data given the slicing parameters
    """
    czyx_data_sliced = czyx_data[
        :,
        czyx_slicing_params[0],
        czyx_slicing_params[1],
        czyx_slicing_params[2],
    ]
    return czyx_data_sliced


def find_lir_slicing_params(
    input_zyx_shape: Tuple,
    target_zyx_shape: Tuple,
    transformation_matrix: np.ndarray,
    plot: bool = False,
) -> Tuple:
    """
    Find the largest internal rectangle between the transformed input and the target
    and return the cropping parameters

    Parameters
    ----------
    input_zyx_shape : Tuple
        shape of input array
    target_zyx_shape : Tuple
        shape of target array
    transformation_matrix : np.ndarray
        transformation matrix between input and target

    Returns
    -------
    Tuple
        Slicing parameters to crop LIR

    """
    print('Starting Largest interior rectangle (LIR) search')

    # Make dummy volumes
    img1 = np.ones(tuple(input_zyx_shape), dtype=np.float32)
    img2 = np.ones(tuple(target_zyx_shape), dtype=np.float32)

    # Conver to ants objects
    target_zyx_ants = ants.from_numpy(img2.astype(np.float32))
    zyx_data_ants = ants.from_numpy(img1.astype(np.float32))

    ants_composed_matrix = numpy_to_ants_transform_zyx(transformation_matrix)

    # Apply affine
    registered_zyx = ants_composed_matrix.apply_to_image(
        zyx_data_ants, reference=target_zyx_ants
    )

    Z_slice, Y_slice, X_slice = find_lir(registered_zyx.numpy(), plot=plot)

    return (Z_slice, Y_slice, X_slice)


def append_channels(input_data_path: Path, target_data_path: Path) -> None:
    """
    Append channels to a target zarr store

    Parameters
    ----------
    input_data_path : Path
        input zarr path = /input.zarr
    target_data_path : Path
        target zarr path  = /target.zarr
    """
    appending_dataset = open_ome_zarr(input_data_path, mode="r")
    appending_channel_names = appending_dataset.channel_names
    with open_ome_zarr(target_data_path, mode="r+") as dataset:
        target_data_channel_names = dataset.channel_names
        num_channels = len(target_data_channel_names) - 1
        print(f"channels in target {target_data_channel_names}")
        print(f"adding channels {appending_channel_names}")
        for name, position in tqdm(dataset.positions(), desc='Positions'):
            for i, appending_channel_idx in enumerate(
                tqdm(appending_channel_names, desc='Channel', leave=False)
            ):
                position.append_channel(appending_channel_idx)
                position["0"][:, num_channels + i + 1] = appending_dataset[str(name)][0][:, i]
        dataset.print_tree()
    appending_dataset.close()


def numpy_to_ants_transform_zyx(T_numpy: np.ndarray):
    """Homogeneous 3D transformation matrix from numpy to ants

    Parameters
    ----------
    numpy_transform :4x4 homogenous matrix

    Returns
    -------
    Ants transformation matrix object
    """
    assert T_numpy.shape == (4, 4)

    T_ants_style = T_numpy[:, :-1].ravel()
    T_ants_style[-3:] = T_numpy[:3, -1]
    T_ants = ants.new_ants_transform(
        transform_type='AffineTransform',
    )
    T_ants.set_parameters(T_ants_style)

    return T_ants


def numpy_to_ants_transform_czyx(T_numpy: np.ndarray):
    """Homogeneous 3D transformation matrix from numpy to ants

    Parameters
    ----------
    numpy_transform :4x4 homogenous matrix

    Returns
    -------
    Ants transformation matrix object
    """
    assert T_numpy.shape == (5, 5)
    shape = T_numpy.shape
    T_ants_style = T_numpy[:, :-1].ravel()
    T_ants_style[-shape[0] + 1 :] = T_numpy[-shape[0] : -1, -1]
    T_ants = ants.new_ants_transform(
        transform_type='AffineTransform',
    )
    T_ants.set_parameters(T_ants_style)

    return T_ants


def ants_to_numpy_transform_zyx(T_ants):
    """
    Convert the ants transformation matrix to numpy 3D homogenous transform

    Modified from Jordao's dexp code

    Parameters
    ----------
    T_ants : Ants transfromation matrix object

    Returns
    -------
    np.array
        Converted Ants to numpy array

    """

    T_numpy = T_ants.parameters.reshape((3, 4), order="F")
    T_numpy[:, :3] = T_numpy[:, :3].transpose()
    T_numpy = np.vstack((T_numpy, np.array([0, 0, 0, 1])))

    # Reference:
    # https://sourceforge.net/p/advants/discussion/840261/thread/9fbbaab7/
    # https://github.com/netstim/leaddbs/blob/a2bb3e663cf7fceb2067ac887866124be54aca7d/helpers/ea_antsmat2mat.m
    # T = original translation offset from A
    # T = T + (I - A) @ centering

    T_numpy[:3, -1] += (np.eye(3) - T_numpy[:3, :3]) @ T_ants.fixed_parameters

    return T_numpy


def apply_stabilization_over_time_ants(
    list_of_shifts: list,
    input_data_path: Path,
    output_path: Path,
    time_indices: list = [0],
    input_channel_idx: list = [],
    output_channel_idx: list = [],
    num_processes: int = 1,
    **kwargs,
) -> None:
    """Apply stabilization over time"""
    # Function to be applied
    # Get the reader and writer
    click.echo(f"Input data path:\t{input_data_path}")
    click.echo(f"Output data path:\t{str(output_path)}")
    input_dataset = open_ome_zarr(str(input_data_path))
    stdout_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer):
        input_dataset.print_tree()
    click.echo(f" Input data tree: {stdout_buffer.getvalue()}")

    T, C, _, _, _ = input_dataset.data.shape

    # Write the settings into the metadata if existing
    # TODO: alternatively we can throw all extra arguments as metadata.
    if 'extra_metadata' in kwargs:
        # For each dictionary in the nest
        with open_ome_zarr(output_path, mode='r+') as output_dataset:
            for params_metadata_keys in kwargs['extra_metadata'].keys():
                output_dataset.zattrs['extra_metadata'] = kwargs['extra_metadata']

    # Loop through (T, C), deskewing and writing as we go
    click.echo(f"\nStarting multiprocess pool with {num_processes} processes")

    if input_channel_idx is None or len(input_channel_idx) == 0:
        # If C is not empty, use itertools.product with both ranges
        _, C, _, _, _ = input_dataset.data.shape
        iterable = itertools.product(time_indices, range(C))
        partial_stabilization_over_time_ants = partial(
            stabilization_over_time_ants,
            input_dataset,
            output_path / Path(*input_data_path.parts[-3:]),
            list_of_shifts,
            None,
            None,
        )
    else:
        # If C is empty, use only the range for time_indices
        iterable = itertools.product(time_indices)
        partial_stabilization_over_time_ants = partial(
            stabilization_over_time_ants,
            input_dataset,
            output_path / Path(*input_data_path.parts[-3:]),
            list_of_shifts,
            input_channel_idx,
            output_channel_idx,
            c_idx=0,
        )

    with mp.Pool(num_processes) as p:
        p.starmap(
            partial_stabilization_over_time_ants,
            iterable,
        )
    input_dataset.close()


def stabilization_over_time_ants(
    position: Position,
    output_path: Path,
    list_of_shifts: np.ndarray,
    input_channel_idx: list,
    output_channel_idx: list,
    t_idx: int,
    c_idx: int,
    **kwargs,
) -> None:
    """Load a zyx array from a Position object, apply a transformation and save the result to file"""

    click.echo(f"Processing c={c_idx}, t={t_idx}")
    tx_shifts = numpy_to_ants_transform_zyx(list_of_shifts[t_idx])

    # Process CZYX vs ZYX
    if input_channel_idx is not None:
        czyx_data = position.data.oindex[t_idx, input_channel_idx]
        if not _check_nan_n_zeros(czyx_data):
            for c in input_channel_idx:
                print(f'czyx_data.shape {czyx_data.shape}')
                zyx_data_ants = ants.from_numpy(czyx_data[0])
                registered_zyx = tx_shifts.apply_to_image(
                    zyx_data_ants, reference=zyx_data_ants
                )
                # Write to file
                with open_ome_zarr(output_path, mode="r+") as output_dataset:
                    output_dataset[0].oindex[
                        t_idx, output_channel_idx
                    ] = registered_zyx.numpy()
            click.echo(f"Finished Writing.. t={t_idx}")
        else:
            click.echo(f"Skipping t={t_idx} due to all zeros or nans")
    else:
        zyx_data = position.data.oindex[t_idx, c_idx]
        # Checking if nans or zeros and skip processing
        if not _check_nan_n_zeros(zyx_data):
            zyx_data_ants = ants.from_numpy(zyx_data)
            # Apply transformation
            registered_zyx = tx_shifts.apply_to_image(zyx_data_ants, reference=zyx_data_ants)

            # Write to file
            with open_ome_zarr(output_path, mode="r+") as output_dataset:
                output_dataset[0][t_idx, c_idx] = registered_zyx.numpy()

            click.echo(f"Finished Writing.. c={c_idx}, t={t_idx}")
        else:
            click.echo(f"Skipping c={c_idx}, t={t_idx} due to all zeros or nans")


def model_to_yaml(model, yaml_path: Path) -> None:
    """
    Save a model's dictionary representation to a YAML file.

    Borrowing from recOrder==0.4.0

    Parameters
    ----------
    model : object
        The model object to convert to YAML.
    yaml_path : Path
        The path to the output YAML file.

    Raises
    ------
    TypeError
        If the `model` object does not have a `dict()` method.

    Notes
    -----
    This function converts a model object into a dictionary representation
    using the `dict()` method. It removes any fields with None values before
    writing the dictionary to a YAML file.

    Examples
    --------
    >>> from my_model import MyModel
    >>> model = MyModel()
    >>> model_to_yaml(model, 'model.yaml')

    """
    yaml_path = Path(yaml_path)

    if not hasattr(model, "dict"):
        raise TypeError("The 'model' object does not have a 'dict()' method.")

    model_dict = model.dict()

    # Remove None-valued fields
    clean_model_dict = {key: value for key, value in model_dict.items() if value is not None}

    with open(yaml_path, "w+") as f:
        yaml.dump(clean_model_dict, f, default_flow_style=False, sort_keys=False)


def yaml_to_model(yaml_path: Path, model):
    """
    Load model settings from a YAML file and create a model instance.

    Borrowing from recOrder==0.4.0

    Parameters
    ----------
    yaml_path : Path
        The path to the YAML file containing the model settings.
    model : class
        The model class used to create an instance with the loaded settings.

    Returns
    -------
    object
        An instance of the model class with the loaded settings.

    Raises
    ------
    TypeError
        If the provided model is not a class or does not have a callable constructor.
    FileNotFoundError
        If the YAML file specified by `yaml_path` does not exist.

    Notes
    -----
    This function loads model settings from a YAML file using `yaml.safe_load()`.
    It then creates an instance of the provided `model` class using the loaded settings.

    Examples
    --------
    >>> from my_model import MyModel
    >>> model = yaml_to_model('model.yaml', MyModel)

    """
    yaml_path = Path(yaml_path)

    if not callable(getattr(model, "__init__", None)):
        raise TypeError("The provided model must be a class with a callable constructor.")

    try:
        with open(yaml_path, "r") as file:
            raw_settings = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The YAML file '{yaml_path}' does not exist.")

    return model(**raw_settings)


# TODO: convert all code to use this function from now on
def create_empty_hcs_zarr(
    store_path: Path,
    position_keys: list[Tuple[str]],
    channel_names: list[str],
    shape: Tuple[int],
    chunks: Tuple[int] = None,
    scale: Tuple[float] = (1, 1, 1, 1, 1),
    dtype: DTypeLike = np.float32,
    max_chunk_size_bytes=500e6,
) -> None:
    """
    If the plate does not exist, create an empty zarr plate.
    If the plate exists, append positions and channels if they are not
    already in the plate.
    Parameters
    ----------
    store_path : Path
        hcs plate path
    position_keys : list[Tuple[str]]
        Position keys, will append if not present in the plate.
        e.g. [("A", "1", "0"), ("A", "1", "1")]
    shape : Tuple[int]
    chunks : Tuple[int]
    scale : Tuple[float]
    channel_names : list[str]
        Channel names, will append if not present in metadata.
    dtype : DTypeLike

    Modifying from recOrder
    https://github.com/mehta-lab/recOrder/blob/d31ad910abf84c65ba927e34561f916651cbb3e8/recOrder/cli/utils.py#L12
    """
    MAX_CHUNK_SIZE = max_chunk_size_bytes  # in bytes
    bytes_per_pixel = np.dtype(dtype).itemsize

    # Limiting the chunking to 500MB
    if chunks is None:
        chunk_zyx_shape = list(shape[-3:])
        # chunk_zyx_shape[-3] > 1 ensures while loop will not stall if single
        # XY image is larger than MAX_CHUNK_SIZE
        while (
            chunk_zyx_shape[-3] > 1
            and np.prod(chunk_zyx_shape) * bytes_per_pixel > MAX_CHUNK_SIZE
        ):
            chunk_zyx_shape[-3] = np.ceil(chunk_zyx_shape[-3] / 2).astype(int)
        chunk_zyx_shape = tuple(chunk_zyx_shape)

        chunks = 2 * (1,) + chunk_zyx_shape

    # Create plate
    output_plate = open_ome_zarr(
        str(store_path), layout="hcs", mode="a", channel_names=channel_names
    )

    # Create positions
    for position_key in position_keys:
        position_key_string = "/".join(position_key)
        # Check if position is already in the store, if not create it
        if position_key_string not in output_plate.zgroup:
            position = output_plate.create_position(*position_key)
            _ = position.create_zeros(
                name="0",
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                transform=[TransformationMeta(type="scale", scale=scale)],
            )
        else:
            position = output_plate[position_key_string]

    # Check if channel_names are already in the store, if not append them
    for channel_name in channel_names:
        # Read channel names directly from metadata to avoid race conditions
        metadata_channel_names = [
            channel.label for channel in position.metadata.omero.channels
        ]
        if channel_name not in metadata_channel_names:
            position.append_channel(channel_name, resize_arrays=True)


def _is_nested(lst):
    return any(isinstance(i, list) for i in lst) or any(isinstance(i, str) for i in lst)


def _check_nan_n_zeros(input_array):
    """
    Checks if any of the channels are all zeros or nans and returns true
    """
    if len(input_array.shape) == 3:
        # Check if all the values are zeros or nans
        if np.all(input_array == 0) or np.all(np.isnan(input_array)):
            # Return true
            return True
    elif len(input_array.shape) == 4:
        # Get the number of channels
        num_channels = input_array.shape[0]
        # Loop through the channels
        for c in range(num_channels):
            # Get the channel
            zyx_array = input_array[c, :, :, :]

            # Check if all the values are zeros or nans
            if np.all(zyx_array == 0) or np.all(np.isnan(zyx_array)):
                # Return true
                return True
    else:
        raise ValueError("Input array must be 3D or 4D")

    # Return false
    return False


# NOTE WIP
def apply_transform_to_zyx_and_save_v2(
    func,
    position: Position,
    output_path: Path,
    input_channel_indices: list[int],
    output_channel_indices: list[int],
    t_idx: int,
    c_idx: int = None,
    **kwargs,
) -> None:
    """Load a zyx array from a Position object, apply a transformation to CZYX or ZYX and save the result to file"""
    click.echo(f"Processing c={c_idx}, t={t_idx}")

    # TODO: temporary fix to slumkit issue
    if _is_nested(input_channel_indices):
        # print(f'input_channel_indices: {input_channel_indices}')
        input_channel_indices = [int(x) for x in input_channel_indices if x.isdigit()]
    if _is_nested(output_channel_indices):
        # print(f'input_channel_indices: {output_channel_indices}')
        output_channel_indices = [int(x) for x in output_channel_indices if x.isdigit()]
    click.echo(f'input_channel_indices: {input_channel_indices}')

    # Process CZYX vs ZYX
    if input_channel_indices is not None:
        czyx_data = position.data.oindex[t_idx, input_channel_indices]
        if not _check_nan_n_zeros(czyx_data):
            transformed_czyx = func(czyx_data, **kwargs)
            # Write to file
            with open_ome_zarr(output_path, mode="r+") as output_dataset:
                output_dataset[0].oindex[t_idx, output_channel_indices] = transformed_czyx
            click.echo(f"Finished Writing.. t={t_idx}")
        else:
            click.echo(f"Skipping t={t_idx} due to all zeros or nans")
    else:
        zyx_data = position.data.oindex[t_idx, c_idx]
        # Checking if nans or zeros and skip processing
        if not _check_nan_n_zeros(zyx_data):
            # Apply transformation
            transformed_zyx = func(zyx_data, **kwargs)

            # Write to file
            with open_ome_zarr(output_path, mode="r+") as output_dataset:
                output_dataset[0][t_idx, c_idx] = transformed_zyx

            click.echo(f"Finished Writing.. c={c_idx}, t={t_idx}")
        else:
            click.echo(f"Skipping c={c_idx}, t={t_idx} due to all zeros or nans")


# TODO: modifiy how we get the time and channesl like recOrder (isinstance(input, list) or instance(input,int) or all)
def process_single_position_v2(
    func,
    input_data_path: Path,
    output_path: Path,
    time_indices: list = [0],
    input_channel_idx: list = [],
    output_channel_idx: list = [],
    num_processes: int = mp.cpu_count(),
    **kwargs,
) -> None:
    """Register a single position with multiprocessing parallelization over T and C"""
    # Function to be applied
    click.echo(f"Function to be applied: \t{func}")

    # Get the reader and writer
    click.echo(f"Input data path:\t{input_data_path}")
    click.echo(f"Output data path:\t{str(output_path)}")
    input_dataset = open_ome_zarr(str(input_data_path))
    stdout_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer):
        input_dataset.print_tree()
    click.echo(f" Input data tree: {stdout_buffer.getvalue()}")

    # Find time indices
    if time_indices == "all":
        time_indices = range(input_dataset.data.shape[0])
    elif isinstance(time_indices, list):
        time_indices = time_indices

    # Check for invalid times
    time_ubound = input_dataset.data.shape[0] - 1
    if np.max(time_indices) > time_ubound:
        raise ValueError(
            f"time_indices = {time_indices} includes a time index beyond the maximum index of the dataset = {time_ubound}"
        )

    # Check the arguments for the function
    all_func_params = inspect.signature(func).parameters.keys()
    # Extract the relevant kwargs for the function 'func'
    func_args = {}
    non_func_args = {}

    for k, v in kwargs.items():
        if k in all_func_params:
            func_args[k] = v
        else:
            non_func_args[k] = v

    # Write the settings into the metadata if existing
    if 'extra_metadata' in non_func_args:
        # For each dictionary in the nest
        with open_ome_zarr(output_path, mode='r+') as output_dataset:
            for params_metadata_keys in kwargs['extra_metadata'].keys():
                output_dataset.zattrs['extra_metadata'] = non_func_args['extra_metadata']

    # Loop through (T, C), deskewing and writing as we go
    if input_channel_idx is None or len(input_channel_idx) == 0:
        # If C is not empty, use itertools.product with both ranges
        _, C, _, _, _ = input_dataset.data.shape
        iterable = itertools.product(time_indices, range(C))
        partial_apply_transform_to_zyx_and_save = partial(
            apply_transform_to_zyx_and_save_v2,
            func,
            input_dataset,
            output_path / Path(*input_data_path.parts[-3:]),
            input_channel_indices=None,
            **func_args,
        )
    else:
        # If C is empty, use only the range for time_indices
        iterable = itertools.product(time_indices)
        partial_apply_transform_to_zyx_and_save = partial(
            apply_transform_to_zyx_and_save_v2,
            func,
            input_dataset,
            output_path / Path(*input_data_path.parts[-3:]),
            input_channel_idx,
            output_channel_idx,
            c_idx=0,
            **func_args,
        )

        click.echo(f"\nStarting multiprocess pool with {num_processes} processes")
        with mp.Pool(num_processes) as p:
            p.starmap(
                partial_apply_transform_to_zyx_and_save,
                iterable,
            )
