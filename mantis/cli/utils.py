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
import numpy as np
import scipy.ndimage as ndi
import torch
import yaml

from iohub.ngff import Position, open_ome_zarr
from iohub.ngff_meta import TransformationMeta
from numpy.typing import DTypeLike
from tqdm import tqdm


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


def apply_transform_to_zyx_and_save(
    func, position: Position, output_path: Path, t_idx: int, c_idx: int, **kwargs
) -> None:
    """Load a zyx array from a Position object, apply a transformation and save the result to file"""
    click.echo(f"Processing c={c_idx}, t={t_idx}")
    zyx_data = position[0][t_idx, c_idx]

    # Apply transformation
    registered_zyx = func(zyx_data, **kwargs)

    # Write to file
    with open_ome_zarr(output_path, mode="r+") as output_dataset:
        output_dataset[0][t_idx, c_idx] = registered_zyx

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
                apply_transform_to_zyx_and_save,
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


def rotate_affine(start_shape_zyx, angle=0.0, end_shape_zyx=None):
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


def ants_affine_transform(
    zyx_data,
    ants_transform_file_list,
    output_shape_zyx,
):
    # Convert the NaN to 0
    zyx_data = np.nan_to_num(zyx_data, nan=0)

    # The output has to be a ANTImage Object
    empty_target_array = np.zeros((output_shape_zyx), dtype=np.float32)
    target_zyx_ants = ants.from_numpy(empty_target_array)

    # NOTE:Matrices the order matters!
    matrices = []
    for mat in ants_transform_file_list:
        matrices.append(ants.read_transform(mat))
    ants_composed_matrix = ants.compose_ants_transforms(matrices)

    zyx_data_ants = ants.from_numpy(zyx_data.astype(np.float32))
    registered_zyx = ants_composed_matrix.apply_to_image(
        zyx_data_ants, reference=target_zyx_ants
    )
    return registered_zyx.numpy()


def copy_n_paste(zyx_data, zyx_slicing_params: list):
    """Load a zyx array and slice"""
    zyx_data = np.nan_to_num(zyx_data, nan=0)
    zyx_data_sliced = zyx_data[
        zyx_slicing_params[0],
        zyx_slicing_params[1],
        zyx_slicing_params[2],
    ]
    return zyx_data_sliced


def find_lir_slicing_params(
    input_zyx_shape, target_zyx_shape, registration_mat_optimized, registration_mat_manual
):
    print(f'Starting Largest interior rectangle (LIR) search')

    # Make dummy volumes
    img1 = np.ones(tuple(input_zyx_shape), dtype=np.float32)
    img2 = np.ones(tuple(target_zyx_shape), dtype=np.float32)

    # Load the matrices
    ants_transform_file_list = [registration_mat_optimized, registration_mat_manual]
    matrices = []
    for mat in ants_transform_file_list:
        matrices.append(ants.read_transform(mat))
    ants_composed_matrix = ants.compose_ants_transforms(matrices)

    # Conver to ants objects
    target_zyx_ants = ants.from_numpy(img2)
    zyx_data_ants = ants.from_numpy(img1.astype(np.float32))

    # Apply affine
    registered_zyx = ants_composed_matrix.apply_to_image(
        zyx_data_ants, reference=target_zyx_ants
    )
    registered_zyx_bool = registered_zyx.numpy().copy()
    registered_zyx_bool = registered_zyx_bool > 0
    # NOTE: we use the center of the volume as reference
    rectangle_coords_yx = lir.lir(registered_zyx_bool[registered_zyx.shape[0] // 2])

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

    # Find the overlap in Z
    registered_zx = registered_zyx.numpy()
    registered_zx = registered_zx.transpose((2, 0, 1)) > 0
    rectangle_coords_zx = lir.lir(registered_zx[registered_zyx.shape[0] // 2].copy())
    x = rectangle_coords_zx[0]
    y = rectangle_coords_zx[1]
    width = rectangle_coords_zx[2]
    height = rectangle_coords_zx[3]
    corner1_zx = (x, y)  # Bottom-left corner
    corner2_zx = (x + width, y)  # Bottom-right corner
    corner3_zx = (x + width, y + height)  # Top-right corner
    corner4_zx = (x, y + height)  # Top-left corner
    rectangle_zx = np.array((corner1_zx, corner2_zx, corner3_zx, corner4_zx))
    Z_slice = slice(rectangle_zx.min(axis=0)[1], rectangle_zx.max(axis=0)[1])
    print(f'Slicing parameters Z:{Z_slice}, Y:{Y_slice}, X:{X_slice}')
    return (Z_slice, Y_slice, X_slice)


def append_channels(input_data_path: Path, target_data_path: Path):
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


def numpy_to_ants_transform_zyx(T_numpy):
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
    T_ants = ants.new_ants_transform()
    T_ants.set_parameters(T_ants_style)

    return T_ants


def ants_to_numpy_transform_zyx(T_ants):
    """
    Homogeneous 3D transformation matrix from ants to numpy

    Parameters
    ----------
    T_ants : Ants transfromation matrix object

    Returns
    -------
    Converted Ants to numpy array

    """
    T_ants_array = np.reshape(T_ants.parameters, (4, 3))
    T_numpy = np.zeros((4, 4))
    # Extract the first 3x3 part
    T_numpy[:3, :3] = T_ants_array[:3, :3]
    # Add the last column from the original matrix as a new column
    T_numpy[:3, -1] = T_ants_array[3, :]
    # Set the last row to [0, 0, 0, 1]
    T_numpy[-1, -1] = 1
    return T_numpy


def apply_stabilization_over_time_ants(
    list_of_shifts_ants_style: list,
    input_data_path: Path,
    output_path: Path,
    num_processes: int = mp.cpu_count(),
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
    with mp.Pool(num_processes) as p:
        p.starmap(
            partial(
                stabilization_over_time_ants,
                input_dataset,
                str(output_path),
                list_of_shifts_ants_style,
            ),
            itertools.product(range(T), range(C)),
        )
    input_dataset.close()


def stabilization_over_time_ants(
    position: Position,
    output_path: Path,
    list_of_shifts,
    t_idx: int,
    c_idx: int,
    **kwargs,
) -> None:
    """Load a zyx array from a Position object, apply a transformation and save the result to file"""
    click.echo(f"Processing c={c_idx}, t={t_idx}")

    zyx_data = position[0][t_idx, c_idx].astype(np.float32)
    # Convert nans to 0
    zyx_data = np.nan_to_num(zyx_data, nan=0)
    zyx_data_ants = ants.from_numpy(zyx_data)

    tx_composition = ants.new_ants_transform()
    tx_composition.set_parameters(list_of_shifts[t_idx])

    # Apply transformation
    registered_zyx = tx_composition.apply_to_image(zyx_data_ants, reference=zyx_data_ants)

    # Write to file
    with open_ome_zarr(output_path, mode="r+") as output_dataset:
        output_dataset[0][t_idx, c_idx] = registered_zyx.numpy()

    click.echo(f"Finished Writing.. c={c_idx}, t={t_idx}")


def model_to_yaml(model, yaml_path: Path) -> None:
    """
    Save a model's dictionary representation to a YAML file.

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
