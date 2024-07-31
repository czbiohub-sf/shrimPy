from typing import Tuple

import ants
import largestinteriorrectangle as lir
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage


def get_3D_rescaling_matrix(start_shape_zyx, scaling_factor_zyx=(1, 1, 1), end_shape_zyx=None):
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


def get_3D_rotation_matrix(
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
    return rotation_matrix


def convert_transform_to_ants(T_numpy: np.ndarray):
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


# def numpy_to_ants_transform_czyx(T_numpy: np.ndarray):
#     """Homogeneous 3D transformation matrix from numpy to ants

#     Parameters
#     ----------
#     numpy_transform :4x4 homogenous matrix

#     Returns
#     -------
#     Ants transformation matrix object
#     """
#     assert T_numpy.shape == (5, 5)
#     shape = T_numpy.shape
#     T_ants_style = T_numpy[:, :-1].ravel()
#     T_ants_style[-shape[0] + 1 :] = T_numpy[-shape[0] : -1, -1]
#     T_ants = ants.new_ants_transform(
#         transform_type='AffineTransform',
#     )
#     T_ants.set_parameters(T_ants_style)

#     return T_ants


def convert_transform_to_numpy(T_ants):
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


def apply_affine_transform(
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

    Z, Y, X = output_shape_zyx
    if crop_output_slicing is not None:
        Z_slice, Y_slice, X_slice = crop_output_slicing
        Z = Z_slice.stop - Z_slice.start
        Y = Y_slice.stop - Y_slice.start
        X = X_slice.stop - X_slice.start

    # TODO: based on the signature of this function, it should not be called on 4D array
    if zyx_data.ndim == 4:
        registered_czyx = np.zeros((zyx_data.shape[0], Z, Y, X), dtype=np.float32)
        for c in range(zyx_data.shape[0]):
            registered_czyx[c] = apply_affine_transform(
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

            T_ants = convert_transform_to_ants(matrix)

            zyx_data_ants = ants.from_numpy(zyx_data.astype(np.float32))
            registered_zyx = T_ants.apply_to_image(
                zyx_data_ants, reference=target_zyx_ants
            ).numpy()

        elif method == 'scipy':
            registered_zyx = scipy.ndimage.affine_transform(zyx_data, matrix, output_shape_zyx)

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

    # NOTE: this method assumes the center of the image is representative of the center of the object to estimate the LIR in Z
    # Find the lir Z using ZX
    zyx_shape = registered_zyx.shape
    registered_zxy_bool = registered_zyx.transpose((2, 0, 1)) > 0
    # Take middle X-slice to find the LIR for Z.
    registered_zx_bool = registered_zxy_bool[zyx_shape[-1] // 2].copy()
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


def find_overlapping_volume(
    input_zyx_shape: Tuple,
    target_zyx_shape: Tuple,
    transformation_matrix: np.ndarray,
    method: str = 'LIR',
    plot: bool = False,
) -> Tuple:
    """
    Find the overlapping rectangular volume after registration of two 3D datasets

    Parameters
    ----------
    input_zyx_shape : Tuple
        shape of input array
    target_zyx_shape : Tuple
        shape of target array
    transformation_matrix : np.ndarray
        affine transformation matrix
    method : str, optional
        method of finding the overlapping volume, by default 'LIR'

    Returns
    -------
    Tuple
        ZYX slices of the overlapping volume after registration

    """

    # Make dummy volumes
    img1 = np.ones(tuple(input_zyx_shape), dtype=np.float32)
    img2 = np.ones(tuple(target_zyx_shape), dtype=np.float32)

    # Conver to ants objects
    target_zyx_ants = ants.from_numpy(img2.astype(np.float32))
    zyx_data_ants = ants.from_numpy(img1.astype(np.float32))

    ants_composed_matrix = convert_transform_to_ants(transformation_matrix)

    # Apply affine
    registered_zyx = ants_composed_matrix.apply_to_image(
        zyx_data_ants, reference=target_zyx_ants
    )

    if method == 'LIR':
        print('Starting Largest interior rectangle (LIR) search')
        Z_slice, Y_slice, X_slice = find_lir(registered_zyx.numpy(), plot=plot)
    else:
        raise ValueError(f'Unknown method {method}')

    return (Z_slice, Y_slice, X_slice)
