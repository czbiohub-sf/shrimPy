import numpy as np
import torch

from monai.transforms.spatial.array import Affine


def _average_n_slices(data, average_window_width=1):
    """Average an array over its first axis

    Parameters
    ----------
    data : np.array

    average_window_width : int, optional
        Averaging window applied to the first axis.

    Returns
    -------
    data_averaged : np.array

    """
    # If first dimension isn't divisible by average_window_width, pad it
    remainder = data.shape[0] % average_window_width
    if remainder > 0:
        padding_width = average_window_width - remainder
        data = np.pad(data, [(0, padding_width)] + [(0, 0)] * (data.ndim - 1), mode='edge')

    # Reshape then average over the first dimension
    new_shape = (data.shape[0] // average_window_width, average_window_width) + data.shape[1:]
    data_averaged = np.mean(data.reshape(new_shape), axis=1)

    return data_averaged


def _get_averaged_shape(deskewed_data_shape: tuple, average_window_width: int) -> tuple:
    """
    Compute the shape of the data returned from `_average_n_slices` function.

    Parameters
    ----------
    deskewed_data_shape : tuple
        Shape of the original data before averaging.

    average_window_width : int
        Averaging window applied to the first axis.

    Returns
    -------
    averaged_shape : tuple
        Shape of the data returned from `_average_n_slices`.
    """
    averaged_shape = (
        int(np.ceil(deskewed_data_shape[0] / average_window_width)),
    ) + deskewed_data_shape[1:]
    return averaged_shape


def _get_transform_matrix(ls_angle_deg: float, px_to_scan_ratio: float):
    """
    Compute affine transformation matrix used to deskew data.

    Parameters
    ----------
    ls_angle_deg : float
    px_to_scan_ratio : float
    keep_overhang : bool

    Returns
    -------
    matrix : np.array
        Affine transformation matrix.
    """
    ct = np.cos(ls_angle_deg * np.pi / 180)

    matrix = np.array(
        [
            [
                -px_to_scan_ratio * ct,
                0,
                px_to_scan_ratio,
                0,
            ],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ]
    )

    return matrix


def get_deskewed_data_shape(
    raw_data_shape: tuple,
    ls_angle_deg: float,
    px_to_scan_ratio: float,
    keep_overhang: bool,
    average_n_slices: int = 1,
    pixel_size_um: float = 1,
):
    """Get the shape of the deskewed data set and its voxel size
    Parameters
    ----------
    raw_data_shape : tuple
        Shape of the raw data, must be len = 3
    ls_angle_deg : float
        Angle of the light sheet relative to the optical axis, in degrees
    px_to_scan_ratio : float
        Ratio of the pixel size to light sheet scan step
    keep_overhang : bool, optional
        If true, the shape of the whole volume within the tilted parallelepiped
        will be returned
        If false, the shape of the deskewed volume within a cuboid region will
        be returned
    average_n_slices : int, optional
        after deskewing, averages every n slices (default = 1 applies no averaging)
    pixel_size_um : float, optional
        Pixel size in micrometers. If not provided, a default value of 1 will be
        used and the returned voxel size will represent a voxel scale
    Returns
    -------
    output_shape : tuple
        Output shape of the deskewed data in ZYX order
    voxel_size : tuple
        Size of the deskewed voxels in micrometers. If the default
        pixel_size_um = 1 is used this parameter will represent the voxel scale
    """

    # Trig
    theta = ls_angle_deg * np.pi / 180
    st = np.sin(theta)
    ct = np.cos(theta)

    # Prepare transforms
    Z, Y, X = raw_data_shape

    if keep_overhang:
        Xp = int(np.ceil((Z / px_to_scan_ratio) + (Y * ct)))
    else:
        Xp = int(np.ceil((Z / px_to_scan_ratio) - (Y * ct)))

    output_shape = (Y, X, Xp)
    voxel_size = (average_n_slices * st * pixel_size_um, pixel_size_um, pixel_size_um)

    averaged_output_shape = _get_averaged_shape(output_shape, average_n_slices)

    return averaged_output_shape, voxel_size


def deskew_data(
    raw_data: np.ndarray,
    ls_angle_deg: float,
    px_to_scan_ratio: float,
    keep_overhang: bool,
    average_n_slices: int = 1,
    device='cpu',
) -> np.ndarray:
    """Deskews fluorescence data from the mantis microscope
    Parameters
    ----------
    raw_data : NDArray with ndim == 3
        raw data from the mantis microscope
        - axis 0 corresponds to the scanning axis
        - axis 1 corresponds to the "tilted" axis
        - axis 2 corresponds to the axis in the plane of the coverslip
    ls_angle_deg : float
        angle of light sheet with respect to the optical axis in degrees
    px_to_scan_ratio : float
        (pixel spacing / scan spacing) in object space
        e.g. if camera pixels = 6.5 um and mag = 1.4*40, then the pixel spacing
        is 6.5/(1.4*40) = 0.116 um. If the scan spacing is 0.3 um, then
        px_to_scan_ratio = 0.116 / 0.3 = 0.386
    keep_overhang : bool
        If true, compute the whole volume within the tilted parallelepiped.
        If false, only compute the deskewed volume within a cuboid region.
    average_n_slices : int, optional
        after deskewing, averages every n slices (default = 1 applies no averaging)
    device : str, optional
        torch device to use for computation. Default is 'cpu'.
    Returns
    -------
    deskewed_data : NDArray with ndim == 3
        axis 0 is the Z axis, normal to the coverslip
        axis 1 is the Y axis, input axis 2 in the plane of the coverslip
        axis 2 is the X axis, the scanning axis
    """
    # Prepare transforms
    matrix = _get_transform_matrix(
        ls_angle_deg,
        px_to_scan_ratio,
    )

    output_shape, _ = get_deskewed_data_shape(
        raw_data.shape, ls_angle_deg, px_to_scan_ratio, keep_overhang
    )

    # convert to tensor on GPU
    # convert raw_data to int32 if it is uint16
    raw_data_tensor = torch.from_numpy(raw_data.astype(np.float32)).to(device)

    # Returns callable
    affine_func = Affine(affine=matrix, padding_mode="zeros", image_only=True)

    # affine_func accepts CZYX array, so for ZYX input we need [None] and for ZYX output we need [0]
    deskewed_data = affine_func(
        raw_data_tensor[None], mode="bilinear", spatial_size=output_shape
    )[0]

    # to numpy array on CPU
    deskewed_data = deskewed_data.cpu().numpy()

    # Apply averaging
    averaged_deskewed_data = _average_n_slices(
        deskewed_data, average_window_width=average_n_slices
    )
    return averaged_deskewed_data
