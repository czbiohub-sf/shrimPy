# %%

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import skimage

from iohub import open_ome_zarr
from numpy.typing import ArrayLike
from skimage.exposure import rescale_intensity
from skimage.feature import match_template
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.registration import phase_cross_correlation

from mantis import logger

# import cv2


# TODO: Make toy datasets for testing
# TODO: multiotsu should have an offset variable from the center of the image (X,Y)
# TODO: Check why PCC is not working
# TODO: write test functions
# TODO: consider splitting this file into two


# %%


def calc_weighted_center(labeled_im):
    """calculates weighted centroid based on the area of the regions

    Parameters
    ----------
    labeled_im : ndarray
        labeled image
    """
    regions = sorted(regionprops(labeled_im), key=lambda r: r.area)
    n_regions = len(regions)
    centers = []
    areas = []
    for i in range(n_regions):
        centroid = regions[i].centroid
        centers.append(centroid)
        area = regions[i].area
        areas.append(area)
    areas_norm = np.array(areas) / np.sum(areas)
    centers = np.array(centers)
    center_weighted = np.zeros(3)
    for j in range(3):
        center_weighted[j] = np.sum(centers[:, j] * areas_norm)

    return center_weighted


def multiotsu_centroid(moving: ArrayLike, reference: ArrayLike) -> list:
    """
    Multiotsu centroid method for finding the shift between two volumes (ZYX)

    Parameters
    ----------
    moving : ndarray
        moving stack ZYX
    reference : ndarray
        reference image stack ZYX

    Returns
    -------
    shifts : list
        list of shifts in z, y, x order
    """
    # Process moving image
    moving = rescale_intensity(moving, in_range='image', out_range=(0, 1.0))
    stack_blur = gaussian(moving, sigma=5.0)
    thresh = skimage.filters.threshold_multiotsu(stack_blur)
    moving = stack_blur > thresh[0]
    moving = label(moving)
    # Process reference image
    reference = rescale_intensity(reference, in_range='image', out_range=(0, 1.0))
    stack_blur = gaussian(reference, sigma=5.0)
    thresh = skimage.filters.threshold_multiotsu(stack_blur)
    reference = stack_blur > thresh[0]
    reference = label(reference)

    # Get the centroids
    moving_center = calc_weighted_center(moving)
    target_center = calc_weighted_center(reference)

    # Find the shifts
    shifts = moving_center - target_center

    logger.debug(
        'moving_center (z,y,x): %f,%f,%f', moving_center[0], moving_center[1], moving_center[2]
    )
    logger.debug(
        'target_center (z,y,x): %f,%f,%f', target_center[0], target_center[1], target_center[2]
    )
    logger.debug('shifts (z,y,x): %f,%f,%f', shifts[0], shifts[1], shifts[2])

    return shifts


_AUTOFOCUS_METHODS = {
    'pcc': phase_cross_correlation,
    'tm': match_template,
    'multiotsu': multiotsu_centroid,
}


# %%
def main():
    print('AUTOTRACKER')
    """
    Toy dataset
    translations = [
        (0, 0, 0),  # Shift for timepoint 0
        (5, -80, 80),  # Shift for timepoint 1
        (9, -50, -50),  # Shift for timepoint 2
        (-5, 30, -60),  # Shift for timepoint 3
        (0, 30, -80),  # Shift for timepoint 4
    ]
    """
    # %%
    input_data_path = (
        '/home/eduardo.hirata/repos/mantis/mantis/tests/x-ed/toy_translate.zarr/0/0/0'
    )
    dataset = open_ome_zarr(input_data_path)
    T, C, Z, Y, X = dataset.data.shape
    # print(channel_names := dataset.channel_names)
    # autofocus_method = 'multiotsu'
    # xy_dapening = (10, 10)

    c_idx = 2
    data_t0 = dataset.data[0, c_idx]
    data_t1 = dataset.data[1, c_idx]
    data_t2 = dataset.data[2, c_idx]
    data_t3 = dataset.data[3, c_idx]

    # subplot
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(data_t0[10])
    ax[1].imshow(data_t1[10])
    ax[2].imshow(data_t2[10])
    plt.show()

    shift_0 = multiotsu_centroid(data_t0, data_t0)
    shift_1 = multiotsu_centroid(data_t1, data_t0)
    shift_2 = multiotsu_centroid(data_t2, data_t0)
    shift_3 = multiotsu_centroid(data_t3, data_t0)

    shifts = [shift_0, shift_1, shift_2, shift_3]

    translations = [
        (0, 0, 0),  # Shift for timepoint 0
        (5, -80, 80),  # Shift for timepoint 1
        (9, -50, -50),  # Shift for timepoint 2
        (-5, 30, -60),  # Shift for timepoint 3
    ]  # Compare shifts with expected translations

    tolerance = (5, 8, 8)  # Define your tolerance level
    for i, (calculated, expected) in enumerate(zip(shifts, translations)):
        is_similar = np.allclose(calculated, expected, atol=tolerance)
        print(
            f'Timepoint {i+1} shift: {calculated}, Expected: {expected}, Similar: {is_similar}'
        )


# %%


def estimate_shift(
    reference_array: ArrayLike,
    moving_array: ArrayLike,
    autofocus_method: Literal['pcc', 'tm', 'multiotsu'],
    xy_dapening: tuple[int] = None,
    **kwargs,
) -> np.ndarray:
    autofocus_method_func = _AUTOFOCUS_METHODS.get(autofocus_method)

    if not autofocus_method_func:
        raise ValueError(f'Unknown autofocus method: {autofocus_method}')

    shifts = autofocus_method_func(**kwargs)

    return shifts


def get_shift_centroid(autofocus_method, im_moving, im_ref):
    """finds the centroid of the images and returns the shift
    between the two volumes in z, y, and x

    Parameters
    ----------
    im_moving : ndarray
        moving image volume
    im_ref : ndarray
        reference image volume

    Returns
    -------
    shift : ndarray
        array of shift in z, y, x order
    """
    Z, Y, X = im_moving.shape
    if autofocus_method == 'multiotsu':
        centroid_moving = multiotsu_centroid(im_moving)
        centroid_ref = [int(Z / 2), int((Y) / 2), int((X) / 2)]
    if autofocus_method == 'com':
        centroid_moving = calc_com(im_moving)
        centroid_ref = [int(Z / 2), int((Y) / 2), int((X) / 2)]

    return centroid_moving - centroid_ref


def calc_com(im):
    """ """
    stack = np.clip(im - np.percentile(im, 10), a_min=0, a_max=None)
    Z, Y, X = im.shape
    center = [int(Z // 2), int(Y // 2), int(X // 2)]
    num_z = 0
    num_y = 0
    num_x = 0
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                val = stack[z, y, x]
                num_z = num_z + (z - center[0]) * val
                num_y = num_y + (y - center[1]) * val
                num_x = num_x + (x - center[2]) * val
    centroid = [
        int(num_z / np.sum(stack)),
        int(num_y / np.sum(stack)),
        int(num_x / np.sum(stack)),
    ]
    centroid = np.array(centroid) + np.array(center)
    print(f'centroid: {centroid}')
    return centroid


# %%
# %%
if __name__ == "__main__":
    main()
