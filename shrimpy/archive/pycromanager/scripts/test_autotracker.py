# %%
from typing import Callable, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import skimage

from iohub import open_ome_zarr
from numpy.typing import ArrayLike
from scipy.fftpack import next_fast_len
from skimage.exposure import rescale_intensity
from skimage.feature import match_template
from skimage.filters import gaussian
from skimage.measure import label, regionprops

from mantis import logger

# FIXME fix the dependencies so that we can install and import dexpv2
# from dexpv2.crosscorr import phase_cross_corr
# from dexpv2.utils import center_crop, pad_to_shape, to_cpu

# TODO: Make toy datasets for testing
# TODO: multiotsu should have an offset variable from the center of the image (X,Y)
# TODO: Check why PCC is not working
# TODO: write test functions
# TODO: consider splitting this file into two


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


def multiotsu_centroid(
    ref_img: ArrayLike,
    mov_img: ArrayLike,
) -> list:
    """
    Computes the translation shifts using a multiotsu threshold approach by finding the centroid of the regions

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
    mov_img = rescale_intensity(mov_img, in_range='image', out_range=(0, 1.0))
    stack_blur = gaussian(mov_img, sigma=5.0)
    thresh = skimage.filters.threshold_multiotsu(stack_blur)
    mov_img = stack_blur > thresh[0]
    mov_img = label(mov_img)
    # Process reference image
    ref_img = rescale_intensity(ref_img, in_range='image', out_range=(0, 1.0))
    stack_blur = gaussian(ref_img, sigma=5.0)
    thresh = skimage.filters.threshold_multiotsu(stack_blur)
    ref_img = stack_blur > thresh[0]
    ref_img = label(ref_img)

    # Get the centroids
    moving_center = calc_weighted_center(mov_img)
    target_center = calc_weighted_center(ref_img)

    # Find the shifts
    shifts = moving_center - target_center

    logger.debug(
        'moving_center (z,y,x): %f,%f,%f',
        moving_center[0],
        moving_center[1],
        moving_center[2],
    )
    logger.debug(
        'target_center (z,y,x): %f,%f,%f',
        target_center[0],
        target_center[1],
        target_center[2],
    )
    logger.debug('shifts (z,y,x): %f,%f,%f', shifts[0], shifts[1], shifts[2])

    return shifts


def template_matching(ref_img, moving_img, template_slicing_zyx):
    """
    Uses template matching to determine shift between two image stacks.

    Parameters:
    - ref_img: Reference 3D image stack (numpy array).
    - moving_img: Moving 3D image stack (numpy array) to be aligned with the reference.
    - template_slicing_zyx: Tuple or list of slice objects defining the region to be used as the template.

    Returns:
    - shift: The shift (displacement) needed to align moving_img with ref_img (numpy array).
    """
    template = ref_img[template_slicing_zyx]

    result = match_template(moving_img, template)
    zyx_1 = np.unravel_index(np.argmax(result), result.shape)

    # Calculate the shift based on template slicing coordinates and match result
    # Subtracting the coordinates of the

    shift = np.array(zyx_1) - np.array([s.start for s in template_slicing_zyx])

    return shift


def to_cpu(arr: ArrayLike) -> ArrayLike:
    """
    Borrowing from Jordao dexpv2.crosscorr https://github.com/royerlab/dexpv2
    Moves array to cpu, if it's already there nothing is done.

    """
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    elif hasattr(arr, "get"):
        arr = arr.get()
    return arr


def _match_shape(img: ArrayLike, shape: Tuple[int, ...]) -> ArrayLike:
    """
    Borrowing from Jordao dexpv2.crosscorr https://github.com/royerlab/dexpv2
    Pad or crop array to match provided shape.
    """

    if np.any(shape > img.shape):
        padded_shape = np.maximum(img.shape, shape)
        img = pad_to_shape(img, padded_shape, mode="reflect")

    if np.any(shape < img.shape):
        img = center_crop(img, shape)

    return img


def center_crop(arr: ArrayLike, shape: Tuple[int, ...]) -> ArrayLike:
    """
    Borrowing from Jordao dexpv2.crosscorr https://github.com/royerlab/dexpv2
    Crops the center of `arr`
    """
    assert arr.ndim == len(shape)

    starts = tuple((cur_s - s) // 2 for cur_s, s in zip(arr.shape, shape))

    assert all(s >= 0 for s in starts)

    slicing = tuple(slice(s, s + d) for s, d in zip(starts, shape))

    logger.info(
        f"center crop: input shape {arr.shape}, output shape {shape}, slicing {slicing}"
    )

    return arr[slicing]


def pad_to_shape(arr: ArrayLike, shape: Tuple[int, ...], mode: str, **kwargs) -> ArrayLike:
    """
    Borrowing from Jordao dexpv2.crosscorr https://github.com/royerlab/dexpv2
    Pads array to shape.

    Parameters
    ----------
    arr : ArrayLike
        Input array.
    shape : Tuple[int]
        Output shape.
    mode : str
        Padding mode (see np.pad).

    Returns
    -------
    ArrayLike
        Padded array.
    """
    assert arr.ndim == len(shape)

    dif = tuple(s - a for s, a in zip(shape, arr.shape))
    assert all(d >= 0 for d in dif)

    pad_width = [[s // 2, s - s // 2] for s in dif]

    logger.info(f"padding: input shape {arr.shape}, output shape {shape}, padding {pad_width}")

    return np.pad(arr, pad_width=pad_width, mode=mode, **kwargs)


def phase_cross_corr(
    ref_img: ArrayLike,
    mov_img: ArrayLike,
    maximum_shift: float = 1.0,
    to_device: Callable[[ArrayLike], ArrayLike] = lambda x: x,
    transform: Optional[Callable[[ArrayLike], ArrayLike]] = np.log1p,
) -> Tuple[int, ...]:
    """
    Borrowing from Jordao dexpv2.crosscorr https://github.com/royerlab/dexpv2

    Computes translation shift using arg. maximum of phase cross correlation.
    Input are padded or cropped for fast FFT computation assuming a maximum translation shift.

    Parameters
    ----------
    ref_img : ArrayLike
        Reference image.
    mov_img : ArrayLike
        Moved image.
    maximum_shift : float, optional
        Maximum location shift normalized by axis size, by default 1.0

    Returns
    -------
    Tuple[int, ...]
        Shift between reference and moved image.
    """
    shape = tuple(
        cast(int, next_fast_len(int(max(s1, s2) * maximum_shift)))
        for s1, s2 in zip(ref_img.shape, mov_img.shape)
    )

    logger.info(
        f"phase cross corr. fft shape of {shape} for arrays of shape {ref_img.shape} and {mov_img.shape} "
        f"with maximum shift of {maximum_shift}"
    )

    ref_img = _match_shape(ref_img, shape)
    mov_img = _match_shape(mov_img, shape)

    ref_img = to_device(ref_img)
    mov_img = to_device(mov_img)

    if transform is not None:
        ref_img = transform(ref_img)
        mov_img = transform(mov_img)

    Fimg1 = np.fft.rfftn(ref_img)
    Fimg2 = np.fft.rfftn(mov_img)
    eps = np.finfo(Fimg1.dtype).eps
    del ref_img, mov_img

    prod = Fimg1 * Fimg2.conj()
    del Fimg1, Fimg2

    norm = np.fmax(np.abs(prod), eps)
    corr = np.fft.irfftn(prod / norm)
    del prod, norm

    corr = np.fft.fftshift(np.abs(corr))

    argmax = to_cpu(np.argmax(corr))
    peak = np.unravel_index(argmax, corr.shape)
    peak = tuple(s // 2 - p for s, p in zip(corr.shape, peak))

    logger.info(f"phase cross corr. peak at {peak}")

    return peak


# %%
class Autotracker(object):
    _AUTOFOCUS_METHODS = {
        'pcc': phase_cross_corr,
        'tm': template_matching,
        'multiotsu': multiotsu_centroid,
    }

    def __init__(self, autofocus_method: str, xy_dapening: tuple[int] = None):
        self.autofocus_method = autofocus_method
        self.xy_dapening = xy_dapening

    def estimate_shift(
        self, reference_array: ArrayLike, moving_array: ArrayLike, **kwargs
    ) -> np.ndarray:
        autofocus_method_func = self._AUTOFOCUS_METHODS.get(self.autofocus_method)

        if not autofocus_method_func:
            raise ValueError(f'Unknown autofocus method: {self.autofocus_method}')

        shifts = autofocus_method_func(**kwargs)

        return shifts


# %%
def main():
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

    c_idx = 0
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
    # %%
    # Testing Multiotsu
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
    # Testing PCC
    shift_0 = phase_cross_corr(data_t0, data_t0)
    shift_1 = phase_cross_corr(data_t0, data_t1)
    shift_2 = phase_cross_corr(data_t0, data_t2)
    shift_3 = phase_cross_corr(data_t0, data_t3)

    shifts = [shift_0, shift_1, shift_2, shift_3]

    for i, (calculated, expected) in enumerate(zip(shifts, translations)):
        is_similar = np.allclose(calculated, expected, atol=tolerance)
        print(
            f'Timepoint {i+1} shift: {calculated}, Expected: {expected}, Similar: {is_similar}'
        )

    # %%
    # Testing template matching

    crop_z = slice(4, 8)
    crop_y = slice(200, 300)
    crop_x = slice(200, 300)
    template = data_t0[crop_z, crop_y, crop_x]

    result = match_template(data_t1, template)
    zyx = np.unravel_index(np.argmax(result), result.shape)

    print(zyx)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(template[0])
    ax[0].set_title('template')
    ax[1].imshow(data_t1[zyx[0]])
    rect = plt.Rectangle(
        (zyx[2], zyx[1]),
        template.shape[2],
        template.shape[1],
        edgecolor='red',
        facecolor='none',
    )
    ax[1].add_patch(rect)
    ax[1].set_title('template matching result')

    # %%
    # Calculate the shift, apply and check the result
    shift_0 = template_matching(data_t0, data_t0, (crop_z, crop_y, crop_x))
    shift_1 = template_matching(data_t0, data_t1, (crop_z, crop_y, crop_x))
    shift_2 = template_matching(data_t0, data_t2, (crop_z, crop_y, crop_x))
    shift_3 = template_matching(data_t0, data_t3, (crop_z, crop_y, crop_x))

    shifts = [shift_0, shift_1, shift_2, shift_3]

    for i, (calculated, expected) in enumerate(zip(shifts, translations)):
        is_similar = np.allclose(calculated, expected, atol=tolerance)
        print(
            f'Timepoint {i+1} shift: {calculated}, Expected: {expected}, Similar: {is_similar}'
        )


# %%
# %%
if __name__ == "__main__":
    main()
