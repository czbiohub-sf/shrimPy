# %%
from pathlib import Path
from time import sleep
from typing import Callable, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import skimage
import torch

from numpy.typing import ArrayLike
from skimage.exposure import rescale_intensity
from skimage.feature import match_template
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from waveorder.models.phase_thick_3d import apply_inverse_transfer_function, calculate_transfer_function

from mantis import logger
from mantis.acquisition.hook_functions import globals

# FIXME fix the dependencies so that we can install and import dexpv2
# from dexpv2.crosscorr import phase_cross_corr
# from dexpv2.utils import center_crop, pad_to_shape, to_cpu

# TODO: write test functions
# TODO: consider splitting this file into two

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    Computes the translation shifts_zyx using a multiotsu threshold approach by finding the centroid of the regions

    Parameters
    ----------
    moving : ndarray
        moving stack ZYX
    reference : ndarray
        reference image stack ZYX

    Returns
    -------
    shifts_zyx : list
        list of shifts_zyx in z, y, x order
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

    # Find the shifts_zyx
    shifts_zyx = moving_center - target_center

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
    logger.debug('shifts_zyx (z,y,x): %f,%f,%f', shifts_zyx[0], shifts_zyx[1], shifts_zyx[2])

    return shifts_zyx


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

 # ensure float tensor
    
def phase_cross_corr(
    ref_img: ArrayLike,
    mov_img: ArrayLike,
    transform: Optional[Callable[[ArrayLike], ArrayLike]] = None,
    normalization: Optional[Literal['magnitude', 'classic']] = None,
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
    Returns
    -------
    Tuple[int, ...]
        Shift between reference and moved image.
    """

    if transform is not None:
        ref_img = transform(ref_img)
        mov_img = transform(mov_img)

    Fimg1 = np.fft.rfftn(ref_img)
    Fimg2 = np.fft.rfftn(mov_img)
    eps = np.finfo(Fimg1.dtype).eps
    del ref_img, mov_img

    prod = Fimg1 * Fimg2.conj()

    if normalization == 'magnitude':
        norm = np.fmax(np.abs(prod), eps)
    elif normalization == 'classic':
        norm = np.abs(Fimg1)*np.abs(Fimg2)
    else:
        norm = 1.0
    corr = np.fft.irfftn(prod / norm)
    del Fimg1, Fimg2

    maxima = np.unravel_index(
        np.argmax(np.abs(corr)), corr.shape
    )
    midpoint = np.array([np.fix(axis_size / 2) for axis_size in corr.shape])

    float_dtype = prod.real.dtype
    del prod, norm

    shift = np.stack(maxima).astype(float_dtype, copy=False)
    shift[shift > midpoint] -= np.array(corr.shape)[shift > midpoint]
    del corr

    logger.info(f"phase cross corr. shift {shift}")

    return shift


# %%
class Autotracker(object):
    _TRACKING_METHODS = {
        'phase_cross_correlation': phase_cross_corr,
        'template_matching': template_matching,
        'multi_otsu': multiotsu_centroid,
    }

    def __init__(
        self,
        tracking_method: str,
        shift_limit: Tuple[float, float, float],
        scale: ArrayLike,
        zyx_shape: Tuple[int, int, int],
        zyx_dampening_factor: ArrayLike = None,
        phase_config: dict = None,
    ):
        """
        Autotracker object

        Parameters
        ----------
        tracking_method : str
            Method to use for autofocus. Options are 'phase_cross_correlation', 'template_matching', 'multi_otsu'
        scale : ArrayLike[float, float, float]
            Scale factor to convert shifts_zyx from px to um
        xy_dampening : tuple[int]
            Dampening factor for xy shifts_zyx
        """
        self.tracking_method = tracking_method
        self.zyx_dampening = zyx_dampening_factor   
        self.shift_limit = shift_limit
        self.scale = scale
        self.shifts_zyx = None
        self.phase_config = phase_config
        self.zyx_shape = zyx_shape
        if self.phase_config is not None:
            # TODO: compute the transfer function
            self.phase_config['transfer_function']['zyx_shape'] = zyx_shape
            self.transfer_function = tuple(tf.to(DEVICE) for tf in calculate_transfer_function(**self.phase_config['transfer_function']))


        
    def estimate_shift(self, ref_img: ArrayLike, mov_img: ArrayLike, **kwargs) -> np.ndarray:
        """
        Estimates the shift between two images using the specified autofocus method.

        Parameters
        ----------
        ref_img : ArrayLike
            Reference image.
        mov_img : ArrayLike
            Image to be aligned with the reference.
        kwargs : dict
            Additional keyword arguments to be passed to the autofocus method.

        Returns
        -------
        np.ndarray
            The estimated shift in scale provided by the user (typically um).
        """

        autofocus_method_func = self._TRACKING_METHODS.get(self.tracking_method)

        if not autofocus_method_func:
            raise ValueError(f'Unknown autofocus method: {self.tracking_method}')

        shifts_zyx_pix = autofocus_method_func(ref_img=ref_img, mov_img=mov_img, **kwargs)

        # shifts_zyx in px to shifts_zyx in um
        shifts_zyx_um = np.array(shifts_zyx_pix) * self.scale

        # Limit the shifts_zyx, preserving the sign of the shift
        self.shifts_zyx = np.sign(shifts_zyx_um) * np.minimum(np.abs(shifts_zyx_um), self.shift_limit)
        if any(self.shifts_zyx != shifts_zyx_um):
            logger.debug('Shifts_zyx limited to %s', self.shifts_zyx)

        if self.zyx_dampening is not None:
            self.shifts_zyx = self.shifts_zyx * self.zyx_dampening
        logger.info(f'shifts_zyx (z,y,x): {self.shifts_zyx}')

        return self.shifts_zyx

    # Function to log the shifts_zyx to a csv file
    def save_shifts_to_file(
        self,
        output_file: str,
        position_id: int,
        timepoint_id: int,
        shifts_zyx: Tuple[int, int, int] = None,
        stage_coords: Tuple[int, int, int] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Saves the computed shifts_zyx to a CSV file.

        Parameters
        ----------
        output_file : str
            Path to the output CSV file.
        shifts_zyx : Tuple[int, int, int]
            The computed shifts_zyx (Z, Y, X).
        position_id : int
            Identifier for the position.
        timepoint_id : int
            Identifier for the timepoint.
        overwrite : bool
            If True, the file will be overwritten if it exists.
        """
        # Convert output_file to a Path object
        output_path = Path(output_file)
        if shifts_zyx is None:
            shifts_zyx = self.shifts_zyx
        if stage_coords is None:
            stage_coords = (0, 0, 0)
        data = {
            "PositionID": [position_id],
            "TimepointID": [timepoint_id],
            "ShiftZ": [shifts_zyx[-3]],
            "ShiftY": [shifts_zyx[-2]],
            "ShiftX": [shifts_zyx[-1]],
            "StageZ": [stage_coords[-3]],
            "StageY": [stage_coords[-2]],
            "StageX": [stage_coords[-1]],
        }

        df = pd.DataFrame(data)

        if overwrite or not output_path.exists():
            # Write the DataFrame to a new file, including the header
            df.to_csv(output_path, mode='w', index=False)
        else:
            # Append the DataFrame to the existing file, without writing the header
            df.to_csv(output_path, mode='a', header=False, index=False)

    def limit_shifts_zyx(
        self, shifts_zyx: Tuple[int, int, int], limits: Tuple[int, int, int] = (5, 5, 5)
    ) -> Tuple[int, int, int]:
        """
        Limits the shifts_zyx to the specified limits.

        Parameters
        ----------
        shifts_zyx : Tuple[int, int, int]
            The computed shifts_zyx (Z, Y, X).
        limits : Tuple[int, int, int]
            The limits for the shifts_zyx (Z, Y, X).

        Returns
        -------
        Tuple[int, int, int]
            The limited shifts_zyx.
        """
        shifts_zyx = np.array(shifts_zyx)
        limits = np.array(limits)
        shifts_zyx = np.where(np.abs(shifts_zyx) > limits, 0, shifts_zyx)
        return tuple(shifts_zyx)


# TODO: logic for handling which t_idx to grab as reference. If the volume changes
# Drastically, we may need to grab the previous timepoint as reference


def get_volume(dataset, axes):
    p_idx, t_idx, autotrack_channel, z_idx = axes
    images = []
    logger.debug(
        f"Getting z-stack for p:{p_idx}, t:{t_idx}, c:{autotrack_channel}, z:{z_idx}"
    )
    for _z in z_idx:
        images.append(
            dataset.read_image(
                **{'channel': autotrack_channel, 'z': _z, 'time': t_idx, 'position': p_idx}
            )
        )
    return np.stack(images)


def autotracker_hook_fn(
    arm,
    autotracker_settings,
    position_settings,
    channel_config,
    z_slice_settings,
    output_shift_path,
    yx_shape,
    axes,
    dataset,

) -> None:
    """
    Pycromanager hook function that is called when an image is saved.

    Parameters
    ----------
    axes : Position, Time, Channel, Z_slice
    dataset: Dataset saved in disk
    """
    # logger.info('Autotracker hook function called for axes %s', axes)

    # TODO: handle the lf acq or ls_a
    if arm == 'lf':
        if axes == globals.lf_last_img_idx:
            globals.lf_acq_finished = True
    elif arm == 'ls':
        if axes == globals.ls_last_img_idx:
            globals.ls_acq_finished = True

    # Get reference to the acquisition engine and it's settings
    # TODO: This is a placeholder, the actual implementation will be different
    z_range = z_slice_settings.z_range
    num_slices = z_slice_settings.num_slices
    scale = autotracker_settings.scale_yx
    shift_limit = autotracker_settings.shift_limit
    tracking_method = autotracker_settings.tracking_method
    tracking_interval = autotracker_settings.tracking_interval
    tracking_channel = channel_config.config_name
    zyx_dampening_factor = autotracker_settings.zyx_dampening_factor
    output_shift_path = Path(output_shift_path)
    zyx_shape = (num_slices, yx_shape[0], yx_shape[1])
    # Get axes info
    p_label = axes['position']
    p_idx = position_settings.position_labels.index(p_label)
    t_idx = axes['time']
    channel = axes['channel']
    z_idx = axes['z']

    tracker = Autotracker(
        tracking_method=tracking_method,
        scale=scale,
        shift_limit=shift_limit,
        zyx_dampening_factor=zyx_dampening_factor,
        zyx_shape=zyx_shape,
    )
    # Get the z_max
    if channel == tracking_channel and z_idx == (num_slices - 1):
        # Skip the 1st timepoint
        if t_idx >= 1:
            if t_idx % tracking_interval != 0:
                logger.debug('Skipping autotracking t %d', t_idx)
                return
            logger.debug("WELCOME TO THE FOCUS ZONE")
            # logger.debug('Curr axes :P:%s, T:%d, C:%s, Z:%d', p_idx, t_idx, channel, z_idx)

            # Logic to get the volumes
            volume_t0_axes = (p_idx, 0, tracking_channel, range(len(z_range)))
            volume_t1_axes = (p_idx, t_idx, tracking_channel, range(len(z_range)))
            # Compute the shifts_zyx
            logger.debug('Instantiating autotracker')
            if globals.demo_run:
                # Random shifting for demo purposes
                shifts_zyx = np.random.randint(-50, 50, 3)
                sleep(3)
                logger.info(
                    'shifts_zyx (z,y,x): %f,%f,%f', shifts_zyx[0], shifts_zyx[1], shifts_zyx[2]
                )
            else:
                volume_t0 = get_volume(dataset, volume_t0_axes)
                volume_t1 = get_volume(dataset, volume_t1_axes)

                if tracker.phase_config is not None:
                    volume_t0 = torch.as_tensor(volume_t0, device=DEVICE, dtype=torch.float32)
                    volume_t1 = torch.as_tensor(volume_t1, device=DEVICE, dtype=torch.float32)
                    volume_t0 = apply_inverse_transfer_function(volume_t0, *tracker.transfer_function, **tracker.phase_config['apply_inverse'], z_padding=tracker.phase_config['transfer_function']['z_padding'])
                    volume_t1 = apply_inverse_transfer_function(volume_t1, *tracker.transfer_function, **tracker.phase_config['apply_inverse'], z_padding=tracker.phase_config['transfer_function']['z_padding'])
                if tracker.vs_config is not None:
                    pass
                    # TODO: apply the vs config
                
                # viewer = napari.Viewer()
                # viewer.add_image(volume_t0)
                # viewer.add_image(volume_t1)

                # Reference and moving volumes
                volume_t0 = volume_t0.detach().cpu().numpy()
                volume_t1 = volume_t1.detach().cpu().numpy()
                
                shifts_zyx = tracker.estimate_shift(volume_t0, volume_t1)
                del volume_t0, volume_t1
                #shifts_zyx = shifts_zyx.cpu().numpy()

            csv_log_filename = f"autotracker_fov_{axes['position']}.csv"
            shift_coord_output = output_shift_path / csv_log_filename

            # Read the previous shifts_zyx and coords
            prev_shifts = pd.read_csv(shift_coord_output)
            prev_shifts = prev_shifts.iloc[-1]

            # Read the previous shifts_zyx
            prev_x = position_settings.xyz_positions_shift[p_idx][0]
            prev_y = position_settings.xyz_positions_shift[p_idx][1]
            prev_z = None
            # Update Z shifts_zyx if available
            if position_settings.xyz_positions_shift[p_idx][2] is not None:
                prev_z = position_settings.xyz_positions_shift[p_idx][2]
            logger.info('Previous shifts (x, y, z): %f, %f, %f', prev_x, prev_y, prev_z)
            # Update the event coordinates
            position_settings.xyz_positions_shift[p_idx][0] = prev_x + shifts_zyx[-1]
            position_settings.xyz_positions_shift[p_idx][1] = prev_y + shifts_zyx[-2]
            # Update Z shifts_zyx if available
            if position_settings.xyz_positions_shift[p_idx][2] is not None:
                position_settings.xyz_positions_shift[p_idx][2] = prev_z + shifts_zyx[-3]
            logger.info(
                'New positions (x, y, z): %f, %f, %f', *position_settings.xyz_positions_shift[p_idx]
            )
            # Save the shifts_zyx
            tracker.save_shifts_to_file(
                shift_coord_output,
                position_id=p_label,
                timepoint_id=t_idx,
                shifts_zyx=shifts_zyx,
                stage_coords=(
                    position_settings.xyz_positions_shift[p_idx][2],
                    position_settings.xyz_positions_shift[p_idx][1],
                    position_settings.xyz_positions_shift[p_idx][0],
                ),
            )
        else:
            # Save the positions at t=0
            csv_log_filename = f"autotracker_fov_{axes['position']}.csv"
            shift_coord_output = output_shift_path / csv_log_filename
            prev_y = position_settings.xyz_positions_shift[p_idx][1]
            prev_x = position_settings.xyz_positions_shift[p_idx][0]
            if position_settings.xyz_positions_shift[p_idx][2] is not None:
                prev_z = position_settings.xyz_positions_shift[p_idx][2]
            else:
                prev_z = None
            tracker.save_shifts_to_file(
                shift_coord_output,
                position_id=p_label,
                timepoint_id=t_idx,
                shifts_zyx=(0, 0, 0),
                stage_coords=(prev_z, prev_y, prev_x),
            )
