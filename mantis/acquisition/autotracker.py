# %%
from pathlib import Path
from re import I
from time import sleep
from typing import Callable, Optional, Tuple, cast

import numpy as np
import pandas as pd
import skimage
import torch
import gc
import tifffile
import importlib

from mantis.acquisition.AcquisitionSettings import AutotrackerSettings
from viscy.translation.engine import AugmentedPredictionVSUNet

from scipy.fftpack import next_fast_len


from numpy.typing import ArrayLike
from skimage.exposure import rescale_intensity
from skimage.feature import match_template
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from waveorder.models.phase_thick_3d import apply_inverse_transfer_function

from mantis import logger
from mantis.acquisition.hook_functions import globals

# FIXME fix the dependencies so that we can install and import dexpv2
# from dexpv2.crosscorr import phase_cross_corr
# from dexpv2.utils import center_crop, pad_to_shape, to_cpu

# TODO: write test functions
# TODO: consider splitting this file into two

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vs_inference_t2t(x: torch.Tensor, cfg: dict, gpu: bool = True) -> torch.Tensor:
    """
    Run virtual staining using a config dictionary and 5D input tensor (B, C, Z, Y, X).
    Returns predicted tensor of shape (B, C_out, Z, Y, X).
    """
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    # Extract model info
    model_cfg = cfg["model"].copy()
    init_args = model_cfg["init_args"]
    class_path = model_cfg["class_path"]

    # Inject ckpt_path from top-level config if needed
    if "ckpt_path" in cfg:
        init_args["ckpt_path"] = cfg["ckpt_path"]

    # Import model class dynamically
    module_path, class_name = class_path.rsplit(".", 1)
    model_class = getattr(importlib.import_module(module_path), class_name)

    # Instantiate model
    model = model_class(**init_args).to(device).eval()

    # Wrap with augmentation logic
    wrapper = (
        AugmentedPredictionVSUNet(
            model=model.model,
            forward_transforms=[lambda t: t],
            inverse_transforms=[lambda t: t],
        )
        .to(x.device)
        .eval()
    )

    wrapper.on_predict_start()
    return wrapper.predict_sliding_windows(x)

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
    
def center_crop(arr: ArrayLike, shape: Tuple[int, ...]) -> ArrayLike:
    """Crops the center of `arr`"""
    assert arr.ndim == len(shape)

    starts = tuple((cur_s - s) // 2 for cur_s, s in zip(arr.shape, shape))

    assert all(s >= 0 for s in starts)

    slicing = tuple(slice(s, s + d) for s, d in zip(starts, shape))

    logger.info(
        f"center crop: input shape {arr.shape}, output shape {shape}, slicing {slicing}"
    )

    return arr[slicing]


def pad_to_shape(arr: ArrayLike, shape: Tuple[int, ...], mode: str, **kwargs) -> ArrayLike:
    """Pads array to shape.

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

    logger.debug(f"padding: input shape {arr.shape}, output shape {shape}, padding {pad_width}")

    return np.pad(arr, pad_width=pad_width, mode=mode, **kwargs)


def _match_shape(img: ArrayLike, shape: Tuple[int, ...]) -> ArrayLike:
    """Pad or crop array to match provided shape."""

    if np.any(shape > img.shape):
        padded_shape = np.maximum(img.shape, shape)
        img = pad_to_shape(img, padded_shape, mode="reflect")

    if np.any(shape < img.shape):
        img = center_crop(img, shape)

    return img


def phase_cross_corr(
    ref_img: ArrayLike,
    mov_img: ArrayLike,
    maximum_shift: float = 1.0,
    normalization: bool = False,
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

    Fimg1 = np.fft.rfftn(ref_img)
    Fimg2 = np.fft.rfftn(mov_img)
    eps = np.finfo(Fimg1.dtype).eps
    del ref_img, mov_img

    prod = Fimg1 * Fimg2.conj()
    del Fimg1, Fimg2

    if normalization:
        norm = np.fmax(np.abs(prod), eps)
    else:
        norm = 1.0
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
    _TRACKING_METHODS = {
        'phase_cross_correlation': phase_cross_corr,
        'template_matching': template_matching,
        'multi_otsu': multiotsu_centroid,
    }

    def __init__(
        self,
        zyx_shape,
        arm,
        position_settings,
        channel_config,
        output_shift_path,
        settings,
        transfer_function,
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
        absolute_shift_limits_um : dict[str, Tuple[float, float]]
            Absolute shift limits in um for each axis
            Dampening factor for xy shifts_zyx
        """

        self.settings = settings 
        self.tracking_method = settings.tracking_method
        self.zyx_dampening = settings.zyx_dampening_factor   
        self.absolute_shift_limits_um = settings.absolute_shift_limits_um
        self.scale = settings.scale_yx
        self.shifts_zyx = None  
        self.transfer_function = transfer_function
        self.ref_volume = None
        self.arm = arm
        self.position_settings = position_settings
        self.channel_config = channel_config
        self.output_shift_path = output_shift_path
        self.zyx_shape = zyx_shape


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

        logger.debug(f'Shifts (z,y,x) pix: {shifts_zyx_pix}')
        logger.debug(f'Scale (um/px): {self.scale}')
        logger.debug(f'Dampening (z,y,x) factor: {self.zyx_dampening}')

        # shifts_zyx in px to shifts_zyx in um
        shifts_zyx_um = np.array(shifts_zyx_pix) * self.scale

        shifts_zyx_um_limited = self.limit_shifts_zyx(shifts_zyx_um)
       
        self.shifts_zyx = shifts_zyx_um_limited
        if any(self.shifts_zyx != shifts_zyx_um_limited):
            logger.debug('Shifts (z,y,x) limited to %s', shifts_zyx_um_limited)

        # Apply dampening if specified
        if self.zyx_dampening is not None:
            self.shifts_zyx = self.shifts_zyx * self.zyx_dampening
        logger.info(f'Shifts (z,y,x) dampened: {self.shifts_zyx}')

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
        self, shifts_zyx: np.ndarray,
    ) -> np.ndarray:
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
        shifts_zyx = np.array(shifts_zyx, dtype=float)
        
        # Map axis order (Z,Y,X) to indices
        axes = ["z", "y", "x"]
       
        # Clamp and threshold shifts automatically
        for i, axis in enumerate(axes):
            min_limit, max_limit = self.absolute_shift_limits_um[axis]
            # Zero out small shifts (below min physical stage threshold)
            if abs(shifts_zyx[i]) < min_limit:
                logger.debug(f'Shifts ({axis}) = {shifts_zyx[i]:.3f} is below the min limit {min_limit}, setting to 0')
                shifts_zyx[i] = 0
                
            # Clip large shifts (above max threshold)
            elif abs(shifts_zyx[i]) > max_limit:
                logger.debug(f'Shifts ({axis}) = {shifts_zyx[i]:.3f} is above the max limit {max_limit}, setting to {np.sign(shifts_zyx[i]) * max_limit:.3f}')
                shifts_zyx[i] = np.sign(shifts_zyx[i]) * max_limit

        return shifts_zyx


# TODO: logic for handling which t_idx to grab as reference. If the volume changes
# Drastically, we may need to grab the previous timepoint as reference

    @staticmethod
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

    def data_preprocessing_labelfree(self, volume_bf: np.ndarray) -> np.ndarray:
        """
        Preprocesses the volume for the autotracker.

        Parameters
        ----------
        volume_bf : np.ndarray
            The volume to preprocess.
        transfer_function : np.ndarray
            The transfer function to use for the phase reconstruction.
        settings : AutotrackerSettings
            The autotracker settings.

        Returns
        -------
        np.ndarray
            The preprocessed volume.
        """
        reconstruction_pipeline = self.settings.reconstruction
        if 'phase' in reconstruction_pipeline:
            logger.info("Reconstructing Phase...")
            tf_tensor = tuple(tf.to(DEVICE) for tf in self.transfer_function)
            t_volume_bf = torch.as_tensor(volume_bf, device=DEVICE, dtype=torch.float32)
            t_volume_phase = apply_inverse_transfer_function(t_volume_bf, *tf_tensor, **self.settings.phase_config['apply_inverse'], z_padding=self.settings.phase_config['transfer_function']['z_padding'])
            volume_phase = t_volume_phase.detach().cpu().numpy()
            del t_volume_bf, tf_tensor
            gc.collect(); torch.cuda.empty_cache()
            if 'vs' in reconstruction_pipeline:
                logger.info("Predicting VS...")
                t_volume_vs = vs_inference_t2t(t_volume_phase.unsqueeze(0).unsqueeze(0), self.settings.vs_config)
                del t_volume_phase
                gc.collect(); torch.cuda.empty_cache()

                volume_nuclei = t_volume_vs.detach().cpu().numpy()[0, 0]
                volume_membrane = t_volume_vs.detach().cpu().numpy()[0, 1]
                del t_volume_vs
                gc.collect(); torch.cuda.empty_cache()
            else:
                volume_phase = t_volume_phase.detach().cpu().numpy()
                del t_volume_phase
                gc.collect(); torch.cuda.empty_cache()

        shift_estimation_channel = self.settings.shift_estimation_channel
        if shift_estimation_channel == 'phase':
            return volume_phase
        elif shift_estimation_channel == 'vs_nuclei':
            return volume_nuclei
        elif shift_estimation_channel == 'vs_membrane':
            return volume_membrane
        elif shift_estimation_channel == 'bf':
            return volume_bf
        else:
            raise ValueError(f"Invalid channel: {shift_estimation_channel}")


    def track(
        self,
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
        if self.arm == 'lf':
            if axes == self.lf_last_img_idx:
                self.lf_acq_finished = True
        elif self.arm == 'ls':
            if axes == self.ls_last_img_idx:
                self.ls_acq_finished = True

        # Get reference to the acquisition engine and it's settings
        # TODO: This is a placeholder, the actual implementation will be different
 
        num_slices = self.zyx_shape[0]
        tracking_interval = self.settings.tracking_interval
        tracking_channel = self.channel_config.config_name
        output_shift_path = Path(self.output_shift_path)
    
        # Get axes info
        p_label = axes['position']
        p_idx = self.position_settings.position_labels.index(p_label)
        t_idx = axes['time']
        channel = axes['channel']
        z_idx = axes['z']

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
                
                volume_ref_axes = (p_idx, 0, tracking_channel, range(num_slices))
                volume_mov_axes = (p_idx, t_idx, tracking_channel, range(num_slices))
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
                    if self.ref_volume is None:
                        logger.info("Reading reference volume...")
                        volume_ref = self.get_volume(dataset, volume_ref_axes)
                        self.ref_volume = self.data_preprocessing_labelfree(volume_ref)
                        del volume_ref
                    else:
                        logger.info("Reference volume in memory, skipping...")

                    logger.info(f"Reading moving volume {t_idx}...")
                    volume_mov = self.get_volume(dataset, volume_mov_axes)
                    volume_mov = self.data_preprocessing_labelfree(volume_mov)

                    # tifffile.imwrite(f"E:\\2025_07_31_test_autotracker\\volume_{t_idx}_0.tiff", volume_t0)
                    # tifffile.imwrite(f"E:\\2025_07_31_test_autotracker\\volume_{t_idx}_1.tiff", volume_t1)
                    
                            
                    # viewer = napari.Viewer()
                    # viewer.add_image(volume_t0)
                    # viewer.add_image(volume_t1)

                    # Reference and moving volumes
                    
                    shifts_zyx = self.estimate_shift(self.ref_volume, volume_mov)
                    del volume_mov
                    #shifts_zyx = shifts_zyx.cpu().numpy()

                csv_log_filename = f"autotracker_fov_{axes['position']}.csv"
                shift_coord_output = output_shift_path / csv_log_filename

                # Read the previous shifts_zyx and coords
                prev_shifts = pd.read_csv(shift_coord_output)
                prev_shifts = prev_shifts.iloc[-1]

                # Read the previous shifts_zyx
                prev_x = self.position_settings.xyz_positions_shift[p_idx][0]
                prev_y = self.position_settings.xyz_positions_shift[p_idx][1]
                prev_z = None
                # Update Z shifts_zyx if available
                if self.position_settings.xyz_positions_shift[p_idx][2] is not None:
                    prev_z = self.position_settings.xyz_positions_shift[p_idx][2]
                logger.info('Previous shifts (x, y, z): %f, %f, %f', prev_x, prev_y, prev_z)
                # Update the event coordinates
                self.position_settings.xyz_positions_shift[p_idx][0] = prev_x + shifts_zyx[-1]
                self.position_settings.xyz_positions_shift[p_idx][1] = prev_y + shifts_zyx[-2]
                # Update Z shifts_zyx if available
                if self.position_settings.xyz_positions_shift[p_idx][2] is not None:
                    self.position_settings.xyz_positions_shift[p_idx][2] = prev_z + shifts_zyx[-3]
                logger.info(
                    'New positions (x, y, z): %f, %f, %f', *self.position_settings.xyz_positions_shift[p_idx]
                )
                # Save the shifts_zyx
                self.save_shifts_to_file(
                    shift_coord_output,
                    position_id=p_label,
                    timepoint_id=t_idx,
                    shifts_zyx=shifts_zyx,
                    stage_coords=(
                        self.position_settings.xyz_positions_shift[p_idx][2],
                        self.position_settings.xyz_positions_shift[p_idx][1],
                        self.position_settings.xyz_positions_shift[p_idx][0],
                    ),
                )
            else:
                # Save the positions at t=0
                csv_log_filename = f"autotracker_fov_{axes['position']}.csv"
                shift_coord_output = output_shift_path / csv_log_filename
                prev_y = self.position_settings.xyz_positions_shift[p_idx][1]
                prev_x = self.position_settings.xyz_positions_shift[p_idx][0]
                if self.position_settings.xyz_positions_shift[p_idx][2] is not None:
                    prev_z = self.position_settings.xyz_positions_shift[p_idx][2]
                else:
                    prev_z = None
                self.save_shifts_to_file(
                    shift_coord_output,
                    position_id=p_label,
                    timepoint_id=t_idx,
                    shifts_zyx=(0, 0, 0),
                    stage_coords=(prev_z, prev_y, prev_x),
                )
