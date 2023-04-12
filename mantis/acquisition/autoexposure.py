# %%
import numpy as np
import os
import logging
from mantis.acquisition.BaseSettings import AutoexposureSettings, ChannelSettings
from skimage import data
from waveorder import visual
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def execute_autoexposure():
    """
    Acquires a z-stack with the selected channel
    Runs the autoexposure and adjusts the laser and exposures
    """
    try:
        autoexposure_succeed = autoexposure()
    except RuntimeWarning(f"Autoexposure failed in channel"):
        autoexposure_succeed = False


def autoxposure(input_stack, autoexposure_settings, channel_settings, method="mean"):
    """
    Parameters
    ----------
    input_image : _type_
        _description_
    autoexposure_settings : _type_
        _description_
    channel_settings : _type_
        _description_
    method : str, optional
        _description_, by default "mean"

    Returns
    -------
    Tuple with the following:
    (autoexposure_succeeded, adjusted exposure time, and adjusted laser power)
    """
    logger.info("Starting autoexposure")
    autoexposure_succeed = True
    flag_exposure = 0  # -1-under , 0-nominal, 1-over exposed
    Z, Y, X = input_stack.shape

    dtype = img_stack.dtype
    dtype_min = np.iinfo(dtype).min
    dtype_max = np.iinfo(dtype).max

    if method == "mean":
        stack_mean = np.mean(input_stack)
        stack_std = np.std(input_stack)
        p_threshold = 0.05

        if (stack_mean + stack_std) > (dtype_max * (1 - p_threshold)):
            logger.info(f"Stack was over-exposed with mean {stack_mean})")
            flag_exposure = 1
            # TODO: Logic for suggesting new laser or camera parameters

        elif (stack_mean - stack_std) < (dtype_min * (1 + p_threshold)):
            logger.info(f"Stack was under-exposed with mean {stack_mean})")
            flag_exposure = -1
            # TODO: Logic for suggesting new laser or camera parameters

        else:
            logger.info(f"Stack was nomially exposed with mean {stack_mean})")
            flag_exposure = 0
            autoexposure_succeed = True

    elif method == "percentile":
        # use a percentile to calculate the 'max' intensity
        # as a defense against hot pixels, anomalous bright spots/dust, etc
        stack_max_intensity = np.percentile(input_stack, 99.99)
        logger.info(f"max_intensity ={stack_max_intensity}")

        # Check for over-exposure
        if stack_max_intensity > autoexposure_settings.max_intensity:
            logger.info(f"Stack was over-exposed)")
            flag_exposure = 1
            # TODO: Logic to change the exposure

        # Check for under-exposure
        intensity_ratio = autoexposure_settings.min_intensity / stack_max_intensity
        if intensity_ratio > 1:
            logger.info(f"Stack was under-exposed)")
            flag_exposure = -1

    # log the final results
    logger.info(
        'The final stack max is %d, the laser power is %0.1f%%, and the exposure time is %dms'
        % (
            stack_max_intensity,
            channel_settings.laser_power or 0,
            channel_settings.exposure_time or 0,
        )
    )

    return autoexposure_succeed

def suggest_exposure_camera(flag_exposure,):
    #TODO: which one should we change? the exposure or the laser?
    if flag_exposure ==1:
        #Check if the exposure if low. Change the laser power instead
        if channel_settings.exposure_time < autoexposure_settings.min_exposure_time:
            logger.info(
                    'The minimum exposure time was exceeded '
                    'so the laser power was reduced to %0.1f%%' % (channel_settings.laser_power)
                )

    elif flag_exposure == -1:
        exposure_time = target_range/mean
        laser_power = target_range/mean
  


def create_autoexposure_test_dataset(img_stack):
    """
    Adjust the exposure of the image stack to create overexposed and
    underexposed stacks

    Parameters
    ----------
    img_stack : (C,Z,Y,X) Image stack
    Returns
    -------
    exposure_stack: with additional dimension containing:
                    under, over and nominally exposed image stacks
    """
    C, Z, Y, X = img_stack.shape
    print(img_stack.shape)
    exposure_stack = np.zeros((3,) + img_stack.shape)
    dtype = img_stack.dtype
    dtype_min = np.iinfo(dtype).min
    dtype_max = np.iinfo(dtype).max

    for c_idx in range(C):
        # Get the 5th and 95th percentile values
        pmin, pmax = np.percentile(img_stack[c_idx], (5, 99.99))
        # Under-exposed
        exposure_stack[0, c_idx] = np.where(
            img_stack[c_idx] < pmin, dtype_min, img_stack[c_idx]
        )
        # Over-exposed
        exposure_stack[1, c_idx] = np.where(
            img_stack[c_idx] < pmax, dtype_max, img_stack[c_idx]
        )
        # Nominaly-exposed
        exposure_stack[2, c_idx] = img_stack[c_idx]
        # exposure_stack[2, c_idx] = np.where(
        #     (img_stack[c_idx] >= pmin) & (img_stack[c_idx] <= pmax),
        #     img_stack[c_idx],
        #     np.interp(img_stack[c_idx], [pmin, pmax], [dtype_min, dtype_max]).astype(
        #         dtype
        #     ),
        # )
    return exposure_stack


def plot_histograms(img_stack):
    """
    Plot the under,over and nominal exposed histograms for all channels

    Parameters
    ----------
    img_stack : Exposures,C,Z,Y,X
    """
    # create a subfigure with three columns
    fig, axes = plt.subplots(
        nrows=img_stack.shape[0], ncols=img_stack.shape[1], figsize=(12, 12)
    )
    for i in range(img_stack.shape[-5]):
        for j in range(img_stack.shape[-4]):
            # compute the histogram and bin values for the i-th image and j-th channel
            hist, bins = np.histogram(img_stack[i, j], bins=50)
            # select the axis for the i-th row and j-th column
            ax = axes[i, j]
            # plot the histogram
            ax.hist(img_stack[i, j].flatten(), bins=bins, alpha=0.5)
            ax.set_title(f"Image {i}, Channel {j}")
            ax.set_xlabel("Pixel Value")
            ax.set_ylabel("Frequency")
    # adjust the layout of the subfigure
    plt.tight_layout()


# %%
if __name__ == "__main__":
    cells = data.cells3d().transpose((1, 0, 2, 3))
    C, Z, Y, X = cells.shape
    img_stack = create_autoexposure_test_dataset(cells)

    # # Calculate the mean and standard deviation of the image stack
    # mean = np.mean(image_stack)
    # std_dev = np.std(image_stack)

# %%
