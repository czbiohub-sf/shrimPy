# %%
import numpy as np
import os
import logging
from mantis.acquisition import BaseSettings
from mantis.acquisition.BaseSettings import (
    ChannelSettings,
    SliceSettings,
    MicroscopeSettings,
    AutoexposureSettings,
    LaserSettings,
)
from skimage import data

from waveorder import visual
import matplotlib.pyplot as plt
import pathlib
import yaml
from mantis import logger


def execute_autoexposure():
    """
    Acquires a z-stack with the selected channel
    Runs the autoexposure and adjusts the laser and exposures
    """
    try:
        autoexposure_succeed = autoexposure()
    except RuntimeWarning(f"Autoexposure failed in channel"):
        autoexposure_succeed = False


def autoxposure(
    input_stack,
    autoexposure_settings: AutoexposureSettings,
    channel_settings: ChannelSettings,
    laser_settings: LaserSettings,
    method: str = "mean",
):
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
    # TODO: Find smart way of changing the laser and channel settings for laser power.
    logger.info("Starting autoexposure")
    autoexposure_succeed = True
    exposure_suggestion = channel_settings.exposure_time_ms[0]
    laser_power_suggestion = laser_settings.lasers["488"].laser_power
    flag_exposure = 0  # 1 over-exposed , 0 nominal, -1 under-exposed
    stack_max_intensity = np.percentile(input_stack, 99.99)

    # Use the dtype for reference
    dtype = img_stack.dtype
    dtype_min = np.iinfo(dtype).min
    dtype_max = np.iinfo(dtype).max

    if method == "mean":
        stack_mean = np.mean(input_stack)
        stack_std = np.std(input_stack)
        p_threshold = 0.05

        ## Over-exposed
        if (stack_mean + stack_std) > (dtype_max * (1 - p_threshold)) or stack_max_intensity >= dtype_max:
            logger.info(
                f"Stack was over-exposed with mean:{stack_mean:.2f} std:{stack_std:.2f})"
            )
            flag_exposure = 1
            autoexposure_succeed = False
            laser_power_suggestion, exposure_suggestion = suggest_exposure_camera(
                flag_exposure, autoexposure_settings, channel_settings, laser_settings
            )

        ## Under-exposed
        elif (stack_mean - stack_std) < (dtype_min * (1 + p_threshold)):
            logger.info(
                f"Stack was under-exposed with mean {stack_mean:.2f} std:{stack_std:.2f}"
            )
            flag_exposure = -1
            autoexposure_succeed = False

            laser_power_suggestion, exposure_suggestion = suggest_exposure_camera(
                flag_exposure, autoexposure_settings, channel_settings, laser_settings
            )

        ## Nominally exposed
        else:
            logger.info(
                f"Stack was nomially exposed with mean {stack_mean:.2f} std:{stack_std:.2f})"
            )
            flag_exposure = 0
            autoexposure_succeed = True

    elif method == "percentile":
        # use a percentile to calculate the 'max' intensity
        # as a defense against hot pixels, anomalous bright spots/dust, etc
        logger.info(f"max_intensity ={stack_max_intensity}")
        # Check for over-exposure
        if stack_max_intensity > autoexposure_settings.max_intensity:
            logger.info(f"Stack was over-exposed)")
            flag_exposure = 1
            autoexposure_succeed = False
            laser_power_suggestion, exposure_suggestion = suggest_exposure_camera(
                flag_exposure, autoexposure_settings, channel_settings, laser_settings
            )

        # Check for under-exposure
        intensity_ratio = autoexposure_settings.min_intensity / stack_max_intensity
        if intensity_ratio > 1:
            logger.info(
                f"Stack was under-exposed with intensity ratio:{intensity_ratio})"
            )
            flag_exposure = -1
            autoexposure_succeed = False

            laser_power_suggestion, exposure_suggestion = suggest_exposure_camera(
                flag_exposure, autoexposure_settings, channel_settings, laser_settings
            )

    # log the final results
    logger.info(
        f"The final stack max is {int(stack_max_intensity)}, "
        f"the suggested laser power is {laser_power_suggestion or 0}, "
        f"and the suggested exposure time is {exposure_suggestion or 0}ms"
    )
    return autoexposure_succeed, exposure_suggestion, laser_power_suggestion


def suggest_exposure_camera(
    flag_exposure,
    autoexposure_settings: AutoexposureSettings,
    channel_settings: ChannelSettings,
    laser_settings: LaserSettings,
):
    laser_power_suggestion = laser_settings.lasers["488"].laser_power
    exposure_suggestion = channel_settings.exposure_time_ms[0]
    # Logic for suggesting new laser or camera parameters
    # Prioritize the laser power bump
    if (
        laser_power_suggestion <= autoexposure_settings.max_laser_power_mW
        or laser_power_suggestion >= autoexposure_settings.min_laser_power_mW
    ):
        laser_power_suggestion = (
            laser_settings.lasers["488"].laser_power
            - (autoexposure_settings.relative_laser_power_step * flag_exposure)
        )

    # Change the exposure if the laser settings is maxed out
    elif exposure_suggestion <= autoexposure_settings.max_exposure_time_ms:
        exposure_suggestion = (
            channel_settings.exposure_time_ms
            - (autoexposure_settings.relative_exposure_step * flag_exposure)
        )
    else:
        logger.Warning(
            f"Autoexposure in channel {1} has reached: "
            f"laser power{1} and exposure {1}"
        )

    return round(laser_power_suggestion), round(exposure_suggestion)


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
    logger.info("Creating Autoexposure Datase")
    C, Z, Y, X = img_stack.shape
    dtype = img_stack.dtype
    exposure_stack = np.zeros((3,) + img_stack.shape).astype(dtype)
    dtype_min = np.iinfo(dtype).min
    dtype_max = np.iinfo(dtype).max

    for c_idx in range(C):
        # Get the 5th and 95th percentile values
        pmin, pmax = np.percentile(img_stack[c_idx], (1, 90))
        # Under-exposed
        exposure_stack[0, c_idx] = np.where(
            img_stack[c_idx] > pmin, dtype_min, img_stack[c_idx]
        ).astype(dtype)
        # Over-exposed
        exposure_stack[1, c_idx] = np.where(
            img_stack[c_idx] > pmax, dtype_max, img_stack[c_idx]
        ).astype(dtype)
        # Nominaly-exposed
        exposure_stack[2, c_idx] = img_stack[c_idx].astype(dtype)
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
    # plot_histograms(img_stack)
    # visual.image_stack_viewer(img_stack[:,:,30])

    SETTINGS_PATH = (
        pathlib.Path(__file__).parent.parent
        / "acquisition"
        / "settings"
        / "demo_acquisition_settings.yaml"
    )
    with open(SETTINGS_PATH, "r") as file:
        mantis_settings = yaml.safe_load(file)
        logger.info(mantis_settings)

    channel_settings = BaseSettings.ChannelSettings(
        **mantis_settings.get("ls_channel_settings")
    )
    stack_settings = BaseSettings.SliceSettings(
        **mantis_settings.get("ls_slice_settings")
    )
    microscope_settings = BaseSettings.MicroscopeSettings(
        **mantis_settings.get("ls_slice_settings")
    )
    autoexposure_settings = BaseSettings.AutoexposureSettings(
        **mantis_settings.get("autoexposure_settings")
    )
    laser_settings = BaseSettings.LaserSettings(**mantis_settings.get("laser_settings"))

    # %%
    # Underexposure
    autoexposure_succeeded, new_laser_power, new_exposure = autoxposure(
        img_stack[0, 1],
        autoexposure_settings,
        channel_settings,
        laser_settings,
        method="mean",
    )
    print(autoexposure_succeeded, new_laser_power, new_exposure)

    autoexposure_succeeded, new_laser_power, new_exposure = autoxposure(
        img_stack[1, 1],
        autoexposure_settings,
        channel_settings,
        laser_settings,
        method="mean",
    )
    print(autoexposure_succeeded, new_laser_power, new_exposure)
    # assert autoexposure_succeeded is False
    # assert new_exposure > channel_settings.exposure_time_ms[0]
    autoexposure_succeeded, new_laser_power, new_exposure = autoxposure(
        img_stack[2, 1],
        autoexposure_settings,
        channel_settings,
        laser_settings,
        method="mean",
    )
    print(autoexposure_succeeded, new_laser_power, new_exposure)

# %%

plot_histograms(img_stack)
visual.image_stack_viewer(img_stack[:, :, 30])
# %%