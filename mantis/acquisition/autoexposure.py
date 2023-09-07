# %%
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import yaml

from skimage import data
from waveorder import visual

from mantis import logger
from mantis.acquisition.AcquisitionSettings import (
    AutoexposureSettings,
    ChannelSettings,
    LaserSettings,
    MicroscopeSettings,
    SliceSettings,
)


def execute_autoexposure(channel_settings: ChannelSettings):
    """
    Acquires a z-stack with the selected channel
    Runs the autoexposure and adjusts the laser and exposures
    """
    try:
        logger.info("Running autoexposure of channel '%s'" % channel_settings.config_name)

        # autoexposure_succeed = autoexposure()
    except RuntimeWarning("Autoexposure failed in channel"):
        # autoexposure_succeed = False
        pass


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
    exposure_suggestion = channel_settings.default_exposure_times_ms[0]
    laser_power_suggestion = laser_settings.lasers["488"].laser_power
    flag_exposure = 0  # 1 over-exposed , 0 nominal, -1 under-exposed

    # use a percentile to calculate the 'max' intensity
    # as a defense against hot pixels, anomalous bright spots/dust, etc
    # (the 99.99th percentile corresponds to ~100 pixels in a 1024x1024 image)
    stack_max_intensity = np.percentile(input_stack, 99.99)

    # Use the dtype for reference
    dtype = img_stack.dtype
    # dtype_min = np.iinfo(dtype).min
    dtype_max = np.iinfo(dtype).max
    # Calculate the min-max percent from dtypes
    max_intensity = dtype_max * (1 - autoexposure_settings.max_intensity_percent / 100.0)
    min_intensity = dtype_max * autoexposure_settings.min_intensity_percent / 100.0

    if method == "mean":
        stack_mean = np.mean(input_stack)
        stack_std = np.std(input_stack)

        # Over-exposed
        if (stack_mean + stack_std) > max_intensity or stack_max_intensity >= dtype_max:
            logger.info(
                f"Stack was over-exposed with mean:{stack_mean:.2f} std:{stack_std:.2f})"
            )
            flag_exposure = 1
            autoexposure_succeed = False
            laser_power_suggestion, exposure_suggestion = suggest_exposure_camera(
                flag_exposure, autoexposure_settings, channel_settings, laser_settings
            )

        # Under-exposed
        elif (stack_mean - stack_std) < min_intensity:
            logger.info(
                f"Stack was under-exposed with mean {stack_mean:.2f} std:{stack_std:.2f}"
            )
            flag_exposure = -1
            autoexposure_succeed = False

            laser_power_suggestion, exposure_suggestion = suggest_exposure_camera(
                flag_exposure, autoexposure_settings, channel_settings, laser_settings
            )

        # Nominally exposed
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
        if stack_max_intensity > max_intensity or stack_max_intensity >= dtype_max:
            logger.info("Stack was over-exposed)")
            flag_exposure = 1
            autoexposure_succeed = False
            laser_power_suggestion, exposure_suggestion = suggest_exposure_camera(
                flag_exposure, autoexposure_settings, channel_settings, laser_settings
            )
        else:
            flag_exposure = 0

        if flag_exposure != 1:
            # Check for under-exposure
            intensity_ratio = min_intensity / stack_max_intensity
            if intensity_ratio > 1:
                logger.info(f"Stack was under-exposed with intensity ratio:{intensity_ratio})")
                flag_exposure = -1
                autoexposure_succeed = False

                laser_power_suggestion, exposure_suggestion = suggest_exposure_camera(
                    flag_exposure,
                    autoexposure_settings,
                    channel_settings,
                    laser_settings,
                )
            # Nominally exposed
            else:
                logger.info(
                    f"Stack was nomially exposed with intensity ratio:{intensity_ratio})"
                )
                flag_exposure = 0
                autoexposure_succeed = True

    elif method == "masked_mean":
        underexpose_thresh = 0.1
        overexpose_thresh = 95
        # Mask out hot pixels
        image_masked = np.where(input_stack < stack_max_intensity, input_stack, 0)
        # Calculate the percentile thresholds for underexposure and overexposure
        underexpose_val = np.percentile(image_masked, underexpose_thresh)
        overexpose_val = np.percentile(image_masked, overexpose_thresh)
        # Check these values fall within desired range
        underexpose_val = underexpose_val if underexpose_val > min_intensity else min_intensity
        overexpose_val = overexpose_val if overexpose_val < max_intensity else max_intensity
        stack_mean = np.mean(image_masked)
        logger.info(f"{stack_mean}, {underexpose_val}, {overexpose_val}")

        if stack_mean < underexpose_val or stack_mean < min_intensity:
            logger.info(
                f"Stack is under-exposed mean {stack_mean:.2f} threshold:{underexpose_val:.2f}"
            )
            flag_exposure = -1
            autoexposure_succeed = False
            laser_power_suggestion, exposure_suggestion = suggest_exposure_camera(
                flag_exposure, autoexposure_settings, channel_settings, laser_settings
            )
        elif stack_mean > overexpose_val or stack_max_intensity >= dtype_max:
            logger.info(
                f"Stack is over-exposed mean {stack_mean:.2f} and threshold: {overexpose_val:.2f}"
            )
            flag_exposure = 1
            autoexposure_succeed = False
            laser_power_suggestion, exposure_suggestion = suggest_exposure_camera(
                flag_exposure, autoexposure_settings, channel_settings, laser_settings
            )
        # Nominally exposed
        else:
            logger.info(f"Stack was nomially exposed with mean {stack_mean:.2f})")
            flag_exposure = 0
            autoexposure_succeed = True

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
    exposure_suggestion = channel_settings.default_exposure_times_ms[0]
    # Logic for suggesting new laser or camera parameters
    # Prioritize the laser power bump
    if (
        laser_power_suggestion <= autoexposure_settings.max_laser_power_mW
        or laser_power_suggestion >= autoexposure_settings.min_laser_power_mW
    ):
        laser_power_suggestion = laser_settings.lasers["488"].laser_power - (
            autoexposure_settings.relative_laser_power_step * flag_exposure
        )

    # Change the exposure if the laser settings is maxed out
    elif exposure_suggestion <= autoexposure_settings.max_exposure_time_ms:
        exposure_suggestion = channel_settings.default_exposure_times_ms - (
            autoexposure_settings.relative_exposure_step * flag_exposure
        )
    else:
        logger.Warning(
            f"Autoexposure in channel {1} has reached: " f"laser power{1} and exposure {1}"
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

    channel_settings = ChannelSettings(**mantis_settings.get("ls_channel_settings"))
    stack_settings = SliceSettings(**mantis_settings.get("ls_slice_settings"))
    microscope_settings = MicroscopeSettings(**mantis_settings.get("ls_slice_settings"))
    autoexposure_settings = AutoexposureSettings(
        **mantis_settings.get("autoexposure_settings")
    )
    laser_settings = LaserSettings(**mantis_settings.get("laser_settings"))

    # %%
    methods = ["mean", "percentile", "masked_mean"]
    for method in methods:
        print(f"Using method: {method}")
        # Underexposure
        autoexposure_succeeded, new_laser_power, new_exposure = autoxposure(
            img_stack[0, 1],
            autoexposure_settings,
            channel_settings,
            laser_settings,
            method=method,
        )
        print(autoexposure_succeeded, new_laser_power, new_exposure)

        autoexposure_succeeded, new_laser_power, new_exposure = autoxposure(
            img_stack[1, 1],
            autoexposure_settings,
            channel_settings,
            laser_settings,
            method=method,
        )
        print(autoexposure_succeeded, new_laser_power, new_exposure)
        # assert autoexposure_succeeded is False
        # assert new_exposure > channel_settings.exposure_time_ms[0]
        autoexposure_succeeded, new_laser_power, new_exposure = autoxposure(
            img_stack[2, 1],
            autoexposure_settings,
            channel_settings,
            laser_settings,
            method=method,
        )
        print(autoexposure_succeeded, new_laser_power, new_exposure)

# %%

plot_histograms(img_stack)
visual.image_stack_viewer(img_stack[:, :, 30])
# %%
