# Methods in this module will take ND image data and return (autoexposure_flag,
# suggested_exposure_time, suggested_laser_power) as defined by the provided
# AutoexposureSettings. autoexposure_flag = 0 is returned for optimally exposed
# images, autoexposure_flag = 1 is returned for over-exposed images,
# autoexposure_flag = -1 is returned for under-exposed images, and
# autoexposure_flag = None is returned when the autoexposure algorithm has
# errored, in which case the current exposure time and laser power will be
# returned

# An external method (e.g. see
# mantis.acquisition.microscope_operations.autoexposure) is responsible for
# adjusting camera exposure and laser power based on these suggestions until
# convergence.

import numpy as np
import pandas as pd

from mantis import logger
from mantis.acquisition.AcquisitionSettings import AutoexposureSettings


def load_manual_illumination_settings(csv_filepath: str) -> pd.DataFrame:
    """
    Import the manual illumination settings from a CSV file.
    The CSV file should have the following columns:
    - well_id
    - exposure_time_ms
    - laser_power_mW
    """

    df = pd.read_csv(csv_filepath, dtype=str)
    if not set(df.columns) == {"well_id", "exposure_time_ms", "laser_power_mW"}:
        raise ValueError(
            "CSV file must contain columns: well_id, exposure_time_ms, laser_power_mW"
        )
    df.set_index("well_id", inplace=True)
    df["exposure_time_ms"] = df["exposure_time_ms"].astype(float)
    df["laser_power_mW"] = df["laser_power_mW"].astype(float)

    return df

def manual_autoexposure(
    current_exposure_time,
    current_laser_power,
    illumination_settings_filepath,
    well_id,
):
    try:
        autoexposure_flag = 0
        illumination_settings = load_manual_illumination_settings(
            illumination_settings_filepath
        )
        suggested_exposure_time = illumination_settings.loc[well_id, "exposure_time_ms"]
        suggested_laser_power = illumination_settings.loc[well_id, "laser_power_mW"]
    except Exception as e:
        logger.error(f"Error reading manual illumination settings: {e}")
        # If autoexposure fails, we return None for autoexposure_flag
        # and keep the current exposure time and laser power
        autoexposure_flag = None
        suggested_exposure_time = current_exposure_time
        suggested_laser_power = current_laser_power

    return autoexposure_flag, suggested_exposure_time, suggested_laser_power


def mean_intensity_autoexposure(
    input_stack,
    current_exposure_time,
    current_laser_power,
    autoexposure_settings: AutoexposureSettings,
):
    logger.info("Starting autoexposure")
    autoexposure_flag = None
    suggested_exposure_time = current_exposure_time
    suggested_laser_power = current_laser_power

    (
        stack_max_intensity,
        dtype_max,
        max_intensity,
        min_intensity,
    ) = _calculate_data_statistics(input_stack, autoexposure_settings)

    stack_mean = np.mean(input_stack)
    stack_std = np.std(input_stack)

    # Over-exposed
    if (stack_mean + stack_std) > max_intensity or stack_max_intensity >= dtype_max:
        logger.info(f"Stack was over-exposed with mean:{stack_mean:.2f} std:{stack_std:.2f})")
        autoexposure_flag = 1
        suggested_laser_power, suggested_exposure_time = _suggest_exposure_camera(
            autoexposure_flag, autoexposure_settings
        )

    # Under-exposed
    elif (stack_mean - stack_std) < min_intensity:
        logger.info(f"Stack was under-exposed with mean {stack_mean:.2f} std:{stack_std:.2f}")
        autoexposure_flag = -1

        suggested_laser_power, suggested_exposure_time = _suggest_exposure_camera(
            autoexposure_flag, autoexposure_settings
        )

    # Nominally exposed
    else:
        logger.info(
            f"Stack was optimally exposed with mean {stack_mean:.2f} std:{stack_std:.2f})"
        )
        autoexposure_flag = 0

    # log the final results
    logger.info(
        f"The final stack max is {int(stack_max_intensity)}, "
        f"the suggested laser power is {suggested_laser_power or 0}, "
        f"and the suggested exposure time is {suggested_exposure_time or 0}ms"
    )
    return autoexposure_flag, suggested_exposure_time, suggested_laser_power


def masked_mean_intensity_autoexposure(
    input_stack,
    current_exposure_time,
    current_laser_power,
    autoexposure_settings: AutoexposureSettings,
):
    logger.info("Starting autoexposure")
    autoexposure_flag = None
    suggested_exposure_time = current_exposure_time
    suggested_laser_power = current_laser_power

    (
        stack_max_intensity,
        dtype_max,
        max_intensity,
        min_intensity,
    ) = _calculate_data_statistics(input_stack, autoexposure_settings)

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
        autoexposure_flag = -1
        suggested_laser_power, suggested_exposure_time = _suggest_exposure_camera(
            autoexposure_flag, autoexposure_settings
        )
    elif stack_mean > overexpose_val or stack_max_intensity >= dtype_max:
        logger.info(
            f"Stack is over-exposed mean {stack_mean:.2f} and threshold: {overexpose_val:.2f}"
        )
        autoexposure_flag = 1
        suggested_laser_power, suggested_exposure_time = _suggest_exposure_camera(
            autoexposure_flag, autoexposure_settings
        )
    # Nominally exposed
    else:
        logger.info(f"Stack was optimally exposed with mean {stack_mean:.2f})")
        autoexposure_flag = 0

    # log the final results
    logger.info(
        f"The final stack max is {int(stack_max_intensity)}, "
        f"the suggested laser power is {suggested_laser_power or 0}, "
        f"and the suggested exposure time is {suggested_exposure_time or 0}ms"
    )
    return autoexposure_flag, suggested_exposure_time, suggested_laser_power


def intensity_percentile_autoexposure(
    input_stack,
    current_exposure_time,
    current_laser_power,
    autoexposure_settings: AutoexposureSettings,
):
    logger.info("Starting autoexposure")
    autoexposure_flag = None
    suggested_exposure_time = current_exposure_time
    suggested_laser_power = current_laser_power

    (
        stack_max_intensity,
        dtype_max,
        max_intensity,
        min_intensity,
    ) = _calculate_data_statistics(input_stack, autoexposure_settings)

    # use a percentile to calculate the 'max' intensity
    # as a defense against hot pixels, anomalous bright spots/dust, etc
    logger.info(f"max_intensity ={stack_max_intensity}")
    # Check for over-exposure
    if stack_max_intensity > max_intensity or stack_max_intensity >= dtype_max:
        logger.info("Stack was over-exposed)")
        autoexposure_flag = 1
        suggested_laser_power, suggested_exposure_time = _suggest_exposure_camera(
            autoexposure_flag, autoexposure_settings
        )
    else:
        autoexposure_flag = 0

    if autoexposure_flag != 1:
        # Check for under-exposure
        intensity_ratio = min_intensity / stack_max_intensity
        if intensity_ratio > 1:
            logger.info(f"Stack was under-exposed with intensity ratio:{intensity_ratio})")
            autoexposure_flag = -1

            suggested_laser_power, suggested_exposure_time = _suggest_exposure_camera(
                autoexposure_flag,
                autoexposure_settings,
            )
        # Nominally exposed
        else:
            logger.info(f"Stack was optimally exposed with intensity ratio:{intensity_ratio})")
            autoexposure_flag = 0

    # log the final results
    logger.info(
        f"The final stack max is {int(stack_max_intensity)}, "
        f"the suggested laser power is {suggested_laser_power or 0}, "
        f"and the suggested exposure time is {suggested_exposure_time or 0}ms"
    )
    return autoexposure_flag, suggested_exposure_time, suggested_laser_power


def _calculate_data_statistics(
    input_stack,
    autoexposure_settings: AutoexposureSettings,
):
    # use a percentile to calculate the 'max' intensity
    # as a defense against hot pixels, anomalous bright spots/dust, etc
    # (the 99.99th percentile corresponds to ~100 pixels in a 1024x1024 image)
    stack_max_intensity = np.percentile(input_stack, 99.99)

    # Use the dtype for reference
    dtype = input_stack.dtype
    dtype_max = np.iinfo(dtype).max
    # Calculate the min-max percent from dtypes
    max_intensity = dtype_max * (1 - autoexposure_settings.max_intensity_percent / 100.0)
    min_intensity = dtype_max * autoexposure_settings.min_intensity_percent / 100.0

    return stack_max_intensity, dtype_max, max_intensity, min_intensity


def _suggest_exposure_camera(
    autoexposure_flag,
    current_exposure_time,
    current_laser_power,
    autoexposure_settings: AutoexposureSettings,
):
    suggested_laser_power = current_laser_power
    suggested_exposure_time = current_exposure_time
    # Logic for suggesting new laser or camera parameters
    # Prioritize the laser power bump
    if (
        suggested_laser_power <= autoexposure_settings.max_laser_power_mW
        or suggested_laser_power >= autoexposure_settings.min_laser_power_mW
    ):
        suggested_laser_power = current_laser_power - (
            autoexposure_settings.relative_laser_power_step * autoexposure_flag
        )

    # Change the exposure if the laser settings is maxed out
    elif suggested_exposure_time <= autoexposure_settings.max_exposure_time_ms:
        suggested_exposure_time = current_exposure_time - (
            autoexposure_settings.relative_exposure_step * autoexposure_flag
        )
    else:
        logger.warning(
            f"Autoexposure in channel {1} has reached: " f"laser power{1} and exposure {1}"
        )

    return round(suggested_laser_power), round(suggested_exposure_time)
