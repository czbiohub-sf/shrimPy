import numpy as np

from mantis import logger
from mantis.acquisition.AcquisitionSettings import AutoexposureSettings


def manual_autoexposure():
    pass


def mean_intensity_autoexposure(
    input_stack,
    current_exposure_time,
    current_laser_power,
    autoexposure_settings: AutoexposureSettings,
):
    logger.info("Starting autoexposure")
    autoexposure_succeed = True
    exposure_suggestion = current_exposure_time
    laser_power_suggestion = current_laser_power
    flag_exposure = 0  # 1 over-exposed , 0 nominal, -1 under-exposed

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
        flag_exposure = 1
        autoexposure_succeed = False
        laser_power_suggestion, exposure_suggestion = _suggest_exposure_camera(
            flag_exposure, autoexposure_settings
        )

    # Under-exposed
    elif (stack_mean - stack_std) < min_intensity:
        logger.info(f"Stack was under-exposed with mean {stack_mean:.2f} std:{stack_std:.2f}")
        flag_exposure = -1
        autoexposure_succeed = False

        laser_power_suggestion, exposure_suggestion = _suggest_exposure_camera(
            flag_exposure, autoexposure_settings
        )

    # Nominally exposed
    else:
        logger.info(
            f"Stack was nomially exposed with mean {stack_mean:.2f} std:{stack_std:.2f})"
        )
        flag_exposure = 0
        autoexposure_succeed = True

    # log the final results
    logger.info(
        f"The final stack max is {int(stack_max_intensity)}, "
        f"the suggested laser power is {laser_power_suggestion or 0}, "
        f"and the suggested exposure time is {exposure_suggestion or 0}ms"
    )
    return autoexposure_succeed, exposure_suggestion, laser_power_suggestion


def masked_mean_intensity_autoexposure(
    input_stack,
    current_exposure_time,
    current_laser_power,
    autoexposure_settings: AutoexposureSettings,
):
    logger.info("Starting autoexposure")
    autoexposure_succeed = True
    exposure_suggestion = current_exposure_time
    laser_power_suggestion = current_laser_power
    flag_exposure = 0  # 1 over-exposed , 0 nominal, -1 under-exposed

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
        flag_exposure = -1
        autoexposure_succeed = False
        laser_power_suggestion, exposure_suggestion = _suggest_exposure_camera(
            flag_exposure, autoexposure_settings
        )
    elif stack_mean > overexpose_val or stack_max_intensity >= dtype_max:
        logger.info(
            f"Stack is over-exposed mean {stack_mean:.2f} and threshold: {overexpose_val:.2f}"
        )
        flag_exposure = 1
        autoexposure_succeed = False
        laser_power_suggestion, exposure_suggestion = _suggest_exposure_camera(
            flag_exposure, autoexposure_settings
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


def intensity_percentile_autoexposure(
    input_stack,
    current_exposure_time,
    current_laser_power,
    autoexposure_settings: AutoexposureSettings,
):
    logger.info("Starting autoexposure")
    autoexposure_succeed = True
    exposure_suggestion = current_exposure_time
    laser_power_suggestion = current_laser_power
    flag_exposure = 0  # 1 over-exposed , 0 nominal, -1 under-exposed

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
        flag_exposure = 1
        autoexposure_succeed = False
        laser_power_suggestion, exposure_suggestion = _suggest_exposure_camera(
            flag_exposure, autoexposure_settings
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

            laser_power_suggestion, exposure_suggestion = _suggest_exposure_camera(
                flag_exposure,
                autoexposure_settings,
            )
        # Nominally exposed
        else:
            logger.info(f"Stack was nomially exposed with intensity ratio:{intensity_ratio})")
            flag_exposure = 0
            autoexposure_succeed = True

    # log the final results
    logger.info(
        f"The final stack max is {int(stack_max_intensity)}, "
        f"the suggested laser power is {laser_power_suggestion or 0}, "
        f"and the suggested exposure time is {exposure_suggestion or 0}ms"
    )
    return autoexposure_succeed, exposure_suggestion, laser_power_suggestion


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
    flag_exposure,
    current_exposure_time,
    current_laser_power,
    autoexposure_settings: AutoexposureSettings,
):
    laser_power_suggestion = current_laser_power
    exposure_suggestion = current_exposure_time
    # Logic for suggesting new laser or camera parameters
    # Prioritize the laser power bump
    if (
        laser_power_suggestion <= autoexposure_settings.max_laser_power_mW
        or laser_power_suggestion >= autoexposure_settings.min_laser_power_mW
    ):
        laser_power_suggestion = current_laser_power - (
            autoexposure_settings.relative_laser_power_step * flag_exposure
        )

    # Change the exposure if the laser settings is maxed out
    elif exposure_suggestion <= autoexposure_settings.max_exposure_time_ms:
        exposure_suggestion = current_exposure_time - (
            autoexposure_settings.relative_exposure_step * flag_exposure
        )
    else:
        logger.warning(
            f"Autoexposure in channel {1} has reached: " f"laser power{1} and exposure {1}"
        )

    return round(laser_power_suggestion), round(exposure_suggestion)
