import logging

logger = logging.getLogger(__name__)

def set_config(mmc, config_group, config_name):
    logger.debug(f'Setting {config_group} config group to {config_name}')

    mmc.set_config(config_group, config_name)

def set_property(mmc, device_name, property_name, property_value):
    logger.debug(f'Setting {device_name} {property_name} to {property_value}')

    mmc.set_property(device_name, property_name, property_value)

    if 'Line Selector' in property_name:
        mmc.update_system_state_cache()

def set_roi(mmc, roi:tuple):
    logger.debug(f'Setting ROI to {roi}')

    mmc.set_roi(*roi)
