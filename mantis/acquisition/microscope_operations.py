import logging
import numpy as np
import time
from typing import Tuple
import nidaqmx
from nidaqmx.constants import AcquisitionType

logger = logging.getLogger(__name__)

def _try_mmc_call(mmc, mmc_call_name, *mmc_carr_args):
    """Wrapper that tries to repeat calls to mmCore if they fail. Largely copied
    from dragonfly_automation MicromanagerInterface

    """
    num_mm_call_tries = 3
    wait_time_between_mm_calls = 5  # in seconds
    call_succeeded = False
    error_occurred = False
    result = None
    
    for _ in range(num_mm_call_tries):
        try:
            result = getattr(mmc, mmc_call_name)(*mmc_carr_args)
            call_succeeded = True
            break
        except Exception as e:
            error_occurred = True
            msg = str(e).split("\n")[0]
            logger.error(f'An error occurred calling method {mmc_call_name}: {msg}')
            time.sleep(wait_time_between_mm_calls)

    if error_occurred and call_succeeded:
        logger.info(f'The call to method {mmc_call_name} succeeded on a subsequent attempt')
    
    if not call_succeeded:
        message = f'Call to method {mmc_call_name} failed after {num_mm_call_tries} tries'
        logger.critical(message)
        raise Exception(message)
    
    return result

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

def get_position_list(mmStudio, z_stage_name):
    mm_pos_list = mmStudio.get_position_list_manager().get_position_list()
    number_of_positions = mm_pos_list.get_number_of_positions()

    xyz_position_list = []
    position_labels = []
    for i in range(number_of_positions):
        _pos = mm_pos_list.get_position(i)
        xyz_position_list.append([
            _pos.get_x(), 
            _pos.get_y(), 
            _pos.get(z_stage_name).get1_d_position()
        ])
        position_labels.append(_pos.get_label())
    
    return xyz_position_list, position_labels

def set_z_position(mmc, z_stage_name:str, z_position: float):
    _try_mmc_call(
        mmc, 
        'set_position', 
        str(z_stage_name), 
        float(z_position)
    )

def set_relative_z_position(mmc, z_stage_name:str, z_offset: float):
    _try_mmc_call(
        mmc, 
        'set_relative_position', 
        str(z_stage_name), 
        float(z_offset)
    )

def set_xy_position(mmc, xy_position: Tuple[float, float]):
    _try_mmc_call(
        mmc, 
        'set_xy_position', 
        float(xy_position[0]), 
        float(xy_position[1])
    )

def set_relative_xy_position(mmc, xy_offset: Tuple[float, float]):
    _try_mmc_call(
        mmc, 
        'set_relative_xy_position', 
        float(xy_offset[0]), 
        float(xy_offset[1])
    )

def setup_daq_counter(
        task:nidaqmx.Task, 
        co_channel, 
        freq, duty_cycle, 
        samples_per_channel, 
        pulse_terminal
):
    
    logger.debug(f'Setting up {task.name} on {co_channel}')
    logger.debug(f'{co_channel} will output {samples_per_channel} samples with {duty_cycle} duty cycle at {freq:.6f} Hz on terminal {pulse_terminal}')
    
    ctr = task.co_channels.add_co_pulse_chan_freq(
        co_channel, 
        freq=freq, 
        duty_cycle=duty_cycle)
    task.timing.cfg_implicit_timing(
        sample_mode=AcquisitionType.FINITE, 
        samps_per_chan=samples_per_channel)
    ctr.co_pulse_term = pulse_terminal

    return ctr

def get_daq_counter_names(CtrTask:nidaqmx.Task or list):
    if not isinstance(CtrTask, list):
        CtrTask = [CtrTask]

    ctr_names = []
    for _task in CtrTask:
        ctr_names.append(_task.name)

    return ctr_names

def start_daq_counter(CtrTask:nidaqmx.Task or list):
    if not isinstance(CtrTask, list):
        CtrTask = [CtrTask]

    for _task in CtrTask:
        if _task.is_task_done():
            _task.stop()  # Counter needs to be stopped before it is restarted
            _task.start()

def get_total_num_daq_counter_samples(CtrTask:nidaqmx.Task or list):
    if not isinstance(CtrTask, list):
        CtrTask = [CtrTask]

    num_counter_samples = 1
    for _task in CtrTask:
        num_counter_samples *= _task.timing.samp_quant_samp_per_chan

    return num_counter_samples

def autofocus():
    logger.debug('Calling autofocus')
    autofocus_success = bool(np.random.choice(2))
    # autofocus_success = True

    return autofocus_success
