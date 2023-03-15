import logging
import nidaqmx
from nidaqmx.constants import AcquisitionType

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