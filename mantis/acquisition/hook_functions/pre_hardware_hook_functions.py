import logging
from mantis.acquisition.microscope_operations import (
    get_daq_counter_names,
    get_total_num_daq_counter_samples,
)

daq_debug = False

logger = logging.getLogger(__name__)

def log_preparing_acquisition(position_labels, events):
    if isinstance(events, list):
        _event = events[0]
    else:
        _event = events  # events is a dict

    t_idx = _event['axes']['time']
    p_idx = _event['axes']['position']
    logger.debug(f'Preparing to acquire timepoint {t_idx} at position {position_labels[p_idx]}')

    return events

def check_num_counter_samples(ctr_tasks, events):
    counter_names = get_daq_counter_names(ctr_tasks)
    num_counter_samples = get_total_num_daq_counter_samples(ctr_tasks)
    if daq_debug:
        logger.debug(f'DAQ counters {counter_names} will generate a total of {num_counter_samples} pulses')

    event_seq_length = len(events)
    if num_counter_samples != event_seq_length:  # here events may be dict
        logger.error(f'Number of counter samples: {num_counter_samples}, is not equal to event sequence length:  {event_seq_length}.')
        logger.error('Aborting acquisition.')
        events = None
    
    return events

def log_preparing_acquisition_check_counter(position_labels, ctr_tasks, events):
    events = log_preparing_acquisition(position_labels, events)
    events = check_num_counter_samples(ctr_tasks, events)

    return events