import logging

from shrimpy.acquisition.microscope_operations import (
    get_daq_counter_names,
    get_total_num_daq_counter_samples,
)

from . import globals

daq_debug = False

logger = logging.getLogger(__name__)


def log_preparing_acquisition(events):
    if isinstance(events, list):
        _event = events[0]
    else:
        _event = events  # events is a dict

    t_idx = _event['axes']['time']
    p_label = _event['axes']['position']

    logger.debug(f'Preparing to acquire timepoint {t_idx} at position {p_label}')

    return events


def check_num_counter_samples(ctr_tasks, events):
    counter_names = get_daq_counter_names(ctr_tasks)
    num_counter_samples = get_total_num_daq_counter_samples(ctr_tasks)
    if daq_debug:
        logger.debug(
            f'DAQ counters {counter_names} will generate a total of {num_counter_samples} pulses'
        )

    event_seq_length = len(events)
    if num_counter_samples != event_seq_length:  # here events may be dict
        logger.error(
            f'Number of counter samples: {num_counter_samples}, is not equal to event sequence length:  {event_seq_length}.'
        )
        logger.error('Aborting acquisition.')
        events = None

    return events


def lf_pre_hardware_hook_function(ctr_tasks, events):
    if globals.lf_acq_aborted:
        logger.debug('Clearing LF acquisition events')
        return

    events = log_preparing_acquisition(events)
    events = check_num_counter_samples(ctr_tasks, events)

    return events


def ls_pre_hardware_hook_function(ctr_tasks, events):
    if globals.ls_acq_aborted:
        logger.debug('Clearing LS acquisition events')
        return

    events = check_num_counter_samples(ctr_tasks, events)

    return events
