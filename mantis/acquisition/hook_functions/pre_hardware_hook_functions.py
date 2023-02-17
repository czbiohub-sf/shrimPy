import logging
from daq_control import get_num_daq_counter_samples

logger = logging.getLogger(__name__)

def pre_hardware_hook_fn(events):
    if isinstance(events, list):
        _event = events[0]
    else:
        _event = events  # events is a dict

    t_idx = _event['axes']['time']
    p_idx = _event['axes']['position']
    logger.debug(f'Preparing to acquire timepoint {t_idx} at position {p_idx}')

    num_counter_samples = get_num_daq_counter_samples([self._lf_z_ctr_task, self._lf_channel_ctr_task], events)
    logger.debug(f'DAQ counters will generate a total of {num_counter_samples}')

    event_seq_length = len(events)
    if num_counter_samples != event_seq_length:  # here events may be dict
        logger.error(f'Number of counter samples: {num_counter_samples}, is not equal to event sequence length:  {event_seq_length}.')
        logger.error('Aborting acquisition.')

    return events