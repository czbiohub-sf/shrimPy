import logging

logger = logging.getLogger(__name__)

def log_acquisition_start(position_labels:list, events):
    if isinstance(events, list):
        _event = events[0]
    else:
        _event = events  # events is a dict

    t_idx = _event['axes']['time']
    p_idx = _event['axes']['position']
    logger.info(f'Starting acquisition of timepoint {t_idx} at position {position_labels[p_idx]}')

    return events