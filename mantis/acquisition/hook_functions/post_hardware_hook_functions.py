import logging

logger = logging.getLogger(__name__)


def log_acquisition_start(position_labels: list, events):
    if isinstance(events, list):
        _event = events[0]
    else:
        _event = events  # events is a dict

    t_idx = _event['axes']['time']
    p_idx = _event['axes']['position']
    logger.info(
        f'Starting acquisition of timepoint {t_idx} at position {position_labels[p_idx]}'
    )

    return events


def update_daq_freq(z_ctr_task, channels: list, acq_rates: list, events):
    if isinstance(events, list):
        _event = events[0]
    else:
        _event = events  # events is a dict

    c_idx = channels.index(_event['axes']['channel'])
    if z_ctr_task.is_task_done():
        z_ctr_task.stop()  # Counter needs to be stopped first
    z_ctr = z_ctr_task.co_channels[0]

    logger.debug(f'Updating {z_ctr.name} pulse frequency to {acq_rates[c_idx]}')
    z_ctr.co_pulse_freq = acq_rates[c_idx]

    return events


def update_daq_freq_log_start_acq(z_ctr_task, channels, acq_rates, position_labels, events):
    events = update_daq_freq(z_ctr_task, channels, acq_rates, events)
    events = log_acquisition_start(position_labels, events)

    return events
