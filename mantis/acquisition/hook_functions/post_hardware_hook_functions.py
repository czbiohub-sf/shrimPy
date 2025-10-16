import logging
import nidaqmx
import useq

from . import globals

logger = logging.getLogger(__name__)


def get_first_acquisition_event(events):
    if isinstance(events, list):
        event = events[0]
    else:
        event = events  # events is a dict

    return event


def log_acquisition_start(events):
    _event = get_first_acquisition_event(events)

    t_idx = _event['axes']['time']
    p_label = _event['axes']['position']

    logger.info(f'Starting acquisition of timepoint {t_idx} at position {p_label}')

    return events


def update_daq_freq(z_ctr_task, c_idx: int):
    if z_ctr_task.is_task_done():
        z_ctr_task.stop()  # Counter needs to be stopped first
    z_ctr = z_ctr_task.co_channels[0]

    acq_rates = globals.ls_slice_acquisition_rates

    logger.debug(f'Updating {z_ctr.name} pulse frequency to {acq_rates[c_idx]}')
    z_ctr.co_pulse_freq = acq_rates[c_idx]


def update_laser_power(lasers, c_idx: int):
    laser = lasers[c_idx]  # will be None if this channel does not use autoexposure

    if laser and globals.new_well:
        laser_name = laser.serial_number
        laser_power = globals.ls_laser_powers[c_idx]

        logger.debug(f'Updating power of laser {laser_name} to {laser_power}')
        # Note, setting laser power takes ~1 s which is slow
        laser.pulse_power = laser_power


def update_ls_hardware(
    z_ctr_task: nidaqmx.Task, 
    event: useq.MDAEvent
) -> useq.MDAEvent:
    if not event:
        logger.debug('Acquisition event is not valid.')
        return

    c_idx = event.index['c']

    update_daq_freq(z_ctr_task, c_idx)
    # As a hack, setting laser power after call to `run_autoexposure`
    # update_laser_power(lasers, c_idx)

    return event
