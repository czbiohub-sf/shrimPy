import logging
from mantis.acquisition.microscope_operations import (
    get_daq_counter_names,
    start_daq_counter,
)

logger = logging.getLogger(__name__)


def start_daq_counters(ctr_tasks, events):
    ctr_names = get_daq_counter_names(ctr_tasks)
    logger.debug(f'Starting DAQ counter tasks: {ctr_names}')
    start_daq_counter(ctr_tasks)
    return events
