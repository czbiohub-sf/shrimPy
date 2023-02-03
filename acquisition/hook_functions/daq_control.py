import nidaqmx
from nidaqmx.constants import AcquisitionType, Slope
from nidaqmx.types import CtrTime

def set_daq_counter_samples(CtrTask, events, extected_sequence_length=None, verbose=False):
    """ Intended to be used as pre-hardware hook function

    """

    event_seq_length = len(events)
    if verbose:
        print(f'Running pre-hardware hook function. Sequence length: {len(event_seq_length)}')
    if extected_sequence_length is not None and event_seq_length != extected_sequence_length:
        raise Warning(f'Sequence length of {event_seq_length} events is not equal to the expected sequence length of {extected_sequence_length} events')
    
    # Counter task needs to be stopped before it is restarted
    if CtrTask.is_task_done():
        CtrTask.stop()
    CtrTask.timing.cfg_implicit_timing(sample_mode=AcquisitionType.FINITE, samps_per_chan=event_seq_length)
    return events

def start_daq_counters(CtrTask: list, events, verbose=False):
    """ Intended to be used as post-camera hook function
    
    """

    if verbose:
        print(f'Running post camera hook fun. Starting tasks {[_task.name for _task in CtrTask]}.')
    for _task in CtrTask:
        if _task.is_task_done():
            _task.start()
    return events
