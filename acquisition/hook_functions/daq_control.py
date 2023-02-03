import nidaqmx
from nidaqmx.constants import AcquisitionType, Slope
from nidaqmx.types import CtrTime

def confirm_num_daq_counter_samples(CtrTask:nidaqmx.Task or list, extected_sequence_length, verbose, events):
    """ Intended to be used as pre-hardware hook function, wrapped by functools.partial

    """

    if not isinstance(CtrTask, list):
        CtrTask = [CtrTask]

    event_seq_length = len(events)
    if verbose:
        print(f'Running *confirm_num_daq_counter_samples* pre-hardware hook function. Sequence length: {event_seq_length}')

    num_counter_samples = 1
    for _task in CtrTask:
        num_counter_samples *= _task.timing.samp_quant_samp_per_chan
    
    if num_counter_samples == event_seq_length:
        if verbose:
            print(f'Number of counter samples is equal to event sequence length: {event_seq_length}.')
    else:
        events = None
        raise Warning(f'Number of counter samples: {num_counter_samples}, is not equal to event sequence length:  {event_seq_length}. Aborting acquisition.')
    
    return events

# def set_daq_counter_samples(CtrTask:nidaqmx.Task or list, extected_sequence_length, verbose, events):
#     """ Intended to be used as pre-hardware hook function, wrapped by functools.partial

#     """

#     if not isinstance(CtrTask, list):
#         CtrTask = [CtrTask]

#     event_seq_length = len(events)
#     if verbose:
#         print(f'Running  pre-hardware hook function. Sequence length: {event_seq_length}')
#     if extected_sequence_length is not None and event_seq_length != extected_sequence_length:
#         raise Warning(f'Sequence length of {event_seq_length} events is not equal to the expected sequence length of {extected_sequence_length} events')
    
#     # Counter task needs to be stopped before it is restarted
#     for _task in CtrTask:
#         if _task.is_task_done():
#             _task.stop()
#             _task.timing.cfg_implicit_timing(sample_mode=AcquisitionType.FINITE, samps_per_chan=event_seq_length)
#     return events

def start_daq_counter(CtrTask:nidaqmx.Task or list, verbose, events):
    """ Intended to be used as post-camera hook function, wrapped by functools.partial
    
    """

    if not isinstance(CtrTask, list):
        CtrTask = [CtrTask]

    if verbose:
        print(f'Running *start_daq_counter* post camera hook fun. Starting tasks {[_task.name for _task in CtrTask]}.')
    for _task in CtrTask:
        if _task.is_task_done():
            _task.start()
    return events
