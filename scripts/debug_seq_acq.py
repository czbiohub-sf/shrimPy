from pycromanager import Core, Acquisition, multi_d_acquisition_events

mmc = Core()

def print_acq_event(event):
    print(event)

    return event

#%% Test non-sequenced acquisition
mmc.set_property('Z', 'UseSequences', 'No')

events = multi_d_acquisition_events(z_start = 0, z_end = 10, z_step = 1)

print('Testing non-sequenced acquisition \n')
with Acquisition(directory=r'D:\temp', name='PM_test2', pre_hardware_hook_fn=print_acq_event) as acq:
    acq.acquire(events)

#%% Test sequenced z-stack
mmc.set_property('Z', 'UseSequences', 'Yes')

events = multi_d_acquisition_events(z_start = 0, z_end = 10, z_step = 1)

print('Testing sequenced z-stack \n')
with Acquisition(directory=r'D:\temp', name='PM_test2', pre_hardware_hook_fn=print_acq_event) as acq:
    acq.acquire(events)

#%% Test sequenced channel acquisition
mmc.set_property('DAC', 'UseSequences', 'Yes')

events = multi_d_acquisition_events(channel_group='DAC-Channel', channels=[f'State{i}' for i in range(5)])

print('Testing sequenced channel acquisition \n')
with Acquisition(directory=r'D:\temp', name='PM_test2', pre_hardware_hook_fn=print_acq_event) as acq:
    acq.acquire(events)
