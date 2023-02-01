#%% import modules
# mantis conda env: acq_pm_latest

import os
import time
import numpy as np
from pycromanager import start_headless, Core, Acquisition, multi_d_acquisition_events

import nidaqmx
from nidaqmx.constants import AcquisitionType, Slope
from nidaqmx.types import CtrTime

#%% Define constants
_verbose = False
PORT1 = 4827
LS_POST_READOUT_DELAY = 0.05 # in ms

#%% Set acquisition parameters
save_path = r'D:\\temp'

n_timepoints = 3
time_interval = 30 # in seconds

ls_exposure_ms = 10

AP_galvo_start = -2/10 # in Volts
AP_galvo_end = 2/10
AP_galvo_step = 0.01

ls_channels = ['GFP EX488 EM525-45', 'mCherry EX561 EM600-37']

#%% Connect to running MM on default port 4827
# MM config: mantis-LS.cfg

mmc2 = Core(port=PORT1)

#%% setup light sheet acquisition

roi = mmc2.get_roi()
ls_roi = [roi.x, roi.y, roi.width, roi.height]
assert ls_roi[3] < 300, 'Please crop camera sensor for light-sheet acquisition'

# mmc2.set_config('Channel - LS', 'GFP EX488 EM525-45')

# Set readout rate and gain
mmc2.set_property('Prime BSI Express', 'ReadoutRate', '200MHz 11bit')
mmc2.set_property('Prime BSI Express', 'Gain', '1-Full well')
# One frame is acquired for every trigger pulse
mmc2.set_property('Prime BSI Express', 'TriggerMode', 'Edge Trigger')
# Rolling Shutter Exposure Out mode is high when all rows are exposing
mmc2.set_property('Prime BSI Express', 'ExposeOutMode', 'Rolling Shutter')

mmc2.set_property('Core', 'Focus', 'AP Galvo')

mmc2.set_property('TS2_DAC03', 'Sequence', 'On') # turn off sequencing to avoid PM bugs
# Illuminate sample only when all rows are exposing, aka pseudo global shutter 
mmc2.set_property('TS2_TTL1-8', 'Blanking', 'On')

mmc2.set_exposure(ls_exposure_ms)
ls_readout_time_ms = np.around(float(mmc2.get_property('Prime BSI Express', 'Timing-ReadoutTimeNs'))*1e-6, decimals=3)
assert ls_readout_time_ms < ls_exposure_ms, f'Exposure time needs to be greater than the {ls_readout_time_ms} sensor readout time'

#%% Setup acquisition events

ls_events = multi_d_acquisition_events(num_time_points=n_timepoints,
                                       time_interval_s=time_interval,
                                       channel_group='Channel - LS', 
                                       channels=ls_channels,
                                       z_start=AP_galvo_start,
                                       z_end=AP_galvo_end,
                                       z_step=AP_galvo_step,
                                       order='tpcz')

#%% Setup DAQ

ls_framerate = 1000 / (ls_exposure_ms + ls_readout_time_ms + LS_POST_READOUT_DELAY)

# Ctr1 triggers LS camera
Ctr1 = nidaqmx.Task('Counter1')
ctr1 = Ctr1.co_channels.add_co_pulse_chan_freq('cDAQ1/_ctr1', freq=ls_framerate, duty_cycle=0.1)
# Ctr1 timing is now set in the prep_daq_counter hook function
# Ctr1.timing.cfg_implicit_timing(sample_mode=AcquisitionType.FINITE, samps_per_chan=80)
# Ctr1.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source='/cDAQ1/PFI0', trigger_edge=Slope.RISING)
ctr1.co_pulse_term = '/cDAQ1/PFI1'

#%% Acquire data

def prep_daq_counter(events):
    event_seq_length = len(events)
    if _verbose:
        print(f'Running pre-hardware hook function. Sequence length: {len(event_seq_length)}')
    # Counter task needs to be stopped before it is restarted
    if Ctr1.is_task_done():
        Ctr1.stop()
    Ctr1.timing.cfg_implicit_timing(sample_mode=AcquisitionType.FINITE, samps_per_chan=event_seq_length)
    return events

def start_daq_counter(events):
    if _verbose:
        print('Running post camera hook fun. Starting DAC Ctr1.')
    Ctr1.start()
    return events


#%%
with Acquisition(directory=save_path, name='ls_acq', port=PORT1, 
                 pre_hardware_hook_fn=prep_daq_counter,
                 post_camera_hook_fn=start_daq_counter,
                 show_display=False) as acq2:
    acq2.acquire(ls_events)

print('Acquisition completed!\n\n\n')

#%% Acquire data v1
# acq2 = Acquisition(directory=save_path, name='ls_acq', port=PORT1,
#                    show_display=False)

# print('Starting acquisition')
# acq2.acquire(ls_events)
# time.sleep(1)  # give time for cameras to setup acquisition

# print('Starting trigger sequence')
# Ctr1.start()

# print('Marking acquisition as finished')
# acq2.mark_finished()

# print('Waiting for acquisition to finish')
# acq2.await_completion(); print('Acq2 finished')

# print('Stop counters')
# Ctr1.stop()


#%% Reset acquisition

# Close counters
Ctr1.close()

mmc2.set_position('AP Galvo', 0)
# mmc2.set_property('TS2_TTL1-8', 'Blanking', 'Off')
mmc2.set_property('Prime BSI Express', 'TriggerMode', 'Internal Trigger')
