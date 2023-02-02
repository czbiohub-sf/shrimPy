#%% 
#  Import modules
######################

# mantis conda env: acq_pm_latest

import os
import time
import numpy as np
from pycromanager import start_headless, Core, Acquisition, multi_d_acquisition_events

import nidaqmx
from nidaqmx.constants import AcquisitionType, Slope
from nidaqmx.types import CtrTime


#%% 
# Define constants
######################

_verbose = False
PORT1 = 4827
PORT2 = 5827   # we need to space out port numbers a bit
LS_POST_READOUT_DELAY = 0.05  # in ms
LS_ROI = (0, 896, 2048, 256)  # centered in FOV

mm_app_path = r'C:\\Program Files\\Micro-Manager-nightly'
config_file = r'C:\\CompMicro_MMConfigs\\mantis\\mantis-LS.cfg'


#%% 
# Set acquisition parameters
######################

save_path = r'D:\\temp'

n_timepoints = 3
time_interval = 30  # in seconds

ls_exposure_ms = 10
lf_exposure_ms = 10

AP_galvo_start = -2/10  # in Volts
AP_galvo_end = 2/10
AP_galvo_step = 0.01

MCL_piezo_start = 0
MCL_piezo_end = 60
MCL_piezo_step = 1

ls_channel_group = 'Channel - LS'
ls_channels = ['GFP EX488 EM525-45', 'mCherry EX561 EM600-37']

lf_channel_group = 'Channel - LF'
lf_channels = [f'State{i}' for i in range(5)]


#%% 
# Connect to running MM on PORT1
######################

# MM config: mantis-LF.cfg
mmc1 = Core(port=PORT1)


#%% 
# Start and connect to headless MM on PORT2
######################

# Start the Java process
start_headless(mm_app_path, config_file, port=PORT2)
mmc2 = Core(port=PORT2)


#%% 
# Setup label-free acquisition on mmc1
######################

mmc1.set_config('Imaging Path', 'Label-free')
mmc1.set_config('Channel - LS', 'External Control')
mmc1.set_config('Channel - LF', 'State0')

mmc1.set_property('Oryx', 'Line Selector', 'Line5'); mmc1.update_system_state_cache()
mmc1.set_property('Oryx', 'Line Mode', 'Output')
mmc1.set_property('Oryx', 'Line Source', 'ExposureActive')
mmc1.set_property('Oryx', 'Line Selector', 'Line2'); mmc1.update_system_state_cache()
mmc1.set_property('Oryx', 'Line Mode', 'Input')
mmc1.set_property('Oryx', 'Trigger Source', 'Line2')
mmc1.set_property('Oryx', 'Trigger Mode', 'On')
oryx_framerate_enabled = mmc1.get_property('Oryx', 'Frame Rate Control Enabled')
oryx_framerate = mmc1.get_property('Oryx', 'Frame Rate')
mmc1.set_property('Oryx', 'Frame Rate Control Enabled', '0')

mmc1.set_property('Core', 'Focus', 'MCL Piezo')

mmc1.set_property('TS1_DAC01', 'Sequence', 'On')
mmc1.set_property('TS1_DAC02', 'Sequence', 'On')
mmc1.set_property('TS1_DAC06', 'Sequence', 'On')

mmc1.set_exposure(lf_exposure_ms)


#%% 
# Setup light sheet acquisition on mmc2
######################

# Set ROI
mmc2.set_roi(*LS_ROI)

roi = mmc2.get_roi()
ls_roi = [roi.x, roi.y, roi.width, roi.height]
assert ls_roi[3] < 300, 'Please crop camera sensor for light-sheet acquisition'

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


#%% 
# Setup acquisition events
######################

lf_events = multi_d_acquisition_events(num_time_points=n_timepoints,
                                       time_interval_s=time_interval,
                                       channel_group=lf_channel_group, 
                                       channels=lf_channels,
                                       z_start=MCL_piezo_start,
                                       z_end=MCL_piezo_end,
                                       z_step=MCL_piezo_step,
                                       order='tpcz')

ls_events = multi_d_acquisition_events(num_time_points=n_timepoints,
                                       time_interval_s=time_interval,
                                       channel_group=ls_channel_group, 
                                       channels=ls_channels,
                                       z_start=AP_galvo_start,
                                       z_end=AP_galvo_end,
                                       z_step=AP_galvo_step,
                                       order='tpcz')


#%% 
# Setup DAQ
######################

ls_framerate = 1000 / (ls_exposure_ms + ls_readout_time_ms + LS_POST_READOUT_DELAY)

# Ctr0 triggers LF camera
Ctr0 = nidaqmx.Task('Counter0')
ctr0 = Ctr0.co_channels.add_co_pulse_chan_freq('cDAQ1/_ctr0', freq=20.0, duty_cycle=0.1)
Ctr0.timing.cfg_implicit_timing(sample_mode=AcquisitionType.FINITE, samps_per_chan=100)
ctr0.co_pulse_term = '/cDAQ1/PFI0'

# Ctr1 triggers LS camera
Ctr1 = nidaqmx.Task('Counter1')
ctr1 = Ctr1.co_channels.add_co_pulse_chan_freq('cDAQ1/_ctr1', freq=ls_framerate, duty_cycle=0.1)
# Ctr1 timing is now set in the prep_daq_counter hook function
# Ctr1.timing.cfg_implicit_timing(sample_mode=AcquisitionType.FINITE, samps_per_chan=50)
Ctr1.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source='/cDAQ1/PFI0', trigger_edge=Slope.RISING)
ctr1.co_pulse_term = '/cDAQ1/PFI1'


#%% 
# Acquire data
######################

save_path = r'D:\2022_11_18 automation'

acq1 = Acquisition(directory=save_path, name='lf_acq', port=PORT1)
acq2 = Acquisition(directory=save_path, name='ls_acq', port=PORT2)

print('Starting acquisition')
acq1.acquire(lf_events)
acq2.acquire(ls_events)
time.sleep(1)  # give time for cameras to setup acquisition

print('Starting trigger sequence')
Ctr1.start()
Ctr0.start()

print('Marking acquisition as finished')
acq1.mark_finished()
acq2.mark_finished()

print('Waiting for acquisition to finish')
acq1.await_completion(); print('Acq1 finished')
acq2.await_completion(); print('Acq2 finished')

print('Stop counters')
Ctr0.stop()
Ctr1.stop()


#%% 
# Reset acquisition
######################

# Close counters
Ctr0.close()
Ctr1.close()

mmc1.set_property('Oryx', 'Trigger Mode', 'Off')
mmc2.set_property('Prime BSI Express', 'TriggerMode', 'Internal Trigger')

mmc1.set_property('Oryx', 'Frame Rate Control Enabled', oryx_framerate_enabled)
if oryx_framerate_enabled == '1': 
    mmc1.set_property('Oryx', 'Frame Rate', oryx_framerate)