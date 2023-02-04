#%% 
#  Import modules
######################

# mantis conda env: acq_pm_latest

import os
import time
import numpy as np
from pycromanager import start_headless, Core, Acquisition, multi_d_acquisition_events

from functools import partial
from hook_functions.daq_control import confirm_num_daq_counter_samples, start_daq_counter
from util.convenience import get_z_range

import nidaqmx
from nidaqmx.constants import AcquisitionType, Slope
from nidaqmx.types import CtrTime


#%% 
# Define constants
######################

_verbose = True
PORT1 = 4827
PORT2 = 5827   # we need to space out port numbers a bit
LS_POST_READOUT_DELAY = 0.05  # in ms
LS_ROI = (0, 896, 2048, 256)  # centered in FOV
MCL_STEP_TIME = 1.5  # in ms
LC_CHANGE_TIME = 20  # in ms

mm_app_path = r'C:\\Program Files\\Micro-Manager-nightly'
config_file = r'C:\\CompMicro_MMConfigs\\mantis\\mantis-LS.cfg'


#%% 
# Set acquisition parameters
######################

save_path = r'D:\\temp'

n_timepoints = 1
time_interval = 0  # in seconds

ls_exposure_ms = 10
lf_exposure_ms = 10

AP_galvo_start = -2/10  # in Volts
AP_galvo_end = 2/10
AP_galvo_step = 0.01
AP_galvo_range = get_z_range(AP_galvo_start, AP_galvo_end, AP_galvo_step)

MCL_piezo_start = 0
MCL_piezo_end = 60
MCL_piezo_step = 5
MCL_piezo_range = get_z_range(MCL_piezo_start, MCL_piezo_end, MCL_piezo_step)

ls_channel_group = 'Channel - LS'
ls_channels = ['GFP EX488 EM525-45']
# ls_channels = ['GFP EX488 EM525-45', 'mCherry EX561 EM600-37']

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
mmc1.set_property('Oryx', 'Trigger Overlap', 'ReadOut')  # required for external triggering at max frame rate
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
ls_readout_time_ms = np.around(
    float(mmc2.get_property('Prime BSI Express', 'Timing-ReadoutTimeNs'))*1e-6, 
    decimals=3)
assert ls_readout_time_ms < ls_exposure_ms, \
    f'Exposure time needs to be greater than the {ls_readout_time_ms} sensor readout time'


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

# TODO: make sure we're not running faster than the max camera framerate
ls_framerate = 1000 / (ls_exposure_ms + ls_readout_time_ms + LS_POST_READOUT_DELAY)
ls_num_slices = len(AP_galvo_range)

lf_num_slices = len(MCL_piezo_range)
lf_num_channels = len(lf_channels)
lf_z_freq = 1000 / (lf_exposure_ms + MCL_STEP_TIME)
lf_channel_freq = 1 / (lf_num_slices/lf_z_freq + LC_CHANGE_TIME/1000)

# LF channel trigger - accommodates longer LC switching times
lf_channel_ctr_task = nidaqmx.Task('LF Channel Counter')
lf_channel_ctr = lf_channel_ctr_task.co_channels.add_co_pulse_chan_freq(
    'cDAQ1/_ctr0', 
    freq=lf_channel_freq, 
    duty_cycle=0.1)
lf_channel_ctr_task.timing.cfg_implicit_timing(
    sample_mode=AcquisitionType.FINITE, 
    samps_per_chan=lf_num_channels)
lf_channel_ctr.co_pulse_term = '/cDAQ1/Ctr0InternalOutput'

# LF Z trigger
lf_z_ctr_task = nidaqmx.Task('LF Z Counter')
lf_z_ctr = lf_z_ctr_task.co_channels.add_co_pulse_chan_freq(
    'cDAQ1/_ctr1', 
    freq=lf_z_freq, 
    duty_cycle=0.1)
lf_z_ctr_task.timing.cfg_implicit_timing(
    sample_mode=AcquisitionType.FINITE, 
    samps_per_chan=lf_num_slices)
lf_z_ctr_task.triggers.start_trigger.cfg_dig_edge_start_trig(
    trigger_source='/cDAQ1/Ctr0InternalOutput', 
    trigger_edge=Slope.RISING)
lf_z_ctr_task.triggers.start_trigger.retriggerable = True  # will always return is_task_done = False after counter is started
lf_z_ctr.co_pulse_term = '/cDAQ1/PFI0'

# LS frame trigger
ls_ctr_task = nidaqmx.Task('LS Frame Counter')
ls_ctr = ls_ctr_task.co_channels.add_co_pulse_chan_freq(
    'cDAQ1/_ctr2', 
    freq=ls_framerate, 
    duty_cycle=0.1)
ls_ctr_task.timing.cfg_implicit_timing(
    sample_mode=AcquisitionType.FINITE, 
    samps_per_chan=ls_num_slices)
ls_ctr_task.triggers.start_trigger.cfg_dig_edge_start_trig(
    trigger_source='/cDAQ1/Ctr0InternalOutput', 
    trigger_edge=Slope.RISING)
ls_ctr.co_pulse_term = '/cDAQ1/PFI1'

#%% 
# Acquire data v2
######################

# LF acquisition
lf_acq = Acquisition(
    directory=save_path, 
    name='lf_acq', 
    port=PORT1,
    pre_hardware_hook_fn=partial(
        confirm_num_daq_counter_samples, 
        [lf_z_ctr_task, lf_channel_ctr_task], 
        lf_num_slices*lf_num_channels, 
        _verbose),
    post_hardware_hook_fn=None  # autofocus
    post_camera_hook_fn=partial(
        start_daq_counter, 
        [lf_z_ctr_task, lf_channel_ctr_task],  # lf_z_ctr_task needs to be started first
        _verbose),
    image_saved_fn=None  # data processing and display
    show_display=False
)

# LS acquisition
ls_acq = Acquisition(
    directory=save_path, 
    name='ls_acq', 
    port=PORT2, 
    pre_hardware_hook_fn=partial(
        confirm_num_daq_counter_samples, 
        ls_ctr_task, 
        ls_num_slices, 
        _verbose), 
    post_camera_hook_fn=partial(
        start_daq_counter, 
        ls_ctr_task, 
        _verbose), 
    show_display=False
)

print('Starting acquisition')
ls_acq.acquire(ls_events)  # it's important to start the LS acquisition first
lf_acq.acquire(lf_events)

if _verbose:
    print('Marking acquisition as finished')
ls_acq.mark_finished()
lf_acq.mark_finished()

if _verbose:
    print('Waiting for acquisition to finish')
ls_acq.await_completion(); print('LS finished')
lf_acq.await_completion(); print('LF finished')

print('Acquisition completed.')

#%% 
# Reset acquisition
######################

if _verbose:
    print('Stop counters')
ls_ctr_task.stop()
lf_z_ctr_task.stop()
lf_channel_ctr_task.stop()

# Close counters
ls_ctr_task.close()
lf_z_ctr_task.close()
lf_channel_ctr_task.close()

#%%

if _verbose:
    print('Resetting microscope hardware')
mmc1.set_property('Oryx', 'Trigger Mode', 'Off')
mmc2.set_property('Prime BSI Express', 'TriggerMode', 'Internal Trigger')

mmc1.set_property('Oryx', 'Frame Rate Control Enabled', oryx_framerate_enabled)
if oryx_framerate_enabled == '1': 
    mmc1.set_property('Oryx', 'Frame Rate', oryx_framerate)


# #%% 
# # Acquire data v1
# ######################

# # LF acquisition
# lf_acq = Acquisition(
#     directory=save_path, 
#     name='lf_acq', 
#     port=PORT1,
#     show_display=False
# )

# # LS acquisition
# ls_acq = Acquisition(
#     directory=save_path, 
#     name='ls_acq', 
#     port=PORT2, 
#     show_display=False
# )

# print('Starting acquisition')
# ls_acq.acquire(ls_events)  # it's important to start the LS acquisition first
# lf_acq.acquire(lf_events)
# time.sleep(1)

# lf_z_ctr_task.start()
# ls_ctr_task.start()
# lf_channel_ctr_task.start()  # triggers other two

# print('Marking acquisition as finished')
# ls_acq.mark_finished()
# lf_acq.mark_finished()

# print('Waiting for acquisition to finish')
# ls_acq.await_completion(); print('LS finished')
# lf_acq.await_completion(); print('LF finished')

# #%% 
# # Reset acquisition
# ######################
# ls_ctr_task.stop()
# lf_z_ctr_task.stop()
# lf_channel_ctr_task.stop()