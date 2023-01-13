#%% import modules
# mantis conda env: acq

import os
import time
import numpy as np
from pycromanager import start_headless, Core, Acquisition, multi_d_acquisition_events

import nidaqmx
from nidaqmx.constants import AcquisitionType, Slope
from nidaqmx.types import CtrTime

#%% Connect to running MM on default port 4827
# MM config: mantis-LF.cfg

PORT1 = 4827
mmc1 = Core(port=PORT1)

print(mmc1)

#%% Start and connect to headless MM on port 5827

mm_app_path = r'C:\\Program Files\\Micro-Manager-2.0_09_01_2022'
config_file = r'C:\\CompMicro_MMConfigs\\mantis\\mantis-LS.cfg'

# Start the Java process
PORT2 = 5827
start_headless(mm_app_path, config_file, port=PORT2)  # we need to space out port numbers a bit

mmc2 = Core(port=PORT2)
print(mmc2)

#%% Setup DAQ

# Ctr0 triggers LF camera
Ctr0 = nidaqmx.Task('Counter0')
ctr0 = Ctr0.co_channels.add_co_pulse_chan_freq('cDAQ1/_ctr0', freq=20.0, duty_cycle=0.1)
Ctr0.timing.cfg_implicit_timing(sample_mode=AcquisitionType.FINITE, samps_per_chan=100)
ctr0.co_pulse_term = '/cDAQ1/PFI0'

# Ctr1 triggers LS camera
Ctr1 = nidaqmx.Task('Counter1')
ctr1 = Ctr1.co_channels.add_co_pulse_chan_freq('cDAQ1/_ctr1', freq=10.0, duty_cycle=0.1)
Ctr1.timing.cfg_implicit_timing(sample_mode=AcquisitionType.FINITE, samps_per_chan=50)
Ctr1.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source='/cDAQ1/PFI0', trigger_edge=Slope.RISING)
ctr1.co_pulse_term = '/cDAQ1/PFI1'

#%% Setup acquisition

# setup label-free acquisition on mmc1
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

mmc1.set_property('TS1_DAC06', 'Sequence', 'Off') # turn off sequencing to avoid PM bugs

mmc1.set_exposure(10)

# setup light sheet acquisition on mmc2
mmc2.set_config('Channel - LS', 'GFP EX488 EM525-45')

# One frame is acquired for every trigger pulse
mmc2.set_property('Prime BSI Express', 'TriggerMode', 'Edge Trigger')
# Rolling Shutter Exposure Out mode is high when all rows are exposing
mmc2.set_property('Prime BSI Express', 'ExposeOutMode', 'Rolling shutter')

mmc2.set_property('Core', 'Focus', 'AP Galvo')

mmc2.set_property('TS2_DAC03', 'Sequence', 'Off') # turn off sequencing to avoid PM bugs

mmc2.set_exposure(80)
prime_bsi_readout_time_ms = np.ceil(float(mmc2.get_property('Prime BSI Express', 'Timing-ReadoutTimeNs'))*1e-6)

#%% Setup acquisition events

lf_events = multi_d_acquisition_events(num_time_points = 100)
print(len(lf_events))

ls_events = multi_d_acquisition_events(num_time_points = 50)
print(len(ls_events))

#%% Acquire data

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

#%% Reset acquisition

# Close counters
Ctr0.close()
Ctr1.close()

mmc1.set_property('Oryx', 'Trigger Mode', 'Off')
mmc2.set_property('Prime BSI Express', 'TriggerMode', 'Internal Trigger')

mmc1.set_property('Oryx', 'Frame Rate Control Enabled', oryx_framerate_enabled)
if oryx_framerate_enabled == '1': 
    mmc1.set_property('Oryx', 'Frame Rate', oryx_framerate)