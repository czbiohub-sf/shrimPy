#%% import modules

import nidaqmx
from nidaqmx.constants import AcquisitionType, Slope
from nidaqmx.types import CtrTime

#%%
lf_exposure_ms = 10
mcl_step_time_ms = 1.5
lc_change_time_ms = 20

num_slices = 60

#%%
lf_z_freq = 1000 / (lf_exposure_ms + mcl_step_time_ms)
lf_channel_freq = 1 / (num_slices/lf_z_freq + lc_change_time_ms/1000)

#%%
# LF channel counter
lf_channel_ctr_task = nidaqmx.Task('LF Channel Counter')
lf_channel_ctr = lf_channel_ctr_task.co_channels.add_co_pulse_chan_freq('cDAQ1/_ctr0', freq=lf_channel_freq, duty_cycle=0.1)
lf_channel_ctr_task.timing.cfg_implicit_timing(sample_mode=AcquisitionType.FINITE, samps_per_chan=5)
lf_channel_ctr.co_pulse_term = '/cDAQ1/Ctr0InternalOutput'

# LF Z counter
lf_z_ctr_task = nidaqmx.Task('LF Z Counter')
lf_z_ctr = lf_z_ctr_task.co_channels.add_co_pulse_chan_freq('cDAQ1/_ctr1', freq=lf_z_freq, duty_cycle=0.1)
lf_z_ctr_task.timing.cfg_implicit_timing(sample_mode=AcquisitionType.FINITE, samps_per_chan=num_slices)
lf_z_ctr.co_pulse_term = '/cDAQ1/PFI0'
lf_z_ctr_task.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source='/cDAQ1/Ctr0InternalOutput', trigger_edge=Slope.RISING)
lf_z_ctr_task.triggers.start_trigger.retriggerable = True

#%%
lf_z_ctr_task.start()
lf_channel_ctr_task.start()

lf_channel_ctr_task.wait_until_done(timeout=5*lf_channel_freq+10)
# lf_z_ctr_task.wait_until_done() # don't use since task is retriggarable?

lf_z_ctr_task.close()
lf_channel_ctr_task.close()
