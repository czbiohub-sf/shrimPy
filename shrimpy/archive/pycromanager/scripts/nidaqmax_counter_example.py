#%% import modules

import nidaqmx
from nidaqmx.constants import AcquisitionType, Slope
from nidaqmx.types import CtrTime

#%% single counter example

with nidaqmx.Task() as task:
    ctr0 = task.co_channels.add_co_pulse_chan_freq('cDAQ1/_ctr0', freq=5.0, duty_cycle=0.1)
    task.timing.cfg_implicit_timing(sample_mode=AcquisitionType.FINITE, samps_per_chan=20)
    ctr0.co_pulse_term = '/cDAQ1/PFI0'
    task.start()
    task.wait_until_done()

#%% two synced timers

Ctr0 = nidaqmx.Task('Counter0')
ctr0 = Ctr0.co_channels.add_co_pulse_chan_freq('cDAQ1/_ctr0', freq=5.0, duty_cycle=0.1)
Ctr0.timing.cfg_implicit_timing(sample_mode=AcquisitionType.FINITE, samps_per_chan=10)
ctr0.co_pulse_term = '/cDAQ1/PFI0'

Ctr1 = nidaqmx.Task('Counter1')
ctr1 = Ctr1.co_channels.add_co_pulse_chan_freq('cDAQ1/_ctr1', freq=10.0, duty_cycle=0.1)
Ctr1.timing.cfg_implicit_timing(sample_mode=AcquisitionType.FINITE, samps_per_chan=20)
Ctr1.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source='/cDAQ1/PFI0', trigger_edge=Slope.RISING)
ctr1.co_pulse_term = '/cDAQ1/PFI1'

Ctr1.start()
Ctr0.start()

Ctr0.wait_until_done()
Ctr1.wait_until_done()

Ctr0.close()
Ctr1.close()