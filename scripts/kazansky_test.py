#%% import libraries

import numpy as np
from pycromanager import Bridge, Acquisition, multi_d_acquisition_events

#%% initialize bridge

bridge = Bridge()
mmc = bridge.get_core()
mmStudio = bridge.get_studio()

#%% define imaging settings

data_directory = r'D:\2022_07_12 kazansky test'
data_name = 'pm_kazansky_test'

# LF channels, 5-state algorithm
lf_channels = [{'group': 'Master Channel', 'config': f'State{i}'} for i in range(5)]
lf_exposure = 5

# relative z, acquired using piezo stage
z_start = -10
z_end = 10
z_step = 0.25
z_range = np.arange(z_start, z_end + z_step, z_step)

num_time_points = 1
time_interval_s = 0  # in seconds

#%% prepare for acquisition

# setup Oryx camera
mmc.set_property('Oryx', 'Frame Rate Control Enabled', '1')
mmc.set_property('Oryx', 'Frame Rate', '10')
mmc.set_property('Oryx', 'Pixel Format', 'Mono12p')
# mmc.set_roi('Oryx', 100, 0, 1024, 1024)

# setup TriggerScope
mmc.set_property('TS_DAC01', 'Sequence', 'On')
mmc.set_property('TS_DAC02', 'Sequence', 'On')
mmc.set_property('TS_DAC05', 'Sequence', 'On')

# set Focus device to piezo drive
mmc.set_property('Core', 'Focus', 'PiezoStage:Q:35')

# setup autofocus
autofocus_manager = mmStudio.get_autofocus_manager()
autofocus_manager.set_autofocus_method_by_name('PFS')

#%% generate acquisition events

events = []
for t_idx in range(num_time_points):

    # lf positions, channel first - channel sequencing does not work
    for z_idx, z in enumerate(z_range):
        for channel in lf_channels:
            events.append({
                'axes': {'time': t_idx, 'z': z_idx},
                'min_start_time': t_idx * time_interval_s,
                'z': z,
                'channel': channel,
                'exposure': lf_exposure,
            })

#%%  acquire data

# turn on autoshutter
mmc.set_auto_shutter(True)

# turn off live preview
snap_live_manager = mmStudio.get_snap_live_manager()
if snap_live_manager.is_live_mode_on():
    snap_live_manager.set_live_mode_on(False)

# acquire data
with Acquisition(directory=data_directory, name=data_name) as acq:
    acq.acquire(events, keep_shutter_open=True)
