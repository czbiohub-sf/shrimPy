#%% import libraries

import numpy as np
from pycromanager import Bridge, Acquisition, multi_d_acquisition_events

#%% initialize bridge

bridge = Bridge()
mmc = bridge.get_core()
mmStudio = bridge.get_studio()

#%% import positions from MM

mm_pos_list = mmStudio.get_position_list_manager().get_position_list()
number_of_positions = mm_pos_list.get_number_of_positions()

xy_position_list = [[mm_pos_list.get_position(i).get_x(),
                     mm_pos_list.get_position(i).get_y()]
                    for i in range(number_of_positions)]

z_position_list = [mm_pos_list.get_position(i).get('ZDrive').get1_d_position()
                   for i in range(number_of_positions)]

#%% define imaging settings

# fluorescence channels
epi_channels = [{'group': 'Master Channel', 'config': 'Epi-GFP'},
                {'group': 'Master Channel', 'config': 'Epi-DSRED'}]
epi_exposure = [100, 100]

# LF channels, 5-state algorithm
lf_channels = [{'group': 'Master Channel', 'config': f'State{i}'} for i in range(5)]
lf_exposure = 5

# relative z, acquired using piezo stage
z_start = -3
z_end = 3
z_step = 0.25
z_range = np.arange(z_start, z_end + z_step, z_step)

num_time_points = 5
time_interval_s = 20*60  # in seconds

#%% prepare for acquisition

# set Focus device to piezo drive
mmc.set_property('Core', 'Focus', 'PiezoStage:Q:35')

# setup autofocus
autofocus_manager = mmStudio.get_autofocus_manager()
autofocus_manager.set_autofocus_method_by_name('PFS')

#%% generate acquisition events

events = []

for t_idx in range(num_time_points):
    for p_idx, p in enumerate(xy_position_list):

        # epi channels, z-first
        for channel, exp in zip(epi_channels, epi_exposure):
            for z_idx, z in enumerate(z_range):
                events.append({
                    'axes': {'time': t_idx, 'position': p_idx, 'z': z_idx},
                    'min_start_time': t_idx * time_interval_s,
                    'x': p[0],
                    'y': p[1],
                    'z': z,
                    'channel': channel,
                    'exposure': exp,
                    'keep_shutter_open': True,
                })

    # epi and lf positions
    for p_idx, p in enumerate(lf_epi_positions):

        # epi channel, z-first
        for z_idx, z in enumerate(z_range + p[2]):
            events.append(
                {
                    'axes': {'time': t, 'position': p_idx + num_epi_positions, 'z': z_idx},
                    'min_start_time': t * time_interval_s,
                    'x': p[0],
                    'y': p[1],
                    'z': z,
                    'channel': epi_channel,
                    'exposure': epi_exposure,
                    'keep_shutter_open': False
                })

        # lf channels, channel-first
        for z_idx, z in enumerate(z_range + p[2]):
            for ch in lf_channels:
                events.append(
                    {
                        'axes': {'time': t, 'position': p_idx + num_epi_positions, 'z': z_idx},
                        'min_start_time': t * time_interval_s,
                        'x': p[0],
                        'y': p[1],
                        'z': z,
                        'channel': ch,
                        'exposure': lf_exposure,
                        'keep_shutter_open': False
                    })