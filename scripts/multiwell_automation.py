#%% import libraries
import time

import numpy as np
from pycromanager import Bridge, Acquisition, multi_d_acquisition_events

#%% initialize bridge

bridge = Bridge()
mmc = bridge.get_core()
mmStudio = bridge.get_studio()

#%% import positions from MM

mm_pos_list = mmStudio.get_position_list_manager().get_position_list()
number_of_positions = mm_pos_list.get_number_of_positions()

xyz_position_list = [[mm_pos_list.get_position(i).get_x(),
                      mm_pos_list.get_position(i).get_y(),
                      mm_pos_list.get_position(i).get('ZDrive').get1_d_position()]
                     for i in range(number_of_positions)]

xy_position_list = [[mm_pos_list.get_position(i).get_x(),
                     mm_pos_list.get_position(i).get_y()]
                    for i in range(number_of_positions)]

z_position_list = [mm_pos_list.get_position(i).get('ZDrive').get1_d_position()
                   for i in range(number_of_positions)]
#%% define imaging settings

# relative z
z_start = -3
z_end = 3
z_step = 0.25
z_range = np.arange(z_start, z_end + z_step, z_step)
num_time_points = 1
time_interval_s = 30

#to save the data
data_directory = r'D:\2022_07_14 automation testing'
data_name = 'BPAE_multi-channel_automation'

# set Focus device to piezo drive
mmc.set_property('Core', 'Focus', 'PiezoStage:Q:35')
# mmc.set_property('Core', 'Focus', 'ZDrive')

# setup autofocus
autofocus_manager = mmStudio.get_autofocus_manager()
autofocus_manager.set_autofocus_method_by_name('PFS')

# set XY stage speed
mmc.set_property('XYStage:XY:31', 'MotorSpeedX-S(mm/s)', '3')
mmc.set_property('XYStage:XY:31', 'MotorSpeedY-S(mm/s)', '3')

#set up channels here
# fluorescence channels
epi_channels = [{'group': 'Master Channel', 'config': 'Epi-GFP'},
                {'group': 'Master Channel', 'config': 'Epi-DSRED'}]
epi_exposure = [150, 80]  # in ms

# LF channels, 5-state algorithm
lf_channels = [{'group': 'Master Channel', 'config': f'State{i}'} for i in range(5)]
lf_exposure = 5

#%% define acquisition events

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
                    'exposure': exp
                })
            # lf positions, channel first - channel sequencing does not work
        for z_idx, z in enumerate(z_range):
            for channel in lf_channels:
                events.append({
                    'axes': {'time': t_idx, 'position': p_idx, 'z': z_idx},
                    'min_start_time': t_idx * time_interval_s,
                    'x': p[0],
                    'y': p[1],
                    'z': z,
                    'channel': channel,
                    'exposure': lf_exposure,
                    })

#%% autofocus function

def autofocus_fn(event, bridge, event_queue):

    if 'axes' in event.keys():  # some events only modify hardware
        if not hasattr(autofocus_fn, 'pos_index'):
            autofocus_fn.pos_index = -1

        # Only run autofocus at new positions (including the first one)
        pos_index = event['axes']['position']
        if pos_index != autofocus_fn.pos_index:
            # get the Micro-Manager core
            mmc = bridge.get_core()

            # apply ZDrive position
            mmc.set_position('ZDrive', z_position_list[pos_index])
            time.sleep(2)  # wait for oil to catch up

            # apply autofocus
            z_old = mmc.get_position('ZDrive')
            print(f'Position before autofocus: {z_old}')

            # autofocus
            print('Calling autofocus')
            try:
                mmc.full_focus()
            except:
                print('First try failed. Move +5 um and retry')
                mmc.set_relative_position('ZDrive', 5)  # try moving up
                try:
                    mmc.full_focus()
                except:
                    print('Second try failed. Move -10 um and retry')
                    mmc.set_relative_position('ZDrive', -10)  # try moving down
                    try:
                        mmc.full_focus()
                    except:
                        print('Autofocus failed')
                    else:
                        print('Autofocus engaged!!')
                else:
                    print('Autofocus engaged!!')
            else:
                print('Autofocus engaged!!')

            z_new = mmc.get_position('ZDrive')
            print(f'Position after autofocus: {z_new}')

        # Keep track of last time point on which autofocus ran
        autofocus_fn.pos_index = pos_index

    return event

#%% start acquisition

# turn off Live Preview if it is running
snap_live_manager = mmStudio.get_snap_live_manager()
if snap_live_manager.is_live_mode_on():
    snap_live_manager.set_live_mode_on(False)

# acquire data
with Acquisition(directory=data_directory, name=data_name, post_hardware_hook_fn=autofocus_fn) as acq:
    acq.acquire(events)
