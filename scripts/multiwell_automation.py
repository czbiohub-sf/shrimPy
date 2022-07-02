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

# set Focus device to piezo drive
mmc.set_property('Core', 'Focus', 'PiezoStage:Q:35')
# mmc.set_property('Core', 'Focus', 'ZDrive')

# setup autofocus
autofocus_manager = mmStudio.get_autofocus_manager()
autofocus_manager.set_autofocus_method_by_name('PFS')


#%% define acquisition events

events = multi_d_acquisition_events(num_time_points=3,
                                    time_interval_s=30,
                                    z_start=z_start,
                                    z_step=z_step,
                                    z_end=z_end,
                                    xy_positions=xy_position_list,
                                    order='tpcz',
                                    keep_shutter_open_between_z_steps=True)

#%% autofocus function

def autofocus_fn(event, bridge, event_queue):
    print(event)

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

            # apply autofocus
            z_old = mmc.get_position('ZDrive')
            print(f'Position before autofocus: {z_old}')

            # autofocus
            try:
                mmc.full_focus()
            except:
                mmc.set_relative_position(5)  # try moving up
                try:
                    mmc.full_focus()
                except:
                    mmc.set_relative_position(-10)  # try moving down
                    mmc.full_focus()

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
with Acquisition(directory=r'D:\2022_06_30 automation', name='beads_PM', post_hardware_hook_fn=autofocus_fn) as acq:
    acq.acquire(events)
