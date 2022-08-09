from pycromanager import Acquisition, Bridge, Dataset, multi_d_acquisition_events
import numpy as np
from skimage.registration import phase_cross_correlation
from copy import deepcopy
import time

#%%
bridge = Bridge()
mmc = bridge.get_core()
mmStudio = bridge.get_studio()

# set Focus device to piezo drive
mmc.set_property('Core', 'Focus', 'PiezoStage:Q:35')

#%%
data_directory = r'E:\2022_08_08 tracker testing'
data_name = 'test'

mm_pos_list = mmStudio.get_position_list_manager().get_position_list()
number_of_positions = mm_pos_list.get_number_of_positions()

xy_position_list = [[mm_pos_list.get_position(i).get_x(),
                     mm_pos_list.get_position(i).get_y()]
                    for i in range(number_of_positions)]

# define the number of time points, it will run through this time point and then stop
n_time = 3

pixel_size = 6.9/100  # in um

# relative z, acquired using piezo stage
z_start = -5
z_end = 5
z_step = 0.8
z_range = np.arange(z_start, z_end + z_step, z_step)

epi_channels = [{'group': 'Master Channel', 'config': 'Epi-GFP'},
                {'group': 'Master Channel', 'config': 'Epi-DSRED'}]
epi_exposure = [150, 100]
pos_init = xy_position_list

#%%
events = []
for channel, exp in zip(epi_channels, epi_exposure):
    for index, z_um in enumerate(z_range):
            evt = {
            'axes' : {'z': index, 'time': 0, 'position': 0},
            'x' : pos_init[0][0],
            'y' : pos_init[0][1],
            'z' : z_um,
            'channel': channel,
            'exposure': exp
                }
            events.append(evt)

#%%
def img_process_fn(image, metadata, bridge, event_queue):
    # mmc = bridge.get_core()
   
    time_index = metadata['Axes']['time']
    print(f'Time index: {time_index}')

    # accumulate the images
    if not hasattr(img_process_fn, "ref_images"):
        img_process_fn.ref_images = []
        img_process_fn.curr_images = []
        
    # end the acquisition if we get through the number of time points
    if time_index >= n_time:
        event_queue.put(None)
        global acq_running
        acq_running = False
    else:
        z_index = metadata['Axes']['z']
        #use DAPI channel for correlation
        channel = metadata['Channel']
        print(f'Channel: {channel}')
        if channel == 'Epi-GFP':
            img_process_fn.curr_images.append(image)
            print('GFP channel added')
       
        # if we're at the end of the z stack
        z_steps = len(z_range)
        if z_index == z_steps:
            print('Entering z_index == len(z_range)-1')
            # if we're in the last channel, DSRED in this case
            #if channel == 'Epi-DSRED':
            if channel == epi_channels[-1]:
                print('Entering channel == Epi-DSRED')
                if time_index > 1:
                    # print('Setting rel xy position to (-5, 0)')
                    # mmc.set_relative_xy_position(-5, 0)
                    # time.sleep(1)

                    print('Entering time_index > 1')
                    x_pos = metadata['XPosition_um_Intended']
                    y_pos = metadata['YPosition_um_Intended']
                    # corr = phase_cross_correlation(ref_image,curr_image)
                    print(len(img_process_fn.ref_images))
                    print(len(img_process_fn.curr_images))
                    corr = phase_cross_correlation(np.stack(img_process_fn.ref_images),
                                                   np.stack(img_process_fn.curr_images))
                    shift = tuple(-corr[0])
                    print(f'Shift{shift}')
                    dx = shift[2] * pixel_size
                    dy = shift[1] * pixel_size
                    dz = shift[0] * z_step
                    #dx = shift[1]
                    #dy = shift[0]
                else:
                    dx = 0
                    dy = 0
                    dz = 0
                    
                time_index = time_index + 1
                # print(time_index)
                # define the next event and add to the queue    
                event = []
                for channel, exp in zip(epi_channels, epi_exposure):
                    for index, z_um in enumerate(z_range + dz):
                        evt = {
                            'axes' : {'z': index, 'time': time_index, 'position': 0},
                            'x' : x_pos + dx,
                            'y' : y_pos + dy,
                            'z' : z_um,
                            'channel' : channel,
                            'exposure' : exp
                        }
                        event.append(evt)
                event_queue.put(event)

                # swap the reference and current images    
                # alternate - always use the first time point as the reference image
                print('Swapping ref and curr images')
                print(len(img_process_fn.curr_images))
                img_process_fn.ref_images = deepcopy(img_process_fn.curr_images)
                img_process_fn.curr_images = []

    return image, metadata

#%%
# turn on autoshutter
mmc.set_auto_shutter(True)

# turn off live preview
snap_live_manager = mmStudio.get_snap_live_manager()
if snap_live_manager.is_live_mode_on():
    snap_live_manager.set_live_mode_on(False)


acq = Acquisition(directory= data_directory, name= data_name,
                    image_process_fn = img_process_fn)


acq.acquire(events)
