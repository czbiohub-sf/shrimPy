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
mmc.set_property('Core', 'Focus', 'ZStage')

#%%
data_directory = r'C:\\Users\\ZebraPhysics\Documents\\2022_10_03_zebrafish_imaging'
data_name = 'autotracker_testing'

mm_pos_list = mmStudio.get_position_list_manager().get_position_list()
number_of_positions = mm_pos_list.get_number_of_positions()

xy_position_list = [[mm_pos_list.get_position(i).get_x(),
                     mm_pos_list.get_position(i).get_y()]
                    for i in range(number_of_positions)]

# define the number of time points, it will run through this time point and then stop
n_time = 10

verbose = False
autotrack_channel = 'c405'
block_avg_window = 4  # in pixels
pixel_size = 4.5/60  # in um

# PID Params
Kp_xy = 1
Kp_z = 0.8

# relative z, acquired using piezo stage
z0 = mmc.get_position('ZStage')
z_start = -10
z_end = 10
z_step = 1
z_range = np.arange(z_start, z_end + z_step, z_step) + z0

epi_channels = [{'group': 'ChConfocal', 'config': 'c405'},
                {'group': 'ChConfocal', 'config': 'c488'}]
epi_exposure = [200, 500]

pos_init = xy_position_list

#%%
# define events for first two timepoints
# events = multi_d_acquisition_events(num_time_points=2,
#                                     channel_group='Master Channel',
#                                     channels=['Epi-GFP', 'Epi-DSRED'],
#                                     channel_exposures_ms=epi_exposure,
#                                     xy_positions=xy_position_list,
#                                     z_start=z_start,
#                                     z_end=z_end,
#                                     z_step=z_step)

events = []
for t_idx, t in enumerate(range(2)):
    for channel, exp in zip(epi_channels, epi_exposure):
        for z_idx, z_um in enumerate(z_range):
                evt = {
                'axes' : {'time': t_idx, 'z': z_idx, 'position': 0},
                'x' : pos_init[0][0],
                'y' : pos_init[0][1],
                'z' : z_um,
                'channel': channel,
                'exposure': exp
                    }
                events.append(evt)

#%%
def block_avg(data, n=2):
    """
    Calculate block average in (n,n) window
    :param data: raw image or stack
    :param n: block window
    :return: block averaged data, in original dtype
    """

    img_shape = data.shape
    dtype = data.dtype
    block_data = np.mean(np.reshape(data, (-1, img_shape[-2]//n, n, img_shape[-1]//n, n)),
                         axis=(-3, -1), dtype=dtype)
    return np.squeeze(block_data)

def rescale_image(image, q=0.001):
    """
    Clip image to (q, 1-q) quantile and cast as 8-bit
    :param image: raw image or stack
    :param q: quantile for clipping, in range (0, 1)
    :return: 8-bit scaled image
    """

    # calculate upper and lower intensity bounds
    lb, ub = np.quantile(image, (q, 1-q))
    # clip and scale to 8-bit range
    image_scaled = (255 * (np.clip(image, lb, ub) - lb) / (ub - lb)).astype('uint8')
    return image_scaled

def img_process_fn(image, metadata, bridge, event_queue):
    # mmc = bridge.get_core()
   
    time_index = metadata['Axes']['time']
    if verbose:
        print(f'Time index: {time_index}')

    # accumulate the images
    if not hasattr(img_process_fn, "ref_images"):
        img_process_fn.ref_images = []
        img_process_fn.curr_images = []
        img_process_fn.shift = np.array([0, 0, 0], dtype='float64')
        
    # end the acquisition if we get through the number of time points
    if time_index >= n_time:
        event_queue.put(None)
        global acq_running
        acq_running = False
    else:
        z_index = metadata['Axes']['z']
        channel = metadata['Channel']

        if channel == autotrack_channel:
            if verbose:
                print(f'Data from {autotrack_channel} channel added to current stack')
            img_process_fn.curr_images.append(image)
       
        # if we're at the end of the z stack
        if z_index == len(z_range)-1:
            if verbose:
                print('Entering z_index == len(z_range)-1 condition')
            if channel == epi_channels[-1]['config']:
                if verbose:
                    print(f'Entering channel == {epi_channels[-1]} condition')
                if time_index > 0:
                    if verbose:
                        print('Entering time_index > 0 condition')
                    x_pos = metadata['XPosition_um_Intended']
                    y_pos = metadata['YPosition_um_Intended']
                    print(f'XY position from metadata: {y_pos, x_pos}')

                    ref_stack = block_avg(rescale_image(np.stack(img_process_fn.ref_images)), n=block_avg_window)
                    curr_stack = block_avg(rescale_image(np.stack(img_process_fn.curr_images)), n=block_avg_window)
                    shift, _, _ = phase_cross_correlation(ref_stack, curr_stack)

                    img_process_fn.shift += np.array([
                        Kp_z  * shift[0] * z_step,
                        Kp_xy * shift[1] * pixel_size * block_avg_window,
                        Kp_xy * shift[2] * pixel_size * block_avg_window
                    ])
                    dz, dy, dx = img_process_fn.shift.copy()

                    # dx = shift[2] * pixel_size * block_avg_window
                    # dy = shift[1] * pixel_size * block_avg_window
                    # dz = shift[0] * z_step
                    print(f'Shift in pixels: {shift}')
                    print(f'Shift in microns: {dz, dy, dx}')

                    if dx < 1:
                        dx = 0
                    if dy < 1:
                        dy = 0

                    # define the next event and add to the queue
                    time_index = time_index + 1
                    new_events = []
                    for channel, exp in zip(epi_channels, epi_exposure):
                        for index, z_um in enumerate(z_range - dz):
                            evt = {
                                'axes': {'z': index, 'time': time_index, 'position': 0},
                                'x': x_pos - dx,
                                'y': y_pos - dy,
                                'z': z_um,
                                'channel': channel,
                                'exposure': exp
                            }
                            new_events.append(evt)
                    event_queue.put(new_events)

                # swap the reference and current images    
                # alternate - always use the first time point as the reference image
                if verbose:
                    print('Swapping ref and curr images')
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
