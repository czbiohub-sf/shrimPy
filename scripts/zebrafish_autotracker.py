from pycromanager import Acquisition, multi_d_acquisition_events
import numpy as np
from skimage.registration import phase_cross_correlation


def img_process_fn(image, metadata, bridge, event_queue):
   
    time_index = metadata['Axes']['time']
    print(time_index)
    
    
    # define the number of time points, it will run through this time point and then stop
    n_time = 5
    # define the number of z_steps
    z_steps = 9

    # accumulate the images
    if not hasattr(img_process_fn, "images"):
        img_process_fn.images = []
        
    # end the acquisition if we get through the number of time points
    if time_index >= n_time:
        event_queue.put(None)
        global acq_running
        acq_running = False
    else:
        z_index = metadata['Axes']['z']
        #use DAPI channel for correlation
        channel = metadata['Channel']
        print(channel)
        if channel == 'DAPI':
            img_process_fn.images.append(image)
            print('DAPI channel added')
       
        print(z_index)
        if z_index == z_steps:
            
        # perform the calculation on the drift
            #print(img_process_fn.images[5]) 
            #img_process_fn.images = []
            print(metadata)

            if time_index > 0: 

                if channel == 'DAPI':
                # calculate the shift in x,y, and z
                # right now, just picking the image in the middle of the stack, need to think more about this
                # ref_image = img_process_fn.images[4]
                # curr_image = img_process_fn.images[int((time_index)*10+4)]
                # compare with the first time point
                    
                    x_pos = metadata['XPosition_um_Intended']
                    y_pos = metadata['YPosition_um_Intended']
                    #ref_stack = np.stack(img_process_fn.images[0:9])
                    ref_stack = np.stack(img_process_fn.images[((time_index-1)*10): ((time_index-1)*10+9)])
                    curr_stack = np.stack(img_process_fn.images[((time_index)*10): ((time_index)*10+9)])
                    # corr = phase_cross_correlation(ref_image,curr_image)
                    corr = phase_cross_correlation(ref_stack, curr_stack)
                    shift = tuple(-corr[0])
                    print(shift)
                    dx = shift[2]
                    dy = shift[1]
                    dz = shift[0]
                    #dx = shift[1]
                    #dy = shift[0]

                    # define the next event and add to the queue
                    time_index = time_index + 1
                    print(time_index)
                    channels = ['DAPI', 'FITC']
                    event = []
                    for c_index, channel in enumerate(channels):
                        for index, z_um in enumerate(np.arange(start=0 + dz, stop=10 + dz, step=1)):
                            evt = {
                                'axes' : {'z': index, 'time': time_index, 'position': 0},
                                'x' : x_pos+ dx,
                                'y' : y_pos + dy,
                                'z' : z_um,
                                'channel' : channel
                            }
                            event.append(evt)
                    
                    event_queue.put(event)

                # event = multi_d_acquisition_events(
                #                     num_time_points=1, time_interval_s=0,
                #                     channel_group='Channel', channels=['DAPI', 'FITC'],
                #                     z_start=0, z_end=10, z_step=1,
                #                     order='tcz')
            
                
            
            else:
                print('time index is zero')
                time_index += 1
                # define the next event and add to the queue
                channels = ['DAPI', 'FITC']
                event = []
                for c_index, channel in enumerate(channels):
                    for index, z_um in enumerate(np.arange(start=0, stop=10, step=1)):
                        evt = {
                            'axes' : {'z': index, 'time': time_index, 'position': 0},
                            'x' : 0,
                            'y' : 0,
                            'z' : z_um,
                            'channel' : {'group' : 'Channel', 'config' : channel}
                        }
                        event.append(evt)
                event_queue.put(event)

                #event = multi_d_acquisition_events(
                                   # num_time_points=1, time_interval_s=0,
                                   # channel_group='Channel', channels=['DAPI', 'FITC'],
                                   # z_start=0, z_end=10, z_step=1,
                                  #  order='tcz')
                
                        
                
            
    return image, metadata

# try multiple positions
#position_list = [(0,0), (100,100)]
# define the first event
events = []
# for p_index in range(len(position_list)):

#     for index, z_um in enumerate(np.arange(start=0, stop=10, step=1)):
#         evt = {
#         'axes' : {'z': index, 'time': 0},
#         'x' : position_list[p_index][0],
#         'y' : position_list[p_index][1],
#         'z' : z_um
#             }
#         events.append(evt)

pos_init = np.array([0,0])
epi_channels = ['DAPI', 'FITC']
for c_index, channel in enumerate(epi_channels):
    for index, z_um in enumerate(np.arange(start=0, stop=10, step=1)):
            evt = {
            'axes' : {'z': index, 'time': 0, 'position': 0},
            'x' : pos_init[0],
            'y' : pos_init[1],
            'z' : z_um,
            'channel': {'group': 'Channel', 'config': channel}
                }
            events.append(evt)

# events = multi_d_acquisition_events(
#                                     num_time_points=2, time_interval_s=0,
#                                     channel_group='Channel', channels=['DAPI', 'FITC'],
#                                     z_start=0, z_end=10, z_step=1,
#                                     order='tcz')

acq = Acquisition(directory='/Users/rachel.banks/Documents/pycromanager_tests/', name='test',
                    image_process_fn = img_process_fn)


acq.acquire(events)

# with Acquisition(directory='/Users/rachel.banks/Documents/pycromanager_tests/', name='test',
#                     image_process_fn = img_process_fn) as acq:
#     events = []
#     for index, z_um in enumerate(np.arange(start=0, stop=10, step=0.5)):
#         evt = {
#             'axes' : {'z': index, 'time': 0},
#             'z' : z_um
#         }
#         events.append(evt)
#     acq.acquire(events)


#     # create some acquisition events here
#     events = []
#     event_0 = {'axes': {'time':0}}
#     event_1 = {'axes': {'time':1}}
#     events = [event_0, event_1]
#     acq.acquire(events)

