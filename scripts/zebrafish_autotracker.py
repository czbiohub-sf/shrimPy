from pycromanager import Acquisition, multi_d_acquisition_events
import numpy as np
from skimage.registration import phase_cross_correlation

# with Acquisition(directory='/Users/rachel.banks/Documents/pycromanager_tests/', name='test') as acq:

#     events = multi_d_acquisition_events(z_start=0, z_end=10, z_step=0.5)

#     acq.acquire(events)

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
        img_process_fn.images.append(image)
       
        #print(z_index)
        if z_index == z_steps:

        # perform the calculation on the drift
            #print(img_process_fn.images[5]) 
            #img_process_fn.images = []
            

            if time_index > 1: 
                # calculate the shift in x,y, and z
                # right now, just picking the image in the middle of the stack, need to think more about this
                #ref_image = img_process_fn.images[int((time_index-2)*9-4)]
                #curr_image = img_process_fn.images[int((time_index-1)*9-4)]
                ref_stack = np.stack(img_process_fn.images[((time_index-2)*10):((time_index-2)*10+9)])
                curr_stack = np.stack(img_process_fn.images[((time_index-1)*10): ((time_index-1)*10+9)])
                #corr = phase_cross_correlation(ref_image,curr_image)
                corr = phase_cross_correlation(ref_stack, curr_stack)
                shift = tuple(-corr[0])
                print(shift)
                dx = shift[2]
                dy = shift[1]
                dz = shift[0]

                # define the next event and add to the queue
                time_index += 1
                event = []
                for index, z_um in enumerate(np.arange(start=0 + dz, stop=10 + dz, step=1)):
                    evt = {
                        'axes' : {'z': index, 'time': time_index},
                        'x' : 0 + dz,
                        'y' : 0 + dy,
                        'z' : z_um
                    }
                    event.append(evt)
            
                event_queue.put(event)
            
            else:
                time_index += 1
                # define the next event and add to the queue
                event = []
                for index, z_um in enumerate(np.arange(start=0, stop=10, step=1)):
                    evt = {
                        'axes' : {'z': index, 'time': time_index},
                        'x' : 0,
                        'y' : 0,
                        'z' : z_um
                    }
                    event.append(evt)
                
                event_queue.put(event)
                 # increment the time index
            
    return image, metadata

# define the first event
events = []
for index, z_um in enumerate(np.arange(start=0, stop=10, step=1)):
    evt = {
     'axes' : {'z': index, 'time': 0},
     'x' : 0,
     'y' : 0,
     'z' : z_um
         }
    events.append(evt)

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

