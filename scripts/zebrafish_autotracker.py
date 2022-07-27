from pycromanager import Acquisition, multi_d_acquisition_events
import numpy as np

# with Acquisition(directory='/Users/rachel.banks/Documents/pycromanager_tests/', name='test') as acq:

#     events = multi_d_acquisition_events(z_start=0, z_end=10, z_step=0.5)

#     acq.acquire(events)

def img_process_fn(image, metadata, bridge, event_queue):

    time_index = metadata['Axes']['time']
    print(time_index)

    if time_index >= 2:
        event_queue.put(None)
    else:
        time_index += 1
        event = []
        for index, z_um in enumerate(np.arange(start=1, stop=11, step=0.5)):
            evt = {
                'axes' : {'z': index, 'time': time_index},
                'z' : z_um
            }
            event.append(evt)
        
        event_queue.put(event)

    return image, metadata

events = []
for index, z_um in enumerate(np.arange(start=0, stop=10, step=0.5)):
    evt = {
     'axes' : {'z': index, 'time': 0},
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

