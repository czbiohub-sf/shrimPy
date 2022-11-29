import os
from pycromanager import Core, Acquisition, multi_d_acquisition_events

PORT1 = 4827
mmc1 = Core(port=PORT1)
print(mmc1)

events1 = multi_d_acquisition_events(num_time_points = 100)
events2 = multi_d_acquisition_events(num_time_points = 50)

save_path = r'D:\temp'

acq1 = Acquisition(directory=save_path, name='acq1', port=PORT1)
acq1.acquire(events1)
acq1.mark_finished()
acq1.await_completion()

# with Acquisition(directory=save_path, name='acq1') as acq1:
#     acq1.acquire(events1)
