import os
from pycromanager import Core, Acquisition, multi_d_acquisition_events, start_headless

PORT1 = 4827
mmc1 = Core(port=PORT1)
print(mmc1)

PORT2 = 5827
mm_app_path = r'C:\Program Files\Micro-Manager-2.0_11_25_2022'
config_file = r'C:\CompMicro_MMConfigs\mantis\mantis-LS.cfg'
start_headless(mm_app_path, config_file, port=PORT2) 
mmc2 = Core(port=PORT2)
print(mmc2)

events1 = multi_d_acquisition_events(num_time_points = 100)
events2 = multi_d_acquisition_events(num_time_points = 50)

save_path = r'D:\temp'

acq1 = Acquisition(directory=save_path, name='acq1', port=PORT1)
acq2 = Acquisition(directory=save_path, name='acq2', port=PORT2)

acq1.acquire(events1)
acq2.acquire(events2)

acq1.mark_finished()
acq2.mark_finished()

acq1.await_completion()
acq2.await_completion()
