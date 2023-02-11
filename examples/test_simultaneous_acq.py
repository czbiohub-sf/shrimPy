#%% import modules

import os
from pycromanager import start_headless, Core, Acquisition, multi_d_acquisition_events

#%% Connect to running MM on default port 4827
# Needs to use a separate MM installation than one launched in headless mode
# In this case mm_app_path = 'C:\Users\Cameron\Micro-Manager-nightly_2'

PORT1 = 4827
mmc1 = Core(port=PORT1)

print(mmc1)

#%% Start and connect to headless MM on port 5827

mm_app_path = r'C:\Program Files\Micro-Manager-nightly'
config_file = os.path.join(mm_app_path, "MMConfig_demo.cfg")

# Start the Java process
PORT2 = 5827
start_headless(mm_app_path, config_file, port=PORT2)  # we need to space out port numbers a bit

mmc2 = Core(port=PORT2)
print(mmc2)

#%% Acquire 100 timepoints on both cameras

events = multi_d_acquisition_events(num_time_points=100)
save_path = 'Q:\Ivan\debug'

# Images can be different size, no problem
mmc2.set_config('Camera', 'LowRes')

# NDViewer needs to be turned off, does not work in this case
acq1 = Acquisition(directory=save_path, name='acq1', port=PORT1, show_display=False)
acq2 = Acquisition(directory=save_path, name='acq2', port=PORT2, show_display=False)

acq1.acquire(events)
acq2.acquire(events)

acq1.mark_finished()
acq2.mark_finished()

acq1.await_completion()
acq2.await_completion()