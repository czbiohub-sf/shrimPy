from pycromanager import Core, Acquisition, multi_d_acquisition_events

mmc = Core()
# mmc.set_config('Camera', 'HighRes')
mmc.set_property('Z', 'UseSequences', 'Yes')

events = multi_d_acquisition_events(num_time_points=3, time_interval_s=20, z_start=-3, z_end=3, z_step=0.01)

with Acquisition(r'D:\\temp', 'pm_acq_bsi', show_display=True) as acq:
    acq.acquire(events)
