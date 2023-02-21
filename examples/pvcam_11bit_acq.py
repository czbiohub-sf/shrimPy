from pycromanager import Core, Acquisition, multi_d_acquisition_events

mmc = Core()

mmc.set_property('Prime BSI Express', 'ReadoutRate', '200MHz 11bit')

with Acquisition(r'D:\temp', 'pvcam_11bit_gui', show_display=False) as acq:
    acq.acquire(
        multi_d_acquisition_events(
            num_time_points=10,
            time_interval_s=0,
        )
    )