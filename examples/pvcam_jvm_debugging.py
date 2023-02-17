from pycromanager import Core, Acquisition, multi_d_acquisition_events

# with Acquisition(r'D:\temp', 'pm_cam_test', show_display=False) as acq:
#     acq.acquire(
#         multi_d_acquisition_events(
#             num_time_points=400,
#             time_interval_s=0
#         )
#     )

# mmc = Core()
# mmc.set_roi(*(0, 896, 2048, 256))
# mmc.set_property('Prime BSI Express', 'ReadoutRate', '200MHz 11bit')
# # mmc.set_property('Prime BSI Express', 'ReadoutRate', '100MHz 16bit')
# mmc.set_property('TS2_DAC03','Sequence','On')

# with Acquisition(r'D:\temp', 'pm_cam_mda_test', show_display=False) as acq:
#     acq.acquire(
#         multi_d_acquisition_events(
#             num_time_points=3,
#             time_interval_s=20,
#             z_start=-3,
#             z_end=3,
#             z_step=0.01
#         )
#     )

# mmc = Core()
# mmc.set_roi(*(0, 896, 2048, 256))
# mmc.set_property('Prime BSI Express', 'ReadoutRate', '200MHz 11bit')
# # mmc.set_property('Prime BSI Express', 'ReadoutRate', '100MHz 16bit')
# # mmc.set_property('TS2_DAC03','Sequence','On')

# with Acquisition(r'D:\temp', 'pm_cam_mda_test', show_display=False) as acq:
#     acq.acquire(
#         multi_d_acquisition_events(
#             num_time_points=1000,
#             time_interval_s=0,
#         )
#     )

# mmc = Core()
# # mmc.set_property('Prime BSI Express', 'ReadoutRate', '100MHz 16bit')
# mmc.set_property('Prime BSI Express', 'ReadoutRate', '200MHz 11bit')

# with Acquisition(r'D:\temp', 'pm_cam_mda_test', show_display=False) as acq:
#     acq.acquire(
#         multi_d_acquisition_events(
#             num_time_points=500,
#             time_interval_s=0,
#         )
#     )

# mmc = Core()
# mmc.set_roi(*(0, 896, 2048, 256))
# mmc.set_property('Prime BSI Express', 'ReadoutRate', '200MHz 11bit')
# # mmc.set_property('Prime BSI Express', 'ReadoutRate', '100MHz 16bit')
# mmc.set_property('TS2_DAC03','Sequence','On')

# with Acquisition(r'D:\temp', 'pm_cam_mda_test', show_display=False) as acq:
#     acq.acquire(
#         multi_d_acquisition_events(
#             num_time_points=3,
#             time_interval_s=20,
#             z_start=-3,
#             z_end=3,
#             z_step=0.01
#         )
#     )

mmc = Core()
# mmc.set_property('Prime BSI Express', 'ReadoutRate', '100MHz 16bit')
mmc.set_property('Prime BSI Express', 'ReadoutRate', '200MHz 11bit')

def image_process_fn(image, metadata):
    return None

with Acquisition(image_process_fn=image_process_fn, show_display=False) as acq:
    acq.acquire(
        multi_d_acquisition_events(
            num_time_points=500,
            time_interval_s=0,
        )
    )
