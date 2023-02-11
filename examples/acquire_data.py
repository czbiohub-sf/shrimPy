from mantis.acquisition.acq_engine import MantisAcquisition, AcquisitionSettings

# it is possible to use different acq rate for LF and LS
num_timepoints = 1
time_internal_s = 0

lf_acq_settings = AcquisitionSettings(
    roi=None,
    exposure_time_ms = 10,
    num_timepoints = num_timepoints,
    time_internal_s = time_internal_s,
    scan_stage = 'MCL Piezo',
    z_start = 0,
    z_end = 60,
    z_step = 5,
    channel_group = 'Channel - LF',
    channels = [f'State{i}' for i in range(5)],
    use_sequence = True,
)

ls_acq_settings = AcquisitionSettings(
    roi = (0, 896, 2048, 256),  # centered in FOV,
    exposure_time_ms = 10,
    num_timepoints = num_timepoints,
    time_internal_s = time_internal_s,
    scan_stage = 'AP Galvo',
    z_start = -2/10,  # in Volts
    z_end = 2/10,
    z_step = 0.01,  # equivalent to 330 nm
    channel_group = 'Channel - LS',
    channels = ['GFP EX488 EM525-45'],
    use_sequence = True,
)

acq = MantisAcquisition(verbose=True)

acq.define_lf_acq_settings(lf_acq_settings)
acq.define_ls_acq_settings(ls_acq_settings)

acq.acquire(directory = r'D:\\temp', name = 'test_acq')

print('done')
