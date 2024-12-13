from mantis.acquisition.acq_engine import (
    MantisAcquisition, 
    ChannelSliceAcquisitionSettings,
    PositionTimeAcquisitionSettings)

# it is possible to use different acq rate for LF and LS

pt_acq_settins = PositionTimeAcquisitionSettings(
    num_timepoints = 3,
    time_interval_s = 15,
    xyz_positions = None,  # will be acquired from MM later
    position_labels = None,
    focus_stage = 'ZDrive',
    use_autofocus = True,
    autofocus_method = 'PFS'
)

lf_acq_settings = ChannelSliceAcquisitionSettings(
    roi=None,
    exposure_time_ms = 10,
    z_scan_stage = 'MCL Piezo',
    z_start = 0,
    z_end = 60,
    z_step = 5,
    channel_group = 'Channel - LF',
    channels = [f'State{i}' for i in range(5)],
    use_sequence = True,
)

ls_acq_settings = ChannelSliceAcquisitionSettings(
    roi = (0, 896, 2048, 256),  # centered in FOV,
    exposure_time_ms = 10,
    z_scan_stage = 'AP Galvo',
    z_start = -2/10,  # in Volts
    z_end = 2/10,
    z_step = 0.01,  # equivalent to 330 nm
    channel_group = 'Channel - LS',
    channels = ['GFP EX488 EM525-45'],
    use_sequence = True,
)

acq = MantisAcquisition(acquisition_directory=r'D:\2023_02_17_mantis_dataset_standard', verbose=False)

acq.define_lf_acq_settings(lf_acq_settings)
acq.define_ls_acq_settings(ls_acq_settings)
acq.defile_position_time_acq_settings(pt_acq_settins)

acq.acquire(name = 'acq1')
acq.close()

print('done')
