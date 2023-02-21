from settings import (
    TimeSettings, PositionSettings, ChannelSettings, SliceSettings, MicroscopeSettings
    )

time_settings = TimeSettings(
    num_timepoints=3,
    time_internal_s=15
)

position_settings = PositionSettings()

lf_channel_settings = ChannelSettings(
    roi=None,
    exposure_time_ms=[10]*5,
    channel_group='Channel - LF',
    channels=[f'State{i}' for i in range(5)],
    use_sequence=True
)

lf_slice_settings = SliceSettings(
    z_scan_stage='MCL Piezo',
    z_start=0,
    z_end=60,
    z_step=5,
    use_sequence=True
)

ls_channel_settings = ChannelSettings(
    roi=(0, 896, 2048, 256),
    exposure_time_ms=[10],
    channel_group='Channel - LS',
    channels=['GFP EX488 EM525-45'],
    use_sequence=False
)

ls_slice_settings = SliceSettings(
    z_scan_stage='AP Galvo',
    z_start=-2/10,  # in Volts
    z_end=2/10,
    z_step=0.01,  # equivalent to 330 nm
    use_sequence=True
)

lf_microscope_settings = MicroscopeSettings(
    config_group_settings=[
        ('Imaging Path', 'Label-free'),
        ('Channel - LS', 'External Control'),
        ('Channel - LF', 'State0')
    ],
    device_property_settings=[
        ('Oryx', 'Line Selector', 'Line5'),
        ('Oryx', 'Line Mode', 'Output'),
        ('Oryx', 'Line Source', 'ExposureActive'),
        ('Oryx', 'Line Selector', 'Line2'),
        ('Oryx', 'Line Mode', 'Input'),
        ('Oryx', 'Trigger Source', 'Line2'),
        ('Oryx', 'Trigger Mode', 'On'),
        ('Oryx', 'Trigger Overlap', 'ReadOut')
    ],
    autofocus_stage='ZDrive',
    use_autofocus=True,
    autofocus_method='PFS'
)

ls_microscope_settings = MicroscopeSettings()
