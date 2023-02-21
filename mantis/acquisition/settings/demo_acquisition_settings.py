from settings import (
    TimeSettings, PositionSettings, ChannelSettings, SliceSettings, MicroscopeSettings
    )

time_settings = TimeSettings(
    num_timepoints=3,
    time_internal_s=5
)

position_settings = PositionSettings()

lf_channel_settings = ChannelSettings(
    roi=None,
    exposure_time_ms=[10, 15],
    channel_group='Channel-Multiband',
    channels=['DAPI', 'FITC'],
    use_sequence=True
)

lf_slice_settings = SliceSettings(
    z_scan_stage='Z',
    z_start=0,
    z_end=60,
    z_step=5,
    use_sequence=True
)

ls_channel_settings = ChannelSettings(
    roi=(0, 896, 2048, 256),
    exposure_time_ms=[20, 30],
    channel_group='Channel',
    channels=['Rhodamine', 'Cy5'],
    use_sequence=False
)

ls_slice_settings = SliceSettings(
    z_scan_stage='Z',
    z_start=-2/10,  # in Volts
    z_end=2/10,
    z_step=0.01,  # equivalent to 330 nm
    use_sequence=True
)

lf_microscope_settings = MicroscopeSettings(
    config_group_settings=[
        ('LightPath', 'Camera-left')
    ],
    device_property_settings=[
        ('Camera', 'OnCameraCCDXSize', '1024'),
        ('Camera', 'OnCameraCCDYSize', '1224'),
        ('Camera', 'BitDepth', '12'),
    ],
)

ls_microscope_settings = MicroscopeSettings(
    config_group_settings=[
        ('LightPath', 'Camera-right')
    ],
    device_property_settings=[
        ('Camera', 'OnCameraCCDXSize', '2048'),
        ('Camera', 'OnCameraCCDYSize', '2048'),
        ('Camera', 'BitDepth', '11'),
    ],
)
