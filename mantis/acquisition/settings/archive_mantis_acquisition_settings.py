from acquisition.BaseSettings import (
    TimeSettings, PositionSettings, ChannelSettings, SliceSettings, MicroscopeSettings
    )

LCA_DAC = 'TS1_DAC01'
LCB_DAC = 'TS1_DAC02'
MCL_DAC = 'TS1_DAC06'
AP_GALVO_DAC = 'TS2_DAC03'

time_settings = TimeSettings(
    num_timepoints=3,
    time_internal_s=15
)

position_settings = PositionSettings()

lf_channel_settings = ChannelSettings(
    exposure_time_ms=[10]*5,
    channel_group='Channel - LF',
    channels=[f'State{i}' for i in range(5)],
    use_sequencing=True,
)

lf_slice_settings = SliceSettings(
    z_stage_name='MCL Piezo',
    z_start=0,
    z_end=60,
    z_step=5,
    use_sequencing=True,
)

ls_channel_settings = ChannelSettings(
    exposure_time_ms=[10],
    channel_group='Channel - LS',
    channels=['GFP EX488 EM525-45'],
    use_sequencing=False
)

ls_slice_settings = SliceSettings(
    z_scan_stage='AP Galvo',
    z_start=-2/10,  # in Volts
    z_end=2/10,
    z_step=0.01,  # equivalent to 330 nm
    use_sequencing=True,
)

lf_microscope_settings = MicroscopeSettings(
    roi=None,
    config_group_settings=[
        ('Imaging Path', 'Label-free'),
        ('Channel - LS', 'External Control'),
    ],
    device_property_settings=[
        ('Oryx', 'Line Selector', 'Line5'),
        ('Oryx', 'Line Mode', 'Output'),
        ('Oryx', 'Line Source', 'ExposureActive'),
        ('Oryx', 'Line Selector', 'Line2'),
        ('Oryx', 'Line Mode', 'Input'),
        ('Oryx', 'Trigger Source', 'Line2'),
        ('Oryx', 'Trigger Mode', 'On'),
        ('Oryx', 'Trigger Overlap', 'ReadOut'),
        ('Oryx', 'Frame Rate Control Enabled', '0'),
    ],
    z_sequencing_settings = [
        (MCL_DAC, 'Sequence', 'On')
    ],
    channel_sequencing_settings=[
        (LCA_DAC, 'Sequence', 'On')
        (LCB_DAC, 'Sequence', 'On')
    ],
    autofocus_stage='ZDrive',
    use_autofocus=True,
    autofocus_method='PFS'
)

ls_microscope_settings = MicroscopeSettings(
    roi=(0, 896, 2048, 256),
    device_property_settings=[
        ('Prime BSI Express', 'ReadoutRate', '200MHz 11bit'),
        ('Prime BSI Express', 'Gain', '1-Full well'),
        ('Prime BSI Express', 'TriggerMode', 'Edge Trigger'),
        ('Prime BSI Express', 'ExposeOutMode', 'Rolling Shutter'),
        ('TS2_TTL1-8', 'Blanking', 'On'),
    ],
    z_sequencing_settings = [
        (AP_GALVO_DAC, 'Sequence', 'On')
    ]
)
