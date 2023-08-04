import pytest

from mantis.acquisition.AcquisitionSettings import (
    ChannelSettings,
    ConfigSettings,
    DevicePropertySettings,
    MicroscopeSettings,
    PositionSettings,
    SliceSettings,
    TimeSettings,
)


def test_config_settings():
    # Test extra parameter
    with pytest.raises(TypeError):
        ConfigSettings(device_str="test")


def test_device_property_settings():
    # Test extra parameter
    with pytest.raises(TypeError):
        DevicePropertySettings(device_str="test")


def test_time_settings():
    # Test extra parameter
    with pytest.raises(TypeError):
        TimeSettings(device_str="test")

    # Test negative time interval
    with pytest.raises(ValueError):
        TimeSettings(time_interval_s=-0.1)

    # Test None value
    with pytest.raises(ValueError):
        TimeSettings(time_interval_s=None)


def test_position_settings():
    # Test extra parameter
    with pytest.raises(TypeError):
        PositionSettings(device_str="test")

    # Test non-equal
    with pytest.raises(AssertionError):
        _ = PositionSettings(xyz_positions=[0, 1], position_labels=['a', 'b', 'c'])


def test_channel_settings():
    # Test extra parameter
    with pytest.raises(TypeError):
        ChannelSettings(device_str="test")

    # Test non-equal
    with pytest.raises(AssertionError):
        ChannelSettings(exposure_time_ms=[0, 1], channels=['GFP'])

    # Test negative
    with pytest.raises(ValueError):
        ChannelSettings(exposure_time_ms=[-0.1], channels=['GFP'])


def test_slice_settings():
    # Test extra parameter
    with pytest.raises(TypeError):
        SliceSettings(device_str="test")

    # Test one of z_stage_name, z_start_, z_end, or z_step not provided
    with pytest.raises(TypeError):
        SliceSettings(z_stage_name="Z")

    with pytest.raises(TypeError):
        SliceSettings(z_start=0, z_end=10, z_step=1)

    with pytest.raises(TypeError):
        SliceSettings(z_stage_name="Z", z_start=0, z_end=10)


def test_example_settings(example_acquisition_settings):
    _, settings = example_acquisition_settings

    # Test constructing settings objects
    TimeSettings(**settings.get('time_settings'))

    raw_position_settings = settings.get('position_settings')
    PositionSettings()
    if raw_position_settings:
        PositionSettings(**raw_position_settings)

    ChannelSettings(**settings.get('lf_channel_settings'))

    SliceSettings(**settings.get('lf_slice_settings'))

    MicroscopeSettings(**settings.get('lf_microscope_settings'))

    ChannelSettings(**settings.get('ls_channel_settings'))

    SliceSettings(**settings.get('ls_slice_settings'))

    MicroscopeSettings(**settings.get('ls_microscope_settings'))


def test_demo_settings(demo_acquisition_settings):
    _, settings = demo_acquisition_settings

    # Test constructing settings objects
    TimeSettings(**settings.get('time_settings'))

    raw_position_settings = settings.get('position_settings')
    PositionSettings()
    if raw_position_settings:
        PositionSettings(**raw_position_settings)

    ChannelSettings(**settings.get('lf_channel_settings'))

    SliceSettings(**settings.get('lf_slice_settings'))

    MicroscopeSettings(**settings.get('lf_microscope_settings'))

    ChannelSettings(**settings.get('ls_channel_settings'))

    SliceSettings(**settings.get('ls_slice_settings'))

    MicroscopeSettings(**settings.get('ls_microscope_settings'))
