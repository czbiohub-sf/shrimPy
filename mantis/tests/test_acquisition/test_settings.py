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

    # Test negative time interval = 0
    # with pytest.raises(ValueError):
    #     TimeSettings(time_interval_s=-0.1)


def test_position_settings():
    # Test extra parameter
    with pytest.raises(TypeError):
        PositionSettings(device_str="test")

    # Test non-equal
    # with pytest.raises(ValueError):
    #     s = PositionSettings(xyz_positions=[0, 1], num_positions=3)


def test_channel_settings():
    # Test extra parameter
    with pytest.raises(TypeError):
        ChannelSettings(device_str="test")

    # Test non-equal
    with pytest.raises(AssertionError):
        ChannelSettings(exposure_time_ms=[0, 1], channels=['GFP'])


def test_example_settings(example_settings):
    settings = example_settings

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


def test_demo_settings(demo_settings):
    settings = demo_settings

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
