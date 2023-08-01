from mantis.acquisition.AcquisitionSettings import (
    AcquisitionSettings,
    ConfigSettings,
    DevicePropertySettings,
    TimeSettings,
)
import pytest


def test_config_settings():
    # Test extra parameter
    with pytest.raises(ValueError):
        s = ConfigSettings(device_str="test")


def test_device_property_settings():
    # Test extra parameter
    with pytest.raises(ValueError):
        s = DevicePropertySettings(device_str="test")


def test_time_settings():
    # Test extra parameter
    with pytest.raises(ValueError):
        s = TimeSettings(device_str="test")

    # Test num_timepoints = 0
    with pytest.raises(ValueError):
        s = TimeSettings(num_timepoints=0)

    # Test negative time interval = 0
    with pytest.raises(ValueError):
        s = TimeSettings(time_interval_s=-0.1)


# def test_example_settings(example_settings):
#     s = AcquisitionSettings(**example_settings)

#     # Check values match example file
#     assert s.ls_microscope_settings.z_sequencing_settings[0].device_name == "TS2_DAC03"
#     assert s.time_settings.num_timepoints == 10


# def test_demo_settings(demo_settings):
#     s = AcquisitionSettings(**demo_settings)

#     # Check values match demo file
#     assert s.time_settings.num_timepoints == 3
#     assert s.lf_channel_settings.exposure_time_ms == [10, 10]
#     assert s.lf_microscope_settings.device_property_settings[0].device_name == "Camera"
#     assert s.lf_microscope_settings.device_property_settings[0].property_value == "1024"

#     # Test extra parameter
#     with pytest.raises(ValueError):
#         s = AcquisitionSettings(**demo_settings, typo_settings=0)
