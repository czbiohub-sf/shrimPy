from acq_engine import MantisAcquisition
from acquisition.settings.mantis_acquisition_settings import (
    time_settings,
    position_settings,
    lf_channel_settings,
    lf_slice_settings,
    lf_microscope_settings,
    ls_channel_settings,
    ls_slice_settings,
    ls_microscope_settings,
)

acq = MantisAcquisition(acquisition_directory=r'D:\\2023_02_13_automation_testing', verbose=False)

acq.time_settings = time_settings
acq.position_settings = position_settings
acq.lf_acq.channel_settings = lf_channel_settings
acq.lf_acq.slice_settings = lf_slice_settings
acq.lf_acq.microscope_settings = lf_microscope_settings
acq.ls_acq.channel_settings = ls_channel_settings
acq.ls_acq.slice_settings = ls_slice_settings
acq.ls_acq.microscope_settings = ls_microscope_settings

acq.setup()
acq.acquire(name='test_acq')
acq.close()

print('done')
