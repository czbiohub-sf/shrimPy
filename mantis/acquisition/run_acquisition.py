from acq_engine import MantisAcquisition
from mantis_acquisition_settings import (
    time_settings,
    position_settings,
    lf_channel_settings,
    lf_slice_settings,
    ls_channel_settings,
    ls_slice_settings,
    autofocus_settings,
)

acq = MantisAcquisition(acquisition_directory=r'D:\\2023_02_13_automation_testing', verbose=False)

acq.define_lf_acq_settings(lf_acq_settings)
acq.define_ls_acq_settings(ls_acq_settings)
acq.defile_position_time_acq_settings(pt_acq_settins)

acq.acquire(name = 'test_acq')
acq.close()

print('done')
