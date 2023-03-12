#%%
import yaml
from mantis.acquisition.acq_engine import MantisAcquisition
from mantis.acquisition.BaseSettings import (
    TimeSettings,
    PositionSettings,
    ChannelSettings,
    SliceSettings,
    MicroscopeSettings,
)

#%%
with open('./acquisition/settings/demo_acquisition_settings.yaml') as file:
    raw_settings = yaml.safe_load(file)

#%%
acquisition_directory = r'D:\\2023_02_13_automation_testing'
mm_app_path = r'C:\\Program Files\\Micro-Manager-nightly'
mm_config_file = r'C:\\CompMicro_MMConfigs\\mantis\\mantis-LS.cfg'

acq = MantisAcquisition(
    acquisition_directory=acquisition_directory,
    mm_app_path=mm_app_path,
    mm_config_file=mm_config_file,
    demo_run=True,
    verbose=False,
)

acq.time_settings = \
    TimeSettings(**raw_settings.get('time_settings'))
acq.position_settings = \
    PositionSettings()
acq.lf_acq.channel_settings = \
    ChannelSettings(**raw_settings.get('lf_channel_settings'))
acq.lf_acq.slice_settings = \
    SliceSettings(**raw_settings.get('lf_slice_settings'))
acq.lf_acq.microscope_settings = \
    MicroscopeSettings(**raw_settings.get('lf_microscope_settings'))
acq.ls_acq.channel_settings = \
    ChannelSettings(**raw_settings.get('ls_channel_settings'))
acq.ls_acq.slice_settings = \
    SliceSettings(**raw_settings.get('ls_slice_settings'))
acq.ls_acq.microscope_settings = \
    MicroscopeSettings(**raw_settings.get('ls_microscope_settings'))

acq.setup()
acq.acquire(name='test_acq')
acq.close()

print('done')
