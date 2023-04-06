import click
import yaml
from mantis.acquisition.acq_engine import MantisAcquisition
from mantis.acquisition.BaseSettings import (
    TimeSettings,
    PositionSettings,
    ChannelSettings,
    SliceSettings,
    MicroscopeSettings,
)

@click.command()
@click.help_option("-h", "--help")
@click.option(
    '--data-dirpath', 
    required=True, 
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help='Directory where acquired data will be saved',
)
@click.option(
    '--name', 
    required=True,
    help='Name of the acquisition',
)
@click.option(
    '--settings', 
    required=True, 
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help='YAML file containing the acquisition settings',
)
@click.option(
    '--mm-app-path', 
    default='C:\\Program Files\\Micro-Manager-nightly', 
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    show_default=True,
    help='''Path to Micro-manager installation directory
      which will run the light-sheet acquisition''',
)
@click.option(
    '--mm-config-file', 
    default='C:\\CompMicro_MMConfigs\\mantis\\mantis-LS.cfg', 
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    show_default=True,
    help='''Path to Micro-manager config file 
      which will run the light-sheet acquisition''',
)
def run_acquisition(
    data_dirpath,
    name,
    settings,
    mm_app_path,
    mm_config_file,
):

    demo_run = True if 'demo' in mm_config_file else False
    
    with open(settings) as file:
        raw_settings = yaml.safe_load(file)

    with MantisAcquisition(
        acquisition_directory=data_dirpath,
        acquisition_name=name,
        mm_app_path=mm_app_path,
        mm_config_file=mm_config_file,
        demo_run=demo_run,
        verbose=False,
    ) as acq:

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
        acq.acquire()


if __name__ == '__main__':
    run_acquisition()