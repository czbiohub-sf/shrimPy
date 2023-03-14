import os
import numpy as np
from dataclasses import asdict
from datetime import datetime
import logging
from collections.abc import Iterable

from mantis.acquisition import microscope_operations
from mantis.acquisition.logger import configure_logger
from mantis.acquisition.BaseSettings import (
    TimeSettings,
    PositionSettings,
    ChannelSettings,
    SliceSettings,
    MicroscopeSettings,
)

from pycromanager import (
    start_headless, 
    Core, 
    Studio,
    Acquisition, 
    multi_d_acquisition_events)
from pycromanager.acq_util import cleanup

from mantis.acquisition.hook_functions.daq_control import (
    get_num_daq_counter_samples, 
    start_daq_counter)

import nidaqmx
from nidaqmx.constants import AcquisitionType, Slope

### Define constants
LF_ZMQ_PORT = 4827
LS_ZMQ_PORT = 5827   # we need to space out port numbers a bit
LS_POST_READOUT_DELAY = 0.05  # in ms
MCL_STEP_TIME = 1.5  # in ms
LC_CHANGE_TIME = 20  # in ms

logger = logging.getLogger(__name__)


class BaseChannelSliceAcquisition(object):

    def __init__(
            self,
            enabled: bool = True,
            mm_app_path: str = None,
            mm_config_file: str = None,
            zmq_port: int = 4827,
            core_log_path: str = '',
    ):
        self.enabled = enabled
        self._channel_settings = ChannelSettings()
        self._slice_settings = SliceSettings()
        self._microscope_settings = MicroscopeSettings()
        self.headless = False if mm_app_path is None else True
        self.type = 'light-sheet' if self.headless else 'label-free'
        self.mmc = None
        self.mmStudio = None

        logger.debug(f'Initializing {self.type} acquisition engine')
        if enabled:
            if self.headless:
                java_loc = None
                if "JAVA_HOME" in os.environ:
                    java_loc = os.environ["JAVA_HOME"]

                logger.debug(f'Starting headless Micro-Manager instance on port {zmq_port}')
                logger.debug(f'Core logs will be saved at: {core_log_path}')
                start_headless(
                    mm_app_path,
                    mm_config_file,
                    java_loc=java_loc,
                    port=zmq_port,
                    core_log_path=core_log_path,
                    buffer_size_mb=2048
                )

            logger.debug(f'Connecting to Micro-Manager on port {zmq_port}')

            self.mmc = Core(port=zmq_port)

            # headless MM instance doesn't have a studio object
            if not self.headless:
                self.mmStudio = Studio(port=zmq_port)

            logger.debug('Successfully connected to Micro-Manager')
            logger.debug(f'{self.mmc.get_version_info()}')  # MMCore Version

            if not self.headless:
                logger.debug(f'MM Studio version: {self.mmStudio.compat().get_version()}')
        else:
            logger.info(f'{self.type.capitalize()} acquisition is not enabled')

    @property
    def channel_settings(self):
        return self._channel_settings
    
    @property
    def slice_settings(self):
        return self._slice_settings
    
    @property
    def microscope_settings(self):
        return self._microscope_settings
    
    @channel_settings.setter
    def channel_settings(self, settings:ChannelSettings):
        logger.debug(
            f'{self.type.capitalize()} acquisition will have the following settings: {asdict(settings)}'
        )
        self._channel_settings = settings

    @slice_settings.setter
    def slice_settings(self, settings:SliceSettings):
        settings_dict = {key: val for key, val in asdict(settings).items() if key != 'z_range'}
        logger.debug(
            f'{self.type.capitalize()} acquisition will have the following settings: {settings_dict}'
        )
        self._slice_settings = settings

    @microscope_settings.setter
    def microscope_settings(self, settings:MicroscopeSettings):
        logger.debug(
            f'{self.type.capitalize()} acquisition will have the following settings: {asdict(settings)}'
        )
        self._microscope_settings = settings

    def setup(self):
        if self.enabled:
        # Apply microscope config settings
            for settings in self.microscope_settings.config_group_settings:
                microscope_operations.set_config(
                    self.mmc,
                    settings.config_group,
                    settings.config_name
                )

            # Apply microscope device property settings
            for settings in self.microscope_settings.device_property_settings:
                microscope_operations.set_property(
                    self.mmc,
                    settings.device_name,
                    settings.property_name,
                    settings.property_value
                )

            # Apply ROI
            if self.microscope_settings.roi is not None:
                microscope_operations.set_roi(
                    self.mmc,
                    self.microscope_settings.roi
                )

            # Setup z scan stage
            microscope_operations.set_property(
                self.mmc,
                'Core',
                'Focus',
                self.slice_settings.z_stage_name
            )

            # TODO: How should we deal with turning off sequencing when use_sequencing=False?
            # Setup z sequencing
            if self.slice_settings.use_sequencing:
                for settings in self.microscope_settings.z_sequencing_settings:
                    microscope_operations.set_property(
                        self.mmc,
                        settings.device_name,
                        settings.property_name,
                        settings.property_value
                    )

            # Setup channel sequencing
            if self.channel_settings.use_sequencing:
                for settings in self.microscope_settings.channel_sequencing_settings:
                    microscope_operations.set_property(
                        self.mmc,
                        settings.device_name,
                        settings.property_name,
                        settings.property_value
                    )


class MantisAcquisition(object):
    """
    Base class for mantis  multimodal acquisition
    """

    def __init__(
            self,
            acquisition_directory: str,
            mm_app_path: str = r'C:\\Program Files\\Micro-Manager-nightly',
            mm_config_file: str = r'C:\\CompMicro_MMConfigs\\mantis\\mantis-LS.cfg',
            enable_ls_acq: bool = True,
            enable_lf_acq: bool = True,
            demo_run: bool = False,
            verbose: bool = False,
            ) -> None:
        
        """Initialize the mantis acquisition class and connect to Micro-manager

        Parameters
        ----------
        acquisition_directory : str
            Directory where acquired data will be saved
        mm_app_path : str, optional
            Path to Micro-manager installation directory which runs the 
            light-sheet acquisition, by default r'C:\Program Files\Micro-Manager-nightly'
        config_file : str, optional
            Path to config file which runs the light-sheet acquisition,
            by default r'C:\CompMicro_MMConfigs\mantis\mantis-LS.cfg'
        enable_ls_acq : bool, optional
            Set to False if only acquiring label-free data, by default True
        enable_lf_acq : bool, optional
            Set to False if only acquiring fluorescence light-sheet data,
            by default True
        verbose : bool, optional
            By default False
        """

        self._acq_dir = acquisition_directory
        self._demo_run = demo_run
        self._verbose = verbose

        # initialize time and position settings
        self._time_settings = TimeSettings()
        self._position_settings = PositionSettings()

        # Setup logger
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        configure_logger(os.path.join(self._acq_dir,
                                      f'mantis_acquisition_log_{timestamp}.txt'))

        if self._demo_run:
            logger.info('NOTE: This is a demo run')
        logger.debug(f'Starting mantis acquisition log at: {self._acq_dir}')
        
        # Connect to MM running LF acq
        self.lf_acq = BaseChannelSliceAcquisition(
            enabled=enable_lf_acq,
            zmq_port=LF_ZMQ_PORT,
        )

        # Connect to MM running LS acq
        self.ls_acq = BaseChannelSliceAcquisition(
            enabled=enable_ls_acq,
            mm_app_path=mm_app_path,
            mm_config_file=mm_config_file,
            zmq_port=LS_ZMQ_PORT,
            core_log_path=os.path.join(mm_app_path, 'CoreLogs',
                                       f'CoreLog{timestamp}_headless.txt')
        )

    @property
    def time_settings(self):
        return self._time_settings
    
    @property
    def position_settings(self):
        return self._position_settings

    @time_settings.setter
    def time_settings(self, settings:TimeSettings):
        logger.debug(
            f'Mantis acquisition will have the following settings: {asdict(settings)}'
        )
        self._time_settings = settings

    @position_settings.setter
    def position_settings(self, settings:PositionSettings):
        logger.debug(
            f'Mantis acquisition will have the following settings: {asdict(settings)}'
        )
        self._position_settings = settings
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        # Close PM bridges
        cleanup()
            
    # def define_lf_acq_settings(self, acq_settings):
    #     logger = logging.getLogger(__name__)
    #     if not self._lf_acq_enabled:
    #         self._lf_acq_enabled = True
    #         logger.warning('Enabling label-free acquisition even though one was not requested.')
        
    #     self.lf_acq_settings = acq_settings
    #     logger.debug(f'Label-free acquisition will have following settings: {acq_settings.__dict__}')

    # def define_ls_acq_settings(self, acq_settings):
    #     logger = logging.getLogger(__name__)
    #     if not self._ls_acq_enabled:
    #         self._ls_acq_enabled = True
    #         logger.warning('Enabling light-sheet acquisition even though one was not requested.')
        
    #     self.ls_acq_settings = acq_settings
    #     logger.debug(f'Light-sheet acquisition will have following settings: {acq_settings.__dict__}')

    def update_position_settings(self):
        mm_pos_list = self.lf_acq.mmStudio.get_position_list_manager().get_position_list()
        mm_number_of_positions = mm_pos_list.get_number_of_positions()

        if self.position_settings.num_positions == 0:
            if mm_number_of_positions > 0:
                logger.debug('Fetching position list from Micro-manager')

                xyz_position_list, position_labels = microscope_operations.get_position_list(
                    self.lf_acq.mmStudio,
                    self.lf_acq.microscope_settings.autofocus_stage
                )
            else:
                logger.debug('Fetching current position from Micro-manager')

                xyz_position_list = [(
                    self.lf_acq.mmc.get_x_position,
                    self.lf_acq.mmc.get_y_position,
                    self.lf_acq.mmc.get_position(self.lf_acq.microscope_settings.autofocus_stage)
                )]
                position_labels = ['Current']

            self.position_settings = PositionSettings(
                xyz_positions=xyz_position_list,
                position_labels=position_labels,
            )
    
    # def defile_position_time_acq_settings(self, acq_settings):
    #     logger = logging.getLogger(__name__)
    #     self.pt_acq_settings = acq_settings
    #     if self.pt_acq_settings.xyz_positions is None:
    #         xyz_position_list, position_labels = self._get_position_list()
    #         if len(xyz_position_list) > 0:
    #             self.pt_acq_settings.xyz_positions = xyz_position_list
    #             self.pt_acq_settings.position_labels = position_labels
    #             self.pt_acq_settings.num_positions = len(xyz_position_list)
    #     logger.debug(f'The following time and position settings will be applied: {acq_settings.__dict__}')

    # def _setup_lf_acq_archive(self):
    #     logger = logging.getLogger(__name__)
    #     if not hasattr(self, 'lf_acq_settings'):
    #         raise Exception('Please define LF acquisition settings.')

    #     # Setup light path
    #     for _config_group, _config in (('Imaging Path', 'Label-free'),
    #                                    ('Channel - LS', 'External Control')):
    #         self._lf_mmc.set_config(_config_group, _config)
    #         logger.debug(f'Setting {_config_group} config group to {_config}')
        
    #     # Setup camera
    #     _apply_device_property_settings(self._lf_mmc, (('Oryx', 'Line Selector', 'Line5'),))
    #     # required for above selection to take effect
    #     self._lf_mmc.update_system_state_cache()

    #     settings = (('Oryx', 'Line Mode', 'Output'),
    #                 ('Oryx', 'Line Source', 'ExposureActive'))
    #     _apply_device_property_settings(self._lf_mmc, settings)


    #     _apply_device_property_settings(self._lf_mmc, 
    #                                     (('Oryx', 'Line Selector', 'Line2'),))
    #     # required for above selection to take effect
    #     self._lf_mmc.update_system_state_cache()

    #     # Trigger Overlap: ReadOut is required for external triggering at max frame rate
    #     settings = (('Oryx', 'Line Mode', 'Input'),
    #                 ('Oryx', 'Trigger Source', 'Line2'),
    #                 ('Oryx', 'Trigger Mode', 'On'),
    #                 ('Oryx', 'Trigger Overlap', 'ReadOut'))
    #     _apply_device_property_settings(self._lf_mmc, settings)

    #     oryx_framerate_enabled = self._lf_mmc.get_property('Oryx', 'Frame Rate Control Enabled')
    #     _apply_device_property_settings(self._lf_mmc, 
    #                                     (('Oryx', 'Frame Rate Control Enabled', '0'),))

    #     # Setup sequencing
    #     _use_seq = 'On' if self.lf_acq_settings.use_sequence else 'Off'
    #     _apply_device_property_settings(
    #         self._lf_mmc, [(d, 'Sequence', _use_seq) for d in (LCA_DAC, LCB_DAC, MCL_DAC)])

    #     # Setup ROI
    #     if self.lf_acq_settings.roi is not None:
    #         self._lf_mmc.set_roi(*self.lf_acq_settings.roi)
    #         logger.debug(f'Setting ROI to {self.lf_acq_settings.roi}')

    #     # Setup scan stage
    #     self._lf_mmc.set_property('Core', 'Focus', self.lf_acq_settings.z_stage)
    #     logger.debug(f'Setting focus stage to {self.lf_acq_settings.z_stage}')

    #     # Setup channels
    #     if self.lf_acq_settings.channels is not None:
    #         self._lf_mmc.set_config(
    #             self.lf_acq_settings.channel_group, 
    #             self.lf_acq_settings.channels[0])
    #         logger.debug(f'Setting first channel as {self.lf_acq_settings.channels[0]}')
            
    #     # Setup exposure
    #     self._lf_mmc.set_exposure(self.lf_acq_settings.exposure_time_ms)
    #     logger.debug(f'Setting exposure to {self.lf_acq_settings.exposure_time_ms} ms')

    #     # Configure acq timing - move to _setup_daq
    #     oryx_framerate = float(self._lf_mmc.get_property('Oryx', 'Frame Rate'))
    #     self.lf_acq_settings.slice_acq_rate = np.minimum(
    #         1000 / (self.lf_acq_settings.exposure_time_ms + MCL_STEP_TIME),
    #         np.floor(oryx_framerate))
    #     self.lf_acq_settings.channel_acq_rate = 1 / (
    #         self.lf_acq_settings.num_slices/self.lf_acq_settings.slice_acq_rate + 
    #         LC_CHANGE_TIME/1000)
    #     logger.debug(f'Maximum Oryx acquisition framerate: {oryx_framerate:.6f}')
    #     logger.debug(f'Current slice acquisition rate: {self.lf_acq_settings.slice_acq_rate:.6f}')
    #     logger.debug(f'Current channel acquisition rate: {self.lf_acq_settings.channel_acq_rate:.6f}')

    #     # Set flag
    #     self._lf_acq_set_up = True

    # def _setup_ls_acq(self):
    #     logger = logging.getLogger(__name__)
    #     if not hasattr(self, 'ls_acq_settings'):
    #         raise Exception('Please define LS acquisition settings.')
        
    #     # Setup camera
    #     # Edge Trigger acquires one frame is acquired for every trigger pulse
    #     # Rolling Shutter Exposure Out mode is high when all rows are exposing
    #     settings = (('Prime BSI Express', 'ReadoutRate', '200MHz 11bit'),
    #                 ('Prime BSI Express', 'Gain', '1-Full well'),
    #                 ('Prime BSI Express', 'TriggerMode', 'Edge Trigger'),
    #                 ('Prime BSI Express', 'ExposeOutMode', 'Rolling Shutter'))
    #     _apply_device_property_settings(self._ls_mmc, settings)

    #     # Setup sequencing
    #     _use_seq = 'On' if self.ls_acq_settings.use_sequence else 'Off'
    #     _apply_device_property_settings(self._ls_mmc, ((AP_GALVO_DAC, 'Sequence', _use_seq),))
    #     # Illuminate sample only when all rows are exposing, aka pseudo global shutter 
    #     _apply_device_property_settings(self._ls_mmc, (('TS2_TTL1-8', 'Blanking', 'On'),))

    #     # Setup ROI
    #     if self.ls_acq_settings.roi is not None:
    #         self._ls_mmc.set_roi(*self.ls_acq_settings.roi)
    #         logger.debug(f'Setting ROI to {self.ls_acq_settings.roi}')

    #     # Setup scan stage
    #     self._ls_mmc.set_property('Core', 'Focus', self.ls_acq_settings.z_stage)
    #     logger.debug(f'Setting focus stage to {self.ls_acq_settings.z_stage}')

    #     # Setup exposure
    #     self._ls_mmc.set_exposure(self.ls_acq_settings.exposure_time_ms)
    #     logger.debug(f'Setting exposure to {self.ls_acq_settings.exposure_time_ms} ms')

    #     # Configure acq timing - move to _setup_daq
    #     ls_readout_time_ms = np.around(
    #         float(self._ls_mmc.get_property('Prime BSI Express', 'Timing-ReadoutTimeNs'))*1e-6, 
    #         decimals=3)
    #     assert ls_readout_time_ms < self.ls_acq_settings.exposure_time_ms, \
    #         f'Exposure time needs to be greater than the {ls_readout_time_ms} sensor readout time'
    #     self.ls_acq_settings.slice_acq_rate = 1000 / (
    #         self.ls_acq_settings.exposure_time_ms + 
    #         ls_readout_time_ms + 
    #         LS_POST_READOUT_DELAY)
    #     _cam_max_fps = int(np.around(1000/ls_readout_time_ms))
    #     logger.debug(f'Maximum Prime BSI Express acquisition framerate: ~{_cam_max_fps}')
    #     logger.debug(f'Current slice acquisition rate: {self.ls_acq_settings.slice_acq_rate:.6f}')

    #     # Set flag
    #     self._ls_acq_set_up = True

    def setup_daq(self):
        # Determine label-free acq timing
        oryx_framerate = float(self._lf_mmc.get_property('Oryx', 'Frame Rate'))
        ## assumes all channels have the same exposure time
        self.lf_acq.slice_settings.acquisition_rate = np.minimum(
            1000 / (self.lf_acq.channel_settings.exposure_time_ms[0] + MCL_STEP_TIME),
            np.floor(oryx_framerate))
        self.lf_acq.channel_settings.acquisition_rate = 1 / (
            self.lf_acq.slice_settings.num_slices/self.lf_acq.slice_settings.acquisition_rate + 
            LC_CHANGE_TIME/1000)
        logger.debug(f'Maximum Oryx acquisition framerate: {oryx_framerate:.6f}')
        logger.debug(f'Current slice acquisition rate: {self.lf_acq.slice_settings.acquisition_rate:.6f}')
        logger.debug(f'Current channel acquisition rate: {self.lf_acq.channel_settings.acquisition_rate:.6f}')

        # Determine light-sheet acq timing
        ls_readout_time_ms = np.around(
            float(self._ls_mmc.get_property('Prime BSI Express', 'Timing-ReadoutTimeNs'))*1e-6, 
            decimals=3)
        for ls_exp_time in self.ls_acq.channel_settings.exposure_time_ms:
            assert ls_readout_time_ms < ls_exp_time, \
                f'Exposure time needs to be greater than the {ls_readout_time_ms} sensor readout time'
        ## TODO: here we are only taking the exposure time of the first channel
        self.ls_acq.slice_settings.acquisition_rate = 1000 / (
            self.ls_acq.channel_settings.exposure_time_ms[0] + 
            ls_readout_time_ms + 
            LS_POST_READOUT_DELAY)
        _cam_max_fps = int(np.around(1000/ls_readout_time_ms))
        logger.debug(f'Maximum Prime BSI Express acquisition framerate: ~{_cam_max_fps}')
        logger.debug(f'Current slice acquisition rate: {self.ls_acq.slice_settings.acquisition_rate:.6f}')

        # LF channel trigger - accommodates longer LC switching times
        self._lf_channel_ctr_task = nidaqmx.Task('LF Channel Counter')
        lf_channel_ctr = microscope_operations.setup_daq_counter(
            self._lf_channel_ctr_task, 
            co_channel='cDAQ1/_ctr0', 
            freq=self.lf_acq.channel_settings.acquisition_rate, 
            duty_cycle=0.1, 
            samples_per_channel=self.lf_acq.channel_settings.num_channels, 
            pulse_terminal='/cDAQ1/Ctr0InternalOutput')

        # LF Z trigger
        self._lf_z_ctr_task = nidaqmx.Task('LF Z Counter')
        lf_z_ctr = microscope_operations.setup_daq_counter(
            self._lf_z_ctr_task, 
            co_channel='cDAQ1/_ctr1', 
            freq=self.lf_acq.slice_settings.acquisition_rate, 
            duty_cycle=0.1, 
            samples_per_channel=self.lf_acq.slice_settings.num_slices, 
            pulse_terminal='/cDAQ1/PFI0')
        
        logger.debug('Setting up cDAQ1/_ctr0 as start trigger for cDAQ1/_ctr1')
        self._lf_z_ctr_task.triggers.start_trigger.cfg_dig_edge_start_trig(
            trigger_source='/cDAQ1/Ctr0InternalOutput', 
            trigger_edge=Slope.RISING)
        # will always return is_task_done = False after counter is started
        logger.debug('Setting up cDAQ1/_ctr1 as retriggerable')
        self._lf_z_ctr_task.triggers.start_trigger.retriggerable = True

        # LS frame trigger
        self._ls_ctr_task = nidaqmx.Task('LS Frame Counter')

        lf_z_ctr = microscope_operations.setup_daq_counter( 
            self._ls_ctr_task, 
            co_channel='cDAQ1/_ctr2', 
            freq=self.ls_acq.slice_settings.acquisition_rate, 
            duty_cycle=0.1, 
            samples_per_channel=self.ls_acq.slice_settings.num_slices, 
            pulse_terminal='/cDAQ1/PFI1')
        
        # TODO: Only trigger by Ctr0 is LF acquisition is also running
        logger.debug('Setting up cDAQ1/_ctr0 as start trigger for cDAQ1/_ctr2')
        self._ls_ctr_task.triggers.start_trigger.cfg_dig_edge_start_trig(
            trigger_source='/cDAQ1/Ctr0InternalOutput', 
            trigger_edge=Slope.RISING)
                
    def setup_autofocus(self):
        pass

    # def _generate_acq_events(self, acq_settings: ChannelSliceAcquisitionSettings):
    #     events =  multi_d_acquisition_events(    
    #         num_time_points = self.pt_acq_settings.num_timepoints,
    #         time_interval_s = self.pt_acq_settings.time_internal_s,
    #         z_start = acq_settings.z_start,
    #         z_end = acq_settings.z_end,
    #         z_step = acq_settings.z_step,
    #         channel_group = acq_settings.channel_group,
    #         channels = acq_settings.channels,
    #         xy_positions=None if self.pt_acq_settings.num_positions==0 else [
    #             self.pt_acq_settings.xyz_positions[p][:2] 
    #             for p in range(self.pt_acq_settings.num_positions)
    #         ],
    #         order = "tpcz")
        
    #     # Make sure all events have time and position axes so that dataset has 
    #     # consistent PTCZYX dimensions. Adding a position axis without giving
    #     # xyz coordinates will not move the microscope
    #     _append_event_axis(events, 'time')
    #     _append_event_axis(events, 'position')

        # return events
                
    def setup(self):
        logger.info('Setting up acquisition')

        logger.debug('Setting up label-free acquisition')
        self.lf_acq.setup()
        logger.debug('Finished setting up label-free acquisition')

        logger.debug('Setting up light-sheet acquisition')
        self.ls_acq.setup()
        logger.debug('Finished setting up light-sheet acquisition')

        logger.debug('Setting up DAQ')
        self.setup_daq()
        logger.debug('Finished setting up DAQ')

        logger.debug('Setting up autofocus')
        self.setup_autofocus()
        logger.debug('Finished setting up autofocus')

        self.update_position_settings()
    
    def acquire(self, name: str):

        def log_and_check_lf_counter(events):
            if isinstance(events, list):
                _event = events[0]
            else:
                _event = events  # events is a dict

            t_idx = _event['axes']['time']
            p_idx = _event['axes']['position']
            logger.debug(f'Preparing to acquire timepoint {t_idx} at position {self.position_settings.position_labels[p_idx]}')

            num_counter_samples = get_num_daq_counter_samples([self._lf_z_ctr_task, self._lf_channel_ctr_task])
            logger.debug(f'DAQ counters will generate a total of {num_counter_samples} pulses')

            event_seq_length = len(events)
            if num_counter_samples != event_seq_length:  # here events may be dict
                logger.error(f'Number of counter samples: {num_counter_samples}, is not equal to event sequence length:  {event_seq_length}.')
                logger.error('Aborting acquisition.')
                events = None

            return events
        
        def check_ls_counter(events):
            num_counter_samples = get_num_daq_counter_samples(self._ls_ctr_task)
            logger.debug(f'DAQ counters will generate a total of {num_counter_samples} pulses')

            event_seq_length = len(events)
            if num_counter_samples != event_seq_length:  # here events may be dict
                logger.error(f'Number of counter samples: {num_counter_samples}, is not equal to event sequence length:  {event_seq_length}.')
                logger.error('Aborting acquisition.')
                events = None
            
            return events
        
        def log_acq_start(events):
            if isinstance(events, list):
                _event = events[0]
            else:
                _event = events  # events is a dict

            t_idx = _event['axes']['time']
            p_idx = _event['axes']['position']
            logger.info(f'Starting acquisition of timepoint {t_idx} at position {self.position_settings.position_labels[p_idx]}')

            return events
        
        def log_and_start_lf_daq_counters(events):
            ctr_names = start_daq_counter([self._lf_z_ctr_task, self._lf_channel_ctr_task])
            logger.debug(f'Started DAQ counter tasks: {ctr_names}.')
            return events
        
        def log_and_start_ls_daq_counters(events):
            ctr_names = start_daq_counter([self._ls_ctr_task])
            logger.debug(f'Started DAQ counter tasks: {ctr_names}.')
            return events
        
        if self._lf_acq_set_up:
            lf_acq = Acquisition(
                directory=self._acq_dir, 
                name=f'{name}_labelfree',
                port=LF_ZMQ_PORT,
                pre_hardware_hook_fn=log_and_check_lf_counter,
                # pre_hardware_hook_fn=partial(
                #     confirm_num_daq_counter_samples, 
                #     [self._lf_z_ctr_task, self._lf_channel_ctr_task], 
                #     self.lf_acq_settings.num_channels*self.lf_acq_settings.num_slices, 
                #     self._verbose),
                post_hardware_hook_fn=log_acq_start,  # autofocus
                post_camera_hook_fn=log_and_start_lf_daq_counters,
                image_saved_fn=None,  # data processing and display
                show_display=False
            )
            
        if self._ls_acq_set_up:
            ls_acq = Acquisition(
                directory=self._acq_dir, 
                name=f'{name}_lightsheet', 
                port=LS_ZMQ_PORT, 
                pre_hardware_hook_fn=check_ls_counter, 
                post_camera_hook_fn=log_and_start_ls_daq_counters, 
                show_display=False
            )
            
        logger.info('Starting acquisition')
        for t_idx in range(self.time_settings.num_timepoints+1):
            for p_idx in range(self.position_settings.num_positions):
                lf_events = _generate_channel_slice_acq_events(self.lf_acq.channel_settings, self.lf_acq.slice_settings)
                ls_events = _generate_channel_slice_acq_events(self.ls_acq.channel_settings, self.ls_acq.slice_settings)

                for _event in lf_events:
                    _event['axes']['time'] = t_idx
                    _event['axes']['position'] = p_idx
                    _event['min_start_time'] = t_idx*self.time_settings.time_internal_s
                    _event['x'] = self.pt_acq_settings.xyz_positions[p_idx][0]
                    _event['y'] = self.pt_acq_settings.xyz_positions[p_idx][1]

                for _event in ls_events:
                    _event['axes']['time'] = t_idx
                    _event['axes']['position'] = p_idx
                    # start LS acq a bit earlier and wait for LF acq to trigger it
                    _event['min_start_time'] = np.maximum(
                        t_idx*self.time_settings.time_internal_s-0.2, 0  
                    )

                ls_acq.acquire(ls_events)  # it's important to start the LS acquisition first
                lf_acq.acquire(lf_events)
        ls_acq.mark_finished()
        lf_acq.mark_finished()


        # if self._ls_acq_enabled:
        #     ls_acq.acquire(ls_events)  # it's important to start the LS acquisition first
        #     ls_acq.mark_finished()
        # if self._lf_acq_enabled:
        #     lf_acq.acquire(lf_events)
        #     lf_acq.mark_finished()

        logger.debug('Waiting for acquisition to finish')
        if self.ls_acq.enabled:
            ls_acq.await_completion() 
            logger.debug('Light-sheet acquisition finished')
        if self.lf_acq.enabled:
            lf_acq.await_completion()
            logger.debug('Label-free acquisition finished')

        logger.debug('Stopping DAQ counter tasks')
        if self.ls_acq.enabled:
            self._ls_ctr_task.stop()
            self._ls_ctr_task.close()

            microscope_operations.set_property(
                self.ls_acq.mmc, 
                *('Prime BSI Express', 'TriggerMode', 'Internal Trigger')
            )

        if self.lf_acq.enabled:
            self._lf_z_ctr_task.stop()
            self._lf_z_ctr_task.close()
            self._lf_channel_ctr_task.stop()
            self._lf_channel_ctr_task.close()

            microscope_operations.set_property(
                self.lf_acq.mmc, 
                *('Oryx', 'Trigger Mode', 'Off')
            )
            # mmc1.set_property('Oryx', 'Frame Rate Control Enabled', oryx_framerate_enabled)
            # if oryx_framerate_enabled == '1': 
            #     mmc1.set_property('Oryx', 'Frame Rate', oryx_framerate)
        
        logger.info('Acquisition finished')

def _generate_channel_slice_acq_events(channel_settings, slice_settings):
    events =  multi_d_acquisition_events(    
        num_time_points = 1,
        time_interval_s = 0,
        z_start = slice_settings.z_start,
        z_end = slice_settings.z_end,
        z_step = slice_settings.z_step,
        channel_group = channel_settings.channel_group,
        channels = channel_settings.channels,
        order = "tpcz")
    
    return events

# def _append_event_axis(events:list, axis:str):
#     if axis not in events[0]['axes'].keys():
#         for _event in events:
#             _event['axes'][axis] = 0

# def _remove_event_key(events:list, key:str):
#     for _event in events:
#         _event.pop(key, None)

# def _apply_device_property_settings(mmc, settings: Iterable):
#     logger = logging.getLogger(__name__)

#     for _dev_settings in settings:
#         _device, _prop_name, _prop_val = _dev_settings
#         logger.debug(f'Setting {_device} {_prop_name} to {_prop_val}')
#         mmc.set_property(_device, _prop_name, _prop_val)

