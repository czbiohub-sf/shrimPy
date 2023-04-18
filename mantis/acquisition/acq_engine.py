import logging
import os
import time

from dataclasses import asdict
from datetime import datetime
from functools import partial

import nidaqmx
import numpy as np

from nidaqmx.constants import Slope
from pycromanager import Acquisition, Core, Studio, multi_d_acquisition_events, start_headless
from pycromanager.acq_util import cleanup

from mantis.acquisition import microscope_operations
from mantis.acquisition.hook_functions import config
from mantis.acquisition.logger import configure_logger

# isort: off
from mantis.acquisition.BaseSettings import (
    TimeSettings,
    PositionSettings,
    ChannelSettings,
    SliceSettings,
    MicroscopeSettings,
)
from mantis.acquisition.hook_functions.pre_hardware_hook_functions import (
    log_preparing_acquisition,
    log_preparing_acquisition_check_counter,
    check_num_counter_samples,
)
from mantis.acquisition.hook_functions.post_hardware_hook_functions import (
    log_acquisition_start,
    update_daq_freq,
)
from mantis.acquisition.hook_functions.post_camera_hook_functions import (
    start_daq_counters,
)
from mantis.acquisition.hook_functions.image_saved_hook_functions import (
    check_lf_acq_finished,
    check_ls_acq_finished,
)

# isort: on


# Define constants
LF_ZMQ_PORT = 4827
LS_ZMQ_PORT = 5827  # we need to space out port numbers a bit
LS_POST_READOUT_DELAY = 0.05  # delay before acquiring next frame, in ms
MCL_STEP_TIME = 1.5  # in ms
LC_CHANGE_TIME = 20  # in ms
LS_CHANGE_TIME = 200  # time needed to change LS filter wheel, in ms

logger = logging.getLogger(__name__)


class BaseChannelSliceAcquisition(object):
    """
    Base class which handles setup of the label-free or light-sheet acquisition

    Parameters
    ----------
    enabled : bool, optional
        Flag if acquisition should be enabled, by default True
    mm_app_path : str, optional
        Path to Micro-manager which will be launched in headless mode, by default None
    mm_config_file : str, optional
        Path to config file for the headless acquisition, by default None
    zmq_port : int, optional
        ZeroMQ port of the acquisition, by default 4827
    core_log_path : str, optional
        Path where the headless acquisition core logs will be saved, by default ''
    """

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
                    buffer_size_mb=2048,
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
    def channel_settings(self, settings: ChannelSettings):
        logger.debug(
            f'{self.type.capitalize()} acquisition will have the following settings: {asdict(settings)}'
        )
        self._channel_settings = settings

    @slice_settings.setter
    def slice_settings(self, settings: SliceSettings):
        settings_dict = {key: val for key, val in asdict(settings).items() if key != 'z_range'}
        logger.debug(
            f'{self.type.capitalize()} acquisition will have the following settings: {settings_dict}'
        )
        self._slice_settings = settings

    @microscope_settings.setter
    def microscope_settings(self, settings: MicroscopeSettings):
        logger.debug(
            f'{self.type.capitalize()} acquisition will have the following settings: {asdict(settings)}'
        )
        self._microscope_settings = settings

    def setup(self):
        """
        Apply acquisition settings as specified by the class properties
        """
        if self.enabled:
            # Apply microscope config settings
            for settings in self.microscope_settings.config_group_settings:
                microscope_operations.set_config(
                    self.mmc, settings.config_group, settings.config_name
                )

            # Apply microscope device property settings
            for settings in self.microscope_settings.device_property_settings:
                microscope_operations.set_property(
                    self.mmc,
                    settings.device_name,
                    settings.property_name,
                    settings.property_value,
                )

            # Apply ROI
            if self.microscope_settings.roi is not None:
                microscope_operations.set_roi(self.mmc, self.microscope_settings.roi)

            # Setup z scan stage
            microscope_operations.set_property(
                self.mmc, 'Core', 'Focus', self.slice_settings.z_stage_name
            )

            # Note: sequencing should be turned off by default
            # Setup z sequencing
            if self.slice_settings.use_sequencing:
                for settings in self.microscope_settings.z_sequencing_settings:
                    microscope_operations.set_property(
                        self.mmc,
                        settings.device_name,
                        settings.property_name,
                        settings.property_value,
                    )

            # Setup channel sequencing
            if self.channel_settings.use_sequencing:
                for settings in self.microscope_settings.channel_sequencing_settings:
                    microscope_operations.set_property(
                        self.mmc,
                        settings.device_name,
                        settings.property_name,
                        settings.property_value,
                    )

    def reset(self):
        """
        Reset the microscope device properties, typically at the end of the acquisition
        """
        if self.enabled:
            # Reset device property settings
            for settings in self.microscope_settings.reset_device_properties:
                microscope_operations.set_property(
                    self.mmc,
                    settings.device_name,
                    settings.property_name,
                    settings.property_value,
                )


class MantisAcquisition(object):
    """
    Acquisition class for simultaneous label-free and light-sheet acquisition on
    the mantis microscope.

    Parameters
    ----------
    acquisition_directory : str
        Directory where acquired data will be saved
    acquisition_name : str
        Name of the acquisition
    mm_app_path : str, optional
        Path to Micro-manager installation directory which runs the light-sheet
        acquisition, by default 'C:\\Program Files\\Micro-Manager-nightly'
    config_file : str, optional
        Path to config file which runs the light-sheet acquisition, by default
        'C:\\CompMicro_MMConfigs\\mantis\\mantis-LS.cfg'
    enable_ls_acq : bool, optional
        Set to False if only acquiring label-free data, by default True
    enable_lf_acq : bool, optional
        Set to False if only acquiring fluorescence light-sheet data, by default
        True
    demo_run : bool, optional
        Set to True if using the MicroManager demo config, by default False
    verbose : bool, optional
        By default False

    Examples
    -----
    This class should be used with a context manager for proper cleanup of the
    acquisition:

    >>> with MantisAcquisition(...) as acq:
            # define acquisition settings
            acq.time_settings = TimeSettings(...)
            acq.lf_acq.channel_settings = ChannelSettings(...)
            ...

            acq.setup()
            acq.acquire()
    """

    def __init__(
        self,
        acquisition_directory: str,
        acquisition_name: str,
        mm_app_path: str = r'C:\\Program Files\\Micro-Manager-nightly',
        mm_config_file: str = r'C:\\CompMicro_MMConfigs\\mantis\\mantis-LS.cfg',
        enable_ls_acq: bool = True,
        enable_lf_acq: bool = True,
        demo_run: bool = False,
        verbose: bool = False,
    ) -> None:

        self._root_dir = acquisition_directory
        self._acq_name = acquisition_name
        self._demo_run = demo_run
        self._verbose = verbose

        if not enable_lf_acq or not enable_ls_acq:
            raise Exception('Disabling LF or LS acquisition is not currently supported')

        # Create acquisition directory
        self._acq_dir = _create_acquisition_directory(self._root_dir, self._acq_name)

        # Setup logger
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        configure_logger(
            os.path.join(self._acq_dir, f'mantis_acquisition_log_{timestamp}.txt')
        )

        # initialize time and position settings
        self._time_settings = TimeSettings()
        self._position_settings = PositionSettings()

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
            core_log_path=os.path.join(
                mm_app_path, 'CoreLogs', f'CoreLog{timestamp}_headless.txt'
            ),
        )

    @property
    def time_settings(self):
        return self._time_settings

    @property
    def position_settings(self):
        return self._position_settings

    @time_settings.setter
    def time_settings(self, settings: TimeSettings):
        logger.debug(
            f'Mantis acquisition will have the following settings: {asdict(settings)}'
        )
        self._time_settings = settings

    @position_settings.setter
    def position_settings(self, settings: PositionSettings):
        logger.debug(
            f'Mantis acquisition will have the following settings: {asdict(settings)}'
        )
        self._position_settings = settings

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        # Shut down DAQ
        if not self._demo_run:
            self.cleanup_daq()

        # Reset LF and LS acquisitions
        self.lf_acq.reset()
        self.ls_acq.reset()

        # Close PM bridges
        cleanup()

    def update_position_settings(self):
        """
        Fetch positions defined in the Micro-manager Position List Manager
        """
        mm_pos_list = self.lf_acq.mmStudio.get_position_list_manager().get_position_list()
        mm_number_of_positions = mm_pos_list.get_number_of_positions()
        autofocus_stage = self.lf_acq.microscope_settings.autofocus_stage

        if self.position_settings.num_positions == 0:
            if mm_number_of_positions > 0:
                logger.debug('Fetching position list from Micro-manager')

                xyz_position_list, position_labels = microscope_operations.get_position_list(
                    self.lf_acq.mmStudio, autofocus_stage
                )
            else:
                logger.debug('Fetching current position from Micro-manager')

                xyz_position_list = [
                    (
                        self.lf_acq.mmc.get_x_position(),
                        self.lf_acq.mmc.get_y_position(),
                        self.lf_acq.mmc.get_position(autofocus_stage)
                        if autofocus_stage
                        else None,
                    )
                ]
                position_labels = ['Current']

            self.position_settings = PositionSettings(
                xyz_positions=xyz_position_list,
                position_labels=position_labels,
            )

    def setup_daq(self):
        """
        Setup the NI DAQ to output trigger pulses for the label-free and
        light-sheet acquisitions. Acquisition can be sequenced across z slices
        and channels
        """
        # Determine label-free acq timing
        oryx_framerate = float(self.lf_acq.mmc.get_property('Oryx', 'Frame Rate'))
        # assumes all channels have the same exposure time
        self.lf_acq.slice_settings.acquisition_rate = np.minimum(
            1000 / (self.lf_acq.channel_settings.exposure_time_ms[0] + MCL_STEP_TIME),
            np.floor(oryx_framerate),
        )
        self.lf_acq.channel_settings.acquisition_rate = 1 / (
            self.lf_acq.slice_settings.num_slices / self.lf_acq.slice_settings.acquisition_rate
            + LC_CHANGE_TIME / 1000
        )
        logger.debug(f'Maximum Oryx acquisition framerate: {oryx_framerate:.6f}')
        logger.debug(
            f'Current label-free slice acquisition rate: {self.lf_acq.slice_settings.acquisition_rate:.6f}'
        )
        logger.debug(
            f'Current label-free channel acquisition rate: {self.lf_acq.channel_settings.acquisition_rate:.6f}'
        )

        # Determine light-sheet acq timing
        ls_readout_time_ms = np.around(
            float(self.ls_acq.mmc.get_property('Prime BSI Express', 'Timing-ReadoutTimeNs'))
            * 1e-6,
            decimals=3,
        )
        _cam_max_fps = int(np.around(1000 / ls_readout_time_ms))
        for ls_exp_time in self.ls_acq.channel_settings.exposure_time_ms:
            assert (
                ls_readout_time_ms < ls_exp_time
            ), f'Exposure time needs to be greater than the {ls_readout_time_ms} ms sensor readout time'
        self.ls_acq.slice_settings.acquisition_rate = [
            1000 / (exp + ls_readout_time_ms + LS_POST_READOUT_DELAY)
            for exp in self.ls_acq.channel_settings.exposure_time_ms
        ]
        # self.ls_acq.channel_settings.acquisition_rate = [
        #     1 / (self.ls_acq.slice_settings.num_slices/acq_rate + LS_CHANGE_TIME/1000)
        #     for acq_rate in self.ls_acq.slice_settings.acquisition_rate
        # ]
        acq_rates = list(np.around(self.ls_acq.slice_settings.acquisition_rate, decimals=6))
        logger.debug(f'Maximum Prime BSI Express acquisition framerate: ~{_cam_max_fps}')
        logger.debug(f'Current light-sheet slice acquisition rate: {acq_rates}')
        # logger.debug(f'Current light-sheet channel acquisition rate: ~{self.ls_acq.channel_settings.acquisition_rate}')

        # LF channel trigger - accommodates longer LC switching times
        self._lf_channel_ctr_task = nidaqmx.Task('LF Channel Counter')
        microscope_operations.setup_daq_counter(
            self._lf_channel_ctr_task,
            co_channel='cDAQ1/_ctr0',
            freq=self.lf_acq.channel_settings.acquisition_rate,
            duty_cycle=0.1,
            samples_per_channel=self.lf_acq.channel_settings.num_channels,
            pulse_terminal='/cDAQ1/Ctr0InternalOutput',
        )

        # Sutter filter wheel is not capable of sequencing, so no need for channel sync here
        # # LS channel trigger
        # self._ls_channel_ctr_task = nidaqmx.Task('LS Channel Counter')
        # ls_channel_ctr = microscope_operations.setup_daq_counter(
        #     self._ls_channel_ctr_task,
        #     co_channel='cDAQ1/_ctr1',
        #     freq=self.ls_acq.channel_settings.acquisition_rate,
        #     duty_cycle=0.1,
        #     samples_per_channel=self.ls_acq.channel_settings.num_channels,
        #     pulse_terminal='/cDAQ1/Ctr1InternalOutput')

        # LF Z trigger
        self._lf_z_ctr_task = nidaqmx.Task('LF Z Counter')
        microscope_operations.setup_daq_counter(
            self._lf_z_ctr_task,
            co_channel='cDAQ1/_ctr2',
            freq=self.lf_acq.slice_settings.acquisition_rate,
            duty_cycle=0.1,
            samples_per_channel=self.lf_acq.slice_settings.num_slices,
            pulse_terminal='/cDAQ1/PFI0',
        )

        # LS Z trigger
        # LS Z counter will start with a software command
        # Counter frequency is updated for each channel in post-camera hook fn
        self._ls_z_ctr_task = nidaqmx.Task('LS Z Counter')
        microscope_operations.setup_daq_counter(
            self._ls_z_ctr_task,
            co_channel='cDAQ1/_ctr3',
            freq=self.ls_acq.slice_settings.acquisition_rate[0],
            duty_cycle=0.1,
            samples_per_channel=self.ls_acq.slice_settings.num_slices,
            pulse_terminal='/cDAQ1/PFI1',
        )

        # # The LF Channel counter task serve as a master start trigger
        # # LF Channel counter triggers LS Channel counter
        # logger.debug('Setting up LF Channel counter as start trigger for LS Channel counter')
        # self._ls_channel_ctr_task.triggers.start_trigger.cfg_dig_edge_start_trig(
        #     trigger_source='/cDAQ1/Ctr0InternalOutput',
        #     trigger_edge=Slope.RISING)

        # # LS Channel counter triggers LS Z counter
        # logger.debug('Setting up LS Channel counter as start trigger for LS Z counter')
        # self._ls_z_ctr_task.triggers.start_trigger.cfg_dig_edge_start_trig(
        #     trigger_source='/cDAQ1/Ctr1InternalOutput',
        #     trigger_edge=Slope.RISING)
        # # will always return is_task_done = False after counter is started
        # logger.debug('Setting up LS Z counter as retriggerable')
        # self._ls_z_ctr_task.triggers.start_trigger.retriggerable = True

        # LF Channel counter triggers LF Z counter
        logger.debug('Setting up LF Channel counter as start trigger for LF Z counter')
        self._lf_z_ctr_task.triggers.start_trigger.cfg_dig_edge_start_trig(
            trigger_source='/cDAQ1/Ctr0InternalOutput', trigger_edge=Slope.RISING
        )
        # will always return is_task_done = False after counter is started
        logger.debug('Setting up LF Z counter as retriggerable')
        self._lf_z_ctr_task.triggers.start_trigger.retriggerable = True

    def cleanup_daq(self):
        logger.debug('Stopping DAQ counter tasks')
        if self.ls_acq.enabled:
            self._ls_z_ctr_task.stop()
            self._ls_z_ctr_task.close()

        if self.lf_acq.enabled:
            self._lf_z_ctr_task.stop()
            self._lf_z_ctr_task.close()
            self._lf_channel_ctr_task.stop()
            self._lf_channel_ctr_task.close()

    def setup_autofocus(self):
        if self.lf_acq.microscope_settings.use_autofocus:
            autofocus_method = self.lf_acq.microscope_settings.autofocus_method
            logger.debug(f'Setting autofocus method as {autofocus_method}')
            self.lf_acq.mmc.set_auto_focus_device(autofocus_method)
        else:
            logger.debug('Autofocus is not enabled')

    def go_to_position(self, position_index: int):
        # Move slowly for short distances such that autofocus can stay engaged.
        # Autofocus typically fails when moving long distances, so we can move
        # quickly and re-engage
        slow_speed = 2.0
        fast_speed = 5.75
        short_distance = 2000  # in um
        p_label = self.position_settings.position_labels[position_index]

        # only change stage speed if using autofocus
        if self.lf_acq.microscope_settings.use_autofocus and not self._demo_run:
            current_xy_position = np.asarray(
                [self.lf_acq.mmc.get_x_position(), self.lf_acq.mmc.get_y_position()]
            )
            target_xy_position = np.asarray(
                self.position_settings.xyz_positions[position_index][:2]
            )
            distance = np.linalg.norm(target_xy_position - current_xy_position)

            speed = slow_speed if distance < short_distance else fast_speed
            microscope_operations.set_property(
                self.lf_acq.mmc, 'XYStage:XY:31', 'MotorSpeedX-S(mm/s)', speed
            )
            microscope_operations.set_property(
                self.lf_acq.mmc, 'XYStage:XY:31', 'MotorSpeedY-S(mm/s)', speed
            )

        logger.debug(
            f'Moving to position {p_label} with coordinates {self.position_settings.xyz_positions[position_index]}'
        )
        microscope_operations.set_xy_position(
            self.lf_acq.mmc, self.position_settings.xyz_positions[position_index][:2]
        )
        microscope_operations.wait_for_device(
            self.lf_acq.mmc, self.lf_acq.mmc.get_xy_stage_device()
        )

        # Note: only set the z position if not using autofocus. Calling
        # set_z_position will disengage continuous autofocus. The autofocus
        # algorithm sets the z position independently
        if not self.lf_acq.microscope_settings.use_autofocus:
            microscope_operations.set_z_position(
                self.lf_acq.mmc,
                self.lf_acq.microscope_settings.autofocus_stage,
                self.position_settings.xyz_positions[position_index][2],
            )
            microscope_operations.wait_for_device(
                self.lf_acq.mmc,
                self.lf_acq.microscope_settings.autofocus_stage,
            )

    def setup(self):
        """
        Setup the mantis acquisition. This method sets up the label-free
        acquisition, the light-sheet acquisition, the NI DAQ, the autofocus, and
        fetches positions defined in the Micro-manager Position List Manager
        """

        logger.info('Setting up acquisition')

        logger.debug('Setting up label-free acquisition')
        self.lf_acq.setup()
        logger.debug('Finished setting up label-free acquisition')

        logger.debug('Setting up light-sheet acquisition')
        self.ls_acq.setup()
        logger.debug('Finished setting up light-sheet acquisition')

        logger.debug('Setting up DAQ')
        if not self._demo_run:
            self.setup_daq()
            logger.debug('Finished setting up DAQ')
        else:
            logger.debug('DAQ setup is not supported in demo mode')

        logger.debug('Setting up autofocus')
        self.setup_autofocus()
        logger.debug('Finished setting up autofocus')

        self.update_position_settings()

    def acquire(self):
        """
        Simultaneously acquire label-free and light-sheet data over multiple
        positions and time points.
        """

        # define LF hook functions
        if self._demo_run:
            lf_pre_hardware_hook_fn = partial(
                log_preparing_acquisition, self.position_settings.position_labels
            )
            lf_post_camera_hook_fn = None
        else:
            lf_pre_hardware_hook_fn = partial(
                log_preparing_acquisition_check_counter,
                self.position_settings.position_labels,
                [self._lf_z_ctr_task, self._lf_channel_ctr_task],
            )
            lf_post_camera_hook_fn = partial(
                start_daq_counters, [self._lf_z_ctr_task, self._lf_channel_ctr_task]
            )
        lf_post_hardware_hook_fn = partial(
            log_acquisition_start, self.position_settings.position_labels
        )
        lf_image_saved_fn = check_lf_acq_finished

        # define LF acquisition
        lf_acq = Acquisition(
            directory=self._acq_dir,
            name=f'{self._acq_name}_labelfree',
            port=LF_ZMQ_PORT,
            pre_hardware_hook_fn=lf_pre_hardware_hook_fn,
            post_hardware_hook_fn=lf_post_hardware_hook_fn,
            post_camera_hook_fn=lf_post_camera_hook_fn,
            image_saved_fn=lf_image_saved_fn,  # data processing and display
            show_display=False,
        )

        # define LS hook functions
        if self._demo_run:
            ls_pre_hardware_hook_fn = None
            ls_post_hardware_hook_fn = None
            ls_post_camera_hook_fn = None
        else:
            ls_pre_hardware_hook_fn = partial(check_num_counter_samples, [self._ls_z_ctr_task])
            ls_post_hardware_hook_fn = partial(
                update_daq_freq,
                self._ls_z_ctr_task,
                self.ls_acq.channel_settings.channels,
                self.ls_acq.slice_settings.acquisition_rate,
            )
            ls_post_camera_hook_fn = partial(start_daq_counters, [self._ls_z_ctr_task])
        ls_image_saved_fn = check_ls_acq_finished

        # define LS acquisition
        ls_acq = Acquisition(
            directory=self._acq_dir,
            name=f'{self._acq_name}_lightsheet',
            port=LS_ZMQ_PORT,
            pre_hardware_hook_fn=ls_pre_hardware_hook_fn,
            post_hardware_hook_fn=ls_post_hardware_hook_fn,
            post_camera_hook_fn=ls_post_camera_hook_fn,
            image_saved_fn=ls_image_saved_fn,
            saving_queue_size=500,
            show_display=False,
        )

        # define acquisition events
        lf_events = _generate_channel_slice_acq_events(
            self.lf_acq.channel_settings, self.lf_acq.slice_settings
        )
        ls_events = _generate_channel_slice_acq_events(
            self.ls_acq.channel_settings, self.ls_acq.slice_settings
        )

        logger.info('Starting acquisition')
        for t_idx in range(self.time_settings.num_timepoints):
            t_start = time.time()
            for p_idx in range(self.position_settings.num_positions):
                p_label = self.position_settings.position_labels[p_idx]

                # move to position
                self.go_to_position(p_idx)

                # autofocus
                if self.lf_acq.microscope_settings.use_autofocus:
                    autofocus_success = microscope_operations.autofocus(
                        self.lf_acq.mmc,
                        self.lf_acq.mmStudio,
                        self.lf_acq.microscope_settings.autofocus_stage,
                        self.position_settings.xyz_positions[p_idx][2],
                    )
                    if not autofocus_success:
                        logger.error(
                            f'Autofocus failed. Aborting acquisition for timepoint {t_idx} at position {p_label}'
                        )
                        continue

                # start acquisition
                for _event in lf_events:
                    _event['axes']['time'] = t_idx
                    _event['axes']['position'] = p_idx
                    _event['min_start_time'] = 0

                for _event in ls_events:
                    _event['axes']['time'] = t_idx
                    _event['axes']['position'] = p_idx
                    _event['min_start_time'] = 0

                config.lf_last_img_idx = lf_events[-1]['axes']
                config.ls_last_img_idx = ls_events[-1]['axes']
                config.lf_acq_finished = False
                config.ls_acq_finished = False

                ls_acq.acquire(ls_events)
                lf_acq.acquire(lf_events)

                # wait for PT acquisition to finish
                while any((not config.lf_acq_finished, not config.ls_acq_finished)):
                    time.sleep(0.2)
                    # if not config.lf_acq_finished:
                    #     print('Waiting for LF acquisition to finish')
                    # if not config.ls_acq_finished:
                    #     print('Waiting for LS acquisition to finish')
            # wait for delay between timepoints
            while time.time() - t_start < self.time_settings.time_internal_s:
                time.sleep(1)

        ls_acq.mark_finished()
        lf_acq.mark_finished()

        logger.debug('Waiting for acquisition to finish')
        if self.ls_acq.enabled:
            ls_acq.await_completion()
            logger.debug('Light-sheet acquisition finished')
        if self.lf_acq.enabled:
            lf_acq.await_completion()
            logger.debug('Label-free acquisition finished')

        # TODO: move scan stages to zero

        # Close ndtiff dataset - not sure why this is necessary
        lf_acq._dataset.close()
        ls_acq._dataset.close()

        logger.info('Acquisition finished')


def _generate_channel_slice_acq_events(channel_settings, slice_settings):
    events = multi_d_acquisition_events(
        num_time_points=1,
        time_interval_s=0,
        z_start=slice_settings.z_start,
        z_end=slice_settings.z_end,
        z_step=slice_settings.z_step,
        channel_group=channel_settings.channel_group,
        channels=channel_settings.channels,
        channel_exposures_ms=channel_settings.exposure_time_ms,
        order="tpcz",
    )

    return events


def _create_acquisition_directory(root_dir, acq_name, idx=1):
    acq_dir = os.path.join(root_dir, f'{acq_name}_{idx}')
    try:
        os.mkdir(acq_dir)
    except OSError:
        return _create_acquisition_directory(root_dir, acq_name, idx + 1)
    return acq_dir
