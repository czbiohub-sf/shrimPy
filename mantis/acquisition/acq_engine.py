import os
import time
from dataclasses import dataclass, field
import numpy as np
from pycromanager import (
    start_headless, 
    Core, 
    Studio,
    Acquisition, 
    multi_d_acquisition_events)

from functools import partial
from .hook_functions.daq_control import (
    confirm_num_daq_counter_samples, 
    start_daq_counter)

import nidaqmx
from nidaqmx.constants import AcquisitionType, Slope
from nidaqmx.types import CtrTime

### Define constants
LF_ZMQ_PORT = 4827
LS_ZMQ_PORT = 5827   # we need to space out port numbers a bit
LS_POST_READOUT_DELAY = 0.05  # in ms
MCL_STEP_TIME = 1.5  # in ms
LC_CHANGE_TIME = 20  # in ms

@dataclass
class AcquisitionSettings:
    roi: tuple = None
    exposure_time_ms: float = 10  # in ms
    num_timepoints: int = 1
    time_internal_s: float = 0  # in seconds
    scan_stage: str = None
    z_start: float = 0
    z_end: float = 1
    z_step: float = 0.1
    channel_group: str = None
    channels: list = None
    use_sequence: bool = True
    num_slices: int = field(init=False)
    num_channels: int = field(init=False)
    slice_acq_rate: float = field(init=False)
    channel_acq_rate: float = field(init=False)

    def __post_init__(self):
        self.num_slices = len(np.arange(self.z_start, self.z_end, self.z_step))
        self.num_channels = len(self.channels)


class MantisAcquisition(object):
    """
    Base class for mantis  multimodal acquisition
    """

    def __init__(
            self,
            mm_app_path: str=r'C:\\Program Files\\Micro-Manager-nightly',
            mm_config_file: str=r'C:\\CompMicro_MMConfigs\\mantis\\mantis-LS.cfg',
            enable_ls_acq: bool=True,
            enable_lf_acq: bool=True,
            verbose: bool=False,
            ) -> None:
        
        """Initialize the mantis acquisition class and connect to Micro-manager

        Parameters
        ----------
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

        self._verbose = verbose
        self._ls_mm_app_path = mm_app_path
        self._ls_mm_config_file = mm_config_file
        self._lf_acq_enabled = enable_lf_acq
        self._ls_acq_enabled = enable_ls_acq
        self._lf_acq_set_up = False
        self._ls_acq_set_up = False
        
        # Connect to MM running LF acq
        if self._lf_acq_enabled:
            self._lf_mmc = Core(port=LF_ZMQ_PORT)
            self._lf_mmStudio = Studio(port=LF_ZMQ_PORT)

        # Connect to MM running LS acq
        if self._ls_acq_enabled:
            start_headless(
                self._ls_mm_app_path,
                self._ls_mm_config_file,
                LS_ZMQ_PORT)
            self._ls_mmc = Core(port=LS_ZMQ_PORT)

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        # Close PM bridges
        pass
            
    def define_lf_acq_settings(self, acq_settings:AcquisitionSettings):
        if not self._lf_acq_enabled:
            self._lf_acq_enabled = True
            raise RuntimeWarning('Enabling label-free acquisition')
        self.lf_acq_settings = acq_settings

    def define_ls_acq_settings(self, acq_settings:AcquisitionSettings):
        if not self._ls_acq_enabled:
            self._ls_acq_enabled = True
            raise RuntimeWarning('Enabling light-sheet acquisition')
        self.ls_acq_settings = acq_settings

    def _setup_lf_acq(self):
        if not hasattr(self, 'lf_acq_settings'):
            raise Exception('Please define LF acquisition settings.')

        # Setup light path
        self._lf_mmc.set_config('Imaging Path', 'Label-free')
        self._lf_mmc.set_config('Channel - LS', 'External Control')
        
        # Setup camera
        self._lf_mmc.set_property('Oryx', 'Line Selector', 'Line5') 
        # required for above selection to take effect
        self._lf_mmc.update_system_state_cache()
        self._lf_mmc.set_property('Oryx', 'Line Mode', 'Output')
        self._lf_mmc.set_property('Oryx', 'Line Source', 'ExposureActive')
        self._lf_mmc.set_property('Oryx', 'Line Selector', 'Line2')
        # required for above selection to take effect
        self._lf_mmc.update_system_state_cache()
        self._lf_mmc.set_property('Oryx', 'Line Mode', 'Input')
        self._lf_mmc.set_property('Oryx', 'Trigger Source', 'Line2')
        self._lf_mmc.set_property('Oryx', 'Trigger Mode', 'On')
        # required for external triggering at max frame rate
        self._lf_mmc.set_property('Oryx', 'Trigger Overlap', 'ReadOut')  
        oryx_framerate_enabled = self._lf_mmc.get_property('Oryx', 'Frame Rate Control Enabled')
        self._lf_mmc.set_property('Oryx', 'Frame Rate Control Enabled', '0')

        # Setup sequencing
        _use_seq = 'On' if self.lf_acq_settings.use_sequence else 'Off'
        self._lf_mmc.set_property('TS1_DAC01', 'Sequence', _use_seq)
        self._lf_mmc.set_property('TS1_DAC02', 'Sequence', _use_seq)
        self._lf_mmc.set_property('TS1_DAC06', 'Sequence', _use_seq)

        # Setup ROI
        if self.lf_acq_settings.roi is not None:
            self._lf_mmc.set_roi(*self.lf_acq_settings.roi)

        # Setup scan stage
        self._lf_mmc.set_property('Core', 'Focus', self.lf_acq_settings.scan_stage)

        # Setup channels
        if self.lf_acq_settings.channels is not None:
            self._lf_mmc.set_config(
                self.lf_acq_settings.channel_group, 
                self.lf_acq_settings.channels[0])
            
        # Setup exposure
        self._lf_mmc.set_exposure(self.lf_acq_settings.exposure_time_ms)

        # Configure acq timing
        oryx_framerate = self._lf_mmc.get_property('Oryx', 'Frame Rate')
        self.lf_acq_settings.slice_acq_rate = np.minimum(
            1000 / (self.lf_acq_settings.exposure_time_ms + MCL_STEP_TIME),
            np.floor(oryx_framerate))
        self.lf_acq_settings.channel_acq_rate = 1 / (
            self.lf_acq_settings.num_channels/self.lf_acq_settings.slice_acq_rate + 
            LC_CHANGE_TIME/1000)

        # Set flag
        self._lf_acq_set_up = True

    def _setup_ls_acq(self):
        if not hasattr(self, 'ls_acq_settings'):
            raise Exception('Please define LS acquisition settings.')
        
        # Setup camera
        # Set readout rate and gain
        self._ls_mmc.set_property('Prime BSI Express', 'ReadoutRate', '200MHz 11bit')
        self._ls_mmc.set_property('Prime BSI Express', 'Gain', '1-Full well')
        # One frame is acquired for every trigger pulse
        self._ls_mmc.set_property('Prime BSI Express', 'TriggerMode', 'Edge Trigger')
        # Rolling Shutter Exposure Out mode is high when all rows are exposing
        self._ls_mmc.set_property('Prime BSI Express', 'ExposeOutMode', 'Rolling Shutter')
        
        # Setup sequencing
        _use_seq = 'On' if self.ls_acq_settings.use_sequence else 'Off'
        self._ls_mmc.set_property('TS2_DAC03', 'Sequence', _use_seq)
        # Illuminate sample only when all rows are exposing, aka pseudo global shutter 
        self._ls_mmc.set_property('TS2_TTL1-8', 'Blanking', 'On')

        # Setup ROI
        if self.ls_acq_settings.roi is not None:
            self._ls_mmc.set_roi(*self.ls_acq_settings.roi)

        # Setup scan stage
        self._ls_mmc.set_property('Core', 'Focus', self.ls_acq_settings.scan_stage)

        # Setup exposure
        self._ls_mmc.set_exposure(self.ls_acq_settings.exposure_time_ms)

        # Configure acq timing
        ls_readout_time_ms = np.around(
            float(self._ls_mmc.get_property('Prime BSI Express', 'Timing-ReadoutTimeNs'))*1e-6, 
            decimals=3)
        assert ls_readout_time_ms < self.ls_acq_settings.exposure_time_ms, \
            f'Exposure time needs to be greater than the {ls_readout_time_ms} sensor readout time'
        self.ls_acq_settings.slice_acq_rate = 1000 / (
            self.ls_acq_settings.exposure_time_ms + 
            ls_readout_time_ms + 
            LS_POST_READOUT_DELAY)

        # Set flag
        self._ls_acq_set_up = True

    def _setup_daq(self):
        if self._lf_acq_set_up:
            # LF channel trigger - accommodates longer LC switching times
            self._lf_channel_ctr_task = nidaqmx.Task('LF Channel Counter')
            lf_channel_ctr = self._lf_channel_ctr_task.co_channels.add_co_pulse_chan_freq(
                'cDAQ1/_ctr0', 
                freq=self.lf_acq_settings.channel_acq_rate, 
                duty_cycle=0.1)
            self._lf_channel_ctr_task.timing.cfg_implicit_timing(
                sample_mode=AcquisitionType.FINITE, 
                samps_per_chan=self.lf_acq_settings.num_channels)
            lf_channel_ctr.co_pulse_term = '/cDAQ1/Ctr0InternalOutput'

            # LF Z trigger
            self._lf_z_ctr_task = nidaqmx.Task('LF Z Counter')
            lf_z_ctr = self._lf_z_ctr_task.co_channels.add_co_pulse_chan_freq(
                'cDAQ1/_ctr1', 
                freq=self.lf_acq_settings.slice_acq_rate, 
                duty_cycle=0.1)
            self._lf_z_ctr_task.timing.cfg_implicit_timing(
                sample_mode=AcquisitionType.FINITE, 
                samps_per_chan=self.lf_acq_settings.num_slices)
            self._lf_z_ctr_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                trigger_source='/cDAQ1/Ctr0InternalOutput', 
                trigger_edge=Slope.RISING)
            self._lf_z_ctr_task.triggers.start_trigger.retriggerable = True  # will always return is_task_done = False after counter is started
            lf_z_ctr.co_pulse_term = '/cDAQ1/PFI0'

        if self._ls_acq_set_up:
            # LS frame trigger
            self._ls_ctr_task = nidaqmx.Task('LS Frame Counter')
            ls_ctr = self._ls_ctr_task.co_channels.add_co_pulse_chan_freq(
                'cDAQ1/_ctr2', 
                freq=self.ls_acq_settings.slice_acq_rate, 
                duty_cycle=0.1)
            self._ls_ctr_task.timing.cfg_implicit_timing(
                sample_mode=AcquisitionType.FINITE, 
                samps_per_chan=self.ls_acq_settings.num_slices)
            ls_ctr.co_pulse_term = '/cDAQ1/PFI1'
            # Only trigger by Ctr0 is LF acquisition is also running
            if self._lf_acq_set_up:
                self._ls_ctr_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                    trigger_source='/cDAQ1/Ctr0InternalOutput', 
                    trigger_edge=Slope.RISING)
                
    def _setup_autofocus(self):
        pass

    def _generate_acq_events(self, acq_settings: AcquisitionSettings):
        events =  multi_d_acquisition_events(    
            num_time_points = acq_settings.num_timepoints,
            time_interval_s = acq_settings,
            z_start = acq_settings.z_start,
            z_end = acq_settings.z_end,
            z_step = acq_settings.z,
            channel_group = acq_settings.channel_group,
            channels = acq_settings.channels,
            channel_exposures_ms = acq_settings.exposure_time_ms,
            # xy_positions=None,
            order = "tpcz")
        
        return events
                
    def acquire(self, directory: str, name: str):
        if self._lf_acq_enabled:
            self._setup_lf_acq()
            lf_events = self._generate_acq_events(self.lf_acq_settings)
        if self._ls_acq_enabled:
            self._setup_ls_acq()
            ls_events = self._generate_acq_events(self.ls_acq_settings)
        self._setup_daq()
        self._setup_autofocus()
        
        if self._lf_acq_set_up:
            lf_acq = Acquisition(
                directory=directory, 
                name=f'{name}_labelfree',
                port=LF_ZMQ_PORT,
                pre_hardware_hook_fn=partial(
                    confirm_num_daq_counter_samples, 
                    [self._lf_z_ctr_task, self._lf_channel_ctr_task], 
                    self.lf_acq_settings.num_channels*self.lf_acq_settings.num_slices, 
                    self._verbose),
                post_hardware_hook_fn=None,  # autofocus
                post_camera_hook_fn=partial(
                    start_daq_counter, 
                    [self._lf_z_ctr_task, self._lf_channel_ctr_task],  # self._lf_z_ctr_task needs to be started first
                    self._verbose),
                image_saved_fn=None,  # data processing and display
                show_display=False)
            
        if self._ls_acq_set_up:
            ls_acq = Acquisition(
                directory=directory, 
                name=f'{name}_lightsheet', 
                port=LS_ZMQ_PORT, 
                pre_hardware_hook_fn=partial(
                    confirm_num_daq_counter_samples, 
                    self._ls_ctr_task, 
                    self.ls_acq_settings.num_slices, 
                    self._verbose), 
                post_camera_hook_fn=partial(
                    start_daq_counter, 
                    self._ls_ctr_task, 
                    self._verbose), 
                show_display=False)
            
        print('Starting acquisition')
        if self._ls_acq_enabled:
            ls_acq.acquire(ls_events)  # it's important to start the LS acquisition first
            ls_acq.mark_finished()
        if self._lf_acq_enabled:
            lf_acq.acquire(lf_events)
            lf_acq.mark_finished()

        if self._verbose:
            print('Waiting for acquisition to finish')
        if self._ls_acq_enabled:
            ls_acq.await_completion(); print('LS finished')
        if self._lf_acq_enabled:
            lf_acq.await_completion(); print('LF finished')

        if self._verbose:
            print('Stopping and closing DAQ counters')
        if self._ls_acq_enabled:
            self._ls_ctr_task.stop()
            self._ls_ctr_task.close()

            self._ls_mmc.set_property('Prime BSI Express', 'TriggerMode', 'Internal Trigger')

        if self._lf_acq_enabled:
            self._lf_z_ctr_task.stop()
            self._lf_z_ctr_task.close()
            self._lf_channel_ctr_task.stop()
            self._lf_channel_ctr_task.close()

            self._lf_mmc.set_property('Oryx', 'Trigger Mode', 'Off')
            # mmc1.set_property('Oryx', 'Frame Rate Control Enabled', oryx_framerate_enabled)
            # if oryx_framerate_enabled == '1': 
            #     mmc1.set_property('Oryx', 'Frame Rate', oryx_framerate)
