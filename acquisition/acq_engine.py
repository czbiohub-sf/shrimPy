import os
import time
from dataclasses import dataclass
import numpy as np
from pycromanager import (
    start_headless, 
    Core, 
    Studio,
    Acquisition, 
    multi_d_acquisition_events)

from functools import partial
from hook_functions.daq_control import (
    confirm_num_daq_counter_samples, 
    start_daq_counter)

import nidaqmx
from nidaqmx.constants import AcquisitionType, Slope
from nidaqmx.types import CtrTime

### Define constants
LF_ZMQ_PORT = 4827
LS_ZMQ_PORT = 5827   # we need to space out port numbers a bit
LS_POST_READOUT_DELAY = 0.05  # in ms
# LS_ROI = (0, 896, 2048, 256)  # centered in FOV
MCL_STEP_TIME = 1.5  # in ms
LC_CHANGE_TIME = 20  # in ms

@dataclass
class AcquisitionSettings:
    roi: list = None
    exposure_time_ms: float = 10  # in ms
    num_timepoints: int = 1
    time_internal_s: float = 0  # in seconds
    scan_stage: str = None
    z_start: float = 0
    z_end: float = 1
    z_step: float = 0.1
    channel_group: str = None
    channels: str = None
    use_sequence: bool = True

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

    def setup_lf_acq(self):
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
        oryx_framerate = self._lf_mmc.get_property('Oryx', 'Frame Rate')
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
        self._lf_mmc.set_exposure(self.ls_acq_settings.exposure_time_ms)

        # Set flag
        self._lf_acq_set_up = True

    def setup_ls_acq(self):
        pass

    def setup_daq(self):
        pass

    def acquire(self):
        pass
