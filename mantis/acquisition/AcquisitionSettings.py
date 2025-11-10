import copy
import warnings

from dataclasses import field
from typing import Dict, List, Literal, Optional, Tuple, Union, Any

import numpy as np

from pydantic.v1 import ConfigDict, NonNegativeFloat, NonNegativeInt, validator
from pydantic.v1.dataclasses import dataclass

config = ConfigDict(extra='forbid')


@dataclass(config=config)
class ConfigSettings:
    config_group: str
    config_name: str


@dataclass(config=config)
class DevicePropertySettings:
    device_name: str
    property_name: str
    property_value: str


@dataclass(config=config)
class TimeSettings:
    num_timepoints: NonNegativeInt = 0
    time_interval_s: NonNegativeFloat = 0  # in seconds


@dataclass(config=config)
class PositionSettings:
    xyz_positions: list = field(default_factory=list)
    position_labels: List[str] = field(default_factory=list)
    num_positions: int = field(init=False, default=0)
    well_ids: List[str] = field(init=False, default_factory=list)
    xyz_positions_shift: list = field(init=False, default_factory=list)

    def __post_init__(self):
        assert len(self.xyz_positions) == len(self.position_labels)
        self.num_positions = len(self.xyz_positions)
        self.xyz_positions_shift = copy.deepcopy(self.xyz_positions)

        try:
            # Look for "'A1-Site_0', 'H12-Site_1', ... " format
            hcs_labels = [pos.split("-Site_") for pos in self.position_labels]
            self.well_ids = [well for well, fov in hcs_labels]
        except ValueError:
            try:
                # Look for "'1-Pos000_000', '2-Pos000_001', ... "
                hcs_labels = [pos.split("-Pos") for pos in self.position_labels]
                self.well_ids = [well for well, fov in hcs_labels]
            except ValueError:
                # Default well is called "0"
                self.well_ids = ["0"] * self.num_positions


@dataclass(config=config)
class ChannelSettings:
    default_exposure_times_ms: Union[
        NonNegativeFloat, List[NonNegativeFloat], None
    ] = None  # in ms
    default_laser_powers: Union[
        NonNegativeFloat, List[NonNegativeFloat], List[None], None
    ] = None
    channel_group: Optional[str] = None
    channels: List[str] = field(default_factory=list)
    use_sequencing: bool = False
    use_autoexposure: Union[bool, List[bool]] = False
    num_channels: int = field(init=False, default=0)
    acquisition_rate: float = field(init=False, default=None)
    light_sources: List = field(init=False, default=None)
    # dictionaries with following structure: {well_id: list_of_exposure_times}
    exposure_times_per_well: Dict = field(init=None, default_factory=dict)
    laser_powers_per_well: Dict = field(init=None, default_factory=dict)
    min_exposure_time: NonNegativeFloat = 0  # in ms

    def __post_init__(self):
        self.num_channels = len(self.channels)
        for attr_name in (
            'default_exposure_times_ms',
            'default_laser_powers',
            'use_autoexposure',
            'light_sources',
        ):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, list):
                assert (
                    len(attr_value) == self.num_channels
                ), f'{attr_name} must be a list of length equal to the number of channels'
            else:
                # Note: [attr_value] * self.num_channels will simply create
                # references, which is not what we want
                setattr(self, attr_name, [attr_value for _ in range(self.num_channels)])


@dataclass(config=config)
class SliceSettings:
    z_stage_name: Optional[str] = None
    z_start: Optional[float] = None
    z_end: Optional[float] = None
    z_step: Optional[float] = None
    use_sequencing: bool = False
    num_slices: int = field(init=False, default=0)
    acquisition_rate: float = field(init=False, default=None)
    z_range: List[float] = field(init=False, repr=False, default_factory=list)

    @validator("z_step")
    def check_z_step(cls, v):
        if v is not None and v < 0.1:
            raise ValueError(
                "Provided z_step size is lower than 0.1 um, you may be using wrong units"
            )
        if v is not None and v == 0.143:
            warnings.warn('You may be using an outdated z_step size of 0.143 um')
        return v

    def __post_init__(self):
        # If one of z_params is provided, then they all need to be provided
        z_params = (self.z_stage_name, self.z_start, self.z_end, self.z_step)
        if any(z_params) and None in z_params:
            raise TypeError(
                'All of z_stage_name, z_start_, z_end, and z_step must be provided'
            )

        if self.z_step is not None:
            self.z_range = list(np.arange(self.z_start, self.z_end + self.z_step, self.z_step))
            self.num_slices = len(self.z_range)


@dataclass(config=config)
class MicroscopeSettings:
    roi: Optional[Tuple[int, int, int, int]] = None
    config_group_settings: List[ConfigSettings] = field(default_factory=list)
    device_property_settings: List[DevicePropertySettings] = field(default_factory=list)
    reset_device_properties: List[DevicePropertySettings] = field(default_factory=list)
    z_sequencing_settings: List[DevicePropertySettings] = field(default_factory=list)
    channel_sequencing_settings: List[DevicePropertySettings] = field(default_factory=list)
    use_autofocus: bool = False
    autofocus_stage: Optional[str] = None
    autofocus_method: Optional[str] = None
    use_o3_refocus: bool = False
    o3_refocus_config: Optional[ConfigSettings] = None
    o3_refocus_interval_min: Optional[int] = None
    o3_refocus_skip_wells: List[str] = field(default_factory=list)
    autotracker_config: Optional[ConfigSettings] = None


@dataclass
class AutoexposureSettings:
    # autoexposure method; currently only "manual" is implemented
    autoexposure_method: Literal['manual'] = None

    # rerun autoexposure for each timepoint at a given well
    rerun_each_timepoint: bool = False

    # min image intensity given as percent of dtype that defines under-exposure
    min_intensity_percent: Optional[float] = None

    # max image intensity given as percent of dtype that defines over-exposure
    max_intensity_percent: Optional[float] = None

    # minimum exposure time
    min_exposure_time_ms: Optional[float] = None

    # maximum exposure time
    max_exposure_time_ms: Optional[float] = None

    # the initial exposure time used when the laser power is lowered
    # TODO: the default exposure time should be the given in Channelsettings.default_exposure_times_ms
    # default_exposure_times_ms: float

    # the minimum laser power
    min_laser_power_mW: Optional[float] = None

    # TODO: get the max laser power from the laser if not provided here
    # TODO: different lasers may have different max power, how do we deal with that?
    max_laser_power_mW: Optional[float] = None

    # factors by which to change the exposure time or laser power
    # if an image is found to be incorrectly exposed
    relative_exposure_step: Optional[float] = None

    relative_laser_power_step: Optional[float] = None

    def __post_init__(self):
        for attr in (
            # "default_exposure_times_ms",
            "min_exposure_time_ms",
            "max_exposure_time_ms",
        ):
            attr_val = getattr(self, attr)
            if attr_val is not None:
                setattr(self, attr, round(attr_val, 1))
        for attr in ("min_laser_power_mW", "max_laser_power_mW"):
            attr_val = getattr(self, attr)
            if attr_val is not None:
                setattr(self, attr, round(attr_val, 1))


@dataclass
class AutotrackerSettings:
    tracking_method: Literal['phase_cross_correlation', 'template_matching', 'multi_otsu']
    tracking_interval: Optional[int] = 1  # TODO: add units
    shift_estimation_channel: Literal['phase', 'vs_nuclei', 'vs_membrane', 'bf'] = 'bf'
    scale_yx: Optional[float] = 1.0  # in um per pixel
    absolute_shift_limits_um: dict[str, Tuple[float, float]] = {'z': (0.5, 2), 'y': (2, 10), 'x': (2, 10)} # in um
    device: Optional[str] = 'cpu'
    zyx_dampening_factor: Optional[Union[Tuple[float, float, float], None]] = None
    # TODO: maybe do the ROI like in the ls_microscope_settings
    template_roi_zyx: Optional[Tuple[int, int, int]] = None
    template_channel: Optional[str] = None
    reconstruction: Optional[List[str]] = field(default_factory=list)
    phase_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    vs_config: Optional[Dict[str, Any]] = field(default_factory=dict)

    @validator("tracking_method")
    def check_tracking_method_options(cls, v):
        # Check if template matching options are provided and are not None
        if v == 'template_matching':
            if not all([cls.template_roi_zyx, cls.template_channel]):
                raise ValueError(
                    'template_roi_zyx and template_channel must be provided for template matching'
                )
        return v
