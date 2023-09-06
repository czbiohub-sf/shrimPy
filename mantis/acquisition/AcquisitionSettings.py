import warnings

from dataclasses import field
from typing import List, Optional, Tuple, Union

import numpy as np

from pydantic import ConfigDict, NonNegativeFloat, NonNegativeInt, validator
from pydantic.dataclasses import dataclass

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

    def __post_init__(self):
        assert len(self.xyz_positions) == len(self.position_labels)
        self.num_positions = len(self.xyz_positions)

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
                self.well_ids = [None] * self.num_positions


@dataclass(config=config)
class ChannelSettings:
    exposure_time_ms: Union[NonNegativeFloat, List[NonNegativeFloat]] = field(
        default_factory=list
    )  # in ms
    channel_group: Optional[str] = None
    channels: List[str] = field(default_factory=list)
    use_sequencing: bool = False
    use_autoexposure: Union[bool, List[bool]] = False
    num_channels: int = field(init=False, default=0)
    acquisition_rate: float = field(init=False, default=None)
    light_sources: List = field(init=False, default=None)

    def __post_init__(self):
        self.num_channels = len(self.channels)
        for attr_name in ('exposure_time_ms', 'use_autoexposure', 'light_sources'):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, list):
                assert (
                    len(attr_value) == self.num_channels
                ), f'{attr_name} must be a list of length equal to the number of channels'
            else:
                setattr(self, attr_name, [attr_value] * self.num_channels)


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


@dataclass
class AutoexposureSettings:
    # min intensity percent to define under-exposure
    min_intensity_percent: float

    # max intensity percent to define over-exposure
    max_intensity_percent: float

    # minimum exposure time used to decide when to lower the laser power
    min_exposure_time_ms: float

    # max exposure time used during adjustment for under-exposure
    max_exposure_time_ms: float

    # the initial exposure time used when the laser power is lowered
    default_exposure_time_ms: float

    # the minimum laser power (used to define autoexposure failure)
    min_laser_power_mW: float

    max_laser_power_mW: float

    # factor by which to decrease the exposure time or laser power
    # if a z-slice is found to be over-exposed
    relative_exposure_step: float

    relative_laser_power_step: float

    def __post_init__(self):
        for attr in (
            "default_exposure_time_ms",
            "min_exposure_time_ms",
            "max_exposure_time_ms",
        ):
            setattr(self, attr, round(getattr(self, attr)))
        for attr in ("min_laser_power_mW", "max_laser_power_mW"):
            setattr(self, attr, round(getattr(self, attr)))
