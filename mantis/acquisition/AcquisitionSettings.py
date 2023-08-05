from dataclasses import field
from typing import List, Optional, Tuple

import numpy as np

from pydantic import ConfigDict, NonNegativeFloat, NonNegativeInt
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

    def __post_init__(self):
        assert len(self.xyz_positions) == len(self.position_labels)
        self.num_positions = len(self.xyz_positions)


@dataclass(config=config)
class ChannelSettings:
    exposure_time_ms: List[NonNegativeFloat] = field(default_factory=list)  # in ms
    channel_group: Optional[str] = None
    channels: List[str] = field(default_factory=list)
    use_sequencing: bool = False
    num_channels: int = field(init=False, default=0)
    acquisition_rate: float = field(init=False, default=None)

    def __post_init__(self):
        assert len(self.exposure_time_ms) == len(
            self.channels
        ), 'Number of channels must equal number of exposure times'
        self.num_channels = len(self.channels)


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
