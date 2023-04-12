import numpy as np
from typing import Optional, List, Tuple, Sequence
from pydantic.dataclasses import dataclass
from dataclasses import field


@dataclass
class ConfigSettings:
    config_group: str
    config_name: str


@dataclass
class DevicePropertySettings:
    device_name: str
    property_name: str
    property_value: str


@dataclass
class TimeSettings:
    num_timepoints: Optional[int] = 0
    time_internal_s: Optional[float] = 0  # in seconds


@dataclass
class PositionSettings:
    xyz_positions: list = field(default_factory=list)
    position_labels: List[str] = field(default_factory=list)
    num_positions: int = field(init=False, default=0)

    def __post_init__(self):
        assert len(self.xyz_positions) == len(self.position_labels)
        self.num_positions = len(self.xyz_positions)


@dataclass
class ChannelSettings:
    exposure_time_ms: List[float] = field(default_factory=list)  # in ms
    channel_group: Optional[str] = None
    channels: List[str] = field(default_factory=list)
    use_sequencing: bool = False
    num_channels: int = field(init=False, default=0)
    acquisition_rate: float = field(init=False, default=None)
    #TODO:associate the laser to which channel
    default_exposure_time: float = None

    def __post_init__(self):
        self.num_channels = len(self.channels)
        assert len(self.exposure_time_ms) == len(self.channels), \
            'Number of channels must equal number of exposure times'
        self.exposure_time_ms = self.default_exposure_time

@dataclass
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
        if self.z_step is not None:
            self.z_range = list(np.arange(self.z_start, self.z_end+self.z_step, self.z_step))
            self.num_slices = len(self.z_range)


@dataclass
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

@dataclass
class AutoexposureSettings:
    # max intensity used to define over-exposure
    max_intensity: int

    # min intensity used to define under-exposure
    min_intensity: int

    # minimum exposure time used to decide when to lower the laser power
    min_exposure_time: float

    # max exposure time used during adjustment for under-exposure
    max_exposure_time: float

    # the initial exposure time used when the laser power is lowered
    default_exposure_time: float

    # the minimum laser power (used to define autoexposure failure)
    min_laser_power: float

    # factor by which to decrease the exposure time or laser power
    # if a z-slice is found to be over-exposed
    relative_exposure_step: float

    # z-step size to use
    z_step_size: float

    # factor by which the exposure time will be rounded using
    # round(exposure_time/rounding_factor)*rounding_factor. When using the
    # Dragonfly spinning disk, the exposure time needs to be set in multiples of
    # 2.5 ms to avoid artifacts introduced by the spinning disk rotation
    rounding_factor: int = 5

    def round_exposure_time(self, exposure_time: float):
        exp_time = round(exposure_time / self.rounding_factor) * self.rounding_factor
        return float(exp_time)

    def __post_init__(self):
        for attr in ('default_exposure_time', 'min_exposure_time', 'max_exposure_time'):
            setattr(self, attr, self.round_exposure_time(getattr(self, attr)))

