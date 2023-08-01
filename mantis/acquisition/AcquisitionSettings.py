from dataclasses import field
from typing import List, Optional, Tuple

import numpy as np

from pydantic.dataclasses import dataclass
from pydantic import BaseModel, Extra, PositiveInt, NonNegativeFloat, NonNegativeInt, validator


# All settings classes inherit from NoExtrasModel,
# which forbids extra parameters to guard against typos
class NoExtrasModel(BaseModel, extra=Extra.forbid):
    pass


class ConfigSettings(NoExtrasModel):
    config_group: str
    config_name: str


class DevicePropertySettings(NoExtrasModel):
    device_name: str
    property_name: str
    property_value: str


class TimeSettings(NoExtrasModel):
    num_timepoints: PositiveInt = 1
    time_interval_s: NonNegativeFloat = 0  # in seconds, must allow zero for MDA


class PositionSettings(NoExtrasModel):
    xyz_positions: list = []
    position_labels: List[str] = []
    num_positions: NonNegativeInt = 0

    @validator("num_positions")
    def check_n_pos(cls, v, values):
        p = len(values.get("xyz_positions"))
        if v != p:
            raise ValueError(f"num_positions = {v} must be equal to len(xyz_positions) = {p}")
        return v


class ChannelSettings(NoExtrasModel):
    exposure_time_ms: List[float] = []  # in ms
    channel_group: Optional[str] = None
    channels: List[str] = []
    use_sequencing: bool = False
    num_channels: int = 0  #
    acquisition_rate: Optional[float] = None

    @validator("channels")
    def check_n_ch(cls, v, values):
        exps = len(values.get("exposure_time_ms"))
        ch = len(v)
        if v != ch:
            raise ValueError(
                f"Number of channels = {ch} must equal number of exposure times {exps}"
            )


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
            self.z_range = list(np.arange(self.z_start, self.z_end + self.z_step, self.z_step))
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
    use_o3_refocus: bool = False
    o3_refocus_config: Optional[ConfigSettings] = None
    o3_refocus_interval_min: Optional[int] = None


class AcquisitionSettings(NoExtrasModel):
    global_settings: Optional[List[str]] = None
    time_settings: TimeSettings
    lf_channel_settings: ChannelSettings
    lf_slice_settings: SliceSettings
    lf_microscope_settings: MicroscopeSettings
    ls_channel_settings: ChannelSettings
    ls_slice_settings: SliceSettings
    ls_microscope_settings: MicroscopeSettings
