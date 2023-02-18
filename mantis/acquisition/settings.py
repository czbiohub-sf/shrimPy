from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class TimeSettings:
    num_timepoints: Optional[int] = 1
    time_internal_s: Optional[float] = 0  # in seconds


@dataclass()
class PositionSettings:
    xyz_positions: Optional[list] = None
    position_labels: Optional[List[str]] = None
    num_positions: int = field(init=False, default=1)

    def __post_init__(self):
        if self.xyz_positions is not None:
            self.num_positions = len(self.xyz_positions)


@dataclass()
class ChannelSettings:
    roi: Optional[Tuple[int, int, int, int]] = None
    exposure_time_ms: List[float] = 10  # in ms
    channel_group: Optional[str] = None
    channels: Optional[List[str]] = None
    use_sequence: bool = False
    num_channels: int = field(init=False, default=1)
    channel_acq_rate: float = field(init=False, default=None)

    def __post_init__(self):
        if self.channels is not None:
            self.num_channels = len(self.channels)
            assert len(self.exposure_time_ms) == len(self.channels), \
                'Number of channels must equal number of exposure times'

@dataclass
class SliceSettings:
    z_scan_stage: Optional[str] = None
    z_start: Optional[float] = None
    z_end: Optional[float] = None
    z_step: Optional[float] = None
    use_sequence: bool = False
    num_slices: int = field(init=False, default=1)
    slice_acq_rate: float = field(init=False, default=None)

    def __post_init__(self):
        if self.z_step is not None:
            self.num_slices = len(np.arange(self.z_start, self.z_end+self.z_step, self.z_step))


@dataclass()
class MicroscopeSettings:
    device_property_settings: Optional[List[Tuple[str, str, str]]] = None
    config_group_settings: Optional[List[Tuple[str, str]]] = None
    use_autofocus: bool = False
    autofocus_stage: Optional[str] = None
    autofocus_method: Optional[str] = None
