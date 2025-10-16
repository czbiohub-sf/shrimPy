import warnings

from dataclasses import field
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from pydantic.v1 import ConfigDict, NonNegativeFloat, NonNegativeInt, PositiveInt, validator
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
                # Default well is called "0"
                self.well_ids = ["0"] * self.num_positions


@dataclass(config=config)
class ChannelSettings:
    default_exposure_times_ms: Union[NonNegativeFloat, List[NonNegativeFloat], None] = (
        None  # in ms
    )
    default_laser_powers: Union[NonNegativeFloat, List[NonNegativeFloat], List[None], None] = (
        None
    )
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


@dataclass(config=config)
class ZarrSettings:
    """Settings for Zarr output configuration using acquire-zarr."""

    # Data type for the Zarr array
    data_type: Literal[
        'UINT8', 'UINT16', 'UINT32', 'INT8', 'INT16', 'INT32', 'FLOAT32', 'FLOAT64'
    ] = 'UINT16'

    # Enable multiscale (pyramids)
    multiscale: bool = False

    # Maximum number of threads for writing
    max_threads: NonNegativeInt = 0  # 0 means use all available

    # Chunking settings as dictionary - unit: pixels per chunk
    chunk_sizes: Dict[str, PositiveInt] = field(
        default_factory=lambda: {
            'x': 64,  # pixels per chunk in X dimension
            'y': 64,  # pixels per chunk in Y dimension
            'z': 1,  # pixels per chunk in Z dimension
            'c': 1,  # pixels per chunk in channel dimension
            'p': 1,  # pixels per chunk in position dimension
            't': 1,  # pixels per chunk in time dimension
        }
    )

    # Sharding settings (Zarr V3 only) - unit is chunks per shard
    shard_sizes: Dict[str, PositiveInt] = field(
        default_factory=lambda: {'x': 1, 'y': 1, 'z': 1, 'c': 1, 'p': 1, 't': 1}
    )

    # Compression settings
    compression_codec: Optional[Literal['blosc', 'gzip', 'lz4', 'zstd']] = None
    compression_level: Optional[int] = None

    # Store settings
    store_path: Optional[str] = None
    overwrite_existing: bool = False

    # HCS (High Content Screening) settings
    use_hcs_layout: bool = False  # Enable HCS zarr structure with wells/plates
    plate_name: Optional[str] = None  # Name of the plate (e.g., "Plate_001")
    plate_description: Optional[str] = None  # Description of the plate

    @validator("chunk_sizes")
    def validate_chunk_sizes(cls, v):
        """Validate chunk_sizes dictionary contains required keys."""
        required_keys = {'x', 'y', 'z', 'c', 'p', 't'}
        if not isinstance(v, dict):
            raise ValueError("chunk_sizes must be a dictionary")

        missing_keys = required_keys - set(v.keys())
        if missing_keys:
            raise ValueError(f"chunk_sizes is missing required keys: {missing_keys}")

        # Ensure all values are positive integers
        for key, value in v.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"chunk_sizes['{key}'] must be a positive integer, got {value}"
                )

        return v

    @validator("shard_sizes")
    def validate_shard_sizes(cls, v):
        """Validate shard_sizes dictionary contains required keys."""
        required_keys = {'x', 'y', 'z', 'c', 'p', 't'}
        if not isinstance(v, dict):
            raise ValueError("shard_sizes must be a dictionary")

        missing_keys = required_keys - set(v.keys())
        if missing_keys:
            raise ValueError(f"shard_sizes is missing required keys: {missing_keys}")

        # Ensure all values are positive integers
        for key, value in v.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"shard_sizes['{key}'] must be a positive integer (chunks per shard), got {value}"
                )

        return v

    @validator("plate_name")
    def validate_plate_name(cls, v, values):
        """Validate plate_name is provided when using HCS layout."""
        use_hcs = values.get('use_hcs_layout', False)
        if use_hcs and v is None:
            raise ValueError("plate_name is required when use_hcs_layout is True")
        return v

    @validator("compression_level")
    def validate_compression_level(cls, v, values):
        """Validate compression level based on codec."""
        if v is not None:
            codec = values.get('compression_codec')
            if codec is None:
                raise ValueError("compression_level requires compression_codec to be set")

            # Validate compression level ranges for different codecs
            if codec == 'gzip' and not (1 <= v <= 9):
                raise ValueError("gzip compression_level must be between 1 and 9")
            elif codec == 'blosc' and not (1 <= v <= 9):
                raise ValueError("blosc compression_level must be between 1 and 9")
            elif codec == 'lz4' and not (1 <= v <= 9):
                raise ValueError("lz4 compression_level must be between 1 and 9")
            elif codec == 'zstd' and not (1 <= v <= 22):
                raise ValueError("zstd compression_level must be between 1 and 22")
        return v

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Ensure chunk sizes are reasonable
        if self.chunk_sizes['x'] > 2048:
            warnings.warn(
                f"Large X chunk size ({self.chunk_sizes['x']}) may impact performance"
            )

        if self.chunk_sizes['y'] > 2048:
            warnings.warn(
                f"Large Y chunk size ({self.chunk_sizes['y']}) may impact performance"
            )

        if self.chunk_sizes['z'] > 512:
            warnings.warn(
                f"Large Z chunk size ({self.chunk_sizes['z']}) may impact performance"
            )

    # Backward compatibility properties
    @property
    def xy_chunk_size(self) -> int:
        """Backward compatibility: returns x chunk size for legacy code that assumes x==y."""
        return self.chunk_sizes['x']

    @property
    def z_chunk_size(self) -> int:
        """Backward compatibility: returns z chunk size."""
        return self.chunk_sizes['z']

    @property
    def t_chunk_size(self) -> int:
        """Backward compatibility: returns t chunk size."""
        return self.chunk_sizes['t']

    @property
    def c_chunk_size(self) -> int:
        """Backward compatibility: returns c chunk size."""
        return self.chunk_sizes['c']

    @property
    def shard_size_chunks(self) -> int:
        """Backward compatibility: returns x shard size for legacy code."""
        return self.shard_sizes['x']

    def get_zarr_version_enum(self):
        """Get the appropriate ZarrVersion enum value for acquire-zarr."""
        from acquire_zarr import ZarrVersion

        return ZarrVersion.V3

    def get_data_type_enum(self):
        """Get the appropriate DataType enum value for acquire-zarr."""
        from acquire_zarr import DataType

        return getattr(DataType, self.data_type)
