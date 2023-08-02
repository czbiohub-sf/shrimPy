from typing import Optional

from pydantic import ConfigDict, PositiveFloat, PositiveInt, validator
from pydantic.dataclasses import dataclass

config = ConfigDict(extra='forbid')


@dataclass(config=config)
class DeskewSettings:
    pixel_size_um: PositiveFloat
    ls_angle_deg: PositiveFloat
    px_to_scan_ratio: Optional[PositiveFloat] = None
    scan_step_um: Optional[PositiveFloat] = None
    keep_overhang: bool = True
    average_n_slices: PositiveInt = 3

    @validator("ls_angle_deg")
    def ls_angle_check(cls, v):
        if v < 0 or v > 45:
            raise ValueError("Light sheet angle must be be between 0 and 45 degrees")
        return round(float(v), 2)

    @validator("px_to_scan_ratio")
    def px_to_scan_ratio_check(cls, v):
        if v is not None:
            return round(float(v), 3)

    def __post_init__(self):
        if self.px_to_scan_ratio is None:
            if self.scan_step_um is not None:
                self.px_to_scan_ratio = round(self.pixel_size_um / self.scan_step_um, 3)
            else:
                raise Exception("px_to_scan_ratio is not valid")
