from pydantic import validator
from pydantic.dataclasses import dataclass

@dataclass
class DeskewSettings:
    pixel_size_um: float
    scan_step_um: float
    ls_angle_deg: float
    px_to_scan_ratio: float = None

    @validator("ls_angle_deg")
    def ls_angle_check(cls, v):
        if v < 0 or v > 45:
            raise ValueError("Light sheet angle must be be between 0 and 45 degrees")
        return round(float(v), 2)
    
    def __post_init__(self):
        if self.px_to_scan_ratio is None:
            self.px_to_scan_ratio = round(self.pixel_size_um / self.scan_step_um, 3)
