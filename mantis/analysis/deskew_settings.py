from pydantic import BaseModel, validator


class DeskewSettings(BaseModel):
    px_to_scan_ratio: float
    theta_deg: float

    @validator("px_to_scan_ratio")
    def px_to_scan_ratio_check(cls, v):
        return round(float(v), 3)

    @validator("theta_deg")
    def theta_check(cls, v):
        if v < 0 or v > 90:
            raise ValueError("theta_deg should be between 0 and 90")
        return round(float(v), 2)
