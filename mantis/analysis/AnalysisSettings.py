from pydantic import validator
from pydantic.dataclasses import dataclass
import numpy as np


@dataclass
class DeskewSettings:
    pixel_size_um: float
    ls_angle_deg: float
    px_to_scan_ratio: float = None
    scan_step_um: float = None
    keep_overhang: bool = True

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


@dataclass
class RegistrationSettings:
    affine_transform_zyx: list
    voxel_size: list
    output_shape: list
    pts_phase_registration: list = None
    pts_fluor_registration: list = None
    phase_channel_path: str = None
    fluor_channel_path: str = None
    fluor_channel_90deg_CCW_rotation: bool = True

    def validate_nx3_array(v):
        if not isinstance(v, list) or len(v) < 3:
            raise ValueError("The input array must be a list with at least 3 rows.")
        for row in v:
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError("Each row of the array must be a list of length 3.")

        return v

    _validate_pts_phase_registration = validator('pts_phase_registration', allow_reuse=True)(
        validate_nx3_array
    )
    _validate_pts_fluor_registration = validator('pts_fluor_registration', allow_reuse=True)(
        validate_nx3_array
    )

    @validator('affine_transform_zyx')
    def check_affine_transform(cls, v):
        if not isinstance(v, list) or len(v) != 4:
            raise ValueError("The input array must be a list of length 3.")

        for row in v:
            if not isinstance(row, list) or len(row) != 4:
                raise ValueError("Each row of the array must be a list of length 3.")

        try:
            # Try converting the list to a 3x3 ndarray to check for valid shape and content
            np_array = np.array(v)
            if np_array.shape != (4, 4):
                raise ValueError("The array must be a 3x3 ndarray.")
        except ValueError:
            raise ValueError("The array must contain valid numerical values.")

        return v

    @validator('pts_phase_registration')
    def validate_same_entries(cls, v, values, **kwargs):
        array_data2 = values.get('pts_fluor_registration')
        if array_data2 and len(v) != len(array_data2):
            raise ValueError(
                "'pts_phase_registration' and 'pts_fluor_registration' must have the same number of entries."
            )
        return v
