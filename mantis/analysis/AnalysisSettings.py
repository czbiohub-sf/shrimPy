from typing import Literal, Optional, Union

import numpy as np

from pydantic import BaseModel, Extra, NonNegativeInt, PositiveFloat, PositiveInt, validator


# All settings classes inherit from MyBaseModel, which forbids extra parameters to guard against typos
class MyBaseModel(BaseModel, extra=Extra.forbid):
    pass


class DeskewSettings(MyBaseModel):
    pixel_size_um: PositiveFloat
    ls_angle_deg: PositiveFloat
    px_to_scan_ratio: Optional[PositiveFloat] = None
    scan_step_um: Optional[PositiveFloat] = None
    keep_overhang: bool = False
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

    def __init__(self, **data):
        if data.get("px_to_scan_ratio") is None:
            if data.get("scan_step_um") is not None:
                data["px_to_scan_ratio"] = round(
                    data["pixel_size_um"] / data["scan_step_um"], 3
                )
            else:
                raise ValueError(
                    "If px_to_scan_ratio is not provided, both pixel_size_um and scan_step_um must be provided"
                )
        super().__init__(**data)


class RegistrationSettings(MyBaseModel):
    source_channel_names: list[str]
    target_channel_name: str
    affine_transform_zyx: list
    keep_overhang: bool = False
    time_indices: Union[NonNegativeInt, list[NonNegativeInt], Literal["all"]] = "all"

    @validator("affine_transform_zyx")
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


class ConcatenateSettings(MyBaseModel):
    concat_data_paths: list[str]
    time_indices: Union[NonNegativeInt, list[NonNegativeInt], Literal["all"]] = "all"
    channel_names: list[Union[str, list[str]]]
    X_slice: list[NonNegativeInt]
    Y_slice: list[NonNegativeInt]
    Z_slice: list[NonNegativeInt]

    @validator("concat_data_paths")
    def check_concat_data_paths(cls, v):
        if not isinstance(v, list) or not all(isinstance(path, str) for path in v):
            raise ValueError("concat_data_paths must be a list of positions.")
        return v

    @validator("channel_names")
    def check_channel_names(cls, v):
        if not isinstance(v, list) or not all(isinstance(name, (str, list)) for name in v):
            raise ValueError("channel_names must be a list of strings or lists of strings.")
        return v

    @validator("X_slice", "Y_slice", "Z_slice")
    def check_slices(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 2
            or not all(isinstance(i, int) and i >= 0 for i in v)
        ):
            raise ValueError("Slices must be lists of two non-negative integers.")
        return v


class StabilizationSettings(MyBaseModel):
    stabilization_estimation_channel: str
    stabilization_type: Literal["z", "xy", "xyz"]
    stabilization_channels: list
    affine_transform_zyx_list: list
    time_indices: Union[NonNegativeInt, list[NonNegativeInt], Literal["all"]] = "all"

    @validator("affine_transform_zyx_list")
    def check_affine_transform_list(cls, v):
        if not isinstance(v, list):
            raise ValueError("affine_transform_list must be a list")

        for arr in v:
            arr = np.array(arr)
            if arr.shape != (4, 4):
                raise ValueError("Each element in affine_transform_list must be a 4x4 ndarray")

        return v
