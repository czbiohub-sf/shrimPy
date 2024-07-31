import numpy as np
import pytest

from pydantic import ValidationError

from mantis.analysis.AnalysisSettings import (
    DeskewSettings,
    RegistrationSettings,
    StabilizationSettings,
)


def test_deskew_settings():
    # Test extra parameter
    with pytest.raises(ValidationError):
        DeskewSettings(
            pixel_size_um=0.116, ls_angle_deg=36, scan_step_um=0.313, typo_param="test"
        )

    # Test negative value
    with pytest.raises(ValidationError):
        DeskewSettings(pixel_size_um=-3, ls_angle_deg=36, scan_step_um=0.313)

    # Test light sheet angle range
    with pytest.raises(ValueError):
        DeskewSettings(pixel_size_um=0.116, ls_angle_deg=90, scan_step_um=0.313)

    # Test px_to_scan_ratio logic
    with pytest.raises(ValueError):
        DeskewSettings(pixel_size_um=0.116, ls_angle_deg=36, scan_step_um=None)


def test_example_deskew_settings(example_deskew_settings):
    _, settings = example_deskew_settings

    DeskewSettings(**settings)


def test_register_settings():
    # Test extra parameter
    with pytest.raises(ValidationError):
        RegistrationSettings(
            source_channel_index=0,
            target_channel_index=0,
            affine_transform_zyx=np.identity(4).tolist(),
            typo_param="test",
        )

    # Test wrong output shape size
    with pytest.raises(ValidationError):
        RegistrationSettings(
            source_channel_index=0,
            target_channel_index=0,
            affine_transform_zyx=np.identity(4).tolist(),
            typo_param="test",
        )

    # Test wrong matrix shape
    with pytest.raises(ValidationError):
        RegistrationSettings(
            source_channel_index=0,
            target_channel_index=0,
            affine_transform_zyx=np.identity(5).tolist(),
            typo_param="test",
        )


def test_example_register_settings(example_register_settings):
    _, settings = example_register_settings
    RegistrationSettings(**settings)


def test_example_stabilize_timelapse_settings(example_stabilize_timelapse_settings):
    _, settings = example_stabilize_timelapse_settings
    StabilizationSettings(**settings)
