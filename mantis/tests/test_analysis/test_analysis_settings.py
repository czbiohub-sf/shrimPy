import numpy as np
import pytest

from mantis.analysis.AnalysisSettings import DeskewSettings, RegistrationSettings


def test_deskew_settings():
    # Test extra parameter
    with pytest.raises(TypeError):
        DeskewSettings(typo_param="test")

    # Test negative value
    with pytest.raises(TypeError):
        DeskewSettings(pixel_size_um=-3)

    # Test px_to_scan_ratio logic
    with pytest.raises(TypeError):
        DeskewSettings(pixel_size_um=0.116, ls_angle_deg=36, scan_step_um=None)


def test_example_deskew_settings(example_deskew_settings):
    _, settings = example_deskew_settings

    DeskewSettings(**settings)


def test_apply_affine_settings():
    # Test extra parameter
    with pytest.raises(TypeError):
        RegistrationSettings(typo_param="test")

    # Test wrong output shape size
    with pytest.raises(ValueError):
        RegistrationSettings(output_shape_zyx=[1, 2, 3, 4])

    # Test wrong matrix shape
    with pytest.raises(ValueError):
        random_array = np.random.rand(5, 5)
        RegistrationSettings(affine_transform_zyx=random_array.tolist())


def test_example_apply_affine_settings(example_apply_affine_settings):
    _, settings = example_apply_affine_settings
    RegistrationSettings(**settings)
