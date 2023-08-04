import pytest

from mantis.analysis.AnalysisSettings import DeskewSettings


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
