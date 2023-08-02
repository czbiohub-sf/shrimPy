import pytest

from mantis.analysis.AnalysisSettings import DeskewSettings


def test_deskew_settings():
    # Test extra parameter
    with pytest.raises(TypeError):
        DeskewSettings(typo_param="test")

    # Test negative value
    with pytest.raises(TypeError):
        DeskewSettings(pixel_size_um=-3)
