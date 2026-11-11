"""Unit tests for the iSIM-specific parts of ISIMEngine.

Behavior shared with the other microscope engines is tested in
``test_base_engine.py``.
"""

from __future__ import annotations

import weakref

from unittest.mock import MagicMock, patch

import pytest

from shrimpy.engines.isim_engine import ORCA_QUEST_CAMERA, ORCA_QUEST_ROI, ISIMEngine


def _make_engine(core: MagicMock) -> ISIMEngine:
    """Build an ISIMEngine on a mock core, bypassing MDAEngine.__init__."""

    def fake_init(self, mmc, *args, **kwargs):
        self._mmcore_ref = weakref.ref(mmc)

    with patch("shrimpy.engines.base_engine.MDAEngine.__init__", fake_init):
        return ISIMEngine(core)


def test_orca_quest_selected_and_cropped(mock_core: MagicMock):
    mock_core.getLoadedDevices.return_value = ("Core", ORCA_QUEST_CAMERA, "ZStage")
    _make_engine(mock_core)
    mock_core.setCameraDevice.assert_called_once_with(ORCA_QUEST_CAMERA)
    mock_core.setROI.assert_called_once_with(*ORCA_QUEST_ROI)


def test_camera_untouched_without_orca_quest(mock_core: MagicMock):
    mock_core.getLoadedDevices.return_value = ("Core", "Camera", "Z")
    _make_engine(mock_core)
    mock_core.setCameraDevice.assert_not_called()
    mock_core.setROI.assert_not_called()


def test_demo_config():
    """With the MM demo config (no ORCA-Quest) the engine leaves the camera alone."""
    from pymmcore_plus import CMMCorePlus

    core = CMMCorePlus()
    try:
        core.loadSystemConfiguration()
    except Exception as e:  # pragma: no cover - no MM install
        pytest.skip(f"Micro-Manager demo config unavailable: {e}")
    roi_before = core.getROI()
    ISIMEngine(core)
    assert core.getCameraDevice() == "Camera"
    assert core.getROI() == roi_before


@pytest.fixture
def multicam_core():
    """ISIMCore on the MM demo config, with a 2-camera Multi Camera selected."""
    from shrimpy.engines.isim_engine import ISIMCore

    core = ISIMCore()
    try:
        core.loadSystemConfiguration()
    except Exception as e:  # pragma: no cover - no MM install
        pytest.skip(f"Micro-Manager demo config unavailable: {e}")
    core.loadDevice("Camera2", "DemoCamera", "DCam")
    core.loadDevice("MultiCam", "Utilities", "Multi Camera")
    core.initializeDevice("MultiCam")
    core.initializeDevice("Camera2")
    core.setProperty("MultiCam", "Physical Camera 1", "Camera")
    core.setProperty("MultiCam", "Physical Camera 2", "Camera2")
    core.setCameraDevice("MultiCam")
    return core


@pytest.mark.parametrize("first_binning", [1, 2])
def test_multicam_mixed_binning_pixel_size(multicam_core, first_binning):
    """Mixed binning makes MMCore report 0; ISIMCore returns the config as defined."""
    from pymmcore_plus import CMMCorePlus

    core = multicam_core
    core.setProperty("Camera", "Binning", str(first_binning))
    core.setProperty("Camera2", "Binning", str(3 - first_binning))  # the other one

    # the Micro-Manager bug being worked around
    assert CMMCorePlus.getPixelSizeUm(core) == 0.0
    assert not any(CMMCorePlus.getPixelSizeAffine(core))

    config = core.getCurrentPixelSizeConfig()
    assert core.getPixelSizeUm() == core.getPixelSizeUmByID(config)
    assert core.getPixelSizeUm(True) == core.getPixelSizeUmByID(config)
    assert core.getPixelSizeAffine() == tuple(core.getPixelSizeAffineByID(config))


def test_multicam_pixel_size_config_keyed_on_camera(multicam_core):
    """As on the iSIM: a pixel-size config for the Multi Camera + objective."""
    core = multicam_core
    objective = core.getProperty("Objective", "Label")
    core.definePixelSizeConfig("60x_MultiCam", "Core", "Camera", "MultiCam")
    core.definePixelSizeConfig("60x_MultiCam", "Objective", "Label", objective)
    core.setPixelSizeUm("60x_MultiCam", 0.075)
    core.setPixelSizeAffine("60x_MultiCam", [0.075, 0.0, 0.0, 0.0, 0.075, 0.0])
    core.definePixelSizeConfig("60x_Single", "Core", "Camera", "Camera2")
    core.definePixelSizeConfig("60x_Single", "Objective", "Label", objective)
    core.setPixelSizeUm("60x_Single", 0.0575)
    core.setProperty("Camera", "Binning", "1")
    core.setProperty("Camera2", "Binning", "2")

    assert core.getCurrentPixelSizeConfig() == "60x_MultiCam"
    assert core.getPixelSizeUm() == 0.075
    assert core.getPixelSizeAffine() == (0.075, 0.0, 0.0, 0.0, 0.075, 0.0)

    # a single camera is unaffected: MMCore scales its config by its binning (2)
    core.setCameraDevice("Camera2")
    assert core.getCurrentPixelSizeConfig() == "60x_Single"
    assert core.getPixelSizeUm() == 0.0575 * 2


def test_multicam_same_binning_pixel_size_unchanged(multicam_core):
    """With matching binning MMCore is right, and ISIMCore changes nothing."""
    from pymmcore_plus import CMMCorePlus

    core = multicam_core
    core.setProperty("Camera", "Binning", "2")
    core.setProperty("Camera2", "Binning", "2")
    assert core.getPixelSizeUm() == CMMCorePlus.getPixelSizeUm(core) != 0.0
    assert core.getPixelSizeAffine() == tuple(CMMCorePlus.getPixelSizeAffine(core))
