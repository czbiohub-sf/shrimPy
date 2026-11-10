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
