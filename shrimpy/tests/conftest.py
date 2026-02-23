"""Shared test fixtures for shrimPy tests.

Provides mock CMMCorePlus instances and helper fixtures used across
unit and integration test modules.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pymmcore_plus.core import CMMCorePlus


@pytest.fixture
def mock_core() -> MagicMock:
    """Create a MagicMock CMMCorePlus with sensible defaults for unit tests.

    This fixture returns a mock that mimics a Mantis-configured CMMCorePlus
    without requiring any real hardware or demo devices. Tests can override
    individual return values as needed.
    """
    core = MagicMock(spec=CMMCorePlus)

    # Device names matching mantis_engine constants
    core.getXYStageDevice.return_value = "XYStage:XY:31"
    core.getFocusDevice.return_value = "ZDrive"

    # Image properties (full sensor before ROI is applied)
    core.getImageWidth.return_value = 2048
    core.getImageHeight.return_value = 2048
    core.getImageBitDepth.return_value = 16
    core.getPixelSizeUm.return_value = 0.115

    # Autofocus defaults — not locked, tests override as needed
    core.isContinuousFocusLocked.return_value = False

    # Stage position tracking (used by _adjust_xy_stage_speed)
    core._last_xy_position = {None: (0.0, 0.0)}

    # MDA sub-object with mock signals for event callbacks
    core.mda = MagicMock()
    core.mda.events = MagicMock()
    core.events = MagicMock()

    return core


@pytest.fixture
def demo_core() -> CMMCorePlus:
    """Create a real CMMCorePlus loaded with the built-in MMConfig_demo.cfg.

    Used by integration tests that exercise the full engine against demo
    devices (DemoCamera, DStage, DXYStage, etc.).
    """
    core = CMMCorePlus()
    core.loadSystemConfiguration()  # loads MMConfig_demo.cfg
    return core
