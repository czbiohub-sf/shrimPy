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


@pytest.fixture
def mantis_metadata() -> dict:
    """Mantis-specific metadata for integration tests, to be used with MMConfig_demo.cfg.

    Returns a fresh dict each test can safely modify (e.g. disable autofocus)
    without affecting other tests. Hardcoded so that changes to demo.yaml
    don't break tests.

    Usage::

        seq = MDASequence(
            channels=[...],
            time_plan={...},
            metadata={"mantis": mantis_metadata},
        )
    """
    return {
        "roi": [225, 880, 1600, 256],
        "z_stage": "Z",
        "use_hardware_sequencing": True,
        "initialization_settings": [
            ["Camera", "OnCameraCCDXSize", "2048"],
            ["Camera", "OnCameraCCDYSize", "2048"],
            ["Camera", "PixelType", "16bit"],
            ["XY", "Velocity", "10000"],
        ],
        "setup_hardware_sequencing_settings": [
            ["Z", "UseSequences", "Yes"],
        ],
        "reset_hardware_sequencing_settings": [
            ["Z", "UseSequences", "No"],
        ],
        "autofocus": {
            "enabled": True,
            "method": "demo-PFS",
            "stage": "Z",
        },
    }
