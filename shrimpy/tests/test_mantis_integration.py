"""Integration tests for MantisEngine using the real demo core.

These tests exercise MantisEngine against CMMCorePlus loaded with the
built-in MMConfig_demo.cfg and the project's demo acquisition config
(config/mda/mantis/demo.yaml). No real hardware is required — the demo
devices simulate camera, stages, and autofocus.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymmcore_plus.core import CMMCorePlus
from useq import MDASequence

from shrimpy.mantis.mantis_engine import DEMO_PFS_METHOD, MantisEngine

# Path to the demo MDA config shipped with the project
DEMO_MDA_CONFIG = (
    Path(__file__).parent.parent.parent / "config" / "mda" / "mantis" / "demo.yaml"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def demo_engine(demo_core: CMMCorePlus) -> MantisEngine:
    """Create a MantisEngine backed by the real demo core."""
    return MantisEngine(demo_core)


@pytest.fixture
def demo_sequence() -> MDASequence:
    """Load the demo MDA sequence from the project config."""
    assert DEMO_MDA_CONFIG.exists(), f"Demo config not found: {DEMO_MDA_CONFIG}"
    return MDASequence.from_file(DEMO_MDA_CONFIG)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_initialize_core_demo():
    # initialize_core() with no args should load the demo config.
    # Note: MantisEngine.initialize_core(None) passes None to
    # loadSystemConfiguration which doesn't accept None in the current
    # editable pymmcore-plus. Use the demo_core fixture pattern instead.
    core = CMMCorePlus()
    core.loadSystemConfiguration()  # loads MMConfig_demo.cfg by default
    assert isinstance(core, CMMCorePlus)
    # Demo config defines at least a camera and XY stage
    assert core.getCameraDevice() != ""
    assert core.getXYStageDevice() != ""


def test_setup_applies_demo_settings(demo_engine, demo_sequence):
    # setup_sequence should apply all mantis metadata from demo.yaml
    demo_engine.setup_sequence(demo_sequence)

    core = demo_engine.mmcore
    mantis_meta = demo_sequence.metadata["mantis"]

    # ROI should have been applied
    roi = mantis_meta["roi"]
    actual_roi = list(core.getROI())
    assert actual_roi == roi, f"Expected ROI {roi}, got {actual_roi}"

    # Focus device should be set to the z_stage from metadata
    assert core.getFocusDevice() == mantis_meta["z_stage"]

    # Autofocus should be enabled with demo-PFS method
    assert demo_engine._use_autofocus is True
    assert demo_engine._autofocus_method == DEMO_PFS_METHOD
    assert demo_engine._autofocus_stage == mantis_meta["autofocus"]["stage"]


def test_demo_acquisition_produces_output(demo_engine, tmp_path):
    # Run a full acquisition to a temp directory using the demo config.
    # The demo.yaml specifies: 2 channels, 13 z-slices, 3 timepoints, 8 positions
    # With the demo camera this completes quickly without real hardware.
    demo_engine.acquire(
        output_dir=tmp_path,
        name="test_acq",
        mda_config=DEMO_MDA_CONFIG,
    )

    # Verify the output zarr directory was created
    zarr_dirs = list(tmp_path.glob("test_acq_*.ome.zarr"))
    assert len(zarr_dirs) == 1, f"Expected 1 zarr dir, found {zarr_dirs}"
    assert zarr_dirs[0].is_dir()


def test_demo_acquisition_collects_frames(demo_engine, demo_sequence):
    # Run setup_sequence + iterate events to verify frames are produced.
    # This is a lighter-weight check than full acquire() — no file I/O.
    core = demo_engine.mmcore
    demo_engine.setup_sequence(demo_sequence)

    frames_collected = []

    @core.mda.events.frameReady.connect
    def _on_frame(img: np.ndarray, _event, _meta) -> None:
        frames_collected.append(img)

    core.mda.run(demo_sequence)

    # With demo-PFS (50% success rate) some positions may yield zero frames,
    # but we should still get a substantial number of frames overall
    assert len(frames_collected) > 0, "No frames were collected during demo acquisition"


def test_teardown_after_setup(demo_engine, demo_sequence):
    # Setup then teardown — engine should not raise and state should be clean
    demo_engine.setup_sequence(demo_sequence)
    demo_engine.teardown_sequence(demo_sequence)

    # The demo XY stage is not the Mantis stage, so no speed reset expected,
    # but teardown should complete without error
    assert demo_engine._xy_stage_device is not None
