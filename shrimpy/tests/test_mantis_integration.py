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

from iohub import open_ome_zarr
from iohub.ngff import Plate
from pymmcore_plus.core import CMMCorePlus
from useq import MDASequence

from shrimpy.mantis.mantis_engine import DEMO_PFS_METHOD, MantisEngine

# Local copy of the demo MDA config, kept in tests/artifacts so test inputs
# are independent of the project's runtime configuration files.
DEMO_MDA_CONFIG = Path(__file__).parent / "artifacts" / "demo_mda_sequence.yaml"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def demo_engine() -> MantisEngine:
    """Create a MantisEngine backed by the real demo core."""
    core = CMMCorePlus()
    core.loadSystemConfiguration()
    return MantisEngine(core)


@pytest.fixture
def demo_mda_sequence() -> MDASequence:
    """Load the full demo MDA sequence from the project config."""
    assert DEMO_MDA_CONFIG.exists(), f"Demo config not found: {DEMO_MDA_CONFIG}"
    return MDASequence.from_file(DEMO_MDA_CONFIG)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_setup_applies_demo_settings(demo_engine, demo_mda_sequence, mantis_metadata):
    # setup_sequence should apply all mantis metadata from demo.yaml,
    # tested in test_setup_sequence_*
    demo_engine.setup_sequence(demo_mda_sequence)

    core = demo_engine.mmcore

    # ROI should have been applied
    roi = mantis_metadata["roi"]
    actual_roi = list(core.getROI())
    assert actual_roi == roi, f"Expected ROI {roi}, got {actual_roi}"

    # Focus device should be set to the z_stage from metadata
    assert core.getFocusDevice() == mantis_metadata["z_stage"]

    # Autofocus should be enabled with demo-PFS method
    assert demo_engine._use_autofocus is True
    assert demo_engine._autofocus_method == DEMO_PFS_METHOD
    assert demo_engine._autofocus_stage == mantis_metadata["autofocus"]["stage"]


def test_demo_acquisition_collects_frames(demo_engine, mantis_metadata):
    # Run setup_sequence + iterate events to verify frames are produced.
    # This is a lighter-weight check than full acquire() — no file I/O.
    core = demo_engine.mmcore
    sequence = MDASequence(
        time_plan={"interval": 0, "loops": 10},
        metadata={"mantis": mantis_metadata},
    )
    demo_engine.setup_sequence(sequence)

    frames_collected = []

    @core.mda.events.frameReady.connect
    def _on_frame(img: np.ndarray, _event, _meta) -> None:
        frames_collected.append(img)

    core.mda.run(sequence)

    # With demo-PFS (50% success rate) some positions may yield zero frames,
    # but we should still get a substantial number of frames overall
    assert len(frames_collected) > 0, "No frames were collected during demo acquisition"


def test_demo_mda_acquisition(demo_engine, demo_mda_sequence, tmp_path):
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

    # Load dataset with iohub and inspect metadata
    dataset = open_ome_zarr(zarr_dirs[0], version="0.5")
    assert isinstance(dataset, Plate)

    positions = list(dataset.positions())
    num_positions = len(positions)
    _pos_name, _pos = positions[0]
    _data = _pos.data

    assert num_positions == 8, f"Expected 8 positions, found {num_positions}"
    # Confirm expected position name
    assert _pos_name == "A/1/fov0"
    assert _data.shape == (
        demo_mda_sequence.sizes["t"],
        demo_mda_sequence.sizes["c"],
        demo_mda_sequence.sizes["z"],
        demo_mda_sequence.metadata["mantis"]["roi"][3],
        demo_mda_sequence.metadata["mantis"]["roi"][2],
    )
    assert all(
        channel.config in dataset.channel_names for channel in demo_mda_sequence.channels
    )


def test_teardown_after_setup(demo_engine, demo_mda_sequence):
    core = demo_engine.mmcore
    # Setup then teardown — engine should not raise and state should be clean
    demo_engine.setup_sequence(demo_mda_sequence)
    assert core.getProperty("Z", "UseSequences") == "Yes"
    demo_engine.teardown_sequence(demo_mda_sequence)
    assert core.getProperty("Z", "UseSequences") == "No"


def test_timelapse_acquisition(demo_engine, mantis_metadata):
    # Acquisition with a timelapse plan
    seq = MDASequence(
        time_plan={"interval": 0, "loops": 10},
        metadata={"mantis": mantis_metadata},
    )

    core = demo_engine.mmcore
    demo_engine.setup_sequence(seq)

    frames_collected = []

    @core.mda.events.frameReady.connect
    def _on_frame(img: np.ndarray, _event, _meta) -> None:
        frames_collected.append(img)

    core.mda.run(seq)

    assert len(frames_collected) == 10, "No frames collected for timelapse acquisition"


def test_autofocus_disabled_acquisition(demo_engine, mantis_metadata):
    # Acquisition with autofocus disabled — all frames should succeed
    mantis_metadata["autofocus"]["enabled"] = False
    seq = MDASequence(
        channels=[{"config": "DAPI", "group": "Channel", "exposure": 5.0}],
        z_plan={"top": 15, "bottom": -15, "step": 15},
        metadata={"mantis": mantis_metadata},
    )

    core = demo_engine.mmcore
    demo_engine.setup_sequence(seq)

    frames_collected = []

    @core.mda.events.frameReady.connect
    def _on_frame(img: np.ndarray, _event, _meta) -> None:
        frames_collected.append(img)

    core.mda.run(seq)

    # With autofocus disabled, every event should produce a frame
    expected = seq.sizes.get("c", 1) * seq.sizes.get("z", 1)
    assert len(frames_collected) == expected, (
        f"Expected {expected} frames, got {len(frames_collected)}"
    )
