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
from useq import MDASequence, Position

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
    # setup_sequence should apply mantis-specific settings from demo.yaml.
    # ROI and device properties are now applied via a setup event in MDASequence,
    # not by MantisEngine.setup_sequence.
    demo_engine.setup_sequence(demo_mda_sequence)

    # Autofocus should be enabled with demo-PFS method
    assert demo_engine._use_autofocus is True
    assert demo_engine._autofocus_method == DEMO_PFS_METHOD
    assert demo_engine._autofocus_stage == mantis_metadata["autofocus"]["stage"]


def test_demo_acquisition_collects_frames(demo_engine, mantis_metadata):
    # Run setup_sequence + iterate events to verify frames are produced.
    # This is a lighter-weight check than full acquire() — no file I/O.
    core = demo_engine.mmcore
    mantis_metadata["autofocus"]["enabled"] = False
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
    dataset = open_ome_zarr(zarr_dirs[0])
    # Verify zarr v3 OME-NGFF v0.5 format
    assert dataset.zgroup.metadata.zarr_format == 3
    assert dataset.version == "0.5"

    assert isinstance(dataset, Plate)
    positions = list(dataset.positions())
    num_positions = len(positions)
    _pos_name, _pos = positions[0]
    _data = _pos.data

    assert num_positions == 8, f"Expected 8 positions, found {num_positions}"
    # Confirm expected position name
    assert _pos_name == "A/1/fov0"
    # ROI is now applied via the setup event in MDASequence
    setup_roi = demo_mda_sequence.setup.roi
    assert _data.shape == (
        demo_mda_sequence.sizes["t"],
        demo_mda_sequence.sizes["c"],
        demo_mda_sequence.sizes["z"],
        setup_roi.height,
        setup_roi.width,
    )
    assert all(
        channel.config in dataset.channel_names for channel in demo_mda_sequence.channels
    )

    # --- Verify zstd compression via blosc codec in sharding pipeline ---
    from zarr.codecs import BloscCodec, ShardingCodec

    codecs = _data.metadata.codecs
    sharding = next((c for c in codecs if isinstance(c, ShardingCodec)), None)
    assert sharding is not None, f"Expected ShardingCodec, got {codecs}"
    blosc = next((c for c in sharding.codecs if isinstance(c, BloscCodec)), None)
    assert blosc is not None, f"Expected BloscCodec in sharding, got {sharding.codecs}"
    assert blosc.cname.value == "zstd", f"Expected zstd compression, got {blosc.cname}"

    # --- Verify z chunk_shape != 1 ---
    # dimension order: (t, c, z, y, x)
    z_chunk = _data.chunks[-3]
    assert z_chunk != 1, f"Expected z chunk_shape != 1, got chunks={_data.chunks}"


def test_summary_metadata_written_to_zarr(demo_engine, demo_mda_sequence, tmp_path):
    """Verify that summary_metadata.json is written at the zarr root."""
    import json

    demo_engine.acquire(
        output_dir=tmp_path,
        name="meta_test",
        mda_config=DEMO_MDA_CONFIG,
    )

    zarr_dirs = list(tmp_path.glob("meta_test_*.ome.zarr"))
    assert len(zarr_dirs) == 1

    meta_path = zarr_dirs[0] / "summary_metadata.json"
    assert meta_path.exists(), "summary_metadata.json not found at zarr root"

    summary = json.loads(meta_path.read_text())
    assert summary["format"] == "summary-dict"
    assert summary["version"] == "1.0"
    assert "devices" in summary
    assert "system_info" in summary
    assert "image_infos" in summary


def test_teardown_after_setup(demo_engine, demo_mda_sequence):
    core = demo_engine.mmcore
    # Setup then teardown — engine should not raise and state should be clean.
    # setup_hardware_sequencing_settings are now applied via a setup event,
    # not by MantisEngine.setup_sequence, so we only check teardown resets.
    demo_engine.setup_sequence(demo_mda_sequence)
    demo_engine.teardown_sequence(demo_mda_sequence)
    assert core.getProperty("Z", "UseSequences") == "No"


def test_timelapse_acquisition(demo_engine, mantis_metadata):
    # Acquisition with a timelapse plan
    mantis_metadata["autofocus"]["enabled"] = False
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


def test_autofocus_failure_pads_with_zeros(demo_engine, mantis_metadata, tmp_path):
    """Verify that when autofocus fails, the written zarr contains zero-padded data.

    Runs a 3-position, 5-timepoint acquisition with sequenced z-slices via
    engine.acquire(). Uses fail_at_index to deterministically fail autofocus
    at specific (t, p) combinations, then reads back the zarr with iohub to
    verify that failed positions contain all-zero data and successful ones
    contain non-zero camera data.
    """
    fail_at_index = [
        {"t": 1, "p": 0},
        {"t": 1, "p": 1},
        {"t": 2, "p": 0},
        {"t": 4, "p": 0},
        {"t": 4, "p": 1},
    ]

    position_names = ["0", "1", "2"]
    position_coords = [(0, 0), (100, 0), (0, 100)]
    seq = MDASequence(
        stage_positions=[
            Position(x=x, y=y, name=name)
            for name, (x, y) in zip(position_names, position_coords, strict=True)
        ],
        time_plan={"interval": 0, "loops": 5},
        channels=[{"config": "DAPI", "exposure": 1.0}],
        z_plan={"top": 15, "bottom": -15, "step": 15},  # 3 z-slices
        metadata={"mantis": mantis_metadata},
    )

    # Set deterministic autofocus failure directly on the engine
    demo_engine._autofocus_fail_at_index = fail_at_index

    # Run full acquisition pipeline (setup, acquire, write to zarr)
    demo_engine.acquire(output_dir=tmp_path, name="af_test", mda_config=seq)

    # Read back the written zarr store
    zarr_dirs = list(tmp_path.glob("af_test_*.ome.zarr"))
    assert len(zarr_dirs) == 1, f"Expected 1 zarr dir, found {zarr_dirs}"

    # Build a set of (t, p) tuples that should fail for failed positions
    fail_set = {(idx["t"], idx["p"]) for idx in fail_at_index}

    c_idx = 0
    for pos_idx, pos_key in enumerate(position_names):
        dataset = open_ome_zarr(zarr_dirs[0] / pos_key, layout="fov")
        data = dataset.data.numpy()
        for t_idx in range(seq.sizes["t"]):
            volume = data[t_idx, c_idx]  # shape: (Z, Y, X)
            if (t_idx, pos_idx) in fail_set:
                assert np.all(volume == 0), (
                    f"Expected zeros at t={t_idx}, p={pos_idx}, got non-zero data"
                )
            else:
                assert np.any(volume != 0), (
                    f"Expected non-zero data at t={t_idx}, p={pos_idx}, got all zeros"
                )
