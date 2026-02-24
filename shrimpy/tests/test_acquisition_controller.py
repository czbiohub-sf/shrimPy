"""Tests for AcquisitionController.

Tests the controller's event flow: autofocus engagement, event queuing
vs frame skipping, stream synchronization, and on_frame_ready hook.
Uses the real demo core with MantisEngine.
"""

from __future__ import annotations

import numpy as np
import pytest

from iohub import open_ome_zarr
from pymmcore_plus.core import CMMCorePlus
from useq import MDASequence, Position

from shrimpy.mantis.acquisition_controller import AcquisitionController
from shrimpy.mantis.mantis_engine import MantisEngine, _get_next_acquisition_name

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def demo_engine() -> MantisEngine:
    """Create a MantisEngine backed by the real demo core."""
    core = CMMCorePlus()
    core.loadSystemConfiguration()
    return MantisEngine(core)


def _run_controller(engine, seq, tmp_path, name="test"):
    """Helper: run AcquisitionController and return the zarr output path."""
    acq_name = _get_next_acquisition_name(tmp_path, name)
    data_path = tmp_path / f"{acq_name}.ome.zarr"
    with engine.create_stream(seq, data_path) as stream:
        controller = AcquisitionController(engine, stream, seq)
        controller.run()
    return data_path


# ---------------------------------------------------------------------------
# Basic acquisition via controller
# ---------------------------------------------------------------------------


def test_controller_runs_single_position(demo_engine, mantis_metadata, tmp_path):
    """Controller completes a single-position acquisition without autofocus."""
    mantis_metadata["autofocus"]["enabled"] = False
    seq = MDASequence(
        stage_positions=[Position(x=0, y=0, name="0")],
        channels=[{"config": "DAPI", "exposure": 1.0}],
        z_plan={"top": 15, "bottom": -15, "step": 15},  # 3 z-slices
        metadata={"mantis": mantis_metadata},
    )

    data_path = _run_controller(demo_engine, seq, tmp_path)

    # Single position: root is the FOV directly
    dataset = open_ome_zarr(data_path, layout="fov")
    data = dataset.data.numpy()
    assert data.shape[0] == 1  # c
    assert data.shape[1] == seq.sizes["z"]  # z
    assert np.any(data != 0)


def test_controller_runs_multi_position(demo_engine, mantis_metadata, tmp_path):
    """Controller handles multiple positions correctly."""
    mantis_metadata["autofocus"]["enabled"] = False
    seq = MDASequence(
        stage_positions=[
            Position(x=0, y=0, name="0"),
            Position(x=100, y=0, name="1"),
            Position(x=0, y=100, name="2"),
        ],
        channels=[{"config": "DAPI", "exposure": 1.0}],
        z_plan={"top": 15, "bottom": -15, "step": 15},
        metadata={"mantis": mantis_metadata},
    )

    data_path = _run_controller(demo_engine, seq, tmp_path)

    for p in range(3):
        dataset = open_ome_zarr(data_path / str(p), layout="fov")
        data = dataset.data.numpy()
        assert np.any(data != 0), f"Position {p} has no data"


def test_controller_runs_timelapse(demo_engine, mantis_metadata, tmp_path):
    """Controller handles timelapse acquisition correctly."""
    mantis_metadata["autofocus"]["enabled"] = False
    seq = MDASequence(
        stage_positions=[Position(x=0, y=0, name="0")],
        time_plan={"interval": 0, "loops": 3},
        channels=[{"config": "DAPI", "exposure": 1.0}],
        z_plan={"top": 15, "bottom": -15, "step": 15},
        metadata={"mantis": mantis_metadata},
    )

    data_path = _run_controller(demo_engine, seq, tmp_path)

    # Single position: root is the FOV directly
    dataset = open_ome_zarr(data_path, layout="fov")
    data = dataset.data.numpy()
    assert data.shape[0] == 3  # t
    for t in range(3):
        assert np.any(data[t] != 0), f"Timepoint {t} has no data"


# ---------------------------------------------------------------------------
# Autofocus skip behavior
# ---------------------------------------------------------------------------


def test_controller_skips_all_positions_on_failure(demo_engine, mantis_metadata, tmp_path):
    """When autofocus fails at every position, all frames are skipped (zeros)."""
    demo_engine._autofocus_fail_at_index = [{"p": 0}, {"p": 1}]

    seq = MDASequence(
        stage_positions=[Position(x=0, y=0, name="0"), Position(x=100, y=0, name="1")],
        channels=[{"config": "DAPI", "exposure": 1.0}],
        z_plan={"top": 15, "bottom": -15, "step": 15},
        metadata={"mantis": mantis_metadata},
    )

    data_path = _run_controller(demo_engine, seq, tmp_path)

    for p in range(2):
        dataset = open_ome_zarr(data_path / str(p), layout="fov")
        data = dataset.data.numpy()
        assert np.all(data == 0), f"Position {p} should be all zeros"


def test_controller_skips_correct_frame_count(demo_engine, mantis_metadata, tmp_path):
    """With 2 channels and 3 z-slices, a failed position skips all frames for
    that position while a successful position has real data."""
    demo_engine._autofocus_fail_at_index = [{"p": 0}]

    seq = MDASequence(
        stage_positions=[Position(x=0, y=0, name="0"), Position(x=100, y=0, name="1")],
        channels=[
            {"config": "DAPI", "exposure": 1.0},
            {"config": "FITC", "exposure": 1.0},
        ],
        z_plan={"top": 15, "bottom": -15, "step": 15},
        metadata={"mantis": mantis_metadata},
    )

    data_path = _run_controller(demo_engine, seq, tmp_path)

    # Position 0: failed → all zeros across both channels and all z-slices
    dataset_0 = open_ome_zarr(data_path / "0", layout="fov")
    data_0 = dataset_0.data.numpy()
    assert np.all(data_0 == 0), "Failed position should be all zeros"

    # Position 1: succeeded → has real data
    dataset_1 = open_ome_zarr(data_path / "1", layout="fov")
    data_1 = dataset_1.data.numpy()
    assert np.any(data_1 != 0), "Successful position should have data"


def test_autofocus_failure_pads_with_zeros(demo_engine, mantis_metadata, tmp_path):
    """Verify that when autofocus fails, the written zarr contains zero-padded data.

    Runs a 3-position, 5-timepoint acquisition with sequenced z-slices.
    Uses fail_at_index to deterministically fail autofocus at specific
    (t, p) combinations, then reads back the zarr to verify that failed
    positions contain all-zero data (from stream.skip()) and successful
    ones contain non-zero camera data.
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

    demo_engine._autofocus_fail_at_index = fail_at_index
    data_path = _run_controller(demo_engine, seq, tmp_path, name="af_test")

    fail_set = {(idx["t"], idx["p"]) for idx in fail_at_index}

    for pos_idx, pos_key in enumerate(position_names):
        dataset = open_ome_zarr(data_path / pos_key, layout="fov")
        data = dataset.data.numpy()
        for t_idx in range(seq.sizes["t"]):
            volume = data[t_idx, 0]  # shape: (Z, Y, X)
            if (t_idx, pos_idx) in fail_set:
                assert np.all(volume == 0), (
                    f"Expected zeros at t={t_idx}, p={pos_idx}, got non-zero data"
                )
            else:
                assert np.any(volume != 0), (
                    f"Expected non-zero data at t={t_idx}, p={pos_idx}, got all zeros"
                )


# ---------------------------------------------------------------------------
# on_frame_ready hook
# ---------------------------------------------------------------------------


def test_controller_on_frame_ready_hook(demo_engine, mantis_metadata, tmp_path):
    """Verify on_frame_ready is called for each acquired frame and can be overridden."""
    mantis_metadata["autofocus"]["enabled"] = False
    seq = MDASequence(
        stage_positions=[Position(x=0, y=0, name="0")],
        time_plan={"interval": 0, "loops": 3},
        channels=[{"config": "DAPI", "exposure": 1.0}],
        metadata={"mantis": mantis_metadata},
    )

    acq_name = _get_next_acquisition_name(tmp_path, "hook_test")
    data_path = tmp_path / f"{acq_name}.ome.zarr"

    frames_seen = []

    class TrackingController(AcquisitionController):
        def on_frame_ready(self, frame, event, meta):
            frames_seen.append(dict(event.index))
            super().on_frame_ready(frame, event, meta)

    with demo_engine.create_stream(seq, data_path) as stream:
        controller = TrackingController(demo_engine, stream, seq)
        controller.run()

    assert len(frames_seen) == 3
    assert all("t" in f for f in frames_seen)
