"""Tests for the ReplayCamera — OME-Zarr-backed UniMMCore camera device.

Creates small synthetic OME-Zarr datasets in temporary directories and
verifies that the ReplayCamera correctly serves frames through UniMMCore.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from iohub.ngff import open_ome_zarr
from pymmcore_plus.experimental.unicore.core._unicore import UniMMCore
from useq import MDAEvent

from shrimpy.mantis.replay_camera import ReplayCamera

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

N_T, N_C, N_Z, N_Y, N_X = 2, 2, 3, 32, 48
N_POS = 2
CHANNEL_NAMES = ["GFP", "BF"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fill_position(pos_node, pos_idx: int) -> None:
    """Fill a position with a deterministic pattern.

    Each pixel value encodes its coordinates so tests can verify
    exact readback: ``value = p*10000 + t*1000 + c*100 + z``.
    """
    data = np.zeros((N_T, N_C, N_Z, N_Y, N_X), dtype=np.uint16)
    for t in range(N_T):
        for c in range(N_C):
            for z in range(N_Z):
                data[t, c, z, :, :] = pos_idx * 10000 + t * 1000 + c * 100 + z
    pos_node["0"][:] = data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def zarr_path(tmp_path):
    """Create a small synthetic OME-Zarr plate and return its path."""
    path = tmp_path / "test_dataset.ome.zarr"
    plate = open_ome_zarr(
        str(path),
        layout="hcs",
        mode="w",
        channel_names=CHANNEL_NAMES,
    )
    for i in range(N_POS):
        pos = plate.create_position("A", "1", f"fov{i}")
        pos.create_zeros("0", shape=(N_T, N_C, N_Z, N_Y, N_X), dtype=np.uint16)
        _fill_position(pos, i)

    return path


@pytest.fixture
def replay_camera(zarr_path) -> ReplayCamera:
    """Return a ReplayCamera backed by the synthetic dataset."""
    return ReplayCamera(zarr_path)


@pytest.fixture
def core_with_camera(replay_camera) -> UniMMCore:
    """Return a UniMMCore with the ReplayCamera loaded as the active camera."""
    core = UniMMCore()
    core.loadPyDevice("Camera", replay_camera)
    core.initializeDevice("Camera")
    core.setCameraDevice("Camera")
    return core


# ---------------------------------------------------------------------------
# ReplayCamera unit tests
# ---------------------------------------------------------------------------


class TestReplayCameraInit:
    def test_opens_dataset(self, replay_camera):
        assert replay_camera.num_positions == N_POS

    def test_sensor_shape(self, replay_camera):
        assert replay_camera.sensor_shape() == (N_Y, N_X)

    def test_dtype(self, replay_camera):
        assert replay_camera.dtype() == np.uint16

    def test_tcz_shape(self, replay_camera):
        assert replay_camera.tcz_shape == (N_T, N_C, N_Z, N_Y, N_X)


class TestGetFrame:
    def test_returns_correct_frame(self, replay_camera):
        frame = replay_camera.get_frame(p=0, t=1, c=0, z=2)
        expected_value = 0 * 10000 + 1 * 1000 + 0 * 100 + 2
        assert frame.shape == (N_Y, N_X)
        assert frame[0, 0] == expected_value

    def test_second_position(self, replay_camera):
        frame = replay_camera.get_frame(p=1, t=0, c=1, z=1)
        expected_value = 1 * 10000 + 0 * 1000 + 1 * 100 + 1
        assert frame[0, 0] == expected_value

    def test_index_wraps_on_overflow(self, replay_camera):
        """Out-of-range t/c/z indices wrap via modulo, not raise."""
        frame = replay_camera.get_frame(p=0, t=N_T + 1, c=0, z=0)
        # (N_T + 1) % N_T == 1
        expected_value = 0 * 10000 + 1 * 1000 + 0 * 100 + 0
        assert frame[0, 0] == expected_value

    def test_position_out_of_range_raises(self, replay_camera):
        with pytest.raises(IndexError, match="Position index"):
            replay_camera.get_frame(p=99, t=0, c=0, z=0)


class TestEmptyDataset:
    def test_raises_on_empty_plate(self, tmp_path):
        """Opening a plate with no positions raises ValueError.

        iohub itself may raise before our check (e.g. if the plate lacks HCS
        metadata), so we just verify that *some* ValueError is raised.
        """
        path = tmp_path / "empty.ome.zarr"
        open_ome_zarr(str(path), layout="hcs", mode="w", channel_names=["ch"])
        with pytest.raises(ValueError):
            ReplayCamera(path)


# ---------------------------------------------------------------------------
# UniMMCore integration tests
# ---------------------------------------------------------------------------


class TestUniMMCoreIntegration:
    def test_image_dimensions(self, core_with_camera):
        assert core_with_camera.getImageWidth() == N_X
        assert core_with_camera.getImageHeight() == N_Y
        assert core_with_camera.getImageBitDepth() == 16

    def test_snap_and_get_image(self, core_with_camera, replay_camera):
        # Set indices via event tracking
        replay_camera._t = 1
        replay_camera._c = 0
        replay_camera._z = 2
        replay_camera._p = 0

        core_with_camera.snapImage()
        img = core_with_camera.getImage()
        expected = 0 * 10000 + 1 * 1000 + 0 * 100 + 2
        assert img.shape == (N_Y, N_X)
        assert img[0, 0] == expected

    def test_snap_updates_with_indices(self, core_with_camera, replay_camera):
        """Verify that changing indices changes the returned frame."""
        replay_camera._t = 0
        replay_camera._c = 0
        replay_camera._z = 0
        replay_camera._p = 0

        core_with_camera.snapImage()
        img1 = core_with_camera.getImage()

        replay_camera._c = 1
        replay_camera._p = 1

        core_with_camera.snapImage()
        img2 = core_with_camera.getImage()

        assert img1[0, 0] != img2[0, 0]
        assert img1[0, 0] == 0  # p=0,t=0,c=0,z=0
        assert img2[0, 0] == 1 * 10000 + 0 * 1000 + 1 * 100 + 0

    def test_exposure_property(self, core_with_camera):
        core_with_camera.setExposure(42.0)
        assert core_with_camera.getExposure() == 42.0

    def test_default_indices_are_zero(self, core_with_camera):
        """Before any event, indices default to (0,0,0,0)."""
        core_with_camera.snapImage()
        img = core_with_camera.getImage()
        assert img[0, 0] == 0  # p=0,t=0,c=0,z=0


# ---------------------------------------------------------------------------
# MDA event tracking tests
# ---------------------------------------------------------------------------


class TestMDAEventTracking:
    def test_connect_and_event_updates_indices(self, core_with_camera, replay_camera):
        replay_camera.connect_to_mda(core_with_camera)
        try:
            event = MDAEvent(index={"t": 1, "c": 0, "z": 2, "p": 0})
            core_with_camera.mda.events.eventStarted.emit(event)

            assert replay_camera._t == 1
            assert replay_camera._c == 0
            assert replay_camera._z == 2
            assert replay_camera._p == 0

            core_with_camera.snapImage()
            img = core_with_camera.getImage()
            expected = 0 * 10000 + 1 * 1000 + 0 * 100 + 2
            assert img[0, 0] == expected
        finally:
            replay_camera.disconnect_from_mda(core_with_camera)

    def test_event_with_missing_indices(self, core_with_camera, replay_camera):
        """Events that lack some index keys use 0 as default."""
        replay_camera.connect_to_mda(core_with_camera)
        try:
            event = MDAEvent(index={"t": 1})  # no p, c, z
            core_with_camera.mda.events.eventStarted.emit(event)

            core_with_camera.snapImage()
            img = core_with_camera.getImage()
            expected = 0 * 10000 + 1 * 1000 + 0 * 100 + 0
            assert img[0, 0] == expected
        finally:
            replay_camera.disconnect_from_mda(core_with_camera)


# ---------------------------------------------------------------------------
# Sequence acquisition tests
# ---------------------------------------------------------------------------


class TestSequenceAcquisition:
    def test_sequence_acquisition(self, core_with_camera, replay_camera):
        replay_camera._t = 0
        replay_camera._c = 0
        replay_camera._z = 0
        replay_camera._p = 0

        n_frames = 3
        core_with_camera.startSequenceAcquisition(n_frames, 0.0, True)

        # Wait for all frames to be available (acquisition runs in bg thread)
        while core_with_camera.getRemainingImageCount() < n_frames:
            time.sleep(0.001)

        frames = []
        for _ in range(n_frames):
            img, _md = core_with_camera.popNextImageAndMD()
            frames.append(img.copy())

        assert len(frames) == n_frames
        # All frames should be the same (same TCZP indices)
        for frame in frames:
            assert frame.shape == (N_Y, N_X)
            assert frame[0, 0] == 0

    def test_sequence_frame_shape(self, core_with_camera, replay_camera):
        replay_camera._t = 1
        replay_camera._c = 1
        replay_camera._z = 2
        replay_camera._p = 0

        core_with_camera.startSequenceAcquisition(1, 0.0, True)

        while core_with_camera.getRemainingImageCount() < 1:
            time.sleep(0.001)

        img, _ = core_with_camera.popNextImageAndMD()

        assert img.shape == (N_Y, N_X)
        expected = 0 * 10000 + 1 * 1000 + 1 * 100 + 2
        assert img[0, 0] == expected
