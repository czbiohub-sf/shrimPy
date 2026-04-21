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

N_T, N_C, N_Z, N_Y, N_X = 3, 2, 10, 32, 48
CHANNEL_NAMES = ["BF", "GFP"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fill_dataset(pos_node) -> None:
    """Fill a FOV with a deterministic pattern.

    Each pixel value encodes its coordinates so tests can verify
    exact readback: ``value = t*10000 + c*1000 + z``.
    """
    data = np.zeros((N_T, N_C, N_Z, N_Y, N_X), dtype=np.uint16)
    for t in range(N_T):
        for c in range(N_C):
            for z in range(N_Z):
                data[t, c, z, :, :] = t * 10000 + c * 1000 + z
    pos_node["0"][:] = data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def zarr_path(tmp_path):
    """Create a small synthetic OME-Zarr FOV and return its path."""
    path = tmp_path / "test_fov.zarr"
    dataset = open_ome_zarr(
        str(path),
        layout="fov",
        mode="w",
        channel_names=CHANNEL_NAMES,
    )
    dataset.create_zeros("0", shape=(N_T, N_C, N_Z, N_Y, N_X), dtype=np.uint16)
    _fill_dataset(dataset)
    dataset.close()
    return path


@pytest.fixture
def replay_camera(zarr_path) -> ReplayCamera:
    """Return an initialized ReplayCamera backed by the synthetic dataset."""
    camera = ReplayCamera()
    camera._data_path = str(zarr_path)
    camera.initialize()
    return camera


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
    def test_sensor_shape(self, replay_camera):
        assert replay_camera.sensor_shape() == (N_Y, N_X)

    def test_dtype(self, replay_camera):
        assert replay_camera.dtype() == np.uint16

    def test_data_shape(self, replay_camera):
        assert replay_camera.data_shape == (N_T, N_C, N_Z, N_Y, N_X)

    def test_channel_names(self, replay_camera):
        assert replay_camera.channel_names == CHANNEL_NAMES

    def test_default_channel(self, replay_camera):
        assert replay_camera._channel_name == "BF"
        assert replay_camera._channel_index == 0

    def test_z_center(self, replay_camera):
        assert replay_camera._z_center == N_Z // 2

    def test_missing_data_path_raises(self):
        camera = ReplayCamera()
        with pytest.raises(RuntimeError, match="DataPath"):
            camera.initialize()


class TestGetFrame:
    def test_returns_correct_frame(self, replay_camera):
        frame = replay_camera.get_frame(t=1, c=0, z=2)
        expected = 1 * 10000 + 0 * 1000 + 2
        assert frame.shape == (N_Y, N_X)
        assert frame[0, 0] == expected

    def test_second_channel(self, replay_camera):
        frame = replay_camera.get_frame(t=0, c=1, z=1)
        expected = 0 * 10000 + 1 * 1000 + 1
        assert frame[0, 0] == expected

    def test_index_wraps_on_overflow(self, replay_camera):
        frame = replay_camera.get_frame(t=N_T + 1, c=0, z=0)
        # (N_T + 1) % N_T == 1
        expected = 1 * 10000
        assert frame[0, 0] == expected


# ---------------------------------------------------------------------------
# Channel switching tests
# ---------------------------------------------------------------------------


class TestChannelSwitching:
    def test_set_valid_channel(self, replay_camera):
        replay_camera._set_channel("GFP")
        assert replay_camera._channel_name == "GFP"
        assert replay_camera._channel_index == 1

    def test_set_invalid_channel_returns_zeros(self, replay_camera):
        replay_camera._set_channel("DAPI")
        assert replay_camera._channel_name == "DAPI"
        assert replay_camera._channel_index == -1

        # Snap should return zeros
        buf = np.ones((N_Y, N_X), dtype=np.uint16)
        replay_camera.snap(buf)
        assert np.all(buf == 0)

    def test_switch_back_to_valid(self, replay_camera):
        replay_camera._set_channel("DAPI")
        replay_camera._set_channel("BF")
        assert replay_camera._channel_index == 0

        buf = np.empty((N_Y, N_X), dtype=np.uint16)
        replay_camera.snap(buf)
        # t=0, c=0 (BF), z=center
        expected = 0 * 10000 + 0 * 1000 + (N_Z // 2)
        assert buf[0, 0] == expected


# ---------------------------------------------------------------------------
# Timepoint auto-increment tests
# ---------------------------------------------------------------------------


class TestTimepointAutoIncrement:
    def test_snap_increments_timepoint(self, replay_camera):
        z_center = N_Z // 2
        buf = np.empty((N_Y, N_X), dtype=np.uint16)

        # First snap: t=0
        replay_camera.snap(buf)
        assert buf[0, 0] == 0 * 10000 + 0 * 1000 + z_center

        # Second snap: t=1
        replay_camera.snap(buf)
        assert buf[0, 0] == 1 * 10000 + 0 * 1000 + z_center

        # Third snap: t=2
        replay_camera.snap(buf)
        assert buf[0, 0] == 2 * 10000 + 0 * 1000 + z_center

    def test_timepoint_wraps(self, replay_camera):
        buf = np.empty((N_Y, N_X), dtype=np.uint16)
        z_center = N_Z // 2

        # Snap N_T times (t=0, 1, 2), counter now at 3
        for _ in range(N_T):
            replay_camera.snap(buf)

        # Next snap wraps: 3 % 3 = 0
        replay_camera.snap(buf)
        assert buf[0, 0] == 0 * 10000 + 0 * 1000 + z_center


# ---------------------------------------------------------------------------
# Z-position tracking tests
# ---------------------------------------------------------------------------


class TestZPositionTracking:
    def test_default_returns_center(self, replay_camera):
        buf = np.empty((N_Y, N_X), dtype=np.uint16)
        replay_camera.snap(buf)
        expected = 0 * 10000 + 0 * 1000 + (N_Z // 2)
        assert buf[0, 0] == expected

    def test_z_offset_shifts_index(self, replay_camera):
        # Move Z by +2 steps worth of z_scale
        replay_camera._z_position = 2 * replay_camera._z_scale
        buf = np.empty((N_Y, N_X), dtype=np.uint16)
        replay_camera.snap(buf)
        expected_z = (N_Z // 2) + 2
        expected = 0 * 10000 + 0 * 1000 + expected_z
        assert buf[0, 0] == expected

    def test_z_clamped_to_bounds(self, replay_camera):
        # Move Z way beyond dataset range
        replay_camera._z_position = 1000 * replay_camera._z_scale
        assert replay_camera._get_z_index() == N_Z - 1

        replay_camera._z_position = -1000 * replay_camera._z_scale
        assert replay_camera._get_z_index() == 0

    def test_negative_z_offset(self, replay_camera):
        replay_camera._z_position = -3 * replay_camera._z_scale
        expected_z = max(0, (N_Z // 2) - 3)
        assert replay_camera._get_z_index() == expected_z


# ---------------------------------------------------------------------------
# UniMMCore integration tests
# ---------------------------------------------------------------------------


class TestUniMMCoreIntegration:
    def test_image_dimensions(self, core_with_camera):
        assert core_with_camera.getImageWidth() == N_X
        assert core_with_camera.getImageHeight() == N_Y
        assert core_with_camera.getImageBitDepth() == 16

    def test_snap_and_get_image(self, core_with_camera, replay_camera):
        core_with_camera.snapImage()
        img = core_with_camera.getImage()
        expected = 0 * 10000 + 0 * 1000 + (N_Z // 2)
        assert img.shape == (N_Y, N_X)
        assert img[0, 0] == expected

    def test_exposure_property(self, core_with_camera):
        core_with_camera.setExposure(42.0)
        assert core_with_camera.getExposure() == 42.0

    def test_channel_property_via_core(self, core_with_camera, replay_camera):
        core_with_camera.setProperty("Camera", "Channel", "GFP")
        assert replay_camera._channel_index == 1

        core_with_camera.snapImage()
        img = core_with_camera.getImage()
        # t=0, c=1 (GFP), z=center
        expected = 0 * 10000 + 1 * 1000 + (N_Z // 2)
        assert img[0, 0] == expected


# ---------------------------------------------------------------------------
# MDA event tracking tests
# ---------------------------------------------------------------------------


class TestMDAEventTracking:
    def test_mda_overrides_timepoint(self, core_with_camera, replay_camera):
        replay_camera.connect_to_mda(core_with_camera)
        try:
            event = MDAEvent(index={"t": 2, "c": 0, "z": 0})
            core_with_camera.mda.events.eventStarted.emit(event)

            core_with_camera.snapImage()
            img = core_with_camera.getImage()
            # t=2, c=0 (BF), z=center (z_pos not set in event)
            expected = 2 * 10000 + 0 * 1000 + (N_Z // 2)
            assert img[0, 0] == expected
        finally:
            replay_camera.disconnect_from_mda()

    def test_mda_disables_auto_increment(self, core_with_camera, replay_camera):
        replay_camera.connect_to_mda(core_with_camera)
        try:
            event = MDAEvent(index={"t": 1})
            core_with_camera.mda.events.eventStarted.emit(event)

            buf1 = np.empty((N_Y, N_X), dtype=np.uint16)
            buf2 = np.empty((N_Y, N_X), dtype=np.uint16)
            replay_camera.snap(buf1)
            replay_camera.snap(buf2)
            # Both snaps should be at t=1 (no auto-increment in MDA mode)
            assert buf1[0, 0] == buf2[0, 0]
        finally:
            replay_camera.disconnect_from_mda()


# ---------------------------------------------------------------------------
# Sequence acquisition tests
# ---------------------------------------------------------------------------


class TestSequenceAcquisition:
    def test_sequence_acquisition(self, core_with_camera, replay_camera):
        n_frames = 3
        core_with_camera.startSequenceAcquisition(n_frames, 0.0, True)

        while core_with_camera.getRemainingImageCount() < n_frames:
            time.sleep(0.001)

        frames = []
        for _ in range(n_frames):
            img, _md = core_with_camera.popNextImageAndMD()
            frames.append(img.copy())

        assert len(frames) == n_frames
        for frame in frames:
            assert frame.shape == (N_Y, N_X)
