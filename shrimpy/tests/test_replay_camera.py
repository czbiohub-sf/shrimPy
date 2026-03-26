"""Tests for the ReplayCamera — OME-Zarr-backed camera patch.

Creates small synthetic OME-Zarr datasets in temporary directories and
verifies that the ReplayCamera correctly serves frames through a patched
CMMCorePlus instance.

These tests do NOT require real Micro-Manager device adapters.  The patched
core tests use a lightweight ``_FakeCore`` that provides just enough of the
CMMCorePlus interface (including psygnal-based MDA signals) to exercise the
replay camera patching logic.
"""

from __future__ import annotations

import numpy as np
import pytest

from iohub.ngff import open_ome_zarr
from psygnal import Signal
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


class _MDAEvents:
    """Minimal signal group matching the MDA events the ReplayCamera connects to."""

    eventStarted = Signal(MDAEvent)
    frameReady = Signal(np.ndarray, MDAEvent, dict)


class _MDA:
    """Minimal stand-in for ``core.mda``."""

    def __init__(self):
        self.events = _MDAEvents()


class _FakeCore:
    """Lightweight fake CMMCorePlus with real psygnal signals.

    Provides the methods that ``patch_with_object`` will try to replace, and
    the ``mda.events`` signal group that ReplayCamera connects to.
    """

    def __init__(self):
        self.mda = _MDA()

    # Methods that ReplayCamera patches — defaults return nonsense so we can
    # verify the patch actually replaced them.
    def snapImage(self):  # noqa: N802
        raise RuntimeError("original snapImage called")

    def getImage(self):  # noqa: N802
        raise RuntimeError("original getImage called")

    def getLastImage(self):  # noqa: N802
        raise RuntimeError("original getLastImage called")

    def getImageWidth(self):  # noqa: N802
        return -1

    def getImageHeight(self):  # noqa: N802
        return -1

    def getImageBitDepth(self):  # noqa: N802
        return -1

    def startSequenceAcquisition(self, n, interval=0, stop=True):  # noqa: N802
        raise RuntimeError("original startSequenceAcquisition called")

    def getRemainingImageCount(self):  # noqa: N802
        return -1

    def isSequenceRunning(self):  # noqa: N802
        return False

    def popNextImageAndMD(self):  # noqa: N802
        raise RuntimeError("original popNextImageAndMD called")

    def stopSequenceAcquisition(self):  # noqa: N802
        raise RuntimeError("original stopSequenceAcquisition called")


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
def fake_core() -> _FakeCore:
    return _FakeCore()


# ---------------------------------------------------------------------------
# ReplayCamera unit tests
# ---------------------------------------------------------------------------


class TestReplayCameraInit:
    def test_opens_dataset(self, replay_camera):
        assert replay_camera.num_positions == N_POS

    def test_shape(self, replay_camera):
        assert replay_camera.shape == (N_T, N_C, N_Z, N_Y, N_X)

    def test_dtype(self, replay_camera):
        assert replay_camera.dtype == np.uint16


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
# Patched core tests
# ---------------------------------------------------------------------------


class TestPatchedCore:
    def test_image_dimensions(self, replay_camera, fake_core):
        with replay_camera.patch(fake_core):
            assert fake_core.getImageWidth() == N_X
            assert fake_core.getImageHeight() == N_Y
            assert fake_core.getImageBitDepth() == 16

    def test_snap_and_get_image(self, replay_camera, fake_core):
        with replay_camera.patch(fake_core):
            # Simulate an event at t=1, c=0, z=2, p=0
            event = MDAEvent(index={"t": 1, "c": 0, "z": 2, "p": 0})
            fake_core.mda.events.eventStarted.emit(event)

            fake_core.snapImage()
            img = fake_core.getImage()
            expected = 0 * 10000 + 1 * 1000 + 0 * 100 + 2
            assert img.shape == (N_Y, N_X)
            assert img[0, 0] == expected

    def test_snap_updates_with_event(self, replay_camera, fake_core):
        """Verify that changing the event changes the returned frame."""
        with replay_camera.patch(fake_core):
            # First event
            e1 = MDAEvent(index={"t": 0, "c": 0, "z": 0, "p": 0})
            fake_core.mda.events.eventStarted.emit(e1)
            fake_core.snapImage()
            img1 = fake_core.getImage()

            # Second event — different position & channel
            e2 = MDAEvent(index={"t": 0, "c": 1, "z": 0, "p": 1})
            fake_core.mda.events.eventStarted.emit(e2)
            fake_core.snapImage()
            img2 = fake_core.getImage()

            assert img1[0, 0] != img2[0, 0]
            assert img1[0, 0] == 0  # p=0,t=0,c=0,z=0
            assert img2[0, 0] == 1 * 10000 + 0 * 1000 + 1 * 100 + 0

    def test_patch_is_temporary(self, replay_camera, fake_core):
        """After exiting the context, core methods are restored."""
        with replay_camera.patch(fake_core):
            # Patched — should not raise
            fake_core.snapImage()

        # Restored — should raise from the original
        with pytest.raises(RuntimeError, match="original snapImage"):
            fake_core.snapImage()

    def test_getLastImage(self, replay_camera, fake_core):
        with replay_camera.patch(fake_core):
            event = MDAEvent(index={"t": 0, "c": 1, "z": 2, "p": 0})
            fake_core.mda.events.eventStarted.emit(event)
            img = fake_core.getLastImage()
            expected = 0 * 10000 + 0 * 1000 + 1 * 100 + 2
            assert img[0, 0] == expected

    def test_default_indices_are_zero(self, replay_camera, fake_core):
        """Before any event is emitted, indices default to (0,0,0,0)."""
        with replay_camera.patch(fake_core):
            img = fake_core.getImage()
            assert img[0, 0] == 0  # p=0,t=0,c=0,z=0

    def test_event_with_missing_indices(self, replay_camera, fake_core):
        """Events that lack some index keys use 0 as default."""
        with replay_camera.patch(fake_core):
            event = MDAEvent(index={"t": 1})  # no p, c, z
            fake_core.mda.events.eventStarted.emit(event)
            img = fake_core.getImage()
            expected = 0 * 10000 + 1 * 1000 + 0 * 100 + 0
            assert img[0, 0] == expected


# ---------------------------------------------------------------------------
# Sequence acquisition tests
# ---------------------------------------------------------------------------


class TestSequenceAcquisition:
    def test_start_and_pop(self, replay_camera, fake_core):
        with replay_camera.patch(fake_core):
            event = MDAEvent(index={"t": 0, "c": 0, "z": 0, "p": 0})
            fake_core.mda.events.eventStarted.emit(event)

            fake_core.startSequenceAcquisition(3)
            assert fake_core.getRemainingImageCount() == 3
            assert fake_core.isSequenceRunning()

            frames = []
            while fake_core.getRemainingImageCount() > 0:
                img, _md = fake_core.popNextImageAndMD()
                frames.append(img)

            assert len(frames) == 3
            assert not fake_core.isSequenceRunning()

    def test_stop_clears_queue(self, replay_camera, fake_core):
        with replay_camera.patch(fake_core):
            event = MDAEvent(index={"t": 0, "c": 0, "z": 0, "p": 0})
            fake_core.mda.events.eventStarted.emit(event)

            fake_core.startSequenceAcquisition(5)
            assert fake_core.getRemainingImageCount() == 5
            fake_core.stopSequenceAcquisition()
            assert fake_core.getRemainingImageCount() == 0
            assert not fake_core.isSequenceRunning()

    def test_pop_from_empty_queue_raises(self, replay_camera, fake_core):
        with replay_camera.patch(fake_core):
            fake_core.startSequenceAcquisition(1)
            fake_core.popNextImageAndMD()  # drain the single frame
            with pytest.raises(RuntimeError, match="No images"):
                fake_core.popNextImageAndMD()

    def test_sequence_frames_have_correct_shape(self, replay_camera, fake_core):
        with replay_camera.patch(fake_core):
            event = MDAEvent(index={"t": 1, "c": 1, "z": 2, "p": 0})
            fake_core.mda.events.eventStarted.emit(event)

            fake_core.startSequenceAcquisition(2)
            img, _ = fake_core.popNextImageAndMD()
            assert img.shape == (N_Y, N_X)
            expected = 0 * 10000 + 1 * 1000 + 1 * 100 + 2
            assert img[0, 0] == expected
