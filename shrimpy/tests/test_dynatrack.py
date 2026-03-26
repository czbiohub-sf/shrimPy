"""Tests for DynaTrackUpdater and supporting functions."""

import numpy as np
import pytest

from shrimpy.mantis.dynatrack import (
    DynaTrackUpdater,
    _center_crop,
    _limit_shifts_zyx,
    _match_shape,
    _pad_to_shape,
    _phase_cross_corr,
)
from shrimpy.mantis.position_update import PositionCoordinates

# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestCenterCrop:
    def test_basic_2d(self):
        arr = np.arange(20).reshape(4, 5)
        result = _center_crop(arr, (2, 3))
        assert result.shape == (2, 3)

    def test_noop_when_same_shape(self):
        arr = np.ones((4, 5))
        result = _center_crop(arr, (4, 5))
        np.testing.assert_array_equal(result, arr)


class TestPadToShape:
    def test_basic_2d(self):
        arr = np.ones((2, 3))
        result = _pad_to_shape(arr, (4, 5), mode="constant")
        assert result.shape == (4, 5)

    def test_noop_when_same_shape(self):
        arr = np.ones((4, 5))
        result = _pad_to_shape(arr, (4, 5), mode="constant")
        np.testing.assert_array_equal(result, arr)


class TestMatchShape:
    def test_pad_smaller(self):
        arr = np.ones((2, 3))
        result = _match_shape(arr, (4, 5))
        assert result.shape == (4, 5)

    def test_crop_larger(self):
        arr = np.ones((6, 7))
        result = _match_shape(arr, (4, 5))
        assert result.shape == (4, 5)

    def test_mixed_pad_and_crop(self):
        arr = np.ones((2, 7))
        result = _match_shape(arr, (4, 5))
        assert result.shape == (4, 5)


# ---------------------------------------------------------------------------
# Phase cross-correlation tests
# ---------------------------------------------------------------------------


class TestPhaseCrossCorr:
    def test_no_shift_returns_zeros(self):
        rng = np.random.default_rng(42)
        img = rng.random((32, 32))
        shifts = _phase_cross_corr(img, img.copy())
        assert shifts == (0, 0)

    def test_known_2d_shift(self):
        """Translate an image by a known amount and verify detected shift."""
        rng = np.random.default_rng(42)
        ref = rng.random((64, 64))
        dy, dx = 3, -5
        mov = np.roll(np.roll(ref, dy, axis=0), dx, axis=1)
        shifts = _phase_cross_corr(ref, mov)
        # The detected shift matches the roll direction
        assert shifts[0] == dy
        assert shifts[1] == dx

    def test_known_3d_shift(self):
        """Translate a 3D stack by a known amount."""
        rng = np.random.default_rng(42)
        ref = rng.random((8, 32, 32))
        dz, dy, dx = 1, 2, -3
        mov = np.roll(np.roll(np.roll(ref, dz, axis=0), dy, axis=1), dx, axis=2)
        shifts = _phase_cross_corr(ref, mov)
        assert shifts[0] == dz
        assert shifts[1] == dy
        assert shifts[2] == dx


# ---------------------------------------------------------------------------
# Shift limiting tests
# ---------------------------------------------------------------------------


class TestLimitShiftsZyx:
    def test_below_min_zeroed(self):
        shifts = np.array([0.5, 0.3, 0.1])
        limits = {"z": (1.0, 10.0), "y": (1.0, 10.0), "x": (1.0, 10.0)}
        result = _limit_shifts_zyx(shifts, limits)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_above_max_clipped(self):
        shifts = np.array([15.0, -12.0, 8.0])
        limits = {"z": (0.1, 10.0), "y": (0.1, 10.0), "x": (0.1, 10.0)}
        result = _limit_shifts_zyx(shifts, limits)
        np.testing.assert_array_equal(result, [10.0, -10.0, 8.0])

    def test_within_range_unchanged(self):
        shifts = np.array([5.0, -3.0, 2.0])
        limits = {"z": (0.1, 10.0), "y": (0.1, 10.0), "x": (0.1, 10.0)}
        result = _limit_shifts_zyx(shifts, limits)
        np.testing.assert_array_equal(result, [5.0, -3.0, 2.0])

    def test_missing_axis_ignored(self):
        shifts = np.array([5.0, 0.01, 2.0])
        limits = {"z": (0.1, 10.0)}  # only z is limited
        result = _limit_shifts_zyx(shifts, limits)
        np.testing.assert_array_equal(result, [5.0, 0.01, 2.0])


# ---------------------------------------------------------------------------
# DynaTrackUpdater._compute_shift tests
# ---------------------------------------------------------------------------


class TestComputeShift:
    def test_pixel_to_micron_conversion(self):
        """Verify that pixel shifts are scaled by the correct factors."""
        rng = np.random.default_rng(42)
        ref = rng.random((8, 64, 64))
        dz, dy, dx = 1, 2, -3
        mov = np.roll(np.roll(np.roll(ref, dz, axis=0), dy, axis=1), dx, axis=2)

        scale_yx = 0.5  # um/px
        scale_z = 2.0  # um/z-step
        updater = DynaTrackUpdater(scale_yx=scale_yx, scale_z=scale_z)

        x_um, y_um, z_um = updater._compute_shift(ref, mov)

        # Phase cross-corr returns shift matching the roll direction, then scaled
        assert x_um == pytest.approx(dx * scale_yx, abs=1e-6)
        assert y_um == pytest.approx(dy * scale_yx, abs=1e-6)
        assert z_um == pytest.approx(dz * scale_z, abs=1e-6)

    def test_dampening_applied(self):
        """Dampening factors should scale the output shift."""
        rng = np.random.default_rng(42)
        ref = rng.random((8, 64, 64))
        dz, dy, dx = 1, 2, -3
        mov = np.roll(np.roll(np.roll(ref, dz, axis=0), dy, axis=1), dx, axis=2)

        scale_yx = 0.5
        scale_z = 2.0
        dampening = (0.5, 0.25, 0.1)  # z, y, x
        updater = DynaTrackUpdater(scale_yx=scale_yx, scale_z=scale_z, dampening=dampening)

        x_um, y_um, z_um = updater._compute_shift(ref, mov)

        expected_x = dx * scale_yx * dampening[2]
        expected_y = dy * scale_yx * dampening[1]
        expected_z = dz * scale_z * dampening[0]

        assert x_um == pytest.approx(expected_x, abs=1e-6)
        assert y_um == pytest.approx(expected_y, abs=1e-6)
        assert z_um == pytest.approx(expected_z, abs=1e-6)

    def test_shift_limits_applied(self):
        """Shift limits should zero out or clip shifts."""
        rng = np.random.default_rng(42)
        ref = rng.random((8, 64, 64))
        # Large shift in x (3 px * 10 um/px = 30 um) should be clipped to 5 um
        dx = -3
        mov = np.roll(ref, dx, axis=2)

        scale_yx = 10.0
        scale_z = 2.0
        shift_limits = {"z": (0.1, 50.0), "y": (0.1, 50.0), "x": (0.1, 5.0)}
        updater = DynaTrackUpdater(
            scale_yx=scale_yx, scale_z=scale_z, shift_limits=shift_limits
        )

        x_um, y_um, z_um = updater._compute_shift(ref, mov)

        # x shift: -3 * 10 = -30 um, clipped to -5 (sign preserved)
        assert x_um == pytest.approx(-5.0, abs=1e-6)
        # y and z should be zero (no shift, below min threshold)
        assert z_um == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Full update() flow tests
# ---------------------------------------------------------------------------


class TestDynaTrackUpdaterFlow:
    def test_first_call_stores_reference(self):
        """First call stores the reference and returns position unchanged."""
        updater = DynaTrackUpdater(scale_yx=0.5, scale_z=2.0)
        pos = PositionCoordinates(x=100.0, y=200.0, z=50.0)
        data = [np.random.default_rng(42).random((64, 64)) for _ in range(8)]

        result = updater.update(0, 0, pos, data)

        assert result.x == 100.0
        assert result.y == 200.0
        assert result.z == 50.0
        assert 0 in updater._reference_stacks

    def test_second_call_detects_shift(self):
        """Second call computes a shift and returns an updated position."""
        rng = np.random.default_rng(42)
        scale_yx = 0.5
        scale_z = 2.0
        updater = DynaTrackUpdater(scale_yx=scale_yx, scale_z=scale_z)
        pos = PositionCoordinates(x=100.0, y=200.0, z=50.0)

        ref_frames = [rng.random((64, 64)) for _ in range(8)]

        # Create moved frames with a known shift
        dy, dx = 2, -3
        mov_frames = [np.roll(np.roll(frame, dy, axis=0), dx, axis=1) for frame in ref_frames]

        # First call: store reference
        updater.update(0, 0, pos, ref_frames)

        # Second call: detect shift
        result = updater.update(1, 0, pos, mov_frames)

        expected_x = 100.0 + dx * scale_yx
        expected_y = 200.0 + dy * scale_yx
        assert result.x == pytest.approx(expected_x, abs=1e-6)
        assert result.y == pytest.approx(expected_y, abs=1e-6)

    def test_no_data_returns_unchanged(self):
        """When data is None, position is returned unchanged."""
        updater = DynaTrackUpdater(scale_yx=0.5, scale_z=2.0)
        pos = PositionCoordinates(x=100.0, y=200.0, z=50.0)

        result = updater.update(0, 0, pos, None)
        assert result.x == 100.0
        assert result.y == 200.0

    def test_empty_data_returns_unchanged(self):
        """When data is an empty list, position is returned unchanged."""
        updater = DynaTrackUpdater(scale_yx=0.5, scale_z=2.0)
        pos = PositionCoordinates(x=100.0, y=200.0, z=50.0)

        result = updater.update(0, 0, pos, [])
        assert result.x == 100.0
        assert result.y == 200.0
