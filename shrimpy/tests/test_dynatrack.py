"""Tests for DynaTrackUpdater and supporting functions."""

import csv

import numpy as np
import pytest
import torch

from shrimpy.mantis.dynatrack import (
    DynaTrackConfig,
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
        t = torch.arange(20).reshape(4, 5)
        result = _center_crop(t, (2, 3))
        assert tuple(result.shape) == (2, 3)

    def test_noop_when_same_shape(self):
        t = torch.ones(4, 5)
        result = _center_crop(t, (4, 5))
        assert torch.equal(result, t)


class TestPadToShape:
    def test_basic_2d(self):
        t = torch.ones(2, 3)
        result = _pad_to_shape(t, (4, 5), mode="constant")
        assert tuple(result.shape) == (4, 5)

    def test_noop_when_same_shape(self):
        t = torch.ones(4, 5)
        result = _pad_to_shape(t, (4, 5), mode="constant")
        assert torch.equal(result, t)


class TestMatchShape:
    def test_pad_smaller(self):
        t = torch.ones(2, 3)
        result = _match_shape(t, (4, 5))
        assert tuple(result.shape) == (4, 5)

    def test_crop_larger(self):
        t = torch.ones(6, 7)
        result = _match_shape(t, (4, 5))
        assert tuple(result.shape) == (4, 5)

    def test_mixed_pad_and_crop(self):
        t = torch.ones(2, 7)
        result = _match_shape(t, (4, 5))
        assert tuple(result.shape) == (4, 5)


# ---------------------------------------------------------------------------
# Phase cross-correlation tests
# ---------------------------------------------------------------------------


class TestPhaseCrossCorr:
    def test_no_shift_returns_zeros(self):
        rng = np.random.default_rng(42)
        img = torch.as_tensor(rng.random((32, 32)), dtype=torch.float32)
        shifts = _phase_cross_corr(img, img.clone())
        assert shifts == (0, 0)

    def test_known_2d_shift(self):
        """Translate an image by a known amount and verify detected shift."""
        rng = np.random.default_rng(42)
        ref = torch.as_tensor(rng.random((64, 64)), dtype=torch.float32)
        dy, dx = 3, -5
        mov = torch.roll(ref, shifts=(dy, dx), dims=(0, 1))
        shifts = _phase_cross_corr(ref, mov)
        # The detected shift matches the roll direction
        assert shifts[0] == dy
        assert shifts[1] == dx

    def test_known_3d_shift(self):
        """Translate a 3D stack by a known amount."""
        rng = np.random.default_rng(42)
        ref = torch.as_tensor(rng.random((8, 32, 32)), dtype=torch.float32)
        dz, dy, dx = 1, 2, -3
        mov = torch.roll(ref, shifts=(dz, dy, dx), dims=(0, 1, 2))
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
# DynaTrackConfig tests
# ---------------------------------------------------------------------------


class TestDynaTrackConfig:
    def test_minimal_config(self):
        cfg = DynaTrackConfig(scale_yx=0.5, scale_z=2.0)
        assert cfg.scale_yx == 0.5
        assert cfg.scale_z == 2.0
        assert cfg.maximum_shift == 1.0
        assert cfg.dampening is None
        assert cfg.shift_limits is None
        assert cfg.tracking_interval == 1
        assert cfg.shift_estimation_channel == "raw"
        assert cfg.preprocessing is None
        assert cfg.shift_log_path is None

    def test_full_config(self):
        cfg = DynaTrackConfig(
            scale_yx=0.075,
            scale_z=0.174,
            dampening=(0.5, 0.8, 0.8),
            shift_limits={"z": (0.5, 2.0), "y": (2.0, 10.0), "x": (2.0, 10.0)},
            tracking_interval=2,
            shift_estimation_channel="phase",
            preprocessing=["phase"],
            phase_config={"wavelength": 0.450},
        )
        assert cfg.dampening == (0.5, 0.8, 0.8)
        assert cfg.tracking_interval == 2
        assert cfg.shift_estimation_channel == "phase"

    def test_config_from_dict(self):
        """Config can be constructed from a metadata dict via **kwargs."""
        meta = {
            "scale_yx": 0.075,
            "scale_z": 0.174,
            "dampening": (0.5, 0.8, 0.8),
        }
        cfg = DynaTrackConfig(**meta)
        assert cfg.scale_yx == 0.075


# ---------------------------------------------------------------------------
# DynaTrackUpdater._compute_shift tests
# ---------------------------------------------------------------------------


class TestComputeShift:
    def _make_updater(self, **kwargs):
        defaults = {"scale_yx": 0.5, "scale_z": 2.0}
        defaults.update(kwargs)
        return DynaTrackUpdater(config=DynaTrackConfig(**defaults))

    def test_pixel_to_micron_conversion(self):
        """Verify that pixel shifts are scaled by the correct factors."""
        rng = np.random.default_rng(42)
        ref = torch.as_tensor(rng.random((8, 64, 64)), dtype=torch.float32)
        dz, dy, dx = 1, 2, -3
        mov = torch.roll(ref, shifts=(dz, dy, dx), dims=(0, 1, 2))

        scale_yx = 0.5  # um/px
        scale_z = 2.0  # um/z-step
        updater = self._make_updater(scale_yx=scale_yx, scale_z=scale_z)

        x_um, y_um, z_um = updater._compute_shift(ref, mov)

        # Phase cross-corr returns shift matching the roll direction, then scaled
        assert x_um == pytest.approx(dx * scale_yx, abs=1e-6)
        assert y_um == pytest.approx(dy * scale_yx, abs=1e-6)
        assert z_um == pytest.approx(dz * scale_z, abs=1e-6)

    def test_dampening_applied(self):
        """Dampening factors should scale the output shift."""
        rng = np.random.default_rng(42)
        ref = torch.as_tensor(rng.random((8, 64, 64)), dtype=torch.float32)
        dz, dy, dx = 1, 2, -3
        mov = torch.roll(ref, shifts=(dz, dy, dx), dims=(0, 1, 2))

        scale_yx = 0.5
        scale_z = 2.0
        dampening = (0.5, 0.25, 0.1)  # z, y, x
        updater = self._make_updater(scale_yx=scale_yx, scale_z=scale_z, dampening=dampening)

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
        ref = torch.as_tensor(rng.random((8, 64, 64)), dtype=torch.float32)
        # Large shift in x (3 px * 10 um/px = 30 um) should be clipped to 5 um
        dx = -3
        mov = torch.roll(ref, shifts=dx, dims=2)

        scale_yx = 10.0
        scale_z = 2.0
        shift_limits = {"z": (0.1, 50.0), "y": (0.1, 50.0), "x": (0.1, 5.0)}
        updater = self._make_updater(
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
    def _make_updater(self, **kwargs):
        defaults = {"scale_yx": 0.5, "scale_z": 2.0}
        defaults.update(kwargs)
        return DynaTrackUpdater(config=DynaTrackConfig(**defaults))

    def test_first_call_stores_reference(self):
        """First call stores the reference and returns position unchanged."""
        updater = self._make_updater()
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
        updater = self._make_updater(scale_yx=scale_yx)
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
        updater = self._make_updater()
        pos = PositionCoordinates(x=100.0, y=200.0, z=50.0)

        result = updater.update(0, 0, pos, None)
        assert result.x == 100.0
        assert result.y == 200.0

    def test_empty_data_returns_unchanged(self):
        """When data is an empty list, position is returned unchanged."""
        updater = self._make_updater()
        pos = PositionCoordinates(x=100.0, y=200.0, z=50.0)

        result = updater.update(0, 0, pos, [])
        assert result.x == 100.0
        assert result.y == 200.0


# ---------------------------------------------------------------------------
# Tracking interval tests
# ---------------------------------------------------------------------------


class TestTrackingInterval:
    def test_skip_non_interval_timepoints(self):
        """Updates are skipped when timepoint is not on the tracking interval."""
        rng = np.random.default_rng(42)
        config = DynaTrackConfig(scale_yx=0.5, scale_z=2.0, tracking_interval=3)
        updater = DynaTrackUpdater(config=config)
        pos = PositionCoordinates(x=100.0, y=200.0, z=50.0)

        ref_frames = [rng.random((64, 64)) for _ in range(8)]
        dy, dx = 2, -3
        mov_frames = [np.roll(np.roll(f, dy, axis=0), dx, axis=1) for f in ref_frames]

        # t=0: store reference
        updater.update(0, 0, pos, ref_frames)

        # t=1: not on interval (1 % 3 != 0), should return unchanged
        result = updater.update(1, 0, pos, mov_frames)
        assert result.x == 100.0
        assert result.y == 200.0

        # t=2: not on interval
        result = updater.update(2, 0, pos, mov_frames)
        assert result.x == 100.0

        # t=3: on interval (3 % 3 == 0), should detect shift
        result = updater.update(3, 0, pos, mov_frames)
        assert result.x != 100.0  # shift detected

    def test_interval_1_tracks_every_timepoint(self):
        """Default interval=1 tracks every timepoint."""
        rng = np.random.default_rng(42)
        updater = DynaTrackUpdater(config=DynaTrackConfig(scale_yx=0.5, scale_z=2.0))
        pos = PositionCoordinates(x=100.0, y=200.0, z=50.0)

        ref_frames = [rng.random((64, 64)) for _ in range(8)]
        dy, dx = 2, -3
        mov_frames = [np.roll(np.roll(f, dy, axis=0), dx, axis=1) for f in ref_frames]

        updater.update(0, 0, pos, ref_frames)
        result = updater.update(1, 0, pos, mov_frames)
        assert result.x != 100.0  # shift detected at t=1


# ---------------------------------------------------------------------------
# Preprocessor hook tests
# ---------------------------------------------------------------------------


class TestPreprocessor:
    def test_preprocessor_is_applied(self):
        """Preprocessor transforms data before shift estimation."""
        rng = np.random.default_rng(42)
        config = DynaTrackConfig(scale_yx=0.5, scale_z=2.0)

        call_count = [0]

        def identity_preprocessor(stack: np.ndarray) -> np.ndarray:
            call_count[0] += 1
            return stack

        updater = DynaTrackUpdater(config=config, preprocessor=identity_preprocessor)
        pos = PositionCoordinates(x=100.0, y=200.0, z=50.0)
        frames = [rng.random((64, 64)) for _ in range(8)]

        # First call: preprocessor applied to reference
        updater.update(0, 0, pos, frames)
        assert call_count[0] == 1

        # Second call: preprocessor applied to current stack
        updater.update(1, 0, pos, frames)
        assert call_count[0] == 2

    def test_preprocessor_affects_shift(self):
        """A preprocessor that introduces a shift should be detected."""
        rng = np.random.default_rng(42)
        config = DynaTrackConfig(scale_yx=1.0, scale_z=1.0)

        # Preprocessor that rolls the stack by 2 pixels in Y
        first_call = [True]

        def shifting_preprocessor(stack: np.ndarray) -> np.ndarray:
            if first_call[0]:
                first_call[0] = False
                return stack
            return np.roll(stack, 2, axis=1)

        updater = DynaTrackUpdater(config=config, preprocessor=shifting_preprocessor)
        pos = PositionCoordinates(x=0.0, y=0.0, z=0.0)
        frames = [rng.random((64, 64)) for _ in range(8)]

        updater.update(0, 0, pos, frames)
        result = updater.update(1, 0, pos, frames)

        # The preprocessor-introduced shift should be detected
        assert result.y == pytest.approx(2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Shift logging tests
# ---------------------------------------------------------------------------


class TestShiftLogging:
    def test_shift_log_created_on_first_write(self, tmp_path):
        """CSV file is created with header on first shift computation."""
        log_path = tmp_path / "shifts.csv"
        config = DynaTrackConfig(scale_yx=0.5, scale_z=2.0, shift_log_path=str(log_path))
        updater = DynaTrackUpdater(config=config)
        pos = PositionCoordinates(x=100.0, y=200.0, z=50.0)

        rng = np.random.default_rng(42)
        ref_frames = [rng.random((64, 64)) for _ in range(8)]
        dy, dx = 2, -3
        mov_frames = [np.roll(np.roll(f, dy, axis=0), dx, axis=1) for f in ref_frames]

        # First call: stores reference, no shift logged
        updater.update(0, 0, pos, ref_frames)
        assert not log_path.exists()

        # Second call: computes shift, log created
        updater.update(1, 0, pos, mov_frames)
        assert log_path.exists()

        with open(log_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header[0] == "position_index"
            assert header[1] == "timepoint_index"
            row = next(reader)
            assert row[0] == "0"  # position_index
            assert row[1] == "1"  # timepoint_index

    def test_multiple_shifts_appended(self, tmp_path):
        """Each shift is appended as a new row."""
        log_path = tmp_path / "shifts.csv"
        config = DynaTrackConfig(scale_yx=0.5, scale_z=2.0, shift_log_path=str(log_path))
        updater = DynaTrackUpdater(config=config)
        pos = PositionCoordinates(x=100.0, y=200.0, z=50.0)

        rng = np.random.default_rng(42)
        ref_frames = [rng.random((64, 64)) for _ in range(8)]
        mov_frames = [np.roll(f, 2, axis=0) for f in ref_frames]

        updater.update(0, 0, pos, ref_frames)  # store ref
        updater.update(1, 0, pos, mov_frames)  # shift 1
        updater.update(2, 0, pos, mov_frames)  # shift 2

        with open(log_path) as f:
            reader = csv.reader(f)
            next(reader)  # header
            rows = list(reader)
            assert len(rows) == 2

    def test_no_log_when_path_is_none(self):
        """No CSV is created when shift_log_path is None."""
        config = DynaTrackConfig(scale_yx=0.5, scale_z=2.0)
        updater = DynaTrackUpdater(config=config)
        pos = PositionCoordinates(x=100.0, y=200.0, z=50.0)

        rng = np.random.default_rng(42)
        frames = [rng.random((64, 64)) for _ in range(8)]
        mov = [np.roll(f, 2, axis=0) for f in frames]

        updater.update(0, 0, pos, frames)
        updater.update(1, 0, pos, mov)
        # No exception, no file created — just verifying no crash
