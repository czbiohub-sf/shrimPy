"""Tests for DynaTrackUpdater and supporting functions."""

import csv

import numpy as np
import pytest

from shrimpy.dynatrack.position_update import PositionCoordinates
from shrimpy.dynatrack.tracking import (
    DynaTrackConfig,
    DynaTrackUpdater,
    _binary_mask,
    _center_crop,
    _center_of_mass,
    _centered_gaussian_blob,
    _gaussian_blur_3d,
    _intensity_center_of_mass,
    _intensity_center_of_mass_to_roi_center,
    _limit_shifts_zyx,
    _match_shape,
    _multiotsu_center_of_mass,
    _multiotsu_pcc,
    _pad_to_shape,
    _percentile,
    _phase_cross_corr,
    _roi_center_pcc,
)

# torch ships only in the optional `dynatrack` dependency group; skip the whole
# module when it is unavailable (neither imported module above requires it).
torch = pytest.importorskip("torch")

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
        cfg = DynaTrackConfig(input_channel="BF", tracking_channel="BF")
        assert cfg.shift.maximum == 1.0
        assert cfg.shift.dampening is None
        assert cfg.shift.limits is None
        assert cfg.tracking_interval == 1
        assert cfg.tracking_channel == "BF"
        assert cfg.preprocessing is None
        assert cfg.shift_log_path is None

    def test_full_config(self):
        cfg = DynaTrackConfig(
            shift={
                "dampening": (0.5, 0.8, 0.8),
                "limits": {"z": (0.5, 2.0), "y": (2.0, 10.0), "x": (2.0, 10.0)},
            },
            tracking_interval=2,
            input_channel="BF",
            tracking_channel="nuclei",
            preprocessing=["phase"],
            phase={"wavelength": 0.450},
        )
        assert cfg.shift.dampening == (0.5, 0.8, 0.8)
        assert cfg.tracking_interval == 2
        assert cfg.tracking_channel == "nuclei"

    def test_config_from_dict(self):
        """Config can be constructed from a metadata dict via **kwargs."""
        meta = {
            "shift": {"dampening": (0.5, 0.8, 0.8)},
            "input_channel": "BF",
            "tracking_channel": "BF",
        }
        cfg = DynaTrackConfig(**meta)
        assert cfg.input_channel == "BF"
        assert cfg.shift.dampening == (0.5, 0.8, 0.8)

    def test_rejects_unknown_key(self):
        """As a pydantic model, unknown keys (e.g. typos) are rejected."""
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            DynaTrackConfig(input_channel="BF", tracking_channel="BF", scale_yx=0.5)

    def test_requires_input_and_tracking_channel(self):
        """input_channel / tracking_channel are required."""
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            DynaTrackConfig(input_channel="BF")  # missing tracking_channel
        with pytest.raises(pydantic.ValidationError):
            DynaTrackConfig(tracking_channel="BF")  # missing input_channel

    def test_coerces_shift_limits_lists_to_tuples(self):
        """YAML lists for shift.limits are coerced to tuples by pydantic."""
        cfg = DynaTrackConfig(
            input_channel="BF",
            tracking_channel="BF",
            shift={"limits": {"z": [0.5, 2.0]}},
        )
        assert cfg.shift.limits["z"] == (0.5, 2.0)


# ---------------------------------------------------------------------------
# DynaTrackUpdater._compute_shift tests
# ---------------------------------------------------------------------------


class TestComputeShift:
    def _make_updater(self, **kwargs):
        # scale_yx/scale_z are derived at runtime and passed to the updater,
        # not config fields.
        scale_yx = kwargs.pop("scale_yx", 0.5)
        scale_z = kwargs.pop("scale_z", 2.0)
        defaults = {"input_channel": "BF", "tracking_channel": "BF"}
        defaults.update(kwargs)
        return DynaTrackUpdater(
            config=DynaTrackConfig(**defaults), scale_yx=scale_yx, scale_z=scale_z
        )

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
        updater = self._make_updater(
            scale_yx=scale_yx, scale_z=scale_z, shift={"dampening": dampening}
        )

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
            scale_yx=scale_yx, scale_z=scale_z, shift={"limits": shift_limits}
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
        # scale_yx/scale_z are derived at runtime and passed to the updater,
        # not config fields.
        scale_yx = kwargs.pop("scale_yx", 0.5)
        scale_z = kwargs.pop("scale_z", 2.0)
        defaults = {"input_channel": "BF", "tracking_channel": "BF"}
        defaults.update(kwargs)
        return DynaTrackUpdater(
            config=DynaTrackConfig(**defaults), scale_yx=scale_yx, scale_z=scale_z
        )

    def test_first_call_stores_reference(self):
        """First call stores the reference and returns position unchanged."""
        updater = self._make_updater()
        pos = PositionCoordinates(x=100.0, y=200.0, z=50.0)
        data = [np.random.default_rng(42).random((64, 64)) for _ in range(8)]

        result = updater.update(0, 0, pos, data)

        assert result.x == 100.0
        assert result.y == 200.0
        assert result.z == 50.0
        assert 0 in updater._reference_stacks_zyx

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

        # Correction is negative feedback: the stage moves OPPOSITE the
        # measured image shift (updated = baseline - shift).
        expected_x = 100.0 - dx * scale_yx
        expected_y = 200.0 - dy * scale_yx
        assert result.x == pytest.approx(expected_x, abs=1e-6)
        assert result.y == pytest.approx(expected_y, abs=1e-6)

    def test_reference_reanchor_skips_correction_and_updates_ref(self):
        """With reference_update_interval=N, every Nth timepoint re-anchors the
        reference to the current stack and applies NO correction."""
        rng = np.random.default_rng(7)
        updater = self._make_updater(reference_update_interval=2)
        pos = PositionCoordinates(x=100.0, y=200.0, z=50.0)
        ref = [rng.random((64, 64)) for _ in range(8)]
        mov = [np.roll(f, 3, axis=1) for f in ref]  # shifted -> would correct

        updater.update(0, 0, pos, ref)  # t=0: store reference
        r1 = updater.update(1, 0, pos, mov)  # t=1: shifted -> correction applied
        assert abs(r1.x - pos.x) > 1.0, "t=1 should produce a correction"

        ref_before = updater._reference_stacks_zyx[0]
        r2 = updater.update(2, 0, pos, mov)  # t=2: re-anchor timepoint
        # No correction applied on the re-anchor timepoint.
        assert (r2.x, r2.y, r2.z) == (pos.x, pos.y, pos.z)
        # Reference replaced with the current (t=2) stack.
        assert updater._reference_stacks_zyx[0] is not ref_before

        # t=3 now compares against the new reference (same content) -> ~zero shift.
        r3 = updater.update(3, 0, pos, mov)
        assert r3.x == pytest.approx(pos.x, abs=1e-6)
        assert r3.y == pytest.approx(pos.y, abs=1e-6)

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
        config = DynaTrackConfig(
            input_channel="BF",
            tracking_channel="BF",
            tracking_interval=3,
        )
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
        updater = DynaTrackUpdater(
            config=DynaTrackConfig(input_channel="BF", tracking_channel="BF")
        )
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
        config = DynaTrackConfig(input_channel="BF", tracking_channel="BF")

        call_count = [0]

        def identity_preprocessor(stack: np.ndarray) -> dict[str, torch.Tensor]:
            call_count[0] += 1
            return {"deskewed": torch.as_tensor(stack)}

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
        config = DynaTrackConfig(input_channel="BF", tracking_channel="BF")

        # Preprocessor that rolls the stack by 2 pixels in Y
        first_call = [True]

        def shifting_preprocessor(stack: np.ndarray) -> dict[str, torch.Tensor]:
            if first_call[0]:
                first_call[0] = False
                return {"deskewed": torch.as_tensor(stack)}
            return {"deskewed": torch.as_tensor(np.roll(stack, 2, axis=1))}

        updater = DynaTrackUpdater(config=config, preprocessor=shifting_preprocessor)
        pos = PositionCoordinates(x=0.0, y=0.0, z=0.0)
        frames = [rng.random((64, 64)) for _ in range(8)]

        updater.update(0, 0, pos, frames)
        result = updater.update(1, 0, pos, frames)

        # The preprocessor-introduced shift should be detected and corrected
        # with negative feedback (stage moves opposite the +2 px image shift).
        assert result.y == pytest.approx(-2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Shift logging tests
# ---------------------------------------------------------------------------


class TestShiftLogging:
    def test_shift_log_created_on_first_write(self, tmp_path):
        """CSV file is created with header on first shift computation."""
        log_path = tmp_path / "shifts.csv"
        config = DynaTrackConfig(
            input_channel="BF",
            tracking_channel="BF",
            shift_log_path=str(log_path),
        )
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
        config = DynaTrackConfig(
            input_channel="BF",
            tracking_channel="BF",
            shift_log_path=str(log_path),
        )
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
        config = DynaTrackConfig(input_channel="BF", tracking_channel="BF")
        updater = DynaTrackUpdater(config=config)
        pos = PositionCoordinates(x=100.0, y=200.0, z=50.0)

        rng = np.random.default_rng(42)
        frames = [rng.random((64, 64)) for _ in range(8)]
        mov = [np.roll(f, 2, axis=0) for f in frames]

        updater.update(0, 0, pos, frames)
        updater.update(1, 0, pos, mov)
        # No exception, no file created — just verifying no crash


# ---------------------------------------------------------------------------
# Multi-Otsu tracking method tests (ported from 230_dynatrack_center_of_mass)
# ---------------------------------------------------------------------------


class TestGaussianBlur3D:
    def test_output_shape_unchanged(self):
        """Blur should preserve the input shape."""
        vol = torch.rand(8, 32, 32)
        result = _gaussian_blur_3d(vol, sigma=2.0)
        assert result.shape == vol.shape

    def test_smooths_values(self):
        """Blurred volume should have a smaller range than the original."""
        vol = torch.rand(8, 32, 32)
        result = _gaussian_blur_3d(vol, sigma=3.0)
        assert (result.max() - result.min()) <= (vol.max() - vol.min())

    def test_zero_sigma_noop(self):
        """Sigma=0 should return the input unchanged."""
        vol = torch.rand(8, 32, 32)
        result = _gaussian_blur_3d(vol, sigma=0.0)
        assert torch.equal(result, vol)


class TestBinaryMask:
    def test_returns_bool_tensor(self):
        """Binary mask should return a boolean tensor on the same device."""
        rng = np.random.default_rng(42)
        vol = rng.random((8, 32, 32)).astype(np.float64) * 0.2
        vol[3:6, 12:20, 12:20] = 0.9
        vol_t = torch.as_tensor(vol, dtype=torch.float32)
        mask = _binary_mask(vol_t, sigma=1.0, otsu_component=0)
        assert mask.dtype == torch.bool
        assert mask.sum() > 0  # at least some True voxels

    def test_otsu_component_selects_threshold(self):
        """Higher otsu_component should produce a stricter (smaller) mask."""
        rng = np.random.default_rng(42)
        vol = rng.random((8, 32, 32)).astype(np.float64) * 0.3
        vol[2:6, 8:24, 8:24] = 0.6
        vol[3:5, 12:20, 12:20] = 0.95
        vol_t = torch.as_tensor(vol, dtype=torch.float32)
        mask_0 = _binary_mask(vol_t, sigma=1.0, otsu_component=0)
        mask_1 = _binary_mask(vol_t, sigma=1.0, otsu_component=1)
        assert mask_0.sum() >= mask_1.sum()


class TestCenterOfMass:
    def test_single_blob(self):
        """Center of mass of a centred blob should be near the middle."""
        mask = torch.zeros(10, 10, 10, dtype=torch.bool)
        mask[3:7, 3:7, 3:7] = True
        center = _center_of_mass(mask)
        expected = torch.tensor([4.5, 4.5, 4.5])
        assert torch.allclose(center, expected, atol=0.5)

    def test_empty_returns_zeros(self):
        """Empty mask should return zero center."""
        mask = torch.zeros(10, 10, 10, dtype=torch.bool)
        center = _center_of_mass(mask)
        assert torch.equal(center, torch.zeros(3))


class TestIntensityCenterOfMass:
    def test_uniform_blob_matches_geometric_center(self):
        """A uniform-intensity region centres at its geometric middle."""
        img = torch.zeros(10, 10, 10)
        img[3:7, 3:7, 3:7] = 1.0
        center = _intensity_center_of_mass(img)
        expected = torch.tensor([4.5, 4.5, 4.5])
        assert torch.allclose(center, expected, atol=0.1)

    def test_brighter_voxels_pull_center(self):
        """Intensity weighting pulls the centroid toward the brightest voxels."""
        img = torch.zeros(10, 10, 10)
        img[3:7, 3:7, 3:7] = 1.0
        img[3:7, 3:7, 6] = 100.0  # one X slice much brighter
        center = _intensity_center_of_mass(img)
        assert center[2] > 4.5  # X pulled toward the bright slice

    def test_zero_weight_returns_geometric_center(self):
        """All-zero weights have an undefined centroid; fall back to the
        geometric centre (not the origin) so a ROI-centre shift is zero rather
        than a spurious half-volume jump toward the corner."""
        img = torch.zeros(10, 10, 10)
        center = _intensity_center_of_mass(img)
        assert torch.allclose(center, torch.tensor([4.5, 4.5, 4.5]))

    def test_negative_values_clamped(self):
        """Negative values are clamped so they don't pull the centroid."""
        img = torch.zeros(10, 10, 10)
        img[3:7, 3:7, 3:7] = 1.0
        img[0, 0, 0] = -1000.0  # would dominate if not clamped
        center = _intensity_center_of_mass(img)
        expected = torch.tensor([4.5, 4.5, 4.5])
        assert torch.allclose(center, expected, atol=0.1)

    def test_background_subtraction_sharpens_centroid(self):
        """Subtracting a background floor removes a uniform pedestal's pull."""
        # Bright blob off-centre on top of a uniform background pedestal.
        img = torch.full((8, 64, 64), 0.2)
        img[2:6, 44:52, 28:36] += 5.0  # off-centre structure
        # Without background subtraction the pedestal pulls the centroid back
        # toward the geometric centre (31.5 in Y).
        c_raw = _intensity_center_of_mass(img)
        # Subtracting the pedestal lets the structure dominate -> centroid moves
        # further toward the structure (higher Y).
        c_sub = _intensity_center_of_mass(img, background=0.2)
        assert c_sub[1] > c_raw[1]


class TestPercentile:
    def test_matches_known_distribution(self):
        """Histogram percentile is close to the true value for a ramp."""
        img = torch.arange(1000, dtype=torch.float32).reshape(10, 10, 10)
        p50 = _percentile(img, 50.0)
        # Median of 0..999 is ~499.5; histogram estimate within one bin width.
        assert abs(p50 - 499.5) < 1000 / 256 + 1

    def test_constant_image_returns_value(self):
        """A flat image returns its single value for any percentile."""
        img = torch.full((4, 8, 8), 3.0)
        assert _percentile(img, 90.0) == pytest.approx(3.0)


class TestIntensityCenterOfMassToRoiCenter:
    def test_centered_structure_zero_shift(self):
        """A structure at the ROI centre yields ~zero shift."""
        img = torch.zeros(8, 64, 64)
        img[2:6, 30:34, 30:34] = 1.0  # centred near (3.5, 31.5, 31.5)
        shift = _intensity_center_of_mass_to_roi_center(img)
        # ROI centre is ((8-1)/2, (64-1)/2, (64-1)/2) = (3.5, 31.5, 31.5)
        assert all(abs(s) < 1.0 for s in shift)

    def test_offset_structure_positive_shift(self):
        """A structure past the centre in +Y yields a positive Y shift."""
        img = torch.zeros(8, 64, 64)
        img[2:6, 40:50, 28:36] = 1.0  # shifted toward higher Y
        shift = _intensity_center_of_mass_to_roi_center(img)
        assert shift[1] > 0  # Y offset from centre is positive

    def test_blank_volume_zero_shift(self):
        """A blank volume (no signal) yields zero shift, not a half-volume jump.

        With the degenerate centroid reported as the origin, the shift would be
        ``-roi_center`` (~half the volume on every axis), commanding a large
        spurious stage move. It must be ~zero instead.
        """
        img = torch.zeros(8, 64, 64)
        shift = _intensity_center_of_mass_to_roi_center(img)
        assert all(abs(s) < 1e-3 for s in shift)

    def test_over_subtracted_volume_zero_shift(self):
        """A uniform pedestal fully removed by background subtraction also has
        no positive mass, so it must yield zero shift rather than a jump."""
        img = torch.full((8, 64, 64), 3.0)
        shift = _intensity_center_of_mass_to_roi_center(img, background_percentile=99.0)
        assert all(abs(s) < 1e-3 for s in shift)
        assert abs(shift[2]) < 2.0  # X stays near centre


class TestCenteredGaussianBlob:
    def test_peak_at_center(self):
        """The blob's maximum sits at the geometric centre of the volume."""
        blob = _centered_gaussian_blob((8, 32, 32), sigma=4.0, device=torch.device("cpu"))
        peak = np.unravel_index(int(torch.argmax(blob)), tuple(blob.shape))
        # Centre for odd/even sizes rounds to floor((n-1)/2) at the peak voxel
        assert peak[0] in (3, 4)
        assert peak[1] in (15, 16)
        assert peak[2] in (15, 16)


class TestRoiCenterPcc:
    def test_centered_structure_zero_shift(self):
        """A blob-like structure at the centre correlates with ~zero shift."""
        img = torch.zeros(8, 64, 64)
        img[2:6, 28:36, 28:36] = 1.0  # near the ROI centre
        shift = _roi_center_pcc(img, blob_sigma=4.0)
        assert abs(shift[1]) <= 2
        assert abs(shift[2]) <= 2

    def test_offset_structure_detected(self):
        """An off-centre bright blob yields a non-zero detected offset in Y."""
        img = torch.zeros(8, 64, 64)
        img[2:6, 44:52, 28:36] = 1.0  # shifted +~16 in Y from centre
        shift = _roi_center_pcc(img, blob_sigma=4.0)
        assert shift[1] > 4  # detected positive Y offset from centre


class TestMultiotsuCenterOfMass:
    def test_detects_shift(self):
        """Center of mass should detect a spatial shift between two volumes."""
        rng = np.random.default_rng(42)
        ref = rng.random((8, 64, 64)).astype(np.float64) * 0.1
        ref[2:6, 20:40, 20:40] = 0.9  # bright blob

        # Shift the blob by +5 in Y
        mov = rng.random((8, 64, 64)).astype(np.float64) * 0.1
        mov[2:6, 25:45, 20:40] = 0.9

        ref_t = torch.as_tensor(ref, dtype=torch.float32)
        mov_t = torch.as_tensor(mov, dtype=torch.float32)
        shift = _multiotsu_center_of_mass(ref_t, mov_t, sigma=1.0, otsu_component=0)

        # Y shift should be approximately +5
        assert abs(shift[1] - 5.0) < 2.0
        # X and Z should be near zero
        assert abs(shift[2]) < 2.0
        assert abs(shift[0]) < 2.0


class TestMultiotsuPcc:
    def test_detects_shift(self):
        """Multiotsu PCC should detect a known shift via binary mask PCC."""
        rng = np.random.default_rng(42)
        ref = rng.random((8, 64, 64)).astype(np.float64) * 0.1
        ref[2:6, 15:50, 15:50] = 0.9

        mov = rng.random((8, 64, 64)).astype(np.float64) * 0.1
        mov[2:6, 18:53, 15:50] = 0.9  # shifted +3 in Y

        ref_t = torch.as_tensor(ref, dtype=torch.float32)
        mov_t = torch.as_tensor(mov, dtype=torch.float32)
        shift = _multiotsu_pcc(ref_t, mov_t, sigma=1.0, otsu_component=0)

        assert abs(shift[1] - 3) <= 1  # Y shift ~3
        assert abs(shift[2]) <= 1  # X near zero


class TestDynaTrackUpdaterMultiotsu:
    def _make_blob_frames(self, rng, y_start, n_z=8, ny=64, nx=64):
        """Create frames with a bright blob at a given Y position."""
        frames = []
        for z in range(n_z):
            frame = rng.random((ny, nx)).astype(np.float64) * 0.1
            if 2 <= z < 6:
                frame[y_start : y_start + 20, 20:40] = 0.9
            frames.append(frame)
        return frames

    def test_multiotsu_center_of_mass_detects_shift(self):
        """update() with multiotsu_center_of_mass detects a blob shift."""
        rng = np.random.default_rng(42)
        config = DynaTrackConfig(
            input_channel="BF",
            tracking_channel="BF",
            tracking_method="multiotsu_center_of_mass",
            segmentation={"otsu_sigma": 1.0},
            preprocessing=[],
        )
        updater = DynaTrackUpdater(config=config)
        pos = PositionCoordinates(x=100.0, y=200.0, z=50.0)

        ref_frames = self._make_blob_frames(rng, y_start=15)
        mov_frames = self._make_blob_frames(rng, y_start=20)  # shifted +5 in Y

        updater.update(0, 0, pos, ref_frames)
        result = updater.update(1, 0, pos, mov_frames)

        # Y position should have changed
        assert result.y != 200.0

    def test_multiotsu_pcc_detects_shift(self):
        """update() with multiotsu_pcc detects a blob shift."""
        rng = np.random.default_rng(42)
        config = DynaTrackConfig(
            input_channel="BF",
            tracking_channel="BF",
            tracking_method="multiotsu_pcc",
            segmentation={"otsu_sigma": 1.0},
            preprocessing=[],
        )
        updater = DynaTrackUpdater(config=config)
        pos = PositionCoordinates(x=100.0, y=200.0, z=50.0)

        ref_frames = self._make_blob_frames(rng, y_start=15)
        mov_frames = self._make_blob_frames(rng, y_start=20)

        updater.update(0, 0, pos, ref_frames)
        result = updater.update(1, 0, pos, mov_frames)

        assert result.y != 200.0

    def test_invalid_method_raises(self):
        """Unknown tracking_method should raise ValueError."""
        rng = np.random.default_rng(42)
        config = DynaTrackConfig(
            input_channel="BF",
            tracking_channel="BF",
            tracking_method="unknown_method",
            preprocessing=[],
        )
        updater = DynaTrackUpdater(config=config)
        pos = PositionCoordinates(x=100.0, y=200.0, z=50.0)

        frames = [rng.random((64, 64)) for _ in range(8)]
        updater.update(0, 0, pos, frames)  # store ref
        with pytest.raises(ValueError, match="Unknown tracking_method"):
            updater.update(1, 0, pos, frames)


# ---------------------------------------------------------------------------
# Referenceless ROI-centre tracking flow (intensity_center_of_mass, roi_center_pcc)
# ---------------------------------------------------------------------------


class TestRoiCenterMethodsFlow:
    def _make_offset_blob_frames(self, rng, y_start, n_z=8, ny=64, nx=64):
        """Frames with a bright blob whose Y position is offset from centre."""
        frames = []
        for z in range(n_z):
            frame = rng.random((ny, nx)).astype(np.float64) * 0.1
            if 2 <= z < 6:
                frame[y_start : y_start + 8, 28:36] = 0.9
            frames.append(frame)
        return frames

    def test_intensity_center_of_mass_corrects_from_t0(self):
        """Referenceless: the first timepoint already applies a correction
        (no reference stored, no unchanged-return)."""
        rng = np.random.default_rng(42)
        config = DynaTrackConfig(
            input_channel="BF",
            tracking_channel="BF",
            tracking_method="intensity_center_of_mass",
            preprocessing=[],
        )
        updater = DynaTrackUpdater(config=config)
        pos = PositionCoordinates(x=100.0, y=200.0, z=50.0)

        # Blob offset toward higher Y -> positive Y image shift -> stage moves -Y.
        frames = self._make_offset_blob_frames(rng, y_start=44)
        result = updater.update(0, 0, pos, frames)

        assert result.y != 200.0  # corrected on the very first timepoint
        assert 0 not in updater._reference_stacks_zyx  # no reference stored

    def test_roi_center_pcc_corrects_from_t0(self):
        """Referenceless PCC-vs-ROI-centre also corrects from t=0."""
        rng = np.random.default_rng(42)
        config = DynaTrackConfig(
            input_channel="BF",
            tracking_channel="BF",
            tracking_method="roi_center_pcc",
            roi_center={"blob_sigma": 4.0},
            preprocessing=[],
        )
        updater = DynaTrackUpdater(config=config)
        pos = PositionCoordinates(x=100.0, y=200.0, z=50.0)

        frames = self._make_offset_blob_frames(rng, y_start=44)
        result = updater.update(0, 0, pos, frames)

        assert result.y != 200.0
        assert 0 not in updater._reference_stacks_zyx


# ---------------------------------------------------------------------------
# wants_reference_refresh: gates the manager's missing-baseline behaviour
# ---------------------------------------------------------------------------


class TestWantsReferenceRefresh:
    def test_reference_based_refreshes_on_interval(self):
        """Reference-based methods re-anchor every Nth timepoint."""
        updater = DynaTrackUpdater(
            config=DynaTrackConfig(
                input_channel="BF",
                tracking_channel="BF",
                tracking_method="pcc",
                reference_update_interval=4,
            )
        )
        assert [updater.wants_reference_refresh(t) for t in (0, 1, 4, 8)] == [
            True,
            False,
            True,
            True,
        ]

    def test_no_interval_never_refreshes(self):
        updater = DynaTrackUpdater(
            config=DynaTrackConfig(
                input_channel="BF",
                tracking_channel="BF",
                tracking_method="pcc",
                reference_update_interval=0,
            )
        )
        assert not any(updater.wants_reference_refresh(t) for t in range(5))

    def test_referenceless_never_refreshes(self):
        """Referenceless methods correct every timepoint and keep no reference,
        so they must never report a refresh -- even with an interval set --
        otherwise a missing baseline would let a real correction run against an
        unanchored (race-prone) store value."""
        for method in ("intensity_center_of_mass", "roi_center_pcc"):
            updater = DynaTrackUpdater(
                config=DynaTrackConfig(
                    input_channel="BF",
                    tracking_channel="BF",
                    tracking_method=method,
                    reference_update_interval=4,
                )
            )
            assert not any(updater.wants_reference_refresh(t) for t in range(9))


class TestReferenceUpdateIntervalWarning:
    def test_warns_for_referenceless_with_interval(self, caplog):
        with caplog.at_level("WARNING", logger="shrimpy.dynatrack.tracking"):
            DynaTrackUpdater(
                config=DynaTrackConfig(
                    input_channel="BF",
                    tracking_channel="BF",
                    tracking_method="intensity_center_of_mass",
                    reference_update_interval=4,
                )
            )
        assert any("ignored for referenceless" in r.message for r in caplog.records)

    def test_no_warning_for_reference_based(self, caplog):
        with caplog.at_level("WARNING", logger="shrimpy.dynatrack.tracking"):
            DynaTrackUpdater(
                config=DynaTrackConfig(
                    input_channel="BF",
                    tracking_channel="BF",
                    tracking_method="pcc",
                    reference_update_interval=4,
                )
            )
        assert not any("ignored for referenceless" in r.message for r in caplog.records)

    def test_no_warning_for_referenceless_without_interval(self, caplog):
        with caplog.at_level("WARNING", logger="shrimpy.dynatrack.tracking"):
            DynaTrackUpdater(
                config=DynaTrackConfig(
                    input_channel="BF",
                    tracking_channel="BF",
                    tracking_method="intensity_center_of_mass",
                    reference_update_interval=0,
                )
            )
        assert not any("ignored for referenceless" in r.message for r in caplog.records)


class TestWorkerPreprocessorWiring:
    """The worker's call into the shared preprocessor.

    ``shrimpy.preprocessing.build_preprocessor`` is shared with FOV selection and
    so takes the reconstruction settings as explicit arguments rather than a
    ``DynaTrackConfig``. That makes the config -> kwargs mapping a real seam: a
    field wired to the wrong argument would leave DynaTrack tracking on a
    silently mis-reconstructed volume, with nothing else to catch it.
    """

    def _build_kwargs(self, cfg, monkeypatch):
        """Run the worker's preprocessor-construction step, capturing the kwargs."""
        import shrimpy.preprocessing as pp

        captured = {}

        def _fake_build(**kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(pp, "build_preprocessor", _fake_build)

        # Mirror the construction in shrimpy.dynatrack.worker._worker_loop. Calling
        # the loop itself would need a live subprocess and queues; this keeps the
        # assertion on the mapping, which is what the refactor changed.
        from shrimpy.preprocessing import build_preprocessor

        build_preprocessor(
            zyx_shape=(16, 64, 64),
            preprocessing=cfg.preprocessing,
            deskew=cfg.deskew,
            phase=cfg.phase,
            virtual_staining=cfg.virtual_staining,
            output_channel=cfg.tracking_channel,
        )
        return captured

    def test_config_fields_map_to_the_right_arguments(self, monkeypatch):
        cfg = DynaTrackConfig(
            input_channel="BF",
            tracking_channel="nuclei",
            preprocessing=["deskew", "phase", "vs"],
            deskew={"pixel_size_um": 0.116, "scan_step_um": 0.31, "ls_angle_deg": 30},
            phase={"transfer_function": {"wavelength_illumination": 0.45}},
            virtual_staining={"model": {"init_args": {}}},
        )
        kwargs = self._build_kwargs(cfg, monkeypatch)

        assert kwargs["preprocessing"] == ["deskew", "phase", "vs"]
        assert kwargs["deskew"] == cfg.deskew
        assert kwargs["phase"] == cfg.phase
        assert kwargs["virtual_staining"] == cfg.virtual_staining
        assert kwargs["zyx_shape"] == (16, 64, 64)
        # output_channel comes from tracking_channel, NOT input_channel: for a VS
        # pipeline it names the VS target the updater tracks.
        assert kwargs["output_channel"] == "nuclei"

    def test_non_vs_pipeline_keys_output_on_the_tracking_channel(self, monkeypatch):
        # Deskew-only tracking: the deskewed volume is emitted under the input
        # channel's name, which tracking_channel also names.
        cfg = DynaTrackConfig(
            input_channel="BF", tracking_channel="BF", preprocessing=["deskew"]
        )
        kwargs = self._build_kwargs(cfg, monkeypatch)
        assert kwargs["output_channel"] == "BF"
        assert kwargs["virtual_staining"] is None

    def test_updater_selects_its_channel_from_the_preprocessor_dict(self):
        """End of the contract: __call__ returns a dict the updater indexes by name."""
        import numpy as np

        cfg = DynaTrackConfig(
            input_channel="BF", tracking_channel="nuclei", preprocessing=["vs"]
        )
        emitted = {"phase": np.zeros((4, 8, 8)), "nuclei": np.ones((4, 8, 8))}
        updater = DynaTrackUpdater(config=cfg, preprocessor=lambda stack: emitted)

        channels = updater._preprocessor(np.zeros((4, 8, 8)))
        assert cfg.tracking_channel in channels
        assert channels[cfg.tracking_channel].mean() == 1.0
