"""Unit tests for the mantis-specific parts of MantisEngine.

Behavior shared with the other microscope engines (autofocus dispatch, demo
PFS, event skipping, hardware reset, ...) is tested in ``test_base_engine.py``.
Tests here use a mock CMMCorePlus to isolate MantisEngine from real hardware
and the parent MDAEngine.
"""

from __future__ import annotations

import weakref

from unittest.mock import MagicMock, call, patch

import pytest

from useq import MDAEvent

from shrimpy.engines.base_engine import BaseEngine
from shrimpy.engines.mantis_engine import (
    FAST_XY_STAGE_SPEED,
    MANTIS_XY_STAGE_NAME,
    SLOW_XY_STAGE_SPEED,
    MantisEngine,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine(mock_core: MagicMock) -> MantisEngine:
    """Create a MantisEngine wired to the mock CMMCorePlus.

    Patches the parent MDAEngine.__init__ so we don't need a real core for
    the super().__init__() call, then manually sets mmcore.
    """
    with patch("shrimpy.engines.base_engine.MDAEngine.__init__", return_value=None):
        eng = MantisEngine(mock_core)
    # Manually assign the core weakref since we bypassed super().__init__
    eng._mmcore_ref = weakref.ref(mock_core)
    return eng


# ---------------------------------------------------------------------------
# MantisEngine.__init__()
# ---------------------------------------------------------------------------


def test_engine_derives_from_base_engine():
    assert issubclass(MantisEngine, BaseEngine)


def test_init_default_attributes(engine):
    # Mantis-specific state starts unset
    assert engine._xy_stage_speed is None


def test_init_acquisition_timeouts(mock_core):
    # Timeouts guard against stalling on dropped frames / missed triggers
    engine = MantisEngine(mock_core)
    assert engine.timeout_base == 10.0
    assert engine.timeout_multiplier == 1.0
    assert engine.timeout_first_frame is None
    assert engine.timeout_action == "warn"
    # ... on top of the shared hardware sequencing defaults
    assert engine.use_hardware_sequencing is True
    assert engine.force_set_xy_position is False


def test_init_timeout_kwargs_override_defaults(mock_core):
    engine = MantisEngine(mock_core, timeout_base=1.0, timeout_action="raise")
    assert engine.timeout_base == 1.0
    assert engine.timeout_action == "raise"


# ---------------------------------------------------------------------------
# _adjust_xy_stage_speed()
# ---------------------------------------------------------------------------


def test_speed_returns_early_when_autofocus_disabled(engine, mock_core):
    # No autofocus → no speed adjustment
    engine._use_autofocus = False
    engine._xy_stage_device = MANTIS_XY_STAGE_NAME
    engine._adjust_xy_stage_speed(MDAEvent(x_pos=100, y_pos=100))
    mock_core.setProperty.assert_not_called()


def test_speed_returns_early_for_non_mantis_stage(engine, mock_core):
    # Non-Mantis XY stage → skip speed adjustment
    engine._use_autofocus = True
    engine._xy_stage_device = "DemoXYStage"
    engine._adjust_xy_stage_speed(MDAEvent(x_pos=100, y_pos=100))
    mock_core.setProperty.assert_not_called()


def test_speed_returns_early_when_no_last_position(engine, mock_core):
    # No previous position recorded → can't compute distance
    engine._use_autofocus = True
    engine._xy_stage_device = MANTIS_XY_STAGE_NAME
    mock_core._last_xy_position = {None: (None, None)}
    engine._adjust_xy_stage_speed(MDAEvent(x_pos=100, y_pos=100))
    mock_core.setProperty.assert_not_called()


def test_speed_short_move_sets_slow_speed(engine, mock_core):
    # Move of 500 µm (< SHORT_DISTANCE threshold) → SLOW_XY_STAGE_SPEED
    engine._use_autofocus = True
    engine._xy_stage_device = MANTIS_XY_STAGE_NAME
    mock_core._last_xy_position = {None: (0.0, 0.0)}
    engine._adjust_xy_stage_speed(MDAEvent(x_pos=500.0, y_pos=0.0))
    mock_core.setProperty.assert_any_call(
        MANTIS_XY_STAGE_NAME, "MotorSpeedX-S(mm/s)", SLOW_XY_STAGE_SPEED
    )
    mock_core.setProperty.assert_any_call(
        MANTIS_XY_STAGE_NAME, "MotorSpeedY-S(mm/s)", SLOW_XY_STAGE_SPEED
    )


def test_speed_long_move_sets_fast_speed(engine, mock_core):
    # Move of 3000 µm (≥ SHORT_DISTANCE threshold) → FAST_XY_STAGE_SPEED
    engine._use_autofocus = True
    engine._xy_stage_device = MANTIS_XY_STAGE_NAME
    mock_core._last_xy_position = {None: (0.0, 0.0)}
    engine._adjust_xy_stage_speed(MDAEvent(x_pos=3000.0, y_pos=0.0))
    mock_core.setProperty.assert_any_call(
        MANTIS_XY_STAGE_NAME, "MotorSpeedX-S(mm/s)", FAST_XY_STAGE_SPEED
    )
    mock_core.setProperty.assert_any_call(
        MANTIS_XY_STAGE_NAME, "MotorSpeedY-S(mm/s)", FAST_XY_STAGE_SPEED
    )


def test_speed_negligible_move_skips_adjustment(engine, mock_core):
    # Move < 1 µm → no speed change
    engine._use_autofocus = True
    engine._xy_stage_device = MANTIS_XY_STAGE_NAME
    mock_core._last_xy_position = {None: (0.0, 0.0)}
    engine._adjust_xy_stage_speed(MDAEvent(x_pos=0.5, y_pos=0.0))
    mock_core.setProperty.assert_not_called()


def test_speed_same_speed_not_set_again(engine, mock_core):
    # If cached speed matches computed speed, skip redundant setProperty calls
    engine._use_autofocus = True
    engine._xy_stage_device = MANTIS_XY_STAGE_NAME
    engine._xy_stage_speed = SLOW_XY_STAGE_SPEED  # already set to slow
    mock_core._last_xy_position = {None: (0.0, 0.0)}
    engine._adjust_xy_stage_speed(MDAEvent(x_pos=500.0, y_pos=0.0))
    mock_core.setProperty.assert_not_called()


# ---------------------------------------------------------------------------
# engage_autofocus() — Nikon PFS
# ---------------------------------------------------------------------------


def test_autofocus_nikon_pfs_dispatched(engine, mock_core):
    # Non-demo method → calls _engage_nikon_pfs with stage name and z position
    engine._use_autofocus = True
    engine._autofocus_method = "PFS"
    engine._autofocus_stage = "ZDrive"
    mock_core.getPosition.return_value = 42.0
    with patch.object(engine, "_engage_nikon_pfs", return_value=True) as mock_pfs:
        engine._engage_autofocus(MDAEvent())
    mock_pfs.assert_called_once_with("ZDrive", 42.0)
    assert engine._autofocus_success is True


def test_autofocus_failure_recorded(engine, mock_core):
    # A failed PFS engagement is recorded so setup_event can skip the event
    engine._use_autofocus = True
    engine._autofocus_method = "PFS"
    engine._autofocus_stage = "ZDrive"
    with patch.object(engine, "_engage_nikon_pfs", return_value=False):
        engine._engage_autofocus(MDAEvent())
    assert engine._autofocus_success is False


# ---------------------------------------------------------------------------
# _engage_nikon_pfs()
# ---------------------------------------------------------------------------


def test_pfs_already_locked_after_fullfocus(engine, mock_core):
    # fullFocus succeeds and focus is locked → immediate success
    mock_core.isContinuousFocusLocked.return_value = True

    with patch("shrimpy.engines.mantis_engine.time.sleep"):
        assert engine._engage_nikon_pfs("ZDrive", 100.0) is True

    mock_core.fullFocus.assert_called_once()
    # Should not enter the z_offset retry loop
    mock_core.enableContinuousFocus.assert_not_called()


def test_pfs_locks_on_first_z_offset(engine, mock_core):
    # fullFocus fails, but first z_offset (0) succeeds
    mock_core.isContinuousFocusLocked.side_effect = [False, True]

    with patch("shrimpy.engines.mantis_engine.time.sleep"):
        assert engine._engage_nikon_pfs("ZDrive", 100.0) is True

    # Should have set position to 100 + 0 = 100
    mock_core.setPosition.assert_called_with("ZDrive", 100.0)


def test_pfs_locks_on_later_z_offset(engine, mock_core):
    # fullFocus fails, first two offsets fail, third (offset=10) succeeds
    mock_core.isContinuousFocusLocked.side_effect = [False, False, False, True]

    with patch("shrimpy.engines.mantis_engine.time.sleep"):
        assert engine._engage_nikon_pfs("ZDrive", 100.0) is True

    # Offsets are [0, -10, 10, ...]; third is 10 → position = 110
    assert any(c == call("ZDrive", 110.0) for c in mock_core.setPosition.call_args_list)


def test_pfs_all_offsets_fail(engine, mock_core):
    # fullFocus fails, none of the 7 z_offsets succeed
    mock_core.isContinuousFocusLocked.return_value = False

    with patch("shrimpy.engines.mantis_engine.time.sleep"):
        assert engine._engage_nikon_pfs("ZDrive", 100.0) is False

    # Z stage should be returned to the original position
    last_set_position = mock_core.setPosition.call_args_list[-1]
    assert last_set_position == call("ZDrive", 100.0)
