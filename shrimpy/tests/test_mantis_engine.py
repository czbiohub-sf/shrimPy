"""Unit tests for MantisEngine and helper functions.

Tests use a mock CMMCorePlus to isolate MantisEngine logic from real
hardware and the parent MDAEngine.
"""

from __future__ import annotations

import weakref

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from useq import MDAEvent, MDASequence

from shrimpy.mantis.mantis_engine import (
    DEFAULT_XY_STAGE_SPEED,
    DEMO_PFS_METHOD,
    MANTIS_XY_STAGE_NAME,
    MantisEngine,
    _get_next_acquisition_name,
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
    with patch("shrimpy.mantis.mantis_engine.MDAEngine.__init__", return_value=None):
        eng = MantisEngine(mock_core)
    # Manually assign the core weakref since we bypassed super().__init__
    eng._mmcore_ref = weakref.ref(mock_core)
    return eng


def _make_sequence(mantis_meta: dict | None = None) -> MDASequence:
    """Helper to create an MDASequence with optional mantis metadata."""
    metadata = {"mantis": mantis_meta} if mantis_meta else {}
    return MDASequence(metadata=metadata)


# ---------------------------------------------------------------------------
# _get_next_acquisition_name() — pure function
# ---------------------------------------------------------------------------


def test_next_name_first_acquisition_in_empty_dir(tmp_path):
    # Empty directory → index starts at 1
    assert _get_next_acquisition_name(tmp_path, "acq") == "acq_1"


def test_next_name_skips_existing_index(tmp_path):
    # acq_1.ome.zarr already exists → should return acq_2
    (tmp_path / "acq_1.ome.zarr").mkdir()
    assert _get_next_acquisition_name(tmp_path, "acq") == "acq_2"


def test_next_name_skips_multiple_existing(tmp_path):
    # acq_1 through acq_3 exist → should return acq_4
    for i in range(1, 4):
        (tmp_path / f"acq_{i}.ome.zarr").mkdir()
    assert _get_next_acquisition_name(tmp_path, "acq") == "acq_4"


def test_next_name_different_base_names_dont_collide(tmp_path):
    # "experiment_1.ome.zarr" exists, but asking for "acq" → acq_1
    (tmp_path / "experiment_1.ome.zarr").mkdir()
    assert _get_next_acquisition_name(tmp_path, "acq") == "acq_1"


def test_next_name_gap_in_indices(tmp_path):
    # acq_1 exists, acq_2 missing, acq_3 exists → returns acq_2
    (tmp_path / "acq_1.ome.zarr").mkdir()
    (tmp_path / "acq_3.ome.zarr").mkdir()
    assert _get_next_acquisition_name(tmp_path, "acq") == "acq_2"


# ---------------------------------------------------------------------------
# MantisEngine.__init__()
# ---------------------------------------------------------------------------


def test_init_default_attributes(engine):
    # All autofocus-related attributes start disabled/unset
    assert engine._use_autofocus is False
    assert engine._autofocus_success is False
    assert engine._autofocus_stage is None
    assert engine._autofocus_method is None
    assert engine._xy_stage_speed is None


def test_init_registers_engine_and_callbacks(mock_core):
    # Verify that __init__ wires up the engine and event callbacks
    with patch("shrimpy.mantis.mantis_engine.MDAEngine.__init__", return_value=None):
        MantisEngine(mock_core)

    mock_core.mda.set_engine.assert_called_once()
    mock_core.events.propertyChanged.connect.assert_called_once()
    mock_core.events.roiSet.connect.assert_called_once()
    mock_core.events.XYStagePositionChanged.connect.assert_called_once()


# ---------------------------------------------------------------------------
# setup_sequence()
# ---------------------------------------------------------------------------


def test_setup_sequence_no_mantis_metadata(engine):
    # Should not raise when metadata is empty
    seq = _make_sequence()
    with patch("shrimpy.mantis.mantis_engine.MDAEngine.setup_sequence"):
        engine.setup_sequence(seq)
    assert engine._use_autofocus is False


def test_setup_sequence_roi_applied(engine, mock_core):
    # ROI from metadata should be forwarded to core.clearROI + core.setROI
    roi = [225, 880, 1600, 256]
    seq = _make_sequence({"roi": roi})
    with patch("shrimpy.mantis.mantis_engine.MDAEngine.setup_sequence"):
        engine.setup_sequence(seq)
    mock_core.clearROI.assert_called_once()
    mock_core.setROI.assert_called_once_with(*roi)


def test_setup_sequence_initialization_settings_applied(engine, mock_core):
    # Each [device, property, value] triple should call core.setProperty
    settings = [
        ["Camera", "OnCameraCCDXSize", "2048"],
        ["Camera", "PixelType", "16bit"],
    ]
    seq = _make_sequence({"initialization_settings": settings})
    with patch("shrimpy.mantis.mantis_engine.MDAEngine.setup_sequence"):
        engine.setup_sequence(seq)

    calls = [
        call("Camera", "OnCameraCCDXSize", "2048"),
        call("Camera", "PixelType", "16bit"),
    ]
    mock_core.setProperty.assert_has_calls(calls, any_order=False)


def test_setup_sequence_z_stage_set(engine, mock_core):
    # z_stage metadata should set Core/Focus property
    seq = _make_sequence({"z_stage": "AP Galvo"})
    with patch("shrimpy.mantis.mantis_engine.MDAEngine.setup_sequence"):
        engine.setup_sequence(seq)
    mock_core.setProperty.assert_called_with("Core", "Focus", "AP Galvo")


def test_setup_sequence_autofocus_enabled(engine, mock_core):
    # Autofocus metadata with enabled=True should configure the engine
    af = {"enabled": True, "stage": "ZDrive", "method": "PFS"}
    seq = _make_sequence({"autofocus": af})
    with patch("shrimpy.mantis.mantis_engine.MDAEngine.setup_sequence"):
        engine.setup_sequence(seq)

    assert engine._use_autofocus is True
    assert engine._autofocus_stage == "ZDrive"
    assert engine._autofocus_method == "PFS"
    # Non-demo method should call setAutoFocusDevice
    mock_core.setAutoFocusDevice.assert_called_once_with("PFS")


def test_setup_sequence_autofocus_disabled(engine):
    # Autofocus explicitly disabled → _use_autofocus stays False
    af = {"enabled": False, "stage": "ZDrive", "method": "PFS"}
    seq = _make_sequence({"autofocus": af})
    with patch("shrimpy.mantis.mantis_engine.MDAEngine.setup_sequence"):
        engine.setup_sequence(seq)
    assert engine._use_autofocus is False


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
    # Move of 500 µm (< 2000 µm threshold) → slow speed 2.0 mm/s
    engine._use_autofocus = True
    engine._xy_stage_device = MANTIS_XY_STAGE_NAME
    mock_core._last_xy_position = {None: (0.0, 0.0)}
    engine._adjust_xy_stage_speed(MDAEvent(x_pos=500.0, y_pos=0.0))
    mock_core.setProperty.assert_any_call(MANTIS_XY_STAGE_NAME, "MotorSpeedX-S(mm/s)", 2.0)
    mock_core.setProperty.assert_any_call(MANTIS_XY_STAGE_NAME, "MotorSpeedY-S(mm/s)", 2.0)


def test_speed_long_move_sets_fast_speed(engine, mock_core):
    # Move of 3000 µm (≥ 2000 µm threshold) → fast speed 5.75 mm/s
    engine._use_autofocus = True
    engine._xy_stage_device = MANTIS_XY_STAGE_NAME
    mock_core._last_xy_position = {None: (0.0, 0.0)}
    engine._adjust_xy_stage_speed(MDAEvent(x_pos=3000.0, y_pos=0.0))
    mock_core.setProperty.assert_any_call(MANTIS_XY_STAGE_NAME, "MotorSpeedX-S(mm/s)", 5.75)
    mock_core.setProperty.assert_any_call(MANTIS_XY_STAGE_NAME, "MotorSpeedY-S(mm/s)", 5.75)


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
    engine._xy_stage_speed = 2.0  # already set to slow
    mock_core._last_xy_position = {None: (0.0, 0.0)}
    engine._adjust_xy_stage_speed(MDAEvent(x_pos=500.0, y_pos=0.0))
    mock_core.setProperty.assert_not_called()


# ---------------------------------------------------------------------------
# _engage_autofocus()
# ---------------------------------------------------------------------------


def test_autofocus_disabled_returns_early(engine):
    # Autofocus disabled → no method calls
    engine._use_autofocus = False
    engine._engage_autofocus(MDAEvent())
    # No crash, no side effects


def test_autofocus_demo_pfs_dispatched(engine):
    # demo-PFS method → calls _engage_demo_pfs
    engine._use_autofocus = True
    engine._autofocus_method = DEMO_PFS_METHOD
    with patch.object(engine, "_engage_demo_pfs") as mock_demo:
        engine._engage_autofocus(MDAEvent())
    mock_demo.assert_called_once_with(success_rate=0.5)


def test_autofocus_nikon_pfs_dispatched(engine, mock_core):
    # Non-demo method → calls _engage_nikon_pfs with stage name and z position
    engine._use_autofocus = True
    engine._autofocus_method = "PFS"
    engine._autofocus_stage = "ZDrive"
    mock_core.getPosition.return_value = 42.0
    with patch.object(engine, "_engage_nikon_pfs") as mock_pfs:
        engine._engage_autofocus(MDAEvent())
    mock_pfs.assert_called_once_with("ZDrive", 42.0)


# ---------------------------------------------------------------------------
# _engage_nikon_pfs()
# ---------------------------------------------------------------------------


def test_pfs_already_locked_after_fullfocus(engine, mock_core):
    # fullFocus succeeds and focus is locked → immediate success
    mock_core.isContinuousFocusLocked.return_value = True

    with patch("shrimpy.mantis.mantis_engine.time.sleep"):
        engine._engage_nikon_pfs("ZDrive", 100.0)

    assert engine._autofocus_success is True
    mock_core.fullFocus.assert_called_once()
    # Should not enter the z_offset retry loop
    mock_core.enableContinuousFocus.assert_not_called()


def test_pfs_locks_on_first_z_offset(engine, mock_core):
    # fullFocus fails, but first z_offset (0) succeeds
    mock_core.isContinuousFocusLocked.side_effect = [False, True]

    with patch("shrimpy.mantis.mantis_engine.time.sleep"):
        engine._engage_nikon_pfs("ZDrive", 100.0)

    assert engine._autofocus_success is True
    # Should have set position to 100 + 0 = 100
    mock_core.setPosition.assert_called_with("ZDrive", 100.0)


def test_pfs_locks_on_later_z_offset(engine, mock_core):
    # fullFocus fails, first two offsets fail, third (offset=10) succeeds
    mock_core.isContinuousFocusLocked.side_effect = [False, False, False, True]

    with patch("shrimpy.mantis.mantis_engine.time.sleep"):
        engine._engage_nikon_pfs("ZDrive", 100.0)

    assert engine._autofocus_success is True
    # Offsets are [0, -10, 10, ...]; third is 10 → position = 110
    assert any(c == call("ZDrive", 110.0) for c in mock_core.setPosition.call_args_list)


def test_pfs_all_offsets_fail(engine, mock_core):
    # fullFocus fails, none of the 7 z_offsets succeed
    mock_core.isContinuousFocusLocked.return_value = False

    with patch("shrimpy.mantis.mantis_engine.time.sleep"):
        engine._engage_nikon_pfs("ZDrive", 100.0)

    assert engine._autofocus_success is False
    # Z stage should be returned to the original position
    last_set_position = mock_core.setPosition.call_args_list[-1]
    assert last_set_position == call("ZDrive", 100.0)


# ---------------------------------------------------------------------------
# exec_event()
# ---------------------------------------------------------------------------


def test_exec_event_no_autofocus_delegates_to_super(engine):
    # When autofocus is off, just delegate to parent
    engine._use_autofocus = False
    event = MDAEvent()
    with patch("shrimpy.mantis.mantis_engine.MDAEngine.exec_event") as mock_super:
        mock_super.return_value = iter([("frame", event, {})])
        results = list(engine.exec_event(event))
    assert len(results) == 1
    mock_super.assert_called_once_with(event)


def test_exec_event_autofocus_success_delegates_to_super(engine):
    # Autofocus on + success → delegate to parent
    engine._use_autofocus = True
    engine._autofocus_success = True
    event = MDAEvent()
    with patch("shrimpy.mantis.mantis_engine.MDAEngine.exec_event") as mock_super:
        mock_super.return_value = iter([("frame", event, {})])
        results = list(engine.exec_event(event))
    assert len(results) == 1
    mock_super.assert_called_once_with(event)


def test_exec_event_autofocus_failure_yields_zero_padded_image(engine, mock_core):
    # Autofocus on + failure → yield zeros with correct shape and dtype
    engine._use_autofocus = True
    engine._autofocus_success = False
    mock_core.getImageHeight.return_value = 2048
    mock_core.getImageWidth.return_value = 2048
    mock_core.getImageBitDepth.return_value = 16
    # get_frame_metadata needs these to build position metadata
    mock_core.getXYPosition.return_value = (0.0, 0.0)
    mock_core.getPosition.return_value = 0.0

    event = MDAEvent()
    results = list(engine.exec_event(event))

    assert len(results) == 1
    img, evt, _meta = results[0]
    assert img.shape == (2048, 2048)
    assert img.dtype == np.uint16
    assert np.all(img == 0)
    assert evt is event


# ---------------------------------------------------------------------------
# teardown_sequence()
# ---------------------------------------------------------------------------


def test_teardown_mantis_stage_speed_reset(engine, mock_core):
    # Mantis XY stage → speed should be reset to default
    engine._xy_stage_device = MANTIS_XY_STAGE_NAME
    seq = MDASequence()
    with patch("shrimpy.mantis.mantis_engine.MDAEngine.teardown_sequence"):
        engine.teardown_sequence(seq)

    mock_core.setProperty.assert_any_call(
        MANTIS_XY_STAGE_NAME, "MotorSpeedX-S(mm/s)", DEFAULT_XY_STAGE_SPEED
    )
    mock_core.setProperty.assert_any_call(
        MANTIS_XY_STAGE_NAME, "MotorSpeedY-S(mm/s)", DEFAULT_XY_STAGE_SPEED
    )


def test_teardown_non_mantis_stage_no_speed_reset(engine, mock_core):
    # Non-Mantis XY stage → no speed reset calls
    engine._xy_stage_device = "DemoXYStage"
    seq = MDASequence()
    with patch("shrimpy.mantis.mantis_engine.MDAEngine.teardown_sequence"):
        engine.teardown_sequence(seq)
    mock_core.setProperty.assert_not_called()
