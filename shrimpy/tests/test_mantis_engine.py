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

from pymmcore_plus.mda import MDAEngine
from useq import MDAEvent, MDASequence

from shrimpy.engines.base_engine import BaseEngine
from shrimpy.engines.mantis_engine import (
    FAST_XY_STAGE_SPEED,
    MANTIS_XY_STAGE_NAME,
    SLOW_XY_STAGE_SPEED,
    MantisEngine,
    _format_duration,
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
    with patch("shrimpy.engines.base_engine.MDAEngine.__init__", return_value=None):
        eng = MantisEngine(mock_core)
    # Manually assign the core weakref since we bypassed super().__init__
    eng._mmcore_ref = weakref.ref(mock_core)
    return eng


# ---------------------------------------------------------------------------
# _get_next_acquisition_name() — pure function
# ---------------------------------------------------------------------------


def test_format_duration_scales_units_with_magnitude():
    # A pre-scan runs from seconds to hours; the unit scales so the number stays readable.
    assert _format_duration(0) == "0.0s"
    assert _format_duration(3.71) == "3.7s"
    assert _format_duration(59.94) == "59.9s"
    assert _format_duration(60) == "1m 00s"
    assert _format_duration(432) == "7m 12s"
    assert _format_duration(3600) == "1h 00m 00s"
    assert _format_duration(3800) == "1h 03m 20s"
    # Rounding must not leave a bare 60 in the seconds/minutes slot.
    assert _format_duration(3599.6) == "1h 00m 00s"
    # A negative span (clock weirdness) must not render as garbage.
    assert _format_duration(-5) == "0.0s"


def test_next_name_first_acquisition_in_empty_dir(tmp_path):
    # Empty directory → the index is still appended; the bare name is never used
    assert _get_next_acquisition_name(tmp_path, "acq") == "acq_1"


def test_next_name_appends_suffix_when_first_index_taken(tmp_path):
    # acq_1.ome.zarr already exists → append the next free suffix, acq_2
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


def test_engine_derives_from_base_engine():
    assert issubclass(MantisEngine, BaseEngine)


def test_init_default_attributes(engine):
    # Mantis-specific state starts unset
    assert engine._xy_stage_speed is None
    assert engine._dynatrack is None


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


# ---------------------------------------------------------------------------
# setup_event() — SkipEvent on autofocus failure
# ---------------------------------------------------------------------------


def test_setup_event_autofocus_failure_raises_skip_event(engine, mock_core):
    # Autofocus on + failure → SkipEvent raised with num_frames=1 for single event
    from pymmcore_plus.mda import SkipEvent

    engine._use_autofocus = True
    engine._autofocus_method = "demo-PFS"
    # Force autofocus to fail at every event
    engine._autofocus_fail_at_index = [{}]

    event = MDAEvent()
    with pytest.raises(SkipEvent, match="autofocus failed") as exc_info:
        engine.setup_event(event)
    assert exc_info.value.num_frames == 1


def test_setup_event_autofocus_failure_sequenced_event_skips_all_frames(engine, mock_core):
    # SkipEvent.num_frames equals len(event.events) for SequencedEvents
    from pymmcore_plus.core._sequencing import SequencedEvent
    from pymmcore_plus.mda import SkipEvent

    engine._use_autofocus = True
    engine._autofocus_method = "demo-PFS"

    engine._autofocus_fail_at_index = [{}]

    sub_events = [MDAEvent(index={"t": 0, "p": 0, "z": i}) for i in range(5)]
    seq_event = SequencedEvent(events=sub_events)

    with pytest.raises(SkipEvent) as exc_info:
        engine.setup_event(seq_event)
    assert exc_info.value.num_frames == 5


def test_setup_event_autofocus_success_does_not_raise(engine, mock_core):
    # Autofocus on + success → no SkipEvent, delegates to parent setup_event
    engine._use_autofocus = True
    engine._autofocus_method = "demo-PFS"
    engine._autofocus_fail_at_index = []

    event = MDAEvent()
    with patch("shrimpy.engines.base_engine.MDAEngine.setup_event"):
        engine.setup_event(event)  # should not raise


# ---------------------------------------------------------------------------
# teardown_sequence()
# ---------------------------------------------------------------------------

# Applying / skipping reset_hardware_sequencing_settings is BaseEngine behavior and is
# covered in test_base_engine.py; only the mantis-specific FOV-selection teardown is
# tested here.


def test_teardown_no_metadata(engine, mock_core):
    # Sequence with no shrimPy metadata at all → no setProperty calls
    seq = MDASequence()
    with patch("shrimpy.engines.base_engine.MDAEngine.teardown_sequence"):
        engine.teardown_sequence(seq)
    mock_core.setProperty.assert_not_called()


def test_teardown_captures_selection_before_debug_writes():
    """A failing debug write must not cost us the selection.

    Regression: finalize_debug_summary() used to run BEFORE _fov_passed_names was
    assigned, so a PermissionError on fov_summary.csv (a spreadsheet holding it open)
    aborted teardown_sequence with the list still empty -- acquire() then skipped the
    timelapse for "no FOVs passed" even though FOVs had scored and passed.
    """
    engine = MantisEngine.__new__(MantisEngine)
    engine._dynatrack = None
    engine._fov_passed_names = []
    core = MagicMock()
    engine._mmcore_ref = weakref.ref(core)  # `mmcore` is a read-only property

    fov = MagicMock()
    fov.passed_position_names.return_value = ["p0_0019", "p0_0021", "p0_0016"]
    fov.finalize_debug_summary.side_effect = PermissionError("fov_summary.csv is locked")
    engine._fov = fov

    sequence = MDASequence(stage_positions=[{"x": 0, "y": 0}])
    with patch.object(MDAEngine, "teardown_sequence"), pytest.raises(PermissionError):
        engine.teardown_sequence(sequence)

    # Even though the debug write blew up, the selection survived.
    assert engine._fov_passed_names == ["p0_0019", "p0_0021", "p0_0016"]


# ---------------------------------------------------------------------------
# _get_next_acquisition_name() — leftovers from crashed runs
# ---------------------------------------------------------------------------
# A run that dies mid-pre-scan writes <name>_fov_debug/ and maybe
# <name>_prescan.ome.zarr but never <name>.ome.zarr, because the pre-scan run passes
# output=None. Testing only the store handed the next run the same name, whose worker
# then appended to the dead run's fov_summary.csv and reused its debug directory.


def test_next_name_avoids_a_crashed_prescan_debug_dir(tmp_path):
    # Store absent, debug dir present -> the name is NOT free.
    (tmp_path / "acq_fov_debug_1").mkdir()
    assert _get_next_acquisition_name(tmp_path, "acq") == "acq_2"


def test_next_name_avoids_a_crashed_prescan_zarr(tmp_path):
    (tmp_path / "acq_prescan_1.ome.zarr").mkdir()
    assert _get_next_acquisition_name(tmp_path, "acq") == "acq_2"


def test_next_name_avoids_indexed_sibling_leftovers(tmp_path):
    # The real failure: stores acq_1..acq_3 exist, but a crashed 4th run left
    # acq_fov_debug_4 and acq_prescan_4. acq_4 looks free by the store alone.
    for suffix in ("_1", "_2", "_3"):
        (tmp_path / f"acq{suffix}.ome.zarr").mkdir()
    for suffix in ("_1", "_2", "_3", "_4"):
        (tmp_path / f"acq_fov_debug{suffix}").mkdir()
    (tmp_path / "acq_prescan_4.ome.zarr").mkdir()

    assert _get_next_acquisition_name(tmp_path, "acq") == "acq_5"


def test_next_name_never_reuses_or_deletes_leftovers(tmp_path):
    # Freshness is about the NAME existing, not about the run having completed: an
    # incomplete folder is skipped and left untouched for inspection.
    debug = tmp_path / "acq_fov_debug_1"
    debug.mkdir()
    (debug / "fov_summary.csv").write_text("name,proba\np0_0000,0.5\n")

    name = _get_next_acquisition_name(tmp_path, "acq")

    assert name == "acq_2"
    assert (debug / "fov_summary.csv").read_text() == "name,proba\np0_0000,0.5\n"


def test_artifact_paths_cover_store_and_siblings(tmp_path):
    from shrimpy.engines.mantis_engine import acquisition_artifact_paths

    assert [p.name for p in acquisition_artifact_paths(tmp_path, "acq_1", 1)] == [
        "acq_1.ome.zarr",
        "acq_fov_debug_1",
        "acq_prescan_1.ome.zarr",
    ]
    # The dedup index lands at the END of each sibling name, not mid-name.
    assert [p.name for p in acquisition_artifact_paths(tmp_path, "acq_2", 2)] == [
        "acq_2.ome.zarr",
        "acq_fov_debug_2",
        "acq_prescan_2.ome.zarr",
    ]
