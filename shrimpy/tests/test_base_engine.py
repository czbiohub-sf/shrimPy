"""Unit tests for BaseEngine and helper functions.

Tests use a mock CMMCorePlus to isolate the shared engine logic from real
hardware and the parent MDAEngine. Microscope-specific behavior is tested in
``test_<microscope>_engine.py``.
"""

from __future__ import annotations

import weakref

from unittest.mock import MagicMock, call, patch

import pytest

from pymmcore_plus.core._constants import Keyword
from pymmcore_plus.core._sequencing import SequencedEvent
from pymmcore_plus.mda import SkipEvent
from useq import MDAEvent, MDASequence

from shrimpy.engines.base_engine import (
    DEMO_PFS_METHOD,
    BaseEngine,
    _get_next_acquisition_name,
)
from shrimpy.engines.dragonfly_engine import DragonflyEngine
from shrimpy.engines.isim_engine import ISIMEngine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine(mock_core: MagicMock) -> BaseEngine:
    """Create a BaseEngine wired to the mock CMMCorePlus.

    Patches the parent MDAEngine.__init__ so we don't need a real core for
    the super().__init__() call, then manually sets mmcore.
    """
    with patch("shrimpy.engines.base_engine.MDAEngine.__init__", return_value=None):
        eng = BaseEngine(mock_core)
    # Manually assign the core weakref since we bypassed super().__init__
    eng._mmcore_ref = weakref.ref(mock_core)
    return eng


def _make_sequence(shrimpy_meta: dict | None = None) -> MDASequence:
    """Helper to create an MDASequence with optional shrimPy metadata sections."""
    return MDASequence(metadata=shrimpy_meta or {})


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
# BaseEngine.__init__()
# ---------------------------------------------------------------------------


def test_init_default_attributes(engine):
    # All autofocus-related attributes start disabled/unset
    assert engine._use_autofocus is False
    assert engine._autofocus_success is False
    assert engine._autofocus_stage is None
    assert engine._autofocus_method is None
    assert engine._autofocus_fail_at_index is None
    assert engine._xy_stage_device is None
    assert engine._data_path is None
    assert engine._dynatrack is None


def test_init_hardware_sequencing_defaults(mock_core):
    # shrimPy defaults: hardware sequencing on, redundant XY moves off
    engine = BaseEngine(mock_core)
    assert engine.use_hardware_sequencing is True
    assert engine.force_set_xy_position is False


def test_init_kwargs_override_defaults(mock_core):
    # Explicit kwargs win over the shrimPy defaults
    engine = BaseEngine(mock_core, use_hardware_sequencing=False, force_set_xy_position=True)
    assert engine.use_hardware_sequencing is False
    assert engine.force_set_xy_position is True


def test_init_registers_engine_and_callbacks(mock_core):
    # Verify that __init__ wires up the engine and event callbacks
    with patch("shrimpy.engines.base_engine.MDAEngine.__init__", return_value=None):
        BaseEngine(mock_core)

    mock_core.mda.set_engine.assert_called_once()
    mock_core.events.propertyChanged.connect.assert_called_once()
    mock_core.events.roiSet.connect.assert_called_once()
    mock_core.events.XYStagePositionChanged.connect.assert_called_once()


# ---------------------------------------------------------------------------
# Logging callbacks
# ---------------------------------------------------------------------------


def test_property_changed_logged(engine, caplog):
    with caplog.at_level("DEBUG", logger="shrimpy.engines.base_engine"):
        engine._on_property_changed("Camera", "Exposure", "10.0")
    assert "Camera.Exposure = 10.0" in caplog.text


def test_property_changed_logs_all_properties_by_default(engine, caplog):
    # BaseEngine has no microscope-specific properties to filter out
    assert engine.NOISY_PROPERTIES == ()
    with caplog.at_level("DEBUG", logger="shrimpy.engines.base_engine"):
        engine._on_property_changed("TIPFSStatus", "PFS Status", "0000001100001010")
    assert "TIPFSStatus.PFS Status = 0000001100001010" in caplog.text


def test_roi_set_logged(engine, caplog):
    with caplog.at_level("DEBUG", logger="shrimpy.engines.base_engine"):
        engine._on_roi_set("Camera", 0, 0, 2048, 512)
    assert "x=0, y=0, width=2048, height=512" in caplog.text


def test_xy_stage_position_changed_logged(engine, caplog):
    with caplog.at_level("DEBUG", logger="shrimpy.engines.base_engine"):
        engine._on_xy_stage_position_changed("XYStage", 100.0, -50.0)
    assert "x=100.00, y=-50.00" in caplog.text


# ---------------------------------------------------------------------------
# setup_sequence()
# ---------------------------------------------------------------------------


def test_setup_sequence_no_shrimpy_metadata(engine):
    # Should not raise when metadata is empty
    seq = _make_sequence()
    with patch("shrimpy.engines.base_engine.MDAEngine.setup_sequence"):
        engine.setup_sequence(seq)
    assert engine._use_autofocus is False


def test_setup_sequence_autofocus_enabled(engine, mock_core):
    # Autofocus metadata with enabled=True should configure the engine
    af = {"enabled": True, "stage": "ZDrive", "method": "PFS"}
    seq = _make_sequence({"autofocus": af})
    with patch("shrimpy.engines.base_engine.MDAEngine.setup_sequence"):
        engine.setup_sequence(seq)

    assert engine._use_autofocus is True
    assert engine._autofocus_stage == "ZDrive"
    assert engine._autofocus_method == "PFS"
    # Non-demo method should call setAutoFocusDevice
    mock_core.setAutoFocusDevice.assert_called_once_with("PFS")


def test_setup_sequence_autofocus_disabled(engine, mock_core):
    # Autofocus explicitly disabled → _use_autofocus stays False
    af = {"enabled": False, "stage": "ZDrive", "method": "PFS"}
    seq = _make_sequence({"autofocus": af})
    with patch("shrimpy.engines.base_engine.MDAEngine.setup_sequence"):
        engine.setup_sequence(seq)
    assert engine._use_autofocus is False
    mock_core.setAutoFocusDevice.assert_not_called()


def test_setup_sequence_demo_pfs_not_set_as_device(engine, mock_core):
    # demo-PFS is simulated in software → never passed to setAutoFocusDevice
    af = {"enabled": True, "stage": "Z", "method": DEMO_PFS_METHOD}
    seq = _make_sequence({"autofocus": af})
    with patch("shrimpy.engines.base_engine.MDAEngine.setup_sequence"):
        engine.setup_sequence(seq)
    assert engine._use_autofocus is True
    mock_core.setAutoFocusDevice.assert_not_called()


def test_setup_sequence_stores_xy_stage_device(engine, mock_core):
    seq = _make_sequence()
    with patch("shrimpy.engines.base_engine.MDAEngine.setup_sequence"):
        engine.setup_sequence(seq)
    assert engine._xy_stage_device == mock_core.getXYStageDevice.return_value


def test_setup_sequence_returns_parent_summary_metadata(engine):
    # The parent's SummaryMetaV1 is passed through unchanged
    sentinel = object()
    seq = _make_sequence()
    with patch("shrimpy.engines.base_engine.MDAEngine.setup_sequence", return_value=sentinel):
        assert engine.setup_sequence(seq) is sentinel


# ---------------------------------------------------------------------------
# _engage_autofocus() — dispatch
# ---------------------------------------------------------------------------


def test_autofocus_disabled_returns_early(engine):
    # Autofocus disabled → no method calls
    engine._use_autofocus = False
    with patch.object(engine, "engage_autofocus") as mock_engage:
        engine._engage_autofocus(MDAEvent())
    mock_engage.assert_not_called()


def test_autofocus_demo_pfs_dispatched(engine):
    # demo-PFS method → calls _engage_demo_pfs
    engine._use_autofocus = True
    engine._autofocus_method = DEMO_PFS_METHOD
    with patch.object(engine, "_engage_demo_pfs", return_value=True) as mock_demo:
        engine._engage_autofocus(MDAEvent())
    mock_demo.assert_called_once()
    assert engine._autofocus_success is True


def test_autofocus_hardware_method_dispatched(engine):
    # Any other method → delegates to the subclass hook and records the result
    engine._use_autofocus = True
    engine._autofocus_method = "PFS"
    event = MDAEvent()
    with patch.object(engine, "engage_autofocus", return_value=False) as mock_engage:
        engine._engage_autofocus(event)
    mock_engage.assert_called_once_with(event)
    assert engine._autofocus_success is False


def test_engage_autofocus_not_implemented_in_base(engine):
    # The base engine has no hardware autofocus routine
    engine._use_autofocus = True
    engine._autofocus_method = "PFS"
    with pytest.raises(NotImplementedError, match="engage_autofocus"):
        engine._engage_autofocus(MDAEvent())


# ---------------------------------------------------------------------------
# _get_autofocus_z_position()
# ---------------------------------------------------------------------------


def test_autofocus_z_position_from_event_properties(engine, mock_core):
    # Z target is read from the event properties when present
    engine._autofocus_stage = "ZDrive"
    event = MDAEvent(properties=[("ZDrive", "Position", 12.5)])
    assert engine._get_autofocus_z_position(event) == 12.5
    mock_core.getPosition.assert_not_called()


def test_autofocus_z_position_falls_back_to_core(engine, mock_core):
    # No matching property → current stage position
    engine._autofocus_stage = "ZDrive"
    mock_core.getPosition.return_value = 42.0
    assert engine._get_autofocus_z_position(MDAEvent()) == 42.0
    mock_core.getPosition.assert_called_once_with("ZDrive")


def test_autofocus_z_position_ignores_other_devices(engine, mock_core):
    # Properties for other devices don't count as the autofocus stage target
    engine._autofocus_stage = "ZDrive"
    mock_core.getPosition.return_value = 42.0
    event = MDAEvent(properties=[("AP Galvo", "Position", 7.0)])
    assert engine._get_autofocus_z_position(event) == 42.0


# ---------------------------------------------------------------------------
# _engage_demo_pfs()
# ---------------------------------------------------------------------------


def test_demo_pfs_fail_at_index_matching_event_fails(engine):
    # Deterministic failure when event index matches fail_at_index entry
    event = MDAEvent(index={"t": 1, "p": 0})
    assert engine._engage_demo_pfs(event=event, fail_at_index=[{"t": 1, "p": 0}]) is False


def test_demo_pfs_fail_at_index_partial_match_fails(engine):
    # Partial key match: {"p": 0} matches any event with p=0
    event = MDAEvent(index={"t": 5, "p": 0})
    assert engine._engage_demo_pfs(event=event, fail_at_index=[{"p": 0}]) is False


def test_demo_pfs_fail_at_index_no_match_succeeds(engine):
    # No matching entry → autofocus succeeds
    event = MDAEvent(index={"t": 0, "p": 1})
    assert engine._engage_demo_pfs(event=event, fail_at_index=[{"t": 1, "p": 0}]) is True


def test_demo_pfs_fail_at_index_empty_dict_fails_all(engine):
    # Empty dict matches every event (all zero keys trivially match)
    event = MDAEvent(index={"t": 3, "p": 2})
    assert engine._engage_demo_pfs(event=event, fail_at_index=[{}]) is False


def test_demo_pfs_fail_at_index_empty_list_succeeds(engine):
    # Empty fail list → no failures
    event = MDAEvent(index={"t": 0, "p": 0})
    assert engine._engage_demo_pfs(event=event, fail_at_index=[]) is True


def test_demo_pfs_random_fallback_when_no_fail_at_index(engine):
    # When fail_at_index is None, uses random success_rate
    assert engine._engage_demo_pfs(event=MDAEvent(), success_rate=1.0) is True
    assert engine._engage_demo_pfs(event=MDAEvent(), success_rate=0.0) is False


def test_demo_pfs_sequenced_event_uses_first_sub_event_index(engine):
    # For SequencedEvents, index matching uses the first sub-event
    sub_events = [
        MDAEvent(index={"t": 0, "p": 1, "z": 0}),
        MDAEvent(index={"t": 0, "p": 1, "z": 1}),
    ]
    seq_event = SequencedEvent(events=sub_events)

    # Partial match on first sub-event's index → should fail
    assert engine._engage_demo_pfs(event=seq_event, fail_at_index=[{"p": 1}]) is False

    # No match → should succeed
    assert engine._engage_demo_pfs(event=seq_event, fail_at_index=[{"p": 2}]) is True


# ---------------------------------------------------------------------------
# _should_engage_autofocus() — once per burst / once per position
# ---------------------------------------------------------------------------


def test_should_engage_autofocus_sequenced_event(engine):
    # A SequencedEvent is one hardware-triggered burst → engage once, up front,
    # regardless of the Z index its first frame carries.
    for z in (0, 1, 5):
        sub_events = [MDAEvent(index={"t": 0, "p": 0, "z": z + i}) for i in range(3)]
        assert engine._should_engage_autofocus(SequencedEvent(events=sub_events)) is True


def test_should_engage_autofocus_single_event_new_position(engine):
    # Nothing engaged yet → the first event of any position engages
    assert engine._should_engage_autofocus(MDAEvent(index={"t": 0, "p": 0, "z": 0})) is True

    # Once a position has been attempted, no event of that position re-engages,
    # whatever its channel or Z index
    engine._last_autofocus_position = engine._autofocus_position_key(
        MDAEvent(index={"t": 0, "p": 0, "c": 0, "z": 0})
    )
    for c in range(4):
        for z in (0, 1, 7):
            event = MDAEvent(index={"t": 0, "p": 0, "c": c, "z": z})
            assert engine._should_engage_autofocus(event) is False

    # A new timepoint, stage position or grid site engages again
    for index in (
        {"t": 1, "p": 0, "z": 0},
        {"t": 0, "p": 1, "z": 0},
        {"t": 0, "p": 0, "g": 1, "z": 0},
    ):
        assert engine._should_engage_autofocus(MDAEvent(index=index)) is True


def test_autofocus_position_key_keeps_time_position_and_grid(engine):
    # Channel and Z are dropped; time, position and grid identify the location,
    # and key order does not depend on index insertion order.
    key = engine._autofocus_position_key(
        MDAEvent(index={"t": 2, "p": 1, "g": 3, "c": 9, "z": 4})
    )
    assert key == (("g", 3), ("p", 1), ("t", 2))
    assert key == engine._autofocus_position_key(
        MDAEvent(index={"z": 0, "c": 0, "g": 3, "t": 2, "p": 1})
    )


def test_should_engage_autofocus_event_without_z_axis(engine):
    # No Z axis in the sequence → the first event of each position still engages
    assert engine._should_engage_autofocus(MDAEvent(index={"t": 0, "p": 0})) is True


def test_engage_autofocus_skipped_within_position_keeps_previous_outcome(engine):
    # Autofocus is not re-run for the rest of a position; the previous outcome
    # (and lock) stands
    engine._use_autofocus = True
    engine._autofocus_method = DEMO_PFS_METHOD
    engine._autofocus_fail_at_index = []

    engine._engage_autofocus(MDAEvent(index={"t": 0, "p": 0, "c": 0, "z": 0}))
    assert engine._autofocus_success is True

    # Would fail if it ran at all — but it must not run again at this position,
    # neither for a later slice nor for a later channel
    engine._autofocus_fail_at_index = [{}]
    engine._engage_autofocus(MDAEvent(index={"t": 0, "p": 0, "c": 0, "z": 1}))
    assert engine._autofocus_success is True
    engine._engage_autofocus(MDAEvent(index={"t": 0, "p": 0, "c": 1, "z": 0}))
    assert engine._autofocus_success is True

    # ... and runs again at the next position
    engine._engage_autofocus(MDAEvent(index={"t": 0, "p": 1, "c": 0, "z": 0}))
    assert engine._autofocus_success is False


def test_engage_autofocus_calls_hardware_once_per_position(engine):
    # The microscope-specific hook is called only for the events that engage:
    # once for a whole 4-channel Z-stack, not once per channel.
    engine._use_autofocus = True
    engine._autofocus_method = "PFS"

    with patch.object(engine, "engage_autofocus", return_value=True) as mock_af:
        for c in range(4):
            for z in range(4):
                engine._engage_autofocus(MDAEvent(index={"t": 0, "p": 0, "c": c, "z": z}))
    assert mock_af.call_count == 1

    # Each new timepoint at the same position re-engages once, not once per
    # channel
    with patch.object(engine, "engage_autofocus", return_value=True) as mock_af:
        for t in range(1, 5):
            for c in range(4):
                for z in range(4):
                    engine._engage_autofocus(MDAEvent(index={"t": t, "p": 0, "c": c, "z": z}))
    assert mock_af.call_count == 4

    # ... and so does a new grid site
    with patch.object(engine, "engage_autofocus", return_value=True) as mock_af:
        for z in range(4):
            engine._engage_autofocus(MDAEvent(index={"t": 5, "p": 0, "g": 1, "z": z}))
    assert mock_af.call_count == 1


def test_engage_autofocus_failure_not_retried_within_position(engine):
    # A failed attempt is recorded, so the remaining channels/slices of the
    # position do not each retry the hardware
    engine._use_autofocus = True
    engine._autofocus_method = "PFS"

    with patch.object(engine, "engage_autofocus", return_value=False) as mock_af:
        for c in range(4):
            engine._engage_autofocus(MDAEvent(index={"t": 0, "p": 0, "c": c, "z": 0}))
    assert mock_af.call_count == 1
    assert engine._autofocus_success is False


def test_setup_sequence_resets_autofocus_position(engine, mock_core):
    # The position key is per-run state: FOV selection runs two sequences
    # through one engine and the timelapse must not inherit the pre-scan's
    # last position.
    engine._last_autofocus_position = (("p", 3),)
    with patch("shrimpy.engines.base_engine.MDAEngine.setup_sequence"):
        engine.setup_sequence(MDASequence())
    assert engine._last_autofocus_position is None


# ---------------------------------------------------------------------------
# Core-Focus homing — autofocus engages at the z_plan's starting position
# ---------------------------------------------------------------------------


def _autofocus_engine(engine, mock_core, focus_device="ZStage", stage="FocusDrive"):
    """Configure ``engine`` as a Dragonfly-style split focus/autofocus setup."""
    engine._use_autofocus = True
    engine._home_focus_device = True
    engine._autofocus_method = "Adaptive Focus Control"
    engine._autofocus_stage = stage
    mock_core.getFocusDevice.return_value = focus_device
    mock_core.getPosition.return_value = 5.0
    return engine


def test_capture_focus_home_records_core_focus_position(engine, mock_core):
    # The Core-Focus device differs from the autofocus stage → track its home
    _autofocus_engine(engine, mock_core)
    engine._capture_focus_home()
    assert engine._focus_device == "ZStage"
    assert engine._focus_home == 5.0


def test_capture_focus_home_skipped_when_focus_is_the_autofocus_stage(engine, mock_core):
    # mantis / demo: the z_plan and autofocus drive the same device, so homing
    # it would fight the z_plan
    _autofocus_engine(engine, mock_core, focus_device="ZDrive", stage="ZDrive")
    engine._capture_focus_home()
    assert engine._focus_device is None
    assert engine._focus_home is None


def test_capture_focus_home_skipped_unless_opted_in(engine, mock_core):
    # mantis: Core-Focus (the LS scan galvo) differs from the autofocus stage,
    # but moves neither sample nor objective, so PFS is indifferent to it and
    # homing would only add a galvo move before every sequenced burst.
    _autofocus_engine(engine, mock_core, focus_device="LS Scan Galvo", stage="ZDrive")
    engine._home_focus_device = False
    engine._capture_focus_home()
    assert engine._focus_device is None
    assert engine._focus_home is None


def test_setup_sequence_clears_autofocus_when_a_later_run_disables_it(engine, mock_core):
    # One engine runs more than one sequence (FOV selection pre-scan, then the
    # timelapse); a run that asks for no autofocus must not inherit the
    # previous run's stage and method.
    engine._use_autofocus = True
    engine._autofocus_stage = "FocusDrive"
    engine._autofocus_method = "AFC"

    with patch("shrimpy.engines.base_engine.MDAEngine.setup_sequence"):
        engine.setup_sequence(MDASequence(metadata={"autofocus": {"enabled": False}}))

    assert engine._use_autofocus is False
    assert engine._autofocus_stage is None
    assert engine._autofocus_method is None


def test_setup_sequence_reads_home_focus_device_from_metadata(engine, mock_core):
    # Opt-in flows from metadata.autofocus.home_focus_device, and does not
    # linger when a later run omits it
    mock_core.getFocusDevice.return_value = "ZStage"
    mock_core.getPosition.return_value = 5.0
    autofocus = {"enabled": True, "method": "AFC", "stage": "FocusDrive"}

    with patch("shrimpy.engines.base_engine.MDAEngine.setup_sequence"):
        engine.setup_sequence(
            MDASequence(metadata={"autofocus": {**autofocus, "home_focus_device": True}})
        )
        assert engine._home_focus_device is True
        assert engine._focus_device == "ZStage"

        engine.setup_sequence(MDASequence(metadata={"autofocus": autofocus}))
        assert engine._home_focus_device is False
        assert engine._focus_device is None


def test_capture_focus_home_skipped_when_autofocus_disabled(engine, mock_core):
    engine._use_autofocus = False
    engine._home_focus_device = True
    mock_core.getFocusDevice.return_value = "ZStage"
    engine._capture_focus_home()
    assert engine._focus_device is None


def test_capture_focus_home_survives_unreadable_position(engine, mock_core):
    _autofocus_engine(engine, mock_core)
    mock_core.getPosition.side_effect = RuntimeError("no reply")
    engine._capture_focus_home()
    assert engine._focus_device is None
    assert engine._focus_home is None


def test_setup_sequence_recaptures_focus_home_per_run(engine, mock_core):
    # Per-run state, like the autofocus position key
    _autofocus_engine(engine, mock_core)
    engine._focus_device = "stale"
    engine._focus_home = 999.0
    sequence = MDASequence(
        metadata={
            "autofocus": {
                "enabled": True,
                "method": "AFC",
                "stage": "FocusDrive",
                "home_focus_device": True,
            }
        }
    )
    with patch("shrimpy.engines.base_engine.MDAEngine.setup_sequence"):
        engine.setup_sequence(sequence)
    assert engine._focus_device == "ZStage"
    assert engine._focus_home == 5.0


def test_return_focus_device_home_moves_and_waits(engine, mock_core):
    engine._focus_device = "ZStage"
    engine._focus_home = 5.0
    mock_core.getPosition.return_value = -2.0  # left at the bottom of a stack

    engine._return_focus_device_home("test")
    mock_core.setPosition.assert_called_once_with("ZStage", 5.0)
    mock_core.waitForDevice.assert_called_once_with("ZStage")


def test_return_focus_device_home_noop_when_already_home(engine, mock_core):
    engine._focus_device = "ZStage"
    engine._focus_home = 5.0
    mock_core.getPosition.return_value = 5.0

    engine._return_focus_device_home("test")
    mock_core.setPosition.assert_not_called()


def test_return_focus_device_home_noop_without_a_home(engine, mock_core):
    engine._focus_device = None
    engine._focus_home = None
    engine._return_focus_device_home("test")
    mock_core.setPosition.assert_not_called()


def test_setup_event_homes_focus_before_autofocus(engine, mock_core):
    # The piezo must be back at its home position *before* fullFocus runs,
    # otherwise autofocus locks onto whatever plane the last stack ended on.
    _autofocus_engine(engine, mock_core)
    engine._focus_device = "ZStage"
    engine._focus_home = 5.0
    mock_core.getPosition.return_value = -2.0

    with (
        patch("shrimpy.engines.base_engine.MDAEngine.setup_event"),
        patch.object(engine, "engage_autofocus", return_value=True) as mock_af,
    ):
        mock_core.attach_mock(mock_af, "engage_autofocus")
        engine.setup_event(MDAEvent(index={"t": 0, "p": 0, "z": 0}))

    names = [c[0] for c in mock_core.mock_calls]
    assert names.index("setPosition") < names.index("engage_autofocus")


def test_setup_event_does_not_home_within_a_position(engine, mock_core):
    # Only the event that engages autofocus unwinds the stack; the remaining
    # slices must not drag the focus device back mid-stack.
    _autofocus_engine(engine, mock_core)
    engine._focus_device = "ZStage"
    engine._focus_home = 5.0
    engine._autofocus_success = True
    engine._last_autofocus_position = engine._autofocus_position_key(
        MDAEvent(index={"t": 0, "p": 0, "z": 0})
    )
    mock_core.getPosition.return_value = -2.0

    with (
        patch("shrimpy.engines.base_engine.MDAEngine.setup_event"),
        patch.object(engine, "engage_autofocus", return_value=True),
    ):
        for z in range(1, 5):
            engine.setup_event(MDAEvent(index={"t": 0, "p": 0, "z": z}))

    mock_core.setPosition.assert_not_called()


def test_teardown_sequence_returns_focus_device_home(engine, mock_core):
    # The last Z-stack has no successor to unwind it in setup_event
    engine._focus_device = "ZStage"
    engine._focus_home = 5.0
    mock_core.getPosition.return_value = 5.0  # value differs below

    mock_core.getPosition.return_value = -2.0
    with patch("shrimpy.engines.base_engine.MDAEngine.teardown_sequence") as mock_super:
        mock_core.attach_mock(mock_super, "super_teardown")
        engine.teardown_sequence(MDASequence())

    mock_core.setPosition.assert_called_once_with("ZStage", 5.0)
    # Before super(), whose _restore_initial_state() may also move Z
    names = [c[0] for c in mock_core.mock_calls]
    assert names.index("setPosition") < names.index("super_teardown")


# ---------------------------------------------------------------------------
# setup_event() — SkipEvent on autofocus failure
# ---------------------------------------------------------------------------


def test_setup_event_autofocus_failure_raises_skip_event(engine, mock_core):
    # Autofocus on + failure → SkipEvent raised with num_frames=1 for single event
    engine._use_autofocus = True
    engine._autofocus_method = DEMO_PFS_METHOD
    # Force autofocus to fail at every event
    engine._autofocus_fail_at_index = [{}]

    event = MDAEvent()
    with pytest.raises(SkipEvent, match="autofocus failed") as exc_info:
        engine.setup_event(event)
    assert exc_info.value.num_frames == 1


def test_setup_event_autofocus_failure_sequenced_event_skips_all_frames(engine, mock_core):
    # SkipEvent.num_frames equals len(event.events) for SequencedEvents
    engine._use_autofocus = True
    engine._autofocus_method = DEMO_PFS_METHOD
    engine._autofocus_fail_at_index = [{}]

    sub_events = [MDAEvent(index={"t": 0, "p": 0, "z": i}) for i in range(5)]
    seq_event = SequencedEvent(events=sub_events)

    with pytest.raises(SkipEvent) as exc_info:
        engine.setup_event(seq_event)
    assert exc_info.value.num_frames == 5


def test_setup_event_failure_at_z0_skips_rest_of_stack(engine, mock_core):
    # Non-sequenced events: autofocus is attempted once, at z=0. If it does not
    # engage, the remaining slices of that stack are skipped too — one SkipEvent
    # each, since the runner only skips the event it was raised from.
    engine._use_autofocus = True
    engine._autofocus_method = "PFS"

    with patch.object(engine, "engage_autofocus", return_value=False) as mock_af:
        for z in range(4):
            event = MDAEvent(index={"t": 0, "p": 0, "z": z})
            with pytest.raises(SkipEvent, match="autofocus failed") as exc_info:
                engine.setup_event(event)
            assert exc_info.value.num_frames == 1
    # ... and the hardware routine was only run once, at z=0
    assert mock_af.call_count == 1

    # The next position engages again; a successful lock acquires the stack
    with (
        patch.object(engine, "engage_autofocus", return_value=True),
        patch("shrimpy.engines.base_engine.MDAEngine.setup_event"),
    ):
        for z in range(4):
            engine.setup_event(MDAEvent(index={"t": 0, "p": 1, "z": z}))  # no raise


def test_setup_event_autofocus_success_does_not_raise(engine, mock_core):
    # Autofocus on + success → no SkipEvent, delegates to parent setup_event
    engine._use_autofocus = True
    engine._autofocus_method = DEMO_PFS_METHOD
    engine._autofocus_fail_at_index = []

    event = MDAEvent()
    with patch("shrimpy.engines.base_engine.MDAEngine.setup_event") as mock_parent:
        engine.setup_event(event)  # should not raise
    mock_parent.assert_called_once_with(event)


def test_setup_event_waits_for_xy_stage(engine, mock_core):
    # The XY move is not blocking, so the engine waits before focusing
    engine._xy_stage_device = "XYStage"
    # Set by MDAEngine.__init__, which the fixture bypasses
    engine.force_set_xy_position = False
    with patch("shrimpy.engines.base_engine.MDAEngine.setup_event"):
        engine.setup_event(MDAEvent(x_pos=10.0, y_pos=20.0))
    mock_core.waitForDevice.assert_any_call("XYStage")


def test_setup_event_does_not_wait_when_xy_position_unchanged(engine, mock_core):
    # Within a Z-stack the XY position repeats: the stage is neither moved nor
    # waited for.
    engine._xy_stage_device = "XYStage"
    engine.force_set_xy_position = False
    mock_core._last_xy_position = {None: (10.0, 20.0)}

    with patch("shrimpy.engines.base_engine.MDAEngine.setup_event"):
        for z in range(3):
            engine.setup_event(MDAEvent(index={"z": z}, x_pos=10.0, y_pos=20.0))
    mock_core.setXYPosition.assert_not_called()
    mock_core.waitForDevice.assert_not_called()


def test_setup_event_does_not_wait_without_xy_position(engine, mock_core):
    # Events with no XY position never move the stage
    engine._xy_stage_device = "XYStage"
    engine.force_set_xy_position = False

    with patch("shrimpy.engines.base_engine.MDAEngine.setup_event"):
        engine.setup_event(MDAEvent())
    mock_core.waitForDevice.assert_not_called()


def test_setup_event_waits_when_xy_position_forced(engine, mock_core):
    # force_set_xy_position re-issues the move even for an unchanged position
    engine._xy_stage_device = "XYStage"
    engine.force_set_xy_position = True
    mock_core._last_xy_position = {None: (10.0, 20.0)}

    with patch("shrimpy.engines.base_engine.MDAEngine.setup_event"):
        engine.setup_event(MDAEvent(x_pos=10.0, y_pos=20.0))
    mock_core.waitForDevice.assert_any_call("XYStage")


def test_setup_event_waits_once_per_position(engine, mock_core):
    # One wait at the bottom of the stack, none for the remaining slices, and
    # another one when the stack moves to a new position.
    engine._xy_stage_device = "XYStage"
    engine.force_set_xy_position = False
    mock_core._last_xy_position = {None: (0.0, 0.0)}

    def _set_xy(event):
        if event.x_pos is not None:
            mock_core._last_xy_position[None] = (event.x_pos, event.y_pos)

    with (
        patch("shrimpy.engines.base_engine.MDAEngine.setup_event"),
        patch.object(engine, "_set_event_xy_position", side_effect=_set_xy),
    ):
        for p, (x, y) in enumerate([(10.0, 20.0), (30.0, 40.0)]):
            for z in range(3):
                engine.setup_event(MDAEvent(index={"p": p, "z": z}, x_pos=x, y_pos=y))
    assert mock_core.waitForDevice.call_count == 2


# ---------------------------------------------------------------------------
# _set_event_properties() — autofocus stage protection
# ---------------------------------------------------------------------------


def test_set_event_properties_skips_z_on_autofocus_stage(engine):
    # With autofocus on, Z positions must not be written to the autofocus stage
    engine._use_autofocus = True
    engine._autofocus_stage = "ZDrive"
    properties = [("ZDrive", Keyword.Position, 10.0), ("Camera", "Exposure", 5.0)]

    with patch("shrimpy.engines.base_engine.MDAEngine._set_event_properties") as mock_parent:
        engine._set_event_properties(properties)

    assert mock_parent.call_args_list == [(([("Camera", "Exposure", 5.0)],),)]


def test_set_event_properties_sets_z_when_autofocus_disabled(engine):
    # With autofocus off, all properties are forwarded to the parent
    engine._use_autofocus = False
    engine._autofocus_stage = "ZDrive"
    properties = [("ZDrive", Keyword.Position, 10.0)]

    with patch("shrimpy.engines.base_engine.MDAEngine._set_event_properties") as mock_parent:
        engine._set_event_properties(properties)

    mock_parent.assert_called_once_with([("ZDrive", Keyword.Position, 10.0)])


def test_set_event_properties_sets_z_on_other_stages(engine):
    # Only the autofocus stage is protected; other Z stages still move
    engine._use_autofocus = True
    engine._autofocus_stage = "ZDrive"
    properties = [("AP Galvo", Keyword.Position, 3.0)]

    with patch("shrimpy.engines.base_engine.MDAEngine._set_event_properties") as mock_parent:
        engine._set_event_properties(properties)

    mock_parent.assert_called_once_with([("AP Galvo", Keyword.Position, 3.0)])


# ---------------------------------------------------------------------------
# teardown_sequence()
# ---------------------------------------------------------------------------


def test_teardown_applies_reset_hardware_sequencing_settings(engine, mock_core):
    # Sequence with reset_hardware_sequencing_settings → applies each setting
    seq = MDASequence(
        metadata={
            "reset_hardware_sequencing_settings": [
                ["Z", "UseSequences", "No"],
            ],
        }
    )
    with patch("shrimpy.engines.base_engine.MDAEngine.teardown_sequence"):
        engine.teardown_sequence(seq)
    mock_core.setProperty.assert_called_once_with("Z", "UseSequences", "No")


def test_teardown_no_reset_settings(engine, mock_core):
    # Sequence without reset_hardware_sequencing_settings → no setProperty calls
    seq = MDASequence(metadata={})
    with patch("shrimpy.engines.base_engine.MDAEngine.teardown_sequence"):
        engine.teardown_sequence(seq)
    mock_core.setProperty.assert_not_called()


def test_teardown_no_shrimpy_metadata(engine, mock_core):
    # Sequence with no metadata at all → no setProperty calls
    seq = MDASequence()
    with patch("shrimpy.engines.base_engine.MDAEngine.teardown_sequence"):
        engine.teardown_sequence(seq)
    mock_core.setProperty.assert_not_called()


# ---------------------------------------------------------------------------
# Placeholder microscope engines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine_cls", [ISIMEngine, DragonflyEngine])
def test_placeholder_engines_inherit_shared_behavior(engine_cls, mock_core):
    # The iSIM / Dragonfly engines are BaseEngine subclasses that inherit the
    # shared defaults; they acquire with autofocus disabled or with demo-PFS.
    with patch("shrimpy.engines.base_engine.MDAEngine.__init__", return_value=None):
        eng = engine_cls(mock_core)
    eng._mmcore_ref = weakref.ref(mock_core)

    assert isinstance(eng, BaseEngine)
    assert eng._engage_demo_pfs(event=MDAEvent(), fail_at_index=[]) is True


def test_isim_engine_has_no_hardware_autofocus(mock_core):
    # iSIM is still a skeleton: no engage_autofocus() implementation.
    with patch("shrimpy.engines.base_engine.MDAEngine.__init__", return_value=None):
        eng = ISIMEngine(mock_core)
    eng._mmcore_ref = weakref.ref(mock_core)

    eng._use_autofocus = True
    eng._autofocus_method = "PFS"
    with pytest.raises(NotImplementedError, match="engage_autofocus"):
        eng._engage_autofocus(MDAEvent())


def test_dragonfly_engage_autofocus_calls_afc(mock_core):
    # Dragonfly implements engage_autofocus() via Leica AFC.
    with patch("shrimpy.engines.base_engine.MDAEngine.__init__", return_value=None):
        eng = DragonflyEngine(mock_core)
    eng._mmcore_ref = weakref.ref(mock_core)

    eng._use_autofocus = True
    eng._autofocus_method = "Adaptive Focus Control"
    eng._autofocus_stage = "FocusDrive"
    mock_core.getPosition.return_value = 100.0

    # Each call uses a distinct position index: autofocus engages once per
    # position, so repeating one would be skipped (see _should_engage_autofocus)
    eng._engage_autofocus(MDAEvent(index={"p": 0}))
    assert eng._autofocus_success is True
    # Locked on the first try. The stage is already at the target, so the
    # zero-distance move is skipped entirely (see _move_focus_stage).
    mock_core.setPosition.assert_not_called()
    mock_core.fullFocus.assert_called_once()

    # A failed call is retried at increasing Z offsets, and the offset at which
    # it locks is the one the stage is left at
    mock_core.reset_mock()
    mock_core.fullFocus.side_effect = [RuntimeError("no lock"), None]
    eng._engage_autofocus(MDAEvent(index={"p": 1}))
    assert eng._autofocus_success is True
    assert mock_core.setPosition.call_args_list == [call("FocusDrive", 90.0)]

    # Exhausting every offset is reported, so setup_event skips the event
    mock_core.reset_mock()
    mock_core.fullFocus.side_effect = RuntimeError("no lock")
    eng._engage_autofocus(MDAEvent(index={"p": 2}))
    assert eng._autofocus_success is False
    # Every offset is tried, then the stage is returned to the target position
    # rather than left at the last (+30 um) offset it failed on. The offset-0
    # attempt and the restore are both no-ops here because the mocked stage
    # never leaves 100.0.
    assert mock_core.setPosition.call_args_list == [
        call("FocusDrive", 100.0 + offset) for offset in (-10, 10, -20, 20, -30, 30)
    ]


def test_engage_leica_afc_does_not_move_the_stage_for_the_zero_offset(mock_core):
    # The first attempt focuses at z_position itself, so the stage is never
    # commanded — not even when it reads far from z_position, which is when a
    # distance-based guard alone would let a move through.
    eng = _dragonfly(mock_core)
    eng._autofocus_stage = "FocusDrive"
    mock_core.getPosition.return_value = 500.0  # nowhere near the target

    assert eng._engage_leica_afc("FocusDrive", 100.0) is True
    mock_core.setPosition.assert_not_called()
    mock_core.fullFocus.assert_called_once()


def test_engage_leica_afc_fallback_offsets_are_relative_to_z_position(mock_core):
    # Once AFC fails at z_position, the fallbacks are absolute targets around
    # it, regardless of where the stage was left
    eng = _dragonfly(mock_core)
    eng._autofocus_stage = "FocusDrive"
    mock_core.getPosition.return_value = 500.0
    mock_core.fullFocus.side_effect = [RuntimeError("no lock"), None]

    assert eng._engage_leica_afc("FocusDrive", 100.0) is True
    mock_core.setPosition.assert_called_once_with("FocusDrive", 90.0)


def _dragonfly(mock_core) -> DragonflyEngine:
    with patch("shrimpy.engines.base_engine.MDAEngine.__init__", return_value=None):
        eng = DragonflyEngine(mock_core)
    eng._mmcore_ref = weakref.ref(mock_core)
    return eng


def test_move_focus_stage_skips_moves_below_threshold(mock_core):
    # Commanding a Leica FocusDrive to the position it already holds hangs
    # waitForDevice: the adapter never sees the $71004 motion flag it clears
    # Busy() on. Anything under MIN_FOCUS_MOVE_UM is reported as already there.
    eng = _dragonfly(mock_core)
    mock_core.getPosition.return_value = 7914.6245

    for target in (7914.6245, 7914.6245 + 0.0999, 7914.6245 - 0.0999):
        assert eng._move_focus_stage("FocusDrive", target) is True
    mock_core.setPosition.assert_not_called()
    mock_core.waitForDevice.assert_not_called()


def test_move_focus_stage_moves_at_the_threshold(mock_core):
    # Exactly MIN_FOCUS_MOVE_UM moves, despite abs(100.1 - 100.0) landing at
    # 0.09999999999999432 in binary floating point
    eng = _dragonfly(mock_core)
    mock_core.getPosition.return_value = 100.0

    assert eng._move_focus_stage("FocusDrive", 100.1) is True
    mock_core.setPosition.assert_called_once_with("FocusDrive", 100.1)
    mock_core.waitForDevice.assert_called_once_with("FocusDrive")


def test_move_focus_stage_moves_well_above_threshold(mock_core):
    eng = _dragonfly(mock_core)
    mock_core.getPosition.return_value = 100.0

    assert eng._move_focus_stage("FocusDrive", 90.0) is True
    mock_core.setPosition.assert_called_once_with("FocusDrive", 90.0)


def test_move_focus_stage_moves_when_position_unreadable(mock_core):
    # Without a position we cannot compute the distance, so command the move
    # rather than silently skipping it
    eng = _dragonfly(mock_core)
    mock_core.getPosition.side_effect = RuntimeError("no reply")

    assert eng._move_focus_stage("FocusDrive", 100.0) is True
    mock_core.setPosition.assert_called_once_with("FocusDrive", 100.0)


def test_move_focus_stage_reports_wait_failure(mock_core):
    # A waitForDevice timeout is non-fatal: the caller tries the next Z offset
    eng = _dragonfly(mock_core)
    mock_core.getPosition.return_value = 100.0
    mock_core.waitForDevice.side_effect = RuntimeError("timed out after 5000ms")

    assert eng._move_focus_stage("FocusDrive", 130.0) is False


def test_move_focus_stage_reports_set_position_failure(mock_core):
    eng = _dragonfly(mock_core)
    mock_core.getPosition.return_value = 100.0
    mock_core.setPosition.side_effect = RuntimeError("out of range")

    assert eng._move_focus_stage("FocusDrive", 130.0) is False
