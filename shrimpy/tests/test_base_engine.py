"""Unit tests for BaseEngine and helper functions.

Tests use a mock CMMCorePlus to isolate the shared engine logic from real
hardware and the parent MDAEngine. Microscope-specific behavior is tested in
``test_<microscope>_engine.py``.
"""

from __future__ import annotations

import weakref

from unittest.mock import MagicMock, patch

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


def test_property_changed_ignores_pfs_status(engine, caplog):
    # Noisy PFS properties are filtered out of the log
    with caplog.at_level("DEBUG", logger="shrimpy.engines.base_engine"):
        engine._on_property_changed("TIPFSStatus", "PFS Status", "0000001100001010")
    assert caplog.text == ""


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
# _should_engage_autofocus() — once per burst / once per Z-stack
# ---------------------------------------------------------------------------


def test_should_engage_autofocus_sequenced_event(engine):
    # A SequencedEvent is one hardware-triggered burst → engage once, up front,
    # regardless of the Z index its first frame carries.
    for z in (0, 1, 5):
        sub_events = [MDAEvent(index={"t": 0, "p": 0, "z": z + i}) for i in range(3)]
        assert engine._should_engage_autofocus(SequencedEvent(events=sub_events)) is True


def test_should_engage_autofocus_single_event_bottom_of_stack(engine):
    # Single events arrive one Z slice at a time → engage only at z=0
    assert engine._should_engage_autofocus(MDAEvent(index={"t": 0, "p": 0, "z": 0})) is True
    for z in (1, 2, 7):
        event = MDAEvent(index={"t": 0, "p": 0, "z": z})
        assert engine._should_engage_autofocus(event) is False


def test_should_engage_autofocus_event_without_z_axis(engine):
    # No Z axis in the sequence → every event engages
    assert engine._should_engage_autofocus(MDAEvent(index={"t": 0, "p": 0})) is True


def test_engage_autofocus_skipped_within_stack_keeps_previous_outcome(engine):
    # Autofocus is not re-run for z>0; the previous outcome (and lock) stands
    engine._use_autofocus = True
    engine._autofocus_method = DEMO_PFS_METHOD
    engine._autofocus_fail_at_index = []

    engine._engage_autofocus(MDAEvent(index={"t": 0, "p": 0, "z": 0}))
    assert engine._autofocus_success is True

    # Would fail if it ran at all — but it must not run within the stack
    engine._autofocus_fail_at_index = [{}]
    engine._engage_autofocus(MDAEvent(index={"t": 0, "p": 0, "z": 1}))
    assert engine._autofocus_success is True

    # ... and runs again at the bottom of the next stack
    engine._engage_autofocus(MDAEvent(index={"t": 0, "p": 1, "z": 0}))
    assert engine._autofocus_success is False


def test_engage_autofocus_calls_hardware_once_per_stack(engine):
    # The microscope-specific hook is called only for the events that engage
    engine._use_autofocus = True
    engine._autofocus_method = "PFS"

    with patch.object(engine, "engage_autofocus", return_value=True) as mock_af:
        for z in range(4):
            engine._engage_autofocus(MDAEvent(index={"t": 0, "p": 0, "z": z}))
    assert mock_af.call_count == 1


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

    eng._engage_autofocus(MDAEvent())
    assert eng._autofocus_success is True
    mock_core.setPosition.assert_called_once_with("FocusDrive", 100.0)
    mock_core.fullFocus.assert_called_once()

    # A failing AFC call is reported, so setup_event skips the event
    mock_core.fullFocus.side_effect = RuntimeError("no lock")
    eng._engage_autofocus(MDAEvent())
    assert eng._autofocus_success is False
