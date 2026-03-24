"""Tests for position update infrastructure.

Covers PositionStore, PositionUpdateManager, and MantisEngine integration.
"""

from __future__ import annotations

import threading
import time
import weakref

from unittest.mock import MagicMock, patch

import pytest

from useq import MDAEvent, MDASequence

from shrimpy.mantis.mantis_engine import MantisEngine
from shrimpy.mantis.position_update import (
    PositionCoordinates,
    PositionStore,
    PositionUpdateConfig,
    PositionUpdateManager,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine(mock_core: MagicMock) -> MantisEngine:
    """Create a MantisEngine wired to the mock CMMCorePlus."""
    with patch("shrimpy.mantis.mantis_engine.MDAEngine.__init__", return_value=None):
        eng = MantisEngine(mock_core)
    eng._mmcore_ref = weakref.ref(mock_core)
    return eng


@pytest.fixture
def position_store() -> PositionStore:
    """Create a PositionStore initialized with 3 positions including Z."""
    store = PositionStore()
    seq = MDASequence(
        stage_positions=[
            {"x": 100, "y": 200, "z": 50},
            {"x": 300, "y": 400, "z": 60},
            {"x": 500, "y": 600, "z": 70},
        ]
    )
    store.initialize_from_sequence(seq)
    return store


@pytest.fixture
def enabled_config() -> PositionUpdateConfig:
    return PositionUpdateConfig(enabled=True)


@pytest.fixture
def disabled_config() -> PositionUpdateConfig:
    return PositionUpdateConfig(enabled=False)


# ---------------------------------------------------------------------------
# PositionStore tests
# ---------------------------------------------------------------------------


class TestPositionStore:
    def test_initialize_from_sequence(self):
        store = PositionStore()
        seq = MDASequence(
            stage_positions=[
                {"x": 10, "y": 20, "z": 5},
                {"x": 30, "y": 40, "z": 15},
            ]
        )
        store.initialize_from_sequence(seq)

        assert store.num_positions == 2
        p0 = store.get_position(0)
        assert p0 is not None
        assert p0.x == 10.0
        assert p0.y == 20.0
        assert p0.z == 5.0
        p1 = store.get_position(1)
        assert p1 is not None
        assert p1.x == 30.0
        assert p1.y == 40.0
        assert p1.z == 15.0

    def test_initialize_with_none_coords_defaults_to_zero(self):
        store = PositionStore()
        seq = MDASequence(stage_positions=[{"x": 5.0}])
        store.initialize_from_sequence(seq)

        p = store.get_position(0)
        assert p is not None
        assert p.x == 5.0
        assert p.y == 0.0
        assert p.z is None

    def test_get_nonexistent_returns_none(self):
        store = PositionStore()
        assert store.get_position(99) is None

    def test_get_returns_copy(self, position_store):
        p = position_store.get_position(0)
        assert p is not None
        p.x = 9999.0
        p_again = position_store.get_position(0)
        assert p_again.x == 100.0

    def test_update_then_get(self, position_store):
        position_store.update_position(0, x=111.0, y=222.0, z=333.0)
        p = position_store.get_position(0)
        assert p.x == 111.0
        assert p.y == 222.0
        assert p.z == 333.0

    def test_get_all_positions(self, position_store):
        all_pos = position_store.get_all_positions()
        assert len(all_pos) == 3
        assert all_pos[0].x == 100.0
        assert all_pos[0].z == 50.0
        assert all_pos[2].x == 500.0
        assert all_pos[2].z == 70.0

    def test_thread_safety(self, position_store):
        """Concurrent reads and writes should not corrupt data."""
        errors = []

        def writer():
            try:
                for i in range(100):
                    position_store.update_position(0, x=float(i), y=float(i), z=float(i))
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    p = position_store.get_position(0)
                    assert p is not None
                    assert p.x == p.y == p.z
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"


# ---------------------------------------------------------------------------
# PositionUpdateManager tests
# ---------------------------------------------------------------------------


class TestPositionUpdateManager:
    def test_disabled_is_noop(self, disabled_config, position_store):
        manager = PositionUpdateManager(disabled_config, position_store)
        manager.start()
        assert manager._executor is None
        manager.on_position_complete(0, 0)
        manager.shutdown()

    def test_calls_updater_fn(self, enabled_config, position_store):
        called_with = {}

        def mock_updater(t_idx, p_idx, positions):
            called_with["t_idx"] = t_idx
            called_with["p_idx"] = p_idx
            called_with["positions"] = positions
            return positions[p_idx]

        manager = PositionUpdateManager(
            enabled_config, position_store, updater_fn=mock_updater
        )
        manager.start()
        manager.on_position_complete(0, 1)
        manager._pending_future.result(timeout=5)
        manager.shutdown()

        assert called_with["t_idx"] == 0
        assert called_with["p_idx"] == 1
        assert len(called_with["positions"]) == 3

    def test_updates_store_with_results(self, enabled_config, position_store):
        def shift_updater(t_idx, p_idx, positions):
            p = positions[p_idx]
            return PositionCoordinates(x=p.x + 10.0, y=p.y + 20.0, z=(p.z or 0) + 5.0)

        manager = PositionUpdateManager(
            enabled_config, position_store, updater_fn=shift_updater
        )
        manager.start()
        manager.on_position_complete(0, 0)
        manager._pending_future.result(timeout=5)
        manager.shutdown()

        p0 = position_store.get_position(0)
        assert p0.x == 110.0
        assert p0.y == 220.0
        assert p0.z == 55.0

    def test_updater_failure_preserves_positions(self, enabled_config, position_store):
        original = position_store.get_position(0)

        def failing_updater(t_idx, p_idx, positions):
            raise RuntimeError("updater crashed")

        manager = PositionUpdateManager(
            enabled_config, position_store, updater_fn=failing_updater
        )
        manager.start()
        manager.on_position_complete(0, 0)
        manager._pending_future.result(timeout=5)
        manager.shutdown()

        p0 = position_store.get_position(0)
        assert p0.x == original.x
        assert p0.y == original.y
        assert p0.z == original.z

    def test_shutdown_waits_for_pending(self, enabled_config, position_store):
        completed = threading.Event()

        def slow_updater(t_idx, p_idx, positions):
            time.sleep(0.2)
            completed.set()
            return positions[p_idx]

        manager = PositionUpdateManager(
            enabled_config, position_store, updater_fn=slow_updater
        )
        manager.start()
        manager.on_position_complete(0, 0)
        manager.shutdown()
        assert completed.is_set()

    def test_default_updater_returns_position_unchanged(self):
        positions = {
            0: PositionCoordinates(x=1.0, y=2.0, z=3.0),
            1: PositionCoordinates(x=4.0, y=5.0, z=6.0),
        }
        result = PositionUpdateManager._default_updater(0, 0, positions)
        assert result.x == 1.0
        assert result.y == 2.0
        assert result.z == 3.0


# ---------------------------------------------------------------------------
# MantisEngine integration tests
# ---------------------------------------------------------------------------


class TestMantisEnginePositionUpdate:
    def test_setup_sequence_initializes_position_update(self, engine, mock_core):
        seq = MDASequence(
            stage_positions=[
                {"x": 10, "y": 20, "z": 5},
                {"x": 30, "y": 40, "z": 15},
            ],
            metadata={"mantis": {"position_update": {"enabled": True}}},
        )
        with patch("shrimpy.mantis.mantis_engine.MDAEngine.setup_sequence"):
            engine.setup_sequence(seq)

        assert engine._position_update_manager is not None
        assert engine._position_update_manager.config.enabled is True
        assert engine._position_update_manager.position_store.num_positions == 2

    def test_setup_sequence_without_position_update(self, engine, mock_core):
        seq = MDASequence(
            stage_positions=[{"x": 10, "y": 20}],
            metadata={"mantis": {}},
        )
        with patch("shrimpy.mantis.mantis_engine.MDAEngine.setup_sequence"):
            engine.setup_sequence(seq)

        assert engine._position_update_manager is None

    def test_setup_sequence_position_update_disabled(self, engine, mock_core):
        seq = MDASequence(
            stage_positions=[{"x": 10, "y": 20}],
            metadata={"mantis": {"position_update": {"enabled": False}}},
        )
        with patch("shrimpy.mantis.mantis_engine.MDAEngine.setup_sequence"):
            engine.setup_sequence(seq)

        assert engine._position_update_manager is None

    def test_apply_position_update_returns_modified_event(self, engine):
        store = PositionStore()
        store.update_position(0, x=999.0, y=888.0, z=777.0)
        engine._position_update_manager = PositionUpdateManager(
            PositionUpdateConfig(enabled=True), store
        )

        event = MDAEvent(x_pos=100.0, y_pos=200.0, z_pos=300.0, index={"t": 0, "p": 0})
        result = engine._apply_position_update(event)

        assert result.x_pos == 999.0
        assert result.y_pos == 888.0
        assert result.z_pos == 777.0
        # Original event should be unchanged (frozen)
        assert event.x_pos == 100.0

    def test_apply_position_update_disabled_passes_event_through(self, engine):
        engine._position_update_manager = None
        event = MDAEvent(x_pos=100.0, y_pos=200.0, index={"t": 0, "p": 0})
        result = engine._apply_position_update(event)
        assert result is event

    def test_apply_position_update_no_position_index(self, engine):
        store = PositionStore()
        store.update_position(0, x=999.0, y=888.0, z=777.0)
        engine._position_update_manager = PositionUpdateManager(
            PositionUpdateConfig(enabled=True), store
        )

        event = MDAEvent(x_pos=100.0, y_pos=200.0, index={"t": 0})
        result = engine._apply_position_update(event)
        assert result is event

    def test_position_boundary_triggers_update(self, engine):
        """When (t, p) changes, update should fire for the previous position."""
        store = PositionStore()
        store.update_position(0, x=10.0, y=20.0, z=5.0)
        store.update_position(1, x=30.0, y=40.0, z=15.0)
        manager = PositionUpdateManager(PositionUpdateConfig(enabled=True), store)
        manager.start()
        engine._position_update_manager = manager
        engine._position_update_last_tp = (-1, -1)

        # First event at (t=0, p=0) — no update yet
        event_p0 = MDAEvent(index={"t": 0, "p": 0})
        engine._check_position_update_boundary(event_p0)
        assert manager._pending_future is None

        # First event at (t=0, p=1) — should trigger update for p=0
        event_p1 = MDAEvent(index={"t": 0, "p": 1})
        engine._check_position_update_boundary(event_p1)
        assert manager._pending_future is not None
        manager._pending_future.result(timeout=5)

        manager.shutdown()

    def test_timepoint_change_triggers_update(self, engine):
        """When t changes, update should fire for the last position."""
        store = PositionStore()
        store.update_position(0, x=10.0, y=20.0, z=5.0)
        manager = PositionUpdateManager(PositionUpdateConfig(enabled=True), store)
        manager.start()
        engine._position_update_manager = manager
        engine._position_update_last_tp = (0, 1)

        event_t1 = MDAEvent(index={"t": 1, "p": 0})
        engine._check_position_update_boundary(event_t1)
        assert manager._pending_future is not None
        manager._pending_future.result(timeout=5)

        manager.shutdown()

    def test_teardown_shuts_down_position_update(self, engine, mock_core):
        store = PositionStore()
        store.update_position(0, x=10.0, y=20.0, z=5.0)
        manager = PositionUpdateManager(PositionUpdateConfig(enabled=True), store)
        manager.start()
        engine._position_update_manager = manager
        engine._position_update_last_tp = (0, 1)

        with patch("shrimpy.mantis.mantis_engine.MDAEngine.teardown_sequence"):
            engine.teardown_sequence(MDASequence(metadata={"mantis": {}}))

        assert engine._position_update_manager is None
        assert manager._executor is None

    def test_setup_event_uses_updated_positions(self, engine, mock_core):
        """Full setup_event flow: updated position should reach _set_event_xy_position."""
        store = PositionStore()
        store.update_position(0, x=777.0, y=666.0, z=555.0)
        engine._position_update_manager = PositionUpdateManager(
            PositionUpdateConfig(enabled=True), store
        )
        engine._position_update_last_tp = (-1, -1)

        event = MDAEvent(x_pos=100.0, y_pos=200.0, z_pos=300.0, index={"t": 0, "p": 0})

        with patch("shrimpy.mantis.mantis_engine.MDAEngine.setup_event"):
            with patch.object(engine, "_set_event_xy_position") as mock_set_xy:
                with patch.object(engine, "_engage_autofocus"):
                    engine.setup_event(event)

        called_event = mock_set_xy.call_args[0][0]
        assert called_event.x_pos == 777.0
        assert called_event.y_pos == 666.0
        assert called_event.z_pos == 555.0


# ---------------------------------------------------------------------------
# Integration test with mock updater
# ---------------------------------------------------------------------------


class TestPositionUpdateIntegration:
    def test_positions_shift_across_acquisitions(self, demo_core, mantis_metadata):
        """End-to-end: a mock updater shifts position by (+1, +1, +0.5) per call.

        After each position's z-stack completes, the updater fires. By
        timepoint 2, positions should reflect accumulated shifts.
        """
        engine = MantisEngine(demo_core)

        def shift_updater(t_idx, p_idx, positions):
            p = positions[p_idx]
            return PositionCoordinates(x=p.x + 1.0, y=p.y + 1.0, z=(p.z or 0) + 0.5)

        mantis_metadata["position_update"] = {"enabled": True}
        seq = MDASequence(
            stage_positions=[
                {"x": 100, "y": 200, "z": 50},
                {"x": 300, "y": 400, "z": 60},
            ],
            time_plan={"interval": 0, "loops": 3},
            metadata={"mantis": mantis_metadata},
        )

        engine.setup_sequence(seq)
        engine._position_update_manager._updater_fn = shift_updater

        xy_positions: list[tuple[int, int, float, float]] = []

        @demo_core.mda.events.frameReady.connect
        def _on_frame(img, event, meta):
            t = event.index.get("t", 0)
            p = event.index.get("p", 0)
            x, y = demo_core.getXYPosition()
            xy_positions.append((t, p, x, y))

        demo_core.mda.run(seq)

        # Group by (t, p) and take the first frame's position for each
        seen = {}
        for t, p, x, y in xy_positions:
            if (t, p) not in seen:
                seen[(t, p)] = (x, y)

        # At t=0, positions should be the originals
        assert seen[(0, 0)] == pytest.approx((100.0, 200.0), abs=0.1)
        assert seen[(0, 1)] == pytest.approx((300.0, 400.0), abs=0.1)

        # By t=2, the updater should have run at least once per position,
        # shifting by +1 each time. Async, so at least one shift.
        x_t2_p0, y_t2_p0 = seen[(2, 0)]
        assert x_t2_p0 > 100.0, f"Expected x > 100 at t=2, got {x_t2_p0}"
        assert y_t2_p0 > 200.0, f"Expected y > 200 at t=2, got {y_t2_p0}"
