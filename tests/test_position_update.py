"""Tests for the position-update infrastructure backing DynaTrack.

Covers PositionStore, PositionUpdater, and PositionUpdateManager. The
engine-facing DynaTrack coordinator is tested in test_dynatrack_manager.py.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from useq import MDAEvent, MDASequence

from shrimpy.dynatrack.position_update import (
    PositionCoordinates,
    PositionStore,
    PositionUpdateManager,
    PositionUpdater,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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

    def test_initialize_reads_z_from_device_property(self):
        """When z_device is set, the initial Z comes from position properties."""
        store = PositionStore()
        seq = MDASequence(
            stage_positions=[
                {"x": 1, "y": 2, "z": 0, "properties": [["ObjectiveZ", "Position", "42"]]},
            ]
        )
        store.initialize_from_sequence(seq, z_device="ObjectiveZ")
        assert store.get_position(0).z == 42.0

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
# PositionUpdater tests
# ---------------------------------------------------------------------------


class TestPositionUpdater:
    def test_default_updater_returns_position_unchanged(self):
        updater = PositionUpdater()
        pos = PositionCoordinates(x=1.0, y=2.0, z=3.0)
        result = updater.update(0, 0, pos)
        assert result.x == 1.0
        assert result.y == 2.0
        assert result.z == 3.0

    def test_default_updater_ignores_data(self):
        updater = PositionUpdater()
        pos = PositionCoordinates(x=1.0, y=2.0, z=3.0)
        frames = [np.zeros((10, 10), dtype=np.uint16)]
        result = updater.update(0, 0, pos, data=frames)
        assert result.x == 1.0

    def test_default_updater_does_not_want_refresh(self):
        assert PositionUpdater().wants_reference_refresh(0) is False

    def test_subclass_receives_data(self):
        """A subclass can use the data parameter."""
        received_data = {}

        class TestUpdater(PositionUpdater):
            def update(self, t_idx, p_idx, position, data=None, **kwargs):
                received_data["frames"] = data
                return position

        updater = TestUpdater()
        pos = PositionCoordinates(x=1.0, y=2.0, z=3.0)
        frames = [np.ones((5, 5), dtype=np.uint16) * 42]
        updater.update(0, 0, pos, data=frames)

        assert received_data["frames"] is frames
        assert received_data["frames"][0][0, 0] == 42


# ---------------------------------------------------------------------------
# PositionUpdateManager tests
# ---------------------------------------------------------------------------


def _record_baseline(manager, t, p):
    """Simulate the event iterator recording the acquisition baseline for
    (t, p) before the stack's frames complete. The real flow always does this
    via ``apply_position_update``; without a baseline the manager now skips the
    correction rather than anchoring it to a race-prone live-store value.
    """
    manager.apply_position_update(MDAEvent(index={"t": t, "p": p}))


class TestPositionUpdateManager:
    def test_calls_updater(self, position_store):
        called_with = {}

        class SpyUpdater(PositionUpdater):
            def update(self, t_idx, p_idx, position, data=None, **kwargs):
                called_with["t_idx"] = t_idx
                called_with["p_idx"] = p_idx
                called_with["position"] = position
                called_with["data"] = data
                return position

        manager = PositionUpdateManager(position_store, updater=SpyUpdater())
        manager.start()
        _record_baseline(manager, 0, 1)
        frames = [np.zeros((4, 4), dtype=np.uint16)]
        manager.on_position_complete(0, 1, data=frames)
        manager._pending_future.result(timeout=5)
        manager.shutdown()

        assert called_with["t_idx"] == 0
        assert called_with["p_idx"] == 1
        assert called_with["position"].x == 300.0
        assert called_with["position"].y == 400.0
        assert called_with["data"] is frames

    def test_noop_before_start(self, position_store):
        """on_position_complete is a no-op until start() creates the executor."""
        manager = PositionUpdateManager(position_store)
        _record_baseline(manager, 0, 0)
        manager.on_position_complete(0, 0)
        assert manager._pending_future is None

    def test_updates_store_with_results(self, position_store):
        class ShiftUpdater(PositionUpdater):
            def update(self, t_idx, p_idx, position, data=None, **kwargs):
                return PositionCoordinates(
                    x=position.x + 10.0,
                    y=position.y + 20.0,
                    z=(position.z or 0) + 5.0,
                )

        manager = PositionUpdateManager(position_store, updater=ShiftUpdater())
        manager.start()
        _record_baseline(manager, 0, 0)
        manager.on_position_complete(0, 0)
        manager._pending_future.result(timeout=5)
        manager.shutdown()

        p0 = position_store.get_position(0)
        assert p0.x == 110.0
        assert p0.y == 220.0
        assert p0.z == 55.0

    def test_updater_failure_preserves_positions(self, position_store):
        original = position_store.get_position(0)

        class FailingUpdater(PositionUpdater):
            def update(self, t_idx, p_idx, position, data=None, **kwargs):
                raise RuntimeError("updater crashed")

        manager = PositionUpdateManager(position_store, updater=FailingUpdater())
        manager.start()
        _record_baseline(manager, 0, 0)
        manager.on_position_complete(0, 0)
        manager._pending_future.result(timeout=5)
        manager.shutdown()

        p0 = position_store.get_position(0)
        assert p0.x == original.x
        assert p0.y == original.y
        assert p0.z == original.z

    def test_shutdown_waits_for_pending(self, position_store):
        completed = threading.Event()

        class SlowUpdater(PositionUpdater):
            def update(self, t_idx, p_idx, position, data=None, **kwargs):
                time.sleep(0.2)
                completed.set()
                return position

        manager = PositionUpdateManager(position_store, updater=SlowUpdater())
        manager.start()
        _record_baseline(manager, 0, 0)
        manager.on_position_complete(0, 0)
        manager.shutdown()
        assert completed.is_set()

    def test_apply_position_update_returns_modified_event(self, position_store):
        manager = PositionUpdateManager(position_store)
        position_store.update_position(0, x=999.0, y=888.0, z=777.0)

        event = MDAEvent(x_pos=100.0, y_pos=200.0, z_pos=300.0, index={"t": 0, "p": 0})
        result = manager.apply_position_update(event)

        assert result.x_pos == 999.0
        assert result.y_pos == 888.0
        assert result.z_pos == 777.0
        # Original event should be unchanged (frozen)
        assert event.x_pos == 100.0

    def test_apply_position_update_writes_z_to_device_property(self, position_store):
        """With z_device set, Z is written to the device property, not z_pos."""
        manager = PositionUpdateManager(position_store, z_device="ObjectiveZ")
        position_store.update_position(0, x=999.0, y=888.0, z=777.0)

        event = MDAEvent(x_pos=1.0, y_pos=2.0, z_pos=3.0, index={"t": 0, "p": 0})
        result = manager.apply_position_update(event)

        assert result.z_pos == 3.0  # untouched
        assert ("ObjectiveZ", "Position", 777.0) in result.properties

    def test_apply_position_update_no_position_index(self, position_store):
        manager = PositionUpdateManager(position_store)

        event = MDAEvent(x_pos=100.0, y_pos=200.0, index={"t": 0})
        result = manager.apply_position_update(event)
        assert result is event

    def test_apply_position_update_unknown_position(self):
        store = PositionStore()
        manager = PositionUpdateManager(store)

        event = MDAEvent(x_pos=100.0, y_pos=200.0, index={"t": 0, "p": 99})
        result = manager.apply_position_update(event)
        assert result is event

    def test_updater_baseline_is_acquired_coords_not_advanced_store(self, position_store):
        """Regression: the shift must be anchored to the coords the stack was
        acquired at, not to a store value a later correction has moved on to.

        Reproduces the pre-fetch race that commit a98c22c fixed: if the store
        is updated between event-apply (acquisition) and on_position_complete,
        the updater must still receive the acquisition baseline so corrections
        don't accumulate against a moving target.
        """
        seen = {}

        class SpyUpdater(PositionUpdater):
            def update(self, t_idx, p_idx, position, data=None):
                seen["position"] = position
                return position

        manager = PositionUpdateManager(position_store, updater=SpyUpdater())
        manager.start()

        # 1. Event for (t=0, p=0) is applied -> baseline recorded from the
        #    store as it stood at acquisition time (100, 200, 50).
        event = MDAEvent(x_pos=0.0, y_pos=0.0, z_pos=0.0, index={"t": 0, "p": 0})
        manager.apply_position_update(event)

        # 2. A later correction lands in the store before this stack completes.
        position_store.update_position(0, x=999.0, y=888.0, z=777.0)

        # 3. Stack completes -> updater must see the acquisition baseline.
        manager.on_position_complete(0, 0)
        manager._pending_future.result(timeout=5)
        manager.shutdown()

        assert seen["position"].x == 100.0
        assert seen["position"].y == 200.0
        assert seen["position"].z == 50.0

    def test_no_baseline_skips_correction(self, position_store):
        """When no acquisition baseline was recorded (event never passed
        through apply_position_update), the correction is skipped rather than
        anchored to a race-prone live-store value. Because shifts anchor to a
        fixed reference, the next timepoint re-centers, so skipping is safe.
        """
        seen = {}

        class SpyUpdater(PositionUpdater):
            def update(self, t_idx, p_idx, position, data=None):
                seen["position"] = position
                return position

        manager = PositionUpdateManager(position_store, updater=SpyUpdater())
        manager.start()
        # No apply_position_update call -> no baseline recorded for (0, 0).
        manager.on_position_complete(0, 0)
        # Nothing is submitted and the updater is never called.
        assert manager._pending_future is None
        assert "position" not in seen
        manager.shutdown()

    def test_no_baseline_proceeds_on_reference_refresh(self, position_store):
        """A missing baseline skips only a *correction*, not a scheduled
        reference refresh. When the updater reports this is a refresh timepoint
        (wants_reference_refresh -> True), the update still runs: the refresh
        applies no correction, so the live-store snapshot is a harmless baseline
        and the reference must still be re-anchored.
        """
        seen = {}

        class RefreshUpdater(PositionUpdater):
            def wants_reference_refresh(self, timepoint_index):
                return True

            def update(self, t_idx, p_idx, position, data=None):
                seen["position"] = position
                return position

        manager = PositionUpdateManager(position_store, updater=RefreshUpdater())
        manager.start()
        # No apply_position_update call -> no baseline recorded for (0, 0), but
        # this is a refresh timepoint, so the updater must still run.
        manager.on_position_complete(0, 0)
        assert manager._pending_future is not None
        manager._pending_future.result(timeout=5)
        manager.shutdown()
        # Proceeded using the live-store snapshot as the baseline.
        assert seen["position"].x == 100.0
        assert seen["position"].y == 200.0
