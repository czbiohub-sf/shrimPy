"""Tests for position update infrastructure.

Covers PositionStore, PositionUpdateManager, PositionUpdater,
and MantisEngine integration.
"""

from __future__ import annotations

import threading
import time
import weakref

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from useq import MDAEvent, MDASequence

from shrimpy.mantis.mantis_engine import MantisEngine
from shrimpy.mantis.position_update import (
    PositionCoordinates,
    PositionStore,
    PositionUpdateConfig,
    PositionUpdateManager,
    PositionUpdater,
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

    def test_subclass_receives_data(self):
        """A subclass can use the data parameter."""
        received_data = {}

        class TestUpdater(PositionUpdater):
            def update(self, t_idx, p_idx, position, data=None):
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


class TestPositionUpdateManager:
    def test_disabled_is_noop(self, disabled_config, position_store):
        manager = PositionUpdateManager(disabled_config, position_store)
        manager.start()
        assert manager._executor is None
        manager.on_position_complete(0, 0)
        manager.shutdown()

    def test_calls_updater(self, enabled_config, position_store):
        called_with = {}

        class SpyUpdater(PositionUpdater):
            def update(self, t_idx, p_idx, position, data=None):
                called_with["t_idx"] = t_idx
                called_with["p_idx"] = p_idx
                called_with["position"] = position
                called_with["data"] = data
                return position

        manager = PositionUpdateManager(enabled_config, position_store, updater=SpyUpdater())
        manager.start()
        frames = [np.zeros((4, 4), dtype=np.uint16)]
        manager.on_position_complete(0, 1, data=frames)
        manager._pending_future.result(timeout=5)
        manager.shutdown()

        assert called_with["t_idx"] == 0
        assert called_with["p_idx"] == 1
        assert called_with["position"].x == 300.0
        assert called_with["position"].y == 400.0
        assert called_with["data"] is frames

    def test_updates_store_with_results(self, enabled_config, position_store):
        class ShiftUpdater(PositionUpdater):
            def update(self, t_idx, p_idx, position, data=None):
                return PositionCoordinates(
                    x=position.x + 10.0,
                    y=position.y + 20.0,
                    z=(position.z or 0) + 5.0,
                )

        manager = PositionUpdateManager(enabled_config, position_store, updater=ShiftUpdater())
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

        class FailingUpdater(PositionUpdater):
            def update(self, t_idx, p_idx, position, data=None):
                raise RuntimeError("updater crashed")

        manager = PositionUpdateManager(
            enabled_config, position_store, updater=FailingUpdater()
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

        class SlowUpdater(PositionUpdater):
            def update(self, t_idx, p_idx, position, data=None):
                time.sleep(0.2)
                completed.set()
                return position

        manager = PositionUpdateManager(enabled_config, position_store, updater=SlowUpdater())
        manager.start()
        manager.on_position_complete(0, 0)
        manager.shutdown()
        assert completed.is_set()

    def test_apply_position_update_returns_modified_event(
        self, enabled_config, position_store
    ):
        manager = PositionUpdateManager(enabled_config, position_store)
        position_store.update_position(0, x=999.0, y=888.0, z=777.0)

        event = MDAEvent(x_pos=100.0, y_pos=200.0, z_pos=300.0, index={"t": 0, "p": 0})
        result = manager.apply_position_update(event)

        assert result.x_pos == 999.0
        assert result.y_pos == 888.0
        assert result.z_pos == 777.0
        # Original event should be unchanged (frozen)
        assert event.x_pos == 100.0

    def test_apply_position_update_no_position_index(self, enabled_config, position_store):
        manager = PositionUpdateManager(enabled_config, position_store)

        event = MDAEvent(x_pos=100.0, y_pos=200.0, index={"t": 0})
        result = manager.apply_position_update(event)
        assert result is event

    def test_apply_position_update_unknown_position(self, enabled_config):
        store = PositionStore()
        manager = PositionUpdateManager(enabled_config, store)

        event = MDAEvent(x_pos=100.0, y_pos=200.0, index={"t": 0, "p": 99})
        result = manager.apply_position_update(event)
        assert result is event


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

    def test_z_slice_count_triggers_update(self, engine):
        """on_position_complete fires when all z-slices for a position arrive."""
        store = PositionStore()
        store.update_position(0, x=10.0, y=20.0, z=5.0)
        manager = PositionUpdateManager(PositionUpdateConfig(enabled=True), store)
        manager.start()
        engine._position_update_manager = manager
        engine._position_update_frames = {}
        engine._position_update_expected_slices = 3

        frame = np.zeros((4, 4), dtype=np.uint16)

        # First two z-slices — not enough yet
        engine._on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 0, "c": 0, "z": 0}))
        engine._on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 0, "c": 0, "z": 1}))
        assert manager._pending_future is None

        # Third z-slice — should trigger update for p=0
        engine._on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 0, "c": 0, "z": 2}))
        assert manager._pending_future is not None
        manager._pending_future.result(timeout=5)

        # Buffer should be cleared for this (t, p)
        assert (0, 0) not in engine._position_update_frames
        manager.shutdown()

    def test_z_slice_count_passes_frames(self, engine):
        """All buffered z-slices should be passed to on_position_complete."""
        store = PositionStore()
        store.update_position(0, x=10.0, y=20.0, z=5.0)

        received_data = {}

        class SpyUpdater(PositionUpdater):
            def update(self, t_idx, p_idx, position, data=None):
                received_data["frames"] = data
                return position

        manager = PositionUpdateManager(
            PositionUpdateConfig(enabled=True), store, updater=SpyUpdater()
        )
        manager.start()
        engine._position_update_manager = manager
        engine._position_update_frames = {}
        engine._position_update_expected_slices = 2

        frame1 = np.ones((4, 4), dtype=np.uint16)
        frame2 = np.ones((4, 4), dtype=np.uint16) * 2
        engine._on_frame_ready(frame1, MDAEvent(index={"t": 0, "p": 0, "c": 0, "z": 0}))
        engine._on_frame_ready(frame2, MDAEvent(index={"t": 0, "p": 0, "c": 0, "z": 1}))
        manager._pending_future.result(timeout=5)

        assert len(received_data["frames"]) == 2
        assert np.array_equal(received_data["frames"][0], frame1)
        assert np.array_equal(received_data["frames"][1], frame2)
        manager.shutdown()

    def test_multiple_positions_update_independently(self, engine):
        """Each (t, p) accumulates slices and flushes independently."""
        store = PositionStore()
        store.update_position(0, x=10.0, y=20.0, z=5.0)
        store.update_position(1, x=30.0, y=40.0, z=15.0)
        manager = PositionUpdateManager(PositionUpdateConfig(enabled=True), store)
        manager.start()
        engine._position_update_manager = manager
        engine._position_update_frames = {}
        engine._position_update_expected_slices = 2

        frame = np.zeros((4, 4), dtype=np.uint16)

        # One slice each for p=0 and p=1 — neither complete yet
        engine._on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 0, "c": 0, "z": 0}))
        engine._on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 1, "c": 0, "z": 0}))
        assert manager._pending_future is None

        # Complete p=0
        engine._on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 0, "c": 0, "z": 1}))
        assert manager._pending_future is not None
        manager._pending_future.result(timeout=5)
        assert (0, 0) not in engine._position_update_frames
        # p=1 still buffering
        assert (0, 1) in engine._position_update_frames

        manager.shutdown()

    def test_teardown_shuts_down_position_update(self, engine, mock_core):
        store = PositionStore()
        store.update_position(0, x=10.0, y=20.0, z=5.0)
        manager = PositionUpdateManager(PositionUpdateConfig(enabled=True), store)
        manager.start()
        engine._position_update_manager = manager
        engine._position_update_frames = {}

        with patch("shrimpy.mantis.mantis_engine.MDAEngine.teardown_sequence"):
            engine.teardown_sequence(MDASequence(metadata={"mantis": {}}))

        assert engine._position_update_manager is None
        assert manager._executor is None

    def test_event_iterator_applies_position_updates(self, demo_core, mantis_metadata):
        """event_iterator should apply position updates before events are logged."""
        from shrimpy.mantis.mantis_engine import MantisEngine

        engine = MantisEngine(demo_core)

        store = PositionStore()
        store.update_position(0, x=777.0, y=666.0, z=555.0)
        engine._position_update_manager = PositionUpdateManager(
            PositionUpdateConfig(enabled=True), store
        )

        event = MDAEvent(x_pos=100.0, y_pos=200.0, z_pos=300.0, index={"t": 0, "p": 0})

        # event_iterator wraps super().event_iterator which does sequencing;
        # for a single event it just yields it back
        results = list(engine.event_iterator([event]))
        assert len(results) == 1
        updated = results[0]
        assert updated.x_pos == 777.0
        assert updated.y_pos == 666.0
        assert updated.z_pos == 555.0

    def test_on_frame_ready_buffers_frames(self, engine):
        """_on_frame_ready should buffer frame copies when position update is active."""
        store = PositionStore()
        engine._position_update_manager = PositionUpdateManager(
            PositionUpdateConfig(enabled=True), store
        )
        engine._position_update_frames = {}
        engine._position_update_expected_slices = 5  # high so no flush

        frame = np.ones((4, 4), dtype=np.uint16) * 42
        event = MDAEvent(index={"t": 0, "p": 0, "c": 0})

        engine._on_frame_ready(frame, event)

        # Frame should be buffered (as a copy)
        buffered = engine._position_update_frames[(0, 0)]
        assert len(buffered) == 1
        assert np.array_equal(buffered[0], frame)
        # Verify it's a copy, not the same object
        assert buffered[0] is not frame

    def test_on_frame_ready_default_caches_channel_0_only(self, engine):
        """Default update_channel=0 should only buffer frames from channel 0."""
        store = PositionStore()
        config = PositionUpdateConfig(enabled=True)  # default update_channel=0
        engine._position_update_manager = PositionUpdateManager(config, store)
        engine._position_update_frames = {}
        engine._position_update_expected_slices = 5

        frame = np.ones((4, 4), dtype=np.uint16)

        # Channel 0 — should be buffered
        engine._on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 0, "c": 0}))
        assert len(engine._position_update_frames.get((0, 0), [])) == 1

        # Channel 1 — should be skipped
        engine._on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 0, "c": 1}))
        assert len(engine._position_update_frames.get((0, 0), [])) == 1

    def test_on_frame_ready_filters_by_channel(self, engine):
        """Only frames from the configured update_channel should be buffered."""
        store = PositionStore()
        config = PositionUpdateConfig(enabled=True, update_channel=1)
        engine._position_update_manager = PositionUpdateManager(config, store)
        engine._position_update_frames = {}
        engine._position_update_expected_slices = 5

        frame = np.ones((4, 4), dtype=np.uint16)

        # Channel 0 frame — should be skipped
        engine._on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 0, "c": 0}))
        assert (0, 0) not in engine._position_update_frames

        # Channel 1 frame — should be buffered
        engine._on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 0, "c": 1}))
        assert len(engine._position_update_frames[(0, 0)]) == 1

        # Channel 2 frame — should be skipped
        engine._on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 0, "c": 2}))
        assert len(engine._position_update_frames[(0, 0)]) == 1

    def test_on_frame_ready_all_channels_when_none(self, engine):
        """When update_channel is None, all channels should be buffered."""
        store = PositionStore()
        config = PositionUpdateConfig(enabled=True, update_channel=None)
        engine._position_update_manager = PositionUpdateManager(config, store)
        engine._position_update_frames = {}
        engine._position_update_expected_slices = 5

        frame = np.ones((4, 4), dtype=np.uint16)
        engine._on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 0, "c": 0}))
        engine._on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 0, "c": 1}))
        assert len(engine._position_update_frames[(0, 0)]) == 2

    def test_on_frame_ready_no_buffer_when_disabled(self, engine):
        """_on_frame_ready should not buffer frames when position update is disabled."""
        engine._position_update_manager = None
        engine._position_update_frames = {}

        frame = np.ones((4, 4), dtype=np.uint16)
        event = MDAEvent(index={"t": 0, "p": 0})

        engine._on_frame_ready(frame, event)

        assert len(engine._position_update_frames) == 0


# ---------------------------------------------------------------------------
# Integration test with mock updater
# ---------------------------------------------------------------------------


class TestBackpressure:
    """Verify that position update drains pending work at timepoint boundaries.

    Without backpressure, a slow updater causes frame data to accumulate
    unboundedly in the ThreadPoolExecutor queue — one full z-stack per
    position per timepoint. This eventually exhausts RAM and triggers
    MemoryError: Buffer overflowed from the Micro-Manager circular buffer.
    """

    def test_slow_updater_queue_bounded_across_timepoints(self, engine):
        """Pending updates must be drained between timepoints.

        Simulate a slow updater (0.5s per position) across 3 timepoints
        with 3 positions and 2 z-slices each. Record when each event is
        *yielded* by event_iterator and when each update *completes*.

        Without backpressure, event_iterator yields timepoint 1 events
        immediately while timepoint 0 updates are still running on the
        executor. With drain, event_iterator blocks at the boundary until
        the pending update finishes, so all t=0 updates complete before
        any t=1 event is yielded.
        """
        store = PositionStore()
        for i in range(3):
            store.update_position(i, x=float(i * 100), y=float(i * 100), z=0.0)

        update_completions: list[tuple[int, int, float]] = []
        event_yields: list[tuple[int, int, float]] = []

        class SlowUpdater(PositionUpdater):
            def update(self, t_idx, p_idx, position, data=None):
                time.sleep(0.5)
                update_completions.append((t_idx, p_idx, time.monotonic()))
                return position

        config = PositionUpdateConfig(enabled=True, update_channel=0)
        manager = PositionUpdateManager(config, store, updater=SlowUpdater())
        manager.start()

        engine._position_update_manager = manager
        engine._position_update_frames = {}
        engine._position_update_expected_slices = 2

        # Build events: 3 timepoints × 3 positions × 2 z-slices
        events = []
        for t in range(3):
            for p in range(3):
                for z in range(2):
                    events.append(MDAEvent(
                        index={"t": t, "p": p, "c": 0, "z": z},
                        x_pos=float(p * 100),
                        y_pos=float(p * 100),
                    ))

        frame = np.zeros((64, 64), dtype=np.uint16)

        with patch("shrimpy.mantis.mantis_engine.MDAEngine.event_iterator", return_value=iter(events)):
            for event in engine.event_iterator(events):
                t_idx = event.index.get("t", 0)
                p_idx = event.index.get("p", 0)
                event_yields.append((t_idx, p_idx, time.monotonic()))
                engine._on_frame_ready(frame, event)

        manager.shutdown()

        # All 9 updates should have completed (3t × 3p)
        assert len(update_completions) == 9

        # The critical check: when event_iterator yields the first event
        # of timepoint 1, all timepoint 0 updates must already be done.
        t0_last_completion = max(ts for t, p, ts in update_completions if t == 0)
        t1_first_yield = min(ts for t, p, ts in event_yields if t == 1)

        assert t0_last_completion < t1_first_yield, (
            f"Timepoint 0 last update completed at {t0_last_completion:.3f}, "
            f"but timepoint 1 first event yielded at {t1_first_yield:.3f} — "
            "event_iterator is not draining pending updates at timepoint boundary"
        )

    def test_executor_queue_depth_bounded(self, engine):
        """The executor should not accumulate more than one pending future.

        Without backpressure, on_position_complete submits to the executor
        without waiting — the internal queue grows by one z-stack per
        position. With drain at timepoint boundaries, the queue is flushed
        before new work is submitted.

        We measure this by counting how many futures are submitted while
        previous ones are still pending.
        """
        store = PositionStore()
        for i in range(3):
            store.update_position(i, x=float(i * 100), y=float(i * 100), z=0.0)

        pending_at_submit: list[int] = []
        _orig_submit = PositionUpdateManager.on_position_complete

        class SlowUpdater(PositionUpdater):
            def update(self, t_idx, p_idx, position, data=None):
                time.sleep(0.3)
                return position

        config = PositionUpdateConfig(enabled=True, update_channel=0)
        manager = PositionUpdateManager(config, store, updater=SlowUpdater())
        manager.start()

        # Monkey-patch to track whether a future is still pending at submit time
        orig_on_position_complete = manager.on_position_complete.__func__

        def tracking_on_position_complete(self_mgr, t_idx, p_idx, data=None):
            if self_mgr._pending_future is not None and not self_mgr._pending_future.done():
                pending_at_submit.append(1)
            else:
                pending_at_submit.append(0)
            return orig_on_position_complete(self_mgr, t_idx, p_idx, data)

        manager.on_position_complete = lambda t, p, data=None: tracking_on_position_complete(manager, t, p, data)

        engine._position_update_manager = manager
        engine._position_update_frames = {}
        engine._position_update_expected_slices = 1

        # 4 timepoints × 3 positions × 1 z-slice
        events = []
        for t in range(4):
            for p in range(3):
                events.append(MDAEvent(
                    index={"t": t, "p": p, "c": 0, "z": 0},
                    x_pos=float(p * 100),
                    y_pos=float(p * 100),
                ))

        frame = np.zeros((64, 64), dtype=np.uint16)

        with patch("shrimpy.mantis.mantis_engine.MDAEngine.event_iterator", return_value=iter(events)):
            for event in engine.event_iterator(events):
                engine._on_frame_ready(frame, event)

        manager.shutdown()

        # Within a single timepoint, overlaps are expected: 3 positions
        # submit faster than the 0.3s updater processes them, so positions
        # 2 and 3 find position 1 still pending. That's fine — the drain
        # only fires between timepoints.
        #
        # Without drain: nearly all submissions overlap (11/12).
        # With drain: at most (positions_per_timepoint - 1) per timepoint,
        # i.e. 2 × 4 = 8 out of 12. The key is no cross-timepoint
        # accumulation — the queue never holds more than one timepoint's
        # worth of work.
        n_positions = 3
        n_timepoints = 4
        max_expected_overlaps = (n_positions - 1) * n_timepoints
        overlaps = sum(pending_at_submit)
        total = len(pending_at_submit)
        assert overlaps <= max_expected_overlaps, (
            f"{overlaps}/{total} submissions found a pending future "
            f"(expected at most {max_expected_overlaps}) — "
            "executor queue is accumulating across timepoints"
        )


class TestPositionUpdateIntegration:
    def test_positions_shift_across_acquisitions(self, demo_core, mantis_metadata):
        """End-to-end: a mock updater shifts position by (+1, +1, +0.5) per call.

        After each position's z-stack completes, the updater fires. By
        timepoint 2, positions should reflect accumulated shifts.
        """
        engine = MantisEngine(demo_core)

        class ShiftUpdater(PositionUpdater):
            def update(self, t_idx, p_idx, position, data=None):
                return PositionCoordinates(
                    x=position.x + 1.0,
                    y=position.y + 1.0,
                    z=(position.z or 0) + 0.5,
                )

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
        engine._position_update_manager._updater = ShiftUpdater()

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

    def test_dynatrack_created_from_metadata(self, demo_core, mantis_metadata):
        """DynaTrackUpdater is created when dynatrack config is in metadata."""
        from shrimpy.mantis.dynatrack import DynaTrackUpdater

        engine = MantisEngine(demo_core)

        mantis_metadata["position_update"] = {
            "enabled": True,
            "update_channel": 0,
            "dynatrack": {
                "scale_yx": 0.075,
                "scale_z": 0.174,
                "dampening": (0.5, 0.8, 0.8),
                "shift_limits": {"z": (0.5, 2.0), "y": (2.0, 10.0), "x": (2.0, 10.0)},
                "tracking_interval": 2,
            },
        }
        seq = MDASequence(
            stage_positions=[{"x": 100, "y": 200, "z": 50}],
            metadata={"mantis": mantis_metadata},
        )

        engine.setup_sequence(seq)

        assert engine._position_update_manager is not None
        updater = engine._position_update_manager._updater
        assert isinstance(updater, DynaTrackUpdater)
        assert updater.config.scale_yx == 0.075
        assert updater.config.scale_z == 0.174
        assert updater.config.tracking_interval == 2
        assert updater.config.dampening == (0.5, 0.8, 0.8)

        engine.teardown_sequence(seq)

    def test_no_dynatrack_uses_default_updater(self, demo_core, mantis_metadata):
        """Without dynatrack config, the default no-op updater is used."""
        from shrimpy.mantis.dynatrack import DynaTrackUpdater

        engine = MantisEngine(demo_core)

        mantis_metadata["position_update"] = {"enabled": True}
        seq = MDASequence(
            stage_positions=[{"x": 100, "y": 200, "z": 50}],
            metadata={"mantis": mantis_metadata},
        )

        engine.setup_sequence(seq)

        assert engine._position_update_manager is not None
        updater = engine._position_update_manager._updater
        assert not isinstance(updater, DynaTrackUpdater)

        engine.teardown_sequence(seq)
