"""Tests for the engine-facing DynaTrack coordinator and its MantisEngine wiring.

The coordinator's tracking work normally runs in a worker subprocess; these
tests inject a lightweight in-process updater so no subprocess (or torch) is
needed. The real worker-spawn path is exercised indirectly by the DynaTrack
unit tests.
"""

from __future__ import annotations

import time
import weakref

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from useq import MDAEvent, MDASequence

from shrimpy.dynatrack import DynaTrack, DynaTrackConfig
from shrimpy.dynatrack.position_update import PositionCoordinates, PositionUpdater
from shrimpy.mantis.mantis_engine import MantisEngine

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def engine(mock_core: MagicMock) -> MantisEngine:
    """Create a MantisEngine wired to the mock CMMCorePlus."""
    with patch("shrimpy.base_engine.MDAEngine.__init__", return_value=None):
        eng = MantisEngine(mock_core)
    eng._mmcore_ref = weakref.ref(mock_core)
    return eng


def _sequence(
    n_positions: int = 1, channels: tuple[str, ...] = ("ch0", "ch1", "ch2")
) -> MDASequence:
    return MDASequence(
        channels=[{"config": c, "group": "Channel"} for c in channels],
        # z_plan.step is the single source of truth for the Z scale (from_metadata).
        z_plan={"top": 1.0, "bottom": -1.0, "step": 0.5},
        stage_positions=[
            {"x": float(i * 100), "y": float(i * 100 + 100), "z": float(i + 5)}
            for i in range(n_positions)
        ],
    )


def _make_dynatrack(
    sequence: MDASequence,
    updater: PositionUpdater,
    input_channel: str = "ch0",
    expected_slices: int = 1,
) -> DynaTrack:
    """Build an in-process DynaTrack (no worker subprocess) and start it."""
    # Raw (non-VS) tracking: tracking_channel must be a valid input channel.
    config = DynaTrackConfig(input_channel=input_channel, tracking_channel="ch0")
    dt = DynaTrack(config, sequence, updater=updater)
    dt._expected_slices = expected_slices
    dt.start()
    return dt


# ---------------------------------------------------------------------------
# DynaTrack.from_metadata
# ---------------------------------------------------------------------------


class TestFromMetadata:
    def test_none_when_meta_absent(self):
        assert DynaTrack.from_metadata(None, _sequence(), pixel_size_um=0.1) is None
        assert DynaTrack.from_metadata({}, _sequence(), pixel_size_um=0.1) is None

    def test_none_when_disabled(self):
        meta = {"enabled": False}
        assert DynaTrack.from_metadata(meta, _sequence(), pixel_size_um=0.1) is None

    def test_none_without_stage_positions(self):
        meta = {"enabled": True}
        assert DynaTrack.from_metadata(meta, MDASequence(), pixel_size_um=0.1) is None

    def test_builds_config_from_metadata(self):
        meta = {
            "enabled": True,
            "input_channel": "ch1",
            "tracking_channel": "ch1",
            "z_device": "ObjectiveZ",
            "tracking_interval": 2,
            "shift": {"dampening": (0.5, 0.8, 0.8)},
        }
        dt = DynaTrack.from_metadata(meta, _sequence(2), pixel_size_um=0.075)
        assert dt is not None
        assert dt.num_positions == 2
        assert dt.config.input_channel == "ch1"
        assert dt._input_channel_index == 1
        assert dt.config.z_device == "ObjectiveZ"
        # scales are derived, not config fields
        assert dt._pixel_size_um == 0.075
        assert dt._z_step_um == 0.5  # _sequence z_plan step
        assert dt.config.tracking_interval == 2
        assert dt.config.shift.dampening == (0.5, 0.8, 0.8)

    def test_derived_scales_injected_into_deskew_and_phase(self):
        meta = {
            "enabled": True,
            "input_channel": "ch0",
            "tracking_channel": "ch0",
            "preprocessing": ["deskew", "phase"],
            "deskew": {"ls_angle_deg": 30.0, "keep_overhang": False},
            "phase": {"transfer_function": {"wavelength_illumination": 0.45}},
        }
        dt = DynaTrack.from_metadata(meta, _sequence(), pixel_size_um=0.11)
        assert dt.config.deskew["pixel_size_um"] == 0.11
        assert dt.config.deskew["scan_step_um"] == 0.5  # z_plan step
        tf = dt.config.phase["transfer_function"]
        assert tf["yx_pixel_size"] == 0.11
        assert tf["z_pixel_size"] == 0.5

    def test_raises_when_pixel_size_unset(self):
        meta = {"enabled": True, "input_channel": "ch0", "tracking_channel": "ch0"}
        with pytest.raises(ValueError, match="pixel size is not set"):
            DynaTrack.from_metadata(meta, _sequence(), pixel_size_um=0)

    def test_raises_when_z_plan_has_no_step(self):
        meta = {"enabled": True, "input_channel": "ch0", "tracking_channel": "ch0"}
        seq = MDASequence(
            channels=[{"config": "ch0", "group": "Channel"}],
            stage_positions=[{"x": 0, "y": 0, "z": 0}],
        )  # no z_plan
        with pytest.raises(ValueError, match="z_plan has no step"):
            DynaTrack.from_metadata(meta, seq, pixel_size_um=0.1)

    def test_sets_shift_log_path_from_data_path(self, tmp_path):
        meta = {"enabled": True, "input_channel": "ch0", "tracking_channel": "ch0"}
        dt = DynaTrack.from_metadata(meta, _sequence(), data_path=tmp_path, pixel_size_um=0.1)
        assert dt.config.shift_log_path == str(tmp_path / "dynatrack_log.csv")

    def test_explicit_shift_log_path_wins(self, tmp_path):
        meta = {
            "enabled": True,
            "input_channel": "ch0",
            "tracking_channel": "ch0",
            "shift_log_path": "/custom/log.csv",
        }
        dt = DynaTrack.from_metadata(meta, _sequence(), data_path=tmp_path, pixel_size_um=0.1)
        assert dt.config.shift_log_path == "/custom/log.csv"

    def test_unknown_input_channel_raises(self):
        """input_channel must match a channel in the sequence."""
        meta = {"enabled": True, "tracking_channel": "ch0", "input_channel": "NOPE"}
        with pytest.raises(ValueError, match="input_channel 'NOPE'"):
            DynaTrack.from_metadata(meta, _sequence(), pixel_size_um=0.1)

    def test_input_channel_is_required(self):
        """input_channel has no default; omitting it is a pydantic error."""
        import pydantic

        meta = {"enabled": True, "tracking_channel": "ch0"}
        with pytest.raises(pydantic.ValidationError):
            DynaTrack.from_metadata(meta, _sequence(), pixel_size_um=0.1)


# ---------------------------------------------------------------------------
# tracking_channel validation
# ---------------------------------------------------------------------------


class TestTrackingChannelValidation:
    def _build(self, **cfg_kwargs):
        cfg_kwargs.setdefault("input_channel", "BF")
        config = DynaTrackConfig(**cfg_kwargs)
        return DynaTrack(config, _sequence(channels=("BF", "GFP")), updater=PositionUpdater())

    @pytest.mark.parametrize("bad", ["phase", "deskewed", "vs_nuclei", "vs_membrane"])
    def test_reserved_names_rejected(self, bad):
        with pytest.raises(ValueError, match="not allowed"):
            self._build(tracking_channel=bad)

    def test_non_vs_must_be_input_channel(self):
        # A valid input channel is accepted (raw pipeline, no preprocessing).
        dt = self._build(tracking_channel="BF")
        assert dt.config.tracking_channel == "BF"
        # A name that is not an acquisition channel is rejected.
        with pytest.raises(ValueError, match="acquisition channels"):
            self._build(tracking_channel="nuclei")

    def test_vs_must_be_target_channel(self):
        common = {
            "preprocessing": ["deskew", "phase", "vs"],
            "virtual_staining": {"target_channels": ["nuclei", "membrane"]},
            "input_channel": "BF",
        }
        dt = self._build(tracking_channel="nuclei", **common)
        assert dt.config.tracking_channel == "nuclei"
        # An input channel name is not a valid VS target.
        with pytest.raises(ValueError, match="target_channels"):
            self._build(tracking_channel="BF", **common)

    def test_tracking_channel_is_required(self):
        """tracking_channel has no default; omitting it is a pydantic error."""
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            DynaTrackConfig(input_channel="BF")


# ---------------------------------------------------------------------------
# Frame buffering / on_frame_ready
# ---------------------------------------------------------------------------


class TestFrameBuffering:
    def test_z_slice_count_triggers_update(self):
        """on_position_complete fires when all z-slices for a position arrive."""
        seen = {}

        class SpyUpdater(PositionUpdater):
            def update(self, t_idx, p_idx, position, data=None, **kwargs):
                seen["data"] = data
                return position

        dt = _make_dynatrack(_sequence(), SpyUpdater(), expected_slices=3)
        dt.apply_position_update(MDAEvent(index={"t": 0, "p": 0}))
        frame = np.zeros((4, 4), dtype=np.uint16)

        dt.on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 0, "c": 0, "z": 0}))
        dt.on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 0, "c": 0, "z": 1}))
        assert dt._manager._pending_future is None

        dt.on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 0, "c": 0, "z": 2}))
        assert dt._manager._pending_future is not None
        dt._manager._pending_future.result(timeout=5)
        assert (0, 0) not in dt._frames
        assert len(seen["data"]) == 3
        dt.shutdown()

    def test_passes_all_buffered_frames(self):
        received = {}

        class SpyUpdater(PositionUpdater):
            def update(self, t_idx, p_idx, position, data=None, **kwargs):
                received["frames"] = data
                return position

        dt = _make_dynatrack(_sequence(), SpyUpdater(), expected_slices=2)
        dt.apply_position_update(MDAEvent(index={"t": 0, "p": 0}))
        frame1 = np.ones((4, 4), dtype=np.uint16)
        frame2 = np.ones((4, 4), dtype=np.uint16) * 2
        dt.on_frame_ready(frame1, MDAEvent(index={"t": 0, "p": 0, "c": 0, "z": 0}))
        dt.on_frame_ready(frame2, MDAEvent(index={"t": 0, "p": 0, "c": 0, "z": 1}))
        dt._manager._pending_future.result(timeout=5)

        assert len(received["frames"]) == 2
        assert np.array_equal(received["frames"][0], frame1)
        assert np.array_equal(received["frames"][1], frame2)
        dt.shutdown()

    def test_buffers_frame_copies(self):
        dt = _make_dynatrack(_sequence(), PositionUpdater(), expected_slices=5)
        frame = np.ones((4, 4), dtype=np.uint16) * 42
        dt.on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 0, "c": 0}))

        buffered = dt._frames[(0, 0)]
        assert len(buffered) == 1
        assert np.array_equal(buffered[0], frame)
        assert buffered[0] is not frame  # copy, not the same object
        dt.shutdown()

    def test_default_caches_first_channel_only(self):
        dt = _make_dynatrack(
            _sequence(), PositionUpdater(), input_channel="ch0", expected_slices=5
        )
        frame = np.ones((4, 4), dtype=np.uint16)

        dt.on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 0, "c": 0}))
        assert len(dt._frames.get((0, 0), [])) == 1
        dt.on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 0, "c": 1}))
        assert len(dt._frames.get((0, 0), [])) == 1  # channel 1 skipped
        dt.shutdown()

    def test_filters_by_configured_channel(self):
        # "ch1" resolves to channel index 1 in the sequence.
        dt = _make_dynatrack(
            _sequence(), PositionUpdater(), input_channel="ch1", expected_slices=5
        )
        frame = np.ones((4, 4), dtype=np.uint16)

        dt.on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 0, "c": 0}))
        assert (0, 0) not in dt._frames
        dt.on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 0, "c": 1}))
        assert len(dt._frames[(0, 0)]) == 1
        dt.on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 0, "c": 2}))
        assert len(dt._frames[(0, 0)]) == 1
        dt.shutdown()

    def test_positions_buffer_independently(self):
        dt = _make_dynatrack(_sequence(2), PositionUpdater(), expected_slices=2)
        dt.apply_position_update(MDAEvent(index={"t": 0, "p": 0}))
        dt.apply_position_update(MDAEvent(index={"t": 0, "p": 1}))
        frame = np.zeros((4, 4), dtype=np.uint16)

        dt.on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 0, "c": 0, "z": 0}))
        dt.on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 1, "c": 0, "z": 0}))
        assert dt._manager._pending_future is None

        dt.on_frame_ready(frame, MDAEvent(index={"t": 0, "p": 0, "c": 0, "z": 1}))
        assert dt._manager._pending_future is not None
        dt._manager._pending_future.result(timeout=5)
        assert (0, 0) not in dt._frames
        assert (0, 1) in dt._frames  # still buffering
        dt.shutdown()


# ---------------------------------------------------------------------------
# MantisEngine wiring
# ---------------------------------------------------------------------------


class TestMantisEngineWiring:
    def test_setup_sequence_initializes_dynatrack(self, engine, mock_core):
        seq = MDASequence(
            channels=[{"config": "BF", "group": "Channel"}],
            z_plan={"top": 1.0, "bottom": -1.0, "step": 0.5},
            stage_positions=[{"x": 10, "y": 20, "z": 5}, {"x": 30, "y": 40, "z": 15}],
            metadata={
                "dynatrack": {
                    "enabled": True,
                    "input_channel": "BF",
                    "tracking_channel": "BF",
                }
            },
        )
        with (
            patch("shrimpy.base_engine.MDAEngine.setup_sequence"),
            patch.object(DynaTrack, "start"),
        ):
            engine.setup_sequence(seq)

        assert engine._dynatrack is not None
        assert engine._dynatrack.num_positions == 2
        mock_core.mda.events.frameReady.connect.assert_called_once_with(
            engine._dynatrack.on_frame_ready
        )

    def test_setup_sequence_without_dynatrack(self, engine):
        seq = MDASequence(stage_positions=[{"x": 10, "y": 20}], metadata={})
        with patch("shrimpy.base_engine.MDAEngine.setup_sequence"):
            engine.setup_sequence(seq)
        assert engine._dynatrack is None

    def test_setup_sequence_dynatrack_disabled(self, engine):
        seq = MDASequence(
            stage_positions=[{"x": 10, "y": 20}],
            metadata={
                "dynatrack": {
                    "enabled": False,
                    "input_channel": "BF",
                    "tracking_channel": "BF",
                }
            },
        )
        with patch("shrimpy.base_engine.MDAEngine.setup_sequence"):
            engine.setup_sequence(seq)
        assert engine._dynatrack is None

    def test_teardown_shuts_down_dynatrack(self, engine, mock_core):
        dt = _make_dynatrack(_sequence(), PositionUpdater())
        engine._dynatrack = dt

        with patch("shrimpy.base_engine.MDAEngine.teardown_sequence"):
            engine.teardown_sequence(MDASequence(metadata={}))

        assert engine._dynatrack is None
        mock_core.mda.events.frameReady.disconnect.assert_called_once_with(dt.on_frame_ready)
        assert dt._manager._executor is None

    def test_event_iterator_applies_position_updates(self, demo_core):
        """event_iterator should apply position updates before events are logged."""
        engine = MantisEngine(demo_core)
        dt = _make_dynatrack(_sequence(), PositionUpdater())
        dt.position_store.update_position(0, x=777.0, y=666.0, z=555.0)
        engine._dynatrack = dt

        event = MDAEvent(x_pos=100.0, y_pos=200.0, z_pos=300.0, index={"t": 0, "p": 0})
        results = list(engine.event_iterator([event]))
        assert len(results) == 1
        assert results[0].x_pos == 777.0
        assert results[0].y_pos == 666.0
        assert results[0].z_pos == 555.0
        dt.shutdown()


# ---------------------------------------------------------------------------
# Backpressure — drains pending work at timepoint boundaries
# ---------------------------------------------------------------------------


class TestBackpressure:
    """Without backpressure a slow updater lets frame data accumulate
    unboundedly in the executor queue. event_iterator must drain pending work
    at timepoint boundaries.
    """

    def test_slow_updater_queue_bounded_across_timepoints(self, engine):
        update_completions: list[tuple[int, int, float]] = []
        event_yields: list[tuple[int, int, float]] = []

        class SlowUpdater(PositionUpdater):
            def update(self, t_idx, p_idx, position, data=None):
                time.sleep(0.5)
                update_completions.append((t_idx, p_idx, time.monotonic()))
                return position

        dt = _make_dynatrack(_sequence(3), SlowUpdater(), expected_slices=2)
        engine._dynatrack = dt

        events = [
            MDAEvent(
                index={"t": t, "p": p, "c": 0, "z": z},
                x_pos=float(p * 100),
                y_pos=float(p * 100),
            )
            for t in range(3)
            for p in range(3)
            for z in range(2)
        ]
        frame = np.zeros((64, 64), dtype=np.uint16)

        with patch("shrimpy.base_engine.MDAEngine.event_iterator", return_value=iter(events)):
            for event in engine.event_iterator(events):
                t_idx = event.index.get("t", 0)
                p_idx = event.index.get("p", 0)
                event_yields.append((t_idx, p_idx, time.monotonic()))
                dt.on_frame_ready(frame, event)

        dt.shutdown()

        assert len(update_completions) == 9  # 3 timepoints x 3 positions

        # When event_iterator yields the first event of timepoint 1, all
        # timepoint 0 updates must already be done.
        t0_last_completion = max(ts for t, p, ts in update_completions if t == 0)
        t1_first_yield = min(ts for t, p, ts in event_yields if t == 1)
        assert t0_last_completion <= t1_first_yield, (
            f"Timepoint 0 last update completed at {t0_last_completion:.3f}, "
            f"but timepoint 1 first event yielded at {t1_first_yield:.3f} — "
            "event_iterator is not draining pending updates at timepoint boundary"
        )

    def test_executor_queue_depth_bounded(self, engine):
        pending_at_submit: list[int] = []

        class SlowUpdater(PositionUpdater):
            def update(self, t_idx, p_idx, position, data=None):
                time.sleep(0.3)
                return position

        dt = _make_dynatrack(_sequence(3), SlowUpdater(), expected_slices=1)
        engine._dynatrack = dt

        manager = dt._manager
        orig_on_position_complete = manager.on_position_complete

        def tracking_on_position_complete(t_idx, p_idx, data=None):
            fut = manager._pending_future
            pending_at_submit.append(1 if fut is not None and not fut.done() else 0)
            return orig_on_position_complete(t_idx, p_idx, data)

        manager.on_position_complete = tracking_on_position_complete

        events = [
            MDAEvent(
                index={"t": t, "p": p, "c": 0, "z": 0},
                x_pos=float(p * 100),
                y_pos=float(p * 100),
            )
            for t in range(4)
            for p in range(3)
        ]
        frame = np.zeros((64, 64), dtype=np.uint16)

        with patch("shrimpy.base_engine.MDAEngine.event_iterator", return_value=iter(events)):
            for event in engine.event_iterator(events):
                dt.on_frame_ready(frame, event)

        dt.shutdown()

        # Within a timepoint, up to (positions - 1) submissions overlap; the
        # drain prevents cross-timepoint accumulation.
        max_expected_overlaps = (3 - 1) * 4
        overlaps = sum(pending_at_submit)
        assert overlaps <= max_expected_overlaps, (
            f"{overlaps}/{len(pending_at_submit)} submissions found a pending future "
            f"(expected at most {max_expected_overlaps}) — "
            "executor queue is accumulating across timepoints"
        )


# ---------------------------------------------------------------------------
# End-to-end integration with a demo acquisition
# ---------------------------------------------------------------------------


class TestDynaTrackIntegration:
    def test_positions_shift_across_acquisitions(self, demo_core, shrimpy_metadata):
        """End-to-end: a mock updater shifts position by (+1, +1, +0.5) per call.

        The engine builds the DynaTrack coordinator from the validated config;
        here we patch from_config to return an in-process coordinator with a
        shifting updater so no worker subprocess is spawned.
        """
        MantisEngine(demo_core)  # registers the engine with demo_core.mda

        class ShiftUpdater(PositionUpdater):
            def update(self, t_idx, p_idx, position, data=None, **kwargs):
                return PositionCoordinates(
                    x=position.x + 1.0,
                    y=position.y + 1.0,
                    z=(position.z or 0) + 0.5,
                )

        # Disable autofocus: demo-PFS fails ~50% of the time, which would
        # randomly drop frames and make the per-(t, p) assertions below flaky.
        shrimpy_metadata["autofocus"]["enabled"] = False
        # Include a channel so events carry a c-axis: on_frame_ready buffers the
        # configured input_channel ("DAPI"), which never matches on a
        # channel-less sequence. Mirrors a real acquisition, which always has
        # channels.
        seq = MDASequence(
            channels=[{"config": "DAPI", "group": "Channel", "exposure": 1.0}],
            stage_positions=[{"x": 100, "y": 200, "z": 50}, {"x": 300, "y": 400, "z": 60}],
            time_plan={"interval": 0, "loops": 3},
            metadata=shrimpy_metadata,
        )

        def _fake_from_config(config, sequence, data_path=None, pixel_size_um=None):
            config = DynaTrackConfig(input_channel="DAPI", tracking_channel="DAPI")
            return DynaTrack(config, sequence, updater=ShiftUpdater())

        xy_positions: list[tuple[int, int, float, float]] = []

        @demo_core.mda.events.frameReady.connect
        def _on_frame(img, event, meta):
            t = event.index.get("t", 0)
            p = event.index.get("p", 0)
            x, y = demo_core.getXYPosition()
            xy_positions.append((t, p, x, y))

        with patch.object(DynaTrack, "from_config", staticmethod(_fake_from_config)):
            demo_core.mda.run(seq)

        # Group by (t, p) and take the first frame's position for each
        seen = {}
        for t, p, x, y in xy_positions:
            seen.setdefault((t, p), (x, y))

        # At t=0, positions should be the originals
        assert seen[(0, 0)] == pytest.approx((100.0, 200.0), abs=0.1)
        assert seen[(0, 1)] == pytest.approx((300.0, 400.0), abs=0.1)

        # By t=2, the updater should have shifted each position at least once.
        x_t2_p0, y_t2_p0 = seen[(2, 0)]
        assert x_t2_p0 > 100.0, f"Expected x > 100 at t=2, got {x_t2_p0}"
        assert y_t2_p0 > 200.0, f"Expected y > 200 at t=2, got {y_t2_p0}"
