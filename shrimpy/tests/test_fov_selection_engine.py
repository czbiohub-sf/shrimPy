"""Engine-level tests for FOV-selection event gating.

FOV selection turns one run into a pre-scan (t=0) + timelapse (t>=1). The
engine:

* yields every event from ``event_iterator`` (never drops -- so the output
  sink's per-event skip accounting stays dense) and drains the streamed
  decision exactly once at the t0->t1 boundary;
* decides in ``_fov_skip_frames`` which events to skip (raise ``SkipEvent``):
  non-``input_channel`` frames during the pre-scan, and non-"good" FOVs during
  the timelapse.

These are checked with a fake ``FovSelection`` (preset verdicts, no
worker/GPU). Hardware sequencing is off so events are plain per-frame events.
"""

from __future__ import annotations

from useq import MDAEvent, MDASequence

from shrimpy.mantis.mantis_engine import MantisEngine

SEQUENCE = MDASequence(
    stage_positions=[
        {"x": 0, "y": 0, "name": "A"},
        {"x": 1, "y": 0, "name": "B"},
        {"x": 2, "y": 0, "name": "C"},
    ],
    channels=[
        {"config": "BF", "group": "Channel"},
        {"config": "GFP", "group": "Channel"},
    ],
    z_plan={"top": 1, "bottom": -1, "step": 1},  # 3 slices
    time_plan={"loops": 2, "interval": 0},  # t0 = pre-scan, t1 = timelapse
)


class _FakeFov:
    """Stand-in for FovSelection: preset good set, records drain calls."""

    input_channel = "BF"

    def __init__(self, good: set[str]) -> None:
        self._good = good
        self.drain_calls = 0

    def drain(self) -> None:
        self.drain_calls += 1

    def is_good(self, name: str) -> bool:
        return name in self._good


def _event(t: int, p: int, name: str, channel: str) -> MDAEvent:
    c_idx = 0 if channel == "BF" else 1
    return MDAEvent(
        index={"t": t, "p": p, "c": c_idx, "z": 0},
        channel={"config": channel, "group": "Channel"},
        pos_name=name,
    )


def test_fov_skip_frames_decisions(demo_core):
    engine = MantisEngine(demo_core, use_hardware_sequencing=False)
    engine._fov = _FakeFov(good={"A", "C"})
    skip = engine._fov_skip_frames

    # Pre-scan (t=0): acquire input channel, skip everything else.
    assert skip(_event(0, 0, "A", "BF")) is None
    assert skip(_event(0, 0, "A", "GFP")) == 1  # non-input channel -> skip
    # Timelapse (t>=1): acquire good FOVs, skip the rest (regardless of channel).
    assert skip(_event(1, 0, "A", "BF")) is None
    assert skip(_event(1, 0, "A", "GFP")) is None
    assert skip(_event(1, 1, "B", "BF")) == 1  # B not good -> skip
    assert skip(_event(1, 2, "C", "GFP")) is None  # C good -> acquire


def test_event_iterator_drains_once_and_yields_all(demo_core):
    # Keep demo_core referenced (engine holds only a weakref). Sequencing off so
    # events are per-frame.
    engine = MantisEngine(demo_core, use_hardware_sequencing=False)
    engine._fov = _FakeFov(good={"A", "C"})

    events = list(engine.event_iterator(SEQUENCE))

    # Nothing is dropped: 2 timepoints x 3 positions x 2 channels x 3 z.
    assert len(events) == 2 * 3 * 2 * 3
    # Decision drained exactly once, at the t0->t1 boundary.
    assert engine._fov.drain_calls == 1


def test_event_iterator_noop_without_fov(demo_core):
    """With no FovSelection, every event passes through unfiltered."""
    engine = MantisEngine(demo_core, use_hardware_sequencing=False)
    assert engine._fov is None

    events = list(engine.event_iterator(SEQUENCE))
    assert len(events) == 2 * 3 * 2 * 3
