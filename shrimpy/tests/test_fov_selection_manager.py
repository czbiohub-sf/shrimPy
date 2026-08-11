"""Tests for the streaming FOV-selection coordinator.

Exercise the frame buffering, per-FOV submission, verdict store, and drain
barrier without a GPU or worker subprocess: an in-process ``decide_fn`` stands
in for the reconstruct->segment->tree decision (which is tested against the real
pipeline elsewhere).
"""

from __future__ import annotations

import numpy as np
import pytest

from useq import MDAEvent, MDASequence

from shrimpy.fov_selection.manager import FovSelection

# 5-slice z_plan, two channels, three candidate positions.
SEQUENCE = MDASequence(
    channels=[
        {"config": "BF", "group": "Channel"},
        {"config": "GFP", "group": "Channel"},
    ],
    z_plan={"top": 2, "bottom": -2, "step": 1},  # -> 5 slices
    stage_positions=[
        {"x": 0, "y": 0, "name": "good0"},
        {"x": 1, "y": 0, "name": "bad0"},
        {"x": 2, "y": 0, "name": "good1"},
    ],
    time_plan={"loops": 3, "interval": 0},
)
N_Z = 5
META = {
    "enabled": True,
    "model": {"type": "classification_tree", "path": "dummy.joblib"},
    "fov_selection_channel": "BF",
    "preprocessing": ["deskew", "phase", "vs", "sum_projection", "segmentation"],
}


def _good_if_positive(bf_zyx: np.ndarray) -> tuple[float, bool]:
    """Decider: a FOV is good iff its stack has a positive mean.

    Encodes the verdict in the frame values so the test controls it.
    """
    is_good = float(bf_zyx.mean()) > 0
    return (1.0 if is_good else 0.0), is_good


def _event(p_idx: int, z_idx: int, channel: str, t_idx: int = 0) -> MDAEvent:
    c_idx = 0 if channel == "BF" else 1
    return MDAEvent(
        index={"t": t_idx, "p": p_idx, "c": c_idx, "z": z_idx},
        channel={"config": channel, "group": "Channel"},
        pos_name=SEQUENCE.stage_positions[p_idx].name,
    )


def _feed_prescan_stack(fov: FovSelection, p_idx: int, value: float, channel: str = "BF"):
    """Emit a full t0 z-stack for one position with constant pixel value."""
    for z in range(N_Z):
        frame = np.full((4, 4), value, dtype=np.float32)
        fov.on_frame_ready(frame, _event(p_idx, z, channel))


def _make_fov() -> FovSelection:
    fov = FovSelection.from_metadata(
        META, SEQUENCE, pixel_size_um=0.1, decide_fn=_good_if_positive
    )
    fov.start(zyx_shape=(N_Z, 4, 4))
    return fov


def test_from_metadata_disabled_returns_none():
    assert FovSelection.from_metadata({"enabled": False}, SEQUENCE, 0.1) is None
    assert FovSelection.from_metadata(None, SEQUENCE, 0.1) is None


def test_from_metadata_requires_model_path():
    with pytest.raises(ValueError, match="model.path"):
        FovSelection.from_metadata({"enabled": True}, SEQUENCE, 0.1)


def test_from_metadata_requires_pixel_size():
    with pytest.raises(ValueError, match="pixel size"):
        FovSelection.from_metadata(META, SEQUENCE, pixel_size_um=0.0)


def test_from_metadata_rejects_unknown_fov_selection_channel():
    with pytest.raises(ValueError, match="fov_selection_channel"):
        FovSelection.from_metadata({**META, "fov_selection_channel": "nope"}, SEQUENCE, 0.1)


def test_streaming_decision_partitions_good_and_bad():
    fov = _make_fov()
    try:
        _feed_prescan_stack(fov, 0, value=1.0)  # good0 -> positive
        _feed_prescan_stack(fov, 1, value=-1.0)  # bad0  -> negative
        _feed_prescan_stack(fov, 2, value=2.0)  # good1 -> positive
        fov.drain()

        assert fov.num_decided == 3
        assert set(fov.good_position_names()) == {"good0", "good1"}
        assert fov.is_good("good0")
        assert not fov.is_good("bad0")
        assert not fov.is_good("unseen")  # undecided -> bad
    finally:
        fov.shutdown()


def test_partial_stack_is_not_decided():
    """A position whose stack is incomplete must not produce a verdict."""
    fov = _make_fov()
    try:
        # Only 3 of 5 slices for position 0.
        for z in range(3):
            fov.on_frame_ready(np.ones((4, 4), np.float32), _event(0, z, "BF"))
        fov.drain()
        assert fov.num_decided == 0
    finally:
        fov.shutdown()


def test_non_input_channel_and_later_timepoints_ignored():
    fov = _make_fov()
    try:
        # GFP frames at t0 and BF frames at t1 must not be buffered for a decision.
        for z in range(N_Z):
            fov.on_frame_ready(np.ones((4, 4), np.float32), _event(0, z, "GFP"))
            fov.on_frame_ready(np.ones((4, 4), np.float32), _event(0, z, "BF", t_idx=1))
        fov.drain()
        assert fov.num_decided == 0
    finally:
        fov.shutdown()


# ---------------------------------------------------------------------------
# fov_selection_channels_type validation
# ---------------------------------------------------------------------------


def test_channels_type_defaults_to_vs_and_requires_vs_step():
    """Default type 'vs' with a 'vs' step is accepted."""
    fov = FovSelection.from_metadata(
        META, SEQUENCE, pixel_size_um=0.1, decide_fn=_good_if_positive
    )
    assert fov is not None


def test_channels_type_vs_without_vs_step_raises():
    meta = {
        **META,
        "fov_selection_channels_type": "vs",
        "preprocessing": ["deskew", "phase", "sum_projection", "segmentation"],
    }
    with pytest.raises(ValueError, match="requires a 'vs' step"):
        FovSelection.from_metadata(
            meta, SEQUENCE, pixel_size_um=0.1, decide_fn=_good_if_positive
        )


def test_channels_type_fluor_does_not_require_vs_step():
    meta = {
        **META,
        "fov_selection_channels_type": "fluor",
        "preprocessing": ["deskew", "phase", "sum_projection", "segmentation"],
    }
    fov = FovSelection.from_metadata(
        meta, SEQUENCE, pixel_size_um=0.1, decide_fn=_good_if_positive
    )
    assert fov is not None


def test_channels_type_invalid_raises():
    meta = {**META, "fov_selection_channels_type": "brightfield"}
    with pytest.raises(ValueError, match="must be one of"):
        FovSelection.from_metadata(
            meta, SEQUENCE, pixel_size_um=0.1, decide_fn=_good_if_positive
        )


# --- debug-summary finalisation ------------------------------------------------------
# `selected` / `rank` are added post-drain over the finished CSV. These pin the values and,
# more importantly, that a filesystem failure there cannot escape into teardown_sequence:
# a debug artifact must never be able to abort an acquisition.


class _StubSelection(FovSelection):
    """Bare FovSelection exposing only what finalize_debug_summary touches."""

    def __init__(self, debug_dir, top_fov, passed):
        self._debug_dir = debug_dir
        self._top_fov = top_fov
        self._passed = passed

    def passed_position_names(self):
        return self._passed


def _summary(tmp_path):
    import pandas as pd

    (tmp_path / "fov_summary.csv").write_text(
        pd.DataFrame({"name": ["A", "B", "C", "D"], "proba": [0.1, 0.9, 0.5, 0.5]}).to_csv(
            index=False
        )
    )
    return tmp_path / "fov_summary.csv"


def test_finalize_adds_selected_and_rank_for_a_ranking_model(tmp_path):
    import pandas as pd

    csv = _summary(tmp_path)
    _StubSelection(tmp_path, top_fov=2, passed=["B", "C"]).finalize_debug_summary()

    df = pd.read_csv(csv)
    assert list(df.columns)[:4] == ["name", "proba", "selected", "rank"]
    assert df.set_index("name")["selected"].to_dict() == {"A": 0, "B": 1, "C": 1, "D": 0}
    # method='first' -> ties get distinct consecutive ranks, not 3.5/3.5.
    assert df.set_index("name")["rank"].to_dict() == {"A": 4.0, "B": 1.0, "C": 2.0, "D": 3.0}


def test_finalize_leaves_rank_nan_for_non_ranking_models(tmp_path):
    import pandas as pd

    csv = _summary(tmp_path)
    _StubSelection(tmp_path, top_fov=None, passed=["B", "D"]).finalize_debug_summary()

    df = pd.read_csv(csv)
    assert df["rank"].isna().all()  # no ordering exists for a pass/fail model
    assert df.set_index("name")["selected"].to_dict() == {"A": 0, "B": 1, "C": 0, "D": 1}


def test_finalize_is_a_noop_without_a_summary(tmp_path):
    _StubSelection(tmp_path, top_fov=2, passed=[]).finalize_debug_summary()  # no CSV
    _StubSelection(None, top_fov=2, passed=[]).finalize_debug_summary()  # save_decision off


def test_finalize_falls_back_when_the_csv_is_not_writable(tmp_path):
    # A spreadsheet app holding fov_summary.csv open must not raise out of
    # teardown_sequence, and must not silently lose the selection either.
    import os
    import stat

    import pandas as pd

    csv = _summary(tmp_path)
    os.chmod(csv, stat.S_IREAD)
    try:
        _StubSelection(tmp_path, top_fov=2, passed=["B", "C"]).finalize_debug_summary()
    finally:
        os.chmod(csv, stat.S_IWRITE)

    fallback = tmp_path / "fov_summary_selected.csv"
    assert fallback.exists(), "selected/rank must survive a locked target"
    assert set(pd.read_csv(fallback).columns) >= {"selected", "rank"}
