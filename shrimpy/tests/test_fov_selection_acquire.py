"""Tests for the two-run FOV-selection orchestration helpers.

FOV selection is a two-run adaptive acquisition (see
``docs/fov_selection_integration_plan.md``): ``acquire`` builds a pre-scan
sequence, runs it to decide which FOVs are "good", then builds a timelapse
sequence on the good FOVs only. These test the pure sequence-builder / position-
filter helpers -- no core, no GPU.
"""

from __future__ import annotations

import pytest

from useq import MDASequence

from shrimpy.mantis.mantis_engine import (
    _build_prescan_sequence,
    _build_timelapse_sequence,
    _enabled_fov_config,
    _filter_good_positions,
)

# Candidate FOVs as a useq WellPlatePlan: wells B3, B4, B5 (1x1 grid) -> the
# expanded positions are named "B3_0000", "B4_0000", "B5_0000".
PLATE = {
    "a1_center_xy": [0.0, 0.0],
    "plate": {
        "rows": 4,
        "columns": 6,
        "name": "24-well",
        "well_size": [15.6, 15.6],
        "well_spacing": [19.0, 19.0],
    },
    "selected_wells": [[1, 1, 1], [2, 3, 4]],
    "well_points_plan": {
        "rows": 1,
        "columns": 1,
        "fov_height": 180.0,
        "fov_width": 180.0,
    },
}


def _sequence(**overrides) -> MDASequence:
    kwargs = dict(
        stage_positions=PLATE,
        time_plan={"loops": 3, "interval": 1},
        z_plan={"top": 1, "bottom": -1, "step": 1},  # 3 slices
        channels=[
            {"config": "BF - Oblique", "group": "Channel"},
            {"config": "GFP", "group": "Channel"},
        ],
        metadata={
            "mantis": {"fov_selection": {"enabled": True, "input_channel": "BF - Oblique"}}
        },
    )
    kwargs.update(overrides)
    return MDASequence(**kwargs)


# ---------------------------------------------------------------------------
# _enabled_fov_config
# ---------------------------------------------------------------------------


def test_enabled_fov_config_returns_block_when_enabled():
    cfg = _enabled_fov_config(_sequence())
    assert cfg is not None
    assert cfg["input_channel"] == "BF - Oblique"


def test_enabled_fov_config_none_when_disabled():
    seq = _sequence(metadata={"mantis": {"fov_selection": {"enabled": False}}})
    assert _enabled_fov_config(seq) is None


def test_enabled_fov_config_none_when_absent():
    assert _enabled_fov_config(_sequence(metadata={})) is None


# ---------------------------------------------------------------------------
# _build_prescan_sequence
# ---------------------------------------------------------------------------


def test_prescan_is_input_channel_only_one_timepoint_full_z():
    seq = _sequence()
    ps = _build_prescan_sequence(seq, "BF - Oblique")

    assert [c.config for c in ps.channels] == ["BF - Oblique"]  # input channel only
    assert ps.sizes["t"] == 1  # single pre-scan timepoint
    assert ps.sizes["z"] == seq.sizes["z"]  # full z retained
    assert ps.sizes["p"] == 3  # all candidates
    # fov_selection stays enabled so setup_sequence builds the coordinator
    assert ps.metadata["mantis"]["fov_selection"]["enabled"] is True


def test_prescan_raises_on_unknown_input_channel():
    with pytest.raises(ValueError, match="not one of the acquisition channels"):
        _build_prescan_sequence(_sequence(), "NoSuchChannel")


# ---------------------------------------------------------------------------
# _build_timelapse_sequence / _filter_good_positions
# ---------------------------------------------------------------------------


def test_timelapse_keeps_only_good_positions_and_full_channels():
    seq = _sequence()
    tl = _build_timelapse_sequence(seq, ["B4_0000", "B5_0000"])

    # names are reduced to the per-well field index ("B4_0000" -> "0000")
    assert [p.name for p in tl.stage_positions] == ["0000", "0000"]
    assert [c.config for c in tl.channels] == ["BF - Oblique", "GFP"]
    assert tl.sizes["t"] == seq.sizes["t"]  # loops used as-is, no +1


def test_timelapse_disables_fov_selection_without_mutating_original():
    seq = _sequence()
    tl = _build_timelapse_sequence(seq, ["B4_0000"])

    assert tl.metadata["mantis"]["fov_selection"]["enabled"] is False
    # the original sequence's metadata must be untouched (deep-copied)
    assert seq.metadata["mantis"]["fov_selection"]["enabled"] is True


def test_filter_converts_plate_coords_and_field_names():
    seq = _sequence()
    good = _filter_good_positions(seq, ["B4_0000", "B5_0000"])

    # int plate coords -> string well names; the FOV name -> per-well field index.
    assert [(p.name, p.plate_row, p.plate_col) for p in good] == [
        ("0000", "B", "4"),
        ("0000", "B", "5"),
    ]


def test_filter_empty_when_no_good_names():
    assert _filter_good_positions(_sequence(), []) == []


def test_timelapse_positions_produce_readable_hcs_plate():
    # The filtered explicit list must yield a proper HCS OME-Zarr plate whose
    # well field/image paths are ALPHANUMERIC (iohub rejects e.g. "B4_0000").
    ome_useq = pytest.importorskip("ome_writers._useq")
    tl = _build_timelapse_sequence(_sequence(), ["B4_0000", "B5_0000"])

    plate = ome_useq._plate_from_useq(tl)
    assert plate is not None
    assert plate.row_names == ["B"]
    assert plate.column_names == ["4", "5"]

    built = ome_useq._build_positions(tl)
    for p in built:
        assert p.name.isalnum(), f"well field path {p.name!r} must be alphanumeric"
    assert [(p.plate_row, p.plate_column, p.name) for p in built] == [
        ("B", "4", "0000"),
        ("B", "5", "0000"),
    ]
