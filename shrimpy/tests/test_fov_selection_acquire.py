"""Tests for the two-run FOV-selection sequence builders.

FOV selection is a two-run adaptive acquisition (see
``docs/fov_selection_integration_plan.md``): ``acquire`` builds a pre-scan
sequence from ``metadata.mantis.fov_selection.prescan_mda``, runs it to decide
which FOVs are "good", then builds a timelapse sequence on the good FOVs only.
These test the pure sequence-builder / position-filter helpers -- no core, no GPU.
"""

from __future__ import annotations

import pytest

from useq import MDASequence

from shrimpy.fov_selection.sequences import (
    _filter_good_positions,
    build_prescan_sequence,
    build_timelapse_sequence,
    expand_candidate_fovs,
    fov_selection_config,
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


def _fov_cfg(**overrides) -> dict:
    """FOV-selection metadata block with a valid nested pre-scan MDASequence."""
    cfg = {
        "enabled": True,
        "fov_selection_channel": "BF - Oblique",
        "prescan_mda": {
            "stage_positions": PLATE,
            "time_plan": {"loops": 1, "interval": 0},
            "z_plan": {"top": 1, "bottom": -1, "step": 1},  # 3 slices
            "channels": [{"config": "BF - Oblique", "group": "Channel"}],
        },
    }
    cfg.update(overrides)
    return cfg


# Candidate FOVs as free-XY centers + a top-level grid_plan (no plate layout):
# two centers, each with a 2x2 FOV grid -> 8 candidates named "<center>_<g>".
def _grid_fov_cfg(**overrides) -> dict:
    """FOV-selection block whose pre-scan uses explicit positions + a grid_plan."""
    cfg = {
        "enabled": True,
        "fov_selection_channel": "BF - Oblique",
        "prescan_mda": {
            "stage_positions": [
                {"x": 21080.0, "y": 24030.0, "z": 5.0, "name": "site0"},
                {"x": 100.0, "y": 200.0},  # unnamed -> "p1"
            ],
            "grid_plan": {
                "rows": 2,
                "columns": 2,
                "fov_height": 180.0,
                "fov_width": 180.0,
            },
            "time_plan": {"loops": 1, "interval": 0},
            "z_plan": {"top": 1, "bottom": -1, "step": 1},  # 3 slices
            "channels": [{"config": "BF - Oblique", "group": "Channel"}],
        },
    }
    cfg.update(overrides)
    return cfg


def _sequence(**overrides) -> MDASequence:
    """Top-level timelapse sequence: empty stage_positions, full channels."""
    kwargs = dict(
        stage_positions=[],  # filled at runtime from good pre-scan FOVs
        time_plan={"loops": 3, "interval": 1},
        z_plan={"top": 1, "bottom": -1, "step": 1},  # 3 slices
        channels=[
            {"config": "BF - Oblique", "group": "Channel"},
            {"config": "GFP", "group": "Channel"},
        ],
        metadata={"mantis": {"fov_selection": _fov_cfg()}},
    )
    kwargs.update(overrides)
    return MDASequence(**kwargs)


# ---------------------------------------------------------------------------
# fov_selection_config
# ---------------------------------------------------------------------------


def test_fov_selection_config_returns_block_when_present():
    cfg = fov_selection_config(_sequence())
    assert cfg.get("enabled") is True
    assert cfg["fov_selection_channel"] == "BF - Oblique"


def test_fov_selection_config_reflects_disabled_flag():
    seq = _sequence(metadata={"mantis": {"fov_selection": {"enabled": False}}})
    assert fov_selection_config(seq).get("enabled") is False


def test_fov_selection_config_empty_when_absent():
    assert fov_selection_config(_sequence(metadata={})) == {}


# ---------------------------------------------------------------------------
# build_prescan_sequence
# ---------------------------------------------------------------------------


def test_prescan_is_fov_selection_channel_only_one_timepoint_full_z():
    seq = _sequence()
    ps = build_prescan_sequence(seq, fov_selection_config(seq))

    assert [c.config for c in ps.channels] == ["BF - Oblique"]  # prescan channel only
    assert ps.sizes["t"] == 1  # single pre-scan timepoint
    assert ps.sizes["z"] == 3  # pre-scan z-plan
    assert ps.sizes["p"] == 3  # all candidates
    # fov_selection stays enabled so setup_sequence builds the coordinator ...
    assert ps.metadata["mantis"]["fov_selection"]["enabled"] is True
    # ... but the nested prescan_mda is dropped to avoid a redundant self-copy
    assert "prescan_mda" not in ps.metadata["mantis"]["fov_selection"]


def test_prescan_injects_shared_mantis_hardware_settings():
    seq = _sequence(
        metadata={
            "mantis": {
                "fov_selection": _fov_cfg(),
                "autofocus": {"enabled": False, "method": "demo-PFS"},
            }
        }
    )
    ps = build_prescan_sequence(seq, fov_selection_config(seq))
    assert ps.metadata["mantis"]["autofocus"]["method"] == "demo-PFS"


def test_prescan_raises_on_unknown_fov_selection_channel():
    seq = _sequence()
    cfg = _fov_cfg(fov_selection_channel="NoSuchChannel")
    with pytest.raises(ValueError, match="not one of the pre-scan channels"):
        build_prescan_sequence(seq, cfg)


def test_prescan_raises_without_prescan_mda():
    cfg = {"enabled": True, "fov_selection_channel": "BF - Oblique"}
    with pytest.raises(ValueError, match="prescan_mda"):
        build_prescan_sequence(_sequence(), cfg)


def test_prescan_raises_on_multiple_timepoints():
    cfg = _fov_cfg()
    cfg["prescan_mda"]["time_plan"] = {"loops": 2, "interval": 0}
    with pytest.raises(ValueError, match="single timepoint"):
        build_prescan_sequence(_sequence(), cfg)


def test_prescan_raises_without_stage_positions():
    cfg = _fov_cfg()
    cfg["prescan_mda"]["stage_positions"] = []
    with pytest.raises(ValueError, match="stage_positions"):
        build_prescan_sequence(_sequence(), cfg)


# ---------------------------------------------------------------------------
# build_timelapse_sequence / _filter_good_positions
# ---------------------------------------------------------------------------


def test_timelapse_keeps_only_good_positions_and_full_channels():
    seq = _sequence()
    ps = build_prescan_sequence(seq, fov_selection_config(seq))
    tl = build_timelapse_sequence(seq, ps, ["B4_0000", "B5_0000"])

    # names are reduced to the per-well field index ("B4_0000" -> "0000")
    assert [p.name for p in tl.stage_positions] == ["0000", "0000"]
    assert [c.config for c in tl.channels] == ["BF - Oblique", "GFP"]
    assert tl.sizes["t"] == seq.sizes["t"]  # loops used as-is, no +1


def test_timelapse_disables_fov_selection_without_mutating_original():
    seq = _sequence()
    ps = build_prescan_sequence(seq, fov_selection_config(seq))
    tl = build_timelapse_sequence(seq, ps, ["B4_0000"])

    assert tl.metadata["mantis"]["fov_selection"]["enabled"] is False
    # the original sequence's metadata must be untouched (deep-copied)
    assert seq.metadata["mantis"]["fov_selection"]["enabled"] is True


def test_filter_converts_plate_coords_and_field_names():
    seq = _sequence()
    ps = build_prescan_sequence(seq, fov_selection_config(seq))
    good = _filter_good_positions(ps, ["B4_0000", "B5_0000"])

    # int plate coords -> string well names; the FOV name -> per-well field index.
    assert [(p.name, p.plate_row, p.plate_col) for p in good] == [
        ("0000", "B", "4"),
        ("0000", "B", "5"),
    ]


def test_filter_empty_when_no_good_names():
    seq = _sequence()
    ps = build_prescan_sequence(seq, fov_selection_config(seq))
    assert _filter_good_positions(ps, []) == []


def test_timelapse_positions_produce_readable_hcs_plate():
    # The filtered explicit list must yield a proper HCS OME-Zarr plate whose
    # well field/image paths are ALPHANUMERIC (iohub rejects e.g. "B4_0000").
    ome_useq = pytest.importorskip("ome_writers._useq")
    seq = _sequence()
    ps = build_prescan_sequence(seq, fov_selection_config(seq))
    tl = build_timelapse_sequence(seq, ps, ["B4_0000", "B5_0000"])

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


# ---------------------------------------------------------------------------
# grid_plan (free-XY) candidate style
# ---------------------------------------------------------------------------


def test_prescan_grid_plan_expands_to_one_position_per_fov():
    # A top-level grid_plan must be flattened so each FOV is its own position
    # (its own p_idx), not left on useq's separate `g` axis.
    seq = _sequence(metadata={"mantis": {"fov_selection": _grid_fov_cfg()}})
    ps = build_prescan_sequence(seq, fov_selection_config(seq))

    assert ps.grid_plan is None  # grid axis collapsed into positions
    assert ps.sizes["p"] == 8  # 2 centers x 2x2 grid
    assert ps.sizes["g"] == 0
    # names are unique and derive from the center (unnamed center -> "p1")
    names = [p.name for p in ps.stage_positions]
    assert names == [f"site0_{g:04d}" for g in range(4)] + [f"p1_{g:04d}" for g in range(4)]
    assert len(set(names)) == len(names)


def test_expand_candidate_fovs_absolute_xy_and_inherited_z():
    seq = _sequence(metadata={"mantis": {"fov_selection": _grid_fov_cfg()}})
    ps = build_prescan_sequence(seq, fov_selection_config(seq))
    by_name = {p.name: p for p in ps.stage_positions}

    # 180 um FOV, 2x2 grid centered on (21080, 24030): +/-90 um in each axis.
    assert (by_name["site0_0000"].x, by_name["site0_0000"].y) == (20990.0, 24120.0)
    assert (by_name["site0_0003"].x, by_name["site0_0003"].y) == (20990.0, 23940.0)
    # z is inherited from the center; the unnamed center had no z.
    assert by_name["site0_0000"].z == 5.0
    assert by_name["p1_0000"].z is None
    # no plate layout for the free-XY style
    assert by_name["site0_0000"].plate_row is None


def test_grid_style_good_positions_pass_through_filter_unchanged():
    # Without plate coords the good FOVs are kept verbatim (name + XY), producing
    # a flat, non-HCS list for the timelapse run.
    seq = _sequence(metadata={"mantis": {"fov_selection": _grid_fov_cfg()}})
    ps = build_prescan_sequence(seq, fov_selection_config(seq))
    good = _filter_good_positions(ps, ["site0_0001", "p1_0003"])

    assert [(g.name, g.x, g.y, g.plate_row) for g in good] == [
        ("site0_0001", 21170.0, 24120.0, None),
        ("p1_0003", 10.0, 110.0, None),
    ]


def test_wellplate_style_is_not_expanded_by_grid_helper():
    # The WellPlatePlan path already expands per-FOV; build_prescan_sequence must
    # leave it as a WellPlatePlan (with plate_row/col) rather than flattening it.
    seq = _sequence()
    ps = build_prescan_sequence(seq, fov_selection_config(seq))
    assert ps.grid_plan is None
    assert all(p.plate_row is not None for p in ps.stage_positions)


def test_explicit_positions_no_grid_pass_through():
    # A plain explicit list with no grid_plan is not expanded by build_prescan.
    cfg = _grid_fov_cfg()
    cfg["prescan_mda"].pop("grid_plan")
    seq = _sequence(metadata={"mantis": {"fov_selection": cfg}})
    ps = build_prescan_sequence(seq, fov_selection_config(seq))
    assert ps.grid_plan is None
    assert ps.sizes["p"] == 2  # the two centers, unexpanded


def test_expand_candidate_fovs_unit():
    # Direct unit check of the expansion helper: 1 center x 2x2 grid -> 4 FOVs.
    seq = MDASequence(
        stage_positions=[{"x": 0.0, "y": 0.0, "name": "c"}],
        grid_plan={"rows": 2, "columns": 2, "fov_height": 180.0, "fov_width": 180.0},
    )
    fovs = expand_candidate_fovs(seq)
    assert [f.name for f in fovs] == ["c_0000", "c_0001", "c_0002", "c_0003"]
    assert {(f.x, f.y) for f in fovs} == {
        (-90.0, 90.0),
        (90.0, 90.0),
        (90.0, -90.0),
        (-90.0, -90.0),
    }
