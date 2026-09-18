"""Tests for the shared plate-coordinate / position-name helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from useq import Position, WellPlatePlan

from shrimpy.fov_selection.plate_naming import (
    col_label,
    file_stem_name,
    plate_labels,
    row_label,
    well_field_name,
    zarr_path_name,
)


@pytest.mark.parametrize(
    "index, expected",
    [(0, "A"), (1, "B"), (25, "Z"), (26, "AA"), (27, "AB"), (51, "AZ"), (52, "BA")],
)
def test_row_label_rolls_over_past_z(index, expected):
    # The replaced replay_camera helper used chr(ord('A') + index), which returned
    # '[' for index 26. Rolling over to 'AA' is what useq does.
    assert row_label(index) == expected


def test_row_label_agrees_with_useq_well_names():
    # A 32-row plate exercises the two-letter rows; labels must match useq's own.
    plan = WellPlatePlan(
        plate={
            "rows": 32,
            "columns": 2,
            "well_size": (1.0, 1.0),
            "well_spacing": (2.0, 2.0),
        },
        a1_center_xy=(0, 0),
    )
    useq_names = list(plan.all_well_names.reshape(-1))
    ours = [f"{row_label(r)}{col_label(c)}" for r in range(32) for c in range(2)]
    assert ours == useq_names


def test_labels_pass_strings_through_unchanged():
    assert (row_label("B"), col_label("2")) == ("B", "2")


def test_col_label_is_one_based():
    assert (col_label(0), col_label(1), col_label(11)) == ("1", "2", "12")


def test_plate_labels_renders_int_coords():
    # useq stores plate coordinates as zero-based ints; plate_labels renders them.
    assert plate_labels(Position(x=0, y=0, plate_row=1, plate_col=1)) == ("B", "2")


def test_plate_labels_passes_rendered_labels_through():
    # Not a useq Position: a plain object standing in for an already-rendered pair,
    # so a label is not double-converted if one reaches plate_labels.
    assert plate_labels(SimpleNamespace(plate_row="B", plate_col="2")) == ("B", "2")


def test_plate_labels_none_when_not_on_a_plate():
    assert plate_labels(Position(x=0, y=0, name="site0")) is None
    assert plate_labels(Position(x=0, y=0, name="site0", plate_row=1)) is None


def test_zarr_path_name_drops_non_alphanumerics():
    assert zarr_path_name("B4_0000", "fallback") == "B40000"
    assert zarr_path_name("1-Pos0000", "fallback") == "1Pos0000"


def test_zarr_path_name_falls_back_when_nothing_survives():
    assert zarr_path_name("___", "p3") == "p3"
    assert zarr_path_name("", "p3") == "p3"


def test_file_stem_name_keeps_underscore_and_dash():
    assert file_stem_name("B4_0000") == "B4_0000"
    assert file_stem_name("1-Pos0000") == "1-Pos0000"
    assert file_stem_name("a/b c") == "a_b_c"
    assert file_stem_name("") == "fov"


def test_the_two_sanitizers_differ_on_the_same_name():
    # Documented divergence: the zarr rule and the filesystem rule are not
    # interchangeable, which is why the call site names the sink.
    assert zarr_path_name("B4_0000", "x") != file_stem_name("B4_0000")


def test_well_field_name_strips_the_well_prefix():
    # plate_row=1, plate_col=3 -> well "B4", so "B4_0000" reduces to the field "0000".
    pos = Position(x=0, y=0, name="B4_0000", plate_row=1, plate_col=3)
    assert well_field_name(pos, "0009") == "0000"


def test_well_field_name_keeps_a_name_with_no_field_suffix():
    # A bare well name has no prefix to strip; it is only sanitized.
    pos = Position(x=0, y=0, name="B4", plate_row=1, plate_col=3)
    assert well_field_name(pos, "0009") == "B4"


def test_well_field_name_falls_back_on_a_bare_well_name():
    pos = Position(x=0, y=0, name="B4_", plate_row=1, plate_col=3)
    assert well_field_name(pos, "0007") == "0007"


def test_well_field_name_leaves_non_plate_names_sanitized_only():
    pos = Position(x=0, y=0, name="site0_0001")
    assert well_field_name(pos, "0009") == "site00001"
