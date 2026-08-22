"""Tests for FOV-selection pre-scan artifact assembly (shrimpy.fov_selection.prescan_artifacts).

`_assemble_debug_channels` stacks every reconstruction stage (deskew / phase /
VS volumes in 3D, plus the 2D projection + mask broadcast across Z) into one
`(C, Z, Y, X)` volume for the debug OME-Zarr. These pin the channel order, the
2D->3D broadcast, and the float cast without needing a GPU or iohub.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shrimpy.fov_selection import prescan_artifacts


def _artifacts(nz=3, ny=4, nx=5):
    rng = np.random.default_rng(0)
    stacks = {
        "deskew": rng.normal(size=(nz, ny, nx)).astype(np.float32),
        "phase": rng.normal(size=(nz, ny, nx)).astype(np.float32),
        "nuclei": rng.normal(size=(nz, ny, nx)).astype(np.float32),
        "membrane": rng.normal(size=(nz, ny, nx)).astype(np.float32),
    }
    projections = {
        "nuclei": rng.normal(size=(ny, nx)).astype(np.float32),
        "membrane": rng.normal(size=(ny, nx)).astype(np.float32),
    }
    # Label masks as integers -> must be cast to float32 in the output.
    masks = {
        "nuclei": rng.integers(0, 7, size=(ny, nx)).astype(np.uint32),
        "membrane": rng.integers(0, 7, size=(ny, nx)).astype(np.uint32),
    }
    return {"stacks": stacks, "projections": projections, "masks": masks}


def test_assemble_channel_order_and_shape():
    names, czyx = prescan_artifacts._assemble_debug_channels(_artifacts())
    assert names == [
        "deskew",
        "phase",
        "nuclei_vs",
        "membrane_vs",
        "nuclei_projection",
        "membrane_projection",
        "nuclei_mask",
        "membrane_mask",
    ]
    assert czyx.shape == (8, 3, 4, 5)
    assert czyx.dtype == np.float32


def test_assemble_broadcasts_2d_across_z():
    art = _artifacts()
    names, czyx = prescan_artifacts._assemble_debug_channels(art)

    # The projection/mask channels are the same 2D plane on every Z slice.
    proj_idx = names.index("nuclei_projection")
    for z in range(czyx.shape[1]):
        np.testing.assert_array_equal(czyx[proj_idx, z], art["projections"]["nuclei"])

    # The 3D VS volume varies across Z (not broadcast).
    vs_idx = names.index("nuclei_vs")
    assert not np.array_equal(czyx[vs_idx, 0], czyx[vs_idx, 1])


def test_assemble_casts_mask_labels_to_float():
    art = _artifacts()
    names, czyx = prescan_artifacts._assemble_debug_channels(art)
    mask_idx = names.index("membrane_mask")
    np.testing.assert_array_equal(
        czyx[mask_idx, 0], art["masks"]["membrane"].astype(np.float32)
    )


def test_assemble_without_3d_stacks_returns_none():
    # No 3D stack -> nothing to anchor the (Z, Y, X) grid -> skip the store.
    names, czyx = prescan_artifacts._assemble_debug_channels(
        {"stacks": {}, "projections": {}, "masks": {}}
    )
    assert names == []
    assert czyx is None


# --------------------------------------------------------------------------
# Calibration feature-viewer layout (write_feature_viewer_artifacts)
# --------------------------------------------------------------------------
# The calibration pre-scan must write exactly what the feature viewer loads:
# <stem>.csv with a `filename` column + sibling prescan_fov / prescan_mask
# folders whose PNG stems equal the CSV `filename`.


def _fv_artifacts(fov_name):
    import pandas as pd

    rng = np.random.default_rng(abs(hash(fov_name)) % (2**32))
    return {
        "projections": {"brightfield": rng.normal(size=(4, 5)).astype(np.float32)},
        "masks": {"brightfield": rng.integers(0, 4, size=(4, 5)).astype(np.uint32)},
        "features": pd.DataFrame([{"coverage_frac": 0.3, "nn_um_mean": 12.0}]),
    }


def test_feature_viewer_layout_matches_standard(tmp_path):
    import pandas as pd

    for name in ("B4_0000", "B4/0001"):  # a slash must be sanitized to a safe stem
        prescan_artifacts.write_feature_viewer_artifacts(tmp_path, name, _fv_artifacts(name))

    # calibration shares the fixed fov_summary.csv name with normal mode
    csv = tmp_path / prescan_artifacts.SUMMARY_CSV_NAME
    assert csv.exists()
    df = pd.read_csv(csv)
    # A `filename` column joins each row to its PNG; the ranking `proba` is NOT written.
    assert "filename" in df.columns
    assert "proba" not in df.columns
    assert {"coverage_frac", "nn_um_mean"} <= set(df.columns)
    assert sorted(df["filename"]) == ["B4_0000", "B4_0001"]

    png_dir = tmp_path / "prescan_fov"
    mask_dir = tmp_path / "prescan_mask"
    assert (png_dir / "B4_0000.png").exists() and (png_dir / "B4_0001.png").exists()
    assert (mask_dir / "B4_0000.png").exists() and (mask_dir / "B4_0001.png").exists()


def test_feature_viewer_layout_loads_in_the_viewer(tmp_path):
    # The written layout must round-trip through the viewer's own loader with the
    # brightfield PNG wired to each row.
    from shrimpy.fov_selection.feature_viewer import data

    for name in ("f0", "f1"):
        prescan_artifacts.write_feature_viewer_artifacts(tmp_path, name, _fv_artifacts(name))

    df = data.load_matrices([tmp_path / prescan_artifacts.SUMMARY_CSV_NAME])
    assert len(df) == 2
    assert all(png and Path(png).exists() for png in df["__png"])
    assert "coverage_frac" in data.feature_columns(df)


def test_normal_mode_fov_summary_loads_in_the_viewer(tmp_path):
    # fov_summary.csv (save_decision, normal mode) must load in the viewer just like the
    # calibration matrix: it carries a `filename` join column and its images live in the same
    # prescan_fov/ folder, while the decision outputs (proba/selected/rank) stay off the axes.
    import pandas as pd

    from shrimpy.fov_selection.feature_viewer import data

    for name, proba in (("f0", 0.2), ("f1", 0.9)):
        prescan_artifacts.write_decision_artifacts(tmp_path, name, proba, _fv_artifacts(name))
    # stamp the whole-run selection columns, as the manager does post-drain
    prescan_artifacts.finalize_summary_csv(
        tmp_path / prescan_artifacts.SUMMARY_CSV_NAME,
        passed={"f1"},
        fov_group={"f0": "P", "f1": "P"},
        top_fov=1,
    )

    csv = tmp_path / prescan_artifacts.SUMMARY_CSV_NAME
    assert {"name", "filename", "proba", "selected", "rank"} <= set(pd.read_csv(csv).columns)

    df = data.load_matrices([csv])
    assert len(df) == 2
    assert all(png and Path(png).exists() for png in df["__png"])  # wired to prescan_fov/
    feats = data.feature_columns(df)
    assert "coverage_frac" in feats
    assert not ({"proba", "selected", "rank"} & set(feats))  # decision outputs are not axes


# --------------------------------------------------------------------------
# Well columns (stamp_well_columns) -> the viewer groups a pre-scan by well
# --------------------------------------------------------------------------


def test_stamp_well_columns_joins_by_filename(tmp_path):
    import pandas as pd

    csv = tmp_path / prescan_artifacts.SUMMARY_CSV_NAME
    pd.DataFrame(
        {"filename": ["A1_0000", "A1_0001", "B2_0000"], "coverage_frac": [0.1, 0.2, 0.3]}
    ).to_csv(csv, index=False)

    prescan_artifacts.stamp_well_columns(
        csv, {"A1_0000": ("A", 1), "A1_0001": ("A", 1), "B2_0000": ("B", 2)}
    )

    df = pd.read_csv(csv).set_index("filename")
    # well_row is the letter label, well_col the one-based int -> "Well B/2" in the viewer
    assert list(df.loc["A1_0000", ["well_row", "well_col"]]) == ["A", 1]
    assert list(df.loc["B2_0000", ["well_row", "well_col"]]) == ["B", 2]


def test_stamp_well_columns_is_a_noop_without_coords_or_filename(tmp_path):
    import pandas as pd

    csv = tmp_path / prescan_artifacts.SUMMARY_CSV_NAME
    pd.DataFrame({"name": ["x"], "coverage_frac": [0.1]}).to_csv(csv, index=False)

    prescan_artifacts.stamp_well_columns(csv, {})  # no coords (off-plate) -> no-op
    prescan_artifacts.stamp_well_columns(csv, {"x": ("A", 1)})  # no filename column -> skip
    assert "well_row" not in pd.read_csv(csv).columns
