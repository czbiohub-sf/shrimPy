"""Tests for the FOV-selection worker's debug-artifact assembly.

`_assemble_debug_channels` stacks every reconstruction stage (deskew / phase /
VS volumes in 3D, plus the 2D projection + mask broadcast across Z) into one
`(C, Z, Y, X)` volume for the debug OME-Zarr. These pin the channel order, the
2D->3D broadcast, and the float cast without needing a GPU or iohub.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shrimpy.fov_selection import worker


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
    names, czyx = worker._assemble_debug_channels(_artifacts())
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
    names, czyx = worker._assemble_debug_channels(art)

    # The projection/mask channels are the same 2D plane on every Z slice.
    proj_idx = names.index("nuclei_projection")
    for z in range(czyx.shape[1]):
        np.testing.assert_array_equal(czyx[proj_idx, z], art["projections"]["nuclei"])

    # The 3D VS volume varies across Z (not broadcast).
    vs_idx = names.index("nuclei_vs")
    assert not np.array_equal(czyx[vs_idx, 0], czyx[vs_idx, 1])


def test_assemble_casts_mask_labels_to_float():
    art = _artifacts()
    names, czyx = worker._assemble_debug_channels(art)
    mask_idx = names.index("membrane_mask")
    np.testing.assert_array_equal(
        czyx[mask_idx, 0], art["masks"]["membrane"].astype(np.float32)
    )


def test_assemble_without_3d_stacks_returns_none():
    # No 3D stack -> nothing to anchor the (Z, Y, X) grid -> skip the store.
    names, czyx = worker._assemble_debug_channels(
        {"stacks": {}, "projections": {}, "masks": {}}
    )
    assert names == []
    assert czyx is None


# --------------------------------------------------------------------------
# Calibration feature-viewer layout (_write_feature_viewer_artifacts)
# --------------------------------------------------------------------------
# The calibration pre-scan must write exactly what the feature viewer loads:
# <stem>.csv with a `filename` column + sibling <stem>_png / <stem>_mask_png
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

    stem = "acq_fov_feature_matrix"
    for name in ("B4_0000", "B4/0001"):  # a slash must be sanitized to a safe stem
        worker._write_feature_viewer_artifacts(tmp_path, stem, name, _fv_artifacts(name))

    csv = tmp_path / f"{stem}.csv"
    assert csv.exists()
    df = pd.read_csv(csv)
    # A `filename` column joins each row to its PNG; the ranking `proba` is NOT written.
    assert "filename" in df.columns
    assert "proba" not in df.columns
    assert {"coverage_frac", "nn_um_mean"} <= set(df.columns)
    assert sorted(df["filename"]) == ["B4_0000", "B4_0001"]

    png_dir = tmp_path / f"{stem}_png"
    mask_dir = tmp_path / f"{stem}_mask_png"
    assert (png_dir / "B4_0000.png").exists() and (png_dir / "B4_0001.png").exists()
    assert (mask_dir / "B4_0000.png").exists() and (mask_dir / "B4_0001.png").exists()


def test_feature_viewer_layout_loads_in_the_viewer(tmp_path):
    # The written layout must round-trip through the viewer's own loader with the
    # brightfield PNG wired to each row.
    from shrimpy.fov_selection.feature_viewer import data

    stem = "acq_fov_feature_matrix"
    for name in ("f0", "f1"):
        worker._write_feature_viewer_artifacts(tmp_path, stem, name, _fv_artifacts(name))

    df = data.load_matrices([tmp_path / f"{stem}.csv"])
    assert len(df) == 2
    assert all(png and Path(png).exists() for png in df["__png"])
    assert "coverage_frac" in data.feature_columns(df)
