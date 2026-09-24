"""Tests for the FOV feature-matrix -> AnnData ND export (nd_export).

The feature-column selection is pure pandas and always runs; the full CSV -> AnnData zarr
round-trip needs the optional `anndata` dependency (the `fov` group), so it importorskips.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shrimpy.fov_selection import nd_export


def _fov_summary_frame() -> pd.DataFrame:
    """A normal-mode fov_summary.csv after finalize: identity / decision / well columns plus the
    model's feature columns."""
    return pd.DataFrame(
        {
            "name": ["A1_0000", "A1_0001", "B2_0000"],
            "filename": ["A1_0000", "A1_0001", "B2_0000"],
            "proba": [0.9, 0.2, 0.5],
            "selected": [1, 0, 1],
            "position": ["A1", "A1", "B2"],
            "rank": [1.0, 2.0, 1.0],
            "well_row": ["A", "A", "B"],
            "well_col": [1, 1, 2],
            "coverage_frac": [0.6, 0.1, 0.4],
            "object_counts": [12, 2, 8],
            "max_empty_radius": [20.0, 55.0, 25.0],
        }
    )


def test_feature_columns_selects_only_numeric_features():
    feats = nd_export.feature_columns(_fov_summary_frame())
    assert feats == ["coverage_frac", "object_counts", "max_empty_radius"]
    # identity / decision / well columns are excluded even though some are numeric
    assert not ({"proba", "selected", "rank", "well_col"} & set(feats))


def test_feature_columns_drops_all_nan_columns():
    frame = _fov_summary_frame()
    frame["empty_grid_frac"] = np.nan  # e.g. a feature undefined for every FOV
    feats = nd_export.feature_columns(frame)
    assert "empty_grid_frac" not in feats
    assert "coverage_frac" in feats


def test_write_feature_anndata_roundtrip(tmp_path):
    pytest.importorskip("anndata")
    csv = tmp_path / "fov_summary.csv"
    _fov_summary_frame().to_csv(csv, index=False)

    out = nd_export.write_feature_anndata(csv)
    assert out == csv.with_suffix(".zarr") and out.exists()

    import anndata as ad

    adata = ad.read_zarr(out)
    assert adata.shape == (3, 3)  # 3 FOVs x 3 features
    assert list(adata.var_names) == ["coverage_frac", "object_counts", "max_empty_radius"]
    assert "X_pca" in adata.obsm
    assert list(adata.obs_names) == ["A1_0000", "A1_0001", "B2_0000"]
    # the model outputs are carried in obs (queryable) but never treated as features
    assert {"proba", "selected", "rank", "well_row", "well_col"} <= set(adata.obs.columns)
