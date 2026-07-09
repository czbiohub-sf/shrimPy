"""Tests for the FOV-selection feature pipeline fast path.

The trained tree only needs a couple of "cheap" aggregate features (object
count / coverage), so ``fov_feature_matrix(..., needed=...)`` skips the
expensive per-object ``regionprops`` extraction. These tests pin the fast path
to be numerically identical to the full path and to compute only what's needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shrimpy.fov_selection import pipeline as P


def _blob_mask(n: int = 40, size: int = 256, seed: int = 0) -> np.ndarray:
    """A label mask with ``n`` disc-shaped objects."""
    from skimage.draw import disk

    rng = np.random.default_rng(seed)
    mask = np.zeros((size, size), np.uint32)
    for lab in range(1, n + 1):
        cy, cx = rng.integers(20, size - 20, 2)
        rr, cc = disk((cy, cx), int(rng.integers(4, 12)), shape=mask.shape)
        mask[rr, cc] = lab
    return mask


def test_parse_needed_features_groups_by_prefix():
    groups = P._parse_needed_features(
        ["nuclei_vs_sum__coverage_frac", "membrane_vs_sum__objects_per_10um2"]
    )
    assert groups == {
        ("nuclei", "vs", "sum"): {"coverage_frac"},
        ("membrane", "vs", "sum"): {"objects_per_10um2"},
    }


def test_cheap_features_match_full_path():
    mask = _blob_mask()
    inten = np.random.default_rng(1).normal(50, 10, mask.shape).astype(np.float32)
    px = 0.1133

    rows = P.object_feature_rows(
        mask, inten, px,
        dataset_tag="live", well_row="", well_col="", fov="",
        timepoint=0, channel="nuclei", source="vs", projection_type="sum",
    )
    full = P.group_features(pd.DataFrame(rows))
    cheap = P._cheap_features(mask, px, set(P.CHEAP_FEATURE_KEYS))

    for key in P.CHEAP_FEATURE_KEYS:
        assert np.isclose(cheap[key], full[key]), key


def test_cheap_features_empty_mask_returns_empty():
    # No objects -> no feature (matches the full path: no rows -> NaN -> imputed).
    assert P._cheap_features(np.zeros((16, 16), np.uint32), 0.1133, {"coverage_frac"}) == {}


def test_matrix_with_needed_computes_only_requested_columns():
    mask = _blob_mask()
    proj = mask.astype(np.float32)
    needed = ["nuclei_vs_sum__coverage_frac", "membrane_vs_sum__objects_per_10um2"]

    matrix = P.fov_feature_matrix(
        {"nuclei": proj, "membrane": proj},
        {"nuclei": mask, "membrane": mask},
        px_um=0.1133,
        projection="sum",
        source="vs",
        needed=needed,
    )
    assert sorted(matrix.columns) == sorted(needed)


def test_matrix_needed_matches_full_matrix_values():
    # The needed-column values must equal the full (all-feature) matrix values.
    mask = _blob_mask()
    proj = mask.astype(np.float32)
    masks = {"nuclei": mask, "membrane": mask}
    projs = {"nuclei": proj, "membrane": proj}

    full = P.fov_feature_matrix(projs, masks, 0.1133, "sum", "vs")
    needed = ["nuclei_vs_sum__coverage_frac", "membrane_vs_sum__objects_per_10um2"]
    fast = P.fov_feature_matrix(projs, masks, 0.1133, "sum", "vs", needed=needed)

    for col in needed:
        assert np.isclose(fast[col].iloc[0], full[col].iloc[0]), col
