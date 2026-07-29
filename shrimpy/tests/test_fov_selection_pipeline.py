"""Tests for the FOV-selection feature pipeline fast path.

The trained tree only needs a couple of "cheap" aggregate features (object
count / coverage), so ``fov_feature_matrix(..., needed=...)`` skips the
expensive per-object ``regionprops`` extraction. These tests pin the fast path
to be numerically identical to the full path and to compute only what's needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shrimpy.fov_selection import pipeline


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
    groups = pipeline._parse_needed_features(
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

    rows = pipeline.object_feature_rows(
        mask,
        inten,
        px,
        dataset_tag="live",
        well_row="",
        well_col="",
        fov="",
        timepoint=0,
        channel="nuclei",
        source="vs",
        projection_type="sum",
    )
    # Mirror what the full path assembles: group_features (per-object aggregates) updated
    # with mask_gap_features (mask-derived). Cheap keys are drawn from BOTH -- coverage_frac
    # from the former, mask_occupancy_entropy from the latter.
    full = pipeline.group_features(pd.DataFrame(rows))
    full.update(pipeline.mask_gap_features(mask, px))
    cheap = pipeline._cheap_features(mask, px, set(pipeline.CHEAP_FEATURE_KEYS))

    for key in pipeline.CHEAP_FEATURE_KEYS:
        assert np.isclose(cheap[key], full[key]), key


def test_mask_occupancy_entropy_bounds_and_ordering():
    # Normalized to [0, 1]: evenly spread foreground -> 1, all foreground in one grid cell -> 0.
    from shrimpy.fov_selection.build_fov_feature_matrix import (
        MASK_OCCUPANCY_GRID as G,
    )
    from shrimpy.fov_selection.build_fov_feature_matrix import (
        mask_occupancy_entropy,
    )

    h, w = 8 * G, 8 * G
    assert mask_occupancy_entropy(np.ones((h, w), np.uint32)) == pytest.approx(1.0)

    one_cell = np.zeros((h, w), np.uint32)
    one_cell[: h // G, : w // G] = 1
    assert mask_occupancy_entropy(one_cell) == pytest.approx(0.0, abs=1e-12)

    half = np.zeros((h, w), np.uint32)
    half[:, : w // 2] = 1
    # Half the cells occupied, uniformly: log(G^2/2)/log(G^2), strictly between the extremes.
    assert 0.0 < mask_occupancy_entropy(half) < 1.0
    assert mask_occupancy_entropy(half) == pytest.approx(np.log(G * G / 2) / np.log(G * G))

    # Undefined (NaN), not 0: an empty mask has no foreground to be spread or concentrated.
    assert np.isnan(mask_occupancy_entropy(np.zeros((h, w), np.uint32)))


def test_mask_occupancy_entropy_independent_of_object_count():
    # The point of the mask-pixel version: one big blob still scores by how it is spread,
    # where the centroid-based occupancy_entropy collapses to exactly 0 (a single centroid
    # falls in a single grid cell no matter how large the object is).
    from shrimpy.fov_selection.build_fov_feature_matrix import mask_occupancy_entropy

    h, w = 256, 256
    one_blob = np.zeros((h, w), np.uint32)
    one_blob[32:224, 32:224] = 1  # a single connected component covering most of the FOV
    assert mask_occupancy_entropy(one_blob) > 0.8

    # Same pixel count concentrated in a corner scores far lower.
    corner = np.zeros((h, w), np.uint32)
    corner[:136, :136] = 1
    assert mask_occupancy_entropy(corner) < mask_occupancy_entropy(one_blob)


def test_mask_occupancy_entropy_is_a_producible_feature_name():
    # Guards the manager's pre-flight check: a model may request the new feature by name.
    from shrimpy.fov_selection.build_fov_feature_matrix import MASK_FEATURE_KEYS

    assert "mask_occupancy_entropy" in MASK_FEATURE_KEYS
    assert "mask_occupancy_entropy" in pipeline.CHEAP_FEATURE_KEYS


def test_cheap_features_empty_mask_reports_zero():
    # No objects is a real measurement, not missing data: the density features must
    # be reported as genuine zeros (so the model can act on an empty FOV) rather than
    # dropped -> NaN -> median-imputed to a typical FOV -> empty FOV misclassified good.
    empty = np.zeros((16, 16), np.uint32)
    assert pipeline._cheap_features(empty, 0.1133, {"coverage_frac"}) == {"coverage_frac": 0.0}

    out = pipeline._cheap_features(empty, 0.1133, set(pipeline.CHEAP_FEATURE_KEYS))
    assert set(out) == set(pipeline.CHEAP_FEATURE_KEYS)
    assert out["coverage_frac"] == 0.0
    # ...except mask_occupancy_entropy: the spread of a nonexistent foreground is genuinely
    # undefined, so NaN is the honest value (there is nothing for a model to act on).
    assert np.isnan(out["mask_occupancy_entropy"])


def test_matrix_empty_mask_reports_zero_density():
    # An FOV with no segmented objects must produce object_count=0 (not a missing
    # column) so the loaded model, not a hardcoded rule, decides it is bad.
    empty = np.zeros((256, 256), np.uint32)
    proj = np.zeros((256, 256), np.float32)
    matrix = pipeline.fov_feature_matrix(
        {"nuclei": proj},
        {"nuclei": empty},
        px_um=0.1133,
        projection="sum",
        source="vs",
        needed=["nuclei_vs_sum__object_count"],
    )
    assert matrix["nuclei_vs_sum__object_count"].iloc[0] == 0


def test_matrix_with_needed_computes_only_requested_columns():
    mask = _blob_mask()
    proj = mask.astype(np.float32)
    needed = ["nuclei_vs_sum__coverage_frac", "membrane_vs_sum__objects_per_10um2"]

    matrix = pipeline.fov_feature_matrix(
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

    full = pipeline.fov_feature_matrix(projs, masks, 0.1133, "sum", "vs")
    needed = ["nuclei_vs_sum__coverage_frac", "membrane_vs_sum__objects_per_10um2"]
    fast = pipeline.fov_feature_matrix(projs, masks, 0.1133, "sum", "vs", needed=needed)

    for col in needed:
        assert np.isclose(fast[col].iloc[0], full[col].iloc[0]), col
