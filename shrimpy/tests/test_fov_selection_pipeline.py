"""Tests for the FOV-selection feature pipeline.

FOV selection segments ONE mask (the target) and emits plain feature keys via
``extract_features``; when a model needs only the "cheap" aggregate features (coverage,
count) the expensive per-object ``regionprops`` extraction is skipped. These tests pin the
cheap path to be numerically identical to the full path, plus the target -> segmentation-input
resolution and the calibration extract-all behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shrimpy.fov_selection import pipeline
from shrimpy.fov_selection.feature_extraction import FeatureExtractor as FE
from shrimpy.fov_selection.segmentation import build_segmenter


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


def test_cheap_features_match_full_path():
    mask = _blob_mask()
    inten = np.random.default_rng(1).normal(50, 10, mask.shape).astype(np.float32)
    px = 0.1133

    rows = FE.object_feature_rows(mask, inten, px)
    # Mirror what the full path assembles: group_features (per-object aggregates) updated
    # with mask_gap_features (mask-derived). Cheap keys are drawn from BOTH -- coverage_frac
    # from the former, mask_occupancy_entropy from the latter.
    full = FE.group_features(pd.DataFrame(rows))
    full.update(FE.mask_gap_features(mask, px))
    cheap = pipeline._cheap_features(mask, set(pipeline.CHEAP_FEATURE_KEYS))

    for key in pipeline.CHEAP_FEATURE_KEYS:
        assert np.isclose(cheap[key], full[key]), key


def test_mask_occupancy_entropy_bounds_and_ordering():
    # Normalized to [0, 1]: evenly spread foreground -> 1, all foreground in one grid cell -> 0.
    from shrimpy.fov_selection.feature_extraction import (
        FeatureExtractor,
        mask_occupancy_entropy,
    )

    G = FeatureExtractor.MASK_OCCUPANCY_GRID

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
    from shrimpy.fov_selection.feature_extraction import mask_occupancy_entropy

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
    from shrimpy.fov_selection.feature_extraction import MASK_FEATURE_KEYS

    assert "mask_occupancy_entropy" in MASK_FEATURE_KEYS
    assert "mask_occupancy_entropy" in pipeline.CHEAP_FEATURE_KEYS


def test_cheap_features_empty_mask_reports_zero():
    # No objects is a real measurement, not missing data: the density features must
    # be reported as genuine zeros (so the model can act on an empty FOV) rather than
    # dropped -> NaN -> median-imputed to a typical FOV -> empty FOV misclassified good.
    empty = np.zeros((16, 16), np.uint32)
    assert pipeline._cheap_features(empty, {"coverage_frac"}) == {"coverage_frac": 0.0}

    out = pipeline._cheap_features(empty, set(pipeline.CHEAP_FEATURE_KEYS))
    assert set(out) == set(pipeline.CHEAP_FEATURE_KEYS)
    assert out["coverage_frac"] == 0.0
    # ...except mask_occupancy_entropy: the spread of a nonexistent foreground is genuinely
    # undefined, so NaN is the honest value (there is nothing for a model to act on).
    assert np.isnan(out["mask_occupancy_entropy"])


def test_extract_features_empty_mask_reports_zero_coverage():
    # An FOV with no segmented objects must report coverage as a real 0.0 (not a missing
    # column / NaN) so the loaded model, not a hardcoded rule, decides it is bad.
    empty = np.zeros((256, 256), np.uint32)
    proj = np.zeros((256, 256), np.float32)
    matrix = pipeline.extract_features(
        {"cells": proj}, {"cells": empty}, pixel_size_um=0.1133, needed=["coverage_frac"]
    )
    assert matrix["coverage_frac"].iloc[0] == 0.0


def test_extract_features_needed_matches_full_values():
    # `needed` restricts to just those plain-key columns, and their values equal the full
    # (all-feature) matrix values -- the cheap-path speed-up is numerically identical.
    mask = _blob_mask()
    proj = mask.astype(np.float32)
    full = pipeline.extract_features({"cells": proj}, {"cells": mask}, 0.1133)
    needed = ["coverage_frac", "nn_um_mean"]
    fast = pipeline.extract_features({"cells": proj}, {"cells": mask}, 0.1133, needed=needed)
    assert sorted(fast.columns) == sorted(needed)
    for col in needed:
        assert np.isclose(fast[col].iloc[0], full[col].iloc[0], equal_nan=True), col


# ---------------------------------------------------------------------------
# decide_fov(extract_all=...) -- calibration extracts every feature
# ---------------------------------------------------------------------------


class _StubModel:
    """A model that reads a single feature; decide_fov must still honor extract_all."""

    feature_names = ["coverage_frac"]

    def predict(self, matrix, threshold):
        return np.array([0.5]), np.array([True])


def _two_blob_stack() -> np.ndarray:
    zyx = np.zeros((3, 64, 64), np.float32)
    zyx[:, 10:20, 10:20] = 1.0
    zyx[:, 40:50, 40:50] = 1.0
    return zyx


def _preprocessor(bf, label="", return_intermediates=False):
    return {"brightfield": bf}


def test_decide_fov_default_extracts_only_model_features():
    _, _, art = pipeline.decide_fov(
        _preprocessor,
        build_segmenter({"model": "otsu"}),
        _StubModel(),
        _two_blob_stack(),
        target="cells",
        target_channels=["brightfield"],
        projection="middle",
        pixel_size_um=0.1,
        return_artifacts=True,
    )
    assert list(art["features"].columns) == ["coverage_frac"]


def test_decide_fov_extract_all_ignores_model_feature_names():
    from shrimpy.fov_selection.feature_extraction import (
        FEATURE_NAMES,
        MASK_FEATURE_KEYS,
    )

    _, _, art = pipeline.decide_fov(
        _preprocessor,
        build_segmenter({"model": "otsu"}),
        _StubModel(),
        _two_blob_stack(),
        target="cells",
        target_channels=["brightfield"],
        projection="middle",
        pixel_size_um=0.1,
        return_artifacts=True,
        extract_all=True,
    )
    produced = set(art["features"].columns)
    # Single segmented channel -> plain (unprefixed) names, and EVERY producible feature.
    assert set(FEATURE_NAMES) | set(MASK_FEATURE_KEYS) <= produced


def test_resolve_seg_input_reduces_to_one_image_by_target():
    n = np.ones((4, 4), np.float32)
    m = np.full((4, 4), 2.0, np.float32)
    # VS 'cells' -> combined nuclei+membrane; 'nuclei' -> nuclei only.
    assert np.array_equal(
        pipeline._resolve_seg_input("cells", {"nuclei": n, "membrane": m}), n + m
    )
    assert np.array_equal(
        pipeline._resolve_seg_input("nuclei", {"nuclei": n, "membrane": m}), n
    )
    # single reconstructed channel -> that channel, regardless of target.
    assert np.array_equal(pipeline._resolve_seg_input("cells", {"brightfield": m}), m)
    assert np.array_equal(pipeline._resolve_seg_input("nuclei", {"brightfield": m}), m)


# ---------------------------------------------------------------------------
# edge_frac -- mask-area share of objects stuck to the image-edge border band
# ---------------------------------------------------------------------------


def test_edge_frac_is_a_producible_mask_feature():
    from shrimpy.fov_selection.feature_extraction import (
        MASK_FEATURE_KEYS,
        mask_gap_features,
    )

    assert "edge_frac" in MASK_FEATURE_KEYS
    mask = np.zeros((64, 64), np.uint32)
    mask[0:6, 0:6] = 1
    mask[30:40, 30:40] = 2
    feats = mask_gap_features(mask, pixel_size_um=0.1)
    assert "edge_frac" in feats and feats["edge_frac"] == 0.5


# ---------------------------------------------------------------------------
# angular_uniformity -- evenness of centroid angles about the FOV center
# ---------------------------------------------------------------------------


def _object_df(centroids_norm, w=100, h=100, px=0.1):
    """A minimal per-object table for group_features (only the columns it reads)."""
    return pd.DataFrame(
        {
            "image_width_px": w,
            "image_height_px": h,
            "pixel_size_um": px,
            "centroid_x_norm": [c[0] for c in centroids_norm],
            "centroid_y_norm": [c[1] for c in centroids_norm],
            "area_px": 10,
        }
    )


def _ring_centroids(n, radius=0.3, cx=0.5, cy=0.5, spread=2 * np.pi, start=0.0):
    """n centroids (normalized) on a circle, angles evenly spaced over `spread` from `start`."""
    angles = start + np.linspace(0, spread, n, endpoint=False)
    return [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]


def test_angular_uniformity_high_when_angles_evenly_spread():
    from shrimpy.fov_selection.feature_extraction import group_features

    rec = group_features(_object_df(_ring_centroids(24)))  # objects all around the center
    # Normalized-entropy uniformity is near 1 for evenly-spread angles (exact 1.0 needs
    # points centered in every bin, not on the edges -- same discreteness as occupancy_entropy).
    assert rec["angular_uniformity"] > 0.9


def test_angular_uniformity_low_when_angles_clustered():
    from shrimpy.fov_selection.feature_extraction import group_features

    # All 8 objects to one side of center (a narrow angular wedge) -> non-uniform.
    clustered = _ring_centroids(8, spread=np.deg2rad(20), start=0.0)
    rec = group_features(_object_df(clustered))
    assert rec["angular_uniformity"] < 0.3


def test_angular_uniformity_ordering_uniform_beats_clustered():
    from shrimpy.fov_selection.feature_extraction import group_features

    uniform = group_features(_object_df(_ring_centroids(12)))["angular_uniformity"]
    clustered = group_features(_object_df(_ring_centroids(12, spread=np.deg2rad(30))))[
        "angular_uniformity"
    ]
    assert uniform > clustered


def test_angular_uniformity_is_in_feature_names():
    from shrimpy.fov_selection.feature_extraction import FEATURE_NAMES

    assert "angular_uniformity" in FEATURE_NAMES
