"""FOV feature extraction at every level, in one place.

:class:`FeatureExtractor` turns a segmentation mask (+ its intensity image) into features,
at two levels:

  object level  -> :meth:`FeatureExtractor.object_feature_rows`
                   one row per segmented object (area, shape, intensity, nearest-neighbor).
  FOV level     -> :meth:`FeatureExtractor.group_features`   (from the per-object table)
                   :meth:`FeatureExtractor.mask_gap_features` (mask-pixel spatial features)
                   one row per FOV (coverage, spacing, voids, non-uniformity).

The class is stateless: every method is a class/static method, and the tuning knobs (grid
resolutions, band fractions, ...) are class attributes, so the extractor is used without
instantiation (``FeatureExtractor.group_features(df)``). :data:`FEATURE_NAMES` /
:data:`MASK_FEATURE_KEYS` name the FOV-level feature schema the rest of the pipeline reads.

This module is general pipeline code: it has NO hardcoded paths and does NO batch I/O. The
offline drivers that build training matrices from whole zarr stores live in
``shrimpy/scripts/`` (build_object_features.py, build_fov_matrix.py) and call these methods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table

# Feature keys that need the label mask itself (not just the per-object centroid table);
# produced by mask_gap_features(), so they are NOT in FEATURE_NAMES (the group_features set).
MASK_FEATURE_KEYS = frozenset(
    {
        "max_gap_um",
        "mask_occupancy_entropy",
        "edge_frac",
        "central_cov_ratio",
        "center_gap_norm",
    }
)

# The per-variant features produced by group_features(), in output order. Coverage +
# spacing + large-scale-void / non-uniformity features (an empty center, ring, or single
# clump that COM / local-NN features miss). max_gap_um is mask-derived (mask_gap_features).
FEATURE_NAMES = [
    "coverage_frac",
    "nn_um_mean",
    "nn_cv",
    "com_offset_norm",
    "centroid_radial_mean",
    "empty_grid_frac",
    "occupancy_entropy",
    "angular_uniformity",
    "max_empty_radius_norm",
    "max_empty_edge_offset_norm",
]


class FeatureExtractor:
    """Compute object-level and FOV-level features from segmentation masks.

    All methods are class/static methods; the tuning knobs below are class attributes so a
    variant extractor can subclass and override them without touching the numeric code.
    """

    # skimage regionprops properties (new-style names) used for per-object features.
    PROPS = (
        "label",
        "centroid",
        "area",
        "equivalent_diameter_area",
        "extent",
        "intensity_mean",
        "intensity_max",
    )

    # Quadrat grid for empty_grid_frac / occupancy_entropy. ADAPTIVE: the grid side scales
    # with the object count so each cell holds ~GRID_LAMBDA objects on average (grid
    # resolution tracks density), clamped to [GRID_MIN, GRID_MAX].
    GRID_LAMBDA = 2.0
    GRID_MIN, GRID_MAX = 2, 24

    # Angular sectors for angular_uniformity, ADAPTIVE like the quadrat grid (~ANGULAR_LAMBDA
    # objects per sector), clamped to [ANGULAR_BINS_MIN, ANGULAR_BINS_MAX].
    ANGULAR_LAMBDA = 2.0
    ANGULAR_BINS_MIN, ANGULAR_BINS_MAX = 4, 16

    EMPTY_RADIUS_SAMPLES = 40  # NxN test-point grid for the largest-empty-circle feature

    # central_cov_ratio: central-disk radius as a fraction of the half-diagonal. center_gap_norm:
    # half-size of the window sampled at the image center, as a fraction of min(H, W).
    CENTRAL_DISK_FRAC = 0.40
    CENTER_WINDOW_FRAC = 0.02

    # Border band for edge_frac: an object counts as "edge" if any of its pixels lie within this
    # fraction of the image WIDTH (left/right) or HEIGHT (top/bottom) from an edge.
    EDGE_BAND_FRAC = 0.1

    # Grid side for mask_occupancy_entropy: the FOV is split into MASK_OCCUPANCY_GRID^2 cells.
    # 8 (64 cells) resolves large-scale voids without being so fine that ordinary gaps read as
    # non-uniformity. Over foreground PIXELS, so it is well defined even for a single blob.
    MASK_OCCUPANCY_GRID = 8

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def organelle_label(channel: str) -> str:
        """Organelle the channel labels: 'nuclei_GFP_maxproj' -> 'nuclei',
        'membrane_prediction_sumproj' -> 'membrane' (first underscore-delimited token)."""
        return (channel or "").split("_")[0]

    @classmethod
    def _adaptive_grid(cls, n: int) -> int:
        """Grid side G = round(sqrt(n / GRID_LAMBDA)), clamped to [GRID_MIN, GRID_MAX]."""
        return int(
            np.clip(
                round(np.sqrt(max(int(n), 1) / cls.GRID_LAMBDA)), cls.GRID_MIN, cls.GRID_MAX
            )
        )

    @classmethod
    def _adaptive_angular_bins(cls, n: int) -> int:
        """Sector count K = round(n / ANGULAR_LAMBDA), clamped to [ANGULAR_BINS_MIN, MAX]."""
        return int(
            np.clip(
                round(max(int(n), 1) / cls.ANGULAR_LAMBDA),
                cls.ANGULAR_BINS_MIN,
                cls.ANGULAR_BINS_MAX,
            )
        )

    # -------------------------------------------------------------- object level
    @classmethod
    def object_feature_rows(
        cls,
        lbl,
        intensity,
        px_um,
        *,
        dataset_tag,
        well_row,
        well_col,
        fov,
        timepoint,
        channel,
        source,
        projection_type,
    ):
        """Per-object feature rows from a single (label mask, intensity) pair.

        Array-based core shared by the batch driver and the online FOV-selection decision,
        so both compute identical features.

        Parameters
        ----------
        lbl : np.ndarray
            2D instance-label mask (Y, X), integer ids (0 = background).
        intensity : np.ndarray | None
            2D intensity image (Y, X) the mask was segmented from, or None.
        px_um : float
            XY pixel size in microns (isotropic).
        channel : str
            Organelle/channel label, e.g. ``'nuclei'`` / ``'membrane'``.
        """
        out: list[dict] = []
        lbl = np.asarray(lbl).astype(np.uint32)
        if lbl.max() == 0:
            return out
        img_h, img_w = lbl.shape
        intensity = np.asarray(intensity, np.float32) if intensity is not None else None
        p = regionprops_table(lbl, intensity_image=intensity, properties=cls.PROPS)
        cy, cx = p["centroid-0"], p["centroid-1"]
        # nearest-neighbor distance among same-channel centroids (px -> um)
        if len(cy) >= 2:
            d, _ = cKDTree(np.column_stack([cy, cx])).query(np.column_stack([cy, cx]), k=2)
            nn_px = d[:, 1]
        else:
            nn_px = np.full(len(cy), np.nan)
        for k in range(len(p["label"])):
            cyk, cxk = float(cy[k]), float(cx[k])
            out.append(
                {
                    "dataset": dataset_tag,
                    "well_row": well_row,
                    "well_col": well_col,
                    "fov": fov,
                    "timepoint": timepoint,
                    "channel": channel,
                    "organelle": cls.organelle_label(channel),
                    "segmentation_source": source,
                    "projection_type": projection_type,
                    "label_id": int(p["label"][k]),
                    "centroid_x_norm": cxk / img_w,
                    "centroid_y_norm": cyk / img_h,
                    "area_px": int(p["area"][k]),
                    "area_um2": float(p["area"][k]) * px_um * px_um,
                    "equivalent_diameter_um": float(p["equivalent_diameter_area"][k]) * px_um,
                    "extent": float(p["extent"][k]),
                    "intensity_mean": float(p["intensity_mean"][k]),
                    "intensity_max": float(p["intensity_max"][k]),
                    "nearest_neighbor_dist_um": float(nn_px[k]) * px_um,
                    "image_width_px": img_w,
                    "image_height_px": img_h,
                    "pixel_size_um": px_um,
                }
            )
        return out

    # ----------------------------------------------------------------- FOV level
    @classmethod
    def group_features(cls, g: pd.DataFrame) -> dict:
        """FOV-level features aggregated from a per-object table (one FOV's rows)."""
        n = len(g)
        width = float(g["image_width_px"].iloc[0])
        height = float(g["image_height_px"].iloc[0])
        px = float(g["pixel_size_um"].iloc[0])
        cx = g["centroid_x_norm"].to_numpy(float) * width
        cy = g["centroid_y_norm"].to_numpy(float) * height

        rec = {
            "coverage_frac": float(g["area_px"].sum() / (width * height)),
        }
        # Nearest-neighbor spacing in PHYSICAL units (um) so it is invariant to
        # magnification (pixel size); NN distance is a local density measure, so it is
        # independent of FOV size too. nn_cv (std/mean) is a unitless ratio.
        if "nearest_neighbor_dist_um" in g.columns:
            nn_um = g["nearest_neighbor_dist_um"].to_numpy(float)
        elif "nearest_neighbor_dist_px" in g.columns:
            nn_um = g["nearest_neighbor_dist_px"].to_numpy(float) * px
        else:
            nn_um = np.full(n, np.nan)
        nn_um = nn_um[~np.isnan(nn_um)]
        rec["nn_um_mean"] = float(nn_um.mean()) if nn_um.size else np.nan
        rec["nn_cv"] = (
            float(nn_um.std() / nn_um.mean()) if nn_um.size and nn_um.mean() else np.nan
        )
        # Center-of-mass offset: distance from the (unweighted) mean object centroid to the
        # FOV center, normalized by the half-diagonal. 0 = centered on average; -> 1 = clumped
        # toward one side / corner (off-center or partially-covered FOV).
        com = np.array([cx.mean(), cy.mean()])
        half_diag = 0.5 * np.hypot(width, height)
        rec["com_offset_norm"] = float(np.hypot(*(com - [width / 2, height / 2])) / half_diag)
        # Mean radial distance of each object centroid from the FOV center, normalized by the
        # half-diagonal (0 = every cell centered, ~1 = every cell in a corner). ASYMMETRIC
        # counterpart to com_offset_norm: two cells in opposite corners score ~1, not 0.
        r_norm = np.hypot(cx - width / 2.0, cy - height / 2.0) / half_diag
        rec["centroid_radial_mean"] = float(r_norm.mean())
        # --- spatial-distribution features (voids / non-uniformity that local NN misses) ---
        grid = cls._adaptive_grid(n)  # grid resolution tracks the object count
        gx = np.clip((cx / width * grid).astype(int), 0, grid - 1)
        gy = np.clip((cy / height * grid).astype(int), 0, grid - 1)
        counts = np.zeros(grid * grid, int)
        np.add.at(counts, gy * grid + gx, 1)
        n_cells = grid * grid
        # Fraction of grid cells with no object -> high for a ring / empty center / one clump.
        rec["empty_grid_frac"] = float((counts == 0).sum() / n_cells)
        # Occupancy spread across the grid (normalized Shannon entropy): uniform->1, concentrated->0.
        p = counts[counts > 0] / counts.sum()
        rec["occupancy_entropy"] = (
            float(-(p * np.log(p)).sum() / np.log(n_cells)) if p.size else np.nan
        )
        # Angular uniformity: how evenly the centroids are spread in ANGLE around the FOV
        # center. Bin angles into adaptive sectors, take normalized Shannon entropy: 1 ->
        # objects surround the center evenly, 0 -> all clustered in one direction.
        k = cls._adaptive_angular_bins(n)
        ang = np.arctan2(cy - height / 2.0, cx - width / 2.0)  # (-pi, pi]
        abin = np.clip(((ang + np.pi) / (2 * np.pi) * k).astype(int), 0, k - 1)
        acounts = np.bincount(abin, minlength=k).astype(float)
        ap = acounts[acounts > 0] / acounts.sum()
        rec["angular_uniformity"] = (
            float(-(ap * np.log(ap)).sum() / np.log(k)) if ap.size and k > 1 else np.nan
        )
        # Largest object-free circle, from the centroids (grid-sampled), normalized by the
        # half-diagonal: the radius of the biggest gap -> large when there is a big empty area.
        s = cls.EMPTY_RADIUS_SAMPLES
        gxs, gys = np.meshgrid(np.linspace(0, width, s), np.linspace(0, height, s))
        pts = np.column_stack([gxs.ravel(), gys.ravel()])
        dmin, _ = cKDTree(np.column_stack([cx, cy])).query(pts)
        imax = int(np.argmax(dmin))
        r_gap = float(dmin[imax])  # radius of the largest empty circle (px)
        rec["max_empty_radius_norm"] = r_gap / half_diag
        # Where that largest void sits relative to the FOV center: signed clearance from the
        # image center to the void's NEAR edge, normalized by the half-diagonal. Higher better:
        #   < 0 -> void engulfs the image center (central hole; worst); > 0 -> void at periphery.
        void_cx, void_cy = float(pts[imax, 0]), float(pts[imax, 1])
        center_to_void = np.hypot(void_cx - width / 2.0, void_cy - height / 2.0)
        rec["max_empty_edge_offset_norm"] = float((center_to_void - r_gap) / half_diag)
        return rec

    @classmethod
    def mask_occupancy_entropy(cls, mask: np.ndarray) -> float:
        """Normalized Shannon entropy of the foreground-PIXEL distribution over a coarse grid.

        The FOV is split into ``MASK_OCCUPANCY_GRID**2`` cells and the foreground pixels are
        binned into them; the value is ``-sum(p log p) / log(n_cells)`` over occupied cells.

            1.0 -> foreground spread evenly across the whole FOV
            ~0  -> all foreground concentrated in a single cell
            NaN -> empty mask (no foreground; the distribution is undefined)

        Mask-pixel counterpart to :data:`FEATURE_NAMES`' ``occupancy_entropy`` (which bins
        object CENTROIDS): binning pixels removes the dependence on object count entirely.
        """
        fg = np.asarray(mask) > 0
        if not fg.any():
            return float("nan")
        g = cls.MASK_OCCUPANCY_GRID
        h, w = fg.shape
        yi = (np.arange(h, dtype=np.int32) * g) // h
        xi = (np.arange(w, dtype=np.int32) * g) // w
        cell = (yi[:, None] * g + xi[None, :]).astype(np.int32)
        counts = np.bincount(cell[fg], minlength=g * g).astype(float)
        p = counts[counts > 0] / counts.sum()
        return float(-(p * np.log(p)).sum() / np.log(g * g))

    @classmethod
    def edge_object_frac(cls, mask: np.ndarray, band_frac: float | None = None) -> float:
        """Fraction of segmented objects touching the image-edge border band.

        An object is an "edge" object if ANY of its pixels lie within ``band_frac`` of the
        image width from the left/right edge OR of the image height from the top/bottom edge.
        Returns ``n_edge_objects / n_objects`` in ``[0, 1]`` (NaN for an empty mask). The
        default band is :attr:`EDGE_BAND_FRAC` of each dimension.
        """
        if band_frac is None:
            band_frac = cls.EDGE_BAND_FRAC
        m = np.asarray(mask)
        labels = np.unique(m)
        labels = labels[labels != 0]
        if labels.size == 0:
            return float("nan")
        h, w = m.shape
        mx = int(round(band_frac * w))
        my = int(round(band_frac * h))
        border = np.zeros(m.shape, bool)
        if mx > 0:
            border[:, :mx] = True
            border[:, w - mx :] = True
        if my > 0:
            border[:my, :] = True
            border[h - my :, :] = True
        edge_labels = np.unique(m[border])
        edge_labels = edge_labels[edge_labels != 0]
        return float(edge_labels.size / labels.size)

    @classmethod
    def mask_gap_features(cls, mask: np.ndarray, px_um: float) -> dict:
        """Spatial features that need the mask itself (not just centroids).

        ``max_gap_um``: radius (um) of the largest object-free region (max over background
        pixels of the distance to the nearest foreground pixel). Empty mask -> NaN.
        ``mask_occupancy_entropy``: foreground-pixel spread (see :meth:`mask_occupancy_entropy`).
        ``edge_frac``: fraction of objects stuck to the border band (see :meth:`edge_object_frac`).
        ``central_cov_ratio``: coverage inside the central disk / whole-FOV coverage
        (coverage-independent; < 1 for a central void or edge ring). Empty mask -> NaN.
        ``center_gap_norm``: radius of empty space AT the image center, normalized by the
        half-diagonal (localized central-blank detector). Empty mask -> NaN.
        """
        m = np.asarray(mask)
        fg = m > 0
        keys = (
            "max_gap_um",
            "mask_occupancy_entropy",
            "edge_frac",
            "central_cov_ratio",
            "center_gap_norm",
        )
        if not fg.any():
            return {k: float("nan") for k in keys}

        dt = distance_transform_edt(~fg)  # background -> nearest-foreground distance
        h, w = fg.shape
        cy0, cx0 = (h - 1) / 2.0, (w - 1) / 2.0
        half = 0.5 * float(np.hypot(h, w))
        # central occupancy relative to overall coverage
        yy, xx = np.ogrid[0:h, 0:w]
        rn = np.hypot(yy - cy0, xx - cx0) / half
        central = rn <= cls.CENTRAL_DISK_FRAC
        overall = float(fg.mean())
        central_cov = float(fg[central].mean()) if central.any() else float("nan")
        central_cov_ratio = central_cov / overall if overall > 0 else float("nan")
        # empty radius sampled at the image center
        win = max(1, int(round(cls.CENTER_WINDOW_FRAC * min(h, w))))
        yc, xc = int(round(cy0)), int(round(cx0))
        center_gap_norm = float(
            dt[yc - win : yc + win + 1, xc - win : xc + win + 1].min() / half
        )
        return {
            "max_gap_um": float(dt.max()) * float(px_um),
            "mask_occupancy_entropy": cls.mask_occupancy_entropy(fg),
            "edge_frac": cls.edge_object_frac(m),
            "central_cov_ratio": central_cov_ratio,
            "center_gap_norm": center_gap_norm,
        }


# Module-level functional API: thin aliases to the FeatureExtractor class methods, for callers
# that prefer plain functions and to keep import sites stable. The class remains the single,
# cohesive home of the feature code -- these are just its bound methods re-exported here.
object_feature_rows = FeatureExtractor.object_feature_rows
group_features = FeatureExtractor.group_features
mask_gap_features = FeatureExtractor.mask_gap_features
mask_occupancy_entropy = FeatureExtractor.mask_occupancy_entropy
edge_object_frac = FeatureExtractor.edge_object_frac
