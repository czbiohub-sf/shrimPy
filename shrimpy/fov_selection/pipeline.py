"""Online per-FOV FOV-selection pipeline glue.

Runs one FOV's decision in memory, reusing the shared feature/segmentation code so online
features match training exactly. Feature extraction is decoupled from the model:

    preprocessing output channel(s)
      -> project (sum / max / middle / logstd) [project_zyx]
      -> segment (cellpose / instanseg / otsu) [Segmenter, from segmentation.py]
      -> per-object features          [FeatureExtractor.object_feature_rows]
      -> per-FOV aggregation          [FeatureExtractor.group_features]
      -> named feature table          [extract_features: plain names for one channel,
                                        <organelle>_vs_<projection>__ prefixed for many]
      -> verdict                      [FovModel.predict, fov_model.py]

The model (thresholding / desirability / trained tree / ...) reads the feature table
by NAME only -- it never sees which channel produced a feature -- so any model type
pairs with any preprocessing (see :mod:`shrimpy.fov_selection.fov_model`).

Heavy imports (cellpose, torch, pandas) are done lazily so importing this module stays cheap.
"""

from __future__ import annotations

import logging

import numpy as np

from shrimpy.fov_selection.feature_extraction import MASK_FEATURE_KEYS, FeatureExtractor

logger = logging.getLogger(__name__)


def save_mask_overlay_png(
    path,
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (255, 0, 255),
    thickness: int = 2,
) -> None:
    """Save a PNG of ``image`` (contrast-stretched grayscale) with the ``mask`` object
    OUTLINES drawn on top as a bright border (default magenta), so the segmentation is
    visible without covering the cells it segments.

    The image is 1-99 percentile stretched to 8-bit grayscale (as the plain projection PNG
    is), converted to RGB, and each object's boundary pixels are painted ``color``.
    ``thickness`` px wide boundary (dilated for visibility). Shared by the live FOV-selection
    worker and the offline PNG exporter so both produce identical overlays.
    """
    import imageio.v3 as iio

    from skimage.segmentation import find_boundaries

    img = np.asarray(image, np.float32)
    finite = img[np.isfinite(img)]
    if finite.size:
        lo, hi = np.percentile(finite, (1.0, 99.0))
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        lo, hi = float(img.min()), float(img.max())
    gray = np.zeros_like(img) if hi <= lo else np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    rgb = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2)
    boundaries = find_boundaries(np.asarray(mask), mode="outer")
    if thickness > 1 and boundaries.any():
        from scipy.ndimage import binary_dilation

        boundaries = binary_dilation(boundaries, iterations=int(thickness) - 1)
    rgb[boundaries] = np.asarray(color, np.uint8)
    iio.imwrite(path, rgb)


def project_zyx(
    zyx: np.ndarray,
    method: str = "sum",
    *,
    px_um: float | None = None,
    best_focus_z: dict | None = None,
    return_index: bool = False,
) -> np.ndarray:
    """Reduce a ``(Z, Y, X)`` volume to a 2D ``(Y, X)`` float32 image.

    ``method``:
      ``'sum'``             -- sum over Z (trained model default, ``*_sum__*`` features);
      ``'max'``             -- max over Z;
      ``'middle'``          -- the single middle Z slice (``zyx[Z // 2]``);
      ``'logstd'``          -- per-pixel standard deviation over Z, shifted to non-negative and
                               ``log1p``-compressed (``log1p(std - std.min())``). Highlights axial
                               texture / contrast, e.g. for label-free brightfield where a flat
                               sum or middle slice carries little signal.
      ``'best_focus_z'`` -- the single IN-FOCUS Z slice, found by waveorder's transverse-band
                               focus metric (:func:`waveorder.focus.focus_from_transverse_band`).
                               Needs ``px_um`` and ``best_focus_z`` optics
                               (``numerical_aperture_detection`` + ``wavelength_illumination``);
                               see :func:`_best_focus_z`.

    All are channel-agnostic: they operate on any z-stack, so a projection is usable on a
    deskewed brightfield stack, a phase volume, a virtual-staining channel, etc. -- selected
    via the projection preprocessing step.

    When ``return_index`` is set, returns ``(image, z_index)`` where ``z_index`` is the
    source Z slice the image came from (``best_focus_z`` -> the picked in-focus slice,
    ``middle`` -> ``Z // 2``) or ``None`` for the reducing projections (sum/max/logstd,
    which combine all slices and have no single source index).
    """
    zyx = np.asarray(zyx)
    if method == "middle":
        idx = zyx.shape[0] // 2
        img = np.asarray(zyx[idx], np.float32)
        return (img, idx) if return_index else img
    if method == "best_focus_z":
        img, idx = _best_focus_z(zyx, px_um, best_focus_z)
        return (img, idx) if return_index else img
    if method == "logstd":
        std = zyx.astype(np.float32).std(axis=0)
        img = np.log1p(std - std.min()).astype(np.float32)
        return (img, None) if return_index else img
    reduce = np.max if method == "max" else np.sum
    img = reduce(zyx, axis=0).astype(np.float32)
    return (img, None) if return_index else img


def _best_focus_z(
    zyx: np.ndarray, px_um: float | None, best_focus_z: dict | None
) -> tuple[np.ndarray, int]:
    """The single best-focus Z slice, via waveorder's transverse-band focus metric.

    :func:`waveorder.focus.focus_from_transverse_band` scores each Z slice by the power in a
    mid spatial-frequency band (set by the detection NA / wavelength / pixel size) and returns
    the index of the extreme (in-focus) slice. Requires ``px_um`` (object-space pixel size, um)
    plus ``best_focus_z`` optics -- ``numerical_aperture_detection`` and ``wavelength_illumination``
    (um, matching ``px_um``); optional ``mode`` ('max'|'min') and ``midband_fractions``.

    Two distinct failure modes, handled differently:
      * MISSING optics (``px_um`` / NA / wavelength) -> :class:`ValueError` (abort). The focus
        metric is meaningless without them, so we refuse rather than silently mis-project.
      * optics present but NO confident in-focus slice -> fall back to the middle slice with a
        warning (a legitimately flat/empty stack should not crash the scan).
    """
    zyx = np.asarray(zyx, np.float32)
    mid = zyx.shape[0] // 2
    best_focus_z = best_focus_z or {}
    na_det = best_focus_z.get("numerical_aperture_detection")
    lambda_ill = best_focus_z.get("wavelength_illumination")
    missing = [
        name
        for name, val in (
            ("px_um", px_um),
            ("best_focus_z.numerical_aperture_detection", na_det),
            ("best_focus_z.wavelength_illumination", lambda_ill),
        )
        if not val
    ]
    if missing:
        raise ValueError(
            "best_focus_z projection requires " + ", ".join(missing) + " (detection NA + "
            "illumination wavelength in um, matching the pixel size); aborting rather than "
            "silently using the middle slice."
        )

    from waveorder.focus import focus_from_transverse_band

    idx = focus_from_transverse_band(
        zyx,
        NA_det=float(na_det),
        lambda_ill=float(lambda_ill),
        pixel_size=float(px_um),
        mode=best_focus_z.get("mode", "max"),
        midband_fractions=tuple(best_focus_z.get("midband_fractions", (0.125, 0.25))),
    )
    if idx is None:  # no confident focus (e.g. threshold_FWHM) -> middle slice
        logger.warning("best_focus_z: no confident in-focus slice found; using z=%d.", mid)
        idx = mid
    idx = int(idx)
    logger.info("best_focus_z: in-focus slice z=%d of %d", idx, zyx.shape[0])
    return np.asarray(zyx[idx], np.float32), idx


# Aggregate features derivable from the label mask alone (total coverage, foreground-pixel
# spread), i.e. WITHOUT regionprops shape props or cKDTree spacing. When the model only needs
# these, the expensive per-object extraction is skipped. Values are identical to the ones the
# full path produces (group_features / mask_gap_features call the same code).
CHEAP_FEATURE_KEYS = frozenset({"coverage_frac", "mask_occupancy_entropy"})


def _parse_needed_features(needed: list[str]) -> dict[tuple[str, str, str], set[str]]:
    """Group model feature columns by ``(organelle, source, projection)`` prefix.

    Column convention: ``<organelle>_<source>_<projection>__<key>``. The last two
    underscore tokens of the prefix are source/projection; the rest is the
    organelle (which may itself contain underscores).
    """
    groups: dict[tuple[str, str, str], set[str]] = {}
    for col in needed:
        prefix, _, key = col.partition("__")
        parts = prefix.split("_")
        if len(parts) < 3:
            continue
        organelle = "_".join(parts[:-2])
        groups.setdefault((organelle, parts[-2], parts[-1]), set()).add(key)
    return groups


def _cheap_features(mask: np.ndarray, px_um: float, keys: set[str]) -> dict[str, float]:
    """Aggregate features for ``keys`` from the mask alone (no regionprops).

    Numerically identical to the full path for the cheap keys (``coverage_frac`` matches
    ``group_features``; ``mask_occupancy_entropy`` calls the same function
    ``mask_gap_features`` does). An empty mask yields a genuine zero for
    ``coverage_frac``: "no objects" is a real measurement, not missing data, so it is
    reported faithfully for the model to act on -- NOT dropped to NaN, which the median
    imputer would then fill with a typical FOV's value and make an empty FOV look good.
    ``mask_occupancy_entropy`` is NaN on an empty mask, because the spread of a
    nonexistent foreground is genuinely undefined rather than "perfectly concentrated".
    """
    m = np.asarray(mask)
    h, w = m.shape
    out: dict[str, float] = {}
    if "coverage_frac" in keys:
        out["coverage_frac"] = float(np.count_nonzero(m) / (w * h))
    if "mask_occupancy_entropy" in keys:
        out["mask_occupancy_entropy"] = FeatureExtractor.mask_occupancy_entropy(m)
    return out


def fov_feature_matrix(
    projections: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    px_um: float,
    projection: str = "sum",
    source: str = "vs",
    needed: list[str] | None = None,
):
    """Build a 1-row feature matrix (variant-prefixed columns) for one FOV.

    ``projections`` / ``masks`` map organelle name (``'nuclei'``, ``'membrane'``)
    to its 2D projection / label mask. Column names follow the training
    convention ``<organelle>_<source>_<projection>__<feature>`` (e.g.
    ``nuclei_vs_sum__coverage_frac``), computed via the shared
    ``object_feature_rows`` + ``group_features``.

    When ``needed`` (the trained model's feature list) is given, only those
    columns are computed: organelles that contribute no needed feature are
    skipped, and organelles whose needed features are all cheap
    (:data:`CHEAP_FEATURE_KEYS`) skip the expensive per-object extraction. This
    is a pure speed-up -- the computed values are identical to the full path.
    """
    import pandas as pd

    groups = _parse_needed_features(needed) if needed is not None else None

    feat: dict[str, float] = {}
    for organelle, proj in projections.items():
        prefix = f"{organelle}_{source}_{projection}"
        keys_needed = None
        if groups is not None:
            keys_needed = groups.get((organelle, source, projection))
            if not keys_needed:
                continue  # this organelle contributes no needed feature

        if keys_needed is not None and keys_needed <= CHEAP_FEATURE_KEYS:
            agg = _cheap_features(masks[organelle], px_um, keys_needed)
        else:
            rows = FeatureExtractor.object_feature_rows(
                masks[organelle],
                proj,
                px_um,
                dataset_tag="live",
                well_row="",
                well_col="",
                fov="",
                timepoint=0,
                channel=organelle,
                source=source,
                projection_type=projection,
            )
            if not rows:
                # No objects segmented: faithfully report coverage_frac as a real zero
                # so the model can act on an empty FOV, instead of dropping the organelle
                # entirely -> every column NaN -> median-imputed to a typical FOV ->
                # empty FOV misclassified good. Spatial features are genuinely undefined
                # with no objects, so they stay absent -> NaN.
                logger.warning(
                    "FOV selection: no %s objects segmented; density features -> 0, "
                    "shape/spatial features -> NaN",
                    organelle,
                )
                cheap = keys_needed if keys_needed is not None else CHEAP_FEATURE_KEYS
                agg = _cheap_features(masks[organelle], px_um, set(cheap) & CHEAP_FEATURE_KEYS)
            else:
                agg = FeatureExtractor.group_features(pd.DataFrame(rows))
                if keys_needed is None or (keys_needed & MASK_FEATURE_KEYS):
                    agg.update(FeatureExtractor.mask_gap_features(masks[organelle], px_um))

        for k, v in agg.items():
            if keys_needed is None or k in keys_needed:
                feat[f"{prefix}__{k}"] = v
    return pd.DataFrame([feat])


def flat_feature_matrix(
    projections: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    px_um: float,
    needed: list[str] | None = None,
):
    """One-row matrix with PLAIN feature-name columns (``coverage_frac``, ``nn_um_mean``,
    ...), i.e. WITHOUT the ``<organelle>_<source>_<projection>__`` prefix that
    :func:`fov_feature_matrix` adds.

    Feature names are then independent of which preprocessing-output channel was
    segmented, so a single config model / ranking profile applies no matter what channel
    (deskewed brightfield, phase, a VS target, ...) produced the mask. Because plain names
    cannot disambiguate multiple channels, exactly one channel must be present (a non-VS
    preprocessing emits a single reconstructed channel).

    ``needed`` restricts the computed columns (the config model's feature keys); when all
    needed keys are cheap (:data:`CHEAP_FEATURE_KEYS`) the per-object extraction is skipped.
    Values match :func:`fov_feature_matrix` (same ``group_features`` / ``_cheap_features``).
    """
    import pandas as pd

    if len(projections) != 1:
        raise ValueError(
            "channel-independent (flat) feature naming requires exactly one "
            f"segmented channel; got {list(projections)}."
        )
    ((organelle, proj),) = projections.items()
    mask = masks[organelle]
    keys_needed = set(needed) if needed is not None else None

    if keys_needed is not None and keys_needed <= CHEAP_FEATURE_KEYS:
        agg = _cheap_features(mask, px_um, keys_needed)
    else:
        rows = FeatureExtractor.object_feature_rows(
            mask,
            proj,
            px_um,
            dataset_tag="live",
            well_row="",
            well_col="",
            fov="",
            timepoint=0,
            channel=organelle,
            source="live",
            projection_type="proj",
        )
        if not rows:
            # No objects: report density features as a real zero (see fov_feature_matrix).
            logger.warning(
                "FOV selection: no %s objects segmented; density features -> 0, "
                "shape/spatial features -> NaN",
                organelle,
            )
            cheap = keys_needed if keys_needed is not None else CHEAP_FEATURE_KEYS
            agg = _cheap_features(mask, px_um, set(cheap) & CHEAP_FEATURE_KEYS)
        else:
            agg = FeatureExtractor.group_features(pd.DataFrame(rows))
            if keys_needed is None or (keys_needed & MASK_FEATURE_KEYS):
                agg.update(FeatureExtractor.mask_gap_features(mask, px_um))

    feat = {k: v for k, v in agg.items() if keys_needed is None or k in keys_needed}
    return pd.DataFrame([feat])


def extract_features(
    projections: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    px_um: float,
    projection: str = "sum",
    needed: list[str] | None = None,
):
    """Model-agnostic per-FOV feature table (preprocessing/segmentation -> named features).

    Naming is decided by the number of segmented channels, NOT by the model:

      one channel   -> PLAIN, channel-independent names (``coverage_frac``, ...) via
                       :func:`flat_feature_matrix`.
      many channels -> ``<organelle>_vs_<projection>__<feature>`` prefixed names via
                       :func:`fov_feature_matrix`, so same-named features from different
                       channels stay distinct.

    Either way the model consumes columns by name (``FovModel.feature_names``); it never
    sees which channel produced a feature. ``needed`` restricts the computed columns.
    """
    if len(projections) == 1:
        return flat_feature_matrix(projections, masks, px_um, needed=needed)
    return fov_feature_matrix(
        projections, masks, px_um, projection, source="vs", needed=needed
    )


def _to_numpy(x) -> np.ndarray:
    """Detach a torch tensor (or pass through an array) to a numpy array."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def decide_fov(
    preprocessor,
    segmenter,
    model,
    bf_zyx: np.ndarray,
    *,
    target_channels: list[str],
    projection: str = "sum",
    px_um: float,
    threshold: float = 0.5,
    best_focus_z: dict | None = None,
    return_artifacts: bool = False,
    return_stacks: bool = False,
    extract_all: bool = False,
    label: str = "",
) -> tuple[float, bool] | tuple[float, bool, dict]:
    """Run one FOV's good/bad decision end to end.

    Reconstructs the input z-stack (``preprocessor``), projects and segments each
    ``target_channels`` channel, extracts a named feature table, and asks ``model``
    (any :class:`~shrimpy.fov_selection.fov_model.FovModel`) for the verdict. Feature
    extraction is model-agnostic (see :func:`extract_features`) and the model reads only
    feature names, so any model type pairs with any preprocessing. Shared by the streaming
    worker and tests.

    When ``return_artifacts`` is set, also returns a dict with the per-channel 2D
    ``projections`` and label ``masks`` and the 1-row ``features`` DataFrame (used
    by the worker for the lightweight PNG/CSV debug artifacts). ``return_stacks``
    (which implies ``return_artifacts``) additionally fills ``stacks`` with the
    per-step 3D volumes (``deskew`` / ``phase`` intermediates + each VS target) for
    the reconstruction OME-Zarr; it is kept separate because those volumes are
    large and only the pre-scan reconstruction store needs them.

    Parameters
    ----------
    preprocessor : callable
        ``build_preprocessor(...)`` result; ``(Z, Y, X) -> {channel: ZYX tensor}``.
    segmenter : Segmenter
        Loaded segmentation backend from
        :func:`shrimpy.fov_selection.segmentation.build_segmenter` (Cellpose / InstanSeg /
        Otsu); it carries its own config, so the segmentation block is not re-passed here.
    model : FovModel
        Any :class:`~shrimpy.fov_selection.fov_model.FovModel` (thresholding, desirability,
        trained tree, ...); consumes the extracted features by name.
    bf_zyx : np.ndarray
        Raw input-channel z-stack ``(Z, Y, X)``.
    target_channels : list[str]
        Reconstructed channels to segment/feature (e.g. ``['nuclei', 'membrane']`` or a
        single ``['brightfield']``).
    projection : str
        ``'sum'`` (trained default), ``'max'``, ``'middle'`` (middle-slice), ``'logstd'``
        (log-normalized per-pixel std over Z), or ``'best_focus_z'`` (the in-focus slice
        picked by waveorder; needs ``best_focus_z`` optics -- see :func:`project_zyx`).
    px_um : float
        XY pixel size in microns (physical feature units; also the focus pixel size).
    threshold : float
        P(good) cutoff.
    best_focus_z : dict | None
        Optics for the ``'best_focus_z'`` projection: ``numerical_aperture_detection`` and
        ``wavelength_illumination`` (um), optional ``mode`` / ``midband_fractions``. Ignored by
        the other projection methods.
    extract_all : bool
        Compute EVERY producible feature column instead of only the ones the model reads
        (``model.feature_names``). Used by the calibration pre-scan, whose whole purpose is
        to populate the feature viewer with all features so the user can choose which to
        rank on; the model still runs (its score is ignored by calibration).
    label : str
        FOV/position name, prefixed to per-step logs so preprocessing and
        segmentation success/failure is attributable to a specific FOV.

    Returns
    -------
    tuple[float, bool]
        ``(proba, good)`` for this FOV -- ``proba`` is the model score (the ranking key for
        ``ranking_by_defined_range``); ``good`` is the model's per-FOV verdict (unused by the
        ranking selection, which is top-K over all FOVs -- see the manager).
    """
    pfx = f"[{label}] " if label else ""
    bf_zyx = np.asarray(bf_zyx)
    # The per-step 3D intermediates (deskew / phase volumes) are only needed for
    # the reconstruction store, which is expensive to build and write -- so pull
    # them only when return_stacks is set. return_artifacts alone (projections /
    # masks / features for the lightweight PNG/CSV debug) stays cheap.
    ### HACK
    if preprocessor:
        channels = preprocessor(
            bf_zyx, label=label, return_intermediates=return_stacks
        )  # {'nuclei', 'membrane', 'phase', ('deskew')}
    else:
        # No reconstruction: the raw input stack is the single channel. Key it by the
        # label the caller chose (target_channels[0]) so the same label reaches the
        # segmenter (which picks the Cellpose diameter from it) downstream.
        channels = {target_channels[0]: bf_zyx}

    # Per-step 3D stacks for the reconstruction store (deskew, phase, and each VS
    # target volume); only populated when return_stacks is set.
    stacks: dict[str, np.ndarray] = {}
    if return_stacks:
        for key in ("deskew", "phase"):
            if key in channels:
                stacks[key] = _to_numpy(channels[key])

    projections: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    # For the 'best_focus_z' projection, remember which Z slice each channel was
    # projected from (and the stack depth) so the worker can log it for debugging.
    best_focus_index: dict[str, dict[str, int]] = {}
    for organelle in target_channels:
        vol = _to_numpy(channels[organelle])
        if return_stacks:
            stacks[organelle] = vol
        proj, z_idx = project_zyx(
            vol, projection, px_um=px_um, best_focus_z=best_focus_z, return_index=True
        )
        if z_idx is not None:
            best_focus_index[organelle] = {"slice": int(z_idx), "n_slices": int(vol.shape[0])}
        projections[organelle] = proj
        try:
            mask = segmenter.segment(proj, organelle, px_um=px_um)
        except Exception as exc:
            logger.error("%ssegment %s FAILED: %s", pfx, organelle, exc)
            raise
        masks[organelle] = mask
        n_objects = int((np.unique(mask) != 0).sum())
        logger.info("%ssegment %s ok (%d objects)", pfx, organelle, n_objects)

    needed = None if extract_all else model.feature_names
    matrix = extract_features(projections, masks, px_um, projection, needed=needed)
    proba, good = model.predict(matrix, threshold)
    if return_artifacts:
        artifacts = {
            "stacks": stacks,
            "projections": projections,
            "masks": masks,
            "features": matrix,
            "best_focus_index": best_focus_index,
        }
        return float(proba[0]), bool(good[0]), artifacts
    return float(proba[0]), bool(good[0])
