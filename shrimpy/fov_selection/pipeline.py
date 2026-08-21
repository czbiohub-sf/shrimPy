"""Online per-FOV FOV-selection pipeline glue.

Runs one FOV's decision in memory, reusing the shared feature/segmentation code so online
features match training exactly. Feature extraction is decoupled from the model:

    preprocessing output channel(s)
      -> project (sum / max / middle / logstd) [project_zyx]
      -> reduce to ONE segmentation input by the target [_resolve_seg_input]
      -> segment (cellpose / instanseg / otsu) [Segmenter, from segmentation.py]
      -> per-object features          [FeatureExtractor.object_feature_rows]
      -> per-FOV aggregation          [FeatureExtractor.group_features]
      -> named feature table          [extract_features: plain feature keys, one mask]
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


def project_zyx(
    zyx: np.ndarray,
    projection: str = "sum",
    *,
    pixel_size_um: float | None = None,
    best_focus_z: dict | None = None,
    return_index: bool = False,
) -> np.ndarray:
    """Reduce a ``(Z, Y, X)`` volume to a 2D ``(Y, X)`` float32 image.

    ``projection``:
      ``'sum'``             -- sum over Z (trained model default, ``*_sum__*`` features);
      ``'max'``             -- max over Z;
      ``'middle'``          -- the single middle Z slice (``zyx[Z // 2]``);
      ``'logstd'``          -- per-pixel standard deviation over Z, shifted to non-negative and
                               ``log1p``-compressed (``log1p(std - std.min())``). Highlights axial
                               texture / contrast, e.g. for label-free brightfield where a flat
                               sum or middle slice carries little signal.
      ``'best_focus_z'`` -- the single IN-FOCUS Z slice, found by waveorder's transverse-band
                               focus metric (:func:`waveorder.focus.focus_from_transverse_band`).
                               Needs ``pixel_size_um`` and ``best_focus_z`` optics
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
    if zyx.ndim == 2:
        # Already a 2D (Y, X) image: there is no Z axis to project over, so return it
        # unchanged for every projection. Lets an already-projected / single-slice input go
        # straight to segmentation without a projection step.
        img = np.asarray(zyx, np.float32)
        return (img, None) if return_index else img
    if projection == "middle":
        idx = zyx.shape[0] // 2
        img = np.asarray(zyx[idx], np.float32)
        return (img, idx) if return_index else img
    if projection == "best_focus_z":
        img, idx = _best_focus_z(zyx, pixel_size_um, best_focus_z)
        return (img, idx) if return_index else img
    if projection == "logstd":
        std = zyx.astype(np.float32).std(axis=0)
        img = np.log1p(std - std.min()).astype(np.float32)
        return (img, None) if return_index else img
    reduce = np.max if projection == "max" else np.sum
    img = reduce(zyx, axis=0).astype(np.float32)
    return (img, None) if return_index else img


def _best_focus_z(
    zyx: np.ndarray, pixel_size_um: float | None, best_focus_z: dict | None
) -> tuple[np.ndarray, int]:
    """The single best-focus Z slice, via waveorder's transverse-band focus metric.

    :func:`waveorder.focus.focus_from_transverse_band` scores each Z slice by the power in a
    mid spatial-frequency band (set by the detection NA / wavelength / pixel size) and returns
    the index of the extreme (in-focus) slice. Requires ``pixel_size_um`` (object-space pixel size, um)
    plus ``best_focus_z`` optics -- ``numerical_aperture_detection`` and ``wavelength_illumination``
    (um, matching ``pixel_size_um``); optional ``mode`` ('max'|'min') and ``midband_fractions``.

    Two distinct failure modes, handled differently:
      * MISSING optics (``pixel_size_um`` / NA / wavelength) -> :class:`ValueError` (abort). The focus
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
            ("pixel_size_um", pixel_size_um),
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
        pixel_size=float(pixel_size_um),
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
CHEAP_FEATURE_KEYS = frozenset({"coverage_frac", "object_counts", "mask_occupancy_entropy"})


def _cheap_features(mask: np.ndarray, keys: set[str]) -> dict[str, float]:
    """Aggregate features for ``keys`` from the mask alone (no regionprops).

    Numerically identical to the full path for the cheap keys (``coverage_frac`` matches
    ``group_features``; ``object_counts`` counts distinct labels, the same as the length of
    the per-object table; ``mask_occupancy_entropy`` calls the same function
    ``mask_gap_features`` does). An empty mask yields a genuine zero for ``coverage_frac`` and
    ``object_counts``: "no objects" is a real measurement, not missing data, so it is reported
    faithfully for the model to act on -- NOT dropped to NaN, which the median imputer would
    then fill with a typical FOV's value and make an empty FOV look good.
    ``mask_occupancy_entropy`` is NaN on an empty mask, because the spread of a nonexistent
    foreground is genuinely undefined rather than "perfectly concentrated".
    """
    m = np.asarray(mask)
    h, w = m.shape
    out: dict[str, float] = {}
    if "coverage_frac" in keys:
        out["coverage_frac"] = float(np.count_nonzero(m) / (w * h))
    if "object_counts" in keys:
        ids = np.unique(m)
        out["object_counts"] = int(ids[ids != 0].size)
    if "mask_occupancy_entropy" in keys:
        out["mask_occupancy_entropy"] = FeatureExtractor.mask_occupancy_entropy(m)
    return out


def extract_features(
    projections: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    pixel_size_um: float,
    needed: list[str] | None = None,
):
    """One-row feature matrix with PLAIN feature-name columns (``coverage_frac``,
    ``nn_um_mean``, ...) from the ONE segmented mask FOV selection produces.

    Feature names are independent of which channel was segmented (the ``target`` -- combined
    VS, nuclei VS, or a single label-free channel), so a single config model / ranking profile
    applies no matter what produced the mask. Exactly one channel must be present.

    ``needed`` restricts the computed columns (the config model's feature keys); when all
    needed keys are cheap (:data:`CHEAP_FEATURE_KEYS`) the per-object extraction is skipped.
    """
    import pandas as pd

    if len(projections) != 1:
        raise ValueError(
            "channel-independent (flat) feature naming requires exactly one "
            f"segmented channel; got {list(projections)}."
        )
    ((channel, proj),) = projections.items()
    mask = masks[channel]
    keys_needed = set(needed) if needed is not None else None

    if keys_needed is not None and keys_needed <= CHEAP_FEATURE_KEYS:
        agg = _cheap_features(mask, keys_needed)
    else:
        rows = FeatureExtractor.object_feature_rows(mask, proj, pixel_size_um)
        if not rows:
            # No objects: report density features (coverage) as a real zero so the model can
            # act on an empty FOV, instead of dropping to NaN (median-imputed -> looks good);
            # shape/spatial features are genuinely undefined with no objects, so stay NaN.
            logger.warning(
                "FOV selection: no %s objects segmented; density features -> 0, "
                "shape/spatial features -> NaN",
                channel,
            )
            cheap = keys_needed if keys_needed is not None else CHEAP_FEATURE_KEYS
            agg = _cheap_features(mask, set(cheap) & CHEAP_FEATURE_KEYS)
        else:
            agg = FeatureExtractor.group_features(pd.DataFrame(rows))
            if keys_needed is None or (keys_needed & MASK_FEATURE_KEYS):
                agg.update(FeatureExtractor.mask_gap_features(mask, pixel_size_um))

    feat = {k: v for k, v in agg.items() if keys_needed is None or k in keys_needed}
    return pd.DataFrame([feat])


def _to_numpy(x) -> np.ndarray:
    """Detach a torch tensor (or pass through an array) to a numpy array."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _resolve_seg_input(target: str, projections: dict[str, np.ndarray]) -> np.ndarray:
    """Reduce the reconstruction-output projections to the ONE 2D image to segment.

    - virtual staining + ``target='cells'``: sum nuclei + membrane into one grayscale (the
      whole cell body); the InstanSeg 'cells' head then separates touching cells.
    - virtual staining + ``target='nuclei'``: the nuclei channel only.
    - single reconstructed channel (label-free): that channel, whatever the target.
    """
    if target == "cells" and "nuclei" in projections and "membrane" in projections:
        return projections["nuclei"] + projections["membrane"]
    if "nuclei" in projections:
        return projections["nuclei"]
    (only,) = projections.values()
    return only


def decide_fov(
    preprocessor,
    segmenter,
    model,
    bf_zyx: np.ndarray,
    *,
    target: str,
    recon_channels: list[str],
    projection: str = "sum",
    pixel_size_um: float,
    threshold: float = 0.5,
    best_focus_z: dict | None = None,
    return_artifacts: bool = False,
    return_stacks: bool = False,
    extract_all: bool = False,
    label: str = "",
) -> tuple[float, bool] | tuple[float, bool, dict]:
    """Run one FOV's good/bad decision end to end.

    Reconstructs the input z-stack (``preprocessor``), projects and segments each
    ``recon_channels`` channel, extracts a named feature table, and asks ``model``
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
    recon_channels : list[str]
        Reconstructed channels to project (e.g. ``['nuclei', 'membrane']`` or a
        single ``['brightfield']``); reduced to ONE segmentation input by ``target``
        (see :func:`_resolve_seg_input`).
    projection : str
        ``'sum'`` (trained default), ``'max'``, ``'middle'`` (middle-slice), ``'logstd'``
        (log-normalized per-pixel std over Z), or ``'best_focus_z'`` (the in-focus slice
        picked by waveorder; needs ``best_focus_z`` optics -- see :func:`project_zyx`).
    pixel_size_um : float
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
    tuple[float, bool | None]
        ``(proba, good)`` for this FOV -- ``proba`` is the model score (the ranking key for
        ``ranking_by_defined_range``); ``good`` is the classification models' per-FOV verdict,
        or ``None`` for the ranking model (which has no per-FOV good/bad notion -- selection is
        top-K per position, see the manager).
    """
    pfx = f"[{label}] " if label else ""
    bf_zyx = np.asarray(bf_zyx)
    # The per-step 3D intermediates (deskew / phase volumes) are only needed for
    # the reconstruction store, which is expensive to build and write -- so pull
    # them only when return_stacks is set. return_artifacts alone (projections /
    # masks / features for the lightweight PNG/CSV debug) stays cheap.
    #
    # `preprocessor` is None for a pipeline with no reconstruction step (raw stack ->
    # segment; build_preprocessor returns None in that case). The raw input stack is then
    # the single channel, keyed by the caller's chosen channel name (recon_channels[0]) so
    # the same label reaches the segmenter (which picks the Cellpose diameter from it).
    if preprocessor:
        channels = preprocessor(
            bf_zyx, label=label, return_intermediates=return_stacks
        )  # {'nuclei', 'membrane', 'phase', ('deskew')}
    else:
        channels = {recon_channels[0]: bf_zyx}

    # Per-step 3D stacks for the reconstruction store (deskew, phase, and each VS
    # target volume); only populated when return_stacks is set.
    stacks: dict[str, np.ndarray] = {}
    if return_stacks:
        for key in ("deskew", "phase"):
            if key in channels:
                stacks[key] = _to_numpy(channels[key])

    # Project each reconstruction-output channel; for the 'best_focus_z' projection remember
    # which Z slice each came from (and the stack depth) so the worker can log it.
    content_proj: dict[str, np.ndarray] = {}
    best_focus_index: dict[str, dict[str, int]] = {}
    for channel in recon_channels:
        vol = _to_numpy(channels[channel])
        if return_stacks:
            stacks[channel] = vol
        proj, z_idx = project_zyx(
            vol,
            projection,
            pixel_size_um=pixel_size_um,
            best_focus_z=best_focus_z,
            return_index=True,
        )
        if z_idx is not None:
            best_focus_index[channel] = {"slice": int(z_idx), "n_slices": int(vol.shape[0])}
        content_proj[channel] = proj

    # FOV selection segments exactly ONE mask: the `target` reduces the reconstruction
    # outputs to a single 2D input (combined nuclei+membrane for 'cells', nuclei-only for
    # 'nuclei', or the single label-free channel). Features are then single-channel (plain
    # keys), so projections/masks are keyed by the target, not per reconstruction channel.
    seg_input = _resolve_seg_input(target, content_proj)
    try:
        mask = segmenter.segment(seg_input, target, pixel_size_um=pixel_size_um)
    except Exception as exc:
        logger.error("%ssegment %s FAILED: %s", pfx, target, exc)
        raise
    logger.info("%ssegment %s ok (%d objects)", pfx, target, int((np.unique(mask) != 0).sum()))
    projections = {target: seg_input}
    masks = {target: mask}

    needed = None if extract_all else model.feature_names
    matrix = extract_features(projections, masks, pixel_size_um, needed=needed)
    proba, good = model.predict(matrix, threshold)
    # `good` is None for a pure-ranking model (no per-FOV verdict; selection is top_fov in the
    # manager) and a per-FOV bool list for the classification models.
    good_val = None if good is None else bool(good[0])
    if return_artifacts:
        artifacts = {
            "stacks": stacks,
            "projections": projections,
            "masks": masks,
            "features": matrix,
            "best_focus_index": best_focus_index,
        }
        return float(proba[0]), good_val, artifacts
    return float(proba[0]), good_val
