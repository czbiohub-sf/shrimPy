"""Online per-FOV FOV-selection pipeline glue.

Runs one FOV's decision in memory, reusing the offline pipeline's feature code
so the online features match training exactly:

    reconstructed channels (nuclei, membrane)
      -> project (sum)               [project_zyx]
      -> cellpose segment            [segment_2d]  (cpdino params from segment_cellpose)
      -> per-object features         [object_feature_rows, from extract_fov_features]
      -> per-FOV aggregation         [group_features, from build_fov_feature_matrix]
      -> variant-prefixed matrix     [fov_feature_matrix]
      -> decision-tree predict       [predict_good]  (the trained .joblib)

Heavy imports (cellpose, joblib, pandas) are done lazily inside functions so
importing this module stays cheap.
"""

from __future__ import annotations

import logging

import numpy as np

from shrimpy.fov_selection.build_fov_feature_matrix import group_features
from shrimpy.fov_selection.extract_fov_features import object_feature_rows

logger = logging.getLogger(__name__)


def project_zyx(zyx: np.ndarray, method: str = "sum") -> np.ndarray:
    """Z-project a ``(Z, Y, X)`` volume to a 2D ``(Y, X)`` float32 image.

    ``method`` is ``'sum'`` or ``'max'`` -- the trained model uses the SUM
    projection (``*_vs_sum__*`` features).
    """
    zyx = np.asarray(zyx)
    reduce = np.max if method == "max" else np.sum
    return reduce(zyx, axis=0).astype(np.float32)


def load_cellpose_model(segmentation: dict | None = None):
    """Load a Cellpose model once (reuse across FOVs).

    ``segmentation`` is the config block: ``model_name`` (defaults to the batch
    script's ``MODEL_NAME``, e.g. ``'cpdino'``) and ``gpu``. This is the
    switchable-segmentation entry point -- other ``model`` backends (e.g.
    watershed) would branch here.
    """
    from cellpose import models

    from shrimpy.fov_selection import segment_cellpose as sc

    seg = segmentation or {}
    backend = seg.get("model", "cellpose")
    if backend != "cellpose":
        raise NotImplementedError(
            f"segmentation.model={backend!r} is not supported yet; only 'cellpose'."
        )
    name = seg.get("model_name") or getattr(sc, "MODEL_NAME", "cpsam")
    gpu = seg.get("gpu", True)
    logger.info("FOV selection: loading Cellpose model %r (gpu=%s)", name, gpu)
    return models.CellposeModel(gpu=gpu, pretrained_model=name)


def _diameter_for(organelle: str, segmentation: dict | None = None) -> float | None:
    """Per-organelle Cellpose diameter (microns); ``None`` means auto-scale.

    Prefers an explicit ``segmentation.diameters[organelle]`` from config; falls
    back to the batch script's membrane-hint policy (membrane gets an explicit
    diameter, nuclei auto-scale).
    """
    diameters = (segmentation or {}).get("diameters") or {}
    if organelle in diameters:
        return diameters[organelle]

    from shrimpy.fov_selection import segment_cellpose as sc

    hint = getattr(sc, "MEMBRANE_HINT", "membrane")
    if hint in organelle.lower():
        return getattr(sc, "MEMBRANE_DIAMETER", 120.0)
    return getattr(sc, "NUCLEI_DIAMETER", None)


def segment_2d(
    img2d: np.ndarray, model, organelle: str, segmentation: dict | None = None
) -> np.ndarray:
    """Segment one 2D projection with Cellpose; returns a uint32 label mask.

    Thresholds / min_size / batch_size / per-organelle diameter come from the
    ``segmentation`` config block, falling back to the batch script's defaults so
    the online segmentation matches training when unset.
    """
    from shrimpy.fov_selection import segment_cellpose as sc

    seg = segmentation or {}
    kwargs = {
        "flow_threshold": seg.get("flow_threshold", getattr(sc, "FLOW_THRESHOLD", 0.4)),
        "cellprob_threshold": seg.get(
            "cellprob_threshold", getattr(sc, "CELLPROB_THRESHOLD", 0.0)
        ),
        "batch_size": seg.get("batch_size", getattr(sc, "BATCH_SIZE", 64)),
        "min_size": seg.get("min_size", getattr(sc, "MIN_SIZE", 15)),
        "normalize": True,
    }
    diam = _diameter_for(organelle, seg)
    if diam is not None:
        kwargs["diameter"] = diam
    # eval returns (masks, flows, styles); masks is a list of 2D masks (one per
    # input image). We pass a single image, so take masks[0].
    masks = model.eval([np.asarray(img2d, np.float32).copy()], **kwargs)[0]
    return np.asarray(masks[0], np.uint32)


# Aggregate features derivable from the label mask alone (object count + total
# coverage), i.e. WITHOUT regionprops shape/intensity props or cKDTree spacing.
# When the trained model only needs these, the expensive per-object extraction is
# skipped. Values are computed identically to group_features (see _cheap_features).
CHEAP_FEATURE_KEYS = frozenset({"object_count", "objects_per_10um2", "coverage_frac"})


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

    Numerically identical to ``group_features`` for the cheap keys. An empty mask
    yields genuine zeros (``object_count=0``, ``coverage_frac=0``,
    ``objects_per_10um2=0``): "no objects" is a real measurement, not missing
    data, so it is reported faithfully for the model to act on -- NOT dropped to
    NaN, which the median imputer would then fill with a typical FOV's value and
    make an empty FOV look good.
    """
    m = np.asarray(mask)
    h, w = m.shape
    n = int((np.unique(m) != 0).sum())
    out: dict[str, float] = {}
    if "object_count" in keys:
        out["object_count"] = n
    if "objects_per_10um2" in keys:
        out["objects_per_10um2"] = (
            10.0 * n / (w * h * px_um * px_um) if px_um else float("nan")
        )
    if "coverage_frac" in keys:
        out["coverage_frac"] = float(np.count_nonzero(m) / (w * h))
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
            rows = object_feature_rows(
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
                # No objects segmented: faithfully report the density features as a
                # real zero (object_count=0, coverage_frac=0, objects_per_10um2=0) so
                # the model can act on an empty FOV, instead of dropping the organelle
                # entirely -> every column NaN -> median-imputed to a typical FOV ->
                # empty FOV misclassified good. Shape/spatial features are genuinely
                # undefined with no objects, so they stay absent -> NaN -> imputed.
                logger.warning(
                    "FOV selection: no %s objects segmented; density features -> 0, "
                    "shape/spatial features -> NaN",
                    organelle,
                )
                cheap = keys_needed if keys_needed is not None else CHEAP_FEATURE_KEYS
                agg = _cheap_features(masks[organelle], px_um, set(cheap) & CHEAP_FEATURE_KEYS)
            else:
                agg = group_features(pd.DataFrame(rows))

        for k, v in agg.items():
            if keys_needed is None or k in keys_needed:
                feat[f"{prefix}__{k}"] = v
    return pd.DataFrame([feat])


def load_fov_model(path: str):
    """Load the trained FOV-goodness model dict ``{imputer, tree, features}``."""
    import joblib

    logger.info("FOV selection: loading model %s", path)
    return joblib.load(path)


def _to_numpy(x) -> np.ndarray:
    """Detach a torch tensor (or pass through an array) to a numpy array."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def decide_fov(
    preprocessor,
    cellpose,
    model: dict,
    bf_zyx: np.ndarray,
    *,
    target_channels: list[str],
    projection: str = "sum",
    px_um: float,
    threshold: float = 0.5,
    segmentation: dict | None = None,
    return_artifacts: bool = False,
    return_stacks: bool = False,
    label: str = "",
) -> tuple[float, bool] | tuple[float, bool, dict]:
    """Run one FOV's good/bad decision end to end.

    Reconstructs the input z-stack (``preprocessor``), projects and segments each
    ``target_channels`` channel, builds the variant-prefixed feature matrix, and
    predicts with the trained tree. Shared by the streaming worker and tests so
    the online decision matches training exactly.

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
    cellpose : CellposeModel
        Loaded segmentation model (see :func:`load_cellpose_model`).
    model : dict
        Trained FOV-goodness model ``{imputer, tree, features}``.
    bf_zyx : np.ndarray
        Raw input-channel z-stack ``(Z, Y, X)``.
    target_channels : list[str]
        Reconstructed channels to segment/feature (e.g. ``['nuclei', 'membrane']``).
    projection : str
        ``'sum'`` (trained default) or ``'max'``.
    px_um : float
        XY pixel size in microns (physical feature units).
    threshold : float
        P(good) cutoff.
    segmentation : dict | None
        Segmentation config block (model, thresholds, per-organelle diameters).
    label : str
        FOV/position name, prefixed to per-step logs so preprocessing and
        segmentation success/failure is attributable to a specific FOV.

    Returns
    -------
    tuple[float, bool]
        ``(proba_good, is_good)`` for this FOV.
    """
    pfx = f"[{label}] " if label else ""
    bf_zyx = np.asarray(bf_zyx)
    # The per-step 3D intermediates (deskew / phase volumes) are only needed for
    # the reconstruction store, which is expensive to build and write -- so pull
    # them only when return_stacks is set. return_artifacts alone (projections /
    # masks / features for the lightweight PNG/CSV debug) stays cheap.
    channels = preprocessor(
        bf_zyx, label=label, return_intermediates=return_stacks
    )  # {'nuclei', 'membrane', 'phase', ('deskew')}

    # Per-step 3D stacks for the reconstruction store (deskew, phase, and each VS
    # target volume); only populated when return_stacks is set.
    stacks: dict[str, np.ndarray] = {}
    if return_stacks:
        for key in ("deskew", "phase"):
            if key in channels:
                stacks[key] = _to_numpy(channels[key])

    projections: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for organelle in target_channels:
        vol = _to_numpy(channels[organelle])
        if return_stacks:
            stacks[organelle] = vol
        proj = project_zyx(vol, projection)
        projections[organelle] = proj
        try:
            mask = segment_2d(proj, cellpose, organelle, segmentation)
        except Exception as exc:
            logger.error("%ssegment %s FAILED: %s", pfx, organelle, exc)
            raise
        masks[organelle] = mask
        n_objects = int((np.unique(mask) != 0).sum())
        logger.info("%ssegment %s ok (%d objects)", pfx, organelle, n_objects)

    matrix = fov_feature_matrix(
        projections, masks, px_um, projection, source="vs", needed=model.get("features")
    )
    proba, good = predict_good(model, matrix, threshold)
    if return_artifacts:
        artifacts = {
            "stacks": stacks,
            "projections": projections,
            "masks": masks,
            "features": matrix,
        }
        return float(proba[0]), bool(good[0]), artifacts
    return float(proba[0]), bool(good[0])


def predict_good(model: dict, matrix_df, threshold: float = 0.5):
    """Predict good/bad from a feature matrix using the trained tree.

    Returns ``(proba, good)`` where ``proba`` is P(good) per row and ``good`` is
    a list of bools. Missing feature columns are filled with NaN and imputed by
    the model's median imputer (same as the offline predictor).
    """
    features = model["features"]
    x = matrix_df.reindex(columns=features)
    x_imputed = model["imputer"].transform(x)
    proba = model["tree"].predict_proba(x_imputed)[:, 1]
    good = [bool(p >= threshold) for p in proba]
    return proba, good
