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


def load_cellpose_model(model_name: str | None = None, gpu: bool = True):
    """Load a Cellpose model once (reuse across FOVs). Defaults to the batch
    script's ``MODEL_NAME`` (e.g. ``'cpdino'``)."""
    from cellpose import models

    from shrimpy.fov_selection import segment_cellpose as sc

    name = model_name or getattr(sc, "MODEL_NAME", "cpsam")
    logger.info("FOV selection: loading Cellpose model %r (gpu=%s)", name, gpu)
    return models.CellposeModel(gpu=gpu, pretrained_model=name)


def _diameter_for(organelle: str) -> float | None:
    """Membrane channels use an explicit diameter; nuclei auto-scale (None)."""
    from shrimpy.fov_selection import segment_cellpose as sc

    hint = getattr(sc, "MEMBRANE_HINT", "membrane")
    if hint in organelle.lower():
        return getattr(sc, "MEMBRANE_DIAMETER", 120.0)
    return getattr(sc, "NUCLEI_DIAMETER", None)


def segment_2d(img2d: np.ndarray, model, organelle: str) -> np.ndarray:
    """Segment one 2D projection with Cellpose; returns a uint32 label mask.

    Uses the same thresholds / min_size / diameter policy as the batch script.
    """
    from shrimpy.fov_selection import segment_cellpose as sc

    kwargs = {
        "flow_threshold": getattr(sc, "FLOW_THRESHOLD", 0.4),
        "cellprob_threshold": getattr(sc, "CELLPROB_THRESHOLD", 0.0),
        "batch_size": getattr(sc, "BATCH_SIZE", 64),
        "min_size": getattr(sc, "MIN_SIZE", 15),
        "normalize": True,
    }
    diam = _diameter_for(organelle)
    if diam is not None:
        kwargs["diameter"] = diam
    # eval returns (masks, flows, styles); masks is a list of 2D masks (one per
    # input image). We pass a single image, so take masks[0].
    masks = model.eval([np.asarray(img2d, np.float32).copy()], **kwargs)[0]
    return np.asarray(masks[0], np.uint32)


def fov_feature_matrix(
    projections: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    px_um: float,
    projection: str = "sum",
    source: str = "vs",
):
    """Build a 1-row feature matrix (variant-prefixed columns) for one FOV.

    ``projections`` / ``masks`` map organelle name (``'nuclei'``, ``'membrane'``)
    to its 2D projection / label mask. Column names follow the training
    convention ``<organelle>_<source>_<projection>__<feature>`` (e.g.
    ``nuclei_vs_sum__coverage_frac``), computed via the shared
    ``object_feature_rows`` + ``group_features``.
    """
    import pandas as pd

    feat: dict[str, float] = {}
    for organelle, proj in projections.items():
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
            logger.warning("FOV selection: no %s objects segmented; features -> NaN", organelle)
            continue
        agg = group_features(pd.DataFrame(rows))
        prefix = f"{organelle}_{source}_{projection}"
        for k, v in agg.items():
            feat[f"{prefix}__{k}"] = v
    return pd.DataFrame([feat])


def load_fov_model(path: str):
    """Load the trained FOV-goodness model dict ``{imputer, tree, features}``."""
    import joblib

    logger.info("FOV selection: loading model %s", path)
    return joblib.load(path)


def predict_good(model: dict, matrix_df, threshold: float = 0.5):
    """Predict good/bad from a feature matrix using the trained tree.

    Returns ``(proba, good)`` where ``proba`` is P(good) per row and ``good`` is
    a list of bools. Missing feature columns are filled with NaN and imputed by
    the model's median imputer (same as the offline predictor).
    """
    features = model["features"]
    X = matrix_df.reindex(columns=features)
    Xi = model["imputer"].transform(X)
    proba = model["tree"].predict_proba(Xi)[:, 1]
    good = [bool(p >= threshold) for p in proba]
    return proba, good
