"""Online per-FOV FOV-selection pipeline glue.

Runs one FOV's decision in memory, reusing the offline feature code so online
features match training exactly. Feature extraction is decoupled from the model:

    preprocessing output channel(s)
      -> project (sum / max / middle / logstd) [project_zyx]
      -> segment (cellpose / instanseg / otsu) [segment_2d]  (cpdino params from
                                        segment_cellpose)
      -> per-object features          [object_feature_rows, from extract_fov_features]
      -> per-FOV aggregation          [group_features, from build_fov_feature_matrix]
      -> named feature table          [extract_features: plain names for one channel,
                                        <organelle>_vs_<projection>__ prefixed for many]
      -> verdict                      [FovModel.predict, fov_model.py]

The model (thresholding / desirability / trained tree / ...) reads the feature table
by NAME only -- it never sees which channel produced a feature -- so any model type
pairs with any preprocessing (see :mod:`shrimpy.fov_selection.fov_model`).

Heavy imports (cellpose, torch, pandas) are done lazily inside functions so importing
this module stays cheap.
"""

from __future__ import annotations

import logging

from pathlib import Path

import numpy as np

from shrimpy.fov_selection.build_fov_feature_matrix import (
    MASK_FEATURE_KEYS,
    group_features,
    mask_gap_features,
    mask_occupancy_entropy,
)
from shrimpy.fov_selection.extract_fov_features import object_feature_rows

logger = logging.getLogger(__name__)


def project_zyx(zyx: np.ndarray, method: str = "sum") -> np.ndarray:
    """Reduce a ``(Z, Y, X)`` volume to a 2D ``(Y, X)`` float32 image.

    ``method``:
      ``'sum'``    -- sum over Z (trained model default, ``*_sum__*`` features);
      ``'max'``    -- max over Z;
      ``'middle'`` -- the single middle Z slice (``zyx[Z // 2]``);
      ``'logstd'`` -- per-pixel standard deviation over Z, shifted to non-negative and
                      ``log1p``-compressed (``log1p(std - std.min())``). Highlights axial
                      texture / contrast, e.g. for label-free brightfield where a flat sum or
                      middle slice carries little signal.

    All are channel-agnostic: they operate on any z-stack, so a projection is usable on a
    deskewed brightfield stack, a phase volume, a virtual-staining channel, etc. -- selected
    via the projection preprocessing step.
    """
    zyx = np.asarray(zyx)
    if method == "middle":
        return np.asarray(zyx[zyx.shape[0] // 2], np.float32)
    if method == "logstd":
        std = zyx.astype(np.float32).std(axis=0)
        return np.log1p(std - std.min()).astype(np.float32)
    reduce = np.max if method == "max" else np.sum
    return reduce(zyx, axis=0).astype(np.float32)


def load_segmenter(segmentation: dict | None = None):
    """Load the segmentation backend once (reused across FOVs); the switchable entry point.

    ``segmentation.model`` selects the backend:
      ``'cellpose'``  -- a Cellpose model (``model_name``, e.g. ``'cpdino'``; ``gpu``), loaded
                         here and passed to :func:`segment_2d`.
      ``'instanseg'`` -- an InstanSeg TorchScript checkpoint (``path``; ``gpu``), loaded here
                         and passed to :func:`segment_2d`. See :func:`_load_instanseg`.
      ``'otsu'``      -- stateless Otsu thresholding (returns ``None``; :func:`segment_2d`
                         thresholds the projection directly, no model / GPU needed).
    """
    seg = segmentation or {}
    backend = seg.get("model", "cellpose")
    if backend == "otsu":
        return None  # stateless; see _segment_otsu
    if backend == "instanseg":
        return _load_instanseg(seg)
    if backend != "cellpose":
        raise NotImplementedError(
            f"segmentation.model={backend!r} is not supported; use 'cellpose', "
            "'instanseg' or 'otsu'."
        )
    from cellpose import models

    from shrimpy.fov_selection import segment_cellpose as sc

    name = seg.get("model_name") or getattr(sc, "MODEL_NAME", "cpsam")
    gpu = seg.get("gpu", True)
    logger.info("FOV selection: loading Cellpose model %r (gpu=%s)", name, gpu)
    return models.CellposeModel(gpu=gpu, pretrained_model=name)


# --- InstanSeg ---------------------------------------------------------------------------
# InstanSeg ships as a TorchScript module, so the backend needs only torch -- not the
# `instanseg` package. Two checkpoint layouts are accepted:
#   * a bioimage.io .zip export (what the model zoo hands you): `instanseg.pt` plus an
#     `rdf.yaml` whose input axis `scale` records the pixel size the model was trained at.
#   * a bare `.pt` TorchScript file, in which case the training pixel size is unknown and
#     must be given as `segmentation.model_pixel_size_um` for rescaling to happen.

# The model's two output heads, in channel order (rdf.yaml outputs[0] channel_names).
INSTANSEG_TARGETS = ("nuclei", "cells")

# Forward kwargs the TorchScript module accepts, with the type TorchScript demands. Passing a
# Python int where the schema says float (or vice versa) is a hard error, so values from YAML
# are coerced here rather than trusted.
INSTANSEG_FORWARD_ARGS = {
    "min_size": int,
    "mask_threshold": float,
    "peak_distance": int,
    "seed_threshold": float,
    "overlap_threshold": float,
    "mean_threshold": float,
    "fg_threshold": float,
    "window_size": int,
    "cleanup_fragments": bool,
    "resolve_cell_and_nucleus": bool,
    "tta": bool,
}

# rdf.yaml preprocessing for this model: per-image percentile scaling over the spatial axes.
INSTANSEG_PERCENTILES = (0.1, 99.9)
INSTANSEG_EPS = 1e-6

# The model requires at least this many pixels per spatial axis (rdf.yaml inputs[0] axes
# size.min); rescaling to the model pixel size must not shrink a FOV below it.
INSTANSEG_MIN_SIZE_PX = 32


class _InstansegModel:
    """A loaded InstanSeg TorchScript module plus the metadata segmentation needs.

    ``pixel_size_um`` is the resolution the network was trained at (from ``rdf.yaml``, or
    ``segmentation.model_pixel_size_um``). :func:`_segment_instanseg` resamples each FOV to
    it before inference -- InstanSeg has no ``diameter`` knob like Cellpose, so matching the
    resolution IS how object scale is communicated. ``None`` disables rescaling.
    """

    def __init__(self, module, pixel_size_um: float | None, device: str) -> None:
        self.module = module
        self.pixel_size_um = pixel_size_um
        self.device = device


def _load_instanseg(seg: dict) -> _InstansegModel:
    """Load an InstanSeg TorchScript checkpoint from a bioimage.io ``.zip`` or a ``.pt``.

    Config keys (under ``fov_selection.segmentation``):
      ``path``                 -- REQUIRED; the ``.zip`` package or bare ``.pt``.
      ``gpu``                  -- default True; falls back to CPU with a warning if no CUDA.
      ``model_pixel_size_um``  -- overrides / supplies the training pixel size. Required for a
                                  bare ``.pt`` if you want rescaling; read from ``rdf.yaml``
                                  automatically for a ``.zip``.
    """
    import torch

    path_str = seg.get("path")
    if not path_str:
        raise ValueError(
            "segmentation.model='instanseg' requires a 'path' to the InstanSeg "
            "checkpoint (a bioimage.io .zip export or a TorchScript .pt)."
        )
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"InstanSeg checkpoint not found: {path}")

    rdf_pixel_size: float | None = None
    if path.suffix.lower() == ".zip":
        import io
        import zipfile

        import yaml

        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            weights = next((n for n in names if n.endswith(".pt")), None)
            if weights is None:
                raise ValueError(f"No TorchScript .pt inside the InstanSeg package {path}")
            # Load straight from memory -- no temp-file extraction of a ~150 MB blob.
            buffer = io.BytesIO(zf.read(weights))
            if "rdf.yaml" in names:
                rdf_pixel_size = _instanseg_rdf_pixel_size(yaml.safe_load(zf.read("rdf.yaml")))
        source = buffer
    else:
        source = str(path)

    gpu = seg.get("gpu", True)
    device = "cuda" if (gpu and torch.cuda.is_available()) else "cpu"
    if gpu and device == "cpu":
        logger.warning("FOV selection: InstanSeg requested gpu=True but CUDA is unavailable")

    module = torch.jit.load(source, map_location=device)
    module.eval()

    pixel_size = seg.get("model_pixel_size_um", rdf_pixel_size)
    pixel_size = float(pixel_size) if pixel_size else None
    logger.info(
        "FOV selection: loaded InstanSeg %s (device=%s, target=%r, model pixel size=%s um)",
        path.name,
        device,
        seg.get("target", INSTANSEG_TARGETS[0]),
        pixel_size if pixel_size else "unknown -- no rescaling",
    )
    if pixel_size is None:
        logger.warning(
            "FOV selection: InstanSeg training pixel size is unknown (no rdf.yaml and no "
            "segmentation.model_pixel_size_um), so the FOV is fed at its native resolution. "
            "If the acquisition pixel size differs from the model's, objects will be the "
            "wrong apparent size and the segmentation will be poor."
        )
    return _InstansegModel(module, pixel_size, device)


def _instanseg_rdf_pixel_size(rdf: dict) -> float | None:
    """Pixel size (um/px) the model was trained at, from a bioimage.io ``rdf.yaml``.

    Reads the ``scale`` of the first spatial input axis, requiring its ``unit`` to be
    micrometer so a model described in other units is not silently misread as um.
    """
    try:
        axes = rdf["inputs"][0]["axes"]
    except (KeyError, IndexError, TypeError):
        return None
    for axis in axes:
        if not isinstance(axis, dict) or axis.get("id") not in ("y", "x"):
            continue
        scale, unit = axis.get("scale"), axis.get("unit")
        if scale and unit in ("micrometer", "um", "µm"):
            return float(scale)
        if scale:
            logger.warning(
                "FOV selection: InstanSeg rdf.yaml axis %r has scale %s in unit %r (not "
                "micrometer); ignoring it for rescaling",
                axis.get("id"),
                scale,
                unit,
            )
            return None
    return None


def _segment_instanseg(
    img2d: np.ndarray,
    model: _InstansegModel,
    segmentation: dict | None = None,
    px_um: float | None = None,
) -> np.ndarray:
    """Segment one 2D projection with InstanSeg into a uint32 label mask.

    Steps, mirroring the model's documented bioimage.io pipeline:

    1. Percentile-scale the projection to ~[0, 1] (``scale_range``, 0.1-99.9 by default,
       overridable via ``segmentation.percentiles``). InstanSeg is trained on scaled input,
       and our projections (log-std, phase, raw brightfield) have wildly different native
       ranges, so this is what makes one checkpoint work across them.
    2. Resample to the model's training pixel size (see :class:`_InstansegModel`), bilinear.
       Skipped when either pixel size is unknown.
    3. Run the network, selecting a single output head via ``target_segmentation`` so only
       the requested one is decoded (``segmentation.target``: ``'nuclei'`` or ``'cells'``).
    4. Resample the label map back to the original pixel grid with NEAREST interpolation, so
       label ids survive and downstream features stay in the acquisition's pixel units
       (``coverage_frac``, ``max_gap_um`` are computed against the original ``px_um``).

    A single channel is fed as-is: the architecture is channel-invariant, so replicating the
    projection to the checkpoint's nominal 3 channels is unnecessary.
    """
    import torch

    seg = segmentation or {}
    target = seg.get("target", INSTANSEG_TARGETS[0])
    if target not in INSTANSEG_TARGETS:
        raise ValueError(
            f"segmentation.target must be one of {list(INSTANSEG_TARGETS)}; got {target!r}"
        )

    img = np.asarray(img2d, np.float32)
    orig_h, orig_w = img.shape

    lo_pct, hi_pct = seg.get("percentiles", INSTANSEG_PERCENTILES)
    lo, hi = np.percentile(img, (float(lo_pct), float(hi_pct)))
    # np.percentile returns float64, which would promote the whole image and then hit the
    # TorchScript module as a DoubleTensor against float weights -- cast back explicitly.
    img = ((img - lo) / (hi - lo + INSTANSEG_EPS)).astype(np.float32)

    x = torch.from_numpy(img)[None, None].to(model.device)

    # Rescale to the model's resolution; this is InstanSeg's equivalent of Cellpose's
    # `diameter`. Clamped so a small FOV never drops under the network's minimum input.
    scale = (px_um / model.pixel_size_um) if (px_um and model.pixel_size_um) else 1.0
    if not np.isclose(scale, 1.0):
        new_h = max(int(round(orig_h * scale)), INSTANSEG_MIN_SIZE_PX)
        new_w = max(int(round(orig_w * scale)), INSTANSEG_MIN_SIZE_PX)
        x = torch.nn.functional.interpolate(
            x, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        logger.debug(
            "InstanSeg: rescaled %s -> %s (%.4f -> %.4f um/px)",
            (orig_h, orig_w),
            (new_h, new_w),
            px_um,
            model.pixel_size_um,
        )

    selector = torch.tensor(
        [1 if t == target else 0 for t in INSTANSEG_TARGETS], device=model.device
    )
    kwargs = {
        name: cast(seg[name])
        for name, cast in INSTANSEG_FORWARD_ARGS.items()
        if seg.get(name) is not None
    }
    with torch.no_grad():
        out = model.module(x, target_segmentation=selector, **kwargs)

    # (1, 1, H, W) -- one head selected above. Nearest-neighbour back to the original grid
    # so label ids are preserved (any smooth interpolation would invent nonexistent ids).
    if out.shape[-2:] != (orig_h, orig_w):
        out = torch.nn.functional.interpolate(out, size=(orig_h, orig_w), mode="nearest")
    return np.asarray(out[0, 0].cpu().numpy(), np.uint32)


# Backward-compatible alias (older callers / scripts imported this name).
load_cellpose_model = load_segmenter


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


def _segment_otsu(img2d: np.ndarray, segmentation: dict | None = None) -> np.ndarray:
    """Otsu-threshold the projection into a cleaned connected-component label mask.

    Threshold by Otsu, optionally morphologically close (``close_radius`` px disk, to bridge
    gaps between the pieces of a cluster), drop connected components below ``min_size`` px,
    fill interior holes (``fill_holes``, default True), then label. Suited to a log-std
    brightfield projection where the foreground is a texture blob rather than crisp objects. A
    flat / empty projection yields an all-zero mask (no foreground). Params from the
    ``segmentation`` config block."""
    from scipy.ndimage import binary_fill_holes
    from skimage.filters import threshold_otsu
    from skimage.measure import label as cc_label
    from skimage.morphology import binary_closing, disk

    seg = segmentation or {}
    img = np.asarray(img2d, np.float32)
    finite = img[np.isfinite(img)]
    if finite.size == 0 or float(finite.min()) == float(finite.max()):
        return np.zeros(img.shape, np.uint32)  # flat image -> no Otsu split -> no foreground
    fg = img > threshold_otsu(img)
    close_radius = int(seg.get("close_radius", 0))
    if close_radius > 0:
        fg = binary_closing(fg, disk(close_radius))
    min_size = int(seg.get("min_size", 0))
    if min_size > 0:
        lab = cc_label(fg)
        sizes = np.bincount(lab.ravel())
        keep = sizes >= min_size
        keep[0] = False  # background
        fg = keep[lab]
    if seg.get("fill_holes", True):
        fg = binary_fill_holes(fg)
    return np.asarray(cc_label(fg), np.uint32)


def segment_2d(
    img2d: np.ndarray,
    model,
    organelle: str,
    segmentation: dict | None = None,
    px_um: float | None = None,
) -> np.ndarray:
    """Segment one 2D projection into a uint32 label mask, per ``segmentation.model``.

    ``'cellpose'`` (default) runs the loaded Cellpose ``model`` with thresholds / min_size /
    batch_size / per-organelle diameter from the ``segmentation`` config block (falling back
    to the batch script's defaults). ``'instanseg'`` runs the loaded TorchScript module (see
    :func:`_segment_instanseg`), which needs ``px_um`` to resample the FOV to the model's
    training resolution. ``'otsu'`` ignores ``model`` and thresholds the projection directly
    (see :func:`_segment_otsu`).
    """
    seg = segmentation or {}
    backend = seg.get("model", "cellpose")
    if backend == "otsu":
        return _segment_otsu(img2d, seg)
    if backend == "instanseg":
        return _segment_instanseg(img2d, model, seg, px_um)

    from shrimpy.fov_selection import segment_cellpose as sc

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
        out["mask_occupancy_entropy"] = mask_occupancy_entropy(m)
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
                agg = group_features(pd.DataFrame(rows))
                if keys_needed is None or (keys_needed & MASK_FEATURE_KEYS):
                    agg.update(mask_gap_features(masks[organelle], px_um))

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
        rows = object_feature_rows(
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
            agg = group_features(pd.DataFrame(rows))
            if keys_needed is None or (keys_needed & MASK_FEATURE_KEYS):
                agg.update(mask_gap_features(mask, px_um))

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
    segmentation: dict | None = None,
    return_artifacts: bool = False,
    return_stacks: bool = False,
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
    segmenter : object | None
        Loaded segmentation backend from :func:`load_segmenter` -- a CellposeModel, or
        ``None`` for the (stateless) Otsu backend.
    model : FovModel
        Any :class:`~shrimpy.fov_selection.fov_model.FovModel` (thresholding, desirability,
        trained tree, ...); consumes the extracted features by name.
    bf_zyx : np.ndarray
        Raw input-channel z-stack ``(Z, Y, X)``.
    target_channels : list[str]
        Reconstructed channels to segment/feature (e.g. ``['nuclei', 'membrane']`` or a
        single ``['brightfield']``).
    projection : str
        ``'sum'`` (trained default), ``'max'``, ``'middle'`` (middle-slice), or ``'logstd'``
        (log-normalized per-pixel std over Z).
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
            mask = segment_2d(proj, segmenter, organelle, segmentation, px_um=px_um)
        except Exception as exc:
            logger.error("%ssegment %s FAILED: %s", pfx, organelle, exc)
            raise
        masks[organelle] = mask
        n_objects = int((np.unique(mask) != 0).sum())
        logger.info("%ssegment %s ok (%d objects)", pfx, organelle, n_objects)

    matrix = extract_features(
        projections, masks, px_um, projection, needed=model.feature_names
    )
    proba, good = model.predict(matrix, threshold)
    if return_artifacts:
        artifacts = {
            "stacks": stacks,
            "projections": projections,
            "masks": masks,
            "features": matrix,
        }
        return float(proba[0]), bool(good[0]), artifacts
    return float(proba[0]), bool(good[0])
