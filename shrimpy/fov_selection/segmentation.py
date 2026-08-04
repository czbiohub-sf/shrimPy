"""All FOV-selection segmentation models, behind one small class hierarchy.

A :class:`Segmenter` turns a 2D projection into a uint32 instance-label mask. Three backends:

    'cellpose'   -> :class:`CellposeSegmenter`   (a Cellpose model, e.g. cpdino)
    'instanseg'  -> :class:`InstansegSegmenter`  (an InstanSeg TorchScript checkpoint)
    'otsu'       -> :class:`OtsuSegmenter`        (stateless Otsu thresholding; no model/GPU)

:func:`build_segmenter` reads the ``fov_selection.segmentation`` config block and returns the
right one, loaded and ready; call :meth:`Segmenter.segment` per FOV. The segmenter carries its
own config, so the caller never re-passes it.

General pipeline code: NO hardcoded paths. Heavy imports (cellpose, torch) are lazy so
importing this module stays cheap. The offline batch driver that segments whole zarr stores
lives in ``shrimpy/scripts/segment_cpdino_batch.py``.
"""

from __future__ import annotations

import logging

from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# The InstanSeg model's two output heads, in channel order (rdf.yaml outputs[0] channel_names).
INSTANSEG_TARGETS = ("nuclei", "cells")


class Segmenter:
    """Interface: segment one 2D projection into a uint32 instance-label mask.

    Subclasses are constructed from the ``fov_selection.segmentation`` config block (which
    they retain), so :meth:`segment` needs only the image, its organelle, and the pixel size.
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}

    def segment(
        self, img2d: np.ndarray, organelle: str, px_um: float | None = None
    ) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class OtsuSegmenter(Segmenter):
    """Otsu-threshold the projection into a cleaned connected-component label mask.

    Threshold by Otsu, optionally morphologically close (``close_radius`` px disk), drop
    connected components below ``min_size`` px, fill interior holes (``fill_holes``, default
    True), then label. Suited to a log-std brightfield projection where the foreground is a
    texture blob. A flat / empty projection yields an all-zero mask. Params from the config.
    """

    def segment(
        self, img2d: np.ndarray, organelle: str = "", px_um: float | None = None
    ) -> np.ndarray:
        from scipy.ndimage import binary_fill_holes
        from skimage.filters import threshold_otsu
        from skimage.measure import label as cc_label
        from skimage.morphology import binary_closing, disk

        seg = self._config
        img = np.asarray(img2d, np.float32)
        finite = img[np.isfinite(img)]
        if finite.size == 0 or float(finite.min()) == float(finite.max()):
            return np.zeros(
                img.shape, np.uint32
            )  # flat image -> no Otsu split -> no foreground
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


class CellposeSegmenter(Segmenter):
    """A loaded Cellpose model (e.g. cpdino) run one image at a time.

    The Cellpose diameter is per organelle: an explicit ``segmentation.diameters[organelle]``
    wins, else membrane channels get :attr:`MEMBRANE_DIAMETER` (whole-cell badly under-covers
    on auto-scale) and everything else (nuclei) auto-scales (:attr:`NUCLEI_DIAMETER` = None).
    """

    MODEL_NAME = "cpdino"  # Cellpose-DINO (fastest; quality ties cyto3/cpsam)
    FLOW_THRESHOLD = 0.4  # mask shape QC
    CELLPROB_THRESHOLD = 0.0  # lower=more/larger masks, higher=stricter
    BATCH_SIZE = 64  # cellpose tiles per forward pass
    MIN_SIZE = 15  # drop masks smaller than this many px
    NUCLEI_DIAMETER: float | None = None  # None = auto-scale (correct for nuclei)
    MEMBRANE_DIAMETER: float | None = 120.0  # whole-cell needs an explicit diameter
    MEMBRANE_HINT = "membrane"  # channels containing this use MEMBRANE_DIAMETER

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        from cellpose import models

        name = self._config.get("model_name") or self.MODEL_NAME
        gpu = self._config.get("gpu", True)
        logger.info("FOV selection: loading Cellpose model %r (gpu=%s)", name, gpu)
        self.model = models.CellposeModel(gpu=gpu, pretrained_model=name)

    def _diameter_for(self, organelle: str) -> float | None:
        """Per-organelle Cellpose diameter (microns); ``None`` means auto-scale."""
        diameters = self._config.get("diameters") or {}
        if organelle in diameters:
            return diameters[organelle]
        if self.MEMBRANE_HINT in (organelle or "").lower():
            return self.MEMBRANE_DIAMETER
        return self.NUCLEI_DIAMETER

    def segment(
        self, img2d: np.ndarray, organelle: str, px_um: float | None = None
    ) -> np.ndarray:
        seg = self._config
        kwargs = {
            "flow_threshold": seg.get("flow_threshold", self.FLOW_THRESHOLD),
            "cellprob_threshold": seg.get("cellprob_threshold", self.CELLPROB_THRESHOLD),
            "batch_size": seg.get("batch_size", self.BATCH_SIZE),
            "min_size": seg.get("min_size", self.MIN_SIZE),
            "normalize": True,
        }
        diam = self._diameter_for(organelle)
        if diam is not None:
            kwargs["diameter"] = diam
        # eval returns (masks, flows, styles); masks is a list (one per input image). We pass
        # a single image, so take masks[0].
        masks = self.model.eval([np.asarray(img2d, np.float32).copy()], **kwargs)[0]
        return np.asarray(masks[0], np.uint32)


class InstansegSegmenter(Segmenter):
    """A loaded InstanSeg TorchScript module plus the metadata segmentation needs.

    InstanSeg ships as a TorchScript module, so the backend needs only torch -- not the
    ``instanseg`` package. Two checkpoint layouts are accepted:
      * a bioimage.io ``.zip`` export: ``instanseg.pt`` plus an ``rdf.yaml`` whose input axis
        ``scale`` records the training pixel size.
      * a bare ``.pt`` TorchScript file, in which case the training pixel size is unknown and
        must be given as ``segmentation.model_pixel_size_um`` for rescaling to happen.

    :attr:`pixel_size_um` is the resolution the network was trained at; :meth:`segment`
    resamples each FOV to it before inference (InstanSeg has no ``diameter`` knob -- matching
    resolution IS how object scale is communicated). ``None`` disables rescaling.
    """

    # Forward kwargs the TorchScript module accepts, with the type TorchScript demands.
    FORWARD_ARGS = {
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
    # rdf.yaml preprocessing: per-image percentile scaling over the spatial axes.
    PERCENTILES = (0.1, 99.9)
    EPS = 1e-6
    # The model requires at least this many pixels per spatial axis (rdf.yaml inputs[0] axes
    # size.min); rescaling to the model pixel size must not shrink a FOV below it.
    MIN_SIZE_PX = 32

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        import torch

        seg = self._config
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
                buffer = io.BytesIO(zf.read(weights))
                if "rdf.yaml" in names:
                    rdf_pixel_size = self._rdf_pixel_size(yaml.safe_load(zf.read("rdf.yaml")))
            source = buffer
        else:
            source = str(path)

        gpu = seg.get("gpu", True)
        self.device = "cuda" if (gpu and torch.cuda.is_available()) else "cpu"
        if gpu and self.device == "cpu":
            logger.warning(
                "FOV selection: InstanSeg requested gpu=True but CUDA is unavailable"
            )

        self.module = torch.jit.load(source, map_location=self.device)
        self.module.eval()

        pixel_size = seg.get("model_pixel_size_um", rdf_pixel_size)
        self.pixel_size_um = float(pixel_size) if pixel_size else None
        logger.info(
            "FOV selection: loaded InstanSeg %s (device=%s, target=%r, model pixel size=%s um)",
            path.name,
            self.device,
            seg.get("target", INSTANSEG_TARGETS[0]),
            self.pixel_size_um if self.pixel_size_um else "unknown -- no rescaling",
        )
        if self.pixel_size_um is None:
            logger.warning(
                "FOV selection: InstanSeg training pixel size is unknown (no rdf.yaml and no "
                "segmentation.model_pixel_size_um), so the FOV is fed at its native "
                "resolution. If the acquisition pixel size differs from the model's, objects "
                "will be the wrong apparent size and the segmentation will be poor."
            )

    @staticmethod
    def _rdf_pixel_size(rdf: dict) -> float | None:
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

    def segment(
        self, img2d: np.ndarray, organelle: str = "", px_um: float | None = None
    ) -> np.ndarray:
        """Segment one 2D projection with InstanSeg into a uint32 label mask.

        Percentile-scale the projection to ~[0, 1], resample to the model's training pixel
        size (InstanSeg's equivalent of Cellpose's ``diameter``; skipped when unknown), run the
        network selecting a single output head via ``segmentation.target``, then resample the
        label map back to the original grid with NEAREST interpolation so label ids survive.
        """
        import torch

        seg = self._config
        target = seg.get("target", INSTANSEG_TARGETS[0])
        if target not in INSTANSEG_TARGETS:
            raise ValueError(
                f"segmentation.target must be one of {list(INSTANSEG_TARGETS)}; got {target!r}"
            )

        img = np.asarray(img2d, np.float32)
        orig_h, orig_w = img.shape

        lo_pct, hi_pct = seg.get("percentiles", self.PERCENTILES)
        lo, hi = np.percentile(img, (float(lo_pct), float(hi_pct)))
        # np.percentile returns float64, which would promote the whole image and hit the
        # TorchScript module as a DoubleTensor against float weights -- cast back explicitly.
        img = ((img - lo) / (hi - lo + self.EPS)).astype(np.float32)

        x = torch.from_numpy(img)[None, None].to(self.device)

        scale = (px_um / self.pixel_size_um) if (px_um and self.pixel_size_um) else 1.0
        if not np.isclose(scale, 1.0):
            new_h = max(int(round(orig_h * scale)), self.MIN_SIZE_PX)
            new_w = max(int(round(orig_w * scale)), self.MIN_SIZE_PX)
            x = torch.nn.functional.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
            logger.debug(
                "InstanSeg: rescaled %s -> %s (%.4f -> %.4f um/px)",
                (orig_h, orig_w),
                (new_h, new_w),
                px_um,
                self.pixel_size_um,
            )

        selector = torch.tensor(
            [1 if t == target else 0 for t in INSTANSEG_TARGETS], device=self.device
        )
        kwargs = {
            name: cast(seg[name])
            for name, cast in self.FORWARD_ARGS.items()
            if seg.get(name) is not None
        }
        with torch.no_grad():
            out = self.module(x, target_segmentation=selector, **kwargs)

        # (1, 1, H, W) -- one head selected above. Nearest-neighbour back to the original grid
        # so label ids are preserved (any smooth interpolation would invent nonexistent ids).
        if out.shape[-2:] != (orig_h, orig_w):
            out = torch.nn.functional.interpolate(out, size=(orig_h, orig_w), mode="nearest")
        return np.asarray(out[0, 0].cpu().numpy(), np.uint32)


def build_segmenter(segmentation: dict | None = None) -> Segmenter:
    """Construct the :class:`Segmenter` for the ``fov_selection.segmentation`` config block.

    ``segmentation.model`` selects the backend: ``'cellpose'`` (default), ``'instanseg'``, or
    ``'otsu'``. The returned segmenter is loaded and ready; call :meth:`Segmenter.segment`.
    """
    seg = segmentation or {}
    backend = seg.get("model", "cellpose")
    if backend == "otsu":
        return OtsuSegmenter(seg)
    if backend == "instanseg":
        return InstansegSegmenter(seg)
    if backend == "cellpose":
        return CellposeSegmenter(seg)
    raise NotImplementedError(
        f"segmentation.model={backend!r} is not supported; use 'cellpose', "
        "'instanseg' or 'otsu'."
    )
