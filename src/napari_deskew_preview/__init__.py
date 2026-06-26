"""Quick on-demand deskew preview of oblique-plane light-sheet data in napari.

The source-agnostic core (:class:`DeskewProjector`, :class:`DeskewedArray`,
:func:`deskewed_layer`, :func:`array_gather`) and the :class:`LazyPlaneArray` base are
numpy-only and importable without napari; the napari dock widget lives in
:mod:`napari_deskew_preview._widget`.
"""

from __future__ import annotations

from napari_deskew_preview._lazy_array import LazyPlaneArray
from napari_deskew_preview.deskew import (
    KEEP_OVERHANG,
    LS_ANGLE_DEG,
    PIXEL_SIZE_UM,
    DeskewedArray,
    DeskewProjector,
    array_gather,
    deskewed_layer,
)

__all__ = [
    "DeskewProjector",
    "DeskewedArray",
    "LazyPlaneArray",
    "deskewed_layer",
    "array_gather",
    "LS_ANGLE_DEG",
    "PIXEL_SIZE_UM",
    "KEEP_OVERHANG",
]

__version__ = "0.1.0"
