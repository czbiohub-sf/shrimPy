"""napari dock widget: quick deskew preview of oblique-plane light-sheet Image layers.

"Display deskewed" / "Display raw" apply to **all** image layers and recenter the view
afterwards (camera + Z slider), since switching shape would otherwise leave the image
off-center. Each layer's last three axes are treated as ``(Z_scan, Y_tilt, X_cover)`` and any
leading axes (T, C, position, ...) as a batch.

The layer data is replaced **in place** with a lazy deskewed view that computes one plane at
a time, so even volumes larger than RAM are viewable. Replacing in place (rather than adding
a second layer) keeps a single, self-consistent set of dimension sliders. The UI is the
shared :class:`~napari_deskew_preview._controls.DeskewControls`.
"""

from __future__ import annotations

import logging

from qtpy.QtWidgets import QVBoxLayout, QWidget

from napari_deskew_preview._controls import DeskewControls
from napari_deskew_preview.deskew import array_gather, deskewed_layer

logger = logging.getLogger(__name__)


class DeskewWidget(QWidget):
    """Deskew all image layers in place, driven by the shared :class:`DeskewControls`."""

    def __init__(self, napari_viewer: object) -> None:
        super().__init__()
        self._viewer = napari_viewer
        # Layers we have deskewed -> their original raw array-like (for rebuild / restore).
        self._sources: dict[object, object] = {}

        self._controls = DeskewControls()
        layout = QVBoxLayout(self)
        layout.addWidget(self._controls)
        self._controls.displayDeskewedRequested.connect(self._display_deskewed)
        self._controls.displayRawRequested.connect(self._display_raw)
        self._controls.geometryChanged.connect(self._on_geometry_changed)

    @staticmethod
    def _raw_of(layer: object) -> object:
        # Multiscale layers expose a list of arrays; deskew the full-resolution level.
        data = layer.data
        return data[0] if isinstance(data, list) else data

    def _display_deskewed(self) -> None:
        """Deskew every image layer in place, then recenter the view."""
        done = skipped = 0
        for layer in list(self._viewer.layers):
            if not hasattr(layer, "data"):
                continue
            # Reuse the stored raw if we already deskewed this layer (idempotent),
            # otherwise capture its current data as the raw source.
            raw = self._sources.get(layer)
            if raw is None:
                raw = self._raw_of(layer)
                if getattr(raw, "ndim", 0) < 3:
                    skipped += 1
                    continue
                self._sources[layer] = raw
            try:
                self._apply(layer, raw)
                done += 1
            except Exception:  # noqa: BLE001 - report, don't crash napari
                logger.exception("Deskew failed for '%s'", getattr(layer, "name", "?"))
                self._sources.pop(layer, None)
                skipped += 1
        self._recenter()
        logger.info("Displaying deskewed: %d layer(s), skipped %d", done, skipped)

    def _display_raw(self) -> None:
        """Restore every layer we deskewed back to its raw data, then recenter."""
        done = 0
        for layer in list(self._viewer.layers):
            raw = self._sources.pop(layer, None)
            if raw is None:
                continue
            layer.data = raw
            done += 1
        self._recenter()
        logger.info("Displaying raw: %d layer(s)", done)

    def _recenter(self) -> None:
        """Recenter after a switch: reset the camera (like Home) and center the Z slider.

        Switching raw<->deskew changes the depth-axis length (e.g. scan 1068 vs deskewed
        depth 256), so the Z slider would otherwise stay put / clamp to an edge.
        """
        try:
            self._viewer.reset_view()
        except Exception:  # noqa: BLE001 - never break on a viewer without reset_view
            logger.debug("reset_view failed (ignored)", exc_info=True)
        try:
            dims = self._viewer.dims
            if dims.ndim >= 3:
                axis = dims.ndim - 3  # the Z (depth) slider, third from last
                step = list(dims.current_step)
                step[axis] = int(dims.nsteps[axis]) // 2
                dims.current_step = tuple(step)
        except Exception:  # noqa: BLE001
            logger.debug("Centering Z slider failed (ignored)", exc_info=True)

    def _apply(self, layer: object, raw: object) -> tuple[int, ...]:
        """Replace ``layer``'s data in place with the deskewed view of ``raw``."""
        batch_sizes = tuple(int(s) for s in raw.shape[:-3])
        raw_zyx = tuple(int(s) for s in raw.shape[-3:])
        data, _ = deskewed_layer(
            array_gather(raw),
            raw_zyx,
            self._controls.scan_step,
            batch_sizes,
            ls_angle_deg=self._controls.angle,
            pixel_size_um=self._controls.pixel_size,
        )
        layer.data = data  # in place: a single layer, sliders match the deskewed shape
        logger.info("Deskew '%s' -> %s", layer.name, data.shape)
        return tuple(data.shape)

    def _on_geometry_changed(self) -> None:
        for layer in list(self._sources):
            if layer not in self._viewer.layers:
                self._sources.pop(layer, None)
                continue
            try:
                self._apply(layer, self._sources[layer])
            except Exception:  # noqa: BLE001
                logger.debug("Rebuild of '%s' failed (ignored)", layer.name, exc_info=True)
