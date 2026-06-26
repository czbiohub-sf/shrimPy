"""napari dock widget: quick deskew preview of a selected oblique-plane Image layer.

Treats the selected layer's last three axes as ``(Z_scan, Y_tilt, X_cover)`` and any
leading axes (T, C, position, ...) as a batch. Replaces the layer's data **in place** with a
lazy deskewed view that computes one plane at a time, so even volumes larger than RAM are
viewable. Replacing in place (rather than adding a second layer) keeps a single,
self-consistent set of dimension sliders -- a raw and a deskewed layer have different axis
sizes (e.g. scan 1068 vs deskewed depth 256), and napari would otherwise union them into
oversized sliders. Use "Restore raw" to put the original data back; editing the geometry
fields rebuilds any layers deskewed by this widget.
"""

from __future__ import annotations

import logging

from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napari_deskew_preview.deskew import (
    LS_ANGLE_DEG,
    PIXEL_SIZE_UM,
    array_gather,
    deskewed_layer,
)

logger = logging.getLogger(__name__)

# Default scan step (um); the user sets this for their acquisition.
DEFAULT_SCAN_STEP_UM = 0.3


class DeskewWidget(QWidget):
    """Deskew the selected image layer in place and keep it in sync with the fields."""

    def __init__(self, napari_viewer: object) -> None:
        super().__init__()
        self._viewer = napari_viewer
        # Layers we have deskewed -> their original raw array-like (for rebuild / restore).
        self._sources: dict[object, object] = {}

        layout = QVBoxLayout(self)
        self._angle = self._spin(LS_ANGLE_DEG, 0.0, 89.9, 1.0, 2)
        self._pixel = self._spin(PIXEL_SIZE_UM, 1e-4, 100.0, 0.001, 4)
        self._scan = self._spin(DEFAULT_SCAN_STEP_UM, 1e-4, 1000.0, 0.01, 4)
        form = QFormLayout()
        form.addRow("Angle (°)", self._angle)
        form.addRow("Pixel size (µm)", self._pixel)
        form.addRow("Scan step (µm)", self._scan)
        layout.addLayout(form)

        deskew_button = QPushButton("Deskew selected layer")
        deskew_button.clicked.connect(self._deskew_selected)
        layout.addWidget(deskew_button)

        restore_button = QPushButton("Restore raw")
        restore_button.clicked.connect(self._restore_selected)
        layout.addWidget(restore_button)

        self._status = QLabel("Select an image layer, set the scan step, then deskew.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        layout.addStretch()

        # Editing geometry rebuilds any layers already deskewed by this widget.
        for spin in (self._angle, self._pixel, self._scan):
            spin.valueChanged.connect(self._on_geometry_changed)

    @staticmethod
    def _spin(
        value: float, lo: float, hi: float, step: float, decimals: int
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setKeyboardTracking(False)  # emit only on enter / focus-out / arrows
        spin.setValue(value)
        return spin

    @staticmethod
    def _raw_of(layer: object) -> object:
        # Multiscale layers expose a list of arrays; deskew the full-resolution level.
        data = layer.data
        return data[0] if isinstance(data, list) else data

    def _deskew_selected(self) -> None:
        layer = self._viewer.layers.selection.active
        if layer is None or not hasattr(layer, "data"):
            self._status.setText("Select an image layer first.")
            return
        # If we already deskewed this layer, rebuild from the stored raw (idempotent);
        # otherwise capture its current data as the raw source.
        raw = self._sources.get(layer)
        if raw is None:
            raw = self._raw_of(layer)
            if getattr(raw, "ndim", 0) < 3:
                self._status.setText("Need a layer with at least 3 dimensions (…, Z, Y, X).")
                return
            self._sources[layer] = raw
        try:
            shape = self._apply(layer, raw)
        except Exception:  # noqa: BLE001 - report, don't crash napari
            logger.exception("Deskew failed")
            self._sources.pop(layer, None)
            self._status.setText("Deskew failed (see console).")
            return
        self._status.setText(f"Deskewed '{layer.name}' in place → {shape}")

    def _restore_selected(self) -> None:
        layer = self._viewer.layers.selection.active
        raw = self._sources.pop(layer, None) if layer is not None else None
        if raw is None:
            self._status.setText("Selected layer was not deskewed by this widget.")
            return
        layer.data = raw
        self._status.setText(f"Restored raw data for '{layer.name}'.")

    def _apply(self, layer: object, raw: object) -> tuple[int, ...]:
        """Replace ``layer``'s data in place with the deskewed view of ``raw``."""
        batch_sizes = tuple(int(s) for s in raw.shape[:-3])
        raw_zyx = tuple(int(s) for s in raw.shape[-3:])
        data, projector = deskewed_layer(
            array_gather(raw),
            raw_zyx,
            self._scan.value(),
            batch_sizes,
            ls_angle_deg=self._angle.value(),
            pixel_size_um=self._pixel.value(),
        )
        layer.data = data  # in place: a single layer, sliders match the deskewed shape
        logger.info("Deskew '%s' -> %s", layer.name, data.shape)
        return tuple(data.shape)

    def _on_geometry_changed(self, *_: object) -> None:
        for layer in list(self._sources):
            if layer not in self._viewer.layers:
                self._sources.pop(layer, None)
                continue
            try:
                self._apply(layer, self._sources[layer])
            except Exception:  # noqa: BLE001
                logger.debug("Rebuild of '%s' failed (ignored)", layer.name, exc_info=True)
