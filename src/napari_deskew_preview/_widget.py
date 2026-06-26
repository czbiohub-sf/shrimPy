"""napari dock widget: quick deskew preview of a selected oblique-plane Image layer.

Treats the selected layer's last three axes as ``(Z_scan, Y_tilt, X_cover)`` and any
leading axes (T, C, position, ...) as a batch. Adds a lazy deskewed layer that computes one
plane at a time, so even volumes larger than RAM are viewable. Editing angle / pixel size /
scan step rebuilds the deskewed layer(s).
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
    """Deskew the selected image layer and keep it in sync with the geometry fields."""

    def __init__(self, napari_viewer: object) -> None:
        super().__init__()
        self._viewer = napari_viewer
        # deskewed-layer-name -> raw array-like it was built from (for rebuilds).
        self._outputs: dict[str, object] = {}

        layout = QVBoxLayout(self)
        self._angle = self._spin(LS_ANGLE_DEG, 0.0, 89.9, 1.0, 2)
        self._pixel = self._spin(PIXEL_SIZE_UM, 1e-4, 100.0, 0.001, 4)
        self._scan = self._spin(DEFAULT_SCAN_STEP_UM, 1e-4, 1000.0, 0.01, 4)
        form = QFormLayout()
        form.addRow("Angle (°)", self._angle)
        form.addRow("Pixel size (µm)", self._pixel)
        form.addRow("Scan step (µm)", self._scan)
        layout.addLayout(form)

        button = QPushButton("Deskew selected layer")
        button.clicked.connect(self._deskew_selected)
        layout.addWidget(button)

        self._status = QLabel("Select an image layer, set the scan step, then deskew.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        layout.addStretch()

        # Editing geometry rebuilds any deskewed layers already created.
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
        raw = self._raw_of(layer)
        if getattr(raw, "ndim", 0) < 3:
            self._status.setText("Need a layer with at least 3 dimensions (…, Z, Y, X).")
            return
        name = f"{layer.name} [deskewed]"
        try:
            self._add_or_update(name, raw, source=layer)
        except Exception:  # noqa: BLE001 - report, don't crash napari
            logger.exception("Deskew failed")
            self._status.setText("Deskew failed (see console).")
            return
        self._status.setText(f"Deskewed → '{name}'")

    def _add_or_update(self, name: str, raw: object, source: object | None = None) -> None:
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
        if name in self._viewer.layers:
            self._viewer.layers[name].data = data
        else:
            kwargs: dict = {}
            if source is not None:
                kwargs = {
                    "colormap": source.colormap,
                    "blending": source.blending,
                    "contrast_limits": source.contrast_limits,
                }
            self._viewer.add_image(data, name=name, **kwargs)
        self._outputs[name] = raw
        logger.info("Deskew '%s' -> %s", name, projector.output_shape)

    def _on_geometry_changed(self, *_: object) -> None:
        for name in list(self._outputs):
            if name not in self._viewer.layers:
                self._outputs.pop(name, None)
                continue
            try:
                self._add_or_update(name, self._outputs[name])
            except Exception:  # noqa: BLE001
                logger.debug("Rebuild of '%s' failed (ignored)", name, exc_info=True)
