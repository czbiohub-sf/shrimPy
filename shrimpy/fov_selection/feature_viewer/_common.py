"""Shared constants, theme, and helper widgets for the FOV feature viewer.

Split out of ``app.py`` so the per-tab modules (label_tab, rank_tab, score_map_tab)
and the main window can all import them without a circular dependency.
"""

from __future__ import annotations

import os

from pathlib import Path

import numpy as np

from qtpy import QtCore, QtGui, QtWidgets

from shrimpy.fov_selection import fov_model as FM

DEFAULT_DIR = os.environ.get(
    "FOV_VIEWER_DIR",
    "/hpc/projects/comp.micro/microscope_dev/smart_fov_selection/fov_selection_output",
)
# Where named ranking profiles (per biological question) are saved/loaded.
PROFILE_DIR = Path(
    os.environ.get("FOV_RANK_PROFILE_DIR", str(Path(DEFAULT_DIR) / "ranking_profiles"))
)
REDUCE_PREFIX = {"PCA": "PCA", "t-SNE": "TSNE", "UMAP": "UMAP"}
MAX_THUMBS = 200  # cap thumbnails rendered at once (refine the selection for more)
# Rank-tab knob-table columns. The two param columns are shape-dependent: each holds a
# prefix-labeled spin for whatever interpretable parameter the row's shape/direction needs
# (center/fwhm, center/fold, midpoint/width), matching the config schema. Two is the maximum
# any shape uses (every current shape has exactly two interpretable params).
RCOL_FEATURE, RCOL_DIR, RCOL_SHAPE, RCOL_P1, RCOL_P2, RCOL_WEIGHT = range(6)
RCOL_PARAMS = (RCOL_P1, RCOL_P2)
# Short label shown as each parameter spin's prefix, per interpretable param key.
RANK_PARAM_LABELS = {
    "center": "center",
    "fwhm": "fwhm",
    "fold": "fold",
    "midpoint": "midpoint",
    "width": "width",
}


def _internal_to_feature(shape, direction, lo, hi, curve_k, weight):
    """Internal (lo, hi, ...) bounds -> a config feature dict with the interpretable params.
    The math is :func:`fov_model.curve_params`; this only packages the result into the config
    schema (center/fwhm, center/fold, or midpoint/width) and drops params the shape ignores."""
    p = FM.curve_params(shape, lo, hi, curve_k)
    feat = {"shape": shape}
    if shape == "gaussian":
        feat.update(center=p["center"], fwhm=p["fwhm"])
    elif shape == "lognormal":
        feat.update(center=p["center"], fold=p["fold"])
    else:  # sigmoid monotonic (direction higher|lower)
        feat.update(midpoint=p["midpoint"], width=p["width"], direction=direction)
    feat["weight"] = weight
    return feat


def _feature_to_internal(feat):
    """A config feature dict -> internal ``(shape, direction, lo, hi, curve_k)``. The current
    interpretable schema is converted via :func:`fov_model.curve_bounds`; legacy ``range``-style
    profiles are read as internal bounds directly so old files still open."""
    shape = feat.get("shape", "gaussian")
    if shape == "linear":
        raise ValueError("the 'linear' shape was removed; use gaussian, lognormal, or sigmoid")
    dir_params = None  # (direction, params) when the current interpretable schema is present
    if shape == "gaussian" and {"center", "fwhm"} <= feat.keys():
        dir_params = ("target", {"center": float(feat["center"]), "fwhm": float(feat["fwhm"])})
    elif shape == "lognormal" and {"center", "fold"} <= feat.keys():
        dir_params = ("target", {"center": float(feat["center"]), "fold": float(feat["fold"])})
    elif shape == "sigmoid" and "midpoint" in feat:
        dir_params = (
            feat.get("direction", "higher"),
            {"midpoint": float(feat["midpoint"]), "width": float(feat["width"])},
        )
    if dir_params is not None:
        direction, params = dir_params
        lo, hi, ck = FM.curve_bounds(shape, params)
        return shape, direction, lo, hi, ck
    # Legacy fallback: `range` (or lo/hi) read straight as the internal bounds (e.g. an old
    # gaussian range = +-1 sigma, or a sigmoid range + curve_k).
    rng = feat.get("range", [feat.get("lo"), feat.get("hi")])
    if rng[0] is None or rng[1] is None:
        raise ValueError(
            f"feature spec {feat!r} lacks the params for a {shape!r} curve "
            "(gaussian: center/fwhm; lognormal: center/fold; sigmoid: midpoint/width/direction)"
        )
    lo, hi = float(rng[0]), float(rng[1])
    direction = feat.get("direction", "target") if shape == "sigmoid" else "target"
    curve_k = float(feat.get("curve_k", 0.0)) if shape == "sigmoid" else 0.0
    return shape, direction, lo, hi, curve_k


DETAIL_SKIP = {"__src", "__png", "png", "__dataset"}
THUMB_CACHE_MAX = 1000  # LRU cap on decoded thumbnails (path,size) -> QPixmap

# highlight color for the clicked FOV: blue border on the thumbnail (Analysis + Rank
# tabs) and the scatter focus ring / Rank histogram value line. Shared so both tabs match.
FOCUS_COLOR = "#2979ff"
RANK_CLICK_COLOR = FOCUS_COLOR  # Rank tab: clicked-FOV border + its value line in the hists
RANK_TITLE_COLOR = "#ffd700"  # gold: the feature-name label on each Rank-tab histogram
THUMB_QSS = "background:#111; color:#666; border:1px solid #333;"
THUMB_FOCUS_QSS = f"background:#111; color:#666; border:3px solid {FOCUS_COLOR};"

# ---- dark-grey theme (lifted off black) ----
MPL_BG = "#3a3a3a"  # scatter/figure background (medium-dark grey)
MPL_FG = "#ececec"  # text / ticks
MPL_GRID = "#5a5a5a"
ACCENT = "#8c8c8c"  # neutral grey accent

# `goodness` is a categorical label, not a continuous feature: color it by class
# (including NaN -> "unlabeled") with fixed, semantic colors.
GOODNESS_CATEGORIES = {1.0: "good (1)", 0.0: "neutral (0)", -1.0: "bad (-1)"}
GOODNESS_COLORS = {
    "good (1)": "#4caf50",  # green
    "neutral (0)": "#ffd24d",  # yellow
    "bad (-1)": "#e53935",  # red
    "unlabeled": "#9e9e9e",  # grey
}
# Label tab columns, in fixed reading order. `value` is the on-disk goodness code
# (None == unlabeled / NaN). A column is shown only if the loaded data has ≥1 FOV in it.
GOODNESS_LABEL_ORDER = [
    ("Good", 1.0, "#4caf50"),
    ("Neutral", 0.0, "#ffd24d"),
    ("Bad", -1.0, "#e53935"),
    ("Nan", None, "#9e9e9e"),
]


def goodness_color(v) -> str:
    """Semantic color for a `goodness` value: good=green, neutral=yellow, bad=red,
    unlabeled (NaN/None)=grey. Used for scatter points and FOV thumbnail borders."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return GOODNESS_COLORS["unlabeled"]
    return GOODNESS_COLORS.get(
        GOODNESS_CATEGORIES.get(float(v), ""), GOODNESS_COLORS["unlabeled"]
    )


def _border_qss(color: str) -> str:
    """Thumbnail stylesheet with a 3px border of the given color."""
    return f"background:#111; color:#666; border:3px solid {color};"


def goodness_border_qss(v) -> str:
    """Thumbnail stylesheet with a 3px border colored by the FOV's goodness label."""
    return _border_qss(goodness_color(v))


DARK_QSS = """
QWidget { background:#3c3c3c; color:#ececec; font-size:12px; }
QScrollArea { border:none; }
QListWidget, QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {
  background:#4a4a4a; border:1px solid #5c5c5c; border-radius:6px; padding:3px 5px; }
/* disabled inputs (e.g. a shape's fixed direction) read as greyed-out and flat */
QComboBox:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled, QLineEdit:disabled {
  background:#3f3f3f; color:#777777; border-color:#4a4a4a; }
QComboBox:disabled::drop-down { border:none; }
QGroupBox { border:1px solid #5c5c5c; border-radius:8px; margin-top:14px; padding:8px; font-weight:600; }
QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; color:#cfcfcf; }
QPushButton { background:#4d4d4d; border:1px solid #5f5f5f; border-radius:6px; padding:6px 10px; }
QPushButton:hover { background:#5a5a5a; }
QPushButton:pressed, QPushButton:checked { background:#8c8c8c; color:#161616; border-color:#8c8c8c; }
QListWidget::item:selected { background:#6f6f6f; color:#ffffff; }
QComboBox QAbstractItemView { background:#4a4a4a; selection-background-color:#6f6f6f; selection-color:#ffffff; }
QSlider::groove:horizontal { height:6px; background:#5c5c5c; border-radius:3px; }
QSlider::handle:horizontal { background:#b0b0b0; width:14px; margin:-5px 0; border-radius:7px; }
QScrollBar:vertical { background:#3c3c3c; width:12px; }
QScrollBar::handle:vertical { background:#5f5f5f; border-radius:6px; min-height:24px; }
QToolTip { background:#4a4a4a; color:#ececec; border:1px solid #5f5f5f; }
QLabel { background:transparent; }
"""


def apply_dark(app):
    """Apply the app-wide dark Fusion palette and stylesheet to the QApplication."""
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    c = QtGui.QColor
    pal.setColor(QtGui.QPalette.Window, c("#3c3c3c"))
    pal.setColor(QtGui.QPalette.WindowText, c("#ececec"))
    pal.setColor(QtGui.QPalette.Base, c("#4a4a4a"))
    pal.setColor(QtGui.QPalette.AlternateBase, c("#525252"))
    pal.setColor(QtGui.QPalette.Text, c("#ececec"))
    pal.setColor(QtGui.QPalette.Button, c("#4d4d4d"))
    pal.setColor(QtGui.QPalette.ButtonText, c("#ececec"))
    pal.setColor(QtGui.QPalette.Highlight, c("#6f6f6f"))
    pal.setColor(QtGui.QPalette.HighlightedText, c("#ffffff"))
    pal.setColor(QtGui.QPalette.ToolTipBase, c("#4a4a4a"))
    pal.setColor(QtGui.QPalette.ToolTipText, c("#ececec"))
    app.setPalette(pal)
    app.setStyleSheet(DARK_QSS)


class _DetailsModel(QtCore.QAbstractTableModel):
    """Virtualized model: renders only visible cells, so resetting is instant."""

    def __init__(self):
        """Initialize the empty table model (no dataframe, rows, or fields yet)."""
        super().__init__()
        self._df = None
        self._rows: list[int] = []
        self._fields: list[str] = []
        self._colpos: list[int] = []

    def set_data(self, df, rows, fields):
        """Point the model at a new dataframe/rows/columns and reset it in one step."""
        self.beginResetModel()
        self._df = df
        self._rows = rows
        self._fields = fields
        self._colpos = [df.columns.get_loc(f) for f in fields] if df is not None else []
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        """Number of selected FOV rows the model exposes (0 until data is set)."""
        return 0 if (parent.isValid() or self._df is None) else len(self._rows)

    def columnCount(self, parent=QtCore.QModelIndex()):
        """Number of displayed field columns (0 until data is set)."""
        return 0 if (parent.isValid() or self._df is None) else len(self._fields)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        """Cell text for display: floats formatted to 4 sig figs (NaN shown as 'NaN')."""
        if role != QtCore.Qt.DisplayRole or not index.isValid():
            return None
        v = self._df.iat[self._rows[index.row()], self._colpos[index.column()]]
        if isinstance(v, (float, np.floating)):
            return "NaN" if v != v else f"{v:.4g}"
        return str(v)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        """Horizontal header shows the field name; vertical headers are suppressed."""
        if role != QtCore.Qt.DisplayRole:
            return None
        return self._fields[section] if orientation == QtCore.Qt.Horizontal else None


class _ThumbLabel(QtWidgets.QLabel):
    """A FOV thumbnail that can be dragged between Label-tab panels. Carries its df
    position as drag payload; the panel it lands on decides the new goodness class."""

    def __init__(self, pos: int):
        """Create a thumbnail label carrying its df position `pos` as drag payload."""
        super().__init__()
        self._pos = pos
        self._press = None

    def mousePressEvent(self, e):
        """Record the left-button press point to detect the start of a drag."""
        if e.button() == QtCore.Qt.LeftButton:
            self._press = e.pos()

    def mouseMoveEvent(self, e):
        """Start a move drag carrying this FOV's df position once the drag threshold is passed."""
        if self._press is None or not (e.buttons() & QtCore.Qt.LeftButton):
            return
        if (
            e.pos() - self._press
        ).manhattanLength() < QtWidgets.QApplication.startDragDistance():
            return
        drag = QtGui.QDrag(self)
        mime = QtCore.QMimeData()
        mime.setText(str(self._pos))
        drag.setMimeData(mime)
        pm = self.pixmap()
        if pm is not None and not pm.isNull():
            drag.setPixmap(pm)
        self._press = None
        drag.exec_(QtCore.Qt.MoveAction)


class _DropPanel(QtWidgets.QWidget):
    """Grid host for one goodness class; accepts thumbnails dropped from any panel."""

    def __init__(self, viewer, value):
        """Create a drop target for goodness class `value`, back-referencing the viewer."""
        super().__init__()
        self._viewer = viewer
        self._value = value  # goodness code this panel represents (NaN for "Nan")
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        """Accept the drag if it carries text (a serialized FOV position)."""
        if e.mimeData().hasText():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        """Keep accepting the drag as it moves while it carries text."""
        if e.mimeData().hasText():
            e.acceptProposedAction()

    def dropEvent(self, e):
        """Relabel the dropped FOV to this panel's goodness class; caveat: silently ignores a non-integer payload."""
        try:
            pos = int(e.mimeData().text())
        except (TypeError, ValueError):
            return
        self._viewer._on_label_drop(pos, self._value)
        e.acceptProposedAction()


class _ParamCell(QtWidgets.QWidget):
    """A Rank-table parameter cell: the param NAME as a small label ABOVE the spin box
    (outside it) instead of an in-box prefix, so the numbers read clearly. Delegates
    value / setValue / blockSignals to the inner spin, so :meth:`RankTabMixin._read_rank_table`
    and :meth:`RankTabMixin._rank_sync_row` can treat the cell exactly like the spin."""

    def __init__(self, key, spin):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 0, 2, 0)
        lay.setSpacing(0)
        label = QtWidgets.QLabel(RANK_PARAM_LABELS.get(key, key))
        label.setStyleSheet("color:#bbbbbb; font-size:10px;")
        lay.addWidget(label)
        lay.addWidget(spin)
        self._spin = spin
        self._param_key = key

    def value(self):
        return self._spin.value()

    def setValue(self, v):
        self._spin.setValue(v)

    def blockSignals(self, b):
        self._spin.blockSignals(b)
        return super().blockSignals(b)
