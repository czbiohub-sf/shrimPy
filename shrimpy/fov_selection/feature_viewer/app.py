"""
FOV Feature Viewer: a Qt GUI to explore FOV-level features, label FOVs, and tune a
ranking model, with the selected FOVs shown as image thumbnails.

Tabs
  Analysis  Explore features. The left column holds the settings (feature axes,
            dimensionality reduction, color, and filters) above an interactive 2D/3D
            scatter (PCA/t-SNE/UMAP; click a point or lasso a region to select); the right
            column shows the selected FOVs as a scrollable PNG grid. Per-feature threshold
            sliders keep only the FOVs whose value falls in a chosen range.
  Label     The loaded FOVs grouped into one panel per goodness class (Good/Neutral/Bad
            and unlabeled). Drag a thumbnail to another panel to relabel that FOV; edits
            are written back to the CSV only when you press Save.
  Rank      Tune the DesirabilityModel. The left column shows each feature's value
            histogram with its desirability curve overlaid, plus a table of the
            range/direction/shoulder knobs and a Re-rank button; the right column lists the
            FOVs as thumbnails ordered best-first by the resulting score.

Data wiring
  Each *_fov_feature_matrix.csv is one row per FOV and must carry a `filename` column.
  data.load_matrices resolves each row to its PNG in the sibling <stem>[_<channel>]_png/
  folder next to the CSV (strict filename match) and stores it as __png / __png_<channel>.

Run:  python -m shrimpy.fov_selection.feature_viewer
"""

from __future__ import annotations

import os
import sys

from collections import OrderedDict

os.environ.setdefault("QT_API", "pyqt6")

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from qtpy import QtCore, QtGui, QtWidgets

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.transforms import IdentityTransform
from mpl_toolkits.mplot3d import proj3d

from shrimpy.fov_selection import fov_model as FM
from shrimpy.scripts import ranking as RANK

from . import data as D

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
RANK_PAGE = 200  # Rank tab: FOV tiles materialized per page (infinite scroll)
# Rank-tab knob-table columns. The four param columns are shape-dependent: each holds a
# prefix-labeled spin for whatever interpretable parameter the row's shape/direction needs
# (center/fwhm, midpoint/width, onset/ideal, ...), matching the config schema.
RCOL_FEATURE, RCOL_DIR, RCOL_SHAPE, RCOL_P1, RCOL_P2, RCOL_P3, RCOL_P4, RCOL_WEIGHT = range(8)
RCOL_PARAMS = (RCOL_P1, RCOL_P2, RCOL_P3, RCOL_P4)
# Short label shown as each parameter spin's prefix, per interpretable param key.
RANK_PARAM_LABELS = {
    "center": "center",
    "fwhm": "fwhm",
    "fold": "fold",
    "midpoint": "midpoint",
    "width": "width",
    "half_band_lo": "band lo",
    "half_band_hi": "band hi",
    "range_lo": "range lo",
    "range_hi": "range hi",
    "soft_left": "shoulder L",
    "soft_right": "shoulder R",
    "onset": "onset",
    "ideal": "ideal",
}


def _legacy_soft(feat, rng):
    """Legacy shoulder widths from a config feature: `soft` scalar/[l, r], or soft_left/right,
    defaulting to the band width."""
    band = max(float(rng[1]) - float(rng[0]), 1e-9)
    soft = feat.get("soft")
    if isinstance(soft, (list, tuple)):
        return float(soft[0]), float(soft[1])
    base = float(soft) if soft else band
    return float(feat.get("soft_left", base)), float(feat.get("soft_right", base))


def _internal_to_feature(shape, direction, lo, hi, soft_left, soft_right, curve_k, weight):
    """Internal (lo, hi, ...) bounds -> a config feature dict with the interpretable params.
    The math is :func:`fov_model.curve_params`; this only packages the result into the config
    schema (arrays for range / half_band / soft) and drops params the shape ignores."""
    p = FM.curve_params(shape, direction, lo, hi, soft_left, soft_right, curve_k)
    feat = {"shape": shape}
    if shape == "gaussian":
        feat.update(center=p["center"], fwhm=p["fwhm"])
    elif shape == "lognormal":
        feat.update(center=p["center"], fold=p["fold"])
    elif shape == "sigmoid" and direction == "target":
        feat.update(half_band=[p["half_band_lo"], p["half_band_hi"]], width=p["width"])
    elif shape == "sigmoid":
        feat.update(midpoint=p["midpoint"], width=p["width"], direction=direction)
    elif direction == "target":  # linear target band
        feat.update(
            range=[p["range_lo"], p["range_hi"]], soft=[p["soft_left"], p["soft_right"]]
        )
    else:  # linear monotonic ramp
        feat.update(onset=p["onset"], ideal=p["ideal"])
    feat["weight"] = weight
    return feat


def _feature_to_internal(feat):
    """A config feature dict -> internal ``(shape, direction, lo, hi, soft_left, soft_right,
    curve_k)``. The current interpretable schema is converted via :func:`fov_model.curve_bounds`;
    legacy ``range``-style profiles are read as internal bounds directly so old files still open."""
    shape = feat.get("shape", "linear")
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
    elif shape == "sigmoid" and "half_band" in feat:
        hb = feat["half_band"]
        dir_params = (
            "target",
            {
                "half_band_lo": float(hb[0]),
                "half_band_hi": float(hb[1]),
                "width": float(feat["width"]),
            },
        )
    elif shape == "linear" and ("onset" in feat or "ideal" in feat):
        onset, ideal = float(feat["onset"]), float(feat["ideal"])
        dir_params = (
            "higher" if ideal > onset else "lower",
            {"onset": onset, "ideal": ideal},
        )
    elif shape == "linear" and "range" in feat and feat.get("direction", "target") == "target":
        rng = feat["range"]
        sl, sr = _legacy_soft(feat, rng)
        dir_params = (
            "target",
            {
                "range_lo": float(rng[0]),
                "range_hi": float(rng[1]),
                "soft_left": sl,
                "soft_right": sr,
            },
        )
    if dir_params is not None:
        direction, params = dir_params
        lo, hi, sl, sr, ck = FM.curve_bounds(shape, direction, params)
        return shape, direction, lo, hi, sl, sr, ck
    # Legacy fallback: `range` (or lo/hi) read straight as the internal bounds (e.g. an old
    # gaussian range = +-1 sigma, a sigmoid range + curve_k, or a linear range + direction).
    rng = feat.get("range", [feat.get("lo"), feat.get("hi")])
    lo, hi = float(rng[0]), float(rng[1])
    direction = feat.get("direction", "target") if shape in ("linear", "sigmoid") else "target"
    sl, sr = _legacy_soft(feat, rng)
    curve_k = float(feat.get("curve_k", 0.0)) if shape == "sigmoid" else 0.0
    return shape, direction, lo, hi, sl, sr, curve_k


DETAIL_SKIP = {"__src", "__png", "png", "__dataset"}
THUMB_CACHE_MAX = 1000  # LRU cap on decoded thumbnails (path,size) -> QPixmap
# highlight color for the clicked FOV: blue border on the thumbnail (Analysis + Rank
# tabs) and the scatter focus ring / Rank histogram value line. Shared so both tabs match.
FOCUS_COLOR = "#2979ff"
RANK_CLICK_COLOR = FOCUS_COLOR  # Rank tab: clicked-FOV border + its value line in the hists
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
    "unlabeled": "#5b8def",  # blue
}
# Label tab columns, in fixed reading order. `value` is the on-disk goodness code
# (None == unlabeled / NaN). A column is shown only if the loaded data has ≥1 FOV in it.
GOODNESS_LABEL_ORDER = [
    ("Good", 1.0, "#4caf50"),
    ("Neutral", 0.0, "#ffd24d"),
    ("Bad", -1.0, "#e53935"),
    ("Nan", None, "#5b8def"),
]


def goodness_color(v) -> str:
    """Semantic color for a `goodness` value: good=green, neutral=yellow, bad=red,
    unlabeled (NaN/None)=blue. Used for scatter points and FOV thumbnail borders."""
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


class FeatureViewer(QtWidgets.QMainWindow):
    def __init__(self):
        """Build the main window: initialize all state, then assemble the Analysis, Label, and Rank tabs."""
        super().__init__()
        self.setWindowTitle("FOV Feature Viewer")
        self.resize(1700, 1000)
        self.df: pd.DataFrame | None = None
        self.filt = np.array([], int)
        self.plot_pos = np.array([], int)
        self.sel_pos: list[int] = []
        self.reduced_cols: list[str] = []
        self.thresholds: dict[str, tuple] = {}  # feature column -> (lo, hi) kept range
        self.lasso_sel = None  # lasso filter: positions, or None = show all
        self.hidden = np.array([], bool)  # True = "removed" (invisible, not deleted)
        self.remove_history: list[list[int]] = []
        self.focus_pos = None  # df position of the highlighted FOV/point
        self._focus_marker = None  # scatter ring artist
        self._focus_label = None  # highlighted thumbnail
        self._picked = False  # was a scatter point hit on this click?
        # Rank tab: per-feature desirability knobs for the production DesirabilityModel.
        # feature -> {"direction","lo","hi","soft_left","soft_right","weight","enabled"}.
        self.rank_ranges: dict = {}
        self.rank_sort = False  # sort the Analysis FOV panel by `score` (legacy hook)
        self._channel = "brightfield"  # which FOV image channel the thumbnails display
        self._ready = False

        left_col = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left_col.addWidget(self._build_settings())
        left_col.addWidget(self._build_center())
        left_col.setStretchFactor(0, 0)
        left_col.setStretchFactor(1, 1)
        left_col.setSizes([450, 620])  # settings fully visible, scatter below

        main = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main.addWidget(left_col)
        main.addWidget(self._build_right())
        main.setStretchFactor(0, 0)  # settings+scatter keep their width; FOV grid takes extra
        main.setStretchFactor(1, 1)
        main.setSizes([820, 980])  # left wide enough to show all 3 settings columns

        # all the existing panels live under an "Analysis" tab; "Label" groups the
        # loaded FOVs into one column per goodness class.
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(main, "Analysis")
        self._label_tab_index = self.tabs.addTab(self._build_label_tab(), "Label")
        self._rank_tab_index = self.tabs.addTab(self._build_rank_tab(), "Rank")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)
        self._ready = True

    # ------------------------------------------------------------- settings
    @staticmethod
    def _button_row(*buttons):
        """HBox of natural-width buttons, left-aligned (not stretched)."""
        h = QtWidgets.QHBoxLayout()
        for b in buttons:
            h.addWidget(b)
        h.addStretch(1)
        return h

    @staticmethod
    def _make_plain_entry(spin):
        """Make a spin box behave like a plain typed entry: no arrows, no wheel."""
        spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        spin.setKeyboardTracking(False)
        spin.wheelEvent = lambda e: e.ignore()
        return spin

    def _build_settings(self):
        """Build the Analysis-tab settings panel (datasets, thresholds, scatter, reduction, removed-points, labeling) and return its widget."""
        w = QtWidgets.QWidget()
        cols = QtWidgets.QHBoxLayout(w)
        colA = QtWidgets.QVBoxLayout()
        colB = QtWidgets.QVBoxLayout()
        colC = QtWidgets.QVBoxLayout()
        # colA = sections 1-2, colB = sections 3-4, colC = the 5|6 row
        for cc, stretch in ((colA, 3), (colB, 4), (colC, 4)):
            cols.addLayout(cc, stretch)

        # --- 1. datasets (col A) ---
        g = QtWidgets.QGroupBox("1. Datasets")
        gv = QtWidgets.QVBoxLayout(g)
        load = QtWidgets.QPushButton("Load CSV(s)…")
        load.clicked.connect(self.on_load)
        rmds = QtWidgets.QPushButton("Remove")
        rmds.clicked.connect(self.on_remove_dataset)
        clr = QtWidgets.QPushButton("Clear all")
        clr.clicked.connect(self.on_clear)
        gv.addLayout(self._button_row(load, rmds, clr))
        self.ds_list = QtWidgets.QListWidget()
        self.ds_list.setMaximumHeight(80)
        gv.addWidget(self.ds_list)
        colA.addWidget(g)

        # --- 2. feature thresholds (col A) ---
        g = QtWidgets.QGroupBox("2. Feature thresholds  (keep in range)")
        gv = QtWidgets.QVBoxLayout(g)
        self.thr_col = QtWidgets.QComboBox()
        self.thr_col.currentTextChanged.connect(self._on_thr_col)
        gv.addWidget(self.thr_col)
        hr = QtWidgets.QHBoxLayout()
        self.thr_lo = QtWidgets.QDoubleSpinBox()
        self.thr_hi = QtWidgets.QDoubleSpinBox()
        for s in (self.thr_lo, self.thr_hi):
            s.setDecimals(4)
            s.setRange(-1e12, 1e12)
            self._make_plain_entry(s)
        hr.addWidget(QtWidgets.QLabel("min"))
        hr.addWidget(self.thr_lo)
        hr.addWidget(QtWidgets.QLabel("max"))
        hr.addWidget(self.thr_hi)
        gv.addLayout(hr)
        adt = QtWidgets.QPushButton("Add / update")
        adt.clicked.connect(self.on_add_threshold)
        rmt = QtWidgets.QPushButton("Remove")
        rmt.clicked.connect(self.on_remove_threshold)
        clt = QtWidgets.QPushButton("Clear")
        clt.clicked.connect(self.on_clear_thresholds)
        gv.addLayout(self._button_row(adt, rmt, clt))
        self.thr_list = QtWidgets.QListWidget()
        self.thr_list.setMaximumHeight(90)
        gv.addWidget(self.thr_list)
        # visible-FOV count after thresholds / lasso (shared by apply_filters)
        self.filter_summary = QtWidgets.QLabel("no filter")
        self.filter_summary.setWordWrap(True)
        gv.addWidget(self.filter_summary)
        colA.addWidget(g)
        colA.addStretch(1)

        # --- 3. scatter (col B) ---
        g = QtWidgets.QGroupBox("3. Scatter plot")
        form = QtWidgets.QFormLayout(g)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["2D", "3D"])
        self.cb_x = QtWidgets.QComboBox()
        self.cb_y = QtWidgets.QComboBox()
        self.cb_z = QtWidgets.QComboBox()
        self.cb_color = QtWidgets.QComboBox()
        self.cb_cmap = QtWidgets.QComboBox()
        self.cb_cmap.addItems(["viridis", "plasma", "coolwarm", "turbo", "tab10", "tab20"])
        for cb in (self.mode, self.cb_x, self.cb_y, self.cb_z, self.cb_color, self.cb_cmap):
            cb.currentIndexChanged.connect(self.update_plot)
        self.mode.currentTextChanged.connect(
            lambda *_: self.cb_z.setEnabled(self.mode.currentText() == "3D")
        )
        form.addRow("Mode", self.mode)
        form.addRow("X", self.cb_x)
        form.addRow("Y", self.cb_y)
        form.addRow("Z", self.cb_z)
        form.addRow("Color by", self.cb_color)
        form.addRow("Colormap", self.cb_cmap)
        colB.addWidget(g)

        # --- 4. dim reduction (col B) ---
        g = QtWidgets.QGroupBox("4. Dim. reduction (3 comp.)")
        form = QtWidgets.QFormLayout(g)
        self.cb_method = QtWidgets.QComboBox()
        self.cb_method.addItems(D.METHODS)
        self.sp_perp = QtWidgets.QDoubleSpinBox()
        self.sp_perp.setRange(2, 200)
        self.sp_perp.setValue(30)
        self._make_plain_entry(self.sp_perp)
        self.sp_nn = QtWidgets.QSpinBox()
        self.sp_nn.setRange(2, 200)
        self.sp_nn.setValue(15)
        self._make_plain_entry(self.sp_nn)
        run = QtWidgets.QPushButton("Run on filtered FOVs")
        run.clicked.connect(self.run_reduction)
        form.addRow("Method", self.cb_method)
        self.lbl_perp = QtWidgets.QLabel("perplexity")  # t-SNE only
        self.lbl_nn = QtWidgets.QLabel("n_neighbors")  # UMAP only
        form.addRow(self.lbl_perp, self.sp_perp)
        form.addRow(self.lbl_nn, self.sp_nn)
        form.addRow(self._button_row(run))
        self.cb_method.currentTextChanged.connect(self._update_reduction_params)
        self._update_reduction_params()
        self.reduce_status = QtWidgets.QLabel("not run")
        self.reduce_status.setWordWrap(True)
        form.addRow(self.reduce_status)
        if not D.HAS_UMAP:
            self.reduce_status.setText(
                "UMAP unavailable (pip install umap-learn). PCA / t-SNE ready."
            )
        colB.addWidget(g)
        colB.addStretch(1)

        # --- 5 + 6 sit side by side in one narrower row (col C) ---
        # --- 5. removed-points history ---
        g6 = QtWidgets.QGroupBox("5. Removed points")
        gv6 = QtWidgets.QVBoxLayout(g6)
        self.history_list = QtWidgets.QListWidget()
        self.history_list.setMaximumHeight(120)
        gv6.addWidget(self.history_list)
        rs = QtWidgets.QPushButton("Restore selected")
        rs.clicked.connect(self.on_restore_selected)
        ra = QtWidgets.QPushButton("Restore all")
        ra.clicked.connect(self.on_restore_all)
        gv6.addLayout(self._button_row(rs, ra))
        self.hidden_count = QtWidgets.QLabel("0 points hidden")
        gv6.addWidget(self.hidden_count)

        # --- 6. label selected FOVs (goodness) ---
        g7 = QtWidgets.QGroupBox("6. Label FOVs  (goodness)")
        gv7 = QtWidgets.QVBoxLayout(g7)
        self.label_value = QtWidgets.QComboBox()
        self.label_value.addItem("Good (1)", 1)
        self.label_value.addItem("Neutral (0)", 0)
        self.label_value.addItem("Bad (-1)", -1)
        self.label_value.addItem("Unlabeled (NaN)", None)  # reset label to nothing
        gv7.addWidget(self.label_value)
        setb = QtWidgets.QPushButton("Set")
        setb.clicked.connect(self.on_set_goodness)
        gv7.addLayout(self._button_row(setb))
        self.label_status = QtWidgets.QLabel("lasso a region or click a point, then Set")
        self.label_status.setWordWrap(True)
        gv7.addWidget(self.label_status)

        row67 = QtWidgets.QHBoxLayout()
        row67.addWidget(g6)
        row67.addWidget(g7)
        colC.addLayout(row67)
        colC.addStretch(1)

        return w

    # --------------------------------------------------------------- center
    def _build_center(self):
        """Build the interactive matplotlib scatter panel (canvas, toolbar, lasso/remove controls) and wire its mouse events."""
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.fig = Figure(figsize=(7, 6))
        self.fig.set_facecolor(MPL_BG)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.toolbar = NavigationToolbar(self.canvas, w)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.toolbar)
        top.addStretch(1)
        v.addLayout(top)
        v.addWidget(self.canvas, 1)
        bottom = QtWidgets.QHBoxLayout()
        self.lasso_btn = QtWidgets.QPushButton("Enable lasso")
        self.lasso_btn.setCheckable(True)
        self.lasso_btn.toggled.connect(self.toggle_lasso)
        self.remove_btn = QtWidgets.QPushButton("Remove selected")
        self.remove_btn.clicked.connect(self.on_remove_selected)
        self.sel_label = QtWidgets.QLabel("selection: 0")
        hint = QtWidgets.QLabel("scroll = zoom · drag = rotate(3D)/pan-tool(2D)")
        hint.setStyleSheet("color:#888;")
        bottom.addWidget(self.lasso_btn)
        bottom.addWidget(self.remove_btn)
        bottom.addWidget(hint)
        bottom.addStretch(1)
        bottom.addWidget(self.sel_label)
        v.addLayout(bottom)
        self.ax = None
        self.scatter = None
        self._lasso_xy: list[tuple[float, float]] = []
        self._lasso_line = None
        self.canvas.mpl_connect("pick_event", self.on_pick)
        self.canvas.mpl_connect("button_press_event", self._lasso_press)
        self.canvas.mpl_connect("motion_notify_event", self._lasso_move)
        self.canvas.mpl_connect("button_release_event", self._lasso_release)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        # allow the scatter column to shrink (the Figure's size hint is large by default)
        self.canvas.setMinimumSize(120, 120)
        w.setMinimumWidth(150)
        return w

    # ---------------------------------------------------------------- right
    def _build_right(self):
        """Build the Analysis-tab right column (selected-FOV thumbnail grid plus the details table) and return its widget."""
        w = QtWidgets.QWidget()
        w.setMinimumWidth(440)
        v = QtWidgets.QVBoxLayout(w)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(QtWidgets.QLabel("<b>Selected FOVs</b>"))
        head.addStretch(1)
        head.addWidget(QtWidgets.QLabel("channel"))
        self.chan_combo = QtWidgets.QComboBox()  # switch displayed FOV channel
        self.chan_combo.currentTextChanged.connect(self._set_channel)
        head.addWidget(self.chan_combo)
        head.addWidget(QtWidgets.QLabel("thumb size"))
        self.size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.size_slider.setRange(60, 800)
        self.size_slider.setValue(150)
        self.size_slider.setFixedWidth(140)
        self.size_slider.sliderReleased.connect(self._rebuild_grid)
        head.addWidget(self.size_slider)
        v.addLayout(head)
        self.grid_info = QtWidgets.QLabel("-")
        self.grid_info.setWordWrap(True)
        v.addWidget(self.grid_info)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.grid_host = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.grid_host)
        self.grid.setSpacing(4)
        self.grid.setAlignment(QtCore.Qt.AlignTop)
        self.scroll.setWidget(self.grid_host)
        self.scroll.verticalScrollBar().valueChanged.connect(self._load_visible_thumbs)
        v.addWidget(self.scroll, 1)
        v.addWidget(QtWidgets.QLabel("<b>FOV details</b>  (one row per selected FOV)"))
        self.details = QtWidgets.QTableView()
        self.details_model = _DetailsModel()
        self.details.setModel(self.details_model)
        self.details.verticalHeader().setVisible(False)
        self.details.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.details.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.details.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.details.horizontalHeader().setDefaultSectionSize(95)
        self.details.setMaximumHeight(280)
        self.details.clicked.connect(lambda idx: self._on_details_click(idx.row()))
        v.addWidget(self.details)
        self._detail_rows: list[int] = []
        self._thumb_items: list[list] = []  # [label, png_path, loaded, pos]
        self._cur_ncols = 0
        self._thumb_cache: OrderedDict = OrderedDict()  # (path,size) -> QPixmap, LRU
        return w

    # ------------------------------------------------------------- label tab
    def _build_label_tab(self):
        """One panel per present goodness class (Good/Neutral/Bad/Nan). Thumbnails can
        be dragged between panels; edits are only written to disk on Save."""
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(
            QtWidgets.QLabel(
                "<b>FOVs grouped by goodness label</b>  ·  drag a FOV to another panel to relabel it"
            )
        )
        head.addStretch(1)
        head.addWidget(QtWidgets.QLabel("thumb size"))
        self.label_size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.label_size_slider.setRange(60, 400)
        self.label_size_slider.setValue(150)
        self.label_size_slider.setFixedWidth(140)
        self.label_size_slider.sliderReleased.connect(self._resize_label_thumbs)
        head.addWidget(self.label_size_slider)
        self.add_class_btn = QtWidgets.QPushButton("Add class")
        self.add_class_btn.clicked.connect(self._show_add_class_menu)
        head.addWidget(self.add_class_btn)
        self.label_save_btn = QtWidgets.QPushButton("Save")
        self.label_save_btn.clicked.connect(self.on_save_labels)
        head.addWidget(self.label_save_btn)
        v.addLayout(head)
        self.label_info = QtWidgets.QLabel("-")
        self.label_info.setWordWrap(True)
        v.addWidget(self.label_info)
        self.label_cols_host = QtWidgets.QWidget()
        self.label_cols_layout = QtWidgets.QHBoxLayout(self.label_cols_host)
        self.label_cols_layout.setSpacing(8)
        v.addWidget(self.label_cols_host, 1)
        self._label_cols: list[dict] = []  # per-panel state (see _build_label_panel)
        self._label_changes: dict[int, float] = {}  # df position -> pending goodness value
        self._extra_classes: set = set()  # classes with no FOVs the user added as empty panels
        self._update_label_save_state()
        return w

    def _on_tab_changed(self, index):
        """Reflow the Rank or Label tab's lazily-laid-out grids once it becomes visible; caveat: the Label tab is (re)built only the first time it is shown so pending drag edits are not discarded."""
        if index == getattr(self, "_rank_tab_index", -1):
            # built while hidden (viewport width unknown) -> reflow with the real width
            QtCore.QTimer.singleShot(0, self._rank_reflow)
            QtCore.QTimer.singleShot(0, self._rank_load_visible)
            return
        if index != getattr(self, "_label_tab_index", -1):
            return
        # build panels the first time the Label tab is shown after data changes; do NOT
        # rebuild on every visit, or unsaved drag edits would be discarded.
        if self.df is not None and not self._label_cols:
            self._refresh_label_tab()
        # panels may have been built while this tab was hidden (viewport width unknown ->
        # too few columns); reflow now that it has its real width. Deferred so the layout
        # settles first.
        QtCore.QTimer.singleShot(0, self._reflow_all_label_panels)

    def _refresh_label_tab(self):
        """Rebuild the Label-tab panels from the current (saved) goodness labels, one panel per present class, discarding any pending drag edits."""
        # rebuild the columns from scratch (cheap: placeholders only; thumbnails are
        # decoded lazily for the cells actually visible in each column's viewport).
        while self.label_cols_layout.count():
            it = self.label_cols_layout.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self._label_cols = []
        self._label_changes = {}
        self._update_label_save_state()
        if self.df is None or len(self.df) == 0:
            self.label_info.setText("no data loaded")
            return
        if "goodness" in self.df.columns:
            gv = self.df["goodness"].to_numpy(float)
        else:
            gv = np.full(len(self.df), np.nan)
        groups = []
        for name, val, color in GOODNESS_LABEL_ORDER:
            mask = np.isnan(gv) if val is None else (gv == val)
            pos = np.where(mask)[0]
            # show a panel if the class has FOVs OR the user added it as an empty target
            if len(pos) or self._class_key(val) in self._extra_classes:
                pval = float("nan") if val is None else float(val)
                groups.append((name, pval, color, pos))
        if not groups:
            self.label_info.setText("no FOVs")
            return
        self.label_info.setText(
            f"{len(self.df)} FOV(s)  ·  "
            + "   ".join(f"{n}: {len(p)}" for n, _, _, p in groups)
        )
        # each class is a panel; panels split the window equally (one class -> full width)
        for name, val, color, pos in groups:
            self._build_label_panel(name, val, color, pos)
        self._relayout_panels()
        QtCore.QTimer.singleShot(0, self._reflow_all_label_panels)

    def _build_label_panel(self, name, value, color, positions):
        """A full panel for one class: a plain title above a wrapping, drop-enabled grid
        of draggable thumbnails. `value` is the goodness code this panel assigns."""
        size = self.label_size_slider.value()
        # show every FOV in the class: only placeholders are created up front, and
        # thumbnails are decoded lazily as they scroll into view, so this stays cheap.
        shown = list(positions)
        panel = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(panel)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(6)
        title = QtWidgets.QLabel()
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet(f"font-size:16px; font-weight:700; color:{color}; padding:4px;")
        pv.addWidget(title)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        host = _DropPanel(self, value)  # accepts thumbnails dropped from any panel
        grid = QtWidgets.QGridLayout(host)
        grid.setSpacing(4)
        grid.setAlignment(QtCore.Qt.AlignTop)
        scroll.setWidget(host)
        pv.addWidget(scroll, 1)
        items = []
        for pos in shown:
            row = self.df.iloc[pos]
            png = row.get("__png", "")
            lab = _ThumbLabel(int(pos))
            lab.setFixedSize(size, size)
            lab.setAlignment(QtCore.Qt.AlignCenter)
            lab.setStyleSheet(THUMB_QSS)
            lab.setCursor(QtCore.Qt.OpenHandCursor)
            lab.setToolTip(f"{row.get('__dataset', '')}\n{self._row_id(row)}")
            lab.setText("…")
            items.append([lab, png, False, int(pos)])
        info = {
            "scroll": scroll,
            "grid": grid,
            "items": items,
            "ncols": 0,
            "value": value,
            "name": name,
            "color": color,
            "count": len(positions),
            "capped": False,
            "title": title,
            "panel": panel,
        }
        self._update_panel_title(info)
        scroll.verticalScrollBar().valueChanged.connect(
            lambda *_, c=info: self._load_visible_label_thumbs(c)
        )
        self._label_cols.append(info)
        return panel

    @staticmethod
    def _same_class(a, b):
        """True if two goodness codes name the same class, treating NaN as equal to NaN."""
        if a != a and b != b:  # both NaN -> same "Nan" class
            return True
        return a == b

    @staticmethod
    def _class_key(val):
        """Hashable identity for a class (NaN isn't equal to itself, so map it to a str)."""
        return "nan" if (val is None or val != val) else val

    @staticmethod
    def _class_rank(val):
        """Position of a class in the fixed Good/Neutral/Bad/Nan order (for panel order)."""
        key = FeatureViewer._class_key(val)
        for i, (_, v, _) in enumerate(GOODNESS_LABEL_ORDER):
            if FeatureViewer._class_key(v) == key:
                return i
        return len(GOODNESS_LABEL_ORDER)

    def _class_meta(self, val):
        """Return the (display name, color) for a goodness class, defaulting to a blue for unknown codes."""
        key = self._class_key(val)
        for name, v, color in GOODNESS_LABEL_ORDER:
            if self._class_key(v) == key:
                return name, color
        return str(val), "#5b8def"

    def _show_add_class_menu(self):
        """Offer the classes not already shown; picking one adds an empty target panel."""
        if self.df is None:
            self.label_info.setText("load data before adding a class")
            return
        present = {self._class_key(i["value"]) for i in self._label_cols}
        menu = QtWidgets.QMenu(self)
        added_any = False
        for name, val, _ in GOODNESS_LABEL_ORDER:
            if self._class_key(val) in present:
                continue
            act = menu.addAction(name)
            act.triggered.connect(lambda _=False, v=val: self._add_class_panel(v))
            added_any = True
        if not added_any:
            act = menu.addAction("All classes already shown")
            act.setEnabled(False)
        menu.exec_(self.add_class_btn.mapToGlobal(self.add_class_btn.rect().bottomLeft()))

    def _add_class_panel(self, val):
        """Add an empty panel for `val` so FOVs can be dragged into a class with no FOVs."""
        self._extra_classes.add(self._class_key(val))
        pval = float("nan") if val is None else float(val)
        if any(self._same_class(i["value"], pval) for i in self._label_cols):
            return  # panel already present
        name, color = self._class_meta(val)
        self._build_label_panel(name, pval, color, np.array([], int))
        info = self._label_cols.pop()  # _build_label_panel appended it at the end
        idx = sum(
            1
            for i in self._label_cols
            if self._class_rank(i["value"]) < self._class_rank(pval)
        )
        self._label_cols.insert(idx, info)
        self._relayout_panels()  # re-add panels + separators in class order
        QtCore.QTimer.singleShot(0, lambda c=info: self._reflow_label_panel(c))

    def _update_panel_title(self, info):
        """Refresh a Label-tab panel's title text with its class name and FOV count."""
        cap = f"  ·  showing first {len(info['items'])}" if info["capped"] else ""
        info["title"].setText(f"{info['name']}   ({info['count']}){cap}")

    def _on_label_drop(self, pos, target_value):
        """Move FOV `pos` into the panel for `target_value` (pending until Save)."""
        source = target = item = si = None
        for info in self._label_cols:
            if self._same_class(info["value"], target_value):
                target = info
            for k, it in enumerate(info["items"]):
                if it[3] == pos:
                    source, item, si = info, it, k
        if source is None or target is None or source is target:
            return
        source["items"].pop(si)
        source["count"] -= 1
        target["items"].append(item)
        target["count"] += 1
        self._label_changes[pos] = target_value
        self._relayout_label_panel_grid(target)  # reparents the moved thumbnail first
        self._relayout_label_panel_grid(source)
        self._update_panel_title(source)
        self._update_panel_title(target)
        self._load_visible_label_thumbs(target)
        self._update_label_save_state()

    def _relayout_label_panel_grid(self, info):
        """Re-place all of the panel's thumbnails after its item set changed."""
        grid = info["grid"]
        while grid.count():
            grid.takeAt(0)  # detach layout items; widgets kept alive
        ncols = max(1, info["ncols"] or 1)
        for i, it in enumerate(info["items"]):
            grid.addWidget(it[0], i // ncols, i % ncols)

    def _update_label_save_state(self):
        """Enable the Save button and show the pending-change count only when edits are outstanding."""
        if not hasattr(self, "label_save_btn"):
            return
        n = len(getattr(self, "_label_changes", {}))
        self.label_save_btn.setEnabled(n > 0)
        self.label_save_btn.setText(f"Save ({n})" if n else "Save")

    def on_save_labels(self):
        """Commit pending Label-tab drag edits to the in-memory table and each row's source CSV, then rebuild the panels from the saved labels."""
        if self.df is None or not self._label_changes:
            return
        if "goodness" not in self.df.columns:
            self.df["goodness"] = np.nan
        goodness_col_index = self.df.columns.get_loc("goodness")
        for pos, val in self._label_changes.items():  # commit to the in-memory table
            self.df.iat[pos, goodness_col_index] = val
        saved, matched = self._persist_changes(self._label_changes)
        n_changes = len(self._label_changes)
        self._label_changes = {}
        self._populate_details()
        if self.cb_color.currentText() == "goodness":
            self.update_plot()
        # drop remembered empty-class panels: after saving, a class shows only if it
        # still has FOVs, so classes emptied by these moves disappear.
        self._extra_classes = set()
        self._refresh_label_tab()  # rebuild panels from the now-saved labels
        self.label_info.setText(
            f"saved {n_changes} change(s) → wrote {matched} row(s) to {saved} CSV(s)"
        )

    def _persist_changes(self, changes):
        """Write pending labels back to each row's own source CSV. Rows are grouped by
        __src so multiple loaded datasets each save to the right file, and matched by
        FOV identity so the on-disk row order is irrelevant."""
        src_i = self.df.columns.get_loc("__src")
        by_src: dict[str, list[tuple[int, float]]] = {}
        for pos, val in changes.items():
            by_src.setdefault(self.df.iat[pos, src_i], []).append((pos, val))
        saved = matched = 0
        for src, rows in by_src.items():
            try:
                disk = pd.read_csv(src)
            except Exception as e:  # noqa: BLE001
                self.label_info.setText(f"read failed: {Path(src).name}: {e}")
                continue
            if "goodness" not in disk.columns:
                disk["goodness"] = np.nan
            keycols = [
                c
                for c in ("filename", "well_row", "well_col", "fov", "timepoint")
                if c in disk.columns and c in self.df.columns
            ]
            if not keycols:
                continue
            lookup = {
                k: i for i, k in enumerate(disk[keycols].astype(str).agg("\x1f".join, axis=1))
            }
            lc = [self.df.columns.get_loc(c) for c in keycols]
            gi = disk.columns.get_loc("goodness")
            for pos, val in rows:
                k = "\x1f".join(str(self.df.iat[pos, c]) for c in lc)
                j = lookup.get(k)
                if j is not None:
                    disk.iat[j, gi] = val
                    matched += 1
            disk.to_csv(src, index=False)
            saved += 1
        return saved, matched

    def _reflow_label_panel(self, info):
        """Lay the panel's thumbnails into as many grid columns as its width allows."""
        items = info["items"]
        if not items:
            return
        size = self.label_size_slider.value()
        vw = info["scroll"].viewport().width() - 12
        ncols = max(1, vw // (size + info["grid"].spacing()))
        if ncols == info["ncols"]:
            return
        info["ncols"] = ncols
        for i, it in enumerate(items):
            info["grid"].addWidget(it[0], i // ncols, i % ncols)
        QtCore.QTimer.singleShot(0, lambda c=info: self._load_visible_label_thumbs(c))

    def _load_visible_label_thumbs(self, info):
        """Decode and set thumbnails only for a Label panel's cells currently in (or near) its viewport, marking each as loaded."""
        items = info["items"]
        if not items:
            return
        scroll = info["scroll"]
        sb = scroll.verticalScrollBar()
        y0, y1 = sb.value(), sb.value() + scroll.viewport().height()
        size = self.label_size_slider.value()
        rowh = size + info["grid"].spacing()
        ncols = max(1, info["ncols"])
        margin = rowh
        for idx, it in enumerate(items):
            lab, png, loaded = it[0], it[1], it[2]
            if loaded:
                continue
            top = (idx // ncols) * rowh
            if top + size < y0 - margin or top > y1 + margin:
                continue
            if png and Path(png).exists():
                pm = self._load_thumb(png, size)
                if not pm.isNull():
                    lab.setText("")
                    lab.setPixmap(pm)
                else:
                    lab.setText("bad png")
            else:
                lab.setText("no png")
            it[2] = True

    def _reflow_all_label_panels(self):
        """Reflow every Label-tab panel to its current width and load the newly-visible thumbnails."""
        for info in self._label_cols:
            self._reflow_label_panel(info)
        self._load_all_label_thumbs()

    def _load_all_label_thumbs(self):
        """Decode the visible thumbnails in every Label-tab panel."""
        for info in self._label_cols:
            self._load_visible_label_thumbs(info)

    @staticmethod
    def _make_separator():
        """Return a thin vertical divider widget used between Label-tab panels."""
        line = QtWidgets.QWidget()
        line.setFixedWidth(2)
        line.setStyleSheet("background:#7a7a7a;")
        line.setProperty("_is_separator", True)
        return line

    def _relayout_panels(self):
        """Re-add every panel to the row in class order with a vertical divider between
        each pair. Panels are detached, not destroyed, so their contents (and any pending
        drag edits) are preserved."""
        while self.label_cols_layout.count():
            it = self.label_cols_layout.takeAt(0)
            w = it.widget()
            if w is not None and w.property("_is_separator"):
                w.deleteLater()  # drop old dividers; keep panel widgets alive
        for k, info in enumerate(self._label_cols):
            if k > 0:
                self.label_cols_layout.addWidget(self._make_separator())
            self.label_cols_layout.addWidget(info["panel"], 1)

    def _resize_label_thumbs(self):
        """Change thumbnail size only -- keep every FOV in its current (possibly dragged,
        not-yet-saved) panel instead of rebuilding from the saved labels."""
        size = self.label_size_slider.value()
        for info in self._label_cols:
            info["ncols"] = 0  # force a reflow at the new size
            for it in info["items"]:
                it[0].setFixedSize(size, size)
                it[0].setText("…")  # clears the old pixmap
                it[2] = False  # re-decode at the new size when visible
        self._reflow_all_label_panels()

    # =============================================================== load
    def on_load(self):
        """Prompt for feature CSV(s) via a file dialog and load any the user selects."""
        start = DEFAULT_DIR if Path(DEFAULT_DIR).exists() else str(Path.home())
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Load feature CSV(s)", start, "CSV (*.csv)"
        )
        if not files:
            return
        self._load_files(files)

    def _load_files(self, files):
        """Load feature CSV(s); each row's image comes from sibling PNG folders."""
        self._ingest(D.load_matrices(files), files)

    def _load_paired(self, pairs):
        """Load explicit (csv, png_folder) pairs, each matrix with its own image folder."""
        self._ingest(D.load_paired(pairs), [c for c, _ in pairs])

    def _ingest(self, df, files):
        """Common post-load wiring for both load paths."""
        self.df = (
            df if self.df is None else pd.concat([self.df, df], ignore_index=True, sort=False)
        )
        for f in files:
            self.ds_list.addItem(Path(f).name)
        self.reduced_cols = []
        self._reset_hidden()
        self._clear_focus()
        self._populate_columns()
        self.apply_filters()
        self._refresh_label_tab()

    def on_remove_dataset(self):
        """Drop the dataset selected in the list from the combined table and refresh all views; caveat: matches by CSV basename, so two loaded files with the same name are both removed."""
        it = self.ds_list.currentItem()
        if it is None or self.df is None:
            return
        name = it.text()
        keep = self.df["__src"].apply(lambda s: Path(s).name != name).to_numpy()
        self.df = self.df[keep].reset_index(drop=True)
        self.ds_list.takeItem(self.ds_list.row(it))
        if len(self.df) == 0:
            self.df = None
        self.reduced_cols = [
            c
            for c in (self.df.columns if self.df is not None else [])
            if D.REDUCED_RE.match(c)
        ]
        self._reset_hidden()
        self._clear_focus()  # positions shifted -> stale highlight
        self._populate_columns()
        self.apply_filters()
        self._refresh_label_tab()

    def on_clear(self):
        """Reset the viewer to the empty state: drop all data, filters, caches, and clear every view."""
        self.df = None
        self.filt = np.array([], int)
        self.reduced_cols = []
        self.thresholds = {}
        self.lasso_sel = None
        self._thumb_cache.clear()
        self.ds_list.clear()
        self._reset_hidden()
        self._clear_focus()
        self._populate_columns()
        self.fig.clear()
        self.canvas.draw_idle()
        self.set_selection([])
        self._extra_classes = set()
        self._refresh_label_tab()

    # ---- removed-points (hidden) model ----
    def _reset_hidden(self):
        """Clear the removed-points mask and history to match the current data length."""
        self.hidden = (
            np.zeros(len(self.df), bool) if self.df is not None else np.array([], bool)
        )
        self.remove_history = []
        if hasattr(self, "history_list"):
            self.history_list.clear()
        self._update_hidden_count()

    def _effective_positions(self):
        """Filtered AND not-hidden positions -- the points actually plotted/reduced."""
        if self.df is None or len(self.filt) == 0:
            return np.array([], int)
        return self.filt[~self.hidden[self.filt]]

    def _update_hidden_count(self):
        """Update the label showing how many points are currently removed (hidden)."""
        n = int(self.hidden.sum()) if self.hidden.size else 0
        if hasattr(self, "hidden_count"):
            self.hidden_count.setText(f"{n} points hidden")

    def _viewing_reduced(self):
        """True if any active scatter axis is a dimensionality-reduction component (PCA/TSNE/UMAP)."""
        axes = [self.cb_x.currentText(), self.cb_y.currentText()]
        if self.mode.currentText() == "3D":
            axes.append(self.cb_z.currentText())
        return any(D.REDUCED_RE.match(a) for a in axes if a)

    def _after_visibility_change(self):
        """After points are removed/restored, re-fit the reduced embedding on the survivors if one is shown, otherwise just replot."""
        # if viewing a reduced embedding, always re-fit it on the survivors
        if self._viewing_reduced() and len(self._effective_positions()) >= 3:
            self._rerun_current_reduction()
        else:
            self.update_plot()

    def _rerun_current_reduction(self):
        """Detect which reduction (PCA/t-SNE/UMAP) the current axes show and re-run it; caveat: falls back to a plain replot if no axis is a reduced component."""
        import re

        for a in (self.cb_x.currentText(), self.cb_y.currentText(), self.cb_z.currentText()):
            if a and D.REDUCED_RE.match(a):
                pref = re.match(r"^(PCA|TSNE|UMAP)", a).group(1)
                method = {"PCA": "PCA", "TSNE": "t-SNE", "UMAP": "UMAP"}[pref]
                if method in D.METHODS:
                    self.cb_method.setCurrentText(method)
                    self.run_reduction()
                    return
        self.update_plot()

    def _reset_to_default_view(self):
        """Back to default: no lasso, no highlight, panel = all visible FOVs."""
        self.lasso_sel = None
        self._clear_lasso_line()
        self._clear_focus()
        self._refresh_panel()

    def on_remove_selected(self):
        """Hide (soft-remove) the explicitly selected FOVs (lasso subset or focused point), recording an undoable history entry; caveat: does nothing on the default 'all visible' set."""
        if self.df is None:
            return
        # remove only an EXPLICIT selection -- a lasso subset, or a highlighted point;
        # never the default "all visible" set
        if self.lasso_sel:
            positions = [p for p in self.lasso_sel if not self.hidden[p]]
        elif self.focus_pos is not None and not self.hidden[self.focus_pos]:
            positions = [self.focus_pos]
        else:
            positions = []
        if not positions:
            return
        self.hidden[positions] = True
        self.remove_history.append(list(positions))
        self.history_list.addItem(f"#{len(self.remove_history)}: removed {len(positions)} pts")
        self._update_hidden_count()
        self._after_visibility_change()  # recompute reduction (if shown) + replot scatter
        self._reset_to_default_view()  # panel/details -> all visible FOVs

    def on_restore_selected(self):
        """Un-hide the FOVs from the history entry selected in the removed-points list."""
        r = self.history_list.currentRow()
        if r < 0 or r >= len(self.remove_history):
            return
        positions = self.remove_history.pop(r)
        self.history_list.takeItem(r)
        self.hidden[positions] = False
        self._update_hidden_count()
        self._after_visibility_change()
        self._reset_to_default_view()

    def on_restore_all(self):
        """Un-hide every removed FOV and clear the removal history."""
        if self.df is None:
            return
        self.hidden[:] = False
        self.remove_history = []
        self.history_list.clear()
        self._update_hidden_count()
        self._after_visibility_change()
        self._reset_to_default_view()

    # ---- label selected FOVs (goodness) ----
    def _label_targets(self):
        """Positions to label: the lasso subset if active, else the single focused
        point, else all currently-visible (filtered, not-hidden) FOVs."""
        if self.lasso_sel:
            return [p for p in self.lasso_sel if not self.hidden[p]]
        if self.focus_pos is not None and not self.hidden[self.focus_pos]:
            return [self.focus_pos]
        return list(self._effective_positions())

    def on_set_goodness(self):
        """Apply the chosen goodness label to the targeted FOVs both in memory and on disk, then refresh the details, scatter, and Label tab."""
        if self.df is None:
            self.label_status.setText("no data loaded")
            return
        positions = self._label_targets()
        if not positions:
            self.label_status.setText("nothing selected")
            return
        raw_value = self.label_value.currentData()
        value = float("nan") if raw_value is None else float(raw_value)  # "Unlabeled" -> NaN
        if "goodness" not in self.df.columns:
            self.df["goodness"] = np.nan
        goodness_col_index = self.df.columns.get_loc("goodness")
        self.df.iloc[positions, goodness_col_index] = value  # update in memory
        saved, matched = self._persist_goodness(positions, value)  # write to disk
        display_value = "NaN" if value != value else str(int(value))
        self.label_status.setText(
            f"set {len(positions)} FOV(s) → {display_value}; "
            f"wrote {matched} row(s) to {saved} CSV(s)"
        )
        self._populate_details()  # details table shows goodness
        if self.cb_color.currentText() == "goodness":
            self.update_plot()
        self._refresh_label_tab()  # move relabeled FOVs to their column

    def _persist_goodness(self, positions, value):
        """Write the label back to each row's own source CSV (rows are grouped by
        __src so multiple loaded datasets each save to the right file). Rows are
        matched by FOV identity, so the on-disk row order is irrelevant."""
        src_i = self.df.columns.get_loc("__src")
        by_src: dict[str, list[int]] = {}
        for p in positions:
            by_src.setdefault(self.df.iat[p, src_i], []).append(p)
        saved = matched = 0
        for src, ps in by_src.items():
            try:
                disk = pd.read_csv(src)
            except Exception as e:  # noqa: BLE001
                self.label_status.setText(f"read failed: {Path(src).name}: {e}")
                continue
            if "goodness" not in disk.columns:
                disk["goodness"] = np.nan
            keycols = [
                c
                for c in ("filename", "well_row", "well_col", "fov", "timepoint")
                if c in disk.columns and c in self.df.columns
            ]
            if not keycols:
                continue
            lookup = {
                k: i for i, k in enumerate(disk[keycols].astype(str).agg("\x1f".join, axis=1))
            }
            lc = [self.df.columns.get_loc(c) for c in keycols]
            gi = disk.columns.get_loc("goodness")
            for p in ps:
                k = "\x1f".join(str(self.df.iat[p, c]) for c in lc)
                j = lookup.get(k)
                if j is not None:
                    disk.iat[j, gi] = value
                    matched += 1
            disk.to_csv(src, index=False)
            saved += 1
        return saved, matched

    def _populate_columns(self):
        """Repopulate the axis / color / threshold combo boxes from the loaded columns, defaulting axes to the best-covered features; caveat: toggles self._ready off/on so combo updates do not trigger premature replots."""
        self._ready = False
        self._refresh_channel_combos()  # channel toggles reflect the loaded data's channels
        feats = D.feature_columns(self.df) if self.df is not None else []
        axis_opts = feats + self.reduced_cols
        all_cols = [
            c
            for c in (self.df.columns if self.df is not None else [])
            if not c.startswith("__") and not c.endswith("__png")
        ]
        # Default axes: prefer features populated across ALL loaded datasets (fewest
        # NaNs). Datasets with different feature sets (concat -> NaN for the columns a
        # dataset lacks) would otherwise leave one dataset's points off the plot on a
        # dataset-specific default axis, so a lasso could only select the other one.
        # Stable sort keeps original order among equally-covered (e.g. single-dataset)
        # columns, so single-dataset behavior is unchanged.
        if self.df is not None and axis_opts:
            na = {c: int(self.df[c].isna().sum()) for c in axis_opts}
            default_order = sorted(axis_opts, key=lambda c: na[c])
        else:
            default_order = axis_opts
        for cb, dflt in [(self.cb_x, 0), (self.cb_y, 1), (self.cb_z, 2)]:
            cur = cb.currentText()
            cb.clear()
            cb.addItems(axis_opts)
            if cur in axis_opts:
                cb.setCurrentText(cur)
            elif len(default_order) > dflt:
                cb.setCurrentText(default_order[dflt])
        cur_color = self.cb_color.currentText()
        self.cb_color.clear()
        self.cb_color.addItems(all_cols)
        if cur_color in all_cols:  # keep the user's color choice across re-runs
            self.cb_color.setCurrentText(cur_color)
        elif "goodness" in all_cols:  # default only on first populate
            self.cb_color.setCurrentText("goodness")
        self.thr_col.clear()
        self.thr_col.addItems(axis_opts)  # numeric features + reduced comps
        self.cb_z.setEnabled(self.mode.currentText() == "3D")
        self._ready = True
        self._on_thr_col(self.thr_col.currentText())
        self._sync_rank_tab()

    # ----- feature thresholds: keep FOVs whose feature is within [lo, hi] -----
    def _on_thr_col(self, feat):
        """Pre-fill min/max from the active threshold, else the feature's data range."""
        if self.df is None or not feat or feat not in self.df.columns:
            return
        if feat in self.thresholds:
            lo, hi = self.thresholds[feat]
        else:
            v = self.df[feat].to_numpy(float)
            v = v[~np.isnan(v)]
            lo, hi = (float(v.min()), float(v.max())) if v.size else (0.0, 0.0)
        self.thr_lo.setValue(lo)
        self.thr_hi.setValue(hi)

    def _refresh_thr_list(self):
        """Rebuild the threshold list widget from the active per-feature keep-ranges."""
        self.thr_list.clear()
        for feat, (lo, hi) in self.thresholds.items():
            self.thr_list.addItem(f"{feat} ∈ [{lo:g}, {hi:g}]")

    def on_add_threshold(self):
        """Add or update the keep-range for the selected feature from the min/max spinboxes and re-filter."""
        feat = self.thr_col.currentText()
        if self.df is None or not feat:
            return
        self.thresholds[feat] = (self.thr_lo.value(), self.thr_hi.value())
        self._refresh_thr_list()
        self.apply_filters()

    def on_remove_threshold(self):
        """Remove the threshold selected in the list and re-filter."""
        r = self.thr_list.currentRow()
        if r < 0:
            return
        feat = list(self.thresholds)[r]
        self.thresholds.pop(feat, None)
        self._refresh_thr_list()
        self.apply_filters()

    def on_clear_thresholds(self):
        """Remove all feature thresholds and re-filter."""
        self.thresholds = {}
        self._refresh_thr_list()
        self.apply_filters()

    def apply_filters(self):
        """Recompute the filtered-FOV set from the active thresholds, reset the lasso, and refresh the scatter and panel."""
        if self.df is None:
            # last dataset removed -> clear the scatter and the FOV panel too
            self.filt = np.array([], int)
            self.plot_pos = np.array([], int)
            self.lasso_sel = None
            self._clear_focus()
            self.filter_summary.setText("no data")
            self.fig.clear()
            self.ax = None
            self.scatter = None
            self.canvas.draw_idle()
            self.set_selection([])
            return
        mask = np.ones(len(self.df), bool)
        for feat, (lo, hi) in self.thresholds.items():
            if feat in self.df.columns:
                values = self.df[feat].to_numpy(float)
                mask &= (values >= lo) & (
                    values <= hi
                )  # NaN comparisons are False -> excluded
        self.filt = np.where(mask)[0]
        self.lasso_sel = None  # changing the filters resets the lasso filter
        if self.focus_pos is not None and self.focus_pos not in set(self.filt.tolist()):
            self._clear_focus()
        parts = [f"{f} ∈ [{lo:g},{hi:g}]" for f, (lo, hi) in self.thresholds.items()]
        desc = "; ".join(parts) if parts else "no filter"
        self.filter_summary.setText(f"{len(self.filt)} / {len(self.df)} FOVs  ·  {desc}")
        # re-fit a reduced embedding on the new survivors (else newly-included FOVs
        # have NaN reduced coords and the scatter goes blank); raw views just replot
        self._refresh_panel()
        self._after_visibility_change()

    def _shown_positions(self):
        """Visible positions shown in the panel: lasso subset if active, else all
        filtered -- always excluding removed (hidden) FOVs. Ordered by `score`
        (descending) when the Rank tab has ranked the FOVs."""
        if self.lasso_sel is not None:
            pos = [p for p in self.lasso_sel if not self.hidden[p]]
        else:
            pos = list(self._effective_positions())
        return self._sort_by_score(pos)

    def _sort_by_score(self, pos):
        """Order positions best-first by the `score` column; NaN scores sink to the end."""
        if not self.rank_sort or self.df is None or "score" not in self.df.columns or not pos:
            return pos
        s = self.df["score"].to_numpy(float)
        return sorted(pos, key=lambda p: np.inf if np.isnan(s[p]) else -s[p])

    def _refresh_panel(self):
        """Sync the FOV panel and details table to the currently-shown positions."""
        self.set_selection(self._shown_positions())

    # ----- dim reduction -----
    def _update_reduction_params(self, *_):
        """Show only the parameter relevant to the chosen method (PCA uses none)."""
        m = self.cb_method.currentText()
        for w, show in (
            (self.lbl_perp, m == "t-SNE"),
            (self.sp_perp, m == "t-SNE"),
            (self.lbl_nn, m == "UMAP"),
            (self.sp_nn, m == "UMAP"),
        ):
            w.setVisible(show)

    def run_reduction(self):
        """Fit the selected dimensionality reduction on the visible FOVs' features, store the 3 components as columns, and switch the scatter axes to them; caveat: needs >=3 visible FOVs."""
        visible_positions = self._effective_positions()
        if self.df is None or len(visible_positions) < 3:
            self.reduce_status.setText("need >=3 visible FOVs")
            return
        method = self.cb_method.currentText()
        feature_cols = D.feature_columns(self.df)
        feature_matrix = self.df.iloc[visible_positions][feature_cols].to_numpy(float)
        self.reduce_status.setText(f"running {method} on {len(visible_positions)} FOVs…")
        QtWidgets.QApplication.processEvents()
        try:
            embedding = D.run_reduction(
                feature_matrix,
                method,
                perplexity=self.sp_perp.value(),
                n_neighbors=self.sp_nn.value(),
            )
        except Exception as e:  # noqa: BLE001
            self.reduce_status.setText(f"error: {e}")
            return
        comp_cols = [f"{REDUCE_PREFIX[method]}{i + 1}" for i in range(3)]
        for c in comp_cols:
            if c not in self.df.columns:
                self.df[c] = np.nan
            if c not in self.reduced_cols:
                self.reduced_cols.append(c)
        self.df.loc[self.df.index[visible_positions], comp_cols] = embedding
        self.reduce_status.setText(f"{method} done -> axes {', '.join(comp_cols)} available")
        self._ready = False
        self._populate_columns()
        for cb, c in zip((self.cb_x, self.cb_y, self.cb_z), comp_cols):
            cb.setCurrentText(c)
        self._ready = True
        self.update_plot()  # explicit: combos may be unchanged (already on PCA*), so no signal

    # =============================================================== plotting
    def _axis_values(self, col):
        """Values of column `col` for the currently-plotted FOVs, as a float array."""
        return self.df.iloc[self.plot_pos][col].to_numpy(float)

    def update_plot(self, *_):
        """Redraw the scatter for the current axes/color/mode: numeric colors get a colorbar, categorical (and goodness) colors a legend, and the focus ring is re-applied; caveat: no-op until self._ready is set."""
        if not self._ready or self.df is None:
            return
        self.plot_pos = self._effective_positions()
        if len(self.plot_pos) == 0:
            self.fig.clear()
            self.canvas.draw_idle()
            return
        x_c, y_c = self.cb_x.currentText(), self.cb_y.currentText()
        if not x_c or not y_c:
            return
        is3d = self.mode.currentText() == "3D"
        color_c, cmap = self.cb_color.currentText(), self.cb_cmap.currentText()
        self.fig.clear()
        self._lasso_line = None
        self._lasso_xy = []  # destroyed by fig.clear()
        self._focus_marker = None
        self.ax = self.fig.add_subplot(111, projection="3d" if is3d else None)
        coords = [self._axis_values(x_c), self._axis_values(y_c)]
        if is3d:
            coords.append(self._axis_values(self.cb_z.currentText()))
        cser = self.df.iloc[self.plot_pos][color_c]
        # goodness is categorical (good/neutral/bad/unlabeled), never a colorbar
        force_cat = color_c == "goodness"
        if pd.api.types.is_numeric_dtype(cser) and not force_cat:
            sc = self.ax.scatter(
                *coords, c=cser.to_numpy(float), cmap=cmap, s=14, alpha=0.85, picker=5
            )
            cb = self.fig.colorbar(sc, ax=self.ax, shrink=0.6, label=color_c)
            cb.ax.tick_params(colors=MPL_FG)
            cb.set_label(color_c, color=MPL_FG)
            cb.outline.set_edgecolor(MPL_GRID)
        else:
            if force_cat:  # NaN -> "unlabeled" (still plotted); fixed semantic colors
                cats = cser.map(
                    lambda v: (
                        "unlabeled"
                        if pd.isna(v)
                        else GOODNESS_CATEGORIES.get(float(v), str(v))
                    )
                )
                order = ["good (1)", "neutral (0)", "bad (-1)", "unlabeled"]
                present = set(cats)
                uniq = [u for u in order if u in present] + sorted(
                    present - set(order), key=str
                )
                pal = matplotlib.colormaps.get_cmap("tab10")
                cmap_d = {
                    u: GOODNESS_COLORS.get(u, pal(i % pal.N)) for i, u in enumerate(uniq)
                }
            else:
                cats = cser.fillna("NaN").astype(str)
                uniq = sorted(cats.unique(), key=str)
                pal = matplotlib.colormaps.get_cmap("tab10" if len(uniq) <= 10 else "tab20")
                cmap_d = {u: pal(i % pal.N) for i, u in enumerate(uniq)}
            sc = self.ax.scatter(
                *coords, c=[cmap_d[v] for v in cats], s=14, alpha=0.85, picker=5
            )
            leg = self.ax.legend(
                handles=[
                    Line2D([], [], marker="o", ls="", color=cmap_d[u], label=u) for u in uniq
                ],
                fontsize=8,
                loc="best",
                title=color_c,
            )
            for t in leg.get_texts():
                t.set_color(MPL_FG)
            leg.get_title().set_color(MPL_FG)
            leg.get_frame().set_facecolor(MPL_BG)
            leg.get_frame().set_edgecolor(MPL_GRID)
        self.scatter = sc
        self.ax.set_xlabel(x_c)
        self.ax.set_ylabel(y_c)
        if is3d:
            self.ax.set_zlabel(self.cb_z.currentText())
            self.ax.mouse_init()  # ensure drag-rotate is enabled
        self._style_axes(self.ax, is3d)
        self.fig.tight_layout()
        self.canvas.draw_idle()
        # reset the toolbar's navigation history so its Home button returns to this
        # full view (axes are recreated each plot; scroll-zoom bypasses the stack)
        try:
            self.toolbar.update()
            self.toolbar.push_current()
        except Exception:  # noqa: BLE001
            pass
        if self.focus_pos is not None:  # re-draw the highlight ring on the new axes
            self._highlight_scatter(self.focus_pos)

    def _style_axes(self, ax, is3d):
        """Apply the dark theme colors to the scatter axes, ticks, grid, and (in 3D) panes."""
        self.fig.set_facecolor(MPL_BG)
        ax.set_facecolor(MPL_BG)
        ax.tick_params(colors=MPL_FG, which="both")
        ax.xaxis.label.set_color(MPL_FG)
        ax.yaxis.label.set_color(MPL_FG)
        ax.title.set_color(MPL_FG)
        if is3d:
            ax.zaxis.label.set_color(MPL_FG)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_pane_color((0.27, 0.27, 0.27, 1.0))
                axis.line.set_color(MPL_GRID)
                axis._axinfo["grid"]["color"] = MPL_GRID
        else:
            for s in ax.spines.values():
                s.set_color(MPL_GRID)
            ax.grid(True, color=MPL_GRID, alpha=0.35)

    def reset_view(self):
        """Reset the scatter to its default view (3D: default angles + autoscale; 2D: toolbar Home)."""
        if self.ax is None:
            return
        if self.mode.currentText() == "3D":
            self.ax.view_init(elev=30, azim=-60)
            self.ax.autoscale()
        else:
            self.toolbar.home()
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        """Zoom the scatter in/out on scroll, centered on the cursor in 2D and on the axis midpoints in 3D."""
        if self.ax is None or event.inaxes is not self.ax:
            return
        factor = 0.83 if event.button == "up" else 1.2  # up = zoom in
        if self.mode.currentText() == "3D":
            for get, setl in [
                (self.ax.get_xlim3d, self.ax.set_xlim3d),
                (self.ax.get_ylim3d, self.ax.set_ylim3d),
                (self.ax.get_zlim3d, self.ax.set_zlim3d),
            ]:
                lo, hi = get()
                mid = (lo + hi) / 2
                half = (hi - lo) / 2 * factor
                setl(mid - half, mid + half)
        else:
            xc = event.xdata if event.xdata is not None else np.mean(self.ax.get_xlim())
            yc = event.ydata if event.ydata is not None else np.mean(self.ax.get_ylim())
            lo, hi = self.ax.get_xlim()
            self.ax.set_xlim(xc - (xc - lo) * factor, xc + (hi - xc) * factor)
            lo, hi = self.ax.get_ylim()
            self.ax.set_ylim(yc - (yc - lo) * factor, yc + (hi - yc) * factor)
        self.canvas.draw_idle()

    # =============================================================== picking
    def on_pick(self, event):
        """Handle clicking a scatter point: focus that FOV (or toggle off if re-clicked), cancelling any active lasso first; caveat: ignored while the lasso tool is on."""
        if event.artist is not self.scatter or self.lasso_btn.isChecked():
            return
        picked_indices = event.ind
        if picked_indices is None or len(picked_indices) == 0:
            return
        self._picked = True  # a point was hit -> don't treat this click as "empty"
        position = int(self.plot_pos[int(picked_indices[0])])
        if self.focus_pos == position:  # click the highlighted point again -> clear highlight
            self._clear_focus()
            return
        # clicking a point cancels the lasso filter -> show ALL visible FOVs + details,
        # then highlight the clicked point
        if self.lasso_sel is not None:
            self.lasso_sel = None
            self._clear_lasso_line()
        self._refresh_panel()
        self._set_focus(position, scroll_grid=True)

    def toggle_lasso(self, on):
        """Enable or disable lasso-select mode, disabling 3D rotation while the lasso is on."""
        self.lasso_btn.setText("Lasso ON (drag to select)" if on else "Enable lasso")
        if on and self.ax is not None and hasattr(self.ax, "disable_mouse_rotation"):
            self.ax.disable_mouse_rotation()
        elif (not on) and self.ax is not None and hasattr(self.ax, "mouse_init"):
            self.ax.mouse_init()

    def _clear_lasso_line(self):
        """Remove the lasso outline artist from the scatter if one is drawn."""
        if self._lasso_line is not None:
            try:
                self._lasso_line.remove()
            except Exception:  # noqa: BLE001  (axis may already be gone)
                pass
            self._lasso_line = None
            self.canvas.draw_idle()

    def _lasso_press(self, event):
        """Begin a lasso outline when lasso mode is on; otherwise treat it as a plain plot click (deferred, to run after any point pick)."""
        if not self.lasso_btn.isChecked():
            # a plain click on the plot: cancel the lasso outline; if it didn't land
            # on a point, also clear the selection + highlight (checked after pick fires)
            self._clear_lasso_line()
            QtCore.QTimer.singleShot(0, self._after_plot_click)
            return
        if event.inaxes is not self.ax:
            return
        self._clear_focus()  # drawing a lasso cancels any highlighted point
        self._clear_lasso_line()  # starting a new lasso replaces the previous one
        self._lasso_xy = [(event.x, event.y)]
        self._lasso_line = Line2D(
            [], [], color="crimson", lw=1.5, transform=IdentityTransform()
        )
        self.ax.add_line(self._lasso_line)

    def _after_plot_click(self):
        """Deferred plot-click handler: if no point was hit, clear the lasso/focus and show all FOVs; caveat: reads self._picked set by on_pick, which fires before this."""
        if not self._picked:  # clicked empty space -> cancel lasso + highlight, show all
            self._clear_focus()
            self.lasso_sel = None
            self._clear_lasso_line()
            self._refresh_panel()
        self._picked = False  # reset for the next click

    def _lasso_move(self, event):
        """Extend the in-progress lasso outline to the current cursor position and redraw."""
        if not self._lasso_xy or event.x is None:
            return
        self._lasso_xy.append((event.x, event.y))
        arr = np.array(self._lasso_xy)
        self._lasso_line.set_data(arr[:, 0], arr[:, 1])
        self.canvas.draw_idle()

    def _lasso_release(self, event):
        """Finish the lasso: select the FOVs whose points fall inside the closed loop (keeping the outline), or clear the filter if the gesture was a click or enclosed nothing."""
        if not self._lasso_xy:
            return
        verts = self._lasso_xy
        self._lasso_xy = []
        if len(verts) >= 3:
            # close the loop visually and KEEP the outline (persists until next
            # lasso / plot click / plot redraw)
            arr = np.array(verts + [verts[0]])
            if self._lasso_line is not None:
                if self.mode.currentText() == "3D":
                    self._lasso_line.set_data(arr[:, 0], arr[:, 1])  # screen space
                else:
                    # anchor the outline to data coords so it tracks pan/zoom
                    dcoords = self.ax.transData.inverted().transform(arr)
                    self._lasso_line.set_data(dcoords[:, 0], dcoords[:, 1])
                    self._lasso_line.set_transform(self.ax.transData)
            inside = MplPath(verts).contains_points(self._display_point_positions())
            chosen = [int(self.plot_pos[i]) for i in np.where(inside)[0]]
            if chosen:
                self.lasso_sel = chosen  # lasso acts as a filter
            else:
                self.lasso_sel = None  # lassoed nothing -> no filter (show all)
                self._clear_lasso_line()
            self._refresh_panel()  # panel always matches the lasso
        else:
            # a click (not a drag) while the lasso is on -> cancel the lasso filter
            self._clear_lasso_line()
            self.lasso_sel = None
            self._clear_focus()
            self._refresh_panel()  # -> show all FOVs + details
        self.canvas.draw_idle()

    def _display_point_positions(self):
        """Screen (display) coordinates of every plotted point, for lasso hit-testing (projected first in 3D)."""
        xs, ys = (
            self._axis_values(self.cb_x.currentText()),
            self._axis_values(self.cb_y.currentText()),
        )
        if self.mode.currentText() == "3D":
            zs = self._axis_values(self.cb_z.currentText())
            xp, yp, _ = proj3d.proj_transform(xs, ys, zs, self.ax.get_proj())
            return self.ax.transData.transform(np.column_stack([xp, yp]))
        return self.ax.transData.transform(np.column_stack([xs, ys]))

    # =============================================================== right grid
    def _row_id(self, row):
        """Human-readable FOV identifier for tooltips: the filename if present, else well/fov/timepoint."""
        fn = row.get("filename", None)
        if isinstance(fn, str) and fn:
            return fn
        return (
            f"{row.get('well_row', '')}/{row.get('well_col', '')}/{row.get('fov', '')} "
            f"t{row.get('timepoint', '')}"
        )

    # ---- FOV image channel (brightfield / mask / fluorescence) ----
    def _refresh_channel_combos(self):
        """Populate the channel toggles from the channels present in the loaded data and
        point __png at the active channel."""
        cols = self.df.columns if self.df is not None else []
        chans = [c for c in D.CHANNELS if f"__png_{c}" in cols]
        if self._channel not in chans:
            self._channel = chans[0] if chans else "brightfield"
        if self.df is not None and f"__png_{self._channel}" in cols:
            self.df["__png"] = self.df[f"__png_{self._channel}"]
        for combo in (
            getattr(self, "chan_combo", None),
            getattr(self, "rank_chan_combo", None),
        ):
            if combo is None:
                continue
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(chans or ["brightfield"])
            combo.setCurrentText(self._channel)
            combo.setEnabled(len(chans) > 1)
            combo.blockSignals(False)

    def _set_channel(self, ch):
        """Switch the displayed FOV channel: repoint __png and reload the thumbnail grids."""
        if not ch or self.df is None or f"__png_{ch}" not in self.df.columns:
            return
        self._channel = ch
        self.df["__png"] = self.df[f"__png_{ch}"]
        for combo in (
            getattr(self, "chan_combo", None),
            getattr(self, "rank_chan_combo", None),
        ):
            if combo is not None and combo.currentText() != ch:
                combo.blockSignals(True)
                combo.setCurrentText(ch)
                combo.blockSignals(False)
        if hasattr(self, "grid"):
            self._rebuild_grid()
        if getattr(self, "_rank_order", None):
            self._rank_rebuild_grid()

    def set_selection(self, positions):
        """Set the selected-FOV set and rebuild the details table and thumbnail grid; caveat: skips the rebuild if the selection is unchanged."""
        positions = list(positions)
        if positions == self.sel_pos:  # unchanged -> skip the rebuild (no freeze on reset)
            return
        self.sel_pos = positions
        if hasattr(self, "sel_label"):
            self.sel_label.setText(f"selection: {len(self.sel_pos)}")
        self._populate_details()
        self._rebuild_grid()

    def _populate_details(self):
        """One row per selected FOV; columns = metadata + features (virtualized)."""
        if self.df is None or not self.sel_pos:
            self._detail_rows = []
            self.details_model.set_data(None, [], [])
            return
        fields = [
            c
            for c in self.df.columns
            if c not in DETAIL_SKIP and not c.endswith("__png") and not c.startswith("__png")
        ]
        self._detail_rows = list(self.sel_pos)  # virtualized: no per-row cost
        self.details_model.set_data(self.df, self._detail_rows, fields)

    def _focus_details_row(self, pos):
        """Select and scroll the details table to the row for FOV `pos`, or clear the selection if it isn't listed."""
        if pos in self._detail_rows:
            r = self._detail_rows.index(pos)
            self.details.selectRow(r)
            self.details.scrollTo(
                self.details_model.index(r, 0), QtWidgets.QAbstractItemView.PositionAtCenter
            )
        else:
            self.details.clearSelection()

    def _on_details_click(self, row):
        """Focus the FOV corresponding to the clicked details-table row (scrolling the grid to it)."""
        if 0 <= row < len(self._detail_rows):
            self._set_focus(self._detail_rows[row], scroll_grid=True)

    def _load_thumb(self, path, size):
        """Return a QPixmap thumbnail for `path` scaled to `size`, using and maintaining the LRU cache."""
        key = (path, size)
        pixmap = self._thumb_cache.get(key)
        if pixmap is not None:  # cache hit -> no disk decode
            self._thumb_cache.move_to_end(key)
            return pixmap
        reader = QtGui.QImageReader(path)
        sz = reader.size()
        if sz.isValid() and sz.width() > 0:
            reader.setScaledSize(sz.scaled(size, size, QtCore.Qt.KeepAspectRatio))
        pixmap = QtGui.QPixmap.fromImage(reader.read())
        self._thumb_cache[key] = pixmap
        if len(self._thumb_cache) > THUMB_CACHE_MAX:
            self._thumb_cache.popitem(last=False)  # evict least-recently-used
        return pixmap

    def _balanced_cap(self, positions, cap):
        """Cap the thumbnails shown, but round-robin across datasets so every loaded
        dataset is represented. A plain prefix (positions[:cap]) would show only the
        first-loaded dataset when it alone exceeds `cap`, hiding the others entirely."""
        if len(positions) <= cap:
            return list(positions)
        src_i = self.df.columns.get_loc("__src")
        groups: dict = {}
        for p in positions:  # preserve per-dataset order
            groups.setdefault(self.df.iat[p, src_i], []).append(p)
        order = list(groups.values())
        out, i = [], 0
        while len(out) < cap:
            advanced = False
            for g in order:
                if i < len(g):
                    out.append(g[i])
                    advanced = True
                    if len(out) >= cap:
                        break
            if not advanced:
                break
            i += 1
        return out

    def _rebuild_grid(self):
        """Rebuild the selected-FOV thumbnail grid with lazily-loaded placeholder tiles (capped and balanced across datasets), then re-apply any focus highlight."""
        # cheap: create placeholder labels only; thumbnails are decoded lazily for
        # the cells actually visible in the viewport (and as the user scrolls).
        while self.grid.count():
            it = self.grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self._thumb_items = []
        self._focus_label = None
        if not self.sel_pos or self.df is None:
            self.grid_info.setText("no selection")
            return
        shown = self._balanced_cap(self.sel_pos, MAX_THUMBS)
        extra = (
            f"  ({len(shown)} of {len(self.sel_pos)}, sampled across datasets, refine selection)"
            if len(self.sel_pos) > len(shown)
            else ""
        )
        self.grid_info.setText(f"{len(shown)} FOV(s){extra}")
        size = self.size_slider.value()
        for pos in shown:
            row = self.df.iloc[pos]
            png = row.get("__png", "")
            lab = QtWidgets.QLabel()
            lab.setFixedSize(size, size)
            lab.setAlignment(QtCore.Qt.AlignCenter)
            # border colored by goodness label (green/yellow/red/blue); stored so the
            # focus highlight can restore it after un-focusing.
            qss = goodness_border_qss(row.get("goodness", np.nan))
            lab._base_qss = qss
            lab.setStyleSheet(qss)
            lab.setCursor(QtCore.Qt.PointingHandCursor)
            lab.setToolTip(f"{row.get('__dataset', '')}\n{self._row_id(row)}")
            lab.setText("…")
            lab.mousePressEvent = lambda e, p=int(pos): self._set_focus(p, scroll_grid=False)
            self._thumb_items.append([lab, png, False, int(pos)])
        self._cur_ncols = 0
        self._reflow_grid()
        QtCore.QTimer.singleShot(0, self._load_visible_thumbs)  # after layout settles
        if self.focus_pos is not None:  # re-apply highlight
            self._highlight_thumb(self.focus_pos, scroll_grid=False)

    def _reflow_grid(self):
        """Re-lay the selected-FOV thumbnails into as many columns as the viewport width allows; caveat: no-op if the column count is unchanged."""
        if not self._thumb_items:
            return
        size = self.size_slider.value()
        vw = self.scroll.viewport().width() - 12
        ncols = max(1, vw // (size + self.grid.spacing()))
        if ncols == self._cur_ncols:
            return
        self._cur_ncols = ncols
        for i, item in enumerate(self._thumb_items):
            self.grid.addWidget(item[0], i // ncols, i % ncols)
        QtCore.QTimer.singleShot(0, self._load_visible_thumbs)

    def _load_visible_thumbs(self, *_):
        """Decode thumbnails only for cells in (or near) the viewport; cache them.

        Cell positions are computed analytically from index/columns (not from
        realized widget geometry), so culling is reliable regardless of layout timing.
        """
        if not self._thumb_items:
            return
        sb = self.scroll.verticalScrollBar()
        y0, y1 = sb.value(), sb.value() + self.scroll.viewport().height()
        size = self.size_slider.value()
        rowh = size + self.grid.spacing()
        vw = self.scroll.viewport().width() - 12
        ncols = max(1, vw // rowh)
        margin = rowh  # preload roughly one extra row each way
        for idx, it in enumerate(self._thumb_items):
            lab, png, loaded = it[0], it[1], it[2]
            if loaded:
                continue
            top = (idx // ncols) * rowh
            if top + size < y0 - margin or top > y1 + margin:
                continue
            if png and Path(png).exists():
                pm = self._load_thumb(png, size)
                if not pm.isNull():
                    lab.setText("")
                    lab.setPixmap(pm)
                else:
                    lab.setText("bad png")
            else:
                lab.setText("no png")
            it[2] = True

    # ---- linked highlight (scatter point <-> FOV thumbnail) ----
    def _set_focus(self, pos, scroll_grid=False):
        """Highlight a FOV across scatter + grid + details (does not change the panel set)."""
        self.focus_pos = int(pos)
        self._highlight_scatter(self.focus_pos)
        self._highlight_thumb(self.focus_pos, scroll_grid)
        self._focus_details_row(self.focus_pos)

    def _clear_focus(self):
        """Clear the current FOV highlight: remove the scatter ring, restore the thumbnail border, and clear the details selection."""
        self.focus_pos = None
        if self._focus_marker is not None:
            try:
                self._focus_marker.remove()
            except Exception:  # noqa: BLE001
                pass
            self._focus_marker = None
            self.canvas.draw_idle()
        if self._focus_label is not None:
            self._focus_label.setStyleSheet(getattr(self._focus_label, "_base_qss", THUMB_QSS))
            self._focus_label = None
        self.details.clearSelection()  # keep the table (it lists the whole selection)

    def _point_coords(self, pos):
        """Data coordinates of FOV `pos` on the current scatter axes (X/Y, plus Z in 3D)."""
        cols = [self.cb_x.currentText(), self.cb_y.currentText()]
        if self.mode.currentText() == "3D":
            cols.append(self.cb_z.currentText())
        return [float(self.df.iloc[pos][c]) for c in cols]

    def _highlight_scatter(self, pos):
        """Draw the focus ring around FOV `pos` on the scatter; caveat: skipped if the point is not currently plotted or has NaN coordinates."""
        if self._focus_marker is not None:
            try:
                self._focus_marker.remove()
            except Exception:  # noqa: BLE001
                pass
            self._focus_marker = None
        if self.ax is not None and pos is not None and pos in set(self.plot_pos.tolist()):
            c = self._point_coords(pos)
            if not any(np.isnan(v) for v in c):
                self._focus_marker = self.ax.scatter(
                    *[[v] for v in c],
                    s=240,
                    facecolors="none",
                    edgecolors=FOCUS_COLOR,
                    linewidths=2.5,
                    zorder=6,
                )
        self.canvas.draw_idle()

    def _highlight_thumb(self, pos, scroll_grid):
        """Give FOV `pos`'s thumbnail the blue focus border (restoring the previous one), optionally scrolling the grid to it."""
        if self._focus_label is not None:
            self._focus_label.setStyleSheet(getattr(self._focus_label, "_base_qss", THUMB_QSS))
            self._focus_label = None
        for it in self._thumb_items:
            if it[3] == pos:
                it[0].setStyleSheet(THUMB_FOCUS_QSS)
                self._focus_label = it[0]
                if scroll_grid:
                    self.scroll.ensureWidgetVisible(it[0])
                    self._load_visible_thumbs()
                break

    # ============================================================== rank tab
    def _build_rank_tab(self):
        """Tune the production DesirabilityModel: per-feature ideal range + shoulder +
        direction. LEFT = feature-value histograms with the desirability curve overlaid
        (dashed) plus a table of the range/direction/shoulder knobs and a Re-rank button.
        RIGHT = the loaded FOVs as thumbnails ordered best-first by the resulting score."""
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # ---------------------------------------- LEFT: parameter tuning
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("<b>Desirability ranges</b>"))
        bar.addStretch(1)
        b_load = QtWidgets.QPushButton("Load…")
        b_load.clicked.connect(self._on_rank_load)
        b_save = QtWidgets.QPushButton("Save…")
        b_save.clicked.connect(self._on_rank_save)
        for b in (b_load, b_save):
            bar.addWidget(b)
        lv.addLayout(bar)

        # top: per-feature histograms with the desirability curve as a dashed overlay
        self.rank_fig = Figure(figsize=(4, 6), facecolor="#3c3c3c")
        self.rank_canvas = FigureCanvas(self.rank_fig)
        # drag the profile's control points directly on the histograms
        self._rank_axes: list[dict] = []  # per-subplot: feature, ax2, profile line, roles
        self._rank_drag = None  # (subplot-meta, control-point index) while dragging
        self.rank_canvas.mpl_connect("button_press_event", self._rank_on_press)
        self.rank_canvas.mpl_connect("motion_notify_event", self._rank_on_motion)
        self.rank_canvas.mpl_connect("button_release_event", self._rank_on_release)
        hist_scroll = QtWidgets.QScrollArea()
        hist_scroll.setWidgetResizable(True)
        hist_scroll.setWidget(self.rank_canvas)
        lv.addWidget(hist_scroll, 3)

        # bottom: the tunable knobs, one row per feature
        lv.addWidget(
            QtWidgets.QLabel(
                "The parameter columns adapt to each row's shape/direction (the labels update): "
                "linear -> onset/ideal or range+shoulders; sigmoid -> midpoint/width or band; "
                "gaussian -> center/fwhm; lognormal -> center/fold. A missing feature scores 0."
            )
        )
        self.rank_table = QtWidgets.QTableWidget(0, 8)
        self.rank_table.setHorizontalHeaderLabels(
            [
                "✓ feature",
                "direction",
                "shape",
                "param 1",
                "param 2",
                "param 3",
                "param 4",
                "weight",
            ]
        )
        self.rank_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch
        )
        self.rank_table.verticalHeader().setVisible(False)
        self.rank_table.setMaximumHeight(240)
        # toggling a feature's checkbox updates which features feed the score
        self.rank_table.itemChanged.connect(self._on_rank_item_changed)
        lv.addWidget(self.rank_table, 2)

        act = QtWidgets.QHBoxLayout()
        rerank = QtWidgets.QPushButton("Re-rank")
        rerank.setStyleSheet("font-weight:bold; padding:4px 16px;")
        rerank.clicked.connect(self._rerank)
        act.addWidget(rerank)
        act.addStretch(1)
        lv.addLayout(act)
        self.rank_status = QtWidgets.QLabel("load data to tune the desirability ranges")
        self.rank_status.setWordWrap(True)
        lv.addWidget(self.rank_status)

        # ---------------------------------------- RIGHT: FOVs ordered by score
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(QtWidgets.QLabel("<b>FOVs ranked by score</b> (best first)"))
        head.addStretch(1)
        head.addWidget(QtWidgets.QLabel("channel"))
        self.rank_chan_combo = QtWidgets.QComboBox()  # switch displayed FOV channel
        self.rank_chan_combo.currentTextChanged.connect(self._set_channel)
        head.addWidget(self.rank_chan_combo)
        head.addWidget(QtWidgets.QLabel("thumb size"))
        self.rank_size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.rank_size_slider.setRange(60, 400)
        self.rank_size_slider.setValue(140)
        self.rank_size_slider.setFixedWidth(140)
        self.rank_size_slider.sliderReleased.connect(self._rank_rebuild_grid)
        head.addWidget(self.rank_size_slider)
        rv.addLayout(head)
        self.rank_grid_info = QtWidgets.QLabel("-")
        rv.addWidget(self.rank_grid_info)
        self.rank_scroll = QtWidgets.QScrollArea()
        self.rank_scroll.setWidgetResizable(True)
        self.rank_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.rank_grid_host = QtWidgets.QWidget()
        self.rank_grid = QtWidgets.QGridLayout(self.rank_grid_host)
        self.rank_grid.setSpacing(4)
        self.rank_grid.setAlignment(QtCore.Qt.AlignTop)
        self.rank_scroll.setWidget(self.rank_grid_host)
        self.rank_scroll.verticalScrollBar().valueChanged.connect(self._rank_load_visible)
        rv.addWidget(self.rank_scroll, 1)
        self._rank_thumb_items: list[list] = []  # [label, png, loaded, pos, score, rank]
        self._rank_ncols = 0
        self._rank_order: list[int] = []
        self._rank_focus_pos = None  # df position of the clicked FOV (blue highlight)
        self._rank_focus_label = None

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 4)
        split.setSizes([760, 900])
        return split

    # ---- rank tab: knob state ----
    def _rank_feature_list(self):
        """The feature columns available to rank in the loaded data (empty if none loaded)."""
        return D.feature_columns(self.df) if self.df is not None else []

    def _seed_range(self, f, direction):
        """Data-derived (lo, hi, soft_left, soft_right) for feature ``f`` at ``direction``:
        label-agnostic quantiles of all values (target -> [q25, q75]; monotone -> [q05, q95])
        with IQR-width shoulders."""
        values = self.df[f].to_numpy(float)
        finite = values[~np.isnan(values)]

        def quantile(a, p):
            """The p-quantile of array `a`, or 0.0 if it is empty."""
            return float(np.quantile(a, p)) if len(a) else 0.0

        if direction == "target":
            lo, hi = quantile(finite, 0.25), quantile(finite, 0.75)
        else:
            lo, hi = quantile(finite, 0.05), quantile(finite, 0.95)
        band = max(hi - lo, 1e-9)
        return lo, hi, band, band

    def _rank_seed_ranges(self):
        """Fresh per-feature knobs: direction from the feature-name default, range/shoulders
        from :meth:`_seed_range` (label-agnostic data quantiles)."""
        ranges = {}
        for f in self._rank_feature_list():
            direction = RANK.DEFAULT_DIRECTION.get(RANK._feature_suffix(f), "target")
            lo, hi, sl, sr = self._seed_range(f, direction)
            ranges[f] = {
                "direction": direction,
                "shape": "linear",  # curve family; user can switch to sigmoid/gaussian
                "lo": lo,
                "hi": hi,
                "soft_left": sl,  # left/right shoulders are adjustable independently
                "soft_right": sr,
                "curve_k": 0.0,  # steepness for sigmoid / lognormal
                "weight": 1.0,
                "enabled": True,  # unchecking a feature drops it from the score
            }
        return ranges

    def _make_spinbox(self, val):
        """Build an arrowless, no-wheel float spinbox initialized to `val` (0.0 for None/NaN)."""
        s = QtWidgets.QDoubleSpinBox()
        s.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        s.setKeyboardTracking(False)
        s.setDecimals(5)
        s.setRange(-1e12, 1e12)
        bad = val is None or (isinstance(val, float) and np.isnan(val))
        s.setValue(0.0 if bad else float(val))
        s.wheelEvent = lambda e: e.ignore()
        return s

    def _rank_populate_table(self):
        """Rebuild the Rank-tab knob table (one row per feature) from self.rank_ranges, wiring the checkbox/direction/shape/param/weight widgets."""
        tbl = self.rank_table
        tbl.blockSignals(True)
        tbl.setRowCount(0)
        feats = list(self.rank_ranges)
        tbl.setRowCount(len(feats))
        for i, f in enumerate(feats):
            spec = self.rank_ranges[f]
            item = QtWidgets.QTableWidgetItem(
                f
            )  # checkbox = include this feature in the score
            item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            item.setCheckState(
                QtCore.Qt.Checked if spec.get("enabled", True) else QtCore.Qt.Unchecked
            )
            tbl.setItem(i, RCOL_FEATURE, item)
            dcombo = QtWidgets.QComboBox()
            dcombo.addItems(list(FM.DesirabilityModel.DIRECTIONS))
            dcombo.setCurrentText(spec["direction"])
            dcombo.currentTextChanged.connect(lambda _t, r=i: self._on_rank_dir_changed(r))
            tbl.setCellWidget(i, RCOL_DIR, dcombo)
            scombo = QtWidgets.QComboBox()  # curve family
            scombo.addItems(list(FM.DesirabilityModel.SHAPES))
            scombo.setCurrentText(spec.get("shape", "linear"))
            scombo.currentTextChanged.connect(lambda _t, r=i: self._on_rank_shape_changed(r))
            tbl.setCellWidget(i, RCOL_SHAPE, scombo)
            self._rank_fill_params(i, spec)  # shape-dependent parameter columns
            # per-feature weight: how much this feature counts in the weighted score (>= 0)
            wspin = self._make_spinbox(spec.get("weight", 1.0))
            wspin.setDecimals(2)
            wspin.setRange(0.0, 1e6)
            wspin.editingFinished.connect(self._rank_refresh_curves)
            tbl.setCellWidget(i, RCOL_WEIGHT, wspin)
            self._rank_update_row_enabled(i)
        tbl.blockSignals(False)

    def _make_param_spinbox(self, key, value):
        """A labeled spin for one interpretable parameter: the param name is the (non-editable)
        prefix, so the column self-describes whatever the row's shape needs. Stores its param
        key on the widget so :meth:`_read_rank_table` knows what it is."""
        s = self._make_spinbox(value)
        s.setPrefix(f"{RANK_PARAM_LABELS.get(key, key)}  ")
        if key in ("fwhm", "width"):  # strictly positive widths
            s.setRange(1e-9, 1e12)
        elif key == "fold":  # multiplicative tolerance must exceed 1
            s.setRange(1.0 + 1e-6, 1e12)
        s._param_key = key
        return s

    def _rank_fill_params(self, row, spec):
        """(Re)build the four parameter columns for ``row`` from its internal spec, showing the
        interpretable params for the current shape/direction (see fov_model.curve_params)."""
        tbl = self.rank_table
        params = FM.curve_params(
            spec.get("shape", "linear"),
            spec["direction"],
            spec["lo"],
            spec["hi"],
            spec["soft_left"],
            spec["soft_right"],
            spec.get("curve_k", 0.0),
        )
        items = list(params.items())
        for idx, col in enumerate(RCOL_PARAMS):
            if idx < len(items):
                key, value = items[idx]
                spin = self._make_param_spinbox(key, value)
                spin.editingFinished.connect(self._rank_refresh_curves)
                tbl.setCellWidget(row, col, spin)
            else:  # unused slot for this shape
                tbl.removeCellWidget(row, col)

    def _rank_update_row_enabled(self, row):
        """Direction applies only to linear / sigmoid; gaussian and lognormal are symmetric
        bells, so force their direction to 'target' and disable the direction combo."""
        dcombo = self.rank_table.cellWidget(row, RCOL_DIR)
        shape = self.rank_table.cellWidget(row, RCOL_SHAPE).currentText()
        if dcombo is None:
            return
        is_bell = shape in ("gaussian", "lognormal")
        if is_bell and dcombo.currentText() != "target":
            dcombo.blockSignals(True)
            dcombo.setCurrentText("target")
            dcombo.blockSignals(False)
        dcombo.setEnabled(not is_bell)

    def _rank_row_feature(self, row):
        """The feature name at table `row`, or None if the row index is out of range."""
        feats = list(self.rank_ranges)
        return feats[row] if 0 <= row < len(feats) else None

    def _on_rank_dir_changed(self, row):
        """Apply a direction change for `row`: store it, rebuild that row's shape-dependent param columns, and redraw the curves."""
        # Direction changes the parameter SET (e.g. linear target range+shoulders <-> onset/
        # ideal ramp), so rebuild this row's param columns from its internal spec, then redraw.
        f = self._rank_row_feature(row)
        if f is None:
            return
        self.rank_ranges[f]["direction"] = self.rank_table.cellWidget(
            row, RCOL_DIR
        ).currentText()
        self.rank_table.blockSignals(True)
        self._rank_fill_params(row, self.rank_ranges[f])
        self.rank_table.blockSignals(False)
        self._rank_refresh_curves()

    def _on_rank_shape_changed(self, row):
        """Apply a shape change for `row`: store it, force target/disable direction for bell shapes, rebuild the param columns from the kept bounds, and redraw."""
        # Shape changes the parameter SET; keep the internal bounds and re-derive the params for
        # the new shape (a linear [lo,hi] becomes a gaussian center/fwhm around the same band).
        f = self._rank_row_feature(row)
        if f is None:
            return
        self.rank_ranges[f]["shape"] = self.rank_table.cellWidget(
            row, RCOL_SHAPE
        ).currentText()
        self._rank_update_row_enabled(row)  # bells force direction=target / disable the combo
        self.rank_ranges[f]["direction"] = self.rank_table.cellWidget(
            row, RCOL_DIR
        ).currentText()
        self.rank_table.blockSignals(True)
        self._rank_fill_params(row, self.rank_ranges[f])
        self.rank_table.blockSignals(False)
        self._rank_refresh_curves()

    def _rank_refresh_curves(self):
        """Redraw the desirability dashed lines from the current knobs (on Enter / commit /
        direction change), WITHOUT re-ranking the FOVs. The histograms and their locked axes
        are unchanged -- only the overlaid profile moves. Re-rank still recomputes scores."""
        if self.df is None or not self.rank_ranges:
            return
        self._read_rank_table()
        self._rank_draw_hists()

    def _read_rank_table(self):
        """Pull the table widgets back into ``self.rank_ranges``. The shape-dependent param
        spins are converted to the internal (lo, hi, soft, curve_k) bounds via
        fov_model.curve_bounds; a transient invalid entry (e.g. fold=1 mid-edit) keeps the
        row's previous values rather than raising."""
        tbl = self.rank_table
        for i, f in enumerate(list(self.rank_ranges)):
            prev = self.rank_ranges[f]
            direction = tbl.cellWidget(i, RCOL_DIR).currentText()
            shape = tbl.cellWidget(i, RCOL_SHAPE).currentText()
            params = {}
            for col in RCOL_PARAMS:
                w = tbl.cellWidget(i, col)
                if w is not None and getattr(w, "_param_key", None) is not None:
                    params[w._param_key] = w.value()
            try:
                lo, hi, sl, sr, curve_k = FM.curve_bounds(shape, direction, params)
            except (ValueError, KeyError):
                lo, hi, sl, sr, curve_k = (
                    prev["lo"],
                    prev["hi"],
                    prev["soft_left"],
                    prev["soft_right"],
                    prev.get("curve_k", 0.0),
                )
            weight = tbl.cellWidget(i, RCOL_WEIGHT).value()
            item = tbl.item(i, RCOL_FEATURE)
            enabled = item is None or item.checkState() == QtCore.Qt.Checked
            self.rank_ranges[f] = {
                "direction": direction,
                "shape": shape,
                "lo": lo,
                "hi": hi,
                "soft_left": sl,
                "soft_right": sr,
                "curve_k": curve_k,
                "weight": weight,
                "enabled": enabled,
            }

    def _rank_reorder_checked_first(self):
        """Order rank_ranges -- and hence the table rows and histograms -- with CHECKED
        features first, keeping the relative order within the checked / unchecked groups."""
        if self.rank_ranges:
            items = sorted(
                self.rank_ranges.items(), key=lambda kv: not kv[1].get("enabled", True)
            )
            self.rank_ranges = dict(items)

    def _on_rank_item_changed(self, item):
        """A feature's include-checkbox toggled: move the checked features (their knob row and
        histogram) to the top, then refresh the profiles (dims the off ones)."""
        if item.column() != RCOL_FEATURE:
            return
        self._read_rank_table()  # capture current knobs + checkbox states
        self._rank_reorder_checked_first()
        self._rank_populate_table()  # rebuild rows in the new (checked-first) order
        self._rank_draw_hists()  # redraw histograms in the same order

    def _rank_model_cfg(self):
        """DesirabilityModel config from the CHECKED features only (unchecked ones are
        excluded from the score, giving control over how many features are used).

        Each feature is emitted with the interpretable params for its shape via
        :func:`_internal_to_feature` (fov_model.curve_params), so the saved profile / config
        never carries values the shape ignores and matches exactly what the model parses."""
        feats = {
            f: _internal_to_feature(
                s.get("shape", "linear"),
                s["direction"],
                s["lo"],
                s["hi"],
                s["soft_left"],
                s["soft_right"],
                s.get("curve_k", 0.0),
                s.get("weight", 1.0),
            )
            for f, s in self.rank_ranges.items()
            if s.get("enabled", True)
        }
        return {"type": "ranking_by_defined_range", "features": feats}

    # ---- rank tab: actions ----
    def _rerank(self):
        """Score every FOV with the current knobs, reorder the right-side grid, and redraw
        the histogram overlays. Bound to the Re-rank button."""
        if self.df is None or not self.rank_ranges:
            self.rank_status.setText("no features to rank; load data first")
            return
        self._read_rank_table()
        cfg = self._rank_model_cfg()
        n_used = len(cfg["features"])
        if n_used == 0:
            self.rank_status.setText("check at least one feature to score the FOVs")
            return
        model = FM.build_fov_model(cfg)
        proba, _good = model.predict(self.df)
        self.df["score"] = np.asarray(proba, float)
        # best-first; NaN scores sink to the end (numpy argsort puts NaN last)
        self._rank_order = list(np.argsort(-self.df["score"].to_numpy(float), kind="stable"))
        self._rank_rebuild_grid()
        self._rank_draw_hists()
        s = self.df["score"]
        acc, npairs = self._rank_pairwise_accuracy(self.df["score"].to_numpy(float))
        acc_txt = (
            f" · pairwise acc {acc:.3f} ({npairs} good/neutral/bad pairs)"
            if acc is not None
            else " · label FOVs (goodness) to get pairwise accuracy"
        )
        self.rank_status.setText(
            f"ranked {len(self.df)} FOV(s) with {n_used}/{len(self.rank_ranges)} feature(s) · "
            f"score in [{s.min():.3f}, {s.max():.3f}]{acc_txt}"
        )

    def _rank_pairwise_accuracy(self, scores):
        """Fraction of labeled (winner, loser) pairs the scores order correctly, where the
        order is the goodness relevance bad<neutral<good. Ties in score count 0.5. Returns
        (accuracy, n_pairs), or (None, 0) if there are no ordered pairs (too few labels)."""
        if self.df is None or "goodness" not in self.df.columns:
            return None, 0
        g = self.df["goodness"].to_numpy(float)
        keep = np.isin(g, (-1.0, 0.0, 1.0)) & ~np.isnan(scores)
        r, sc = g[keep], np.asarray(scores)[keep]
        if r.size < 2:
            return None, 0
        rd = r[:, None] - r[None, :]  # relevance of i minus j
        sd = sc[:, None] - sc[None, :]
        wins = rd > 0  # pairs where i should rank above j
        npairs = int(wins.sum())
        if npairs == 0:
            return None, 0
        correct = float((sd[wins] > 0).sum() + 0.5 * (sd[wins] == 0).sum())
        return correct / npairs, npairs

    @staticmethod
    def _profile_points(spec):
        """Draggable anchor points of the profile, as (x, y, role): 'lo'/'hi' move the band
        edges, 'soft_left'/'soft_right' the shoulder widths. The curve BETWEEN anchors is set
        by the shape (drawn by sampling); these are just the drag handles. Only the LINEAR
        target has the two shoulder-base handles; every other shape exposes just lo/hi (placed
        at the curve's height there)."""
        lo, hi = spec["lo"], spec["hi"]
        if spec.get("shape", "linear") == "linear" and spec["direction"] == "target":
            return [
                (lo - spec["soft_left"], 0.0, "soft_left"),
                (lo, 1.0, "lo"),
                (hi, 1.0, "hi"),
                (hi + spec["soft_right"], 0.0, "soft_right"),
            ]

        def desirability_at(x):
            """The profile's desirability height at value `x` for this feature's spec."""
            return float(
                FM.DesirabilityModel._desirability(
                    np.array([x]),
                    lo,
                    hi,
                    spec["direction"],
                    spec["soft_left"],
                    spec["soft_right"],
                    spec.get("shape", "linear"),
                    spec.get("curve_k", 0.0),
                )[0]
            )

        return [(lo, desirability_at(lo), "lo"), (hi, desirability_at(hi), "hi")]

    @staticmethod
    def _sample_profile(spec, lo_x, hi_x, n=240):
        """(xs, ds) of the desirability curve across [lo_x, hi_x] using the model shape."""
        xs = np.linspace(lo_x, hi_x, n)
        ds = FM.DesirabilityModel._desirability(
            xs,
            spec["lo"],
            spec["hi"],
            spec["direction"],
            spec["soft_left"],
            spec["soft_right"],
            spec.get("shape", "linear"),
            spec.get("curve_k", 0.0),
        )
        return xs, ds

    def _rank_draw_hists(self):
        """One histogram of measured values per feature, with the desirability profile drawn
        as a dashed line through DRAGGABLE control points (the shoulder / ramp corners).

        Axis limits are LOCKED to the full data range of each feature, so the whole
        histogram is always shown and tuning the range never rescales the axes."""
        fig = self.rank_fig
        fig.clear()
        self._rank_axes = []
        feats = list(self.rank_ranges)
        if self.df is None or not feats:
            self.rank_canvas.draw_idle()
            return
        ncol = 2
        nrow = int(np.ceil(len(feats) / ncol))
        # give each row a fixed pixel height so the canvas grows (and scrolls) with the
        # feature count instead of squeezing every histogram into the viewport.
        row_px = 230
        self.rank_canvas.setMinimumHeight(row_px * nrow)
        for i, f in enumerate(feats):
            ax = fig.add_subplot(nrow, ncol, i + 1, facecolor="#2b2b2b")
            spec = self.rank_ranges[f]
            v = self.df[f].to_numpy(float)
            finite = ~np.isnan(v)
            vv = v[finite]
            if vv.size:
                lo_x, hi_x = float(vv.min()), float(vv.max())
                bins = np.linspace(lo_x, hi_x, 31)
                # total distribution in grey, then each goodness class overlaid as a step
                ax.hist(vv, bins=bins, color="#9a9a9a", alpha=0.55)
                if "goodness" in self.df.columns:
                    gcol = self.df["goodness"].to_numpy(float)
                    for val, c in ((1.0, "#4caf50"), (0.0, "#ffd24d"), (-1.0, "#e53935")):
                        cv = v[finite & (gcol == val)]
                        if cv.size:
                            ax.hist(cv, bins=bins, histtype="step", color=c, lw=1.3)
            else:
                lo_x, hi_x = float(spec["lo"]), float(spec["hi"])
            if hi_x <= lo_x:
                hi_x = lo_x + 1e-9
            # desirability profile (right axis, 0..1): smooth SAMPLED curve for the chosen
            # shape + draggable anchor markers. Unchecked features are dimmed.
            on = spec.get("enabled", True)
            prof_c = "#ffffff" if on else "#666666"
            ax2 = ax.twinx()
            xs_c, ds_c = self._sample_profile(spec, lo_x, hi_x)
            (curve,) = ax2.plot(xs_c, ds_c, "--", color=prof_c, lw=1.6)
            pts = self._profile_points(spec)
            (markers,) = ax2.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                "o",
                color=prof_c,
                ms=7,
                mfc=prof_c,
                mec="#2b2b2b",
            )
            ax2.set_ylim(-0.05, 1.08)
            ax2.tick_params(colors="#ffffff", labelsize=6)
            # the clicked FOV's value for this feature (blue dashed line) + its desirability:
            # a dot where the line meets the profile, labeled "value -> desirability".
            if self._rank_focus_pos is not None:
                fv = (
                    float(self.df.iloc[self._rank_focus_pos][f])
                    if f in self.df.columns
                    else np.nan
                )
                if not np.isnan(fv):
                    ax.axvline(fv, color=RANK_CLICK_COLOR, lw=1.6, ls="--")
                    fd = float(
                        FM.DesirabilityModel._desirability(
                            np.array([fv]),
                            spec["lo"],
                            spec["hi"],
                            spec["direction"],
                            spec["soft_left"],
                            spec["soft_right"],
                            spec.get("shape", "linear"),
                            spec.get("curve_k", 0.0),
                        )[0]
                    )
                    ax2.plot([fv], [fd], "o", color=RANK_CLICK_COLOR, ms=6, zorder=5)
                    ax2.annotate(
                        f"{fv:.3g} → {fd:.2f}",
                        (fv, fd),
                        textcoords="offset points",
                        xytext=(4, 4),
                        color=RANK_CLICK_COLOR,
                        fontsize=6,
                    )
            # lock x to the full histogram, independent of the knobs; disable autoscale so
            # a dragged control point can never rescale it.
            for a in (ax, ax2):
                a.set_xlim(lo_x, hi_x)
                a.set_autoscalex_on(False)
            # feature label INSIDE the axes (top-left) instead of a title band above it, so
            # rows can be packed with almost no vertical gap.
            title = f"{f}  ({spec['direction']})" + ("" if on else "  [off]")
            ax.text(
                0.02,
                0.94,
                title,
                transform=ax.transAxes,
                va="top",
                ha="left",
                color="#ececec" if on else "#888888",
                fontsize=7,
                bbox=dict(facecolor="#2b2b2b", alpha=0.6, pad=1, lw=0),
            )
            ax.tick_params(colors="#bbbbbb", labelsize=6)
            for sp in ax.spines.values():
                sp.set_color("#666")
            self._rank_axes.append(
                {
                    "feature": f,
                    "ax": ax,
                    "ax2": ax2,
                    "curve": curve,
                    "markers": markers,
                    "roles": [p[2] for p in pts],
                    "xlim": (lo_x, hi_x),
                }
            )
        # Explicit tight margins (fractions of the figure) fill the canvas with almost no
        # blank border; small hspace packs the rows since the label is inside each axes.
        fig.subplots_adjust(
            left=0.08, right=0.94, top=0.995, bottom=0.03, hspace=0.18, wspace=0.28
        )
        self.rank_canvas.draw_idle()

    # ---- rank tab: drag the profile control points ----
    def _rank_on_press(self, event):
        """Begin dragging the nearest profile control point (within 10 px) across all rank histograms."""
        if event.x is None:
            return
        best = None  # (meta, point index, pixel distance)
        for meta in self._rank_axes:
            xs, ys = meta["markers"].get_data()
            for j, (x, y) in enumerate(zip(xs, ys)):
                px, py = meta["ax2"].transData.transform((x, y))
                dist = np.hypot(px - event.x, py - event.y)
                if dist <= 10 and (best is None or dist < best[2]):
                    best = (meta, j, dist)
        if best is not None:
            self._rank_drag = (best[0], best[1])

    def _rank_on_motion(self, event):
        """While dragging, move the grabbed control point (lo/hi/shoulder), sync the table row, and reshape the overlaid curve."""
        if self._rank_drag is None or event.x is None:
            return
        meta, j = self._rank_drag
        f = meta["feature"]
        spec = self.rank_ranges[f]
        lo_x, hi_x = meta["xlim"]
        # pixel -> this subplot's data x (works even if the cursor drifts to another subplot)
        xdata, _ = meta["ax2"].transData.inverted().transform((event.x, event.y))
        x = min(max(float(xdata), lo_x), hi_x)
        role = meta["roles"][j]
        lo, hi = spec["lo"], spec["hi"]
        sl, sr = spec["soft_left"], spec["soft_right"]
        if role == "lo":
            lo = min(x, hi)
        elif role == "hi":
            hi = max(x, lo)
        elif role == "soft_left":
            sl = max(lo - x, 1e-9)
        elif role == "soft_right":
            sr = max(x - hi, 1e-9)
        spec.update(lo=lo, hi=hi, soft_left=sl, soft_right=sr)
        self._rank_sync_row(f)  # keep the table knobs in sync
        pts = self._profile_points(spec)
        meta["markers"].set_data([p[0] for p in pts], [p[1] for p in pts])
        meta["roles"] = [p[2] for p in pts]
        xs_c, ds_c = self._sample_profile(spec, lo_x, hi_x)  # reshape the sampled curve
        meta["curve"].set_data(xs_c, ds_c)
        self.rank_canvas.draw_idle()

    def _rank_on_release(self, _event):
        """End any in-progress control-point drag."""
        self._rank_drag = None

    def _rank_sync_row(self, feature):
        """After a drag moved a feature's internal bounds, refresh its parameter spinboxes in
        place (no signals). The drag keeps the shape/direction fixed, so the param SET is
        unchanged -- just re-derive and set each param's value."""
        feats = list(self.rank_ranges)
        if feature not in feats:
            return
        r = feats.index(feature)
        spec = self.rank_ranges[feature]
        params = FM.curve_params(
            spec.get("shape", "linear"),
            spec["direction"],
            spec["lo"],
            spec["hi"],
            spec["soft_left"],
            spec["soft_right"],
            spec.get("curve_k", 0.0),
        )
        for col in RCOL_PARAMS:
            w = self.rank_table.cellWidget(r, col)
            key = getattr(w, "_param_key", None) if w is not None else None
            if key is not None and key in params:
                w.blockSignals(True)
                w.setValue(float(params[key]))
                w.blockSignals(False)

    # ---- rank tab: right-side thumbnail grid (ordered by score) ----
    def _annotate_pixmap(self, pm, text):
        """Return a copy of ``pm`` with a small rank/score caption painted top-left."""
        out = QtGui.QPixmap(pm)
        p = QtGui.QPainter(out)
        p.fillRect(0, 0, out.width(), 16, QtGui.QColor(0, 0, 0, 130))
        p.setPen(QtGui.QColor("#ffffff"))
        p.drawText(3, 12, text)
        p.end()
        return out

    def _rank_rebuild_grid(self):
        """Clear the Rank-tab thumbnail grid and materialize the first page of score-ordered FOVs."""
        while self.rank_grid.count():
            it = self.rank_grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self._rank_thumb_items = []
        self._rank_ncols = 0
        self._rank_focus_label = None  # tiles are recreated; re-applied by _rank_grow
        if self.df is None or not self._rank_order:
            self.rank_grid_info.setText("no FOVs")
            return
        self._rank_grow(reset=True)  # first page; more pages load as the user scrolls

    def _rank_grow(self, reset: bool = False):
        """Create placeholder tiles for the next page of ranked FOVs (infinite scroll):
        ALL FOVs are shown, but tiles are materialized RANK_PAGE at a time as the user
        scrolls toward the bottom. Thumbnails within a page still decode lazily."""
        order = self._rank_order
        built = len(self._rank_thumb_items)
        if not reset and built >= len(order):
            return
        target = min(len(order), built + RANK_PAGE)
        size = self.rank_size_slider.value()
        for rank in range(built, target):
            pos = order[rank]
            row = self.df.iloc[pos]
            png = row.get("__png", "")
            score = float(row.get("score", float("nan")))
            lab = QtWidgets.QLabel()
            lab.setFixedSize(size, size)
            lab.setAlignment(QtCore.Qt.AlignCenter)
            lab.setCursor(QtCore.Qt.PointingHandCursor)
            lab._base_qss = goodness_border_qss(row.get("goodness", np.nan))
            if pos == self._rank_focus_pos:  # keep the blue highlight across rebuilds
                lab.setStyleSheet(_border_qss(RANK_CLICK_COLOR))
                self._rank_focus_label = lab
            else:
                lab.setStyleSheet(lab._base_qss)
            lab.setToolTip(f"#{rank + 1}  score={score:.4f}\n{self._row_id(row)}")
            lab.setText("…")
            lab.mousePressEvent = lambda e, p=int(pos): self._rank_on_click(p)
            self._rank_thumb_items.append([lab, png, False, int(pos), score, rank + 1])
        loaded = len(self._rank_thumb_items)
        more = "  · scroll for more" if loaded < len(order) else ""
        self.rank_grid_info.setText(f"{loaded} of {len(order)} FOV(s) loaded{more}")
        self._rank_ncols = 0  # force re-place of all current tiles
        self._rank_reflow()
        QtCore.QTimer.singleShot(0, self._rank_load_visible)

    def _rank_on_click(self, pos: int):
        """Select a FOV: blue border on its tile + a vertical line at its value in each
        histogram. Clicking the selected FOV again clears the selection."""
        if self._rank_focus_label is not None:  # un-highlight the previous tile
            self._rank_focus_label.setStyleSheet(
                getattr(self._rank_focus_label, "_base_qss", THUMB_QSS)
            )
            self._rank_focus_label = None
        toggled_off = pos == self._rank_focus_pos
        self._rank_focus_pos = None if toggled_off else int(pos)
        if not toggled_off:
            for it in self._rank_thumb_items:
                if it[3] == pos:
                    it[0].setStyleSheet(_border_qss(RANK_CLICK_COLOR))
                    self._rank_focus_label = it[0]
                    break
        self._rank_draw_hists()  # (re)draw the value lines for the current selection

    def _rank_reflow(self):
        """Re-lay the Rank-tab thumbnails into as many columns as the viewport allows; caveat: no-op if the column count is unchanged."""
        if not self._rank_thumb_items:
            return
        size = self.rank_size_slider.value()
        vw = self.rank_scroll.viewport().width() - 12
        ncols = max(1, vw // (size + self.rank_grid.spacing()))
        if ncols == self._rank_ncols:
            return
        self._rank_ncols = ncols
        for i, item in enumerate(self._rank_thumb_items):
            self.rank_grid.addWidget(item[0], i // ncols, i % ncols)
        QtCore.QTimer.singleShot(0, self._rank_load_visible)

    def _rank_load_visible(self, *_):
        """Decode the visible Rank-tab thumbnails (captioned with rank/score) and, near the bottom, trigger loading the next page."""
        if not self._rank_thumb_items:
            return
        sb = self.rank_scroll.verticalScrollBar()
        y0, y1 = sb.value(), sb.value() + self.rank_scroll.viewport().height()
        size = self.rank_size_slider.value()
        rowh = size + self.rank_grid.spacing()
        vw = self.rank_scroll.viewport().width() - 12
        ncols = max(1, vw // rowh)
        margin = rowh
        for idx, it in enumerate(self._rank_thumb_items):
            lab, png, loaded = it[0], it[1], it[2]
            if loaded:
                continue
            top = (idx // ncols) * rowh
            if top + size < y0 - margin or top > y1 + margin:
                continue
            if png and Path(png).exists():
                pm = self._load_thumb(png, size)
                if not pm.isNull():
                    lab.setText("")
                    lab.setPixmap(self._annotate_pixmap(pm, f"#{it[5]}  {it[4]:.4f}"))
                else:
                    lab.setText("bad png")
            else:
                lab.setText(f"#{it[5]}  {it[4]:.4f}\n(no png)")
            it[2] = True
        # infinite scroll: near the bottom and more FOVs remain -> materialize next page
        if (
            len(self._rank_thumb_items) < len(self._rank_order)
            and sb.value() >= sb.maximum() - 2 * rowh
        ):
            QtCore.QTimer.singleShot(0, self._rank_grow)

    # ---- rank tab: sync + profile IO ----
    def _sync_rank_tab(self):
        """Rebuild the Rank tab for the currently loaded data (called after load/remove)."""
        if not hasattr(self, "rank_table"):
            return
        feats = self._rank_feature_list()
        if not feats:
            self.rank_ranges = {}
            self.rank_table.setRowCount(0)
            self.rank_fig.clear()
            self.rank_canvas.draw_idle()
            self._rank_order = []
            self._rank_rebuild_grid()
            self.rank_status.setText("no features found in the loaded data")
            return
        if list(self.rank_ranges) != feats:  # new / changed feature set -> reseed
            self.rank_ranges = self._rank_seed_ranges()
            self._rank_populate_table()
        self._rerank()

    def _on_rank_save(self):
        """Save the current desirability model config (checked features only) to a JSON file via a file dialog."""
        if not self.rank_ranges:
            self.rank_status.setText("nothing to save; load data first")
            return
        self._read_rank_table()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save desirability ranges",
            str(PROFILE_DIR / "desirability_ranges.json"),
            "JSON (*.json)",
        )
        if not path:
            return
        import json

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self._rank_model_cfg(), indent=2))
        self.rank_status.setText(
            f"saved ranges → {Path(path).name} (paste under fov_selection.model in the config)"
        )

    def _on_rank_load(self):
        """Load a desirability profile JSON, merging its (checked) features over the data-seeded (unchecked) ones, then re-rank; caveat: legacy `range`-style profiles are still parsed."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load desirability ranges", str(PROFILE_DIR), "JSON (*.json)"
        )
        if not path:
            return
        import json

        try:
            cfg = json.loads(Path(path).read_text())
            feats = cfg.get("features", cfg) if isinstance(cfg, dict) else {}
            loaded = {}
            for f, feat_cfg in feats.items():
                # curve_bounds does the shape math; legacy `range` profiles still open.
                shape, direction, lo, hi, sl, sr, curve_k = _feature_to_internal(feat_cfg)
                loaded[f] = {
                    "direction": direction,
                    "shape": shape,
                    "lo": lo,
                    "hi": hi,
                    "soft_left": sl,
                    "soft_right": sr,
                    "curve_k": curve_k,
                    "weight": float(feat_cfg.get("weight", 1.0)),
                    "enabled": True,  # a feature in the file is one to use
                }
        except Exception as e:  # noqa: BLE001
            self.rank_status.setText(f"load failed: {e}")
            return
        # Keep all data features in the table so they can be re-enabled: seed from the data,
        # leave the ones absent from the file unchecked, and overlay the loaded ones (checked).
        if self.df is not None and self._rank_feature_list():
            merged = self._rank_seed_ranges()
            for spec in merged.values():
                spec["enabled"] = False
            merged.update(loaded)
            self.rank_ranges = merged
        else:
            self.rank_ranges = loaded
        self._rank_reorder_checked_first()  # loaded (checked) features on top
        self._rank_populate_table()
        if self.df is not None:
            self._rerank()
        self.rank_status.setText(f"loaded {len(loaded)} feature(s) from {Path(path).name}")

    def resizeEvent(self, e):
        """On window resize, reflow all thumbnail grids (Analysis, Label, Rank) to the new width."""
        super().resizeEvent(e)
        self._reflow_grid()
        self._load_visible_thumbs()
        self._reflow_all_label_panels()
        if hasattr(self, "rank_grid"):
            self._rank_reflow()
            self._rank_load_visible()


def main():
    """CLI entry point: parse args, launch the dark-themed viewer, and auto-load any CSVs given (paired --csv/--png-folder or positional)."""
    import argparse

    ap = argparse.ArgumentParser(description="FOV feature viewer")
    ap.add_argument(
        "--csv",
        action="append",
        default=[],
        help="matrix CSV to auto-load (repeatable; pair with --png-folder)",
    )
    ap.add_argument(
        "--png-folder",
        action="append",
        default=[],
        help="override folder of that matrix's brightfield PNGs (repeatable; paired with "
        "--csv). Omit to use the sibling <stem>_png/ folder next to the CSV.",
    )
    ap.add_argument(
        "csvs",
        nargs="*",
        help="feature CSV(s) to auto-load; images come from sibling <stem>[_<channel>]_png/ "
        "folders next to each CSV",
    )
    args = ap.parse_args()

    if args.csv and args.png_folder and len(args.csv) != len(args.png_folder):
        ap.error(
            f"--csv ({len(args.csv)}) and --png-folder ({len(args.png_folder)}) "
            "must be given the same number of times (paired by order)"
        )

    app = QtWidgets.QApplication(sys.argv[:1])  # keep argv out of Qt's parser
    apply_dark(app)
    win = FeatureViewer()
    win.show()
    if args.csv:  # explicit brightfield-folder override per CSV
        folders = args.png_folder or [None] * len(args.csv)
        win._load_paired(
            list(
                zip(
                    [str(Path(c)) for c in args.csv],
                    [str(Path(f)) if f else None for f in folders],
                )
            )
        )
    elif args.csvs:  # sibling-folder auto-wiring
        win._load_files([str(Path(c)) for c in args.csvs])
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
