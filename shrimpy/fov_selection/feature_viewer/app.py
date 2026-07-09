"""
FOV Feature Viewer -- a Qt GUI to explore FOV-level features, cluster them, and
find features/thresholds that separate goodness classes.

Layout (two columns):
  LEFT  column = settings (top) + interactive scatter (bottom), in one wide column.
  RIGHT column = grid of the selected FOV PNGs (scrollable, with a size slider).

Scatter: 2D pan (toolbar) + scroll-to-zoom; 3D drag-rotate + scroll-to-zoom; Home
button; Lasso toggle (works in 2D and 3D). Click a point or lasso a region -> the
right grid updates. Dim reduction (PCA/t-SNE/UMAP) always computes 3 components.

Wiring: each *_fov_feature_matrix.csv is one row per FOV; data.load_matrices resolves
each row to its composite image under fov_composites/ (by FOV identity) as __png.

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

from . import data as D

DEFAULT_DIR = "/hpc/projects/comp.micro/microscope_dev/smart_fov_selection/fov_selection_output/fov_features"
REDUCE_PREFIX = {"PCA": "PCA", "t-SNE": "TSNE", "UMAP": "UMAP"}
MAX_THUMBS = 200  # cap thumbnails rendered at once (refine the selection for more)
DETAIL_SKIP = {"__src", "__png", "png", "__dataset"}
THUMB_CACHE_MAX = 1000  # LRU cap on decoded thumbnails (path,size) -> QPixmap
FOCUS_COLOR = "#ffd24d"  # highlight color for the clicked point / thumbnail
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
    "neutral (0)": "#bdbdbd",  # grey
    "bad (-1)": "#e53935",  # red
    "unlabeled": "#5b8def",  # blue
}
# Label tab columns, in fixed reading order. `value` is the on-disk goodness code
# (None == unlabeled / NaN). A column is shown only if the loaded data has ≥1 FOV in it.
GOODNESS_LABEL_ORDER = [
    ("Good", 1.0, "#4caf50"),
    ("Neutral", 0.0, "#bdbdbd"),
    ("Bad", -1.0, "#e53935"),
    ("Nan", None, "#5b8def"),
]
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
        super().__init__()
        self._df = None
        self._rows: list[int] = []
        self._fields: list[str] = []
        self._colpos: list[int] = []

    def set_data(self, df, rows, fields):
        self.beginResetModel()
        self._df = df
        self._rows = rows
        self._fields = fields
        self._colpos = [df.columns.get_loc(f) for f in fields] if df is not None else []
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if (parent.isValid() or self._df is None) else len(self._rows)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if (parent.isValid() or self._df is None) else len(self._fields)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole or not index.isValid():
            return None
        v = self._df.iat[self._rows[index.row()], self._colpos[index.column()]]
        if isinstance(v, (float, np.floating)):
            return "NaN" if v != v else f"{v:.4g}"
        return str(v)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return None
        return self._fields[section] if orientation == QtCore.Qt.Horizontal else None


class _ThumbLabel(QtWidgets.QLabel):
    """A FOV thumbnail that can be dragged between Label-tab panels. Carries its df
    position as drag payload; the panel it lands on decides the new goodness class."""

    def __init__(self, pos: int):
        super().__init__()
        self._pos = pos
        self._press = None

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:
            self._press = e.pos()

    def mouseMoveEvent(self, e):
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
        super().__init__()
        self._viewer = viewer
        self._value = value  # goodness code this panel represents (NaN for "Nan")
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasText():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasText():
            e.acceptProposedAction()

    def dropEvent(self, e):
        try:
            pos = int(e.mimeData().text())
        except (TypeError, ValueError):
            return
        self._viewer._on_label_drop(pos, self._value)
        e.acceptProposedAction()


class FeatureViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FOV Feature Viewer")
        self.resize(1700, 1000)
        self.df: pd.DataFrame | None = None
        self.filt = np.array([], int)
        self.plot_pos = np.array([], int)
        self.sel_pos: list[int] = []
        self.reduced_cols: list[str] = []
        self.col_filters: dict[str, set] = {}  # metadata column -> kept value set (str)
        self.thresholds: dict[str, tuple] = {}  # feature column -> (lo, hi) kept range
        self.lasso_sel = None  # lasso filter: positions, or None = show all
        self.hidden = np.array([], bool)  # True = "removed" (invisible, not deleted)
        self.remove_history: list[list[int]] = []
        self.focus_pos = None  # df position of the highlighted FOV/point
        self._focus_marker = None  # scatter ring artist
        self._focus_label = None  # highlighted thumbnail
        self._picked = False  # was a scatter point hit on this click?
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
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)
        self._ready = True

    # ------------------------------------------------------------- settings
    @staticmethod
    def _btn_row(*buttons):
        """HBox of natural-width buttons, left-aligned (not stretched)."""
        h = QtWidgets.QHBoxLayout()
        for b in buttons:
            h.addWidget(b)
        h.addStretch(1)
        return h

    @staticmethod
    def _as_entry(spin):
        """Make a spin box behave like a plain typed entry: no arrows, no wheel."""
        spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        spin.setKeyboardTracking(False)
        spin.wheelEvent = lambda e: e.ignore()
        return spin

    def _build_settings(self):
        w = QtWidgets.QWidget()
        cols = QtWidgets.QHBoxLayout(w)
        colA = QtWidgets.QVBoxLayout()
        colB = QtWidgets.QVBoxLayout()
        colC = QtWidgets.QVBoxLayout()
        # colA = sections 1-2, colB = sections 3-4, colC = section 5 + the 6|7 row
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
        gv.addLayout(self._btn_row(load, rmds, clr))
        self.ds_list = QtWidgets.QListWidget()
        self.ds_list.setMaximumHeight(80)
        gv.addWidget(self.ds_list)
        colA.addWidget(g)

        # --- 2. filter (col A): pick a column, keep the selected values ---
        g = QtWidgets.QGroupBox("2. Filter FOVs  (keep selected values)")
        gv = QtWidgets.QVBoxLayout(g)
        self.filter_col = QtWidgets.QComboBox()
        self.filter_col.currentTextChanged.connect(self._on_filter_col)
        gv.addWidget(self.filter_col)
        self.f_vals = QtWidgets.QListWidget()
        self.f_vals.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.f_vals.setMaximumHeight(150)
        gv.addWidget(self.f_vals)
        ap = QtWidgets.QPushButton("Apply")
        ap.clicked.connect(self.on_apply_filter)
        ca = QtWidgets.QPushButton("Cancel all")
        ca.clicked.connect(self.on_cancel_filters)
        gv.addLayout(self._btn_row(ap, ca))
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
        self._as_entry(self.sp_perp)
        self.sp_nn = QtWidgets.QSpinBox()
        self.sp_nn.setRange(2, 200)
        self.sp_nn.setValue(15)
        self._as_entry(self.sp_nn)
        run = QtWidgets.QPushButton("Run on filtered FOVs")
        run.clicked.connect(self.run_reduction)
        form.addRow("Method", self.cb_method)
        self.lbl_perp = QtWidgets.QLabel("perplexity")  # t-SNE only
        self.lbl_nn = QtWidgets.QLabel("n_neighbors")  # UMAP only
        form.addRow(self.lbl_perp, self.sp_perp)
        form.addRow(self.lbl_nn, self.sp_nn)
        form.addRow(self._btn_row(run))
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

        # --- 5. feature thresholds (col C) ---
        g = QtWidgets.QGroupBox("5. Feature thresholds  (keep in range)")
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
            self._as_entry(s)
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
        gv.addLayout(self._btn_row(adt, rmt, clt))
        self.thr_list = QtWidgets.QListWidget()
        self.thr_list.setMaximumHeight(90)
        gv.addWidget(self.thr_list)
        colC.addWidget(g)

        # --- 6 + 7 sit side by side in one narrower row under section 5 ---
        # --- 6. removed-points history ---
        g6 = QtWidgets.QGroupBox("6. Removed points")
        gv6 = QtWidgets.QVBoxLayout(g6)
        self.history_list = QtWidgets.QListWidget()
        self.history_list.setMaximumHeight(120)
        gv6.addWidget(self.history_list)
        rs = QtWidgets.QPushButton("Restore selected")
        rs.clicked.connect(self.on_restore_selected)
        ra = QtWidgets.QPushButton("Restore all")
        ra.clicked.connect(self.on_restore_all)
        gv6.addLayout(self._btn_row(rs, ra))
        self.hidden_count = QtWidgets.QLabel("0 points hidden")
        gv6.addWidget(self.hidden_count)

        # --- 7. label selected FOVs (goodness) ---
        g7 = QtWidgets.QGroupBox("7. Label FOVs  (goodness)")
        gv7 = QtWidgets.QVBoxLayout(g7)
        self.label_value = QtWidgets.QComboBox()
        self.label_value.addItem("Good (1)", 1)
        self.label_value.addItem("Neutral (0)", 0)
        self.label_value.addItem("Bad (-1)", -1)
        self.label_value.addItem("Unlabeled (NaN)", None)  # reset label to nothing
        gv7.addWidget(self.label_value)
        setb = QtWidgets.QPushButton("Set")
        setb.clicked.connect(self.on_set_goodness)
        gv7.addLayout(self._btn_row(setb))
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
        w = QtWidgets.QWidget()
        w.setMinimumWidth(440)
        v = QtWidgets.QVBoxLayout(w)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(QtWidgets.QLabel("<b>Selected FOVs</b>"))
        head.addStretch(1)
        head.addWidget(QtWidgets.QLabel("thumb size"))
        self.size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.size_slider.setRange(60, 800)
        self.size_slider.setValue(150)
        self.size_slider.setFixedWidth(140)
        self.size_slider.sliderReleased.connect(self._rebuild_grid)
        head.addWidget(self.size_slider)
        v.addLayout(head)
        self.grid_info = QtWidgets.QLabel("—")
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
        self.label_info = QtWidgets.QLabel("—")
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
        self._regrid_label_panel(target)  # reparents the moved thumbnail first
        self._regrid_label_panel(source)
        self._update_panel_title(source)
        self._update_panel_title(target)
        self._load_visible_label_thumbs(target)
        self._update_label_save_state()

    def _regrid_label_panel(self, info):
        """Re-place all of the panel's thumbnails after its item set changed."""
        grid = info["grid"]
        while grid.count():
            grid.takeAt(0)  # detach layout items; widgets kept alive
        ncols = max(1, info["ncols"] or 1)
        for i, it in enumerate(info["items"]):
            grid.addWidget(it[0], i // ncols, i % ncols)

    def _update_label_save_state(self):
        if not hasattr(self, "label_save_btn"):
            return
        n = len(getattr(self, "_label_changes", {}))
        self.label_save_btn.setEnabled(n > 0)
        self.label_save_btn.setText(f"Save ({n})" if n else "Save")

    def on_save_labels(self):
        if self.df is None or not self._label_changes:
            return
        if "goodness" not in self.df.columns:
            self.df["goodness"] = np.nan
        gi = self.df.columns.get_loc("goodness")
        for pos, val in self._label_changes.items():  # commit to the in-memory table
            self.df.iat[pos, gi] = val
        saved, matched = self._persist_changes(self._label_changes)
        n = len(self._label_changes)
        self._label_changes = {}
        self._populate_details()
        if self.cb_color.currentText() == "goodness":
            self.update_plot()
        # drop remembered empty-class panels: after saving, a class shows only if it
        # still has FOVs, so classes emptied by these moves disappear.
        self._extra_classes = set()
        self._refresh_label_tab()  # rebuild panels from the now-saved labels
        self.label_info.setText(
            f"saved {n} change(s) → wrote {matched} row(s) to {saved} CSV(s)"
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
        for info in self._label_cols:
            self._reflow_label_panel(info)
        self._load_all_label_thumbs()

    def _load_all_label_thumbs(self):
        for info in self._label_cols:
            self._load_visible_label_thumbs(info)

    @staticmethod
    def _make_separator():
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
        start = DEFAULT_DIR if Path(DEFAULT_DIR).exists() else str(Path.home())
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Load feature CSV(s)", start, "CSV (*.csv)"
        )
        if not files:
            return
        df = D.load_matrices(files)
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
        self.df = None
        self.filt = np.array([], int)
        self.reduced_cols = []
        self.col_filters = {}
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
        self.hidden = (
            np.zeros(len(self.df), bool) if self.df is not None else np.array([], bool)
        )
        self.remove_history = []
        if hasattr(self, "history_list"):
            self.history_list.clear()
        self._update_hidden_count()

    def _effective(self):
        """Filtered AND not-hidden positions -- the points actually plotted/reduced."""
        if self.df is None or len(self.filt) == 0:
            return np.array([], int)
        return self.filt[~self.hidden[self.filt]]

    def _update_hidden_count(self):
        n = int(self.hidden.sum()) if self.hidden.size else 0
        if hasattr(self, "hidden_count"):
            self.hidden_count.setText(f"{n} points hidden")

    def _viewing_reduced(self):
        axes = [self.cb_x.currentText(), self.cb_y.currentText()]
        if self.mode.currentText() == "3D":
            axes.append(self.cb_z.currentText())
        return any(D.REDUCED_RE.match(a) for a in axes if a)

    def _after_visibility_change(self):
        # if viewing a reduced embedding, always re-fit it on the survivors
        if self._viewing_reduced() and len(self._effective()) >= 3:
            self._rerun_current_reduction()
        else:
            self.update_plot()

    def _rerun_current_reduction(self):
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
        if self.df is None:
            return
        # remove only an EXPLICIT selection -- a lasso subset, or a highlighted point;
        # never the default "all visible" set
        if self.lasso_sel:
            pos = [p for p in self.lasso_sel if not self.hidden[p]]
        elif self.focus_pos is not None and not self.hidden[self.focus_pos]:
            pos = [self.focus_pos]
        else:
            pos = []
        if not pos:
            return
        self.hidden[pos] = True
        self.remove_history.append(list(pos))
        self.history_list.addItem(f"#{len(self.remove_history)}: removed {len(pos)} pts")
        self._update_hidden_count()
        self._after_visibility_change()  # recompute reduction (if shown) + replot scatter
        self._reset_to_default_view()  # panel/details -> all visible FOVs

    def on_restore_selected(self):
        r = self.history_list.currentRow()
        if r < 0 or r >= len(self.remove_history):
            return
        pos = self.remove_history.pop(r)
        self.history_list.takeItem(r)
        self.hidden[pos] = False
        self._update_hidden_count()
        self._after_visibility_change()
        self._reset_to_default_view()

    def on_restore_all(self):
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
        return list(self._effective())

    def on_set_goodness(self):
        if self.df is None:
            self.label_status.setText("no data loaded")
            return
        positions = self._label_targets()
        if not positions:
            self.label_status.setText("nothing selected")
            return
        raw = self.label_value.currentData()
        value = float("nan") if raw is None else float(raw)  # "Unlabeled" -> NaN
        if "goodness" not in self.df.columns:
            self.df["goodness"] = np.nan
        gi = self.df.columns.get_loc("goodness")
        self.df.iloc[positions, gi] = value  # update in memory
        saved, matched = self._persist_goodness(positions, value)  # write to disk
        shown = "NaN" if value != value else str(int(value))
        self.label_status.setText(
            f"set {len(positions)} FOV(s) → {shown}; wrote {matched} row(s) to {saved} CSV(s)"
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
        self._ready = False
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
        # filterable columns: categorical / low-cardinality (value-pick filter);
        # datasets are managed in section 1, so "dataset" is excluded here.
        fcols = []
        if self.df is not None:
            for c in all_cols:
                if c == "dataset":
                    continue
                if (
                    2 <= self.df[c].nunique(dropna=False) <= 50
                ):  # categorical / low-cardinality
                    fcols.append(c)
        self.filter_col.clear()
        self.filter_col.addItems(fcols)
        self.thr_col.clear()
        self.thr_col.addItems(axis_opts)  # numeric features + reduced comps
        self.cb_z.setEnabled(self.mode.currentText() == "3D")
        self._ready = True
        self._on_filter_col(self.filter_col.currentText())
        self._on_thr_col(self.thr_col.currentText())

    # ----- filters: pick a column, keep the selected values -----
    def _on_filter_col(self, col):
        """Show the column's values; pre-select the active keep-set (or all)."""
        if self.df is None or not col or col not in self.df.columns:
            self.f_vals.clear()
            return
        self.f_vals.blockSignals(True)
        self.f_vals.clear()
        kept = self.col_filters.get(col)
        for v in sorted(self.df[col].fillna("NaN").astype(str).unique(), key=str):
            it = QtWidgets.QListWidgetItem(v)
            self.f_vals.addItem(it)
            it.setSelected(kept is None or v in kept)
        self.f_vals.blockSignals(False)

    def on_apply_filter(self):
        """Keep only the selected values for the current column (AND across columns)."""
        col = self.filter_col.currentText()
        if self.df is None or not col:
            return
        sel = {i.text() for i in self.f_vals.selectedItems()}
        allv = {self.f_vals.item(i).text() for i in range(self.f_vals.count())}
        if sel and sel != allv:
            self.col_filters[col] = sel
        else:  # all (or none) selected -> no restriction on this column
            self.col_filters.pop(col, None)
        self.apply_filters()

    def on_cancel_filters(self):
        self.col_filters = {}
        self.apply_filters()
        self._on_filter_col(self.filter_col.currentText())

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
        self.thr_list.clear()
        for feat, (lo, hi) in self.thresholds.items():
            self.thr_list.addItem(f"{feat} ∈ [{lo:g}, {hi:g}]")

    def on_add_threshold(self):
        feat = self.thr_col.currentText()
        if self.df is None or not feat:
            return
        self.thresholds[feat] = (self.thr_lo.value(), self.thr_hi.value())
        self._refresh_thr_list()
        self.apply_filters()

    def on_remove_threshold(self):
        r = self.thr_list.currentRow()
        if r < 0:
            return
        feat = list(self.thresholds)[r]
        self.thresholds.pop(feat, None)
        self._refresh_thr_list()
        self.apply_filters()

    def on_clear_thresholds(self):
        self.thresholds = {}
        self._refresh_thr_list()
        self.apply_filters()

    def apply_filters(self):
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
        for col, kept in self.col_filters.items():
            if col in self.df.columns:
                mask &= self.df[col].fillna("NaN").astype(str).isin(kept).to_numpy()
        for feat, (lo, hi) in self.thresholds.items():
            if feat in self.df.columns:
                v = self.df[feat].to_numpy(float)
                mask &= (v >= lo) & (v <= hi)  # NaN comparisons are False -> excluded
        self.filt = np.where(mask)[0]
        self.lasso_sel = None  # changing the filters resets the lasso filter
        if self.focus_pos is not None and self.focus_pos not in set(self.filt.tolist()):
            self._clear_focus()
        parts = [f"{c} ∈ {{{','.join(sorted(v))}}}" for c, v in self.col_filters.items()]
        parts += [f"{f} ∈ [{lo:g},{hi:g}]" for f, (lo, hi) in self.thresholds.items()]
        desc = "; ".join(parts) if parts else "no filter"
        self.filter_summary.setText(f"{len(self.filt)} / {len(self.df)} FOVs  ·  {desc}")
        # re-fit a reduced embedding on the new survivors (else newly-included FOVs
        # have NaN reduced coords and the scatter goes blank); raw views just replot
        self._refresh_panel()
        self._after_visibility_change()

    def _shown(self):
        """Visible positions shown in the panel: lasso subset if active, else all
        filtered -- always excluding removed (hidden) FOVs."""
        if self.lasso_sel is not None:
            return [p for p in self.lasso_sel if not self.hidden[p]]
        return list(self._effective())

    def _refresh_panel(self):
        self.set_selection(self._shown())

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
        eff = self._effective()
        if self.df is None or len(eff) < 3:
            self.reduce_status.setText("need >=3 visible FOVs")
            return
        method = self.cb_method.currentText()
        feats = D.feature_columns(self.df)
        X = self.df.iloc[eff][feats].to_numpy(float)
        self.reduce_status.setText(f"running {method} on {len(eff)} FOVs…")
        QtWidgets.QApplication.processEvents()
        try:
            emb = D.run_reduction(
                X, method, perplexity=self.sp_perp.value(), n_neighbors=self.sp_nn.value()
            )
        except Exception as e:  # noqa: BLE001
            self.reduce_status.setText(f"error: {e}")
            return
        cols = [f"{REDUCE_PREFIX[method]}{i + 1}" for i in range(3)]
        for c in cols:
            if c not in self.df.columns:
                self.df[c] = np.nan
            if c not in self.reduced_cols:
                self.reduced_cols.append(c)
        self.df.loc[self.df.index[eff], cols] = emb
        self.reduce_status.setText(f"{method} done -> axes {', '.join(cols)} available")
        self._ready = False
        self._populate_columns()
        for cb, c in zip((self.cb_x, self.cb_y, self.cb_z), cols):
            cb.setCurrentText(c)
        self._ready = True
        self.update_plot()  # explicit: combos may be unchanged (already on PCA*), so no signal

    # =============================================================== plotting
    def _axis_values(self, col):
        return self.df.iloc[self.plot_pos][col].to_numpy(float)

    def update_plot(self, *_):
        if not self._ready or self.df is None:
            return
        self.plot_pos = self._effective()
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
        if self.ax is None:
            return
        if self.mode.currentText() == "3D":
            self.ax.view_init(elev=30, azim=-60)
            self.ax.autoscale()
        else:
            self.toolbar.home()
        self.canvas.draw_idle()

    def _on_scroll(self, event):
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
        if event.artist is not self.scatter or self.lasso_btn.isChecked():
            return
        ind = event.ind
        if ind is None or len(ind) == 0:
            return
        self._picked = True  # a point was hit -> don't treat this click as "empty"
        pos = int(self.plot_pos[int(ind[0])])
        if self.focus_pos == pos:  # click the highlighted point again -> clear highlight
            self._clear_focus()
            return
        # clicking a point cancels the lasso filter -> show ALL visible FOVs + details,
        # then highlight the clicked point
        if self.lasso_sel is not None:
            self.lasso_sel = None
            self._clear_lasso_line()
        self._refresh_panel()
        self._set_focus(pos, scroll_grid=True)

    def toggle_lasso(self, on):
        self.lasso_btn.setText("Lasso ON (drag to select)" if on else "Enable lasso")
        if on and self.ax is not None and hasattr(self.ax, "disable_mouse_rotation"):
            self.ax.disable_mouse_rotation()
        elif (not on) and self.ax is not None and hasattr(self.ax, "mouse_init"):
            self.ax.mouse_init()

    def _clear_lasso_line(self):
        if self._lasso_line is not None:
            try:
                self._lasso_line.remove()
            except Exception:  # noqa: BLE001  (axis may already be gone)
                pass
            self._lasso_line = None
            self.canvas.draw_idle()

    def _lasso_press(self, event):
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
        if not self._picked:  # clicked empty space -> cancel lasso + highlight, show all
            self._clear_focus()
            self.lasso_sel = None
            self._clear_lasso_line()
            self._refresh_panel()
        self._picked = False  # reset for the next click

    def _lasso_move(self, event):
        if not self._lasso_xy or event.x is None:
            return
        self._lasso_xy.append((event.x, event.y))
        arr = np.array(self._lasso_xy)
        self._lasso_line.set_data(arr[:, 0], arr[:, 1])
        self.canvas.draw_idle()

    def _lasso_release(self, event):
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
            inside = MplPath(verts).contains_points(self._points_display())
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

    def _points_display(self):
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
        fn = row.get("filename", None)
        if isinstance(fn, str) and fn:
            return fn
        return (
            f"{row.get('well_row', '')}/{row.get('well_col', '')}/{row.get('fov', '')} "
            f"t{row.get('timepoint', '')}"
        )

    def set_selection(self, positions):
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
            c for c in self.df.columns if c not in DETAIL_SKIP and not c.endswith("__png")
        ]
        self._detail_rows = list(self.sel_pos)  # virtualized: no per-row cost
        self.details_model.set_data(self.df, self._detail_rows, fields)

    def _focus_details_row(self, pos):
        if pos in self._detail_rows:
            r = self._detail_rows.index(pos)
            self.details.selectRow(r)
            self.details.scrollTo(
                self.details_model.index(r, 0), QtWidgets.QAbstractItemView.PositionAtCenter
            )
        else:
            self.details.clearSelection()

    def _on_details_click(self, row):
        if 0 <= row < len(self._detail_rows):
            self._set_focus(self._detail_rows[row], scroll_grid=True)

    def _load_thumb(self, path, size):
        key = (path, size)
        pm = self._thumb_cache.get(key)
        if pm is not None:  # cache hit -> no disk decode
            self._thumb_cache.move_to_end(key)
            return pm
        reader = QtGui.QImageReader(path)
        sz = reader.size()
        if sz.isValid() and sz.width() > 0:
            reader.setScaledSize(sz.scaled(size, size, QtCore.Qt.KeepAspectRatio))
        pm = QtGui.QPixmap.fromImage(reader.read())
        self._thumb_cache[key] = pm
        if len(self._thumb_cache) > THUMB_CACHE_MAX:
            self._thumb_cache.popitem(last=False)  # evict least-recently-used
        return pm

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
            f"  ({len(shown)} of {len(self.sel_pos)}, sampled across datasets — refine selection)"
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
            lab.setStyleSheet(THUMB_QSS)
            lab.setCursor(QtCore.Qt.PointingHandCursor)
            lab.setToolTip(f"{row.get('__dataset', '')}\n{self._row_id(row)}")
            lab.setText("…")
            lab.mousePressEvent = lambda e, p=int(pos): self._set_focus(p, scroll_grid=False)
            self._thumb_items.append([lab, png, False, int(pos)])
        self._cur_ncols = 0
        self._reflow()
        QtCore.QTimer.singleShot(0, self._load_visible_thumbs)  # after layout settles
        if self.focus_pos is not None:  # re-apply highlight
            self._highlight_thumb(self.focus_pos, scroll_grid=False)

    def _reflow(self):
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
        self.focus_pos = None
        if self._focus_marker is not None:
            try:
                self._focus_marker.remove()
            except Exception:  # noqa: BLE001
                pass
            self._focus_marker = None
            self.canvas.draw_idle()
        if self._focus_label is not None:
            self._focus_label.setStyleSheet(THUMB_QSS)
            self._focus_label = None
        self.details.clearSelection()  # keep the table (it lists the whole selection)

    def _point_coords(self, pos):
        cols = [self.cb_x.currentText(), self.cb_y.currentText()]
        if self.mode.currentText() == "3D":
            cols.append(self.cb_z.currentText())
        return [float(self.df.iloc[pos][c]) for c in cols]

    def _highlight_scatter(self, pos):
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
        if self._focus_label is not None:
            self._focus_label.setStyleSheet(THUMB_QSS)
            self._focus_label = None
        for it in self._thumb_items:
            if it[3] == pos:
                it[0].setStyleSheet(THUMB_FOCUS_QSS)
                self._focus_label = it[0]
                if scroll_grid:
                    self.scroll.ensureWidgetVisible(it[0])
                    self._load_visible_thumbs()
                break

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._reflow()
        self._load_visible_thumbs()
        self._reflow_all_label_panels()


def main():
    app = QtWidgets.QApplication(sys.argv)
    apply_dark(app)
    win = FeatureViewer()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
