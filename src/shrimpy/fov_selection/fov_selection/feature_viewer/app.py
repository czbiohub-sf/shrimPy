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
            shape/direction/param knobs and a Re-rank button; the right column lists the
            FOVs as thumbnails ordered best-first by the resulting score.

Data wiring
  Each feature CSV (fov_summary.csv, or a legacy *_fov_feature_matrix.csv) is one row per
  FOV and must carry a `filename` column.
  data.load_matrices resolves each row to its PNG in the sibling prescan_fov/ (brightfield),
  prescan_mask/, prescan_fluor/ folder next to the CSV (strict filename match; legacy
  <stem>[_<channel>]_png/ folders still open) and stores it as __png / __png_<channel>.

Run:  python -m shrimpy.fov_selection.feature_viewer
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_API", "pyqt6")

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from qtpy import QtCore, QtGui, QtWidgets

matplotlib.use("QtAgg")

from . import data as D
from ._common import (
    DEFAULT_DIR,
    REDUCE_PREFIX,
    THUMB_CACHE_MAX,
    apply_dark,
)
from .analysis_tab import AnalysisTabMixin
from .label_tab import LabelTabMixin
from .rank_tab import RankTabMixin
from .score_map_tab import ScoreMapTabMixin


class FeatureViewer(
    AnalysisTabMixin,
    ScoreMapTabMixin,
    RankTabMixin,
    LabelTabMixin,
    QtWidgets.QMainWindow,
):
    def __init__(self):
        """Build the main window: initialize all state, then assemble the Analysis, Label, and Rank tabs."""
        super().__init__()
        self.setWindowTitle("FOV Feature Viewer")
        # Size to fill the available screen (never larger), so the Analysis-tab settings
        # panel shows all its sections completely; on smaller displays the window still
        # fits on-screen. Cap at a large preferred size and move on-screen.
        screen = QtWidgets.QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else None
        w = min(1900, avail.width()) if avail is not None else 1900
        h = min(1200, avail.height()) if avail is not None else 1200
        self.resize(w, h)
        if avail is not None:
            self.move(avail.left(), avail.top())
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
        # feature -> {"shape","direction","lo","hi","curve_k","weight","enabled"}.
        self.rank_ranges: dict = {}
        self.rank_sort = False  # sort the Analysis FOV panel by `score` (legacy hook)
        # which FOV image channel the thumbnails display; default to the mask overlay so the
        # segmentation is visible on load (falls back to brightfield/first when absent).
        self._channel = "mask"
        self._ready = False

        left_col = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        # Wrap the settings panel in a scroll area so the left column can shrink below the
        # panel's natural width (a horizontal scrollbar appears only when truly too narrow),
        # keeping the whole window able to fit on a small screen.
        settings_scroll = QtWidgets.QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        settings_scroll.setWidget(self._build_settings())
        left_col.addWidget(settings_scroll)
        left_col.addWidget(self._build_center())
        left_col.setStretchFactor(0, 0)
        left_col.setStretchFactor(1, 1)
        left_col.setSizes(
            [540, 660]
        )  # settings tall enough to show all sections; scatter gets the rest

        main = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main.addWidget(left_col)
        main.addWidget(self._build_right())
        main.setStretchFactor(0, 0)  # settings+scatter keep their width; FOV grid takes extra
        main.setStretchFactor(1, 1)
        main.setSizes(
            [1120, 780]
        )  # left holds settings (all 3 columns) + scatter; FOV panel gets the rest

        # all the existing panels live under an "Analysis" tab; "Label" groups the
        # loaded FOVs into one column per goodness class.
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(main, "Analysis")
        self._label_tab_index = self.tabs.addTab(self._build_label_tab(), "Label")
        self._rank_tab_index = self.tabs.addTab(self._build_rank_tab(), "Rank")
        self._map_tab_index = self.tabs.addTab(self._build_score_map_tab(), "Score map")
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
        # colA = sections 1-2, colB = sections 3-4, colC = sections 5+6. Stretch factors
        # (not fixed widths) so all three columns scale PROPORTIONALLY as the settings panel
        # is resized; colA is given the smallest share so sections 1-2 stay relatively narrow.
        for cc, stretch in ((colA, 2), (colB, 4), (colC, 4)):
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

        # --- 5 (removed points) stacked ABOVE 6 (label FOVs) in col C ---
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

        colC.addWidget(g6)  # 5. removed points
        colC.addWidget(g7)  # 6. label FOVs, stacked directly below section 5
        colC.addStretch(1)

        return w

    def _on_tab_changed(self, index):
        """Reflow the Rank or Label tab's lazily-laid-out grids once it becomes visible; caveat: the Label tab is (re)built only the first time it is shown so pending drag edits are not discarded."""
        if index == getattr(self, "_rank_tab_index", -1):
            # built while hidden (viewport width unknown) -> reflow with the real width
            QtCore.QTimer.singleShot(0, self._rank_reflow)
            QtCore.QTimer.singleShot(0, self._rank_load_visible)
            return
        if index == getattr(self, "_map_tab_index", -1):
            # repopulate the pair combos from the currently-checked features, then draw
            self._refresh_score_map_controls()
            QtCore.QTimer.singleShot(0, self._update_score_map)
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
        self._img_aspect_cache = None  # recompute the tile aspect from the new data's PNGs
        self._refresh_channel_combos()  # channel toggles reflect the loaded data's channels
        feats = D.feature_columns(self.df) if self.df is not None else []
        cols = list(self.df.columns) if self.df is not None else []
        # goodness / goodness_probability are labels / model outputs (kept out of the
        # reduction inputs, feature_columns), but the user still wants them selectable as
        # plot axes and color -- expose them alongside the features.
        label_opts = [
            c for c in ("goodness", "goodness_probability") if c in cols and c not in feats
        ]
        axis_opts = feats + self.reduced_cols + label_opts
        # Color-by options: every non-internal column EXCEPT `filename` -- a unique per-FOV
        # string, meaningless as a color and pure clutter. goodness / goodness_probability /
        # dataset / well_col etc. remain available.
        all_cols = [
            c
            for c in cols
            if not c.startswith("__") and not c.endswith("__png") and c != "filename"
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

        # Preferred default axis for a combo when the column is present; otherwise fall back
        # to the best-covered order.
        def _pref_col(name):
            return name if name in axis_opts else None

        preferred = {id(self.cb_y): "nn_um_mean"}
        for cb, dflt in [(self.cb_x, 0), (self.cb_y, 1), (self.cb_z, 2)]:
            cur = cb.currentText()
            cb.clear()
            cb.addItems(axis_opts)
            if cur in axis_opts:  # keep the user's choice across re-runs
                cb.setCurrentText(cur)
                continue
            pref = _pref_col(preferred[id(cb)]) if id(cb) in preferred else None
            if pref is not None:
                cb.setCurrentText(pref)
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
        # strict: comp_cols is always 3 long (built from range(3)), matching the 3 combos.
        for cb, c in zip((self.cb_x, self.cb_y, self.cb_z), comp_cols, strict=True):
            cb.setCurrentText(c)
        self._ready = True
        self.update_plot()  # explicit: combos may be unchanged (already on PCA*), so no signal

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
            # default to the mask overlay when present, else the first available channel
            if "mask" in chans:
                self._channel = "mask"
            else:
                self._channel = chans[0] if chans else "brightfield"
        if self.df is not None and f"__png_{self._channel}" in cols:
            self.df["__png"] = self.df[f"__png_{self._channel}"]
        for combo in (
            getattr(self, "chan_combo", None),
            getattr(self, "rank_chan_combo", None),
            getattr(self, "label_chan_combo", None),
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
        """Switch the displayed FOV channel EVERYWHERE: repoint __png, sync the channel combo
        on all three tabs, and reload the Analysis, Rank, and Label thumbnail grids so the
        choice is unified (changing it on one tab applies to the others)."""
        if not ch or self.df is None or f"__png_{ch}" not in self.df.columns:
            return
        self._channel = ch
        self.df["__png"] = self.df[f"__png_{ch}"]
        for combo in (
            getattr(self, "chan_combo", None),
            getattr(self, "rank_chan_combo", None),
            getattr(self, "label_chan_combo", None),
        ):
            if combo is not None and combo.currentText() != ch:
                combo.blockSignals(True)
                combo.setCurrentText(ch)
                combo.blockSignals(False)
        if hasattr(self, "grid_host"):
            self._rebuild_grid()
        if getattr(self, "_rank_order", None):
            self._rank_rebuild_grid()
        self._relabel_label_channel(ch)

    def _relabel_label_channel(self, ch):
        """Repoint the Label-tab thumbnails to channel ``ch`` IN PLACE (so pending, unsaved
        drag edits are preserved -- unlike a full _refresh_label_tab rebuild), then redecode
        the visible ones."""
        col = f"__png_{ch}"
        if not getattr(self, "_label_cols", None) or col not in self.df.columns:
            return
        pngs = self.df[col]
        for info in self._label_cols:
            for it in info["items"]:
                pos = it[3]
                it[1] = pngs.iloc[pos] if 0 <= pos < len(pngs) else ""  # new channel's png
                it[2] = False  # mark not-loaded so it re-decodes
                it[0].setPixmap(QtGui.QPixmap())  # drop the old channel's image
                it[0].setText("…")
            self._load_visible_label_thumbs(info)

    def _image_aspect(self) -> float:
        """Representative FOV image aspect ratio (width / height).

        Lets thumbnail tiles match the image shape instead of a fixed square (wide FOVs
        otherwise get black side-margins). Read once from the first available PNG header and
        cached; 1.0 (square) when unknown. The cache is cleared on every data change (see
        :meth:`_refresh_axis_selectors`)."""
        a = getattr(self, "_img_aspect_cache", None)
        if a is not None:
            return a
        a = 1.0
        if self.df is not None and "__png" in self.df.columns:
            for p in self.df["__png"]:
                if p and Path(p).exists():
                    sz = QtGui.QImageReader(str(p)).size()
                    if sz.isValid() and sz.width() > 0 and sz.height() > 0:
                        a = sz.width() / sz.height()
                        break
        self._img_aspect_cache = a
        return a

    def _tile_dims(self, size: int) -> tuple[int, int]:
        """(width, height) of a thumbnail tile at scale `size`, matching the image aspect
        (longer side == ``size``) so wide FOVs render without black side-margins. Mirrors the
        KeepAspectRatio scaling :meth:`_load_thumb` applies to the pixmap, so the tile fits it
        exactly."""
        a = self._image_aspect()
        if a >= 1.0:
            return int(size), max(1, int(round(size / a)))
        return max(1, int(round(size * a))), int(size)

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

    def _group_positions_by_well(self, positions):
        """Group df positions by (well_row, well_col), preserving each well's incoming order,
        and return ``(per_well, wells)`` with ``wells`` in plate order (row, then numeric col).
        Falls back to a single ``None`` well ("All FOVs") when the data has no well columns."""
        df = self.df
        per_well: dict = {}
        if "well_row" in df.columns and "well_col" in df.columns:
            wr = df.columns.get_loc("well_row")
            wc = df.columns.get_loc("well_col")
            for pos in positions:
                per_well.setdefault((df.iat[pos, wr], df.iat[pos, wc]), []).append(pos)

            def _well_sort_key(k):
                row, col = k
                try:
                    return (str(row), 0, int(col))
                except (TypeError, ValueError):
                    return (str(row), 1, str(col))

            return per_well, sorted(per_well, key=_well_sort_key)
        per_well[None] = list(positions)
        return per_well, [None]

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
        "--csv). Omit to use the sibling prescan_fov/ folder next to the CSV.",
    )
    ap.add_argument(
        "csvs",
        nargs="*",
        help="feature CSV(s) to auto-load; images come from sibling prescan_fov/ "
        "(prescan_mask/, prescan_fluor/) folders next to each CSV",
    )
    ap.add_argument(
        "--start-tab",
        choices=["analysis", "label", "rank", "map"],
        default="analysis",
        help="which tab to open on launch (calibration pre-scans open 'rank')",
    )
    ap.add_argument(
        "--rank-profile",
        default=None,
        help="YAML/JSON desirability profile (a bare features mapping or a full model "
        "dict) to seed the Rank tab on launch, merged over the data-seeded defaults",
    )
    ap.add_argument(
        "--rank-profile-json",
        default=None,
        help="same as --rank-profile but the profile is passed inline as a JSON string "
        "rather than a file path (a calibration pre-scan passes the config's "
        "fov_selection.model here, so no profile file is written to disk)",
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
                # strict: ap.error above already rejects mismatched --csv / --png-folder
                # counts, and the fallback is one None per CSV, so the two are equal here.
                zip(
                    [str(Path(c)) for c in args.csv],
                    [str(Path(f)) if f else None for f in folders],
                    strict=True,
                )
            )
        )
    elif args.csvs:  # sibling-folder auto-wiring
        win._load_files([str(Path(c)) for c in args.csvs])
    # Seed the Rank tab from a profile (after loading, so data features are merged in).
    if args.rank_profile:
        win._apply_rank_profile(str(Path(args.rank_profile)))
    elif args.rank_profile_json:
        import json

        try:
            cfg = json.loads(args.rank_profile_json)
        except Exception:  # noqa: BLE001
            cfg = None
        if cfg:
            win._apply_rank_profile_cfg(cfg, source="config model")
    # Open the requested tab (after loading, so the Rank tab has data to lay out).
    tab_index = {
        "analysis": 0,
        "label": win._label_tab_index,
        "rank": win._rank_tab_index,
        "map": win._map_tab_index,
    }[args.start_tab]
    win.tabs.setCurrentIndex(tab_index)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
