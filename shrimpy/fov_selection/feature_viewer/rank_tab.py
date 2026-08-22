"""Rank tab: tune the DesirabilityModel (per-feature curves) and rank FOVs by score."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy import QtCore, QtGui, QtWidgets

from shrimpy.fov_selection import fov_model as FM

from . import data as D
from ._common import (
    PROFILE_DIR,
    RANK_CLICK_COLOR,
    RANK_TITLE_COLOR,
    RCOL_DIR,
    RCOL_FEATURE,
    RCOL_PARAMS,
    RCOL_SHAPE,
    RCOL_WEIGHT,
    THUMB_QSS,
    _border_qss,
    _feature_to_internal,
    _internal_to_feature,
    _ParamCell,
    goodness_border_qss,
)


class RankTabMixin:
    # ============================================================== rank tab
    def _build_rank_tab(self):
        """Tune the production DesirabilityModel: per-feature shape + direction + params.
        LEFT = feature-value histograms with the desirability curve overlaid (dashed) plus a
        table of the shape/direction/param knobs and a Re-rank button.
        RIGHT = the loaded FOVs as thumbnails ordered best-first by the resulting score."""
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # ---------------------------------------- LEFT: parameter tuning
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("<b>Set score function for each feature</b>"))
        bar.addStretch(1)
        b_load = QtWidgets.QPushButton("Load…")
        b_load.clicked.connect(self._on_rank_load)
        b_save = QtWidgets.QPushButton("Save…")
        b_save.clicked.connect(self._on_rank_save)
        for b in (b_load, b_save):
            bar.addWidget(b)
        lv.addLayout(bar)

        # top: per-feature histograms with the desirability curve as a dashed overlay.
        # Kept narrow (small default width + a modest left splitter share below) so the left
        # tuning panel doesn't crowd out the ranked-FOV grid on the right.
        self.rank_fig = Figure(figsize=(3, 6), facecolor="#3c3c3c")
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
        table_help = QtWidgets.QLabel("Check a feature to include it in the score.")
        table_help.setWordWrap(True)
        lv.addWidget(table_help)
        self.rank_table = QtWidgets.QTableWidget(0, 6)
        self.rank_table.setHorizontalHeaderLabels(
            [
                "✓ feature",
                "direction",
                "shape",
                "param 1",
                "param 2",
                "weight",
            ]
        )
        hh = self.rank_table.horizontalHeader()
        # Feature name fits its longest value; direction / shape / weight fit their widgets;
        # the two parameter-entry columns stretch to take all the remaining width, so the
        # spin boxes are roomy instead of the feature name hogging the table.
        hh.setSectionResizeMode(RCOL_FEATURE, QtWidgets.QHeaderView.ResizeToContents)
        for c in (RCOL_DIR, RCOL_SHAPE, RCOL_WEIGHT):
            hh.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        for c in RCOL_PARAMS:
            hh.setSectionResizeMode(c, QtWidgets.QHeaderView.Stretch)
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
        # how the per-feature scores combine into the final score
        act.addWidget(QtWidgets.QLabel("combine"))
        self.rank_agg_combo = QtWidgets.QComboBox()
        self.rank_agg_combo.addItems(
            list(FM.DesirabilityModel.AGGREGATIONS)
        )  # sum/product/gaussian
        # Start on the same rule an unset `aggregation` gets during acquisition, so what you
        # tune here is what runs.
        self.rank_agg_combo.setCurrentText(FM.DesirabilityModel.DEFAULT_AGGREGATION)
        self.rank_agg_combo.setToolTip(
            "sum: weighted mean (compensatory)\n"
            "product: weighted geometric mean (one weak feature vetoes)\n"
            "gaussian: joint N-D gaussian (strongest veto)"
        )
        self.rank_agg_combo.currentTextChanged.connect(lambda *_: self._rerank())
        act.addWidget(self.rank_agg_combo)
        act.addStretch(1)
        write_scores = QtWidgets.QPushButton("Write proba/rank to CSV")
        write_scores.setToolTip(
            "Write the current ranking's score (as `proba`) and best-first `rank` back to "
            "each loaded FOV's source CSV, so the calibration matrix carries the tuned "
            "selection. Re-ranks first if needed."
        )
        write_scores.clicked.connect(self._on_rank_write_scores)
        act.addWidget(write_scores)
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
        # The ranked FOVs are grouped into one section per well (a header + a wrapping grid),
        # stacked vertically; scrolling flows through the well sections in order.
        self.rank_sections_layout = QtWidgets.QVBoxLayout(self.rank_grid_host)
        self.rank_sections_layout.setSpacing(4)
        self.rank_sections_layout.setContentsMargins(0, 0, 0, 0)
        self.rank_sections_layout.setAlignment(QtCore.Qt.AlignTop)
        self.rank_scroll.setWidget(self.rank_grid_host)
        self.rank_scroll.verticalScrollBar().valueChanged.connect(self._rank_load_visible)
        rv.addWidget(self.rank_scroll, 1)
        # flat list of all tiles [label, png, loaded, pos, score, caption] for lazy decode
        self._rank_thumb_items: list[list] = []
        self._rank_sections: list[dict] = []  # per-well: {"grid": QGridLayout, "items": [...]}
        self._rank_ncols = 0
        self._rank_order: list[int] = []
        self._rank_focus_pos = None  # df position of the clicked FOV (blue highlight)
        self._rank_focus_label = None

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 2)  # narrower tuning panel (histograms + knob table)
        split.setStretchFactor(1, 5)  # more room for the ranked-FOV grid
        split.setSizes([560, 1140])
        return split

    # ---- rank tab: knob state ----
    def _rank_feature_list(self):
        """The feature columns available to rank in the loaded data (empty if none loaded)."""
        return D.feature_columns(self.df) if self.df is not None else []

    def _seed_range(self, f, direction):
        """Data-derived (lo, hi) for feature ``f`` at ``direction``: label-agnostic quantiles
        of all values (target -> [q25, q75]; monotone -> [q05, q95])."""
        values = self.df[f].to_numpy(float)
        finite = values[~np.isnan(values)]

        def quantile(a, p):
            """The p-quantile of array `a`, or 0.0 if it is empty."""
            return float(np.quantile(a, p)) if len(a) else 0.0

        if direction == "target":
            return quantile(finite, 0.25), quantile(finite, 0.75)
        return quantile(finite, 0.05), quantile(finite, 0.95)

    def _rank_seed_ranges(self):
        """Fresh per-feature knobs: direction from the feature-name default, range from
        :meth:`_seed_range` (label-agnostic data quantiles)."""
        ranges = {}
        feats = self._rank_feature_list()
        # Default selection: only coverage_frac is checked (feeds the score); the user enables
        # the rest as needed. Fall back to the first feature if coverage_frac is absent.
        default_on = {f for f in feats if f == "coverage_frac"}
        if not default_on and feats:
            default_on = {feats[0]}
        for f in feats:
            # Default to a gaussian bell: it is symmetric, so the direction is 'target'
            # (a gaussian's direction combo is forced to 'target' anyway). Seed its center /
            # fwhm from the target-band quantiles via _seed_range.
            direction = "target"
            lo, hi = self._seed_range(f, direction)
            ranges[f] = {
                "direction": direction,
                "shape": "gaussian",  # curve family; user can switch to sigmoid/lognormal
                "lo": lo,
                "hi": hi,
                "curve_k": 0.0,  # steepness for sigmoid / lognormal
                "weight": 1.0,
                "enabled": f in default_on,  # unchecking a feature drops it from the score
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
            scombo.setCurrentText(spec.get("shape", "gaussian"))
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
        """A spin for one interpretable parameter. The param name is shown OUTSIDE the box (a
        label above it, added by :class:`_ParamCell`) rather than as an in-box prefix. Stores
        its param key on the widget so :meth:`_read_rank_table` knows what it is."""
        s = self._make_spinbox(value)
        if key in ("fwhm", "width"):  # strictly positive widths
            s.setRange(1e-9, 1e12)
        elif key == "fold":  # multiplicative tolerance must exceed 1
            s.setRange(1.0 + 1e-6, 1e12)
        s._param_key = key
        return s

    def _rank_fill_params(self, row, spec):
        """(Re)build the three parameter columns for ``row`` from its internal spec, showing the
        interpretable params for the current shape/direction (see fov_model.curve_params)."""
        tbl = self.rank_table
        params = FM.curve_params(
            spec.get("shape", "gaussian"),
            spec["lo"],
            spec["hi"],
            spec.get("curve_k", 0.0),
        )
        items = list(params.items())
        for idx, col in enumerate(RCOL_PARAMS):
            if idx < len(items):
                key, value = items[idx]
                spin = self._make_param_spinbox(key, value)
                spin.editingFinished.connect(self._rank_refresh_curves)
                tbl.setCellWidget(
                    row, col, _ParamCell(key, spin)
                )  # name label outside the box
            else:  # unused slot for this shape
                tbl.removeCellWidget(row, col)

    def _rank_update_row_enabled(self, row):
        """Constrain the direction combo to the row's shape: gaussian and lognormal are
        symmetric bells (direction is always 'target', combo disabled); sigmoid is monotonic
        (direction 'higher' or 'lower' only). The combo's items are rebuilt to the allowed set
        and its selection is taken from the feature's stored direction when still valid."""
        dcombo = self.rank_table.cellWidget(row, RCOL_DIR)
        shape_w = self.rank_table.cellWidget(row, RCOL_SHAPE)
        if dcombo is None or shape_w is None:
            return
        shape = shape_w.currentText()
        allowed = ["target"] if shape in ("gaussian", "lognormal") else ["higher", "lower"]
        f = self._rank_row_feature(row)
        desired = (
            self.rank_ranges[f]["direction"] if f in self.rank_ranges else dcombo.currentText()
        )
        dcombo.blockSignals(True)
        if [dcombo.itemText(i) for i in range(dcombo.count())] != allowed:
            dcombo.clear()
            dcombo.addItems(allowed)
        dcombo.setCurrentText(desired if desired in allowed else allowed[0])
        dcombo.blockSignals(False)
        dcombo.setEnabled(len(allowed) > 1)  # bells have a single fixed direction

    def _rank_row_feature(self, row):
        """The feature name at table `row`, or None if the row index is out of range."""
        feats = list(self.rank_ranges)
        return feats[row] if 0 <= row < len(feats) else None

    def _on_rank_dir_changed(self, row):
        """Apply a direction change for `row`: store it, rebuild that row's param columns, and redraw the curves."""
        # Only sigmoid has an editable direction (higher <-> lower); it flips the curve while
        # keeping the same params (midpoint/width), so rebuild the columns and redraw.
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
        """Apply a shape change for `row`: store it, constrain the direction combo to the new shape, rebuild the param columns from the kept bounds, and redraw."""
        # Shape changes the parameter SET; keep the internal bounds and re-derive the params for
        # the new shape (a gaussian center/fwhm becomes a sigmoid midpoint/width around the band).
        f = self._rank_row_feature(row)
        if f is None:
            return
        self.rank_ranges[f]["shape"] = self.rank_table.cellWidget(
            row, RCOL_SHAPE
        ).currentText()
        # bells -> target (disabled); sigmoid -> higher/lower; may coerce the current direction
        self._rank_update_row_enabled(row)
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
        spins are converted to the internal (lo, hi, curve_k) bounds via
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
                lo, hi, curve_k = FM.curve_bounds(shape, params)
            except (ValueError, KeyError):
                lo, hi, curve_k = prev["lo"], prev["hi"], prev.get("curve_k", 0.0)
            weight = tbl.cellWidget(i, RCOL_WEIGHT).value()
            item = tbl.item(i, RCOL_FEATURE)
            enabled = item is None or item.checkState() == QtCore.Qt.Checked
            self.rank_ranges[f] = {
                "direction": direction,
                "shape": shape,
                "lo": lo,
                "hi": hi,
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
                s.get("shape", "gaussian"),
                s["direction"],
                s["lo"],
                s["hi"],
                s.get("curve_k", 0.0),
                s.get("weight", 1.0),
            )
            for f, s in self.rank_ranges.items()
            if s.get("enabled", True) and (self.df is None or f in self.df.columns)
        }
        agg = (
            self.rank_agg_combo.currentText()
            if hasattr(self, "rank_agg_combo")
            else FM.DesirabilityModel.DEFAULT_AGGREGATION
        )
        # top_fov is required by DesirabilityModel but is a SELECTION quota applied by the
        # manager; the viewer only scores/orders FOVs and never selects, so pass the minimal
        # valid value to satisfy the constructor.
        return {
            "type": "ranking_by_defined_range",
            "top_fov": 1,
            "aggregation": agg,
            "features": feats,
        }

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
        # keep the Analysis-tab FOV grid in the same (score) order as the Rank tab
        if hasattr(self, "grid_host") and self.sel_pos:
            self._rebuild_grid()
        s = self.df["score"]
        scores = self.df["score"].to_numpy(float)
        acc, npairs = self._rank_pairwise_accuracy(scores)
        if acc is not None:
            acc_txt = f" · pairwise acc {acc:.3f} ({npairs} good/neutral/bad pairs)"
            topk, k = self._rank_topk_accuracy(scores)
            if topk is not None:
                acc_txt += f" · top-{k} acc {topk:.3f} (good in top-{k} / {k} good)"
        else:
            acc_txt = " · label FOVs (goodness) to get pairwise accuracy"
        self.rank_status.setText(
            f"ranked {len(self.df)} FOV(s) with {n_used}/{len(self.rank_ranges)} feature(s) · "
            f"score in [{s.min():.3f}, {s.max():.3f}]{acc_txt}"
        )

    def _rank_topk_accuracy(self, scores):
        """Precision@k over the LABELED FOVs, where k = number of Good-labeled FOVs: rank the
        labeled FOVs by score, take the top-k, and report the fraction that are Good (== recall
        here). Directly measures the failure mode where a neutral/bad FOV is scored too high.
        Returns (precision, k), or (None, 0) when there are no Good labels / too few labels."""
        if self.df is None or "goodness" not in self.df.columns:
            return None, 0
        g = self.df["goodness"].to_numpy(float)
        keep = np.isin(g, (-1.0, 0.0, 1.0)) & ~np.isnan(np.asarray(scores))
        gl, sc = g[keep], np.asarray(scores)[keep]
        k = int((gl == 1.0).sum())
        if k == 0 or gl.size < 2:
            return None, k
        top = np.argsort(-sc, kind="stable")[:k]
        return float((gl[top] == 1.0).mean()), k

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
        edges. The curve BETWEEN anchors is set by the shape (drawn by sampling); these are
        just the drag handles, placed at the curve's height at lo/hi."""
        lo, hi = spec["lo"], spec["hi"]

        def desirability_at(x):
            """The profile's desirability height at value `x` for this feature's spec."""
            return float(
                FM.DesirabilityModel._desirability(
                    np.array([x]),
                    lo,
                    hi,
                    spec["direction"],
                    spec.get("shape", "gaussian"),
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
            spec.get("shape", "gaussian"),
            spec.get("curve_k", 0.0),
        )
        return xs, ds

    def _rank_draw_hists(self):
        """One histogram of measured values per feature, with the desirability profile drawn
        as a dashed line through DRAGGABLE control points (the lo/hi band edges).

        Axis limits are LOCKED to the full data range of each feature, so the whole
        histogram is always shown and tuning the range never rescales the axes."""
        fig = self.rank_fig
        fig.clear()
        self._rank_axes = []
        # only draw features actually present in the loaded matrix; a ranking profile may
        # reference features this matrix does not have (skip them instead of KeyError-crashing).
        feats = (
            [] if self.df is None else [f for f in self.rank_ranges if f in self.df.columns]
        )
        if self.df is None or not feats:
            self.rank_canvas.draw_idle()
            return
        ncol = 2
        nrow = int(np.ceil(len(feats) / ncol))
        # give each row a fixed pixel height so the canvas grows (and scrolls) with the
        # feature count instead of squeezing every histogram into the viewport.
        row_px = 250  # a little taller so the x/y axis labels are not clipped
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
            # Widen the x-axis to 1.5x the data span (0.25*span of padding on each side) so
            # the distribution and its desirability curve have breathing room on both sides.
            # Histogram BINS stay at the data range [lo_x, hi_x]; only the view limits and the
            # sampled curve extend into the padding.
            pad_x = 0.25 * (hi_x - lo_x)
            xlo, xhi = lo_x - pad_x, hi_x + pad_x
            # desirability profile (right axis, 0..1): smooth SAMPLED curve for the chosen
            # shape + draggable anchor markers. Unchecked features are dimmed.
            on = spec.get("enabled", True)
            prof_c = "#ffffff" if on else "#666666"
            ax2 = ax.twinx()
            xs_c, ds_c = self._sample_profile(spec, xlo, xhi)
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
            if self._rank_focus_pos is not None and 0 <= self._rank_focus_pos < len(self.df):
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
                            spec.get("shape", "gaussian"),
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
            # lock x to the padded range, independent of the knobs; disable autoscale so
            # a dragged control point can never rescale it.
            for a in (ax, ax2):
                a.set_xlim(xlo, xhi)
                a.set_autoscalex_on(False)
            # feature name as the axes TITLE (above the plot) with a small pad to save space.
            # Bold + gold so it reads at a glance (dimmed gold when the feature is unchecked).
            title = f"{f}  ({spec['direction']})" + ("" if on else "  [off]")
            ax.set_title(
                title,
                color=RANK_TITLE_COLOR if on else "#8a7a3a",
                fontsize=8,
                fontweight="bold",
                pad=2,
            )
            # axis labels: feature value (x), FOV count (y, left), desirability (y, right).
            ax.set_xlabel("value", color="#bbbbbb", fontsize=6, labelpad=1)
            ax.set_ylabel("count", color="#bbbbbb", fontsize=6, labelpad=1)
            ax2.set_ylabel("desirability", color="#ffffff", fontsize=6, labelpad=1)
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
                    "xlim": (xlo, xhi),
                }
            )
        # Margins leave room for the title above each plot and the x/y axis labels (left for
        # 'count', right for 'desirability', bottom for 'value'); hspace fits each row's title
        # with a small pad, wspace keeps the y labels off the neighbouring axes.
        fig.subplots_adjust(
            left=0.12, right=0.88, top=0.95, bottom=0.08, hspace=0.5, wspace=0.42
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
            # strict: Line2D.get_data() returns equal-length x/y by construction.
            for j, (x, y) in enumerate(zip(xs, ys, strict=True)):
                px, py = meta["ax2"].transData.transform((x, y))
                dist = np.hypot(px - event.x, py - event.y)
                if dist <= 10 and (best is None or dist < best[2]):
                    best = (meta, j, dist)
        if best is not None:
            self._rank_drag = (best[0], best[1])

    def _rank_on_motion(self, event):
        """While dragging, move the grabbed control point (lo/hi), sync the table row, and reshape the overlaid curve."""
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
        if role == "lo":
            lo = min(x, hi)
        elif role == "hi":
            hi = max(x, lo)
        spec.update(lo=lo, hi=hi)
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
            spec.get("shape", "gaussian"),
            spec["lo"],
            spec["hi"],
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
        """Rebuild the Rank-tab FOV grid as one section per WELL: a header + a wrapping grid of
        that well's FOVs ordered best-first by score, with the wells laid out in plate order and
        stacked vertically (scrolling flows through the sections). Thumbnails decode lazily."""
        while self.rank_sections_layout.count():
            it = self.rank_sections_layout.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self._rank_sections = []
        self._rank_thumb_items = []
        self._rank_ncols = 0
        self._rank_focus_label = None  # tiles are recreated below; re-applied per section
        if self.df is None or not self._rank_order:
            self.rank_grid_info.setText("no FOVs")
            return

        # Group the globally score-sorted positions by well; iterating _rank_order keeps each
        # well's FOVs in best-first score order. Wells are then shown in plate order (row, col).
        per_well, wells = self._group_positions_by_well(self._rank_order)

        size = self.rank_size_slider.value()
        tw, th = self._tile_dims(size)
        for key in wells:
            self._rank_add_section(key, per_well[key], tw, th)
        self.rank_grid_info.setText(
            f"{len(self._rank_order)} FOV(s) in {len(wells)} well(s), best-first per well"
        )
        self._rank_ncols = 0  # force a reflow into the current column count
        self._rank_reflow()
        QtCore.QTimer.singleShot(0, self._rank_load_visible)

    def _rank_add_section(self, key, positions: list[int], tw: int, th: int) -> None:
        """Append one well section: a header label (acts as the divider) above a grid of the
        well's FOV tiles, in best-first score order. Tiles are placeholders; thumbnails decode
        lazily in :meth:`_rank_load_visible`."""
        title = "All FOVs" if key is None else f"Well {key[0]}/{key[1]}"
        header = QtWidgets.QLabel(f"{title}   ({len(positions)} FOV(s))")
        header.setStyleSheet(
            f"font-weight:700; color:{RANK_TITLE_COLOR}; background:#333; "
            "padding:4px 6px; border-radius:4px; margin-top:6px;"
        )
        self.rank_sections_layout.addWidget(header)
        grid_host = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(grid_host)
        grid.setSpacing(4)
        grid.setContentsMargins(0, 0, 0, 4)
        grid.setAlignment(QtCore.Qt.AlignTop)
        self.rank_sections_layout.addWidget(grid_host)

        items = []
        for rank_in_well, pos in enumerate(positions, start=1):
            row = self.df.iloc[pos]
            png = row.get("__png", "")
            score = float(row.get("score", float("nan")))
            caption = f"#{rank_in_well}  {score:.4f}"
            lab = QtWidgets.QLabel()
            lab.setFixedSize(tw, th)
            lab.setAlignment(QtCore.Qt.AlignCenter)
            lab.setCursor(QtCore.Qt.PointingHandCursor)
            lab._base_qss = goodness_border_qss(row.get("goodness", np.nan))
            if pos == self._rank_focus_pos:  # keep the blue highlight across rebuilds
                lab.setStyleSheet(_border_qss(RANK_CLICK_COLOR))
                self._rank_focus_label = lab
            else:
                lab.setStyleSheet(lab._base_qss)
            lab.setToolTip(f"{caption}  (well rank)\n{self._row_id(row)}")
            lab.setText("…")
            lab.mousePressEvent = lambda e, p=int(pos): self._rank_on_click(p)
            item = [lab, png, False, int(pos), score, caption]
            items.append(item)
            self._rank_thumb_items.append(item)
        self._rank_sections.append({"grid": grid, "items": items})

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
        """Re-lay each well section's thumbnails into as many columns as the viewport allows;
        caveat: no-op if the column count is unchanged."""
        if not self._rank_sections:
            return
        size = self.rank_size_slider.value()
        tw, _th = self._tile_dims(size)
        vw = self.rank_scroll.viewport().width() - 12
        ncols = max(1, vw // (tw + 4))
        if ncols == self._rank_ncols:
            return
        self._rank_ncols = ncols
        for sec in self._rank_sections:
            grid = sec["grid"]
            for i, item in enumerate(sec["items"]):
                grid.addWidget(item[0], i // ncols, i % ncols)
        QtCore.QTimer.singleShot(0, self._rank_load_visible)

    def _rank_load_visible(self, *_):
        """Decode the thumbnails currently visible in the scroll viewport (captioned with the
        per-well rank + score). Visibility is taken from each tile's clipped visible region, so
        it works regardless of the variable-height well sections above it."""
        if not self._rank_thumb_items:
            return
        size = self.rank_size_slider.value()
        for it in self._rank_thumb_items:
            lab, png, loaded, caption = it[0], it[1], it[2], it[5]
            if loaded or lab.visibleRegion().isEmpty():  # skip loaded / offscreen tiles
                continue
            if png and Path(png).exists():
                pm = self._load_thumb(png, size)
                if not pm.isNull():
                    lab.setText("")
                    lab.setPixmap(self._annotate_pixmap(pm, caption))
                else:
                    lab.setText("bad png")
            else:
                lab.setText(f"{caption}\n(no png)")
            it[2] = True

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
        self._refresh_score_map_controls()  # keep the Score-map pair combos in sync

    def _csv_dir(self) -> Path | None:
        """Directory the loaded feature CSV was read from (its ``__src``), or ``None`` when
        no data is loaded. Used to default the profile Save/Load dialogs next to the CSV."""
        if self.df is None or "__src" not in self.df.columns or len(self.df) == 0:
            return None
        src = self.df["__src"].iloc[0]
        return Path(src).parent if src else None

    def _on_rank_save(self):
        """Save the checked features' desirability config as a YAML mapping ready to drop under fov_selection.model.features in the acquisition config; caveat: writes only the features block (not type/top_fov), and unchecked features are omitted."""
        if not self.rank_ranges:
            self.rank_status.setText("nothing to save; load data first")
            return
        self._read_rank_table()
        # Default to the loaded CSV's directory so the profile saves next to the data it was
        # tuned on; fall back to the configured profile dir when no data is loaded.
        default_dir = self._csv_dir() or PROFILE_DIR
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save desirability ranges",
            str(Path(default_dir) / "desirability_ranges.yaml"),
            "YAML (*.yaml *.yml)",
        )
        if not path:
            return
        import yaml

        # Match the config's model.features layout: every feature is a block mapping. Any
        # numeric list would stay inline as [lo, hi] (no current shape emits one, but keep the
        # rule so future list-valued params render compactly) via a custom dumper.
        class _ProfileDumper(yaml.SafeDumper):
            pass

        _ProfileDumper.add_representer(
            list,
            lambda dumper, data: dumper.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            ),
        )
        features = self._rank_model_cfg()["features"]
        # sort_keys=False keeps the checked-first feature order and the shape/params/weight order.
        text = yaml.dump(
            features,
            Dumper=_ProfileDumper,
            sort_keys=False,
            default_flow_style=False,
            indent=2,
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
        self.rank_status.setText(
            f"saved {len(features)} feature(s) → {Path(path).name} "
            "(paste under fov_selection.model.features in the config)"
        )

    def _on_rank_write_scores(self):
        """Write the current ranking's ``proba`` (score) and ``rank`` back to each loaded FOV's
        source CSV, so the calibration matrix carries the tuned selection. Bound to the
        "Write proba/rank to CSV" button.

        Re-ranks first when no score exists yet. ``rank`` is global, best-first (1 = highest
        score, ties broken by row order so ranks are distinct consecutive ints), NaN scores
        last -- matching the best-first order shown on the right. The columns are named
        ``proba`` / ``rank`` so the calibration CSV lines up with the normal-mode
        ``fov_summary.csv`` and the viewer keeps them off the feature axes (META_BLACKLIST).
        """
        import pandas as pd

        if self.df is None or not self.rank_ranges:
            self.rank_status.setText("no data to score; load a matrix first")
            return
        if "score" not in self.df.columns:
            self._rerank()
            if "score" not in self.df.columns:
                return  # _rerank already reported why (e.g. no checked features)
        scores = self.df["score"].to_numpy(float)
        # method='first' -> ties resolve to distinct consecutive ranks; NaN scores rank last.
        rank = pd.Series(scores, index=self.df.index).rank(ascending=False, method="first")
        self.df["proba"] = scores
        self.df["rank"] = rank.to_numpy(float)
        saved, matched = self._persist_scores()
        if saved:
            self.rank_status.setText(
                f"wrote proba/rank for {matched} FOV(s) to {saved} CSV(s)"
            )

    def _persist_scores(self):
        """Write the in-memory ``proba`` / ``rank`` columns back to each row's source CSV.

        Rows are grouped by ``__src`` so multiple loaded datasets each save to the right file,
        and matched by FOV identity (``filename`` etc.) so the on-disk row order is irrelevant
        -- the same join the Label tab uses (:meth:`_persist_changes`). Never raises: a
        per-file failure is reported in the status line and skipped. Returns (files, rows)."""
        import pandas as pd

        if "__src" not in self.df.columns:
            return 0, 0
        src_i = self.df.columns.get_loc("__src")
        by_src: dict[str, list[int]] = {}
        for pos in range(len(self.df)):
            by_src.setdefault(self.df.iat[pos, src_i], []).append(pos)
        saved = matched = 0
        for src, rows in by_src.items():
            if not src:
                continue
            try:
                disk = pd.read_csv(src)
            except Exception as e:  # noqa: BLE001
                self.rank_status.setText(f"read failed: {Path(src).name}: {e}")
                continue
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
            for col in ("proba", "rank"):
                if col not in disk.columns:
                    disk[col] = np.nan
            pi, ri = disk.columns.get_loc("proba"), disk.columns.get_loc("rank")
            spi, sri = self.df.columns.get_loc("proba"), self.df.columns.get_loc("rank")
            for pos in rows:
                k = "\x1f".join(str(self.df.iat[pos, c]) for c in lc)
                j = lookup.get(k)
                if j is not None:
                    disk.iat[j, pi] = self.df.iat[pos, spi]
                    disk.iat[j, ri] = self.df.iat[pos, sri]
                    matched += 1
            disk.to_csv(src, index=False)
            saved += 1
        return saved, matched

    def _on_rank_load(self):
        """Load a desirability profile (YAML or legacy JSON), merging its (checked) features over the data-seeded (unchecked) ones, then re-rank; caveat: accepts either a bare features mapping or a full model dict, and legacy `range`-style profiles are still parsed."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load desirability ranges",
            str(PROFILE_DIR),
            "profile (*.yaml *.yml *.json)",
        )
        if not path:
            return
        self._apply_rank_profile(path)

    def _apply_rank_profile(self, path, *, source=None):
        """Merge a desirability profile FILE over the data-seeded ranges, then re-rank.

        Used by the Rank-tab Load button and the ``--rank-profile`` launch flag. Reads a
        YAML/JSON file holding either a bare features mapping or a full model dict, then
        delegates to :meth:`_apply_rank_profile_cfg`. ``source`` overrides the name shown in
        the status line. Never raises: a bad/absent file is reported in the status line.
        """
        import yaml

        source = source or Path(path).name
        try:
            cfg = yaml.safe_load(Path(path).read_text())  # YAML is a superset of JSON
        except Exception as e:  # noqa: BLE001
            self.rank_status.setText(f"load failed: {e}")
            return
        self._apply_rank_profile_cfg(cfg, source=source)

    def _apply_rank_profile_cfg(self, cfg, *, source):
        """Merge an already-parsed desirability profile over the data-seeded ranges, re-rank.

        Shared by :meth:`_apply_rank_profile` (file load) and the ``--rank-profile-json``
        launch flag (a calibration pre-scan seeds the viewer straight from the config's
        ``fov_selection.model``, passed inline so no profile file is written to disk).
        Accepts either a bare features mapping or a full model dict (``{type, features,
        ...}``); legacy ``range``-style profiles are still parsed. Never raises: a malformed
        profile is reported in the status line.
        """
        try:
            feats = cfg.get("features", cfg) if isinstance(cfg, dict) else {}
            loaded = {}
            for f, feat_cfg in feats.items():
                # curve_bounds does the shape math; legacy `range` profiles still open.
                shape, direction, lo, hi, curve_k = _feature_to_internal(feat_cfg)
                loaded[f] = {
                    "direction": direction,
                    "shape": shape,
                    "lo": lo,
                    "hi": hi,
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
        msg = f"loaded {len(loaded)} feature(s) from {source}"
        if self.df is not None:
            missing = [f for f in loaded if f not in self.df.columns]
            if missing:
                msg += f" · {len(missing)} not in this matrix (ignored): {', '.join(missing)}"
        self.rank_status.setText(msg)
