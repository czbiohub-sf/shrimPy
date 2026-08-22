"""Score-map tab: 2D desirability heatmaps over pairs of features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from qtpy import QtWidgets

from shrimpy.fov_selection import fov_model as FM

from ._common import (
    MPL_BG,
    MPL_FG,
    _internal_to_feature,
    goodness_color,
)


class ScoreMapTabMixin:
    # =============================================================== score-map tab
    def _build_score_map_tab(self):
        """Tab: the 2-feature desirability score surface as a 2D filled contour + a 3D
        surface, with the loaded FOVs overlaid (colored by goodness). Two combos pick which
        pair of the CHECKED scoring features to view; the surface is the score of THAT pair
        alone (a 2-feature sub-model with the same shapes/weights/aggregation), so you always
        look at one pair at a time even when more than two features feed the full score."""
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("<b>Score map</b>"))
        bar.addWidget(QtWidgets.QLabel("X"))
        self.map_x_combo = QtWidgets.QComboBox()
        self.map_x_combo.setMinimumWidth(170)
        bar.addWidget(self.map_x_combo)
        bar.addWidget(QtWidgets.QLabel("Y"))
        self.map_y_combo = QtWidgets.QComboBox()
        self.map_y_combo.setMinimumWidth(170)
        bar.addWidget(self.map_y_combo)
        bar.addWidget(QtWidgets.QLabel("combine"))
        self.map_agg_combo = QtWidgets.QComboBox()
        self.map_agg_combo.addItems(list(FM.DesirabilityModel.AGGREGATIONS))
        if hasattr(self, "rank_agg_combo"):  # default to the Rank tab's aggregation
            self.map_agg_combo.setCurrentText(self.rank_agg_combo.currentText())
        bar.addWidget(self.map_agg_combo)
        self.map_points_cb = QtWidgets.QCheckBox("show FOVs")
        self.map_points_cb.setChecked(True)
        bar.addWidget(self.map_points_cb)
        bar.addStretch(1)
        v.addLayout(bar)

        self.map_status = QtWidgets.QLabel(
            "check at least 2 features in the Rank tab, then open this tab"
        )
        self.map_status.setWordWrap(True)
        v.addWidget(self.map_status)

        self.map_fig = Figure(figsize=(11, 5), facecolor=MPL_BG)
        self.map_canvas = FigureCanvas(self.map_fig)
        self.map_toolbar = NavigationToolbar(self.map_canvas, w)
        v.addWidget(self.map_toolbar)
        v.addWidget(self.map_canvas, 1)

        self._map_updating = False  # guard so combo repopulation does not trigger redraws
        for cb in (self.map_x_combo, self.map_y_combo, self.map_agg_combo):
            cb.currentIndexChanged.connect(self._update_score_map)
        self.map_points_cb.toggled.connect(self._update_score_map)
        return w

    def _score_map_features(self):
        """The checked (enabled) Rank-tab features -- the ones that feed the score."""
        return [f for f, s in self.rank_ranges.items() if s.get("enabled", True)]

    def _refresh_score_map_controls(self):
        """Repopulate the X/Y pair combos from the currently-checked features, keeping the
        prior picks when still valid and otherwise defaulting to the first two distinct ones."""
        if not hasattr(self, "map_x_combo"):
            return
        feats = self._score_map_features()
        prev_x, prev_y = self.map_x_combo.currentText(), self.map_y_combo.currentText()
        self._map_updating = True
        try:
            for cb in (self.map_x_combo, self.map_y_combo):
                cb.clear()
                cb.addItems(feats)
            if feats:
                x = prev_x if prev_x in feats else feats[0]
                y = (
                    prev_y
                    if (prev_y in feats and prev_y != x)
                    else next((f for f in feats if f != x), x)
                )
                self.map_x_combo.setCurrentText(x)
                self.map_y_combo.setCurrentText(y)
        finally:
            self._map_updating = False

    def _map_pair_cfg(self, fx, fy):
        """A 2-feature DesirabilityModel config for the (fx, fy) pair, using each feature's
        current shape/direction/range/weight and the map tab's aggregation."""
        feats = {}
        for f in (fx, fy):
            s = self.rank_ranges[f]
            feats[f] = _internal_to_feature(
                s.get("shape", "gaussian"),
                s["direction"],
                s["lo"],
                s["hi"],
                s.get("curve_k", 0.0),
                s.get("weight", 1.0),
            )
        # top_fov is a manager-side selection quota, unused for scoring; set to the minimal
        # valid value so DesirabilityModel (which requires it) can be constructed here.
        return {
            "type": "ranking_by_defined_range",
            "top_fov": 1,
            "aggregation": self.map_agg_combo.currentText(),
            "features": feats,
        }

    @staticmethod
    def _style_map_axes(ax):
        """Apply the dark theme to a score-map axes (2D or 3D)."""
        ax.set_facecolor(MPL_BG)
        ax.tick_params(colors=MPL_FG, labelsize=8)
        ax.title.set_color(MPL_FG)
        ax.xaxis.label.set_color(MPL_FG)
        ax.yaxis.label.set_color(MPL_FG)
        if hasattr(ax, "zaxis"):
            ax.zaxis.label.set_color(MPL_FG)
            ax.zaxis.set_tick_params(colors=MPL_FG)

    def _update_score_map(self, *_):
        """Draw the score contour (2D) + surface (3D) for the selected pair, with the loaded
        FOVs overlaid. Bound to the pair/aggregation combos and the FOV checkbox."""
        if getattr(self, "_map_updating", False) or not hasattr(self, "map_fig"):
            return
        self.map_fig.clear()
        if self.df is None:
            self.map_status.setText("no data loaded")
            self.map_canvas.draw_idle()
            return
        # pull the latest Rank-table edits so shapes / weights / direction are current
        if hasattr(self, "rank_table") and self.rank_ranges:
            self._read_rank_table()
        feats = self._score_map_features()
        if len(feats) < 2:
            self.map_status.setText(
                "check at least 2 features in the Rank tab to draw a score map"
            )
            self.map_canvas.draw_idle()
            return
        fx, fy = self.map_x_combo.currentText(), self.map_y_combo.currentText()
        if fx not in feats or fy not in feats:
            fx, fy = feats[0], feats[1]
        if fx == fy:
            self.map_status.setText("pick two DIFFERENT features for X and Y")
            self.map_canvas.draw_idle()
            return

        try:
            model = FM.build_fov_model(self._map_pair_cfg(fx, fy))
        except Exception as e:  # noqa: BLE001
            self.map_status.setText(f"cannot build score model: {e}")
            self.map_canvas.draw_idle()
            return

        x = self.df[fx].to_numpy(float)
        y = self.df[fy].to_numpy(float)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 1:
            self.map_status.setText(f"no FOVs with finite {fx} and {fy}")
            self.map_canvas.draw_idle()
            return

        xs = np.linspace(float(x[finite].min()), float(x[finite].max()), 120)
        ys = np.linspace(float(y[finite].min()), float(y[finite].max()), 120)
        gx, gy = np.meshgrid(xs, ys)
        grid_df = pd.DataFrame({fx: gx.ravel(), fy: gy.ravel()})
        z_grid = np.asarray(model.predict(grid_df)[0], float).reshape(gx.shape)
        fov_scores = np.asarray(model.predict(self.df)[0], float)
        agg = self.map_agg_combo.currentText()

        ax2 = self.map_fig.add_subplot(1, 2, 1)
        cf = ax2.contourf(gx, gy, z_grid, levels=20, cmap="viridis")
        ax2.contour(gx, gy, z_grid, levels=10, colors="k", linewidths=0.3, alpha=0.35)
        ax3 = self.map_fig.add_subplot(1, 2, 2, projection="3d")
        ax3.plot_surface(
            gx, gy, z_grid, cmap="viridis", alpha=0.55, linewidth=0, antialiased=True
        )

        if self.map_points_cb.isChecked():
            g = (
                self.df["goodness"].to_numpy(float)
                if "goodness" in self.df.columns
                else np.full(len(self.df), np.nan)
            )
            for code, name in (
                (1.0, "good"),
                (0.0, "neutral"),
                (-1.0, "bad"),
                (None, "unlabeled"),
            ):
                m = (np.isnan(g) if code is None else (g == code)) & finite
                if not m.any():
                    continue
                col = goodness_color(code)
                ax2.scatter(
                    x[m],
                    y[m],
                    s=22,
                    c=col,
                    edgecolors="k",
                    linewidths=0.4,
                    label=name,
                    zorder=5,
                )
                ax3.scatter(
                    x[m],
                    y[m],
                    fov_scores[m],
                    s=16,
                    c=col,
                    edgecolors="k",
                    linewidths=0.3,
                    depthshade=False,
                )
            leg = ax2.legend(loc="best", fontsize=8, framealpha=0.85)
            if leg is not None:
                for txt in leg.get_texts():
                    txt.set_color("#111")

        ax2.set_xlabel(fx)
        ax2.set_ylabel(fy)
        ax2.set_title(f"2D contour  ·  {agg}")
        ax3.set_xlabel(fx)
        ax3.set_ylabel(fy)
        ax3.set_zlabel("score")
        ax3.set_title(f"3D surface  ·  {agg}")
        ax3.view_init(elev=28, azim=-125)
        self._style_map_axes(ax2)
        self._style_map_axes(ax3)
        cbar = self.map_fig.colorbar(cf, ax=ax2, fraction=0.046, pad=0.04, label="score")
        cbar.ax.yaxis.label.set_color(MPL_FG)
        cbar.ax.tick_params(colors=MPL_FG)
        self.map_fig.tight_layout()
        self.map_canvas.draw_idle()

        n_more = len(feats) - 2
        extra = (
            f"  ·  {n_more} other checked feature(s) not shown (pair view)"
            if n_more > 0
            else ""
        )
        self.map_status.setText(f"score = f({fx}, {fy}) via {agg}{extra}")
