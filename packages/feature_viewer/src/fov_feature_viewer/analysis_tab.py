"""Analysis tab: interactive 2D/3D feature scatter (PCA / t-SNE / UMAP) and the selected-FOV thumbnail grid."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.transforms import IdentityTransform
from mpl_toolkits.mplot3d import proj3d
from qtpy import QtCore, QtWidgets

from ._common import (
    DETAIL_SKIP,
    FOCUS_COLOR,
    GOODNESS_CATEGORIES,
    GOODNESS_COLORS,
    MAX_THUMBS,
    MPL_BG,
    MPL_FG,
    MPL_GRID,
    RANK_TITLE_COLOR,
    THUMB_FOCUS_QSS,
    THUMB_QSS,
    _DetailsModel,
    goodness_border_qss,
)


class AnalysisTabMixin:
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
        self.canvas.setMinimumSize(100, 100)
        w.setMinimumWidth(100)
        return w

    # ---------------------------------------------------------------- right
    def _build_right(self):
        """Build the Analysis-tab right column (selected-FOV thumbnail grid plus the details table) and return its widget."""
        w = QtWidgets.QWidget()
        w.setMinimumWidth(240)
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
        # Selected FOVs are grouped into one section per well (a header + a wrapping grid),
        # stacked vertically like the Rank tab; scrolling flows through the well sections.
        self.sections_layout = QtWidgets.QVBoxLayout(self.grid_host)
        self.sections_layout.setSpacing(4)
        self.sections_layout.setContentsMargins(0, 0, 0, 0)
        self.sections_layout.setAlignment(QtCore.Qt.AlignTop)
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
        self._thumb_items: list[list] = []  # flat [label, png_path, loaded, pos], lazy decode
        self._grid_sections: list[dict] = []  # per-well {"grid": QGridLayout, "items": [...]}
        self._cur_ncols = 0
        self._thumb_cache: OrderedDict = OrderedDict()  # (path,size) -> QPixmap, LRU
        return w

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

    def _rebuild_grid(self):
        """Rebuild the selected-FOV grid as one section per WELL (a header + a wrapping grid of
        that well's FOVs, best-first by score), mirroring the Rank tab. Tiles are lazy
        placeholders (capped and balanced across datasets); any focus highlight is re-applied."""
        while self.sections_layout.count():
            it = self.sections_layout.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self._thumb_items = []
        self._grid_sections = []
        self._focus_label = None
        self._cur_ncols = 0
        if not self.sel_pos or self.df is None:
            self.grid_info.setText("no selection")
            return
        shown = self._balanced_cap(self.sel_pos, MAX_THUMBS)
        # Match the Rank tab's ordering: best-first by `score` once it has been computed
        # (NaN scores sink to the end). Before any Re-rank there is no score column, so the
        # selection order is kept.
        if "score" in self.df.columns:
            sc = self.df["score"].to_numpy(float)
            shown = sorted(
                shown,
                key=lambda p: (np.isnan(sc[p]), -(sc[p] if not np.isnan(sc[p]) else 0.0)),
            )
        extra = (
            f"  ({len(shown)} of {len(self.sel_pos)}, sampled across datasets, refine selection)"
            if len(self.sel_pos) > len(shown)
            else ""
        )
        per_well, wells = self._group_positions_by_well(shown)
        self.grid_info.setText(f"{len(shown)} FOV(s) in {len(wells)} well(s){extra}")
        size = self.size_slider.value()
        tw, th = self._tile_dims(size)
        for key in wells:
            self._grid_add_section(key, per_well[key], tw, th)
        self._reflow_grid()
        QtCore.QTimer.singleShot(0, self._load_visible_thumbs)  # after layout settles
        if self.focus_pos is not None:  # re-apply highlight
            self._highlight_thumb(self.focus_pos, scroll_grid=False)

    def _grid_add_section(self, key, positions, tw, th):
        """Append one well section to the selected-FOV grid: a header (the divider) above a grid
        of the well's goodness-bordered, click-to-focus tiles. Tiles are placeholders; thumbnails
        decode lazily in :meth:`_load_visible_thumbs`."""
        title = "All FOVs" if key is None else f"Well {key[0]}/{key[1]}"
        header = QtWidgets.QLabel(f"{title}   ({len(positions)} FOV(s))")
        header.setStyleSheet(
            f"font-weight:700; color:{RANK_TITLE_COLOR}; background:#333; "
            "padding:4px 6px; border-radius:4px; margin-top:6px;"
        )
        self.sections_layout.addWidget(header)
        grid_host = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(grid_host)
        grid.setSpacing(4)
        grid.setContentsMargins(0, 0, 0, 4)
        grid.setAlignment(QtCore.Qt.AlignTop)
        self.sections_layout.addWidget(grid_host)
        items = []
        for pos in positions:
            row = self.df.iloc[pos]
            png = row.get("__png", "")
            lab = QtWidgets.QLabel()
            lab.setFixedSize(tw, th)
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
            item = [lab, png, False, int(pos)]
            items.append(item)
            self._thumb_items.append(item)
        self._grid_sections.append({"grid": grid, "items": items})

    def _reflow_grid(self):
        """Re-lay each well section's thumbnails into as many columns as the viewport width allows; caveat: no-op if the column count is unchanged."""
        if not self._grid_sections:
            return
        size = self.size_slider.value()
        tw, _th = self._tile_dims(size)
        vw = self.scroll.viewport().width() - 12
        ncols = max(1, vw // (tw + 4))
        if ncols == self._cur_ncols:
            return
        self._cur_ncols = ncols
        for sec in self._grid_sections:
            grid = sec["grid"]
            for i, item in enumerate(sec["items"]):
                grid.addWidget(item[0], i // ncols, i % ncols)
        QtCore.QTimer.singleShot(0, self._load_visible_thumbs)

    def _load_visible_thumbs(self, *_):
        """Decode the thumbnails currently visible in the scroll viewport; cache them.
        Visibility is taken from each tile's clipped visible region, so it works regardless of
        the variable-height well sections above it."""
        if not self._thumb_items:
            return
        size = self.size_slider.value()
        for it in self._thumb_items:
            lab, png, loaded = it[0], it[1], it[2]
            if loaded or lab.visibleRegion().isEmpty():  # skip loaded / offscreen tiles
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
        # also drop the rank-tab (blue) highlight: it stores a df position that goes stale
        # when the data changes, and a smaller new dataset would make it index out of bounds.
        self._rank_focus_pos = None
        if self._rank_focus_label is not None:
            self._rank_focus_label.setStyleSheet(
                getattr(self._rank_focus_label, "_base_qss", THUMB_QSS)
            )
            self._rank_focus_label = None
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
