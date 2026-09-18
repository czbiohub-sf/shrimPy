"""Label tab: group loaded FOVs into per-goodness-class panels; drag to relabel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qtpy import QtCore, QtWidgets

from ._common import (
    GOODNESS_LABEL_ORDER,
    THUMB_QSS,
    _DropPanel,
    _ThumbLabel,
)


class LabelTabMixin:
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
        head.addWidget(QtWidgets.QLabel("channel"))
        self.label_chan_combo = QtWidgets.QComboBox()  # shared channel toggle (all tabs)
        self.label_chan_combo.currentTextChanged.connect(self._set_channel)
        head.addWidget(self.label_chan_combo)
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
        tw, th = self._tile_dims(size)
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
            lab.setFixedSize(tw, th)
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
        key = LabelTabMixin._class_key(val)
        for i, (_, v, _) in enumerate(GOODNESS_LABEL_ORDER):
            if LabelTabMixin._class_key(v) == key:
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
        """Commit pending Label-tab drag edits to the in-memory table and each row's source
        CSV, then update the save state in place. The panels already show the moved thumbnails
        (drags relayout them and update counts live), so nothing is rebuilt -- each panel keeps
        its current scroll position instead of jumping back to the top."""
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
        self._update_label_save_state()  # disable Save; panels keep their scroll position
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
        tw, _th = self._tile_dims(size)
        vw = info["scroll"].viewport().width() - 12
        ncols = max(1, vw // (tw + info["grid"].spacing()))
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
        _tw, th = self._tile_dims(size)
        rowh = th + info["grid"].spacing()
        ncols = max(1, info["ncols"])
        margin = rowh
        for idx, it in enumerate(items):
            lab, png, loaded = it[0], it[1], it[2]
            if loaded:
                continue
            top = (idx // ncols) * rowh
            if top + th < y0 - margin or top > y1 + margin:
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
