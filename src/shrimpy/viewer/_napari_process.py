"""The napari window: a live or finished acquisition read straight from its OME-Zarr.

This module is the viewer itself. It runs in a *separate process* from an acquisition
(launched by :class:`shrimpy.viewer.live.LiveViewer`) and in the main process for
``shrimpy view``; it must therefore be importable without napari installed -- all
napari/Qt imports are deferred into :func:`run_viewer` -- so the acquisition process can
import the launcher regardless.

Every pixel comes from the store on disk (see :class:`shrimpy.viewer.store`), never from
the acquisition process, so the *whole* acquisition is browsable while it runs and
nothing the viewer does can stall or crash the acquisition. A timer re-reads the array
metadata, grows the layers as timepoints are appended, and follows the newest complete
volume until the user scrubs away.

Each channel is its own napari image layer over ``(p, t, z, y, x)``; not-yet-acquired
indices render as blank planes.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue as _queue

from contextlib import contextmanager
from pathlib import Path

import numpy as np

from napari_deskew_preview import (
    LS_ANGLE_DEG,
    PIXEL_SIZE_UM,
    LazyPlaneArray,
    deskewed_layer,
)

from shrimpy.viewer.store import DEFAULT_CACHE_MB, AcquisitionStore

# DeskewControls is imported lazily inside _add_deskew_widget (it needs Qt).

logger = logging.getLogger(__name__)

# Index axes, in the order they appear as leading dimensions of each layer.
# "position" folds stage positions and grid FOVs into a single slider, matching the
# one-array-per-position layout on disk.
_INDEX_AXES = ("position", "t", "z")

# Slider labels shown in napari (position axis shown simply as "p"), plus y/x.
_AXIS_LABELS = ("p", "t", "z", "y", "x")

# Leading axes the viewer auto-advances. z is deliberately excluded: it stays under
# the user's control, so scrubbing z never pauses following and following preserves it.
_FOLLOW_AXES = tuple(i for i, ax in enumerate(_INDEX_AXES) if ax != "z")

# How often (ms) the viewer re-reads the store and repaints while acquiring.
DEFAULT_REFRESH_MS = 500


@contextmanager
def _quietly(logger_name: str):
    """Raise ``logger_name``'s threshold to errors for the duration of the block."""
    quietened = logging.getLogger(logger_name)
    previous = quietened.level
    quietened.setLevel(logging.ERROR)
    try:
        yield
    finally:
        quietened.setLevel(previous)


class LazyStoreArray(LazyPlaneArray):
    """Raw, read-only view of one channel of the store as ``(p, t, z, y, x)``.

    Each plane is fetched from the OME-Zarr on demand; indices not yet acquired come
    back as zeros. Deskewed display is handled separately by
    :class:`napari_deskew_preview.DeskewedArray`.
    """

    def __init__(self, store: AcquisitionStore, channel: int) -> None:
        self._store = store
        self._channel = channel
        self._index_sizes = store.index_sizes
        self._frame_shape = store.frame_shape
        self.dtype = store.dtype
        self._init_shape()

    def _plane(self, position: int, t: int, z: int) -> np.ndarray:
        return self._store.plane(position, t, self._channel, z)


def store_gather(store: AcquisitionStore, channel: int):
    """A :data:`~napari_deskew_preview.deskew.Gather` over one channel of the store.

    The deskew core is source-agnostic, so this is the store's counterpart to
    :func:`napari_deskew_preview.array_gather`: given ``(position, t)`` and a tilt-axis
    row, it returns that row across the whole scan stack.
    """
    blank_shape = (store.n_z, store.frame_shape[1])

    def gather(leading: tuple[int, ...], tilt_row: int) -> np.ndarray:
        position, t = leading
        row = store.tilt_row(position, t, channel, tilt_row)
        # Only complete volumes may be deskewed: a deskewed plane mixes every z slice,
        # so a stack still being written would render as a corrupt image.
        return np.zeros(blank_shape, dtype=store.dtype) if row is None else row

    return gather


class _ViewerState:
    """Owns the napari layers and keeps them in step with the store on disk."""

    def __init__(self, viewer: object, *, deskew: bool, cache_mb: float) -> None:
        self._viewer = viewer
        self._deskew_requested = deskew
        self._cache_mb = cache_mb
        self._store: AcquisitionStore | None = None
        self._path: Path | None = None
        self._layers: list[object] = []
        # Channels whose contrast limits have been auto-set from their first real data.
        self._contrast_done: set[int] = set()
        # Auto-advance state. Sliders follow the newest complete volume until the user
        # scrubs away (any manual current_step change pauses it); Home resumes it.
        self._following = True
        # Guards current_step writes we make ourselves, so they aren't mistaken for
        # user scrubbing.
        self._programmatic = False
        # (position, t) we last auto-advanced to; also where Home snaps back to.
        self._follow_target: tuple[int, int] | None = None
        # Last current_step we observed, to tell which axis a user change touched.
        self._last_step: tuple[int, ...] | None = None
        # Deskew state. Both raw and deskewed array views are built up front (lazy and
        # cheap) and the Deskew widget swaps which one backs each layer at runtime.
        self._deskew = False
        self._deskew_available = False
        self._projector: object | None = None
        self._index_sizes: tuple[int, ...] = ()
        self._raw_arrays: list[object] = []
        self._deskew_arrays: list[object] = []
        self._controls: object | None = None  # shared DeskewControls (built in the widget)

    # -- lifecycle --------------------------------------------------------------

    def set_path(self, path: str | Path) -> None:
        """Point the viewer at a store; it opens on the next tick, retrying until there."""
        self._path = Path(path)

    def tick(self) -> None:
        """Open the store if needed, pick up new data, and repaint.

        Repaints only when the store reports something new, which for a live
        acquisition is any volume whose chunks have landed since the last tick -- so
        following costs nothing while the writer is between volumes, and nothing at
        all once the acquisition is over.
        """
        if self._store is None:
            self._try_open()
            if self._store is None:
                return
        elif not self._store.refresh():
            return
        self._sync_extent()
        self._autoset_contrast()
        if self._following:
            self._follow_latest()
        self._refresh_layers()

    def _try_open(self) -> None:
        """Open the store, tolerating the window between 'the acquisition started' and
        'the writer finished creating the store'.

        Reading a half-created store is expected here, not exceptional, so iohub's
        per-position "skipped invalid item" warnings are quietened for the attempt --
        they would otherwise land in the acquisition log once per position per retry.
        Nothing real is hidden: a position's array can only exist after the writer has
        created *every* position's group, so an attempt that finds image data (the
        only kind this accepts) has by then seen the complete set of positions.
        """
        if self._path is None:
            return
        try:
            with _quietly("iohub.ngff"):
                self._store = AcquisitionStore(self._path, cache_mb=self._cache_mb)
        except Exception:  # noqa: BLE001 - not written yet (or not yet valid); retry
            logger.debug("Store %s not readable yet", self._path, exc_info=True)
            return
        # Confirm the newest volume before the first paint, so the viewer opens on
        # real data rather than on a stack that may still be mid-flush.
        self._store.refresh()
        self._build_layers()

    # -- layer construction -----------------------------------------------------

    def _build_layers(self) -> None:
        store = self._store
        assert store is not None
        self._index_sizes = store.index_sizes
        self._build_arrays()

        # Start on the deskewed view where it is available; "Display raw" switches back.
        self._deskew = self._deskew_available
        arrays = self._deskew_arrays if self._deskew else self._raw_arrays
        clim = _default_contrast_limits(store.dtype)
        for channel, name in enumerate(store.channels):
            self._layers.append(
                self._viewer.add_image(
                    arrays[channel],
                    name=name,
                    contrast_limits=clim,
                    colormap=_colormap_for_channel(name),
                    blending="additive" if store.n_channels > 1 else "translucent",
                )
            )
        # Label the leading sliders; y/x are the displayed image dims.
        self._viewer.dims.axis_labels = _AXIS_LABELS
        self._connect_follow_controls()
        # napari centers every slider by default; start following from p=0, t=0 instead.
        # Guarded (via _set_step) so it doesn't trip the auto-advance pause, and z is
        # left under the user's control.
        self._set_step((0, 0))
        self._last_step = tuple(self._viewer.dims.current_step)
        if self._deskew_available:
            self._add_deskew_widget()
        logger.info(
            "Viewing %s: %d position(s), %d channel(s), %d z, %d timepoint(s) so far",
            store.path.name,
            store.n_positions,
            store.n_channels,
            store.n_z,
            store.n_timepoints,
        )

    def _build_arrays(self) -> None:
        """(Re)build the raw and deskewed views for the store's current extent."""
        store = self._store
        assert store is not None
        self._raw_arrays = [LazyStoreArray(store, c) for c in range(store.n_channels)]

        # Deskew needs the microscope to be an oblique-plane one (``deskew``) and the
        # data to support it: a recorded scan step and a real z-stack.
        self._deskew_available = (
            self._deskew_requested and bool(store.scan_step_um) and store.n_z > 1
        )
        if not self._deskew_available:
            self._deskew_arrays = []
            return
        angle, pixel, scan = self._geometry()
        self._build_deskew_arrays(angle, pixel, scan)

    def _geometry(self) -> tuple[float, float, float]:
        """Deskew geometry: the widget's values once it exists, else the initial ones.

        The scan step comes from the store, where the acquisition recorded it; the
        angle and lateral pixel size are the microscope's defaults, which the Deskew
        widget can override (the store's lateral scale is not trustworthy -- see
        :class:`~shrimpy.viewer.store.AcquisitionStore`).
        """
        store = self._store
        assert store is not None
        if self._controls is not None:
            return (self._controls.angle, self._controls.pixel_size, self._controls.scan_step)
        return (LS_ANGLE_DEG, PIXEL_SIZE_UM, store.scan_step_um or 0.0)

    def _build_deskew_arrays(self, angle: float, pixel: float, scan: float) -> None:
        store = self._store
        assert store is not None
        arrays: list[object] = []
        for channel in range(store.n_channels):
            array, self._projector = deskewed_layer(
                store_gather(store, channel),
                raw_zyx_shape=(store.n_z, *store.frame_shape),
                scan_step_um=scan,
                batch_sizes=(store.n_positions, store.n_timepoints),
                ls_angle_deg=angle,
                pixel_size_um=pixel,
            )
            arrays.append(array)
        self._deskew_arrays = arrays

    def _sync_extent(self) -> None:
        """Grow the layers when the acquisition appends timepoints."""
        store = self._store
        assert store is not None
        if store.index_sizes == self._index_sizes or not self._layers:
            return
        self._index_sizes = store.index_sizes
        self._build_arrays()
        self._reassign_layer_data()

    def _reassign_layer_data(self) -> None:
        arrays = self._deskew_arrays if self._deskew else self._raw_arrays
        for layer, array in zip(self._layers, arrays, strict=True):
            layer.data = array  # napari resets dims/extent to the new shape
        self._last_step = tuple(self._viewer.dims.current_step)

    # -- deskew widget ----------------------------------------------------------

    def _add_deskew_widget(self) -> None:
        """Add the shared Deskew dock widget (Display deskewed / raw + geometry).

        The deskewed *view* does not need the widget, only the controls for editing its
        geometry do -- so a viewer running without Qt (a test harness) still deskews.
        """
        try:
            from napari_deskew_preview.controls import DeskewControls
            from qtpy.QtWidgets import QApplication

            if QApplication.instance() is None:
                # Constructing a QWidget without a QApplication aborts the process
                # rather than raising, so this has to be checked, not caught.
                logger.debug("No Qt application; skipping the Deskew widget.")
                return

            angle, pixel, scan = self._geometry()
            self._controls = DeskewControls(
                angle_deg=angle, pixel_size_um=pixel, scan_step_um=scan
            )
            self._controls.displayDeskewedRequested.connect(self._display_deskewed)
            self._controls.displayRawRequested.connect(self._display_raw)
            self._controls.geometryChanged.connect(self._on_geometry_changed)
            self._viewer.window.add_dock_widget(self._controls, name="Deskew", area="right")
        except Exception:  # noqa: BLE001 - widget is optional; never break the viewer
            logger.debug("Could not add deskew widget", exc_info=True)

    def _display_deskewed(self) -> None:
        self._apply_deskew(True)

    def _display_raw(self) -> None:
        self._apply_deskew(False)

    def _apply_deskew(self, on: bool) -> None:
        """Swap every layer between its raw and deskewed view."""
        if not self._deskew_available or bool(on) == self._deskew:
            return
        self._deskew = bool(on)
        self._reassign_layer_data()
        # The z axis changes meaning (scan <-> deskewed depth); reset follow bookkeeping.
        self._follow_target = None
        self._viewer.dims.axis_labels = _AXIS_LABELS
        self._refresh_layers()
        logger.info("Deskew display %s", "ON" if self._deskew else "OFF")

    def _on_geometry_changed(self) -> None:
        """Rebuild deskewed views from the edited angle / pixel size / scan step."""
        if not self._deskew_available or self._controls is None:
            return
        try:
            self._build_deskew_arrays(*self._geometry())
            if self._deskew:  # currently showing deskew -> swap to the rebuilt arrays
                self._reassign_layer_data()
                self._follow_target = None
                self._refresh_layers()
            logger.info(
                "Deskew geometry: angle %.2f°, pixel %.4f um, scan %.4f um -> %s",
                self._controls.angle,
                self._controls.pixel_size,
                self._controls.scan_step,
                self._projector.output_shape,
            )
        except Exception:  # noqa: BLE001 - a bad value must not break the viewer
            logger.debug("Deskew geometry update failed (ignored)", exc_info=True)

    # -- following --------------------------------------------------------------

    def _connect_follow_controls(self) -> None:
        """Wire up auto-advance pause (user scrub) and resume (Home button)."""
        self._last_step = tuple(self._viewer.dims.current_step)
        try:
            self._viewer.dims.events.current_step.connect(self._on_dims_step_changed)
        except Exception:  # noqa: BLE001
            logger.debug("Could not connect dims event; auto-advance pause disabled.")
        try:
            button = self._viewer.window._qt_viewer.viewerButtons.resetViewButton
            button.clicked.connect(self._on_home_clicked)
        except Exception:  # noqa: BLE001
            logger.debug("Could not connect Home button; auto-advance resume disabled.")

    def _on_dims_step_changed(self, *_: object) -> None:
        """Pause auto-advance when the user moves a followed axis (position or t).

        Moving z alone does not pause -- z is user-controlled. Our own writes (guarded
        by ``_programmatic``) never pause.
        """
        new = tuple(self._viewer.dims.current_step)
        previous = self._last_step
        self._last_step = new
        if self._programmatic or not self._following or previous is None:
            return
        moved = any(
            a < len(new) and a < len(previous) and new[a] != previous[a] for a in _FOLLOW_AXES
        )
        if moved:
            self._following = False
            logger.info("Auto-advance paused; press Home in the viewer to resume following.")

    def _on_home_clicked(self, *_: object) -> None:
        """The Home button resumes auto-advance and snaps to the newest complete volume."""
        self._following = True
        target = self._store.latest_complete if self._store is not None else None
        if target is not None:
            self._follow_target = target
            self._set_step(target)
        logger.info("Auto-advance resumed.")

    def _follow_latest(self) -> None:
        target = self._store.latest_complete if self._store is not None else None
        if target is None or target == self._follow_target:
            return
        self._follow_target = target
        self._set_step(target)

    def _set_step(self, target: tuple[int, int]) -> None:
        """Advance the followed sliders (position, t) to ``target``, preserving z.

        Does not trip the pause logic (writes are made under ``_programmatic``).
        """
        step = list(self._viewer.dims.current_step)
        for axis in _FOLLOW_AXES:
            if axis < len(step) and axis < len(target):
                step[axis] = target[axis]
        self._programmatic = True
        try:
            self._viewer.dims.current_step = tuple(step)
        finally:
            self._programmatic = False

    # -- painting ---------------------------------------------------------------

    def _refresh_layers(self) -> None:
        try:
            for layer in self._layers:
                layer.refresh()
        except Exception:  # noqa: BLE001 - a stale/closed layer must not kill the loop
            logger.debug("Layer refresh failed (ignored)", exc_info=True)

    def _autoset_contrast(self) -> None:
        """Set each channel's contrast limits from its first real data, once.

        Avoids the all-zeros default range (which renders typical low-intensity frames
        as black) without scanning the whole, mostly-empty array.
        """
        store = self._store
        if store is None or store.latest_complete is None:
            return
        position, t = store.latest_complete
        for channel in range(len(self._layers)):
            if channel in self._contrast_done:
                continue
            try:
                plane = store.plane(position, t, channel, store.n_z // 2)
                low, high = float(plane.min()), float(plane.max())
                if high > low:
                    self._layers[channel].contrast_limits = (low, high)
                    self._contrast_done.add(channel)
            except Exception:  # noqa: BLE001
                logger.debug("Auto-contrast failed (ignored)", exc_info=True)
                self._contrast_done.add(channel)


# Channel-name substring -> napari colormap. First match wins; default is "gray".
_CHANNEL_COLORMAPS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("gfp", "fitc"), "green"),
    (("mcherry", "rhodamine"), "magenta"),
    (("dapi",), "bop blue"),
)


def _colormap_for_channel(name: str) -> str:
    """Pick a napari colormap from the channel name (case-insensitive substring match)."""
    lowered = name.lower()
    for keys, colormap in _CHANNEL_COLORMAPS:
        if any(key in lowered for key in keys):
            return colormap
    return "gray"


def _default_contrast_limits(dtype: np.dtype) -> tuple[float, float]:
    """Pick contrast limits without scanning the (mostly empty) array."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return (float(info.min), float(info.max))
    return (0.0, 1.0)


def run_viewer(
    path: str | Path | None = None,
    control: mp.Queue | None = None,
    *,
    deskew: bool = False,
    cache_mb: float = DEFAULT_CACHE_MB,
    refresh_ms: int = DEFAULT_REFRESH_MS,
) -> None:
    """Open napari on an acquisition store and follow it until the window closes.

    Parameters
    ----------
    path : str | Path | None
        The ``.ome.zarr`` to view. May be None when ``control`` will deliver it -- a
        live acquisition names its store only once it starts, and the window is opened
        before then so napari's slow import overlaps with the first frames.
    control : multiprocessing.Queue | None
        Lifecycle messages from the acquisition process: ``{"kind": "open", "path":
        ...}`` and ``{"kind": "finish"}``. None for offline viewing.
    deskew : bool
        Offer deskewed display (oblique-plane microscopes such as mantis).
    cache_mb, refresh_ms
        Volume cache budget and how often the store is re-read while acquiring.

    Imports napari lazily so this module stays importable in the acquisition process,
    which has no napari/Qt dependency.
    """
    # A spawned child has no logging handlers; give it a basic console one.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - [viewer] %(levelname)s - %(message)s"
    )
    try:
        import napari

        from qtpy.QtCore import QTimer
    except Exception:  # noqa: BLE001
        logger.exception(
            "napari is not installed; cannot launch viewer. "
            "Install it with: pip install 'shrimpy[viewer]'"
        )
        return

    viewer = napari.Viewer(title=_title(path) if path is not None else "shrimpy")
    state = _ViewerState(viewer, deskew=deskew, cache_mb=cache_mb)
    if path is not None:
        state.set_path(path)

    def _tick() -> None:
        try:
            _drain(control, state, viewer)
            state.tick()
        except Exception:  # noqa: BLE001 - a bad tick must not stop the viewer
            logger.debug("Viewer refresh failed (ignored)", exc_info=True)

    timer = QTimer()
    timer.timeout.connect(_tick)
    timer.start(refresh_ms)

    try:
        napari.run()
    finally:
        timer.stop()


def _title(path: str | Path, status: str = "") -> str:
    return f"shrimpy — {Path(path).name}" + (f" ({status})" if status else "")


def _drain(control: mp.Queue | None, state: _ViewerState, viewer: object) -> None:
    """Apply any pending lifecycle messages from the acquisition process."""
    if control is None:
        return
    while True:
        try:
            message = control.get_nowait()
        except _queue.Empty:
            return
        kind = message.get("kind")
        if kind == "open":
            state.set_path(message["path"])
            viewer.title = _title(message["path"], "acquiring")
        elif kind == "finish":
            viewer.title = str(viewer.title).replace("(acquiring)", "(finished)")
