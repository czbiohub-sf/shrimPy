"""The viewer child process: a napari window backed by a shared-memory ring.

This module runs in a *separate process* from the acquisition (launched by
:class:`shrimpy.viewer.feeder.ViewerFeeder`). It must therefore be importable without
napari installed -- all napari/Qt imports are deferred into :func:`run_viewer` -- so that
the acquisition process can import the feeder regardless.

Each channel becomes its own napari image layer, backed by a :class:`LazyRingArray` that
advertises the full ``(p, t, z, y, x)`` shape but only holds the most-recent frames in the
ring. Indices that have been evicted (or not yet acquired) render as blank planes, giving a
"follow acquisition" experience with bounded RAM and limited scrollback.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue as _queue

import numpy as np

from napari_deskew_preview import (
    LS_ANGLE_DEG,
    PIXEL_SIZE_UM,
    LazyPlaneArray,
    deskewed_layer,
)

from shrimpy.viewer.ring_buffer import RingBuffer

# Imported lazily inside _add_deskew_widget (needs Qt) to keep this module import-safe.

logger = logging.getLogger(__name__)

# Index axes, in the order they appear as leading dimensions of each layer.
# "position" folds stage positions and grid FOVs into a single slider.
_INDEX_AXES = ("position", "t", "z")

# Slider labels shown in napari (position axis shown simply as "p"), plus y/x.
_AXIS_LABELS = ("p", "t", "z", "y", "x")

# Leading axes the viewer auto-advances. z is deliberately excluded: it stays under
# the user's control, so scrubbing z never pauses following and following preserves it.
_FOLLOW_AXES = tuple(i for i, ax in enumerate(_INDEX_AXES) if ax != "z")

# How often (ms) the viewer drains the coordinate queue and repaints.
_DRAIN_INTERVAL_MS = 100

# Drain at most this many messages per timer tick to keep the GUI responsive.
_QUEUE_DRAIN_BATCH = 512


class LazyRingArray(LazyPlaneArray):
    """Raw, read-only view over the ring for a single channel.

    Advertises shape ``(position, t, z_scan, y, x)``; each plane is the frame for
    ``(channel, position, t, z_scan)`` copied from the ring, or zeros if not (yet) present.
    Deskewed display is handled separately by :class:`napari_deskew_preview.DeskewedArray`.
    """

    def __init__(
        self,
        ring: RingBuffer,
        channel: int,
        index_sizes: tuple[int, ...],
        plane_shape: tuple[int, ...],
        index_map: dict[tuple[int, ...], int],
        dtype: np.dtype | None = None,
    ) -> None:
        self._ring = ring
        self._channel = channel
        self._index_sizes = tuple(index_sizes)
        self._frame_shape = tuple(plane_shape)
        self._index_map = index_map
        self.dtype = np.dtype(dtype) if dtype is not None else ring.dtype
        self._init_shape()

    def _plane(self, position: int, t: int, z: int) -> np.ndarray:
        slot = self._index_map.get((self._channel, position, t, z))
        if slot is None:
            return np.zeros(self._frame_shape, dtype=self.dtype)
        return self._ring.read(slot)


class _ViewerState:
    """Owns the napari layers and applies incoming coordinate messages."""

    def __init__(self, viewer: object) -> None:
        self._viewer = viewer
        self._ring: RingBuffer | None = None
        self._channels: list[str] = []
        # (c, p, t, z) -> slot, kept exactly in sync with ring contents.
        self._index_map: dict[tuple[int, ...], int] = {}
        # slot -> the key currently stored there, for precise eviction on overwrite.
        self._slot_owner: list[tuple[int, ...] | None] = []
        self._layers: list[object] = []
        # Channels whose contrast limits have been auto-set from their first frame.
        self._contrast_done: set[int] = set()
        # Auto-advance state. Sliders follow the latest frame until the user scrubs
        # away (any manual current_step change pauses it); the Home button resumes it.
        self._following = True
        # Guards current_step writes we make ourselves, so they aren't mistaken for
        # user scrubbing.
        self._programmatic = False
        # (position, t) we last auto-advanced to -- the latest coordinate where ALL
        # channels have data; also where Home snaps back to.
        self._follow_target: tuple[int, ...] | None = None
        # Last current_step we observed, to tell which axis a user change touched.
        self._last_step: tuple[int, ...] | None = None
        # Deskew state. Both raw and deskewed array views are built up front (lazy, cheap)
        # and the Deskew widget swaps which one backs each layer at runtime.
        self._deskew = False
        self._deskew_available = False
        self._projector: object | None = None
        self._n_zscan = 0
        self._n_channels = 0
        self._n_position = 1
        self._n_t = 1
        self._frame_shape: tuple[int, ...] = ()
        self._raw_arrays: list[object] = []
        self._deskew_arrays: list[object] = []
        self._controls: object | None = None  # shared DeskewControls (built in the widget)
        self._started = False

    def handle(self, msg: dict) -> None:
        kind = msg.get("kind")
        if kind == "frame":
            self._on_frame(msg)
        elif kind == "start":
            self._on_start(msg)
        elif kind == "finish":
            self._on_finish()

    def _on_start(self, msg: dict) -> None:
        if self._started:
            return
        self._started = True
        sizes = msg["sizes"]
        frame_shape = tuple(msg["frame_shape"])
        raw_dtype = np.dtype(msg["dtype"])
        self._channels = list(msg["channels"])
        self._n_channels = len(self._channels)
        self._ring = RingBuffer.attach(
            msg["shm_name"], int(msg["n_slots"]), frame_shape, raw_dtype
        )
        self._slot_owner = [None] * int(msg["n_slots"])

        self._n_position = int(sizes.get("position", 1))
        self._n_t = int(sizes.get("t", 1))
        n_z = int(sizes.get("z", 1))
        self._n_zscan = n_z
        self._frame_shape = frame_shape
        scan_step_um = float(msg.get("scan_step_um", 0.0))

        # Raw views (always available).
        self._raw_arrays = [
            LazyRingArray(
                self._ring, c, (self._n_position, self._n_t, n_z), frame_shape, self._index_map
            )
            for c in range(self._n_channels)
        ]

        # Deskew is available when the microscope enables it (msg["deskew"], e.g. mantis)
        # and the data supports it (a scan step and a real z-stack). Built with the default
        # geometry; the Deskew widget can rebuild them with edited values.
        self._deskew_available = bool(msg.get("deskew")) and scan_step_um > 0 and n_z > 1
        if self._deskew_available:
            self._build_deskew_arrays(LS_ANGLE_DEG, PIXEL_SIZE_UM, scan_step_um)
            logger.info(
                "Deskew available: raw %s -> deskewed %s (scan step %.4f um)",
                (n_z, *frame_shape),
                self._projector.output_shape,
                scan_step_um,
            )

        # When deskew is available (e.g. mantis), start on the deskewed view; the widget's
        # "Display raw" button switches back.
        self._deskew = self._deskew_available
        init_arrays = self._deskew_arrays if self._deskew else self._raw_arrays
        clim = _default_contrast_limits(init_arrays[0].dtype)
        for c, name in enumerate(self._channels):
            layer = self._viewer.add_image(
                init_arrays[c],
                name=name,
                contrast_limits=clim,
                colormap=_colormap_for_channel(name),
                blending="additive" if self._n_channels > 1 else "translucent",
            )
            self._layers.append(layer)
        # Label the leading sliders; y/x are the displayed image dims.
        self._viewer.dims.axis_labels = _AXIS_LABELS
        self._connect_follow_controls()
        # napari centers every slider by default; start following from p=0, t=0 instead.
        # Guarded (via _set_step) so it doesn't trip the auto-advance pause, and z is left
        # under the user's control.
        self._set_step((0, 0))
        self._last_step = tuple(self._viewer.dims.current_step)
        if self._deskew_available:
            self._add_deskew_widget(scan_step_um, (n_z, *frame_shape))
        logger.info("Viewer initialized: %d channel(s), sizes=%s", self._n_channels, sizes)

    def _ring_gather(self, channel: int):
        """A source gather for :func:`deskewed_layer`: one tilt row across the scan stack."""
        n_zscan = self._n_zscan

        def gather(leading: tuple[int, ...], tilt_row: int) -> np.ndarray:
            position, t = leading
            # Only complete volumes may be previewed: a deskewed plane mixes every z
            # slice, so a partial stack (still filling, or partly evicted) would render
            # as a corrupt half-deskewed image. Blank it until the whole stack is present.
            if not self._volume_complete(channel, position, t):
                return np.zeros((n_zscan, self._frame_shape[1]), dtype=self._ring.dtype)
            slots = [self._index_map.get((channel, position, t, zs)) for zs in range(n_zscan)]
            return self._ring.read_rows(slots, tilt_row)

        return gather

    def _build_deskew_arrays(self, angle: float, pixel: float, scan: float) -> None:
        """(Re)build the per-channel deskewed views for the given geometry."""
        arrays: list[object] = []
        for c in range(self._n_channels):
            arr, self._projector = deskewed_layer(
                self._ring_gather(c),
                raw_zyx_shape=(self._n_zscan, *self._frame_shape),
                scan_step_um=scan,
                batch_sizes=(self._n_position, self._n_t),
                ls_angle_deg=angle,
                pixel_size_um=pixel,
            )
            arrays.append(arr)
        self._deskew_arrays = arrays

    def _add_deskew_widget(self, scan_step_um: float, raw_shape: tuple[int, ...]) -> None:
        """Add the shared Deskew dock widget (Display deskewed / Display raw + geometry)."""
        try:
            from napari_deskew_preview._controls import DeskewControls

            self._controls = DeskewControls(scan_step_um=scan_step_um)
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
        arrays = self._deskew_arrays if self._deskew else self._raw_arrays
        for layer, array in zip(self._layers, arrays, strict=True):
            layer.data = array  # napari resets dims/extent to the new shape
        # The z axis changes meaning (scan <-> deskewed depth); reset follow bookkeeping.
        self._follow_target = None
        self._last_step = tuple(self._viewer.dims.current_step)
        self._viewer.dims.axis_labels = _AXIS_LABELS
        for layer in self._layers:
            layer.refresh()
        logger.info("Deskew display %s", "ON" if self._deskew else "OFF")

    def _on_geometry_changed(self) -> None:
        """Rebuild deskewed views from the edited angle / pixel size / scan step."""
        if not self._deskew_available or self._controls is None:
            return
        try:
            self._build_deskew_arrays(
                self._controls.angle, self._controls.pixel_size, self._controls.scan_step
            )
            if self._deskew:  # currently showing deskew -> swap to the rebuilt arrays
                for layer, array in zip(self._layers, self._deskew_arrays, strict=True):
                    layer.data = array
                self._follow_target = None
                self._last_step = tuple(self._viewer.dims.current_step)
                for layer in self._layers:
                    layer.refresh()
            logger.info(
                "Deskew geometry: angle %.2f°, pixel %.4f um, scan %.4f um -> %s",
                self._controls.angle,
                self._controls.pixel_size,
                self._controls.scan_step,
                self._projector.output_shape,
            )
        except Exception:  # noqa: BLE001 - bad value must not break the viewer
            logger.debug("Deskew geometry update failed (ignored)", exc_info=True)

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
        prev = self._last_step
        self._last_step = new
        if self._programmatic or not self._following or prev is None:
            return
        changed_followed = any(
            a < len(new) and a < len(prev) and new[a] != prev[a] for a in _FOLLOW_AXES
        )
        if changed_followed:
            self._following = False
            logger.info("Auto-advance paused; press Home in the viewer to resume following.")

    def _on_home_clicked(self, *_: object) -> None:
        """The Home button resumes auto-advance and snaps to the latest complete frame."""
        self._following = True
        if self._follow_target is not None:
            self._set_step(self._follow_target)
        logger.info("Auto-advance resumed.")

    def _complete_at(self, position: int, t: int, z: int) -> bool:
        """True once every channel has a frame at (position, t, z) (raw mode)."""
        return all((c, position, t, z) in self._index_map for c in range(self._n_channels))

    def _volume_complete(self, channel: int, position: int, t: int) -> bool:
        """True once every z slice of one channel's scan stack at (position, t) is present.

        Requires *all* z indices (not merely the last) so that neither a still-filling
        stack nor a partially-evicted one is ever treated as complete.
        """
        return all((channel, position, t, z) in self._index_map for z in range(self._n_zscan))

    def _set_step(self, target: tuple[int, ...]) -> None:
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

    def _evict_volume(self, channel: int, position: int, t: int) -> None:
        """Drop every slot of one channel's scan stack at (position, t) from the index.

        Called when the ring begins overwriting a volume: rather than let the buffer hold
        a half-evicted stack, we retire the whole volume at once. The not-yet-overwritten
        slots are simply released back to the index and reused as acquisition continues.
        """
        for z in range(self._n_zscan):
            k = (channel, position, t, z)
            s = self._index_map.pop(k, None)
            if s is not None and self._slot_owner[s] == k:
                self._slot_owner[s] = None

    def _on_frame(self, msg: dict) -> None:
        if not self._started or self._ring is None:
            return
        slot = int(msg["slot"])
        key = (int(msg["c"]), int(msg["position"]), int(msg["t"]), int(msg["z"]))

        # The ring just overwrote this slot. Evict the ENTIRE volume the old frame
        # belonged to -- not just that one frame -- so the buffer never holds a
        # half-overwritten stack (some z present, some gone). Whole volumes enter and
        # leave the buffer together.
        old = self._slot_owner[slot]
        if old is not None and self._index_map.get(old) == slot:
            self._evict_volume(old[0], old[1], old[2])
        self._index_map[key] = slot
        self._slot_owner[slot] = key

        channel, position, t, z = key
        self._autoset_contrast(channel, slot)

        # Auto-advance follows the latest coordinate whose displayed plane is ready; z
        # stays under the user's control.
        try:
            if self._deskew:
                # Reveal each channel the moment its OWN scan stack is complete -- a slow
                # channel must not hold up the ones already acquired (channels arrive as
                # sequential z-stacks: all of ch0's z, then all of ch1's z). Refreshing
                # just the completed layer repaints it even when the position slider does
                # not move -- e.g. the very first stack, while we are parked at p=0, t=0.
                if self._volume_complete(channel, position, t):
                    if self._following:
                        self._follow_target = (position, t)
                        self._set_step((position, t))
                    self._layers[channel].refresh()
            else:
                if self._following:
                    disp_z = int(self._viewer.dims.current_step[_INDEX_AXES.index("z")])
                    if self._complete_at(position, t, disp_z):
                        self._follow_target = (position, t)
                        self._set_step((position, t))
                # Repaint so a frame written into the currently displayed plane shows up.
                for layer in self._layers:
                    layer.refresh()
        except Exception:  # noqa: BLE001 - a stale/closed layer must not kill the loop
            logger.debug("Layer refresh failed (ignored)", exc_info=True)

    def _autoset_contrast(self, channel: int, slot: int) -> None:
        """Set a channel's contrast limits from its first real frame, once.

        Avoids the all-zeros default range (which renders typical low-intensity frames
        as black) without scanning the whole mostly-empty array.
        """
        if channel in self._contrast_done or self._ring is None:
            return
        if not (0 <= channel < len(self._layers)):
            return
        try:
            frame = self._ring.read(slot)
            lo, hi = float(frame.min()), float(frame.max())
            if hi > lo:
                self._layers[channel].contrast_limits = (lo, hi)
            self._contrast_done.add(channel)
        except Exception:  # noqa: BLE001
            logger.debug("Auto-contrast failed (ignored)", exc_info=True)

    def _on_finish(self) -> None:
        try:
            self._viewer.title = "shrimpy — acquisition finished"
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        if self._ring is not None:
            self._ring.close()
            self._ring = None


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


def run_viewer(queue: mp.Queue) -> None:
    """Child-process entry point: open napari and drain ``queue`` until the window closes.

    Imports napari lazily so this module stays importable in the acquisition process,
    which has no napari/Qt dependency.
    """
    # The child is spawned fresh with no logging handlers; give it a basic console one.
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

    viewer = napari.Viewer(title="shrimpy — live acquisition")
    state = _ViewerState(viewer)

    def _drain() -> None:
        for _ in range(_QUEUE_DRAIN_BATCH):
            try:
                msg = queue.get_nowait()
            except _queue.Empty:
                break
            try:
                state.handle(msg)
            except Exception:  # noqa: BLE001 - one bad message must not stop the viewer
                logger.debug("Error handling viewer message (ignored)", exc_info=True)

    timer = QTimer()
    timer.timeout.connect(_drain)
    timer.start(_DRAIN_INTERVAL_MS)

    try:
        napari.run()
    finally:
        timer.stop()
        state.close()
