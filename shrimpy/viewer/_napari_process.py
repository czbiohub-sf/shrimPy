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

import itertools
import logging
import multiprocessing as mp
import queue as _queue

import numpy as np

from shrimpy.viewer.ring_buffer import RingBuffer

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


class LazyRingArray:
    """A read-only, numpy-like array view over the ring for a single channel.

    Advertises shape ``(*index_sizes, *frame_shape)`` but resolves each requested plane
    on demand via ``index_map``: present frames are copied out of the ring; missing ones
    return zeros.

    Parameters
    ----------
    ring : RingBuffer
        The shared ring holding pixel data.
    channel : int
        The channel index this array represents.
    index_sizes : tuple[int, ...]
        Sizes of the leading (p, t, z) axes.
    frame_shape : tuple[int, ...]
        Shape of a single (y, x) frame.
    index_map : dict
        Shared mapping ``(c, p, t, z) -> slot`` maintained by :class:`_ViewerState`.
    """

    def __init__(
        self,
        ring: RingBuffer,
        channel: int,
        index_sizes: tuple[int, ...],
        frame_shape: tuple[int, ...],
        index_map: dict[tuple[int, ...], int],
    ) -> None:
        self._ring = ring
        self._channel = channel
        self._index_sizes = tuple(index_sizes)
        self._frame_shape = tuple(frame_shape)
        self._index_map = index_map
        self.shape = (*self._index_sizes, *self._frame_shape)
        self.dtype = ring.dtype
        self.ndim = len(self.shape)

    def __array__(self, dtype: object = None) -> np.ndarray:
        # napari occasionally coerces the array (e.g. to probe). Materialize the
        # current first plane only -- coercing the whole 5D volume would be huge.
        plane = self[tuple(0 for _ in self._index_sizes)]
        return np.asarray(plane, dtype=dtype) if dtype is not None else plane

    def __getitem__(self, key: object) -> np.ndarray:
        """Resolve an index with numpy semantics (int drops an axis, slice keeps it).

        napari's slicing keeps the non-displayed axes and projects (``np.mean``) over
        them, so we must preserve dimensionality exactly as a real ndarray would.
        """
        key = self._normalize_key(key)
        n_idx = len(self._index_sizes)
        lead, trail = key[:n_idx], key[n_idx:]

        # Per leading axis: the selected integer coordinates, and whether to keep the axis.
        per_axis: list[list[int]] = []
        keep: list[bool] = []
        for axis, k in enumerate(lead):
            size = self._index_sizes[axis]
            if isinstance(k, slice):
                per_axis.append(list(range(*k.indices(size))))
                keep.append(True)
            elif isinstance(k, (int, np.integer)):
                per_axis.append([int(k) % size])
                keep.append(False)
            else:  # array-like fancy index
                per_axis.append([int(v) % size for v in np.atleast_1d(k)])
                keep.append(True)

        # Fetch each selected plane (real or blank), applying the trailing y/x index.
        planes = []
        for combo in itertools.product(*per_axis):
            slot = self._index_map.get((self._channel, *combo))
            plane = (
                self._ring.read(slot)
                if slot is not None
                else np.zeros(self._frame_shape, dtype=self.dtype)
            )
            planes.append(plane[trail] if trail else plane)

        plane_shape = planes[0].shape
        full_lead_shape = tuple(len(c) for c in per_axis)
        stacked = np.stack(planes).reshape(*full_lead_shape, *plane_shape)
        # Drop axes that were indexed with an integer (matching numpy semantics).
        dropper = tuple(slice(None) if keep[a] else 0 for a in range(n_idx))
        return stacked[dropper]

    def _normalize_key(self, key: object) -> tuple:
        """Expand to a full-ndim tuple, resolving Ellipsis and padding trailing axes."""
        if not isinstance(key, tuple):
            key = (key,)
        if any(k is Ellipsis for k in key):
            i = next(j for j, k in enumerate(key) if k is Ellipsis)
            n_fill = self.ndim - (len(key) - 1)
            key = key[:i] + (slice(None),) * n_fill + key[i + 1 :]
        return key + (slice(None),) * (self.ndim - len(key))


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
        index_sizes = tuple(int(sizes.get(ax, 1)) for ax in _INDEX_AXES)
        frame_shape = tuple(msg["frame_shape"])
        dtype = np.dtype(msg["dtype"])
        self._channels = list(msg["channels"])
        self._ring = RingBuffer.attach(
            msg["shm_name"], int(msg["n_slots"]), frame_shape, dtype
        )
        self._slot_owner = [None] * int(msg["n_slots"])

        clim = _default_contrast_limits(dtype)
        for c, name in enumerate(self._channels):
            data = LazyRingArray(self._ring, c, index_sizes, frame_shape, self._index_map)
            layer = self._viewer.add_image(
                data,
                name=name,
                contrast_limits=clim,
                colormap=_colormap_for_channel(name),
                blending="additive" if len(self._channels) > 1 else "translucent",
            )
            self._layers.append(layer)
        # Label the leading sliders; y/x are the displayed image dims.
        self._viewer.dims.axis_labels = _AXIS_LABELS
        self._connect_follow_controls()
        logger.info("Viewer initialized: %d channel(s), sizes=%s", len(self._channels), sizes)

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
        """True once every channel has a frame at (position, t, z)."""
        return all((c, position, t, z) in self._index_map for c in range(len(self._channels)))

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

    def _on_frame(self, msg: dict) -> None:
        if not self._started or self._ring is None:
            return
        slot = int(msg["slot"])
        key = (int(msg["c"]), int(msg["position"]), int(msg["t"]), int(msg["z"]))

        # Evict whatever previously lived in this slot (the ring just overwrote it).
        old = self._slot_owner[slot]
        if old is not None and self._index_map.get(old) == slot:
            del self._index_map[old]
        self._index_map[key] = slot
        self._slot_owner[slot] = key

        channel, position, t, z = key
        self._autoset_contrast(channel, slot)

        # Auto-advance only to (position, t) where ALL channels already have data at the
        # displayed z. Channels are acquired as interleaved z-stacks (all DAPI z, then
        # all FITC z), so following the raw latest frame would jump to a position the
        # lagging channel hasn't reached yet -- showing it black. Gating on completeness
        # keeps every channel visible. z stays under the user's control.
        try:
            if self._following:
                z_axis = _INDEX_AXES.index("z")
                disp_z = int(self._viewer.dims.current_step[z_axis])
                if self._complete_at(position, t, disp_z):
                    self._follow_target = (position, t)
                    self._set_step((position, t))
            # Repaint so a frame written into the currently displayed plane shows up,
            # whether or not we advanced.
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
