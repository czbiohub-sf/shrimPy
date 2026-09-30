"""Read an acquisition's OME-Zarr -- including one that is still being written.

:class:`AcquisitionStore` opens the store produced by
:meth:`~shrimpy.engines.base_engine.BaseEngine.acquire` **read-only** and serves image
planes from it. Because acquire-zarr no longer holds ``zarr.json`` locked while
streaming (acquire-zarr#234), the same store can be read while the acquisition writes
to it: a live acquisition and a finished dataset go through exactly the same code path,
and the viewer shows the *whole* acquisition rather than the tail of it.

Layout, ordering, channel names, axes and scales come from :func:`iohub.open_ome_zarr`,
which already understands every layout ome-writers writes (a single image, a
bioformats2raw series list, an HCS plate) and re-reads an array's metadata on each
``Position.data`` access, so a growing store needs no special handling. What this class
adds on top is what reading a *live* store needs:

- **Volumes, not planes.** shrimPy chunks a whole z-stack into one chunk, so reading a
  single plane decompresses the entire stack -- ~275 ms for a mantis-sized volume,
  every time. :class:`AcquisitionStore` therefore caches decompressed ``(z, y, x)``
  volumes under a byte budget, which is what makes scrubbing z (and deskew, where each
  displayed plane mixes all of them) interactive rather than seconds per frame.
- **Knowing what is actually readable.** Chunks are flushed by a writer thread once
  complete, so "the frame was acquired" is not "the frame is readable", and an
  unflushed chunk reads back as zeros. :meth:`volume_ready` settles that from the files
  on disk, which both blanks half-written stacks instead of deskewing garbage and keeps
  a blank volume from being cached for the rest of the session.
- **Positions that do not exist yet.** acquire-zarr creates a position's array only
  when that position is first written, so ``Position.data`` raises until then; such a
  position reads as blank and is picked up by :meth:`refresh`.
- **Following the acquisition.** :attr:`latest_complete` tracks the newest volume whose
  chunks are all on disk, which is the coordinate the viewer parks on.
"""

from __future__ import annotations

import itertools
import warnings

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from iohub import open_ome_zarr
from iohub.ngff import Position

# Resolution level within each image group; iohub's ``Position.data`` uses the same.
FULL_RESOLUTION = "0"

# Volume cache budget (MB). Deliberately not a CLI option: every plane is readable from
# disk at any index, so this only trades RAM against re-decompressing a volume and
# changes nothing about what the viewer can show. Sized to hold the frame on screen --
# a deskewed plane mixes every channel's volume at the current (position, t) -- with
# room left to scrub between frames.
DEFAULT_CACHE_MB = 4096.0


class AcquisitionStore:
    """Read-only, refreshable view of one OME-Zarr acquisition.

    Parameters
    ----------
    path : str | Path
        The ``.ome.zarr`` store. It may still be open for writing by an acquisition.
    cache_mb : float
        Budget for the decompressed-volume cache, in megabytes. At least one volume is
        always kept, however large.

    Notes
    -----
    Every read is bounded by what has actually been acquired: indices past the written
    extent, and chunks that have not been flushed, come back as zeros rather than
    raising, so the viewer can advertise the whole array and let it fill in.
    """

    def __init__(self, path: str | Path, *, cache_mb: float = DEFAULT_CACHE_MB) -> None:
        self.path = Path(path)
        node = open_ome_zarr(self.path, mode="r")
        # A single-position acquisition *is* the image; everything else iterates its
        # positions in acquisition order.
        images = [(".", node)] if isinstance(node, Position) else list(node.positions())
        if not images:
            raise ValueError(f"No images found in {self.path}")
        self.position_names = tuple(name for name, _ in images)
        self._images = [image for _, image in images]
        self._array_paths = [_array_path(image) for image in self._images]
        self._arrays = [self._fetch(i) for i in range(len(self._images))]
        # (mtime_ns, size) of each array's zarr.json, so refresh() only re-reads the
        # metadata of arrays that actually changed.
        self._meta_keys: list[object] = [None] * len(self._arrays)

        first = self._images[0]
        self.axes: tuple[str, ...] = tuple(axis.name for axis in first.axes)
        self._ax = {name: i for i, name in enumerate(self.axes)}
        self.channels: tuple[str, ...] = tuple(first.channel_names)
        # Only the scan step is taken from the store. The lateral scale is *not*: MM
        # resolves getPixelSizeUm() from whichever pixel-size config currently matches
        # the device properties, which has silently changed the deskew ratio between
        # otherwise-identical runs (see BaseEngine._setup_dynatrack). The deskew
        # default, and the widget, are the reliable source for it.
        self.scan_step_um = _recorded_scale(first.scale, self._ax.get("z"))

        # Every position shares one shape and chunk grid (ome-writers builds them from a
        # single image model) and only the outermost axis grows -- so read them once
        # from whichever position has been written, and let the rest fill in later.
        template = next((a for a in self._arrays if a is not None), None)
        if template is None:
            raise ValueError(f"{self.path} has no image data yet")
        self.frame_shape: tuple[int, int] = (template.shape[-2], template.shape[-1])
        self.dtype = np.dtype(template.dtype)
        self.n_channels = self._axis_size(template, "c")
        self.n_z = self._axis_size(template, "z")

        self._cache: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()
        self._cache_budget = int(cache_mb * 1e6)
        self._cache_bytes = 0
        # Volumes whose chunks are all on disk. Monotonic: files never go away.
        self._ready: set[tuple[int, int, int]] = set()
        # Highest timepoint known complete per position, and the position that reached
        # its latest one most recently -- the coordinate the viewer follows. Seeded one
        # behind the written extent so that the newest timepoint, the one that may
        # still be mid-flush, is always verified by the first refresh(); everything
        # before it the writer has long since moved past.
        self._frontier = [max(-1, self.t_extent(p) - 2) for p in range(self.n_positions)]
        self._latest: tuple[int, int] | None = _argmax_frontier(self._frontier)

    # -- shape ------------------------------------------------------------------

    @property
    def n_positions(self) -> int:
        return len(self._arrays)

    @property
    def n_timepoints(self) -> int:
        """Timepoints written at *any* position, so the slider spans the whole run."""
        return max(1, *(self.t_extent(p) for p in range(self.n_positions)))

    @property
    def index_sizes(self) -> tuple[int, int, int]:
        """Leading sizes of a per-channel raw layer: ``(position, t, z)``."""
        return (self.n_positions, self.n_timepoints, self.n_z)

    def t_extent(self, position: int) -> int:
        """Timepoints written so far at ``position``; 0 if it has not been visited."""
        array = self._arrays[position]
        if array is None:
            return 0
        return array.shape[self._ax["t"]] if "t" in self._ax else 1

    def _axis_size(self, array: Any, axis: str) -> int:
        """Size of a named axis, or 1 for an axis this store does not have."""
        i = self._ax.get(axis)
        return array.shape[i] if i is not None else 1

    # -- refresh ----------------------------------------------------------------

    def refresh(self) -> bool:
        """Re-read the array metadata; return whether anything new became visible.

        Cheap when nothing changed: each position costs one ``stat`` of its
        ``zarr.json``, and only arrays whose metadata was rewritten are re-read.
        """
        changed = False
        for i in range(self.n_positions):
            key = self._meta_key(i)
            if key is not None and key == self._meta_keys[i]:
                continue
            self._meta_keys[i] = key
            extent = self.t_extent(i)
            self._arrays[i] = self._fetch(i)
            changed |= self.t_extent(i) != extent

        # Walk each position's completed-volume frontier forward. Only positions with
        # an unconfirmed timepoint are probed, which in steady state is the one or two
        # currently being written.
        for i in range(self.n_positions):
            t = self._frontier[i]
            while t + 1 < self.t_extent(i) and self._all_channels_ready(i, t + 1):
                t += 1
            if t != self._frontier[i]:
                self._frontier[i] = t
                self._latest = (i, t)
                changed = True
        return changed

    @property
    def latest_complete(self) -> tuple[int, int] | None:
        """The ``(position, t)`` most recently completed -- what the viewer follows."""
        return self._latest

    def _fetch(self, position: int):
        """This position's array, re-read from the store, or None if not written yet."""
        try:
            with warnings.catch_warnings():
                # iohub builds its KeyError message from Group.array_keys(), which
                # walks the group -- and zarr warns there about an array directory
                # acquire-zarr has created but not yet given a zarr.json. That is the
                # very state this method exists to report, not something to print.
                warnings.simplefilter("ignore")
                return self._images[position].data
        except KeyError:
            # acquire-zarr has not created this position's array; nothing acquired here.
            return None

    def _meta_key(self, position: int) -> object | None:
        """Change key for a position's array metadata, or None if it can't be stat'd."""
        try:
            st = (self.path / self._array_paths[position] / "zarr.json").stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    # -- reading ----------------------------------------------------------------

    def plane(self, position: int, t: int, channel: int, z: int) -> np.ndarray:
        """One ``(y, x)`` plane, or zeros where nothing has been acquired yet."""
        if t >= self.t_extent(position) or z >= self.n_z:
            return np.zeros(self.frame_shape, dtype=self.dtype)
        volume = self._cache_get((position, t, channel))
        if volume is None:
            if self.volume_ready(position, t, channel):
                volume = self._load_volume(position, t, channel)
            else:
                # Mid-flush: read the single plane rather than pin a partial volume.
                return self._read(position, t, channel, z)
        return volume[z]

    def tilt_row(self, position: int, t: int, channel: int, row: int) -> np.ndarray | None:
        """One tilt row across the whole scan stack, ``(n_z, x)``, for deskew.

        Returns None until every chunk of the volume is on disk: a deskewed plane mixes
        all z slices, so a half-written stack would render as a corrupt image.
        """
        if not self.volume_ready(position, t, channel):
            return None
        volume = self._cache_get((position, t, channel))
        if volume is None:
            volume = self._load_volume(position, t, channel)
        return volume[:, row, :]

    def volume_ready(self, position: int, t: int, channel: int) -> bool:
        """True once every chunk backing the ``(position, t, channel)`` stack exists.

        Chunks appear only when acquire-zarr flushes them, which lags the frames by up
        to one chunk, so "the frame was acquired" is not "the frame is readable".
        Once true this stays true, so a volume is only ever checked until it lands.
        """
        key = (position, t, channel)
        if key in self._ready:
            return True
        array = self._arrays[position]
        if array is None or t >= self.t_extent(position):
            return False
        if self.path.is_dir():
            base = self.path / self._array_paths[position]
            encode = array.metadata.chunk_key_encoding.encode_chunk_key
            for coord in self._chunk_coords(position, t, channel):
                if not (base / encode(coord)).exists():
                    return False
        # Otherwise the store is not on this filesystem and its chunk files cannot be
        # stat'd, so the written extent is all there is to go on. shrimPy writes local
        # stores; this only keeps a remote path degrading gracefully.
        self._ready.add(key)
        return True

    def _all_channels_ready(self, position: int, t: int) -> bool:
        return all(self.volume_ready(position, t, c) for c in range(self.n_channels))

    def _chunk_coords(self, position: int, t: int, channel: int):
        """Chunk-grid coordinates of every chunk holding this volume."""
        array = self._arrays[position]
        grid = array.shards or array.chunks
        per_axis = []
        for axis, name in enumerate(self.axes):
            if name == "t":
                per_axis.append([t // grid[axis]])
            elif name == "c":
                per_axis.append([channel // grid[axis]])
            else:
                per_axis.append(range(-(-array.shape[axis] // grid[axis])))
        return itertools.product(*per_axis)

    def _index(self, t: int, channel: int, z: int | slice) -> tuple:
        """Index tuple for one plane or one whole stack, in the store's axis order."""
        index: list[Any] = []
        for name in self.axes[:-2]:
            if name == "t":
                index.append(t)
            elif name == "c":
                index.append(channel)
            elif name == "z":
                index.append(z)
            else:  # an axis this viewer does not surface; show its first entry
                index.append(0)
        return tuple(index)

    def _read(self, position: int, t: int, channel: int, z: int | slice) -> np.ndarray:
        array = self._arrays[position]
        return np.asarray(array[self._index(t, channel, z)])

    def _load_volume(self, position: int, t: int, channel: int) -> np.ndarray:
        """Decompress a whole ``(z, y, x)`` stack and add it to the cache."""
        volume = self._read(position, t, channel, slice(None))
        if volume.ndim == 2:  # no z axis in this store
            volume = volume[np.newaxis]
        self._cache_put((position, t, channel), volume)
        return volume

    def _cache_get(self, key: tuple[int, int, int]) -> np.ndarray | None:
        """A cached volume, marked as the most recently used."""
        volume = self._cache.get(key)
        if volume is not None:
            self._cache.move_to_end(key)
        return volume

    def _cache_put(self, key: tuple[int, int, int], volume: np.ndarray) -> None:
        self._cache[key] = volume
        self._cache_bytes += volume.nbytes
        while len(self._cache) > 1 and self._cache_bytes > self._cache_budget:
            _, evicted = self._cache.popitem(last=False)
            self._cache_bytes -= evicted.nbytes


def _array_path(image: Position) -> str:
    """Path of an image's full-resolution array, relative to the store root."""
    group = image.zgroup.path
    return f"{group}/{FULL_RESOLUTION}" if group else FULL_RESOLUTION


def _recorded_scale(scales: list[float], axis: int | None) -> float | None:
    """A physical scale recorded for an axis, or None if the writer left it at zero."""
    if axis is None or axis >= len(scales):
        return None
    return scales[axis] or None


def _argmax_frontier(frontier: list[int]) -> tuple[int, int] | None:
    """The ``(position, t)`` with the highest completed timepoint, if any."""
    if not frontier:
        return None
    position = max(range(len(frontier)), key=frontier.__getitem__)
    return (position, frontier[position]) if frontier[position] >= 0 else None
