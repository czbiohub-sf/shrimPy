"""Base class for read-only, napari-friendly array-likes resolved one plane at a time.

napari slices a layer by indexing its data, keeping the non-displayed axes and projecting
(``np.mean``) over them. So a lazy data source must implement numpy-consistent fancy
indexing (int drops an axis, slice keeps it). This base provides that machinery; subclasses
only implement ``_plane(*leading) -> 2D ndarray`` and declare their axis sizes/dtype.
"""

from __future__ import annotations

import itertools

import numpy as np


class LazyPlaneArray:
    """Resolve an N-D array on demand, one (y, x) plane per leading-index combination.

    Subclasses must set ``self._index_sizes`` (leading axis sizes), ``self._frame_shape``
    (plane shape), and ``self.dtype``, then call ``self._init_shape()``; and implement
    ``_plane(*leading) -> np.ndarray`` returning the 2D plane for those leading indices.
    """

    _index_sizes: tuple[int, ...]
    _frame_shape: tuple[int, ...]
    dtype: np.dtype

    def _init_shape(self) -> None:
        self.shape = (*self._index_sizes, *self._frame_shape)
        self.ndim = len(self.shape)

    def _plane(self, *leading: int) -> np.ndarray:
        raise NotImplementedError

    def __array__(self, dtype: object = None) -> np.ndarray:
        # napari occasionally coerces the array (e.g. to probe). Materialize the first
        # plane only -- coercing the whole volume would be huge.
        plane = self[tuple(0 for _ in self._index_sizes)]
        return np.asarray(plane, dtype=dtype) if dtype is not None else plane

    def __getitem__(self, key: object) -> np.ndarray:
        """Index with numpy semantics (int drops an axis, slice keeps it)."""
        key = self._normalize_key(key)
        n_idx = len(self._index_sizes)
        lead, trail = key[:n_idx], key[n_idx:]

        # Per leading axis: the selected coordinates, and whether to keep the axis.
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

        planes = []
        for combo in itertools.product(*per_axis):
            plane = self._plane(*combo)
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
