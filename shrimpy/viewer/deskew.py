"""On-demand single-plane deskew for oblique-plane light-sheet display.

Deskew is a pure affine (permute + flip + shear + anisotropic scale) in which only
the scan axis needs fractional interpolation -- see biahub's ``fast_deskew_zyx`` /
``_build_deskew_grid``. That structure means a single *deskewed axial plane*
(``Z_out = j``) maps to one raw tilt-row of the volume (``raw[:, n_tilt-1-j, :]``)
resampled in 1-D along the scan axis. So to show one plane we never materialize the
~1 GB deskewed volume: we gather ~3.5 MB (one tilt row across the scan stack) and do a
single vectorized 1-D interpolation (~ms, CPU).

The math here mirrors biahub exactly (linear interpolation, zero padding, the same
offset/grid), with ``average_n_slices = 1`` (no z-averaging for display).

Raw axis convention (matching biahub): ``(Z_scan, Y_tilt, X_cover)``.
Deskewed output: ``(Z_out, Y_out, X_out)`` where ``Z_out`` is normal to the coverslip
(from ``Y_tilt``), ``Y_out`` is ``X_cover``, and ``X_out`` is the scan direction.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from shrimpy.viewer._lazy_array import LazyPlaneArray

# A gather callable abstracts the raw source (ring, zarr, dask, numpy): given the leading
# indices (e.g. (position, t)) and a tilt-axis row, it returns that row across the whole
# scan stack, shape ``(n_scan, n_cover)`` -- i.e. ``raw[..., :, tilt_row, :]``.
Gather = Callable[[tuple[int, ...], int], np.ndarray]

# Fixed acquisition geometry for the mantis light-sheet arm.
LS_ANGLE_DEG = 30.0
PIXEL_SIZE_UM = 0.1133
KEEP_OVERHANG = True


class DeskewProjector:
    """Computes single deskewed planes on demand from raw scan frames.

    Parameters
    ----------
    raw_zyx_shape : tuple[int, int, int]
        Raw volume shape ``(n_scan, n_tilt, n_cover)`` = (Z_scan, Y_tilt, X_cover).
    scan_step_um : float
        Light-sheet scan step in micrometers (from ``MDASequence.z_plan.step``).
    """

    def __init__(self, raw_zyx_shape: tuple[int, int, int], scan_step_um: float) -> None:
        self.n_scan, self.n_tilt, self.n_cover = (int(v) for v in raw_zyx_shape)
        self.scan_step_um = float(scan_step_um)
        # px_to_scan_ratio = lateral pixel size / scan step (object space)
        self.ratio = PIXEL_SIZE_UM / self.scan_step_um
        self.ct = float(np.cos(np.deg2rad(LS_ANGLE_DEG)))

        # Output extents (no z-averaging): Z_out = n_tilt, Y_out = n_cover.
        self.z_out = self.n_tilt
        self.y_out = self.n_cover
        overhang = self.n_tilt * self.ct
        scan_extent = self.n_scan / self.ratio
        self.x_out = int(
            np.ceil(scan_extent + overhang if KEEP_OVERHANG else scan_extent - overhang)
        )

        # Scan-axis sampling position: in_z = ratio*x_out - ratio*ct*z_out + offset.
        # Split into the z_out-independent base (computed once) and the per-plane shift.
        self.offset = (
            self.ratio * self.ct * (self.z_out - 1) / 2
            - self.ratio * (self.x_out - 1) / 2
            + (self.n_scan - 1) / 2
        )
        self._s_base = self.ratio * np.arange(self.x_out, dtype=np.float32) + self.offset

    @property
    def output_shape(self) -> tuple[int, int, int]:
        """Deskewed volume shape ``(Z_out, Y_out, X_out)``."""
        return (self.z_out, self.y_out, self.x_out)

    @property
    def plane_shape(self) -> tuple[int, int]:
        """Shape of one deskewed axial plane ``(Y_out, X_out)``."""
        return (self.y_out, self.x_out)

    def tilt_row(self, z_out: int) -> int:
        """Raw tilt-axis row that maps to deskewed axial plane ``z_out`` (with flip)."""
        return self.n_tilt - 1 - int(z_out)

    def deskew_plane(self, scan_cover: np.ndarray, z_out: int) -> np.ndarray:
        """Deskew one axial plane.

        Parameters
        ----------
        scan_cover : np.ndarray
            The gathered tilt row across the scan stack, shape ``(n_scan, n_cover)``
            (i.e. ``raw[:, tilt_row(z_out), :]``).
        z_out : int
            Deskewed axial-plane index.

        Returns
        -------
        np.ndarray
            Deskewed plane, shape ``(Y_out, X_out)``, dtype float32.
        """
        # Flip the coverslip axis (X_cover -> Y_out) and interpolate along scan.
        cover_flipped = np.asarray(scan_cover, dtype=np.float32)[:, ::-1]
        s = self._s_base - self.ratio * self.ct * int(z_out)  # (X_out,)
        lower_idx = np.floor(s).astype(np.int64)
        weight = (s - lower_idx).astype(np.float32)

        # Linear interp between the two bracketing scan slices, with each out-of-range
        # neighbor contributing 0 (matches grid_sample padding_mode="zeros").
        last = self.n_scan - 1
        lo_ok = ((lower_idx >= 0) & (lower_idx <= last))[:, None]
        hi_ok = ((lower_idx + 1 >= 0) & (lower_idx + 1 <= last))[:, None]
        lower = cover_flipped[np.clip(lower_idx, 0, last)] * lo_ok
        upper = cover_flipped[np.clip(lower_idx + 1, 0, last)] * hi_ok
        plane_t = (1.0 - weight)[:, None] * lower + weight[:, None] * upper  # (X_out, n_cover)
        return plane_t.T  # (Y_out, X_out)


class DeskewedArray(LazyPlaneArray):
    """Source-agnostic lazy deskewed view over a raw oblique-plane volume.

    Presents shape ``(*batch_sizes, Z_out, Y_out, X_out)``; each axial plane is computed
    on demand from a ``gather`` callable via :class:`DeskewProjector`. The gather hides the
    source (live ring, on-disk zarr, dask, numpy), so this class -- and the deskew math --
    are identical for live and saved data.

    Parameters
    ----------
    gather : Gather
        ``gather(batch_index, tilt_row) -> (n_scan, n_cover)`` for the raw source.
    projector : DeskewProjector
        Supplies geometry and the per-plane deskew.
    batch_sizes : tuple[int, ...]
        Sizes of the leading batch axes (e.g. ``(n_position, n_t)``); ``Z_out`` is appended
        automatically so the array is ``(*batch_sizes, Z_out, Y_out, X_out)``.
    """

    def __init__(
        self,
        gather: Gather,
        projector: DeskewProjector,
        batch_sizes: tuple[int, ...],
    ) -> None:
        self._gather = gather
        self._projector = projector
        self._index_sizes = (*tuple(batch_sizes), projector.z_out)
        self._frame_shape = projector.plane_shape
        self.dtype = np.dtype(np.float32)
        self._init_shape()

    def _plane(self, *leading_z: int) -> np.ndarray:
        *batch, z_out = leading_z
        scan_cover = self._gather(tuple(batch), self._projector.tilt_row(z_out))
        return self._projector.deskew_plane(scan_cover, z_out)


def deskewed_layer(
    gather: Gather,
    raw_zyx_shape: tuple[int, int, int],
    scan_step_um: float,
    batch_sizes: tuple[int, ...] = (),
) -> tuple[DeskewedArray, DeskewProjector]:
    """Build a lazy deskewed array from a source-agnostic ``gather`` callable.

    ``raw_zyx_shape`` is the per-volume raw shape ``(n_scan, n_tilt, n_cover)``;
    ``batch_sizes`` are any leading axes over volumes (e.g. ``(n_position, n_t)``, or ``()``
    for a single volume). Returns the array (napari layer data) and the projector (geometry,
    e.g. ``output_shape``). See :func:`array_gather` for a plain array-like source.
    """
    projector = DeskewProjector(raw_zyx_shape, scan_step_um)
    return DeskewedArray(gather, projector, batch_sizes), projector


def array_gather(raw: object) -> Gather:
    """A :data:`Gather` for a plain array-like indexed as ``(*leading, z_scan, y, x)``.

    Suitable for saved data (zarr/dask/numpy), where the slice ``raw[..., :, row, :]`` is
    itself lazy/efficient. Used by tests and a future saved-data deskew path.
    """

    def gather(leading: tuple[int, ...], tilt_row: int) -> np.ndarray:
        return np.asarray(raw[(*leading, slice(None), tilt_row, slice(None))])

    return gather
