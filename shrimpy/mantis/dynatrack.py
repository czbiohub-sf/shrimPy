"""DynaTrack — neuromast tracking for zebrafish embryo growth compensation.

Tracks neuromast motion across timepoints by comparing acquired z-stacks
to reference images and computing X/Y/Z shifts. Plugs into the position
update infrastructure via the PositionUpdater interface.
"""

from __future__ import annotations

import logging

import numpy as np

from shrimpy.mantis.position_update import PositionCoordinates, PositionUpdater

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions ported from archive/pycromanager/autotracker.py
# ---------------------------------------------------------------------------


def _next_fast_len(n: int) -> int:
    """Return the smallest integer >= *n* whose prime factors are only 2, 3, 5.

    This mirrors ``scipy.fftpack.next_fast_len`` without requiring scipy.
    """
    if n <= 1:
        return 1
    # Brute-force: increment until we find a 5-smooth number
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def _center_crop(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Crop the center of *arr* to *shape*."""
    assert arr.ndim == len(shape)
    starts = tuple((cur_s - s) // 2 for cur_s, s in zip(arr.shape, shape, strict=True))
    assert all(s >= 0 for s in starts)
    slicing = tuple(slice(s, s + d) for s, d in zip(starts, shape, strict=True))
    return arr[slicing]


def _pad_to_shape(
    arr: np.ndarray, shape: tuple[int, ...], mode: str = "reflect"
) -> np.ndarray:
    """Pad *arr* to *shape* using *mode* (see ``np.pad``)."""
    assert arr.ndim == len(shape)
    dif = tuple(s - a for s, a in zip(shape, arr.shape, strict=True))
    assert all(d >= 0 for d in dif)
    pad_width = [[s // 2, s - s // 2] for s in dif]
    return np.pad(arr, pad_width=pad_width, mode=mode)


def _match_shape(img: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Pad or crop *img* to match *shape*."""
    shape_arr = np.array(shape)
    if np.any(shape_arr > np.array(img.shape)):
        padded_shape = tuple(np.maximum(img.shape, shape))
        img = _pad_to_shape(img, padded_shape, mode="reflect")
    if np.any(shape_arr < np.array(img.shape)):
        img = _center_crop(img, tuple(shape))
    return img


def _phase_cross_corr(
    ref_img: np.ndarray,
    mov_img: np.ndarray,
    maximum_shift: float = 1.0,
) -> tuple[int, ...]:
    """FFT-based phase cross-correlation returning pixel shifts in ZYX order.

    Parameters
    ----------
    ref_img : np.ndarray
        Reference image (2-D or 3-D).
    mov_img : np.ndarray
        Moved image, same dimensionality as *ref_img*.
    maximum_shift : float
        Maximum translation normalised by axis size (controls FFT padding).

    Returns
    -------
    Tuple[int, ...]
        Pixel shift for each axis (positive = *mov_img* shifted in the
        positive direction relative to *ref_img*).
    """
    shape = tuple(
        _next_fast_len(int(max(s1, s2) * maximum_shift))
        for s1, s2 in zip(ref_img.shape, mov_img.shape, strict=True)
    )

    logger.debug(
        "phase cross corr: fft shape %s for arrays %s and %s (max_shift=%.2f)",
        shape,
        ref_img.shape,
        mov_img.shape,
        maximum_shift,
    )

    ref_img = _match_shape(ref_img, shape)
    mov_img = _match_shape(mov_img, shape)

    fimg1 = np.fft.rfftn(ref_img)
    fimg2 = np.fft.rfftn(mov_img)

    prod = fimg1 * fimg2.conj()
    del fimg1, fimg2

    corr = np.fft.irfftn(prod)
    del prod

    corr = np.fft.fftshift(np.abs(corr))

    argmax = np.argmax(corr)
    peak = np.unravel_index(argmax, corr.shape)
    peak = tuple(s // 2 - p for s, p in zip(corr.shape, peak, strict=True))

    logger.debug("phase cross corr: peak at %s", peak)

    return peak


# ---------------------------------------------------------------------------
# Shift limiting
# ---------------------------------------------------------------------------


def _limit_shifts_zyx(
    shifts_zyx: np.ndarray,
    shift_limits: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Apply per-axis min/max limits to *shifts_zyx* (in microns).

    For each axis, shifts below the minimum absolute value are zeroed out
    (below stage resolution threshold) and shifts above the maximum are
    clipped in magnitude while preserving sign.

    Parameters
    ----------
    shifts_zyx : np.ndarray
        Shift values for (z, y, x).
    shift_limits : dict
        Mapping of axis name ("z", "y", "x") to (min, max) in microns.

    Returns
    -------
    np.ndarray
        Limited shift values (z, y, x).
    """
    shifts_zyx = np.array(shifts_zyx, dtype=float)
    axes = ["z", "y", "x"]

    for i, axis in enumerate(axes):
        if axis not in shift_limits:
            continue
        min_limit, max_limit = shift_limits[axis]
        if abs(shifts_zyx[i]) < min_limit:
            logger.debug(
                "Shift (%s) = %.3f below min limit %.3f, zeroing",
                axis,
                shifts_zyx[i],
                min_limit,
            )
            shifts_zyx[i] = 0.0
        elif abs(shifts_zyx[i]) > max_limit:
            logger.debug(
                "Shift (%s) = %.3f above max limit %.3f, clipping",
                axis,
                shifts_zyx[i],
                max_limit,
            )
            shifts_zyx[i] = np.sign(shifts_zyx[i]) * max_limit

    return shifts_zyx


# ---------------------------------------------------------------------------
# DynaTrackUpdater
# ---------------------------------------------------------------------------


class DynaTrackUpdater(PositionUpdater):
    """Position updater that tracks neuromast drift across timepoints.

    On the first call for a given position index, the z-stack is stored as the
    reference image. On subsequent calls, the current z-stack is compared to the
    reference to compute a translational shift in X, Y, and Z, which is applied
    to the position coordinates.

    Parameters
    ----------
    scale_yx : float
        Pixel size in microns per pixel for Y and X axes.
    scale_z : float
        Step size in microns for the Z axis.
    maximum_shift : float
        Maximum translation normalised by axis size for FFT padding
        (default 1.0).
    dampening : tuple[float, float, float] | None
        Optional (z, y, x) dampening factors applied multiplicatively
        to the computed shift.
    shift_limits : dict[str, tuple[float, float]] | None
        Optional per-axis (min, max) limits in microns. Keys are "z",
        "y", "x". Shifts below min are zeroed; shifts above max are
        clipped.
    """

    def __init__(
        self,
        scale_yx: float,
        scale_z: float,
        maximum_shift: float = 1.0,
        dampening: tuple[float, float, float] | None = None,
        shift_limits: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self._reference_stacks: dict[int, np.ndarray] = {}
        self._scale_yx = scale_yx
        self._scale_z = scale_z
        self._maximum_shift = maximum_shift
        self._dampening = dampening
        self._shift_limits = shift_limits

    def update(
        self,
        timepoint_index: int,
        position_index: int,
        position: PositionCoordinates,
        data: list[np.ndarray] | None = None,
    ) -> PositionCoordinates:
        """Compute updated position by tracking drift from reference z-stack.

        Parameters
        ----------
        timepoint_index : int
            The current timepoint index.
        position_index : int
            The position that was just acquired.
        position : PositionCoordinates
            Current coordinates for this position.
        data : list[np.ndarray] | None
            Frames acquired for this position (one 2D array per z-slice).

        Returns
        -------
        PositionCoordinates
            Position corrected for neuromast drift.
        """
        if data is None or len(data) == 0:
            logger.warning(
                f"DynaTrack: no data for p={position_index} at t={timepoint_index}, "
                "returning position unchanged"
            )
            return position

        current_stack = np.stack(data)

        # Store reference on first encounter
        if position_index not in self._reference_stacks:
            self._reference_stacks[position_index] = current_stack
            logger.info(
                f"DynaTrack: stored reference stack for p={position_index} "
                f"(shape={current_stack.shape})"
            )
            return position

        reference_stack = self._reference_stacks[position_index]
        shift = self._compute_shift(reference_stack, current_stack)

        logger.info(
            f"DynaTrack: p={position_index} t={timepoint_index} "
            f"shift=({shift[0]:.2f}, {shift[1]:.2f}, {shift[2]:.2f})"
        )

        return PositionCoordinates(
            x=position.x + shift[0],
            y=position.y + shift[1],
            z=(position.z or 0) + shift[2] if position.z is not None else None,
        )

    def _compute_shift(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compute the (x, y, z) shift between reference and current z-stacks.

        Parameters
        ----------
        reference : np.ndarray
            Reference z-stack, shape (Z, Y, X).
        current : np.ndarray
            Current z-stack, shape (Z, Y, X).

        Returns
        -------
        tuple[float, float, float]
            Estimated shift in (x, y, z) in stage coordinates (microns).
        """
        # 1. Phase cross-correlation in pixel space (returns ZYX order)
        shifts_zyx_px = _phase_cross_corr(reference, current, self._maximum_shift)

        # 2. Convert pixels to microns
        shifts_zyx_um = np.array(
            [
                shifts_zyx_px[0] * self._scale_z,
                shifts_zyx_px[1] * self._scale_yx,
                shifts_zyx_px[2] * self._scale_yx,
            ],
            dtype=float,
        )

        # 3. Apply shift limits (zero below min, clip above max)
        if self._shift_limits is not None:
            shifts_zyx_um = _limit_shifts_zyx(shifts_zyx_um, self._shift_limits)

        # 4. Apply dampening
        if self._dampening is not None:
            shifts_zyx_um = shifts_zyx_um * np.array(self._dampening, dtype=float)

        # 5. Reorder from (z, y, x) to (x, y, z) for PositionCoordinates
        x_um = float(shifts_zyx_um[2])
        y_um = float(shifts_zyx_um[1])
        z_um = float(shifts_zyx_um[0])

        return (x_um, y_um, z_um)
