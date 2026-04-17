"""DynaTrack — neuromast tracking for zebrafish embryo growth compensation.

Tracks neuromast motion across timepoints by comparing acquired z-stacks
to reference images and computing X/Y/Z shifts. Plugs into the position
update infrastructure via the PositionUpdater interface.
"""

from __future__ import annotations

import csv
import logging
import os

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import psutil

from shrimpy.mantis.position_update import PositionCoordinates, PositionUpdater

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


_PROC = psutil.Process(os.getpid())


def _rss_gb() -> float:
    return _PROC.memory_info().rss / (1024**3)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DynaTrackConfig:
    """Configuration for DynaTrack position tracking.

    Parameters
    ----------
    scale_yx : float
        Pixel size in microns per pixel for Y and X axes.
    scale_z : float
        Step size in microns for the Z axis.
    maximum_shift : float
        Maximum translation normalised by axis size for FFT padding.
    dampening : tuple[float, float, float] | None
        Optional (z, y, x) dampening factors applied multiplicatively
        to the computed shift.
    shift_limits : dict[str, tuple[float, float]] | None
        Optional per-axis (min, max) limits in microns. Keys are "z",
        "y", "x". Shifts below min are zeroed; shifts above max are clipped.
    tracking_interval : int
        Track every N timepoints (1 = every timepoint).
    shift_estimation_channel : str
        Which representation to use for shift estimation:
        ``'raw'`` (default, no preprocessing), ``'phase'`` (phase
        reconstruction), ``'vs_nuclei'`` or ``'vs_membrane'`` (virtual
        staining).
    preprocessing : list[str] | None
        Pipeline steps, e.g. ``['phase']`` or ``['phase', 'vs']``.
        Used by external factory functions to build the preprocessor callable.
    phase_config : dict[str, Any] | None
        Optical parameters for phase reconstruction (waveorder).
    deskew_config : dict[str, Any] | None
        Deskew parameters for light-sheet data (biahub). Keys:
        ``ls_angle_deg``, ``px_to_scan_ratio``, ``keep_overhang``,
        ``average_n_slices``.
    vs_config : dict[str, Any] | None
        Model and checkpoint config for virtual staining (viscy).
    shift_log_path : str | Path | None
        Path to a CSV file for incremental shift logging. Each computed
        shift is appended immediately after calculation. Typically set
        automatically by MantisEngine to ``<zarr_store>/dynatrack_log.csv``.
    """

    scale_yx: float
    scale_z: float
    maximum_shift: float = 1.0
    dampening: tuple[float, float, float] | None = None
    shift_limits: dict[str, tuple[float, float]] | None = None
    tracking_interval: int = 1
    shift_estimation_channel: str = "raw"
    preprocessing: list[str] | None = None
    deskew_config: dict[str, Any] | None = None
    phase_config: dict[str, Any] | None = None
    vs_config: dict[str, Any] | None = None
    shift_log_path: str | Path | None = None
    save_debug: bool = False


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

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_t = torch.from_numpy(ref_img.astype(np.float32, copy=False)).to(
        device, non_blocking=True
    )
    fimg1 = torch.fft.rfftn(ref_t)
    del ref_t

    mov_t = torch.from_numpy(mov_img.astype(np.float32, copy=False)).to(
        device, non_blocking=True
    )
    fimg2 = torch.fft.rfftn(mov_t)
    del mov_t

    prod = fimg1 * fimg2.conj()
    del fimg1, fimg2

    corr = torch.fft.irfftn(prod)
    del prod

    corr = torch.fft.fftshift(corr.abs())

    argmax = int(torch.argmax(corr).item())
    corr_shape = tuple(corr.shape)
    del corr
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    peak = np.unravel_index(argmax, corr_shape)
    peak = tuple(int(s // 2) - int(p) for s, p in zip(corr_shape, peak, strict=True))

    logger.debug("phase cross corr: peak at %s (device=%s)", peak, device)

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
# Shift logging
# ---------------------------------------------------------------------------

_SHIFT_LOG_HEADER = [
    "position_index",
    "timepoint_index",
    "shift_z_um",
    "shift_y_um",
    "shift_x_um",
    "stage_x",
    "stage_y",
    "stage_z",
]


def _append_shift_log(
    path: Path,
    position_index: int,
    timepoint_index: int,
    shift_zyx_um: tuple[float, float, float],
    stage_coords: PositionCoordinates,
) -> None:
    """Append a single shift record to the CSV log file.

    Creates the file with a header row if it does not already exist.
    """
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(_SHIFT_LOG_HEADER)
        writer.writerow(
            [
                position_index,
                timepoint_index,
                f"{shift_zyx_um[0]:.4f}",
                f"{shift_zyx_um[1]:.4f}",
                f"{shift_zyx_um[2]:.4f}",
                f"{stage_coords.x:.4f}",
                f"{stage_coords.y:.4f}",
                f"{stage_coords.z:.4f}" if stage_coords.z is not None else "",
            ]
        )


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
    config : DynaTrackConfig
        Tracking configuration.
    preprocessor : Callable[[np.ndarray], dict[str, np.ndarray]] | None
        Optional callable that transforms a z-stack and returns a dict of
        channel name to ZYX array (e.g. ``{'phase': ..., 'vs_nuclei': ...}``).
        The channel specified by ``config.shift_estimation_channel`` is used
        for phase cross-correlation. When ``None``, the raw z-stack is used.
    """

    def __init__(
        self,
        config: DynaTrackConfig,
        preprocessor: Callable[[np.ndarray], dict[str, np.ndarray]] | None = None,
    ) -> None:
        self._config = config
        self._preprocessor = preprocessor
        self._reference_stacks: dict[int, np.ndarray] = {}
        self._shift_log_path: Path | None = (
            Path(config.shift_log_path) if config.shift_log_path else None
        )
        # Debug HCS zarr store for preprocessed stacks (set by MantisEngine)
        self._debug_zarr_path: Path | None = None
        self._debug_store = None
        self._debug_position_names: dict[int, str] = {}  # set by MantisEngine

    @property
    def config(self) -> DynaTrackConfig:
        return self._config

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

        import time as _time

        logger.info(
            f"DynaTrack[mem]: entry p={position_index} t={timepoint_index} "
            f"rss={_rss_gb():.2f} GB"
        )

        current_stack = np.stack(data)
        logger.info(
            f"DynaTrack: p={position_index} t={timepoint_index} "
            f"stack shape={current_stack.shape} dtype={current_stack.dtype} "
            f"size={current_stack.nbytes / 1024**3:.2f} GB"
        )
        logger.info(
            f"DynaTrack[mem]: after np.stack p={position_index} t={timepoint_index} "
            f"rss={_rss_gb():.2f} GB"
        )

        # Apply optional preprocessing (e.g. phase reconstruction, VS)
        if self._preprocessor is not None:
            t0 = _time.monotonic()
            channels = self._preprocessor(current_stack)
            logger.info(
                f"DynaTrack: preprocessing took {_time.monotonic() - t0:.1f}s "
                f"(channels={list(channels.keys())})"
            )
            ch_bytes = sum(a.nbytes for a in channels.values())
            logger.info(
                f"DynaTrack[mem]: after preprocessor p={position_index} t={timepoint_index} "
                f"rss={_rss_gb():.2f} GB channels_total={ch_bytes / 1024**3:.2f} GB"
            )
            self._save_debug_channels(channels, timepoint_index, position_index)
            # Select the configured channel for shift estimation
            channel_name = self._config.shift_estimation_channel
            if channel_name in channels:
                current_stack = channels[channel_name]
            else:
                logger.warning(
                    f"DynaTrack: channel '{channel_name}' not in preprocessor "
                    f"output {list(channels.keys())}, using first channel"
                )
                current_stack = next(iter(channels.values()))
            logger.info(
                f"DynaTrack[mem]: after channel select p={position_index} t={timepoint_index} "
                f"rss={_rss_gb():.2f} GB"
            )

        # Store reference on first encounter
        if position_index not in self._reference_stacks:
            self._reference_stacks[position_index] = current_stack
            ref_total = sum(a.nbytes for a in self._reference_stacks.values())
            logger.info(
                f"DynaTrack: stored reference stack for p={position_index} "
                f"(shape={current_stack.shape})"
            )
            logger.info(
                f"DynaTrack[mem]: after store_ref p={position_index} t={timepoint_index} "
                f"rss={_rss_gb():.2f} GB refs={len(self._reference_stacks)} "
                f"refs_total={ref_total / 1024**3:.2f} GB"
            )
            return position

        # Skip tracking if not on a tracking interval
        if (
            self._config.tracking_interval > 1
            and timepoint_index % self._config.tracking_interval != 0
        ):
            logger.debug(
                f"DynaTrack: skipping p={position_index} at t={timepoint_index} "
                f"(interval={self._config.tracking_interval})"
            )
            return position

        reference_stack = self._reference_stacks[position_index]
        logger.info(
            f"DynaTrack[mem]: before compute_shift p={position_index} t={timepoint_index} "
            f"rss={_rss_gb():.2f} GB"
        )
        t_pcc = _time.monotonic()
        shift_xyz = self._compute_shift(reference_stack, current_stack)
        logger.info(f"DynaTrack: phase cross corr took {_time.monotonic() - t_pcc:.2f}s")
        logger.info(
            f"DynaTrack[mem]: after compute_shift p={position_index} t={timepoint_index} "
            f"rss={_rss_gb():.2f} GB"
        )

        updated = PositionCoordinates(
            x=position.x + shift_xyz[0],
            y=position.y + shift_xyz[1],
            z=(position.z or 0) + shift_xyz[2] if position.z is not None else None,
        )

        logger.info(
            f"DynaTrack: p={position_index} t={timepoint_index} "
            f"shift=({shift_xyz[0]:.2f}, {shift_xyz[1]:.2f}, {shift_xyz[2]:.2f})"
        )

        # Log shift to CSV immediately
        if self._shift_log_path is not None:
            _append_shift_log(
                self._shift_log_path,
                position_index,
                timepoint_index,
                shift_zyx_um=self._last_shift_zyx_um,
                stage_coords=updated,
            )

        logger.info(
            f"DynaTrack[mem]: exit p={position_index} t={timepoint_index} "
            f"rss={_rss_gb():.2f} GB"
        )
        return updated

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
        cfg = self._config

        # 1. Phase cross-correlation in pixel space (returns ZYX order)
        shifts_zyx_px = _phase_cross_corr(reference, current, cfg.maximum_shift)

        # 2. Convert pixels to microns
        shifts_zyx_um = np.array(
            [
                shifts_zyx_px[0] * cfg.scale_z,
                shifts_zyx_px[1] * cfg.scale_yx,
                shifts_zyx_px[2] * cfg.scale_yx,
            ],
            dtype=float,
        )

        # 3. Apply shift limits (zero below min, clip above max)
        if cfg.shift_limits is not None:
            shifts_zyx_um = _limit_shifts_zyx(shifts_zyx_um, cfg.shift_limits)

        # 4. Apply dampening
        if cfg.dampening is not None:
            shifts_zyx_um = shifts_zyx_um * np.array(cfg.dampening, dtype=float)

        # Store for shift logging (ZYX order, microns)
        self._last_shift_zyx_um = (
            float(shifts_zyx_um[0]),
            float(shifts_zyx_um[1]),
            float(shifts_zyx_um[2]),
        )

        # 5. Reorder from (z, y, x) to (x, y, z) for PositionCoordinates
        x_um = float(shifts_zyx_um[2])
        y_um = float(shifts_zyx_um[1])
        z_um = float(shifts_zyx_um[0])

        return (x_um, y_um, z_um)

    def _save_debug_channels(
        self,
        channels: dict[str, np.ndarray],
        timepoint_index: int,
        position_index: int,
    ) -> None:
        """Save all preprocessed channels to an HCS OME-Zarr store.

        Each MDA position maps to a position in the HCS plate, using
        matching position names from the acquisition sequence.
        """
        if self._debug_zarr_path is None:
            return

        from iohub.ngff import open_ome_zarr

        channel_names = sorted(channels.keys())
        czyx = np.stack([channels[name] for name in channel_names])
        nc, nz, ny, nx = czyx.shape

        # Create the HCS store on first call
        if self._debug_store is None:
            self._debug_store = open_ome_zarr(
                str(self._debug_zarr_path),
                layout="hcs",
                mode="w",
                channel_names=channel_names,
                version="0.5",
            )
            logger.info(
                "DynaTrack: debug store created at %s (channels=%s)",
                self._debug_zarr_path,
                channel_names,
            )

        # Create position on first encounter
        pos_name = self._debug_position_names.get(position_index, f"p{position_index}")
        pos_key = f"0/{position_index}/{pos_name}"
        if pos_key not in dict(self._debug_store.positions()):
            pos = self._debug_store.create_position("0", str(position_index), pos_name)
            pos.create_zeros(
                "0",
                shape=(0, nc, nz, ny, nx),
                chunks=(1, 1, min(32, nz), ny, nx),
                dtype=czyx.dtype,
            )
            logger.info("DynaTrack: debug position '%s' created", pos_name)

        _, pos_node = next((k, v) for k, v in self._debug_store.positions() if k == pos_key)
        pos_node["0"].append(czyx[np.newaxis], axis=0)
        logger.debug(
            "DynaTrack: saved debug t=%d p=%d '%s' (shape=%s)",
            timepoint_index,
            position_index,
            pos_name,
            pos_node["0"].shape,
        )
