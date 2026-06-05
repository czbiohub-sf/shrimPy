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

    import torch


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
    tracking_method : str
        Shift estimation algorithm. One of ``'pcc'`` (phase cross-correlation,
        default), ``'multiotsu_center_of_mass'`` (multi-Otsu threshold then
        area-weighted centroid), or ``'multiotsu_pcc'`` (multi-Otsu threshold
        then PCC on the binary masks).
    otsu_sigma : float
        Gaussian blur sigma for the multi-Otsu methods (default 5.0).
    otsu_component : int
        Which multi-Otsu threshold to use: 0 = lower, 1 = upper (default 0).
    reference_update_interval : int
        Re-anchor the per-position reference every N timepoints (0 = never,
        i.e. keep the fixed t=0 reference). On a re-anchor timepoint the
        current stack becomes the new reference and NO shift correction is
        applied (the current stage position is accepted as the new baseline),
        which avoids a jump. Useful for long timelapses where the sample
        changes enough that phase-correlation against a stale reference
        becomes unreliable.
    shift_estimation_channel : str
        Which representation to use for shift estimation:
        ``'deskewed'`` (default; the deskewed input volume, e.g. deskew-only
        tracking), ``'phase'`` (phase reconstruction), ``'vs_nuclei'`` or
        ``'vs_membrane'`` (virtual staining).
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
    tracking_method: str = "pcc"
    otsu_sigma: float = 5.0
    otsu_component: int = 0
    reference_update_interval: int = 0
    shift_estimation_channel: str = "deskewed"
    preprocessing: list[str] | None = None
    deskew_config: dict[str, Any] | None = None
    phase_config: dict[str, Any] | None = None
    vs_config: dict[str, Any] | None = None
    image_to_stage_matrix_xyz: list[list[float]] | None = None
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


def _center_crop(t: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Crop the center of *t* to *shape*."""
    assert t.ndim == len(shape)
    starts = tuple((cur_s - s) // 2 for cur_s, s in zip(t.shape, shape, strict=True))
    assert all(s >= 0 for s in starts)
    slicing = tuple(slice(s, s + d) for s, d in zip(starts, shape, strict=True))
    return t[slicing]


def _pad_to_shape(
    t: torch.Tensor, shape: tuple[int, ...], mode: str = "reflect"
) -> torch.Tensor:
    """Pad *t* to *shape* using *mode* (``torch.nn.functional.pad`` semantics)."""
    from torch.nn.functional import pad as torch_pad

    assert t.ndim == len(shape)
    dif = [s - a for s, a in zip(shape, t.shape, strict=True)]
    assert all(d >= 0 for d in dif)
    if all(d == 0 for d in dif):
        return t
    # pad sizes are ordered from the last axis to the first
    pad_arg: list[int] = []
    for d in reversed(dif):
        left = d // 2
        pad_arg.extend([left, d - left])
    # reflect/replicate require ndim == 2 + n_pad_dims; wrap in unit
    # batch+channel dims so the trick works for any input rank.
    orig_shape = tuple(t.shape)
    t = t.reshape((1, 1) + orig_shape)
    t = torch_pad(t, pad_arg, mode=mode)
    return t.reshape(tuple(t.shape[2:]))


def _match_shape(t: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Pad or crop *t* to match *shape*."""
    if any(s > d for s, d in zip(shape, t.shape, strict=True)):
        padded_shape = tuple(max(d, s) for d, s in zip(t.shape, shape, strict=True))
        t = _pad_to_shape(t, padded_shape, mode="reflect")
    if any(s < d for s, d in zip(shape, t.shape, strict=True)):
        t = _center_crop(t, tuple(shape))
    return t


def _phase_cross_corr(
    ref_img: torch.Tensor,
    mov_img: torch.Tensor,
    maximum_shift: float = 1.0,
) -> tuple[int, ...]:
    """FFT-based phase cross-correlation returning pixel shifts in ZYX order.

    Parameters
    ----------
    ref_img : torch.Tensor
        Reference image (2-D or 3-D) on the target compute device.
    mov_img : torch.Tensor
        Moved image, same dimensionality and device as *ref_img*.
    maximum_shift : float
        Maximum translation normalised by axis size (controls FFT padding).

    Returns
    -------
    Tuple[int, ...]
        Pixel shift for each axis (positive = *mov_img* shifted in the
        positive direction relative to *ref_img*).
    """
    import torch

    ref_t = ref_img.to(dtype=torch.float32)
    mov_t = mov_img.to(dtype=torch.float32)

    shape = tuple(
        _next_fast_len(int(max(s1, s2) * maximum_shift))
        for s1, s2 in zip(ref_t.shape, mov_t.shape, strict=True)
    )

    logger.debug(
        "phase cross corr: fft shape %s for arrays %s and %s (max_shift=%.2f)",
        shape,
        tuple(ref_t.shape),
        tuple(mov_t.shape),
        maximum_shift,
    )

    ref_t = _match_shape(ref_t, shape)
    mov_t = _match_shape(mov_t, shape)

    fimg1 = torch.fft.rfftn(ref_t)
    del ref_t
    fimg2 = torch.fft.rfftn(mov_t)
    del mov_t

    prod = fimg1 * fimg2.conj()
    del fimg1, fimg2

    corr = torch.fft.irfftn(prod, s=shape)
    del prod

    corr = torch.fft.fftshift(corr.abs())

    argmax = int(torch.argmax(corr).item())
    corr_shape = tuple(corr.shape)
    device = corr.device
    del corr
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    peak = np.unravel_index(argmax, corr_shape)
    peak = tuple(int(s // 2) - int(p) for s, p in zip(corr_shape, peak, strict=True))

    logger.debug("phase cross corr: peak at %s (device=%s)", peak, device)

    return peak


# ---------------------------------------------------------------------------
# Multi-Otsu thresholding helpers (GPU)
# ---------------------------------------------------------------------------


def _gaussian_blur_3d(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply separable 3-D Gaussian blur entirely on *img*'s device.

    Uses three sequential 1-D convolutions (one per axis) with reflect
    padding, matching the behaviour of ``skimage.filters.gaussian``.
    """
    import torch
    import torch.nn.functional as F  # noqa: N812

    if sigma <= 0:
        return img

    max_radius = int(4 * sigma + 0.5)

    vol = img[None, None]  # (1, 1, Z, Y, X)

    # Convolve each spatial axis with a 1-D Gaussian kernel.
    # Reflect padding requires pad < dim, so clamp per axis.
    for spatial_idx, axis in enumerate((2, 3, 4)):
        r = min(max_radius, vol.shape[axis] - 1)
        x = torch.arange(-r, r + 1, device=img.device, dtype=img.dtype)
        k1d = torch.exp(-0.5 * (x / sigma) ** 2)
        k1d = k1d / k1d.sum()

        # F.pad expects (x_l, x_r, y_l, y_r, z_l, z_r) — last dim first.
        # spatial_idx: 0=Z(axis2), 1=Y(axis3), 2=X(axis4)
        pad = [0] * 6
        pad_pos = 2 * (2 - spatial_idx)  # Z→4, Y→2, X→0
        pad[pad_pos] = r
        pad[pad_pos + 1] = r
        vol = F.pad(vol, pad, mode="reflect")

        k_shape = [1, 1, 1, 1, 1]
        k_shape[axis] = len(k1d)
        vol = F.conv3d(vol, k1d.reshape(k_shape))

    return vol[0, 0]


def _multiotsu_threshold(
    img_blur: torch.Tensor,
    otsu_component: int = 0,
    nbins: int = 256,
) -> float:
    """Compute multi-Otsu threshold entirely on GPU.

    Builds a histogram with ``torch.histc``, then finds the two
    thresholds that maximise inter-class variance (3-class Otsu) via a
    fully vectorised search over all bin-pair splits — all on the
    tensor's device.

    Parameters
    ----------
    img_blur : torch.Tensor
        Pre-blurred volume on GPU.
    otsu_component : int
        Which threshold to return (0 = lower, 1 = upper).
    nbins : int
        Number of histogram bins.

    Returns
    -------
    float
        The selected threshold value.
    """
    import torch

    vmin = img_blur.min()
    vmax = img_blur.max()
    if vmin == vmax:
        return float(vmin)

    hist = torch.histc(img_blur, bins=nbins, min=float(vmin), max=float(vmax))
    hist = hist / hist.sum()  # normalise to probability

    bin_centers = torch.linspace(float(vmin), float(vmax), nbins, device=img_blur.device)

    # Cumulative sums for fast inter-class variance computation
    cum_w = torch.cumsum(hist, dim=0)  # cumulative weight
    cum_wm = torch.cumsum(hist * bin_centers, dim=0)  # cumulative weighted mean
    total_mean = cum_wm[-1]
    del hist

    # 3-class Otsu: choose split boundaries a < b (bin indices) that maximise
    # the between-class variance, evaluated over all (a, b) pairs at once
    # instead of a Python double loop (this is the expensive part).
    #   class 0 = bins [0..a]   class 1 = bins [a+1..b]   class 2 = bins [b+1..]
    eps = 1e-10
    w0 = cum_w.unsqueeze(1)  # [N, 1]  weight of class 0 (boundary a)
    w1 = cum_w.unsqueeze(0) - cum_w.unsqueeze(1)  # [N, N]  class 1 (a < b)
    w2 = 1.0 - cum_w.unsqueeze(0)  # [1, N]  class 2 (boundary b)
    m0 = cum_wm.unsqueeze(1) / w0.clamp_min(eps)
    m1 = (cum_wm.unsqueeze(0) - cum_wm.unsqueeze(1)) / w1.clamp_min(eps)
    m2 = (total_mean - cum_wm.unsqueeze(0)) / w2.clamp_min(eps)
    sigma = (
        w0 * (m0 - total_mean) ** 2 + w1 * (m1 - total_mean) ** 2 + w2 * (m2 - total_mean) ** 2
    )

    bins = torch.arange(nbins, device=img_blur.device)
    valid = (
        (bins.unsqueeze(0) > bins.unsqueeze(1))  # b > a
        & (bins.unsqueeze(0) <= nbins - 2)  # class 2 non-empty
        & (w0 > eps)
        & (w1 > eps)
        & (w2 > eps)
    )
    sigma = sigma.masked_fill(~valid, -1.0)

    flat = int(torch.argmax(sigma))
    best_a, best_b = divmod(flat, nbins)
    # the original loop indexed thresholds at t1 = a + 1, t2 = b + 1
    thresholds = (float(bin_centers[best_a + 1]), float(bin_centers[best_b + 1]))
    del cum_w, cum_wm, bin_centers, sigma
    logger.debug("multi-Otsu thresholds: %s (using component %d)", thresholds, otsu_component)
    idx = min(otsu_component, 1)
    return thresholds[idx]


def _binary_mask(
    img: torch.Tensor,
    sigma: float = 5.0,
    otsu_component: int = 0,
) -> torch.Tensor:
    """Rescale, blur, and threshold a 3-D volume on GPU.

    Parameters
    ----------
    img : torch.Tensor
        Input volume (Z, Y, X) on the target device.
    sigma : float
        Gaussian blur sigma applied before thresholding.
    otsu_component : int
        Which multi-Otsu threshold to use (0 = lower, 1 = upper).

    Returns
    -------
    torch.Tensor
        Boolean mask on the same device as *img*.
    """
    import torch

    img = img.to(dtype=torch.float32)

    # Rescale to [0, 1]
    vmin = img.min()
    vmax = img.max()
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    else:
        return torch.zeros_like(img, dtype=torch.bool)

    img_blur = _gaussian_blur_3d(img, sigma)
    del img
    threshold = _multiotsu_threshold(img_blur, otsu_component)
    mask = img_blur > threshold
    del img_blur
    return mask


def _center_of_mass(mask: torch.Tensor) -> torch.Tensor:
    """Compute the center of mass of a boolean mask on GPU.

    Equivalent to the area-weighted centroid: every True voxel
    contributes equally, so larger connected regions naturally
    dominate.

    Parameters
    ----------
    mask : torch.Tensor
        Boolean mask (Z, Y, X).

    Returns
    -------
    torch.Tensor
        Center-of-mass coordinates, shape ``(ndim,)``, on the same device.
    """
    import torch

    coords = torch.nonzero(mask, as_tuple=False).to(dtype=torch.float32)
    if coords.shape[0] == 0:
        return torch.zeros(mask.ndim, device=mask.device)
    center = coords.mean(dim=0)
    del coords
    return center


def _multiotsu_center_of_mass(
    ref_img: torch.Tensor,
    mov_img: torch.Tensor,
    sigma: float = 5.0,
    otsu_component: int = 0,
) -> tuple[float, ...]:
    """Compute shift via multi-Otsu thresholding + center of mass on GPU.

    Both images are thresholded independently and the shift is the
    difference between their centres of mass (ZYX pixel order).
    All heavy computation stays on the input tensor's device.
    """
    ref_mask = _binary_mask(ref_img, sigma=sigma, otsu_component=otsu_component)
    mov_mask = _binary_mask(mov_img, sigma=sigma, otsu_component=otsu_component)

    ref_center = _center_of_mass(ref_mask)
    del ref_mask
    mov_center = _center_of_mass(mov_mask)
    del mov_mask

    shift_zyx = mov_center - ref_center
    logger.debug(
        "multiotsu_center_of_mass: ref_center=%s mov_center=%s shift=%s",
        ref_center.tolist(),
        mov_center.tolist(),
        shift_zyx.tolist(),
    )
    del ref_center, mov_center
    return tuple(float(s) for s in shift_zyx)


def _multiotsu_pcc(
    ref_img: torch.Tensor,
    mov_img: torch.Tensor,
    sigma: float = 5.0,
    otsu_component: int = 0,
    maximum_shift: float = 1.0,
) -> tuple[int, ...]:
    """Compute shift via multi-Otsu thresholding + PCC on binary masks.

    Both images are thresholded on GPU, then phase cross-correlation
    is run on the binary volumes entirely on device.
    """
    import torch

    ref_mask = _binary_mask(ref_img, sigma=sigma, otsu_component=otsu_component)
    mov_mask = _binary_mask(mov_img, sigma=sigma, otsu_component=otsu_component)

    ref_binary_f = ref_mask.to(dtype=torch.float32)
    del ref_mask
    mov_binary_f = mov_mask.to(dtype=torch.float32)
    del mov_mask

    result = _phase_cross_corr(ref_binary_f, mov_binary_f, maximum_shift)
    del ref_binary_f, mov_binary_f
    return result


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
    preprocessor : Callable[[np.ndarray], dict[str, torch.Tensor]] | None
        Optional callable that transforms a z-stack and returns a dict of
        channel name to ZYX tensor on the target device (e.g.
        ``{'phase': ..., 'vs_nuclei': ...}``). The channel specified by
        ``config.shift_estimation_channel`` is used for phase
        cross-correlation. When ``None``, the raw z-stack is used.
    """

    def __init__(
        self,
        config: DynaTrackConfig,
        preprocessor: Callable[[np.ndarray], dict[str, torch.Tensor]] | None = None,
    ) -> None:
        self._config = config
        self._preprocessor = preprocessor
        self._reference_stacks_zyx: dict[int, torch.Tensor] = {}
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
            Stage coordinates the stack was acquired at. The computed shift is
            added to this value, so corrections compensate the drift between
            the reference and where the stack actually was -- not against a
            store value that a later update may already have moved on.
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

        logger.debug(
            f"DynaTrack[mem]: entry p={position_index} t={timepoint_index} "
            f"rss={_rss_gb():.2f} GB"
        )

        import torch

        raw_stack = np.stack(data)
        logger.info(
            f"DynaTrack: p={position_index} t={timepoint_index} "
            f"stack shape={raw_stack.shape} dtype={raw_stack.dtype} "
            f"size={raw_stack.nbytes / 1024**3:.2f} GB"
        )
        logger.debug(
            f"DynaTrack[mem]: after np.stack p={position_index} t={timepoint_index} "
            f"rss={_rss_gb():.2f} GB"
        )

        # All downstream ops run on tensors. Move to CUDA if available;
        # the preprocessor (if any) already targets the same device.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Apply optional preprocessing (e.g. phase reconstruction, VS).
        # Preprocessor returns torch tensors on device; convert to numpy at
        # the PCC/saving boundaries below.
        if self._preprocessor is not None:
            t0 = _time.monotonic()
            channels_zyx = self._preprocessor(raw_stack)
            logger.info(
                f"DynaTrack: preprocessing took {_time.monotonic() - t0:.1f}s "
                f"(channels={list(channels_zyx.keys())})"
            )
            ch_bytes = sum(a.nbytes for a in channels_zyx.values())
            logger.debug(
                f"DynaTrack[mem]: after preprocessor p={position_index} t={timepoint_index} "
                f"rss={_rss_gb():.2f} GB channels_total={ch_bytes / 1024**3:.2f} GB"
            )
            # Select the configured channel for shift estimation. Stays as a
            # torch tensor on device; PCC consumes tensors directly.
            channel_name = self._config.shift_estimation_channel
            if channel_name in channels_zyx:
                selected = channels_zyx[channel_name]
            else:
                logger.warning(
                    f"DynaTrack: channel '{channel_name}' not in preprocessor "
                    f"output {list(channels_zyx.keys())}, using first channel"
                )
                selected = next(iter(channels_zyx.values()))
            self._save_debug_channels(channels_zyx, timepoint_index, position_index)
            # For center-of-mass tracking, save a PNG of the thresholded mask
            # (Z max-projection) with the computed centroid marked, for a quick
            # visual check of where the tracker is centring.
            if (
                self._debug_zarr_path is not None
                and self._config.tracking_method == "multiotsu_center_of_mass"
            ):
                self._save_center_png(selected, timepoint_index, position_index)
            # Clone into a compact standalone tensor. A bare ``.detach()`` is a
            # view that keeps the *entire* preprocessor output alive (all
            # channels share one parent tensor), so a stored reference would
            # pin every channel, not just this one. Cloning lets the other
            # channels + parent be freed below.
            current_stack_zyx = selected.detach().clone()
            # Drop the dict + view and return the freed blocks to the GPU so
            # the caching allocator does not stay pinned at the VS peak.
            del channels_zyx, selected
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug(
                f"DynaTrack[mem]: after channel select p={position_index} t={timepoint_index} "
                f"rss={_rss_gb():.2f} GB"
            )
        else:
            # No preprocessing: move the raw stack to the device directly.
            current_stack_zyx = torch.as_tensor(raw_stack, device=device, dtype=torch.float32)

        # (Re)anchor the reference: store it on first encounter, and re-anchor
        # every ``reference_update_interval`` timepoints. On a re-anchor
        # timepoint we adopt the current image as the new reference and apply
        # NO correction (return position unchanged) -- the current stage
        # position becomes the new baseline; correcting here would jump the
        # stage against a reference we are about to discard.
        interval = self._config.reference_update_interval
        if (position_index not in self._reference_stacks_zyx) or (
            interval and timepoint_index % interval == 0
        ):
            self._reference_stacks_zyx[position_index] = current_stack_zyx
            ref_total = sum(a.nbytes for a in self._reference_stacks_zyx.values())
            logger.info(
                f"DynaTrack: stored reference stack for p={position_index} from t={timepoint_index} "
                f"(zyx_shape={current_stack_zyx.shape})"
            )
            logger.debug(
                f"DynaTrack[mem]: after store_ref p={position_index} t={timepoint_index} "
                f"rss={_rss_gb():.2f} GB refs={len(self._reference_stacks_zyx)} "
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

        reference_stack_zyx = self._reference_stacks_zyx[position_index]
        logger.debug(
            f"DynaTrack[mem]: before compute_shift p={position_index} t={timepoint_index} "
            f"rss={_rss_gb():.2f} GB"
        )
        t_pcc = _time.monotonic()
        shift_image_xyz = self._compute_shift(reference_stack_zyx, current_stack_zyx)
        logger.info(f"DynaTrack: phase cross corr took {_time.monotonic() - t_pcc:.2f}s")
        logger.debug(
            f"DynaTrack[mem]: after compute_shift p={position_index} t={timepoint_index} "
            f"rss={_rss_gb():.2f} GB"
        )

        # 1. Decouple position in image space from position in stage space
        # First get the stage position in image space with configurable transform matrix
        # Add the shift to get the new position in image space
        # Convert the new position in image space to stage position
        # Update the position in stage space

        if self._config.image_to_stage_matrix_xyz is not None:
            transform_xyz = np.asarray(self._config.image_to_stage_matrix_xyz)
            shift_stage_xyz = transform_xyz @ shift_image_xyz
            logger.info(
                f"DynaTrack: applied image-to-stage matrix transform to shift: "
                f"image_xyz=({shift_image_xyz[0]:.2f}, {shift_image_xyz[1]:.2f}, {shift_image_xyz[2]:.2f}) um -> "
                f"stage_xyz=({shift_stage_xyz[0]:.2f}, {shift_stage_xyz[1]:.2f}, {shift_stage_xyz[2]:.2f}) um"
            )
        else:
            shift_stage_xyz = shift_image_xyz

        # Compensate the drift between the reference and where the stack was
        # actually acquired. `position` is the commanded stage coords at
        # acquisition time (so we don't accumulate against a store value a
        # later update may already have moved on). The shift is the measured
        # drift of the current image relative to the reference, so the stage
        # must move in the OPPOSITE direction to recenter -- hence subtract.
        baseline = position
        logger.info(
            f"DynaTrack: baseline p={position_index} t={timepoint_index} "
            f"x={baseline.x} y={baseline.y} z={baseline.z}"
        )
        _x = baseline.x - shift_stage_xyz[0]
        _y = baseline.y - shift_stage_xyz[1]
        _z = (baseline.z or 0) - shift_stage_xyz[2] if baseline.z is not None else None
        updated = PositionCoordinates(_x, _y, _z)
        logger.info(
            f"DynaTrack: updated position p={position_index} t={timepoint_index} "
            f"x={updated.x} y={updated.y} z={updated.z}"
        )
        logger.info(
            f"DynaTrack: p={position_index} t={timepoint_index} "
            f"shift_xyz_um=({shift_image_xyz[0]:.2f}, {shift_image_xyz[1]:.2f}, {shift_image_xyz[2]:.2f})"
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

        logger.debug(
            f"DynaTrack[mem]: exit p={position_index} t={timepoint_index} "
            f"rss={_rss_gb():.2f} GB"
        )
        return updated

    def _compute_shift(
        self,
        reference_zyx: torch.Tensor,
        current_zyx: torch.Tensor,
    ) -> tuple[float, float, float]:
        """Compute the (x, y, z) shift between reference and current z-stacks.

        Parameters
        ----------
        reference_zyx : torch.Tensor
            Reference z-stack, shape (Z, Y, X).
        current_zyx : torch.Tensor
            Current z-stack, shape (Z, Y, X).

        Returns
        -------
        tuple[float, float, float]
            Estimated shift in (x, y, z) in stage coordinates (microns).
        """
        cfg = self._config

        # 1. Compute pixel shifts using the configured method (returns ZYX order)
        method = cfg.tracking_method
        if method == "pcc":
            shifts_zyx_px = _phase_cross_corr(reference_zyx, current_zyx, cfg.maximum_shift)
        elif method == "multiotsu_center_of_mass":
            shifts_zyx_px = _multiotsu_center_of_mass(
                reference_zyx,
                current_zyx,
                sigma=cfg.otsu_sigma,
                otsu_component=cfg.otsu_component,
            )
        elif method == "multiotsu_pcc":
            shifts_zyx_px = _multiotsu_pcc(
                reference_zyx,
                current_zyx,
                sigma=cfg.otsu_sigma,
                otsu_component=cfg.otsu_component,
                maximum_shift=cfg.maximum_shift,
            )
        else:
            raise ValueError(
                f"Unknown tracking_method={method!r}. "
                "Use 'pcc', 'multiotsu_center_of_mass', or 'multiotsu_pcc'."
            )

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
        channels: dict[str, torch.Tensor],
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
        czyx = np.stack([channels[name].detach().cpu().numpy() for name in channel_names])
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

        # Create position on first encounter. OME-Zarr (iohub) requires
        # alphanumeric path names, so strip non-alphanumeric characters from
        # the acquisition position name (e.g. "1-Pos0000" -> "1Pos0000").
        raw_name = self._debug_position_names.get(position_index, f"p{position_index}")
        pos_name = "".join(ch for ch in raw_name if ch.isalnum()) or f"p{position_index}"
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

    def _save_center_png(
        self,
        volume: torch.Tensor,
        timepoint_index: int,
        position_index: int,
    ) -> None:
        """Save a PNG of the thresholded mask (Z max-projection) with the
        center-of-mass marked, for a quick visual check of the tracker's
        centring. Written to a ``dynatrack_centers/`` folder next to the
        debug zarr, one PNG per position per timepoint.
        """
        if self._debug_zarr_path is None:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mask = _binary_mask(
            volume,
            sigma=self._config.otsu_sigma,
            otsu_component=self._config.otsu_component,
        )
        cz, cy, cx = (float(c) for c in _center_of_mass(mask).tolist())
        mip = mask.any(dim=0).detach().cpu().numpy()  # (Y, X) projection over Z

        raw_name = self._debug_position_names.get(position_index, f"p{position_index}")
        pos_name = "".join(ch for ch in raw_name if ch.isalnum()) or f"p{position_index}"
        out_dir = self._debug_zarr_path.with_name("dynatrack_centers")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{pos_name}_t{timepoint_index:04d}.png"

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(mip, cmap="gray", interpolation="nearest")
        ax.plot(cx, cy, marker="x", color="red", markersize=14, markeredgewidth=2)
        ax.set_title(f"{pos_name} t={timepoint_index}  com=({cz:.1f}, {cy:.1f}, {cx:.1f})")
        ax.axis("off")
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info("DynaTrack: saved center PNG %s", out_path)
