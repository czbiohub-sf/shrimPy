"""FOV-selection debug artifacts: the PNG / CSV / OME-Zarr traces of a pre-scan.

Separated from the worker's process orchestration (:mod:`shrimpy.fov_selection.worker`) so
all the "what a pre-scan writes to disk, and in what layout" logic lives in one place. Three
kinds of output, all optional:

- lightweight per-FOV debug (``save_decision``): a projection PNG, a magenta mask-overlay
  PNG, and one row appended to ``fov_summary.csv`` -- :func:`write_decision_artifacts`.
- calibration feature-viewer layout: the same images under the ``<stem>``-prefixed folder
  names and a ``<stem>.csv`` with a ``filename`` join column that
  :mod:`shrimpy.fov_selection.feature_viewer.data` loads directly --
  :func:`write_feature_viewer_artifacts`.
- the full per-step reconstruction OME-Zarr (``save_pre_scan_omezarr``): every
  reconstruction stage (deskew / phase / VS volumes in 3D + the projection / mask broadcast
  across Z) as one HCS store -- :func:`write_reconstruction_zarr`.

Plus :func:`finalize_summary_csv`, which stamps the whole-run ``selected`` / ``position`` /
``rank`` columns onto ``fov_summary.csv`` after every FOV is scored, and
:func:`save_mask_overlay_png`, the shared magenta-outline renderer.

FOV selection segments exactly ONE mask (the ``target``), so the ``projections`` / ``masks``
that :func:`shrimpy.fov_selection.pipeline.decide_fov` returns each hold a single channel;
the per-FOV writers below are single-channel accordingly. Only the reconstruction store is
multi-channel (its 3D ``stacks`` still carry deskew / phase / nuclei / membrane).

The writers run in the worker subprocess one FOV at a time, so every CSV/store
read-append-write is unsynchronized-safe. Every call site guards these functions: debug I/O
must never invalidate an already-computed decision.
"""

from __future__ import annotations

import logging

from pathlib import Path

import numpy as np

from shrimpy.fov_selection.plate_naming import file_stem_name, zarr_path_name

logger = logging.getLogger(__name__)

# The combined per-FOV debug table (one row per FOV) written under `debug_dir` when
# `save_decision` is set. Named here because the manager reopens it post-drain to add the
# selected/rank columns (see finalize_summary_csv).
SUMMARY_CSV_NAME = "fov_summary.csv"

# Fallback stem for the calibration feature-viewer CSV / PNG folders when the manager did
# not pass one. The viewer derives the sibling PNG folders from the CSV stem
# (<stem>_png / <stem>_mask_png -- see feature_viewer/data.py), so the CSV name and the
# folder names all share this stem.
DEFAULT_MATRIX_STEM = "fov_feature_matrix"


def save_mask_overlay_png(
    path,
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (255, 0, 255),
    thickness: int = 2,
) -> None:
    """Save a PNG of ``image`` (contrast-stretched grayscale) with the ``mask`` object
    OUTLINES drawn on top as a bright border (default magenta), so the segmentation is
    visible without covering the cells it segments.

    The image is 1-99 percentile stretched to 8-bit grayscale (as the plain projection PNG
    is), converted to RGB, and each object's boundary pixels are painted ``color``.
    ``thickness`` px wide boundary (dilated for visibility). Shared by the live FOV-selection
    worker and the offline PNG exporter so both produce identical overlays.
    """
    import imageio.v3 as iio

    from skimage.segmentation import find_boundaries

    img = np.asarray(image, np.float32)
    finite = img[np.isfinite(img)]
    if finite.size:
        lo, hi = np.percentile(finite, (1.0, 99.0))
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        lo, hi = float(img.min()), float(img.max())
    gray = np.zeros_like(img) if hi <= lo else np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    rgb = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2)
    boundaries = find_boundaries(np.asarray(mask), mode="outer")
    if thickness > 1 and boundaries.any():
        from scipy.ndimage import binary_dilation

        boundaries = binary_dilation(boundaries, iterations=int(thickness) - 1)
    rgb[boundaries] = np.asarray(color, np.uint8)
    iio.imwrite(path, rgb)


def _single(mapping: dict | None) -> tuple[str, np.ndarray] | None:
    """The sole ``(channel, array)`` of a single-channel projections/masks dict, or ``None``.

    FOV selection segments exactly one mask, so :func:`decide_fov` returns one-entry
    ``projections`` / ``masks`` dicts. Returns ``None`` for an empty/absent dict so a caller
    can skip the artifact rather than crash.
    """
    if not mapping:
        return None
    return next(iter(mapping.items()))


def write_decision_artifacts(
    debug_dir: Path,
    name: str,
    proba: float,
    artifacts: dict,
) -> None:
    """Persist one FOV's lightweight decision artifacts (save_decision).

    Layout -- one folder per image kind so the projections and mask overlays can be flipped
    through separately::

        <debug_dir>/projection_png/<name>.png   # grayscale, min/max stretched
        <debug_dir>/mask_png/<name>.png         # magenta segmentation outline over it
        <debug_dir>/fov_summary.csv             # one row per FOV

    The combined ``fov_summary.csv`` gets a row per FOV -- ``name, proba`` followed by every
    feature column. The ``selected`` / ``position`` / ``rank`` decision columns are added
    post-drain by the manager (via :func:`finalize_summary_csv`).

    The full per-step 3D reconstruction is written separately (see
    :func:`write_reconstruction_zarr`, gated by ``save_pre_scan_omezarr``).
    """
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    safe = file_stem_name(name)

    proj = _single(artifacts.get("projections"))
    mask = _single(artifacts.get("masks"))
    if proj is not None:
        _save_projection_png(_fov_png_path(debug_dir, "projection_png", safe), proj[1])
    if mask is not None:
        mask_path = _fov_png_path(debug_dir, "mask_png", safe)
        # magenta segmentation outline over the projection it segments (cells stay visible);
        # fall back to a colorized label map if the projection is somehow missing.
        if proj is not None:
            save_mask_overlay_png(mask_path, proj[1], mask[1])
        else:
            _save_label_png(mask_path, mask[1])

    _append_summary_row(debug_dir, name, proba, artifacts.get("features"))


def write_feature_viewer_artifacts(
    debug_dir: Path,
    matrix_stem: str | None,
    name: str,
    artifacts: dict,
) -> None:
    """Persist one FOV's artifacts in the feature viewer's STANDARD layout (calibration).

    Same images as :func:`write_decision_artifacts`, but under the ``<stem>``-prefixed folder
    names and with the ``filename`` join column that
    :mod:`shrimpy.fov_selection.feature_viewer.data` requires, so a calibration pre-scan drops
    straight into the viewer with no conversion::

        <debug_dir>/<stem>.csv            # one row per FOV; carries a `filename` column
        <debug_dir>/<stem>_png/           # projection (brightfield slot) PNG per FOV
        <debug_dir>/<stem>_mask_png/      # segmentation-mask PNG per FOV

    where ``<stem>`` is ``matrix_stem`` (e.g. ``<acq>_fov_feature_matrix``). The CSV
    ``filename`` column equals each PNG's stem (the sanitized FOV name), the strict 1:1 join
    the viewer relies on. Every producible feature column is written (the worker ran with
    ``extract_all``); the ranking ``proba`` is deliberately omitted so it does not show up as
    a pseudo-feature axis in the viewer. The single segmented channel's projection wires to
    the viewer's default (brightfield) thumbnail slot.
    """
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = matrix_stem or DEFAULT_MATRIX_STEM
    safe = file_stem_name(name)

    proj = _single(artifacts.get("projections"))
    masks = artifacts.get("masks") or {}
    if proj is not None:
        channel, proj_img = proj
        _save_projection_png(_fov_png_path(debug_dir, f"{stem}_png", safe), proj_img)
        if channel in masks:
            # magenta segmentation outline over the projection (cells stay visible)
            save_mask_overlay_png(
                _fov_png_path(debug_dir, f"{stem}_mask_png", safe), proj_img, masks[channel]
            )

    _append_feature_viewer_row(debug_dir, stem, safe, artifacts.get("features"))


def _fov_png_path(debug_dir: Path, subfolder: str, safe_name: str) -> Path:
    """``<debug_dir>/<subfolder>/<safe_name>.png``, creating the subfolder."""
    folder = debug_dir / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{safe_name}.png"


def _append_feature_viewer_row(debug_dir: Path, stem: str, filename: str, features) -> None:
    """Append one FOV's row (``filename`` + all feature columns) to ``<stem>.csv``.

    Uses pandas concat so FOVs with different feature columns (e.g. no objects -> only the
    cheap features) still align, missing columns filled with NaN -- the viewer treats an
    all-NaN column within a dataset as absent. ``filename`` (== the PNG stem) is inserted
    first so the viewer's 1:1 CSV-to-image join works.
    """
    import pandas as pd

    row = features.copy() if features is not None else pd.DataFrame([{}])
    row.insert(0, "filename", filename)

    csv_path = debug_dir / f"{stem}.csv"
    if csv_path.exists():
        row = pd.concat([pd.read_csv(csv_path), row], ignore_index=True)
    row.to_csv(csv_path, index=False)


def append_best_focus_z_row(
    debug_dir: Path,
    stem: str | None,
    name: str,
    z_step_um: float,
    artifacts: dict,
) -> None:
    """Append the detected best-focus Z slice(s) for one FOV to a debug CSV.

    Written only when ``fov_selection.save_best_focus_z_for_debug`` is set and the
    projection is ``best_focus_z``. One row per FOV per segmented channel::

        fov, channel, best_focus_slice, n_slices, z_step_um, best_focus_z_um

    ``best_focus_slice`` is the 0-based Z index picked by the focus finder and
    ``best_focus_z_um = best_focus_slice * z_step_um`` its depth from the first slice
    (exact when no deskew is applied; the acquisition Z step otherwise only approximates
    the reconstructed axial spacing). Written to ``<stem>_best_focus_z.csv`` (or
    ``best_focus_z.csv`` when there is no stem).
    """
    import pandas as pd

    index = artifacts.get("best_focus_index") or {}
    if not index:
        return

    rows = [
        {
            "fov": name,
            "channel": channel,
            "best_focus_slice": info["slice"],
            "n_slices": info["n_slices"],
            "z_step_um": z_step_um,
            "best_focus_z_um": info["slice"] * z_step_um,
        }
        for channel, info in index.items()
    ]
    row = pd.DataFrame(rows)

    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    fname = f"{stem}_best_focus_z.csv" if stem else "best_focus_z.csv"
    csv_path = Path(debug_dir) / fname
    if csv_path.exists():
        row = pd.concat([pd.read_csv(csv_path), row], ignore_index=True)
    row.to_csv(csv_path, index=False)


def _assemble_debug_channels(artifacts: dict) -> tuple[list[str], np.ndarray | None]:
    """Stack every debug artifact into one ``(C, Z, Y, X)`` float32 volume.

    Channel order (only those present are emitted, and the set is fixed by the
    pipeline config so it is identical for every FOV in a run):

        ``deskew``, ``phase``, ``<ch>_vs`` (per VS target),
        ``<ch>_projection``, ``<ch>_mask``

    The 3D ``deskew`` / ``phase`` / ``<ch>_vs`` stacks share one ``(Z, Y, X)`` (the
    deskewed shape that phase and VS both operate on). The 2D projection and mask
    are broadcast to that same Z (identical plane on every slice) so a single
    channel axis lines them all up in one store. Labels are cast to float32 (a
    single OME-Zarr array is one dtype); exact for label ids up to 2**24.
    """
    stacks = artifacts.get("stacks") or {}
    projections = artifacts.get("projections") or {}
    masks = artifacts.get("masks") or {}

    reference = next((v for v in stacks.values() if getattr(v, "ndim", 0) == 3), None)
    if reference is None:
        return [], None
    nz = reference.shape[0]  # the (Z, Y, X) 2D planes are broadcast to; only Z is needed here

    names: list[str] = []
    arrays: list[np.ndarray] = []

    def _add_3d(name: str, vol: np.ndarray) -> None:
        names.append(name)
        arrays.append(np.asarray(vol, np.float32))

    def _add_2d_broadcast(name: str, plane: np.ndarray) -> None:
        plane = np.asarray(plane, np.float32)
        names.append(name)
        arrays.append(np.broadcast_to(plane, (nz, *plane.shape)))

    for key in ("deskew", "phase"):
        if key in stacks:
            _add_3d(key, stacks[key])
    for channel in (k for k in stacks if k not in ("deskew", "phase")):
        _add_3d(f"{channel}_vs", stacks[channel])
    for channel, proj in projections.items():
        _add_2d_broadcast(f"{channel}_projection", proj)
    for channel, mask in masks.items():
        _add_2d_broadcast(f"{channel}_mask", mask)

    return names, np.stack(arrays, axis=0)


def write_reconstruction_zarr(zarr_path: Path, p_idx: int, name: str, artifacts: dict) -> None:
    """Append one FOV's reconstruction stages to the shared pre-scan HCS store.

    Writes ``<name>_prescan.ome.zarr`` with one position per FOV (path
    ``0/<p_idx>/<name>``) and the channels from :func:`_assemble_debug_channels` --
    every reconstruction step (deskew / phase / vs volumes in 3D, plus the 2D
    projection / mask broadcast across Z). Created on the first FOV (with the
    channel names) and reopened read/write for each subsequent FOV; the worker
    decides one FOV at a time, so no locking is needed.
    """
    channel_names, czyx = _assemble_debug_channels(artifacts)
    if czyx is None:
        logger.warning("FOV selection: no 3D stacks to write for %s; skipping zarr", name)
        return

    from iohub.ngff import open_ome_zarr

    nc, nz, ny, nx = czyx.shape
    zarr_path = Path(zarr_path)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    if zarr_path.exists():
        store = open_ome_zarr(str(zarr_path), mode="r+")
    else:
        store = open_ome_zarr(
            str(zarr_path),
            layout="hcs",
            mode="w",
            channel_names=channel_names,
            version="0.5",
        )
    try:
        # iohub path parts must be alphanumeric (e.g. "B4_0000" -> "B40000").
        pos_name = zarr_path_name(name, f"p{p_idx}")
        pos = store.create_position("0", str(p_idx), pos_name)
        image = pos.create_zeros(
            "0",
            shape=(1, nc, nz, ny, nx),
            chunks=(1, 1, min(32, nz), ny, nx),
            dtype=np.float32,
        )
        image[0] = czyx
        logger.info(
            "FOV selection: wrote pre-scan reconstruction for %s (channels=%s, zyx=%s)",
            name,
            channel_names,
            (nz, ny, nx),
        )
    finally:
        store.close()


def _append_summary_row(debug_dir: Path, name: str, proba: float, features) -> None:
    """Append one FOV's row (name, proba, + all features) to fov_summary.csv.

    Uses pandas concat so FOVs with different feature columns (e.g. no objects) still align
    -- missing columns are filled with NaN.

    The ``selected`` / ``position`` / ``rank`` columns are NOT written here: they are
    properties of the whole pre-scan (a per-position top-K ranking), not of one FOV, and
    are only known after every FOV has been scored. The manager fills them in once,
    post-drain (:func:`finalize_summary_csv`). The per-FOV ``good`` flag is deliberately
    not recorded either -- for a ranking model it is a threshold artifact that has nothing
    to do with what actually gets imaged.
    """
    import pandas as pd

    row = features.copy() if features is not None else pd.DataFrame([{}])
    row.insert(0, "proba", float(proba))
    row.insert(0, "name", name)

    summary_path = debug_dir / SUMMARY_CSV_NAME
    if summary_path.exists():
        row = pd.concat([pd.read_csv(summary_path), row], ignore_index=True)
    row.to_csv(summary_path, index=False)


def finalize_summary_csv(
    summary_path: Path, passed: set[str], fov_group: dict[str, str], top_fov: int | None
) -> None:
    """Add the whole-run ``selected`` / ``position`` / ``rank`` columns to ``fov_summary.csv``.

    The worker appends one row per FOV as it is decided (``name, proba, <features>``), but
    whether a FOV is actually imaged is a whole-run property -- for
    ``ranking_by_defined_range`` the top-``top_fov`` cut over every score, which does not exist
    until the last FOV is scored. So the decision columns are written here, once, over the
    finished table:

    ``selected`` : 1 for the FOVs the timelapse images (``passed``), 0 otherwise.
    ``position`` : the well / grid center the FOV belongs to (``fov_group``).
    ``rank``     : 1 = highest score WITHIN its position (ties broken by score order, so
                   ``selected`` is exactly ``rank <= top_fov``). Only meaningful for ranking
                   models; left NaN when ``top_fov`` is None (per-FOV pass/fail, no ordering).

    Every filesystem step is guarded: this runs at the very end of the pre-scan and must not
    be able to raise out of ``teardown_sequence`` and take the acquisition down. The
    ``summary_path`` is assumed to exist (the caller checks); a read failure is logged, not
    raised.
    """
    import pandas as pd

    summary_path = Path(summary_path)
    try:
        df = pd.read_csv(summary_path)
    except Exception:
        logger.exception("FOV selection: could not read %s to finalize", summary_path)
        return
    if "name" not in df.columns:
        logger.warning("FOV selection: %s has no 'name' column; skipping", summary_path)
        return

    df = df.drop(
        columns=[c for c in ("selected", "rank", "position", "good") if c in df.columns]
    )
    names = df["name"].astype(str)
    selected = names.isin(passed).astype(int)
    position = names.map(lambda n: fov_group.get(n, n))
    if top_fov is not None:
        # Rank WITHIN each position, matching the per-position top_fov quota, so that
        # `selected == (rank <= top_fov)` holds inside every position. method='first' so
        # ties resolve to distinct consecutive integers (a dense/average rank would emit
        # 1.5-style values); NaN scores rank last.
        rank = df.groupby(position)["proba"].rank(ascending=False, method="first")
    else:
        rank = pd.Series(np.nan, index=df.index)
    df.insert(2, "rank", rank)
    df.insert(2, "position", position)
    df.insert(2, "selected", selected)

    try:
        df.to_csv(summary_path, index=False)
    except OSError as exc:
        # Typically the file is locked by a spreadsheet app watching the run (Windows
        # gives PermissionError). Don't lose the columns: write them beside the original
        # so the selection is still recoverable, and say plainly what happened.
        fallback = summary_path.with_name(f"{summary_path.stem}_selected.csv")
        try:
            df.to_csv(fallback, index=False)
        except OSError:
            logger.exception(
                "FOV selection: could not write selected/rank to %s or %s; the "
                "per-FOV scores in the log are the only record of the selection",
                summary_path,
                fallback,
            )
            return
        logger.warning(
            "FOV selection: could not write %s (%s) -- is it open in another program? "
            "Wrote the selected/rank table to %s instead.",
            summary_path,
            exc,
            fallback,
        )
        summary_path = fallback

    logger.info(
        "FOV selection: wrote selected/rank for %d FOVs (%d selected) to %s",
        len(df),
        int(selected.sum()),
        summary_path,
    )


def _save_projection_png(path: Path, proj: np.ndarray) -> None:
    """Save a contrast-stretched 8-bit grayscale PNG of a float projection.

    Stretches the 1st-99th percentile to 0-255 so the (wide-range, possibly
    negative) float32 projection is visible instead of near-black.
    """
    import imageio.v3 as iio

    proj = np.asarray(proj, np.float32)
    lo, hi = np.percentile(proj, (1.0, 99.0))
    if hi <= lo:
        lo, hi = float(proj.min()), float(proj.max())
    scaled = np.zeros_like(proj) if hi <= lo else (proj - lo) / (hi - lo)
    iio.imwrite(path, (np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8))


def _save_label_png(path: Path, mask: np.ndarray) -> None:
    """Save a colorized-label PNG of an integer label mask.

    Each object gets a distinct color (background black); a grayscale PNG of
    small label IDs would still look near-black.
    """
    import imageio.v3 as iio

    from skimage.color import label2rgb

    rgb = label2rgb(np.asarray(mask), bg_label=0)
    iio.imwrite(path, (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8))
