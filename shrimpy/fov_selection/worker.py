"""FOV-selection worker process.

Runs the heavy per-FOV decision (deskew, phase reconstruction, virtual
staining, Cellpose segmentation, tree prediction) in a separate process with
its own GIL and GPU context, so torch/OpenMP does not interfere with the
acquisition thread -- the same isolation DynaTrack uses (see
:class:`shrimpy.dynatrack.worker.DynaTrackWorker`).

The worker builds its preprocessor / Cellpose model / trained tree once at
startup, then decides one FOV per ``decide`` message and returns
``(proba, good)``.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time as _time

from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# The combined per-FOV debug table (one row per FOV) written under `debug_dir` when
# `save_decision` is set. Named here because the manager reopens it post-drain to add the
# selected/rank columns (FovSelection.finalize_debug_summary).
SUMMARY_CSV_NAME = "fov_summary.csv"

# Fallback stem for the calibration feature-viewer CSV / PNG folders when the manager did
# not pass one. The viewer derives the sibling PNG folders from the CSV stem
# (<stem>_png / <stem>_mask_png -- see feature_viewer/data.py), so the CSV name and the
# folder names all share this stem.
DEFAULT_MATRIX_STEM = "fov_feature_matrix"


class FovSelectionWorker:
    """Manages a subprocess that runs the FOV-selection decision.

    Parameters
    ----------
    recon : dict
        Scale-injected reconstruction sub-config (``preprocessing``, ``deskew``,
        ``phase``, ``virtual_staining``) for :func:`shrimpy.preprocessing.build_preprocessor`.
    target_channels : list[str]
        Reconstructed channels to segment/feature (e.g. ``['nuclei', 'membrane']``).
    segmentation : dict
        Segmentation config block (model, thresholds, per-organelle diameters).
    model_cfg : dict
        The ``fov_selection.model`` config block, passed to
        :func:`shrimpy.fov_selection.fov_model.build_fov_model`. Its ``type`` (one of
        :data:`shrimpy.fov_selection.fov_model.MODEL_TYPES`) selects the model;
        ``classification_tree`` additionally carries a ``path`` to a trained ``.joblib``.
    projection : str
        Projection method (``'sum'`` / ``'max'``).
    threshold : float
        P(good) cutoff.
    px_um : float
        XY pixel size in microns.
    zyx_shape : tuple[int, int, int]
        Acquired (Z, Y, X) stack shape (for the phase transfer function).
    log_file_path : Path | None
        Log file the subprocess should append to.
    calibration_mode : bool
        Extract every producible feature (not just the model's) and write the debug
        artifacts in the feature viewer's standard layout (see
        :func:`_write_feature_viewer_artifacts`).
    matrix_stem : str | None
        CSV / PNG-folder stem for the calibration feature-viewer output, e.g.
        ``"<acq>_fov_feature_matrix"``.
    """

    def __init__(
        self,
        recon: dict,
        target_channels: list[str],
        segmentation: dict,
        model_cfg: dict,
        projection: str,
        threshold: float,
        px_um: float,
        zyx_shape: tuple[int, int, int],
        log_file_path: Path | None = None,
        debug_dir: Path | None = None,
        recon_zarr_path: Path | None = None,
        require_gpu: bool = True,
        calibration_mode: bool = False,
        matrix_stem: str | None = None,
        best_focus_z: dict | None = None,
        z_step_um: float = 1.0,
        save_best_focus_z: bool = False,
        write_debug_artifacts: bool = True,
    ) -> None:
        self._recon = recon
        self._target_channels = target_channels
        self._segmentation = segmentation
        self._model_cfg = model_cfg
        self._projection = projection
        self._threshold = threshold
        self._px_um = px_um
        self._zyx_shape = zyx_shape
        self._log_file_path = log_file_path
        # When set, the lightweight per-FOV debug artifacts (projection/mask PNGs +
        # fov_summary.csv) are written here (save_decision).
        self._debug_dir = debug_dir
        # When set, the per-step reconstruction OME-Zarr (deskew / phase / vs /
        # projection / mask channels, one position per FOV) is written to this path
        # -- the <name>_prescan.ome.zarr store (save_pre_scan_omezarr).
        self._recon_zarr_path = recon_zarr_path
        # Fail fast if the reconstruction can't run on a GPU (fov_selection.require_gpu).
        self._require_gpu = require_gpu
        # Calibration mode: extract EVERY producible feature (not just the model's) and
        # write the debug artifacts in the feature viewer's standard layout
        # (<matrix_stem>.csv with a `filename` column + sibling <matrix_stem>_png/
        # <matrix_stem>_mask_png/ folders) so the pre-scan output loads straight into the
        # viewer. matrix_stem is the CSV/folder stem, e.g. "<acq>_fov_feature_matrix".
        self._calibration_mode = calibration_mode
        self._matrix_stem = matrix_stem
        # Optics for the 'best_focus_z' projection (numerical_aperture_detection,
        # wavelength_illumination, ...); None for the other projection methods.
        self._best_focus_z = best_focus_z
        # Acquisition Z step (um), used to turn the best-focus slice index into a depth.
        self._z_step_um = z_step_um
        # When set (fov_selection.save_best_focus_z_for_debug) and the projection is
        # 'best_focus_z', append the detected slice / depth per FOV to a debug CSV.
        self._save_best_focus_z = save_best_focus_z
        # Write the standard PNG/feature debug artifacts (save_decision / calibration).
        # Kept separate so save_best_focus_z_for_debug can add its CSV without also
        # forcing the projection/mask PNGs when only the focus CSV was requested.
        self._write_debug_artifacts = write_debug_artifacts
        self._process: mp.Process | None = None
        self._input_queue: mp.Queue | None = None
        self._output_queue: mp.Queue | None = None

    def start(self) -> None:
        """Spawn the worker process and wait for it to be ready."""
        ctx = mp.get_context("spawn")
        self._input_queue = ctx.Queue()
        self._output_queue = ctx.Queue()

        self._process = ctx.Process(
            target=_worker_loop,
            args=(
                self._recon,
                self._target_channels,
                self._segmentation,
                self._model_cfg,
                self._projection,
                self._threshold,
                self._px_um,
                self._zyx_shape,
                self._input_queue,
                self._output_queue,
                self._log_file_path,
                self._debug_dir,
                self._recon_zarr_path,
                self._require_gpu,
                self._calibration_mode,
                self._matrix_stem,
                self._best_focus_z,
                self._z_step_um,
                self._save_best_focus_z,
                self._write_debug_artifacts,
            ),
            daemon=True,
        )
        self._process.start()

        msg = self._output_queue.get(timeout=600)
        if msg["type"] != "ready":
            raise RuntimeError(f"FOV-selection worker failed to start: {msg}")
        logger.info("FOV-selection worker process started (pid=%d)", self._process.pid)

    def submit(
        self, timepoint_index: int, position_index: int, name: str, data: list[np.ndarray]
    ) -> None:
        """Send a decision job to the worker (non-blocking)."""
        self._input_queue.put(
            {
                "type": "decide",
                "timepoint_index": timepoint_index,
                "position_index": position_index,
                "name": name,
                "data": data,
            }
        )

    def get_result(self, timeout: float = 600) -> dict | None:
        """Block for the next result; return the message dict or ``None``."""
        try:
            msg = self._output_queue.get(timeout=timeout)
            if msg["type"] == "result":
                return msg
            if msg["type"] == "error":
                logger.error("FOV-selection worker error: %s", msg["error"])
                return None
        except Exception:
            return None
        return None

    def shutdown(self, timeout: float = 120) -> None:
        """Signal the worker to stop and wait for it to finish."""
        if self._input_queue is not None:
            self._input_queue.put({"type": "shutdown"})
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                logger.warning("FOV-selection worker did not exit, terminating")
                self._process.terminate()
        self._process = None


def _worker_loop(
    recon: dict,
    target_channels: list[str],
    segmentation: dict,
    model_cfg: dict,
    projection: str,
    threshold: float,
    px_um: float,
    zyx_shape: tuple[int, int, int],
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    log_file_path: Path | None = None,
    debug_dir: Path | None = None,
    recon_zarr_path: Path | None = None,
    require_gpu: bool = True,
    calibration_mode: bool = False,
    matrix_stem: str | None = None,
    best_focus_z: dict | None = None,
    z_step_um: float = 1.0,
    save_best_focus_z: bool = False,
    write_debug_artifacts: bool = True,
) -> None:
    """Main loop for the FOV-selection worker process."""
    # Mirror DynaTrack's subprocess logging: file handler to the parent's log
    # file plus a console handler for live visibility.
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s.%(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    shrimpy_logger = logging.getLogger("shrimpy")
    shrimpy_logger.setLevel(logging.DEBUG)
    if log_file_path is not None:
        file_handler = logging.FileHandler(str(log_file_path))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        shrimpy_logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)
    shrimpy_logger.addHandler(stream_handler)

    log = logging.getLogger("shrimpy.fov_selection.worker")

    try:
        from shrimpy.fov_selection import fov_model as fov_model_lib
        from shrimpy.fov_selection import pipeline
        from shrimpy.fov_selection.segmentation import build_segmenter
        from shrimpy.preprocessing import build_preprocessor

        log.info("FOV-selection worker: initializing for shape %s...", zyx_shape)

        preprocessor = build_preprocessor(
            zyx_shape=zyx_shape,
            preprocessing=recon.get("preprocessing"),
            deskew=recon.get("deskew"),
            phase=recon.get("phase"),
            virtual_staining=recon.get("virtual_staining"),
            output_channel=recon.get("output_channel", "phase"),
            require_gpu=require_gpu,
        )
        if preprocessor is None:
            raise ValueError(
                "fov_selection.reconstruction produced no preprocessor; a "
                "'deskew'/'phase'/'vs' pipeline is required to make nuclei/membrane."
            )
        segmenter = build_segmenter(segmentation)
        # Pluggable FOV model built from the config (trained .joblib via 'path', or a
        # hand-tuned 'type'); decide_fov calls model.predict, agnostic to the type.
        model = fov_model_lib.build_fov_model(model_cfg)

        output_queue.put({"type": "ready"})
        log.info(
            "FOV-selection worker: reconstruction + %s segmentation + %s (features=%s) ready",
            (segmentation or {}).get("model", "cellpose"),
            type(model).__name__,
            model.feature_names,
        )

    except Exception as e:
        output_queue.put({"type": "error", "error": str(e)})
        return

    while True:
        try:
            msg = input_queue.get()
        except Exception:
            break

        if msg["type"] == "shutdown":
            log.info("FOV-selection worker: shutting down")
            break

        if msg["type"] == "decide":
            t0 = _time.monotonic()
            t_idx = msg["timepoint_index"]
            p_idx = msg["position_index"]
            name = msg["name"]
            data = msg["data"]
            bf_zyx = np.stack(data, axis=0) if isinstance(data, list) else np.asarray(data)

            log.info(
                "FOV-selection worker: deciding %s (p=%d, %d slices)",
                name,
                p_idx,
                bf_zyx.shape[0],
            )
            try:
                # Artifacts are needed for either debug output; the heavy per-step
                # 3D stacks are needed only for the reconstruction store.
                need_artifacts = debug_dir is not None or recon_zarr_path is not None
                result = pipeline.decide_fov(
                    preprocessor,
                    segmenter,
                    model,
                    bf_zyx,
                    target_channels=target_channels,
                    projection=projection,
                    px_um=px_um,
                    threshold=threshold,
                    best_focus_z=best_focus_z,
                    return_artifacts=need_artifacts,
                    return_stacks=recon_zarr_path is not None,
                    extract_all=calibration_mode,
                    label=name,
                )
                if need_artifacts:
                    proba, good, artifacts = result
                    # Debug output must NEVER invalidate a decision. These writers touch the
                    # filesystem, so they fail for reasons that have nothing to do with the
                    # science -- most commonly fov_summary.csv being locked by a spreadsheet
                    # app while the run is watched. Letting that propagate would discard an
                    # already-computed verdict (the FOV is scored NaN/bad) and, if it happens
                    # to enough FOVs, leave nothing to image. Log and carry on instead.
                    if recon_zarr_path is not None:
                        try:
                            _write_reconstruction_zarr(recon_zarr_path, p_idx, name, artifacts)
                        except Exception:
                            log.exception(
                                "FOV-selection worker: could not write the pre-scan "
                                "reconstruction for %s; the decision is unaffected",
                                name,
                            )
                    if debug_dir is not None and save_best_focus_z:
                        try:
                            _append_best_focus_z_row(
                                debug_dir, matrix_stem, name, z_step_um, artifacts
                            )
                        except Exception:
                            log.exception(
                                "FOV-selection worker: could not write the best-focus-Z "
                                "debug CSV for %s; the decision is unaffected",
                                name,
                            )
                    if debug_dir is not None and write_debug_artifacts:
                        try:
                            if calibration_mode:
                                _write_feature_viewer_artifacts(
                                    debug_dir, matrix_stem, name, artifacts
                                )
                            else:
                                _write_decision_artifacts(
                                    debug_dir, name, proba, good, artifacts
                                )
                        except Exception:
                            log.exception(
                                "FOV-selection worker: could not write debug artifacts for "
                                "%s (is %s open in another program?); the decision is "
                                "unaffected",
                                name,
                                Path(debug_dir) / SUMMARY_CSV_NAME,
                            )
                else:
                    proba, good = result
                elapsed = _time.monotonic() - t0
                output_queue.put(
                    {
                        "type": "result",
                        "position_index": p_idx,
                        "timepoint_index": t_idx,
                        "name": name,
                        "proba": proba,
                        "good": good,
                        "elapsed": elapsed,
                    }
                )
                log.info(
                    "FOV-selection worker: %s -> score=%.3f (%.1fs)",
                    name,
                    proba,
                    elapsed,
                )
            except Exception as e:
                log.exception("FOV-selection worker: decision failed for %s", name)
                output_queue.put(
                    {
                        "type": "error",
                        "error": str(e),
                        "position_index": p_idx,
                        "timepoint_index": t_idx,
                        "name": name,
                    }
                )

            # Release this job's GPU working set (as DynaTrack does).
            try:
                import torch as _torch

                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass


def _write_decision_artifacts(
    debug_dir: Path,
    name: str,
    proba: float,
    good: bool,
    artifacts: dict,
) -> None:
    """Persist one FOV's lightweight decision artifacts (save_decision).

    Layout: all files live directly in ``<debug_dir>/`` (no per-FOV subfolders, so
    every FOV's images can be browsed together). Per FOV per target channel:
    ``<name>_<ch>_projection.png`` (grayscale, min/max stretched) and
    ``<name>_<ch>_mask.png`` (colorized labels) for quick visual QC without a zarr
    viewer. One combined ``fov_summary.csv`` gets a row per FOV -- ``name, proba``
    followed by every feature column -- so all FOVs are viewable at a glance. The
    ``selected`` / ``rank`` decision columns are added post-drain by the manager
    (see :meth:`~shrimpy.fov_selection.manager.FovSelection.finalize_debug_summary`).

    The full per-step 3D reconstruction is written separately as the pre-scan
    OME-Zarr (see :func:`_write_reconstruction_zarr`, gated by
    ``save_pre_scan_omezarr``). Runs in the worker subprocess one FOV at a time, so
    the summary read/append/write is unsynchronized-safe.
    """
    from pathlib import Path as _Path

    debug_dir = _Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if (c.isalnum() or c in "_-") else "_" for c in str(name)) or "fov"

    from shrimpy.fov_selection.pipeline import save_mask_overlay_png

    projections = artifacts.get("projections") or {}
    masks = artifacts.get("masks") or {}
    for channel, proj in projections.items():
        _save_projection_png(debug_dir / f"{safe}_{channel}_projection.png", proj)
    for channel, mask in masks.items():
        # the mask PNG is the segmentation OUTLINE (magenta) over the projection it segments,
        # so the cells stay visible. Fall back to a colorized label map if the projection is
        # somehow missing for this channel.
        mask_path = debug_dir / f"{safe}_{channel}_mask.png"
        if channel in projections:
            save_mask_overlay_png(mask_path, projections[channel], mask)
        else:
            _save_label_png(mask_path, mask)

    _append_summary_row(debug_dir, name, proba, good, artifacts.get("features"))


def _write_feature_viewer_artifacts(
    debug_dir: Path,
    matrix_stem: str | None,
    name: str,
    artifacts: dict,
) -> None:
    """Persist one FOV's artifacts in the feature viewer's STANDARD layout (calibration).

    Unlike :func:`_write_decision_artifacts` (the flat ``<name>_<ch>_*.png`` + ``proba``
    layout), this writes exactly what :mod:`shrimpy.fov_selection.feature_viewer.data`
    loads without any conversion, so a calibration pre-scan drops straight into the viewer::

        <debug_dir>/<stem>.csv            # one row per FOV; carries a `filename` column
        <debug_dir>/<stem>_png/           # projection (brightfield) PNG per FOV
        <debug_dir>/<stem>_mask_png/      # segmentation-mask PNG per FOV

    where ``<stem>`` is ``matrix_stem`` (e.g. ``<acq>_fov_feature_matrix``). The CSV
    ``filename`` column equals each PNG's stem (the sanitized FOV name), which is the strict
    1:1 join the viewer relies on. Every producible feature column is written (the worker
    ran with ``extract_all``); the ranking ``proba`` is deliberately omitted so it does not
    show up as a pseudo-feature axis in the viewer. Runs in the worker subprocess one FOV at
    a time, so the CSV read/append/write is unsynchronized-safe.

    A single segmented channel is the common (non-VS) calibration case: its projection is
    the brightfield thumbnail, its mask the mask thumbnail. With several channels (VS), the
    first is used as brightfield and the second (if any) as the fluor channel, matching the
    viewer's channel toggles.
    """
    from pathlib import Path as _Path

    debug_dir = _Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = matrix_stem or DEFAULT_MATRIX_STEM
    safe = "".join(c if (c.isalnum() or c in "_-") else "_" for c in str(name)) or "fov"

    from shrimpy.fov_selection.pipeline import save_mask_overlay_png

    # Map segmented channels onto the viewer's brightfield/fluor thumbnail slots (brightfield
    # is the default toggle); the mask always goes to the mask slot. Folder suffixes mirror
    # feature_viewer/data.py's _channel_png_folder ("_png" / "_<channel>_png").
    projections = list((artifacts.get("projections") or {}).items())
    masks = artifacts.get("masks") or {}
    for i, (channel, proj) in enumerate(projections):
        slot = "brightfield" if i == 0 else ("fluor" if i == 1 else channel)
        folder = debug_dir / (f"{stem}_png" if slot == "brightfield" else f"{stem}_{slot}_png")
        folder.mkdir(parents=True, exist_ok=True)
        _save_projection_png(folder / f"{safe}.png", proj)
        if channel in masks:
            mask_folder = debug_dir / f"{stem}_mask_png"
            mask_folder.mkdir(parents=True, exist_ok=True)
            # magenta segmentation outline over the projection (cells stay visible)
            save_mask_overlay_png(mask_folder / f"{safe}.png", proj, masks[channel])

    _append_feature_viewer_row(debug_dir, stem, safe, artifacts.get("features"))


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


def _append_best_focus_z_row(
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
    ``best_focus_z.csv`` when there is no stem). Runs one FOV at a time in the worker
    subprocess, so the read/append/write is unsynchronized-safe.
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
    nz, ny, nx = reference.shape

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
    for organelle in (k for k in stacks if k not in ("deskew", "phase")):
        _add_3d(f"{organelle}_vs", stacks[organelle])
    for organelle, proj in projections.items():
        _add_2d_broadcast(f"{organelle}_projection", proj)
    for organelle, mask in masks.items():
        _add_2d_broadcast(f"{organelle}_mask", mask)

    return names, np.stack(arrays, axis=0)


def _write_reconstruction_zarr(
    zarr_path: Path, p_idx: int, name: str, artifacts: dict
) -> None:
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

    from pathlib import Path

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
        pos_name = "".join(c for c in str(name) if c.isalnum()) or f"p{p_idx}"
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


def _append_summary_row(
    debug_dir: Path, name: str, proba: float, good: bool, features
) -> None:
    """Append one FOV's row (name, proba, + all features) to fov_summary.csv.

    Uses pandas concat so FOVs with different feature columns (e.g. no membrane
    objects) still align -- missing columns are filled with NaN.

    The ``selected`` / ``rank`` columns are NOT written here: both are properties of
    the whole pre-scan (top-K ranking across every FOV), not of one FOV, and are only
    known after every FOV has been scored. The manager fills them in once, post-drain
    (:meth:`shrimpy.fov_selection.manager.FovSelection.finalize_debug_summary`). The
    per-FOV ``good`` flag is deliberately not written -- for a ranking model it is a
    threshold artifact that has nothing to do with what actually gets imaged.
    """
    import pandas as pd

    row = features.copy() if features is not None else pd.DataFrame([{}])
    row.insert(0, "proba", float(proba))
    row.insert(0, "name", name)

    summary_path = debug_dir / SUMMARY_CSV_NAME
    if summary_path.exists():
        row = pd.concat([pd.read_csv(summary_path), row], ignore_index=True)
    row.to_csv(summary_path, index=False)


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
