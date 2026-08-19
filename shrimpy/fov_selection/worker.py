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

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shrimpy.fov_selection import debug_artifacts

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerConfig:
    """Everything the worker subprocess needs, assembled once by the manager and pickled
    into the process as a single value.

    Collecting the settings here (rather than threading ~20 positional args through
    ``FovSelectionWorker`` -> ``Process(args=...)`` -> ``_worker_loop``) means each field is
    named exactly once at the construction site and read by name in the loop; there is no
    positional args tuple to keep in sync. Frozen so it is a stable snapshot for the run.

    Fields
    ------
    recon : dict
        Scale-injected reconstruction sub-config (``preprocessing``, ``deskew``, ``phase``,
        ``virtual_staining``) for :func:`shrimpy.preprocessing.build_preprocessor`.
    target : str
        The object to segment and score (``'cells'`` | ``'nuclei'``); also the InstanSeg head.
    recon_channels : list[str]
        Reconstruction output channels to project (e.g. ``['nuclei', 'membrane']`` for a VS
        ``cells`` run, ``['nuclei']`` for VS ``nuclei``, or a single label-free channel);
        reduced to ONE segmentation input by ``target`` (see pipeline._resolve_seg_input).
    segmentation : dict
        Segmentation config block (model, thresholds, per-target diameters).
    model_cfg : dict
        The ``fov_selection.model`` config block, passed to
        :func:`shrimpy.fov_selection.fov_model.build_fov_model`.
    projection : str
        Projection method (``'sum'`` / ``'max'`` / ``'middle'`` / ``'logstd'`` /
        ``'best_focus_z'``).
    threshold : float
        P(good) cutoff.
    pixel_size_um : float
        XY pixel size in microns.
    zyx_shape : tuple[int, int, int]
        Acquired (Z, Y, X) stack shape (for the phase transfer function).
    log_file_path : Path | None
        Log file the subprocess should append to.
    debug_dir : Path | None
        Sibling directory for the lightweight per-FOV debug artifacts (projection/mask PNGs +
        fov_summary.csv); ``save_decision``.
    recon_zarr_path : Path | None
        Destination for the per-step reconstruction OME-Zarr (one position per FOV);
        ``save_pre_scan_omezarr``.
    require_gpu : bool
        Fail fast if the reconstruction cannot run on a GPU (``fov_selection.require_gpu``).
    calibration_mode : bool
        Extract every producible feature (not just the model's) and write the debug artifacts
        in the feature viewer's standard layout (``<matrix_stem>.csv`` + sibling PNG folders),
        so the pre-scan output loads straight into the viewer.
    matrix_stem : str | None
        CSV / PNG-folder stem for the calibration feature-viewer output, e.g.
        ``"<acq>_fov_feature_matrix"``.
    best_focus_z : dict | None
        Optics for the ``'best_focus_z'`` projection; ``None`` for the other methods.
    z_step_um : float
        Acquisition Z step (um), used to turn the best-focus slice index into a depth.
    save_best_focus_z : bool
        Append the detected best-focus slice / depth per FOV to a debug CSV
        (``save_best_focus_z_for_debug``); only meaningful with the ``best_focus_z`` projection.
    write_debug_artifacts : bool
        Write the standard PNG/feature debug artifacts (``save_decision`` / calibration). Kept
        separate from ``save_best_focus_z`` so the focus CSV can be requested on its own.
    """

    recon: dict
    target: str
    recon_channels: list[str]
    segmentation: dict
    model_cfg: dict
    projection: str
    threshold: float
    pixel_size_um: float
    zyx_shape: tuple[int, int, int]
    log_file_path: Path | None = None
    debug_dir: Path | None = None
    recon_zarr_path: Path | None = None
    require_gpu: bool = True
    calibration_mode: bool = False
    matrix_stem: str | None = None
    best_focus_z: dict | None = None
    z_step_um: float = 1.0
    save_best_focus_z: bool = False
    write_debug_artifacts: bool = True


class FovSelectionWorker:
    """Manages a subprocess that runs the FOV-selection decision.

    Built from a single :class:`WorkerConfig` (the manager assembles it once); the config is
    pickled into the spawned process as one value, so there is no long positional args list to
    keep aligned across ``start`` and :func:`_worker_loop`.
    """

    def __init__(self, config: WorkerConfig) -> None:
        self._config = config
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
            args=(self._config, self._input_queue, self._output_queue),
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
    config: WorkerConfig,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
) -> None:
    """Main loop for the FOV-selection worker process."""
    # Unpack the config once into locals so the body below reads plainly; every name comes
    # from the single WorkerConfig the manager built (no positional args to keep in sync).
    recon = config.recon
    target = config.target
    recon_channels = config.recon_channels
    segmentation = config.segmentation
    model_cfg = config.model_cfg
    projection = config.projection
    threshold = config.threshold
    pixel_size_um = config.pixel_size_um
    zyx_shape = config.zyx_shape
    log_file_path = config.log_file_path
    debug_dir = config.debug_dir
    recon_zarr_path = config.recon_zarr_path
    require_gpu = config.require_gpu
    calibration_mode = config.calibration_mode
    matrix_stem = config.matrix_stem
    best_focus_z = config.best_focus_z
    z_step_um = config.z_step_um
    save_best_focus_z = config.save_best_focus_z
    write_debug_artifacts = config.write_debug_artifacts

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
        # preprocessor may be None: a pipeline with no reconstruction step (raw brightfield
        # -> segment) is supported; decide_fov keys the raw stack as the single channel.
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
                    target=target,
                    recon_channels=recon_channels,
                    projection=projection,
                    pixel_size_um=pixel_size_um,
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
                            debug_artifacts.write_reconstruction_zarr(
                                recon_zarr_path, p_idx, name, artifacts
                            )
                        except Exception:
                            log.exception(
                                "FOV-selection worker: could not write the pre-scan "
                                "reconstruction for %s; the decision is unaffected",
                                name,
                            )
                    if debug_dir is not None and save_best_focus_z:
                        try:
                            debug_artifacts.append_best_focus_z_row(
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
                                debug_artifacts.write_feature_viewer_artifacts(
                                    debug_dir, matrix_stem, name, artifacts
                                )
                            else:
                                debug_artifacts.write_decision_artifacts(
                                    debug_dir, name, proba, artifacts
                                )
                        except Exception:
                            log.exception(
                                "FOV-selection worker: could not write debug artifacts for "
                                "%s (is %s open in another program?); the decision is "
                                "unaffected",
                                name,
                                Path(debug_dir) / debug_artifacts.SUMMARY_CSV_NAME,
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
