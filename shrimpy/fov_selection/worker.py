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

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


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
    model_path : str
        Path to the trained FOV-goodness ``.joblib``.
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
    """

    def __init__(
        self,
        recon: dict,
        target_channels: list[str],
        segmentation: dict,
        model_path: str,
        projection: str,
        threshold: float,
        px_um: float,
        zyx_shape: tuple[int, int, int],
        log_file_path: Path | None = None,
        debug_dir: Path | None = None,
        require_gpu: bool = True,
    ) -> None:
        self._recon = recon
        self._target_channels = target_channels
        self._segmentation = segmentation
        self._model_path = model_path
        self._projection = projection
        self._threshold = threshold
        self._px_um = px_um
        self._zyx_shape = zyx_shape
        self._log_file_path = log_file_path
        # When set, per-FOV debug artifacts (projections, masks, features,
        # decisions.csv) are written here (save_decision).
        self._debug_dir = debug_dir
        # Fail fast if the reconstruction can't run on a GPU (fov_selection.require_gpu).
        self._require_gpu = require_gpu
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
                self._model_path,
                self._projection,
                self._threshold,
                self._px_um,
                self._zyx_shape,
                self._input_queue,
                self._output_queue,
                self._log_file_path,
                self._debug_dir,
                self._require_gpu,
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
    model_path: str,
    projection: str,
    threshold: float,
    px_um: float,
    zyx_shape: tuple[int, int, int],
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    log_file_path: Path | None = None,
    debug_dir: Path | None = None,
    require_gpu: bool = True,
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
        from shrimpy.fov_selection import pipeline
        from shrimpy.preprocessing import build_preprocessor

        log.info("FOV-selection worker: initializing for shape %s...", zyx_shape)

        preprocessor = build_preprocessor(
            zyx_shape=zyx_shape,
            preprocessing=recon.get("preprocessing"),
            deskew=recon.get("deskew"),
            phase=recon.get("phase"),
            virtual_staining=recon.get("virtual_staining"),
            require_gpu=require_gpu,
        )
        if preprocessor is None:
            raise ValueError(
                "fov_selection.reconstruction produced no preprocessor; a "
                "'deskew'/'phase'/'vs' pipeline is required to make nuclei/membrane."
            )
        cellpose = pipeline.load_cellpose_model(segmentation)
        model = pipeline.load_fov_model(model_path)

        output_queue.put({"type": "ready"})
        log.info("FOV-selection worker: reconstruction + Cellpose + tree ready")

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
                result = pipeline.decide_fov(
                    preprocessor,
                    cellpose,
                    model,
                    bf_zyx,
                    target_channels=target_channels,
                    projection=projection,
                    px_um=px_um,
                    threshold=threshold,
                    segmentation=segmentation,
                    return_artifacts=debug_dir is not None,
                    label=name,
                )
                if debug_dir is not None:
                    proba, good, artifacts = result
                    _write_decision_artifacts(debug_dir, name, proba, good, artifacts)
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
                    "FOV-selection worker: %s -> proba=%.3f %s (%.1fs)",
                    name,
                    proba,
                    "GOOD" if good else "bad",
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
    """Persist one FOV's decision artifacts (save_decision).

    Layout: all files live directly in ``<debug_dir>/`` (no per-FOV subfolders, so
    every FOV's images can be browsed together). Per FOV per target channel:
    ``<name>_<ch>_projection.png`` (grayscale, min/max stretched) and
    ``<name>_<ch>_mask.png`` (colorized labels) for quick visual QC. One combined
    ``<debug_dir>/fov_summary.csv`` gets a row per FOV -- ``name, proba, good``
    followed by every feature column -- so all FOVs are viewable at a glance in one
    file. Runs in the worker subprocess one FOV at a time, so the summary
    read/append/write is unsynchronized-safe.
    """
    from pathlib import Path as _Path

    debug_dir = _Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if (c.isalnum() or c in "_-") else "_" for c in str(name)) or "fov"

    for channel, proj in (artifacts.get("projections") or {}).items():
        _save_projection_png(debug_dir / f"{safe}_{channel}_projection.png", proj)
    for channel, mask in (artifacts.get("masks") or {}).items():
        _save_label_png(debug_dir / f"{safe}_{channel}_mask.png", mask)

    _append_summary_row(debug_dir, name, proba, good, artifacts.get("features"))


def _append_summary_row(
    debug_dir: Path, name: str, proba: float, good: bool, features
) -> None:
    """Append one FOV's row (name, proba, good, + all features) to fov_summary.csv.

    Uses pandas concat so FOVs with different feature columns (e.g. no membrane
    objects) still align -- missing columns are filled with NaN.
    """
    import pandas as pd

    row = features.copy() if features is not None else pd.DataFrame([{}])
    row.insert(0, "good", int(bool(good)))
    row.insert(0, "proba", float(proba))
    row.insert(0, "name", name)

    summary_path = debug_dir / "fov_summary.csv"
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
