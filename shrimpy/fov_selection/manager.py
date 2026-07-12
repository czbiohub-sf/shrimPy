"""Engine-facing coordinator for online, streaming FOV selection.

Mirrors :class:`shrimpy.dynatrack.manager.DynaTrack`: an acquisition engine
builds a :class:`FovSelection` from the ``fov_selection`` metadata section and
interacts with that object only. It turns the pre-scan run (a single-timepoint
run on ``fov_selection_channel`` over all candidate FOVs) into a per-FOV good/bad verdict:

    BF z-stack -> reconstruct (deskew -> phase -> virtual stain)   [preprocessing.py]
               -> project -> segment -> features -> tree predict   [pipeline.py]

The decision is streamed: as each pre-scan FOV's z-stack completes in
``on_frame_ready`` it is submitted to a worker subprocess (torch/GPU isolation,
like DynaTrack). ``drain`` is awaited after the pre-scan run, and
``good_position_names`` selects which FOVs the timelapse run images.

Config lives under ``metadata.mantis.fov_selection``. Scale parameters (XY pixel
size, Z step) are the single source of truth injected into the deskew/phase
sub-configs (as DynaTrack does), so they are not duplicated in the config.
"""

from __future__ import annotations

import copy
import logging
import threading
import time

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from useq import MDAEvent, MDASequence

logger = logging.getLogger(__name__)

# Channels the decision needs (segmented + fed to the model).
DEFAULT_TARGET_CHANNELS = ["nuclei", "membrane"]

# Timepoint used for the pre-scan (first timepoint of the run).
PRESCAN_TIMEPOINT = 0


class FovSelection:
    """Coordinates the online, streaming FOV-selection decision for one run.

    Parameters
    ----------
    config : dict
        The ``fov_selection`` metadata block.
    sequence : MDASequence
        The acquisition sequence being run; provides the pre-scan channel list
        and the number of z-slices per stack.
    pixel_size_um : float
        XY pixel size (microns) -- injected into deskew/phase and used for
        physical feature units.
    z_step_um : float
        Z step (microns) -- injected into deskew (``scan_step_um``) and phase
        (``z_pixel_size``).
    decide_fn : callable | None
        Optional in-process decider ``(bf_zyx) -> (proba, good)``. When given,
        decisions run on the executor thread instead of the worker subprocess
        (used by tests and custom deciders); ``start`` then skips spawning the
        worker.
    """

    def __init__(
        self,
        config: dict,
        sequence: MDASequence,
        pixel_size_um: float,
        z_step_um: float,
        data_path: Path | None = None,
        decide_fn: Callable[[np.ndarray], tuple[float, bool]] | None = None,
    ) -> None:
        self.config = config
        self._pixel_size_um = pixel_size_um
        self._z_step_um = z_step_um
        # Optional per-FOV debug artifacts (segmentation, features, goodness),
        # written by the worker to a sibling directory next to the output store.
        self._save_decision = bool(config.get("save_decision", False))
        self._debug_dir = self._debug_dir_for(data_path) if self._save_decision else None
        # Fail fast if reconstruction can't run on a GPU. Default True; set
        # fov_selection.require_gpu: false to allow a (slow) CPU run for debugging.
        self._require_gpu = bool(config.get("require_gpu", True))
        # Acquired channel imaged during the pre-scan and fed to reconstruction.
        self._fov_selection_channel = config.get("fov_selection_channel", "BF - Oblique")
        # Ordered preprocessing steps (DynaTrack style), e.g.
        # ['deskew', 'phase', 'vs', 'sum_projection', 'segmentation']. The
        # reconstruction steps are consumed by build_preprocessor; projection and
        # segmentation are consumed here / in the pipeline.
        self._steps = list(config.get("preprocessing") or [])
        self._projection = self._projection_from_steps(self._steps)
        self._segmentation = config.get("segmentation", {}) or {}
        model_cfg = config.get("model", {}) or {}
        self._threshold = float(model_cfg.get("threshold", 0.5))
        # What the fov_selection_channels are: 'vs' (virtual-stained -> requires a
        # 'vs' preprocessing step) or 'fluor' (acquired fluorescence -> no VS).
        self._channels_type = config.get("fov_selection_channels_type", "vs")
        # Channels the decision is computed on (segmented + fed to the model);
        # may be raw input channels or preprocessed (VS) channels. Defaults to
        # the virtual-staining target channels.
        vs_cfg = config.get("virtual_staining", {}) or {}
        self._target_channels = list(
            config.get("fov_selection_channels")
            or vs_cfg.get("target_channels")
            or DEFAULT_TARGET_CHANNELS
        )

        self._validate_fov_selection_channel(sequence)
        self._validate_channels_type()
        self._require_segmentation_step()
        self._expected_slices = max(sequence.sizes.get("z", 1), 1)

        # Per-(timepoint, position) frame buffering for the pre-scan stacks.
        self._frames: dict[tuple[int, int], list[np.ndarray]] = {}
        self._names: dict[int, str] = {}

        # Verdicts, keyed by position name. Written from the executor thread,
        # read from the acquisition thread -> guarded by a lock.
        self._verdicts: dict[str, tuple[float, bool]] = {}
        self._verdicts_lock = threading.Lock()

        # Timing: when each FOV's z-stack finished acquiring (set in
        # on_frame_ready), used to measure stack-complete -> verdict latency.
        self._stack_done_at: dict[str, float] = {}
        self._decision_latencies: list[float] = []

        # Single-worker executor with at most one in-flight decision, so only
        # one FOV's frames are held past the acquisition of the next stack.
        self._executor: ThreadPoolExecutor | None = None
        self._pending: Future | None = None
        self._worker = None  # FovSelectionWorker (subprocess) unless decide_fn set
        self._decide_fn = decide_fn

    # -- construction ------------------------------------------------------

    @staticmethod
    def _debug_dir_for(data_path: Path | None) -> Path | None:
        """Sibling ``<name>_fov_debug/`` directory next to the output store."""
        if data_path is None:
            return None
        data_path = Path(data_path)
        name = data_path.name
        for suffix in (".ome.zarr", ".zarr"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        return data_path.with_name(f"{name}_fov_debug")

    @classmethod
    def from_metadata(
        cls,
        meta: dict | None,
        sequence: MDASequence,
        pixel_size_um: float,
        data_path: Path | None = None,
        decide_fn: Callable[[np.ndarray], tuple[float, bool]] | None = None,
    ) -> FovSelection | None:
        """Build the coordinator from the ``fov_selection`` metadata block.

        Returns ``None`` when FOV selection is disabled. Raises when it is
        enabled but no ``model_path`` is configured (fail before acquiring) or
        when the pixel size / z step needed for reconstruction are missing.
        """
        if not meta or not meta.get("enabled", False):
            return None
        if not (meta.get("model", {}) or {}).get("path"):
            raise ValueError(
                "FOV selection is enabled but no 'model.path' is configured under "
                "metadata.mantis.fov_selection. Provide a trained FOV-selection model "
                "(.joblib), or disable fov_selection. Aborting before acquisition."
            )
        if not pixel_size_um:
            raise ValueError(
                "FOV selection: pixel size is not set (core.getPixelSizeUm() returned "
                "0 or None); calibrate the pixel size in Micro-Manager."
            )
        z_step_um = getattr(sequence.z_plan, "step", None) if sequence.z_plan else None
        if not z_step_um:
            raise ValueError(
                "FOV selection: the sequence z_plan has no step; a stepped z_plan is "
                "required to derive the Z scale for reconstruction."
            )
        return cls(
            config=meta,
            sequence=sequence,
            pixel_size_um=pixel_size_um,
            z_step_um=z_step_um,
            data_path=data_path,
            decide_fn=decide_fn,
        )

    def _validate_fov_selection_channel(self, sequence: MDASequence) -> None:
        names = [ch.config for ch in sequence.channels]
        if self._fov_selection_channel not in names:
            raise ValueError(
                f"FOV selection fov_selection_channel {self._fov_selection_channel!r} is not one of "
                f"the acquisition channels {names}."
            )

    def _validate_channels_type(self) -> None:
        """Validate ``fov_selection_channels_type`` and guard the preprocessing steps.

        Only ``'vs'`` (virtual-stained) and ``'fluor'`` (acquired fluorescence)
        are supported. ``'vs'`` requires a ``'vs'`` step in the preprocessing list
        (the decision runs on virtual-stained channels); ``'fluor'`` does not.
        """
        valid = ("vs", "fluor")
        if self._channels_type not in valid:
            raise ValueError(
                f"fov_selection.fov_selection_channels_type must be one of {valid}; "
                f"got {self._channels_type!r}."
            )
        if self._channels_type == "vs" and "vs" not in self._steps:
            raise ValueError(
                "fov_selection_channels_type='vs' requires a 'vs' step in "
                f"fov_selection.preprocessing; got {self._steps}."
            )

    def _require_segmentation_step(self) -> None:
        """The trained tree model consumes segmented features -> require the step."""
        if "segmentation" not in self._steps:
            raise ValueError(
                "fov_selection.preprocessing must include a 'segmentation' step; the "
                f"trained tree model requires segmented features. Got {self._steps}."
            )

    @staticmethod
    def _projection_from_steps(steps: list[str]) -> str:
        """Derive the projection method from the preprocessing step list."""
        if "max_projection" in steps:
            return "max"
        if "sum_projection" in steps:
            return "sum"
        raise ValueError(
            "fov_selection.preprocessing must include a projection step "
            "('sum_projection' or 'max_projection')."
        )

    def _recon_config(self) -> dict:
        """Assemble the reconstruction sub-config for build_preprocessor.

        The deskew/phase/virtual_staining blocks live directly under
        ``fov_selection`` (DynaTrack style); only the reconstruction steps of the
        preprocessing list are relevant to the preprocessor (it ignores
        projection/segmentation).
        """
        return {
            "preprocessing": self._steps,
            "deskew": self.config.get("deskew"),
            "phase": self.config.get("phase"),
            "virtual_staining": self.config.get("virtual_staining"),
        }

    def _inject_scales(self, recon: dict) -> dict:
        """Inject XY pixel size / Z step into the deskew and phase sub-configs.

        Single source of truth (as DynaTrack does), so the pixel/step values are
        never duplicated in the config and cannot drift.
        """
        recon = copy.deepcopy(recon)
        deskew = recon.get("deskew")
        if deskew is not None:
            deskew["pixel_size_um"] = self._pixel_size_um
            deskew["scan_step_um"] = self._z_step_um
        phase = recon.get("phase")
        if phase is not None:
            tf = phase.setdefault("transfer_function", {})
            tf["yx_pixel_size"] = self._pixel_size_um
            tf["z_pixel_size"] = self._z_step_um
        return recon

    # -- lifecycle ---------------------------------------------------------

    def start(
        self,
        zyx_shape: tuple[int, int, int],
        log_file_path: Path | None = None,
    ) -> None:
        """Start the worker subprocess (unless a ``decide_fn`` was injected).

        The worker needs the acquired frame shape, so call this after hardware
        setup has applied the ROI.
        """
        if self._decide_fn is None:
            from shrimpy.fov_selection.worker import FovSelectionWorker

            recon = self._inject_scales(self._recon_config())
            logger.info("FOV selection: starting worker process for shape %s", zyx_shape)
            self._worker = FovSelectionWorker(
                recon=recon,
                target_channels=self._target_channels,
                segmentation=self._segmentation,
                model_path=self.config["model"]["path"],
                projection=self._projection,
                threshold=self._threshold,
                px_um=self._pixel_size_um,
                zyx_shape=zyx_shape,
                log_file_path=log_file_path,
                debug_dir=self._debug_dir,
                require_gpu=self._require_gpu,
            )
            self._worker.start()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending = None

    def on_frame_ready(self, img: np.ndarray, event: MDAEvent) -> None:
        """Buffer pre-scan frames per position and submit completed stacks.

        Connect to the core's ``frameReady`` signal. Only the pre-scan timepoint
        (t=0) and the ``fov_selection_channel`` are buffered; frames are matched by
        channel *name* (not index) since the pre-scan phase yields only the
        prescan channel. When all z-slices for a position have arrived, the stack
        is submitted for a decision.
        """
        channel = getattr(getattr(event, "channel", None), "config", None)
        if channel != self._fov_selection_channel:
            return
        if event.index.get("t", 0) != PRESCAN_TIMEPOINT:
            return

        p_idx = event.index.get("p", 0)
        tp = (PRESCAN_TIMEPOINT, p_idx)
        self._frames.setdefault(tp, []).append(img.copy())
        self._names[p_idx] = event.pos_name or f"p{p_idx}"

        if len(self._frames[tp]) >= self._expected_slices:
            frames = self._frames.pop(tp)
            # Stamp the moment the z-stack finished acquiring, so _record can
            # measure the acquired -> good/bad-decision latency for this FOV.
            self._stack_done_at[self._names[p_idx]] = time.monotonic()
            self._on_position_complete(p_idx, self._names[p_idx], frames)

    def _on_position_complete(self, p_idx: int, name: str, frames: list[np.ndarray]) -> None:
        """Submit one completed pre-scan stack for a decision (bounded).

        Waits for the previous decision to finish before submitting the next, so
        at most one FOV's frames are in flight -- this is the backpressure that
        keeps the pre-scan from buffering a whole plate in memory. Runs on the
        acquisition thread (``frameReady``), so waiting here pauses acquisition.
        """
        if self._executor is None:
            return
        self._await_pending()
        self._pending = self._executor.submit(self._decide_task, p_idx, name, frames)

    def _await_pending(self, timeout: float = 600) -> None:
        if self._pending is not None:
            try:
                self._pending.result(timeout=timeout)
            except Exception:
                logger.exception("FOV selection: pending decision failed")
            self._pending = None

    def _decide_task(self, p_idx: int, name: str, frames: list[np.ndarray]) -> None:
        """Run one decision (worker subprocess or in-process) and store the verdict."""
        if self._decide_fn is not None:
            bf_zyx = np.stack(frames, axis=0)
            del frames
            proba, good = self._decide_fn(bf_zyx)
            self._record(name, proba, good)
            return

        self._worker.submit(PRESCAN_TIMEPOINT, p_idx, name, frames)
        del frames  # free the main-process copy once pickled to the queue
        result = self._worker.get_result()
        if result is None:
            logger.warning(
                "FOV selection: no result from worker for %s; treating as bad", name
            )
            self._record(name, float("nan"), False)
            return
        self._record(name, result["proba"], result["good"])

    def _record(self, name: str, proba: float, good: bool) -> None:
        with self._verdicts_lock:
            self._verdicts[name] = (float(proba), bool(good))
        started = self._stack_done_at.pop(name, None)
        latency = time.monotonic() - started if started is not None else None
        if latency is not None:
            self._decision_latencies.append(latency)
        logger.info(
            "FOV selection: %s -> proba=%.3f %s%s",
            name,
            proba,
            "GOOD" if good else "bad",
            f" (acquired->decision {latency:.1f}s)" if latency is not None else "",
        )

    def _log_latency_summary(self) -> None:
        """Log the average acquired-stack -> good/bad-decision latency."""
        lat = self._decision_latencies
        if not lat:
            return
        logger.info(
            "FOV selection: acquired->decision latency avg %.1fs over %d FOVs "
            "(min %.1fs, max %.1fs)",
            sum(lat) / len(lat),
            len(lat),
            min(lat),
            max(lat),
        )

    def drain(self, timeout: float = 600) -> None:
        """Block until all submitted pre-scan decisions have completed.

        Awaited in ``teardown_sequence`` after the pre-scan run finishes, before
        ``good_position_names`` is read to build the timelapse run.
        """
        self._await_pending(timeout=timeout)

    def is_good(self, name: str) -> bool:
        """Whether ``name`` was decided good. Unknown/undecided names are bad."""
        with self._verdicts_lock:
            verdict = self._verdicts.get(name)
        return bool(verdict[1]) if verdict is not None else False

    def good_position_names(self) -> list[str]:
        """Names of positions decided good, in decision order."""
        with self._verdicts_lock:
            return [name for name, (_p, good) in self._verdicts.items() if good]

    @property
    def fov_selection_channel(self) -> str:
        """Acquisition channel used for the pre-scan (fed to reconstruction)."""
        return self._fov_selection_channel

    @property
    def num_decided(self) -> int:
        with self._verdicts_lock:
            return len(self._verdicts)

    def shutdown(self) -> None:
        """Finish any in-flight decision and shut down the worker + executor."""
        self._await_pending()
        self._log_latency_summary()
        if self._worker is not None:
            self._worker.shutdown()
            self._worker = None
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._frames = {}
        self._names = {}
