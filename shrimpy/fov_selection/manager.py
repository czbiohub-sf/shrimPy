"""Engine-facing coordinator for online, streaming FOV selection.

Mirrors :class:`shrimpy.dynatrack.manager.DynaTrack`: an acquisition engine
builds a :class:`FovSelection` from the ``fov_selection`` metadata section and
interacts with that object only. It turns the pre-scan run (a single-timepoint
run on ``fov_selection_channel`` over all candidate FOVs) into a per-FOV pass/skip verdict:

    BF z-stack -> reconstruct (deskew -> phase -> virtual stain)   [preprocessing.py]
               -> project -> segment -> features -> tree predict   [pipeline.py]

The decision is streamed: as each pre-scan FOV's z-stack completes in
``on_frame_ready`` it is submitted to a worker subprocess (torch/GPU isolation,
like DynaTrack). ``drain`` is awaited after the pre-scan run, and
``passed_position_names`` selects which FOVs the timelapse run images.

Config lives under ``metadata.fov_selection``. Scale parameters (XY pixel
size, Z step) are the single source of truth injected into the deskew/phase
sub-configs (as DynaTrack does), so they are not duplicated in the config.
"""

from __future__ import annotations

import copy
import logging
import re
import threading
import time

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from shrimpy.fov_selection import prescan_artifacts
from shrimpy.fov_selection.plate_naming import file_stem_name, plate_labels

if TYPE_CHECKING:
    from collections.abc import Callable

    from useq import MDAEvent, MDASequence

logger = logging.getLogger(__name__)

# Trailing per-well/per-center field index in an expanded FOV name ("site0_0003" -> "site0").
# Anchored and 4-digit so it only strips a real field suffix (see _build_fov_groups).
_FOV_FIELD_SUFFIX = re.compile(r"_\d{4}$")

# The object FOV selection segments and scores; also the InstanSeg head name.
TARGETS = ("cells", "nuclei")

# Timepoint used for the pre-scan (first timepoint of the run).
PRESCAN_TIMEPOINT = 0


def sibling_artifact_paths(data_path: Path | None, run_index: int | None = None) -> list[Path]:
    """Every path a FOV-selection run may create NEXT TO its output store.

    The pre-scan writes no store of its own, so these are the only on-disk traces a run
    leaves if it dies before the timelapse starts. Name deduplication has to test them
    too -- otherwise a crashed pre-scan leaves ``<name>_fov_debug/`` behind while
    ``<name>.ome.zarr`` is still free, the next run picks the same name, and its worker
    appends to the dead run's ``fov_summary.csv``.

    The selected-FOV config artifact is deliberately NOT here: it uses the fixed
    ``config_for_recovery.yaml`` name, not the acquisition's, so it does not vary with the
    candidate name -- including it would make every candidate look taken.
    """
    if data_path is None:
        return []
    return [
        p
        for p in (
            FovSelection._debug_dir_for(data_path, run_index),
            FovSelection._prescan_recon_path_for(data_path, run_index),
        )
        if p is not None
    ]


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
    run_index : int | None
        Deduplication index the engine appended to the acquisition name when the
        bare name was already taken (``acq`` -> ``acq_1``). Moved to the end of the
        sibling artifact names, so they read ``acq_fov_debug_1`` /
        ``acq_prescan_1.ome.zarr`` rather than burying the index mid-name. See
        :meth:`_sibling_path`.
    """

    def __init__(
        self,
        config: dict,
        sequence: MDASequence,
        pixel_size_um: float,
        z_step_um: float,
        data_path: Path | None = None,
        decide_fn: Callable[[np.ndarray], tuple[float, bool]] | None = None,
        run_index: int | None = None,
    ) -> None:
        self.config = config
        self._pixel_size_um = pixel_size_um
        self._z_step_um = z_step_um
        # Calibration mode: the engine runs the pre-scan ONLY (no timelapse) and opens the
        # feature viewer on its output. To feed the viewer, the worker extracts EVERY
        # producible feature (not just the model's) and writes the debug artifacts in the
        # viewer's standard layout, so save_decision is forced on regardless of config.
        self._calibration_mode = bool(config.get("calibration_mode", False))
        # Optional lightweight per-FOV debug artifacts (projection/mask PNGs +
        # fov_summary.csv), written by the worker to a sibling directory next to
        # the output store. Always on in calibration mode.
        self._save_decision = (
            bool(config.get("save_decision", False)) or self._calibration_mode
        )
        # Optional per-FOV best-focus-Z debug CSV (detected slice + depth), written by the
        # worker only when the projection is 'best_focus_z'. Independent of save_decision, but
        # it also needs the sibling debug directory, so it opens one when it is the only thing on.
        self._save_best_focus_z = bool(config.get("save_best_focus_z_for_debug", False))
        self._debug_dir = (
            self._debug_dir_for(data_path, run_index)
            if (self._save_decision or self._save_best_focus_z)
            else None
        )
        # Feature-viewer CSV/PNG-folder stem for calibration output (the viewer derives the
        # sibling PNG folders from the CSV stem); None outside calibration.
        self._matrix_stem = (
            self._matrix_stem_for(data_path) if self._calibration_mode else None
        )
        # When set, the per-step reconstruction OME-Zarr (deskew / phase / vs /
        # projection / mask channels) is written to <name>_prescan.ome.zarr next to
        # the output store -- replacing the raw pre-scan store, so the pre-scan run
        # itself writes nothing to disk. Independent of save_decision.
        self._save_pre_scan_omezarr = bool(config.get("save_pre_scan_omezarr", False))
        self._recon_zarr_path = (
            self._prescan_recon_path_for(data_path, run_index)
            if self._save_pre_scan_omezarr
            else None
        )
        # Fail fast if reconstruction can't run on a GPU. Default True; set
        # fov_selection.require_gpu: false to allow a (slow) CPU run for debugging.
        self._require_gpu = bool(config.get("require_gpu", True))
        # Acquired channel imaged during the pre-scan and fed to reconstruction.
        # No default: it must be declared in the acquisition config so the pipeline
        # always images whatever the YAML specifies.
        self._fov_selection_channel = config.get("fov_selection_channel")
        if not self._fov_selection_channel:
            raise ValueError(
                "FOV selection requires fov_selection_channel in the acquisition config "
                "(metadata.fov_selection.fov_selection_channel); there is no default."
            )
        # Ordered preprocessing steps (DynaTrack style), e.g.
        # ['deskew', 'phase', 'vs', 'sum_projection', 'segmentation']. The
        # reconstruction steps are consumed by build_preprocessor; projection and
        # segmentation are consumed here / in the pipeline.
        self._steps = list(config.get("preprocessing") or [])
        self._projection = self._projection_from_steps(self._steps)
        # Optics for the 'best_focus_z' projection (waveorder focus): required only when that
        # projection is selected; fail before acquiring if it is and they are missing.
        self._best_focus_z = config.get("best_focus_z") or None
        if self._projection == "best_focus_z":
            self._validate_best_focus_z()
        self._segmentation = config.get("segmentation", {}) or {}
        model_cfg = config.get("model", {}) or {}
        # threshold is a classification-only knob: TrainedTreeModel uses it (proba >= threshold);
        # the thresholding box and the ranking model both ignore it.
        self._threshold = float(model_cfg.get("threshold", 0.5))
        # The model type drives the SELECTION rule (see passed_position_names): the ranking
        # model selects by top_fov per position, every classification model by its per-FOV
        # `good` verdict. Keyed on the type, not on whether top_fov happens to be set, so a
        # classification model never needs top_fov.
        self._model_type = model_cfg.get("type")
        # top_fov (ranking_by_defined_range): keep the N highest-proba FOVs PER POSITION (per
        # well / per grid center -- see _build_fov_groups). Required for ranking (validated in
        # from_metadata / the model), unused by the classification models.
        top_fov = model_cfg.get("top_fov")
        self._top_fov = int(top_fov) if top_fov is not None else None
        # FOV name -> the position it belongs to, so top_fov is applied within each position.
        self._fov_group = self._build_fov_groups(sequence)
        # FOV filename -> (well_row, well_col) labels, stamped onto fov_summary.csv so the
        # feature viewer can group the pre-scan FOVs by well (see _build_well_coords).
        self._well_coords = self._build_well_coords(sequence)
        self._is_vs = "vs" in self._steps
        # `target` (cells | nuclei) is the ONE object FOV selection segments and scores. It
        # drives: (a) how the reconstruction outputs are reduced to a single segmentation
        # input (pipeline._resolve_seg_input), (b) the InstanSeg head (below), and (c) which
        # channels VS must predict. Selection always produces ONE mask -> single-channel
        # (plain) feature names, so `target` is recorded as run metadata, not in column names.
        self._target = str(config.get("target", "")).lower()
        if self._target not in TARGETS:
            raise ValueError(
                "fov_selection.target must be one of "
                f"{TARGETS} (the object to segment and score); got {config.get('target')!r}. "
                "Aborting before acquisition."
            )
        # Reconstruction output channels to project. VS 'cells' combines nuclei+membrane into
        # one grayscale, so both are predicted; VS 'nuclei' segments the nuclei channel only,
        # so membrane is not predicted. Non-VS is the single reconstructed channel.
        if self._is_vs:
            self._recon_channels = (
                ["nuclei", "membrane"] if self._target == "cells" else ["nuclei"]
            )
        else:
            self._recon_channels = [self._target]
        # Drive the InstanSeg head from the single `target` field (cells|nuclei match the
        # InstanSeg heads), so the segmentation block does not carry a second source of truth.
        self._segmentation = {**self._segmentation, "target": self._target}

        self._validate_fov_selection_channel(sequence)
        self._require_segmentation_step()
        self._expected_slices = max(sequence.sizes.get("z", 1), 1)

        # Per-(timepoint, position) frame buffering for the pre-scan stacks.
        self._frames: dict[tuple[int, int], list[np.ndarray]] = {}
        self._names: dict[int, str] = {}

        # Verdicts, keyed by position name. Written from the executor thread,
        # read from the acquisition thread -> guarded by a lock. `good` is None for the
        # ranking model (no per-FOV verdict; selection is top_fov).
        self._verdicts: dict[str, tuple[float, bool | None]] = {}
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

        # Fail before acquiring if the model asks for a feature the configured preprocessing
        # cannot produce (a typo would otherwise be read as a silently-missing column). Skipped
        # when a decide_fn stands in for the pipeline -- it never extracts features.
        if self._decide_fn is None:
            self._validate_feature_names(model_cfg)
            self._validate_segmentation()

    # -- construction ------------------------------------------------------

    @staticmethod
    def _sibling_path(
        data_path: Path | None, suffix: str, run_index: int | None = None
    ) -> Path | None:
        """Sibling artifact path next to the output store: ``<base><suffix>[_<run_index>]``.

        ``run_index`` is the index the engine appends to every acquisition name
        (``acq`` -> ``acq_1``). It is stripped off the store name and re-appended at the
        END of the sibling's own name, so the first run of ``acq`` yields
        ``acq_fov_debug_1`` -- not ``acq_1_fov_debug``, which buries the index mid-name
        and sorts the runs' folders apart.

        Passing the index explicitly (rather than pattern-matching a trailing ``_<n>``
        off the path) keeps a user-supplied name that genuinely ends in a number intact:
        the first run of ``plate_2`` is stored as ``plate_2_1`` and yields
        ``plate_2_fov_debug_1``, which cannot collide with the second run of ``plate``
        (``plate_fov_debug_2``).
        """
        if data_path is None:
            return None
        data_path = Path(data_path)
        name = data_path.name
        for ext in (".ome.zarr", ".zarr"):
            if name.endswith(ext):
                name = name[: -len(ext)]
                break
        tail = ""
        if run_index is not None and name.endswith(f"_{run_index}"):
            name = name[: -len(f"_{run_index}")]
            tail = f"_{run_index}"
        return data_path.with_name(f"{name}{suffix}{tail}")

    @classmethod
    def _debug_dir_for(
        cls, data_path: Path | None, run_index: int | None = None
    ) -> Path | None:
        """Sibling ``<name>_fov_debug[_<n>]/`` directory next to the output store."""
        return cls._sibling_path(data_path, "_fov_debug", run_index)

    @staticmethod
    def _matrix_stem_for(data_path: Path | None) -> str | None:
        """Stem for the optional best-focus-Z debug CSV, ``<acq>_fov_feature_matrix``.

        Derived from the output store name (``<acq>.ome.zarr`` -> ``<acq>``). The main
        calibration table and its image folders now use fixed names (``fov_summary.csv`` /
        ``prescan_fov`` / ``prescan_mask``); this stem only labels the best-focus-Z CSV.
        """
        if data_path is None:
            return None
        name = Path(data_path).name
        for ext in (".ome.zarr", ".zarr"):
            if name.endswith(ext):
                name = name[: -len(ext)]
                break
        return f"{name}_fov_feature_matrix"

    @classmethod
    def _prescan_recon_path_for(
        cls, data_path: Path | None, run_index: int | None = None
    ) -> Path | None:
        """Sibling ``<name>_prescan[_<n>].ome.zarr`` store next to the output store."""
        path = cls._sibling_path(data_path, "_prescan", run_index)
        return None if path is None else path.with_name(f"{path.name}.ome.zarr")

    @classmethod
    def from_metadata(
        cls,
        meta: dict | None,
        sequence: MDASequence,
        pixel_size_um: float,
        data_path: Path | None = None,
        decide_fn: Callable[[np.ndarray], tuple[float, bool]] | None = None,
        run_index: int | None = None,
    ) -> FovSelection | None:
        """Build the coordinator from the ``fov_selection`` metadata block.

        Returns ``None`` when FOV selection is disabled. Raises (fail before
        acquiring) when it is enabled but no usable model is configured, the pixel
        size is missing, or a deskew/phase reconstruction needs the Z step but the
        sequence z_plan has none.
        """
        if not meta or not meta.get("enabled", False):
            return None
        from shrimpy.fov_selection.fov_model import MODEL_TYPES

        model_cfg = meta.get("model", {}) or {}
        model_type = model_cfg.get("type")
        if model_type not in MODEL_TYPES:
            raise ValueError(
                "FOV selection is enabled but metadata.fov_selection.model.type "
                f"must be one of {sorted(MODEL_TYPES)}; got {model_type!r}. Aborting "
                "before acquisition."
            )
        if model_type == "classification_tree" and not model_cfg.get("path"):
            raise ValueError(
                "fov_selection.model.type='classification_tree' requires a 'path' to a "
                "trained FOV-selection .joblib. Aborting before acquisition."
            )
        if model_type == "ranking_by_defined_range":
            top_fov = model_cfg.get("top_fov")
            if top_fov is None or int(top_fov) < 1:
                raise ValueError(
                    "fov_selection.model.type='ranking_by_defined_range' selects by pure "
                    "ranking and requires 'top_fov' (a positive int): the N highest-ranked "
                    "FOVs OF EACH POSITION (well / grid center) pass. Aborting before "
                    "acquisition."
                )
        if not pixel_size_um:
            raise ValueError(
                "FOV selection: pixel size is not set (core.getPixelSizeUm() returned "
                "0 or None); calibrate the pixel size in Micro-Manager."
            )
        z_step_um = getattr(sequence.z_plan, "step", None) if sequence.z_plan else None
        # Deskew and phase reconstruction need the Z step; other pipelines (raw -> segment,
        # flatfield-only) do not, so only require it when a deskew/phase block is configured.
        # _inject_scales feeds it into DeskewSettings.scan_step_um / PhaseSettings.z_pixel_size;
        # a missing step would otherwise crash the worker mid-run (or, with a hand-set
        # px_to_scan_ratio, silently use a wrong axial scale) and select nothing.
        if (meta.get("deskew") or meta.get("phase")) and not z_step_um:
            raise ValueError(
                "FOV selection: reconstruction includes deskew/phase, which need the Z step, "
                "but the sequence z_plan has no step. Add a stepped z_plan before acquiring."
            )
        return cls(
            config=meta,
            sequence=sequence,
            pixel_size_um=pixel_size_um,
            z_step_um=z_step_um,
            data_path=data_path,
            decide_fn=decide_fn,
            run_index=run_index,
        )

    @staticmethod
    def _build_fov_groups(sequence: MDASequence) -> dict[str, str]:
        """Map each candidate FOV name -> the *position* it belongs to.

        ``top_fov`` is a per-position quota, so the FOVs of one well / one grid center have to
        be identifiable as a group. Both candidate styles name a FOV ``"{position}_{field}"``
        (see :func:`shrimpy.fov_selection.sequences.expand_candidate_fovs`), but they carry the
        position differently:

        * on a plate (``WellPlatePlan``, or centers with ``plate_row``/``plate_col``) the well
          IS the position, and it is available structurally -- ``"B2_0007" -> "B2"``;
        * off a plate the only record of the grid center is the name prefix, so a trailing
          ``_<4-digit field>`` is stripped -- ``"site0_0003" -> "site0"``. The 4-digit shape is
          matched exactly so a position whose own name contains an underscore and no field
          suffix (explicit ``stage_positions`` with no ``grid_plan``) stays whole and simply
          forms a group of one.
        """
        groups: dict[str, str] = {}
        for idx, pos in enumerate(sequence.stage_positions):
            name = pos.name or f"p{idx}"
            well = plate_labels(pos)
            if well is not None:
                groups[name] = f"{well[0]}{well[1]}"
            else:
                groups[name] = _FOV_FIELD_SUFFIX.sub("", name) or name
        return groups

    @staticmethod
    def _build_well_coords(sequence: MDASequence) -> dict[str, tuple[str, int]]:
        """Map each candidate FOV's *filename* -> its ``(well_row, well_col)`` plate labels.

        The feature viewer groups FOVs by the ``well_row`` / ``well_col`` columns
        (:meth:`FeatureViewer._group_positions_by_well`), so writing them onto
        ``fov_summary.csv`` (:func:`shrimpy.fov_selection.prescan_artifacts.stamp_well_columns`)
        lets the viewer group a pre-scan by well. The labels are the human plate form the rest
        of the codebase uses -- ``well_row`` a letter (``"B"``), ``well_col`` a one-based int
        (``4``) -- so the viewer's "Well B/4" headers match the ``position`` column and the
        OME-Zarr paths.

        Keyed by ``filename`` (``file_stem_name(name)``, the CSV's join column and the PNG
        stem) so it lines up with both the normal and calibration CSV regardless of how a name
        sanitizes. Only positions on a plate (carrying ``plate_row``/``plate_col``) are
        included; off-plate grid candidates have no well and are omitted, so the viewer falls
        back to a single "All FOVs" group for them.
        """
        coords: dict[str, tuple[str, int]] = {}
        for idx, pos in enumerate(sequence.stage_positions):
            well = plate_labels(pos)
            if well is None:
                continue
            name = pos.name or f"p{idx}"
            coords[file_stem_name(name)] = (well[0], int(well[1]))
        return coords

    def _validate_fov_selection_channel(self, sequence: MDASequence) -> None:
        names = [ch.config for ch in sequence.channels]
        if self._fov_selection_channel not in names:
            raise ValueError(
                f"FOV selection fov_selection_channel {self._fov_selection_channel!r} is not one of "
                f"the acquisition channels {names}."
            )

    def _require_segmentation_step(self) -> None:
        """Features are computed from segmentation masks, so the step is required."""
        if "segmentation" not in self._steps:
            raise ValueError(
                "fov_selection.preprocessing must include a 'segmentation' step "
                f"(features come from segmentation masks). Got {self._steps}."
            )

    def _producible_feature_names(self) -> set[str]:
        """Every feature-column name the configured preprocessing/segmentation can emit.

        FOV selection segments exactly ONE mask (the ``target``), so the columns are always
        the plain feature keys (``coverage_frac``, ...) -- no channel prefix.
        """
        from shrimpy.fov_selection.feature_extraction import (
            FEATURE_NAMES,
            MASK_FEATURE_KEYS,
        )

        return set(FEATURE_NAMES) | set(MASK_FEATURE_KEYS)

    def _validate_feature_names(self, model_cfg: dict) -> None:
        """Fail before acquiring if the model asks for a feature the pipeline cannot produce.

        Checks the config-defined models (``ranking_by_defined_range`` /
        ``classification_by_thresholding``), whose feature names are hand-typed: a typo would
        otherwise surface only as a silently-missing column (NaN) at decision time rather than
        an error. A trained ``classification_tree``'s names come from training (not the config)
        and are validated when the worker builds the model, so they are not checked here.
        """
        if model_cfg.get("type") not in (
            "ranking_by_defined_range",
            "classification_by_thresholding",
        ):
            return
        requested = list(model_cfg.get("features") or {})
        producible = self._producible_feature_names()
        unknown = [name for name in requested if name not in producible]
        if unknown:
            raise ValueError(
                "FOV selection: model requests feature name(s) the configured preprocessing "
                f"cannot produce: {unknown}. Available feature names: {sorted(producible)}. "
                "Fix the names under fov_selection.model.features (feature keys are plain, "
                "e.g. 'coverage_frac' -- no channel prefix). Aborting before acquisition."
            )

    def _validate_segmentation(self) -> None:
        """Fail before acquiring on an unusable ``segmentation`` block.

        The backend is only loaded inside the worker subprocess, which does not start until
        the pre-scan is already running -- so a typo'd backend name or a missing InstanSeg
        checkpoint would otherwise surface as a mid-acquisition worker crash. Checked here
        instead, alongside the model feature names.
        """
        from shrimpy.fov_selection.segmentation import INSTANSEG_TARGETS

        backend = self._segmentation.get("model", "cellpose")
        if backend not in ("cellpose", "instanseg", "otsu"):
            raise ValueError(
                f"fov_selection.segmentation.model must be 'cellpose', 'instanseg' or "
                f"'otsu'; got {backend!r}. Aborting before acquisition."
            )
        if backend != "instanseg":
            return

        path = self._segmentation.get("path")
        if not path:
            raise ValueError(
                "fov_selection.segmentation.model='instanseg' requires a 'path' to the "
                "InstanSeg checkpoint (a bioimage.io .zip export or a TorchScript .pt). "
                "Aborting before acquisition."
            )
        if not Path(path).exists():
            raise FileNotFoundError(
                f"fov_selection.segmentation.path: InstanSeg checkpoint not found: {path}. "
                "Aborting before acquisition."
            )
        target = self._segmentation.get("target", INSTANSEG_TARGETS[0])
        if target not in INSTANSEG_TARGETS:
            raise ValueError(
                f"fov_selection.segmentation.target must be one of "
                f"{list(INSTANSEG_TARGETS)}; got {target!r}. Aborting before acquisition."
            )

    def _validate_best_focus_z(self) -> None:
        """Fail before acquiring if the 'best_focus_z' projection lacks its optics.

        :func:`shrimpy.fov_selection.pipeline.project_zyx` would otherwise fall back to the
        middle slice at run time; catching it here makes the misconfiguration explicit.
        """
        best_focus_z = self._best_focus_z or {}
        missing = [
            k
            for k in ("numerical_aperture_detection", "wavelength_illumination")
            if not best_focus_z.get(k)
        ]
        if missing:
            raise ValueError(
                "fov_selection.preprocessing selects 'best_focus_z', which needs "
                f"fov_selection.best_focus_z with {missing} (detection NA + illumination wavelength in "
                "um). Add a 'best_focus_z' block or choose another projection step."
            )

    @staticmethod
    def _projection_from_steps(steps: list[str]) -> str:
        """Derive the projection method from the preprocessing step list."""
        if "max_projection" in steps:
            return "max"
        if "sum_projection" in steps:
            return "sum"
        if "middle_slice_projection" in steps:
            return "middle"
        if "logstd_projection" in steps:
            return "logstd"
        if "best_focus_z" in steps:
            return "best_focus_z"
        # No explicit projection step: default to 'sum' (the trained-model default and
        # project_zyx's own default; channel-agnostic, and a no-op for a single-slice
        # stack). A projection step is not mandatory in the config.
        return "sum"

    def _recon_config(self) -> dict:
        """Assemble the reconstruction sub-config for build_preprocessor.

        The deskew/phase/virtual_staining blocks live directly under
        ``fov_selection`` (DynaTrack style); only the reconstruction steps of the
        preprocessing list are relevant to the preprocessor (it ignores
        projection/segmentation).
        """
        recon = {
            "preprocessing": self._steps,
            "deskew": self.config.get("deskew"),
            "phase": self.config.get("phase"),
            "virtual_staining": self.config.get("virtual_staining"),
        }
        if self._is_vs:
            # VS predicts exactly the channels the target needs ('cells' -> nuclei+membrane,
            # 'nuclei' -> nuclei only), so a nuclei-only run does not pay for membrane VS.
            vs = dict(recon.get("virtual_staining") or {})
            vs["target_channels"] = self._recon_channels
            recon["virtual_staining"] = vs
        else:
            # Non-VS: name the single preprocessor output after our one channel.
            recon["output_channel"] = self._recon_channels[0]
        return recon

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
            from shrimpy.fov_selection.worker import FovSelectionWorker, WorkerConfig

            recon = self._inject_scales(self._recon_config())
            logger.info("FOV selection: starting worker process for shape %s", zyx_shape)
            self._worker = FovSelectionWorker(
                WorkerConfig(
                    recon=recon,
                    target=self._target,
                    recon_channels=self._recon_channels,
                    segmentation=self._segmentation,
                    model_cfg=self.config.get("model", {}) or {},
                    projection=self._projection,
                    threshold=self._threshold,
                    pixel_size_um=self._pixel_size_um,
                    zyx_shape=zyx_shape,
                    log_file_path=log_file_path,
                    debug_dir=self._debug_dir,
                    recon_zarr_path=self._recon_zarr_path,
                    require_gpu=self._require_gpu,
                    calibration_mode=self._calibration_mode,
                    matrix_stem=self._matrix_stem,
                    best_focus_z=self._best_focus_z,
                    z_step_um=self._z_step_um,
                    save_best_focus_z=self._save_best_focus_z,
                    write_prescan_artifacts=self._save_decision,
                )
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

    def _record(self, name: str, proba: float, good: bool | None) -> None:
        with self._verdicts_lock:
            self._verdicts[name] = (float(proba), None if good is None else bool(good))
        started = self._stack_done_at.pop(name, None)
        latency = time.monotonic() - started if started is not None else None
        if latency is not None:
            self._decision_latencies.append(latency)
        # Per-FOV score only: the Passed/Skipped verdict is a per-position top-K ranking result,
        # so it needs every FOV of a position scored first (see log_selection_summary).
        logger.info(
            "FOV selection: %s -> score=%.3f%s",
            name,
            proba,
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
        ``passed_position_names`` is read to build the timelapse run.
        """
        self._await_pending(timeout=timeout)

    def passed_position_names(self) -> list[str]:
        """Names of the FOVs that passed FOV selection (imaged in the timelapse run).

        The selection rule is chosen by MODEL TYPE, not by whether ``top_fov`` is set:

        ``ranking_by_defined_range``: the ``top_fov`` highest-scoring FOVs **of each position**
        -- the quota is per well / per grid center, not across the whole pre-scan, so every
        position contributes its own best FOVs and a dense well cannot crowd out a sparser one.
        With ``top_fov: 3`` and 4 positions you get up to 12 FOVs. Ordering is still globally
        best-first (ties broken by decision order); ranking is pure, so a passing FOV is only
        the best available in its position, not necessarily "good".

        Classification models (``classification_by_thresholding`` / ``classification_tree``):
        every FOV the model decided good (its per-FOV verdict), in decision order -- a per-FOV
        pass/fail, so ``top_fov`` does not apply and is not needed.
        """
        with self._verdicts_lock:
            items = list(self._verdicts.items())
        if self._model_type != "ranking_by_defined_range":
            return [name for name, (_p, good) in items if good]
        ranked = sorted(items, key=lambda kv: kv[1][0], reverse=True)
        kept: list[str] = []
        per_position: dict[str, int] = {}
        for name, _verdict in ranked:
            position = self._fov_group.get(name, name)
            if per_position.get(position, 0) >= self._top_fov:
                continue
            per_position[position] = per_position.get(position, 0) + 1
            kept.append(name)
        return kept

    def log_selection_summary(self) -> None:
        """Log the final selection after the drain (call in EVERY mode).

        One INFO line with the count and the PASSED FOV names, then each SKIPPED FOV (with its
        score) at DEBUG. The per-FOV scores are already logged at decision time
        (:meth:`_record`), so this does not repeat every score at INFO -- it records which FOVs
        the selection kept. In calibration mode there is no timelapse, but this still reports
        which FOVs the current model WOULD select, so the summary is meaningful in every mode.

        The Passed/Skipped split for ``ranking_by_defined_range`` is the per-position top-K
        outcome, known only once every FOV has been scored -- hence a post-drain summary.
        """
        passed = self.passed_position_names()
        passed_set = set(passed)
        with self._verdicts_lock:
            items = sorted(self._verdicts.items(), key=lambda kv: kv[1][0], reverse=True)
        logger.info(
            "FOV selection: %d/%d FOVs passed selection: %s",
            len(passed_set),
            len(items),
            passed,
        )
        for name, (proba, _good) in items:
            if name not in passed_set:
                logger.debug("FOV selection: %s -> score=%.3f Skipped", name, proba)

    def finalize_debug_summary(self) -> None:
        """Stamp the whole-run columns onto ``fov_summary.csv`` (call after the drain).

        The worker appends one row per FOV as it is decided (``name, filename, proba,
        <features>``); a few columns are properties of the WHOLE pre-scan and can only be
        written once every FOV is scored, so they are added here, over the finished table:

        ``well_row`` / ``well_col`` : the FOV's plate well (:meth:`_build_well_coords`), written
                       in BOTH normal and calibration mode so the feature viewer can group the
                       pre-scan FOVs by well.
        ``selected`` : 1 for the FOVs the timelapse images (:meth:`passed_position_names`),
                       0 otherwise -- for every model type (normal mode only).
        ``position`` : the well / grid center the FOV belongs to (:meth:`_build_fov_groups`) --
                       the group ``rank`` and the ``top_fov`` quota are computed within.
        ``rank``     : 1 = highest score WITHIN its position, ties broken by score order (so
                       ``selected`` is exactly ``rank <= top_fov``). Only meaningful for
                       ranking models; for the threshold/tree model types selection is a
                       per-FOV pass/fail with no ordering, so ``rank`` is left NaN.

        It then gathers just the selected FOVs' projection PNGs into ``selected_fov/``
        (:func:`shrimpy.fov_selection.prescan_artifacts.save_selected_fov_pngs`) so the fields
        the timelapse will image can be browsed on their own.

        A no-op when ``save_decision`` is off or the CSV was never written. The selection
        columns are skipped in calibration mode -- there is no timelapse, so nothing is
        "selected"; its scores are filled in later from the viewer's Rank tab. The CSV
        mechanics live in :mod:`shrimpy.fov_selection.prescan_artifacts`; this method supplies
        the whole-run inputs (well labels, passed set, groups, quota). Every filesystem step
        there is guarded: this is written at the very end of the pre-scan, and it must not be
        able to raise out of ``teardown_sequence`` and take the acquisition down with it.
        """
        if self._debug_dir is None:
            return
        summary_path = Path(self._debug_dir) / prescan_artifacts.SUMMARY_CSV_NAME
        if not summary_path.exists():
            return

        # well_row/well_col first, in BOTH modes, so the viewer's per-well grouping works.
        prescan_artifacts.stamp_well_columns(summary_path, self._well_coords)

        # Calibration applies no selection (no timelapse), so there is nothing more to add.
        if self._calibration_mode:
            return

        passed = set(self.passed_position_names())
        prescan_artifacts.finalize_summary_csv(
            summary_path,
            passed=passed,
            fov_group=self._fov_group,
            top_fov=self._top_fov,
        )
        # Gather the selected FOVs' projection PNGs into their own folder so the fields the
        # timelapse will image can be browsed without hunting through every candidate.
        prescan_artifacts.save_selected_fov_pngs(self._debug_dir, passed, self._fov_group)

    @property
    def calibration_mode(self) -> bool:
        """Whether this is a calibration pre-scan (pre-scan only + feature viewer, all
        features extracted; see :meth:`__init__`)."""
        return self._calibration_mode

    @property
    def calibration_matrix_csv(self) -> Path | None:
        """Feature-viewer CSV the calibration pre-scan writes (``None`` outside calibration
        or when no debug directory is set). The engine opens the viewer on this file. Shares
        the fixed ``fov_summary.csv`` name with the normal-mode decision table."""
        if not self._calibration_mode or self._debug_dir is None:
            return None
        return Path(self._debug_dir) / prescan_artifacts.SUMMARY_CSV_NAME

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
