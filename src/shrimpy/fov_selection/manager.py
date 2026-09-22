"""Engine-facing coordinator for online, streaming FOV selection.

Mirrors :class:`shrimpy.dynatrack.manager.DynaTrack`: an acquisition engine
builds a :class:`FOVSelection` from the ``fov_selection`` metadata section and
interacts with that object only. It turns the pre-scan run (a single-timepoint
run on ``fov_selection_channel`` over all candidate FOVs) into a per-FOV pass/skip verdict:

    BF z-stack -> reconstruct (deskew -> phase -> virtual stain)   [preprocessing.py]
               -> project -> segment -> features -> tree predict   [pipeline.py]

The decision is streamed: as each pre-scan FOV's z-stack completes in
``on_frame_ready`` it is submitted to a worker subprocess (torch/GPU isolation,
like DynaTrack). ``drain`` is awaited after the pre-scan run, then :meth:`outcome`
hands the engine a :class:`PrescanOutcome` naming the FOVs the timelapse run images --
the coordinator itself does not outlive ``teardown_sequence``.

Config lives under ``metadata.fov_selection``, validated by
:class:`shrimpy.fov_selection.config.FOVSelectionConfig` when the acquisition config is
loaded -- the coordinator takes that object, not a raw mapping, and only adds the checks
that need the acquisition too (see :meth:`FOVSelection.from_metadata`). Scale parameters
(XY pixel size, Z step) are the single source of truth injected into the deskew/phase
sub-configs (as DynaTrack does), so they are not duplicated in the config.
"""

from __future__ import annotations

import copy
import logging
import re
import threading
import time

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from shrimpy.fov_selection import prescan_artifacts
from shrimpy.fov_selection.plate_naming import file_stem_name, plate_labels

if TYPE_CHECKING:
    from collections.abc import Callable

    from useq import MDAEvent, MDASequence

    from shrimpy.fov_selection.config import FOVSelectionConfig

logger = logging.getLogger(__name__)

# Trailing per-well/per-center field index in an expanded FOV name ("site0_0003" -> "site0").
# Anchored and 4-digit so it only strips a real field suffix (see _build_fov_groups).
_FOV_FIELD_SUFFIX = re.compile(r"_\d{4}$")

# Timepoint used for the pre-scan (first timepoint of the run).
PRESCAN_TIMEPOINT = 0


@dataclass(frozen=True)
class PrescanOutcome:
    """What a pre-scan run hands back to the engine, captured by :meth:`FOVSelection.outcome`.

    An MDA run has no return value: the runner passes only the sequence to
    ``setup_sequence`` / ``teardown_sequence``, so the coordinator is built, used, and
    dropped entirely inside one ``mmc.mda.run()`` call. Anything the *next* run needs has
    to be lifted out before it goes -- and it is exactly these two things, so they are
    named here rather than kept as loose attributes on the engine.

    Attributes
    ----------
    selected_fovs : list[str]
        Names of the candidate FOVs the timelapse run should image, in pre-scan order
        (see :meth:`FOVSelection.passed_position_names`). Empty both when nothing passed
        and in calibration mode, which runs no timelapse at all -- the engine treats the
        two the same way, and calibration is told apart by ``calibration_csv``.
    calibration_csv : Path | None
        The feature viewer's ``fov_summary.csv``, for the engine to open the viewer on.
        ``None`` outside calibration mode, and when no debug directory was written.
    """

    selected_fovs: list[str] = field(default_factory=list)
    calibration_csv: Path | None = None


def sibling_artifact_paths(data_path: Path | None) -> list[Path]:
    """Every path a FOV-selection run may create NEXT TO its output store.

    The pre-scan writes no store of its own, so these are the only on-disk traces a run
    leaves if it dies before the timelapse starts. Name deduplication has to test them
    too -- otherwise a crashed pre-scan leaves ``<name>_fov_debug/`` behind while
    ``<name>.ome.zarr`` is still free, the next run picks the same name, and its worker
    appends to the dead run's ``fov_summary.csv``.

    The selected-FOV config backup is one of these: it is written between the two runs,
    so a run that dies right after it (before the timelapse creates the store) leaves it
    as the only trace of the name.
    """
    if data_path is None:
        return []
    return [
        p
        for p in (
            FOVSelection._debug_dir_for(data_path),
            FOVSelection._prescan_recon_path_for(data_path),
            FOVSelection._config_backup_path_for(data_path),
        )
        if p is not None
    ]


class FOVSelection:
    """Coordinates the online, streaming FOV-selection decision for one run.

    Parameters
    ----------
    config : FOVSelectionConfig
        The validated ``fov_selection`` metadata block. Everything checkable from
        the config alone (unknown keys, the pipeline, curve parameters, ``top_fov``)
        has already been rejected by
        :class:`~shrimpy.fov_selection.config.FOVSelectionConfig`; what is left here
        are the checks that need the acquisition too -- the channel against the
        sequence, the feature names against the configured preprocessing, and the
        segmentation checkpoint against the filesystem.
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
        config: FOVSelectionConfig,
        sequence: MDASequence,
        pixel_size_um: float,
        z_step_um: float,
        data_path: Path | None = None,
        decide_fn: Callable[[np.ndarray], tuple[float, bool]] | None = None,
    ) -> None:
        self.config = config
        self._pixel_size_um = pixel_size_um
        self._z_step_um = z_step_um
        # Calibration mode: the engine runs the pre-scan ONLY (no timelapse) and opens the
        # feature viewer on its output. To feed the viewer, the worker extracts EVERY
        # producible feature (not just the model's) and writes the debug artifacts in the
        # viewer's standard layout, so save_decision is forced on regardless of config.
        self._calibration_mode = config.calibration_mode
        # Optional AnnData/zarr export of the feature matrix for Embedding Atlas (nd_export).
        # It reads fov_summary.csv after the drain, so it forces save_decision on (the CSV must
        # exist); the export itself runs post-drain in export_prescan_nd().
        self._save_pre_scan_nd = config.save_pre_scan_nd
        # Optional lightweight per-FOV debug artifacts (projection/mask PNGs +
        # fov_summary.csv), written by the worker to a sibling directory next to
        # the output store. Always on in calibration mode and when save_pre_scan_nd is set.
        self._save_decision = (
            config.save_decision or self._calibration_mode or self._save_pre_scan_nd
        )
        # Optional per-FOV best-focus-Z debug CSV (detected slice + depth), written by the
        # worker only when the projection is 'best_focus_z'. Independent of save_decision, but
        # it also needs the sibling debug directory, so it opens one when it is the only thing on.
        self._save_best_focus_z = config.save_best_focus_z_for_debug
        self._debug_dir = (
            self._debug_dir_for(data_path)
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
        self._save_pre_scan_omezarr = config.save_pre_scan_omezarr
        self._recon_zarr_path = (
            self._prescan_recon_path_for(data_path) if self._save_pre_scan_omezarr else None
        )
        # Fail fast if reconstruction can't run on a GPU. Default True; set
        # fov_selection.require_gpu: false to allow a (slow) CPU run for debugging.
        self._require_gpu = config.require_gpu
        # Acquired channel imaged during the pre-scan and fed to reconstruction.
        self._fov_selection_channel = config.fov_selection_channel
        # Ordered preprocessing steps (DynaTrack style), e.g.
        # ['deskew', 'phase', 'vs', 'sum_projection', 'segmentation']. The
        # reconstruction steps are consumed by build_preprocessor; projection and
        # segmentation are consumed here / in the pipeline.
        self._steps = list(config.preprocessing)
        self._projection = config.projection
        # Optics for the 'best_focus_z' projection (waveorder focus). The pipeline reads them
        # as a plain mapping, so the sub-blocks are dumped here rather than passed as models:
        # they cross into the worker subprocess, and every consumer of them (build_segmenter,
        # build_fov_model, project_zyx) is config-object agnostic by design.
        self._best_focus_z = (
            config.best_focus_z.model_dump(mode="json") if config.best_focus_z else None
        )
        self._segmentation = config.segmentation.model_dump(mode="json", exclude_none=True)
        self._model_cfg = config.model.model_dump(mode="json", exclude_none=True)
        # threshold is a classification-only knob: TrainedTreeModel uses it (proba >= threshold);
        # the thresholding box and the ranking model both ignore it (and do not declare it).
        self._threshold = getattr(config.model, "threshold", 0.5)
        # The model type drives the SELECTION rule (see passed_position_names): the ranking
        # model selects by top_fov per position, every classification model by its per-FOV
        # `good` verdict. Keyed on the type, not on whether top_fov happens to be set, so a
        # classification model never needs top_fov.
        self._model_type = config.model.type
        # top_fov (ranking_by_defined_range): keep the N highest-proba FOVs PER POSITION (per
        # well / per grid center -- see _build_fov_groups). Required by the ranking model's
        # schema, absent from the classification models.
        self._top_fov = getattr(config.model, "top_fov", None)
        # FOV name -> the position it belongs to, so top_fov is applied within each position.
        self._fov_group = self._build_fov_groups(sequence)
        # FOV filename -> (well_row, well_col) labels, stamped onto fov_summary.csv so the
        # feature viewer can group the pre-scan FOVs by well (see _build_well_coords).
        self._well_coords = self._build_well_coords(sequence)
        self._is_vs = config.uses_virtual_staining
        # `target` (cells | nuclei) is the ONE object FOV selection segments and scores. It
        # drives: (a) how the reconstruction outputs are reduced to a single segmentation
        # input (pipeline._resolve_seg_input), (b) the InstanSeg head (below), and (c) which
        # channels VS must predict. Selection always produces ONE mask -> single-channel
        # (plain) feature names, so `target` is recorded as run metadata, not in column names.
        self._target = config.target
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
        self._worker = None  # FOVSelectionWorker (subprocess) unless decide_fn set
        self._decide_fn = decide_fn

        # Fail before acquiring if the model asks for a feature the configured preprocessing
        # cannot produce (a typo would otherwise be read as a silently-missing column). Skipped
        # when a decide_fn stands in for the pipeline -- it never extracts features.
        if self._decide_fn is None:
            self._validate_feature_names(config)
            self._validate_segmentation_checkpoint()

    # -- construction ------------------------------------------------------

    @staticmethod
    def _store_stem(data_path: Path | None) -> str | None:
        """The output store's name without its zarr extension: ``acq_1.ome.zarr`` -> ``acq_1``.

        The single source of truth for naming every sibling artifact. The engine always
        indexes the acquisition name (``acq`` -> ``acq_1``), so the index rides along
        inside the stem and nothing else has to be threaded down here.
        """
        if data_path is None:
            return None
        name = Path(data_path).name
        for ext in (".ome.zarr", ".zarr"):
            if name.endswith(ext):
                return name[: -len(ext)]
        return name

    @classmethod
    def _sibling_path(cls, data_path: Path | None, suffix: str) -> Path | None:
        """Sibling artifact path next to the output store: ``<store stem><suffix>``.

        ``acq_1.ome.zarr`` yields ``acq_1_fov_debug`` / ``acq_1_prescan.ome.zarr``, so a
        run's artifacts all share its store's name and sort beside it. Derived purely
        from the store path -- there is no separate index to keep in sync, and a
        user-supplied name that itself ends in a digit (``plate_2`` -> ``plate_2_1``) is
        never mis-parsed, because nothing here parses.
        """
        stem = cls._store_stem(data_path)
        if stem is None:
            return None
        return Path(data_path).with_name(f"{stem}{suffix}")

    @classmethod
    def _debug_dir_for(cls, data_path: Path | None) -> Path | None:
        """Sibling ``<acq>_fov_debug/`` directory next to the output store."""
        return cls._sibling_path(data_path, "_fov_debug")

    @classmethod
    def _matrix_stem_for(cls, data_path: Path | None) -> str | None:
        """Stem for the optional best-focus-Z debug CSV, ``<acq>_fov_feature_matrix``.

        The main calibration table and its image folders use fixed names
        (``fov_summary.csv`` / ``prescan_fov`` / ``prescan_mask``); this stem only labels
        the best-focus-Z CSV.
        """
        stem = cls._store_stem(data_path)
        return None if stem is None else f"{stem}_fov_feature_matrix"

    @classmethod
    def _prescan_recon_path_for(cls, data_path: Path | None) -> Path | None:
        """Sibling ``<acq>_prescan.ome.zarr`` store next to the output store."""
        path = cls._sibling_path(data_path, "_prescan")
        return None if path is None else path.with_name(f"{path.name}.ome.zarr")

    @classmethod
    def _config_backup_path_for(cls, data_path: Path | None) -> Path | None:
        """Sibling ``<acq>_config_backup.yaml`` next to the output store.

        The acquisition config with the SELECTED FOVs filled in; written by
        :func:`shrimpy.fov_selection.acquisition_artifacts.save_selected_config`. Named
        from the store like every other sibling, so each run gets its own backup and no
        run can overwrite another's.
        """
        return cls._sibling_path(data_path, "_config_backup.yaml")

    @classmethod
    def from_metadata(
        cls,
        meta: FOVSelectionConfig | None,
        sequence: MDASequence,
        pixel_size_um: float,
        data_path: Path | None = None,
        decide_fn: Callable[[np.ndarray], tuple[float, bool]] | None = None,
    ) -> FOVSelection | None:
        """Build the coordinator from the validated ``fov_selection`` metadata block.

        Returns ``None`` when FOV selection is disabled. The block itself was already
        validated by :class:`~shrimpy.fov_selection.config.FOVSelectionConfig` when the
        acquisition config was loaded; what is checked here is the pairing of that block
        with *this acquisition* -- the pixel size the hardware reports, and the sequence's
        Z step. Both raise before acquiring.
        """
        if meta is None or not meta.enabled:
            return None
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
        if (meta.deskew or meta.phase) and not z_step_um:
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

    def _validate_feature_names(self, config: FOVSelectionConfig) -> None:
        """Fail before acquiring if the model asks for a feature the pipeline cannot produce.

        Checks the config-defined models (``ranking_by_defined_range`` /
        ``classification_by_thresholding``), whose feature names are hand-typed: a typo would
        otherwise surface only as a silently-missing column (NaN) at decision time rather than
        an error. A trained ``classification_tree``'s names come from training (not the config)
        and are validated when the worker builds the model, so it declares no ``features``
        block and is skipped here.
        """
        requested = list(getattr(config.model, "features", None) or {})
        producible = self._producible_feature_names()
        unknown = [name for name in requested if name not in producible]
        if unknown:
            raise ValueError(
                "FOV selection: model requests feature name(s) the configured preprocessing "
                f"cannot produce: {unknown}. Available feature names: {sorted(producible)}. "
                "Fix the names under fov_selection.model.features (feature keys are plain, "
                "e.g. 'coverage_frac' -- no channel prefix). Aborting before acquisition."
            )

    def _validate_segmentation_checkpoint(self) -> None:
        """Fail before acquiring when the InstanSeg checkpoint is not on this machine.

        The backend is only loaded inside the worker subprocess, which does not start until
        the pre-scan is already running, so a missing checkpoint would otherwise surface as a
        mid-acquisition worker crash. The rest of the block (backend name, required ``path``)
        is the schema's job -- this is the one check that depends on the filesystem, which a
        config written on another machine cannot know about.
        """
        path = self._segmentation.get("path")
        if path and not Path(path).exists():
            raise FileNotFoundError(
                f"fov_selection.segmentation.path: InstanSeg checkpoint not found: {path}. "
                "Aborting before acquisition."
            )

    def _recon_config(self) -> dict:
        """Assemble the reconstruction sub-config for build_preprocessor.

        The deskew/phase/virtual_staining blocks live directly under
        ``fov_selection`` (DynaTrack style); only the reconstruction steps of the
        preprocessing list are relevant to the preprocessor (it ignores
        projection/segmentation).
        """
        recon = {
            "preprocessing": self._steps,
            "deskew": self.config.deskew,
            "phase": self.config.phase,
            "virtual_staining": self.config.virtual_staining,
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
            from shrimpy.fov_selection.worker import FOVSelectionWorker, WorkerConfig

            recon = self._inject_scales(self._recon_config())
            logger.info("FOV selection: starting worker process for shape %s", zyx_shape)
            self._worker = FOVSelectionWorker(
                WorkerConfig(
                    recon=recon,
                    target=self._target,
                    recon_channels=self._recon_channels,
                    segmentation=self._segmentation,
                    model_cfg=self._model_cfg,
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
        :meth:`outcome` is read to build the timelapse run.
        """
        self._await_pending(timeout=timeout)

    def outcome(self) -> PrescanOutcome:
        """Everything the timelapse run needs from this pre-scan (see :class:`PrescanOutcome`).

        Call after :meth:`drain` and *before* the debug writers
        (:meth:`log_selection_summary`, :meth:`finalize_debug_summary`,
        :meth:`export_prescan_nd`). Those are individually guarded now, but the ordering
        is what keeps the science independent of them: the capture used to run last, so a
        PermissionError writing ``fov_summary.csv`` -- a spreadsheet app holding it open --
        aborted teardown with nothing captured, and the timelapse was skipped for "no FOVs
        passed" despite a perfectly good selection.

        Safe to call after :meth:`shutdown` (that clears the frame buffers, not the
        verdicts), though the engine has no reason to: the coordinator does not outlive
        ``teardown_sequence``.
        """
        if self._calibration_mode:
            # Calibration stops after the pre-scan, so there is no selection to hand on --
            # only the feature matrix for the viewer. log_selection_summary still reports
            # which FOVs the model WOULD have selected.
            return PrescanOutcome(
                selected_fovs=[], calibration_csv=self.calibration_matrix_csv
            )
        return PrescanOutcome(selected_fovs=self.passed_position_names())

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

    def export_prescan_nd(self) -> None:
        """Export ``fov_summary.csv`` to an AnnData zarr for Embedding Atlas (call post-drain).

        Gated by ``fov_selection.save_pre_scan_nd`` (which forces ``save_decision`` on so the
        CSV exists). Runs in every mode, after :meth:`finalize_debug_summary` so the CSV carries
        the well and (normal-mode) decision columns. Best-effort: a missing optional dependency
        (anndata) or a write failure is logged, never raised, so it cannot take the acquisition
        down.
        """
        if not self._save_pre_scan_nd or self._debug_dir is None:
            return
        summary_path = Path(self._debug_dir) / prescan_artifacts.SUMMARY_CSV_NAME
        if not summary_path.exists():
            logger.warning(
                "FOV selection: %s not found; skipping the ND (Embedding Atlas) export",
                summary_path,
            )
            return
        try:
            from shrimpy.fov_selection import nd_export

            out = nd_export.write_feature_anndata(summary_path)
            logger.info("FOV selection: wrote the ND feature export to %s", out)
        except Exception:
            logger.exception(
                "FOV selection: ND export failed for %s; the acquisition is unaffected",
                summary_path,
            )

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
