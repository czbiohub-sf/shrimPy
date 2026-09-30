"""Validated schema for the ``metadata.fov_selection`` acquisition config block.

:class:`FOVSelectionConfig` is the single source of truth for what a FOV-selection
config may contain. :class:`shrimpy.config.ShrimpyMetadata` declares the section as
``fov_selection: FOVSelectionConfig | None``, so ``load_config`` rejects a mistyped
key, a missing required field, or an unusable curve *before* any hardware is touched
-- the same fail-before-acquiring contract :class:`~shrimpy.dynatrack.tracking.DynaTrackConfig`
gives DynaTrack.

Written in the acquisition YAML as::

    metadata:
      fov_selection:
        enabled: true
        prescan_mda: {...}                 # a complete nested MDASequence
        fov_selection_channel: BF
        target: cells                      # cells | nuclei
        preprocessing: ['deskew', 'phase', 'best_focus_z', 'segmentation']
        deskew: {...}                      # biahub DeskewSettings
        phase: {...}                       # waveorder phase Settings
        best_focus_z: {...}                # optics for the best_focus_z projection
        segmentation: {model: instanseg, path: ...}
        model: {type: ranking_by_defined_range, top_fov: 3, features: {...}}

What is modelled here and what is not
-------------------------------------
Typed: every setting shrimPy itself defines -- the pre-scan sequence, the
preprocessing pipeline, the segmentation backend, and the FOV-goodness model.

Left as plain ``dict``: ``deskew``, ``phase``, and ``virtual_staining``, which are
validated against their upstream schemas where they are consumed (``biahub``'s
``DeskewSettings``, ``waveorder``'s phase ``Settings``, cytoland's ``VSUNet`` -- see
:func:`shrimpy.preprocessing.build_preprocessor`). Mirroring those here would be a
second source of truth *and* would make this module unimportable without the optional
reconstruction dependencies. Same choice as :class:`DynaTrackConfig`.

Defaults
--------
A field carries a default here only when this config layer owns it (``enabled``,
``require_gpu``, the ``save_*`` flags). Backend tuning knobs whose default is a tuned
constant on the consuming class -- the Cellpose thresholds, the InstanSeg forward
arguments -- are ``| None = None`` instead, and the consumer's own default applies when
they are unset. The manager dumps the sub-blocks with ``exclude_none=True``, so an unset
knob never reaches the backend as an explicit ``None`` that would shadow its default.

Presence, not ``enabled``, drives validation: a section that is present is validated in
full even with ``enabled: false`` (the timelapse run of an adaptive acquisition is
handed exactly such a block -- see
:func:`shrimpy.fov_selection.sequences.build_main_sequence`). Omit the section
entirely to disable FOV selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from useq import MDASequence

from shrimpy.preprocessing import RECON_STEPS

__all__ = [
    "PROJECTION_STEPS",
    "BestFocusZSettings",
    "CellposeSettings",
    "ClassificationTreeModelSettings",
    "FOVSelectionConfig",
    "GaussianFeature",
    "InstansegSettings",
    "LognormalFeature",
    "OtsuSettings",
    "RankingModelSettings",
    "SigmoidFeature",
    "ThresholdingModelSettings",
]

# Projection step name (as written in `preprocessing`) -> the method
# :func:`shrimpy.fov_selection.pipeline.project_zyx` takes. Exactly one may appear in a
# pipeline: they are alternative ways to reduce the same z-stack to one 2D image, so two
# of them is a config error rather than a precedence question.
PROJECTION_STEPS = {
    "sum_projection": "sum",
    "max_projection": "max",
    "middle_slice_projection": "middle",
    "logstd_projection": "logstd",
    "best_focus_z": "best_focus_z",
}

# Projection used when the pipeline names none. 'sum' is also project_zyx's own default;
# it is channel-agnostic and a no-op for a single-slice stack, so a 2D pre-scan needs no
# projection step at all.
DEFAULT_PROJECTION = "sum"

# Features are computed from segmentation masks, so this step is mandatory.
SEGMENTATION_STEP = "segmentation"

# Every step name a `preprocessing` list may contain.
PIPELINE_STEPS = (*RECON_STEPS, *PROJECTION_STEPS, SEGMENTATION_STEP)

# The object FOV selection segments and scores; also the InstanSeg head name.
TARGETS = ("cells", "nuclei")


# ---------------------------------------------------------------------------
# Projection optics
# ---------------------------------------------------------------------------


class BestFocusZSettings(BaseModel):
    """Optics for the ``best_focus_z`` projection (waveorder's transverse-band focus).

    Required as a block whenever ``best_focus_z`` is the projection step: without the
    NA and the wavelength the focus metric is meaningless, and
    :func:`shrimpy.fov_selection.pipeline.project_zyx` would refuse mid-run rather than
    silently mis-project. The XY pixel size the metric also needs is *not* a field: it
    comes from ``core.getPixelSizeUm()`` at acquisition start (single source of truth).

    Parameters
    ----------
    numerical_aperture_detection : float
        Detection (objective) NA.
    wavelength_illumination : float
        Illumination wavelength in microns, matching the pixel-size unit.
    mode : {'max', 'min'}
        Pick the slice of extreme mid-band power: ``'max'`` for bright objects on a
        dark background (the default), ``'min'`` for the inverse.
    midband_fractions : tuple[float, float]
        The scored spatial-frequency band, as a fraction of the cutoff frequency.
    """

    model_config = ConfigDict(extra="forbid")

    numerical_aperture_detection: float = Field(gt=0)
    wavelength_illumination: float = Field(gt=0)
    mode: Literal["max", "min"] = "max"
    midband_fractions: tuple[float, float] = (0.125, 0.25)


# ---------------------------------------------------------------------------
# Segmentation backends
# ---------------------------------------------------------------------------


class _SegmentationSettings(BaseModel):
    """Shared config for the segmentation backends in :mod:`.segmentation`.

    ``protected_namespaces=()`` because the backends are selected by a field literally
    named ``model`` and configured with ``model_name`` / ``model_pixel_size_um``; those
    names are the config's, and pydantic's ``model_`` reservation would otherwise warn
    about them.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def _reject_target(cls, data: Any) -> Any:
        """``target`` is a top-level field, not a per-backend one.

        It names the ONE object selection segments and scores, and drives more than
        segmentation (the VS channels predicted, how reconstruction outputs reduce to a
        single segmentation input, the InstanSeg head). The manager injects it into the
        segmentation block from the top-level field, so a copy here would be silently
        overwritten -- worth naming explicitly, since earlier example configs wrote it.
        """
        if isinstance(data, dict) and "target" in data:
            raise ValueError(
                "fov_selection.segmentation.target is not a segmentation setting: move it "
                "to the top level as 'fov_selection.target' (cells | nuclei), which also "
                "drives the InstanSeg head."
            )
        return data


class CellposeSettings(_SegmentationSettings):
    """Cellpose instance segmentation (:class:`~.segmentation.CellposeSegmenter`).

    Every tuning knob is optional; unset ones fall back to the class constants on
    ``CellposeSegmenter`` (the model's own training defaults).

    Parameters
    ----------
    model_name : str | None
        Cellpose model to load, e.g. ``"cpdino"``.
    gpu : bool
        Run the model on the GPU when one is available.
    flow_threshold, cellprob_threshold : float | None
        Mask-shape QC and the mask-probability cutoff (lower = more/larger masks).
    batch_size : int | None
        Cellpose tiles per forward pass.
    min_size : int | None
        Drop masks smaller than this many pixels.
    diameters : dict[{'cells', 'nuclei'}, float | None] | None
        Object diameter in microns, keyed by ``target``; ``None`` lets Cellpose
        auto-scale, which is correct for nuclei but under-covers whole cells. Keyed by
        ``target`` and not by a channel name -- ``CellposeSegmenter._diameter_for``
        looks the value up by target, so any other key is silently never read.
    """

    model: Literal["cellpose"] = "cellpose"
    model_name: str | None = None
    gpu: bool = True
    flow_threshold: float | None = None
    cellprob_threshold: float | None = None
    batch_size: int | None = Field(default=None, gt=0)
    min_size: int | None = Field(default=None, ge=0)
    diameters: dict[Literal["cells", "nuclei"], float | None] | None = None


class InstansegSettings(_SegmentationSettings):
    """InstanSeg TorchScript segmentation (:class:`~.segmentation.InstansegSegmenter`).

    Parameters
    ----------
    path : Path
        The checkpoint: a bioimage.io ``.zip`` export (which carries the training pixel
        size in its ``rdf.yaml``) or a bare TorchScript ``.pt``. Existence is checked by
        the coordinator before acquiring, not here -- a config is often written on a
        different machine from the one that runs it.
    gpu : bool
        Run on CUDA when available.
    model_pixel_size_um : float | None
        Resolution the network was trained at. InstanSeg has no ``diameter`` knob --
        resampling each FOV to this pixel size IS how object scale is communicated.
        Read from a ``.zip``'s ``rdf.yaml`` when present; set it explicitly for a bare
        ``.pt``, or object scale will be wrong.
    percentiles : tuple[float, float] | None
        Per-image percentile scaling applied before inference.

    The remaining fields are the TorchScript module's own forward arguments; each is
    passed through only when set, so the module's default otherwise applies.
    """

    model: Literal["instanseg"]
    path: Path
    gpu: bool = True
    model_pixel_size_um: float | None = Field(default=None, gt=0)
    percentiles: tuple[float, float] | None = None

    # Forward kwargs of the TorchScript module (segmentation.InstansegSegmenter.FORWARD_ARGS).
    min_size: int | None = None
    mask_threshold: float | None = None
    peak_distance: int | None = None
    seed_threshold: float | None = None
    overlap_threshold: float | None = None
    mean_threshold: float | None = None
    fg_threshold: float | None = None
    window_size: int | None = None
    cleanup_fragments: bool | None = None
    resolve_cell_and_nucleus: bool | None = None
    tta: bool | None = None


class OtsuSettings(_SegmentationSettings):
    """Otsu thresholding (:class:`~.segmentation.OtsuSegmenter`) -- no model, no GPU.

    Threshold, optionally close, drop small components, fill holes, label. Suited to a
    ``logstd_projection`` brightfield image whose foreground is a texture blob. All
    sizes are in pixels.

    Parameters
    ----------
    close_radius : int
        Morphological-closing disk radius; 0 skips the closing.
    min_size : int
        Drop foreground components smaller than this; 0 keeps all of them.
    fill_holes : bool
        Fill interior holes before labelling.
    """

    model: Literal["otsu"]
    close_radius: int = Field(default=0, ge=0)
    min_size: int = Field(default=0, ge=0)
    fill_holes: bool = True


SegmentationSettings = Annotated[
    CellposeSettings | InstansegSettings | OtsuSettings,
    Field(discriminator="model"),
]


# ---------------------------------------------------------------------------
# FOV-goodness model
# ---------------------------------------------------------------------------


class _RankingFeature(BaseModel):
    """One desirability curve of a :class:`RankingModelSettings` profile.

    ``weight`` (default 1.0) scales the feature's contribution to the aggregated score.
    The shape-specific parameters are the interpretable ones the feature viewer's Rank
    tab edits and writes, not the internal bounds
    :class:`~shrimpy.fov_selection.fov_model.DesirabilityModel` stores.
    """

    model_config = ConfigDict(extra="forbid")

    weight: float = Field(default=1.0, ge=0)


class GaussianFeature(_RankingFeature):
    """Symmetric bell: desirability 1 at ``center``, 0.5 at ``center ± fwhm/2``."""

    shape: Literal["gaussian"]
    center: float
    fwhm: float = Field(gt=0)


class LognormalFeature(_RankingFeature):
    """Right-skewed bell over ``x > 0``: peak at ``center``, desirability 0.5 at
    ``center * fold`` and ``center / fold``."""

    shape: Literal["lognormal"]
    center: float = Field(gt=0)
    fold: float = Field(gt=1)


class SigmoidFeature(_RankingFeature):
    """Monotonic logistic: desirability 0.5 at ``midpoint``, rising (``direction:
    higher``) or falling (``lower``) over a 10%->90% span of ``width``."""

    shape: Literal["sigmoid"]
    midpoint: float
    width: float = Field(gt=0)
    direction: Literal["higher", "lower"]


RankingFeature = Annotated[
    GaussianFeature | LognormalFeature | SigmoidFeature,
    Field(discriminator="shape"),
]


class FeatureRange(BaseModel):
    """A ``[lo, hi]`` acceptance band for :class:`ThresholdingModelSettings`.

    Accepts either the explicit ``{range: [lo, hi]}`` mapping or the bare ``[lo, hi]``
    list, which is what the configs are usually written with.
    """

    model_config = ConfigDict(extra="forbid")

    range: tuple[float, float]

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_pair(cls, data: Any) -> Any:
        if isinstance(data, (list, tuple)):
            return {"range": data}
        return data

    @model_validator(mode="after")
    def _ordered(self) -> FeatureRange:
        lo, hi = self.range
        if lo > hi:
            raise ValueError(f"range must be [lo, hi] with lo <= hi; got [{lo}, {hi}]")
        return self


class RankingModelSettings(BaseModel):
    """``type: ranking_by_defined_range`` -- weighted desirability, selected by ranking.

    Each feature gets a curve (:data:`RankingFeature`); the curves combine into one score
    per FOV by ``aggregation``, and the ``top_fov`` highest-scoring FOVs OF EACH POSITION
    (well / grid centre) pass. There is no per-FOV good/bad verdict, so ``top_fov`` is
    required and ``threshold`` does not apply -- it is not a field of this model.

    ``aggregation`` mirrors
    :attr:`~shrimpy.fov_selection.fov_model.DesirabilityModel.AGGREGATIONS`: ``gaussian``
    (one N-dimensional gaussian over every feature at once -- the strongest veto, and
    the default), ``product`` (weighted geometric mean; one weak feature vetoes), or
    ``sum`` (compensatory weighted mean). See that class for the full derivation.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["ranking_by_defined_range"]
    top_fov: int = Field(ge=1)
    aggregation: Literal["sum", "product", "gaussian"] = "gaussian"
    features: dict[str, RankingFeature] = Field(min_length=1)


class ThresholdingModelSettings(BaseModel):
    """``type: classification_by_thresholding`` -- a hard QC box.

    A FOV passes iff every feature is inside its ``[lo, hi]`` range. The verdict is
    per-FOV, so there is no ``top_fov`` and no ``threshold``.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["classification_by_thresholding"]
    features: dict[str, FeatureRange] = Field(min_length=1)


class ClassificationTreeModelSettings(BaseModel):
    """``type: classification_tree`` -- a trained ``.joblib`` (median imputer + tree).

    The feature names come from training rather than the config, so there is no
    ``features`` block. ``threshold`` is the ``P(good)`` cutoff; it is a field of this
    model alone, because the other two ignore it.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["classification_tree"]
    path: Path
    threshold: float = Field(default=0.5, ge=0, le=1)


FOVModelSettings = Annotated[
    RankingModelSettings | ThresholdingModelSettings | ClassificationTreeModelSettings,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# The fov_selection block
# ---------------------------------------------------------------------------


class FOVSelectionConfig(BaseModel):
    """The ``metadata.fov_selection`` block of an acquisition config.

    Parameters
    ----------
    enabled : bool
        Master switch. Engines only build a
        :class:`~shrimpy.fov_selection.manager.FOVSelection` coordinator when True; the
        rest of the block is validated either way (see the module docstring).
    calibration_mode : bool
        Run the pre-scan ONLY -- no timelapse. Every producible feature is extracted
        (not just the model's), the debug artifacts are written in the feature viewer's
        layout, and the viewer opens on the result. Forces ``save_decision`` on.
    prescan_mda : MDASequence | None
        The pre-scan run: a complete nested sequence carrying the candidate FOVs
        (``stage_positions``, optionally plus a ``grid_plan``, or a ``WellPlatePlan``),
        its own ``z_plan`` -- which may be a single 2D slice, independent of the
        timelapse -- and a single timepoint.

        Optional at this level rather than required-when-enabled, because the pre-scan
        run's *own* metadata carries this same block with ``prescan_mda`` stripped (it
        would otherwise nest a redundant copy of itself).
        :func:`~shrimpy.fov_selection.sequences.build_prescan_sequence` raises when it
        is genuinely missing.
    fov_selection_channel : str
        The acquired channel imaged during the pre-scan and fed to reconstruction. Must
        name one of ``prescan_mda``'s channels (checked against the sequence, which is
        not visible here).
    target : {'cells', 'nuclei'}
        The ONE object selection segments and scores. Drives the InstanSeg head, which
        channels virtual staining predicts, and how reconstruction outputs reduce to a
        single segmentation input. Selection always produces one mask, so feature names
        stay plain (no channel prefix) and this is recorded as run metadata instead.
    preprocessing : list[str]
        Ordered steps: reconstruction (any subset of ``flatfield``, ``deskew``,
        ``phase``, ``vs``), then at most ONE projection
        (:data:`PROJECTION_STEPS`), then ``segmentation``. Unknown names, duplicates, a
        second projection, or a missing ``segmentation`` are all errors.
    require_gpu : bool
        Fail fast when reconstruction cannot run on a GPU. The per-FOV decision has to
        keep up with the pre-scan, which is impractical on CPU; set False only to debug.
    deskew, phase, virtual_staining : dict | None
        Reconstruction sub-configs, validated against their upstream schemas at
        preprocessor-build time (see the module docstring). The XY pixel size and Z step
        are injected from the acquisition, never written here.
    best_focus_z : BestFocusZSettings | None
        Optics for the ``best_focus_z`` projection; required when that step is selected.
    segmentation : CellposeSettings | InstansegSettings | OtsuSettings
        The segmentation backend, chosen by its ``model`` field.
    model : RankingModelSettings | ThresholdingModelSettings | ClassificationTreeModelSettings
        The FOV-goodness model, chosen by its ``type`` field. The type also decides the
        SELECTION rule: ranking keeps ``top_fov`` per position, the classification models
        select on their per-FOV verdict.
    save_decision : bool
        Write the per-FOV debug artifacts (projection/mask PNGs + ``fov_summary.csv``)
        to a sibling ``<name>_fov_debug/`` directory.
    save_pre_scan_omezarr : bool
        Write the per-step reconstruction to a sibling ``<name>_prescan.ome.zarr``. The
        pre-scan run writes nothing itself, so this replaces the raw store rather than
        adding to it.
    save_pre_scan_nd : bool
        Also export ``fov_summary.csv`` to an AnnData zarr for Embedding Atlas. Reads
        the CSV after the drain, so it forces ``save_decision`` on.
    save_best_focus_z_for_debug : bool
        Append the detected best-focus slice and depth per FOV to a debug CSV. Only
        meaningful with the ``best_focus_z`` projection.

    Raises
    ------
    pydantic.ValidationError
        On an unknown key anywhere in the block, an invalid ``preprocessing`` pipeline,
        or a ``best_focus_z`` projection with no ``best_focus_z`` optics.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    calibration_mode: bool = False
    prescan_mda: MDASequence | None = None
    fov_selection_channel: str
    target: Literal["cells", "nuclei"]
    preprocessing: list[str]
    require_gpu: bool = True
    deskew: dict[str, Any] | None = None
    phase: dict[str, Any] | None = None
    virtual_staining: dict[str, Any] | None = None
    best_focus_z: BestFocusZSettings | None = None
    segmentation: SegmentationSettings
    model: FOVModelSettings
    save_decision: bool = False
    save_pre_scan_omezarr: bool = False
    save_pre_scan_nd: bool = False
    save_best_focus_z_for_debug: bool = False

    @model_validator(mode="after")
    def _check_pipeline(self) -> FOVSelectionConfig:
        """Validate the ``preprocessing`` list and the blocks its steps depend on.

        Every one of these used to surface late or not at all: an unknown or duplicated
        step was silently ignored, two projections resolved by a hidden precedence order,
        and a reconstruction step whose sub-config was missing was quietly dropped by
        ``build_preprocessor`` -- so the run reconstructed something other than what the
        config said, and only the features looked wrong.
        """
        unknown = [step for step in self.preprocessing if step not in PIPELINE_STEPS]
        if unknown:
            raise ValueError(
                f"unknown preprocessing step(s) {unknown}; "
                f"choose from {sorted(PIPELINE_STEPS)}"
            )

        duplicates = sorted(
            {step for step in self.preprocessing if self.preprocessing.count(step) > 1}
        )
        if duplicates:
            raise ValueError(f"preprocessing step(s) repeated: {duplicates}")

        projections = [step for step in self.preprocessing if step in PROJECTION_STEPS]
        if len(projections) > 1:
            raise ValueError(
                f"preprocessing names {len(projections)} projection steps {projections}; "
                "they are alternative ways to reduce one z-stack to a 2D image, so pick one"
            )

        if SEGMENTATION_STEP not in self.preprocessing:
            raise ValueError(
                f"preprocessing must include a {SEGMENTATION_STEP!r} step "
                f"(features come from segmentation masks). Got {self.preprocessing}."
            )

        # A reconstruction step with no sub-config is a no-op inside build_preprocessor,
        # which would reconstruct something other than what the config asked for.
        for step, block in (
            ("deskew", self.deskew),
            ("phase", self.phase),
            ("vs", self.virtual_staining),
        ):
            field = "virtual_staining" if step == "vs" else step
            if step in self.preprocessing and not block:
                raise ValueError(
                    f"preprocessing includes {step!r} but there is no "
                    f"'fov_selection.{field}' block to configure it"
                )

        if self.projection == "best_focus_z" and self.best_focus_z is None:
            raise ValueError(
                "preprocessing selects the 'best_focus_z' projection, which needs a "
                "'fov_selection.best_focus_z' block (detection NA + illumination "
                "wavelength in um); add one or choose another projection step"
            )
        return self

    @property
    def projection(self) -> str:
        """Projection method for :func:`shrimpy.fov_selection.pipeline.project_zyx`.

        Derived from the single projection step in ``preprocessing``
        (:data:`PROJECTION_STEPS`), or :data:`DEFAULT_PROJECTION` when the pipeline names
        none -- a projection step is not mandatory.
        """
        for step in self.preprocessing:
            if step in PROJECTION_STEPS:
                return PROJECTION_STEPS[step]
        return DEFAULT_PROJECTION

    @property
    def uses_virtual_staining(self) -> bool:
        """Whether the pipeline reconstructs virtual-stain channels."""
        return "vs" in self.preprocessing

    def block(self, *, exclude: set[str] | None = None) -> dict[str, Any]:
        """This block as a plain JSON-safe mapping, ready to write back into metadata.

        ``mode="json"`` so ``Path`` and the nested ``MDASequence`` come out as data that
        survives a YAML round-trip, a re-validation, and the pickle into the worker
        subprocess. ``exclude_none`` drops every unset optional field, so a backend knob
        that was never configured reaches its consumer as *absent* rather than as an
        explicit ``None`` that would shadow the consumer's own default.
        """
        return self.model_dump(mode="json", exclude_none=True, exclude=exclude)
