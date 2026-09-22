"""Tests for the ``metadata.fov_selection`` schema.

:class:`~shrimpy.fov_selection.config.FOVSelectionConfig` is the fail-before-acquiring
gate for FOV selection. Most of what it rejects used to surface either as a late
exception from the coordinator or -- worse -- as a silently *different* run: an unknown
step ignored, two projections resolved by a hidden precedence order, a reconstruction
step whose sub-config was missing quietly skipped. Each test below pins one of those to
a config error instead.

The last section guards against drift: the ``Literal``\\ s here have to keep naming
exactly the backends, model types, and curve shapes their consumers implement.
"""

from __future__ import annotations

from typing import get_args

import pytest

from pydantic import ValidationError

from shrimpy.fov_selection.config import (
    PIPELINE_STEPS,
    PROJECTION_STEPS,
    TARGETS,
    CellposeSettings,
    ClassificationTreeModelSettings,
    FOVSelectionConfig,
    GaussianFeature,
    InstansegSettings,
    LognormalFeature,
    OtsuSettings,
    RankingModelSettings,
    SigmoidFeature,
    ThresholdingModelSettings,
)

RANKING_FEATURE = {"shape": "gaussian", "center": 0.5, "fwhm": 0.2}
BEST_FOCUS_Z = {"numerical_aperture_detection": 1.35, "wavelength_illumination": 0.45}


def _cfg(**overrides) -> dict:
    """A minimal valid block; ``overrides`` replace whole top-level fields."""
    cfg = {
        "enabled": True,
        "fov_selection_channel": "BF",
        "target": "cells",
        "preprocessing": ["segmentation"],
        "segmentation": {"model": "otsu"},
        "model": {
            "type": "ranking_by_defined_range",
            "top_fov": 3,
            "features": {"coverage_frac": RANKING_FEATURE},
        },
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# the block as a whole
# ---------------------------------------------------------------------------


def test_minimal_block_parses():
    cfg = FOVSelectionConfig.model_validate(_cfg())
    assert cfg.enabled is True
    assert cfg.require_gpu is True  # on by default: a CPU decision cannot keep up
    assert cfg.prescan_mda is None  # optional here; build_prescan_sequence requires it
    assert cfg.projection == "sum"  # no projection step named -> the default


def test_unknown_key_is_rejected():
    with pytest.raises(ValidationError, match="fov_selection_chanel"):
        FOVSelectionConfig.model_validate(_cfg(fov_selection_chanel="BF"))


@pytest.mark.parametrize(
    "field", ["fov_selection_channel", "target", "preprocessing", "segmentation", "model"]
)
def test_required_fields(field):
    cfg = _cfg()
    del cfg[field]
    with pytest.raises(ValidationError, match=field):
        FOVSelectionConfig.model_validate(cfg)


def test_target_must_be_a_segmentable_object():
    with pytest.raises(ValidationError, match="target"):
        FOVSelectionConfig.model_validate(_cfg(target="brightfield"))
    for target in TARGETS:
        assert FOVSelectionConfig.model_validate(_cfg(target=target)).target == target


def test_prescan_mda_is_parsed_as_a_sequence():
    cfg = FOVSelectionConfig.model_validate(
        _cfg(
            prescan_mda={
                "channels": [{"config": "BF", "group": "Channel"}],
                "stage_positions": [{"x": 0, "y": 0}],
                "time_plan": {"loops": 1, "interval": 0},
            }
        )
    )
    assert [c.config for c in cfg.prescan_mda.channels] == ["BF"]
    # an invalid nested sequence fails here rather than at build_prescan_sequence time
    with pytest.raises(ValidationError, match="z_plan"):
        FOVSelectionConfig.model_validate(_cfg(prescan_mda={"z_plan": {"top": 1}}))


# ---------------------------------------------------------------------------
# preprocessing pipeline
# ---------------------------------------------------------------------------


def test_unknown_step_is_rejected():
    with pytest.raises(ValidationError, match="unknown preprocessing step"):
        FOVSelectionConfig.model_validate(_cfg(preprocessing=["sum", "segmentation"]))


def test_repeated_step_is_rejected():
    with pytest.raises(ValidationError, match="repeated"):
        FOVSelectionConfig.model_validate(
            _cfg(preprocessing=["flatfield", "flatfield", "segmentation"])
        )


def test_two_projections_are_rejected():
    # They are alternative reductions of one z-stack, so naming both is a config error,
    # not a precedence question.
    with pytest.raises(ValidationError, match="projection steps"):
        FOVSelectionConfig.model_validate(
            _cfg(preprocessing=["sum_projection", "max_projection", "segmentation"])
        )


def test_segmentation_step_is_required():
    with pytest.raises(ValidationError, match="'segmentation' step"):
        FOVSelectionConfig.model_validate(_cfg(preprocessing=["sum_projection"]))


@pytest.mark.parametrize(
    ("step", "field"),
    [("deskew", "deskew"), ("phase", "phase"), ("vs", "virtual_staining")],
)
def test_reconstruction_step_needs_its_block(step, field):
    # build_preprocessor treats a step with no sub-config as a no-op, so the run would
    # reconstruct something other than what the config says and only the features would
    # look wrong.
    with pytest.raises(ValidationError, match=f"fov_selection.{field}"):
        FOVSelectionConfig.model_validate(_cfg(preprocessing=[step, "segmentation"]))

    cfg = FOVSelectionConfig.model_validate(
        _cfg(**{"preprocessing": [step, "segmentation"], field: {"some": "setting"}})
    )
    assert step in cfg.preprocessing


@pytest.mark.parametrize(("step", "projection"), sorted(PROJECTION_STEPS.items()))
def test_projection_is_derived_from_the_step(step, projection):
    extra = {"best_focus_z": BEST_FOCUS_Z} if step == "best_focus_z" else {}
    cfg = FOVSelectionConfig.model_validate(
        _cfg(preprocessing=[step, "segmentation"], **extra)
    )
    assert cfg.projection == projection


def test_best_focus_z_projection_requires_its_optics():
    with pytest.raises(ValidationError, match="best_focus_z"):
        FOVSelectionConfig.model_validate(_cfg(preprocessing=["best_focus_z", "segmentation"]))


def test_best_focus_z_optics_are_validated():
    with pytest.raises(ValidationError, match="wavelength_illumination"):
        FOVSelectionConfig.model_validate(
            _cfg(best_focus_z={"numerical_aperture_detection": 1.35})
        )
    cfg = FOVSelectionConfig.model_validate(_cfg(best_focus_z=BEST_FOCUS_Z))
    assert cfg.best_focus_z.mode == "max"
    assert cfg.best_focus_z.midband_fractions == (0.125, 0.25)


# ---------------------------------------------------------------------------
# segmentation backends
# ---------------------------------------------------------------------------


def test_segmentation_backend_is_discriminated_by_model():
    assert isinstance(FOVSelectionConfig.model_validate(_cfg()).segmentation, OtsuSettings)
    cfg = FOVSelectionConfig.model_validate(
        _cfg(segmentation={"model": "instanseg", "path": "model.zip"})
    )
    assert isinstance(cfg.segmentation, InstansegSettings)
    assert isinstance(
        FOVSelectionConfig.model_validate(
            _cfg(segmentation={"model": "cellpose"})
        ).segmentation,
        CellposeSettings,
    )


def test_unknown_segmentation_backend_is_rejected():
    with pytest.raises(ValidationError, match="cellpose"):
        FOVSelectionConfig.model_validate(_cfg(segmentation={"model": "stardist"}))


def test_instanseg_requires_a_checkpoint_path():
    with pytest.raises(ValidationError, match="path"):
        FOVSelectionConfig.model_validate(_cfg(segmentation={"model": "instanseg"}))


def test_segmentation_target_points_at_the_top_level_field():
    # `target` drives more than the InstanSeg head, so the coordinator injects it from
    # the top level -- a copy inside the segmentation block would be overwritten.
    with pytest.raises(ValidationError, match="move it to the top level"):
        FOVSelectionConfig.model_validate(
            _cfg(segmentation={"model": "instanseg", "path": "m.zip", "target": "cells"})
        )


def test_cellpose_diameters_are_keyed_by_target():
    # CellposeSegmenter._diameter_for looks the diameter up by target, so any other key
    # (a channel name, say) would silently never be read.
    with pytest.raises(ValidationError, match="diameters"):
        FOVSelectionConfig.model_validate(
            _cfg(segmentation={"model": "cellpose", "diameters": {"membrane": 120.0}})
        )
    cfg = FOVSelectionConfig.model_validate(
        _cfg(segmentation={"model": "cellpose", "diameters": {"cells": 120.0, "nuclei": None}})
    )
    assert cfg.segmentation.diameters == {"cells": 120.0, "nuclei": None}


# ---------------------------------------------------------------------------
# selection models
# ---------------------------------------------------------------------------


def test_unknown_model_type_is_rejected():
    with pytest.raises(ValidationError, match="type"):
        FOVSelectionConfig.model_validate(_cfg(model={"type": "random_forest"}))


def test_ranking_requires_top_fov_and_features():
    # Pure ranking: without a quota there is no selection rule at all.
    with pytest.raises(ValidationError, match="top_fov"):
        FOVSelectionConfig.model_validate(
            _cfg(
                model={
                    "type": "ranking_by_defined_range",
                    "features": {"coverage_frac": RANKING_FEATURE},
                }
            )
        )
    with pytest.raises(ValidationError, match="top_fov"):
        FOVSelectionConfig.model_validate(
            _cfg(
                model={
                    "type": "ranking_by_defined_range",
                    "top_fov": 0,
                    "features": {"coverage_frac": RANKING_FEATURE},
                }
            )
        )
    with pytest.raises(ValidationError, match="features"):
        FOVSelectionConfig.model_validate(
            _cfg(model={"type": "ranking_by_defined_range", "top_fov": 1, "features": {}})
        )


def test_threshold_belongs_to_the_classification_tree_only():
    # It is a P(good) cutoff; the ranking model has no per-FOV verdict to cut and the
    # thresholding box is a hard AND, so both would silently ignore it.
    assert (
        FOVSelectionConfig.model_validate(
            _cfg(model={"type": "classification_tree", "path": "t.joblib", "threshold": 0.8})
        ).model.threshold
        == 0.8
    )
    with pytest.raises(ValidationError, match="threshold"):
        FOVSelectionConfig.model_validate(
            _cfg(
                model={
                    "type": "ranking_by_defined_range",
                    "top_fov": 1,
                    "threshold": 0.8,
                    "features": {"coverage_frac": RANKING_FEATURE},
                }
            )
        )


def test_classification_tree_requires_a_path():
    with pytest.raises(ValidationError, match="path"):
        FOVSelectionConfig.model_validate(_cfg(model={"type": "classification_tree"}))


@pytest.mark.parametrize(
    "feature",
    [
        {"shape": "gaussian", "center": 0.5},  # no fwhm
        {"shape": "gaussian", "center": 0.5, "fwhm": 0.0},  # fwhm must be > 0
        {"shape": "lognormal", "center": 1.0, "fold": 1.0},  # fold must be > 1
        {"shape": "lognormal", "center": 0.0, "fold": 2.0},  # center must be > 0
        {"shape": "sigmoid", "midpoint": 1.0, "width": 1.0},  # no direction
        {"shape": "sigmoid", "midpoint": 1.0, "width": 1.0, "direction": "up"},
        {"center": 0.5, "fwhm": 0.2},  # shape is required, never inferred
        {"shape": "linear", "center": 0.5},  # unknown shape
    ],
)
def test_invalid_ranking_curves_are_rejected(feature):
    with pytest.raises(ValidationError):
        FOVSelectionConfig.model_validate(
            _cfg(
                model={
                    "type": "ranking_by_defined_range",
                    "top_fov": 1,
                    "features": {"coverage_frac": feature},
                }
            )
        )


def test_thresholding_accepts_a_bare_pair_or_an_explicit_range():
    cfg = FOVSelectionConfig.model_validate(
        _cfg(
            model={
                "type": "classification_by_thresholding",
                "features": {
                    "coverage_frac": [0.1, 0.9],
                    "com_offset_norm": {"range": [0.0, 0.4]},
                },
            }
        )
    )
    assert cfg.model.features["coverage_frac"].range == (0.1, 0.9)
    assert cfg.model.features["com_offset_norm"].range == (0.0, 0.4)

    with pytest.raises(ValidationError, match="lo <= hi"):
        FOVSelectionConfig.model_validate(
            _cfg(
                model={
                    "type": "classification_by_thresholding",
                    "features": {"coverage_frac": [0.9, 0.1]},
                }
            )
        )


# ---------------------------------------------------------------------------
# block(): what gets written back into the pre-scan metadata
# ---------------------------------------------------------------------------


def test_block_is_json_safe_and_round_trips():
    cfg = FOVSelectionConfig.model_validate(
        _cfg(
            segmentation={"model": "instanseg", "path": "C:/models/instanseg.zip"},
            prescan_mda={
                "channels": [{"config": "BF", "group": "Channel"}],
                "stage_positions": [{"x": 0, "y": 0}],
                "time_plan": {"loops": 1, "interval": 0},
            },
        )
    )
    block = cfg.block(exclude={"prescan_mda"})

    import json

    json.dumps(block)  # must survive a YAML/JSON round-trip and the worker pickle
    assert "prescan_mda" not in block  # no redundant nested copy of itself
    assert isinstance(block["segmentation"]["path"], str)
    # unset backend knobs are absent, not None: an explicit None would shadow the
    # backend's own default (e.g. CellposeSegmenter.FLOW_THRESHOLD).
    assert "model_pixel_size_um" not in block["segmentation"]
    assert FOVSelectionConfig.model_validate(block).segmentation == cfg.segmentation


def test_block_keeps_none_inside_a_mapping_value():
    # `diameters: {nuclei: null}` means "auto-scale"; exclude_none must not eat it.
    cfg = FOVSelectionConfig.model_validate(
        _cfg(segmentation={"model": "cellpose", "diameters": {"nuclei": None}})
    )
    assert cfg.block()["segmentation"]["diameters"] == {"nuclei": None}


# ---------------------------------------------------------------------------
# drift guards: the Literals here vs the consumers that implement them
# ---------------------------------------------------------------------------


def _literal(model, field):
    return set(get_args(model.model_fields[field].annotation))


def test_model_types_match_fov_model():
    from shrimpy.fov_selection.fov_model import MODEL_TYPES

    declared = set()
    for settings in (
        RankingModelSettings,
        ThresholdingModelSettings,
        ClassificationTreeModelSettings,
    ):
        declared |= _literal(settings, "type")
    assert declared == set(MODEL_TYPES)


def test_aggregations_and_shapes_match_the_desirability_model():
    from shrimpy.fov_selection.fov_model import DesirabilityModel

    assert _literal(RankingModelSettings, "aggregation") == set(DesirabilityModel.AGGREGATIONS)
    assert (
        RankingModelSettings.model_fields["aggregation"].default
        == DesirabilityModel.DEFAULT_AGGREGATION
    )
    shapes = set()
    for feature in (GaussianFeature, LognormalFeature, SigmoidFeature):
        shapes |= _literal(feature, "shape")
    assert shapes == set(DesirabilityModel.SHAPES)


def test_segmentation_backends_match_the_builder():
    from shrimpy.fov_selection.segmentation import INSTANSEG_TARGETS, SEGMENTATION_BACKENDS

    declared = set()
    for settings in (CellposeSettings, InstansegSettings, OtsuSettings):
        declared |= _literal(settings, "model")
    assert declared == set(SEGMENTATION_BACKENDS)
    # `target` selects the InstanSeg head, so the two name the same objects
    assert set(TARGETS) == set(INSTANSEG_TARGETS)


def test_pipeline_steps_cover_the_shared_reconstruction_steps():
    from shrimpy.preprocessing import RECON_STEPS

    assert set(RECON_STEPS) <= set(PIPELINE_STEPS)
