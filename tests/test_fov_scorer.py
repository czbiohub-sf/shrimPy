"""DesirabilityScorer: the ranking model as the feature viewer's Scorer interface.

The viewer tunes a ranking profile only through this object, so these tests pin that
every spec it hands out is a valid ``fov_selection.model.features`` entry and that its
scores are exactly the acquisition's (DesirabilityModel.predict).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shrimpy.fov_selection.fov_model import DesirabilityScorer, build_fov_model

GAUSS = {"shape": "gaussian", "center": 0.5, "fwhm": 0.2, "weight": 1.0}
LOGN = {"shape": "lognormal", "center": 4.0, "fold": 2.0, "weight": 2.0}
SIG = {"shape": "sigmoid", "midpoint": 10.0, "width": 4.0, "direction": "lower", "weight": 1.0}


@pytest.fixture
def scorer():
    return DesirabilityScorer()


def _approx(a: dict, b: dict):
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == pytest.approx(b[k]) if isinstance(a[k], float) else a[k] == b[k]


def test_schema(scorer):
    assert scorer.default_shape in scorer.shapes
    assert scorer.default_aggregation in scorer.aggregations
    assert scorer.directions("gaussian") == ("target",)
    assert scorer.directions("sigmoid") == ("higher", "lower")
    assert [n for n, _ in scorer.params("lognormal")] == ["center", "fold"]


@pytest.mark.parametrize("aggregation", DesirabilityScorer.aggregations)
def test_score_matches_the_acquisition_model(scorer, aggregation):
    df = pd.DataFrame(
        {"a": [0.5, 0.2, np.nan, 0.9], "b": [4.0, 1.0, 8.0, 0.0], "c": [3, 9, 12, 20]}
    )
    features = {"a": GAUSS, "b": LOGN, "c": SIG}
    expected = build_fov_model(
        {
            "type": "ranking_by_defined_range",
            "top_fov": 1,
            "aggregation": aggregation,
            "features": features,
        }
    ).predict(df)[0]
    np.testing.assert_allclose(scorer.score(df, features, aggregation), expected)


def test_curve_peaks_at_center_and_is_half_at_fwhm(scorer):
    d = scorer.curve(GAUSS, np.array([0.5, 0.4, 0.6, np.nan]))
    np.testing.assert_allclose(d, [1.0, 0.5, 0.5, 0.0])


@pytest.mark.parametrize("spec", [GAUSS, LOGN, SIG])
def test_reshape_to_own_shape_is_identity(scorer, spec):
    _approx(scorer.reshape(spec), spec)


def test_reshape_keeps_the_band_and_coerces_direction(scorer):
    as_sigmoid = scorer.reshape(GAUSS, "sigmoid")
    assert as_sigmoid["direction"] == "higher"  # bells have no higher/lower
    assert as_sigmoid["midpoint"] == pytest.approx(GAUSS["center"])
    lo, hi = (x for _, x in scorer.handles(GAUSS))
    assert [x for _, x in scorer.handles(as_sigmoid)] == pytest.approx([lo, hi])
    # back to a bell: 'target' is its only direction, so no direction key
    assert "direction" not in scorer.reshape(as_sigmoid, "gaussian")
    assert scorer.reshape(SIG, direction="higher")["direction"] == "higher"


def test_reshape_rejects_invalid_params(scorer):
    with pytest.raises(ValueError):
        scorer.reshape({**LOGN, "fold": 1.0})
    with pytest.raises(ValueError):
        scorer.reshape({**SIG, "width": 0.0})
    with pytest.raises(ValueError):
        scorer.reshape({"shape": "gaussian", "center": 1.0})  # no fwhm


def test_drag_moves_one_edge_and_never_crosses(scorer):
    (_, lo), (_, hi) = scorer.handles(GAUSS)
    moved = scorer.drag(GAUSS, "hi", hi + 0.1)
    assert [x for _, x in scorer.handles(moved)] == pytest.approx([lo, hi + 0.1])
    crossed = scorer.drag(GAUSS, "lo", hi + 1.0)
    new_lo, new_hi = (x for _, x in scorer.handles(crossed))
    assert new_lo < new_hi == pytest.approx(hi)
    scorer.reshape(crossed)  # still a valid spec


def test_seed_centers_on_the_data(scorer):
    spec = scorer.seed(np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan]))
    assert spec["shape"] == scorer.default_shape
    assert spec["center"] == pytest.approx(3.0)
    for values in ([], [np.nan], [7.0, 7.0]):
        scorer.reshape(scorer.seed(np.array(values)))  # degenerate data -> still valid


def test_read_profile_accepts_model_dict_and_upgrades_legacy(scorer):
    model_cfg = {"type": "ranking_by_defined_range", "top_fov": 2, "features": {"a": GAUSS}}
    _approx(scorer.read_profile(model_cfg)["a"], GAUSS)
    legacy = scorer.read_profile({"a": {"shape": "gaussian", "range": [0.4, 0.6]}})["a"]
    assert legacy["center"] == pytest.approx(0.5)
    assert legacy["weight"] == 1.0
    with pytest.raises(ValueError):
        scorer.read_profile({"a": {"shape": "linear", "range": [0, 1]}})
    with pytest.raises(ValueError):
        scorer.read_profile(["not", "a", "mapping"])
