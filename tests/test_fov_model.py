"""FOV-selection model behavior.

Pins the model contract after the ranking/classification split: the ranking model REQUIRES
``top_fov`` and has no per-FOV good/bad verdict (``predict`` returns ``good=None``); the
classification models return a per-FOV ``good`` and do not need ``top_fov``. ``threshold`` is a
classification-only knob.
"""

from __future__ import annotations

import pandas as pd
import pytest

from shrimpy.fov_selection.fov_model import build_fov_model

_RANKING = {
    "type": "ranking_by_defined_range",
    "features": {"coverage_frac": {"shape": "gaussian", "center": 0.5, "fwhm": 0.2}},
    "top_fov": 2,
}


def test_ranking_requires_top_fov():
    without = {k: v for k, v in _RANKING.items() if k != "top_fov"}
    with pytest.raises(ValueError, match="top_fov"):
        build_fov_model(without)
    with pytest.raises(ValueError, match="top_fov"):
        build_fov_model({**_RANKING, "top_fov": 0})  # must be a positive int


def test_ranking_predict_has_no_good_verdict_and_ignores_threshold():
    model = build_fov_model(_RANKING)
    df = pd.DataFrame({"coverage_frac": [0.5, 0.1]})
    # threshold is irrelevant to a ranking model; good is None regardless of its value.
    proba, good = model.predict(df, threshold=0.99)
    assert good is None
    assert len(proba) == 2 and proba[0] > proba[1]  # the on-center FOV ranks higher


def test_thresholding_returns_good_box_and_needs_no_top_fov():
    model = build_fov_model(
        {
            "type": "classification_by_thresholding",
            "features": {"coverage_frac": {"range": [0.2, 0.8]}},
        }
    )
    df = pd.DataFrame({"coverage_frac": [0.5, 0.9]})
    # the hard [lo, hi] box ignores threshold; good is the in-range verdict.
    proba, good = model.predict(df, threshold=0.0)
    assert good == [True, False]
    assert proba[0] == 1.0 and proba[1] == 0.0
