"""The Rank tab runs on any Scorer, not just shrimpy's.

A toy scorer -- one linear "ramp" shape between two draggable points -- drives the real
widgets headless, so the viewer is shown to need nothing from a particular model.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# qtpy imports fine without a binding and then raises QtBindingsNotFoundError, which is
# not an ImportError, so importorskip("qtpy") is not enough.
try:
    from qtpy import QtWidgets
except Exception:  # pragma: no cover - depends on the environment
    pytest.skip("no Qt bindings", allow_module_level=True)
pytest.importorskip("matplotlib")
pytest.importorskip("fov_feature_viewer")

from fov_feature_viewer.app import FeatureViewer  # noqa: E402
from fov_feature_viewer.scorer import Scorer, load_scorer  # noqa: E402


class RampScorer:
    """Desirability rises linearly from 0 at ``start`` to 1 at ``end``; scores average."""

    shapes = ("ramp",)
    default_shape = "ramp"
    aggregations = ("mean",)
    default_aggregation = "mean"

    def directions(self, shape):
        return ("up",)

    def params(self, shape):
        return (("start", None), ("end", None))

    def seed(self, values):
        v = np.asarray(values, float)
        v = v[~np.isnan(v)]
        return {"shape": "ramp", "start": float(v.min()), "end": float(v.max()), "weight": 1.0}

    def reshape(self, spec, shape=None, direction=None):
        if not spec["end"] > spec["start"]:
            raise ValueError("end must exceed start")
        return {k: spec[k] for k in ("shape", "start", "end", "weight")}

    def handles(self, spec):
        return [("start", spec["start"]), ("end", spec["end"])]

    def drag(self, spec, handle, x):
        return self.reshape({**spec, handle: float(x)})

    def curve(self, spec, x):
        d = (np.asarray(x, float) - spec["start"]) / (spec["end"] - spec["start"])
        return np.nan_to_num(np.clip(d, 0.0, 1.0))

    def score(self, df, features, aggregation):
        return np.mean([self.curve(s, df[f].to_numpy(float)) for f, s in features.items()], 0)

    def read_profile(self, cfg):
        return {f: self.reshape(s) for f, s in cfg.items()}


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def viewer(qapp, tmp_path):
    df = pd.DataFrame({"filename": ["a", "b", "c"], "coverage_frac": [0.0, 0.5, 1.0]})
    csv = tmp_path / "fov_summary.csv"
    df.to_csv(csv, index=False)
    win = FeatureViewer(RampScorer())
    win._load_files([str(csv)])
    return win


def test_toy_scorer_satisfies_the_protocol():
    assert isinstance(RampScorer(), Scorer)
    assert isinstance(load_scorer(f"{__name__}:RampScorer"), RampScorer)


def test_rank_tab_scores_with_any_scorer(viewer):
    np.testing.assert_allclose(viewer.df["score"].to_numpy(float), [0.0, 0.5, 1.0])
    row = list(viewer.rank_ranges).index("coverage_frac")
    assert not viewer.rank_table.cellWidget(row, 1).isEnabled()  # one direction = fixed


def test_rank_table_edits_go_through_the_scorer(viewer):
    row = list(viewer.rank_ranges).index("coverage_frac")
    start, end = (viewer.rank_table.cellWidget(row, c) for c in viewer._rank_param_cols)
    end.setValue(0.5)
    viewer._rerank()
    np.testing.assert_allclose(viewer.df["score"].to_numpy(float), [0.0, 1.0, 1.0])
    end.setValue(-1.0)  # invalid for this scorer: the previous spec is kept
    viewer._read_rank_table()
    assert viewer.rank_ranges["coverage_frac"]["spec"]["end"] == 0.5
