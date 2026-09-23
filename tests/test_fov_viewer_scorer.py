"""The feature viewer's Rank and Score-map tabs, run headless on shrimpy's scorer.

The viewer does no model math of its own; these tests drive its real widgets through a
tuning session and check that everything it computes or saves comes from the scorer.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# qtpy imports fine without a binding and then raises QtBindingsNotFoundError (not an
# ImportError), so importorskip("qtpy") is not enough.
try:
    from qtpy import QtWidgets
except Exception:  # pragma: no cover - depends on which extras are installed
    pytest.skip("no Qt bindings (install the `fov` extra)", allow_module_level=True)
pytest.importorskip("matplotlib")

from shrimpy.fov_selection.feature_viewer.app import FeatureViewer  # noqa: E402
from shrimpy.fov_selection.feature_viewer.scorer import Scorer, load_scorer  # noqa: E402
from shrimpy.fov_selection.fov_model import DesirabilityScorer  # noqa: E402

SCORER_PATH = "shrimpy.fov_selection.fov_model:DesirabilityScorer"


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def csv(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "filename": [f"f{i}" for i in range(20)],
            "coverage_frac": rng.uniform(0.1, 0.9, 20),
            "nn_um_mean": rng.uniform(5.0, 50.0, 20),
            "goodness": [1.0, 0.0, -1.0, np.nan] * 5,
        }
    )
    path = tmp_path / "fov_summary.csv"
    df.to_csv(path, index=False)
    return path


def _viewer(qapp, csv, scorer):
    win = FeatureViewer(scorer)
    win._load_files([str(csv)])
    return win


def test_shrimpy_scorer_satisfies_the_viewer_interface():
    assert isinstance(DesirabilityScorer(), Scorer)
    assert isinstance(load_scorer(SCORER_PATH), DesirabilityScorer)
    with pytest.raises(ValueError, match="Scorer"):
        load_scorer("builtins:dict")  # a class, but not a scorer
    with pytest.raises(ValueError, match="unknown scorer"):
        load_scorer("no-such-scorer")


def test_no_scorer_hides_the_rank_and_map_tabs(qapp, csv):
    win = _viewer(qapp, csv, None)
    tabs = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert tabs == ["Analysis", "Label"]
    assert win.df is not None and "score" not in win.df.columns


def test_rank_scores_are_the_scorers(qapp, csv):
    scorer = DesirabilityScorer()
    win = _viewer(qapp, csv, scorer)
    assert [win.tabs.tabText(i) for i in range(win.tabs.count())][2:] == ["Rank", "Score map"]
    features = win._rank_features()
    assert list(features) == ["coverage_frac"]  # the default-checked feature
    expected = scorer.score(win.df, features, scorer.default_aggregation)
    np.testing.assert_allclose(win.df["score"].to_numpy(float), expected)


def test_shape_change_drag_and_profile_round_trip(qapp, csv):
    scorer = DesirabilityScorer()
    win = _viewer(qapp, csv, scorer)
    row = list(win.rank_ranges).index("coverage_frac")

    # switch the curve to a sigmoid through the table's shape combo
    win.rank_table.cellWidget(row, 2).setCurrentText("sigmoid")
    spec = win.rank_ranges["coverage_frac"]["spec"]
    assert spec["shape"] == "sigmoid" and spec["direction"] in scorer.directions("sigmoid")
    assert win.rank_table.cellWidget(row, 1).isEnabled()  # sigmoid has two directions

    # drag the upper handle; the table's param spins follow the new spec
    moved = scorer.drag(spec, "hi", dict(scorer.handles(spec))["hi"] + 0.05)
    win.rank_ranges["coverage_frac"]["spec"] = moved
    win._rank_sync_row("coverage_frac")
    win._read_rank_table()
    assert win.rank_ranges["coverage_frac"]["spec"]["width"] == pytest.approx(moved["width"])

    # the saved profile is what read_profile reads back, and loading it keeps the ranking
    win._rerank()
    saved = win._rank_features()
    scores = win.df["score"].to_numpy(float)
    win._apply_rank_profile_cfg(saved, source="test")
    assert win.rank_ranges["coverage_frac"]["enabled"]
    assert not win.rank_ranges["nn_um_mean"]["enabled"]
    np.testing.assert_allclose(win.df["score"].to_numpy(float), scores)


def test_score_map_scores_the_pair_with_the_scorer(qapp, csv):
    scorer = DesirabilityScorer()
    win = _viewer(qapp, csv, scorer)
    for state in win.rank_ranges.values():
        state["enabled"] = True
    win._rank_populate_table()
    win._refresh_score_map_controls()
    win._update_score_map()
    assert win.map_status.text().startswith("score = f(")
    pair = {f: win.rank_ranges[f]["spec"] for f in ("coverage_frac", "nn_um_mean")}
    np.testing.assert_allclose(
        win._map_pair_score(win.df, "coverage_frac", "nn_um_mean"),
        scorer.score(win.df, pair, win.map_agg_combo.currentText()),
    )
