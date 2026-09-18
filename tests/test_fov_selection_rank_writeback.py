"""Rank-tab write-back: `proba` / `rank` are persisted to each row's source CSV.

The Rank tab's "Write proba/rank to CSV" button drives ``_persist_scores``, which joins the
in-memory scores back to the on-disk feature CSV by FOV identity (``filename``). This pins
that the join is order-independent and touches only the ``proba`` / ``rank`` columns, so a
calibration ``fov_summary.csv`` ends up carrying the ranking tuned in the viewer.
"""

from __future__ import annotations

import types

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("matplotlib")


def _stub(df):
    """A minimal stand-in for the FeatureViewer, exposing only what _persist_scores reads."""
    return types.SimpleNamespace(
        df=df, rank_status=types.SimpleNamespace(setText=lambda *_: None)
    )


def test_persist_scores_writes_proba_and_rank_by_filename(tmp_path):
    import pandas as pd

    from shrimpy.fov_selection.feature_viewer.rank_tab import RankTabMixin

    # a source CSV as the calibration pre-scan writes it (filename + features), with rows in
    # a DIFFERENT order than the in-memory table to exercise the identity join.
    src = tmp_path / "fov_summary.csv"
    pd.DataFrame({"filename": ["f1", "f0"], "coverage_frac": [0.4, 0.2]}).to_csv(
        src, index=False
    )

    df = pd.DataFrame(
        {
            "filename": ["f0", "f1"],
            "coverage_frac": [0.2, 0.4],
            "proba": [0.3, 0.9],
            "rank": [2.0, 1.0],
            "__src": [str(src), str(src)],
        }
    )
    saved, matched = RankTabMixin._persist_scores(_stub(df))
    assert (saved, matched) == (1, 2)

    disk = pd.read_csv(src).set_index("filename")
    # scores land on the right rows despite the on-disk order, existing columns untouched
    assert disk.loc["f0", "proba"] == 0.3 and disk.loc["f0", "rank"] == 2.0
    assert disk.loc["f1", "proba"] == 0.9 and disk.loc["f1", "rank"] == 1.0
    assert disk.loc["f0", "coverage_frac"] == 0.2 and disk.loc["f1", "coverage_frac"] == 0.4
