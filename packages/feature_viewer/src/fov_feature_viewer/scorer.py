"""The scoring interface the viewer's Rank and Score-map tabs run on.

The viewer does no model math. A :class:`Scorer` supplies the curve schema and every
computation, and the viewer only draws and edits what it hands back. Each feature's
settings travel as a *spec*: a plain dict owned by the scorer (for shrimpy's ranking model,
the exact ``fov_selection.model.features`` entry). The viewer reads only ``spec["shape"]``,
``spec.get("direction")``, ``spec.get("weight", 1.0)`` and the params named by
:meth:`Scorer.params`, and treats everything else as opaque.

A scorer is found through the ``fov_feature_viewer.scorers`` entry-point group (shrimpy
registers its ranking model there), or named explicitly with ``--scorer module:attr``.
Without one the viewer still explores and labels data; the Rank and Score-map tabs are
simply not shown.
"""

from __future__ import annotations

import importlib

from collections.abc import Sequence
from importlib.metadata import entry_points
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

ENTRY_POINT_GROUP = "fov_feature_viewer.scorers"


@runtime_checkable
class Scorer(Protocol):
    """A tunable per-feature scoring model.

    Every method that returns a spec returns a *valid* one, and every method that takes a
    spec raises ``ValueError`` if it is not.
    """

    #: Curve shapes a feature can use, and the one a fresh feature starts with.
    shapes: Sequence[str]
    default_shape: str
    #: Ways the per-feature curves combine into one score, and the default one.
    aggregations: Sequence[str]
    default_aggregation: str

    def directions(self, shape: str) -> Sequence[str]:
        """Directions ``shape`` allows; the first is the default. One entry = fixed."""

    def params(self, shape: str) -> Sequence[tuple[str, float | None]]:
        """``(name, minimum)`` of each editable param of ``shape``; None = unbounded."""

    def seed(self, values: np.ndarray) -> dict:
        """A default spec fitted to one feature's measured values (may contain NaN)."""

    def reshape(
        self, spec: dict, shape: str | None = None, direction: str | None = None
    ) -> dict:
        """``spec`` converted to ``shape`` / ``direction`` (None = keep), or normalized."""

    def handles(self, spec: dict) -> Sequence[tuple[str, float]]:
        """``(name, x)`` of each point of the curve that can be dragged along x."""

    def drag(self, spec: dict, handle: str, x: float) -> dict:
        """``spec`` with the named handle moved to ``x``."""

    def curve(self, spec: dict, x: np.ndarray) -> np.ndarray:
        """Desirability in ``[0, 1]`` of ``spec`` at each value of ``x``."""

    def score(
        self, df: pd.DataFrame, features: dict[str, dict], aggregation: str
    ) -> np.ndarray:
        """One score per row of ``df`` from ``features`` (column name -> spec)."""

    def read_profile(self, cfg) -> dict[str, dict]:
        """Feature specs (name -> spec) from a parsed profile file or config block."""


def available_scorers() -> dict[str, str]:
    """Registered scorers: entry-point name -> ``module:attr``."""
    return {ep.name: ep.value for ep in entry_points(group=ENTRY_POINT_GROUP)}


def load_scorer(name: str | None = None) -> Scorer | None:
    """Build a scorer from ``name`` (``module:attr`` or a registered entry-point name).

    With no ``name``, the first registered scorer (by name) is used, or ``None`` if there
    is none. ``attr`` may be a class or factory (called with no arguments) or an instance.
    Raises ``ValueError`` if ``name`` does not resolve or does not satisfy :class:`Scorer`.
    """
    registered = available_scorers()
    if name is None:
        if not registered:
            return None
        name = sorted(registered)[0]
    target = registered.get(name, name)
    module_name, sep, attr = target.partition(":")
    if not sep:
        raise ValueError(
            f"unknown scorer {name!r}: give module:attr or one of {sorted(registered)}"
        )
    obj = getattr(importlib.import_module(module_name), attr)
    # A class can itself pass the structural check (its attributes and methods exist), so
    # classes are always instantiated; other callables only when they are not a scorer.
    if isinstance(obj, type) or (callable(obj) and not isinstance(obj, Scorer)):
        obj = obj()
    scorer = obj
    if not isinstance(scorer, Scorer):
        raise ValueError(f"{target} does not implement the viewer's Scorer interface")
    return scorer
