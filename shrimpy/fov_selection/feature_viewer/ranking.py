"""Shared feature-name helper for the feature viewer.

Historically this module also held a tunable ``RankingProfile`` subsystem (five
interchangeable ranking models trained from labeled FOVs). That machinery was never wired
into the viewer and has been removed in the cleanup; only the feature-name helper the Rank
tab actually calls remains here. The live per-feature desirability curves live in
:mod:`shrimpy.fov_selection.fov_model` (``DesirabilityModel``), which the Rank tab builds
directly from the config schema.
"""

from __future__ import annotations


def _feature_suffix(col: str) -> str:
    """'nuclei_vs_max__objects_per_10um2' -> 'objects_per_10um2' (the token after '__')."""
    return col.split("__")[-1]
