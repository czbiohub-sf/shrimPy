"""Online FOV selection — the runtime decision made during acquisition.

This is the seam between the acquisition engine and the FOV-selection science.
The engine runs a pre-scan over all positions, hands the position list (and,
later, the pre-scan data) to ``select_good_fovs``, and then runs the timelapse
only over the returned "good" positions.

M1 (this file): a *dummy* selector that keeps a subset of positions so the
two-phase adaptive acquisition can be exercised end to end without any
reconstruction/segmentation/model. In M2 this is replaced by the real decision
(reconstruct -> project -> segment -> features -> trained model), likely wrapped
in a ``FovSelection`` manager class mirroring ``shrimpy.dynatrack.manager``.
The input/output contract stays the same: positions in -> good positions out.
"""

from __future__ import annotations

import logging

from collections.abc import Sequence
from typing import TypeVar

logger = logging.getLogger(__name__)

# A position is a useq ``Position``; kept generic so this stub stays dependency-free.
P = TypeVar("P")


def select_good_fovs(positions: Sequence[P]) -> list[P]:
    """Return the "good" subset of pre-scan positions.

    Parameters
    ----------
    positions : sequence of useq Position
        The positions acquired during the pre-scan (in acquisition order).

    Returns
    -------
    list of useq Position
        The subset to acquire in the timelapse.

    Notes
    -----
    M1 stub: keep every other position. The real M2 implementation will also
    receive the pre-scan data path and run the full decision pipeline per FOV.
    """
    good = [p for i, p in enumerate(positions) if i % 2 == 0]
    logger.info(
        "FOV selection (M1 dummy): kept %d/%d positions: %s",
        len(good),
        len(positions),
        [getattr(p, "name", None) for p in good],
    )
    return good
