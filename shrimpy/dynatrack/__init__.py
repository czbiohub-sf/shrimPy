"""DynaTrack — sample tracking and dynamic stage-position updating.

Microscope-agnostic: an acquisition engine builds a :class:`DynaTrack`
coordinator from its ``dynatrack`` metadata section and interacts with that
object only. The position-update infrastructure (:class:`PositionStore`,
:class:`PositionUpdater`, :class:`PositionUpdateManager`) is an
implementation detail of this package, exposed for tests and for writing
custom trackers.
"""

from shrimpy.dynatrack.manager import DynaTrack
from shrimpy.dynatrack.position_update import (
    PositionCoordinates,
    PositionStore,
    PositionUpdateManager,
    PositionUpdater,
)
from shrimpy.dynatrack.tracking import DynaTrackConfig, DynaTrackUpdater

__all__ = [
    "DynaTrack",
    "DynaTrackConfig",
    "DynaTrackUpdater",
    "PositionCoordinates",
    "PositionStore",
    "PositionUpdateManager",
    "PositionUpdater",
]
