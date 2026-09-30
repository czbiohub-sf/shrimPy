"""napari viewer for shrimPy acquisitions, live or finished.

The viewer reads the acquisition's OME-Zarr store directly
(:class:`~shrimpy.viewer.store.AcquisitionStore`), so browsing a run in progress and
browsing a finished dataset are the same thing -- the whole acquisition is available in
both cases, not just the most recent frames. During an acquisition
:class:`~shrimpy.viewer.live.LiveViewer` runs it in a separate process, so a viewer
crash or hang can never disrupt the run; ``shrimpy view`` runs the same window on a
store that is already on disk.
"""

from __future__ import annotations

from shrimpy.viewer.live import LiveViewer
from shrimpy.viewer.store import AcquisitionStore

__all__ = ["AcquisitionStore", "LiveViewer"]
