"""Out-of-process napari viewer for live shrimpy acquisitions.

The viewer runs in a separate process and is fed best-effort over shared memory, so a
viewer crash or hang can never disrupt an ongoing acquisition. See
:class:`~shrimpy.viewer.feeder.ViewerFeeder`.
"""

from __future__ import annotations

from shrimpy.viewer.feeder import ViewerFeeder

__all__ = ["ViewerFeeder"]
