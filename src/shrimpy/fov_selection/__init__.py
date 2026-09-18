"""FOV selection -- online, streaming selection of "good" fields of view.

Microscope-agnostic: an acquisition engine builds a :class:`FovSelection`
coordinator from its ``fov_selection`` metadata section and interacts with that
object only. The per-FOV decision pipeline (reconstruct -> project -> segment ->
features -> tree) and the worker subprocess are implementation details of this
package.
"""

from shrimpy.fov_selection.manager import FovSelection

__all__ = ["FovSelection"]
