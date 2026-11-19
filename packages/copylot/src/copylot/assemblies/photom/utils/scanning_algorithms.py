"""Calibration-grid helpers.

Vendored from coPylot ``copylot/assemblies/photom/utils/scanning_algorithms.py``
(main @ 0bcc3a3), BSD-3-Clause; see the package LICENSE.

Only ``calculate_rectangle_corners`` and ``generate_grid_points`` (used by
:class:`~copylot.assemblies.photom.photom.PhotomAssembly`) are vendored; the
``ScanAlgorithm`` pattern generator, used only by coPylot's photom GUI, is not.
``generate_grid_points`` drops upstream's unused ``flip_xy`` argument.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from numpy.typing import ArrayLike


def calculate_rectangle_corners(
    window_size: tuple[float, float], center: Sequence[float] = (0.0, 0.0)
) -> list[list[float]]:
    """Corners of a ``(width, height)`` rectangle around ``center``, counter-clockwise."""
    x0y0 = [
        -window_size[0] / 2 + center[0],
        -window_size[1] / 2 + center[1],
    ]
    x1y0 = [x0y0[0] + window_size[0], x0y0[1]]
    x1y1 = [x0y0[0] + window_size[0], x0y0[1] + window_size[1]]
    x0y1 = [x0y0[0], x0y0[1] + window_size[1]]
    return [x0y0, x1y0, x1y1, x0y1]


def generate_grid_points(rectangle_size: ArrayLike, n_points: int = 5) -> np.ndarray:
    """An ``n_points x n_points`` grid of (x, y) points spanning a rectangle.

    Parameters
    ----------
    rectangle_size : array-like
        ``[[start_x, start_y], [end_x, end_y]]``.
    n_points : int
        Points per row and per column.

    Examples
    --------
    >>> generate_grid_points([[-1, -1], [1, 1]], 3)[:4]
    array([[-1., -1.],
           [ 0., -1.],
           [ 1., -1.],
           [-1.,  0.]], dtype=float32)
    """
    start_x, start_y = rectangle_size[0]
    end_x, end_y = rectangle_size[1]
    interval_x = (end_x - start_x) / (n_points - 1)
    interval_y = (end_y - start_y) / (n_points - 1)

    grid_points = np.zeros((n_points * n_points, 2), dtype=np.float32)
    for i in range(n_points):
        for j in range(n_points):
            grid_points[i * n_points + j] = [
                start_x + j * interval_x,
                start_y + i * interval_y,
            ]
    return grid_points
