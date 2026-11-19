"""Drawn ROI shapes and the ablation patterns that fill them.

Vendored from coPylot ``copylot/assemblies/photom/utils/pattern_tracing.py``
(main @ 0bcc3a3), BSD-3-Clause; see the package LICENSE. Ported from PyQt5 to
qtpy.

Changes from upstream: module logger instead of the root logger and ``print``;
``_pattern_spiral(num_points=None)`` uses the maximum number of points that fit
(upstream raised ``TypeError`` comparing ``None``); points are passed to
``QPoint`` as ``int`` (PyQt6 rejects floats).
"""

from __future__ import annotations

import logging

from collections import deque

import numpy as np

from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QPolygon

logger = logging.getLogger(__name__)


class ShapeTrace(QPolygon):
    def __init__(self, border_points: list[QPoint]) -> None:
        """A drawn shape.

        Parameters
        ----------
        border_points : list[QPoint]
            The shape's border, in window pixels.
        """
        super().__init__(border_points)
        self.border_points = border_points
        self.pattern_style = None
        self.ablation_points = []  # points to be ablated
        self.pattern_points = []  # points plotted in gray to visualize the pattern

        self.gap = 5
        self.points_per_cycle = 80

    def _pattern_bidirectional(
        self, vertical_spacing: int, horizontal_spacing: int, num_points: int | None = None
    ) -> None:
        """Fill the shape with a grid of points, breadth-first from its center.

        Parameters
        ----------
        vertical_spacing, horizontal_spacing : int
            Pixels between points; at least :attr:`gap`.
        num_points : int, optional
            Maximum number of points.
        """
        if horizontal_spacing < self.gap:
            logger.warning(
                f"horizontal spacing is too small for the gap. defaulting to gap ({self.gap})."
            )
            horizontal_spacing = self.gap
        if vertical_spacing < self.gap:
            logger.warning(
                f"vertical spacing is too small for the gap. defaulting to gap ({self.gap})."
            )
            vertical_spacing = self.gap
        self.border_style = "Bidirectional"
        rect = self.boundingRect()
        center_x = round((rect.left() + rect.right()) / 2)
        center_y = round((rect.top() + rect.bottom()) / 2)

        visited = {(center_x, center_y)}
        queue = deque([(center_x, center_y)])
        directions = [
            (0, -vertical_spacing),  # up
            (horizontal_spacing, 0),  # right
            (0, vertical_spacing),  # down
            (-horizontal_spacing, 0),  # left
        ]

        while queue and (num_points is None or len(self.ablation_points) < num_points):
            curr_x, curr_y = queue.popleft()
            self.ablation_points.append((curr_x, curr_y))

            for dx, dy in directions:
                new_x, new_y = curr_x + dx, curr_y + dy
                if (
                    self.containsPoint(QPoint(new_x, new_y), Qt.FillRule.OddEvenFill)
                    and (new_x, new_y) not in visited
                ):
                    queue.append((new_x, new_y))
                    visited.add((new_x, new_y))
        if not self.ablation_points:
            self.ablation_points.append((center_x, center_y))
            logger.warning("spacing configuration is too large for the shape.")

    def _pattern_spiral(self, num_points: int | None = None) -> None:
        """Add a spiral of ablation points inside the shape.

        Parameters
        ----------
        num_points : int, optional
            Number of points; defaults to (and is capped at) the most that fit
            with :attr:`gap` between them.
        """
        rect = self.boundingRect()
        width = rect.width()
        height = rect.height()

        max_radius = min(width, height) / 2

        # at least 4 turns, more until turns are at least `gap` apart
        num_turns = 4
        distance_bt_turns = max_radius / num_turns
        while distance_bt_turns < self.gap:
            num_turns += 1
            distance_bt_turns = max_radius / num_turns

        theta = np.linspace(0, num_turns * np.pi, 1000)
        radius = theta

        # polar -> cartesian, normalized to the bounding box
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        x = (x - x.min()) / (x.max() - x.min()) * width
        y = (y - y.min()) / (y.max() - y.min()) * height

        arc_length = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        arc_length = np.insert(arc_length, 0, 0)

        total_length = arc_length[-1]
        max_num_points = int(total_length // self.gap)

        if num_points is None:
            num_points = max_num_points
        elif num_points > max_num_points:
            logger.warning(
                f"num_points: {num_points} is greater than max_num_points: "
                f"{max_num_points}, defaulting to max_num_points."
            )
            num_points = max_num_points

        # equidistant points along the arc, at least `gap` apart
        target_lengths = np.linspace(0, total_length, num_points)
        indices = set(np.searchsorted(arc_length, target_lengths))

        for idx in range(len(x)):
            plot_x = int(x[idx]) + rect.left()
            plot_y = int(y[idx]) + rect.top()
            if self.containsPoint(QPoint(plot_x, plot_y), Qt.FillRule.OddEvenFill):
                if idx in indices:
                    self.ablation_points.append((plot_x, plot_y))
                else:
                    self.pattern_points.append((plot_x, plot_y))
