"""DynaTrack — neuromast tracking for zebrafish embryo growth compensation.

Tracks neuromast motion across timepoints by comparing acquired z-stacks
to reference images and computing X/Y/Z shifts. Plugs into the position
update infrastructure via the PositionUpdater interface.
"""

from __future__ import annotations

import logging

import numpy as np

from shrimpy.mantis.position_update import PositionCoordinates, PositionUpdater

logger = logging.getLogger(__name__)


class DynaTrackUpdater(PositionUpdater):
    """Position updater that tracks neuromast drift across timepoints.

    On the first call for a given position index, the z-stack is stored as the
    reference image. On subsequent calls, the current z-stack is compared to the
    reference to compute a translational shift in X, Y, and Z, which is applied
    to the position coordinates.
    """

    def __init__(self) -> None:
        self._reference_stacks: dict[int, np.ndarray] = {}
        # TODO: add any model state, parameters, or configuration needed
        # for the tracking algorithm (e.g. cross-correlation settings,
        # maximum allowed shift, subpixel refinement options)

    def update(
        self,
        timepoint_index: int,
        position_index: int,
        position: PositionCoordinates,
        data: list[np.ndarray] | None = None,
    ) -> PositionCoordinates:
        """Compute updated position by tracking drift from reference z-stack.

        Parameters
        ----------
        timepoint_index : int
            The current timepoint index.
        position_index : int
            The position that was just acquired.
        position : PositionCoordinates
            Current coordinates for this position.
        data : list[np.ndarray] | None
            Frames acquired for this position (one 2D array per z-slice).

        Returns
        -------
        PositionCoordinates
            Position corrected for neuromast drift.
        """
        if data is None or len(data) == 0:
            logger.warning(
                f"DynaTrack: no data for p={position_index} at t={timepoint_index}, "
                "returning position unchanged"
            )
            return position

        current_stack = np.stack(data)

        # Store reference on first encounter
        if position_index not in self._reference_stacks:
            self._reference_stacks[position_index] = current_stack
            logger.info(
                f"DynaTrack: stored reference stack for p={position_index} "
                f"(shape={current_stack.shape})"
            )
            return position

        reference_stack = self._reference_stacks[position_index]
        shift = self._compute_shift(reference_stack, current_stack)

        # TODO: decide whether to update the reference stack after each
        # timepoint (track cumulative drift) or always compare to the
        # original reference (track absolute drift from t=0)

        logger.info(
            f"DynaTrack: p={position_index} t={timepoint_index} "
            f"shift=({shift[0]:.2f}, {shift[1]:.2f}, {shift[2]:.2f})"
        )

        return PositionCoordinates(
            x=position.x + shift[0],
            y=position.y + shift[1],
            z=(position.z or 0) + shift[2] if position.z is not None else None,
        )

    def _compute_shift(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compute the (x, y, z) shift between reference and current z-stacks.

        Parameters
        ----------
        reference : np.ndarray
            Reference z-stack, shape (Z, Y, X).
        current : np.ndarray
            Current z-stack, shape (Z, Y, X).

        Returns
        -------
        tuple[float, float, float]
            Estimated shift in (x, y, z) in stage coordinates (microns).
        """
        # TODO: implement the actual tracking algorithm. Options include:
        # - 3D cross-correlation (e.g. scipy.signal.fftconvolve or
        #   skimage.registration.phase_cross_correlation)
        # - 2D cross-correlation on a max-projection, plus separate
        #   Z shift estimation from axial profile correlation
        # - Feature-based tracking (detect neuromast blobs, match across frames)
        #
        # The returned shift should be in stage coordinate units (microns),
        # so pixel shifts need to be multiplied by the pixel size.

        # TODO: convert pixel shifts to stage coordinates using pixel size
        # and z-step size from the acquisition metadata

        return (0.0, 0.0, 0.0)
