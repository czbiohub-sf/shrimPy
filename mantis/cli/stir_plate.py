import logging
import time

import click

from pycromanager import Core, Studio

from mantis.acquisition.acq_engine import LF_ZMQ_PORT
from mantis.acquisition.microscope_operations import get_position_list, set_xy_position

logger = logging.getLogger(__name__)


def stir_plate(duration_hours: float, dwell_time_min: int) -> None:
    """Move through all positions defined in the Micro-manager position list,
    dwelling at each position for dwell_time_min until duration_hours is reached.
    This helps distribute the silicone oil evenly across the plate.

    Parameters
    ----------
    duration_hours : float
        Total duration for stirring the plate in hours.
    dwell_time_min : int
        Time spent at each position in minutes.
    """

    # Connect to the Micro-manager Studio
    mmc = Core(port=LF_ZMQ_PORT)
    mmStudio = Studio(port=LF_ZMQ_PORT)
    z_stage = None

    # Import positions from the Micro-manager position list
    positions, position_labels = get_position_list(mmStudio, z_stage)
    num_positions = len(positions)
    if num_positions == 0:
        raise RuntimeError(
            "No positions found in the Micro-manager position list. "
            "Please create a position list before running this command."
        )

    p_idx = 0
    t_start = time.time()
    t_end = t_start + duration_hours * 3600  # Convert hours to seconds
    while time.time() < t_end:
        # Move to the next position
        logger.info(f"Moving to position: {position_labels[p_idx]}")
        set_xy_position(mmc, positions[p_idx][:2])

        # Dwell for specified amount of time
        time.sleep(dwell_time_min * 60)

        # Increment position index, wrap around if necessary
        p_idx = (p_idx + 1) % num_positions

    # Move back to the first position
    logger.info("Stirring completed, moving to the first position.")
    set_xy_position(mmc, positions[0][:2])


@click.command("stir-plate")
@click.option(
    "--duration-hours",
    type=float,
    required=True,
    help="Total duration for stirring the plate in hours.",
)
@click.option(
    "--dwell-time-min",
    type=int,
    required=False,
    default=1,
    help="Time spent at each position in minutes, by default 1 minute.",
)
def stir_plate_cli(duration_hours: float, dwell_time_min: int) -> None:
    """Stir the plate by moving through all positions in the
    Micro-manager position list.
    """
    stir_plate(duration_hours, dwell_time_min)
