"""Manual autoexposure support for Mantis acquisitions.

The user supplies an ``illumination.csv`` file with per-well exposure time
and laser power. The engine applies those values whenever the acquisition
moves into a new well.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ILLUMINATION_CSV_COLUMNS = {
    "well_id",
    "channel_name",
    "exposure_time_ms",
    "laser_name",
    "laser_power_mW",
}


def load_manual_illumination_settings(csv_filepath: str | Path) -> pd.DataFrame:
    """Load the manual illumination settings CSV.

    The CSV must have columns ``well_id``, ``channel_name``,
    ``exposure_time_ms``, ``laser_name``, and ``laser_power_mW``. The
    returned DataFrame is indexed by ``(well_id, channel_name)``, so each
    row describes the illumination to apply when the acquisition enters a
    given well in a given channel. Rows must be unique on the index.
    """
    df = pd.read_csv(csv_filepath, dtype=str)
    if set(df.columns) != ILLUMINATION_CSV_COLUMNS:
        raise ValueError(
            f"CSV file {csv_filepath} must contain columns: {sorted(ILLUMINATION_CSV_COLUMNS)}"
        )
    df["exposure_time_ms"] = df["exposure_time_ms"].astype(float)
    df["laser_power_mW"] = df["laser_power_mW"].astype(float)
    df.set_index(["well_id", "channel_name"], inplace=True, verify_integrity=True)
    return df
