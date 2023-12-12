from mantis.cli.stabilization import (
    estimate_position_focus,
    combine_dataframes,
    get_mean_z_positions,
    calculate_z_drift,
    calculate_yx_stabilization,
)
import pandas as pd
import numpy as np


def test_estimate_position_focus():
    # Create input data
    z_positions = [1, 2, 3, 4, 5]
    focus_scores = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Call the function
    result = estimate_position_focus(z_positions, focus_scores)

    # Check the result
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)


def test_mean_z_positions():
    # Create input data
    df = pd.DataFrame(
        {
            "channel": ["GFP"],
            "channel_idx": [0, 0, 0, 0, 0, 0, 0, 0],
            "time_min": [1.0, 1.0, 1.0 , 1.0 , 2.0 , 2.0 , 2.0 , 2.0],
            "focal_idx": [20, 25, 19, 16, 24, 29, 21, 18],
            "position_idx": [1, 2, 3, 4, 1, 2, 3, 4],
        }
    )

    # Call the function
    result = get_mean_z_positions(df, 0)
    pos_1 = np.array([20, 25, 19, 16]).mean()
    pos_2 = np.array([24, 29, 21, 18]).mean()

    # Check the result
    assert result == [pos_1,pos_2]

