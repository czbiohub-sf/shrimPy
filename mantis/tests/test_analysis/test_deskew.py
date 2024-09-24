import numpy as np
import pytest

from mantis.analysis.deskew import (
    _average_n_slices,
    _get_averaged_shape,
    _get_transform_matrix,
    deskew_data,
    get_deskewed_data_shape,
)

def test_average_n_slices():
    data = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]], [[13, 14], [15, 16]]]
    )
    # Non-divisible window
    averaged_data1 = _average_n_slices(data, average_window_width=3)
    expected_data1 = np.array([[[5, 6], [7, 8]], [[13, 14], [15, 16]]])
    assert np.array_equal(averaged_data1, expected_data1)
    assert averaged_data1.shape == _get_averaged_shape(data.shape, 3)

    # Divisible window
    averaged_data2 = _average_n_slices(data, average_window_width=2)
    expected_data2 = np.array([[[3, 4], [5, 6]], [[11, 12], [13, 14]]])
    assert np.array_equal(averaged_data2, expected_data2)
    assert averaged_data2.shape == _get_averaged_shape(data.shape, 2)

    # Window = 1
    averaged_data3 = _average_n_slices(data, average_window_width=1)
    assert np.array_equal(averaged_data3, data)
    assert averaged_data3.shape == _get_averaged_shape(data.shape, 1)


def test_deskew_data():
    raw_data = np.random.random((2, 3, 4))
    px_to_scan_ratio = 0.386
    pixel_size_um = 1.0
    ls_angle_deg = 36
    average_n_slices = 1
    keep_overhang = True
    deskewed_data = deskew_data(
        raw_data, ls_angle_deg, px_to_scan_ratio, keep_overhang, average_n_slices
    )
    assert deskewed_data.shape[1] == 4
    assert deskewed_data[0, 0, 0] != 0  # indicates incorrect shifting

    assert (
        deskewed_data.shape
        == get_deskewed_data_shape(
            raw_data.shape,
            ls_angle_deg,
            px_to_scan_ratio,
            keep_overhang,
            pixel_size_um=pixel_size_um,
        )[0]
    )


@pytest.mark.parametrize(
    "function_to_test, parameters",
    [
        (_average_n_slices, {"data": np.random.rand(10, 10, 10)}),
        (
            _get_averaged_shape,
            {"deskewed_data_shape": (10, 10, 10), "average_window_width": 1},
        ),
        (_get_transform_matrix, {"ls_angle_deg": 0.5, "px_to_scan_ratio": 0.5}),
        (
            get_deskewed_data_shape,
            {
                "raw_data_shape": (10, 10, 10),
                "ls_angle_deg": 0.5,
                "px_to_scan_ratio": 0.5,
                "keep_overhang": False,
            },
        ),
        (
            deskew_data,
            {
                "raw_data": np.random.rand(10, 10, 10),
                "ls_angle_deg": 0.5,
                "px_to_scan_ratio": 0.5,
                "keep_overhang": False,
            },
        ),
    ],
)
def test_error(function_to_test, parameters):
    with pytest.raises(ValueError):
        function_to_test(**parameters)
   
