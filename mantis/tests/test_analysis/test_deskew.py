import numpy as np

from mantis.analysis import deskew


def test_deskew_data():
    raw_data = np.random.random((2, 3, 4))
    px_to_scan_ratio = 0.386
    pixel_size_um = 1.0
    ls_angle_deg = 36
    ko = True
    deskewed_data = deskew.deskew_data(
        raw_data, px_to_scan_ratio, ls_angle_deg, keep_overhang=ko
    )
    assert deskewed_data.shape[1] == 4

    assert (
        deskewed_data.shape
        == deskew.get_deskewed_data_shape(
            raw_data.shape, pixel_size_um, ls_angle_deg, px_to_scan_ratio, keep_overhang=ko
        )[0]
    )
