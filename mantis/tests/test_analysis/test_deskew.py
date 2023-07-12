import numpy as np

from mantis.analysis import deskew


def test_deskew_data():
    raw_data = np.random.random((2, 3, 4))
    px_to_scan_ratio = 0.386
    pixel_size_um = 1.0
    ls_angle_deg = 36
    keep_overhang = True
    deskewed_data = deskew.deskew_data(raw_data, ls_angle_deg, px_to_scan_ratio, keep_overhang)
    assert deskewed_data.shape[1] == 4

    assert (
        deskewed_data.shape
        == deskew.get_deskewed_data_shape(
            raw_data.shape,
            ls_angle_deg,
            px_to_scan_ratio,
            keep_overhang,
            pixel_size_um=pixel_size_um,
        )[0]
    )
