import tempfile

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mantis.analysis.analyze_psf import (
    _adjust_fig,
    _generate_html,
    _make_plots,
    analyze_psf,
    calculate_peak_widths,
    detect_peaks,
    extract_beads,
    generate_report,
    plot_fwhm_vs_acq_axes,
    plot_psf_amp,
    plot_psf_slices,
)


@pytest.fixture
def test_data():
    return np.random.rand(10, 10, 10)


@pytest.fixture
def tempdir():
    with tempfile.TemporaryDirectory() as temp:
        yield Path(temp)


@pytest.fixture
def beads():
    return [np.random.random((10, 10, 10)) for _ in range(5)]


@pytest.fixture
def df_gaussian_fit():
    data = {
        'x_mu': np.random.rand(5),
        'y_mu': np.random.rand(5),
        'z_mu': np.random.rand(5),
        'zyx_x_fwhm': np.random.rand(5),
        'zyx_y_fwhm': np.random.rand(5),
        'zyx_z_fwhm': np.random.rand(5),
        'zyx_amp': np.random.rand(5),
        'zyx_pc3_fwhm': np.random.rand(5),
        'zyx_pc2_fwhm': np.random.rand(5),
        'zyx_pc1_fwhm': np.random.rand(5),
    }
    return pd.DataFrame(data)


@pytest.fixture
def df_1d_peak_width():
    return pd.DataFrame(
        {
            '1d_x_fwhm': np.random.rand(5),
            '1d_y_fwhm': np.random.rand(5),
            '1d_z_fwhm': np.random.rand(5),
        }
    )


def test_make_plots(tempdir, beads, df_gaussian_fit, df_1d_peak_width):
    scale = (0.5, 0.5, 0.5)
    axis_labels = ("T", "Z", "Y", "X")
    with pytest.warns(DeprecationWarning) as record:
        _make_plots(tempdir, beads, df_gaussian_fit, df_1d_peak_width, scale, axis_labels)
    assert "biahub" in str(record.list[0].message), "Deprecation warning was not found."


def test_generate_report(tempdir, beads, test_data, df_gaussian_fit, df_1d_peak_width, mocker):
    mock_open = mocker.patch('webbrowser.open')
    scale = (0.5, 0.5, 0.5)
    axis_labels = ("T", "Z", "Y", "X")
    with pytest.warns(DeprecationWarning) as record:
        generate_report(
            tempdir,
            tempdir,
            "mock_dataset",
            beads,
            test_data,
            df_gaussian_fit,
            df_1d_peak_width,
            scale,
            axis_labels,
        )
    assert "biahub" in str(record.list[0].message), "Deprecation warning was not found."
    mock_open.assert_called_once()


def test_extract_beads(test_data):
    points = np.random.rand(5, 3)
    scale = (0.5, 0.5, 0.5)
    with pytest.warns(DeprecationWarning) as record:
        extract_beads(test_data, points, scale, (1, 1, 1))
    assert "biahub" in str(record.list[0].message), "Deprecation warning was not found."


def test_calculate_peak_widths(test_data):
    scale = (0.5, 0.5, 0.5)
    with pytest.warns(DeprecationWarning) as record:
        calculate_peak_widths(test_data, scale)
    assert "biahub" in str(record.list[0].message), "Deprecation warning was not found."


def test_plot_psf_slices(tempdir, beads):
    scale = (0.5, 0.5, 0.5)
    axis_labels = ("T", "Z", "Y", "X")
    with pytest.warns(DeprecationWarning) as record:
        plot_psf_slices(tempdir, beads, scale, axis_labels, beads)
    assert "biahub" in str(record.list[0].message), "Deprecation warning was not found."


def test_detect_peaks(test_data):
    with pytest.warns(DeprecationWarning) as record:
        detect_peaks(test_data)
    assert "biahub" in str(record.list[0].message), "Deprecation warning was not found."


def test_adjust_fig():
    fig, ax = plt.subplots(2, 2)  # Creates a 2x2 grid of Axes objects
    with pytest.warns(DeprecationWarning) as record:
        _adjust_fig(fig, ax)
    assert "biahub" in str(record.list[0].message), "Deprecation warning was not found."


def test_plot_fwhm_vs_acq_axes(tempdir):
    num_data_points = 100
    x = np.linspace(0, 10, num_data_points)
    y = np.linspace(0, 20, num_data_points)
    z = np.linspace(0, 30, num_data_points)
    fwhm_x = np.random.normal(2, 0.1, num_data_points)
    fwhm_y = np.random.normal(2, 0.1, num_data_points)
    fwhm_z = np.random.normal(2, 0.1, num_data_points)
    axis_labels = ('Z', 'Y', 'X')

    with pytest.warns(DeprecationWarning) as record:
        plot_fwhm_vs_acq_axes(tempdir, x, y, z, fwhm_x, fwhm_y, fwhm_z, axis_labels)
    assert "biahub" in str(record.list[0].message), "Deprecation warning was not found."


def test_plot_psf_amp(tempdir):
    num_points = 100
    x = np.linspace(0, 10, num_points)
    y = np.linspace(0, 20, num_points)
    z = np.linspace(0, 30, num_points)
    amp = np.random.rand(num_points)
    axis_labels = ('Z', 'Y', 'X')

    with pytest.warns(DeprecationWarning) as record:
        plot_psf_amp(tempdir, x, y, z, amp, axis_labels)
    assert "biahub" in str(record.list[0].message), "Deprecation warning was not found."


def test_generate_html():
    # Define all the data directly inside the test function
    dataset_name = "Test Dataset"
    data_path = "/path/to/dataset"
    dataset_scale = (0.5, 0.1, 0.2)  # Example scale in z, y, x
    num_beads_total_good_bad = (100, 90, 10)  # Total beads, good beads, bad beads
    fwhm_1d_mean = (1.5, 1.6, 1.7)
    fwhm_1d_std = (0.05, 0.06, 0.07)
    fwhm_3d_mean = (1.2, 1.3, 1.4)
    fwhm_3d_std = (0.03, 0.04, 0.05)
    fwhm_pc_mean = (1.0, 0.9, 0.8)
    bead_psf_slices_paths = [
        "/path/to/bead_xy.png",
        "/path/to/bead_xz.png",
        "/path/to/bead_yz.png",
    ]
    fwhm_vs_acq_axes_paths = [
        "/path/to/fwhm_z.png",
        "/path/to/fwhm_y.png",
        "/path/to/fwhm_x.png",
    ]
    psf_amp_paths = ["/path/to/psf_amp_xy.png", "/path/to/psf_amp_z.png"]
    axis_labels = ("Z", "Y", "X")

    # Call the function with all parameters
    with pytest.warns(DeprecationWarning) as record:
        _generate_html(
            dataset_name,
            data_path,
            dataset_scale,
            num_beads_total_good_bad,
            fwhm_1d_mean,
            fwhm_1d_std,
            fwhm_3d_mean,
            fwhm_3d_std,
            fwhm_pc_mean,
            bead_psf_slices_paths,
            fwhm_vs_acq_axes_paths,
            psf_amp_paths,
            axis_labels,
        )
    assert "biahub" in str(record.list[0].message), "Deprecation warning was not found."


# # ANALYZE_PSF FUNCTION -> anlyze_psf.py file change the += in line 191 to 193
def test_analyze_psf():
    zyx_patches = [np.random.rand(10, 10, 10) for _ in range(3)]
    bead_offsets = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    scale = (0.1, 0.1, 0.1)
    with pytest.warns(DeprecationWarning) as record:
        analyze_psf(zyx_patches, bead_offsets, scale)
    assert "biahub" in str(record.list[0].message), "Deprecation warning was not found."
