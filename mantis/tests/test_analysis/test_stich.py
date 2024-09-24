import tempfile

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from iohub import open_ome_zarr

from mantis.analysis.AnalysisSettings import ProcessingSettings
from mantis.analysis.stitch import (
    blend,
    cleanup_shifts,
    compute_total_translation,
    consolidate_zarr_fov_shifts,
    estimate_shift,
    estimate_zarr_fov_shifts,
    get_grid_rows_cols,
    get_image_shift,
    get_stitch_output_shape,
    preprocess_and_shift,
    process_dataset,
    shift_image,
)


@pytest.fixture
def processing_settings():
    return ProcessingSettings()


def create_test_csv(path, data, filename):
    """Helper function to create dummy CSV files."""
    filepath = path / filename
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    return filepath


def test_estimate_shift(test_data):
    test_data = np.random.rand(10, 10, 10, 10)

    with pytest.raises(ValueError):
        estimate_shift(test_data, test_data, 0.5, "row")


def test_process_dataset():
    test_data = np.random.rand(10, 10, 10, 10)
    processing_settings = ProcessingSettings()
    with pytest.raises(ValueError):

        process_dataset(test_data, processing_settings)


def test_preprocess_and_shift():
    processing_settings = ProcessingSettings()
    test_data_int = np.random.randint(0, 255, (10, 10, 10, 10), dtype=np.uint8)
    shift = (30, 30)
    scale_x = 0.1
    scale_y = 0.1
    with pytest.raises(ValueError):
        preprocess_and_shift(test_data_int, processing_settings, shift, scale_x, scale_y)


def test_blend():
    test_data_int = np.random.randint(0, 256, (30, 30, 30, 30), dtype=np.int32)
    with pytest.raises(ValueError):
        blend(test_data_int)


def test_get_stitch_output_shape():
    n_rows = 2
    n_cols = 2
    sizeY = 100
    sizeX = 100
    col_translation = (
        10,
        10,
    )
    row_translation = (10, 10)

    with pytest.raises(ValueError):
        get_stitch_output_shape(n_rows, n_cols, sizeY, sizeX, col_translation, row_translation)


def test_get_image_shift():

    col_idx = 1
    row_idx = 1
    col_translation = (10, 5)
    row_translation = (15, 10)
    global_translation = (20, 25)

    with pytest.raises(ValueError):
        get_image_shift(col_idx, row_idx, col_translation, row_translation, global_translation)


def test_shift_image():
    czyx_data = np.random.rand(2, 3, 100, 100)
    yx_output_shape = (120, 120)
    yx_shift = (10, 15)

    with pytest.raises(ValueError):
        shift_image(czyx_data, yx_output_shape, yx_shift, verbose=True)


def test_compute_total_translation():
    csv_filepath = '/fake/path/to/shifts.csv'

    mock_data = {
        'fov0': ['001', '002'],
        'fov1': ['002', '003'],
        'well': ['A', 'A'],
        'direction': ['col', 'row'],
        'shift-x': [1, -1],
        'shift-y': [1, -1],
    }
    df = pd.DataFrame(mock_data)

    with patch('pandas.read_csv', return_value=df):
        with pytest.raises(ValueError):
            compute_total_translation(csv_filepath)


def test_cleanup_shifts():
    csv_filepath = '/fake/path/to/shifts.csv'
    pixel_size_um = 0.5

    mock_data = {
        'well': ['W1', 'W1', 'W2'],
        'fov0': ['001', '002', '003'],
        'fov1': ['002', '003', '004'],
        'shift-x': [1, 2, 3],
        'shift-y': [1, 2, 3],
        'direction': ['row', 'col', 'row'],
    }
    df = pd.DataFrame(mock_data)

    with patch('pandas.read_csv', return_value=df):
        with patch('pandas.DataFrame.to_csv'):
            with pytest.raises(ValueError):
                cleanup_shifts(csv_filepath, pixel_size_um)


def test_estimate_zarr_fov_shifts():
    T, Y, X = 1, 10, 10

    position_list = (
        ("A", "3", "7"),
        ("A", "3", "8"),
        ("A", "3", "9"),
        ("B", "3", "7"),
        ("B", "3", "8"),
        ("B", "3", "9"),
        ("B", "4", "6"),
        ("B", "4", "7"),
        ("B", "4", "8"),
        ("B", "4", "9"),
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        store_path = Path(tmpdirname) / "test_dataset.zarr"

        with open_ome_zarr(
            store_path, layout="hcs", mode="w-", channel_names=["Nuclei_prediction_labels"]
        ) as dataset:
            for row, col, fov in position_list:
                position = dataset.create_position(row, col, fov)
                position["0"] = np.random.randint(
                    0, np.iinfo(np.uint32).max, size=(T, 1, 1, Y, X), dtype=np.uint32
                )

        fov0_zarr_path = store_path / "A" / "3" / "7"
        fov1_zarr_path = store_path / "A" / "3" / "8"

        tcz_index = (0, 0, 0)
        percent_overlap = 0.5
        fliplr = False
        flipud = False
        direction = "col"

        with pytest.raises(ValueError):

            estimate_zarr_fov_shifts(
                str(fov0_zarr_path),
                str(fov1_zarr_path),
                tcz_index,
                percent_overlap,
                fliplr,
                flipud,
                direction,
            )


def test_consolidate_zarr_fov_shifts():
    with tempfile.TemporaryDirectory() as tmpdirname:
        input_dir = Path(tmpdirname)

        create_test_csv(
            input_dir, {'fov0': ['001', '002'], 'fov1': ['003', '004']}, 'test1_shift.csv'
        )
        create_test_csv(
            input_dir, {'fov0': ['005', '006'], 'fov1': ['007', '008']}, 'test2_shift.csv'
        )

        output_filepath = input_dir / "output.csv"

        with pytest.raises(ValueError):
            consolidate_zarr_fov_shifts(str(input_dir), str(output_filepath))


def test_get_grid_rows_cols():
    T, Y, X = 1, 10, 10

    position_list = (
        ("A", "3", "7"),
        ("A", "3", "8"),
        ("A", "3", "9"),
        ("B", "3", "7"),
        ("B", "3", "8"),
        ("B", "3", "9"),
        ("B", "4", "6"),
        ("B", "4", "7"),
        ("B", "4", "8"),
        ("B", "4", "9"),
    )

    with tempfile.TemporaryDirectory() as input_dir:
        input_data_path = Path(input_dir) / "input.zarr"

        with open_ome_zarr(
            input_data_path,
            layout="hcs",
            mode="w-",
            channel_names=["Nuclei_prediction_labels"],
        ) as dataset:
            for row, col, fov in position_list:
                position = dataset.create_position(row, col, fov)
                position["0"] = np.random.randint(
                    0, np.iinfo(np.uint32).max, size=(T, 1, 1, Y, X), dtype=np.uint32
                )

        with pytest.raises(ValueError):

            get_grid_rows_cols(str(input_data_path))
