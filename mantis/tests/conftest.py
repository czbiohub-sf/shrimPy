import numpy as np
import pytest
import yaml

from iohub.ngff import open_ome_zarr

# These fixtures return paired
# - paths for testing CLIs
# - objects for testing underlying functions


@pytest.fixture(scope="function")
def demo_acquisition_settings():
    settings_path = "./mantis/acquisition/settings/demo_acquisition_settings.yaml"
    with open(settings_path) as file:
        settings = yaml.safe_load(file)
    yield settings_path, settings


@pytest.fixture(scope="function")
def example_acquisition_settings():
    settings_path = "./mantis/acquisition/settings/example_acquisition_settings.yaml"
    with open(settings_path) as file:
        settings = yaml.safe_load(file)
    yield settings_path, settings


@pytest.fixture(scope="function")
def example_deskew_settings():
    settings_path = "./mantis/analysis/settings/example_deskew_settings.yml"
    with open(settings_path) as file:
        settings = yaml.safe_load(file)
    yield settings_path, settings


@pytest.fixture(scope="function")
def example_register_settings():
    settings_path = "./mantis/analysis/settings/example_register_settings.yml"
    with open(settings_path) as file:
        settings = yaml.safe_load(file)
    yield settings_path, settings


@pytest.fixture(scope="function")
def example_stabilize_timelapse_settings():
    settings_path = "./mantis/analysis/settings/example_stabilize_timelapse_settings.yml"
    with open(settings_path) as file:
        settings = yaml.safe_load(file)
    yield settings_path, settings


@pytest.fixture(scope="function")
def example_concatenate_settings():
    settings_path = "./mantis/analysis/settings/example_concatenate_settings.yml"
    with open(settings_path) as file:
        settings = yaml.safe_load(file)
    yield settings_path, settings


@pytest.fixture(scope="function")
def example_plate(tmp_path):
    plate_path = tmp_path / "plate.zarr"

    position_list = (
        ("A", "1", "0"),
        ("B", "1", "0"),
        ("B", "2", "0"),
    )

    # Generate input dataset
    plate_dataset = open_ome_zarr(
        plate_path,
        layout="hcs",
        mode="w",
        channel_names=["GFP", "RFP", "Phase3D", "Orientation", "Retardance", "Birefringence"],
    )

    for row, col, fov in position_list:
        position = plate_dataset.create_position(row, col, fov)
        position["0"] = np.random.uniform(0.0, 255.0, size=(3, 6, 4, 5, 6)).astype(np.float32)

    yield plate_path, plate_dataset


@pytest.fixture(scope="function")
def example_plate_2(tmp_path):
    plate_path = tmp_path / "plate.zarr"

    position_list = (
        ("A", "1", "0"),
        ("B", "1", "0"),
        ("B", "2", "0"),
    )

    # Generate input dataset
    plate_dataset = open_ome_zarr(
        plate_path,
        layout="hcs",
        mode="w",
        channel_names=["GFP", "RFP"],
    )

    for row, col, fov in position_list:
        position = plate_dataset.create_position(row, col, fov)
        position["0"] = np.random.randint(
            0, np.iinfo(np.uint16).max, size=(3, 2, 4, 5, 6), dtype=np.uint16
        )
    yield plate_path, plate_dataset
