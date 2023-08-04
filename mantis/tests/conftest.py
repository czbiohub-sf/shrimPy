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
def example_fov(tmp_path):
    fov_path = tmp_path / "fov.zarr"

    # Generate input dataset
    fov_dataset = open_ome_zarr(
        fov_path,
        layout="fov",
        mode="w",
        channel_names=["GFP", "RFP"],
    )

    fov_dataset.create_zeros("0", (3, 2, 4, 5, 6), dtype=np.uint)
    yield fov_path, fov_dataset
