import pytest
import yaml


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
